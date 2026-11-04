"""Hierarchical next-basket scorer: stage-1 key -> type x attributes -> item.

P(i | u) = theta_u(c) * tau_u(t | c) * A_u(x(i) | t) * phi_u(i | t, x(i)), ranked by the sum of logs.
c is the stage-1 key (Instacart aisle, CJ commodity), t a type nested in c (the unit is the (c, t)
pair), x(i) the item's attributes that apply to t (C_a^type), and the cell (t, x) the item's group in
the last stage. All statistics come from one split's history inside the food scope F; targets are only
used for metrics and candidate diagnostics.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import sparse

from .attributes import (
    applicable_categories,
    attribute_column,
    category_sets,
    exclude_nonfood_categories,
)
from .ingredient_types import stage_columns, type_attribute_categories
from .repeat_explore import (
    PROBABILITY_CLIP,
    Interactions,
    PreferenceModel,
    build_interactions,
    build_preferences,
    item_popularity,
    repeat_scores,
    scope_mask,
    scope_targets,
)


# Added per unit to every population share (theta_bar, tau_bar, pop_cell) so all probabilities are positive.
PSEUDOCOUNT = 0.01
# Variant name -> (hierarchy, k override). noattr drops attributes (A = 1, cells = types); notype uses
# the stage-1 key as the type, so the type stage disappears and cells are (c, x).
VARIANTS = {
    "hier_full": ("full", None),
    "hier_noattr": ("noattr", None),
    "hier_notype": ("notype", None),
    "hier_noknn": ("full", 0),
}


@dataclass(frozen=True)
class Params:
    alpha: float
    mu: float
    lambda_type: float
    k: int


def csr_from(values: np.ndarray, rows: np.ndarray, cols: np.ndarray, shape: tuple[int, int]) -> sparse.csr_matrix:
    """CSR with duplicate coordinates summed and sorted indices."""
    matrix = sparse.csr_matrix((values.astype(np.float64), (rows, cols)), shape=shape)
    matrix.sum_duplicates()
    matrix.sort_indices()
    return matrix


def lookup(matrix: sparse.csr_matrix, rows: np.ndarray, cols: np.ndarray) -> np.ndarray:
    """matrix[rows[j], cols[j]] for every j, as a dense vector (0 where the entry is absent)."""
    matrix = matrix.tocsr()
    matrix.sort_indices()
    if matrix.nnz == 0 or len(rows) == 0:
        return np.zeros(len(rows))
    width = np.int64(matrix.shape[1])
    entry_rows = np.repeat(np.arange(matrix.shape[0], dtype=np.int64), np.diff(matrix.indptr))
    keys = entry_rows * width + matrix.indices.astype(np.int64)
    query = np.asarray(rows, dtype=np.int64) * width + np.asarray(cols, dtype=np.int64)
    position = np.minimum(np.searchsorted(keys, query), len(keys) - 1)
    return np.where(keys[position] == query, matrix.data[position], 0.0)


def grouped_members(groups: np.ndarray, n_groups: int, popularity: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """CSR-style (ptr, members) of items per group, by descending popularity (ties by item index)."""
    items = np.flatnonzero(groups >= 0)
    order = np.lexsort((items, -popularity[items], groups[items]))
    members = items[order]
    ptr = np.zeros(n_groups + 1, dtype=np.int64)
    np.cumsum(np.bincount(groups[members], minlength=n_groups), out=ptr[1:])
    return ptr, members


def user_neighbors(profiles: np.ndarray, k: int, block: int) -> tuple[np.ndarray, np.ndarray]:
    """Top-k cosine neighbours per user from L2-normalised profiles, computed in user blocks.

    The user itself is excluded and only positive similarities count; missing neighbours are -1.
    Ties are broken by user index. Never materialises the user x user matrix.
    """
    n_users = len(profiles)
    neighbors = np.full((n_users, k), -1, dtype=np.int64)
    sims = np.zeros((n_users, k))
    if k == 0 or n_users == 0:
        return neighbors, sims
    norms = np.linalg.norm(profiles, axis=1)
    unit = profiles / np.where(norms > 0, norms, 1.0)[:, None]
    width = min(k, n_users - 1)
    for start in range(0, n_users, block):
        stop = min(start + block, n_users)
        scores = unit[start:stop] @ unit.T
        scores[np.arange(stop - start), np.arange(start, stop)] = -np.inf
        order = np.argsort(-scores, axis=1, kind="stable")[:, :width]
        top = np.take_along_axis(scores, order, axis=1)
        valid = top > 0
        neighbors[start:stop, :width] = np.where(valid, order, -1)
        sims[start:stop, :width] = np.where(valid, top, 0.0)
    return neighbors, sims


def neighbor_matrix(neighbors: np.ndarray, rows: np.ndarray, k: int, n_users: int) -> sparse.csr_matrix:
    """rows x users 0/1 matrix of each row's first k neighbours."""
    chosen = neighbors[rows, :k]
    local, position = np.nonzero(chosen >= 0)
    return csr_from(np.ones(len(local)), local, chosen[local, position], (len(rows), n_users))


def attribute_categories(
    config: dict[str, object], dataset: str, items: pd.DataFrame, names: list[str]
) -> dict[str, set[str]]:
    """Aisle-level C_a (existing min_with / min_without rule, non-food aisles removed)."""
    settings = config["attributes"]
    table = applicable_categories(items, names, int(settings["min_with"]), int(settings["min_without"]))
    food_table = exclude_nonfood_categories(table, items, list(settings["exclude_departments"][dataset]))
    sets = category_sets(food_table)
    return {name: sets.get(name, set()) for name in names}


@dataclass
class Hierarchy:
    """Types nested in stage-1 keys, applicable attributes and cells over F, with history aggregates."""

    attributes: list[str]
    type_stage1: np.ndarray  # T, stage-1 index of each type
    type_labels: np.ndarray  # T, "stage1|type"
    item_type: np.ndarray  # items, -1 outside F
    item_cell: np.ndarray  # items, -1 outside F
    item_bits: np.ndarray  # items x A, attribute flags kept only where they apply to the item's type
    item_flag: np.ndarray  # items x A, raw attribute flags
    type_applicable: np.ndarray  # T x A, (c, t) in C_a^type
    type_ptr: np.ndarray
    type_members: np.ndarray  # F items per type by descending popularity
    cell_ptr: np.ndarray
    cell_members: np.ndarray  # F items per cell by descending popularity
    pop_cell: np.ndarray  # items, smoothed popularity share inside the cell
    type_counts: sparse.csr_matrix  # users x T, s_u(t)
    stage1_mass: np.ndarray  # users x C, sum over t in c of s_u(t)
    tau_hat: sparse.csr_matrix  # users x T, s_u(t) / stage1_mass_u(c(t))
    tau_bar: np.ndarray  # T, population type share inside its stage-1 key
    n_type: sparse.csr_matrix  # users x T, purchases n_u(t)
    n_attr_type: list[sparse.csr_matrix]  # per attribute, users x T, n_u(a, t)
    type_population: np.ndarray  # T x A, population pi_bar(a | t) (pi_bar(a) when nobody bought in t)
    cell_repeat: sparse.csr_matrix  # users x cells, sum of s_rep over the cell
    type_seen: sparse.csr_matrix  # users x T, distinct history items per type

    @property
    def n_types(self) -> int:
        return len(self.type_stage1)


@dataclass
class HierState:
    """History statistics of one split shared by all hierarchies and parameters."""

    dataset: str
    gamma: float
    interactions: Interactions
    in_scope: np.ndarray
    stage1_labels: np.ndarray  # C
    item_stage1: np.ndarray  # items, -1 outside F
    popularity: np.ndarray
    repeat: sparse.csr_matrix  # users x items, s_rep inside F
    seen: sparse.csr_matrix  # users x items, I_u inside F
    stage1_counts: np.ndarray  # users x C, s_u(c)
    theta_hat: np.ndarray  # users x C, s_u(c) / sum_c s_u(c) (0 for users without food history)
    theta_bar: np.ndarray  # C
    neighbors: np.ndarray  # users x k_max, -1 padded
    neighbor_sims: np.ndarray
    aisle_model: PreferenceModel  # pi_u(a) at the aisle level
    hierarchies: dict[str, Hierarchy]
    history_items: dict[str, set[str]]
    targets: dict[str, set[str]]  # food targets of history users with a non-empty food target
    excluded_users: int
    target_rows: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.int64))
    # One entry per (target user, food target item): user row, item index, item not in I_u.
    target_user: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.int64))
    target_item: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.int64))
    target_new: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=bool))

    @property
    def n_users(self) -> int:
        return len(self.interactions.users)

    @property
    def n_items(self) -> int:
        return len(self.interactions.items)


def _unique_weighted(
    users: np.ndarray, baskets: np.ndarray, units: np.ndarray, weights: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """One (user, basket, unit) row per basket that contains the unit; the weight is the basket's."""
    frame = pd.DataFrame({"user": users, "basket": baskets, "unit": units, "weight": weights})
    frame = frame.drop_duplicates(["user", "basket", "unit"])
    return frame["user"].to_numpy(), frame["unit"].to_numpy(), frame["weight"].to_numpy()


def build_hierarchy(
    state: HierState,
    item_table: pd.DataFrame,
    type_labels: pd.Series,
    attributes: list[str],
    min_with: int,
    min_without: int,
) -> Hierarchy:
    """Types (stage-1 key, type label) over F, C_a^type from the item list, cells, and history aggregates."""
    interactions, in_scope = state.interactions, state.in_scope
    n_users, n_items = state.n_users, state.n_items
    stage1 = state.item_stage1
    pair = pd.Series(
        np.where(in_scope, state.stage1_labels[np.maximum(stage1, 0)].astype(str), ""), dtype=object
    ) + "|" + type_labels.astype(str).to_numpy()
    codes, labels = pd.factorize(pair.where(in_scope), sort=True)
    item_type = np.where(in_scope, codes, -1).astype(np.int64)
    n_types = len(labels)
    type_stage1 = np.zeros(n_types, dtype=np.int64)
    type_stage1[item_type[in_scope]] = stage1[in_scope]

    flags = np.zeros((n_items, len(attributes)), dtype=np.int8)
    for column, name in enumerate(attributes):
        flags[:, column] = item_table[attribute_column(name)].fillna(0).to_numpy(np.int8)
    applicable = np.zeros((n_types, len(attributes)), dtype=bool)
    if attributes:
        frame = pd.DataFrame(
            {"stage1": "s", "type": item_type[in_scope].astype(str)}
            | {attribute_column(name): flags[in_scope, column] for column, name in enumerate(attributes)}
        )
        table = type_attribute_categories(frame, "stage1", "type", attributes, min_with, min_without)
        for column, name in enumerate(attributes):
            chosen = table[table["attribute"].eq(name) & table["included"]]["type"].astype(int).to_numpy()
            applicable[chosen, column] = True
    bits = flags * applicable[np.maximum(item_type, 0)] * in_scope[:, None]
    weights = 1 << np.arange(len(attributes), dtype=np.int64)
    cell_key = np.where(in_scope, item_type * (1 << len(attributes)) + bits.astype(np.int64) @ weights, -1)
    cell_codes, cell_values = pd.factorize(pd.Series(cell_key).where(in_scope), sort=True)
    item_cell = np.where(in_scope, cell_codes, -1).astype(np.int64)
    n_cells = len(cell_values)

    popularity = state.popularity
    type_ptr, type_members = grouped_members(item_type, n_types, popularity)
    cell_ptr, cell_members = grouped_members(item_cell, n_cells, popularity)
    cell_size = np.bincount(item_cell[in_scope], minlength=n_cells)
    cell_pop = np.bincount(item_cell[in_scope], weights=popularity[in_scope], minlength=n_cells)
    pop_cell = np.zeros(n_items)
    pop_cell[in_scope] = (popularity[in_scope] + PSEUDOCOUNT) / (
        cell_pop[item_cell[in_scope]] + PSEUDOCOUNT * cell_size[item_cell[in_scope]]
    )

    rows = in_scope[interactions.item_index]
    users = interactions.user_index[rows]
    items = interactions.item_index[rows]
    weight = np.power(state.gamma, interactions.exponent[rows].astype(np.float64))
    unit_users, unit_types, unit_weights = _unique_weighted(users, interactions.basket_seq[rows], item_type[items], weight)
    type_counts = csr_from(unit_weights, unit_users, unit_types, (n_users, n_types))
    to_stage1 = csr_from(np.ones(n_types), np.arange(n_types), type_stage1, (n_types, len(state.stage1_labels)))
    stage1_mass = np.asarray((type_counts @ to_stage1).todense())
    tau_hat = type_counts.copy()
    entry_rows = np.repeat(np.arange(n_users), np.diff(tau_hat.indptr))
    tau_hat.data = tau_hat.data / stage1_mass[entry_rows, type_stage1[tau_hat.indices]]
    type_total = np.asarray(type_counts.sum(axis=0)).ravel()
    types_per_stage1 = np.bincount(type_stage1, minlength=len(state.stage1_labels))
    stage1_total = np.bincount(type_stage1, weights=type_total, minlength=len(state.stage1_labels))
    tau_bar = (type_total + PSEUDOCOUNT) / (stage1_total[type_stage1] + PSEUDOCOUNT * types_per_stage1[type_stage1])

    n_type = csr_from(np.ones(len(users)), users, item_type[items], (n_users, n_types))
    n_attr_type = [
        csr_from(flags[items, column].astype(np.float64), users, item_type[items], (n_users, n_types))
        for column in range(len(attributes))
    ]
    type_purchases = np.asarray(n_type.sum(axis=0)).ravel()
    type_population = np.tile(state.aisle_model.prior[: len(attributes)], (n_types, 1)) if attributes else np.zeros((n_types, 0))
    for column in range(len(attributes)):
        attribute_purchases = np.asarray(n_attr_type[column].sum(axis=0)).ravel()
        bought = type_purchases > 0
        type_population[bought, column] = attribute_purchases[bought] / type_purchases[bought]
    type_population = np.clip(type_population, PROBABILITY_CLIP, 1.0 - PROBABILITY_CLIP)
    cell_repeat = csr_from(weight, users, item_cell[items], (n_users, n_cells))
    seen = state.seen.tocoo()
    type_seen = csr_from(np.ones(seen.nnz), seen.row, item_type[seen.col], (n_users, n_types))
    return Hierarchy(
        attributes=list(attributes),
        type_stage1=type_stage1,
        type_labels=np.asarray(labels, dtype=object),
        item_type=item_type,
        item_cell=item_cell,
        item_bits=bits.astype(np.int8),
        item_flag=flags,
        type_applicable=applicable,
        type_ptr=type_ptr,
        type_members=type_members,
        cell_ptr=cell_ptr,
        cell_members=cell_members,
        pop_cell=pop_cell,
        type_counts=type_counts,
        stage1_mass=stage1_mass,
        tau_hat=tau_hat,
        tau_bar=tau_bar,
        n_type=n_type,
        n_attr_type=n_attr_type,
        type_population=type_population,
        cell_repeat=cell_repeat,
        type_seen=type_seen,
    )


def build_hier_state(
    config: dict[str, object],
    dataset: str,
    items: pd.DataFrame,
    history: pd.DataFrame,
    target: pd.DataFrame,
    gamma: float,
    variants: tuple[str, ...] = ("full", "noattr", "notype"),
) -> HierState:
    """History statistics, user neighbours and the requested hierarchies; target only fills `targets`."""
    settings = config["hierarchical"]
    stage1_column, type_column = stage_columns(config, dataset)
    missing = [column for column in [stage1_column, type_column] if column not in items.columns]
    if missing:
        raise KeyError(f"Item table lacks {missing}; run prepare-types first")
    attributes = list(settings["attributes"][dataset])
    interactions = build_interactions(history, items)
    nonfood = list(config["scope"]["nonfood_departments"][dataset])
    in_scope = scope_mask(items, interactions.items, "food", nonfood)
    table = items.drop_duplicates("item_id").assign(item_id=lambda frame: frame["item_id"].astype(str))
    table = table.set_index("item_id").reindex(interactions.items)
    stage1_values = table[stage1_column].astype("string").fillna("UNKNOWN").astype(str)
    stage1_codes, stage1_labels = pd.factorize(stage1_values.where(in_scope), sort=True)
    item_stage1 = np.where(in_scope, stage1_codes, -1).astype(np.int64)
    n_users, n_items, n_stage1 = len(interactions.users), len(interactions.items), len(stage1_labels)

    popularity = item_popularity(interactions)
    scope_diagonal = sparse.diags(in_scope.astype(np.float64))
    repeat = (repeat_scores(interactions, gamma) @ scope_diagonal).tocsr()
    repeat.eliminate_zeros()
    repeat.sort_indices()
    seen = repeat.copy()
    seen.data[:] = 1.0

    rows = in_scope[interactions.item_index]
    weight = np.power(float(gamma), interactions.exponent[rows].astype(np.float64))
    unit_users, unit_stage1, unit_weights = _unique_weighted(
        interactions.user_index[rows],
        interactions.basket_seq[rows],
        item_stage1[interactions.item_index[rows]],
        weight,
    )
    stage1_counts = np.asarray(csr_from(unit_weights, unit_users, unit_stage1, (n_users, n_stage1)).todense())
    totals = stage1_counts.sum(axis=1, keepdims=True)
    theta_hat = np.divide(stage1_counts, totals, out=np.zeros_like(stage1_counts), where=totals > 0)
    column_total = stage1_counts.sum(axis=0)
    theta_bar = (column_total + PSEUDOCOUNT) / (column_total.sum() + PSEUDOCOUNT * n_stage1)
    k_max = max(int(value) for value in settings["knn_k"])
    neighbors, sims = user_neighbors(stage1_counts, k_max, int(settings.get("user_block", 1000)))

    categories = attribute_categories(config, dataset, items, attributes)
    aisle_model = build_preferences(history, items, interactions, categories, float(config["attributes"]["smoothing"]))
    item_ids = interactions.items.to_numpy()
    all_seen = sparse.csr_matrix(
        (np.ones(len(interactions.user_index)), (interactions.user_index, interactions.item_index)),
        shape=(n_users, n_items),
    )
    history_items = {
        str(user): set(item_ids[all_seen.indices[all_seen.indptr[row] : all_seen.indptr[row + 1]]].astype(str))
        for row, user in enumerate(interactions.users)
    }
    frame = target.assign(user_id=target["user_id"].astype(str), item_id=target["item_id"].astype(str))
    target_sets = {user: set(group) for user, group in frame.groupby("user_id")["item_id"] if user in history_items}
    targets, excluded = scope_targets(target_sets, set(item_ids[in_scope].astype(str)))
    state = HierState(
        dataset=dataset,
        gamma=float(gamma),
        interactions=interactions,
        in_scope=in_scope,
        stage1_labels=np.asarray(stage1_labels, dtype=object),
        item_stage1=item_stage1,
        popularity=popularity,
        repeat=repeat,
        seen=seen,
        stage1_counts=stage1_counts,
        theta_hat=theta_hat,
        theta_bar=theta_bar,
        neighbors=neighbors,
        neighbor_sims=sims,
        aisle_model=aisle_model,
        hierarchies={},
        history_items=history_items,
        targets=targets,
        excluded_users=excluded,
        target_rows=np.sort(interactions.users.get_indexer(list(targets))).astype(np.int64),
    )
    pairs = [(user, item) for user in sorted(targets) for item in sorted(targets[user])]
    if pairs:
        pair_users, pair_items = map(list, zip(*pairs))
        state.target_user = interactions.users.get_indexer(pair_users).astype(np.int64)
        state.target_item = interactions.items.get_indexer(pair_items).astype(np.int64)
        state.target_new = np.array([item not in history_items[user] for user, item in pairs], dtype=bool)
    min_with, min_without = int(settings["type_attr_min_with"]), int(settings["type_attr_min_without"])
    type_labels = {"full": table[type_column], "noattr": table[type_column], "notype": table[stage1_column]}
    for variant in variants:
        names = [] if variant == "noattr" else attributes
        state.hierarchies[variant] = build_hierarchy(
            state, table, type_labels[variant].astype("string").fillna("UNKNOWN"), names, min_with, min_without
        )
    return state


def stage_probabilities(
    state: HierState, hierarchy: Hierarchy, params: Params, rows: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """theta_u(c) (rows x C) and tau_u(t | c) (rows x T) for the given user rows.

    theta_u(c) = (s_u(c) + alpha * theta~_u(c)) / (sum_c s_u(c) + alpha), and
    tau_u(t | c) = (s_u(t) + alpha * tau~_u(t | c)) / (sum_{t' in c} s_u(t') + alpha). The tau
    denominator sums over the types of c (not s_u(c)) so that tau sums to one inside c. The neighbour
    priors average the neighbours' shares with the population share counted as one extra neighbour;
    for tau only neighbours who bought in c count, so without such neighbours the prior is tau_bar.
    """
    alpha, k = float(params.alpha), int(params.k)
    counts = state.stage1_counts[rows]
    type_counts = hierarchy.type_counts[rows].toarray()
    mass = hierarchy.stage1_mass[rows][:, hierarchy.type_stage1]
    if k == 0:
        theta_prior = np.broadcast_to(state.theta_bar, counts.shape)
        tau_prior = np.broadcast_to(hierarchy.tau_bar, type_counts.shape)
    else:
        near = neighbor_matrix(state.neighbors, rows, k, state.n_users)
        theta_prior = (near @ state.theta_hat + state.theta_bar) / (np.asarray(near.sum(axis=1)) + 1.0)
        covered = near @ (hierarchy.stage1_mass > 0).astype(np.float64)
        tau_prior = ((near @ hierarchy.tau_hat).toarray() + hierarchy.tau_bar) / (
            covered[:, hierarchy.type_stage1] + 1.0
        )
    theta = (counts + alpha * theta_prior) / (counts.sum(axis=1, keepdims=True) + alpha)
    tau = (type_counts + alpha * tau_prior) / (mass + alpha)
    return theta, tau


def type_preferences(
    state: HierState, hierarchy: Hierarchy, lambda_type: float, users: np.ndarray, types: np.ndarray
) -> np.ndarray:
    """pi_u(a | t) = (n_u(a, t) + lambda_t * pi_u(a)) / (n_u(t) + lambda_t), one row per (user, type)."""
    n_type = lookup(hierarchy.n_type, users, types)
    pi_user = state.aisle_model.pi[users][:, : len(hierarchy.attributes)]
    n_attr = np.column_stack([lookup(matrix, users, types) for matrix in hierarchy.n_attr_type]) if hierarchy.attributes else np.zeros((len(users), 0))
    pi = (n_attr + lambda_type * pi_user) / (n_type[:, None] + lambda_type)
    return np.clip(pi, PROBABILITY_CLIP, 1.0 - PROBABILITY_CLIP)


def attribute_log_prob(
    state: HierState, hierarchy: Hierarchy, lambda_type: float, users: np.ndarray, items: np.ndarray
) -> np.ndarray:
    """log A_u(x(i) | t(i)): Bernoulli log-likelihood summed over the attributes that apply to t(i)."""
    if not hierarchy.attributes:
        return np.zeros(len(users))
    types = hierarchy.item_type[items]
    pi = type_preferences(state, hierarchy, lambda_type, users, types)
    terms = np.where(hierarchy.item_bits[items] == 1, np.log(pi), np.log(1.0 - pi))
    return (terms * hierarchy.type_applicable[types]).sum(axis=1)


def item_cell_prob(
    hierarchy: Hierarchy, mu: float, repeat: sparse.csr_matrix, local: np.ndarray, users: np.ndarray, items: np.ndarray
) -> np.ndarray:
    """phi_u(i | t, x) = (s_rep(u, i) + mu * pop_cell(i)) / (sum_{j in cell} s_rep(u, j) + mu).

    `repeat` holds the rows of `local`; `users` are the matching global rows.
    """
    rep = lookup(repeat, local, items)
    cell_rep = lookup(hierarchy.cell_repeat, users, hierarchy.item_cell[items])
    return (rep + mu * hierarchy.pop_cell[items]) / (cell_rep + mu)


def log_scores(
    state: HierState,
    hierarchy: Hierarchy,
    params: Params,
    rows: np.ndarray,
    theta: np.ndarray,
    tau: np.ndarray,
    local: np.ndarray,
    items: np.ndarray,
) -> np.ndarray:
    """log theta + log tau + log A + log phi for (rows[local], items)."""
    users = rows[local]
    log_theta = np.log(theta[local, state.item_stage1[items]])
    log_tau = np.log(tau[local, hierarchy.item_type[items]])
    log_attr = attribute_log_prob(state, hierarchy, float(params.lambda_type), users, items)
    phi = item_cell_prob(hierarchy, float(params.mu), state.repeat[rows], local, users, items)
    return log_theta + log_tau + log_attr + np.log(phi)


def _pair_keys(local: np.ndarray, items: np.ndarray, n_items: int) -> np.ndarray:
    return np.asarray(local, dtype=np.int64) * np.int64(n_items) + np.asarray(items, dtype=np.int64)


def generate_candidates(
    state: HierState,
    hierarchy: Hierarchy,
    joint: np.ndarray,
    rows: np.ndarray,
    n_cells: int,
    per_cell: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(local row, item, seen) candidates: I_u inside F plus, for the n_cells types with the largest
    theta * tau, the per_cell most popular items of the type that are not in I_u."""
    n_items = state.n_items
    width = min(n_cells, hierarchy.n_types)
    top = np.argsort(-joint, axis=1, kind="stable")[:, :width]
    pair_local = np.repeat(np.arange(len(rows), dtype=np.int64), width)
    pair_type = top.ravel()
    seen_counts = lookup(hierarchy.type_seen[rows], pair_local, pair_type).astype(np.int64)
    sizes = hierarchy.type_ptr[pair_type + 1] - hierarchy.type_ptr[pair_type]
    lengths = np.minimum(per_cell + seen_counts, sizes)
    pair_id = np.repeat(np.arange(len(pair_type)), lengths)
    offsets = np.arange(lengths.sum()) - np.repeat(np.cumsum(lengths) - lengths, lengths)
    new_items = hierarchy.type_members[hierarchy.type_ptr[pair_type[pair_id]] + offsets]
    new_local = pair_local[pair_id]
    seen_block = state.seen[rows].tocoo()
    seen_keys = np.sort(_pair_keys(seen_block.row, seen_block.col, n_items))
    unseen = ~np.isin(_pair_keys(new_local, new_items, n_items), seen_keys, assume_unique=False)
    pair_id, new_local, new_items = pair_id[unseen], new_local[unseen], new_items[unseen]
    starts = np.flatnonzero(np.r_[True, pair_id[1:] != pair_id[:-1]]) if len(pair_id) else np.zeros(0, dtype=np.int64)
    rank = np.arange(len(pair_id)) - np.repeat(starts, np.diff(np.r_[starts, len(pair_id)]))
    keep = rank < per_cell
    local = np.concatenate([seen_block.row.astype(np.int64), new_local[keep]])
    items = np.concatenate([seen_block.col.astype(np.int64), new_items[keep]])
    seen_flag = np.concatenate([np.ones(seen_block.nnz, dtype=bool), np.zeros(int(keep.sum()), dtype=bool)])
    return local, items, seen_flag


def _per_user_mean(users: np.ndarray, values: np.ndarray) -> np.ndarray:
    """Mean of values per distinct user (users without rows are absent)."""
    unique, inverse = np.unique(users, return_inverse=True)
    return np.bincount(inverse, weights=values.astype(np.float64), minlength=len(unique)) / np.bincount(
        inverse, minlength=len(unique)
    )


@dataclass
class HierResult:
    lists: dict[str, list[str]]
    candidate_recall: float
    new_item_candidate_recall: float
    candidates_per_user: float


def recommend_hier(
    state: HierState,
    variant: str,
    params: Params,
    config: dict[str, object],
    rows: np.ndarray | None = None,
) -> HierResult:
    """Exact top-K among the candidates for each user row (target users by default), in user blocks."""
    settings = config["hierarchical"]
    top_k = int(config["evaluation"]["top_k"])
    hierarchy = state.hierarchies[VARIANTS[variant][0] if variant in VARIANTS else variant]
    override = VARIANTS.get(variant, (None, None))[1]
    params = params if override is None else Params(params.alpha, params.mu, params.lambda_type, int(override))
    rows = state.target_rows if rows is None else np.sort(np.asarray(rows, dtype=np.int64))
    block = int(settings.get("user_block", 1000))
    user_ids = state.interactions.users.to_numpy()
    item_ids = state.interactions.items.to_numpy()
    lists: dict[str, list[str]] = {}
    found_all = np.zeros(len(state.target_user), dtype=bool)
    evaluated = np.isin(state.target_user, rows)
    candidate_total = 0
    for start in range(0, len(rows), block):
        chunk = rows[start : start + block]
        theta, tau = stage_probabilities(state, hierarchy, params, chunk)
        joint = theta[:, hierarchy.type_stage1] * tau
        local, items, _ = generate_candidates(
            state, hierarchy, joint, chunk, int(settings["candidate_cells"]), int(settings["items_per_cell"])
        )
        candidate_total += len(items)
        scores = log_scores(state, hierarchy, params, chunk, theta, tau, local, items)
        order = np.lexsort((items, -scores, local))
        local_sorted, items_sorted = local[order], items[order]
        starts = np.searchsorted(local_sorted, np.arange(len(chunk)))
        ends = np.searchsorted(local_sorted, np.arange(len(chunk)), side="right")
        for position, row in enumerate(chunk):
            top = items_sorted[starts[position] : min(ends[position], starts[position] + top_k)]
            lists[str(user_ids[row])] = item_ids[top].astype(str).tolist()
        in_chunk = np.flatnonzero(np.isin(state.target_user, chunk))
        target_local = np.searchsorted(chunk, state.target_user[in_chunk])
        found_all[in_chunk] = np.isin(
            _pair_keys(target_local, state.target_item[in_chunk], state.n_items),
            _pair_keys(local, items, state.n_items),
        )
    recall = _per_user_mean(state.target_user[evaluated], found_all[evaluated])
    new = evaluated & state.target_new
    new_recall = _per_user_mean(state.target_user[new], found_all[new])
    return HierResult(
        lists=lists,
        candidate_recall=float(recall.mean()) if len(recall) else 0.0,
        new_item_candidate_recall=float(new_recall.mean()) if len(new_recall) else 0.0,
        candidates_per_user=candidate_total / max(len(rows), 1),
    )
