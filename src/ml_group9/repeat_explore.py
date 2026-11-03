"""Repeat-explore recommender: recency-weighted repeats, item-CF exploration, preference log-ratios.

All statistics are computed from one split's history; targets are only used for metrics.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import sparse

from .attributes import attribute_column, population_rates, preference_counts, smoothed_preferences
from .recommender import item_cosine_neighbors


PROBABILITY_CLIP = 1e-6


@dataclass
class Interactions:
    """History rows deduplicated by (user, basket, item) and mapped to matrix indices."""

    users: pd.Index
    items: pd.Index
    user_index: np.ndarray
    item_index: np.ndarray
    basket_seq: np.ndarray
    exponent: np.ndarray  # t_u - t_b; the user's latest basket has exponent 0

    @property
    def shape(self) -> tuple[int, int]:
        return len(self.users), len(self.items)


def build_interactions(history: pd.DataFrame, items: pd.DataFrame) -> Interactions:
    """Index history purchases of catalogue items; users are those with at least one such purchase."""
    item_ids = pd.Index(items["item_id"].astype(str).drop_duplicates().to_numpy())
    implicit = history[["user_id", "basket_id", "basket_seq", "item_id"]].copy()
    for column in ["user_id", "basket_id", "item_id"]:
        implicit[column] = implicit[column].astype(str)
    implicit = implicit.drop_duplicates(["user_id", "basket_id", "item_id"])
    implicit = implicit[implicit["item_id"].isin(item_ids)]
    user_ids = pd.Index(np.sort(implicit["user_id"].unique()))
    latest = implicit.groupby("user_id")["basket_seq"].transform("max")
    return Interactions(
        users=user_ids,
        items=item_ids,
        user_index=user_ids.get_indexer(implicit["user_id"]),
        item_index=item_ids.get_indexer(implicit["item_id"]),
        basket_seq=implicit["basket_seq"].to_numpy(np.int64),
        exponent=(latest - implicit["basket_seq"]).to_numpy(np.int64),
    )


def item_fine_categories(items: pd.DataFrame, interactions: Interactions) -> np.ndarray:
    """fine_category per interaction item ("UNKNOWN" when missing)."""
    table = items.drop_duplicates("item_id").assign(item_id=lambda frame: frame["item_id"].astype(str))
    fine = table.set_index("item_id")["fine_category"].reindex(interactions.items)
    return fine.astype("string").fillna("UNKNOWN").astype(str).to_numpy(dtype=object)


def item_popularity(interactions: Interactions) -> np.ndarray:
    """Number of history baskets containing each item."""
    return np.bincount(interactions.item_index, minlength=interactions.shape[1]).astype(np.float64)


def seen_matrix(interactions: Interactions) -> sparse.csr_matrix:
    """Binary user x item matrix of I_u."""
    matrix = sparse.csr_matrix(
        (np.ones(len(interactions.user_index)), (interactions.user_index, interactions.item_index)),
        shape=interactions.shape,
    )
    matrix.data[:] = 1.0
    return matrix


def repeat_scores(interactions: Interactions, gamma: float) -> sparse.csr_matrix:
    """s_rep(u, i) = sum over history baskets b of gamma^(t_u - t_b) * 1[i in b]."""
    weights = np.power(float(gamma), interactions.exponent.astype(np.float64))
    # Duplicate (user, item) coordinates are summed when converting to CSR.
    return sparse.csr_matrix((weights, (interactions.user_index, interactions.item_index)), shape=interactions.shape)


def similarity_matrix(
    history: pd.DataFrame, item_ids: pd.Index, top_k: int, block_size: int
) -> sparse.csr_matrix:
    """Item x item matrix with Sim[i, j] = cosine(i, j) for j among i's top-k neighbours."""
    neighbors = item_cosine_neighbors(history, top_k=top_k, block_size=block_size)
    rows: list[int] = []
    cols: list[int] = []
    values: list[float] = []
    sources = item_ids.get_indexer(list(neighbors))
    for source, pairs in zip(sources, neighbors.values()):
        if source < 0 or not pairs:
            continue
        targets = item_ids.get_indexer([item for item, _ in pairs])
        for target, (_, score) in zip(targets, pairs):
            if target >= 0:
                rows.append(int(source))
                cols.append(int(target))
                values.append(score)
    return sparse.csr_matrix((values, (rows, cols)), shape=(len(item_ids), len(item_ids)))


def cf_scores(
    repeat: sparse.csr_matrix, similarity: sparse.csr_matrix, seen: sparse.csr_matrix, in_scope: np.ndarray
) -> sparse.csr_matrix:
    """s_cf = S_rep @ Sim^T, zeroed on I_u and outside the scope F."""
    scores = (repeat @ similarity.T).tocsr()
    scores = scores - scores.multiply(seen)
    scores = (scores @ sparse.diags(in_scope.astype(np.float64))).tocsr()
    scores.eliminate_zeros()
    return scores


@dataclass
class PreferenceModel:
    """Smoothed per-user attribute preferences pi(u, a) aligned to interaction indices."""

    attributes: list[str]
    prior: np.ndarray  # pi_bar(a), clipped
    pi: np.ndarray  # users x attributes, clipped; users without C_a purchases get pi_bar
    n_c: np.ndarray  # users x attributes
    item_flag: np.ndarray  # items x attributes, attr_a(i)
    item_applicable: np.ndarray  # items x attributes, fine_category(i) in C_a

    def log_ratio(self, user_index: np.ndarray, item_index: np.ndarray) -> np.ndarray:
        """l_u(i) = sum over a in A_i of the attribute log-likelihood ratio; 0 when A_i is empty."""
        if not self.attributes:
            return np.zeros(len(user_index))
        pi = self.pi[user_index]
        positive = np.log(pi / self.prior)
        negative = np.log((1.0 - pi) / (1.0 - self.prior))
        terms = np.where(self.item_flag[item_index] == 1, positive, negative)
        return (terms * self.item_applicable[item_index]).sum(axis=1)

    def preferring_users(self, min_n_c: int) -> np.ndarray:
        """users x attributes mask of pi(u, a) > pi_bar(a) and n_C(u, a) >= min_n_c."""
        return (self.pi > self.prior) & (self.n_c >= min_n_c)


def build_preferences(
    history: pd.DataFrame,
    items: pd.DataFrame,
    interactions: Interactions,
    categories: dict[str, set[str]],
    smoothing: float,
) -> PreferenceModel:
    """Estimate pi(u, a) and pi_bar(a) from history purchases inside C_a."""
    names = list(categories)
    counts = preference_counts(history, items, categories)
    rates = population_rates(counts, names)
    preferences = smoothed_preferences(counts, rates, smoothing)
    clip = (PROBABILITY_CLIP, 1.0 - PROBABILITY_CLIP)
    prior = np.clip(np.array([rates[name] for name in names], dtype=np.float64), *clip)
    pi = np.tile(prior, (len(interactions.users), 1))
    n_c = np.zeros_like(pi)
    if len(counts):
        users = interactions.users.get_indexer(counts["user_id"].astype(str))
        columns = pd.Index(names).get_indexer(counts["attribute"])
        keep = users >= 0
        pi[users[keep], columns[keep]] = [
            preferences[(str(user), str(name))]
            for user, name in zip(counts["user_id"][keep], counts["attribute"][keep])
        ]
        n_c[users[keep], columns[keep]] = counts["n_c"].to_numpy()[keep]
    pi = np.clip(pi, *clip)
    table = items.drop_duplicates("item_id").assign(item_id=lambda frame: frame["item_id"].astype(str))
    table = table.set_index("item_id").reindex(interactions.items)
    fine = table["fine_category"].astype("string").fillna("UNKNOWN")
    item_flag = np.zeros((len(interactions.items), len(names)), dtype=np.int8)
    item_applicable = np.zeros_like(item_flag, dtype=bool)
    for column, name in enumerate(names):
        item_flag[:, column] = table[attribute_column(name)].fillna(0).to_numpy(np.int8)
        item_applicable[:, column] = fine.isin(categories[name]).to_numpy()
    return PreferenceModel(names, prior, pi, n_c, item_flag, item_applicable)


def explore_candidates(
    cf: sparse.csr_matrix,
    seen: sparse.csr_matrix,
    in_scope: np.ndarray,
    preferences: PreferenceModel,
    item_fine: np.ndarray,
    popularity: np.ndarray,
    settings: dict[str, object],
    top_k: int,
) -> pd.DataFrame:
    """E_u: CF neighbours (a), preferred-attribute items (b), popular fill-ins (c); all in F and not in I_u.

    (b) and (c) are always kept; (a) is truncated by s_cf to respect max_candidates, so the
    candidate set does not depend on beta and beta = 0 is a like-for-like CF-only comparison.
    """
    n_users = cf.shape[0]
    coo = cf.tocoo()
    from_cf = pd.DataFrame({"user": coo.row, "item": coo.col, "s_cf": coo.data})

    seen_sets = [set(seen.indices[seen.indptr[row] : seen.indptr[row + 1]]) for row in range(n_users)]
    per_attr = int(settings["pref_items_per_attr"])
    preferring = preferences.preferring_users(int(settings["pref_min_n_c"]))
    pref_rows: list[tuple[int, int]] = []
    user_fine_sets: dict[int, set[str]] = {}
    order = np.lexsort((np.arange(len(popularity)), -popularity))
    for column, _ in enumerate(preferences.attributes):
        eligible = order[
            (preferences.item_flag[order, column] == 1)
            & preferences.item_applicable[order, column]
            & in_scope[order]
        ]
        by_fine: dict[str, list[int]] = {}
        for item in eligible:
            by_fine.setdefault(item_fine[item], []).append(int(item))
        applicable_fines = set(item_fine[preferences.item_applicable[:, column]])
        for user in np.flatnonzero(preferring[:, column]):
            if user not in user_fine_sets:
                user_fine_sets[user] = {item_fine[item] for item in seen_sets[user]}
            user_fines = user_fine_sets[user] & applicable_fines
            chosen: list[int] = []
            for fine in user_fines:
                taken = 0
                for item in by_fine.get(fine, []):
                    if item in seen_sets[user]:
                        continue
                    chosen.append(item)
                    taken += 1
                    if taken == per_attr:
                        break
            chosen.sort(key=lambda item: (-popularity[item], item))
            pref_rows.extend((int(user), item) for item in chosen[:per_attr])
    from_pref = pd.DataFrame(pref_rows, columns=["user", "item"]).drop_duplicates()
    from_pref["from_pref"] = True

    candidates = from_cf.merge(from_pref, on=["user", "item"], how="outer")
    candidates["s_cf"] = candidates["s_cf"].fillna(0.0)
    candidates["from_pref"] = candidates["from_pref"].eq(True)
    candidates["from_popular"] = False

    counts = np.bincount(candidates["user"].to_numpy(np.int64), minlength=n_users)
    short_users = np.flatnonzero(counts < top_k)
    if len(short_users):
        popular = [int(item) for item in order if in_scope[item]]
        existing = candidates.groupby("user")["item"].apply(set).to_dict()
        fill_rows: list[tuple[int, int]] = []
        for user in short_users:
            have = existing.get(user, set())
            need = top_k - len(have)
            for item in popular:
                if need == 0:
                    break
                if item in have or item in seen_sets[user]:
                    continue
                fill_rows.append((int(user), item))
                need -= 1
        fill = pd.DataFrame(fill_rows, columns=["user", "item"]).assign(s_cf=0.0, from_pref=False, from_popular=True)
        candidates = pd.concat([candidates, fill], ignore_index=True)

    limit = int(settings["max_candidates"])
    candidates["keep_always"] = candidates["from_pref"] | candidates["from_popular"]
    candidates = candidates.sort_values(
        ["user", "keep_always", "s_cf", "item"], ascending=[True, False, False, True], kind="mergesort"
    )
    candidates = candidates[candidates.groupby("user").cumcount() < limit]
    candidates["ell"] = preferences.log_ratio(
        candidates["user"].to_numpy(np.int64), candidates["item"].to_numpy(np.int64)
    )
    return candidates.drop(columns="keep_always").reset_index(drop=True)


def explore_scores(candidates: pd.DataFrame, beta: float, epsilon: float) -> np.ndarray:
    """s_exp(u, i) = log(s_cf + epsilon) + beta * l_u(i)."""
    return np.log(candidates["s_cf"].to_numpy() + epsilon) + beta * candidates["ell"].to_numpy()
