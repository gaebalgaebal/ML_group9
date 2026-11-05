"""Repeat-explore recommender: recency-weighted repeats, item-CF exploration, preference log-ratios.

All statistics are computed from one split's history; targets are only used for metrics.
Library module used by re_pipeline, health_subgroup and the hierarchical recommender.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import sparse

from .attributes import (
    attribute_column,
    item_departments,
    population_rates,
    preference_counts,
    smoothed_preferences,
)
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
    n_a: np.ndarray | None = None  # users x attributes, attribute purchases inside C_a
    recent_n_a: np.ndarray | None = None  # users x attributes, attribute purchases in recent baskets
    recent_n_c: np.ndarray | None = None
    early_n_a: np.ndarray | None = None
    early_n_c: np.ndarray | None = None
    late_n_a: np.ndarray | None = None
    late_n_c: np.ndarray | None = None

    def log_ratio(
        self, user_index: np.ndarray, item_index: np.ndarray, user_mask: np.ndarray | None = None
    ) -> np.ndarray:
        """l_u(i) = sum over a in A_i of the attribute log-likelihood ratio; 0 when A_i is empty.

        user_mask (users x attributes, bool) keeps only the terms of attributes enabled for the user.
        """
        if not self.attributes:
            return np.zeros(len(user_index))
        pi = self.pi[user_index]
        positive = np.log(pi / self.prior)
        negative = np.log((1.0 - pi) / (1.0 - self.prior))
        terms = np.where(self.item_flag[item_index] == 1, positive, negative)
        terms = terms * self.item_applicable[item_index]
        if user_mask is not None:
            terms = terms * user_mask[user_index]
        return terms.sum(axis=1)

    def preferring_users(self, min_n_c: int) -> np.ndarray:
        """users x attributes mask of pi(u, a) > pi_bar(a) and n_C(u, a) >= min_n_c."""
        return (self.pi > self.prior) & (self.n_c >= min_n_c)


def build_preferences(
    history: pd.DataFrame,
    items: pd.DataFrame,
    interactions: Interactions,
    categories: dict[str, set[str]],
    smoothing: float,
    recent_baskets: int = 5,
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
    n_a = np.zeros_like(pi)
    if len(counts):
        users = interactions.users.get_indexer(counts["user_id"].astype(str))
        columns = pd.Index(names).get_indexer(counts["attribute"])
        keep = users >= 0
        pi[users[keep], columns[keep]] = [
            preferences[(str(user), str(name))]
            for user, name in zip(counts["user_id"][keep], counts["attribute"][keep])
        ]
        n_c[users[keep], columns[keep]] = counts["n_c"].to_numpy()[keep]
        n_a[users[keep], columns[keep]] = counts["n_a"].to_numpy()[keep]
    pi = np.clip(pi, *clip)
    table = items.drop_duplicates("item_id").assign(item_id=lambda frame: frame["item_id"].astype(str))
    table = table.set_index("item_id").reindex(interactions.items)
    fine = table["fine_category"].astype("string").fillna("UNKNOWN")
    item_flag = np.zeros((len(interactions.items), len(names)), dtype=np.int8)
    item_applicable = np.zeros_like(item_flag, dtype=bool)
    for column, name in enumerate(names):
        item_flag[:, column] = table[attribute_column(name)].fillna(0).to_numpy(np.int8)
        item_applicable[:, column] = fine.isin(categories[name]).to_numpy()

    recent_n_a = np.zeros_like(pi)
    recent_n_c = np.zeros_like(pi)
    early_n_a = np.zeros_like(pi)
    early_n_c = np.zeros_like(pi)
    late_n_a = np.zeros_like(pi)
    late_n_c = np.zeros_like(pi)
    if names:
        annotated = history[["user_id", "basket_id", "basket_seq", "item_id"]].copy()
        annotated[["user_id", "basket_id", "item_id"]] = annotated[["user_id", "basket_id", "item_id"]].astype(str)
        annotated = annotated.drop_duplicates(["user_id", "basket_id", "item_id"])
        baskets = annotated[["user_id", "basket_id", "basket_seq"]].drop_duplicates()
        baskets = baskets.sort_values(
            ["user_id", "basket_seq", "basket_id"], ascending=[True, False, False], kind="mergesort"
        )
        baskets["recent_rank"] = baskets.groupby("user_id").cumcount()
        chronological = baskets.sort_values(
            ["user_id", "basket_seq", "basket_id"], ascending=[True, True, True], kind="mergesort"
        )
        chronological["basket_index"] = chronological.groupby("user_id").cumcount()
        chronological["basket_count"] = chronological.groupby("user_id")["basket_id"].transform("size")
        annotated = annotated.merge(
            baskets[["user_id", "basket_id", "recent_rank"]], on=["user_id", "basket_id"]
        ).merge(
            chronological[["user_id", "basket_id", "basket_index", "basket_count"]],
            on=["user_id", "basket_id"],
        )
        catalogue = items.drop_duplicates("item_id").assign(item_id=lambda frame: frame["item_id"].astype(str))
        annotated = annotated.merge(
            catalogue[["item_id", "fine_category"] + [attribute_column(name) for name in names]],
            on="item_id",
            how="left",
        )

        def store_counts(matrix: np.ndarray, selected: pd.DataFrame, column: int) -> None:
            counts_selected = selected.groupby("user_id").size()
            user_rows = interactions.users.get_indexer(counts_selected.index.astype(str))
            keep = user_rows >= 0
            matrix[user_rows[keep], column] = counts_selected.to_numpy()[keep]

        for column, name in enumerate(names):
            applicable = annotated["fine_category"].astype("string").isin(categories[name])
            attributed = annotated[attribute_column(name)].fillna(0).eq(1)
            recent = annotated["recent_rank"] < int(recent_baskets)
            early = 2 * annotated["basket_index"] < annotated["basket_count"]
            store_counts(recent_n_c, annotated[applicable & recent], column)
            store_counts(recent_n_a, annotated[applicable & attributed & recent], column)
            store_counts(early_n_c, annotated[applicable & early], column)
            store_counts(early_n_a, annotated[applicable & attributed & early], column)
            store_counts(late_n_c, annotated[applicable & ~early], column)
            store_counts(late_n_a, annotated[applicable & attributed & ~early], column)
    return PreferenceModel(
        names,
        prior,
        pi,
        n_c,
        item_flag,
        item_applicable,
        n_a=n_a,
        recent_n_a=recent_n_a,
        recent_n_c=recent_n_c,
        early_n_a=early_n_a,
        early_n_c=early_n_c,
        late_n_a=late_n_a,
        late_n_c=late_n_c,
    )


def seen_item_sets(seen: sparse.csr_matrix) -> list[set[int]]:
    """I_u as Python sets of item indices, one per user row."""
    return [set(seen.indices[seen.indptr[row] : seen.indptr[row + 1]].tolist()) for row in range(seen.shape[0])]


def popularity_order(popularity: np.ndarray) -> np.ndarray:
    """Item indices by descending history popularity (ties by index)."""
    return np.lexsort((np.arange(len(popularity)), -popularity))


def preference_candidates(
    seen_sets: list[set[int]],
    in_scope: np.ndarray,
    preferences: PreferenceModel,
    item_fine: np.ndarray,
    popularity: np.ndarray,
    settings: dict[str, object],
) -> pd.DataFrame:
    """Candidates (b): for each attribute with pi > pi_bar and n_C >= pref_min_n_c, the most popular
    unseen in-scope attribute items from the user's history aisles that belong to C_a.

    Independent of gamma and beta, so it can be computed once per split and scope.
    """
    per_attr = int(settings["pref_items_per_attr"])
    preferring = preferences.preferring_users(int(settings["pref_min_n_c"]))
    order = popularity_order(popularity)
    rows: list[tuple[int, int]] = []
    user_fine_sets: dict[int, set[str]] = {}
    for column, _ in enumerate(preferences.attributes):
        eligible = order[
            (preferences.item_flag[order, column] == 1) & preferences.item_applicable[order, column] & in_scope[order]
        ]
        by_fine: dict[str, list[int]] = {}
        for item in eligible:
            by_fine.setdefault(item_fine[item], []).append(int(item))
        for user in np.flatnonzero(preferring[:, column]):
            if user not in user_fine_sets:
                user_fine_sets[user] = {item_fine[item] for item in seen_sets[user]}
            seen_items = seen_sets[user]
            chosen: list[int] = []
            # Each aisle contributes at most per_attr unseen items, which is enough for an exact top per_attr.
            for fine in user_fine_sets[user].intersection(by_fine):
                taken = 0
                for item in by_fine[fine]:
                    if item in seen_items:
                        continue
                    chosen.append(item)
                    taken += 1
                    if taken == per_attr:
                        break
            chosen.sort(key=lambda item: (-popularity[item], item))
            rows.extend((int(user), item) for item in chosen[:per_attr])
    return pd.DataFrame(rows, columns=["user", "item"]).drop_duplicates().reset_index(drop=True)


def health_preference_candidates(
    seen_sets: list[set[int]],
    in_scope: np.ndarray,
    preferences: PreferenceModel,
    item_fine: np.ndarray,
    popularity: np.ndarray,
    settings: dict[str, object],
) -> pd.DataFrame:
    """Popular unseen health items from each eligible user's previously purchased aisles.

    Unlike ``preference_candidates``, the quota is applied per aisle before a per-user
    cap.  This prevents one high-volume aisle from consuming every health candidate.
    Eligibility and all popularity statistics come from history only.
    """
    if not bool(settings.get("candidate_enabled", True)):
        return pd.DataFrame(columns=["user", "item"])
    columns = [
        preferences.attributes.index(str(name))
        for name in settings.get("attributes", [])
        if str(name) in preferences.attributes
    ]
    if not columns:
        return pd.DataFrame(columns=["user", "item"])
    per_aisle = int(settings["candidate_items_per_aisle"])
    max_per_user = int(settings["candidate_max_per_user"])
    order = popularity_order(popularity)
    rows: list[tuple[int, int]] = []
    user_fines = [{item_fine[item] for item in seen} for seen in seen_sets]
    for column in columns:
        preferring = (
            (preferences.n_c[:, column] >= int(settings["min_n_c"]))
            & ((preferences.pi[:, column] - preferences.prior[column]) >= float(settings["min_lift"]))
        )
        eligible_items = order[
            (preferences.item_flag[order, column] == 1)
            & preferences.item_applicable[order, column]
            & in_scope[order]
        ]
        by_fine: dict[str, list[int]] = {}
        for item in eligible_items:
            by_fine.setdefault(str(item_fine[item]), []).append(int(item))
        for user in np.flatnonzero(preferring):
            for fine in sorted(user_fines[user].intersection(by_fine)):
                unseen = [item for item in by_fine[fine] if item not in seen_sets[user]][:per_aisle]
                rows.extend((int(user), item) for item in unseen)
    if not rows:
        return pd.DataFrame(columns=["user", "item"])
    frame = pd.DataFrame(rows, columns=["user", "item"]).drop_duplicates()
    frame["popularity"] = popularity[frame["item"].to_numpy(np.int64)]
    frame = frame.sort_values(
        ["user", "popularity", "item"], ascending=[True, False, True], kind="mergesort"
    )
    return frame[frame.groupby("user").cumcount() < max_per_user][["user", "item"]].reset_index(drop=True)


def explore_candidates(
    cf: sparse.csr_matrix,
    seen: sparse.csr_matrix,
    in_scope: np.ndarray,
    preferences: PreferenceModel,
    item_fine: np.ndarray,
    popularity: np.ndarray,
    settings: dict[str, object],
    top_k: int,
    preferred: pd.DataFrame | None = None,
    health_preferred: pd.DataFrame | None = None,
    seen_sets: list[set[int]] | None = None,
) -> pd.DataFrame:
    """E_u: CF neighbours (a), preferred-attribute items (b), popular fill-ins (c); all in F and not in I_u.

    (b) and (c) are always kept; (a) is truncated by s_cf to respect max_candidates, so the
    candidate set does not depend on beta and beta = 0 is a like-for-like CF-only comparison.
    """
    n_users = cf.shape[0]
    if seen_sets is None:
        seen_sets = seen_item_sets(seen)
    if preferred is None:
        preferred = preference_candidates(seen_sets, in_scope, preferences, item_fine, popularity, settings)
    if health_preferred is None:
        health_preferred = pd.DataFrame(columns=["user", "item"])
    coo = cf.tocoo()
    from_cf = pd.DataFrame({"user": coo.row, "item": coo.col, "s_cf": coo.data, "from_cf": True})
    candidates = from_cf.merge(preferred.assign(from_pref=True), on=["user", "item"], how="outer")
    candidates = candidates.merge(
        health_preferred.assign(from_health=True), on=["user", "item"], how="outer"
    )
    candidates["s_cf"] = candidates["s_cf"].fillna(0.0)
    candidates["from_cf"] = candidates["from_cf"].eq(True)
    candidates["from_pref"] = candidates["from_pref"].eq(True)
    candidates["from_health"] = candidates["from_health"].eq(True)
    candidates["from_popular"] = False

    counts = np.bincount(candidates["user"].to_numpy(np.int64), minlength=n_users)
    short_users = np.flatnonzero(counts < top_k)
    if len(short_users):
        popular = [int(item) for item in popularity_order(popularity) if in_scope[item]]
        existing = candidates[candidates["user"].isin(short_users)].groupby("user")["item"].apply(set).to_dict()
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
        fill = pd.DataFrame(fill_rows, columns=["user", "item"]).assign(
            s_cf=0.0, from_cf=False, from_pref=False, from_health=False, from_popular=True
        )
        candidates = pd.concat([candidates, fill], ignore_index=True)

    limit = int(settings["max_candidates"])
    candidates["keep_always"] = candidates["from_pref"] | candidates["from_health"] | candidates["from_popular"]
    candidates["health_only"] = candidates["from_health"] & ~(
        candidates["from_cf"] | candidates["from_pref"] | candidates["from_popular"]
    )
    extra = candidates.groupby("user")["health_only"].transform("sum").to_numpy(np.int64)
    candidates = candidates.sort_values(
        ["user", "keep_always", "s_cf", "item"], ascending=[True, False, False, True], kind="mergesort"
    )
    candidates = candidates[candidates.groupby("user").cumcount().to_numpy() < limit + extra[candidates.index]]
    candidates["ell"] = preferences.log_ratio(
        candidates["user"].to_numpy(np.int64), candidates["item"].to_numpy(np.int64)
    )
    return candidates.drop(columns=["keep_always", "health_only"]).reset_index(drop=True)


def explore_scores(candidates: pd.DataFrame, beta: float, epsilon: float) -> np.ndarray:
    """s_exp(u, i) = log(s_cf + epsilon) + beta * l_u(i)."""
    return np.log(candidates["s_cf"].to_numpy() + epsilon) + beta * candidates["ell"].to_numpy()


def scope_mask(items: pd.DataFrame, item_ids: pd.Index, mode: str, nonfood_departments: list[str]) -> np.ndarray:
    """F as a boolean mask over item_ids: every item for "all", non-food departments removed for "food"."""
    if mode == "all":
        return np.ones(len(item_ids), dtype=bool)
    if mode == "food":
        departments = item_departments(items).reindex(item_ids)
        return ~departments.isin(nonfood_departments).to_numpy()
    raise ValueError(f"Unknown scope mode: {mode!r}")


def scope_targets(
    target_sets: dict[str, set[str]], allowed: set[str] | None
) -> tuple[dict[str, set[str]], int]:
    """Restrict targets to F; users whose restricted target is empty are dropped and counted."""
    if allowed is None:
        return {user: set(truth) for user, truth in target_sets.items() if truth}, 0
    restricted = {user: truth & allowed for user, truth in target_sets.items()}
    kept = {user: truth for user, truth in restricted.items() if truth}
    return kept, len(restricted) - len(kept)


def repeat_counts(interactions: Interactions, in_scope: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """n(u): in-scope (basket, item) pairs from the user's second basket on; n_rep(u): those already bought."""
    frame = pd.DataFrame(
        {"user": interactions.user_index, "item": interactions.item_index, "seq": interactions.basket_seq}
    )
    first_basket = frame.groupby("user")["seq"].transform("min").to_numpy()
    first_purchase = frame.groupby(["user", "item"])["seq"].transform("min").to_numpy()
    seq = frame["seq"].to_numpy()
    later = (seq > first_basket) & in_scope[frame["item"].to_numpy()]
    repeated = later & (seq > first_purchase)
    n_users = interactions.shape[0]
    users = frame["user"].to_numpy()
    return (
        np.bincount(users[later], minlength=n_users).astype(np.float64),
        np.bincount(users[repeated], minlength=n_users).astype(np.float64),
    )


def repeat_ratio(n: np.ndarray, n_rep: np.ndarray, kappa: float) -> tuple[np.ndarray, float]:
    """rho_u = (n_rep(u) + kappa * rho_bar) / (n(u) + kappa) with rho_bar pooled over the same history."""
    rho_bar = float(n_rep.sum() / n.sum()) if n.sum() else 0.0
    denominator = n + kappa
    rho = np.divide(n_rep + kappa * rho_bar, denominator, out=np.full_like(n, rho_bar), where=denominator > 0)
    return rho, rho_bar


def repeat_ranking(repeat: sparse.csr_matrix, in_scope: np.ndarray) -> pd.DataFrame:
    """I_u intersected with F, ranked by s_rep (ties by item index)."""
    coo = repeat.tocoo()
    frame = pd.DataFrame({"user": coo.row, "item": coo.col, "score": coo.data})
    frame = frame[in_scope[frame["item"].to_numpy()]]
    frame = frame.sort_values(["user", "score", "item"], ascending=[True, False, True], kind="mergesort")
    frame["rank"] = frame.groupby("user").cumcount()
    return frame.reset_index(drop=True)


def explore_ranking(candidates: pd.DataFrame, scores: np.ndarray) -> pd.DataFrame:
    """E_u ranked by s_exp (ties by item index)."""
    frame = candidates.assign(score=scores)
    frame = frame.sort_values(["user", "score", "item"], ascending=[True, False, True], kind="mergesort")
    frame["rank"] = frame.groupby("user").cumcount()
    return frame.reset_index(drop=True)


def health_preference_scores(
    user_index: np.ndarray,
    item_index: np.ndarray,
    preferences: PreferenceModel,
    settings: dict[str, object],
    require_recent: bool = False,
    persistence_mode: str = "none",
) -> np.ndarray:
    """Reliability-weighted positive preference lift for configured health attributes."""
    configured = [str(name) for name in settings.get("attributes", [])]
    columns = [preferences.attributes.index(name) for name in configured if name in preferences.attributes]
    if not columns or len(user_index) == 0:
        return np.zeros(len(user_index), dtype=np.float64)

    prior = preferences.prior[columns]
    lift = preferences.pi[:, columns] - prior
    n_c = preferences.n_c[:, columns]
    min_n_c = float(settings["min_n_c"])
    eligible = (n_c >= min_n_c) & (lift >= float(settings["min_lift"]))
    if require_recent and persistence_mode == "none":
        persistence_mode = "recent_1"
    if persistence_mode == "recent_1":
        if preferences.recent_n_a is None:
            return np.zeros(len(user_index), dtype=np.float64)
        eligible &= preferences.recent_n_a[:, columns] >= int(settings["recent_min_purchases"])
    elif persistence_mode == "recent_2":
        if preferences.recent_n_a is None:
            return np.zeros(len(user_index), dtype=np.float64)
        eligible &= preferences.recent_n_a[:, columns] >= int(settings["recent_strict_min_purchases"])
    elif persistence_mode == "recent_rate":
        if preferences.recent_n_a is None or preferences.recent_n_c is None:
            return np.zeros(len(user_index), dtype=np.float64)
        recent_rate = np.divide(
            preferences.recent_n_a[:, columns],
            preferences.recent_n_c[:, columns],
            out=np.zeros_like(preferences.recent_n_a[:, columns]),
            where=preferences.recent_n_c[:, columns] > 0,
        )
        eligible &= preferences.recent_n_c[:, columns] >= int(settings["recent_rate_min_n_c"])
        eligible &= recent_rate > prior
    elif persistence_mode == "both_halves":
        matrices = [preferences.early_n_a, preferences.early_n_c, preferences.late_n_a, preferences.late_n_c]
        if any(matrix is None for matrix in matrices):
            return np.zeros(len(user_index), dtype=np.float64)
        early_rate = np.divide(
            preferences.early_n_a[:, columns],
            preferences.early_n_c[:, columns],
            out=np.zeros_like(preferences.early_n_a[:, columns]),
            where=preferences.early_n_c[:, columns] > 0,
        )
        late_rate = np.divide(
            preferences.late_n_a[:, columns],
            preferences.late_n_c[:, columns],
            out=np.zeros_like(preferences.late_n_a[:, columns]),
            where=preferences.late_n_c[:, columns] > 0,
        )
        minimum = int(settings["half_min_n_c"])
        eligible &= (preferences.early_n_c[:, columns] >= minimum) & (preferences.late_n_c[:, columns] >= minimum)
        eligible &= (early_rate > prior) & (late_rate > prior)
    elif persistence_mode != "none":
        raise ValueError(f"Unknown health persistence mode: {persistence_mode!r}")
    # Reliability is 0.5 at the evidence threshold and approaches 1 with more history.
    reliability = n_c / (n_c + min_n_c)
    user_weight = np.where(eligible, np.maximum(lift, 0.0) * reliability, 0.0)
    item_match = (
        preferences.item_flag[:, columns].astype(bool)
        & preferences.item_applicable[:, columns]
    )
    return (user_weight[user_index] * item_match[item_index]).sum(axis=1)


def health_soft_explore_ranking(
    candidates: pd.DataFrame,
    scores: np.ndarray,
    preferences: PreferenceModel,
    settings: dict[str, object],
    health_lambda: float,
    require_recent: bool = False,
    persistence_mode: str = "none",
) -> pd.DataFrame:
    """Add a soft health-affinity term to Item-CF scores without reserving slots.

    ``adjusted_score = base_score + health_lambda * health_score``.  The health score
    uses only configured attributes and history-derived preference statistics.  A zero
    lambda, absent attributes, or ineligible users reproduces the base ranking exactly.
    """
    weight = max(0.0, float(health_lambda))
    users = candidates["user"].to_numpy(np.int64)
    items = candidates["item"].to_numpy(np.int64)
    health = health_preference_scores(
        users,
        items,
        preferences,
        settings,
        require_recent=require_recent,
        persistence_mode=persistence_mode,
    )
    adjusted = np.asarray(scores, dtype=np.float64) + weight * health
    ranked = explore_ranking(candidates, adjusted)
    # Re-align from candidate rows after explore_ranking's sort/reset.
    lookup = candidates[["user", "item"]].assign(
        base_score=np.asarray(scores, dtype=np.float64), health_score=health
    )
    ranked = ranked.merge(lookup, on=["user", "item"], how="left")
    ranked["health_boosted"] = ranked["health_score"].gt(0) & (weight > 0)
    ranked["health_promoted"] = False
    return ranked.sort_values(["user", "rank"], kind="mergesort").reset_index(drop=True)


def health_gated_explore_ranking(
    candidates: pd.DataFrame,
    scores: np.ndarray,
    preferences: PreferenceModel,
    settings: dict[str, object],
    health_slots: int,
) -> pd.DataFrame:
    """Promote at most ``health_slots`` validated-health candidates per user.

    A candidate is eligible only when it has one of the configured attributes and the
    user's history for that attribute has both enough applicable purchases and a
    smoothed preference above the population rate by ``min_lift``.  Promoted products
    are ordered by a reliability-weighted preference lift; all remaining products keep
    their original Item-CF order.  With zero slots, unavailable attributes, or no
    eligible candidates this is exactly :func:`explore_ranking`.

    This function only changes the exploration ranking.  Repeat scores and repeat-slot
    allocation are deliberately outside its scope.
    """
    base = explore_ranking(candidates, scores).assign(health_score=0.0, health_promoted=False)
    slots = max(0, int(health_slots))
    if slots == 0 or base.empty:
        return base

    users = base["user"].to_numpy(np.int64)
    items = base["item"].to_numpy(np.int64)
    base["health_score"] = health_preference_scores(users, items, preferences, settings)
    eligible_rows = base[base["health_score"].gt(0)].sort_values(
        ["user", "health_score", "score", "item"],
        ascending=[True, False, False, True],
        kind="mergesort",
    )
    promoted_keys = eligible_rows.groupby("user", sort=False).head(slots)[["user", "item"]]
    if promoted_keys.empty:
        return base

    marked = base.merge(promoted_keys.assign(health_promoted=True), on=["user", "item"], how="left", suffixes=("", "_new"))
    marked["health_promoted"] = marked["health_promoted_new"].eq(True)
    marked = marked.drop(columns="health_promoted_new")
    promoted = marked[marked["health_promoted"]].sort_values(
        ["user", "health_score", "score", "item"], ascending=[True, False, False, True], kind="mergesort"
    )
    promoted["health_order"] = promoted.groupby("user").cumcount()
    promoted["health_priority"] = 0
    remainder = marked[~marked["health_promoted"]].copy()
    remainder["health_order"] = remainder["rank"]
    remainder["health_priority"] = 1
    ranked = pd.concat([promoted, remainder], ignore_index=True)
    ranked = ranked.sort_values(
        ["user", "health_priority", "health_order", "item"], kind="mergesort"
    ).drop(columns=["health_priority", "health_order"])
    ranked["rank"] = ranked.groupby("user").cumcount()
    return ranked.reset_index(drop=True)


def slot_counts(rho: np.ndarray, top_k: int, available: np.ndarray) -> np.ndarray:
    """r_u = round(K * rho_u), clipped to [0, K] and to |I_u intersected with F|."""
    slots = np.clip(np.floor(top_k * rho + 0.5), 0, top_k).astype(np.int64)
    return np.minimum(slots, available.astype(np.int64))


def allocate_slots(
    repeat_ranked: pd.DataFrame,
    explore_ranked: pd.DataFrame,
    repeat_slots: np.ndarray,
    top_k: int,
) -> pd.DataFrame:
    """Top r_u repeats followed by top K - r_u explores; a short side is filled from the other.

    Explore rows already in the repeat ranking are dropped first, so lists have no duplicates.
    """
    n_users = len(repeat_slots)
    explore = explore_ranked.merge(
        repeat_ranked[["user", "item"]], on=["user", "item"], how="left", indicator=True
    )
    explore = explore[explore["_merge"].eq("left_only")].drop(columns="_merge")
    explore = explore.sort_values(["user", "rank"], kind="mergesort")
    explore["rank"] = explore.groupby("user").cumcount()
    repeat_available = np.bincount(repeat_ranked["user"].to_numpy(np.int64), minlength=n_users)
    explore_available = np.bincount(explore["user"].to_numpy(np.int64), minlength=n_users)
    repeat_take = np.minimum(repeat_slots, repeat_available)
    explore_take = np.minimum(top_k - repeat_take, explore_available)
    repeat_take = np.minimum(repeat_available, top_k - explore_take)
    repeat_users = repeat_ranked["user"].to_numpy(np.int64)
    explore_users = explore["user"].to_numpy(np.int64)
    chosen_repeat = repeat_ranked[repeat_ranked["rank"].to_numpy() < repeat_take[repeat_users]]
    chosen_explore = explore[explore["rank"].to_numpy() < explore_take[explore_users]]
    result = pd.concat(
        [
            chosen_repeat.assign(slot="repeat", position=chosen_repeat["rank"]),
            chosen_explore.assign(
                slot="explore",
                position=chosen_explore["rank"] + repeat_take[chosen_explore["user"].to_numpy(np.int64)],
            ),
        ],
        ignore_index=True,
    )
    return result.sort_values(["user", "position"], kind="mergesort").reset_index(drop=True)


def to_lists(
    ranked: pd.DataFrame, users: pd.Index, items: pd.Index, order: str = "position"
) -> dict[str, list[str]]:
    """Per-user item-id lists sorted by the given column."""
    frame = ranked.sort_values(["user", order], kind="mergesort")
    item_ids = items.to_numpy()[frame["item"].to_numpy(np.int64)]
    user_ids = users.to_numpy()[frame["user"].to_numpy(np.int64)]
    return {str(user): list(group) for user, group in pd.Series(item_ids).groupby(user_ids, sort=False)}
