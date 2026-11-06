"""Per-user evaluation metrics and a paired bootstrap over users.

All per_user_* functions return a float64 Series indexed by the users that
have at least one row in the target frame (sorted by user_id).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

KEY_COLUMNS = ["user_id", "item_id"]


def _target_sets(target: pd.DataFrame) -> pd.Series:
    """Map user_id -> set of target item_id (duplicates removed), sorted by user."""
    deduped = target[KEY_COLUMNS].drop_duplicates()
    return deduped.groupby("user_id", sort=True)["item_id"].agg(set)


def _check_k(k: int) -> None:
    if isinstance(k, bool) or not isinstance(k, (int, np.integer)) or k < 1:
        raise ValueError(f"k must be a positive integer, got {k!r}")


def _topk_lists(lists: pd.DataFrame, k: int) -> pd.Series:
    """Map user_id -> list of item_id in rank order (ranks 1..k only)."""
    _check_k(k)
    top = lists[lists["rank"] <= k].sort_values(["user_id", "rank"])
    return top.groupby("user_id", sort=True)["item_id"].agg(list)


def per_user_recall(lists: pd.DataFrame, target: pd.DataFrame, k: int) -> pd.Series:
    """Recall@k = |L_u(k) & T_u| / |T_u| per user of ``target``, L_u(k) = top-k rows of ``lists``; users without a list score 0."""
    targets = _target_sets(target)
    ranked = _topk_lists(lists, k)
    values = []
    for user, items in targets.items():
        hits = sum(1 for item in ranked.get(user, []) if item in items)
        values.append(hits / len(items))
    return pd.Series(values, index=targets.index, dtype="float64", name="recall")


def per_user_ndcg(lists: pd.DataFrame, target: pd.DataFrame, k: int) -> pd.Series:
    """NDCG@k of ``lists`` against ``target`` with binary relevance; IDCG uses min(|T_u|, k) ideal hits."""
    targets = _target_sets(target)
    ranked = _topk_lists(lists, k)
    discounts = 1.0 / np.log2(np.arange(2, k + 2))  # position r (1-based) -> 1/log2(r+1)
    values = []
    for user, items in targets.items():
        dcg = sum(
            discounts[pos] for pos, item in enumerate(ranked.get(user, [])) if item in items
        )
        idcg = float(discounts[: min(len(items), k)].sum())
        values.append(dcg / idcg)
    return pd.Series(values, index=targets.index, dtype="float64", name="ndcg")


def per_user_category_hit(
    category_rank: pd.DataFrame, new_target: pd.DataFrame, k: int, level: str = "L2"
) -> pd.Series:
    """Share of a user's new target rows whose ``level`` category is in the top-k categories.

    ``category_rank`` has columns [user_id, category, rank]; ``new_target`` must
    carry the ``level`` column. (user_id, item_id) duplicates are removed.
    """
    _check_k(k)
    deduped = new_target.drop_duplicates(subset=KEY_COLUMNS)
    top = category_rank[category_rank["rank"] <= k]
    top_sets = top.groupby("user_id", sort=True)["category"].agg(set)
    users = []
    values = []
    for user, group in deduped.groupby("user_id", sort=True):
        cats = top_sets.get(user, set())
        hit = int(group[level].astype(str).isin(cats).sum())
        users.append(user)
        values.append(hit / len(group))
    index = pd.Index(users, name="user_id")
    return pd.Series(values, index=index, dtype="float64", name="category_hit")


def paired_bootstrap(
    a: pd.Series,
    b: pd.Series,
    n: int,
    seed: int,
    levels: list[float],
    chunk_size: int = 200,
) -> dict[str, Any]:
    """Percentile bootstrap CI of mean(b) - mean(a) over users: ``n`` resamples with replacement, one CI per level in ``levels``.

    ``a`` and ``b`` must share the same index set. Resamples are drawn in
    chunks of ``chunk_size`` replicates to bound memory. Uses
    numpy.random.default_rng(seed).integers(low, high, size) to draw user
    indices and numpy.percentile(values, q) for the CI bounds.
    """
    if set(a.index) != set(b.index):
        raise ValueError("paired_bootstrap: a and b must have the same user index set")
    if len(a) == 0 or not a.index.is_unique or not b.index.is_unique:
        raise ValueError("paired_bootstrap: series must be non-empty with unique user index")
    if n < 1:
        raise ValueError(f"paired_bootstrap: n must be >= 1, got {n}")
    if any(not 0.0 < level < 1.0 for level in levels):
        raise ValueError(f"paired_bootstrap: levels must lie in (0, 1), got {levels}")
    a = a.sort_index()
    b = b.reindex(a.index)
    diff = b.to_numpy(dtype="float64") - a.to_numpy(dtype="float64")
    if not np.isfinite(diff).all():
        raise ValueError("paired_bootstrap: a and b must be finite")
    n_users = len(diff)
    rng = np.random.default_rng(seed)
    means = np.empty(n, dtype="float64")
    done = 0
    while done < n:
        size = min(chunk_size, n - done)
        idx = rng.integers(0, n_users, size=(size, n_users))
        means[done : done + size] = diff[idx].mean(axis=1)
        done += size
    ci = {}
    for level in levels:
        alpha = (1.0 - level) / 2.0
        lo, hi = np.percentile(means, [100 * alpha, 100 * (1 - alpha)])
        ci[level] = [float(lo), float(hi)]
    return {
        "n_users": int(n_users),
        "mean_a": float(a.mean()),
        "mean_b": float(b.mean()),
        "diff": float(diff.mean()),
        "ci": ci,
    }
