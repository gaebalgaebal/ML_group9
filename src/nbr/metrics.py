# *------------------- 2026 Fall Semester Machine Learning Term-project: Team 9 --------------------*
# Code First written at 2026-11-05
# Code Final Edited at 2026-11-13
# Editor: Yejun Kim

'''
Purpose : Per-user evaluation metrics and a paired bootstrap over users.
Inputs  : Recommendation lists (columns user_id, item_id, rank), target frames (columns
          user_id, item_id), a category ranking (columns user_id, category, rank) and two
          per-user metric Series for the bootstrap.
Outputs : All per_user_* functions return a float64 Series indexed by the users that
          have at least one row in the target frame (sorted by user_id).
          paired_bootstrap returns a dict with n_users, mean_a, mean_b, diff and one CI per level.
Usage   :
    from nbr.metrics import paired_bootstrap, per_user_ndcg, per_user_recall
    stage3[r] = float(per_user_ndcg(final, target, K).mean())
    result = paired_bootstrap(a, b, int(boot["n"]), int(boot["seed"]), levels)
'''

from __future__ import annotations

# ---------------------------------------------------------------------------
# imports
# ---------------------------------------------------------------------------
from typing import Any

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# constants
# ---------------------------------------------------------------------------
KEY_COLUMNS = ["user_id", "item_id"]


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _target_sets(target: pd.DataFrame) -> pd.Series:
    """Map user_id -> set of target item_id (duplicates removed), sorted by user."""
    # Drop repeated (user, item) rows of the target.
    deduped = target[KEY_COLUMNS].drop_duplicates()
    # Collect each user's target items into a set.
    return deduped.groupby("user_id", sort=True)["item_id"].agg(set)


def _check_k(k: int) -> None:
    '''Raise ValueError unless ``k`` is a positive integer (bool is rejected).'''
    if isinstance(k, bool) or not isinstance(k, (int, np.integer)) or k < 1:
        raise ValueError(f"k must be a positive integer, got {k!r}")


def _topk_lists(lists: pd.DataFrame, k: int) -> pd.Series:
    """Map user_id -> list of item_id in rank order (ranks 1..k only)."""
    # Validate the cutoff.
    _check_k(k)
    # Keep ranks 1..k and order the rows by user and rank.
    top = lists[lists["rank"] <= k].sort_values(["user_id", "rank"])
    # Collect each user's items into a list in rank order.
    return top.groupby("user_id", sort=True)["item_id"].agg(list)


# ---------------------------------------------------------------------------
# per-user metrics
# ---------------------------------------------------------------------------
def per_user_recall(lists: pd.DataFrame, target: pd.DataFrame, k: int) -> pd.Series:
    """Recall@k = |L_u(k) & T_u| / |T_u| per user of ``target``, L_u(k) = top-k rows of ``lists``; users without a list score 0."""
    # Build the target set and the top-k list of every user.
    targets = _target_sets(target)
    ranked = _topk_lists(lists, k)
    values = []
    for user, items in targets.items():
        # Count the listed items that are in the user's target set.
        hits = sum(1 for item in ranked.get(user, []) if item in items)
        # Divide the hits by the size of the target set.
        values.append(hits / len(items))
    # Return one value per target user under the name "recall".
    return pd.Series(values, index=targets.index, dtype="float64", name="recall")


def per_user_ndcg(lists: pd.DataFrame, target: pd.DataFrame, k: int) -> pd.Series:
    """NDCG@k of ``lists`` against ``target`` with binary relevance; IDCG uses min(|T_u|, k) ideal hits."""
    # Build the target set and the top-k list of every user.
    targets = _target_sets(target)
    ranked = _topk_lists(lists, k)
    # Precompute the discount of every list position.
    discounts = 1.0 / np.log2(np.arange(2, k + 2))  # position r (1-based) -> 1/log2(r+1)
    values = []
    for user, items in targets.items():
        # Sum the discounts of the positions that hold a target item.
        dcg = sum(
            discounts[pos] for pos, item in enumerate(ranked.get(user, [])) if item in items
        )
        # Sum the discounts of the best possible list for this user.
        idcg = float(discounts[: min(len(items), k)].sum())
        # Normalise the discounted gain by the ideal one.
        values.append(dcg / idcg)
    # Return one value per target user under the name "ndcg".
    return pd.Series(values, index=targets.index, dtype="float64", name="ndcg")


def per_user_category_hit(
    category_rank: pd.DataFrame, new_target: pd.DataFrame, k: int, level: str = "L2"
) -> pd.Series:
    """Share of a user's new target rows whose ``level`` category is in the top-k categories.

    ``category_rank`` has columns [user_id, category, rank]; ``new_target`` must
    carry the ``level`` column. (user_id, item_id) duplicates are removed.
    """
    # Validate the cutoff.
    _check_k(k)
    # Drop repeated (user, item) rows of the target.
    deduped = new_target.drop_duplicates(subset=KEY_COLUMNS)
    # Collect each user's top-k categories into a set.
    top = category_rank[category_rank["rank"] <= k]
    top_sets = top.groupby("user_id", sort=True)["category"].agg(set)
    users = []
    values = []
    for user, group in deduped.groupby("user_id", sort=True):
        # Use an empty set for a user without a category ranking.
        cats = top_sets.get(user, set())
        # Count the target rows whose category, read as a string, is in the set.
        hit = int(group[level].astype(str).isin(cats).sum())
        users.append(user)
        # Divide the hits by the number of target rows of the user.
        values.append(hit / len(group))
    # Return one value per target user under the name "category_hit".
    index = pd.Index(users, name="user_id")
    return pd.Series(values, index=index, dtype="float64", name="category_hit")


# ---------------------------------------------------------------------------
# paired bootstrap
# ---------------------------------------------------------------------------
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
    # Require the same set of users in both series.
    if set(a.index) != set(b.index):
        raise ValueError("paired_bootstrap: a and b must have the same user index set")
    # Require at least one user and no repeated user in either series.
    if len(a) == 0 or not a.index.is_unique or not b.index.is_unique:
        raise ValueError("paired_bootstrap: series must be non-empty with unique user index")
    # Require at least one resample.
    if n < 1:
        raise ValueError(f"paired_bootstrap: n must be >= 1, got {n}")
    # Require every confidence level to lie strictly between 0 and 1.
    if any(not 0.0 < level < 1.0 for level in levels):
        raise ValueError(f"paired_bootstrap: levels must lie in (0, 1), got {levels}")
    # Align both series on the sorted user index.
    a = a.sort_index()
    b = b.reindex(a.index)
    # Compute the per-user difference b - a.
    diff = b.to_numpy(dtype="float64") - a.to_numpy(dtype="float64")
    # Reject missing or infinite differences.
    if not np.isfinite(diff).all():
        raise ValueError("paired_bootstrap: a and b must be finite")
    n_users = len(diff)
    # Create the seeded random generator and the array of resample means.
    rng = np.random.default_rng(seed)
    means = np.empty(n, dtype="float64")
    done = 0
    while done < n:
        # Draw the user indices of the next chunk of resamples.
        size = min(chunk_size, n - done)
        idx = rng.integers(0, n_users, size=(size, n_users))
        # Store the mean difference of each resample in the chunk.
        means[done : done + size] = diff[idx].mean(axis=1)
        done += size
    ci = {}
    for level in levels:
        # Take the two-sided percentile bounds of the resample means for this level.
        alpha = (1.0 - level) / 2.0
        lo, hi = np.percentile(means, [100 * alpha, 100 * (1 - alpha)])
        ci[level] = [float(lo), float(hi)]
    # Return the user count, both means, the observed difference and the intervals.
    return {
        "n_users": int(n_users),
        "mean_a": float(a.mean()),
        "mean_b": float(b.mean()),
        "diff": float(diff.mean()),
        "ci": ci,
    }
