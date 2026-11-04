"""Basket-level Top-K evaluation metrics.

Library module: ranking and list metrics, candidate and subset recall, and paired bootstrap CIs.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd


def ranking_metrics(
    frame: pd.DataFrame,
    target_sets: dict[str, set[str]],
    score_column: str,
    top_k: int,
    catalog_size: int,
) -> dict[str, float]:
    """Compute macro Top-K metrics and catalog coverage."""
    per_user: list[tuple[float, float, float, float]] = []
    recommended: set[str] = set()
    for user, group in frame.groupby("user_id", observed=True):
        truth = target_sets.get(str(user), set())
        ranked = group.sort_values([score_column, "item_id"], ascending=[False, True]).head(top_k)
        items = ranked["item_id"].astype(str).tolist()
        recommended.update(items)
        hits = [int(item in truth) for item in items]
        hit_count = sum(hits)
        precision = hit_count / top_k
        recall = hit_count / len(truth) if truth else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        dcg = sum(hit / math.log2(rank + 2) for rank, hit in enumerate(hits))
        ideal = sum(1 / math.log2(rank + 2) for rank in range(min(len(truth), top_k)))
        ndcg = dcg / ideal if ideal else 0.0
        per_user.append((precision, recall, f1, ndcg))
    values = np.asarray(per_user, dtype=float)
    if values.size == 0:
        values = np.zeros((1, 4), dtype=float)
    return {
        "precision_at_k": float(values[:, 0].mean()),
        "recall_at_k": float(values[:, 1].mean()),
        "f1_at_k": float(values[:, 2].mean()),
        "ndcg_at_k": float(values[:, 3].mean()),
        "catalog_coverage": len(recommended) / max(catalog_size, 1),
        "users_evaluated": int(len(per_user)),
    }


def candidate_recall(frame: pd.DataFrame, target_sets: dict[str, set[str]]) -> float:
    """Macro fraction of true target items present in each candidate set."""
    candidate_map = frame.groupby("user_id", observed=True)["item_id"].apply(lambda s: set(s.astype(str))).to_dict()
    values = []
    for user, truth in target_sets.items():
        if truth:
            values.append(len(candidate_map.get(user, set()) & truth) / len(truth))
    return float(np.mean(values)) if values else 0.0


def list_metrics(
    lists: dict[str, list[str]], targets: dict[str, set[str]], top_k: int
) -> tuple[dict[str, float], pd.DataFrame]:
    """Macro Top-K metrics from explicit per-user lists; users without a list score zero.

    Returns the summary and the per-user values (indexed by user) for paired comparisons.
    """
    rows: list[tuple[str, float, float, float, float]] = []
    for user, truth in targets.items():
        items = lists.get(user, [])[:top_k]
        hits = [int(item in truth) for item in items]
        hit_count = sum(hits)
        precision = hit_count / top_k
        recall = hit_count / len(truth)
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        dcg = sum(hit / math.log2(rank + 2) for rank, hit in enumerate(hits))
        ideal = sum(1 / math.log2(rank + 2) for rank in range(min(len(truth), top_k)))
        rows.append((user, precision, recall, f1, dcg / ideal))
    per_user = pd.DataFrame(
        rows, columns=["user_id", "precision_at_k", "recall_at_k", "f1_at_k", "ndcg_at_k"]
    ).set_index("user_id")
    summary: dict[str, float] = {
        column: float(per_user[column].mean()) if len(per_user) else 0.0 for column in per_user.columns
    }
    summary["users_evaluated"] = int(len(per_user))
    return summary, per_user


def subset_recall(
    lists: dict[str, list[str]], relevant: dict[str, set[str]], top_k: int
) -> tuple[float, pd.Series]:
    """Macro recall of each top-K list against a per-user relevant subset; empty subsets are skipped."""
    values = {
        user: len(set(lists.get(user, [])[:top_k]) & items) / len(items)
        for user, items in relevant.items()
        if items
    }
    series = pd.Series(values, dtype=float)
    return (float(series.mean()) if len(series) else 0.0), series


def paired_bootstrap(
    first: pd.Series, second: pd.Series, resamples: int, seed: int, chunk: int = 100
) -> dict[str, float]:
    """Mean of first - second over shared users with a 95% percentile CI from user resampling."""
    shared = first.index.intersection(second.index)
    differences = (first.loc[shared] - second.loc[shared]).to_numpy(dtype=float)
    if len(differences) == 0:
        return {"users": 0, "resamples": resamples, "mean_difference": 0.0, "ci_low": 0.0, "ci_high": 0.0}
    rng = np.random.default_rng(seed)
    means: list[np.ndarray] = []
    for start in range(0, resamples, chunk):
        draws = rng.integers(0, len(differences), size=(min(chunk, resamples - start), len(differences)))
        means.append(differences[draws].mean(axis=1))
    low, high = np.percentile(np.concatenate(means), [2.5, 97.5])
    return {
        "users": int(len(differences)),
        "resamples": int(resamples),
        "mean_difference": float(differences.mean()),
        "ci_low": float(low),
        "ci_high": float(high),
    }
