"""Basket-level Top-K evaluation metrics."""

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
