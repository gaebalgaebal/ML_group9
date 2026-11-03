"""Transferable logistic-ranker training and persistence."""

from __future__ import annotations

import hashlib
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .metrics import ranking_metrics
from .recommender import FEATURE_COLUMNS


def _negative_sample(frame: pd.DataFrame, ratio: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    pieces: list[pd.DataFrame] = []
    for _, group in frame.groupby("user_id", observed=True):
        positive = group[group["label"].eq(1)]
        negative = group[group["label"].eq(0)]
        if positive.empty:
            continue
        count = min(len(negative), max(len(positive) * ratio, 1))
        if count < len(negative):
            negative = negative.iloc[rng.choice(len(negative), size=count, replace=False)]
        pieces.extend([positive, negative])
    if not pieces:
        raise ValueError("No positive candidates were generated; cannot train the ranker")
    return pd.concat(pieces, ignore_index=True).sample(frac=1, random_state=seed).reset_index(drop=True)


def _new_model(alpha: float, max_iter: int, seed: int) -> Pipeline:
    return Pipeline(
        [
            ("scale", StandardScaler()),
            (
                "classifier",
                SGDClassifier(
                    loss="log_loss",
                    penalty="l2",
                    alpha=alpha,
                    class_weight="balanced",
                    max_iter=max_iter,
                    tol=1e-3,
                    random_state=seed,
                ),
            ),
        ]
    )


def train_ranker(
    frame: pd.DataFrame, target_sets: dict[str, set[str]], config: dict[str, object]
) -> tuple[dict[str, object], dict[str, float]]:
    """Select alpha on a user holdout, then refit on all sampled candidates."""
    seed = int(config["seed"])
    training = config["training"]
    users = frame[["user_id"]].drop_duplicates()
    splitter = GroupShuffleSplit(
        n_splits=1, test_size=float(training["validation_fraction"]), random_state=seed
    )
    train_user_idx, valid_user_idx = next(splitter.split(users, groups=users["user_id"]))
    train_users = set(users.iloc[train_user_idx]["user_id"].astype(str))
    valid_users = set(users.iloc[valid_user_idx]["user_id"].astype(str))
    train_frame = frame[frame["user_id"].astype(str).isin(train_users)]
    valid_frame = frame[frame["user_id"].astype(str).isin(valid_users)].copy()
    sampled_train = _negative_sample(train_frame, int(training["negative_ratio"]), seed)

    best_alpha = None
    best_ndcg = -1.0
    alpha_scores: dict[str, float] = {}
    valid_targets = {user: truth for user, truth in target_sets.items() if user in valid_users}
    for alpha in training["alphas"]:
        model = _new_model(float(alpha), int(training["max_iter"]), seed)
        model.fit(sampled_train[FEATURE_COLUMNS], sampled_train["label"])
        valid_frame["score_lr"] = model.predict_proba(valid_frame[FEATURE_COLUMNS])[:, 1]
        metrics = ranking_metrics(
            valid_frame,
            valid_targets,
            "score_lr",
            int(config["evaluation"]["top_k"]),
            catalog_size=valid_frame["item_id"].nunique(),
        )
        alpha_scores[str(alpha)] = metrics["ndcg_at_k"]
        if metrics["ndcg_at_k"] > best_ndcg:
            best_alpha, best_ndcg = float(alpha), metrics["ndcg_at_k"]

    sampled_all = _negative_sample(frame, int(training["negative_ratio"]), seed)
    final_model = _new_model(float(best_alpha), int(training["max_iter"]), seed)
    final_model.fit(sampled_all[FEATURE_COLUMNS], sampled_all["label"])
    artifact = {
        "model": final_model,
        "feature_columns": FEATURE_COLUMNS,
        "alpha": best_alpha,
        "seed": seed,
        "training_rows": len(sampled_all),
    }
    return artifact, alpha_scores


def save_model(artifact: dict[str, object], path: Path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, path)
    return file_checksum(path)


def load_model(path: Path) -> dict[str, object]:
    return joblib.load(path)


def file_checksum(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
