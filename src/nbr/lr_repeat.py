"""L2 logistic-regression repeat ranker over the features of nbr.features (W6, H6/H7).

Training rows are all (user, item) pairs of the fit history; the label is 1
when the pair occurs in the fit target (a repurchase). C is selected on a
user-level validation split by repeat Recall@10, the model is refit on all
training rows, and lists are produced by scoring the pairs of an (eval)
history. Scoring functions never read targets or files.
"""

from __future__ import annotations

import math
import time
from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from nbr.features import DEFAULT_DECAY_GAMMAS, feature_names, repeat_features
from nbr.metrics import per_user_recall
from nbr.repeat import PAIR_KEY, rank_scores

SOURCE = "repeat_lr"


def build_training(
    fit_history: pd.DataFrame,
    fit_target: pd.DataFrame,
    decay_gammas: Sequence[float] = DEFAULT_DECAY_GAMMAS,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """Features X (float64 columns, MultiIndex (user_id, item_id)), labels y (0/1), groups (user_id).

    X rows are exactly the (user, item) pairs of ``fit_history``; y[r] = 1 when
    the pair of row r also occurs in ``fit_target``. Targets only define y,
    never a feature. ``decay_gammas`` are the gammas of the dec_g{gamma} features (must contain 1.0).
    """
    features = repeat_features(fit_history, decay_gammas)
    keys = pd.MultiIndex.from_frame(features[PAIR_KEY])
    positives = pd.MultiIndex.from_frame(fit_target[PAIR_KEY].drop_duplicates())
    y = keys.isin(positives).astype("int64")
    X = features.set_index(keys)[feature_names(decay_gammas)]
    groups = features["user_id"].to_numpy()
    return X, y, groups


def split_users(users: np.ndarray, val_fraction: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Deterministic user-level split: (train_users, val_users), sorted.

    The sorted unique users are permuted with numpy.random.default_rng(seed)
    and round(n * val_fraction) of them (at least 1, at most n - 1) go to validation.
    """
    unique = np.array(sorted(set(users)))
    if len(unique) < 2:
        raise ValueError("split_users: need at least two users")
    if not 0.0 < val_fraction < 1.0:
        raise ValueError(f"split_users: val_fraction must lie in (0, 1), got {val_fraction}")
    n_val = min(max(int(round(len(unique) * val_fraction)), 1), len(unique) - 1)
    order = np.random.default_rng(seed).permutation(len(unique))
    val = np.sort(unique[order[:n_val]])
    train = np.sort(unique[order[n_val:]])
    return train, val


def select_c(table: dict[float, float]) -> float:
    """argmax of ``table`` (C -> validation Recall@10); numerically tied values -> smallest C."""
    best = max(table.values())
    return min(c for c, value in table.items() if math.isclose(value, best, rel_tol=1e-12, abs_tol=1e-12))


def _fit(X: pd.DataFrame, y: np.ndarray, C: float, max_iter: int, seed: int) -> tuple[StandardScaler, LogisticRegression]:
    """StandardScaler (zero mean, unit variance per column) + L2 LogisticRegression.

    LogisticRegression(penalty="l2", C=C, solver="lbfgs", max_iter=max_iter): C is
    the inverse regularisation strength, lbfgs is a deterministic quasi-Newton
    optimiser (random_state only matters for other solvers).
    """
    scaler = StandardScaler().fit(X.to_numpy(dtype="float64"))
    model = LogisticRegression(penalty="l2", C=float(C), solver="lbfgs", max_iter=int(max_iter), random_state=seed)
    model.fit(scaler.transform(X.to_numpy(dtype="float64")), y)
    return scaler, model


def score_frame(X: pd.DataFrame, scaler: StandardScaler, model: LogisticRegression) -> pd.DataFrame:
    """[user_id, item_id, score] for the rows of ``X``: ``model``.decision_function on ``scaler``-standardized features."""
    scores = model.decision_function(scaler.transform(X.to_numpy(dtype="float64")))
    frame = X.index.to_frame(index=False)[PAIR_KEY].copy()
    frame["score"] = scores.astype("float64")
    return frame


def validation_recall(X: pd.DataFrame, y: np.ndarray, scaler: StandardScaler, model: LogisticRegression, k: int) -> float:
    """Mean repeat Recall@``k`` of (``scaler``, ``model``) on the validation rows ``X`` with labels ``y``, over users with >= 1 positive row.

    Lists are the top-k rows per user by score with the contract tie rule
    (score desc, pop desc, item_id asc); pop is recovered from log_pop.
    """
    frame = score_frame(X, scaler, model)
    pop = pd.Series(np.expm1(X["log_pop"].to_numpy()), index=X.index.get_level_values("item_id"))
    pop = pop[~pop.index.duplicated()]
    lists = rank_scores(frame, pop, k, SOURCE)
    target = frame.loc[np.asarray(y) == 1, PAIR_KEY]
    return float(per_user_recall(lists, target, k).mean())


def fit_select(
    X: pd.DataFrame,
    y: np.ndarray,
    users: np.ndarray,
    C_grid: Sequence[float],
    val_fraction: float,
    seed: int,
    max_iter: int = 1000,
    k: int = 10,
) -> dict[str, Any]:
    """Select C from ``C_grid`` on a user-level split of ``X`` / ``y`` (``users`` per row; ``val_fraction``, ``seed`` as in
    split_users; validation repeat Recall@``k``), refit on all rows (``max_iter`` iterations), return the bundle.

    Returns a dict with keys: scaler, model (refit on all rows at C*), C,
    validation (list of rows with C, val_recall, n_iter, fit_seconds),
    coefficients (feature -> standardized coefficient), intercept, features,
    n_rows, positive_rate, n_train_users, n_val_users, timing_seconds, and
    (W8) train_scaler / train_model (the C* fit on the training users only),
    train_users, val_users (sorted arrays of the user split).
    """
    users = np.asarray(users)
    y = np.asarray(y, dtype="int64")
    if len(X) != len(y) or len(X) != len(users):
        raise ValueError("fit_select: X, y and users must have the same length")
    if y.sum() == 0 or y.sum() == len(y):
        raise ValueError("fit_select: both classes must be present")
    train_users, val_users = split_users(users, val_fraction, seed)
    # pandas Index.isin (hash based) instead of numpy.isin, which is quadratic on object (string) arrays.
    is_train = pd.Index(users).isin(train_users)
    X_train, y_train = X[is_train], y[is_train]
    X_val, y_val = X[~is_train], y[~is_train]

    timings: dict[str, float] = {}
    validation = []
    table = {}
    train_fits = {}
    t_sel = time.perf_counter()
    for C in C_grid:
        t0 = time.perf_counter()
        scaler, model = _fit(X_train, y_train, C, max_iter, seed)
        seconds = time.perf_counter() - t0
        recall = validation_recall(X_val, y_val, scaler, model, k)
        table[float(C)] = recall
        train_fits[float(C)] = (scaler, model)
        validation.append({"C": float(C), "val_recall": recall, "n_iter": int(model.n_iter_[0]),
                           "fit_seconds": seconds})
    C_star = select_c(table)
    timings["select_C"] = time.perf_counter() - t_sel
    train_scaler, train_model = train_fits[C_star]

    t0 = time.perf_counter()
    scaler, model = _fit(X, y, C_star, max_iter, seed)
    timings["refit"] = time.perf_counter() - t0
    features = list(X.columns)
    return {
        "scaler": scaler,
        "model": model,
        "train_scaler": train_scaler,
        "train_model": train_model,
        "train_users": train_users,
        "val_users": val_users,
        "C": C_star,
        "validation": validation,
        "coefficients": {name: float(value) for name, value in zip(features, model.coef_[0])},
        "intercept": float(model.intercept_[0]),
        "n_iter_refit": int(model.n_iter_[0]),
        "features": features,
        "n_rows": int(len(y)),
        "positive_rate": float(y.mean()),
        "n_train_users": int(len(train_users)),
        "n_val_users": int(len(val_users)),
        "val_fraction": float(val_fraction),
        "seed": int(seed),
        "max_iter": int(max_iter),
        "timing_seconds": timings,
    }


def lr_repeat_scores(history: pd.DataFrame, bundle: dict[str, Any]) -> pd.DataFrame:
    """Score every (user, item) pair of ``history`` with the fitted bundle -> [user_id, item_id, score]."""
    gammas = bundle.get("decay_gammas")
    if gammas is None:  # recover the gammas from the feature names "dec_g{gamma}"
        gammas = [float(name[len("dec_g"):]) for name in bundle["features"] if name.startswith("dec_g")]
    features = repeat_features(history, gammas)
    X = features.set_index(pd.MultiIndex.from_frame(features[PAIR_KEY]))[bundle["features"]]
    return score_frame(X, bundle["scaler"], bundle["model"])


def lr_repeat_list(
    history: pd.DataFrame, bundle: dict[str, Any], k: int, pop: pd.Series, users: Any = None
) -> pd.DataFrame:
    """Top-k repurchase candidates per user ranked by the logistic model of ``bundle``, ties by ``pop`` (source = "repeat_lr").

    Features are always computed on the whole ``history`` (item- and population-level
    statistics); with ``users`` the returned lists are restricted to those users.
    """
    scores = lr_repeat_scores(history, bundle)
    if users is not None:
        scores = scores[scores["user_id"].isin(pd.Index(users))]
    return rank_scores(scores, pop, k, SOURCE)


def train_only_bundle(bundle: dict[str, Any]) -> dict[str, Any]:
    """View of ``bundle`` that scores with the C* model fitted on the training users only (W8 slot selection)."""
    return {**bundle, "scaler": bundle["train_scaler"], "model": bundle["train_model"]}
