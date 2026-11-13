# *------------------- 2026 Fall Semester Machine Learning Term-project: Team 9 --------------------*
# Code First written at 2026-11-05
# Code Final Edited at 2026-11-13
# Editor: Yejun Kim, Bonjun Gu

'''
Purpose : L2 logistic-regression repeat ranker (R2) over the features of nbr.features
          (W6, H6/H7). C is selected on a user-level validation split by repeat Recall@10,
          the model is refit on all training rows, and lists are produced by scoring the
          pairs of an (eval) history. Scoring functions never read targets or files.
          H6 compares R2 with the decayed purchase frequency ranker (R1) on repeat
          Recall@10; H7 compares the combination of R2 with the hierarchical explore list
          (C1) with the combination of R1 with the same explore list (C0) on NDCG@10.
          PREREG v1.5 (R2', the R2 ranker plus the item-CF similarity feature): with
          ``icf_neighbors`` (config key lr_repeat.itemcf_feature, default false) the item-CF
          similarity feature of nbr.features_icf is appended as a 13th column, computed on
          the same history as the other features (fit history for training rows, eval
          history for lists); the bundle records ``icf_neighbors``. Without it every
          function behaves exactly as before.
Inputs  : Training rows are all (user, item) pairs of the fit history; the label is 1
          when the pair occurs in the fit target (a repurchase). Also the grid of C, the
          validation fraction, the seed and, for lists, a history and a popularity Series.
Outputs : build_training returns (X, y, groups); fit_select returns the fitted bundle (dict);
          lr_repeat_scores returns [user_id, item_id, score]; lr_repeat_list returns the
          top-k list frame built by nbr.repeat.rank_scores.
Usage   :
    from nbr.lr_repeat import build_training, lr_repeat_list
    X, y, groups = build_training(history, target, gammas, icf_neighbors=icf_neighbors)
    repeat_lr = lr_repeat_list(history, lr_bundle, K, pop)
'''

from __future__ import annotations

# ---------------------------------------------------------------------------
# imports
# ---------------------------------------------------------------------------
import math
import time
from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from nbr.features import DEFAULT_DECAY_GAMMAS, feature_names, repeat_features
from nbr.features_icf import ICF_COLUMN, icf_feature
from nbr.metrics import per_user_recall
from nbr.repeat import PAIR_KEY, rank_scores

# ---------------------------------------------------------------------------
# constants
# ---------------------------------------------------------------------------
SOURCE = "repeat_lr"
SOURCE_ICF = "repeat_lri"


# ---------------------------------------------------------------------------
# item-CF feature
# ---------------------------------------------------------------------------
def add_icf(
    features: pd.DataFrame, history: pd.DataFrame, n_neighbors: int, neighbors: pd.DataFrame | None = None
) -> pd.DataFrame:
    """``features`` (one row per (user, item) pair of ``history``, nbr.features.repeat_features) with the column "icf"
    of nbr.features_icf.icf_feature(history, n_neighbors) appended (``neighbors``: precomputed neighbour frame)."""
    # Compute the item-CF feature of every (user, item) pair.
    icf = icf_feature(history, n_neighbors, neighbors=neighbors)
    # Join it to the feature rows one to one.
    out = features.merge(icf, on=PAIR_KEY, how="left", validate="one_to_one")
    # Reject any feature row without an item-CF value.
    if out[ICF_COLUMN].isna().any():
        raise ValueError("add_icf: every feature row needs an icf value")
    return out


# ---------------------------------------------------------------------------
# training data
# ---------------------------------------------------------------------------
def build_training(
    fit_history: pd.DataFrame,
    fit_target: pd.DataFrame,
    decay_gammas: Sequence[float] = DEFAULT_DECAY_GAMMAS,
    icf_neighbors: int | None = None,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """Features X (float64 columns, MultiIndex (user_id, item_id)), labels y (0/1), groups (user_id).

    X rows are exactly the (user, item) pairs of ``fit_history``; y[r] = 1 when
    the pair of row r also occurs in ``fit_target``. Targets only define y,
    never a feature. ``decay_gammas`` are the gammas of the dec_g{gamma} features (must contain 1.0).
    With ``icf_neighbors`` (PREREG v1.5, R2') the column "icf" of nbr.features_icf computed on
    ``fit_history`` with that neighbour count is appended as the last feature.
    """
    # Compute the feature rows and the feature column names.
    features = repeat_features(fit_history, decay_gammas)
    columns = feature_names(decay_gammas)
    # Append the item-CF feature when a neighbour count is given.
    if icf_neighbors is not None:
        features = add_icf(features, fit_history, int(icf_neighbors))
        columns = columns + [ICF_COLUMN]
    # Index the rows by (user, item) and collect the distinct target pairs.
    keys = pd.MultiIndex.from_frame(features[PAIR_KEY])
    positives = pd.MultiIndex.from_frame(fit_target[PAIR_KEY].drop_duplicates())
    # Label a row 1 when its pair is in the target.
    y = keys.isin(positives).astype("int64")
    # Build the feature matrix and the user of every row.
    X = features.set_index(keys)[columns]
    groups = features["user_id"].to_numpy()
    return X, y, groups


def split_users(users: np.ndarray, val_fraction: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Deterministic user-level split: (train_users, val_users), sorted.

    The sorted unique users are permuted with numpy.random.default_rng(seed)
    and round(n * val_fraction) of them (at least 1, at most n - 1) go to validation.
    """
    # Sort the distinct users.
    unique = np.array(sorted(set(users)))
    # Require at least two users.
    if len(unique) < 2:
        raise ValueError("split_users: need at least two users")
    # Require a validation fraction strictly between 0 and 1.
    if not 0.0 < val_fraction < 1.0:
        raise ValueError(f"split_users: val_fraction must lie in (0, 1), got {val_fraction}")
    # Round the validation size and clip it to 1..n-1.
    n_val = min(max(int(round(len(unique) * val_fraction)), 1), len(unique) - 1)
    # Permute the users with the seeded generator.
    order = np.random.default_rng(seed).permutation(len(unique))
    # Take the first part as validation users and the rest as training users, both sorted.
    val = np.sort(unique[order[:n_val]])
    train = np.sort(unique[order[n_val:]])
    return train, val


def select_c(table: dict[float, float]) -> float:
    """argmax of ``table`` (C -> validation Recall@10); numerically tied values -> smallest C."""
    # Find the best validation value.
    best = max(table.values())
    # Return the smallest C whose value is numerically equal to it.
    return min(c for c, value in table.items() if math.isclose(value, best, rel_tol=1e-12, abs_tol=1e-12))


# ---------------------------------------------------------------------------
# model fitting and scoring
# ---------------------------------------------------------------------------
def _fit(X: pd.DataFrame, y: np.ndarray, C: float, max_iter: int, seed: int) -> tuple[StandardScaler, LogisticRegression]:
    """StandardScaler (zero mean, unit variance per column) + L2 LogisticRegression.

    LogisticRegression(penalty="l2", C=C, solver="lbfgs", max_iter=max_iter): C is
    the inverse regularisation strength, lbfgs is a deterministic quasi-Newton
    optimiser (random_state only matters for other solvers).
    """
    # Fit the scaler on the feature matrix.
    scaler = StandardScaler().fit(X.to_numpy(dtype="float64"))
    # Create the logistic model and fit it on the standardized features.
    model = LogisticRegression(penalty="l2", C=float(C), solver="lbfgs", max_iter=int(max_iter), random_state=seed)
    model.fit(scaler.transform(X.to_numpy(dtype="float64")), y)
    return scaler, model


def score_frame(X: pd.DataFrame, scaler: StandardScaler, model: LogisticRegression) -> pd.DataFrame:
    """[user_id, item_id, score] for the rows of ``X``: ``model``.decision_function on ``scaler``-standardized features."""
    # Compute the decision values on the standardized features.
    scores = model.decision_function(scaler.transform(X.to_numpy(dtype="float64")))
    # Return them next to the (user, item) keys of the rows.
    frame = X.index.to_frame(index=False)[PAIR_KEY].copy()
    frame["score"] = scores.astype("float64")
    return frame


def validation_recall(X: pd.DataFrame, y: np.ndarray, scaler: StandardScaler, model: LogisticRegression, k: int) -> float:
    """Mean repeat Recall@``k`` of (``scaler``, ``model``) on the validation rows ``X`` with labels ``y``, over users with >= 1 positive row.

    Lists are the top-k rows per user by score with the contract tie rule
    (score desc, pop desc, item_id asc); pop is recovered from log_pop.
    """
    # Score the validation rows.
    frame = score_frame(X, scaler, model)
    # Recover the item popularity from log_pop, one value per item.
    pop = pd.Series(np.expm1(X["log_pop"].to_numpy()), index=X.index.get_level_values("item_id"))
    pop = pop[~pop.index.duplicated()]
    # Rank the scores into top-k lists.
    lists = rank_scores(frame, pop, k, SOURCE)
    # Use the rows with label 1 as the target.
    target = frame.loc[np.asarray(y) == 1, PAIR_KEY]
    # Average the per-user recall.
    return float(per_user_recall(lists, target, k).mean())


# ---------------------------------------------------------------------------
# selection of C
# ---------------------------------------------------------------------------
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
    # Convert the users and the labels to arrays.
    users = np.asarray(users)
    y = np.asarray(y, dtype="int64")
    # Require the same number of rows in X, y and users.
    if len(X) != len(y) or len(X) != len(users):
        raise ValueError("fit_select: X, y and users must have the same length")
    # Require both classes in the labels.
    if y.sum() == 0 or y.sum() == len(y):
        raise ValueError("fit_select: both classes must be present")
    # Split the users into training and validation users.
    train_users, val_users = split_users(users, val_fraction, seed)
    # pandas Index.isin (hash based) instead of numpy.isin, which is quadratic on object (string) arrays.
    is_train = pd.Index(users).isin(train_users)
    # Split the rows by the user split.
    X_train, y_train = X[is_train], y[is_train]
    X_val, y_val = X[~is_train], y[~is_train]

    # Create the timing record and the per-C results.
    timings: dict[str, float] = {}
    validation = []
    table = {}
    train_fits = {}
    t_sel = time.perf_counter()
    for C in C_grid:
        # Fit on the training rows and time the fit.
        t0 = time.perf_counter()
        scaler, model = _fit(X_train, y_train, C, max_iter, seed)
        seconds = time.perf_counter() - t0
        # Compute the validation recall of this C.
        recall = validation_recall(X_val, y_val, scaler, model, k)
        # Store the recall, the fitted pair and the validation row.
        table[float(C)] = recall
        train_fits[float(C)] = (scaler, model)
        validation.append({"C": float(C), "val_recall": recall, "n_iter": int(model.n_iter_[0]),
                           "fit_seconds": seconds})
    # Select C and keep the training-only fit of the selected C.
    C_star = select_c(table)
    timings["select_C"] = time.perf_counter() - t_sel
    train_scaler, train_model = train_fits[C_star]

    # Refit on all rows at the selected C and time it.
    t0 = time.perf_counter()
    scaler, model = _fit(X, y, C_star, max_iter, seed)
    timings["refit"] = time.perf_counter() - t0
    # Return the bundle with both fits, the user split, the coefficients and the settings.
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


# ---------------------------------------------------------------------------
# scoring and lists
# ---------------------------------------------------------------------------
def lr_repeat_scores(
    history: pd.DataFrame, bundle: dict[str, Any], neighbors: pd.DataFrame | None = None
) -> pd.DataFrame:
    """Score every (user, item) pair of ``history`` with the fitted bundle -> [user_id, item_id, score].

    A bundle with ``icf_neighbors`` (R2') gets the "icf" column computed on ``history`` with that neighbour
    count (``neighbors``: a precomputed item_neighbors frame of ``history`` to share with the item-CF lists).
    """
    # Read the decay gammas of the bundle.
    gammas = bundle.get("decay_gammas")
    if gammas is None:  # recover the gammas from the feature names "dec_g{gamma}"
        gammas = [float(name[len("dec_g"):]) for name in bundle["features"] if name.startswith("dec_g")]
    # Compute the features of the history.
    features = repeat_features(history, gammas)
    # Append the item-CF feature when the bundle records a neighbour count.
    if bundle.get("icf_neighbors") is not None:
        features = add_icf(features, history, int(bundle["icf_neighbors"]), neighbors=neighbors)
    # Order the columns as in the bundle and score the rows.
    X = features.set_index(pd.MultiIndex.from_frame(features[PAIR_KEY]))[bundle["features"]]
    return score_frame(X, bundle["scaler"], bundle["model"])


def lr_repeat_list(
    history: pd.DataFrame, bundle: dict[str, Any], k: int, pop: pd.Series, users: Any = None,
    source: str = SOURCE, neighbors: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Top-k repurchase candidates per user ranked by the logistic model of ``bundle``, ties by ``pop`` (``source`` = "repeat_lr").

    Features are always computed on the whole ``history`` (item- and population-level
    statistics); with ``users`` the returned lists are restricted to those users. ``neighbors`` is
    passed to lr_repeat_scores (R2' bundles only).
    """
    # Score every (user, item) pair of the history.
    scores = lr_repeat_scores(history, bundle, neighbors=neighbors)
    # Keep the requested users when given.
    if users is not None:
        scores = scores[scores["user_id"].isin(pd.Index(users))]
    # Rank the scores into top-k lists with the popularity tie rule.
    return rank_scores(scores, pop, k, source)


def train_only_bundle(bundle: dict[str, Any]) -> dict[str, Any]:
    """View of ``bundle`` that scores with the C* model fitted on the training users only (W8 slot selection)."""
    return {**bundle, "scaler": bundle["train_scaler"], "model": bundle["train_model"]}
