# *------------------- 2026 Fall Semester Machine Learning Term-project: Team 9 --------------------*
# Code First written at 2026-11-06
# Code Final Edited at 2026-11-13
# Editor: Yejun Kim, Bonjun Gu

'''
Purpose : Binned purchase-cycle features on top of nbr.features.repeat_features (W9), the
          features of the logistic-regression repeat ranker with binned cycle features (R3).
          Every row is a (user_id, item_id) pair of the history; the 12 columns of the
          logistic-regression repeat ranker (R2) from ``repeat_features`` are kept and the
          following are appended:
            item_cadence    median mean_gap of the users who bought the item in >= 2 baskets
                            (over the whole history); items without such a buyer get the
                            global median of those mean_gap values (1.0 if there is none)
            cadence_ratio   recency / item_cadence, capped at CADENCE_RATIO_CAP (10)
            ob_<lo>_<hi>    one-hot bins of overdue (right-open [lo, hi)); the first bin is
                            the reference and has no column; pairs with once == 1 have every
                            ob_<lo>_<hi> = 0 and the separate column ob_once = 1
            rb_<lo>_<hi>    one-hot bins of recency, first bin = reference (no column)
            cb_<lo>_<hi>    one-hot bins of cadence_ratio, first bin = reference (no column)
Inputs  : A history DataFrame (the columns required by ``repeat_features``), a config mapping
          with the bin edges (configs/r3.yaml, key "r3") and a sequence of decay factors.
          Only the history frame is read; targets are never seen.
Outputs : cycle_features returns [user_id, item_id] + cycle_feature_names(cfg) as float64,
          one row per history pair. Values outside [edges[0], edges[-1]) raise ValueError
          instead of being clipped silently.
Usage   :
    from nbr.features_cycle import cycle_features
    features = cycle_features(fit_history, cfg, decay_gammas)
'''

from __future__ import annotations

# ---------------------------------------------------------------------------
# imports
# ---------------------------------------------------------------------------
from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd

from nbr.features import DEFAULT_DECAY_GAMMAS, feature_names, repeat_features
from nbr.repeat import PAIR_KEY

# ---------------------------------------------------------------------------
# constants
# ---------------------------------------------------------------------------
CADENCE_RATIO_CAP = 10.0
ONCE_COLUMN = "ob_once"
# (column prefix, config key, source column) of the three dummy groups, in output order.
BIN_GROUPS = (
    ("ob", "overdue_bins", "overdue"),
    ("rb", "recency_bins", "recency"),
    ("cb", "item_cadence_ratio_bins", "cadence_ratio"),
)


# ---------------------------------------------------------------------------
# bin edges, labels and indices
# ---------------------------------------------------------------------------
def _fmt(value: float) -> str:
    """Edge label without a trailing ".0": 1.0 -> "1", 0.5 -> "0.5", 10.01 -> "10.01"."""
    # Convert the value to a float.
    value = float(value)
    # Print whole numbers as integers and other values with repr.
    return str(int(value)) if value.is_integer() else repr(value)


def bin_edges(cfg: dict[str, Any], key: str) -> list[float]:
    """Strictly increasing float edges of ``cfg[key]`` (at least two edges)."""
    # Read the edges of the config key as floats.
    edges = [float(e) for e in cfg[key]]
    # Reject fewer than two edges and edges that do not increase strictly.
    if len(edges) < 2 or any(b <= a for a, b in zip(edges[:-1], edges[1:])):
        raise ValueError(f"{key}: edges must be strictly increasing with >= 2 values, got {edges}")
    return edges


def bin_labels(edges: Sequence[float]) -> list[str]:
    """Interval labels "[lo, hi)" of every bin of ``edges``, including the reference bin."""
    return [f"[{_fmt(lo)}, {_fmt(hi)})" for lo, hi in zip(edges[:-1], edges[1:])]


def bin_columns(prefix: str, edges: Sequence[float]) -> list[str]:
    """Dummy column names "<prefix>_<lo>_<hi>" of every bin of ``edges`` except the first (reference)."""
    return [f"{prefix}_{_fmt(lo)}_{_fmt(hi)}" for lo, hi in zip(edges[1:-1], edges[2:])]


def bin_index(values: np.ndarray, edges: Sequence[float]) -> np.ndarray:
    """0-based bin of each value for right-open bins [edges[j], edges[j+1]).

    Uses numpy.searchsorted(edges, values, side="right") (index of the first
    edge strictly greater than the value) minus one. Values below edges[0] or
    at/above edges[-1] raise ValueError.
    """
    # Convert the values and the edges to float64 arrays.
    values = np.asarray(values, dtype="float64")
    edges = np.asarray(edges, dtype="float64")
    # Reject NaN and infinite values.
    if not np.isfinite(values).all():
        raise ValueError("bin_index: values must be finite")
    # Reject values outside the range covered by the edges.
    if (values < edges[0]).any() or (values >= edges[-1]).any():
        raise ValueError(f"bin_index: values outside [{edges[0]}, {edges[-1]})")
    # Return the index of the last edge that is not greater than each value.
    return np.searchsorted(edges, values, side="right") - 1


# ---------------------------------------------------------------------------
# feature column names
# ---------------------------------------------------------------------------
def cycle_feature_names(cfg: dict[str, Any], decay_gammas: Sequence[float] = DEFAULT_DECAY_GAMMAS) -> list[str]:
    """Feature columns of ``cycle_features`` in output order: R2 columns for ``decay_gammas`` first, then the bins of ``cfg``."""
    # Start with the repeat-feature columns and the two cadence columns.
    names = feature_names(decay_gammas) + ["item_cadence", "cadence_ratio"]
    for prefix, key, _ in BIN_GROUPS:
        # Append the dummy columns of the group.
        names += bin_columns(prefix, bin_edges(cfg, key))
        # Append the single-purchase column after the overdue dummies.
        if prefix == "ob":
            names.append(ONCE_COLUMN)
    return names


# ---------------------------------------------------------------------------
# item cadence
# ---------------------------------------------------------------------------
def item_cadence(features: pd.DataFrame) -> pd.Series:
    """item_id -> median mean_gap over the pairs of ``features`` (repeat_features) with once == 0 (users who repeated the item).

    Items without a repeating buyer are absent from the result; callers fill
    them with the global median returned by ``global_cadence``.
    """
    # Keep the pairs bought in more than one basket.
    repeated = features[features["once"] == 0.0]
    # Take the median mean_gap of each item.
    return repeated.groupby("item_id")["mean_gap"].median()


def global_cadence(features: pd.DataFrame) -> float:
    """Global median of mean_gap over the pairs of ``features`` with once == 0 (1.0 if no pair repeated)."""
    # Keep the mean_gap values of the pairs bought in more than one basket.
    repeated = features.loc[features["once"] == 0.0, "mean_gap"]
    # Return their median, or 1.0 when there is no such pair.
    return float(repeated.median()) if len(repeated) else 1.0


# ---------------------------------------------------------------------------
# cycle features
# ---------------------------------------------------------------------------
def cycle_features(
    history: pd.DataFrame, cfg: dict[str, Any], decay_gammas: Sequence[float] = DEFAULT_DECAY_GAMMAS
) -> pd.DataFrame:
    """[user_id, item_id] + cycle_feature_names(cfg) as float64, one row per history pair.

    Row order equals ``repeat_features(history, decay_gammas)``. Dummies are 0/1
    floats; every dummy column exists even when its bin is empty in ``history``
    so that fit and eval frames always share the same columns.
    """
    # Compute the repeat features and work on a copy.
    base = repeat_features(history, decay_gammas)
    out = base.copy()
    # Map each item to its cadence and fill the items without one with the global cadence.
    cadence = out["item_id"].map(item_cadence(base)).fillna(global_cadence(base))
    out["item_cadence"] = cadence.astype("float64")
    # Divide recency by the item cadence and cap the ratio at CADENCE_RATIO_CAP.
    out["cadence_ratio"] = np.minimum(out["recency"] / out["item_cadence"], CADENCE_RATIO_CAP)

    # Mark the pairs bought in exactly one basket.
    once = out["once"].to_numpy() == 1.0
    for prefix, key, source in BIN_GROUPS:
        # Find the bin of every row of the source column.
        edges = bin_edges(cfg, key)
        index = bin_index(out[source].to_numpy(), edges)
        columns = bin_columns(prefix, edges)
        for j, column in enumerate(columns, start=1):  # bin 0 is the reference
            # Set the dummy to 1 for the rows that fall into bin j.
            dummy = index == j
            # Set the overdue dummies to 0 for single-purchase pairs.
            if prefix == "ob":
                dummy = dummy & ~once
            out[column] = dummy.astype("float64")
        # Add the single-purchase column to the overdue group.
        if prefix == "ob":
            out[ONCE_COLUMN] = once.astype("float64")

    # Return the key and feature columns in output order as float64 with a fresh row index.
    columns = cycle_feature_names(cfg, decay_gammas)
    out = out[PAIR_KEY + columns].copy()
    out[columns] = out[columns].astype("float64")
    return out.reset_index(drop=True)
