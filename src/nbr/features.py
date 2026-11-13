# *------------------- 2026 Fall Semester Machine Learning Term-project: Team 9 --------------------*
# Code First written at 2026-11-05
# Code Final Edited at 2026-11-13
# Editor: Yejun Kim, Bonjun Gu

'''
Purpose : Repeat-candidate features for every (user, item) pair of a user's history (W6).
          All features are computed from the history frame passed in; targets are never read.
          Basket age is the usual contract (distinct basket_seq per user, most recent = 0).
          A purchase "gap" is the basket_seq difference between two consecutive baskets of the
          same user that contain the item; in the processed data basket_seq is contiguous
          from 1, so this equals the age difference.
Inputs  : A history DataFrame with the required columns user_id, basket_seq, item_id and a
          sequence of decay factors (decay_gammas, must contain 1.0).
Outputs : repeat_features returns one row per (user_id, item_id) pair with float64 feature
          columns; feature_names returns the feature column names in output order.
Usage   :
    from nbr.features import repeat_features
    features = repeat_features(fit_history, decay_gammas)
'''

from __future__ import annotations

# ---------------------------------------------------------------------------
# imports
# ---------------------------------------------------------------------------
from collections.abc import Sequence

import numpy as np
import pandas as pd

from nbr.baselines import item_popularity
from nbr.repeat import BASKET_KEY, PAIR_KEY, basket_ages

# ---------------------------------------------------------------------------
# constants
# ---------------------------------------------------------------------------
DEFAULT_DECAY_GAMMAS = (0.5, 0.8, 1.0)
OVERDUE_CAP = 10.0


# ---------------------------------------------------------------------------
# feature column names
# ---------------------------------------------------------------------------
def decay_column(gamma: float) -> str:
    """Column name of the decayed-frequency feature for ``gamma`` (e.g. "dec_g0.8")."""
    return f"dec_g{float(gamma)}"


def feature_names(decay_gammas: Sequence[float] = DEFAULT_DECAY_GAMMAS) -> list[str]:
    """Feature columns of ``repeat_features`` in output order, one dec_g column per gamma in ``decay_gammas``."""
    # List the decayed-frequency columns first and the fixed feature columns after them.
    return [decay_column(g) for g in decay_gammas] + [
        "recency", "mean_gap", "once", "overdue", "item_repeat_rate", "log_pop",
        "user_repeat_rate", "log_hist_len", "share",
    ]


# ---------------------------------------------------------------------------
# repeat features
# ---------------------------------------------------------------------------
def repeat_features(
    history: pd.DataFrame, decay_gammas: Sequence[float] = DEFAULT_DECAY_GAMMAS
) -> pd.DataFrame:
    """One row per (user_id, item_id) pair in ``history`` with float64 features (one dec_g column per gamma in ``decay_gammas``).

    Columns (after the keys user_id, item_id):
      dec_g{gamma}      sum over baskets containing the item of gamma ** age (gamma = 1 -> basket count)
      recency           age of the most recent basket containing the item (most recent basket = 0)
      mean_gap          mean basket_seq difference between consecutive purchases of the item;
                        items bought once get the median mean_gap of that item over all users
                        (then the global median; then 1.0 if no user repeated any item)
      once              1.0 if the item was bought in exactly one basket, else 0.0
      overdue           recency / mean_gap, capped at 10
      item_repeat_rate  share of the item's buyers who bought it in >= 2 baskets
      log_pop           log(1 + pop(i)), pop = number of baskets containing the item
      user_repeat_rate  share of (basket, item) events in the user's 2nd and later baskets whose
                        item was already bought in an earlier basket (0 for single-basket users)
      log_hist_len      log(1 + number of baskets of the user)
      share             dec_g1.0 / number of baskets of the user

    Uses DataFrame.groupby(...).cumcount() (0-based position of each row within
    its group) and groupby(...).diff() (difference to the previous row of the
    group) to find first purchases and purchase gaps after sorting by basket_seq.
    """
    # Convert the decay factors to floats and require 1.0 among them.
    gammas = [float(g) for g in decay_gammas]
    if 1.0 not in gammas:
        raise ValueError("decay_gammas must contain 1.0 (basket count), needed for the share feature")
    # Keep one row per (user, basket, item) event.
    events = history[BASKET_KEY + ["item_id"]].drop_duplicates()
    # Attach the age of each basket.
    events = events.merge(basket_ages(history), on=BASKET_KEY, how="left")
    # Order the events by user, item and basket_seq with a stable sort.
    events = events.sort_values(PAIR_KEY + ["basket_seq"], kind="mergesort").reset_index(drop=True)
    # Add one column gamma ** age per decay factor.
    age = events["age"].to_numpy(dtype="float64")
    for gamma in gammas:
        events[decay_column(gamma)] = np.power(gamma, age)
    # Number the purchases of each pair from 0 and take the basket_seq gap to the previous one.
    by_pair = events.groupby(PAIR_KEY, sort=True)
    events["purchase_index"] = by_pair.cumcount()
    events["gap"] = by_pair["basket_seq"].diff().astype("float64")

    # Aggregate per pair: decayed sums, smallest age, number of purchases and mean gap.
    pairs = by_pair.agg(**{decay_column(g): (decay_column(g), "sum") for g in gammas},
                        recency=("age", "min"), n_purchases=("age", "size"), mean_gap=("gap", "mean"))
    pairs = pairs.reset_index()
    pairs["recency"] = pairs["recency"].astype("float64")
    # Flag the pairs with exactly one purchase.
    pairs["once"] = (pairs["n_purchases"] == 1).astype("float64")

    # mean_gap fallback for single-purchase items: item median over users, then global median, then 1.0.
    item_median = pairs.groupby("item_id")["mean_gap"].median()
    global_median = float(pairs["mean_gap"].median()) if pairs["mean_gap"].notna().any() else 1.0
    fallback = pairs["item_id"].map(item_median).fillna(global_median)
    pairs["mean_gap"] = pairs["mean_gap"].fillna(fallback).astype("float64")
    # Divide recency by mean_gap and cap the ratio at OVERDUE_CAP.
    pairs["overdue"] = np.minimum(pairs["recency"] / pairs["mean_gap"], OVERDUE_CAP)

    # Item-level features: repeat rate over the item's buyers, log popularity.
    repeat_rate = (pairs["n_purchases"] >= 2).groupby(pairs["item_id"]).mean()
    pairs["item_repeat_rate"] = pairs["item_id"].map(repeat_rate).astype("float64")
    pop = item_popularity(history)
    pairs["log_pop"] = np.log1p(pairs["item_id"].map(pop).to_numpy(dtype="float64"))

    # User-level features: repeat share of events after the first basket, history length.
    n_baskets = events[BASKET_KEY].drop_duplicates().groupby("user_id").size()
    # Keep the events that lie after each user's first basket.
    first_seq = events.groupby("user_id")["basket_seq"].transform("min")
    later = events[events["basket_seq"] > first_seq]
    # Take the share of those events whose item was bought before, 0 for users without any.
    user_rate = (later["purchase_index"] > 0).groupby(later["user_id"]).mean()
    pairs["user_repeat_rate"] = pairs["user_id"].map(user_rate).fillna(0.0).astype("float64")
    # Add the log basket count of the user and the basket-count share of the item.
    user_len = pairs["user_id"].map(n_baskets).to_numpy(dtype="float64")
    pairs["log_hist_len"] = np.log1p(user_len)
    pairs["share"] = pairs[decay_column(1.0)] / user_len

    # Return the key and feature columns in output order as float64 with a fresh row index.
    columns = feature_names(gammas)
    out = pairs[PAIR_KEY + columns].copy()
    out[columns] = out[columns].astype("float64")
    return out.reset_index(drop=True)
