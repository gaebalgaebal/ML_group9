# *------------------- 2026 Fall Semester Machine Learning Term-project: Team 9 --------------------*
# Code First written at 2026-11-06
# Code Final Edited at 2026-11-13
# Editor: Yejun Kim

'''
Purpose : Day-based recency decay for the repeat scorer (W14; post-hoc, development data, not
          confirmatory).
          score(u, i) = sum over baskets b in H_u containing i of lam ** (d_last(u) - d_b),
          with lam = 2 ** (-1 / h), h = half-life in days and d_last(u) the day of the user's
          most recent history basket; h = inf gives lam = 1, i.e. personal purchase frequency
          (nbr.repeat.repeat_scores with gamma = 1).
          Functions only use the frames they receive; the loader reads the basket-day file of
          a dataset and nothing else.
Inputs  : A history frame (user_id, basket_id, basket_seq, item_id), a basket-day frame, a
          half-life in days, the list length k and an item popularity Series.
          Basket days come from <dataset_dir>/basket_days.parquet
          (scripts/make_basket_days.py; columns user_id, basket_id, day).
Outputs : day_decay_scores returns [user_id, item_id, score]; day_decay_list returns a list
          frame with at most k rows per user; inter_basket_gaps returns an int64 Series of
          day gaps; decay_rate and basket_half_life return a float.
Usage   :
    from nbr.day_decay import HALF_LIFE_GRID, day_decay_list, load_basket_days
    days = load_basket_days(config, dataset)
    lists = {h: day_decay_list(history, days, h, K, pop) for h in HALF_LIFE_GRID}
'''

from __future__ import annotations

# ---------------------------------------------------------------------------
# imports
# ---------------------------------------------------------------------------
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from nbr.data import dataset_dir
from nbr.repeat import PAIR_KEY, rank_scores

# ---------------------------------------------------------------------------
# constants
# ---------------------------------------------------------------------------
BASKET_ID_KEY = ["user_id", "basket_id"]
DAY_COLUMNS = ["user_id", "basket_id", "day"]
# Half-life grid in days of the post-hoc analysis (inf = personal frequency).
HALF_LIFE_GRID = [7.0, 14.0, 30.0, 60.0, 90.0, 180.0, 365.0, math.inf]


# ---------------------------------------------------------------------------
# basket-day loading and validation
# ---------------------------------------------------------------------------
def basket_days_path(config: dict[str, Any], dataset: str) -> Path:
    """<dataset_dir(config, dataset)>/basket_days.parquet."""
    return dataset_dir(config, dataset) / "basket_days.parquet"


def check_basket_days(days: pd.DataFrame) -> pd.DataFrame:
    """Validate a basket-day frame: columns user_id, basket_id, day; no nulls; one row per (user_id, basket_id).

    Returns a copy restricted to those columns with ``day`` as int64.
    """
    # Fail when a required column is absent.
    missing = [column for column in DAY_COLUMNS if column not in days.columns]
    if missing:
        raise ValueError(f"basket days: missing columns {missing}")
    # Copy the three required columns.
    out = days[DAY_COLUMNS].copy()
    # Fail when any of them holds a null value.
    if out.isna().any().any():
        raise ValueError("basket days: null values present")
    # Cast the day to int64.
    out["day"] = out["day"].astype("int64")
    # Fail when a (user_id, basket_id) pair occurs more than once.
    if out.duplicated(BASKET_ID_KEY).any():
        raise ValueError("basket days: (user_id, basket_id) must be unique")
    return out.reset_index(drop=True)


def load_basket_days(config: dict[str, Any], dataset: str) -> pd.DataFrame:
    """Read and validate basket_days_path(config, dataset)."""
    return check_basket_days(pd.read_parquet(basket_days_path(config, dataset)))


# ---------------------------------------------------------------------------
# decay rate and half-life
# ---------------------------------------------------------------------------
def decay_rate(half_life: float) -> float:
    """lam = 2 ** (-1 / half_life) for a half-life in days (> 0); half_life = inf -> 1.0."""
    # Reject a bool or a value that is not a real number.
    if isinstance(half_life, bool) or not isinstance(half_life, (int, float, np.integer, np.floating)):
        raise ValueError(f"half_life must be a positive number, got {half_life!r}")
    # Reject NaN, zero and negative values.
    if math.isnan(half_life) or half_life <= 0:
        raise ValueError(f"half_life must be > 0, got {half_life!r}")
    # An infinite half-life means no decay.
    if math.isinf(half_life):
        return 1.0
    # Compute the per-day decay factor.
    return 2.0 ** (-1.0 / float(half_life))


def basket_half_life(gamma: float) -> float:
    """Half-life in baskets of the basket decay ``gamma`` in (0, 1]: ln(0.5) / ln(gamma); gamma = 1 -> inf."""
    # Reject a gamma outside (0, 1].
    if not 0.0 < gamma <= 1.0:
        raise ValueError(f"gamma must lie in (0, 1], got {gamma}")
    # A gamma of 1 means no decay, so the half-life is infinite.
    if gamma == 1.0:
        return math.inf
    # Compute the number of baskets after which the weight is halved.
    return math.log(0.5) / math.log(float(gamma))


# ---------------------------------------------------------------------------
# day-decayed repeat scores and list
# ---------------------------------------------------------------------------
def history_basket_days(history: pd.DataFrame, days: pd.DataFrame) -> pd.DataFrame:
    """[user_id, basket_id, basket_seq, day] for every distinct basket of ``history``.

    ``days`` is a basket-day frame (check_basket_days). Every history basket must
    have exactly one day (ValueError otherwise). The id columns keep the dtypes of
    ``history``.
    """
    # Keep one row per distinct basket of the history.
    baskets = history[BASKET_ID_KEY + ["basket_seq"]].drop_duplicates().reset_index(drop=True)
    # Fail when one (user_id, basket_id) pair has several basket_seq values.
    if baskets.duplicated(BASKET_ID_KEY).any():
        raise ValueError("history: a (user_id, basket_id) pair maps to several basket_seq values")
    # Validate the basket-day frame.
    lookup = check_basket_days(days)
    # Cast its id columns to the dtypes of the history.
    for column in BASKET_ID_KEY:
        lookup[column] = lookup[column].astype(baskets[column].dtype)
    # Attach the day of every history basket.
    merged = baskets.merge(lookup, on=BASKET_ID_KEY, how="left", validate="one_to_one")
    # Fail when a history basket has no day.
    if merged["day"].isna().any():
        sample = merged.loc[merged["day"].isna(), BASKET_ID_KEY].head(5).to_dict("records")
        raise ValueError(f"{int(merged['day'].isna().sum())} history baskets have no day, e.g. {sample}")
    # Cast the day to int64.
    merged["day"] = merged["day"].astype("int64")
    return merged[BASKET_ID_KEY + ["basket_seq", "day"]]


def day_decay_scores(history: pd.DataFrame, days: pd.DataFrame, half_life: float) -> pd.DataFrame:
    """Day-decayed repurchase score of every (user, item) pair of ``history`` with half-life ``half_life`` (days).

    s(u, i) = sum over baskets b of user u containing i of lam ** (d_last(u) - d_b),
    lam = decay_rate(half_life), d_last(u) = max day of the user's history baskets.
    An item appearing several times in one basket counts once. With half_life = inf
    the score equals the number of baskets containing the item. Returns
    [user_id, item_id, score(float64)] sorted by (user_id, item_id).
    """
    # Convert the half-life into the per-day decay factor.
    lam = decay_rate(half_life)
    # Get the day of every history basket.
    baskets = history_basket_days(history, days)
    # Find the day of each user's most recent basket.
    last_day = baskets.groupby("user_id", sort=False)["day"].transform("max")
    # Compute the age of every basket in days relative to that day.
    age_days = (last_day - baskets["day"]).to_numpy(dtype="float64")
    # Weight every basket by the decay factor raised to its age.
    baskets = baskets.assign(weight=np.power(lam, age_days))
    # Keep one row per (user, basket, item).
    events = history[BASKET_ID_KEY + ["item_id"]].drop_duplicates()
    # Attach the basket weight to every row.
    events = events.merge(baskets[BASKET_ID_KEY + ["weight"]], on=BASKET_ID_KEY, how="left")
    # Sum the weights of every (user, item) pair.
    scores = events.groupby(PAIR_KEY, sort=True, as_index=False)["weight"].sum()
    # Rename the sum to score and store it as float64.
    scores = scores.rename(columns={"weight": "score"})
    scores["score"] = scores["score"].astype("float64")
    return scores[PAIR_KEY + ["score"]]


def day_decay_list(
    history: pd.DataFrame, days: pd.DataFrame, half_life: float, k: int, pop: pd.Series
) -> pd.DataFrame:
    """Top-``k`` repurchase candidates per user by day_decay_scores(history, days, half_life).

    Same list schema, tie-breaking (score desc, pop desc, item_id asc) and length
    rule (at most k rows per user, no padding) as nbr.repeat.repeat_list; source =
    "repeat".
    """
    return rank_scores(day_decay_scores(history, days, half_life), pop, k, "repeat")


# ---------------------------------------------------------------------------
# inter-basket gaps
# ---------------------------------------------------------------------------
def inter_basket_gaps(history: pd.DataFrame, days: pd.DataFrame) -> pd.Series:
    """Day gap between consecutive baskets of each user of ``history`` (baskets ordered by day, then basket_seq).

    Returns an int64 Series with one entry per consecutive pair (users with one
    basket contribute nothing); same-day baskets give a gap of 0.
    """
    # Order each user's baskets by day and then by basket_seq, with a stable sort.
    baskets = history_basket_days(history, days).sort_values(["user_id", "day", "basket_seq"], kind="mergesort")
    # Take the day difference to the previous basket of the same user.
    gaps = baskets.groupby("user_id", sort=False)["day"].diff().dropna()
    # Return the gaps as an int64 Series named gap_days with a fresh index.
    return gaps.astype("int64").reset_index(drop=True).rename("gap_days")
