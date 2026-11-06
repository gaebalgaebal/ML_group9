"""Day-based recency decay for the repeat scorer (W14; post-hoc, development data, not confirmatory).

score(u, i) = sum over baskets b in H_u containing i of lam ** (d_last(u) - d_b), with
lam = 2 ** (-1 / h), h = half-life in days and d_last(u) the day of the user's most
recent history basket; h = inf gives lam = 1, i.e. personal purchase frequency
(nbr.repeat.repeat_scores with gamma = 1). Basket days come from
<dataset_dir>/basket_days.parquet (scripts/make_basket_days.py; columns user_id,
basket_id, day). Functions only use the frames they receive; the loader reads the
basket-day file of a dataset and nothing else.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from nbr.data import dataset_dir
from nbr.repeat import PAIR_KEY, rank_scores

BASKET_ID_KEY = ["user_id", "basket_id"]
DAY_COLUMNS = ["user_id", "basket_id", "day"]
# Half-life grid in days of the post-hoc analysis (inf = personal frequency).
HALF_LIFE_GRID = [7.0, 14.0, 30.0, 60.0, 90.0, 180.0, 365.0, math.inf]


def basket_days_path(config: dict[str, Any], dataset: str) -> Path:
    """<dataset_dir(config, dataset)>/basket_days.parquet."""
    return dataset_dir(config, dataset) / "basket_days.parquet"


def check_basket_days(days: pd.DataFrame) -> pd.DataFrame:
    """Validate a basket-day frame: columns user_id, basket_id, day; no nulls; one row per (user_id, basket_id).

    Returns a copy restricted to those columns with ``day`` as int64.
    """
    missing = [column for column in DAY_COLUMNS if column not in days.columns]
    if missing:
        raise ValueError(f"basket days: missing columns {missing}")
    out = days[DAY_COLUMNS].copy()
    if out.isna().any().any():
        raise ValueError("basket days: null values present")
    out["day"] = out["day"].astype("int64")
    if out.duplicated(BASKET_ID_KEY).any():
        raise ValueError("basket days: (user_id, basket_id) must be unique")
    return out.reset_index(drop=True)


def load_basket_days(config: dict[str, Any], dataset: str) -> pd.DataFrame:
    """Read and validate basket_days_path(config, dataset)."""
    return check_basket_days(pd.read_parquet(basket_days_path(config, dataset)))


def decay_rate(half_life: float) -> float:
    """lam = 2 ** (-1 / half_life) for a half-life in days (> 0); half_life = inf -> 1.0."""
    if isinstance(half_life, bool) or not isinstance(half_life, (int, float, np.integer, np.floating)):
        raise ValueError(f"half_life must be a positive number, got {half_life!r}")
    if math.isnan(half_life) or half_life <= 0:
        raise ValueError(f"half_life must be > 0, got {half_life!r}")
    if math.isinf(half_life):
        return 1.0
    return 2.0 ** (-1.0 / float(half_life))


def basket_half_life(gamma: float) -> float:
    """Half-life in baskets of the basket decay ``gamma`` in (0, 1]: ln(0.5) / ln(gamma); gamma = 1 -> inf."""
    if not 0.0 < gamma <= 1.0:
        raise ValueError(f"gamma must lie in (0, 1], got {gamma}")
    if gamma == 1.0:
        return math.inf
    return math.log(0.5) / math.log(float(gamma))


def history_basket_days(history: pd.DataFrame, days: pd.DataFrame) -> pd.DataFrame:
    """[user_id, basket_id, basket_seq, day] for every distinct basket of ``history``.

    ``days`` is a basket-day frame (check_basket_days). Every history basket must
    have exactly one day (ValueError otherwise). The id columns keep the dtypes of
    ``history``.
    """
    baskets = history[BASKET_ID_KEY + ["basket_seq"]].drop_duplicates().reset_index(drop=True)
    if baskets.duplicated(BASKET_ID_KEY).any():
        raise ValueError("history: a (user_id, basket_id) pair maps to several basket_seq values")
    lookup = check_basket_days(days)
    for column in BASKET_ID_KEY:
        lookup[column] = lookup[column].astype(baskets[column].dtype)
    merged = baskets.merge(lookup, on=BASKET_ID_KEY, how="left", validate="one_to_one")
    if merged["day"].isna().any():
        sample = merged.loc[merged["day"].isna(), BASKET_ID_KEY].head(5).to_dict("records")
        raise ValueError(f"{int(merged['day'].isna().sum())} history baskets have no day, e.g. {sample}")
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
    lam = decay_rate(half_life)
    baskets = history_basket_days(history, days)
    last_day = baskets.groupby("user_id", sort=False)["day"].transform("max")
    age_days = (last_day - baskets["day"]).to_numpy(dtype="float64")
    baskets = baskets.assign(weight=np.power(lam, age_days))
    events = history[BASKET_ID_KEY + ["item_id"]].drop_duplicates()
    events = events.merge(baskets[BASKET_ID_KEY + ["weight"]], on=BASKET_ID_KEY, how="left")
    scores = events.groupby(PAIR_KEY, sort=True, as_index=False)["weight"].sum()
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


def inter_basket_gaps(history: pd.DataFrame, days: pd.DataFrame) -> pd.Series:
    """Day gap between consecutive baskets of each user of ``history`` (baskets ordered by day, then basket_seq).

    Returns an int64 Series with one entry per consecutive pair (users with one
    basket contribute nothing); same-day baskets give a gap of 0.
    """
    baskets = history_basket_days(history, days).sort_values(["user_id", "day", "basket_seq"], kind="mergesort")
    gaps = baskets.groupby("user_id", sort=False)["day"].diff().dropna()
    return gaps.astype("int64").reset_index(drop=True).rename("gap_days")
