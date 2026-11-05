"""Decayed repurchase scores and the repeat (repurchase-only) recommendation list.

Required history columns: user_id, basket_seq, item_id. Functions only use the
frames they receive and never read files or targets.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

BASKET_KEY = ["user_id", "basket_seq"]
PAIR_KEY = ["user_id", "item_id"]
LIST_COLUMNS = ["user_id", "rank", "item_id", "source"]


def basket_ages(history: pd.DataFrame) -> pd.DataFrame:
    """Return [user_id, basket_seq, age] for every distinct user basket.

    Within a user, distinct basket_seq values sorted ascending get
    age = 0 for the most recent basket, 1 for the one before, and so on.
    """
    baskets = history[BASKET_KEY].drop_duplicates().sort_values(BASKET_KEY)
    # Rank descending per user with method="first" (equivalent to dense after de-duplication).
    order = baskets.groupby("user_id", sort=False)["basket_seq"].rank(ascending=False, method="first")
    baskets = baskets.assign(age=(order - 1).astype("int64")).reset_index(drop=True)
    return baskets


def repeat_scores(history: pd.DataFrame, gamma: float) -> pd.DataFrame:
    """Decayed repurchase score for every (user, item) in I(H_u).

    s_rep(u, i) = sum over baskets b of user u containing i of gamma ** age(b).
    An item appearing several times in one basket counts once. With gamma = 1
    the score equals the number of baskets containing the item (personal
    purchase frequency). Returns [user_id, item_id, score(float64)].
    """
    if not 0.0 <= gamma <= 1.0:
        raise ValueError(f"gamma must lie in [0, 1], got {gamma}")
    events = history[BASKET_KEY + ["item_id"]].drop_duplicates()
    events = events.merge(basket_ages(history), on=BASKET_KEY, how="left")
    events["weight"] = np.power(float(gamma), events["age"].to_numpy(dtype="float64"))
    scores = events.groupby(PAIR_KEY, sort=True, as_index=False)["weight"].sum()
    scores = scores.rename(columns={"weight": "score"})
    scores["score"] = scores["score"].astype("float64")
    return scores[PAIR_KEY + ["score"]]


def rank_scores(scores: pd.DataFrame, pop: pd.Series, k: int, source: str) -> pd.DataFrame:
    """Turn a score frame into a list frame with at most k items per user.

    Ties are broken by score desc, then pop desc, then item_id asc. Items
    missing from ``pop`` get popularity 0. Output: [user_id, rank, item_id, source].
    """
    if k < 1:
        raise ValueError(f"k must be >= 1, got {k}")
    frame = scores[PAIR_KEY + ["score"]].copy()
    frame["pop"] = frame["item_id"].map(pop).fillna(0).to_numpy(dtype="float64")
    frame = frame.sort_values(
        ["user_id", "score", "pop", "item_id"], ascending=[True, False, False, True], kind="mergesort"
    )
    frame["rank"] = frame.groupby("user_id", sort=False).cumcount() + 1
    frame = frame[frame["rank"] <= k]
    frame["source"] = source
    return frame[LIST_COLUMNS].reset_index(drop=True)


def repeat_list(history: pd.DataFrame, gamma: float, k: int, pop: pd.Series) -> pd.DataFrame:
    """Top-k repurchase candidates per user (source = "repeat"); may be shorter than k."""
    return rank_scores(repeat_scores(history, gamma), pop, k, "repeat")
