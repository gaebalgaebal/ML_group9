# *------------------- 2026 Fall Semester Machine Learning Term-project: Team 9 --------------------*
# Code First written at 2026-11-05
# Code Final Edited at 2026-11-13
# Editor: Yejun Kim, Bonjun Gu

'''
Purpose : Decayed repurchase scores and the repeat (repurchase-only) recommendation list,
          i.e. the decayed purchase frequency repeat module (R1).
Inputs  : A history DataFrame with the required columns user_id, basket_seq, item_id, a decay
          factor gamma, the list length k and an item popularity Series for tie-breaking.
          Functions only use the frames they receive and never read files or targets.
Outputs : basket_ages returns [user_id, basket_seq, age]; repeat_scores returns
          [user_id, item_id, score]; rank_scores and repeat_list return a list frame with the
          columns of LIST_COLUMNS ([user_id, rank, item_id, source]).
Usage   :
    from nbr.repeat import repeat_list
    repeat_star = repeat_list(history, gamma_star, K, pop)
'''

from __future__ import annotations

# ---------------------------------------------------------------------------
# imports
# ---------------------------------------------------------------------------
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# constants
# ---------------------------------------------------------------------------
BASKET_KEY = ["user_id", "basket_seq"]
PAIR_KEY = ["user_id", "item_id"]
LIST_COLUMNS = ["user_id", "rank", "item_id", "source"]


# ---------------------------------------------------------------------------
# basket ages
# ---------------------------------------------------------------------------
def basket_ages(history: pd.DataFrame) -> pd.DataFrame:
    """Return [user_id, basket_seq, age] for every distinct basket of ``history``.

    Within a user, distinct basket_seq values sorted ascending get
    age = 0 for the most recent basket, 1 for the one before, and so on.
    """
    # Keep one row per basket, sorted by user and basket_seq.
    baskets = history[BASKET_KEY].drop_duplicates().sort_values(BASKET_KEY)
    # Rank descending per user with method="first" (equivalent to dense after de-duplication).
    order = baskets.groupby("user_id", sort=False)["basket_seq"].rank(ascending=False, method="first")
    # Store rank - 1 as the integer age and reset the row index.
    baskets = baskets.assign(age=(order - 1).astype("int64")).reset_index(drop=True)
    return baskets


# ---------------------------------------------------------------------------
# decayed repurchase scores
# ---------------------------------------------------------------------------
def repeat_scores(history: pd.DataFrame, gamma: float) -> pd.DataFrame:
    """Decayed repurchase score for every (user, item) pair of ``history`` (I(H_u)) with decay ``gamma``.

    s_rep(u, i) = sum over baskets b of user u containing i of gamma ** age(b).
    An item appearing several times in one basket counts once. With gamma = 1
    the score equals the number of baskets containing the item (personal
    purchase frequency). Returns [user_id, item_id, score(float64)].
    """
    # Reject a decay factor outside [0, 1].
    if not 0.0 <= gamma <= 1.0:
        raise ValueError(f"gamma must lie in [0, 1], got {gamma}")
    # Keep one row per (user, basket, item) event.
    events = history[BASKET_KEY + ["item_id"]].drop_duplicates()
    # Attach the age of each basket.
    events = events.merge(basket_ages(history), on=BASKET_KEY, how="left")
    # Weight each event by gamma ** age.
    events["weight"] = np.power(float(gamma), events["age"].to_numpy(dtype="float64"))
    # Sum the weights of each (user, item) pair.
    scores = events.groupby(PAIR_KEY, sort=True, as_index=False)["weight"].sum()
    # Rename the sum to "score" and return it as float64 next to the pair key.
    scores = scores.rename(columns={"weight": "score"})
    scores["score"] = scores["score"].astype("float64")
    return scores[PAIR_KEY + ["score"]]


# ---------------------------------------------------------------------------
# ranked lists
# ---------------------------------------------------------------------------
def rank_scores(scores: pd.DataFrame, pop: pd.Series, k: int, source: str) -> pd.DataFrame:
    """Turn ``scores`` ([user_id, item_id, score]) into a list frame with at most ``k`` items per user, tagged ``source``.

    Ties are broken by score desc, then pop desc, then item_id asc. Items
    missing from ``pop`` get popularity 0. Output: [user_id, rank, item_id, source].
    """
    # Reject a list length below 1.
    if k < 1:
        raise ValueError(f"k must be >= 1, got {k}")
    # Copy the score rows and attach the item popularity, 0 for unknown items.
    frame = scores[PAIR_KEY + ["score"]].copy()
    frame["pop"] = frame["item_id"].map(pop).fillna(0).to_numpy(dtype="float64")
    # Order each user's rows by score, popularity and item_id with a stable sort.
    frame = frame.sort_values(
        ["user_id", "score", "pop", "item_id"], ascending=[True, False, False, True], kind="mergesort"
    )
    # Number the rows of each user from 1.
    frame["rank"] = frame.groupby("user_id", sort=False).cumcount() + 1
    # Cut each user's list at k rows.
    frame = frame[frame["rank"] <= k]
    # Tag the rows with the source name and return the list columns with a fresh row index.
    frame["source"] = source
    return frame[LIST_COLUMNS].reset_index(drop=True)


def repeat_list(history: pd.DataFrame, gamma: float, k: int, pop: pd.Series) -> pd.DataFrame:
    """Top-``k`` repurchase candidates per user of ``history`` by repeat_scores(history, gamma), ties by ``pop``
    (source = "repeat"); may be shorter than k."""
    return rank_scores(repeat_scores(history, gamma), pop, k, "repeat")
