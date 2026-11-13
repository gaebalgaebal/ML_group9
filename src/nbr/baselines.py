# *------------------- 2026 Fall Semester Machine Learning Term-project: Team 9 --------------------*
# Code First written at 2026-11-05
# Code Final Edited at 2026-11-13
# Editor: Yejun Kim

'''
Purpose : Popularity and personal-purchase-frequency baselines, plus list padding.
          All functions are vectorised with pandas groupby/merge (no per-user Python loops)
          and use only the frames they receive.
Inputs  : A history frame (required history columns: user_id, basket_seq, item_id), the users
          to build lists for, the list length k, an item popularity Series and, for padding,
          a list frame and a filler list frame with the columns of nbr.repeat.LIST_COLUMNS.
Outputs : item_popularity returns a Series named "pop" indexed by item_id.
          popularity_list, pad_list and personal_frequency_list return a list frame with the
          columns of LIST_COLUMNS and at most k rows per user.
Usage   :
    from nbr.baselines import item_popularity, pad_list, popularity_list
    pop = item_popularity(history)
    filler = popularity_list(history, short, k, pop, exclude_history=True)
    lists = pad_list(lists, filler, k)
'''

from __future__ import annotations

# ---------------------------------------------------------------------------
# imports
# ---------------------------------------------------------------------------
from collections.abc import Iterable

import pandas as pd

from nbr.repeat import LIST_COLUMNS, PAIR_KEY, repeat_list

# ---------------------------------------------------------------------------
# constants
# ---------------------------------------------------------------------------
SOURCE_POPULARITY = "popularity"


# ---------------------------------------------------------------------------
# item popularity
# ---------------------------------------------------------------------------
def item_popularity(history: pd.DataFrame) -> pd.Series:
    """pop(i) = number of distinct (user_id, basket_seq) baskets of ``history`` containing item i.

    Returns a Series named "pop" indexed by item_id (sorted by item_id).
    """
    # Keep one row per (user, basket, item).
    events = history[["user_id", "basket_seq", "item_id"]].drop_duplicates()
    # Count the baskets of every item.
    pop = events.groupby("item_id", sort=True).size().astype("int64")
    # Name the Series and its index.
    pop.name = "pop"
    pop.index.name = "item_id"
    return pop


def _global_ranking(pop: pd.Series) -> pd.DataFrame:
    """Items ordered by pop desc, item_id asc -> [item_id, pop, order]."""
    # Turn the popularity Series into a frame with the columns item_id and pop.
    ranking = pop.rename("pop").rename_axis("item_id").reset_index()
    # Cast the item ids to str.
    ranking["item_id"] = ranking["item_id"].astype(str)
    # Order the items by popularity descending and item_id ascending, with a stable sort.
    ranking = ranking.sort_values(["pop", "item_id"], ascending=[False, True], kind="mergesort")
    # Number the items in that order from 0.
    ranking["order"] = range(len(ranking))
    return ranking.reset_index(drop=True)


# ---------------------------------------------------------------------------
# popularity list
# ---------------------------------------------------------------------------
def popularity_list(
    history: pd.DataFrame,
    users: Iterable[str],
    k: int,
    pop: pd.Series,
    exclude_history: bool,
) -> pd.DataFrame:
    """Top-k globally popular items (by ``pop``) for each user in ``users`` (source = "popularity").

    With ``exclude_history`` the user's own history items I(H_u) are removed
    first (flat popularity baseline for new items). Vectorised: every user is
    cross-joined with the global top-M items, where M = k + max |I(H_u)| so no
    user can run short, then the user's history pairs are anti-joined away.
    """
    # Reject a list length below 1.
    if k < 1:
        raise ValueError(f"k must be >= 1, got {k}")
    # Build a frame with one row per distinct user, with the user ids as str.
    user_frame = pd.DataFrame({"user_id": pd.unique(pd.Series(list(users), dtype="object"))})
    user_frame["user_id"] = user_frame["user_id"].astype(str)
    # Order all items by global popularity.
    ranking = _global_ranking(pop)

    if exclude_history:
        # Collect the distinct (user, item) history pairs of the requested users.
        seen = history[PAIR_KEY].drop_duplicates()
        seen = seen[seen["user_id"].isin(user_frame["user_id"])]
        # Find the largest number of history items of one user.
        max_seen = int(seen.groupby("user_id").size().max()) if len(seen) else 0
        # Take the top k + max_seen items so that k items remain for every user.
        top = ranking.head(k + max_seen)
        # Pair every user with every one of these items.
        candidates = user_frame.merge(top, how="cross")
        # Flag the pairs that occur in the user's history and drop them.
        candidates = candidates.merge(seen, on=PAIR_KEY, how="left", indicator=True)
        candidates = candidates[candidates["_merge"] == "left_only"].drop(columns="_merge")
    else:
        # Pair every user with the global top-k items.
        candidates = user_frame.merge(ranking.head(k), how="cross")

    # Order each user's candidates by the global order, with a stable sort.
    candidates = candidates.sort_values(["user_id", "order"], kind="mergesort")
    # Number the ranks from 1 within each user.
    candidates["rank"] = candidates.groupby("user_id", sort=False).cumcount() + 1
    # Cut each user's list at k rows.
    candidates = candidates[candidates["rank"] <= k]
    # Label the rows with the popularity source.
    candidates["source"] = SOURCE_POPULARITY
    # Return the list columns with a fresh row index.
    return candidates[LIST_COLUMNS].reset_index(drop=True)


# ---------------------------------------------------------------------------
# list padding
# ---------------------------------------------------------------------------
def pad_list(lists: pd.DataFrame, filler: pd.DataFrame, k: int) -> pd.DataFrame:
    """Extend lists shorter than k with filler items, in filler order, without duplicates.

    Users present only in ``filler`` are treated as having an empty list.
    Appended rows keep the filler's source; ranks are renumbered 1..k. Rows
    beyond k are dropped. Output sorted by user_id, rank.
    """
    # Reject a list length below 1.
    if k < 1:
        raise ValueError(f"k must be >= 1, got {k}")
    # Mark the rows of the given lists as the first part.
    base = lists[LIST_COLUMNS].assign(part=0)
    # Flag every filler row by whether its (user, item) pair also occurs in the given lists.
    extra = filler[LIST_COLUMNS].merge(lists[PAIR_KEY], on=PAIR_KEY, how="left", indicator=True)
    # Keep the filler rows that are not in the given lists as the second part.
    extra = extra[extra["_merge"] == "left_only"].drop(columns="_merge").assign(part=1)
    # Stack the two parts into one frame.
    combined = pd.concat([base, extra], ignore_index=True)
    # Order each user's rows by part and then by the original rank, with a stable sort.
    combined = combined.sort_values(["user_id", "part", "rank"], kind="mergesort")
    # Renumber the ranks from 1 within each user.
    combined["rank"] = combined.groupby("user_id", sort=False).cumcount() + 1
    # Cut each user's list at k rows.
    combined = combined[combined["rank"] <= k]
    # Return the list columns with a fresh row index.
    return combined[LIST_COLUMNS].reset_index(drop=True)


# ---------------------------------------------------------------------------
# personal frequency baseline
# ---------------------------------------------------------------------------
def personal_frequency_list(history: pd.DataFrame, k: int, pop: pd.Series) -> pd.DataFrame:
    """Top-``k`` list per user of ``history``: personal purchase frequency (gamma = 1) padded with ``pop``-popularity outside I(H_u)."""
    # Rank each user's history items by purchase frequency without decay.
    repeat = repeat_list(history, 1.0, k, pop)
    # Collect the distinct users of the history.
    users = history["user_id"].drop_duplicates()
    # Build the popularity list without the user's history items as the filler.
    filler = popularity_list(history, users, k, pop, exclude_history=True)
    # Fill the lists shorter than k with the filler items.
    return pad_list(repeat, filler, k)
