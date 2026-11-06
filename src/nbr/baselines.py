"""Popularity and personal-purchase-frequency baselines, plus list padding.

Required history columns: user_id, basket_seq, item_id. All functions are
vectorised with pandas groupby/merge (no per-user Python loops) and use only
the frames they receive.
"""

from __future__ import annotations

from collections.abc import Iterable

import pandas as pd

from nbr.repeat import LIST_COLUMNS, PAIR_KEY, repeat_list

SOURCE_POPULARITY = "popularity"


def item_popularity(history: pd.DataFrame) -> pd.Series:
    """pop(i) = number of distinct (user_id, basket_seq) baskets of ``history`` containing item i.

    Returns a Series named "pop" indexed by item_id (sorted by item_id).
    """
    events = history[["user_id", "basket_seq", "item_id"]].drop_duplicates()
    pop = events.groupby("item_id", sort=True).size().astype("int64")
    pop.name = "pop"
    pop.index.name = "item_id"
    return pop


def _global_ranking(pop: pd.Series) -> pd.DataFrame:
    """Items ordered by pop desc, item_id asc -> [item_id, pop, order]."""
    ranking = pop.rename("pop").rename_axis("item_id").reset_index()
    ranking["item_id"] = ranking["item_id"].astype(str)
    ranking = ranking.sort_values(["pop", "item_id"], ascending=[False, True], kind="mergesort")
    ranking["order"] = range(len(ranking))
    return ranking.reset_index(drop=True)


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
    if k < 1:
        raise ValueError(f"k must be >= 1, got {k}")
    user_frame = pd.DataFrame({"user_id": pd.unique(pd.Series(list(users), dtype="object"))})
    user_frame["user_id"] = user_frame["user_id"].astype(str)
    ranking = _global_ranking(pop)

    if exclude_history:
        seen = history[PAIR_KEY].drop_duplicates()
        seen = seen[seen["user_id"].isin(user_frame["user_id"])]
        max_seen = int(seen.groupby("user_id").size().max()) if len(seen) else 0
        top = ranking.head(k + max_seen)
        candidates = user_frame.merge(top, how="cross")
        candidates = candidates.merge(seen, on=PAIR_KEY, how="left", indicator=True)
        candidates = candidates[candidates["_merge"] == "left_only"].drop(columns="_merge")
    else:
        candidates = user_frame.merge(ranking.head(k), how="cross")

    candidates = candidates.sort_values(["user_id", "order"], kind="mergesort")
    candidates["rank"] = candidates.groupby("user_id", sort=False).cumcount() + 1
    candidates = candidates[candidates["rank"] <= k]
    candidates["source"] = SOURCE_POPULARITY
    return candidates[LIST_COLUMNS].reset_index(drop=True)


def pad_list(lists: pd.DataFrame, filler: pd.DataFrame, k: int) -> pd.DataFrame:
    """Extend lists shorter than k with filler items, in filler order, without duplicates.

    Users present only in ``filler`` are treated as having an empty list.
    Appended rows keep the filler's source; ranks are renumbered 1..k. Rows
    beyond k are dropped. Output sorted by user_id, rank.
    """
    if k < 1:
        raise ValueError(f"k must be >= 1, got {k}")
    base = lists[LIST_COLUMNS].assign(part=0)
    extra = filler[LIST_COLUMNS].merge(lists[PAIR_KEY], on=PAIR_KEY, how="left", indicator=True)
    extra = extra[extra["_merge"] == "left_only"].drop(columns="_merge").assign(part=1)
    combined = pd.concat([base, extra], ignore_index=True)
    combined = combined.sort_values(["user_id", "part", "rank"], kind="mergesort")
    combined["rank"] = combined.groupby("user_id", sort=False).cumcount() + 1
    combined = combined[combined["rank"] <= k]
    return combined[LIST_COLUMNS].reset_index(drop=True)


def personal_frequency_list(history: pd.DataFrame, k: int, pop: pd.Series) -> pd.DataFrame:
    """Top-``k`` list per user of ``history``: personal purchase frequency (gamma = 1) padded with ``pop``-popularity outside I(H_u)."""
    repeat = repeat_list(history, 1.0, k, pop)
    users = history["user_id"].drop_duplicates()
    filler = popularity_list(history, users, k, pop, exclude_history=True)
    return pad_list(repeat, filler, k)
