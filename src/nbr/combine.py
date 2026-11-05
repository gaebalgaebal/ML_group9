"""Slot combination of a repeat list and an explore list into one final list."""

from __future__ import annotations

import pandas as pd

from nbr.repeat import LIST_COLUMNS, PAIR_KEY


def combine(repeat: pd.DataFrame, explore: pd.DataFrame, r: int, k: int = 10) -> pd.DataFrame:
    """Final list: top min(r, |repeat_u|) repeat items, then explore items, then leftover repeat items.

    Slots: the first r positions go to the user's repeat list (fewer if the
    user has fewer repeat candidates); the remaining positions are filled
    with the explore list in its order; if the explore list runs out, the
    rest of the repeat list (ranks > r) fills up. Items keep their original
    source; no duplicates per user; ranks are renumbered 1..k. Users present
    in only one of the inputs are handled the same way. r = k gives the
    repeat-only list with explore filling only when repeat candidates are
    short of k. Output sorted by user_id, rank.
    """
    if not 0 <= r <= k:
        raise ValueError(f"r must lie in 0..k, got r={r}, k={k}")
    head = repeat[LIST_COLUMNS][repeat["rank"] <= r].assign(part=0)
    tail = repeat[LIST_COLUMNS][repeat["rank"] > r].assign(part=2)
    middle = explore[LIST_COLUMNS].merge(repeat[PAIR_KEY], on=PAIR_KEY, how="left", indicator=True)
    middle = middle[middle["_merge"] == "left_only"].drop(columns="_merge").assign(part=1)
    combined = pd.concat([head, middle, tail], ignore_index=True)
    combined = combined.sort_values(["user_id", "part", "rank"], kind="mergesort")
    combined["rank"] = combined.groupby("user_id", sort=False).cumcount() + 1
    combined = combined[combined["rank"] <= k]
    return combined[LIST_COLUMNS].reset_index(drop=True)
