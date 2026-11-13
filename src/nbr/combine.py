# *------------------- 2026 Fall Semester Machine Learning Term-project: Team 9 --------------------*
# Code First written at 2026-11-05
# Code Final Edited at 2026-11-13
# Editor: Yejun Kim

'''
Purpose : Slot combination of a repeat list and an explore list into one final list.
Inputs  : A repeat list and an explore list (DataFrames with the columns of
          nbr.repeat.LIST_COLUMNS), the number of repeat slots r and the list length k.
Outputs : One DataFrame with the columns of LIST_COLUMNS, at most k rows per user,
          sorted by user_id, rank.
Usage   :
    from nbr.combine import combine
    final = combine(repeat_star, explore_star, r, K)
'''

from __future__ import annotations

# ---------------------------------------------------------------------------
# imports
# ---------------------------------------------------------------------------
import pandas as pd

from nbr.repeat import LIST_COLUMNS, PAIR_KEY


# ---------------------------------------------------------------------------
# slot combination
# ---------------------------------------------------------------------------
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
    # Reject a number of repeat slots outside 0..k.
    if not 0 <= r <= k:
        raise ValueError(f"r must lie in 0..k, got r={r}, k={k}")
    # Mark the repeat rows with rank <= r as the first part of the list.
    head = repeat[LIST_COLUMNS][repeat["rank"] <= r].assign(part=0)
    # Mark the repeat rows with rank > r as the last part of the list.
    tail = repeat[LIST_COLUMNS][repeat["rank"] > r].assign(part=2)
    # Flag every explore row by whether its (user, item) pair also occurs in the repeat list.
    middle = explore[LIST_COLUMNS].merge(repeat[PAIR_KEY], on=PAIR_KEY, how="left", indicator=True)
    # Keep the explore rows that are not in the repeat list as the middle part.
    middle = middle[middle["_merge"] == "left_only"].drop(columns="_merge").assign(part=1)
    # Stack the three parts into one frame.
    combined = pd.concat([head, middle, tail], ignore_index=True)
    # Order each user's rows by part and then by the original rank, with a stable sort.
    combined = combined.sort_values(["user_id", "part", "rank"], kind="mergesort")
    # Renumber the ranks from 1 within each user.
    combined["rank"] = combined.groupby("user_id", sort=False).cumcount() + 1
    # Cut each user's list at k rows.
    combined = combined[combined["rank"] <= k]
    # Return the list columns with a fresh row index.
    return combined[LIST_COLUMNS].reset_index(drop=True)
