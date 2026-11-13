# *------------------- 2026 Fall Semester Machine Learning Term-project: Team 9 --------------------*
# Code First written at 2026-11-05
# Code Final Edited at 2026-11-13
# Editor: Yejun Kim

'''
Purpose : Tests for nbr.combine on synthetic lists.
Inputs  : Synthetic repeat and explore list frames (columns user_id, rank, item_id, source)
          built in this file.
Outputs : pytest pass/fail results; no file is read or written.
Usage   :
    python -m pytest tests/test_combine.py
'''

# ---------------------------------------------------------------------------
# imports
# ---------------------------------------------------------------------------
import pandas as pd
import pytest

from nbr.combine import combine


# ---------------------------------------------------------------------------
# fixtures and helpers
# ---------------------------------------------------------------------------
def _lists(entries, source):
    '''Build a list frame from ``entries`` (user -> ordered items) with the given ``source``.'''
    rows = [(u, r + 1, item, source) for u, items in entries.items() for r, item in enumerate(items)]
    return pd.DataFrame(rows, columns=["user_id", "rank", "item_id", "source"])


REPEAT = _lists({"u1": ["r1", "r2", "r3", "r4"], "u2": ["r1"], "u3": ["r1", "r2", "r3", "r4", "r5", "r6"]}, "repeat")
EXPLORE = _lists({"u1": ["e1", "e2", "e3", "e4", "e5", "e6"], "u2": ["e1", "e2"], "u3": ["e1"], "u4": ["e1", "e2"]},
                 "explore")


def _items(frame, user):
    '''Return the item, source and rank lists of ``user`` in the list frame ``frame``.'''
    rows = frame[frame["user_id"] == user]
    return list(rows["item_id"]), list(rows["source"]), list(rows["rank"])


# ---------------------------------------------------------------------------
# slot rule
# ---------------------------------------------------------------------------
def test_combine_basic_and_sources():
    # Combine both lists with two repeat slots and a list length of five.
    out = combine(REPEAT, EXPLORE, r=2, k=5)
    items, sources, ranks = _items(out, "u1")
    # User u1 gets two repeat items, then three explore items, at ranks 1 to 5.
    assert items == ["r1", "r2", "e1", "e2", "e3"]
    assert sources == ["repeat", "repeat", "explore", "explore", "explore"]
    assert ranks == [1, 2, 3, 4, 5]
    assert list(out.columns) == ["user_id", "rank", "item_id", "source"]


def test_r_boundaries():
    # Combine with no repeat slot.
    zero = combine(REPEAT, EXPLORE, r=0, k=5)
    # The list holds explore items first and repeat items only after explore runs out.
    assert _items(zero, "u1")[0] == ["e1", "e2", "e3", "e4", "e5"]
    assert _items(zero, "u2")[0] == ["e1", "e2", "r1"]  # explore exhausted -> repeat fills
    # Combine with as many repeat slots as the list length.
    full = combine(REPEAT, EXPLORE, r=5, k=5)
    # The list holds repeat items first and explore items only after repeat runs out.
    assert _items(full, "u1")[0] == ["r1", "r2", "r3", "r4", "e1"]  # repeat short of k -> explore fills
    assert _items(full, "u3")[0] == ["r1", "r2", "r3", "r4", "r5"]  # repeat-only, truncated at k


def test_fewer_repeat_candidates_than_r():
    # Combine with three repeat slots for user u2, who has one repeat item.
    out = combine(REPEAT, EXPLORE, r=3, k=5)
    items, sources, _ = _items(out, "u2")
    # The single repeat item is followed by the explore items.
    assert items == ["r1", "e1", "e2"]
    assert sources == ["repeat", "explore", "explore"]


def test_explore_short_then_repeat_rest_fills():
    # Combine with two repeat slots for user u3, who has one explore item.
    out = combine(REPEAT, EXPLORE, r=2, k=5)
    items, sources, ranks = _items(out, "u3")
    # The leftover repeat items fill the list after the explore item.
    assert items == ["r1", "r2", "e1", "r3", "r4"]
    assert sources == ["repeat", "repeat", "explore", "repeat", "repeat"]
    assert ranks == [1, 2, 3, 4, 5]


# ---------------------------------------------------------------------------
# users, duplicates and validation
# ---------------------------------------------------------------------------
def test_users_in_one_input_only_and_no_duplicates():
    # Append to u1's explore list an item that is also in the repeat list and a new item.
    explore = pd.concat([EXPLORE, _lists({"u1": ["r2", "x1"]}, "explore").assign(rank=[7, 8])], ignore_index=True)
    out = combine(REPEAT, explore, r=2, k=10)
    # No item is listed twice, the explore-only user is kept and ranks are consecutive.
    assert not out.duplicated(["user_id", "item_id"]).any()
    assert _items(out, "u1")[0] == ["r1", "r2", "e1", "e2", "e3", "e4", "e5", "e6", "x1", "r3"]
    assert _items(out, "u4")[0] == ["e1", "e2"]  # explore-only user
    assert out.equals(out.sort_values(["user_id", "rank"]).reset_index(drop=True))
    assert all(ranks == list(range(1, len(ranks) + 1)) for ranks in out.groupby("user_id")["rank"].agg(list))


def test_r_validation():
    # A repeat slot count above the list length or below zero raises ValueError.
    with pytest.raises(ValueError):
        combine(REPEAT, EXPLORE, r=6, k=5)
    with pytest.raises(ValueError):
        combine(REPEAT, EXPLORE, r=-1, k=5)
