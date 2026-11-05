"""Tests for nbr.combine on synthetic lists."""

import pandas as pd
import pytest

from nbr.combine import combine


def _lists(entries, source):
    rows = [(u, r + 1, item, source) for u, items in entries.items() for r, item in enumerate(items)]
    return pd.DataFrame(rows, columns=["user_id", "rank", "item_id", "source"])


REPEAT = _lists({"u1": ["r1", "r2", "r3", "r4"], "u2": ["r1"], "u3": ["r1", "r2", "r3", "r4", "r5", "r6"]}, "repeat")
EXPLORE = _lists({"u1": ["e1", "e2", "e3", "e4", "e5", "e6"], "u2": ["e1", "e2"], "u3": ["e1"], "u4": ["e1", "e2"]},
                 "explore")


def _items(frame, user):
    rows = frame[frame["user_id"] == user]
    return list(rows["item_id"]), list(rows["source"]), list(rows["rank"])


def test_combine_basic_and_sources():
    out = combine(REPEAT, EXPLORE, r=2, k=5)
    items, sources, ranks = _items(out, "u1")
    assert items == ["r1", "r2", "e1", "e2", "e3"]
    assert sources == ["repeat", "repeat", "explore", "explore", "explore"]
    assert ranks == [1, 2, 3, 4, 5]
    assert list(out.columns) == ["user_id", "rank", "item_id", "source"]


def test_r_boundaries():
    zero = combine(REPEAT, EXPLORE, r=0, k=5)
    assert _items(zero, "u1")[0] == ["e1", "e2", "e3", "e4", "e5"]
    assert _items(zero, "u2")[0] == ["e1", "e2", "r1"]  # explore exhausted -> repeat fills
    full = combine(REPEAT, EXPLORE, r=5, k=5)
    assert _items(full, "u1")[0] == ["r1", "r2", "r3", "r4", "e1"]  # repeat short of k -> explore fills
    assert _items(full, "u3")[0] == ["r1", "r2", "r3", "r4", "r5"]  # repeat-only, truncated at k


def test_fewer_repeat_candidates_than_r():
    out = combine(REPEAT, EXPLORE, r=3, k=5)
    items, sources, _ = _items(out, "u2")
    assert items == ["r1", "e1", "e2"]
    assert sources == ["repeat", "explore", "explore"]


def test_explore_short_then_repeat_rest_fills():
    out = combine(REPEAT, EXPLORE, r=2, k=5)
    items, sources, ranks = _items(out, "u3")
    assert items == ["r1", "r2", "e1", "r3", "r4"]
    assert sources == ["repeat", "repeat", "explore", "repeat", "repeat"]
    assert ranks == [1, 2, 3, 4, 5]


def test_users_in_one_input_only_and_no_duplicates():
    explore = pd.concat([EXPLORE, _lists({"u1": ["r2", "x1"]}, "explore").assign(rank=[7, 8])], ignore_index=True)
    out = combine(REPEAT, explore, r=2, k=10)
    assert not out.duplicated(["user_id", "item_id"]).any()
    assert _items(out, "u1")[0] == ["r1", "r2", "e1", "e2", "e3", "e4", "e5", "e6", "x1", "r3"]
    assert _items(out, "u4")[0] == ["e1", "e2"]  # explore-only user
    assert out.equals(out.sort_values(["user_id", "rank"]).reset_index(drop=True))
    assert all(ranks == list(range(1, len(ranks) + 1)) for ranks in out.groupby("user_id")["rank"].agg(list))


def test_r_validation():
    with pytest.raises(ValueError):
        combine(REPEAT, EXPLORE, r=6, k=5)
    with pytest.raises(ValueError):
        combine(REPEAT, EXPLORE, r=-1, k=5)
