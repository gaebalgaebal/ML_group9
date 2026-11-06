"""Tests for nbr.baselines on synthetic baskets."""

import inspect

import pandas as pd
import pytest

from nbr.baselines import item_popularity, pad_list, personal_frequency_list, popularity_list


def _history(rows):
    frame = pd.DataFrame(rows, columns=["user_id", "basket_seq", "item_id"])
    return frame.astype({"basket_seq": "int32"})


HISTORY = _history([
    ("u1", 1, "a"), ("u1", 1, "b"), ("u1", 2, "a"), ("u1", 2, "a"),
    ("u2", 1, "a"), ("u2", 1, "c"),
    ("u3", 1, "d"),
])
# pop: a=3 (u1 twice as distinct baskets, u2 once), b=1, c=1, d=1


def _lists(rows, source):
    return pd.DataFrame(rows, columns=["user_id", "rank", "item_id"]).assign(source=source)


def test_item_popularity():
    pop = item_popularity(HISTORY)
    assert pop.name == "pop" and pop.index.name == "item_id"
    assert pop.to_dict() == {"a": 3, "b": 1, "c": 1, "d": 1}


def test_popularity_list_global_order_and_ties():
    pop = item_popularity(HISTORY)
    lists = popularity_list(HISTORY, ["u1", "u2"], 3, pop, exclude_history=False)
    assert list(lists.columns) == ["user_id", "rank", "item_id", "source"]
    assert (lists["source"] == "popularity").all()
    assert list(lists[lists["user_id"] == "u1"]["item_id"]) == ["a", "b", "c"]  # pop desc, id asc
    assert list(lists[lists["user_id"] == "u2"]["item_id"]) == ["a", "b", "c"]
    assert list(lists["user_id"]) == ["u1"] * 3 + ["u2"] * 3


def test_popularity_list_exclude_history():
    pop = item_popularity(HISTORY)
    lists = popularity_list(HISTORY, ["u1", "u2", "u3", "u9"], 2, pop, exclude_history=True)
    by_user = lists.groupby("user_id")["item_id"].agg(list)
    assert by_user["u1"] == ["c", "d"]       # a, b excluded
    assert by_user["u2"] == ["b", "d"]       # a, c excluded
    assert by_user["u3"] == ["a", "b"]       # d excluded
    assert by_user["u9"] == ["a", "b"]       # no history -> nothing excluded
    assert all(ranks == [1, 2] for ranks in lists.groupby("user_id")["rank"].agg(list))
    # Catalogue smaller than k + history: lists may be shorter than k.
    short = popularity_list(HISTORY, ["u1"], 5, pop, exclude_history=True)
    assert list(short["item_id"]) == ["c", "d"]


def test_pad_list():
    lists = _lists([("u1", 1, "a"), ("u1", 2, "b"), ("u2", 1, "x")], "repeat")
    filler = _lists([("u1", 1, "b"), ("u1", 2, "c"), ("u1", 3, "d"),
                     ("u2", 1, "x"), ("u2", 2, "y"), ("u2", 3, "z"),
                     ("u3", 1, "p"), ("u3", 2, "q")], "popularity")
    padded = pad_list(lists, filler, 3)
    by_user = padded.groupby("user_id")["item_id"].agg(list)
    assert by_user["u1"] == ["a", "b", "c"]   # b not duplicated, c appended
    assert by_user["u2"] == ["x", "y", "z"]
    assert by_user["u3"] == ["p", "q"]        # user only in filler
    assert (padded.groupby("user_id")["rank"].agg(list).map(lambda r: r == list(range(1, len(r) + 1)))).all()
    sources = padded.set_index(["user_id", "item_id"])["source"]
    assert sources[("u1", "a")] == "repeat" and sources[("u1", "c")] == "popularity"
    assert sources[("u2", "y")] == "popularity"
    assert not padded.duplicated(["user_id", "item_id"]).any()
    assert padded.equals(padded.sort_values(["user_id", "rank"]).reset_index(drop=True))


def test_personal_frequency_list_short_history():
    pop = item_popularity(HISTORY)
    lists = personal_frequency_list(HISTORY, 3, pop)
    by_user = lists.groupby("user_id")["item_id"].agg(list)
    assert by_user["u1"] == ["a", "b", "c"]   # a (2 baskets), b (1), then popular c
    assert by_user["u2"] == ["a", "c", "b"]   # a, c tie on 1 basket -> pop desc; then b
    assert by_user["u3"] == ["d", "a", "b"]
    sources = lists.set_index(["user_id", "item_id"])["source"]
    assert sources[("u1", "a")] == "repeat" and sources[("u1", "c")] == "popularity"
    assert sources[("u3", "d")] == "repeat" and sources[("u3", "b")] == "popularity"
    assert (lists.groupby("user_id").size() == 3).all()
    assert not lists.duplicated(["user_id", "item_id"]).any()


def test_deterministic():
    pop = item_popularity(HISTORY)
    first = personal_frequency_list(HISTORY, 3, pop)
    second = personal_frequency_list(HISTORY.sample(frac=1, random_state=5), 3, pop)
    pd.testing.assert_frame_equal(first, second)


def test_no_target_argument():
    for function in (item_popularity, popularity_list, pad_list, personal_frequency_list):
        assert "target" not in inspect.signature(function).parameters


def test_k_validation():
    pop = item_popularity(HISTORY)
    with pytest.raises(ValueError):
        popularity_list(HISTORY, ["u1"], 0, pop, exclude_history=True)
