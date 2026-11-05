"""Hand-computed tests for nbr.repeat on synthetic baskets."""

import inspect

import pandas as pd
import pytest

from nbr.repeat import basket_ages, rank_scores, repeat_list, repeat_scores


def _history(rows):
    frame = pd.DataFrame(rows, columns=["user_id", "basket_seq", "item_id"])
    return frame.astype({"basket_seq": "int32"})


# User u1: three baskets (seq 1, 3, 7). Items: a in all three, b in the two
# oldest, c only in the most recent (listed twice there). User u2: one basket.
HISTORY = _history([
    ("u1", 1, "a"), ("u1", 1, "b"),
    ("u1", 3, "a"), ("u1", 3, "b"),
    ("u1", 7, "a"), ("u1", 7, "c"), ("u1", 7, "c"),
    ("u2", 5, "b"),
])
POP = pd.Series({"a": 3, "b": 3, "c": 1}, name="pop")


def _scores(frame):
    return {(u, i): s for u, i, s in frame[["user_id", "item_id", "score"]].itertuples(index=False)}


def test_basket_ages():
    ages = basket_ages(HISTORY)
    assert list(ages.columns) == ["user_id", "basket_seq", "age"]
    expected = {("u1", 7): 0, ("u1", 3): 1, ("u1", 1): 2, ("u2", 5): 0}
    assert {(u, s): a for u, s, a in ages.itertuples(index=False)} == expected
    assert len(ages) == 4


def test_repeat_scores_gamma_half_hand_computed():
    scores = _scores(repeat_scores(HISTORY, 0.5))
    assert scores[("u1", "a")] == pytest.approx(1 + 0.5 + 0.25)
    assert scores[("u1", "b")] == pytest.approx(0.5 + 0.25)
    assert scores[("u1", "c")] == pytest.approx(1.0)   # duplicate in one basket counts once
    assert scores[("u2", "b")] == pytest.approx(1.0)
    assert len(scores) == 4


def test_repeat_scores_gamma_one_equals_basket_count():
    scores = _scores(repeat_scores(HISTORY, 1.0))
    assert scores == {("u1", "a"): 3.0, ("u1", "b"): 2.0, ("u1", "c"): 1.0, ("u2", "b"): 1.0}
    frame = repeat_scores(HISTORY, 1.0)
    assert frame["score"].dtype == "float64"
    assert list(frame.columns) == ["user_id", "item_id", "score"]


def test_repeat_list_order_and_tie_breaking():
    # gamma = 1: a=3, b=2, c=1 for u1 -> order a, b, c; k caps the list.
    lists = repeat_list(HISTORY, 1.0, 2, POP)
    assert list(lists.columns) == ["user_id", "rank", "item_id", "source"]
    u1 = lists[lists["user_id"] == "u1"]
    assert list(u1["item_id"]) == ["a", "b"] and list(u1["rank"]) == [1, 2]
    assert (lists["source"] == "repeat").all()
    u2 = lists[lists["user_id"] == "u2"]
    assert list(u2["item_id"]) == ["b"]  # shorter than k

    # Equal scores: pop desc, then item_id asc.
    scores = pd.DataFrame({"user_id": ["u"] * 4, "item_id": ["d", "c", "b", "a"],
                           "score": [1.0, 1.0, 1.0, 2.0]})
    pop = pd.Series({"a": 1, "b": 5, "c": 9, "d": 9})
    ranked = rank_scores(scores, pop, 4, "repeat")
    assert list(ranked["item_id"]) == ["a", "c", "d", "b"]
    assert list(ranked["rank"]) == [1, 2, 3, 4]
    # Items absent from pop count as popularity 0.
    ranked = rank_scores(scores, pd.Series({"b": 1}), 4, "repeat")
    assert list(ranked["item_id"]) == ["a", "b", "c", "d"]


def test_repeat_list_sorted_and_unique():
    lists = repeat_list(HISTORY, 0.5, 10, POP)
    assert lists.equals(lists.sort_values(["user_id", "rank"]).reset_index(drop=True))
    assert not lists.duplicated(["user_id", "item_id"]).any()
    per_user = lists.groupby("user_id")["rank"].agg(list)
    assert all(ranks == list(range(1, len(ranks) + 1)) for ranks in per_user)


def test_deterministic_and_input_untouched():
    original = HISTORY.copy()
    shuffled = HISTORY.sample(frac=1, random_state=3)
    first = repeat_list(HISTORY, 0.7, 3, POP)
    second = repeat_list(shuffled, 0.7, 3, POP)
    pd.testing.assert_frame_equal(first, second)
    pd.testing.assert_frame_equal(HISTORY, original)


def test_gamma_range():
    with pytest.raises(ValueError):
        repeat_scores(HISTORY, 1.5)


def test_no_target_argument():
    for function in (basket_ages, repeat_scores, repeat_list, rank_scores):
        assert "target" not in inspect.signature(function).parameters
