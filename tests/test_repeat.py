# *------------------- 2026 Fall Semester Machine Learning Term-project: Team 9 --------------------*
# Code First written at 2026-11-05
# Code Final Edited at 2026-11-13
# Editor: Yejun Kim, Bonjun Gu

'''
Purpose : Hand-computed tests for nbr.repeat on synthetic baskets.
Inputs  : A hand-built history frame (columns user_id, basket_seq, item_id) of two users and
          a hand-built item popularity Series.
Outputs : Pytest pass / fail results; no files are written.
Usage   :
    python -m pytest tests/test_repeat.py
'''

# ---------------------------------------------------------------------------
# imports
# ---------------------------------------------------------------------------
import inspect

import pandas as pd
import pytest

from nbr.repeat import basket_ages, rank_scores, repeat_list, repeat_scores


# ---------------------------------------------------------------------------
# fixtures and helpers
# ---------------------------------------------------------------------------
def _history(rows):
    '''Build a history frame from ``rows`` = (user_id, basket_seq, item_id).'''
    # Build the frame and cast basket_seq to int32.
    frame = pd.DataFrame(rows, columns=["user_id", "basket_seq", "item_id"])
    return frame.astype({"basket_seq": "int32"})


# User u1 has three baskets (seq 1, 3, 7).
# Item a is in all three, b in the two oldest, c only in the most recent (listed twice there).
# User u2 has one basket.
HISTORY = _history([
    ("u1", 1, "a"), ("u1", 1, "b"),
    ("u1", 3, "a"), ("u1", 3, "b"),
    ("u1", 7, "a"), ("u1", 7, "c"), ("u1", 7, "c"),
    ("u2", 5, "b"),
])
# Item popularity used to break ties between equal scores.
POP = pd.Series({"a": 3, "b": 3, "c": 1}, name="pop")


def _scores(frame):
    '''Map (user_id, item_id) -> score for the rows of the score ``frame``.'''
    return {(u, i): s for u, i, s in frame[["user_id", "item_id", "score"]].itertuples(index=False)}


# ---------------------------------------------------------------------------
# basket ages and repeat scores
# ---------------------------------------------------------------------------
def test_basket_ages():
    # Compute the basket ages of the shared history.
    ages = basket_ages(HISTORY)
    # Expect age 0 for each user's most recent basket and one row per basket.
    assert list(ages.columns) == ["user_id", "basket_seq", "age"]
    expected = {("u1", 7): 0, ("u1", 3): 1, ("u1", 1): 2, ("u2", 5): 0}
    assert {(u, s): a for u, s, a in ages.itertuples(index=False)} == expected
    assert len(ages) == 4


def test_repeat_scores_gamma_half_hand_computed():
    # Score the shared history with gamma = 0.5.
    scores = _scores(repeat_scores(HISTORY, 0.5))
    # Expect each score to be the sum of 0.5 ** age over the baskets that hold the item.
    assert scores[("u1", "a")] == pytest.approx(1 + 0.5 + 0.25)
    assert scores[("u1", "b")] == pytest.approx(0.5 + 0.25)
    assert scores[("u1", "c")] == pytest.approx(1.0)   # duplicate in one basket counts once
    assert scores[("u2", "b")] == pytest.approx(1.0)
    assert len(scores) == 4


def test_repeat_scores_gamma_one_equals_basket_count():
    # Score the shared history with gamma = 1.
    scores = _scores(repeat_scores(HISTORY, 1.0))
    # Expect each score to equal the number of baskets that hold the item.
    assert scores == {("u1", "a"): 3.0, ("u1", "b"): 2.0, ("u1", "c"): 1.0, ("u2", "b"): 1.0}
    # Expect a float64 score column and the three output columns.
    frame = repeat_scores(HISTORY, 1.0)
    assert frame["score"].dtype == "float64"
    assert list(frame.columns) == ["user_id", "item_id", "score"]


# ---------------------------------------------------------------------------
# repeat lists and ranking
# ---------------------------------------------------------------------------
def test_repeat_list_order_and_tie_breaking():
    # Build the top-2 repeat lists of the shared history with gamma = 1.
    # gamma = 1: a=3, b=2, c=1 for u1 -> order a, b, c; k caps the list.
    lists = repeat_list(HISTORY, 1.0, 2, POP)
    # Expect the items in score order with ranks 1, 2 and the source name repeat.
    assert list(lists.columns) == ["user_id", "rank", "item_id", "source"]
    u1 = lists[lists["user_id"] == "u1"]
    assert list(u1["item_id"]) == ["a", "b"] and list(u1["rank"]) == [1, 2]
    assert (lists["source"] == "repeat").all()
    u2 = lists[lists["user_id"] == "u2"]
    assert list(u2["item_id"]) == ["b"]  # shorter than k

    # Build four scored items of which three share the same score.
    # Equal scores: pop desc, then item_id asc.
    scores = pd.DataFrame({"user_id": ["u"] * 4, "item_id": ["d", "c", "b", "a"],
                           "score": [1.0, 1.0, 1.0, 2.0]})
    pop = pd.Series({"a": 1, "b": 5, "c": 9, "d": 9})
    # Expect the highest score first and the ties ordered by popularity, then by item_id.
    ranked = rank_scores(scores, pop, 4, "repeat")
    assert list(ranked["item_id"]) == ["a", "c", "d", "b"]
    assert list(ranked["rank"]) == [1, 2, 3, 4]
    # Items absent from pop count as popularity 0.
    ranked = rank_scores(scores, pd.Series({"b": 1}), 4, "repeat")
    assert list(ranked["item_id"]) == ["a", "b", "c", "d"]


def test_repeat_list_sorted_and_unique():
    # Build the repeat lists of the shared history with gamma = 0.5 and k = 10.
    lists = repeat_list(HISTORY, 0.5, 10, POP)
    # Expect rows sorted by user and rank, unique items per user and ranks 1..n.
    assert lists.equals(lists.sort_values(["user_id", "rank"]).reset_index(drop=True))
    assert not lists.duplicated(["user_id", "item_id"]).any()
    per_user = lists.groupby("user_id")["rank"].agg(list)
    assert all(ranks == list(range(1, len(ranks) + 1)) for ranks in per_user)


# ---------------------------------------------------------------------------
# determinism and argument checks
# ---------------------------------------------------------------------------
def test_deterministic_and_input_untouched():
    # Build the repeat lists from the shared history and from a row-shuffled copy.
    original = HISTORY.copy()
    shuffled = HISTORY.sample(frac=1, random_state=3)
    first = repeat_list(HISTORY, 0.7, 3, POP)
    second = repeat_list(shuffled, 0.7, 3, POP)
    # Expect equal lists for both row orders and an unchanged input frame.
    pd.testing.assert_frame_equal(first, second)
    pd.testing.assert_frame_equal(HISTORY, original)


def test_gamma_range():
    # Expect ValueError for a gamma of 1.5.
    with pytest.raises(ValueError):
        repeat_scores(HISTORY, 1.5)


def test_no_target_argument():
    # Expect none of the four public functions to take a parameter named target.
    for function in (basket_ages, repeat_scores, repeat_list, rank_scores):
        assert "target" not in inspect.signature(function).parameters
