# *------------------- 2026 Fall Semester Machine Learning Term-project: Team 9 --------------------*
# Code First written at 2026-11-06
# Code Final Edited at 2026-11-13
# Editor: Yejun Kim

'''
Purpose : Hand-computed tests for nbr.day_decay (W14 post-hoc day-based recency decay) on
          synthetic baskets.
Inputs  : A synthetic history frame (columns user_id, basket_id, basket_seq, item_id), a
          basket-day frame (columns user_id, basket_id, day) and a popularity Series built
          in this file.
Outputs : pytest pass/fail results; no file is read or written.
Usage   :
    python -m pytest tests/test_day_decay.py
'''

# ---------------------------------------------------------------------------
# imports
# ---------------------------------------------------------------------------
import inspect
import math

import pandas as pd
import pytest

from nbr.day_decay import (
    basket_half_life,
    check_basket_days,
    day_decay_list,
    day_decay_scores,
    decay_rate,
    history_basket_days,
    inter_basket_gaps,
)
from nbr.repeat import repeat_list, repeat_scores


# ---------------------------------------------------------------------------
# fixtures and helpers
# ---------------------------------------------------------------------------
def _history(rows):
    '''Build a history frame from ``rows`` of (user_id, basket_id, basket_seq, item_id).'''
    frame = pd.DataFrame(rows, columns=["user_id", "basket_id", "basket_seq", "item_id"])
    return frame.astype({"basket_seq": "int32"})


def _days(rows):
    '''Build a basket-day frame from ``rows`` of (user_id, basket_id, day) tuples.'''
    return pd.DataFrame(rows, columns=["user_id", "basket_id", "day"])


# User u1 has three baskets (b1 day 0, b3 day 10, b7 day 30).
# Item a is in all three, b in the two oldest, c only in the most recent (listed twice there).
# User u2 has one basket on day 100.
HISTORY = _history([
    ("u1", "b1", 1, "a"), ("u1", "b1", 1, "b"),
    ("u1", "b3", 3, "a"), ("u1", "b3", 3, "b"),
    ("u1", "b7", 7, "a"), ("u1", "b7", 7, "c"), ("u1", "b7", 7, "c"),
    ("u2", "b5", 5, "b"),
])
DAYS = _days([("u1", "b1", 0), ("u1", "b3", 10), ("u1", "b7", 30), ("u2", "b5", 100)])
POP = pd.Series({"a": 3, "b": 3, "c": 1}, name="pop")


def _scores(frame):
    '''Map (user_id, item_id) to the score of each row of the score frame ``frame``.'''
    return {(u, i): s for u, i, s in frame[["user_id", "item_id", "score"]].itertuples(index=False)}


# ---------------------------------------------------------------------------
# decay rate and half-life conversion
# ---------------------------------------------------------------------------
def test_decay_rate():
    # An infinite half-life gives rate 1 and a finite one gives a weight of 0.5 after that time.
    assert decay_rate(math.inf) == 1.0
    assert decay_rate(10) == pytest.approx(0.5 ** 0.1)
    assert decay_rate(10) ** 10 == pytest.approx(0.5)       # weight halves after one half-life
    assert decay_rate(7.0) ** 7 == pytest.approx(0.5)
    # A half-life that is not a positive number raises ValueError.
    for bad in (0, -1, math.nan, True, "7", None):
        with pytest.raises(ValueError):
            decay_rate(bad)


def test_basket_half_life():
    # Rate 1 gives an infinite half-life and a rate g gives log(0.5) / log(g).
    assert basket_half_life(1.0) == math.inf
    assert basket_half_life(0.5) == pytest.approx(1.0)
    assert 0.97 ** basket_half_life(0.97) == pytest.approx(0.5)
    assert basket_half_life(0.9) == pytest.approx(math.log(0.5) / math.log(0.9))
    # A rate of 0, below 0 or above 1 raises ValueError.
    for bad in (0.0, -0.5, 1.5):
        with pytest.raises(ValueError):
            basket_half_life(bad)


# ---------------------------------------------------------------------------
# scores
# ---------------------------------------------------------------------------
def test_scores_hand_computed_half_life_10():
    # u1: d_last = 30; ages 30, 20, 0 days -> weights 2**-3, 2**-2, 1.
    scores = _scores(day_decay_scores(HISTORY, DAYS, 10))
    # Each score is the sum of the weights of the baskets that hold the item.
    assert scores[("u1", "a")] == pytest.approx(1 + 0.25 + 0.125)
    assert scores[("u1", "b")] == pytest.approx(0.25 + 0.125)
    assert scores[("u1", "c")] == pytest.approx(1.0)   # duplicate in one basket counts once
    assert scores[("u2", "b")] == pytest.approx(1.0)   # single basket: age 0 whatever its day
    assert len(scores) == 4
    frame = day_decay_scores(HISTORY, DAYS, 10)
    assert list(frame.columns) == ["user_id", "item_id", "score"]
    assert frame["score"].dtype == "float64"


def test_half_life_inf_equals_repeat_list_gamma_one():
    # With an infinite half-life the scores and lists equal those of nbr.repeat with gamma 1.
    pd.testing.assert_frame_equal(day_decay_scores(HISTORY, DAYS, math.inf), repeat_scores(HISTORY, 1.0))
    pd.testing.assert_frame_equal(day_decay_list(HISTORY, DAYS, math.inf, 10, POP), repeat_list(HISTORY, 1.0, 10, POP))
    pd.testing.assert_frame_equal(day_decay_list(HISTORY, DAYS, math.inf, 2, POP), repeat_list(HISTORY, 1.0, 2, POP))
    # ... whatever the days are
    other = DAYS.assign(day=[5, 5, 5, 0])
    pd.testing.assert_frame_equal(day_decay_list(HISTORY, other, math.inf, 10, POP), repeat_list(HISTORY, 1.0, 10, POP))


def test_same_day_baskets_get_equal_weight():
    # b1 and b3 on the same day (10), b7 on day 30, half-life 20: weights 0.5, 0.5, 1.
    days = DAYS.assign(day=[10, 10, 30, 100])
    scores = _scores(day_decay_scores(HISTORY, days, 20))
    # The two same-day baskets add 0.5 each to the scores of their items.
    assert scores[("u1", "b")] == pytest.approx(0.5 + 0.5)
    assert scores[("u1", "a")] == pytest.approx(0.5 + 0.5 + 1.0)
    # A second basket on the last day weighs the same as the last basket (1), independent of basket_seq.
    history = pd.concat([HISTORY, _history([("u1", "b2", 2, "d")])], ignore_index=True)
    days = pd.concat([days, _days([("u1", "b2", 30)])], ignore_index=True)
    scores = _scores(day_decay_scores(history, days, 20))
    assert scores[("u1", "d")] == pytest.approx(1.0)
    assert scores[("u1", "c")] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# ranking
# ---------------------------------------------------------------------------
def test_same_day_tie_order_follows_rank_scores_rule():
    # Add a basket b2 with item d on the last day and put b1 and b3 on the same day.
    history = pd.concat([HISTORY, _history([("u1", "b2", 2, "d")])], ignore_index=True)
    days = DAYS.assign(day=[10, 10, 30, 100])
    days = pd.concat([days, _days([("u1", "b2", 30)])], ignore_index=True)
    lists = day_decay_list(history, days, 20, 10, POP)
    u1 = list(lists.loc[lists["user_id"] == "u1", "item_id"])
    # scores: a 2.0, b 1.0 (pop 3), c 1.0 (pop 1), d 1.0 (pop 0): score desc, pop desc, item_id asc
    assert u1 == ["a", "b", "c", "d"]


def test_shifting_all_days_does_not_change_the_ranking():
    # Compute the reference lists and scores for a half-life of 10 days.
    reference_lists = day_decay_list(HISTORY, DAYS, 10, 10, POP)
    reference_scores = day_decay_scores(HISTORY, DAYS, 10)
    # Adding the same number of days to every basket gives the same lists and scores.
    for shift in (7, 1000, -3, 10 ** 6):
        shifted = DAYS.assign(day=DAYS["day"] + shift)
        pd.testing.assert_frame_equal(day_decay_list(HISTORY, shifted, 10, 10, POP), reference_lists)
        pd.testing.assert_frame_equal(day_decay_scores(HISTORY, shifted, 10), reference_scores)
    # a per-user shift (only u2 moved) changes nothing either: ages are relative to each user's last basket
    shifted = DAYS.assign(day=[0, 10, 30, 5])
    pd.testing.assert_frame_equal(day_decay_list(HISTORY, shifted, 10, 10, POP), reference_lists)


def test_longer_half_life_moves_towards_frequency():
    # With a short half-life the recent-only item c ranks above b (bought twice, long ago); with h = inf b wins.
    short = day_decay_list(HISTORY, DAYS, 5, 10, POP)
    long = day_decay_list(HISTORY, DAYS, math.inf, 10, POP)
    short_u1 = list(short.loc[short["user_id"] == "u1", "item_id"])
    long_u1 = list(long.loc[long["user_id"] == "u1", "item_id"])
    # The order of b and c differs between the two lists of u1.
    assert short_u1 == ["a", "c", "b"]
    assert long_u1 == ["a", "b", "c"]


def test_list_schema_and_length_rule():
    # Build lists of at most two items with a half-life of 30 days.
    lists = day_decay_list(HISTORY, DAYS, 30, 2, POP)
    # The lists have the four list columns, the repeat source and only history items.
    assert list(lists.columns) == ["user_id", "rank", "item_id", "source"]
    assert (lists["source"] == "repeat").all()
    assert lists.groupby("user_id")["rank"].agg(list).to_dict() == {"u1": [1, 2], "u2": [1]}   # no padding
    assert not lists.duplicated(["user_id", "item_id"]).any()


# ---------------------------------------------------------------------------
# basket days and gaps
# ---------------------------------------------------------------------------
def test_history_basket_days_and_errors():
    # Join the day of every history basket.
    merged = history_basket_days(HISTORY, DAYS)
    # The result has one row per basket with an int64 day.
    assert list(merged.columns) == ["user_id", "basket_id", "basket_seq", "day"]
    assert len(merged) == 4 and merged["day"].dtype == "int64"
    # Each invalid input below raises ValueError.
    with pytest.raises(ValueError):   # u2's basket has no day
        history_basket_days(HISTORY, DAYS[DAYS["user_id"] != "u2"])
    with pytest.raises(ValueError):   # duplicated (user_id, basket_id)
        history_basket_days(HISTORY, pd.concat([DAYS, DAYS.iloc[:1]]))
    with pytest.raises(ValueError):   # null day
        history_basket_days(HISTORY, DAYS.assign(day=[0, None, 30, 100]))
    with pytest.raises(ValueError):   # missing column
        check_basket_days(DAYS.drop(columns="day"))
    with pytest.raises(ValueError):   # a basket_id mapping to two basket_seq values
        history_basket_days(pd.concat([HISTORY, _history([("u1", "b1", 9, "z")])]), DAYS)


def test_string_dtype_ids_are_preserved():
    # Cast the id columns of the history to the pandas string dtype.
    history = HISTORY.astype({"user_id": "string", "basket_id": "string", "item_id": "string"})
    scores = day_decay_scores(history, DAYS, 10)   # days carry object ids
    # The scores keep the id dtype and equal those of the object-dtype history.
    assert scores["user_id"].dtype == history["user_id"].dtype
    assert _scores(scores) == pytest.approx(_scores(day_decay_scores(HISTORY, DAYS, 10)))
    pd.testing.assert_frame_equal(day_decay_scores(history, DAYS, math.inf), repeat_scores(history, 1.0))


def test_inter_basket_gaps():
    # Compute the day gaps between consecutive baskets of each user.
    gaps = inter_basket_gaps(HISTORY, DAYS)
    # The gaps of u1 are returned as int64, a same-day pair gives 0 and one basket gives no gap.
    assert list(gaps) == [10, 20] and gaps.dtype == "int64"   # u2 has one basket: no gap
    same_day = inter_basket_gaps(HISTORY, DAYS.assign(day=[10, 10, 30, 100]))
    assert list(same_day) == [0, 20]
    assert len(inter_basket_gaps(HISTORY[HISTORY["user_id"] == "u2"], DAYS)) == 0


# ---------------------------------------------------------------------------
# determinism and signatures
# ---------------------------------------------------------------------------
def test_deterministic_and_input_untouched():
    # Keep copies of the inputs and shuffle the rows of both frames.
    original_history, original_days = HISTORY.copy(), DAYS.copy()
    shuffled = HISTORY.sample(frac=1, random_state=3)
    shuffled_days = DAYS.sample(frac=1, random_state=5)
    # Build the lists from the original and from the shuffled frames.
    first = day_decay_list(HISTORY, DAYS, 14, 3, POP)
    second = day_decay_list(shuffled, shuffled_days, 14, 3, POP)
    # Both lists are equal and the input frames are unchanged.
    pd.testing.assert_frame_equal(first, second)
    pd.testing.assert_frame_equal(HISTORY, original_history)
    pd.testing.assert_frame_equal(DAYS, original_days)


def test_no_target_argument():
    # No day-decay function has a parameter named target.
    for function in (day_decay_scores, day_decay_list, history_basket_days, inter_basket_gaps):
        assert "target" not in inspect.signature(function).parameters
