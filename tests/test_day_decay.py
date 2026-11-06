"""Hand-computed tests for nbr.day_decay (W14 post-hoc day-based recency decay) on synthetic baskets."""

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


def _history(rows):
    frame = pd.DataFrame(rows, columns=["user_id", "basket_id", "basket_seq", "item_id"])
    return frame.astype({"basket_seq": "int32"})


def _days(rows):
    return pd.DataFrame(rows, columns=["user_id", "basket_id", "day"])


# User u1: three baskets (b1 day 0, b3 day 10, b7 day 30). Items: a in all three, b in the two
# oldest, c only in the most recent (listed twice there). User u2: one basket on day 100.
HISTORY = _history([
    ("u1", "b1", 1, "a"), ("u1", "b1", 1, "b"),
    ("u1", "b3", 3, "a"), ("u1", "b3", 3, "b"),
    ("u1", "b7", 7, "a"), ("u1", "b7", 7, "c"), ("u1", "b7", 7, "c"),
    ("u2", "b5", 5, "b"),
])
DAYS = _days([("u1", "b1", 0), ("u1", "b3", 10), ("u1", "b7", 30), ("u2", "b5", 100)])
POP = pd.Series({"a": 3, "b": 3, "c": 1}, name="pop")


def _scores(frame):
    return {(u, i): s for u, i, s in frame[["user_id", "item_id", "score"]].itertuples(index=False)}


def test_decay_rate():
    assert decay_rate(math.inf) == 1.0
    assert decay_rate(10) == pytest.approx(0.5 ** 0.1)
    assert decay_rate(10) ** 10 == pytest.approx(0.5)       # weight halves after one half-life
    assert decay_rate(7.0) ** 7 == pytest.approx(0.5)
    for bad in (0, -1, math.nan, True, "7", None):
        with pytest.raises(ValueError):
            decay_rate(bad)


def test_basket_half_life():
    assert basket_half_life(1.0) == math.inf
    assert basket_half_life(0.5) == pytest.approx(1.0)
    assert 0.97 ** basket_half_life(0.97) == pytest.approx(0.5)
    assert basket_half_life(0.9) == pytest.approx(math.log(0.5) / math.log(0.9))
    for bad in (0.0, -0.5, 1.5):
        with pytest.raises(ValueError):
            basket_half_life(bad)


def test_scores_hand_computed_half_life_10():
    # u1: d_last = 30; ages 30, 20, 0 days -> weights 2**-3, 2**-2, 1.
    scores = _scores(day_decay_scores(HISTORY, DAYS, 10))
    assert scores[("u1", "a")] == pytest.approx(1 + 0.25 + 0.125)
    assert scores[("u1", "b")] == pytest.approx(0.25 + 0.125)
    assert scores[("u1", "c")] == pytest.approx(1.0)   # duplicate in one basket counts once
    assert scores[("u2", "b")] == pytest.approx(1.0)   # single basket: age 0 whatever its day
    assert len(scores) == 4
    frame = day_decay_scores(HISTORY, DAYS, 10)
    assert list(frame.columns) == ["user_id", "item_id", "score"]
    assert frame["score"].dtype == "float64"


def test_half_life_inf_equals_repeat_list_gamma_one():
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
    assert scores[("u1", "b")] == pytest.approx(0.5 + 0.5)
    assert scores[("u1", "a")] == pytest.approx(0.5 + 0.5 + 1.0)
    # A second basket on the last day weighs the same as the last basket (1), independent of basket_seq.
    history = pd.concat([HISTORY, _history([("u1", "b2", 2, "d")])], ignore_index=True)
    days = pd.concat([days, _days([("u1", "b2", 30)])], ignore_index=True)
    scores = _scores(day_decay_scores(history, days, 20))
    assert scores[("u1", "d")] == pytest.approx(1.0)
    assert scores[("u1", "c")] == pytest.approx(1.0)


def test_same_day_tie_order_follows_rank_scores_rule():
    history = pd.concat([HISTORY, _history([("u1", "b2", 2, "d")])], ignore_index=True)
    days = DAYS.assign(day=[10, 10, 30, 100])
    days = pd.concat([days, _days([("u1", "b2", 30)])], ignore_index=True)
    lists = day_decay_list(history, days, 20, 10, POP)
    u1 = list(lists.loc[lists["user_id"] == "u1", "item_id"])
    # scores: a 2.0, b 1.0 (pop 3), c 1.0 (pop 1), d 1.0 (pop 0): score desc, pop desc, item_id asc
    assert u1 == ["a", "b", "c", "d"]


def test_shifting_all_days_does_not_change_the_ranking():
    reference_lists = day_decay_list(HISTORY, DAYS, 10, 10, POP)
    reference_scores = day_decay_scores(HISTORY, DAYS, 10)
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
    assert short_u1 == ["a", "c", "b"]
    assert long_u1 == ["a", "b", "c"]


def test_list_schema_and_length_rule():
    lists = day_decay_list(HISTORY, DAYS, 30, 2, POP)
    assert list(lists.columns) == ["user_id", "rank", "item_id", "source"]
    assert (lists["source"] == "repeat").all()
    assert lists.groupby("user_id")["rank"].agg(list).to_dict() == {"u1": [1, 2], "u2": [1]}   # no padding
    assert not lists.duplicated(["user_id", "item_id"]).any()


def test_history_basket_days_and_errors():
    merged = history_basket_days(HISTORY, DAYS)
    assert list(merged.columns) == ["user_id", "basket_id", "basket_seq", "day"]
    assert len(merged) == 4 and merged["day"].dtype == "int64"
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
    history = HISTORY.astype({"user_id": "string", "basket_id": "string", "item_id": "string"})
    scores = day_decay_scores(history, DAYS, 10)   # days carry object ids
    assert scores["user_id"].dtype == history["user_id"].dtype
    assert _scores(scores) == pytest.approx(_scores(day_decay_scores(HISTORY, DAYS, 10)))
    pd.testing.assert_frame_equal(day_decay_scores(history, DAYS, math.inf), repeat_scores(history, 1.0))


def test_inter_basket_gaps():
    gaps = inter_basket_gaps(HISTORY, DAYS)
    assert list(gaps) == [10, 20] and gaps.dtype == "int64"   # u2 has one basket: no gap
    same_day = inter_basket_gaps(HISTORY, DAYS.assign(day=[10, 10, 30, 100]))
    assert list(same_day) == [0, 20]
    assert len(inter_basket_gaps(HISTORY[HISTORY["user_id"] == "u2"], DAYS)) == 0


def test_deterministic_and_input_untouched():
    original_history, original_days = HISTORY.copy(), DAYS.copy()
    shuffled = HISTORY.sample(frac=1, random_state=3)
    shuffled_days = DAYS.sample(frac=1, random_state=5)
    first = day_decay_list(HISTORY, DAYS, 14, 3, POP)
    second = day_decay_list(shuffled, shuffled_days, 14, 3, POP)
    pd.testing.assert_frame_equal(first, second)
    pd.testing.assert_frame_equal(HISTORY, original_history)
    pd.testing.assert_frame_equal(DAYS, original_days)


def test_no_target_argument():
    for function in (day_decay_scores, day_decay_list, history_basket_days, inter_basket_gaps):
        assert "target" not in inspect.signature(function).parameters
