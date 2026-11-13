# *------------------- 2026 Fall Semester Machine Learning Term-project: Team 9 --------------------*
# Code First written at 2026-11-05
# Code Final Edited at 2026-11-13
# Editor: Yejun Kim, Bonjun Gu

'''
Purpose : Tests for nbr.features.repeat_features: hand-computed values, item/user rates,
          leakage.
Inputs  : Synthetic history and target frames (columns user_id, basket_seq, item_id) built
          in this file and the decay rates GAMMAS.
Outputs : pytest pass/fail results; no file is read or written.
Usage   :
    python -m pytest tests/test_features.py
'''

# ---------------------------------------------------------------------------
# imports
# ---------------------------------------------------------------------------
import math

import numpy as np
import pandas as pd
import pytest

from nbr.features import feature_names, repeat_features
from nbr.lr_repeat import build_training

# ---------------------------------------------------------------------------
# constants
# ---------------------------------------------------------------------------
GAMMAS = (0.5, 0.8, 1.0)


# ---------------------------------------------------------------------------
# fixtures and helpers
# ---------------------------------------------------------------------------
def _history(rows):
    '''Build a history frame from ``rows`` of (user_id, basket_seq, item_id) tuples.'''
    return pd.DataFrame(rows, columns=["user_id", "basket_seq", "item_id"]).astype({"basket_seq": "int32"})


def _rows(features, user):
    '''Return the rows of ``user`` in the feature frame ``features``, indexed by item_id.'''
    return features[features["user_id"] == user].set_index("item_id")


# user u, four baskets: {A,B}, {A,C}, {B,C,D}, {A,C}  (ages 3, 2, 1, 0)
USER_U = [("u", 1, "A"), ("u", 1, "B"), ("u", 2, "A"), ("u", 2, "C"), ("u", 3, "B"), ("u", 3, "C"), ("u", 3, "D"),
          ("u", 4, "A"), ("u", 4, "C"), ("u", 4, "C")]  # C repeated inside basket 4 counts once


# ---------------------------------------------------------------------------
# hand-computed feature values
# ---------------------------------------------------------------------------
def test_hand_computed_single_user():
    # Compute the features of the single user u.
    features = repeat_features(_history(USER_U), GAMMAS)
    # The frame has one float64 column per feature and one row per item, with the values below.
    assert list(features.columns) == ["user_id", "item_id"] + feature_names(GAMMAS)
    assert (features.dtypes[feature_names(GAMMAS)] == "float64").all()
    rows = _rows(features, "u")
    assert sorted(rows.index) == ["A", "B", "C", "D"]

    a = rows.loc["A"]
    assert a["dec_g0.5"] == pytest.approx(0.5**3 + 0.5**2 + 1.0)
    assert a["dec_g0.8"] == pytest.approx(0.8**3 + 0.8**2 + 1.0)
    assert a["dec_g1.0"] == 3.0
    assert a["recency"] == 0.0 and a["once"] == 0.0
    assert a["mean_gap"] == pytest.approx(1.5)  # gaps 2-1 and 4-2
    assert a["overdue"] == 0.0
    assert a["share"] == pytest.approx(3 / 4)

    b = rows.loc["B"]
    assert b["dec_g0.5"] == pytest.approx(0.5**3 + 0.5**1)
    assert b["recency"] == 1.0 and b["mean_gap"] == 2.0 and b["overdue"] == pytest.approx(0.5)
    assert b["share"] == pytest.approx(0.5)

    c = rows.loc["C"]
    assert c["dec_g1.0"] == 3.0 and c["mean_gap"] == 1.0 and c["recency"] == 0.0

    d = rows.loc["D"]
    assert d["once"] == 1.0 and d["recency"] == 1.0
    assert d["mean_gap"] == pytest.approx(1.5)  # no other buyer of D -> global median of {1.5, 2, 1}
    assert d["overdue"] == pytest.approx(1.0 / 1.5)
    assert d["share"] == pytest.approx(0.25)

    # user-level: baskets 2.. hold 7 events of which 5 are repeats (A, B, C, A, C)
    assert np.allclose(rows["user_repeat_rate"], 5 / 7)
    assert np.allclose(rows["log_hist_len"], math.log(5))
    # item-level with a single user: every item bought >= 2 times has rate 1, D has 0
    assert rows.loc[["A", "B", "C"], "item_repeat_rate"].tolist() == [1.0, 1.0, 1.0]
    assert rows.loc["D", "item_repeat_rate"] == 0.0
    assert rows.loc["A", "log_pop"] == pytest.approx(math.log(4)) and rows.loc["D", "log_pop"] == pytest.approx(math.log(2))


def test_item_and_user_rates_with_two_users():
    # user v: {D}, {E}, {D}  -> D bought twice (gap 2), E once
    rows = USER_U + [("v", 1, "D"), ("v", 2, "E"), ("v", 3, "D")]
    features = repeat_features(_history(rows), GAMMAS)
    u, v = _rows(features, "u"), _rows(features, "v")
    # The item-level and user-level features equal the values derived in the comments below.
    # item_repeat_rate(D): buyers u (once) and v (twice) -> 0.5; E: only v, once -> 0
    assert u.loc["D", "item_repeat_rate"] == 0.5 and v.loc["D", "item_repeat_rate"] == 0.5
    assert v.loc["E", "item_repeat_rate"] == 0.0
    # u's single purchase of D takes the item median over users (v's gap 2), not the global median
    assert u.loc["D", "mean_gap"] == 2.0 and u.loc["D", "overdue"] == pytest.approx(0.5)
    # v's E has no other buyer -> global median of {1.5, 2, 1, 2} = 1.75
    assert v.loc["E", "mean_gap"] == pytest.approx(1.75)
    # user_repeat_rate(v): events after basket 1 are E (new) and D (repeat) -> 0.5
    assert (v["user_repeat_rate"] == 0.5).all()
    assert np.allclose(v["log_hist_len"], math.log(4))
    assert v.loc["D", "log_pop"] == pytest.approx(math.log(1 + 3))  # D in 3 baskets overall


# ---------------------------------------------------------------------------
# caps, fallbacks and validation
# ---------------------------------------------------------------------------
def test_overdue_cap_and_single_basket_user():
    # Build user w, who buys Y in 12 baskets and X only in the first, and one-basket user z.
    rows = [("w", 1, "X"), ("w", 1, "Y")] + [("w", s, "Y") for s in range(2, 13)] + [("z", 1, "X")]
    features = repeat_features(_history(rows), GAMMAS)
    w, z = _rows(features, "w"), _rows(features, "z")
    # The overdue value of X is capped at 10 and the one-basket user has a repeat rate of 0.
    assert w.loc["Y", "mean_gap"] == 1.0 and w.loc["Y", "overdue"] == 0.0
    assert w.loc["X", "recency"] == 11.0 and w.loc["X", "mean_gap"] == 1.0  # global median
    assert w.loc["X", "overdue"] == 10.0  # 11 / 1 capped at 10
    assert (z["user_repeat_rate"] == 0.0).all() and np.allclose(z["log_hist_len"], math.log(2))
    assert z.loc["X", "share"] == 1.0


def test_all_single_purchases_fall_back_to_one():
    # Compute the features of a history in which every item is bought once.
    features = repeat_features(_history([("a", 1, "X"), ("a", 2, "Y")]), GAMMAS)
    # The mean gap is 1 and the once flag is 1 for every row.
    assert (features["mean_gap"] == 1.0).all() and (features["once"] == 1.0).all()


def test_decay_gammas_must_include_one():
    # Decay rates without 1.0 raise ValueError.
    with pytest.raises(ValueError):
        repeat_features(_history(USER_U), (0.5, 0.8))


# ---------------------------------------------------------------------------
# leakage
# ---------------------------------------------------------------------------
def test_features_do_not_depend_on_target():
    # Build the training set from the history of u and a one-row target.
    history = _history(USER_U)
    target = _history([("u", 5, "A")])
    X1, y1, g1 = build_training(history, target, GAMMAS)
    # Build it again with a target that adds a history item, an unseen item and a new user.
    extra = _history([("u", 5, "A"), ("u", 5, "B"), ("u", 5, "NEVER"), ("newuser", 5, "A")])
    X2, y2, g2 = build_training(history, extra, GAMMAS)
    # The feature rows and groups are identical and only the labels differ.
    pd.testing.assert_frame_equal(X1, X2)
    assert np.array_equal(g1, g2)
    assert y1.sum() == 1 and y2.sum() == 2  # only labels change; no row for NEVER or newuser
