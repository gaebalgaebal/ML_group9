# *------------------- 2026 Fall Semester Machine Learning Term-project: Team 9 --------------------*
# Code First written at 2026-11-06
# Code Final Edited at 2026-11-13
# Editor: Yejun Kim, Bonjun Gu

'''
Purpose : Tests for nbr.features_cycle: hand-computed cadence values, bin boundaries, once
          handling, reference bin, leakage.
Inputs  : Small hand-written purchase histories (columns user_id, basket_seq, item_id) and a
          bin configuration (overdue_bins, recency_bins, item_cadence_ratio_bins).
Outputs : None; the tests assert on the names, bin indices and feature frames returned by
          nbr.features_cycle and on the training frames of nbr.r3_eval.build_training_r3.
Usage   :
    python -m pytest tests/test_features_cycle.py
'''

# ---------------------------------------------------------------------------
# imports
# ---------------------------------------------------------------------------
import numpy as np
import pandas as pd
import pytest

from nbr.features import feature_names, repeat_features
from nbr.features_cycle import (
    ONCE_COLUMN,
    bin_columns,
    bin_index,
    bin_labels,
    cycle_feature_names,
    cycle_features,
    global_cadence,
    item_cadence,
)

# ---------------------------------------------------------------------------
# constants
# ---------------------------------------------------------------------------
GAMMAS = (0.5, 0.8, 1.0)
CFG = {
    "overdue_bins": [0.0, 0.5, 1.0, 1.5, 2.5, 5.0, 10.01],
    "recency_bins": [0, 1, 2, 3, 5, 10, 1000000],
    "item_cadence_ratio_bins": [0.0, 0.5, 1.0, 1.5, 2.5, 5.0, 10.01],
}
OB = ["ob_0.5_1", "ob_1_1.5", "ob_1.5_2.5", "ob_2.5_5", "ob_5_10.01"]
RB = ["rb_1_2", "rb_2_3", "rb_3_5", "rb_5_10", "rb_10_1000000"]
CB = ["cb_0.5_1", "cb_1_1.5", "cb_1.5_2.5", "cb_2.5_5", "cb_5_10.01"]

# user u, four baskets {A,B}, {A,C}, {B,C,D}, {A,C}; user v: {D}, {E}, {D}
ROWS = [("u", 1, "A"), ("u", 1, "B"), ("u", 2, "A"), ("u", 2, "C"), ("u", 3, "B"), ("u", 3, "C"), ("u", 3, "D"),
        ("u", 4, "A"), ("u", 4, "C"), ("v", 1, "D"), ("v", 2, "E"), ("v", 3, "D")]


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _history(rows):
    '''Build a history frame (user_id, basket_seq, item_id) from the tuples in ``rows``.'''
    return pd.DataFrame(rows, columns=["user_id", "basket_seq", "item_id"]).astype({"basket_seq": "int32"})


def _row(features, user, item):
    '''Return the row of ``features`` that belongs to the pair (``user``, ``item``).'''
    return features[(features["user_id"] == user) & (features["item_id"] == item)].iloc[0]


def _dummies(row, columns):
    '''Return the names in ``columns`` whose value in ``row`` equals 1.0.'''
    return [c for c in columns if row[c] == 1.0]


# ---------------------------------------------------------------------------
# names and bin helpers
# ---------------------------------------------------------------------------
def test_feature_names_extend_r2_in_order():
    # Build the feature names of the test configuration.
    names = cycle_feature_names(CFG, GAMMAS)
    # Expect the names of the logistic-regression repeat ranker (R2) first, then the cycle names.
    assert names[: len(feature_names(GAMMAS))] == feature_names(GAMMAS)
    assert names[len(feature_names(GAMMAS)):] == ["item_cadence", "cadence_ratio"] + OB + [ONCE_COLUMN] + RB + CB
    assert len(names) == 30 and len(set(names)) == 30


def test_bin_columns_drop_reference_and_labels_cover_all_bins():
    # Expect dummy columns for every bin except the first and one label per bin.
    assert bin_columns("ob", CFG["overdue_bins"]) == OB
    assert bin_columns("rb", CFG["recency_bins"]) == RB
    assert bin_labels(CFG["overdue_bins"]) == ["[0, 0.5)", "[0.5, 1)", "[1, 1.5)", "[1.5, 2.5)", "[2.5, 5)", "[5, 10.01)"]


def test_bin_index_right_open_boundaries():
    # Take values on and next to the edges of the overdue bins.
    edges = CFG["overdue_bins"]
    values = [0.0, 0.49, 0.5, 0.99, 1.0, 1.5, 2.5, 4.99, 5.0, 10.0]
    # Expect a value on an edge to fall into the bin that starts at that edge.
    assert bin_index(np.array(values), edges).tolist() == [0, 0, 1, 1, 2, 3, 4, 4, 5, 5]
    recency = CFG["recency_bins"]
    assert bin_index(np.array([0, 1, 2, 3, 4, 5, 9, 10, 11, 500]), recency).tolist() == [0, 1, 2, 3, 3, 4, 4, 5, 5, 5]
    # Expect ValueError for a value outside the range of the edges.
    with pytest.raises(ValueError):
        bin_index(np.array([10.01]), edges)  # at the last edge: outside the right-open range
    with pytest.raises(ValueError):
        bin_index(np.array([-0.1]), edges)


# ---------------------------------------------------------------------------
# hand-computed cadence features
# ---------------------------------------------------------------------------
def test_item_cadence_and_ratio_by_hand():
    # Compute the R2 features of the two-user history.
    history = _history(ROWS)
    base = repeat_features(history, GAMMAS)
    # repeated pairs and their mean_gap: (u,A) 1.5, (u,B) 2, (u,C) 1, (v,D) 2 -> item medians A 1.5, B 2, C 1, D 2
    cadence = item_cadence(base)
    assert cadence.to_dict() == {"A": 1.5, "B": 2.0, "C": 1.0, "D": 2.0}
    assert global_cadence(base) == pytest.approx(1.75)  # median of {1.5, 2, 1, 2}

    # Compute the cycle features and expect the hand-computed cadence and ratio of four pairs.
    features = cycle_features(history, CFG, GAMMAS)
    assert list(features.columns) == ["user_id", "item_id"] + cycle_feature_names(CFG, GAMMAS)
    assert (features.dtypes[cycle_feature_names(CFG, GAMMAS)] == "float64").all()
    # the R2 columns are unchanged
    pd.testing.assert_frame_equal(features[base.columns], base)

    u_d = _row(features, "u", "D")  # bought once by u; v repeated it with gap 2
    assert u_d["item_cadence"] == 2.0 and u_d["recency"] == 1.0
    assert u_d["cadence_ratio"] == pytest.approx(0.5)
    v_e = _row(features, "v", "E")  # nobody repeated E -> global median
    assert v_e["item_cadence"] == pytest.approx(1.75)
    assert v_e["cadence_ratio"] == pytest.approx(1.0 / 1.75)
    u_b = _row(features, "u", "B")
    assert u_b["item_cadence"] == 2.0 and u_b["cadence_ratio"] == pytest.approx(0.5)
    u_a = _row(features, "u", "A")
    assert u_a["item_cadence"] == 1.5 and u_a["cadence_ratio"] == 0.0


def test_cadence_ratio_is_capped_and_global_fallback_is_one_without_repeats():
    # Build a user w who buys X once in basket 1 and Y in each of 29 baskets.
    rows = [("w", 1, "X"), ("w", 1, "Y")] + [("w", s, "Y") for s in range(2, 30)] + [("z", 1, "X")]
    features = cycle_features(_history(rows), CFG, GAMMAS)
    # Expect the ratio of (w, X) to be capped at 10 and its dummies in the last bins.
    w_x = _row(features, "w", "X")
    assert w_x["recency"] == 28.0 and w_x["item_cadence"] == 1.0  # global median over the only repeated pair (Y, gap 1)
    assert w_x["cadence_ratio"] == 10.0 and w_x["overdue"] == 10.0
    assert w_x["once"] == 1.0 and w_x[ONCE_COLUMN] == 1.0 and _dummies(w_x, OB) == []  # once: no ob_* bin
    assert _dummies(w_x, CB) == ["cb_5_10.01"] and _dummies(w_x, RB) == ["rb_10_1000000"]
    w_y = _row(features, "w", "Y")
    assert w_y["overdue"] == 0.0 and _dummies(w_y, OB + [ONCE_COLUMN] + RB + CB) == []
    # Expect an item cadence of 1.0 for a history without any repeated pair.
    only_once = cycle_features(_history([("a", 1, "X"), ("a", 2, "Y")]), CFG, GAMMAS)
    assert (only_once["item_cadence"] == 1.0).all()


# ---------------------------------------------------------------------------
# dummies: boundaries, once handling, reference bin
# ---------------------------------------------------------------------------
def test_dummies_once_and_reference():
    # Compute the cycle features of the two-user history.
    features = cycle_features(_history(ROWS), CFG, GAMMAS)
    # Expect the active dummies of three pairs of user u as noted on each row.
    u_a = _row(features, "u", "A")  # overdue 0, recency 0, ratio 0 -> every group in its reference bin
    assert _dummies(u_a, OB + [ONCE_COLUMN] + RB + CB) == []
    u_b = _row(features, "u", "B")  # overdue 0.5 -> second bin (boundary value belongs to the upper bin)
    assert _dummies(u_b, OB) == ["ob_0.5_1"] and _dummies(u_b, RB) == ["rb_1_2"] and _dummies(u_b, CB) == ["cb_0.5_1"]
    assert u_b[ONCE_COLUMN] == 0.0
    u_d = _row(features, "u", "D")  # once -> ob_once only, no ob_* bin even though overdue = 0.5
    assert u_d["once"] == 1.0 and u_d[ONCE_COLUMN] == 1.0 and _dummies(u_d, OB) == []
    assert _dummies(u_d, RB) == ["rb_1_2"] and _dummies(u_d, CB) == ["cb_0.5_1"]
    # every row of a group has at most one active dummy
    for group in (OB, RB, CB):
        assert features[group].sum(axis=1).le(1.0).all()
    assert ((features[OB].sum(axis=1) == 0.0) | (features[ONCE_COLUMN] == 0.0)).all()


def test_recency_bins_edges_through_features():
    # item Q bought in basket 1 only by a user with 12 baskets -> recency 11; others give recency 0, 1, 2, 3, 4, 5, 9, 10
    rows = []
    for recency in (0, 1, 2, 3, 4, 5, 9, 10, 11):
        user = f"r{recency}"
        rows += [(user, s, "F") for s in range(1, 13)]  # filler item in every basket
        rows.append((user, 12 - recency, "Q"))
    features = cycle_features(_history(rows), CFG, GAMMAS)
    # Expect each recency value to activate the dummy of the bin that contains it.
    expected = {0: [], 1: ["rb_1_2"], 2: ["rb_2_3"], 3: ["rb_3_5"], 4: ["rb_3_5"], 5: ["rb_5_10"], 9: ["rb_5_10"],
                10: ["rb_10_1000000"], 11: ["rb_10_1000000"]}
    for recency, dummies in expected.items():
        row = _row(features, f"r{recency}", "Q")
        assert row["recency"] == float(recency) and _dummies(row, RB) == dummies


def test_every_dummy_column_exists_even_when_its_bin_is_empty():
    # Compute the features of one user who buys one item in two baskets.
    features = cycle_features(_history([("a", 1, "X"), ("a", 2, "X")]), CFG, GAMMAS)
    # Expect every dummy column to be present and the bin dummies to be 0.
    assert set(OB + [ONCE_COLUMN] + RB + CB) <= set(features.columns)
    assert (features[RB + CB + OB] == 0.0).all().all()


def test_invalid_bins_rejected():
    # Expect ValueError for overdue edges that are not increasing.
    with pytest.raises(ValueError):
        cycle_features(_history(ROWS), {**CFG, "overdue_bins": [0.0, 1.0, 0.5]}, GAMMAS)
    # Expect ValueError for recency edges that do not cover the recency values of the history.
    with pytest.raises(ValueError):
        cycle_features(_history(ROWS), {**CFG, "recency_bins": [0, 1]}, GAMMAS)  # recency 1 falls outside [0, 1)


# ---------------------------------------------------------------------------
# leakage: the target never changes a feature
# ---------------------------------------------------------------------------
def test_features_do_not_depend_on_target():
    from nbr.r3_eval import build_training_r3

    # Build the training rows of the cycle-feature repeat ranker (R3) with a one-row target.
    history = _history(ROWS)
    X1, y1, g1 = build_training_r3(history, _history([("u", 5, "A")]), CFG, GAMMAS)
    # Build them again with a larger target that adds an unseen item and a new user.
    extra = _history([("u", 5, "A"), ("u", 5, "B"), ("u", 5, "NEVER"), ("v", 5, "E"), ("new", 5, "A")])
    X2, y2, g2 = build_training_r3(history, extra, CFG, GAMMAS)
    # Expect identical features and groups for both targets.
    pd.testing.assert_frame_equal(X1, X2)
    assert np.array_equal(g1, g2)
    assert y1.sum() == 1 and y2.sum() == 3  # only labels change; no row for NEVER or the new user
    assert list(X1.columns) == cycle_feature_names(CFG, GAMMAS)
