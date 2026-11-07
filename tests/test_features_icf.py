"""Tests for nbr.features_icf (PREREG v1.5 item-CF feature of R2'): hand-computed toy example, zero cases, block independence."""

import inspect
import math

import numpy as np
import pandas as pd
import pytest

from nbr.features_icf import COLUMNS, icf_feature
from nbr.itemcf import item_neighbors


def _history(rows):
    return pd.DataFrame(rows, columns=["user_id", "basket_seq", "item_id"]).astype({"basket_seq": "int32"})


# Binary user x item matrix (3 users x 4 items):
#   u1: a, b        u2: a, b, c        u3: a, d
# |U_a| = 3, |U_b| = 2, |U_c| = 1, |U_d| = 1
# cos(a, b) = 2/sqrt(6), cos(a, c) = 1/sqrt(3), cos(a, d) = 1/sqrt(3), cos(b, c) = 1/sqrt(2), cos(b, d) = cos(c, d) = 0
TOY = _history([
    ("u1", 1, "a"), ("u1", 1, "b"),
    ("u2", 1, "a"), ("u2", 2, "b"), ("u2", 2, "c"), ("u2", 2, "a"),  # a twice: counts once
    ("u3", 1, "a"), ("u3", 2, "d"),
])
AB, AC, BC = 2 / math.sqrt(6), 1 / math.sqrt(3), 1 / math.sqrt(2)


def _values(frame):
    return {(u, i): v for u, i, v in frame.itertuples(index=False)}


def test_toy_example_all_neighbours():
    out = icf_feature(TOY, n_neighbors=20)
    assert list(out.columns) == COLUMNS == ["user_id", "item_id", "icf"]
    assert out["icf"].dtype == "float64"
    assert out[["user_id", "item_id"]].values.tolist() == [["u1", "a"], ["u1", "b"], ["u2", "a"], ["u2", "b"], ["u2", "c"], ["u3", "a"], ["u3", "d"]]
    values = _values(out)
    expected = {
        ("u1", "a"): math.log1p(AB), ("u1", "b"): math.log1p(AB),
        ("u2", "a"): math.log1p(AB + AC), ("u2", "b"): math.log1p(AB + BC), ("u2", "c"): math.log1p(AC + BC),
        ("u3", "a"): math.log1p(AC), ("u3", "d"): math.log1p(AC),
    }
    assert set(values) == set(expected)
    for key, value in expected.items():
        assert values[key] == pytest.approx(value, rel=1e-6), key


def test_toy_example_truncated_neighbour_lists():
    # n_neighbors = 1: a -> {b}, b -> {a}, c -> {b}, d -> {a}; sim(j, i) is taken from j's stored list only.
    out = icf_feature(TOY, n_neighbors=1)
    values = _values(out)
    assert values[("u1", "a")] == pytest.approx(math.log1p(AB), rel=1e-6)   # b's list contains a
    assert values[("u1", "b")] == pytest.approx(math.log1p(AB), rel=1e-6)   # a's list contains b
    assert values[("u2", "a")] == pytest.approx(math.log1p(AB), rel=1e-6)   # b -> a yes, c -> a no (c's list is {b})
    assert values[("u2", "b")] == pytest.approx(math.log1p(AB + BC), rel=1e-6)  # a -> b and c -> b
    assert values[("u2", "c")] == 0.0                                        # c is in no stored list of a or b
    assert values[("u3", "a")] == pytest.approx(math.log1p(AC), rel=1e-6)   # d -> a
    assert values[("u3", "d")] == 0.0                                        # a's list is {b}


def test_single_item_user_and_lonely_history():
    history = _history([*TOY.itertuples(index=False), ("u4", 1, "a"), ("u4", 2, "a")])
    out = icf_feature(history, n_neighbors=20)
    assert _values(out)[("u4", "a")] == 0.0
    # A history with one item has no neighbour at all: icf = 0 for every pair.
    lonely = _history([("x", 1, "only"), ("y", 1, "only"), ("y", 2, "only")])
    out = icf_feature(lonely, n_neighbors=20)
    assert out["user_id"].tolist() == ["x", "y"] and (out["icf"] == 0.0).all()


def test_independent_of_block_size_and_row_order():
    rng = np.random.default_rng(0)
    rows = [(f"u{u:02d}", s, f"i{i:02d}") for u in range(37) for s in (1, 2) for i in rng.choice(25, size=5, replace=False)]
    history = _history(rows)
    reference = icf_feature(history, n_neighbors=5, block_size=2000)
    assert reference["user_id"].nunique() == 37 and (reference["icf"] >= 0).all() and (reference["icf"] > 0).any()
    for block_size in (1, 3, 7, 37, 1000):
        pd.testing.assert_frame_equal(icf_feature(history, n_neighbors=5, block_size=block_size), reference, check_exact=True)
    shuffled = history.sample(frac=1, random_state=1)
    pd.testing.assert_frame_equal(icf_feature(shuffled, n_neighbors=5, block_size=4), reference, check_exact=True)
    # Precomputed neighbours give the same result; neighbour rows of unknown items are ignored.
    neighbors = item_neighbors(history, 5)
    pd.testing.assert_frame_equal(icf_feature(history, n_neighbors=5, neighbors=neighbors), reference, check_exact=True)
    extra = pd.concat([neighbors, pd.DataFrame({"item_id": ["zz"], "neighbor_id": ["i00"], "sim": [0.5]})], ignore_index=True)
    pd.testing.assert_frame_equal(icf_feature(history, n_neighbors=5, neighbors=extra), reference, check_exact=True)


def test_matches_dense_reference():
    rng = np.random.default_rng(2)
    rows = [(f"u{u}", 1, f"i{i}") for u in range(12) for i in rng.choice(9, size=4, replace=False)]
    history = _history(rows)
    neighbors = item_neighbors(history, 3)
    sims = {(j, i): s for j, i, s in neighbors.itertuples(index=False)}
    bought = history.groupby("user_id")["item_id"].agg(set)
    out = _values(icf_feature(history, n_neighbors=3, block_size=5))
    for (user, item), value in out.items():
        expected = math.log1p(sum(sims.get((j, item), 0.0) for j in bought[user] if j != item))
        assert value == pytest.approx(expected, rel=1e-9, abs=1e-12)


def test_rejects_bad_arguments_and_has_no_target_parameter():
    with pytest.raises(ValueError):
        icf_feature(TOY, n_neighbors=0)
    with pytest.raises(ValueError):
        icf_feature(TOY, block_size=0)
    assert "target" not in inspect.signature(icf_feature).parameters
