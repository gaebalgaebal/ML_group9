"""Tests for nbr.itemcf on synthetic data."""

import inspect
import math

import numpy as np
import pandas as pd
import pytest

from nbr.itemcf import cf_list, cf_scores, item_neighbors


def _history(rows):
    return pd.DataFrame(rows, columns=["user_id", "basket_seq", "item_id"]).astype({"basket_seq": "int32"})


# User x item binary matrix (3 x 3):
#   u1: a, b        u2: a, b, c        u3: a (twice, counts once)
# |U_a| = 3, |U_b| = 2, |U_c| = 1
# cos(a, b) = 2 / sqrt(6), cos(a, c) = 1 / sqrt(3), cos(b, c) = 1 / sqrt(2)
HISTORY = _history([
    ("u1", 1, "a"), ("u1", 1, "b"),
    ("u2", 1, "a"), ("u2", 2, "b"), ("u2", 2, "c"),
    ("u3", 1, "a"), ("u3", 2, "a"),
])


def _sims(neighbors):
    return {(i, j): s for i, j, s in neighbors.itertuples(index=False)}


def test_cosine_hand_computed():
    neighbors = item_neighbors(HISTORY, n_neighbors=20)
    assert list(neighbors.columns) == ["item_id", "neighbor_id", "sim"]
    sims = _sims(neighbors)
    assert sims[("a", "b")] == pytest.approx(2 / math.sqrt(6), rel=1e-6)
    assert sims[("b", "a")] == pytest.approx(2 / math.sqrt(6), rel=1e-6)
    assert sims[("a", "c")] == pytest.approx(1 / math.sqrt(3), rel=1e-6)
    assert sims[("b", "c")] == pytest.approx(1 / math.sqrt(2), rel=1e-6)
    assert ("a", "a") not in sims and len(sims) == 6


def test_neighbor_truncation_and_blocks():
    rng = np.random.default_rng(0)
    rows = [(f"u{u}", 1, f"i{i:02d}") for u in range(15) for i in rng.choice(30, size=8, replace=False)]
    history = _history(rows)
    full = item_neighbors(history, n_neighbors=100, block_size=7)
    top3 = item_neighbors(history, n_neighbors=3, block_size=4)
    assert (top3.groupby("item_id").size() <= 3).all()
    assert (top3["sim"] > 0).all()
    # Top-3 equals the first three rows of the full ranking (sim desc, neighbor_id asc).
    expected = full.groupby("item_id", sort=False).head(3).reset_index(drop=True)
    pd.testing.assert_frame_equal(top3, expected)
    # Block size does not change the result.
    pd.testing.assert_frame_equal(item_neighbors(history, n_neighbors=3, block_size=256), top3)


def test_neighbor_ties_resolved_by_id():
    # Items p, q, r all co-occur identically with s: equal similarity to s.
    history = _history([("u1", 1, "s"), ("u1", 1, "p"), ("u1", 1, "q"), ("u1", 1, "r")])
    neighbors = item_neighbors(history, n_neighbors=2)
    s = neighbors[neighbors["item_id"] == "s"]
    assert list(s["neighbor_id"]) == ["p", "q"]


def test_cf_scores_excludes_bought():
    neighbors = item_neighbors(HISTORY, n_neighbors=20)
    scores = cf_scores(HISTORY, neighbors)
    assert list(scores.columns) == ["user_id", "item_id", "score"]
    by_pair = {(u, i): s for u, i, s in scores.itertuples(index=False)}
    assert set(by_pair) == {("u1", "c"), ("u3", "b"), ("u3", "c")}
    assert by_pair[("u1", "c")] == pytest.approx(1 / math.sqrt(3) + 1 / math.sqrt(2), rel=1e-6)
    assert by_pair[("u3", "b")] == pytest.approx(2 / math.sqrt(6), rel=1e-6)
    assert by_pair[("u3", "c")] == pytest.approx(1 / math.sqrt(3), rel=1e-6)


def test_cf_list():
    neighbors = item_neighbors(HISTORY, n_neighbors=20)
    pop = pd.Series({"a": 3, "b": 2, "c": 1}, name="pop")
    lists = cf_list(HISTORY, neighbors, 5, pop)
    assert list(lists.columns) == ["user_id", "rank", "item_id", "source"]
    assert (lists["source"] == "itemcf").all()
    assert list(lists[lists["user_id"] == "u3"]["item_id"]) == ["b", "c"]
    assert list(lists[lists["user_id"] == "u1"]["item_id"]) == ["c"]
    assert "u2" not in set(lists["user_id"])  # bought everything


def test_deterministic():
    neighbors = item_neighbors(HISTORY, n_neighbors=20)
    shuffled = HISTORY.sample(frac=1, random_state=2)
    pd.testing.assert_frame_equal(item_neighbors(shuffled, n_neighbors=20), neighbors)
    pd.testing.assert_frame_equal(cf_scores(shuffled, neighbors), cf_scores(HISTORY, neighbors))
    pd.testing.assert_frame_equal(cf_scores(HISTORY, neighbors, chunk_users=1), cf_scores(HISTORY, neighbors))


def test_no_target_argument():
    for function in (item_neighbors, cf_scores, cf_list):
        assert "target" not in inspect.signature(function).parameters
