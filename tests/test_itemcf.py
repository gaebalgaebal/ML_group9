# *------------------- 2026 Fall Semester Machine Learning Term-project: Team 9 --------------------*
# Code First written at 2026-11-05
# Code Final Edited at 2026-11-13
# Editor: Yejun Kim

'''
Purpose : Tests for nbr.itemcf on synthetic data.
Inputs  : Small hand-written and seeded random purchase histories (columns user_id, basket_seq,
          item_id) and a hand-written popularity Series.
Outputs : None; the tests assert on the frames returned by item_neighbors (item_id,
          neighbor_id, sim), cf_scores (user_id, item_id, score) and cf_list (user_id, rank,
          item_id, source).
Usage   :
    python -m pytest tests/test_itemcf.py
'''

# ---------------------------------------------------------------------------
# imports
# ---------------------------------------------------------------------------
import inspect
import math

import numpy as np
import pandas as pd
import pytest

from nbr.itemcf import cf_list, cf_scores, item_neighbors


# ---------------------------------------------------------------------------
# helpers and toy data
# ---------------------------------------------------------------------------
def _history(rows):
    '''Build a history frame (user_id, basket_seq, item_id) from the tuples in ``rows``.'''
    return pd.DataFrame(rows, columns=["user_id", "basket_seq", "item_id"]).astype({"basket_seq": "int32"})


# The toy history is a user x item binary matrix (3 x 3).
# The users buy u1: a, b; u2: a, b, c; u3: a (twice, counts once).
# The user counts of the items are |U_a| = 3, |U_b| = 2, |U_c| = 1.
# The cosines are cos(a, b) = 2 / sqrt(6), cos(a, c) = 1 / sqrt(3), cos(b, c) = 1 / sqrt(2).
HISTORY = _history([
    ("u1", 1, "a"), ("u1", 1, "b"),
    ("u2", 1, "a"), ("u2", 2, "b"), ("u2", 2, "c"),
    ("u3", 1, "a"), ("u3", 2, "a"),
])


def _sims(neighbors):
    '''Map (item_id, neighbor_id) -> sim for the three-column frame ``neighbors``.'''
    return {(i, j): s for i, j, s in neighbors.itertuples(index=False)}


# ---------------------------------------------------------------------------
# item neighbours
# ---------------------------------------------------------------------------
def test_cosine_hand_computed():
    # Compute the neighbours of the toy history with every neighbour kept.
    neighbors = item_neighbors(HISTORY, n_neighbors=20)
    assert list(neighbors.columns) == ["item_id", "neighbor_id", "sim"]
    # Expect the hand-computed cosines in both directions and no self pair.
    sims = _sims(neighbors)
    assert sims[("a", "b")] == pytest.approx(2 / math.sqrt(6), rel=1e-6)
    assert sims[("b", "a")] == pytest.approx(2 / math.sqrt(6), rel=1e-6)
    assert sims[("a", "c")] == pytest.approx(1 / math.sqrt(3), rel=1e-6)
    assert sims[("b", "c")] == pytest.approx(1 / math.sqrt(2), rel=1e-6)
    assert ("a", "a") not in sims and len(sims) == 6


def test_neighbor_truncation_and_blocks():
    # Build a seeded random history of 15 users with one basket of 8 items each.
    rng = np.random.default_rng(0)
    rows = [(f"u{u}", 1, f"i{i:02d}") for u in range(15) for i in rng.choice(30, size=8, replace=False)]
    history = _history(rows)
    # Compute the full ranking and the top-3 ranking with different block sizes.
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
    # Expect the two kept neighbours of s to be the first two ids in ascending order.
    s = neighbors[neighbors["item_id"] == "s"]
    assert list(s["neighbor_id"]) == ["p", "q"]


# ---------------------------------------------------------------------------
# scores and lists
# ---------------------------------------------------------------------------
def test_cf_scores_excludes_bought():
    # Score the toy history with its own neighbours.
    neighbors = item_neighbors(HISTORY, n_neighbors=20)
    scores = cf_scores(HISTORY, neighbors)
    assert list(scores.columns) == ["user_id", "item_id", "score"]
    # Expect scores only for items the user never bought, each the sum of the cosines.
    by_pair = {(u, i): s for u, i, s in scores.itertuples(index=False)}
    assert set(by_pair) == {("u1", "c"), ("u3", "b"), ("u3", "c")}
    assert by_pair[("u1", "c")] == pytest.approx(1 / math.sqrt(3) + 1 / math.sqrt(2), rel=1e-6)
    assert by_pair[("u3", "b")] == pytest.approx(2 / math.sqrt(6), rel=1e-6)
    assert by_pair[("u3", "c")] == pytest.approx(1 / math.sqrt(3), rel=1e-6)


def test_cf_list():
    # Build the lists of the toy history with a hand-written popularity Series.
    neighbors = item_neighbors(HISTORY, n_neighbors=20)
    pop = pd.Series({"a": 3, "b": 2, "c": 1}, name="pop")
    lists = cf_list(HISTORY, neighbors, 5, pop)
    # Expect the source itemcf, unseen items in score order and no list for u2.
    assert list(lists.columns) == ["user_id", "rank", "item_id", "source"]
    assert (lists["source"] == "itemcf").all()
    assert list(lists[lists["user_id"] == "u3"]["item_id"]) == ["b", "c"]
    assert list(lists[lists["user_id"] == "u1"]["item_id"]) == ["c"]
    assert "u2" not in set(lists["user_id"])  # bought everything


# ---------------------------------------------------------------------------
# determinism and signatures
# ---------------------------------------------------------------------------
def test_deterministic():
    # Compute the neighbours of the toy history and shuffle its rows.
    neighbors = item_neighbors(HISTORY, n_neighbors=20)
    shuffled = HISTORY.sample(frac=1, random_state=2)
    # Expect the same frames for the shuffled rows and for a user chunk size of 1.
    pd.testing.assert_frame_equal(item_neighbors(shuffled, n_neighbors=20), neighbors)
    pd.testing.assert_frame_equal(cf_scores(shuffled, neighbors), cf_scores(HISTORY, neighbors))
    pd.testing.assert_frame_equal(cf_scores(HISTORY, neighbors, chunk_users=1), cf_scores(HISTORY, neighbors))


def test_no_target_argument():
    # Expect none of the three functions to have a parameter named target.
    for function in (item_neighbors, cf_scores, cf_list):
        assert "target" not in inspect.signature(function).parameters
