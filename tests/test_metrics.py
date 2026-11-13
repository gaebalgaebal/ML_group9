# *------------------- 2026 Fall Semester Machine Learning Term-project: Team 9 --------------------*
# Code First written at 2026-11-03
# Code Final Edited at 2026-11-13
# Editor: Yejun Kim

'''
Purpose : Hand-computed tests for nbr.metrics.
Inputs  : Small list and target frames built in this file and per-user Series for the
          bootstrap.
Outputs : pytest pass/fail results; no file is written.
Usage   :
    python -m pytest tests/test_metrics.py
'''

# ---------------------------------------------------------------------------
# imports
# ---------------------------------------------------------------------------
import math

import numpy as np
import pandas as pd
import pytest

from nbr.metrics import (
    paired_bootstrap,
    per_user_category_hit,
    per_user_ndcg,
    per_user_recall,
)


# ---------------------------------------------------------------------------
# helpers and fixtures
# ---------------------------------------------------------------------------
def _lists(entries):
    """entries: {user: [item, ...]} -> list frame [user_id, rank, item_id, source]."""
    # Give every item its 1-based position in the user's list as rank.
    rows = [(u, r + 1, item, "test") for u, items in entries.items() for r, item in enumerate(items)]
    # Build the list frame.
    return pd.DataFrame(rows, columns=["user_id", "rank", "item_id", "source"])


def _target(entries):
    '''Build a target frame [user_id, item_id] from ``entries`` ({user: [item, ...]}).'''
    # Write one row per user and target item.
    rows = [(u, item) for u, items in entries.items() for item in items]
    # Build the target frame.
    return pd.DataFrame(rows, columns=["user_id", "item_id"])


# Shared list frame of the hand-computed tests.
LISTS = _lists({
    "u1": ["a", "b", "c"],            # 1 target, hit at rank 2
    "u2": ["x", "a", "y", "b", "c"],  # 3 targets, hits at ranks 2 and 4 (c at rank 5 > k=4)
    "u3": ["a"],                      # list shorter than k, 2 targets, hit at rank 1
    "u9": ["a"],                      # not a target user -> not in the output
})
# Shared target frame of the hand-computed tests.
TARGET = _target({"u1": ["b"], "u2": ["a", "b", "c"], "u3": ["a", "z"], "u4": ["q"]})


def _discount(rank):
    '''Return the NDCG discount 1 / log2(rank + 1) of the 1-based position ``rank``.'''
    return 1 / math.log2(rank + 1)


# ---------------------------------------------------------------------------
# recall and ndcg
# ---------------------------------------------------------------------------
def test_recall_hand_computed():
    # Score the shared lists against the shared targets at k = 4.
    recall = per_user_recall(LISTS, TARGET, k=4)
    # Expect one float value per target user that equals the hand-computed recall.
    assert list(recall.index) == ["u1", "u2", "u3", "u4"]
    assert recall.dtype == np.dtype("float64")
    assert recall["u1"] == 1.0
    assert recall["u2"] == pytest.approx(2 / 3)
    assert recall["u3"] == 0.5
    assert recall["u4"] == 0.0  # no list


def test_ndcg_hand_computed():
    # Score the shared lists against the shared targets at k = 4.
    ndcg = per_user_ndcg(LISTS, TARGET, k=4)
    d = _discount
    # Expect the hand-computed ratio of discounted gain to ideal gain for every target user.
    assert list(ndcg.index) == ["u1", "u2", "u3", "u4"]
    assert ndcg["u1"] == pytest.approx(d(2) / d(1))
    assert ndcg["u2"] == pytest.approx((d(2) + d(4)) / (d(1) + d(2) + d(3)))
    assert ndcg["u3"] == pytest.approx(d(1) / (d(1) + d(2)))
    assert ndcg["u4"] == 0.0


def test_idcg_capped_at_k():
    # Build a user with two listed items and three target items.
    lists = _lists({"u": ["a", "b"]})
    target = _target({"u": ["a", "b", "c"]})
    # Expect NDCG 1 because the ideal gain counts only k = 2 hits, and recall 2/3.
    assert per_user_ndcg(lists, target, k=2)["u"] == pytest.approx(1.0)
    assert per_user_recall(lists, target, k=2)["u"] == pytest.approx(2 / 3)


def test_target_duplicates_ignored():
    # Repeat every row of the target frame.
    target = pd.concat([TARGET, TARGET], ignore_index=True)
    # Expect the same recall and NDCG as with the original target.
    pd.testing.assert_series_equal(per_user_recall(LISTS, target, 4), per_user_recall(LISTS, TARGET, 4))
    pd.testing.assert_series_equal(per_user_ndcg(LISTS, target, 4), per_user_ndcg(LISTS, TARGET, 4))


@pytest.mark.parametrize("k", [0, -1, 1.5, True])
def test_invalid_k(k):
    # Expect ValueError for a cutoff that is not a positive integer.
    with pytest.raises(ValueError):
        per_user_recall(LISTS, TARGET, k)


# ---------------------------------------------------------------------------
# category hit
# ---------------------------------------------------------------------------
def test_category_hit_hand_computed():
    # Build a category ranking for u1 and u2 and new-target rows with one repeated row.
    category_rank = pd.DataFrame(
        [("u1", "c1", 1), ("u1", "c2", 2), ("u1", "c3", 3), ("u2", "c9", 1)],
        columns=["user_id", "category", "rank"],
    )
    new_target = pd.DataFrame(
        [("u1", "i1", "c1"), ("u1", "i2", "c3"), ("u1", "i3", "c5"), ("u1", "i3", "c5"),
         ("u2", "i4", "c1"), ("u3", "i5", "c1")],
        columns=["user_id", "item_id", "L2"],
    )
    # Expect the share of each user's distinct target rows whose category is in the top k.
    hit = per_user_category_hit(category_rank, new_target, k=2)
    assert list(hit.index) == ["u1", "u2", "u3"]
    assert hit["u1"] == pytest.approx(1 / 3)  # c1 in top-2; c3 (rank 3) and c5 are not
    assert hit["u2"] == 0.0
    assert hit["u3"] == 0.0  # no category ranking
    assert per_user_category_hit(category_rank, new_target, k=3)["u1"] == pytest.approx(2 / 3)
    # Expect the same value when the category column is chosen through the level argument.
    renamed = new_target.rename(columns={"L2": "L1"})
    assert per_user_category_hit(category_rank, renamed, k=2, level="L1")["u1"] == pytest.approx(1 / 3)


# ---------------------------------------------------------------------------
# paired bootstrap
# ---------------------------------------------------------------------------
def test_bootstrap_identical_series():
    # Compare a series with a copy of itself.
    a = pd.Series([0.1, 0.5, 0.9], index=["u1", "u2", "u3"])
    out = paired_bootstrap(a, a.copy(), n=50, seed=1, levels=[0.95, 0.99375])
    # Expect a zero difference and zero-width intervals at both levels.
    assert out["n_users"] == 3
    assert out["diff"] == 0.0
    assert out["ci"] == {0.95: [0.0, 0.0], 0.99375: [0.0, 0.0]}
    assert out["mean_a"] == out["mean_b"] == pytest.approx(0.5)


def test_bootstrap_deterministic_and_chunked():
    # Draw two random per-user series for 40 users and run the same bootstrap twice.
    rng = np.random.default_rng(0)
    idx = [f"u{i}" for i in range(40)]
    a = pd.Series(rng.random(40), index=idx)
    b = pd.Series(rng.random(40), index=idx)
    first = paired_bootstrap(a, b, n=300, seed=7, levels=[0.95, 0.99])
    second = paired_bootstrap(a, b, n=300, seed=7, levels=[0.95, 0.99])
    # Expect equal results, the observed mean difference and nested intervals.
    assert first == second
    assert first["diff"] == pytest.approx(float((b - a).mean()))
    lo, hi = first["ci"][0.95]
    assert lo <= first["diff"] <= hi
    lo99, hi99 = first["ci"][0.99]
    assert lo99 <= lo and hi <= hi99
    # Index order must not matter.
    shuffled = paired_bootstrap(a, b.sample(frac=1, random_state=3), n=300, seed=7, levels=[0.95])
    assert shuffled["ci"][0.95] == first["ci"][0.95]
    # Chunk size must not change the result.
    small_chunks = paired_bootstrap(a, b, n=300, seed=7, levels=[0.95, 0.99], chunk_size=7)
    assert small_chunks == first
    # Different seed -> (almost surely) different CI.
    assert paired_bootstrap(a, b, n=300, seed=8, levels=[0.95])["ci"][0.95] != first["ci"][0.95]


def test_bootstrap_single_user_constant_diff():
    # Build one user whose difference is 0.6.
    a = pd.Series([0.2], index=["u"])
    b = pd.Series([0.8], index=["u"])
    out = paired_bootstrap(a, b, n=5, seed=1, levels=[0.95])
    # Expect the difference and both interval bounds to equal 0.6.
    assert out["diff"] == pytest.approx(0.6)
    assert out["ci"][0.95] == pytest.approx([0.6, 0.6])


def test_bootstrap_index_mismatch():
    # Build two series whose user indices differ.
    a = pd.Series([1.0, 2.0], index=["u1", "u2"])
    b = pd.Series([1.0, 2.0], index=["u1", "u3"])
    # Expect ValueError.
    with pytest.raises(ValueError):
        paired_bootstrap(a, b, n=10, seed=0, levels=[0.95])


@pytest.mark.parametrize("kind", ["empty", "duplicate", "nan", "n_zero", "bad_level"])
def test_bootstrap_invalid_inputs(kind):
    # Start from a valid series and break one input according to the parameter.
    a = pd.Series([0.1, 0.5], index=["a", "b"])
    n, levels = 10, [0.95]
    if kind == "empty":
        a = a.iloc[:0]
    elif kind == "duplicate":
        a.index = ["a", "a"]
    elif kind == "nan":
        a.iloc[0] = np.nan
    elif kind == "n_zero":
        n = 0
    else:
        levels = [1.0]
    # Expect ValueError for every kind of invalid input.
    with pytest.raises(ValueError):
        paired_bootstrap(a, a, n, 0, levels)
