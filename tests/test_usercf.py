# *------------------- 2026 Fall Semester Machine Learning Term-project: Team 9 --------------------*
# Code First written at 2026-11-05
# Code Final Edited at 2026-11-13
# Editor: Yejun Kim

'''
Purpose : Synthetic, hand-calculated user-CF contracts: tests of the user-based
          collaborative filtering functions of nbr.usercf (user vectors, neighbours, scores
          and recommendation lists).
Inputs  : Small purchase histories built in this file (columns user_id, basket_seq, item_id).
Outputs : pytest pass/fail results; no files are written.
Usage   :
    python -m pytest tests/test_usercf.py
'''

# ---------------------------------------------------------------------------
# imports
# ---------------------------------------------------------------------------
import inspect
import math

import numpy as np
import pandas as pd
import pytest

from nbr.baselines import item_popularity
from nbr.usercf import _list_from_vectors, ucf_list, ucf_scores, user_neighbors, user_vectors


# ---------------------------------------------------------------------------
# fixtures and helpers
# ---------------------------------------------------------------------------
def history(rows):
    '''Build a (user_id, basket_seq, item_id) frame from ``rows`` with an int32 basket_seq.'''
    return pd.DataFrame(rows, columns=["user_id", "basket_seq", "item_id"]).astype({"basket_seq": "int32"})


HISTORY = history([
    ("u1", 1, "a"), ("u1", 1, "b"),
    ("u2", 1, "a"), ("u2", 2, "a"), ("u2", 2, "c"),
    ("u3", 1, "b"), ("u3", 1, "d"),
])


# ---------------------------------------------------------------------------
# user vectors and neighbours
# ---------------------------------------------------------------------------
def test_vectors_and_cosine_hand_calculated():
    # Build the binary user x item matrix of the three-user history.
    X, users, items = user_vectors(HISTORY, "binary")
    # Expect sorted user and item labels and the hand-written 0/1 matrix.
    assert users.tolist() == ["u1", "u2", "u3"]
    assert items.tolist() == ["a", "b", "c", "d"]
    np.testing.assert_array_equal(X.toarray(), [[1, 1, 0, 0], [1, 0, 1, 0], [0, 1, 0, 1]])
    # Compute the two nearest neighbours of every user.
    neighbors = user_neighbors(X, 2)
    # Expect only the pairs that share an item, each with cosine similarity 0.5.
    pairs = {(u, v): s for u, v, s in neighbors.itertuples(index=False)}
    assert set(pairs) == {("u1", "u2"), ("u1", "u3"), ("u2", "u1"), ("u3", "u1")}
    assert all(value == pytest.approx(.5) for value in pairs.values())
    # Repeat one history row and build the log1p-weighted matrix.
    weighted, _, _ = user_vectors(pd.concat([HISTORY, HISTORY.iloc[[2]]]), "log1p")
    # Expect log(1 + basket count) entries and the hand-calculated similarity of u1 and u2.
    assert weighted[1, 0] == pytest.approx(math.log(3))
    np.testing.assert_allclose(weighted.toarray()[0], [math.log(2), math.log(2), 0, 0])
    sims = user_neighbors(weighted, 2).set_index(["user_id", "neighbor_id"])["sim"]
    expected = math.log(3) / math.sqrt(2 * (math.log(3)**2 + math.log(2)**2))
    assert sims.loc[("u1", "u2")] == pytest.approx(expected)
    assert sims.loc[("u1", "u2")] != pytest.approx(.5)


def test_neighbor_topk_self_ties_and_blocks():
    # Build four users with identical baskets so that every similarity is tied.
    h = history([(u, 1, i) for u in ["2", "10", "3", "1"] for i in ["a", "b"]])
    X, _, _ = user_vectors(h, "binary")
    # Take the first two rows per user of the full neighbour table as the reference.
    full = user_neighbors(X, 10, 1)
    expected = full.groupby("user_id", sort=False).head(2).reset_index(drop=True)
    # Expect the same top-2 table for every block size, without self pairs.
    for size in (1, 2, 256):
        result = user_neighbors(X, 2, size)
        pd.testing.assert_frame_equal(result, expected)
        assert not (result["user_id"] == result["neighbor_id"]).any()
    # Expect ties to be ordered by the neighbour id read as a string.
    assert expected[expected["user_id"] == "3"]["neighbor_id"].tolist() == ["1", "10"]


# ---------------------------------------------------------------------------
# scores and recommendation lists
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("weighting", ["binary", "log1p"])
def test_scores_hand_calculated_unseen_only(weighting):
    # Score the items of the three-user history with all neighbours and block size 1.
    X, _, _ = user_vectors(HISTORY, weighting)
    neighbors = user_neighbors(X, 20)
    scores = ucf_scores(HISTORY, neighbors, X, 1)
    # Expect scores only for unseen items, equal to similarity times the neighbour's weight.
    actual = scores.set_index(["user_id", "item_id"])["score"]
    assert set(actual.index) == {("u1", "c"), ("u1", "d"), ("u2", "b"), ("u3", "a")}
    sims = neighbors.set_index(["user_id", "neighbor_id"])["sim"]
    unit = 1 if weighting == "binary" else math.log(2)
    assert actual.loc[("u1", "c")] == pytest.approx(sims.loc[("u1", "u2")] * unit)
    assert actual.loc[("u1", "d")] == pytest.approx(.5 * unit)
    assert actual.loc[("u2", "b")] == pytest.approx(sims.loc[("u2", "u1")] * unit)
    # Expect the same scores with block size 256.
    pd.testing.assert_frame_equal(scores, ucf_scores(HISTORY, neighbors, X, 256))


def test_list_ties_padding_sources_unique_and_lengths():
    # Add a user z who shares no item with the other users.
    h = pd.concat([HISTORY, history([("z", 1, i) for i in ["e", "f", "g", "h"]])], ignore_index=True)
    pop = item_popularity(h)
    # Equal score: popularity precedes lexical ID.
    pop.loc["d"] = 9
    # Build lists of length 3 with two neighbours.
    lists = ucf_list(h, 2, "binary", 3, pop)
    # Expect three distinct unseen items per user, scored items first and popularity padding last.
    assert lists.columns.tolist() == ["user_id", "rank", "item_id", "source"]
    assert lists.groupby("user_id").size().eq(3).all()
    assert not lists.duplicated(["user_id", "item_id"]).any()
    assert lists.merge(h, on=["user_id", "item_id"]).empty
    u1 = lists[lists["user_id"] == "u1"]
    assert u1["item_id"].tolist()[:2] == ["d", "c"]
    assert u1["source"].tolist() == ["usercf", "usercf", "popularity"]
    assert lists[lists["user_id"] == "z"]["source"].eq("popularity").all()
    # Expect the same lists with block size 1.
    pd.testing.assert_frame_equal(lists, ucf_list(h, 2, "binary", 3, pop, block_size=1))


def test_two_clusters_recommend_within_cluster():
    # Build two user pairs with disjoint items and one-item lists from one neighbour.
    h = history([("a1", 1, "x"), ("a2", 1, "x"), ("a2", 1, "y"),
                 ("b1", 1, "p"), ("b2", 1, "p"), ("b2", 1, "q")])
    lists = ucf_list(h, 1, "binary", 1)
    # Expect each user to get the extra item of the other user of the same pair.
    assert lists.set_index("user_id").loc["a1", "item_id"] == "y"
    assert lists.set_index("user_id").loc["b1", "item_id"] == "q"


# ---------------------------------------------------------------------------
# determinism and edge cases
# ---------------------------------------------------------------------------
def test_determinism_cache_equivalence_and_no_target_input():
    pop = item_popularity(HISTORY)
    for weighting in ("binary", "log1p"):
        # Build one list from the first row per user of a full neighbour table.
        X, _, _ = user_vectors(HISTORY, weighting)
        full = user_neighbors(X, 100)
        prefix = full.groupby("user_id", sort=False).head(1)
        cached = _list_from_vectors(HISTORY, prefix, X, 10, pop, 1)
        # Expect the same list from the public function with one neighbour.
        direct = ucf_list(HISTORY, 1, weighting, pop=pop)
        pd.testing.assert_frame_equal(cached, direct)
        # Expect the same list when the history rows are shuffled.
        shuffled = HISTORY.sample(frac=1, random_state=42)
        pd.testing.assert_frame_equal(direct, ucf_list(shuffled, 1, weighting, pop=pop))
    # Expect no public function to take a parameter named target.
    for function in (user_vectors, user_neighbors, ucf_scores, ucf_list):
        assert "target" not in inspect.signature(function).parameters


def test_empty_singleton_and_insufficient_catalogue():
    # Expect empty neighbours, scores and lists for an empty history.
    empty = HISTORY.iloc[:0]
    X, _, _ = user_vectors(empty, "binary")
    assert user_neighbors(X, 2).empty
    assert ucf_scores(empty, user_neighbors(X, 2), X).empty
    assert ucf_list(empty, 2, "binary").empty
    # Expect an empty list for a single user with a single item.
    one = history([("u", 1, "a")])
    assert ucf_list(one, 2, "binary").empty
    # Expect two rows per user when the catalogue has only two unseen items per user.
    result = ucf_list(HISTORY, 1, "binary")
    assert result.groupby("user_id").size().eq(2).all()


# ---------------------------------------------------------------------------
# invalid arguments
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("k,size", [(0, 256), (1, 0), (-1, 1), (True, 1)])
def test_invalid_neighbor_parameters(k, size):
    # Build a valid matrix of the three-user history.
    X, _, _ = user_vectors(HISTORY, "binary")
    # Expect a ValueError for a neighbour count or block size that is not a positive integer.
    with pytest.raises(ValueError):
        user_neighbors(X, k, size)


def test_invalid_weighting_and_unlabelled_matrix():
    # Expect a ValueError for an unknown weighting name.
    with pytest.raises(ValueError):
        user_vectors(HISTORY, "quantity")
    # Expect a ValueError for a matrix copy that does not carry the user and item labels.
    X, _, _ = user_vectors(HISTORY, "binary")
    with pytest.raises(ValueError):
        user_neighbors(X.copy(), 1)
