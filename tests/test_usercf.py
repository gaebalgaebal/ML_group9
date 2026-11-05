"""Synthetic, hand-calculated user-CF contracts."""

import inspect
import math

import numpy as np
import pandas as pd
import pytest

from nbr.baselines import item_popularity
from nbr.usercf import _list_from_vectors, ucf_list, ucf_scores, user_neighbors, user_vectors


def history(rows):
    return pd.DataFrame(rows, columns=["user_id", "basket_seq", "item_id"]).astype({"basket_seq": "int32"})


HISTORY = history([
    ("u1", 1, "a"), ("u1", 1, "b"),
    ("u2", 1, "a"), ("u2", 2, "a"), ("u2", 2, "c"),
    ("u3", 1, "b"), ("u3", 1, "d"),
])


def test_vectors_and_cosine_hand_calculated():
    X, users, items = user_vectors(HISTORY, "binary")
    assert users.tolist() == ["u1", "u2", "u3"]
    assert items.tolist() == ["a", "b", "c", "d"]
    np.testing.assert_array_equal(X.toarray(), [[1, 1, 0, 0], [1, 0, 1, 0], [0, 1, 0, 1]])
    neighbors = user_neighbors(X, 2)
    pairs = {(u, v): s for u, v, s in neighbors.itertuples(index=False)}
    assert set(pairs) == {("u1", "u2"), ("u1", "u3"), ("u2", "u1"), ("u3", "u1")}
    assert all(value == pytest.approx(.5) for value in pairs.values())
    weighted, _, _ = user_vectors(pd.concat([HISTORY, HISTORY.iloc[[2]]]), "log1p")
    assert weighted[1, 0] == pytest.approx(math.log(3))
    np.testing.assert_allclose(weighted.toarray()[0], [math.log(2), math.log(2), 0, 0])
    sims = user_neighbors(weighted, 2).set_index(["user_id", "neighbor_id"])["sim"]
    expected = math.log(3) / math.sqrt(2 * (math.log(3)**2 + math.log(2)**2))
    assert sims.loc[("u1", "u2")] == pytest.approx(expected)
    assert sims.loc[("u1", "u2")] != pytest.approx(.5)


def test_neighbor_topk_self_ties_and_blocks():
    h = history([(u, 1, i) for u in ["2", "10", "3", "1"] for i in ["a", "b"]])
    X, _, _ = user_vectors(h, "binary")
    full = user_neighbors(X, 10, 1)
    expected = full.groupby("user_id", sort=False).head(2).reset_index(drop=True)
    for size in (1, 2, 256):
        result = user_neighbors(X, 2, size)
        pd.testing.assert_frame_equal(result, expected)
        assert not (result["user_id"] == result["neighbor_id"]).any()
    assert expected[expected["user_id"] == "3"]["neighbor_id"].tolist() == ["1", "10"]


@pytest.mark.parametrize("weighting", ["binary", "log1p"])
def test_scores_hand_calculated_unseen_only(weighting):
    X, _, _ = user_vectors(HISTORY, weighting)
    neighbors = user_neighbors(X, 20)
    scores = ucf_scores(HISTORY, neighbors, X, 1)
    actual = scores.set_index(["user_id", "item_id"])["score"]
    assert set(actual.index) == {("u1", "c"), ("u1", "d"), ("u2", "b"), ("u3", "a")}
    sims = neighbors.set_index(["user_id", "neighbor_id"])["sim"]
    unit = 1 if weighting == "binary" else math.log(2)
    assert actual.loc[("u1", "c")] == pytest.approx(sims.loc[("u1", "u2")] * unit)
    assert actual.loc[("u1", "d")] == pytest.approx(.5 * unit)
    assert actual.loc[("u2", "b")] == pytest.approx(sims.loc[("u2", "u1")] * unit)
    pd.testing.assert_frame_equal(scores, ucf_scores(HISTORY, neighbors, X, 256))


def test_list_ties_padding_sources_unique_and_lengths():
    h = pd.concat([HISTORY, history([("z", 1, i) for i in ["e", "f", "g", "h"]])], ignore_index=True)
    pop = item_popularity(h)
    # Equal score: popularity precedes lexical ID.
    pop.loc["d"] = 9
    lists = ucf_list(h, 2, "binary", 3, pop)
    assert lists.columns.tolist() == ["user_id", "rank", "item_id", "source"]
    assert lists.groupby("user_id").size().eq(3).all()
    assert not lists.duplicated(["user_id", "item_id"]).any()
    assert lists.merge(h, on=["user_id", "item_id"]).empty
    u1 = lists[lists["user_id"] == "u1"]
    assert u1["item_id"].tolist()[:2] == ["d", "c"]
    assert u1["source"].tolist() == ["usercf", "usercf", "popularity"]
    assert lists[lists["user_id"] == "z"]["source"].eq("popularity").all()
    pd.testing.assert_frame_equal(lists, ucf_list(h, 2, "binary", 3, pop, block_size=1))


def test_two_clusters_recommend_within_cluster():
    h = history([("a1", 1, "x"), ("a2", 1, "x"), ("a2", 1, "y"),
                 ("b1", 1, "p"), ("b2", 1, "p"), ("b2", 1, "q")])
    lists = ucf_list(h, 1, "binary", 1)
    assert lists.set_index("user_id").loc["a1", "item_id"] == "y"
    assert lists.set_index("user_id").loc["b1", "item_id"] == "q"


def test_determinism_cache_equivalence_and_no_target_input():
    pop = item_popularity(HISTORY)
    for weighting in ("binary", "log1p"):
        X, _, _ = user_vectors(HISTORY, weighting)
        full = user_neighbors(X, 100)
        prefix = full.groupby("user_id", sort=False).head(1)
        cached = _list_from_vectors(HISTORY, prefix, X, 10, pop, 1)
        direct = ucf_list(HISTORY, 1, weighting, pop=pop)
        pd.testing.assert_frame_equal(cached, direct)
        shuffled = HISTORY.sample(frac=1, random_state=42)
        pd.testing.assert_frame_equal(direct, ucf_list(shuffled, 1, weighting, pop=pop))
    for function in (user_vectors, user_neighbors, ucf_scores, ucf_list):
        assert "target" not in inspect.signature(function).parameters


def test_empty_singleton_and_insufficient_catalogue():
    empty = HISTORY.iloc[:0]
    X, _, _ = user_vectors(empty, "binary")
    assert user_neighbors(X, 2).empty
    assert ucf_scores(empty, user_neighbors(X, 2), X).empty
    assert ucf_list(empty, 2, "binary").empty
    one = history([("u", 1, "a")])
    assert ucf_list(one, 2, "binary").empty
    result = ucf_list(HISTORY, 1, "binary")
    assert result.groupby("user_id").size().eq(2).all()


@pytest.mark.parametrize("k,size", [(0, 256), (1, 0), (-1, 1), (True, 1)])
def test_invalid_neighbor_parameters(k, size):
    X, _, _ = user_vectors(HISTORY, "binary")
    with pytest.raises(ValueError):
        user_neighbors(X, k, size)


def test_invalid_weighting_and_unlabelled_matrix():
    with pytest.raises(ValueError):
        user_vectors(HISTORY, "quantity")
    X, _, _ = user_vectors(HISTORY, "binary")
    with pytest.raises(ValueError):
        user_neighbors(X.copy(), 1)
