import numpy as np
import pandas as pd
import pytest
from scipy import sparse

from ml_group9.repeat_explore import (
    PreferenceModel,
    build_interactions,
    cf_scores,
    explore_candidates,
    explore_scores,
    item_popularity,
    repeat_scores,
    seen_matrix,
)


COLUMNS = ["dataset", "user_id", "basket_id", "basket_seq", "item_id", "coarse_category", "fine_category", "quantity"]


def frame(rows):
    return pd.DataFrame([("d", u, b, s, i, "food", "f", 1) for u, b, s, i in rows], columns=COLUMNS)


def catalogue(ids):
    return pd.DataFrame({"item_id": ids, "coarse_category": "food", "fine_category": "f"})


def test_repeat_score_matches_hand_calculation():
    history = frame(
        [
            ("u1", "b1", 1, "a"), ("u1", "b1", 1, "b"),
            ("u1", "b2", 2, "a"),
            ("u1", "b3", 3, "a"), ("u1", "b3", 3, "c"), ("u1", "b3", 3, "c"),
        ]
    )
    interactions = build_interactions(history, catalogue(["a", "b", "c", "d"]))
    scores = repeat_scores(interactions, gamma=0.5).toarray()[0]
    assert scores == pytest.approx([0.25 + 0.5 + 1.0, 0.25, 1.0, 0.0])


def test_cf_scores_match_double_loop_and_mask_seen_and_out_of_scope():
    history = frame([("u1", "b1", 1, "a"), ("u1", "b2", 2, "b"), ("u2", "b3", 1, "c")])
    interactions = build_interactions(history, catalogue(["a", "b", "c", "d", "e"]))
    rng = np.random.default_rng(0)
    dense_sim = rng.random((5, 5)) * (rng.random((5, 5)) > 0.4)
    np.fill_diagonal(dense_sim, 0.0)
    in_scope = np.array([True, True, True, True, False])
    repeat = repeat_scores(interactions, gamma=0.8)
    seen = seen_matrix(interactions)
    result = cf_scores(repeat, sparse.csr_matrix(dense_sim), seen, in_scope).toarray()

    rep = repeat.toarray()
    expected = np.zeros_like(rep)
    for user in range(rep.shape[0]):
        for item in range(rep.shape[1]):
            if rep[user, item] > 0 or not in_scope[item]:
                continue
            expected[user, item] = sum(rep[user, j] * dense_sim[item, j] for j in range(rep.shape[1]))
    assert result == pytest.approx(expected)
    assert result[0, 0] == 0 and result[0, 1] == 0
    assert (result[:, 4] == 0).all()


def preference_model(pi_u1):
    # Items: 0 = attribute item in C_a, 1 = non-attribute item in C_a, 2 = outside every C_a.
    return PreferenceModel(
        attributes=["organic"],
        prior=np.array([0.3]),
        pi=np.array([[pi_u1], [0.3]]),
        n_c=np.array([[20.0], [0.0]]),
        item_flag=np.array([[1], [0], [1]], dtype=np.int8),
        item_applicable=np.array([[True], [True], [False]]),
    )


def test_log_ratio_signs():
    model = preference_model(0.6)
    users = np.array([0, 0, 0, 1, 1])
    items = np.array([0, 1, 2, 0, 1])
    ell = model.log_ratio(users, items)
    assert ell[0] == pytest.approx(np.log(0.6 / 0.3))
    assert ell[0] > 0
    assert ell[1] == pytest.approx(np.log(0.4 / 0.7))
    assert ell[1] < 0
    assert ell[2] == 0.0  # A_i is empty
    assert ell[3] == pytest.approx(0.0) and ell[4] == pytest.approx(0.0)  # pi = pi_bar


def test_explore_candidates_cover_cf_preference_and_popular_fill():
    history = frame([("u1", "b1", 1, "a"), ("u2", "b2", 1, "b"), ("u2", "b2", 1, "c"), ("u3", "b3", 1, "c")])
    interactions = build_interactions(history, catalogue(["a", "b", "c", "d", "e"]))
    seen = seen_matrix(interactions)
    in_scope = np.array([True, True, True, True, False])
    cf = sparse.csr_matrix(([0.5], ([0], [1])), shape=interactions.shape)
    model = PreferenceModel(
        attributes=["organic"],
        prior=np.array([0.3]),
        pi=np.array([[0.9], [0.1], [0.3]]),
        n_c=np.array([[20.0], [20.0], [20.0]]),
        item_flag=np.array([[1], [0], [0], [1], [1]], dtype=np.int8),
        item_applicable=np.ones((5, 1), dtype=bool),
    )
    settings = {"pref_items_per_attr": 20, "pref_min_n_c": 10, "max_candidates": 500}
    fine = np.array(["f"] * 5, dtype=object)
    candidates = explore_candidates(cf, seen, in_scope, model, fine, item_popularity(interactions), settings, top_k=3)
    by_user = candidates.groupby("user")["item"].apply(set).to_dict()

    assert by_user[0] >= {1, 3}  # CF item b and preferred organic item d; e is out of scope
    assert 4 not in set(candidates["item"])
    assert not candidates.merge(pd.DataFrame({"user": interactions.user_index, "item": interactions.item_index})).shape[0]
    assert candidates.loc[candidates["user"].eq(0) & candidates["item"].eq(3), "from_pref"].item()
    assert candidates.groupby("user").size().min() >= 2  # u2 has only a, d, e left and e is out of scope
    assert candidates.loc[candidates["user"].eq(1), "from_popular"].all()
    scores = explore_scores(candidates, beta=0.0, epsilon=1e-4)
    assert scores[candidates["s_cf"].eq(0).to_numpy()] == pytest.approx(np.log(1e-4))
