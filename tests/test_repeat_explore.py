import numpy as np
import pandas as pd
import pytest
from scipy import sparse

from ml_group9.repeat_explore import (
    PreferenceModel,
    allocate_slots,
    build_interactions,
    cf_scores,
    explore_candidates,
    explore_scores,
    item_popularity,
    repeat_counts,
    repeat_ranking,
    repeat_ratio,
    repeat_scores,
    scope_mask,
    scope_targets,
    seen_matrix,
    slot_counts,
    to_lists,
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


def ranked(rows):
    frame = pd.DataFrame(rows, columns=["user", "item", "score"])
    frame = frame.sort_values(["user", "score"], ascending=[True, False])
    frame["rank"] = frame.groupby("user").cumcount()
    return frame


def test_slot_counts_are_rounded_clipped_and_bounded_by_repeat_items():
    rho = np.array([-0.2, 0.0, 0.34, 0.35, 0.96, 1.4])
    available = np.array([10, 10, 10, 10, 10, 3])
    assert slot_counts(rho, 10, available).tolist() == [0, 0, 3, 4, 10, 3]


def test_allocation_fills_short_side_without_duplicates():
    # User 0: 2 repeats and many explores, 6 repeat slots -> 2 repeats + 8 explores.
    # User 1: 12 repeats and 1 explore, 3 repeat slots -> 9 repeats + 1 explore.
    repeats = ranked([(0, item, 10 - item) for item in range(2)] + [(1, item, 20 - item) for item in range(12)])
    explores = ranked(
        [(0, 100 + item, -item) for item in range(15)] + [(0, 1, 5.0)] + [(1, 200, 1.0)]
    )
    lists = allocate_slots(repeats, explores, np.array([6, 3]), top_k=10)
    for user, group in lists.groupby("user"):
        assert len(group) == 10
        assert not group["item"].duplicated().any()
        assert group["position"].tolist() == list(range(10))
    user0 = lists[lists["user"].eq(0)]
    assert user0["slot"].value_counts().to_dict() == {"explore": 8, "repeat": 2}
    assert 1 in set(user0.loc[user0["slot"].eq("repeat"), "item"])  # the explore duplicate of a repeat is dropped
    user1 = lists[lists["user"].eq(1)]
    assert user1["slot"].value_counts().to_dict() == {"repeat": 9, "explore": 1}


def test_food_scope_excludes_non_food_items_from_lists_and_targets():
    items = pd.DataFrame(
        {
            "item_id": ["milk", "bread", "soap", "dog"],
            "coarse_category": ["dairy eggs", "bakery", "personal care", "pets"],
            "fine_category": ["milk", "bread", "soap", "dog food care"],
        }
    )
    history = frame([("u1", "b1", 1, "milk"), ("u1", "b1", 1, "soap"), ("u1", "b2", 2, "dog")])
    interactions = build_interactions(history, items)
    in_scope = scope_mask(items, interactions.items, "food", ["personal care", "pets"])
    assert in_scope.tolist() == [True, True, False, False]
    repeat = repeat_ranking(repeat_scores(interactions, 0.9), in_scope)
    explore = ranked([(0, 1, 1.0), (0, 2, 0.5)])
    explore = explore[in_scope[explore["item"].to_numpy()]]
    lists = to_lists(allocate_slots(repeat, explore, np.array([5]), top_k=10), interactions.users, interactions.items)
    assert lists == {"u1": ["milk", "bread"]}

    targets, excluded = scope_targets({"u1": {"milk", "soap"}, "u2": {"dog"}}, {"milk", "bread"})
    assert targets == {"u1": {"milk"}} and excluded == 1
    assert scope_targets({"u1": {"soap"}}, None) == ({"u1": {"soap"}}, 0)


def test_repeat_ratio_counts_from_second_basket_within_scope():
    history = frame(
        [
            ("u1", "b1", 1, "a"),
            ("u1", "b2", 2, "a"), ("u1", "b2", 2, "b"),
            ("u1", "b3", 3, "a"), ("u1", "b3", 3, "b"), ("u1", "b3", 3, "c"),
            ("u2", "b4", 1, "a"),
        ]
    )
    interactions = build_interactions(history, catalogue(["a", "b", "c"]))
    n, n_rep = repeat_counts(interactions, np.array([True, True, True]))
    assert n.tolist() == [5, 0] and n_rep.tolist() == [3, 0]
    rho, rho_bar = repeat_ratio(n, n_rep, kappa=5)
    assert rho_bar == pytest.approx(3 / 5)
    assert rho.tolist() == pytest.approx([(3 + 5 * 0.6) / (5 + 5), 0.6])
    n_food, n_rep_food = repeat_counts(interactions, np.array([True, False, True]))
    assert n_food.tolist() == [3, 0] and n_rep_food.tolist() == [2, 0]
