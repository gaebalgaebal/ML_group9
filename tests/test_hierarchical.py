import numpy as np
import pandas as pd
import pytest

from ml_group9.hierarchical import (
    PSEUDOCOUNT,
    Params,
    attribute_log_prob,
    build_hier_state,
    generate_candidates,
    item_cell_prob,
    recommend_hier,
    stage_probabilities,
    type_preferences,
    user_neighbors,
)


COLUMNS = ["dataset", "user_id", "basket_id", "basket_seq", "item_id", "coarse_category", "fine_category", "quantity"]
GAMMA = 0.5


def config(knn=(0, 2)):
    return {
        "attributes": {
            "min_with": 1,
            "min_without": 1,
            "smoothing": 2,
            "exclude_departments": {"instacart": ["personal care"]},
        },
        "scope": {"nonfood_departments": {"instacart": ["personal care"]}},
        "evaluation": {"top_k": 3},
        "hierarchical": {
            "stage1_key": {"instacart": "fine_category"},
            "type_key": {"instacart": "ingredient_type"},
            "attributes": {"instacart": ["fat_reduced"]},
            "type_attr_min_with": 2,
            "type_attr_min_without": 2,
            "knn_k": list(knn),
            "candidate_cells": 2,
            "items_per_cell": 2,
            "user_block": 2,
        },
    }


def items():
    # item, department, aisle, type, fat_reduced
    rows = [
        ("m1", "dairy", "milk", "i_milk", 1), ("m2", "dairy", "milk", "i_milk", 1),
        ("m3", "dairy", "milk", "i_milk", 0), ("m4", "dairy", "milk", "i_milk", 0),
        ("c1", "dairy", "milk", "i_cream", 0), ("c2", "dairy", "milk", "i_cream", 1),
        ("b1", "bakery", "bread", "i_bread", 0), ("b2", "bakery", "bread", "i_bread", 0),
        ("b3", "bakery", "bread", "aisle:bread", 0),
        ("s1", "personal care", "soap", "aisle:soap", 1), ("s2", "personal care", "soap", "aisle:soap", 0),
    ]
    frame = pd.DataFrame(
        rows, columns=["item_id", "coarse_category", "fine_category", "ingredient_type", "attr_fat_reduced"]
    )
    return frame.astype({"attr_fat_reduced": "int8"})


def purchases(rows):
    category = items().set_index("item_id")
    return pd.DataFrame(
        [("instacart", u, b, s, i, category.loc[i, "coarse_category"], category.loc[i, "fine_category"], 1) for u, b, s, i in rows],
        columns=COLUMNS,
    )


def history():
    return purchases(
        [
            ("u1", "b1", 1, "m1"), ("u1", "b1", 1, "m2"), ("u1", "b1", 1, "b1"),
            ("u1", "b2", 2, "m1"), ("u1", "b2", 2, "s1"),
            ("u2", "b3", 1, "m3"), ("u2", "b3", 1, "c1"), ("u2", "b4", 2, "b2"), ("u2", "b4", 2, "b3"),
            ("u3", "b5", 1, "b1"), ("u3", "b6", 2, "b1"), ("u3", "b6", 2, "b2"),
            ("u4", "b7", 1, "s2"), ("u4", "b8", 2, "m4"),
        ]
    )


def target(extra=()):
    return purchases([("u1", "t1", 3, "m3"), ("u2", "t2", 3, "m1"), ("u3", "t3", 3, "b3")] + list(extra))


def state(knn=(0, 2), history_frame=None, target_frame=None):
    return build_hier_state(
        config(knn), "instacart", items(), history() if history_frame is None else history_frame,
        target() if target_frame is None else target_frame, GAMMA,
    )


def rows_of(hier_state, users):
    return hier_state.interactions.users.get_indexer(users).astype(np.int64)


@pytest.mark.parametrize("k", [0, 2])
def test_theta_and_tau_are_positive_and_sum_to_one(k):
    hier_state = state()
    hierarchy = hier_state.hierarchies["full"]
    rows = np.arange(hier_state.n_users)
    theta, tau = stage_probabilities(hier_state, hierarchy, Params(alpha=3.0, mu=1.0, lambda_type=2.0, k=k), rows)
    assert (theta > 0).all() and (tau > 0).all()
    assert theta.sum(axis=1) == pytest.approx(np.ones(len(rows)))
    for stage1 in range(len(hier_state.stage1_labels)):
        types = hierarchy.type_stage1 == stage1
        assert tau[:, types].sum(axis=1) == pytest.approx(np.ones(len(rows)))


def test_theta_and_tau_match_hand_calculation_without_neighbours():
    hier_state = state()
    hierarchy = hier_state.hierarchies["full"]
    assert hier_state.stage1_labels.tolist() == ["bread", "milk"]
    row = rows_of(hier_state, ["u1"])
    alpha = 2.0
    theta, tau = stage_probabilities(hier_state, hierarchy, Params(alpha, 1.0, 2.0, 0), row)
    # u1: milk in b1 (weight 0.5) and b2 (1.0); bread in b1. The soap purchase is outside F.
    s_milk, s_bread = 1.5, 0.5
    population = np.array([0.5 + 1.0 + 0.5 + 1.0, 1.5 + 0.5 + 1.0]) + PSEUDOCOUNT  # bread, milk over users
    theta_bar = population / population.sum()
    assert hier_state.theta_bar == pytest.approx(theta_bar)
    expected = (np.array([s_bread, s_milk]) + alpha * theta_bar) / (s_milk + s_bread + alpha)
    assert theta[0] == pytest.approx(expected)
    labels = hierarchy.type_labels.tolist()
    milk, cream = labels.index("milk|i_milk"), labels.index("milk|i_cream")
    tau_bar_milk = (1.5 + 0.5 + 1.0 + PSEUDOCOUNT) / (1.5 + 0.5 + 1.0 + 0.5 + 2 * PSEUDOCOUNT)
    assert hierarchy.tau_bar[milk] == pytest.approx(tau_bar_milk)
    assert tau[0, milk] == pytest.approx((1.5 + alpha * tau_bar_milk) / (1.5 + alpha))
    assert tau[0, cream] == pytest.approx((alpha * (1 - tau_bar_milk)) / (1.5 + alpha))


def test_neighbour_prior_averages_neighbour_shares_with_population():
    hier_state = state()
    hierarchy = hier_state.hierarchies["full"]
    row = rows_of(hier_state, ["u3"])
    neighbors = hier_state.neighbors[row[0], :2]
    valid = neighbors[neighbors >= 0]
    alpha = 4.0
    theta, tau = stage_probabilities(hier_state, hierarchy, Params(alpha, 1.0, 2.0, 2), row)
    prior = (hier_state.theta_hat[valid].sum(axis=0) + hier_state.theta_bar) / (len(valid) + 1)
    counts = hier_state.stage1_counts[row[0]]
    assert theta[0] == pytest.approx((counts + alpha * prior) / (counts.sum() + alpha))
    # u3 bought only bread; neighbours who never bought milk do not inform the milk types.
    labels = hierarchy.type_labels.tolist()
    milk = labels.index("milk|i_milk")
    buyers = [v for v in valid if hierarchy.stage1_mass[v, hierarchy.type_stage1[milk]] > 0]
    tau_prior = (sum(hierarchy.tau_hat[v, milk] for v in buyers) + hierarchy.tau_bar[milk]) / (len(buyers) + 1)
    assert tau[0, milk] == pytest.approx(alpha * tau_prior / alpha)


def test_attribute_stage_sums_to_one_and_matches_two_level_shrinkage():
    hier_state = state()
    hierarchy = hier_state.hierarchies["full"]
    labels = hierarchy.type_labels.tolist()
    milk = labels.index("milk|i_milk")
    assert hierarchy.type_applicable[:, 0].tolist() == [label == "milk|i_milk" for label in labels]
    index = hier_state.interactions.items
    with_attr, without_attr = index.get_loc("m1"), index.get_loc("m3")
    lambda_type = 2.0
    for user in ["u1", "u2", "u3", "u4"]:
        row = rows_of(hier_state, [user])
        users = np.repeat(row, 2)
        probability = np.exp(attribute_log_prob(hier_state, hierarchy, lambda_type, users, np.array([with_attr, without_attr])))
        assert probability.sum() == pytest.approx(1.0) and (probability > 0).all()
    # u1 bought m1 twice and m2 once inside i_milk: n_u(t) = 3, n_u(a, t) = 3.
    row = rows_of(hier_state, ["u1"])
    pi_user = hier_state.aisle_model.pi[row[0], 0]
    pi = type_preferences(hier_state, hierarchy, lambda_type, row, np.array([milk]))[0, 0]
    assert pi == pytest.approx((3 + lambda_type * pi_user) / (3 + lambda_type))
    # u3 never bought i_milk: n_u(t) = 0, so pi_u(a | t) = pi_u(a).
    row = rows_of(hier_state, ["u3"])
    pi = type_preferences(hier_state, hierarchy, lambda_type, row, np.array([milk]))[0, 0]
    assert pi == pytest.approx(hier_state.aisle_model.pi[row[0], 0])
    # Items of types outside C_a^type have A = 1.
    cream = index.get_loc("c2")
    assert attribute_log_prob(hier_state, hierarchy, lambda_type, row, np.array([cream]))[0] == 0.0


def test_item_stage_sums_to_one_inside_each_cell():
    hier_state = state()
    hierarchy = hier_state.hierarchies["full"]
    mu = 1.5
    for user in ["u1", "u2"]:
        row = rows_of(hier_state, [user])
        repeat = hier_state.repeat[row]
        for cell in range(len(hierarchy.cell_ptr) - 1):
            members = hierarchy.cell_members[hierarchy.cell_ptr[cell] : hierarchy.cell_ptr[cell + 1]]
            phi = item_cell_prob(
                hierarchy, mu, repeat, np.zeros(len(members), dtype=np.int64), np.repeat(row, len(members)), members
            )
            assert phi.sum() == pytest.approx(1.0) and (phi > 0).all()
    # u1, cell (i_milk, fat_reduced = 1) = {m1, m2}: s_rep = 0.5 + 1 and 0.5; popularity 2 and 1.
    index = hier_state.interactions.items
    m1, m2 = index.get_loc("m1"), index.get_loc("m2")
    row = rows_of(hier_state, ["u1"])
    phi = item_cell_prob(hierarchy, mu, hier_state.repeat[row], np.zeros(2, dtype=np.int64), np.repeat(row, 2), np.array([m1, m2]))
    pop = np.array([2 + PSEUDOCOUNT, 1 + PSEUDOCOUNT]) / (3 + 2 * PSEUDOCOUNT)
    assert phi == pytest.approx((np.array([1.5, 0.5]) + mu * pop) / (2.0 + mu))


def test_neighbours_exclude_self_and_blocks_match_dense():
    rng = np.random.default_rng(3)
    profiles = rng.random((9, 4)) * (rng.random((9, 4)) > 0.3)
    profiles[4] = 0.0
    neighbors, sims = user_neighbors(profiles, 3, block=2)
    unit = profiles / np.maximum(np.linalg.norm(profiles, axis=1), 1e-12)[:, None]
    dense = unit @ unit.T
    np.fill_diagonal(dense, -np.inf)
    for user in range(9):
        assert user not in neighbors[user]
        order = np.argsort(-dense[user], kind="stable")[:3]
        expected = [int(v) for v in order if dense[user, v] > 0]
        assert neighbors[user][: len(expected)].tolist() == expected
        assert (neighbors[user][len(expected) :] == -1).all()
        assert sims[user][: len(expected)] == pytest.approx(dense[user, expected])
    assert (neighbors[4] == -1).all()
    assert (neighbors[:, :] != 4).all()


def test_k_zero_uses_the_population_prior():
    hier_state = state()
    hierarchy = hier_state.hierarchies["full"]
    rows = np.arange(hier_state.n_users)
    theta, tau = stage_probabilities(hier_state, hierarchy, Params(1e9, 1.0, 2.0, 0), rows)
    assert theta == pytest.approx(np.tile(hier_state.theta_bar, (len(rows), 1)), rel=1e-6)
    assert tau == pytest.approx(np.tile(hierarchy.tau_bar, (len(rows), 1)), rel=1e-6)


def test_ablation_switches():
    hier_state = state()
    noattr = hier_state.hierarchies["noattr"]
    rows = np.repeat(rows_of(hier_state, ["u1"]), hier_state.n_items)
    in_scope = np.flatnonzero(hier_state.in_scope)
    assert (attribute_log_prob(hier_state, noattr, 2.0, rows[: len(in_scope)], in_scope) == 0).all()
    # Without attributes every type is a single cell.
    assert len(set(zip(noattr.item_type[in_scope], noattr.item_cell[in_scope]))) == len(set(noattr.item_type[in_scope]))
    notype = hier_state.hierarchies["notype"]
    assert len(notype.type_stage1) == len(hier_state.stage1_labels)
    _, tau = stage_probabilities(hier_state, notype, Params(2.0, 1.0, 2.0, 2), np.arange(hier_state.n_users))
    assert tau == pytest.approx(np.ones_like(tau))
    # notype applies the attribute to the whole aisle: milk has 3 fat-reduced and 3 other items.
    assert notype.type_applicable[:, 0].tolist() == [False, True]


def test_candidates_contain_history_items_in_f_and_nothing_outside_f():
    hier_state = state()
    hierarchy = hier_state.hierarchies["full"]
    rows = np.arange(hier_state.n_users)
    theta, tau = stage_probabilities(hier_state, hierarchy, Params(2.0, 1.0, 2.0, 0), rows)
    local, items_, seen = generate_candidates(hier_state, hierarchy, theta[:, hierarchy.type_stage1] * tau, rows, 2, 2)
    assert hier_state.in_scope[items_].all()
    pairs = set(zip(local.tolist(), items_.tolist()))
    assert len(pairs) == len(local)
    index = hier_state.interactions.items
    for user, row in zip(hier_state.interactions.users, rows):
        own = {int(index.get_loc(item)) for item in hier_state.history_items[user] if hier_state.in_scope[index.get_loc(item)]}
        assert own <= {item for r, item in pairs if r == row}
    assert seen.sum() == hier_state.seen.nnz
    # u4's soap purchase is not a candidate.
    assert index.get_loc("s2") not in items_


def test_recommendations_rank_candidates_exactly():
    hier_state = state()
    result = recommend_hier(hier_state, "hier_full", Params(2.0, 1.0, 2.0, 2), config())
    assert set(result.lists) == set(hier_state.targets)
    for user, items_ in result.lists.items():
        assert len(items_) == len(set(items_)) <= 3
        assert not {"s1", "s2"} & set(items_)
    assert 0.0 <= result.candidate_recall <= 1.0


def snapshot(target_frame):
    hier_state = state(target_frame=target_frame)
    hierarchy = hier_state.hierarchies["full"]
    rows = np.arange(hier_state.n_users)
    params = Params(2.0, 1.0, 2.0, 2)
    theta, tau = stage_probabilities(hier_state, hierarchy, params, rows)
    in_scope = np.flatnonzero(hier_state.in_scope)
    users = np.repeat(rows, len(in_scope))
    items_ = np.tile(in_scope, len(rows))
    return {
        "theta": theta,
        "tau": tau,
        "pi": type_preferences(hier_state, hierarchy, 2.0, users, hierarchy.item_type[items_]),
        "phi": item_cell_prob(hierarchy, 1.0, hier_state.repeat, users, users, items_),
        "neighbors": hier_state.neighbors,
        "applicable": hierarchy.type_applicable,
        "pop_cell": hierarchy.pop_cell,
    }


def test_target_only_purchases_do_not_change_history_statistics():
    base = snapshot(target())
    leaked = snapshot(target([("u3", "t3", 3, "m1"), ("u3", "t3", 3, "m2"), ("u4", "t4", 3, "c2"), ("u4", "t4", 3, "b1")]))
    for key, value in base.items():
        assert np.array_equal(value, leaked[key]), key
