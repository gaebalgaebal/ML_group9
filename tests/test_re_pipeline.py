import numpy as np
import pandas as pd
import pandas.testing as pdt

from ml_group9.re_pipeline import build_gamma, build_scope, build_state, recommend, tune_split


COLUMNS = ["dataset", "user_id", "basket_id", "basket_seq", "item_id", "coarse_category", "fine_category", "quantity"]


def config():
    return {
        "item_cf": {"top_k": 5, "block_size": 4},
        "evaluation": {"top_k": 3},
        "attributes": {
            "min_with": 1,
            "min_without": 1,
            "smoothing": 2,
            "distinct_min_n": 1,
            "exclude_departments": {"d": ["personal care"]},
        },
        "explore": {"epsilon": 1e-4, "max_candidates": 50, "pref_items_per_attr": 2, "pref_min_n_c": 1},
        "scope": {"modes": ["all", "food"], "nonfood_departments": {"d": ["personal care"]}},
        "repeat_explore": {"gammas": [0.8, 1.0], "kappas": [5], "betas": [0, 1], "global_r": [1, 2]},
        "health_rerank": {
            "attributes": ["fat_reduced", "gluten_free", "low_sodium"],
            "min_n_c": 10,
            "min_lift": 0.05,
            "slot_grid": [0, 1, 2],
            "soft_lambdas": [0, 0.1, 0.5],
            "recent_baskets": 2,
            "recent_min_purchases": 1,
            "recent_strict_min_purchases": 2,
            "recent_rate_min_n_c": 1,
            "half_min_n_c": 1,
            "persistence_modes": ["recent_2", "recent_rate", "both_halves"],
            "candidate_enabled": True,
            "candidate_items_per_aisle": 2,
            "candidate_max_per_user": 10,
            "candidate_soft_lambdas": [0, 1, 5],
        },
    }


def items():
    rows = [
        ("m1", "dairy", "milk", 1), ("m2", "dairy", "milk", 0), ("m3", "dairy", "milk", 1),
        ("y1", "dairy", "yogurt", 1), ("y2", "dairy", "yogurt", 0),
        ("s1", "personal care", "soap", 1), ("s2", "personal care", "soap", 0),
    ]
    frame = pd.DataFrame(rows, columns=["item_id", "coarse_category", "fine_category", "attr_organic"])
    frame["attr_organic"] = frame["attr_organic"].astype("int8")
    return frame


def history():
    category = items().set_index("item_id")
    rows = [
        ("u1", "b1", 1, "m1"), ("u1", "b1", 1, "y1"), ("u1", "b2", 2, "m1"), ("u1", "b2", 2, "s1"),
        ("u2", "b3", 1, "m2"), ("u2", "b3", 1, "y2"), ("u2", "b4", 2, "m2"), ("u2", "b4", 2, "m1"),
        ("u3", "b5", 1, "y1"), ("u3", "b5", 1, "s2"), ("u3", "b6", 2, "m3"),
    ]
    return pd.DataFrame(
        [("d", u, b, s, i, category.loc[i, "coarse_category"], category.loc[i, "fine_category"], 1) for u, b, s, i in rows],
        columns=COLUMNS,
    )


def target(extra=()):
    rows = [("u1", "t1", 3, "m3"), ("u2", "t2", 3, "m2"), ("u3", "t3", 3, "s1")] + list(extra)
    return pd.DataFrame([("d", u, b, s, i, "x", "x", 1) for u, b, s, i in rows], columns=COLUMNS)


def snapshot(target_frame):
    state = build_state(config(), "d", items(), history(), target_frame)
    result = {
        "pi": state.preferences.pi.copy(),
        "prior": state.preferences.prior.copy(),
        "recent_n_a": state.preferences.recent_n_a.copy(),
        "recent_n_c": state.preferences.recent_n_c.copy(),
        "early_n_a": state.preferences.early_n_a.copy(),
        "early_n_c": state.preferences.early_n_c.copy(),
        "late_n_a": state.preferences.late_n_a.copy(),
        "late_n_c": state.preferences.late_n_c.copy(),
    }
    for mode in ["all", "food"]:
        scope = build_scope(state, mode, config())
        gamma_state = build_gamma(state, scope, 0.8, config())
        result[mode] = (scope.n.copy(), scope.n_rep.copy(), gamma_state.repeat_ranked, gamma_state.candidates)
    return result


def test_target_only_purchases_do_not_change_history_statistics():
    base = snapshot(target())
    # Target-only organic purchases, a target-only item and a repeat of a history item.
    leaked = snapshot(target([("u2", "t2", 3, "m3"), ("u2", "t2", 3, "y1"), ("u3", "t3", 3, "m1")]))
    assert np.array_equal(base["pi"], leaked["pi"])
    assert np.array_equal(base["prior"], leaked["prior"])
    for key in ["recent_n_a", "recent_n_c", "early_n_a", "early_n_c", "late_n_a", "late_n_c"]:
        assert np.array_equal(base[key], leaked[key])
    for mode in ["all", "food"]:
        assert np.array_equal(base[mode][0], leaked[mode][0])
        assert np.array_equal(base[mode][1], leaked[mode][1])
        pdt.assert_frame_equal(base[mode][2], leaked[mode][2])
        pdt.assert_frame_equal(base[mode][3], leaked[mode][3])


def test_recent_attribute_counts_use_history_baskets_only():
    state = build_state(config(), "d", items(), history(), target())
    column = state.preferences.attributes.index("organic")
    rows = state.interactions.users.get_indexer(["u1", "u2", "u3"])
    assert state.preferences.recent_n_a[rows, column].tolist() == [3.0, 1.0, 2.0]
    assert state.preferences.recent_n_c[rows, column].tolist() == [3.0, 4.0, 2.0]
    assert state.preferences.early_n_a[rows, column].tolist() == [2.0, 0.0, 1.0]
    assert state.preferences.early_n_c[rows, column].tolist() == [2.0, 2.0, 1.0]
    assert state.preferences.late_n_a[rows, column].tolist() == [1.0, 1.0, 1.0]
    assert state.preferences.late_n_c[rows, column].tolist() == [1.0, 2.0, 1.0]


def test_food_scope_drops_non_food_targets_and_items_end_to_end():
    state = build_state(config(), "d", items(), history(), target())
    assert state.category_sizes == {
        "before_nonfood_exclusion": {"organic": 3},
        "after": {"organic": 2},
    }
    scope = build_scope(state, "food", config())
    assert scope.excluded_users == 1  # u3's only target item is soap
    assert set(scope.targets) == {"u1", "u2"}
    recommendation = recommend(state, scope, build_gamma(state, scope, 0.9, config()), 5.0, 1.0, config())
    for user, items_ in recommendation.final.items():
        assert not {"s1", "s2"} & set(items_)
        assert len(items_) == len(set(items_)) <= 3


def test_tuning_selects_from_grids_per_scope():
    state = build_state(config(), "d", items(), history(), target())
    result = tune_split(config(), state)
    assert set(result["scopes"]) == {"all", "food"}
    for scope in result["scopes"].values():
        assert len(scope["grid_gamma_kappa"]) == 2
        assert len(scope["grid_beta"]) == 2 and len(scope["grid_global_r"]) == 2
        assert len(scope["grid_health_slots"]) == 3
        assert len(scope["grid_health_soft"]) == 3
        assert len(scope["grid_health_soft_recent"]) == 3
        assert set(scope["grid_health_persistence"]) == {"recent_2", "recent_rate", "both_halves"}
        assert all(len(rows) == 3 for rows in scope["grid_health_persistence"].values())
        assert len(scope["grid_health_candidate_soft"]) == 3
        selected = scope["selected"]
        assert selected["gamma"] in {0.8, 1.0} and selected["beta"] in {0.0, 1.0} and selected["global_r"] in {1, 2}
        assert selected["health_slots"] in {0, 1, 2}
        assert selected["health_lambda"] in {0.0, 0.1, 0.5}
        assert selected["health_recent_lambda"] in {0.0, 0.1, 0.5}
        assert set(selected["health_persistence_lambdas"]) == {"recent_2", "recent_rate", "both_halves"}
        assert selected["health_candidate_lambda"] in {0.0, 1.0, 5.0}


def test_health_gate_never_changes_repeat_ranking_or_repeat_slots():
    settings = config()
    settings["health_rerank"] = {
        "attributes": ["organic"],
        "min_n_c": 1,
        "min_lift": 0.0,
        "slot_grid": [0, 1, 2],
        "soft_lambdas": [0, 0.5],
        "recent_baskets": 2,
        "recent_min_purchases": 1,
        "recent_strict_min_purchases": 2,
        "recent_rate_min_n_c": 1,
        "half_min_n_c": 1,
        "persistence_modes": ["recent_2", "recent_rate", "both_halves"],
        "candidate_enabled": True,
        "candidate_items_per_aisle": 2,
        "candidate_max_per_user": 10,
        "candidate_soft_lambdas": [0, 1, 5],
    }
    state = build_state(settings, "d", items(), history(), target())
    scope = build_scope(state, "all", settings)
    gamma_state = build_gamma(state, scope, 0.8, settings)
    base = recommend(state, scope, gamma_state, 5.0, 0.0, settings, health_slots=0)
    gated = recommend(state, scope, gamma_state, 5.0, 0.0, settings, health_slots=2)
    assert base.repeat == gated.repeat
    base_repeat = base.allocation[base.allocation["slot"].eq("repeat")][["user", "item", "position"]]
    gated_repeat = gated.allocation[gated.allocation["slot"].eq("repeat")][["user", "item", "position"]]
    pdt.assert_frame_equal(base_repeat.reset_index(drop=True), gated_repeat.reset_index(drop=True))
    promoted = gated.allocation[gated.allocation["slot"].eq("explore")].groupby("user")["health_promoted"].sum()
    assert (promoted <= 2).all()

    soft = recommend(state, scope, gamma_state, 5.0, 0.0, settings, health_lambda=0.5)
    assert base.repeat == soft.repeat
    soft_repeat = soft.allocation[soft.allocation["slot"].eq("repeat")][["user", "item", "position"]]
    pdt.assert_frame_equal(base_repeat.reset_index(drop=True), soft_repeat.reset_index(drop=True))
