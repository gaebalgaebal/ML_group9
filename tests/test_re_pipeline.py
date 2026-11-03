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
    result = {"pi": state.preferences.pi.copy(), "prior": state.preferences.prior.copy()}
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
    for mode in ["all", "food"]:
        assert np.array_equal(base[mode][0], leaked[mode][0])
        assert np.array_equal(base[mode][1], leaked[mode][1])
        pdt.assert_frame_equal(base[mode][2], leaked[mode][2])
        pdt.assert_frame_equal(base[mode][3], leaked[mode][3])


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
        selected = scope["selected"]
        assert selected["gamma"] in {0.8, 1.0} and selected["beta"] in {0.0, 1.0} and selected["global_r"] in {1, 2}
