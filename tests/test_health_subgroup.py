import numpy as np
import pandas as pd
import pytest

from ml_group9.health_subgroup import (
    build_health_split,
    config_subgroups,
    health_categories,
    health_preferences,
    select_attributes,
    subgroups,
)
from ml_group9.repeat_explore import PreferenceModel, build_interactions


COLUMNS = ["dataset", "user_id", "basket_id", "basket_seq", "item_id", "coarse_category", "fine_category", "quantity"]


def config():
    return {
        "item_cf": {"top_k": 5, "block_size": 4},
        "evaluation": {"top_k": 3},
        "attributes": {
            "min_with": 1,
            "min_without": 1,
            "smoothing": 2,
            "exclude_departments": {"instacart": ["personal care"]},
        },
        "explore": {"epsilon": 1e-4, "max_candidates": 50, "pref_items_per_attr": 2, "pref_min_n_c": 1},
        "scope": {"modes": ["food"], "nonfood_departments": {"instacart": ["personal care"]}},
        "health_subgroup": {
            "attributes": ["fat_reduced", "gluten_free"],
            "scopes": ["food"],
            "min_attribute_purchases": 1,
            "strict": {"min_n_c": 2, "min_gap": 0.2},
        },
    }


def items():
    # Columns: item, department, aisle, fat_reduced, gluten_free.
    rows = [
        ("m1", "dairy", "milk", 1, 0), ("m2", "dairy", "milk", 0, 0), ("m3", "dairy", "milk", 1, 0),
        ("b1", "bakery", "bread", 0, 1), ("b2", "bakery", "bread", 0, 0), ("b3", "bakery", "bread", 0, 1),
        ("s1", "personal care", "soap", 1, 1), ("s2", "personal care", "soap", 0, 0),
    ]
    frame = pd.DataFrame(
        rows, columns=["item_id", "coarse_category", "fine_category", "attr_fat_reduced", "attr_gluten_free"]
    )
    return frame.astype({"attr_fat_reduced": "int8", "attr_gluten_free": "int8"})


def purchases(rows):
    category = items().set_index("item_id")
    return pd.DataFrame(
        [("instacart", u, b, s, i, category.loc[i, "coarse_category"], category.loc[i, "fine_category"], 1) for u, b, s, i in rows],
        columns=COLUMNS,
    )


def history():
    return purchases(
        [
            # u1: one in-C_a fat-reduced purchase.
            ("u1", "b1", 1, "m1"), ("u1", "b1", 1, "b2"), ("u1", "b2", 2, "m2"),
            # u2: fat-reduced and gluten-free only in the non-food soap aisle.
            ("u2", "b3", 1, "s1"), ("u2", "b3", 1, "m2"), ("u2", "b4", 2, "b2"),
            # u3: no attribute purchase in history.
            ("u3", "b5", 1, "m2"), ("u3", "b5", 1, "b2"),
            # u4: gluten-free buyer with many bread purchases.
            ("u4", "b6", 1, "b1"), ("u4", "b6", 1, "b3"), ("u4", "b7", 2, "b1"), ("u4", "b7", 2, "m3"),
        ]
    )


def model_from(history_frame):
    interactions = build_interactions(history_frame, items())
    return interactions, health_preferences(config(), "instacart", items(), history_frame, interactions)


def test_health_categories_drop_non_food_aisles():
    assert health_categories(config(), "instacart", items()) == {"fat_reduced": {"milk"}, "gluten_free": {"bread"}}


def test_members_need_an_attribute_purchase_inside_c_a():
    interactions, model = model_from(history())
    groups = config_subgroups(model, config())
    member = {
        name: set(interactions.users[groups.members[:, column]]) for column, name in enumerate(groups.attributes)
    }
    assert member["fat_reduced"] == {"u1", "u4"}  # u4 bought m3; u2's soap purchase is outside C_a
    assert member["gluten_free"] == {"u4"}
    assert set(interactions.users[groups.any_member]) == {"u1", "u4"}
    row = interactions.users.get_loc("u1")
    assert model.n_a[row].tolist() == [1, 0] and model.n_c[row].tolist() == [2, 1]


def target(extra=()):
    return purchases([("u1", "t1", 3, "m2"), ("u2", "t2", 3, "b2"), ("u3", "t3", 3, "m2")] + list(extra))


def health_snapshot(target_frame):
    split = build_health_split(config(), "instacart", items(), history(), target_frame)
    shape = split.state.interactions.shape
    users = np.repeat(np.arange(shape[0]), shape[1])
    item_rows = np.tile(np.arange(shape[1]), shape[0])
    return {
        "pi": split.model.pi,
        "prior": split.model.prior,
        "n_a": split.model.n_a,
        "n_c": split.model.n_c,
        "members": split.groups.members,
        "strict": split.groups.strict,
        "ell": split.model.log_ratio(users, item_rows),
        "ell_masked": split.model.log_ratio(users, item_rows, user_mask=split.groups.members),
    }


def test_target_only_purchases_do_not_change_subgroups_or_preferences():
    base = health_snapshot(target())
    # u3 buys fat-reduced and gluten-free items only in the target; u2 a new C_a attribute item.
    leaked = health_snapshot(target([("u3", "t3", 3, "m1"), ("u3", "t3", 3, "b1"), ("u2", "t2", 3, "m3")]))
    for key, value in base.items():
        assert np.array_equal(value, leaked[key]), key
    assert not base["members"].any(axis=1)[2]  # u3 stays outside G_H


def strict_model(pi, n_c, n_a):
    return PreferenceModel(
        attributes=["fat_reduced"],
        prior=np.array([0.3]),
        pi=np.array(pi, dtype=float).reshape(-1, 1),
        n_c=np.array(n_c, dtype=float).reshape(-1, 1),
        item_flag=np.zeros((1, 1), dtype=np.int8),
        item_applicable=np.zeros((1, 1), dtype=bool),
        n_a=np.array(n_a, dtype=float).reshape(-1, 1),
    )


def test_strict_subgroup_boundaries():
    model = strict_model(
        pi=[0.5, 0.5, 0.49, 0.1, 0.9, 0.5],
        n_c=[10, 9, 10, 30, 30, 10],
        n_a=[5, 5, 5, 1, 27, 0],
    )
    groups = subgroups(model, min_attribute_purchases=1, min_n_c=10, min_gap=0.2)
    assert groups.members[:, 0].tolist() == [True, True, True, True, True, False]
    # n_C = 10 and gap 0.20 pass; n_C = 9, gap 0.19, pi < pi_bar, and non-members fail.
    assert groups.strict[:, 0].tolist() == [True, False, False, False, True, False]
    assert groups.any_strict.tolist() == groups.strict[:, 0].tolist()


def masked_model():
    # Items: 0 = fat-reduced milk, 1 = plain milk, 2 = gluten-free bread, 3 = fat-reduced and gluten-free bread.
    return PreferenceModel(
        attributes=["fat_reduced", "gluten_free"],
        prior=np.array([0.3, 0.2]),
        pi=np.array([[0.6, 0.5], [0.1, 0.4]]),
        n_c=np.array([[20.0, 10.0], [20.0, 10.0]]),
        item_flag=np.array([[1, 0], [0, 0], [0, 1], [1, 1]], dtype=np.int8),
        item_applicable=np.array([[True, False], [True, False], [False, True], [True, True]]),
        n_a=np.array([[12.0, 5.0], [2.0, 4.0]]),
    )


def test_log_ratio_user_mask():
    model = masked_model()
    users = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    items_ = np.array([0, 1, 2, 3, 0, 1, 2, 3])
    base = model.log_ratio(users, items_)
    assert np.array_equal(model.log_ratio(users, items_, user_mask=None), base)
    assert np.array_equal(model.log_ratio(users, items_, user_mask=np.ones((2, 2), dtype=bool)), base)
    mask = np.array([[True, False], [False, True]])
    masked = model.log_ratio(users, items_, user_mask=mask)
    fat = np.log(0.6 / 0.3)
    gluten = np.log(0.4 / 0.2)
    assert masked[:4] == pytest.approx([fat, np.log(0.4 / 0.7), 0.0, fat])
    assert masked[4:] == pytest.approx([0.0, 0.0, gluten, gluten])


def test_single_attribute_model_ignores_other_attribute_flags():
    model = masked_model()
    users = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    items_ = np.array([0, 1, 2, 3, 0, 1, 2, 3])
    single = select_attributes(model, ["fat_reduced"])
    flipped = select_attributes(model, ["fat_reduced", "gluten_free"])
    flipped.item_flag = flipped.item_flag.copy()
    flipped.item_flag[:, 1] = 1 - flipped.item_flag[:, 1]
    flipped_single = select_attributes(flipped, ["fat_reduced"])
    expected = model.log_ratio(users, items_, user_mask=np.array([[True, False], [True, False]]))
    assert single.attributes == ["fat_reduced"]
    assert single.log_ratio(users, items_) == pytest.approx(expected)
    assert np.array_equal(flipped_single.log_ratio(users, items_), single.log_ratio(users, items_))
    assert single.n_a[:, 0].tolist() == [12.0, 2.0]


def test_single_attribute_model_matches_a_model_built_from_one_c_a():
    interactions, model = model_from(history())
    single_config = {**config(), "health_subgroup": {**config()["health_subgroup"], "attributes": ["gluten_free"]}}
    direct = health_preferences(single_config, "instacart", items(), history(), interactions)
    sliced = select_attributes(model, ["gluten_free"])
    for field in ["prior", "pi", "n_c", "n_a", "item_flag", "item_applicable"]:
        assert np.array_equal(getattr(direct, field), getattr(sliced, field)), field
