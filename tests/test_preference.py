import pandas as pd
import pytest

from ml_group9.attributes import congruence, distinct_preference_users
from ml_group9.recommender import FEATURE_COLUMNS_PREF, build_context, build_feature_frame


COLUMNS = ["dataset", "user_id", "basket_id", "basket_seq", "item_id", "coarse_category", "fine_category", "quantity"]


def config():
    return {
        "candidates": {"recent_items": 10, "neighbors_per_item": 20, "preferred_categories": 3, "category_popular": 50, "global_popular": 50, "max_per_user": 50},
        "item_cf": {"top_k": 20, "block_size": 2},
        "attributes": {"min_with": 1, "min_without": 1, "smoothing": 2},
    }


def items():
    # "dairy" is the only category with both organic and non-organic items, so C_organic = {dairy}.
    return pd.DataFrame(
        {
            "item_id": ["org1", "org2", "plain", "snack", "org_snack_free"],
            "coarse_category": ["food"] * 5,
            "fine_category": ["dairy", "dairy", "dairy", "snacks", "produce"],
            "attr_organic": pd.Series([1, 1, 0, 0, 1], dtype="int8"),
        }
    )


def history():
    rows = [
        ("d", "u1", "b1", 1, "org1"),
        ("d", "u1", "b1", 1, "org1"),  # duplicate line in the same basket counts once
        ("d", "u1", "b2", 2, "org2"),
        ("d", "u1", "b2", 2, "plain"),
        ("d", "u1", "b2", 2, "snack"),  # outside C_organic
        ("d", "u2", "b3", 1, "plain"),
        ("d", "u2", "b3", 1, "snack"),
    ]
    category = {"org1": "dairy", "org2": "dairy", "plain": "dairy", "snack": "snacks", "org_snack_free": "produce"}
    return pd.DataFrame(
        [(d, u, b, s, i, "food", category[i], 1) for d, u, b, s, i in rows], columns=COLUMNS
    )


def test_smoothed_preference_matches_hand_calculation():
    context = build_context(history(), items(), config())
    # u1: n_a = 2, n_C = 3; u2: n_a = 0, n_C = 1; pi_bar = 2 / 4.
    assert context.attribute_prior["organic"] == pytest.approx(0.5)
    assert context.attribute_preferences[("u1", "organic")] == pytest.approx((2 + 2 * 0.5) / (3 + 2))
    assert context.attribute_preferences[("u2", "organic")] == pytest.approx((0 + 2 * 0.5) / (1 + 2))
    flags = context.item_attribute_flags
    assert congruence(flags["org1"], "u1", context.attribute_preferences, context.attribute_prior) == pytest.approx(0.6)
    assert congruence(flags["plain"], "u1", context.attribute_preferences, context.attribute_prior) == pytest.approx(0.4)
    # A user without history in C_a falls back to pi_bar.
    assert congruence(flags["org1"], "new", context.attribute_preferences, context.attribute_prior) == pytest.approx(0.5)
    assert distinct_preference_users(context.attribute_counts, context.attribute_preferences, context.attribute_prior, 2, 0.05) == {"u1"}
    assert distinct_preference_users(context.attribute_counts, context.attribute_preferences, context.attribute_prior, 2, 0.2) == set()


def test_items_outside_applicable_categories_get_neutral_congruence():
    target = pd.DataFrame([("d", "u1", "t1", 3, "snack", "food", "snacks", 1)], columns=COLUMNS)
    features, _, context = build_feature_frame(history(), items(), target, config())
    assert set(FEATURE_COLUMNS_PREF).issubset(features.columns)
    assert "snack" not in context.item_attribute_flags
    snack = features[features["item_id"].eq("snack")].iloc[0]
    assert snack["pref_congruence"] == 0.5
    assert snack["pref_applicable"] == 0
    org1 = features[features["item_id"].eq("org1")].iloc[0]
    assert org1["pref_applicable"] == 1
    assert org1["pref_congruence"] == pytest.approx(0.6)


def test_target_only_attribute_purchases_do_not_change_preferences():
    plain_target = pd.DataFrame([("d", "u2", "t1", 2, "plain", "food", "dairy", 1)], columns=COLUMNS)
    organic_target = pd.DataFrame(
        [("d", "u2", "t1", 2, item, "food", "dairy", 1) for item in ["org1", "org2"]], columns=COLUMNS
    )
    _, _, plain_context = build_feature_frame(history(), items(), plain_target, config())
    _, _, organic_context = build_feature_frame(history(), items(), organic_target, config())
    assert organic_context.attribute_preferences == plain_context.attribute_preferences
    assert organic_context.attribute_prior == plain_context.attribute_prior
