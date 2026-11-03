import pandas as pd

from ml_group9.recommender import FEATURE_COLUMNS, build_feature_frame, item_cosine_neighbors


def config():
    return {
        "candidates": {"recent_items": 10, "neighbors_per_item": 20, "preferred_categories": 3, "category_popular": 50, "global_popular": 50, "max_per_user": 5},
        "item_cf": {"top_k": 20, "block_size": 2},
    }


def synthetic_history():
    return pd.DataFrame(
        [
            ("d", "u1", "b1", 1, "a", "food", "fruit", 1),
            ("d", "u1", "b1", 1, "b", "food", "dairy", 1),
            ("d", "u1", "b2", 2, "a", "food", "fruit", 1),
            ("d", "u2", "b3", 1, "a", "food", "fruit", 1),
            ("d", "u2", "b3", 1, "c", "food", "fruit", 1),
            ("d", "u2", "b4", 2, "c", "food", "fruit", 1),
        ],
        columns=["dataset", "user_id", "basket_id", "basket_seq", "item_id", "coarse_category", "fine_category", "quantity"],
    )


def test_sparse_cosine_finds_cooccurring_item():
    neighbors = item_cosine_neighbors(synthetic_history(), top_k=2, block_size=2)
    assert neighbors["a"][0][0] in {"b", "c"}
    assert all(item != "a" for item, _ in neighbors["a"])


def test_candidates_are_unique_bounded_and_target_does_not_enter_context():
    history = synthetic_history()
    items = pd.DataFrame({"item_id": ["a", "b", "c", "future"], "coarse_category": ["food"] * 4, "fine_category": ["fruit", "dairy", "fruit", "fruit"]})
    target = pd.DataFrame([("d", "u1", "target", 3, "future", "food", "fruit", 1)], columns=history.columns)
    features, target_sets, context = build_feature_frame(history, items, target, config())
    assert set(FEATURE_COLUMNS).issubset(features.columns)
    assert features.groupby("user_id")["item_id"].nunique().max() <= 5
    assert not features.duplicated(["user_id", "item_id"]).any()
    assert "future" not in context.item_popularity
    assert target_sets["u1"] == {"future"}
