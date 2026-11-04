import math

import pandas as pd
import pytest

from ml_group9.metrics import candidate_recall, list_metrics, paired_bootstrap, ranking_metrics, subset_recall


def test_ranking_metrics_match_hand_calculation():
    frame = pd.DataFrame({"user_id": ["u1", "u1", "u1"], "item_id": ["a", "b", "c"], "score": [0.9, 0.8, 0.7]})
    truth = {"u1": {"a", "c"}}
    metrics = ranking_metrics(frame, truth, "score", top_k=2, catalog_size=3)
    assert metrics["precision_at_k"] == 0.5
    assert metrics["recall_at_k"] == 0.5
    assert metrics["catalog_coverage"] == 2 / 3
    assert candidate_recall(frame, truth) == 1.0


def test_list_metrics_match_ranking_metrics_and_score_missing_lists_as_zero():
    lists = {"u1": ["a", "b", "c"]}
    truth = {"u1": {"a", "c"}, "u2": {"x"}}
    summary, per_user = list_metrics(lists, truth, top_k=2)
    assert per_user.loc["u1", "recall_at_k"] == 0.5
    assert per_user.loc["u1", "ndcg_at_k"] == pytest.approx(1 / (1 + 1 / math.log2(3)))
    assert per_user.loc["u2"].tolist() == [0.0, 0.0, 0.0, 0.0]
    assert summary["users_evaluated"] == 2
    assert summary["precision_at_k"] == pytest.approx(0.25)


def test_new_and_repeat_recall_hand_calculation_skip_empty_subsets():
    history_items = {"u1": {"a", "b"}, "u2": {"c"}, "u3": {"d"}}
    targets = {"u1": {"a", "x", "y"}, "u2": {"c"}, "u3": {"z"}}
    new_items = {user: truth - history_items[user] for user, truth in targets.items()}
    repeat_items = {user: truth & history_items[user] for user, truth in targets.items()}
    explore_lists = {"u1": ["x", "q"], "u2": ["w"], "u3": ["q", "z"]}
    repeat_lists = {"u1": ["b", "a"], "u2": ["c"], "u3": ["d"]}

    new_mean, new_values = subset_recall(explore_lists, new_items, top_k=2)
    assert new_values.to_dict() == {"u1": 0.5, "u3": 1.0}  # u2 has no new target item
    assert new_mean == pytest.approx(0.75)
    repeat_mean, repeat_values = subset_recall(repeat_lists, repeat_items, top_k=1)
    assert repeat_values.to_dict() == {"u1": 0.0, "u2": 1.0}  # u3 has no repeat target item
    assert repeat_mean == pytest.approx(0.5)


def test_paired_bootstrap_is_reproducible_and_contains_zero_for_identical_inputs():
    first = pd.Series([0.1, 0.4, 0.0, 0.9, 0.3], index=list("abcde"))
    second = pd.Series([0.2, 0.1, 0.0, 0.5, 0.3], index=list("abcde"))
    assert paired_bootstrap(first, second, 200, seed=7) == paired_bootstrap(first, second, 200, seed=7)
    same = paired_bootstrap(first, first, 200, seed=7)
    assert same["ci_low"] <= 0.0 <= same["ci_high"]
    result = paired_bootstrap(first, second.drop("e"), 200, seed=7)
    assert result["users"] == 4
    assert result["mean_difference"] == pytest.approx((-0.1 + 0.3 + 0.0 + 0.4) / 4)
