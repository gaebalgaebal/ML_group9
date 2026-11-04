import numpy as np
import pytest

from ml_group9.hier_pipeline import (
    auc_score,
    bootstrap_auc_difference,
    cell_ranks,
    item_stage,
    log_loss,
    set_ranking_metrics,
    stage_rankings,
    target_units,
    tune_state,
)
from ml_group9.hierarchical import Params, recommend_hier

from test_hierarchical import config, state


def test_set_ranking_metrics_match_hand_calculation():
    ranked = np.array([[3, 1, 2], [0, 4, 5], [7, 8, 9]])
    truth = [{1, 9}, {6}, set()]
    result = set_ranking_metrics(ranked, truth, [1, 2])
    assert result["at_1"] == {"recall": 0.0, "hit_rate": 0.0, "precision": 0.0, "users": 2}
    assert result["at_2"]["recall"] == pytest.approx((0.5 + 0.0) / 2)
    assert result["at_2"]["hit_rate"] == pytest.approx(0.5)
    assert result["at_2"]["precision"] == pytest.approx((0.5 + 0.0) / 2)


def test_auc_matches_pair_counting_with_ties():
    scores = np.array([0.9, 0.4, 0.4, 0.2, 0.7])
    labels = np.array([1, 1, 0, 0, 0])
    # Positive pairs: 0.9 beats all 3 negatives; 0.4 beats 0.2, ties 0.4, loses to 0.7.
    assert auc_score(scores, labels) == pytest.approx((3 + 1 + 0.5) / 6)
    assert auc_score(scores, np.zeros(5)) is None


def test_weighted_auc_equals_duplicated_rows():
    rng = np.random.default_rng(1)
    scores = rng.integers(0, 4, 30).astype(float)
    labels = rng.integers(0, 2, 30)
    weights = rng.integers(0, 3, 30)
    repeated = np.repeat(np.arange(30), weights)
    assert auc_score(scores, labels, weights) == pytest.approx(auc_score(scores[repeated], labels[repeated]))


def test_log_loss_and_bootstrap_are_reproducible():
    labels = np.array([1, 0, 1, 0])
    assert log_loss(np.array([0.8, 0.2, 0.6, 0.4]), labels) == pytest.approx(-np.mean(np.log([0.8, 0.8, 0.6, 0.6])))
    users = np.array([0, 0, 1, 2, 3, 3, 4, 5])
    y = np.array([1, 0, 1, 0, 1, 0, 1, 0])
    first = np.array([0.9, 0.1, 0.8, 0.3, 0.6, 0.4, 0.7, 0.2])
    second = np.array([0.5, 0.5, 0.4, 0.6, 0.5, 0.5, 0.3, 0.7])
    a = bootstrap_auc_difference(users, y, first, second, 200, 42)
    b = bootstrap_auc_difference(users, y, first, second, 200, 42)
    assert a == b
    assert a["mean_difference"] == pytest.approx(auc_score(first, y) - auc_score(second, y))
    assert a["ci_low"] <= a["mean_difference"] <= a["ci_high"]


def test_stage_one_ranking_hit_rate_by_hand():
    hier_state = state()
    # Targets: u1 -> m3 (milk), u2 -> m1 (milk), u3 -> b3 (bread).
    truth = target_units(hier_state, hier_state.item_stage1)
    labels = hier_state.stage1_labels.tolist()
    assert [{labels[unit] for unit in units} for units in truth] == [{"milk"}, {"milk"}, {"bread"}]
    ranked, _ = stage_rankings(hier_state, "hier_full", Params(1e-9, 1.0, 2.0, 0), 2, 2)
    # Personal frequency: u1 has milk 1.5 > bread 0.5; u2 bread 1.0 > milk 0.5; u3 only bread.
    assert [labels[row[0]] for row in ranked] == ["milk", "bread", "bread"]
    result = set_ranking_metrics(ranked, truth, [1])
    assert result["at_1"]["hit_rate"] == pytest.approx(2 / 3)


def test_cell_rank_and_top3_hit_by_hand():
    hier_state = state()
    mu = 1.0
    frame = cell_ranks(hier_state, Params(1.0, mu, 2.0, 0))
    users = hier_state.interactions.users
    by_user = {users[row]: entry for row, entry in zip(frame["user"], frame.itertuples())}
    # u1 -> m3 in cell (i_milk, x = 0) = {m3, m4}; neither bought by u1, both popularity 1: tie by index.
    assert by_user["u1"].cell_size == 2 and by_user["u1"].phi_rank == 0
    # u2 -> m1 in cell (i_milk, x = 1) = {m1 (pop 2), m2 (pop 1)}; u2 bought neither.
    assert by_user["u2"].phi_rank == 0 and by_user["u2"].popularity_rank == 0
    # u3 -> b3, alone in its fallback type.
    assert by_user["u3"].cell_size == 1 and bool(by_user["u3"].new)
    assert item_stage(hier_state, Params(1.0, mu, 2.0, 0))["all"]["phi_hit_at_3"] == 1.0


def test_tuning_follows_the_fixed_order_and_keeps_the_best_setting():
    settings = config()
    settings["hierarchical"].update(
        {"alphas": [1, 5], "mus": [0.1, 10], "knn_k": [0, 2], "lambda_type": [2, 10], "lambda_type_initial": 10}
    )
    hier_state = state()
    report = tune_state(settings, hier_state, "test")
    steps = [row["step"] for row in report["grid"]]
    assert steps == ["1_alpha_mu"] * 4 + ["2_k"] * 2 + ["3_lambda_type"] * 2
    first = [row for row in report["grid"] if row["step"] == "1_alpha_mu"]
    assert all(row["k"] == 0 and row["lambda_type"] == 10 for row in first)
    best = max(first, key=lambda row: row["ndcg_at_k"])
    second = [row for row in report["grid"] if row["step"] == "2_k"]
    assert all(row["alpha"] == best["alpha"] and row["mu"] == best["mu"] for row in second)
    final = report["selected"]
    third = [row for row in report["grid"] if row["step"] == "3_lambda_type"]
    assert max(row["ndcg_at_k"] for row in third) == pytest.approx(
        next(row["ndcg_at_k"] for row in third if row["lambda_type"] == final["lambda_type"])
    )


def test_noknn_variant_equals_full_with_k_zero():
    hier_state = state()
    settings = config()
    full_k0 = recommend_hier(hier_state, "hier_full", Params(2.0, 1.0, 2.0, 0), settings)
    noknn = recommend_hier(hier_state, "hier_noknn", Params(2.0, 1.0, 2.0, 2), settings)
    assert full_k0.lists == noknn.lists
