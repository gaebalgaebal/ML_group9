"""Synthetic checks for follow-up inference, reproduction and direct hit accounting."""

import numpy as np
import pandas as pd
import pytest

from ml_group9.hs_followup import (
    bootstrap_means,
    bootstrap_p,
    displacement_counts,
    holm_adjust,
    percentile_interval,
    reproduction_check,
    summarize_displacement,
)
from ml_group9.metrics import paired_bootstrap


def test_holm_hand_calculation_preserves_order_and_step_down():
    # Sorted raw p: .001, .01, .03, .04; scaled: .004, .03, .06, .04.
    actual = holm_adjust([0.03, 0.001, 0.04, 0.01])
    np.testing.assert_allclose(actual, [0.06, 0.004, 0.06, 0.03])
    assert (actual <= 0.05).tolist() == [False, True, False, True]


def test_holm_ties_and_clipping():
    np.testing.assert_allclose(holm_adjust([0.02, 0.02, 0.9]), [0.06, 0.06, 0.9])
    np.testing.assert_allclose(holm_adjust([0.8, 0.9]), [1, 1])


@pytest.mark.parametrize("values", [[-0.1], [1.01], [np.nan], [[0.1]]])
def test_holm_rejects_invalid_p(values):
    with pytest.raises(ValueError):
        holm_adjust(values)


def test_bootstrap_p_symmetric_and_inclusive_zero():
    assert bootstrap_p(np.array([-2, -1, 1, 2])) == 1
    # Zero belongs to both tails: 2 * min(2/4, 3/4) = 1.
    assert bootstrap_p(np.array([-1, 0, 1, 2])) == 1
    assert bootstrap_p(np.zeros(4)) == 1


def test_bootstrap_p_asymmetric_and_resolution_floor():
    assert bootstrap_p(np.array([-1, 1, 2, 3, 4])) == pytest.approx(0.4)
    assert bootstrap_p(np.array([-4, -3, -2, -1, 1])) == pytest.approx(0.4)
    assert bootstrap_p(np.array([1, 2, 3, 4])) == pytest.approx(0.25)
    assert bootstrap_p(np.ones(10_000)) == pytest.approx(0.0001)


def test_seed_and_legacy_bootstrap_reproduction():
    before = pd.Series([0.2, 0.5, 0.0, 0.4], index=["1", "2", "3", "4"])
    after = pd.Series([0.4, 0.0, 0.3, 0.4], index=before.index)
    differences = (after - before).to_numpy()
    first = bootstrap_means(differences, 10_000, 42)
    np.testing.assert_array_equal(first, bootstrap_means(differences, 10_000, 42))
    np.testing.assert_array_equal(first[:1_000], bootstrap_means(differences, 1_000, 42))
    assert not np.array_equal(first, bootstrap_means(differences, 10_000, 43))
    original = paired_bootstrap(after, before, 1_000, 42)
    actual = percentile_interval(first[:1_000], 0.95)
    assert actual["ci_low"] == pytest.approx(original["ci_low"], abs=1e-15)
    assert actual["ci_high"] == pytest.approx(original["ci_high"], abs=1e-15)


@pytest.mark.parametrize("differences,resamples", [([], 100), ([np.nan], 100), ([1], 0)])
def test_bootstrap_rejects_empty_nonfinite_or_no_draws(differences, resamples):
    with pytest.raises(ValueError):
        bootstrap_means(np.asarray(differences), resamples, 42)


def test_displacement_hand_calculation_and_top_k():
    before = ["a_lost", "o_lost", "keep", "irrelevant_old"]
    after = ["a_gain", "o_gain", "keep", "irrelevant_new", "outside_top"]
    truth = {"a_lost", "o_lost", "a_gain", "o_gain", "keep", "outside_top"}
    attributes = {"a_lost", "a_gain", "irrelevant_new", "outside_top"}
    assert displacement_counts(before, after, truth, attributes, top_k=4) == {
        "gained_attr": 1, "lost_attr": 1, "gained_other": 1, "lost_other": 1,
        "changed_slots": 3, "entered_attr": 2, "net": 0,
    }


def test_displacement_empty_truth_union_attributes_and_duplicate_sets():
    result = displacement_counts(["old", "old"], ["new", "new", "other"], set(), {"new"})
    assert result == {
        "gained_attr": 0, "lost_attr": 0, "gained_other": 0, "lost_other": 0,
        "changed_slots": 2, "entered_attr": 1, "net": 0,
    }
    union = {"a1"} | {"a2"}
    result = displacement_counts(["o1", "o2"], ["a1", "a2"], {"o1", "a1", "a2"}, union)
    assert result["gained_attr"] == 2
    assert result["lost_other"] == 1
    assert result["net"] == 1


def test_displacement_summary_filters_empty_truth_and_pools_slots():
    rows = []
    for user, before, after, truth, attrs in [
        ("u1", ["o1", "o2"], ["a1", "a2"], {"o1", "a1", "a2"}, {"a1", "a2"}),
        ("u2", ["o3"], ["noise"], {"o3"}, set()),
        ("empty", ["x"], ["a3"], set(), {"a3"}),
    ]:
        rows.append({
            "user_id": user, "new_items_count": len(truth),
            **displacement_counts(before, after, truth, attrs),
        })
    result = summarize_displacement(pd.DataFrame(rows))
    assert result["users"] == 2
    assert result["sums"] == {
        "gained_attr": 2, "lost_attr": 0, "gained_other": 0, "lost_other": 2, "changed_slots": 3,
    }
    assert result["means"]["gained_attr"] == 1
    assert result["net_mean"] == 0
    assert result["net_ci_95"]["ci_low"] == -1
    assert result["net_ci_95"]["ci_high"] == 1
    assert result["changed_slot_attribute_share"] == pytest.approx(2 / 3)
    assert result["users_gained_attr_and_lost_other"] == 1
    assert result["lost_other_among_users_gaining_attr"] == 1


def test_no_changed_slots_has_undefined_share():
    counts = displacement_counts(["a"], ["a"], {"a"}, {"a"})
    result = summarize_displacement(pd.DataFrame([{
        "user_id": "u", "new_items_count": 1, **counts,
    }]))
    assert result["changed_slot_attribute_share"] is None
    assert result["net_ci_95"]["ci_low"] == 0
    assert result["net_ci_95"]["ci_high"] == 0


def test_reproduction_accepts_tolerance_and_detects_mismatch():
    check = reproduction_check({"new": 0.1000000001, "attr": 0.2}, {"new": 0.1, "attr": 0.2})
    assert check["new"]["passed"]
    assert check["new"]["absolute_difference"] < 1e-9
    with pytest.raises(ValueError, match="Reproduction mismatch"):
        reproduction_check({"new": 0.100000002}, {"new": 0.1})


@pytest.mark.parametrize("actual,expected", [
    ({"new": np.nan}, {"new": 0.1}), ({"new": None}, {"new": 0.1}),
    ({"new": 0.1}, {"attr": 0.1}), ({}, {}),
])
def test_reproduction_rejects_missing_or_invalid_means(actual, expected):
    with pytest.raises(ValueError):
        reproduction_check(actual, expected)
