"""Tests for scripts/check_policy_dev.py (W17 development check of the policy code): guards and the pure comparison helpers."""

import importlib.util

import pytest

from nbr.config import REPO_ROOT


def _load_script():
    path = REPO_ROOT / "scripts" / "check_policy_dev.py"
    spec = importlib.util.spec_from_file_location("check_policy_dev", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


cpd = _load_script()


def test_parse_args_defaults_and_guards(tmp_path):
    args = cpd.parse_args(["--reference-tuning-dir", "ref"])
    assert args.profile == "medium" and args.datasets == ["instacart", "complete_journey"] and args.reference_tuning_dir == "ref"
    with pytest.raises(SystemExit):
        cpd.parse_args(["--reference-tuning-dir", "ref", "--profile", "holdout3"])
    with pytest.raises(SystemExit):
        cpd.parse_args([])
    with pytest.raises(ValueError):
        cpd.parse_datasets("holdout3")
    for bad in ("final3", "holdout3", "final"):
        with pytest.raises(PermissionError):
            cpd.guard_path(tmp_path / "reports" / bad / "x.json")
    assert cpd.guard_path(tmp_path / "reports" / "metrics") == tmp_path / "reports" / "metrics"
    assert cpd.EXPECTED_C3_NDCG == {("medium", "instacart"): 0.4274, ("medium", "complete_journey"): 0.2046}


def test_short_bins_and_expected_from_w15():
    labels = ["(0,4]", "(4,6]", "(6,10]", "(10,20]", "(20,inf)"]
    assert cpd.short_bins(4, labels) == ["(0,4]"]
    assert cpd.short_bins(6, labels) == ["(0,4]", "(4,6]"]
    assert cpd.short_bins(10, labels) == ["(0,4]", "(4,6]", "(6,10]"]
    assert cpd.short_bins(3, labels) == [] and cpd.short_bins(1000, labels) == labels[:4]
    assert cpd.bin_edges(labels) == [4.0, 6.0, 10.0, 20.0]
    assert cpd.checkable(4, labels) and cpd.checkable(10, labels) and not cpd.checkable(5, labels) and not cpd.checkable(1000, labels)
    cj = ["(0,20]", "(20,50]", "(50,100]", "(100,inf)"]
    assert cpd.short_bins(4, cj) == [] and not cpd.checkable(4, cj) and not cpd.checkable(6, cj) and cpd.checkable(20, cj)
    bins = {"ndcg": {"(0,4]": {"n_users": 4000, "diff": -0.006}, "(4,6]": {"n_users": 3000, "diff": -0.008}, "(6,10]": {"n_users": 0}}}
    assert cpd.expected_from_w15(bins, "ndcg", ["(0,4]"], 20000) == pytest.approx(-0.006 * 4000 / 20000)
    assert cpd.expected_from_w15(bins, "ndcg", ["(0,4]", "(4,6]", "(6,10]"], 20000) == pytest.approx((-0.006 * 4000 - 0.008 * 3000) / 20000)
    assert cpd.expected_from_w15(bins, "ndcg", [], 20000) == 0.0


def test_compare_selections_reports_every_quantity():
    run = {"selections": {"gamma": 0.8, "gamma_c": 0.9, "alpha": 100.0, "r": 10}, "lr_repeat": {"C": 0.001},
           "integration": {"selections": {"k": 100, "weighting": "binary", "r": {"final_lr_r": 10, "final_lr_ucf": 10, "final_star_ucf": 10}}},
           "meta": {"git_commit": "a", "utc_time": "t"}}
    same = cpd.compare_selections(run, {**run, "selections": {**run["selections"], "alpha": 100}})
    assert same["all_equal"] and [row["quantity"] for row in same["selections"]] == list(cpd.SELECTION_PATHS)
    other = cpd.compare_selections(run, {**run, "integration": {"selections": {**run["integration"]["selections"], "k": 50}}})
    assert not other["all_equal"] and [row for row in other["selections"] if not row["equal"]][0]["quantity"] == "usercf_k"
    assert not cpd.compare_selections(run, {})["all_equal"]
