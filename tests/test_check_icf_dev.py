"""Tests for scripts/check_icf_dev.py (W19 development check of the PREREG v1.5 code): guards and the pure check helpers."""

import importlib.util

import pytest

from nbr.config import REPO_ROOT


def _load_script():
    path = REPO_ROOT / "scripts" / "check_icf_dev.py"
    spec = importlib.util.spec_from_file_location("check_icf_dev", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


cid = _load_script()


def test_parse_args_defaults_and_guards(tmp_path):
    args = cid.parse_args(["--reference-dir", "ref"])
    assert args.profile == "medium" and args.datasets == ["instacart", "complete_journey"] and args.reference_dir == "ref"
    assert args.skip_run is False and cid.parse_args(["--reference-dir", "ref", "--skip-run"]).skip_run is True
    with pytest.raises(SystemExit):
        cid.parse_args(["--reference-dir", "ref", "--profile", "holdout4"])
    with pytest.raises(SystemExit):
        cid.parse_args([])
    with pytest.raises(ValueError):
        cid.parse_datasets("holdout4")
    for bad in ("final4", "holdout4", "final"):
        with pytest.raises(PermissionError):
            cid.guard_path(tmp_path / "reports" / bad / "x.json")
    assert cid.guard_path(tmp_path / "reports" / "metrics") == tmp_path / "reports" / "metrics"
    assert cid.EXPECTED_C3_NDCG == {("medium", "instacart"): 0.4274, ("medium", "complete_journey"): 0.2046}
    assert cid.EXPECTED_APPROX["instacart"] == {"repeat_lr": 0.5757, "repeat_lri": 0.5759}
    assert cid.LABEL == "development check of the PREREG v1.5 code; not confirmatory"
    with pytest.raises(PermissionError):
        cid.run_pipeline("holdout3", ["instacart"])


def test_compare_selections_reports_every_quantity():
    run = {"selections": {"gamma": 0.8, "gamma_c": 0.9, "alpha": 100.0, "r": 10}, "lr_repeat": {"C": 0.001},
           "integration": {"selections": {"k": 100, "weighting": "binary", "r": {"final_lr_r": 10, "final_lr_ucf": 10, "final_star_ucf": 10}}},
           "meta": {"git_commit": "a", "utc_time": "t"}}
    same = cid.compare_selections(run, {**run, "selections": {**run["selections"], "alpha": 100}})
    assert same["all_equal"] and [row["quantity"] for row in same["selections"]] == list(cid.SELECTION_PATHS)
    other = cid.compare_selections(run, {**run, "integration": {"selections": {**run["integration"]["selections"], "k": 50}}})
    assert not other["all_equal"] and [row for row in other["selections"] if not row["equal"]][0]["quantity"] == "usercf_k"
    assert not cid.compare_selections(run, {})["all_equal"]


def test_c3_and_r2prime_checks():
    evaluation = {"list_metrics": {"final_lr_ucf": {"ndcg": 0.42738}}, "v15": {"primary": {"H14_ndcg": {"comparator_mean": 0.42738}},
                  "lr_repeat_icf": {"features": ["a", "b", "icf"]}}}
    reference = {"list_metrics": {"final_lr_ucf": {"ndcg": 0.42738}}}
    c3 = cid.check_c3(evaluation, reference, "medium", "instacart")
    assert c3["matched"] and c3["equals_reference"] and c3["v15_equals_list_metric"]
    assert not cid.check_c3(evaluation, {"list_metrics": {"final_lr_ucf": {"ndcg": 0.4274}}}, "medium", "instacart")["equals_reference"]
    assert not cid.check_c3(evaluation, None, "medium", "complete_journey")["matched"]
    assert cid.check_c3(evaluation, None, "sample", "instacart")["matched"]  # no expectation for the sample profile
    tuning = {"lr_repeat": {"n_val_users": 6, "C_grid": [0.1, 1.0], "coefficients": [{"feature": "a"}, {"feature": "b"}]},
              "v15": {"icf_neighbors": 5, "lr": {"n_val_users": 6, "C_grid": [0.1, 1.0], "same_validation_users_as_R2": True},
                      "lr_repeat_icf": {"coefficients": [{"feature": "a"}, {"feature": "b"}, {"feature": "icf"}]}}}
    assert cid.check_r2prime(tuning, evaluation)["ok"]
    broken = {**tuning, "v15": {**tuning["v15"], "lr": {**tuning["v15"]["lr"], "n_val_users": 5}}}
    assert not cid.check_r2prime(broken, evaluation)["ok"]
    assert not cid.check_r2prime(tuning, {**evaluation, "v15": {**evaluation["v15"], "lr_repeat_icf": {"features": ["a", "icf", "b"]}}})["ok"]


def test_check_artifact(tmp_path):
    file = tmp_path / "x.joblib"
    file.write_bytes(b"abc")
    digest = cid.sha256_of(file)
    assert cid.check_artifact(digest, file, digest)["matches"]
    assert not cid.check_artifact(digest, file, "other")["matches"]
    assert not cid.check_artifact(digest, tmp_path / "missing.joblib", digest)["matches"]


def test_policy_constants_of_the_check():
    assert cid.POLICY_KEYS == ("H11_ndcg", "H12_new_hit_share", "H13_new_hit_share")
    assert cid.EXPECTED_POLICY_APPROX == {"instacart": {"ndcg": -0.0013, "new_hit_share": 0.010}}
    assert cid.POLICY_LEVELS == (0.95, 0.9833) and cid.POLICY_PRIMARY_LEVEL == "0.9833"
    assert cid.LEVELS == (0.95, 0.975) and cid.PRIMARY_LEVEL == "0.975"
