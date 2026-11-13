# *------------------- 2026 Fall Semester Machine Learning Term-project: Team 9 --------------------*
# Code First written at 2026-11-07
# Code Final Edited at 2026-11-13
# Editor: Yejun Kim

'''
Purpose : Tests for scripts/check_policy_dev.py (W17 development check of the policy code):
          guards and the pure comparison helpers.
Inputs  : The script module loaded from scripts/check_policy_dev.py by file path, hand-written
          bin labels and tuning dicts, and the pytest tmp_path directory.
Outputs : pytest pass/fail results; no file is written.
Usage   :
    python -m pytest tests/test_check_policy_dev.py
'''

# ---------------------------------------------------------------------------
# imports
# ---------------------------------------------------------------------------
import importlib.util

import pytest

from nbr.config import REPO_ROOT


# ---------------------------------------------------------------------------
# fixtures and helpers
# ---------------------------------------------------------------------------
def _load_script():
    '''Import scripts/check_policy_dev.py from its file path and return the module.'''
    path = REPO_ROOT / "scripts" / "check_policy_dev.py"
    spec = importlib.util.spec_from_file_location("check_policy_dev", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


cpd = _load_script()


# ---------------------------------------------------------------------------
# argument parsing and guards
# ---------------------------------------------------------------------------
def test_parse_args_defaults_and_guards(tmp_path):
    # Parse a command line that gives only the reference tuning directory.
    args = cpd.parse_args(["--reference-tuning-dir", "ref"])
    # The defaults are the medium profile and both datasets.
    assert args.profile == "medium" and args.datasets == ["instacart", "complete_journey"] and args.reference_tuning_dir == "ref"
    # The confirmatory profile of PREREG v1.4 (holdout3) and a missing directory are refused.
    with pytest.raises(SystemExit):
        cpd.parse_args(["--reference-tuning-dir", "ref", "--profile", "holdout3"])
    with pytest.raises(SystemExit):
        cpd.parse_args([])
    with pytest.raises(ValueError):
        cpd.parse_datasets("holdout3")
    # A path under a final or holdout report folder raises PermissionError, other paths pass.
    for bad in ("final3", "holdout3", "final"):
        with pytest.raises(PermissionError):
            cpd.guard_path(tmp_path / "reports" / bad / "x.json")
    assert cpd.guard_path(tmp_path / "reports" / "metrics") == tmp_path / "reports" / "metrics"
    assert cpd.EXPECTED_C3_NDCG == {("medium", "instacart"): 0.4274, ("medium", "complete_journey"): 0.2046}


# ---------------------------------------------------------------------------
# history-length bins
# ---------------------------------------------------------------------------
def test_short_bins_and_expected_from_w15():
    # Define five bin labels with the upper edges 4, 6, 10 and 20.
    labels = ["(0,4]", "(4,6]", "(6,10]", "(10,20]", "(20,inf)"]
    # short_bins returns the bins up to the threshold and checkable is true only on a bin edge.
    assert cpd.short_bins(4, labels) == ["(0,4]"]
    assert cpd.short_bins(6, labels) == ["(0,4]", "(4,6]"]
    assert cpd.short_bins(10, labels) == ["(0,4]", "(4,6]", "(6,10]"]
    assert cpd.short_bins(3, labels) == [] and cpd.short_bins(1000, labels) == labels[:4]
    assert cpd.bin_edges(labels) == [4.0, 6.0, 10.0, 20.0]
    assert cpd.checkable(4, labels) and cpd.checkable(10, labels) and not cpd.checkable(5, labels) and not cpd.checkable(1000, labels)
    # Define four bin labels with the upper edges 20, 50 and 100.
    cj = ["(0,20]", "(20,50]", "(50,100]", "(100,inf)"]
    # Thresholds 4 and 6 give no short bin and are not checkable, while 20 is checkable.
    assert cpd.short_bins(4, cj) == [] and not cpd.checkable(4, cj) and not cpd.checkable(6, cj) and cpd.checkable(20, cj)
    # Build per-bin user counts and differences of the ndcg metric.
    bins = {"ndcg": {"(0,4]": {"n_users": 4000, "diff": -0.006}, "(4,6]": {"n_users": 3000, "diff": -0.008}, "(6,10]": {"n_users": 0}}}
    # The expected value is the user-weighted sum of the bin differences divided by 20000.
    assert cpd.expected_from_w15(bins, "ndcg", ["(0,4]"], 20000) == pytest.approx(-0.006 * 4000 / 20000)
    assert cpd.expected_from_w15(bins, "ndcg", ["(0,4]", "(4,6]", "(6,10]"], 20000) == pytest.approx((-0.006 * 4000 - 0.008 * 3000) / 20000)
    assert cpd.expected_from_w15(bins, "ndcg", [], 20000) == 0.0


# ---------------------------------------------------------------------------
# selection comparison
# ---------------------------------------------------------------------------
def test_compare_selections_reports_every_quantity():
    # Build a tuning document that holds every selected quantity.
    run = {"selections": {"gamma": 0.8, "gamma_c": 0.9, "alpha": 100.0, "r": 10}, "lr_repeat": {"C": 0.001},
           "integration": {"selections": {"k": 100, "weighting": "binary", "r": {"final_lr_r": 10, "final_lr_ucf": 10, "final_star_ucf": 10}}},
           "meta": {"git_commit": "a", "utc_time": "t"}}
    # A copy whose alpha is the integer 100 compares equal and reports every selection path.
    same = cpd.compare_selections(run, {**run, "selections": {**run["selections"], "alpha": 100}})
    assert same["all_equal"] and [row["quantity"] for row in same["selections"]] == list(cpd.SELECTION_PATHS)
    # A copy with another neighbour count k is unequal in the usercf_k row.
    other = cpd.compare_selections(run, {**run, "integration": {"selections": {**run["integration"]["selections"], "k": 50}}})
    assert not other["all_equal"] and [row for row in other["selections"] if not row["equal"]][0]["quantity"] == "usercf_k"
    # An empty reference document is unequal.
    assert not cpd.compare_selections(run, {})["all_equal"]
