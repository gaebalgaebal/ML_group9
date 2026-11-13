# *------------------- 2026 Fall Semester Machine Learning Term-project: Team 9 --------------------*
# Code First written at 2026-11-07
# Code Final Edited at 2026-11-13
# Editor: Yejun Kim

'''
Purpose : Tests for scripts/posthoc_explore_slot.py: argument guard, per-basket novelty,
          aggregation, rank hits, slot metrics, setup check, and an end-to-end run
          (tune + analysis) on synthetic sample data.
Inputs  : Small hand-built frames, hand-built tuning documents and synthetic parquet files
          written under pytest's tmp_path.
Outputs : Pytest pass / fail results; files are written only under tmp_path.
Usage   :
    python -m pytest tests/test_posthoc_explore_slot.py
'''

# ---------------------------------------------------------------------------
# imports
# ---------------------------------------------------------------------------
import importlib.util
import json
import math

import numpy as np
import pandas as pd
import pytest

import nbr.pipeline as pipeline
from nbr.config import REPO_ROOT, load_config


# ---------------------------------------------------------------------------
# fixtures and helpers
# ---------------------------------------------------------------------------
def _load_script():
    '''Import scripts/posthoc_explore_slot.py from its file path and return the module.'''
    # Build the path of the script under the repository root.
    path = REPO_ROOT / "scripts" / "posthoc_explore_slot.py"
    # Create a module from the file and execute it.
    spec = importlib.util.spec_from_file_location("posthoc_explore_slot", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# Load the script once for all tests.
pes = _load_script()
# Column names of the synthetic basket frames.
COLUMNS = ["dataset", "user_id", "basket_id", "basket_seq", "item_id", "coarse_category", "fine_category", "quantity"]


def _frame(rows, dataset="instacart"):
    """rows = (user, seq, item); basket_id = f"{user}-{seq}"."""
    # Build one row per (user, seq, item) with constant categories and quantity 1.
    frame = pd.DataFrame([(dataset, u, f"{u}-{s}", s, i, "c", "f", 1) for u, s, i in rows], columns=COLUMNS)
    # Cast the id columns to string and basket_seq to int32.
    return frame.astype({"user_id": "string", "basket_id": "string", "basket_seq": "int32", "item_id": "string"})


def _lists(rows, source="repeat"):
    """rows = (user, rank, item)."""
    # Build one row per (user, rank, item) and attach the list name.
    frame = pd.DataFrame(rows, columns=["user_id", "rank", "item_id"])
    frame["source"] = source
    # Cast the id columns to string and rank to int64.
    return frame.astype({"user_id": "string", "item_id": "string", "rank": "int64"})


# ---------------------------------------------------------------------------
# arguments and helpers
# ---------------------------------------------------------------------------
def test_parse_args_defaults():
    # Parse an empty argument list.
    args = pes.parse_args([])
    # Expect the default value of every checked argument.
    assert args.profile == "medium" and args.datasets == ["instacart", "complete_journey"]
    assert args.out_dir.endswith("posthoc") and args.reference_tuning_dir is None and args.model_root is None
    assert (args.bootstrap_n, args.seed) == (1000, 42)


@pytest.mark.parametrize("argv", [["--profile", "holdout"], ["--profile", "holdout2"], ["--datasets", "holdout"], ["--datasets", ","]])
def test_parse_args_rejects_non_development_inputs(argv):
    # Expect each holdout profile, holdout dataset or empty dataset list to exit.
    with pytest.raises(SystemExit):
        pes.parse_args(argv)


def test_guard_path_refuses_confirmatory_folders(tmp_path):
    # Expect a path under reports/metrics to be returned unchanged.
    assert pes.guard_path(tmp_path / "reports" / "metrics") == tmp_path / "reports" / "metrics"
    # Expect paths inside the final and holdout folders to be refused.
    for bad in ("final", "final2", "holdout", "holdout2"):
        with pytest.raises(PermissionError):
            pes.guard_path(tmp_path / "reports" / bad / "x.json")


def test_bin_labels_match_pipeline_t5_bins():
    # Expect the same bin labels as the pipeline for every dataset.
    for dataset, edges in pipeline.HISTORY_BINS.items():
        assert pes.bin_labels(edges) == pipeline._bin_labels(edges), dataset
    # Expect the Instacart labels and the basket-position bins in the listed order.
    assert [name for name, _, _ in pes.bin_labels(pipeline.HISTORY_BINS["instacart"])] == ["(0,4]", "(4,6]", "(6,10]", "(10,20]", "(20,inf)"]
    assert [name for name, _, _ in pes.POSITION_BINS] == ["2-4", "5-10", "11-20", "21-50", ">50"]
    assert pes.POSITION_BINS[-1][2] == math.inf and pes.POSITION_BINS[0][1] == 2


def test_repeat_modules_merge_gamma_star():
    # Expect the logistic-regression repeat ranker (R2) first and an added gamma 0.8 module.
    names = [m["name"] for m in pes.repeat_modules(0.8)]
    assert names == ["R2", "R1_g0.8", "R1_g0.9", "R1_g0.97", "R1_g1"]
    # Expect gamma 1.0 to merge into the decayed purchase frequency (R1) module of gamma 1.
    modules = pes.repeat_modules(1.0)
    assert [m["name"] for m in modules] == ["R2", "R1_g0.9", "R1_g0.97", "R1_g1"]
    assert modules[-1]["is_gamma_star"] and modules[-1]["label"] == "R1 gamma 1 (gamma*)" and not modules[1]["is_gamma_star"]
    assert modules[0] == {"name": "R2", "label": "R2 (LR repeat ranker)", "kind": "R2", "gamma": None, "is_gamma_star": False}
    # Expect model_config to keep repo_root unless a model root is given.
    assert pes.model_config({"repo_root": "a", "profile": "sample"}, None)["repo_root"] == "a"
    assert str(pes.model_config({"repo_root": "a"}, "b")["repo_root"]) == "b"


def test_compare_and_compare_by_bin():
    # Build two per-user series whose difference is 1 for two of the four users.
    index = pd.Index(["u1", "u2", "u3", "u4"], name="user_id")
    a = pd.Series([0.0, 0.0, 0.0, 0.0], index=index)
    b = pd.Series([1.0, 1.0, 0.0, 0.0], index=index)
    # Expect a mean difference of 0.5 that lies inside its bootstrap interval.
    result = pes.compare(a, b, 50, 0)
    assert result["n_users"] == 4 and result["diff"] == pytest.approx(0.5) and result["ci_0.95"][0] <= 0.5 <= result["ci_0.95"][1]
    # Expect one row per Instacart history bin with the per-bin user counts and differences.
    lengths = pd.Series({"u1": 2, "u2": 3, "u3": 8, "u4": 30, "u9": 5})
    rows = pes.compare_by_bin(a, b, lengths, "instacart", 50, 0)
    assert [row["bin"] for row in rows] == ["(0,4]", "(4,6]", "(6,10]", "(10,20]", "(20,inf)"]
    assert [row["n_users"] for row in rows] == [2, 0, 1, 0, 1]
    assert rows[0]["diff"] == pytest.approx(1.0) and rows[2]["diff"] == 0.0 and "diff" not in rows[1]


# ---------------------------------------------------------------------------
# A. per-basket novelty
# ---------------------------------------------------------------------------
def test_basket_novelty_hand_checked():
    # Build three history baskets for user A, one for user B and one target basket each.
    history = _frame([("A", 1, "x"), ("A", 1, "y"), ("A", 2, "x"), ("A", 2, "z"), ("A", 3, "z"), ("A", 3, "w"), ("A", 3, "v"), ("A", 3, "v"),
                      ("B", 1, "a")])
    target = _frame([("A", 4, "x"), ("A", 4, "q"), ("B", 2, "a")])
    # Expect one row per basket after the first with hand-computed sizes and new-item counts.
    out = pes.basket_novelty(history, target)
    assert list(out.columns) == ["user_id", "basket_id", "position", "is_target", "size", "n_new", "new_share", "any_new"]
    assert out["basket_id"].tolist() == ["A-2", "A-3", "A-4", "B-2"]            # first baskets A-1, B-1 dropped
    assert out["position"].tolist() == [2, 3, 4, 2]
    assert out["is_target"].tolist() == [False, False, True, True]
    assert out["size"].tolist() == [2, 3, 2, 1]                                  # v counted once in A-3
    assert out["n_new"].tolist() == [1, 2, 1, 0]                                 # z; w, v; q; none
    assert out["new_share"].tolist() == pytest.approx([0.5, 2 / 3, 0.5, 0.0])
    assert out["any_new"].tolist() == [1.0, 1.0, 1.0, 0.0]


def test_basket_novelty_prior_includes_target_order_and_errors():
    # Build two users whose target items were bought before only by the other user.
    history = _frame([("A", 1, "x"), ("B", 1, "q"), ("B", 2, "q")])
    target = _frame([("A", 2, "q"), ("B", 3, "x")])
    # Expect an item to count as new only when the same user has not bought it before.
    out = pes.basket_novelty(history, target).set_index("basket_id")
    assert out.loc["A-2", "n_new"] == 1 and out.loc["B-3", "n_new"] == 1 and out.loc["B-2", "n_new"] == 0
    # Expect ValueError for each of the two malformed inputs.
    with pytest.raises(ValueError):                                              # target not after the history
        pes.basket_novelty(_frame([("A", 1, "x"), ("A", 3, "y")]), _frame([("A", 2, "z")]))
    with pytest.raises(ValueError):                                              # one basket_id with two basket_seq values
        broken = _frame([("A", 1, "x"), ("A", 2, "y")])
        broken.loc[1, "basket_id"] = "A-1"
        pes.basket_novelty(broken, _frame([("A", 3, "z")]))


def _baskets():
    '''Return a hand-built per-basket frame with three baskets of user A and one of user B.'''
    # user A: three baskets, user B: one basket -> basket-weighted and user-weighted means differ
    return pd.DataFrame({
        "user_id": pd.array(["A", "A", "A", "B"], dtype="string"), "basket_id": ["A-2", "A-3", "A-4", "B-2"],
        "position": [2, 3, 12, 60], "is_target": [False, False, True, True], "size": [2, 4, 2, 4],
        "n_new": [1, 2, 0, 3], "new_share": [0.5, 0.5, 0.0, 0.75], "any_new": [1.0, 1.0, 0.0, 1.0]})


def test_summarise_weightings_and_empty():
    # Summarise the four hand-built baskets.
    summary = pes.summarise(_baskets())
    # Expect the hand-computed basket-weighted and user-weighted means.
    assert summary["n_baskets"] == 4 and summary["n_users"] == 2
    assert summary["basket_weighted"]["mean_new_share"] == pytest.approx((0.5 + 0.5 + 0.0 + 0.75) / 4)
    assert summary["user_weighted"]["mean_new_share"] == pytest.approx(((0.5 + 0.5 + 0.0) / 3 + 0.75) / 2)
    assert summary["basket_weighted"]["p_any_new"] == pytest.approx(0.75) and summary["user_weighted"]["p_any_new"] == pytest.approx((2 / 3 + 1) / 2)
    assert summary["basket_weighted"]["mean_n_new"] == pytest.approx(1.5) and summary["user_weighted"]["mean_n_new"] == pytest.approx((1 + 3) / 2)
    assert summary["basket_weighted"]["mean_size"] == pytest.approx(3.0)
    # Expect an empty summary for an empty frame and equal shares of n_new 0, 1, 2 and 3+.
    empty = pes.summarise(_baskets().iloc[0:0])
    assert empty == {"n_baskets": 0, "n_users": 0, "basket_weighted": None, "user_weighted": None}
    assert pes.n_new_distribution(_baskets()) == pytest.approx({"0": 0.25, "1": 0.25, "2": 0.25, "3+": 0.25})
    assert pes.n_new_distribution(_baskets().iloc[0:0]) is None


def test_by_position_and_by_history_length():
    # Group the hand-built baskets by basket position.
    baskets = _baskets()
    rows = pes.by_position(baskets)
    # Expect five position bins that hold 2, 0, 1, 0 and 1 baskets.
    assert [row["bin"] for row in rows] == ["2-4", "5-10", "11-20", "21-50", ">50"]
    assert [row["n_baskets"] for row in rows] == [2, 0, 1, 0, 1]
    assert rows[0]["user_weighted"]["mean_new_share"] == pytest.approx(0.5) and rows[1]["basket_weighted"] is None
    # Expect the history-length bins of each dataset with the hand-computed basket counts.
    lengths = pd.Series({"A": 3, "B": 59})
    rows = pes.by_history_length(baskets, lengths, "instacart")
    assert [row["bin"] for row in rows] == ["(0,4]", "(4,6]", "(6,10]", "(10,20]", "(20,inf)"]
    assert [row["n_baskets"] for row in rows] == [3, 0, 0, 0, 1] and rows[4]["basket_weighted"]["mean_n_new"] == 3.0
    rows = pes.by_history_length(baskets, lengths, "complete_journey")
    assert [row["bin"] for row in rows] == ["(0,20]", "(20,50]", "(50,100]", "(100,inf)"] and [row["n_baskets"] for row in rows] == [3, 0, 1, 0]
    # Expect ValueError when a user has no history length.
    with pytest.raises(ValueError):
        pes.by_history_length(baskets, pd.Series({"A": 3}), "instacart")


# ---------------------------------------------------------------------------
# B. rank hits
# ---------------------------------------------------------------------------
def test_rank_hits_hand_checked():
    # Build ranked lists for users A, B, C and a target frame with one duplicated row.
    lists = _lists([("A", 1, "x"), ("A", 2, "y"), ("A", 3, "z"), ("B", 1, "p"), ("C", 1, "m"), ("C", 2, "n")])
    target = _frame([("A", 9, "y"), ("A", 9, "z"), ("B", 9, "q"), ("C", 9, "n"), ("C", 9, "n")])
    # Expect per-rank hit and presence indicators for every target user.
    hit, present = pes.rank_hits(lists, target, 3)
    assert list(hit.index) == ["A", "B", "C"] and list(hit.columns) == [1, 2, 3] and hit.columns.name is None
    assert hit.loc["A"].tolist() == [0.0, 1.0, 1.0] and hit.loc["B"].tolist() == [0.0, 0.0, 0.0] and hit.loc["C"].tolist() == [0.0, 1.0, 0.0]
    assert present.loc["A"].tolist() == [1.0, 1.0, 1.0] and present.loc["B"].tolist() == [1.0, 0.0, 0.0] and present.loc["C"].tolist() == [1.0, 1.0, 0.0]
    # Expect the per-rank hit probability, presence share and conditional hit probability.
    table = pes.rank_table(hit, present)
    assert table[1] == {"rank": 2, "p_hit": pytest.approx(2 / 3), "share_present": pytest.approx(2 / 3), "p_hit_given_present": pytest.approx(1.0)}
    assert table[2]["p_hit"] == pytest.approx(1 / 3) and table[2]["p_hit_given_present"] == pytest.approx(1.0)
    # a user without any list row gets zeros
    hit2, present2 = pes.rank_hits(lists[lists["user_id"] != "B"], target, 2)
    assert hit2.loc["B"].tolist() == [0.0, 0.0] and present2.loc["B"].tolist() == [0.0, 0.0]
    # Expect no conditional hit probability when no user has an item at the rank.
    empty_table = pes.rank_table(*pes.rank_hits(lists.iloc[0:0], target, 1))
    assert empty_table == [{"rank": 1, "p_hit": 0.0, "share_present": 0.0, "p_hit_given_present": None}]


# ---------------------------------------------------------------------------
# C. list metrics and the reproduction check
# ---------------------------------------------------------------------------
def test_list_metrics_new_hit_share_over_all_users():
    # Build a history, a target and final lists for users A, B and C.
    history = _frame([("A", 1, "x"), ("A", 2, "y"), ("B", 1, "p"), ("C", 1, "m")])
    target = _frame([("A", 3, "x"), ("A", 3, "n1"), ("B", 3, "p"), ("C", 3, "n2")])     # A: repeat + new, B: repeat only, C: new only
    # Split the target into its repeat part and its new-item part.
    data = {"target": target, **dict(zip(("repeat_target", "new_target"), pipeline.split_target(history, target)))}
    final = _lists([("A", 1, "x"), ("A", 2, "n1"), ("B", 1, "p"), ("B", 2, "zz"), ("C", 1, "m"), ("C", 2, "n9")])
    users = pd.Index(["A", "B", "C"])
    # Expect each metric to be indexed by the users it is defined for.
    metrics = pes.list_metrics(final, data, users)
    assert set(metrics) == set(pes.LIST_METRICS)
    assert metrics["ndcg"].index.tolist() == ["A", "B", "C"] and metrics["recall"].tolist() == pytest.approx([1.0, 1.0, 0.0])
    assert metrics["new_recall"].index.tolist() == ["A", "C"] and metrics["new_recall"].tolist() == pytest.approx([1.0, 0.0])
    assert metrics["new_hit_share"].index.tolist() == ["A", "B", "C"] and metrics["new_hit_share"].tolist() == [1.0, 0.0, 0.0]
    assert metrics["repeat_recall"].index.tolist() == ["A", "B"] and metrics["repeat_recall"].tolist() == pytest.approx([1.0, 1.0])


def test_reproduction_check_uses_sanity_numbers():
    # Build a slot table with one row per repeat module at r = 10.
    entry = {"eval": {"slots": {"table": [
        {"module": "R2", "r": 10, "ndcg": 0.42744, "repeat_list_recall": 0.5},
        {"module": "R1_g0.97", "r": 10, "ndcg": 0.3, "repeat_list_recall": 0.27171},
        {"module": "R1_g1", "r": 10, "ndcg": 0.3, "repeat_list_recall": 0.2497}]}}}
    # Expect no checks for the sample profile.
    assert pes.reproduction_check("sample", "instacart", entry) == {}
    # Expect the medium Instacart check to match the entry.
    checks = pes.reproduction_check("medium", "instacart", entry)
    assert set(checks) == {"R2_E2_r10_ndcg", "all_matched"} and checks["all_matched"] and checks["R2_E2_r10_ndcg"]["observed"] == 0.42744
    # Expect the Complete Journey NDCG check to fail while both recall checks match.
    checks = pes.reproduction_check("medium", "complete_journey", entry)
    assert checks["R2_E2_r10_ndcg"]["matched"] is False and checks["R1_g0.97_repeat_list_recall"]["matched"] and checks["R1_g1_repeat_list_recall"]["matched"]
    assert not checks["all_matched"]


# ---------------------------------------------------------------------------
# setup check
# ---------------------------------------------------------------------------
def _tuning_document(k_rows, gamma=0.8, commit="ref"):
    '''Return a tuning document with a usercf grid over ``k_rows`` and both weightings.

    ``gamma`` is the selected gamma and ``commit`` the git commit stored in the meta block.
    '''
    # Build one grid row per weighting and k.
    grid = [{"k": k, "weighting": w, "new_recall@10": 0.01 * k + (0.001 if w == "log1p" else 0.0), "seconds": 1.0}
            for w in ("binary", "log1p") for k in k_rows]
    # Assemble the meta block, the selections, the stage grids and the integration block.
    return {
        "meta": {"git_commit": commit, "utc_time": "2026-11-06T00:00:00+00:00"},
        "selections": {"gamma": gamma, "gamma_c": 0.9, "alpha": 100.0, "r": 10},
        "stage1_repeat_recall": [{"gamma": 0.5, "recall": 0.1}, {"gamma": 1.0, "recall": 0.2}],
        "stage2_explore_new_recall": [{"gamma_c": 0.9, "alpha": 10.0, "recall": 0.05}],
        "stage3_final_ndcg": [{"r": 9, "ndcg": 0.3}, {"r": 10, "ndcg": 0.31}],
        "lr_repeat": {"C": 0.001, "validation": [{"C": 0.001, "val_recall": 0.58}, {"C": 0.01, "val_recall": 0.57}],
                      "sha256": "abc", "artifact": "missing.joblib"},
        "integration": {"selections": {"C": 0.001, "k": 100, "weighting": "binary", "r": {"final_lr_r": 10, "final_lr_ucf": 10, "final_star_ucf": 10}},
                        "usercf": {"grid": grid}},
    }


def test_compare_tuning_equal_apart_from_k10_rows():
    # Build a run document with k = 10 grid rows and a reference document without them.
    run, reference = _tuning_document([10, 20, 50, 100], commit="run"), _tuning_document([20, 50, 100])
    # Expect every selection, grid and hash to compare equal apart from the k = 10 rows.
    result = pes.compare_tuning(run, reference)
    assert result["all_selections_equal"] and all(row["equal"] for row in result["selections"])
    assert [row["quantity"] for row in result["selections"]] == list(pes.SELECTION_PATHS)
    assert result["usercf_grid_excluding_k10_equal"] and len(result["usercf_k10_rows_run"]) == 2 and result["usercf_k10_rows_reference"] == []
    assert all(result["stage_grids_equal"].values()) and result["lr_validation_equal"] and result["lr_artifact_sha256"]["equal"]
    assert result["run_meta"]["git_commit"] == "run" and result["reference_meta"]["git_commit"] == "ref"


def test_compare_tuning_detects_differences():
    # Build a run document that differs in gamma, k, one NDCG value and one grid row.
    reference = _tuning_document([20, 50])
    run = _tuning_document([10, 20, 50], gamma=0.9)
    run["integration"]["selections"]["k"] = 10
    run["stage3_final_ndcg"][0]["ndcg"] = 0.29
    run["integration"]["usercf"]["grid"][1]["new_recall@10"] += 0.01     # the k = 20 binary row
    # Expect the two changed selections and the two changed grids to be reported.
    result = pes.compare_tuning(run, reference)
    unequal = {row["quantity"] for row in result["selections"] if not row["equal"]}
    assert unequal == {"gamma", "usercf_k"} and not result["all_selections_equal"]
    assert not result["usercf_grid_excluding_k10_equal"] and not result["stage_grids_equal"]["stage3_final_ndcg"]
    assert result["stage_grids_equal"]["stage1_repeat_recall"]
    # Expect _equal to use a numeric tolerance and to tell bool and None from numbers.
    assert pes._equal(0.1, 0.1 + 1e-12) and not pes._equal(0.1, 0.1 + 1e-6) and pes._equal("binary", "binary") and not pes._equal(True, 1)
    assert pes._equal(10, 10.0) and not pes._equal(None, 0)


def test_setup_check_reads_both_folders_and_stops_on_a_difference(tmp_path):
    # Write a reference and a run tuning document that differ only in the k = 10 rows.
    run_dir, ref_dir = tmp_path / "run" / "reports" / "metrics", tmp_path / "ref" / "metrics"
    run_dir.mkdir(parents=True), ref_dir.mkdir(parents=True)
    models = {"profile": "sample", "repo_root": tmp_path / "run"}
    (ref_dir / "sample_instacart_tuning.json").write_text(json.dumps(_tuning_document([20, 50])), encoding="utf-8")
    (run_dir / "sample_instacart_tuning.json").write_text(json.dumps(_tuning_document([10, 20, 50], commit="run")), encoding="utf-8")
    # Expect the check to pass and to record the hash of the reference file.
    result = pes.setup_check(models, ["instacart"], ref_dir)
    assert result["passed"] and result["datasets"]["instacart"]["lr_artifact"]["matches"] is False
    assert result["datasets"]["instacart"]["files"]["reference_sha256"] == pes.sha256_of(ref_dir / "sample_instacart_tuning.json")
    # Rewrite the run document with another gamma and expect the check to stop the run.
    (run_dir / "sample_instacart_tuning.json").write_text(json.dumps(_tuning_document([10, 20, 50], gamma=0.5)), encoding="utf-8")
    with pytest.raises(SystemExit, match="gamma"):
        pes.setup_check(models, ["instacart"], ref_dir)
    # Expect a missing run document and a reference folder named final2 to raise.
    with pytest.raises(FileNotFoundError):
        pes.setup_check(models, ["complete_journey"], ref_dir)
    with pytest.raises(PermissionError):
        pes.setup_check(models, ["instacart"], tmp_path / "reports" / "final2")


# ---------------------------------------------------------------------------
# end-to-end run on synthetic data (sample profile, NBR_DATA_DIR redirected, tune run into tmp)
# ---------------------------------------------------------------------------
def _synthetic_dataset(folder, dataset, seed, n_users=40, n_items=30):
    """Users with 3-9 history baskets plus one target basket; items repeat from a per-user favourite set; items catalogue with every level column."""
    # Seed the random generator.
    rng = np.random.default_rng(seed)
    rows = []
    for user in range(n_users):
        # Draw the number of history baskets and the user's five favourite items.
        n_baskets = int(rng.integers(3, 10))
        favourites = rng.choice(n_items, size=5, replace=False)
        for seq in range(1, n_baskets + 2):
            # Fill the basket with favourite items plus random items.
            items = set(rng.choice(favourites, size=3).tolist()) | set(rng.choice(n_items, size=2).tolist())
            rows.extend((dataset, str(user), f"{user}-{seq}", seq, str(item), f"c{item % 3}", f"f{item % 6}", 1) for item in sorted(items))
    # Build the basket frame and cast its columns.
    frame = pd.DataFrame(rows, columns=COLUMNS).astype(
        {"user_id": "string", "basket_id": "string", "basket_seq": "int32", "item_id": "string", "quantity": "int16"})
    # Write each user's last basket as eval target and the baskets before it as eval history.
    last = frame.groupby("user_id")["basket_seq"].transform("max")
    folder.mkdir(parents=True, exist_ok=True)
    frame[frame["basket_seq"] == last].to_parquet(folder / "eval_target.parquet", index=False)
    frame[frame["basket_seq"] < last].to_parquet(folder / "eval_history.parquet", index=False)
    # Write the second-to-last basket as fit target and the baskets before it as fit history.
    frame[frame["basket_seq"] == last - 1].to_parquet(folder / "fit_target.parquet", index=False)
    frame[frame["basket_seq"] < last - 1].to_parquet(folder / "fit_history.parquet", index=False)
    # Write the item catalogue with the category columns of every item.
    item_ids = [str(item) for item in range(n_items)]
    catalogue = pd.DataFrame({"item_id": item_ids, "coarse_category": [f"c{int(i) % 3}" for i in item_ids],
                              "fine_category": [f"f{int(i) % 6}" for i in item_ids], "ingredient_type": [f"t{int(i) % 9}" for i in item_ids],
                              "DEPARTMENT": [f"c{int(i) % 3}" for i in item_ids], "commodity": [f"f{int(i) % 6}" for i in item_ids]})
    catalogue.to_parquet(folder / "items.parquet", index=False)


def test_end_to_end_on_synthetic_sample(tmp_path, monkeypatch):
    # Write synthetic data for both datasets and point NBR_DATA_DIR at it.
    data_dir = tmp_path / "processed"
    for seed, dataset in enumerate(pes.DATASETS):
        _synthetic_dataset(data_dir / "sample" / dataset, dataset, seed)
    monkeypatch.setenv("NBR_DATA_DIR", str(data_dir))
    # Run the tuning step of the pipeline for both datasets under tmp_path.
    config = {**load_config("sample"), "repo_root": tmp_path}          # tuning documents and LR artifacts land under tmp_path
    for dataset in pes.DATASETS:
        pipeline.run_tune(config, dataset)
    # Run the script on the sample profile with 40 bootstrap resamples.
    out = tmp_path / "out"
    json_path = pes.main(["--profile", "sample", "--out-dir", str(out), "--model-root", str(tmp_path), "--bootstrap-n", "40", "--seed", "1"])
    # Expect the label, profile, parts, bootstrap settings and input hashes in the document.
    document = json.loads(json_path.read_text(encoding="utf-8"))
    assert document["label"] == pes.LABEL and document["profile"] == "sample" and document["setup_check"] is None
    assert document["parts"] == ["setup check", "A", "B", "C", "D"] and document["bootstrap"] == {"n": 40, "seed": 1, "level": 0.95}
    assert set(document["datasets"]) == set(pes.DATASETS) and document["model_root"] == str(tmp_path)
    assert set(document["inputs_sha256"]["instacart"]) == set(pes.INPUT_FILES)
    assert document["meta"]["git_commit"] is None or len(document["meta"]["git_commit"]) == 40
    for dataset, entry in document["datasets"].items():
        # Expect the selections, history bins and modules to agree with the tuning document.
        tuning = json.loads((tmp_path / "reports" / "metrics" / f"sample_{dataset}_tuning.json").read_text(encoding="utf-8"))
        assert entry["selections"]["gamma"] == tuning["selections"]["gamma"] and entry["selections"]["usercf_k"] == tuning["integration"]["selections"]["k"]
        assert entry["history_bins"] == [name for name, _, _ in pes.bin_labels(pipeline.HISTORY_BINS[dataset])]
        names = [module["name"] for module in entry["modules"]]
        assert names[0] == "R2" and len(names) in (4, 5) and sum(module["is_gamma_star"] for module in entry["modules"]) == 1
        # A on both splits
        for split in ("eval", "fit"):
            block = entry[split]["novelty"]
            checks, sets = block["checks"], block["basket_sets"]
            assert checks["n_users"] == 40 and checks["matches_split_target"] and checks["target_size_min"] >= 1
            assert checks["n_baskets_analysed"] == checks["n_history_baskets"] + checks["n_target_baskets"] - checks["n_first_baskets_dropped"]
            assert checks["n_first_baskets_dropped"] == 40 and checks["target_seq_minus_last_history_seq"] == {"1": 40}
            assert set(sets) == set(pes.BASKET_SETS) and sets["target"]["n_baskets"] == 40
            assert sets["target"]["basket_weighted"] == sets["target"]["user_weighted"]
            assert sets["target"]["basket_weighted"]["p_any_new"] == pytest.approx(checks["target_any_new_users"] / 40)
            for block_set in sets.values():
                assert [row["bin"] for row in block_set["by_position"]] == document["position_bins"]
                assert sum(row["n_baskets"] for row in block_set["by_position"]) == block_set["n_baskets"]
                assert sum(row["n_baskets"] for row in block_set["by_history_length"]) == block_set["n_baskets"]
                assert sum(block_set["n_new_distribution"].values()) == pytest.approx(1.0)
        # B
        hits = entry["eval"]["rank_hits"]
        assert hits["n_users"] == 40 and [row["rank"] for row in hits["repeat_R2"]] == list(range(1, 13))
        assert [row["rank"] for row in hits["explore_E2"]] == [1, 2, 3]
        for row in hits["repeat_R2"] + hits["explore_E2"]:
            assert 0.0 <= row["p_hit"] <= row["share_present"] <= 1.0
        assert hits["explore_E2"][0]["share_present"] == 1.0                   # E2 pads with popularity: every user has a rank-1 item
        for item in hits["comparisons"].values():
            assert item["n_users"] == 40 and item["ci_0.95"][0] <= item["diff"] <= item["ci_0.95"][1]
            assert sum(row["n_users"] for row in item["by_history_length"]) == 40
        # C eval
        slots = entry["eval"]["slots"]
        table = {(row["module"], row["r"]): row for row in slots["table"]}
        assert set(table) == {(name, r) for name in names for r in (10, 9, 8)}
        for row in slots["table"]:
            assert 0.0 <= row["ndcg"] <= 1.0 and 0.0 <= row["new_hit_share"] <= 1.0 and row["ndcg_n_users"] == 40
        assert table[("R2", 10)]["repeat_list_recall"] == table[("R2", 9)]["repeat_list_recall"]
        for diff in slots["differences"]:
            assert set(diff["metrics"]) == set(pes.LIST_METRICS) and diff["r"] in (9, 8)
            for metric, item in diff["metrics"].items():
                a = table[(diff["module"], 10)][metric]
                assert item["diff"] == pytest.approx(table[(diff["module"], diff["r"])][metric] - a)
                assert item["ci_0.95"][0] <= item["diff"] <= item["ci_0.95"][1]
                assert len(item["by_history_length"]) == len(entry["history_bins"])
        # C fit curves: R2 curve equals the tuning document's final_lr_ucf curve, gamma* curve equals final_star_ucf
        curves = entry["fit"]["slot_curves"]
        assert curves["lr_consistency"]["C_equal"] and curves["lr_consistency"]["max_abs_coefficient_difference"] < 1e-6
        by_module = {curve["module"]: curve for curve in curves["curves"]}
        assert [item["r"] for item in by_module["R2"]["ndcg_by_r"]] == list(range(11)) and 0 <= by_module["R2"]["r_star"] <= 10
        assert by_module["R2"]["tuning_document_curve"]["name"] == "final_lr_ucf" and by_module["R2"]["tuning_document_curve"]["equal"]
        star = next(curve for curve in curves["curves"] if curve.get("tuning_document_curve", {}).get("name") == "final_star_ucf")
        assert star["tuning_document_curve"]["equal"] and star["r_star"] == tuning["integration"]["selections"]["r"]["final_star_ucf"]
        assert by_module["R2"]["r_star"] == tuning["integration"]["selections"]["r"]["final_lr_ucf"]
        for curve in curves["curves"]:
            best = max(item["ndcg"] for item in curve["ndcg_by_r"])
            assert curve["r_star_below_10"] == (curve["r_star"] < 10)
            assert next(item["ndcg"] for item in curve["ndcg_by_r"] if item["r"] == curve["r_star"]) == pytest.approx(best)
        assert entry["reproduction_check"] == {}                               # reference numbers exist for medium only
    # figures and markdown
    for figure in document["figures"].values():
        assert (out / figure["png"]).exists() and (out / figure["csv"]).exists() and figure["points"] > 0
    assert len(pd.read_csv(out / "fig_new_rate_by_position.csv")) == 2 * (len(pes.POSITION_BINS) + 1)
    assert len(pd.read_csv(out / "fig_hit_probability_by_rank.csv")) == 2 * (pes.REPEAT_RANKS + pes.EXPLORE_RANKS)
    slots_csv = pd.read_csv(out / "fig_slot_effects.csv")
    assert set(slots_csv["metric"]) == {"ndcg", "new_hit_share"} and set(slots_csv["r"]) == {8, 9}
    markdown = (out / "explore_slot_sample.md").read_text(encoding="utf-8")
    assert pes.LABEL in markdown and "## Setup check" in markdown and "not checked" in markdown
    for heading in ("## A. New-item purchase rates", "## B. Hit probability by rank", "## C. Forced exploration slot", "## C. Fit-split check",
                    "## Reproduction checks", "## Figures", "### Instacart, eval split (main)", "### Complete Journey, fit split (robustness check)"):
        assert heading in markdown
    assert "(20,inf)" in markdown and "(100,inf)" in markdown and ">50" in markdown
    for word in ("significant", "novel", "outperform"):
        assert word not in markdown.lower()
    assert "confirmatory" not in markdown.replace("not confirmatory", "").replace("no confirmatory claim is made", "")
