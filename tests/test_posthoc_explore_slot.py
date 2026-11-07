"""Tests for scripts/posthoc_explore_slot.py: argument guard, per-basket novelty, aggregation, setup check, an end-to-end run on synthetic data."""

import importlib.util
import json
import math

import numpy as np
import pandas as pd
import pytest

import nbr.pipeline as pipeline
from nbr.config import REPO_ROOT


def _load_script():
    path = REPO_ROOT / "scripts" / "posthoc_explore_slot.py"
    spec = importlib.util.spec_from_file_location("posthoc_explore_slot", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


pes = _load_script()
COLUMNS = ["dataset", "user_id", "basket_id", "basket_seq", "item_id", "coarse_category", "fine_category", "quantity"]


def _frame(rows, dataset="instacart"):
    """rows = (user, seq, item); basket_id = f"{user}-{seq}"."""
    frame = pd.DataFrame([(dataset, u, f"{u}-{s}", s, i, "c", "f", 1) for u, s, i in rows], columns=COLUMNS)
    return frame.astype({"user_id": "string", "basket_id": "string", "basket_seq": "int32", "item_id": "string"})


# ---------------------------------------------------------------------------
# arguments and helpers
# ---------------------------------------------------------------------------
def test_parse_args_defaults():
    args = pes.parse_args([])
    assert args.profile == "medium" and args.datasets == ["instacart", "complete_journey"]
    assert args.out_dir.endswith("posthoc") and args.reference_tuning_dir is None


@pytest.mark.parametrize("argv", [["--profile", "holdout"], ["--profile", "holdout2"], ["--datasets", "holdout"], ["--datasets", ","]])
def test_parse_args_rejects_non_development_inputs(argv):
    with pytest.raises(SystemExit):
        pes.parse_args(argv)


def test_guard_path_refuses_confirmatory_folders(tmp_path):
    assert pes.guard_path(tmp_path / "reports" / "metrics") == tmp_path / "reports" / "metrics"
    for bad in ("final", "final2", "holdout", "holdout2"):
        with pytest.raises(PermissionError):
            pes.guard_path(tmp_path / "reports" / bad / "x.json")


def test_bin_labels_match_pipeline_t5_bins():
    for dataset, edges in pipeline.HISTORY_BINS.items():
        assert pes.bin_labels(edges) == pipeline._bin_labels(edges), dataset
    assert [name for name, _, _ in pes.bin_labels(pipeline.HISTORY_BINS["instacart"])] == ["(0,4]", "(4,6]", "(6,10]", "(10,20]", "(20,inf)"]
    assert [name for name, _, _ in pes.POSITION_BINS] == ["2-4", "5-10", "11-20", "21-50", ">50"]
    assert pes.POSITION_BINS[-1][2] == math.inf and pes.POSITION_BINS[0][1] == 2


# ---------------------------------------------------------------------------
# per-basket novelty
# ---------------------------------------------------------------------------
def test_basket_novelty_hand_checked():
    history = _frame([("A", 1, "x"), ("A", 1, "y"), ("A", 2, "x"), ("A", 2, "z"), ("A", 3, "z"), ("A", 3, "w"), ("A", 3, "v"), ("A", 3, "v"),
                      ("B", 1, "a")])
    target = _frame([("A", 4, "x"), ("A", 4, "q"), ("B", 2, "a")])
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
    # an item bought in the target only is new even if another user bought it before
    history = _frame([("A", 1, "x"), ("B", 1, "q"), ("B", 2, "q")])
    target = _frame([("A", 2, "q"), ("B", 3, "x")])
    out = pes.basket_novelty(history, target).set_index("basket_id")
    assert out.loc["A-2", "n_new"] == 1 and out.loc["B-3", "n_new"] == 1 and out.loc["B-2", "n_new"] == 0
    with pytest.raises(ValueError):                                              # target not after the history
        pes.basket_novelty(_frame([("A", 1, "x"), ("A", 3, "y")]), _frame([("A", 2, "z")]))
    with pytest.raises(ValueError):                                              # one basket_id with two basket_seq values
        broken = _frame([("A", 1, "x"), ("A", 2, "y")])
        broken.loc[1, "basket_id"] = "A-1"
        pes.basket_novelty(broken, _frame([("A", 3, "z")]))


# ---------------------------------------------------------------------------
# aggregation
# ---------------------------------------------------------------------------
def _baskets():
    # user A: three baskets, user B: one basket -> basket-weighted and user-weighted means differ
    return pd.DataFrame({
        "user_id": pd.array(["A", "A", "A", "B"], dtype="string"), "basket_id": ["A-2", "A-3", "A-4", "B-2"],
        "position": [2, 3, 12, 60], "is_target": [False, False, True, True], "size": [2, 4, 2, 4],
        "n_new": [1, 2, 0, 3], "new_share": [0.5, 0.5, 0.0, 0.75], "any_new": [1.0, 1.0, 0.0, 1.0]})


def test_summarise_weightings_and_empty():
    summary = pes.summarise(_baskets())
    assert summary["n_baskets"] == 4 and summary["n_users"] == 2
    assert summary["basket_weighted"]["mean_new_share"] == pytest.approx((0.5 + 0.5 + 0.0 + 0.75) / 4)
    assert summary["user_weighted"]["mean_new_share"] == pytest.approx(((0.5 + 0.5 + 0.0) / 3 + 0.75) / 2)
    assert summary["basket_weighted"]["p_any_new"] == pytest.approx(0.75) and summary["user_weighted"]["p_any_new"] == pytest.approx((2 / 3 + 1) / 2)
    assert summary["basket_weighted"]["mean_n_new"] == pytest.approx(1.5) and summary["user_weighted"]["mean_n_new"] == pytest.approx((1 + 3) / 2)
    assert summary["basket_weighted"]["mean_size"] == pytest.approx(3.0)
    empty = pes.summarise(_baskets().iloc[0:0])
    assert empty == {"n_baskets": 0, "n_users": 0, "basket_weighted": None, "user_weighted": None}
    assert pes.n_new_distribution(_baskets()) == pytest.approx({"0": 0.25, "1": 0.25, "2": 0.25, "3+": 0.25})
    assert pes.n_new_distribution(_baskets().iloc[0:0]) is None


def test_by_position_and_by_history_length():
    baskets = _baskets()
    rows = pes.by_position(baskets)
    assert [row["bin"] for row in rows] == ["2-4", "5-10", "11-20", "21-50", ">50"]
    assert [row["n_baskets"] for row in rows] == [2, 0, 1, 0, 1]
    assert rows[0]["user_weighted"]["mean_new_share"] == pytest.approx(0.5) and rows[1]["basket_weighted"] is None
    lengths = pd.Series({"A": 3, "B": 59})
    rows = pes.by_history_length(baskets, lengths, "instacart")
    assert [row["bin"] for row in rows] == ["(0,4]", "(4,6]", "(6,10]", "(10,20]", "(20,inf)"]
    assert [row["n_baskets"] for row in rows] == [3, 0, 0, 0, 1] and rows[4]["basket_weighted"]["mean_n_new"] == 3.0
    rows = pes.by_history_length(baskets, lengths, "complete_journey")
    assert [row["bin"] for row in rows] == ["(0,20]", "(20,50]", "(50,100]", "(100,inf)"] and [row["n_baskets"] for row in rows] == [3, 0, 1, 0]
    with pytest.raises(ValueError):
        pes.by_history_length(baskets, pd.Series({"A": 3}), "instacart")


# ---------------------------------------------------------------------------
# setup check
# ---------------------------------------------------------------------------
def _tuning_document(k_rows, gamma=0.8, commit="ref"):
    grid = [{"k": k, "weighting": w, "new_recall@10": 0.01 * k + (0.001 if w == "log1p" else 0.0), "seconds": 1.0}
            for w in ("binary", "log1p") for k in k_rows]
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
    run, reference = _tuning_document([10, 20, 50, 100], commit="run"), _tuning_document([20, 50, 100])
    result = pes.compare_tuning(run, reference)
    assert result["all_selections_equal"] and all(row["equal"] for row in result["selections"])
    assert [row["quantity"] for row in result["selections"]] == list(pes.SELECTION_PATHS)
    assert result["usercf_grid_excluding_k10_equal"] and len(result["usercf_k10_rows_run"]) == 2 and result["usercf_k10_rows_reference"] == []
    assert all(result["stage_grids_equal"].values()) and result["lr_validation_equal"] and result["lr_artifact_sha256"]["equal"]
    assert result["run_meta"]["git_commit"] == "run" and result["reference_meta"]["git_commit"] == "ref"


def test_compare_tuning_detects_differences():
    reference = _tuning_document([20, 50])
    run = _tuning_document([10, 20, 50], gamma=0.9)
    run["integration"]["selections"]["k"] = 10
    run["stage3_final_ndcg"][0]["ndcg"] = 0.29
    run["integration"]["usercf"]["grid"][1]["new_recall@10"] += 0.01     # the k = 20 binary row
    result = pes.compare_tuning(run, reference)
    unequal = {row["quantity"] for row in result["selections"] if not row["equal"]}
    assert unequal == {"gamma", "usercf_k"} and not result["all_selections_equal"]
    assert not result["usercf_grid_excluding_k10_equal"] and not result["stage_grids_equal"]["stage3_final_ndcg"]
    assert result["stage_grids_equal"]["stage1_repeat_recall"]
    assert pes._equal(0.1, 0.1 + 1e-12) and not pes._equal(0.1, 0.1 + 1e-6) and pes._equal("binary", "binary") and not pes._equal(True, 1)


def test_setup_check_reads_both_folders_and_stops_on_a_difference(tmp_path, monkeypatch):
    run_dir, ref_dir = tmp_path / "run" / "metrics", tmp_path / "ref" / "metrics"
    run_dir.mkdir(parents=True), ref_dir.mkdir(parents=True)
    monkeypatch.setenv("NBR_TUNING_DIR", str(run_dir))
    config = {"profile": "sample", "repo_root": tmp_path}
    (ref_dir / "sample_instacart_tuning.json").write_text(json.dumps(_tuning_document([20, 50])), encoding="utf-8")
    (run_dir / "sample_instacart_tuning.json").write_text(json.dumps(_tuning_document([10, 20, 50], commit="run")), encoding="utf-8")
    result = pes.setup_check(config, ["instacart"], ref_dir)
    assert result["passed"] and result["datasets"]["instacart"]["lr_artifact"]["matches"] is False
    assert result["datasets"]["instacart"]["files"]["reference_sha256"] == pes.sha256_of(ref_dir / "sample_instacart_tuning.json")
    (run_dir / "sample_instacart_tuning.json").write_text(json.dumps(_tuning_document([10, 20, 50], gamma=0.5)), encoding="utf-8")
    with pytest.raises(SystemExit, match="gamma"):
        pes.setup_check(config, ["instacart"], ref_dir)
    with pytest.raises(FileNotFoundError):
        pes.setup_check(config, ["complete_journey"], ref_dir)
    with pytest.raises(PermissionError):
        pes.setup_check(config, ["instacart"], tmp_path / "reports" / "final2")


# ---------------------------------------------------------------------------
# end-to-end run on synthetic data (sample profile, NBR_DATA_DIR redirected)
# ---------------------------------------------------------------------------
def _synthetic_dataset(folder, dataset, seed, n_users=40, n_items=30):
    """Users with 3-9 history baskets plus one target basket; items repeat from a per-user favourite set."""
    rng = np.random.default_rng(seed)
    rows = []
    for user in range(n_users):
        n_baskets = int(rng.integers(3, 10))
        favourites = rng.choice(n_items, size=5, replace=False)
        for seq in range(1, n_baskets + 2):
            items = set(rng.choice(favourites, size=3).tolist()) | set(rng.choice(n_items, size=2).tolist())
            rows.extend((dataset, str(user), f"{user}-{seq}", seq, str(item), "c", "f", 1) for item in sorted(items))
    frame = pd.DataFrame(rows, columns=COLUMNS).astype(
        {"user_id": "string", "basket_id": "string", "basket_seq": "int32", "item_id": "string", "quantity": "int16"})
    last = frame.groupby("user_id")["basket_seq"].transform("max")
    folder.mkdir(parents=True, exist_ok=True)
    frame[frame["basket_seq"] == last].to_parquet(folder / "eval_target.parquet", index=False)
    frame[frame["basket_seq"] < last].to_parquet(folder / "eval_history.parquet", index=False)
    frame[frame["basket_seq"] == last - 1].to_parquet(folder / "fit_target.parquet", index=False)
    frame[frame["basket_seq"] < last - 1].to_parquet(folder / "fit_history.parquet", index=False)


def test_end_to_end_on_synthetic_sample(tmp_path, monkeypatch):
    data_dir = tmp_path / "processed"
    for seed, dataset in enumerate(pes.DATASETS):
        _synthetic_dataset(data_dir / "sample" / dataset, dataset, seed)
    monkeypatch.setenv("NBR_DATA_DIR", str(data_dir))
    out = tmp_path / "out"
    json_path = pes.main(["--profile", "sample", "--out-dir", str(out)])
    document = json.loads(json_path.read_text(encoding="utf-8"))
    assert document["label"] == pes.LABEL and document["profile"] == "sample" and document["setup_check"] is None
    assert document["parts"] == ["setup check", "A"] and document["position_bins"] == ["2-4", "5-10", "11-20", "21-50", ">50"]
    assert set(document["datasets"]) == set(pes.DATASETS)
    assert set(document["inputs_sha256"]["instacart"]) == set(pes.INPUT_FILES)
    assert document["meta"]["git_commit"] is None or len(document["meta"]["git_commit"]) == 40
    for dataset, entry in document["datasets"].items():
        assert entry["history_bins"] == [name for name, _, _ in pes.bin_labels(pipeline.HISTORY_BINS[dataset])]
        for split in ("eval", "fit"):
            block = entry[split]
            checks, sets = block["checks"], block["basket_sets"]
            assert checks["n_users"] == 40 and checks["matches_split_target"] and checks["target_size_min"] >= 1
            assert checks["n_baskets_analysed"] == checks["n_history_baskets"] + checks["n_target_baskets"] - checks["n_first_baskets_dropped"]
            assert checks["n_first_baskets_dropped"] == 40 and checks["history_seq_starts_at_one_share"] == 1.0
            assert checks["target_seq_minus_last_history_seq"] == {"1": 40} and checks["users_with_skipped_baskets_before_target"] == 0
            assert set(sets) == set(pes.BASKET_SETS)
            assert sets["history_all"]["n_baskets"] == checks["n_history_baskets"] - 40 and sets["target"]["n_baskets"] == 40
            assert sets["history_min3"]["n_baskets"] <= sets["history_all"]["n_baskets"]
            target = sets["target"]
            assert target["basket_weighted"] == target["user_weighted"]                   # one target basket per user
            assert target["basket_weighted"]["p_any_new"] == pytest.approx(checks["target_any_new_users"] / 40)
            for name, block_set in sets.items():
                assert [row["bin"] for row in block_set["by_position"]] == document["position_bins"]
                assert sum(row["n_baskets"] for row in block_set["by_position"]) == block_set["n_baskets"]
                assert [row["bin"] for row in block_set["by_history_length"]] == entry["history_bins"]
                assert sum(row["n_baskets"] for row in block_set["by_history_length"]) == block_set["n_baskets"]
                dist = block_set["n_new_distribution"]
                assert sum(dist.values()) == pytest.approx(1.0)
                bw = block_set["basket_weighted"]
                assert 0.0 <= bw["mean_new_share"] <= bw["p_any_new"] <= 1.0 and bw["mean_n_new"] <= bw["mean_size"]
    markdown = (out / "explore_slot_sample.md").read_text(encoding="utf-8")
    assert pes.LABEL in markdown and "## Setup check" in markdown and "not checked" in markdown
    assert "### Instacart, eval split (main)" in markdown and "### Complete Journey, fit split (robustness check)" in markdown
    assert "(20,inf)" in markdown and "(100,inf)" in markdown and ">50" in markdown
    for word in ("significant", "novel", "outperform"):
        assert word not in markdown.lower()
    assert "confirmatory" not in markdown.replace("not confirmatory", "").replace("no confirmatory claim is made", "")
