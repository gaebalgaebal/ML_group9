# *------------------- 2026 Fall Semester Machine Learning Term-project: Team 9 --------------------*
# Code First written at 2026-11-06
# Code Final Edited at 2026-11-13
# Editor: Yejun Kim

'''
Purpose : Tests for scripts/posthoc_day_decay.py: argument guard, selection and binning
          helpers, an end-to-end run on synthetic data.
Inputs  : The script loaded from its file path and synthetic parquet files written to
          pytest's tmp_path (NBR_DATA_DIR is redirected there).
Outputs : pytest pass/fail results; the script's report files are written under tmp_path
          only.
Usage   :
    python -m pytest tests/test_posthoc_day_decay.py
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

from nbr.config import REPO_ROOT


# ---------------------------------------------------------------------------
# script loading and constants
# ---------------------------------------------------------------------------
def _load_script():
    '''Load scripts/posthoc_day_decay.py as a module from its file path and return it.'''
    # Locate the script in the repository.
    path = REPO_ROOT / "scripts" / "posthoc_day_decay.py"
    # Execute the file as the module posthoc_day_decay.
    spec = importlib.util.spec_from_file_location("posthoc_day_decay", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# Load the script once for all tests.
pdd = _load_script()
COLUMNS = ["dataset", "user_id", "basket_id", "basket_seq", "item_id", "coarse_category", "fine_category", "quantity"]


# ---------------------------------------------------------------------------
# arguments and helpers
# ---------------------------------------------------------------------------
def test_parse_args_defaults():
    # Parse an empty argument list.
    args = pdd.parse_args([])
    # Expect the medium profile, both datasets and the default bootstrap settings.
    assert args.profile == "medium" and args.datasets == ["instacart", "complete_journey"]
    assert (args.bootstrap_n, args.seed) == (1000, 42) and args.out_dir.endswith("posthoc")


@pytest.mark.parametrize("argv", [["--profile", "holdout"], ["--profile", "holdout2"], ["--datasets", "holdout"], ["--datasets", ","]])
def test_parse_args_rejects_non_development_inputs(argv):
    # Expect the parser to exit for a holdout profile or an invalid dataset list.
    with pytest.raises(SystemExit):
        pdd.parse_args(argv)


def test_select_param_ties_prefer_larger():
    # Expect the best score to win and equal scores to go to the larger parameter.
    assert pdd.select_param({7.0: 0.3, 30.0: 0.3, math.inf: 0.2}) == 30.0
    assert pdd.select_param({7.0: 0.3, math.inf: 0.3}) == math.inf
    assert pdd.select_param({0.5: 0.4, 0.9: 0.4, 1.0: 0.3}) == 0.9
    assert pdd.select_param({0.5: 0.4 + 1e-15, 0.9: 0.4}) == 0.9    # numerically equal -> tie rule
    assert pdd.select_param({0.5: 0.5, 0.9: 0.4}) == 0.5


def test_bin_labels_and_grids():
    # Expect the six history-length bins and the fixed gamma and half-life grids.
    assert [name for name, _, _ in pdd.bin_labels(pdd.HISTORY_BIN_EDGES)] == [
        "(0,4]", "(4,10]", "(10,20]", "(20,50]", "(50,100]", "(100,inf)"]
    assert pdd.GAMMA_GRID == [0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.97, 0.98, 0.99, 0.995, 1.0]
    assert pdd.HALF_LIFE_GRID[:-1] == [7, 14, 30, 60, 90, 180, 365] and math.isinf(pdd.HALF_LIFE_GRID[-1])


def test_by_history_length():
    # Build per-user recall for two half-lives and one gamma, and five history lengths.
    index = pd.Index(["u1", "u2", "u3", "u4"], name="user_id")
    day = {7.0: pd.Series([1.0, 0.0, 0.5, 0.5], index=index), math.inf: pd.Series([0.0, 1.0, 0.5, 0.0], index=index)}
    basket = {0.5: pd.Series([0.2, 0.4, 0.6, 0.8], index=index)}
    lengths = pd.Series({"u1": 2, "u2": 4, "u3": 12, "u4": 200, "u9": 7})   # u9 has no repeat target: not scored
    rows = pdd.by_history_length(day, basket, lengths)
    # Expect one row per bin with the user count and the mean recall, None for an empty bin.
    assert [row["bin"] for row in rows] == ["(0,4]", "(4,10]", "(10,20]", "(20,50]", "(50,100]", "(100,inf)"]
    assert [row["n_users"] for row in rows] == [2, 0, 1, 0, 0, 1]
    assert rows[0]["day"] == [{"h": 7.0, "recall": 0.5}, {"h": "inf", "recall": 0.5}]
    assert rows[0]["basket"] == [{"gamma": 0.5, "recall": pytest.approx(0.3)}]
    assert rows[1]["day"][0]["recall"] is None and rows[1]["basket"][0]["recall"] is None
    assert rows[5]["day"] == [{"h": 7.0, "recall": 0.5}, {"h": "inf", "recall": 0.0}]


def test_unified_scale_and_param_helpers():
    # Convert three gammas with a mean gap of 3 days and a median gap of 2 days.
    rows = pdd.unified_scale([0.5, 0.97, 1.0], mean_gap=3.0, median_gap=2.0)
    # Expect basket half-lives, their day equivalents and "inf" for gamma = 1.
    assert rows[0] == {"gamma": 0.5, "basket_half_life": pytest.approx(1.0), "day_equivalent_mean_gap": pytest.approx(3.0),
                       "day_equivalent_median_gap": pytest.approx(2.0)}
    assert rows[1]["basket_half_life"] == pytest.approx(math.log(0.5) / math.log(0.97))
    assert rows[2] == {"gamma": 1.0, "basket_half_life": "inf", "day_equivalent_mean_gap": "inf", "day_equivalent_median_gap": "inf"}
    # Expect the parameter helpers to map infinity to and from the string "inf".
    assert pdd.jsonable(math.inf) == "inf" and pdd.jsonable(0.97) == 0.97
    assert pdd.fmt_param(math.inf) == "inf" and pdd.fmt_param(0.97) == "0.97" and pdd.fmt_param(30.0) == "30"
    assert pdd._param("inf") == math.inf and pdd._param(30.0) == 30.0


def test_gap_summary_and_compare():
    # Summarise three gaps for three users, one of them with a single basket.
    gaps = pd.Series([0, 5, 10], dtype="int64")
    lengths = pd.Series({"a": 1, "b": 3, "c": 2})
    summary = pdd.gap_summary(gaps, lengths)
    # Expect the gap count, the mean, the median and the share of zero gaps.
    assert summary["n_gaps"] == 3 and summary["users_with_one_basket"] == 1
    assert summary["mean_days"] == pytest.approx(5.0) and summary["median_days"] == 5.0
    assert summary["share_zero"] == pytest.approx(1 / 3) and summary["mean_days_excluding_zero"] == pytest.approx(7.5)
    # Compare an all-zero series with an all-one series.
    index = pd.Index(["u1", "u2", "u3"], name="user_id")
    result = pdd.compare(pd.Series([0.0, 0.0, 0.0], index=index), pd.Series([1.0, 1.0, 1.0], index=index), 20, 0)
    # Expect a difference of 1 with both interval bounds at 1.
    assert result["diff"] == pytest.approx(1.0) and result["ci_0.95"] == [pytest.approx(1.0), pytest.approx(1.0)]
    assert result["n_users"] == 3 and result["mean_a"] == 0.0 and result["mean_b"] == 1.0


# ---------------------------------------------------------------------------
# end-to-end run on synthetic data (sample profile, NBR_DATA_DIR redirected)
# ---------------------------------------------------------------------------
def _synthetic_dataset(folder, dataset, seed, n_users=40, n_items=30):
    """Users with 3-9 history baskets on ascending days plus one target basket; items repeat from a per-user favourite set."""
    # Seed the generator of this dataset.
    rng = np.random.default_rng(seed)
    rows, days = [], []
    for user in range(n_users):
        # Draw the basket count, the favourite items and the first day of the user.
        n_baskets = int(rng.integers(3, 10))
        favourites = rng.choice(n_items, size=5, replace=False)
        day = int(rng.integers(0, 30))
        for seq in range(1, n_baskets + 2):
            # Fill the basket with three favourite draws and two draws from all items.
            basket_id = f"{user}-{seq}"
            items = set(rng.choice(favourites, size=3).tolist()) | set(rng.choice(n_items, size=2).tolist())
            rows.extend((dataset, str(user), basket_id, seq, str(item), "c", "f", 1) for item in sorted(items))
            days.append((str(user), basket_id, day))
            # Move the day forward by 0 to 14 days.
            day += int(rng.integers(0, 15))
    # Build the typed transaction frame.
    frame = pd.DataFrame(rows, columns=COLUMNS).astype(
        {"user_id": "string", "basket_id": "string", "basket_seq": "int32", "item_id": "string", "quantity": "int16"})
    # Write the last basket as eval target and the one before it as fit target.
    last = frame.groupby("user_id")["basket_seq"].transform("max")
    folder.mkdir(parents=True, exist_ok=True)
    frame[frame["basket_seq"] == last].to_parquet(folder / "eval_target.parquet", index=False)
    frame[frame["basket_seq"] < last].to_parquet(folder / "eval_history.parquet", index=False)
    frame[frame["basket_seq"] == last - 1].to_parquet(folder / "fit_target.parquet", index=False)
    frame[frame["basket_seq"] < last - 1].to_parquet(folder / "fit_history.parquet", index=False)
    # Write the day of every basket.
    day_frame = pd.DataFrame(days, columns=["user_id", "basket_id", "day"]).astype(
        {"user_id": "string", "basket_id": "string", "day": "int64"})
    day_frame.to_parquet(folder / "basket_days.parquet", index=False)


def test_end_to_end_on_synthetic_sample(tmp_path, monkeypatch):
    # Write both synthetic datasets, point NBR_DATA_DIR at them and run the sample profile.
    data_dir = tmp_path / "processed"
    for seed, dataset in enumerate(pdd.DATASETS):
        _synthetic_dataset(data_dir / "sample" / dataset, dataset, seed)
    monkeypatch.setenv("NBR_DATA_DIR", str(data_dir))
    out = tmp_path / "out"
    json_path = pdd.main(["--profile", "sample", "--out-dir", str(out), "--bootstrap-n", "50", "--seed", "1"])
    document = json.loads(json_path.read_text(encoding="utf-8"))
    # Expect the document header, per-dataset grids and comparisons, and the transfer entries.
    assert document["label"] == pdd.LABEL and document["profile"] == "sample"
    assert document["bootstrap"] == {"n": 50, "seed": 1, "level": 0.95}
    assert set(document["datasets"]) == set(pdd.DATASETS)
    assert document["reproduction_check"] == {}            # reference numbers exist for medium only
    assert set(document["inputs_sha256"]["instacart"]) == {"fit_history", "fit_target", "eval_history", "eval_target", "basket_days"}
    assert document["meta"]["git_commit"] is None or len(document["meta"]["git_commit"]) == 40
    for dataset, entry in document["datasets"].items():
        fit, ev = entry["fit"], entry["eval"]
        assert [row["h"] for row in fit["day"]] == [7, 14, 30, 60, 90, 180, 365, "inf"]
        assert [row["gamma"] for row in fit["basket"]] == pdd.GAMMA_GRID
        assert fit["day"][-1]["recall"] == pytest.approx(fit["basket"][-1]["recall"])    # h = inf == gamma = 1
        assert ev["day"][-1]["recall"] == pytest.approx(ev["basket"][-1]["recall"])
        assert fit["h_star"] in [7, 14, 30, 60, 90, 180, 365, "inf"] and fit["gamma_star"] in pdd.GAMMA_GRID
        comparisons = ev["comparisons"]
        assert set(comparisons) == {"day_hstar_vs_frequency", "day_hstar_vs_basket_gstar", "basket_gstar_vs_frequency", "basket_0.97_vs_frequency"}
        assert comparisons["day_hstar_vs_frequency"]["mean_a"] == pytest.approx(ev["basket"][-1]["recall"])
        assert comparisons["day_hstar_vs_frequency"]["n_users"] == entry["n_users"]["eval_repeat_target"]
        for item in comparisons.values():
            assert item["ci_0.95"][0] <= item["diff"] <= item["ci_0.95"][1]
        assert len(entry["by_history_length"]) == 6 and sum(b["n_users"] for b in entry["by_history_length"]) == entry["n_users"]["fit_repeat_target"]
        assert len(entry["unified_scale"]) == len(pdd.GAMMA_GRID) and entry["unified_scale"][-1]["basket_half_life"] == "inf"
        assert entry["gaps"]["n_gaps"] == entry["n_history_baskets"]["fit"] - entry["n_users"]["fit_all"]
    assert set(document["transfer"]) == {"instacart_day", "instacart_basket", "complete_journey_day", "complete_journey_basket"}
    for item in document["transfer"].values():
        assert item["transfer_loss"] == pytest.approx(item["own_recall"] - item["transferred_recall"])
        if item["own_param"] == item["transferred_param"]:
            assert item["transfer_loss"] == 0.0
    # Expect the report, figure and CSV files, and a report without the excluded words.
    assert (out / "day_decay_sample.md").exists() and (out / "fig_recall_vs_halflife.png").exists()
    csv = pd.read_csv(out / "fig_recall_vs_halflife.csv")
    assert len(csv) == 2 * (len(pdd.HALF_LIFE_GRID) + len(pdd.GAMMA_GRID)) and csv["selected"].sum() == 4
    markdown = (out / "day_decay_sample.md").read_text(encoding="utf-8")
    assert pdd.LABEL in markdown and "## (c) Transfer" in markdown and "(100,inf)" in markdown
    for word in ("significant", "novel", "outperform", "confirmatory claim is made"):
        assert word not in markdown.replace("No confirmatory claim is made", "")
