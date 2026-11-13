# *------------------- 2026 Fall Semester Machine Learning Term-project: Team 9 --------------------*
# Code First written at 2026-11-06
# Code Final Edited at 2026-11-13
# Editor: Yejun Kim, Bonjun Gu

'''
Purpose : Tests for nbr.r3_eval: shared user split, non-monotone cycle signal (R3 > R2),
          H10 structure/direction, outputs. R3 is the repeat ranker with the R2 features plus
          binned cycle features, R2 is the logistic-regression repeat ranker and H10 is the
          comparison of R3 against R2 on repeat Recall@10.
Inputs  : Synthetic baskets built in this file and pytest's tmp_path and monkeypatch fixtures.
Outputs : pytest pass/fail results; files are written only below tmp_path.
Usage   :
    python -m pytest tests/test_r3_eval.py
'''

# ---------------------------------------------------------------------------
# imports
# ---------------------------------------------------------------------------
import json

import numpy as np
import pandas as pd
import pytest

from nbr.baselines import item_popularity
from nbr.features import repeat_features
from nbr.features_cycle import ONCE_COLUMN, cycle_feature_names
from nbr.io import write_json
from nbr.lr_repeat import build_training, lr_repeat_list, split_users
from nbr.metrics import per_user_recall
from nbr.r3_eval import (
    BOOTSTRAP,
    HYPOTHESES,
    R1_SOURCE,
    R2_SOURCE,
    R3_SOURCE,
    build_training_r3,
    coefficient_table,
    compare,
    fit_r2_r3,
    load_gamma_star,
    load_r3,
    load_r3_config,
    overdue_table,
    r3_list,
    report_path,
    run,
)

# ---------------------------------------------------------------------------
# constants
# ---------------------------------------------------------------------------
GAMMAS = (0.5, 0.8, 1.0)
CFG = {
    "C": [0.1, 1.0],
    "overdue_bins": [0.0, 0.5, 1.0, 1.5, 2.5, 5.0, 10.01],
    "recency_bins": [0, 1, 2, 3, 5, 10, 1000000],
    "item_cadence_ratio_bins": [0.0, 0.5, 1.0, 1.5, 2.5, 5.0, 10.01],
    "val_fraction": 0.2,
    "seed": 42,
    "max_iter": 500,
}
COLUMNS = ["dataset", "user_id", "basket_id", "basket_seq", "item_id", "coarse_category", "fine_category", "quantity"]


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _frame(rows):
    '''Build a (user_id, basket_seq, item_id) frame from ``rows`` with an int32 basket_seq.'''
    return pd.DataFrame(rows, columns=["user_id", "basket_seq", "item_id"]).astype({"basket_seq": "int32"})


def _cycle_data(rng, n_users=150, n_items=40, n_baskets=12, basket_size=6, p_in=0.9, p_out=0.02):
    """Random baskets; a history pair is repurchased with probability p_in when its overdue lies in
    [0.5, 1.5) ("around the cycle") and p_out otherwise: a non-monotone signal in overdue."""
    # Name the items of the catalogue.
    items = [f"i{j:02d}" for j in range(n_items)]
    rows = []
    for u in range(n_users):
        user = f"u{u:03d}"
        for s in range(1, n_baskets + 1):
            # Draw the distinct items of one basket.
            rows += [(user, s, item) for item in rng.choice(items, size=basket_size, replace=False)]
    history = _frame(rows)
    # Compute the overdue value of every (user, item) pair of the history.
    base = repeat_features(history, GAMMAS)
    overdue = base["overdue"].to_numpy()
    # Set the repurchase probability of each pair from its overdue value.
    p = np.where((overdue >= 0.5) & (overdue < 1.5), p_in, p_out)
    # Draw the repurchased pairs and put them into one target basket per user.
    chosen = base.loc[rng.random(len(p)) < p, ["user_id", "item_id"]]
    target = _frame([(user, n_baskets + 1, item) for user, item in chosen.itertuples(index=False)])
    return history, target


# ---------------------------------------------------------------------------
# training rows and the shared user split
# ---------------------------------------------------------------------------
def test_r2_and_r3_share_rows_labels_and_user_split(monkeypatch):
    import nbr.lr_repeat as module

    # Build the training rows of both models from the same synthetic baskets.
    history, target = _cycle_data(np.random.default_rng(1), n_users=40)
    X2, y2, g2 = build_training(history, target, GAMMAS)
    X3, y3, g3 = build_training_r3(history, target, CFG, GAMMAS)
    # Expect the same rows, labels, users and validation split, with the R2 columns unchanged.
    assert X3.index.equals(X2.index) and np.array_equal(y2, y3) and np.array_equal(g2, g3)
    assert list(X3.columns) == cycle_feature_names(CFG, GAMMAS)
    pd.testing.assert_frame_equal(X3[X2.columns], X2)
    assert split_users(g2, 0.2, 42)[1].tolist() == split_users(g3, 0.2, 42)[1].tolist()

    # Replace the fitting function by a wrapper that records the users of every call.
    seen = []
    original = module._fit

    def spy(X_part, y_part, C, max_iter, seed):
        '''Record the users of ``X_part`` and call the original fit with the same arguments.'''
        seen.append(set(X_part.index.get_level_values("user_id")))
        return original(X_part, y_part, C, max_iter, seed)

    monkeypatch.setattr(module, "_fit", spy)
    # Fit both models on the same data.
    fitted = fit_r2_r3(history, target, CFG, GAMMAS)
    # Expect the same training users in the calls of both models and matching fit summaries.
    n_grid = len(CFG["C"])
    assert len(seen) == 2 * (n_grid + 1)  # per model: one fit per C on train users, then the refit on everybody
    r2_calls, r3_calls = seen[: n_grid + 1], seen[n_grid + 1:]
    assert r2_calls[0] == r3_calls[0] == set(split_users(g2, 0.2, 42)[0])  # same training users (seed 42)
    assert r2_calls[-1] == r3_calls[-1] == set(g2)
    assert fitted["r2"]["n_val_users"] == fitted["r3"]["n_val_users"] == 8
    assert fitted["r2"]["seed"] == fitted["r3"]["seed"] == 42
    assert [row["C"] for row in fitted["r2"]["validation"]] == [row["C"] for row in fitted["r3"]["validation"]] == [0.1, 1.0]
    assert fitted["r3"]["bins"]["overdue_bins"] == CFG["overdue_bins"]
    assert fitted["r3"]["decay_gammas"] == list(GAMMAS)
    assert set(fitted["r3"]["coefficients"]) == set(cycle_feature_names(CFG, GAMMAS))
    assert isinstance(fitted["r2"]["at_grid_edge"], bool)


def test_fit_is_reproducible_with_the_same_seed():
    # Fit both models twice on the same synthetic baskets.
    history, target = _cycle_data(np.random.default_rng(2), n_users=40)
    a = fit_r2_r3(history, target, CFG, GAMMAS)
    b = fit_r2_r3(history, target, CFG, GAMMAS)
    # Expect the same selected C, coefficients and validation recall in both runs.
    assert a["r3"]["C"] == b["r3"]["C"] and a["r2"]["C"] == b["r2"]["C"]
    np.testing.assert_allclose(a["r3"]["model"].coef_, b["r3"]["model"].coef_)
    assert a["r3"]["validation"][0]["val_recall"] == b["r3"]["validation"][0]["val_recall"]


# ---------------------------------------------------------------------------
# non-monotone cycle signal: the binned model must win
# ---------------------------------------------------------------------------
def test_r3_beats_r2_when_repurchase_peaks_near_the_cycle():
    # Fit both models on baskets whose repurchase probability is high only near the cycle.
    history, target = _cycle_data(np.random.default_rng(3))
    fitted = fit_r2_r3(history, target, CFG, GAMMAS)
    # Build both top-10 lists on a second synthetic data set drawn with another seed.
    eval_history, eval_target = _cycle_data(np.random.default_rng(103))
    pop = item_popularity(eval_history)
    r2_lists = lr_repeat_list(eval_history, fitted["r2"], 10, pop)
    r3_lists = r3_list(eval_history, fitted["r3"], 10, pop)
    # Expect lists with the fixed columns, the right source and at most 10 distinct items per user.
    assert list(r3_lists.columns) == ["user_id", "rank", "item_id", "source"]
    assert (r3_lists["source"] == R3_SOURCE).all() and (r2_lists["source"] == R2_SOURCE).all()
    assert not r3_lists.duplicated(["user_id", "item_id"]).any()
    assert r3_lists.groupby("user_id")["rank"].max().le(10).all()
    # Expect the mean repeat recall of R3 to exceed that of R2 by more than 0.02.
    r2_recall = per_user_recall(r2_lists, eval_target, 10).mean()
    r3_recall = per_user_recall(r3_lists, eval_target, 10).mean()
    assert r3_recall > r2_recall + 0.02
    coefficients = fitted["r3"]["coefficients"]
    # bins inside the peak get positive weight, the bins past it do not exceed them
    assert coefficients["ob_0.5_1"] > 0 and coefficients["ob_1_1.5"] > 0
    assert coefficients["ob_1.5_2.5"] < coefficients["ob_0.5_1"]


# ---------------------------------------------------------------------------
# H10 structure and direction
# ---------------------------------------------------------------------------
def test_hypothesis_names_and_compare_structure():
    # Expect the fixed (comparator, model) pair of every hypothesis and the bootstrap settings.
    assert HYPOTHESES["H10_repeat_recall"] == (R2_SOURCE, R3_SOURCE)
    assert HYPOTHESES["aux_r3_vs_r1_repeat_recall"] == (R1_SOURCE, R3_SOURCE)
    assert HYPOTHESES["aux_r2_vs_r1_repeat_recall"] == (R1_SOURCE, R2_SOURCE)
    assert BOOTSTRAP == {"n": 1000, "seed": 42, "levels": [0.95, 0.99375]}
    # Compare a per-user series with a copy shifted up by 0.5.
    comparator = pd.Series([0.1, 0.2, 0.3, 0.4], index=["a", "b", "c", "d"])
    entry = compare(comparator, comparator + 0.5, R2_SOURCE, R3_SOURCE)
    # Expect all entry keys, a difference of 0.5 and the decision "improvement observed".
    assert set(entry) == {"comparator", "model", "metric", "users", "n_users", "comparator_mean", "model_mean", "diff",
                          "ci", "decision"}
    assert entry["comparator"] == "repeat_lr" and entry["model"] == "repeat_r3"
    assert entry["metric"] == "repeat Recall@10" and entry["users"] == "users with repeat target"
    assert entry["n_users"] == 4 and set(entry["ci"]) == set(entry["decision"]) == {"0.95", "0.99375"}
    assert entry["diff"] == pytest.approx(0.5) and entry["diff"] == pytest.approx(entry["model_mean"] - entry["comparator_mean"])
    assert entry["decision"]["0.95"] == entry["decision"]["0.99375"] == "improvement observed"
    # Swap the two series and expect a difference of -0.5 with the decision "lower".
    worse = compare(comparator + 0.5, comparator, R2_SOURCE, R3_SOURCE)
    assert worse["diff"] == pytest.approx(-0.5) and worse["decision"]["0.95"] == "lower"
    # Add mixed-sign changes and expect the wider interval to contain the narrower one.
    noisy = compare(comparator, comparator + pd.Series([0.3, -0.3, 0.2, -0.2], index=comparator.index), R2_SOURCE, R3_SOURCE)
    assert noisy["ci"]["0.95"][0] <= noisy["diff"] <= noisy["ci"]["0.95"][1]
    assert noisy["ci"]["0.99375"][0] <= noisy["ci"]["0.95"][0] and noisy["ci"]["0.99375"][1] >= noisy["ci"]["0.95"][1]


# ---------------------------------------------------------------------------
# inputs: gamma* source and the r3 config
# ---------------------------------------------------------------------------
def test_load_gamma_star_reads_w4_file_or_fails(tmp_path):
    # Expect an error while the tuning file does not exist.
    with pytest.raises(FileNotFoundError):
        load_gamma_star("medium", "instacart", tmp_path)
    # Write a tuning file with a selections block.
    path = tmp_path / "medium_instacart_tuning.json"
    path.write_text(json.dumps({"selections": {"gamma": 0.8, "gamma_c": 0.9, "alpha": 100.0, "r": 10}}), encoding="utf-8")
    # Expect the stored gamma together with the path of the file.
    assert load_gamma_star("medium", "instacart", tmp_path) == (0.8, path)


def test_repo_r3_config_is_valid():
    # Load the R3 configuration stored in the repository.
    cfg = load_r3_config()
    # Expect the fixed C grid, bin edges and split settings, and 30 feature names.
    assert cfg["C"] == [0.001, 0.01, 0.1, 1.0, 10.0]
    assert cfg["overdue_bins"][-1] > 10.0 and cfg["item_cadence_ratio_bins"][-1] > 10.0
    assert cfg["recency_bins"] == [0, 1, 2, 3, 5, 10, 1000000]
    assert cfg["val_fraction"] == 0.2 and cfg["seed"] == 42 and cfg["max_iter"] == 2000
    assert len(cycle_feature_names(cfg)) == 30


# ---------------------------------------------------------------------------
# interpretation tables
# ---------------------------------------------------------------------------
def test_overdue_and_coefficient_tables():
    # Fit both models on synthetic baskets and build the overdue table of R3.
    history, target = _cycle_data(np.random.default_rng(4), n_users=40)
    fitted = fit_r2_r3(history, target, CFG, GAMMAS)
    r3 = fitted["r3"]
    table = overdue_table(r3)
    # Expect one row per overdue bin plus "once", a zero reference row and unscaled effects.
    assert table["bin"].tolist() == ["[0, 0.5)", "[0.5, 1)", "[1, 1.5)", "[1.5, 2.5)", "[2.5, 5)", "[5, 10.01)", "once"]
    assert table.loc[0, "column"] == "(reference)" and table.loc[0, "effect_log_odds"] == 0.0
    assert table.loc[6, "column"] == ONCE_COLUMN
    j = r3["features"].index("ob_0.5_1")
    assert table.loc[1, "coef_std"] == pytest.approx(r3["coefficients"]["ob_0.5_1"])
    assert table.loc[1, "effect_log_odds"] == pytest.approx(r3["coefficients"]["ob_0.5_1"] / r3["scaler"].scale_[j])
    assert table.loc[1, "train_mean"] == pytest.approx(r3["scaler"].mean_[j])

    # Build the table of all coefficients of both models.
    full = coefficient_table(r3, fitted["r2"])
    # Expect every R3 feature plus the intercept, with R2 values only for the R2 features.
    assert full["feature"].tolist() == r3["features"] + ["intercept"]
    assert full["r2_coef_std"].notna().sum() == len(fitted["r2"]["features"]) + 1
    assert full.loc[full["feature"] == "recency", "r2_coef_std"].item() == pytest.approx(fitted["r2"]["coefficients"]["recency"])
    assert np.isnan(full.loc[full["feature"] == "ob_once", "r2_coef_std"].item())


# ---------------------------------------------------------------------------
# full run on a synthetic workspace
# ---------------------------------------------------------------------------
def _write_dataset(root, profile, dataset, rng):
    '''Write synthetic parquet files of ``dataset`` to ``root``/``profile``/``dataset``.

    Baskets are drawn with ``rng``; the files are the history and target of the fit and eval
    splits and items.parquet.
    '''
    # Create the folder of the data set.
    folder = root / profile / dataset
    folder.mkdir(parents=True, exist_ok=True)
    items = set()
    for split in ("fit", "eval"):
        # Draw the history and the target of this split.
        history, target = _cycle_data(rng, n_users=60, n_baskets=8)
        for name, frame in (("history", history), ("target", target)):
            # Add the remaining stored columns with constant values and write the frame.
            stored = pd.DataFrame({
                "dataset": dataset, "user_id": frame["user_id"], "basket_id": "b" + frame["basket_seq"].astype(str),
                "basket_seq": frame["basket_seq"], "item_id": frame["item_id"], "coarse_category": "c",
                "fine_category": "f", "quantity": 1,
            })[COLUMNS]
            stored.to_parquet(folder / f"{split}_{name}.parquet", index=False)
        # Collect the items seen in this split.
        items |= set(history["item_id"]) | set(target["item_id"])
    # Write the item table with one of five aisles per item.
    items = sorted(items)
    pd.DataFrame({"item_id": items, "dept": "d", "aisle": [f"a{j % 5}" for j in range(len(items))], "leaf": "l"}
                 ).to_parquet(folder / "items.parquet", index=False)


def _config(root, role="smoke"):
    '''Return a run configuration of the sample profile rooted at ``root`` with role ``role``.'''
    return {
        "profile": "sample", "role": role, "repo_root": root, "data_dir": root / "data",
        "datasets": ["instacart"],
        "levels": {"instacart": {"L1": "dept", "L2": "aisle", "L3": "leaf"}},
        "bootstrap": {"n": 200, "seed": 42, "levels": [0.95, 0.99375]},
        "lr_repeat": {"decay_gammas": [0.5, 0.8, 1.0], "C": [0.1, 1.0], "val_fraction": 0.2, "seed": 42, "max_iter": 300},
    }


def test_run_refuses_final_profiles(tmp_path):
    # Expect a PermissionError for a configuration whose role is "final".
    with pytest.raises(PermissionError):
        run(_config(tmp_path, role="final"), CFG, "instacart", tmp_path)


def test_run_end_to_end_on_synthetic_workspace(tmp_path):
    # Write a synthetic data set and a tuning file with gamma 0.8 below tmp_path.
    _write_dataset(tmp_path / "data", "sample", "instacart", np.random.default_rng(5))
    gamma_dir = tmp_path / "w4"
    gamma_dir.mkdir()
    (gamma_dir / "sample_instacart_tuning.json").write_text(json.dumps({"selections": {"gamma": 0.8}}), encoding="utf-8")
    # Run the evaluation, write the report and read it back.
    config = _config(tmp_path)
    document = run(config, CFG, "instacart", gamma_dir)
    path = write_json(report_path(config, "instacart"), document, config)
    assert path == tmp_path / "reports" / "metrics" / "sample_instacart_r3.json"
    stored = json.loads(path.read_text(encoding="utf-8"))

    # Expect the label, the gamma source, both fit blocks, the hypotheses and the breakdowns.
    assert stored["label"] == "smoke (sample)" and stored["k"] == 10
    assert stored["gamma_star"] == {"value": 0.8, "source": str(gamma_dir / "sample_instacart_tuning.json")}
    assert stored["fit"]["same_user_split"] is True
    for name in ("r2", "r3"):
        block = stored["fit"][name]
        assert block["C_grid"] == [0.1, 1.0] and block["C"] in (0.1, 1.0)
        assert [row["C"] for row in block["validation"]] == [0.1, 1.0]
        assert all({"C", "val_recall", "n_iter", "fit_seconds"} <= set(row) for row in block["validation"])
        assert isinstance(block["at_grid_edge"], bool)
    assert len(stored["fit"]["r3"]["coefficients"]) == 30 and len(stored["fit"]["r2"]["coefficients"]) == 12
    assert set(stored["hypotheses"]) == {"H10_repeat_recall", "aux_r3_vs_r1_repeat_recall", "aux_r2_vs_r1_repeat_recall"}
    h10 = stored["hypotheses"]["H10_repeat_recall"]
    assert h10["comparator"] == "repeat_lr" and h10["model"] == "repeat_r3"
    assert h10["n_users"] == stored["n_users"]["eval_repeat_target"]
    assert h10["diff"] == pytest.approx(h10["model_mean"] - h10["comparator_mean"])
    assert h10["ci"]["0.95"][0] <= h10["diff"] <= h10["ci"]["0.95"][1]
    assert set(stored["list_metrics"]) == {"repeat_star", "repeat_lr", "repeat_r3"}
    assert h10["model_mean"] == pytest.approx(stored["list_metrics"]["repeat_r3"]["repeat_recall"])
    assert [row["bin"] for row in stored["by_history_length"]] == ["(0,4]", "(4,6]", "(6,10]", "(10,20]", "(20,inf)"]
    assert stored["by_history_length"][2]["n_users"] == 60  # every user has 8 baskets
    assert stored["by_history_length"][2]["H10_repeat_recall"]["n_users"] == h10["n_users"]
    assert stored["by_history_length"][0]["H10_repeat_recall"] is None
    assert [row["bin"] for row in stored["overdue_coefficients"]][-1] == "once"
    assert stored["meta"]["profile"] == "sample" and "git_commit" in stored["meta"]
    assert stored["timing_seconds"]["total"] > 0

    # Expect the figure and both coefficient tables on disk.
    figure = tmp_path / "reports" / "figures" / "sample_instacart_r3_overdue_coef.png"
    assert figure.exists() and figure.stat().st_size > 1000 and stored["outputs"]["figure"] == str(figure)
    overdue_csv = pd.read_csv(tmp_path / "reports" / "tables" / "sample_instacart_r3_overdue_coef.csv")
    assert overdue_csv["bin"].tolist() == [row["bin"] for row in stored["overdue_coefficients"]]
    coefficients_csv = pd.read_csv(tmp_path / "reports" / "tables" / "sample_instacart_r3_coefficients.csv")
    assert len(coefficients_csv) == 31

    # Expect the stored model bundle to exist and to match the report.
    artifact = tmp_path / "artifacts" / "sample" / "instacart_r3.joblib"
    assert stored["artifact"]["path"] == str(artifact) and artifact.exists()
    bundle = load_r3(config, "instacart")
    assert bundle["C"] == stored["fit"]["r3"]["C"] and bundle["features"] == stored["fit"]["r3"]["features"]
    assert bundle["bins"]["recency_bins"] == CFG["recency_bins"]
    assert bundle["r2_refit"]["C"] == stored["fit"]["r2"]["C"]
    # the stored model reproduces the evaluated R3 list
    from nbr.pipeline import prepare

    data = prepare(config, "instacart", "eval")
    lists = r3_list(data["history"], bundle, 10, data["pop"])
    recall = per_user_recall(lists, data["repeat_target"], 10).mean()
    assert recall == pytest.approx(stored["list_metrics"]["repeat_r3"]["repeat_recall"])
