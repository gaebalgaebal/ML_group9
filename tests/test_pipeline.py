# *------------------- 2026 Fall Semester Machine Learning Term-project: Team 9 --------------------*
# Code First written at 2026-11-05
# Code Final Edited at 2026-11-13
# Editor: Yejun Kim

'''
Purpose : Tests for nbr.pipeline: selection tie rules, tuning/evaluation on synthetic parquet
          data, leakage, holdout guard.
Inputs  : Synthetic history, target and item-catalogue parquet files written to pytest's
          tmp_path and a configuration dict built in this file.
Outputs : pytest pass/fail results; tuning and evaluation JSON files and model artifacts are
          written under tmp_path only.
Usage   :
    python -m pytest tests/test_pipeline.py
'''

# ---------------------------------------------------------------------------
# imports
# ---------------------------------------------------------------------------
import json

import numpy as np
import pandas as pd
import pytest

from nbr.pipeline import (
    build_lists,
    decision,
    evaluate,
    load_selections,
    run_evaluate,
    run_tune,
    select_explore,
    select_gamma,
    select_r,
    tune,
)

# ---------------------------------------------------------------------------
# constants
# ---------------------------------------------------------------------------
COLUMNS = ["dataset", "user_id", "basket_id", "basket_seq", "item_id", "coarse_category", "fine_category", "quantity"]


# ---------------------------------------------------------------------------
# selection tie rules
# ---------------------------------------------------------------------------
def test_select_gamma_ties_prefer_closest_to_one():
    # Expect the best score to win and equal scores to go to the gamma closest to 1.
    assert select_gamma({0.5: 0.3, 0.7: 0.3, 1.0: 0.2}) == 0.7
    assert select_gamma({0.5: 0.3, 0.7: 0.3, 1.0: 0.3}) == 1.0
    assert select_gamma({0.5: 0.4, 0.7: 0.3, 1.0: 0.3}) == 0.5
    assert select_gamma({0.5: 0.3, 0.7: 0.3 + 1e-15}) == 0.7  # numerically equal -> tie rule


def test_select_explore_ties_prefer_larger_alpha_then_gamma_c():
    # Build scores in which four (gamma_c, alpha) pairs share the best value.
    scores = {(0.8, 1.0): 0.2, (0.9, 10.0): 0.2, (1.0, 10.0): 0.2, (1.0, 1.0): 0.2, (0.8, 100.0): 0.1}
    # Expect ties to go to the larger alpha, then the larger gamma_c, and a higher score to win.
    assert select_explore(scores) == (1.0, 10.0)
    scores[(0.8, 100.0)] = 0.2
    assert select_explore(scores) == (0.8, 100.0)
    scores[(0.9, 1.0)] = 0.25
    assert select_explore(scores) == (0.9, 1.0)


def test_select_r_ties_prefer_larger_r():
    # Expect the best score to win and equal scores to go to the larger r.
    assert select_r({5: 0.1, 6: 0.2, 7: 0.2, 10: 0.15}) == 7
    assert select_r({5: 0.1, 10: 0.1}) == 10


def test_decision_wording():
    # Expect the wording that matches the position of the interval relative to zero.
    assert decision([0.01, 0.05]) == "improvement observed"
    assert decision([-0.01, 0.05]) == "no difference established"
    assert decision([-0.05, -0.01]) == "lower"
    assert decision([0.0, 0.05]) == "no difference established"


# ---------------------------------------------------------------------------
# synthetic dataset on disk
# ---------------------------------------------------------------------------
def _frame(rows, dataset):
    '''Build a transaction frame for ``dataset`` from ``rows`` of (user, basket_seq, item).'''
    return pd.DataFrame(
        [(dataset, u, f"b{s}", s, i, "c", "f", 1) for u, s, i in rows], columns=COLUMNS
    ).astype({"basket_seq": "int32"})


def _synthetic(rng, n_users=30, n_items=40, n_baskets=5, basket_size=4):
    '''Draw history and target rows for ``n_users`` users with the generator ``rng``.

    Each user gets ``n_baskets`` history baskets of ``basket_size`` items out of ``n_items``.
    '''
    # Name the items and give them popularity weights that fall with the item index.
    items = [f"i{j:02d}" for j in range(n_items)]
    weights = 1.0 / (np.arange(n_items) + 1.0)
    weights /= weights.sum()
    history, target = [], []
    for u in range(n_users):
        user = f"u{u:02d}"
        # Draw the history baskets of the user without repeated items inside a basket.
        for s in range(1, n_baskets + 1):
            chosen = rng.choice(items, size=basket_size, replace=False, p=weights)
            history += [(user, s, item) for item in chosen]
        # Target basket: two repeat items (from history) and two fresh items.
        seen = sorted({item for _, _, item in history if _ == user} if False else {i for uu, _, i in history if uu == user})
        repeat = list(rng.choice(seen, size=2, replace=False))
        unseen = [item for item in items if item not in seen]
        fresh = list(rng.choice(unseen, size=2, replace=False)) if len(unseen) >= 2 else []
        target += [(user, n_baskets + 1, item) for item in repeat + fresh]
    # Return the history rows, the target rows and the item names.
    return history, target, items


def _write_dataset(root, profile, dataset, rng):
    '''Write the synthetic parquet files of ``dataset`` to ``root``/``profile``; uses ``rng``.'''
    # Create the dataset folder.
    folder = root / profile / dataset
    folder.mkdir(parents=True, exist_ok=True)
    # Draw the rows and write the same history and target for the fit and eval splits.
    history, target, items = _synthetic(rng)
    for split in ("fit", "eval"):
        _frame(history, dataset).to_parquet(folder / f"{split}_history.parquet", index=False)
        _frame(target, dataset).to_parquet(folder / f"{split}_target.parquet", index=False)
    # Write an item catalogue with three category columns.
    catalogue = pd.DataFrame({
        "item_id": items,
        "dept": [f"d{j % 2}" for j in range(len(items))],
        "aisle": [f"a{j % 5}" for j in range(len(items))],
        "leaf": [f"l{j % 8}" for j in range(len(items))],
    })
    catalogue.to_parquet(folder / "items.parquet", index=False)
    return folder


def _config(root, profile="sample", role="smoke"):
    '''Return a small pipeline configuration rooted at ``root`` for ``profile`` and ``role``.'''
    return {
        "profile": profile,
        "role": role,
        "repo_root": root,
        "data_dir": root / "data",
        "datasets": ["instacart", "complete_journey"],
        "levels": {
            "instacart": {"L1": "dept", "L2": "aisle", "L3": "leaf"},
            "complete_journey": {"L1": "dept", "L2": "aisle", "L3": "leaf"},
        },
        "repeat": {"gammas": [0.5, 0.8, 1.0]},
        "explore": {"level": "L2", "gamma_c": [0.8, 1.0], "alphas": [1, 10], "itemcf": {"neighbors": 5}},
        "combine": {"r": [5, 8, 10]},
        "bootstrap": {"n": 200, "seed": 42, "levels": [0.95, 0.99375]},
        "lr_repeat": {"decay_gammas": [0.5, 0.8, 1.0], "C": [0.1, 1.0], "val_fraction": 0.2, "seed": 42, "max_iter": 300},
        "usercf": {"k": [2, 5], "weighting": ["binary", "log1p"], "block_size": 7},
        "integration": {"r": list(range(0, 11))},
    }


@pytest.fixture
def workspace(tmp_path):
    '''Write both synthetic datasets of the sample profile under ``tmp_path`` and return it.'''
    # Seed the generator so that every run writes the same data.
    rng = np.random.default_rng(0)
    _write_dataset(tmp_path / "data", "sample", "instacart", rng)
    _write_dataset(tmp_path / "data", "sample", "complete_journey", rng)
    return tmp_path


# ---------------------------------------------------------------------------
# tuning, evaluation, holdout guard and leakage
# ---------------------------------------------------------------------------
def test_tune_and_evaluate_end_to_end(workspace):
    # Tune on the synthetic Instacart data and read the tuning document.
    config = _config(workspace)
    path = run_tune(config, "instacart")
    assert path == workspace / "reports" / "metrics" / "sample_instacart_tuning.json"
    document = json.loads(path.read_text(encoding="utf-8"))
    # Expect the four selections, one row per grid point of each stage and the profile name.
    assert set(document["selections"]) == {"gamma", "gamma_c", "alpha", "r"}
    assert len(document["stage1_repeat_recall"]) == 3
    assert len(document["stage2_explore_new_recall"]) == 4
    assert len(document["stage3_final_ndcg"]) == 3
    assert set(document["at_grid_edge"]) == {"gamma", "gamma_c", "alpha", "r"}
    assert document["meta"]["profile"] == "sample"

    # Evaluate the eval split with the tuned selections.
    selections = load_selections(config, "instacart")
    result = evaluate(config, "instacart", selections)
    # Expect the primary comparisons, the item-CF entry, secondary entries and history bins.
    assert set(result["primary"]) == {"H1_repeat_recall", "H2_new_recall", "H3_ndcg", "H3_recall", "H4_category_hit"}
    for key, entry in result["primary"].items():
        assert set(entry["ci"]) == {"0.95", "0.99375"}
        assert set(entry["decision"]) == {"0.95", "0.99375"}
        # Direction: diff = model - comparator, so a better model gives diff > 0.
        assert entry["diff"] == pytest.approx(entry["model_mean"] - entry["comparator_mean"])
        assert entry["ci"]["0.95"][0] <= entry["diff"] <= entry["ci"]["0.95"][1]
    assert result["primary"]["H1_repeat_recall"]["model"] == "repeat_star"
    h5 = result["H5"]
    assert h5["label"] == "secondary confirmatory; generated on development data"
    assert h5["comparator"] == "flat_explore" and h5["model"] == "itemcf_flat"
    assert set(h5["ci"]) == {"0.95"} and set(h5["decision"]) == {"0.95"}
    assert h5["n_users"] == result["primary"]["H2_new_recall"]["n_users"]
    assert h5["diff"] == pytest.approx(h5["model_mean"] - h5["comparator_mean"])  # item-CF better -> diff > 0
    assert h5["ci"]["0.95"][0] <= h5["diff"] <= h5["ci"]["0.95"][1]
    assert evaluate(config, "instacart", selections, include_itemcf=False)["H5"] is None
    assert result["primary"]["H4_category_hit"]["comparator"] == "population_category_ranking"
    assert "decision" not in next(iter(result["secondary"].values()))
    assert {"final_cf_vs_final_ndcg", "final_vs_popularity_recall", "explore_L3_vs_explore_star_new_recall"} <= set(result["secondary"])
    assert set(result["list_metrics"]) >= {"popularity", "personal_frequency", "final", "final_cf", "final_r10"}
    assert [row["bin"] for row in result["by_history_length"]] == ["(0,4]", "(4,6]", "(6,10]", "(10,20]", "(20,inf)"]
    assert result["by_history_length"][1]["n_users"] == 30  # every user has 5 baskets
    assert result["by_history_length"][0]["H3_ndcg"] is None

    # Transfer: CJ evaluated with Instacart selections, written under the _transfer name.
    transfer_path = run_evaluate(config, "complete_journey", transfer_from="instacart")
    assert transfer_path.name == "sample_complete_journey_eval_transfer.json"
    transfer = json.loads(transfer_path.read_text(encoding="utf-8"))
    assert transfer["transfer_from"] == "instacart"
    assert transfer["selections"]["gamma"] == selections["gamma"]


def test_bootstrap_direction_with_strictly_better_model(workspace):
    # With explicit per-user series the sign convention is checked directly through evaluate's helper.
    from nbr.pipeline import _compare

    # Build a model series that is 0.5 above the comparator for every user.
    config = _config(workspace)
    comparator = pd.Series([0.1, 0.2, 0.3], index=["a", "b", "c"])
    model = comparator + 0.5
    entry = _compare(comparator, model, config, primary=True)
    # Expect a positive difference with an improvement decision, and "lower" when swapped.
    assert entry["diff"] == pytest.approx(0.5)
    assert entry["decision"]["0.95"] == "improvement observed"
    lower = _compare(model, comparator, config, primary=True)
    assert lower["decision"]["0.95"] == "lower"


def test_h5_direction_item_cf_better(workspace, monkeypatch):
    """H5 diff > 0 when item-CF beats flat popularity: force itemcf_flat to equal the new target."""
    import nbr.pipeline as pipeline

    # Patch build_lists so that the item-CF list equals the distinct new-target rows.
    config = _config(workspace)
    original = pipeline.build_lists

    def patched(history, pop, selections, config, *, include_itemcf=True, lr_bundle=None, integration=None, **extra):
        '''Call the original build_lists and replace ``lists["itemcf_flat"]`` by the new target.'''
        lists, timings = original(history, pop, selections, config, include_itemcf=include_itemcf,
                                  lr_bundle=lr_bundle, integration=integration, **extra)
        data = pipeline.prepare(config, "instacart", "eval")
        perfect = data["new_target"][["user_id", "item_id"]].drop_duplicates().copy()
        perfect["rank"] = perfect.groupby("user_id").cumcount() + 1
        perfect["source"] = "itemcf"
        lists["itemcf_flat"] = perfect[["user_id", "rank", "item_id", "source"]]
        return lists, timings

    monkeypatch.setattr(pipeline, "build_lists", patched)
    result = pipeline.evaluate(config, "instacart", {"gamma": 0.8, "gamma_c": 1.0, "alpha": 10.0, "r": 8})
    # Expect a mean of 1 for the item-CF list and an improvement decision.
    assert result["H5"]["model_mean"] == pytest.approx(1.0)
    assert result["H5"]["diff"] > 0
    assert result["H5"]["decision"]["0.95"] == "improvement observed"


def test_holdout_guard_in_evaluate(workspace):
    # Write a holdout profile and use a configuration with the role final.
    rng = np.random.default_rng(1)
    _write_dataset(workspace / "data", "holdout", "instacart", rng)
    config = _config(workspace, profile="holdout", role="final")
    selections = {"gamma": 0.8, "gamma_c": 1.0, "alpha": 10.0, "r": 8}
    # Expect PermissionError for the eval split unless allow_holdout_eval is set.
    with pytest.raises(PermissionError):
        evaluate(config, "instacart", selections)
    run_tune(config, "instacart")  # fit split is allowed under role final
    with pytest.raises(PermissionError):
        run_evaluate(config, "instacart")  # no --final
    result = evaluate(config, "instacart", selections, allow_holdout_eval=True)
    assert result["profile"] == "holdout"


def test_no_leakage_from_target(workspace):
    # Tune once, add a new user with never-seen items to the fit target only and tune again.
    config = _config(workspace)
    before = tune(config, "instacart")
    folder = workspace / "data" / "sample" / "instacart"
    target = pd.read_parquet(folder / "fit_target.parquet")
    # Extra purchases that exist only in the target: a new user buying never-seen items.
    extra = _frame([("zz_new", 9, "never1"), ("zz_new", 9, "never2")], "instacart")
    items = pd.read_parquet(folder / "items.parquet")
    items = pd.concat([items, pd.DataFrame({"item_id": ["never1", "never2"], "dept": "d0", "aisle": "a0", "leaf": "l0"})])
    pd.concat([target, extra]).to_parquet(folder / "fit_target.parquet", index=False)
    items.to_parquet(folder / "items.parquet", index=False)
    after = tune(config, "instacart")
    # Expect unchanged selections, one more user and no target-only user or item in the lists.
    assert after["selections"] == before["selections"]
    assert after["n_users"]["all"] == before["n_users"]["all"] + 1

    from nbr.pipeline import prepare

    data = prepare(config, "instacart", "fit")
    lists, _ = build_lists(data["history"], data["pop"], after["selections"], config)
    assert "zz_new" not in set(lists["final"]["user_id"])
    assert not lists["explore_star"]["item_id"].isin(["never1", "never2"]).any()


# ---------------------------------------------------------------------------
# W6: logistic-regression repeat ranker (H6, H7)
# ---------------------------------------------------------------------------
# Key sets expected in the lr_repeat block of the evaluation document.
LR_PRIMARY = {"H6_repeat_recall", "H7_ndcg", "H7_recall"}
LR_SECONDARY_KEYS = {"repeat_lr_vs_repeat_pf_repeat_recall", "final_lr_vs_personal_frequency_ndcg"}


def test_tune_writes_lr_repeat_block_and_artifact(workspace):
    from nbr.pipeline import lr_artifact_path, load_lr_repeat, sha256_of

    # Tune and read the tuning document.
    config = _config(workspace)
    document = json.loads(run_tune(config, "instacart").read_text(encoding="utf-8"))
    # Expect the lr_repeat block with its grid, validation rows, coefficients and split sizes.
    assert document["label"] == "smoke (sample)"
    block = document["lr_repeat"]
    assert block["C_grid"] == [0.1, 1.0] and block["C"] in (0.1, 1.0)
    assert [row["C"] for row in block["validation"]] == [0.1, 1.0]
    assert {"val_recall", "n_iter", "fit_seconds"} <= set(block["validation"][0])
    assert [row["feature"] for row in block["coefficients"]][:3] == ["dec_g0.5", "dec_g0.8", "dec_g1.0"]
    assert block["n_rows"] > 0 and 0 < block["positive_rate"] < 1
    assert block["n_train_users"] == 24 and block["n_val_users"] == 6
    # Expect the model artifact on disk with the recorded hash and the same coefficients.
    path = lr_artifact_path(config, "instacart")
    assert path == workspace / "artifacts" / "sample" / "instacart_lr_repeat.joblib"
    assert path.exists() and block["artifact"] == str(path) and block["sha256"] == sha256_of(path)
    bundle = load_lr_repeat(config, "instacart")
    assert bundle["C"] == block["C"] and bundle["sha256"] == block["sha256"]
    assert {name: value for name, value in bundle["coefficients"].items()} == {
        row["feature"]: row["coef"] for row in block["coefficients"]
    }


def test_evaluate_h6_h7_structure_and_h1_h5_unchanged(workspace):
    from nbr.pipeline import load_lr_repeat

    # Tune, then evaluate once with and once without the logistic-regression bundle.
    config = _config(workspace)
    run_tune(config, "instacart")
    selections = load_selections(config, "instacart")
    with_lr = evaluate(config, "instacart", selections, lr_bundle=load_lr_repeat(config, "instacart"))
    without = evaluate(config, "instacart", selections)
    # Expect no lr_repeat block and no repeat_lr list without the bundle.
    assert without["lr_repeat"] is None and "repeat_lr" not in without["list_metrics"]
    # H1-H5, secondary comparisons and the history-length breakdown do not depend on the LR model.
    for key in ("primary", "H5", "secondary", "by_history_length", "n_users", "selections"):
        assert with_lr[key] == without[key]
    assert {name: with_lr["list_metrics"][name] for name in without["list_metrics"]} == without["list_metrics"]

    # Expect the lr_repeat block with its primary, secondary and history-length entries.
    assert with_lr["label"] == "smoke (sample)"
    block = with_lr["lr_repeat"]
    assert block["label"] == "smoke (sample)" and block["sha256"] and block["C"] in (0.1, 1.0)
    assert set(block["primary"]) == LR_PRIMARY
    for key, entry in block["primary"].items():
        assert set(entry["ci"]) == {"0.95", "0.99375"} and set(entry["decision"]) == {"0.95", "0.99375"}
        assert entry["diff"] == pytest.approx(entry["model_mean"] - entry["comparator_mean"])
        assert entry["ci"]["0.95"][0] <= entry["diff"] <= entry["ci"]["0.95"][1]
    h6, h7 = block["primary"]["H6_repeat_recall"], block["primary"]["H7_ndcg"]
    assert (h6["comparator"], h6["model"]) == ("repeat_star", "repeat_lr")
    assert h6["n_users"] == with_lr["primary"]["H1_repeat_recall"]["n_users"]
    assert (h7["comparator"], h7["model"]) == ("final", "final_lr") and h7["n_users"] == with_lr["n_users"]["all"]
    assert set(block["secondary"]) == LR_SECONDARY_KEYS
    assert "decision" not in block["secondary"]["final_lr_vs_personal_frequency_ndcg"]
    assert [row["bin"] for row in block["by_history_length"]] == [row["bin"] for row in with_lr["by_history_length"]]
    assert set(block["by_history_length"][1]) == {"bin", "n_users"} | LR_PRIMARY
    assert {"repeat_lr", "final_lr"} <= set(with_lr["list_metrics"])
    assert with_lr["list_metrics"]["final_lr"]["ndcg"] == pytest.approx(h7["model_mean"])

    # CLI path: run_evaluate loads the artifact; --no-lr equivalent skips it.
    document = json.loads(run_evaluate(config, "instacart").read_text(encoding="utf-8"))
    assert set(document["lr_repeat"]["primary"]) == LR_PRIMARY
    assert json.loads(run_evaluate(config, "instacart", include_lr=False).read_text(encoding="utf-8"))["lr_repeat"] is None


def test_h6_h7_direction_with_perfect_lr_lists(workspace, monkeypatch):
    """diff > 0 when the LR lists are better: replace repeat_lr / final_lr by the (dedup) targets."""
    import nbr.pipeline as pipeline

    # Tune, load the bundle and patch build_lists to return the targets as the two LR lists.
    config = _config(workspace)
    run_tune(config, "instacart")
    bundle = pipeline.load_lr_repeat(config, "instacart")
    original = pipeline.build_lists

    def perfect(frame, source):
        '''Turn the (user, item) rows of ``frame`` into a ranked list frame labelled ``source``.'''
        out = frame[["user_id", "item_id"]].drop_duplicates().copy()
        out["rank"] = out.groupby("user_id").cumcount() + 1
        out["source"] = source
        return out[["user_id", "rank", "item_id", "source"]]

    def patched(history, pop, selections, config, *, include_itemcf=True, lr_bundle=None, integration=None, **extra):
        '''Call the original build_lists and replace the two LR lists by the eval targets.'''
        lists, timings = original(history, pop, selections, config, include_itemcf=include_itemcf,
                                  lr_bundle=lr_bundle, integration=integration, **extra)
        data = pipeline.prepare(config, "instacart", "eval")
        lists["repeat_lr"] = perfect(data["repeat_target"], "repeat_lr")
        lists["final_lr"] = perfect(data["target"], "final_lr")
        return lists, timings

    monkeypatch.setattr(pipeline, "build_lists", patched)
    result = pipeline.evaluate(config, "instacart", load_selections(config, "instacart"), lr_bundle=bundle)
    primary = result["lr_repeat"]["primary"]
    # Expect a mean of 1 and an improvement decision for every primary LR comparison.
    assert primary["H6_repeat_recall"]["model_mean"] == pytest.approx(1.0)
    assert primary["H7_ndcg"]["model_mean"] == pytest.approx(1.0)
    for key in LR_PRIMARY:
        assert primary[key]["diff"] > 0
        assert primary[key]["decision"]["0.95"] == "improvement observed"
    assert result["lr_repeat"]["secondary"]["repeat_lr_vs_repeat_pf_repeat_recall"]["diff"] > 0


# ---------------------------------------------------------------------------
# integration block: final_lr_ucf versus final
# ---------------------------------------------------------------------------
def test_evaluate_h7prime_c3_vs_c0_keys(workspace):
    """PREREG v1.3 H7' = C3 (final_lr_ucf) vs C0 (final): NDCG@10 and Recall@10, both CI levels, decision wording."""
    # Tune, evaluate through run_evaluate and read the secondary integration entries.
    config = _config(workspace)
    run_tune(config, "instacart")
    document = json.loads(run_evaluate(config, "instacart").read_text(encoding="utf-8"))
    secondary = document["integration"]["secondary"]
    # Expect both metrics of final_lr_ucf versus final with intervals and decisions at two levels.
    for key, metric in (("final_lr_ucf_vs_final_ndcg", "ndcg"), ("final_lr_ucf_vs_final_recall", "recall")):
        entry = secondary[key]
        assert (entry["comparator"], entry["model"], entry["metric"]) == ("final", "final_lr_ucf", metric)
        assert entry["n_users"] == document["n_users"]["all"]
        assert set(entry["ci"]) == {"0.95", "0.99375"} and set(entry["decision"]) == {"0.95", "0.99375"}
        assert entry["diff"] == pytest.approx(entry["model_mean"] - entry["comparator_mean"])
        assert entry["comparator_mean"] == pytest.approx(document["list_metrics"]["final"][metric])
        assert entry["model_mean"] == pytest.approx(document["list_metrics"]["final_lr_ucf"][metric])
        assert entry["decision"]["0.95"] == decision(entry["ci"]["0.95"])
        assert entry["decision"]["0.99375"] == decision(entry["ci"]["0.99375"])
    # The other integration comparisons stay CI-only; the primary entries carry both levels.
    for key in ("final_star_ucf_vs_final_ndcg", "final_lr_r_vs_final_lr_ndcg",
                "final_lr_ucf_vs_personal_frequency_ndcg", "usercf_vs_itemcf_flat_new_recall"):
        assert key in secondary and "decision" not in secondary[key]
    for key in ("H8_new_recall", "H9_ndcg", "H9_recall"):
        assert set(document["integration"]["primary"][key]["ci"]) == {"0.95", "0.99375"}
