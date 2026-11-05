"""Tests for nbr.pipeline: selection tie rules, tuning/evaluation on synthetic parquet data, leakage, holdout guard."""

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

COLUMNS = ["dataset", "user_id", "basket_id", "basket_seq", "item_id", "coarse_category", "fine_category", "quantity"]


# ---------------------------------------------------------------------------
# selection tie rules
# ---------------------------------------------------------------------------
def test_select_gamma_ties_prefer_closest_to_one():
    assert select_gamma({0.5: 0.3, 0.7: 0.3, 1.0: 0.2}) == 0.7
    assert select_gamma({0.5: 0.3, 0.7: 0.3, 1.0: 0.3}) == 1.0
    assert select_gamma({0.5: 0.4, 0.7: 0.3, 1.0: 0.3}) == 0.5
    assert select_gamma({0.5: 0.3, 0.7: 0.3 + 1e-15}) == 0.7  # numerically equal -> tie rule


def test_select_explore_ties_prefer_larger_alpha_then_gamma_c():
    scores = {(0.8, 1.0): 0.2, (0.9, 10.0): 0.2, (1.0, 10.0): 0.2, (1.0, 1.0): 0.2, (0.8, 100.0): 0.1}
    assert select_explore(scores) == (1.0, 10.0)
    scores[(0.8, 100.0)] = 0.2
    assert select_explore(scores) == (0.8, 100.0)
    scores[(0.9, 1.0)] = 0.25
    assert select_explore(scores) == (0.9, 1.0)


def test_select_r_ties_prefer_larger_r():
    assert select_r({5: 0.1, 6: 0.2, 7: 0.2, 10: 0.15}) == 7
    assert select_r({5: 0.1, 10: 0.1}) == 10


def test_decision_wording():
    assert decision([0.01, 0.05]) == "improvement observed"
    assert decision([-0.01, 0.05]) == "no difference established"
    assert decision([-0.05, -0.01]) == "lower"
    assert decision([0.0, 0.05]) == "no difference established"


# ---------------------------------------------------------------------------
# synthetic dataset on disk
# ---------------------------------------------------------------------------
def _frame(rows, dataset):
    return pd.DataFrame(
        [(dataset, u, f"b{s}", s, i, "c", "f", 1) for u, s, i in rows], columns=COLUMNS
    ).astype({"basket_seq": "int32"})


def _synthetic(rng, n_users=30, n_items=40, n_baskets=5, basket_size=4):
    items = [f"i{j:02d}" for j in range(n_items)]
    weights = 1.0 / (np.arange(n_items) + 1.0)
    weights /= weights.sum()
    history, target = [], []
    for u in range(n_users):
        user = f"u{u:02d}"
        for s in range(1, n_baskets + 1):
            chosen = rng.choice(items, size=basket_size, replace=False, p=weights)
            history += [(user, s, item) for item in chosen]
        # Target basket: two repeat items (from history) and two fresh items.
        seen = sorted({item for _, _, item in history if _ == user} if False else {i for uu, _, i in history if uu == user})
        repeat = list(rng.choice(seen, size=2, replace=False))
        unseen = [item for item in items if item not in seen]
        fresh = list(rng.choice(unseen, size=2, replace=False)) if len(unseen) >= 2 else []
        target += [(user, n_baskets + 1, item) for item in repeat + fresh]
    return history, target, items


def _write_dataset(root, profile, dataset, rng):
    folder = root / profile / dataset
    folder.mkdir(parents=True, exist_ok=True)
    history, target, items = _synthetic(rng)
    for split in ("fit", "eval"):
        _frame(history, dataset).to_parquet(folder / f"{split}_history.parquet", index=False)
        _frame(target, dataset).to_parquet(folder / f"{split}_target.parquet", index=False)
    catalogue = pd.DataFrame({
        "item_id": items,
        "dept": [f"d{j % 2}" for j in range(len(items))],
        "aisle": [f"a{j % 5}" for j in range(len(items))],
        "leaf": [f"l{j % 8}" for j in range(len(items))],
    })
    catalogue.to_parquet(folder / "items.parquet", index=False)
    return folder


def _config(root, profile="sample", role="smoke"):
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
    }


@pytest.fixture
def workspace(tmp_path):
    rng = np.random.default_rng(0)
    _write_dataset(tmp_path / "data", "sample", "instacart", rng)
    _write_dataset(tmp_path / "data", "sample", "complete_journey", rng)
    return tmp_path


def test_tune_and_evaluate_end_to_end(workspace):
    config = _config(workspace)
    path = run_tune(config, "instacart")
    assert path == workspace / "reports" / "metrics" / "sample_instacart_tuning.json"
    document = json.loads(path.read_text(encoding="utf-8"))
    assert set(document["selections"]) == {"gamma", "gamma_c", "alpha", "r"}
    assert len(document["stage1_repeat_recall"]) == 3
    assert len(document["stage2_explore_new_recall"]) == 4
    assert len(document["stage3_final_ndcg"]) == 3
    assert set(document["at_grid_edge"]) == {"gamma", "gamma_c", "alpha", "r"}
    assert document["meta"]["profile"] == "sample"

    selections = load_selections(config, "instacart")
    result = evaluate(config, "instacart", selections)
    assert set(result["primary"]) == {"H1_repeat_recall", "H2_new_recall", "H3_ndcg", "H3_recall", "H4_category_hit"}
    for key, entry in result["primary"].items():
        assert set(entry["ci"]) == {"0.95", "0.99375"}
        assert set(entry["decision"]) == {"0.95", "0.99375"}
        # Direction: diff = model - comparator, so a better model gives diff > 0.
        assert entry["diff"] == pytest.approx(entry["model_mean"] - entry["comparator_mean"])
        assert entry["ci"]["0.95"][0] <= entry["diff"] <= entry["ci"]["0.95"][1]
    assert result["primary"]["H1_repeat_recall"]["model"] == "repeat_star"
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

    config = _config(workspace)
    comparator = pd.Series([0.1, 0.2, 0.3], index=["a", "b", "c"])
    model = comparator + 0.5
    entry = _compare(comparator, model, config, primary=True)
    assert entry["diff"] == pytest.approx(0.5)
    assert entry["decision"]["0.95"] == "improvement observed"
    lower = _compare(model, comparator, config, primary=True)
    assert lower["decision"]["0.95"] == "lower"


def test_holdout_guard_in_evaluate(workspace):
    rng = np.random.default_rng(1)
    _write_dataset(workspace / "data", "holdout", "instacart", rng)
    config = _config(workspace, profile="holdout", role="final")
    selections = {"gamma": 0.8, "gamma_c": 1.0, "alpha": 10.0, "r": 8}
    with pytest.raises(PermissionError):
        evaluate(config, "instacart", selections)
    run_tune(config, "instacart")  # fit split is allowed under role final
    with pytest.raises(PermissionError):
        run_evaluate(config, "instacart")  # no --final
    result = evaluate(config, "instacart", selections, allow_holdout_eval=True)
    assert result["profile"] == "holdout"


def test_no_leakage_from_target(workspace):
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
    assert after["selections"] == before["selections"]
    assert after["n_users"]["all"] == before["n_users"]["all"] + 1

    from nbr.pipeline import prepare

    data = prepare(config, "instacart", "fit")
    lists, _ = build_lists(data["history"], data["pop"], after["selections"], config)
    assert "zz_new" not in set(lists["final"]["user_id"])
    assert not lists["explore_star"]["item_id"].isin(["never1", "never2"]).any()
