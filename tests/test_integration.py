"""W8 tests: slot re-selection without label leakage, combination lists, H8/H9 structure and direction."""

import json

import numpy as np
import pandas as pd
import pytest

import nbr.pipeline as pipeline
from nbr.combine import combine
from nbr.pipeline import (
    INTEGRATION_COMBOS,
    build_lists,
    evaluate,
    load_integration_selections,
    load_lr_repeat,
    load_selections,
    run_evaluate,
    run_tune,
    select_r,
    tune,
)
from tests.test_pipeline import _config, _frame, workspace  # noqa: F401  (fixture re-used)

PRIMARY = {"H8_new_recall", "H9_ndcg", "H9_recall"}
SECONDARY = {"final_star_ucf_vs_final_ndcg", "final_lr_r_vs_final_lr_ndcg", "final_lr_ucf_vs_final_ndcg",
             "final_lr_ucf_vs_personal_frequency_ndcg", "usercf_vs_itemcf_flat_new_recall"}
COMBOS = {"final_lr_ucf", "final_star_ucf", "final_lr_r"}


def _lists(rows, source):
    frame = pd.DataFrame(rows, columns=["user_id", "rank", "item_id"])
    frame["source"] = source
    return frame


def test_combine_r_zero_is_explore_only_and_r_ten_is_repeat_only_with_fill():
    repeat = _lists([("u", i, f"r{i}") for i in range(1, 7)], "repeat")  # 6 repeat candidates
    explore = _lists([("u", i, f"e{i}") for i in range(1, 11)], "explore")
    zero = combine(repeat, explore, 0, 10)
    assert zero["item_id"].tolist() == [f"e{i}" for i in range(1, 11)] and (zero["source"] == "explore").all()
    ten = combine(repeat, explore, 10, 10)
    assert ten["item_id"].tolist() == [f"r{i}" for i in range(1, 7)] + [f"e{i}" for i in range(1, 5)]
    assert ten["rank"].tolist() == list(range(1, 11))


def test_slot_tie_rule_prefers_larger_r():
    assert select_r({r: 0.3 for r in range(11)}) == 10
    assert select_r({0: 0.3, 3: 0.4, 7: 0.4, 10: 0.2}) == 7
    assert select_r({0: 0.4, 10: 0.4 - 1e-15}) == 10


def test_tune_integration_block(workspace):
    config = _config(workspace)
    document = json.loads(run_tune(config, "instacart").read_text(encoding="utf-8"))
    block = document["integration"]
    assert block["label"] == "smoke (sample)"
    assert set(block["selections"]) == {"k", "weighting", "C", "r"} and set(block["selections"]["r"]) == COMBOS
    assert block["selections"]["k"] in (2, 5) and block["selections"]["weighting"] in ("binary", "log1p")
    assert block["selections"]["C"] == document["lr_repeat"]["C"]
    assert len(block["usercf"]["grid"]) == 4 and block["usercf"]["selections"] == {
        "k": block["selections"]["k"], "weighting": block["selections"]["weighting"]}
    assert block["r_grid"] == list(range(11))
    for name in COMBOS:
        scores = block["r_scores"][name]
        assert [row["r"] for row in scores] == list(range(11))
        assert block["selections"]["r"][name] == select_r({row["r"]: row["ndcg"] for row in scores})
        assert block["at_grid_edge"]["r"][name] == (block["selections"]["r"][name] in (0, 10))
    # LR combinations are scored on the LR validation users only; the other one on all fit users.
    n_val = document["lr_repeat"]["n_val_users"]
    assert block["r_selection_users"]["final_lr_ucf"]["n_users"] == n_val == 6
    assert block["r_selection_users"]["final_lr_r"]["n_users"] == n_val
    assert block["r_selection_users"]["final_star_ucf"]["n_users"] == document["n_users"]["all"] == 30
    assert set(load_integration_selections(config, "instacart")) == {"k", "weighting", "C", "r", "source"}


def test_lr_slot_selection_uses_only_validation_users_not_trained_on(workspace, monkeypatch):
    """NDCG for the LR combinations is computed on validation users; the scoring model never saw them."""
    config = _config(workspace)
    data = pipeline.prepare(config, "instacart", "fit")
    history, target, pop = data["history"], data["target"], data["pop"]
    seen_fit_users = []
    import nbr.lr_repeat as lr_module

    original_fit = lr_module._fit

    def spy_fit(X_part, y_part, C, max_iter, seed):
        seen_fit_users.append(set(X_part.index.get_level_values("user_id")))
        return original_fit(X_part, y_part, C, max_iter, seed)

    monkeypatch.setattr(lr_module, "_fit", spy_fit)
    bundle = pipeline.tune_lr_repeat(history, target, config)
    train_users, val_users = set(bundle["train_users"]), set(bundle["val_users"])
    assert not train_users & val_users and train_users | val_users == set(history["user_id"])
    assert len(val_users) == 6
    # The model used for slot selection is the C* fit on the training users: it was fitted on train users only.
    train_fits = [users for users in seen_fit_users if users == train_users]
    assert train_fits and all(not users & val_users for users in train_fits)
    assert bundle["train_model"] is not bundle["model"]

    scored = {}
    original_ndcg = pipeline.per_user_ndcg

    def spy_ndcg(lists, frame_target, k):
        result = original_ndcg(lists, frame_target, k)
        scored.setdefault(frozenset(lists["source"].unique()), set()).update(result.index)
        return result

    monkeypatch.setattr(pipeline, "per_user_ndcg", spy_ndcg)
    from nbr.explore import explore_list
    from nbr.repeat import repeat_list

    lists = {"repeat_star": repeat_list(history, 0.8, 10, pop), "explore_star": explore_list(history, 10, 1.0, 10.0, "L2")}
    block = pipeline.tune_integration(history, target, pop, bundle, lists, config)
    lr_scored = [users for sources, users in scored.items() if "repeat_lr" in sources]
    assert lr_scored and all(users <= val_users for users in lr_scored)
    assert all(not users & train_users for users in lr_scored)
    assert block["r_selection_users"]["final_lr_ucf"]["n_users"] == len(val_users)


def test_combination_lists_are_valid(workspace):
    config = _config(workspace)
    run_tune(config, "instacart")
    data = pipeline.prepare(config, "instacart", "eval")
    lists, _ = build_lists(data["history"], data["pop"], load_selections(config, "instacart"), config,
                           lr_bundle=load_lr_repeat(config, "instacart"),
                           integration=load_integration_selections(config, "instacart"))
    users = set(data["history"]["user_id"])
    for name in COMBOS | {"usercf"}:
        frame = lists[name]
        assert set(frame["user_id"]) == users
        assert not frame.duplicated(["user_id", "item_id"]).any()
        sizes = frame.groupby("user_id")["rank"].agg(["size", "min", "max"])
        assert (sizes["size"] == 10).all() and (sizes["min"] == 1).all() and (sizes["max"] == 10).all()
        assert list(frame.columns) == ["user_id", "rank", "item_id", "source"]
    selections = load_integration_selections(config, "instacart")
    for name, (repeat_name, explore_name, _) in INTEGRATION_COMBOS.items():
        expected = combine(lists[repeat_name], lists[explore_name], selections["r"][name], 10)
        pd.testing.assert_frame_equal(lists[name], expected)


def test_evaluate_integration_structure_and_h1_h7_unchanged(workspace):
    config = _config(workspace)
    run_tune(config, "instacart")
    selections = load_selections(config, "instacart")
    bundle = load_lr_repeat(config, "instacart")
    integration = load_integration_selections(config, "instacart")
    with_int = evaluate(config, "instacart", selections, lr_bundle=bundle, integration=integration)
    without = evaluate(config, "instacart", selections, lr_bundle=bundle)
    assert without["integration"] is None
    for key in ("primary", "H5", "secondary", "by_history_length", "lr_repeat", "n_users", "selections"):
        assert with_int[key] == without[key]
    assert {name: with_int["list_metrics"][name] for name in without["list_metrics"]} == without["list_metrics"]
    assert COMBOS | {"usercf"} <= set(with_int["list_metrics"])

    block = with_int["integration"]
    assert block["label"] == "smoke (sample)" and block["selections"]["r"] == integration["r"]
    assert set(block["primary"]) == PRIMARY and set(block["secondary"]) == SECONDARY
    for key, entry in block["primary"].items():
        assert set(entry["ci"]) == {"0.95", "0.99375"} and set(entry["decision"]) == {"0.95", "0.99375"}
        assert entry["diff"] == pytest.approx(entry["model_mean"] - entry["comparator_mean"])
        assert entry["ci"]["0.95"][0] <= entry["diff"] <= entry["ci"]["0.95"][1]
    h8, h9 = block["primary"]["H8_new_recall"], block["primary"]["H9_ndcg"]
    assert (h8["comparator"], h8["model"]) == ("flat_explore", "usercf")
    assert h8["n_users"] == with_int["primary"]["H2_new_recall"]["n_users"]
    assert (h9["comparator"], h9["model"]) == ("final_lr", "final_lr_ucf") and h9["n_users"] == with_int["n_users"]["all"]
    assert h9["model_mean"] == pytest.approx(with_int["list_metrics"]["final_lr_ucf"]["ndcg"])
    assert "decision" not in block["secondary"]["final_lr_ucf_vs_final_ndcg"]
    assert [row["bin"] for row in block["by_history_length"]] == [row["bin"] for row in with_int["by_history_length"]]
    assert set(block["by_history_length"][1]) == {"bin", "n_users"} | PRIMARY
    assert block["process_peak_working_set_mib"] is None or block["process_peak_working_set_mib"] > 0

    document = json.loads(run_evaluate(config, "instacart").read_text(encoding="utf-8"))
    assert set(document["integration"]["primary"]) == PRIMARY
    assert json.loads(run_evaluate(config, "instacart", include_integration=False).read_text(encoding="utf-8"))["integration"] is None


def test_h9_direction_with_perfect_combination(workspace, monkeypatch):
    config = _config(workspace)
    run_tune(config, "instacart")
    original = pipeline.build_lists

    def perfect(frame, source):
        out = frame[["user_id", "item_id"]].drop_duplicates().copy()
        out["rank"] = out.groupby("user_id").cumcount() + 1
        out["source"] = source
        return out[["user_id", "rank", "item_id", "source"]]

    def patched(history, pop, selections, config, *, include_itemcf=True, lr_bundle=None, integration=None):
        lists, timings = original(history, pop, selections, config, include_itemcf=include_itemcf,
                                  lr_bundle=lr_bundle, integration=integration)
        data = pipeline.prepare(config, "instacart", "eval")
        lists["final_lr_ucf"] = perfect(data["target"], "final_lr_ucf")
        lists["usercf"] = perfect(data["new_target"], "usercf")
        return lists, timings

    monkeypatch.setattr(pipeline, "build_lists", patched)
    result = evaluate(config, "instacart", load_selections(config, "instacart"),
                      lr_bundle=load_lr_repeat(config, "instacart"),
                      integration=load_integration_selections(config, "instacart"))
    primary = result["integration"]["primary"]
    assert primary["H9_ndcg"]["model_mean"] == pytest.approx(1.0) and primary["H8_new_recall"]["model_mean"] == pytest.approx(1.0)
    for key in PRIMARY:
        assert primary[key]["diff"] > 0 and primary[key]["decision"]["0.95"] == "improvement observed"
    assert result["integration"]["secondary"]["final_lr_ucf_vs_final_ndcg"]["diff"] > 0


def test_no_leakage_from_eval_target_and_fit_target_only_users(workspace):
    config = _config(workspace)
    before = tune(config, "instacart")
    data = pipeline.prepare(config, "instacart", "eval")
    bundle = load_lr_repeat(config, "instacart")
    lists_before, _ = build_lists(data["history"], data["pop"], before["selections"], config,
                                  lr_bundle=bundle, integration=before["integration"]["selections"])
    folder = workspace / "data" / "sample" / "instacart"
    items = pd.read_parquet(folder / "items.parquet")
    items = pd.concat([items, pd.DataFrame({"item_id": ["never1", "never2"], "dept": "d0", "aisle": "a0", "leaf": "l0"})])
    items.to_parquet(folder / "items.parquet", index=False)
    # Eval target gains purchases (existing users buying unseen items): lists are target-free.
    eval_target = pd.read_parquet(folder / "eval_target.parquet")
    extra = _frame([("u00", 9, "never1"), ("u01", 9, "never2"), ("u00", 9, "i39")], "instacart")
    pd.concat([eval_target, extra]).to_parquet(folder / "eval_target.parquet", index=False)
    data_after = pipeline.prepare(config, "instacart", "eval")
    lists_after, _ = build_lists(data_after["history"], data_after["pop"], before["selections"], config,
                                 lr_bundle=bundle, integration=before["integration"]["selections"])
    for name in lists_before:
        pd.testing.assert_frame_equal(lists_before[name], lists_after[name])
    # Fit target gains a user without history (not an LR training row): selections k, weighting, C*, r* unchanged.
    fit_target = pd.read_parquet(folder / "fit_target.parquet")
    extra = _frame([("zz_new", 9, "never1"), ("zz_new", 9, "never2")], "instacart")
    pd.concat([fit_target, extra]).to_parquet(folder / "fit_target.parquet", index=False)
    after = tune(config, "instacart")
    assert after["selections"] == before["selections"]
    assert after["integration"]["selections"] == before["integration"]["selections"]
    assert after["lr_repeat"]["n_rows"] == before["lr_repeat"]["n_rows"]
