# *------------------- 2026 Fall Semester Machine Learning Term-project: Team 9 --------------------*
# Code First written at 2026-11-07
# Code Final Edited at 2026-11-13
# Editor: Yejun Kim

'''
Purpose : W19 tests: the PREREG v1.5 two-CF final system (R2' = R2 + icf, C4 = R2' + E2) in
          nbr.v15 and its wiring into nbr.pipeline (tune: separate C and r selection; evaluate:
          the "v15" block; nothing else changes). R2 is the logistic-regression repeat ranker,
          icf is the item-CF similarity feature and E2 is user-based collaborative filtering.
Inputs  : The synthetic workspace fixture and the _config helper of tests.test_pipeline.
          The default-value test also loads the sample configuration of the repository.
Outputs : pytest pass/fail results; files are written only below the workspace of the fixture.
Usage   :
    python -m pytest tests/test_v15.py
'''

# ---------------------------------------------------------------------------
# imports
# ---------------------------------------------------------------------------
import json

import numpy as np
import pandas as pd
import pytest

import nbr.pipeline as pipeline
import nbr.v15 as v15
from nbr.combine import combine
from nbr.config import load_config
from nbr.features import feature_names
from nbr.lr_repeat import SOURCE_ICF, lr_repeat_list
from nbr.pipeline import ICF_SUFFIX, icf_enabled, load_v15_selections, lr_artifact_path, sha256_of
from tests.test_pipeline import _config, workspace  # noqa: F401  (fixture re-used)

# ---------------------------------------------------------------------------
# constants
# ---------------------------------------------------------------------------
PRIMARY = {"H14_ndcg", "H15_ndcg"}
SECONDARY = {"repeat_lri_vs_repeat_lr_repeat_recall", "repeat_lri_vs_repeat_star_repeat_recall", "final_lri_ucf_vs_final_lr_ucf_recall"}
VOLATILE = {"timing_seconds", "peak_memory_mib", "process_peak_working_set_mib", "utc_time", "meta", "fit_seconds", "seconds"}
FEATURES = feature_names([0.5, 0.8, 1.0])


# ---------------------------------------------------------------------------
# fixtures and helpers
# ---------------------------------------------------------------------------
def _icf_config(root):
    '''Return the test configuration rooted at ``root`` with the item-CF feature flag set.'''
    # Start from the configuration of tests.test_pipeline.
    config = _config(root)
    # Switch on the itemcf_feature flag of the lr_repeat block.
    config["lr_repeat"] = {**config["lr_repeat"], "itemcf_feature": True}
    return config


def _stable(value):
    '''Return ``value`` with the keys of VOLATILE removed from every nested dict.'''
    # Drop the volatile keys of a dict and clean its values.
    if isinstance(value, dict):
        return {k: _stable(v) for k, v in value.items() if k not in VOLATILE}
    # Clean every element of a list and return any other value unchanged.
    return [_stable(v) for v in value] if isinstance(value, list) else value


# ---------------------------------------------------------------------------
# constants fixed by PREREG v1.5
# ---------------------------------------------------------------------------
def test_prereg_constants_and_config_default():
    # Expect the fixed margin, levels, list names and combination of the module constants.
    assert v15.DELTA == 0.003 and v15.LEVELS == (0.95, 0.975) and v15.PRIMARY_LEVEL == "0.975" and v15.PREREG == "v1.5"
    assert v15.K == pipeline.K == 10
    assert v15.LIST_SOURCES == {"R1": "repeat_star", "R2": "repeat_lr", "R2'": "repeat_lri", "E2": "usercf",
                                "C3": "final_lr_ucf", "C4": "final_lri_ucf", "PF": "personal_frequency"}
    assert pipeline.V15_COMBOS == {"final_lri_ucf": ("repeat_lri", "usercf")} and ICF_SUFFIX == "_icf"
    # Load the sample configuration of the repository.
    common = load_config("sample")
    # Expect the item-CF feature flag to be off by default and 20 item-CF neighbours.
    assert common["lr_repeat"]["itemcf_feature"] is False and not icf_enabled(common)
    assert not icf_enabled({}) and icf_enabled({"lr_repeat": {"itemcf_feature": True}})
    assert pipeline.icf_neighbor_count(common) == 20


# ---------------------------------------------------------------------------
# tune: R2' and the C4 slot selection
# ---------------------------------------------------------------------------
def test_tune_writes_v15_block_and_icf_artifact(workspace):
    # Run the tuning with the item-CF feature flag on and read the tuning document.
    config = _icf_config(workspace)
    document = json.loads(pipeline.run_tune(config, "instacart").read_text(encoding="utf-8"))
    block = document["v15"]
    # Expect the v15 block with the selected C, the slot count r of C4 and its score grid.
    assert block["label"] == "smoke (sample)" and block["prereg"] == "v1.5" and block["icf_neighbors"] == 5
    assert set(block["selections"]) == {"C", "r"} and set(block["selections"]["r"]) == {"final_lri_ucf"}
    assert block["selections"]["C"] in (0.1, 1.0) and block["lr"]["C"] == block["selections"]["C"]
    assert block["lr"]["C_grid"] == [0.1, 1.0] and block["lr"]["same_validation_users_as_R2"] is True
    assert block["lr"]["n_val_users"] == document["lr_repeat"]["n_val_users"] == 6
    assert block["r_grid"] == list(range(11))
    scores = block["r_scores"]["final_lri_ucf"]
    assert [row["r"] for row in scores] == list(range(11))
    assert block["selections"]["r"]["final_lri_ucf"] == pipeline.select_r({row["r"]: row["ndcg"] for row in scores})
    assert block["at_grid_edge"]["r"]["final_lri_ucf"] == (block["selections"]["r"]["final_lri_ucf"] in (0, 10))
    assert block["r_selection_users"]["final_lri_ucf"] == {"users": "LR validation users (train-only model)", "n_users": 6}
    # R2' summary: 13 features with icf last; R2 (the existing block) keeps its 12 features.
    icf = block["lr_repeat_icf"]
    assert [row["feature"] for row in icf["coefficients"]] == FEATURES + ["icf"] and icf["icf_neighbors"] == 5
    assert [row["feature"] for row in document["lr_repeat"]["coefficients"]] == FEATURES
    assert "icf_neighbors" not in document["lr_repeat"]
    assert [row["C"] for row in icf["validation"]] == [0.1, 1.0] and icf["n_rows"] == document["lr_repeat"]["n_rows"]
    # Expect the stored R2' model file to match the block and the R2 bundle to lack icf_neighbors.
    path = lr_artifact_path(config, "instacart", ICF_SUFFIX)
    assert path == workspace / "artifacts" / "sample" / "instacart_lr_repeat_icf.joblib" and path.exists()
    assert icf["artifact"] == str(path) and icf["sha256"] == sha256_of(path)
    bundle = pipeline.load_lr_repeat(config, "instacart", ICF_SUFFIX)
    assert bundle["icf_neighbors"] == 5 and bundle["features"] == FEATURES + ["icf"] and bundle["C"] == icf["C"]
    assert "icf_neighbors" not in pipeline.load_lr_repeat(config, "instacart")
    assert set(load_v15_selections(config, "instacart")) == {"C", "r", "source"}


def test_tune_v15_selects_on_the_same_validation_users(workspace):
    # Tune R2 and R2' on the fit split of the workspace.
    config = _icf_config(workspace)
    data = pipeline.prepare(config, "instacart", "fit")
    history, target, pop = data["history"], data["target"], data["pop"]
    lr_bundle = pipeline.tune_lr_repeat(history, target, config)
    icf_bundle = pipeline.tune_lr_repeat(history, target, config, icf_neighbors=5)
    # Expect the same validation users for both models and the icf feature only in R2'.
    assert list(icf_bundle["val_users"]) == list(lr_bundle["val_users"]) and icf_bundle["icf_neighbors"] == 5
    assert icf_bundle["features"] == FEATURES + ["icf"] and lr_bundle["features"] == FEATURES
    from nbr.explore import explore_list
    from nbr.repeat import repeat_list

    # Run the integration tuning and keep its lists, then run the v15 tuning.
    kept = {}
    lists = {"repeat_star": repeat_list(history, 0.8, 10, pop), "explore_star": explore_list(history, 10, 1.0, 10.0, "L2")}
    integration = pipeline.tune_integration(history, target, pop, lr_bundle, lists, config, keep_lists=kept)
    assert set(kept) == {"usercf", "repeat_lr"}
    document = {"artifact": "x", "sha256": "y"}
    block = pipeline.tune_v15(history, target, pop, lr_bundle, icf_bundle, kept["usercf"], document, config)
    assert block["lr_repeat_icf"] is document
    # The C4 slot scores are computed on the validation users only with the train-only R2' model.
    val_users = pd.Index(lr_bundle["val_users"])
    repeat = lr_repeat_list(history, pipeline.train_only_bundle(icf_bundle), 10, pop, users=val_users, source=SOURCE_ICF)
    explore = kept["usercf"][kept["usercf"]["user_id"].isin(val_users)]
    val_target = target[target["user_id"].isin(val_users)]
    for row in block["r_scores"]["final_lri_ucf"]:
        expected = float(pipeline.per_user_ndcg(combine(repeat, explore, row["r"], 10), val_target, 10).mean())
        assert row["ndcg"] == pytest.approx(expected)
    # C3's selection is not touched by the v15 selection.
    assert integration["selections"]["r"]["final_lr_ucf"] == pipeline.tune_integration(history, target, pop, lr_bundle, lists, config)["selections"]["r"]["final_lr_ucf"]
    # Expect a ValueError when the two bundles hold different validation users.
    other = dict(icf_bundle, val_users=np.array(sorted(set(lr_bundle["val_users"]) ^ {"u00"})))
    with pytest.raises(ValueError, match="validation users"):
        pipeline.tune_v15(history, target, pop, lr_bundle, other, kept["usercf"], document, config)


def test_tune_without_the_flag_is_unchanged(workspace):
    # Run the tuning with the item-CF feature flag off.
    plain = _config(workspace)
    before = json.loads(pipeline.run_tune(plain, "instacart").read_text(encoding="utf-8"))
    # Expect no v15 block, no R2' model file and an error when the v15 selections are loaded.
    assert "v15" not in before and not lr_artifact_path(plain, "instacart", ICF_SUFFIX).exists()
    with pytest.raises(ValueError, match="v15"):
        load_v15_selections(plain, "instacart")
    # Run the tuning again with the flag on and expect only v15 keys to be added.
    after = json.loads(pipeline.run_tune(_icf_config(workspace), "instacart").read_text(encoding="utf-8"))
    assert set(after) == set(before) | {"v15"}
    for key in set(before) - VOLATILE:
        assert _stable(after[key]) == _stable(before[key]), key
    assert set(after["timing_seconds"]) == set(before["timing_seconds"]) | {"lr_repeat_icf", "v15"}


# ---------------------------------------------------------------------------
# evaluate: lists and the v15 block
# ---------------------------------------------------------------------------
def _loaded(config, dataset="instacart"):
    '''Load the stored selections, R2 bundle and integration selections of ``dataset``.

    The paths come from ``config``; the result is a dict of keyword arguments.
    '''
    return dict(selections=pipeline.load_selections(config, dataset), lr_bundle=pipeline.load_lr_repeat(config, dataset),
                integration=pipeline.load_integration_selections(config, dataset))


def test_c4_lists_are_valid_and_neighbours_are_shared(workspace):
    # Tune with the item-CF feature flag on and load the stored models and selections.
    config = _icf_config(workspace)
    pipeline.run_tune(config, "instacart")
    loaded = _loaded(config)
    icf_bundle = pipeline.load_lr_repeat(config, "instacart", ICF_SUFFIX)
    selections = load_v15_selections(config, "instacart")
    # Build all lists of the eval split.
    data = pipeline.prepare(config, "instacart", "eval")
    lists, timings = pipeline.build_lists(data["history"], data["pop"], loaded["selections"], config, lr_bundle=loaded["lr_bundle"],
                                          integration=loaded["integration"], icf_bundle=icf_bundle, v15=selections)
    # Expect well-formed R2' and C4 lists, and C4 equal to the slot combination of R2' and E2.
    assert {"repeat_lri", "final_lri_ucf"} <= set(lists) and {"repeat_lri", "final_lri_ucf"} <= set(timings)
    users = set(data["history"]["user_id"])
    for name in ("repeat_lri", "final_lri_ucf"):
        frame = lists[name]
        assert list(frame.columns) == ["user_id", "rank", "item_id", "source"] and set(frame["user_id"]) == users
        assert not frame.duplicated(["user_id", "item_id"]).any() and frame.groupby("user_id")["rank"].max().le(10).all()
    assert (lists["repeat_lri"]["source"] == "repeat_lri").all() and (lists["repeat_lr"]["source"] == "repeat_lr").all()
    sizes = lists["final_lri_ucf"].groupby("user_id")["rank"].agg(["size", "max"])
    assert (sizes["size"] == 10).all() and (sizes["max"] == 10).all()
    r = selections["r"]["final_lri_ucf"]
    pd.testing.assert_frame_equal(lists["final_lri_ucf"], combine(lists["repeat_lri"], lists["usercf"], r, 10))
    # Sharing the item-CF neighbours changes nothing: the list equals the one computed with its own neighbours.
    own = lr_repeat_list(data["history"], icf_bundle, 10, data["pop"], source=SOURCE_ICF)
    pd.testing.assert_frame_equal(lists["repeat_lri"], own)
    without_itemcf, _ = pipeline.build_lists(data["history"], data["pop"], loaded["selections"], config, include_itemcf=False,
                                             lr_bundle=loaded["lr_bundle"], integration=loaded["integration"],
                                             icf_bundle=icf_bundle, v15=selections)
    pd.testing.assert_frame_equal(without_itemcf["repeat_lri"], own)
    # R2' ranks differently from R2 for at least one user (the icf column is used), but both are repeat-only lists.
    assert not lists["repeat_lri"].drop(columns="source").equals(lists["repeat_lr"].drop(columns="source"))
    history_pairs = pd.MultiIndex.from_frame(data["history"][["user_id", "item_id"]].drop_duplicates())
    assert pd.MultiIndex.from_frame(lists["repeat_lri"][["user_id", "item_id"]]).isin(history_pairs).all()
    # Expect a ValueError when the R2' bundle is passed without the v15 selections.
    with pytest.raises(ValueError, match="v15"):
        pipeline.build_lists(data["history"], data["pop"], loaded["selections"], config, lr_bundle=loaded["lr_bundle"],
                             integration=loaded["integration"], icf_bundle=icf_bundle)


def test_evaluate_v15_block_structure(workspace):
    # Tune and evaluate with the item-CF feature flag on and read the evaluation document.
    config = _icf_config(workspace)
    pipeline.run_tune(config, "instacart")
    document = json.loads(pipeline.run_evaluate(config, "instacart", skip_l3=True).read_text(encoding="utf-8"))
    block = document["v15"]
    # Expect the fixed settings, the selections and the summary of the R2' model in the block.
    assert block["label"] == "smoke (sample)" and block["prereg"] == "v1.5" and block["delta"] == 0.003 and block["k"] == 10
    assert block["levels"] == ["0.95", "0.975"] and block["primary_level"] == "0.975"
    assert block["lists"] == v15.LIST_SOURCES and block["icf"]["neighbors"] == 5
    assert block["selections"] == {"C": document["v15"]["lr_repeat_icf"]["C"], "r": {"final_lri_ucf": block["selections"]["r"]["final_lri_ucf"]}}
    assert block["selections_source"].endswith("sample_instacart_tuning.json")
    icf = block["lr_repeat_icf"]
    assert icf["features"] == FEATURES + ["icf"] and set(icf["coefficients"]) == set(FEATURES + ["icf"]) and icf["icf_neighbors"] == 5
    assert isinstance(icf["intercept"], float) and icf["sha256"] == sha256_of(lr_artifact_path(config, "instacart", ICF_SUFFIX))
    assert block["n_users"] == document["n_users"]
    # Expect both primary entries with nested intervals, a decision per level and the fixed rules.
    assert set(block["primary"]) == PRIMARY and set(block["secondary"]) == SECONDARY
    for key, entry in block["primary"].items():
        assert set(entry["ci"]) == {"0.95", "0.975"} and set(entry["decision"]) == {"0.95", "0.975"}
        assert entry["diff"] == pytest.approx(entry["model_mean"] - entry["comparator_mean"])
        assert entry["ci"]["0.95"][0] <= entry["diff"] <= entry["ci"]["0.95"][1]
        assert entry["ci"]["0.975"][0] <= entry["ci"]["0.95"][0] and entry["ci"]["0.95"][1] <= entry["ci"]["0.975"][1]
        assert entry["n_users"] == document["n_users"]["all"] and entry["users"] == "all users" and entry["metric"] == "NDCG@10"
    h14, h15 = block["primary"]["H14_ndcg"], block["primary"]["H15_ndcg"]
    assert (h14["comparator"], h14["model"]) == ("final_lr_ucf", "final_lri_ucf") and h14["delta"] == 0.003
    assert h14["rule"] == "non-inferior if the 0.975 lower bound > -0.003"
    assert h14["decision"]["0.975"] == v15.noninferiority_decision(h14["ci"]["0.975"])
    assert h14["decision"]["0.975"] in ("non-inferior", "non-inferiority not established")
    assert (h15["comparator"], h15["model"]) == ("personal_frequency", "final_lri_ucf")
    assert h15["decision"]["0.975"] == v15.improvement_decision(h15["ci"]["0.975"]) and h15["rule"] == "improvement observed if the 0.975 lower bound > 0"
    # Means in the block are the list metrics of the main document.
    assert h14["comparator_mean"] == pytest.approx(document["list_metrics"]["final_lr_ucf"]["ndcg"])
    assert h14["model_mean"] == pytest.approx(document["list_metrics"]["final_lri_ucf"]["ndcg"])
    assert h15["comparator_mean"] == pytest.approx(document["list_metrics"]["personal_frequency"]["ndcg"])
    assert block["list_metrics"]["final_lri_ucf"]["ndcg"] == pytest.approx(h14["model_mean"])
    assert block["list_metrics"]["final_lri_ucf"]["recall"] == pytest.approx(document["list_metrics"]["final_lri_ucf"]["recall"])
    # Expect the secondary entries without a decision and consistent with the main document.
    sec = block["secondary"]
    for entry in sec.values():
        assert "decision" not in entry and set(entry["ci"]) == {"0.95", "0.975"}
    assert (sec["repeat_lri_vs_repeat_lr_repeat_recall"]["comparator"], sec["repeat_lri_vs_repeat_lr_repeat_recall"]["model"]) == ("repeat_lr", "repeat_lri")
    assert sec["repeat_lri_vs_repeat_lr_repeat_recall"]["n_users"] == document["n_users"]["repeat_target"]
    assert sec["repeat_lri_vs_repeat_lr_repeat_recall"]["comparator_mean"] == pytest.approx(document["lr_repeat"]["primary"]["H6_repeat_recall"]["model_mean"])
    assert sec["repeat_lri_vs_repeat_star_repeat_recall"]["comparator_mean"] == pytest.approx(document["lr_repeat"]["primary"]["H6_repeat_recall"]["comparator_mean"])
    assert sec["repeat_lri_vs_repeat_star_repeat_recall"]["model_mean"] == pytest.approx(sec["repeat_lri_vs_repeat_lr_repeat_recall"]["model_mean"])
    assert sec["final_lri_ucf_vs_final_lr_ucf_recall"]["metric"] == "Recall@10" and sec["final_lri_ucf_vs_final_lr_ucf_recall"]["n_users"] == document["n_users"]["all"]
    assert block["list_metrics"]["repeat_lri"]["repeat_recall"] == pytest.approx(sec["repeat_lri_vs_repeat_lr_repeat_recall"]["model_mean"])
    # Expect the history-length rows of the main document and the added timing and memory keys.
    assert [row["bin"] for row in block["by_history_length"]] == [row["bin"] for row in document["by_history_length"]]
    assert set(block["by_history_length"][1]) == {"bin", "n_users", "H14_ndcg"} and block["by_history_length"][0]["H14_ndcg"] is None
    assert block["by_history_length"][1]["H14_ndcg"]["n_users"] == 30 and "ci_0.95" in block["by_history_length"][1]["H14_ndcg"]
    assert {"list_repeat_lri", "list_final_lri_ucf", "v15_metrics"} <= set(document["timing_seconds"])
    assert {"repeat_lri", "final_lri_ucf", "v15"} <= set(document["peak_memory_mib"])


def test_evaluate_other_keys_unchanged_with_and_without_v15(workspace):
    # Tune with the item-CF feature flag on and load the stored selections and R2 bundle.
    config = _icf_config(workspace)
    pipeline.run_tune(config, "instacart")
    loaded = _loaded(config)
    # Evaluate once with and once without the R2' bundle and the v15 selections.
    with_v15 = pipeline.evaluate(config, "instacart", skip_l3=True, icf_bundle=pipeline.load_lr_repeat(config, "instacart", ICF_SUFFIX),
                                 v15=load_v15_selections(config, "instacart"), **loaded)
    without = pipeline.evaluate(config, "instacart", skip_l3=True, **loaded)
    # Expect the v15 run to add only v15 keys and to leave every other stable value equal.
    assert "v15" not in without and set(with_v15) == set(without) | {"v15"}
    for key in set(without) - VOLATILE - {"list_metrics"}:
        assert _stable(with_v15[key]) == _stable(without[key]), key
    assert set(with_v15["list_metrics"]) == set(without["list_metrics"]) | {"repeat_lri", "final_lri_ucf"}
    assert {name: with_v15["list_metrics"][name] for name in without["list_metrics"]} == without["list_metrics"]
    assert set(with_v15["timing_seconds"]) == set(without["timing_seconds"]) | {"list_repeat_lri", "list_final_lri_ucf", "v15_metrics"}
    # Partial arguments are refused.
    with pytest.raises(ValueError, match="v15"):
        pipeline.evaluate(config, "instacart", skip_l3=True, icf_bundle=pipeline.load_lr_repeat(config, "instacart", ICF_SUFFIX), **loaded)
    with pytest.raises(ValueError, match="v15"):
        pipeline.evaluate(config, "instacart", loaded["selections"], lr_bundle=loaded["lr_bundle"],
                          icf_bundle=pipeline.load_lr_repeat(config, "instacart", ICF_SUFFIX), v15=load_v15_selections(config, "instacart"))


def test_run_evaluate_flags_and_transfer(workspace):
    # Tune with the item-CF feature flag on.
    config = _icf_config(workspace)
    pipeline.run_tune(config, "instacart")
    # Expect no v15 block when the flag, the integration lists or the R2 lists are switched off.
    plain = json.loads(pipeline.run_evaluate(_config(workspace), "instacart", skip_l3=True).read_text(encoding="utf-8"))
    assert "v15" not in plain and "repeat_lri" not in plain["list_metrics"]  # flag off: nothing of v1.5 is evaluated
    assert "v15" not in json.loads(pipeline.run_evaluate(config, "instacart", include_integration=False).read_text(encoding="utf-8"))
    assert "v15" not in json.loads(pipeline.run_evaluate(config, "instacart", include_lr=False).read_text(encoding="utf-8"))
    # Expect the v15 block next to the policy block in a full run.
    full = json.loads(pipeline.run_evaluate(config, "instacart", skip_l3=True, include_policy=True).read_text(encoding="utf-8"))
    assert set(full["v15"]["primary"]) == PRIMARY and full["policy"] is not None and full["skipped"] == ["explore_L3"]
    # Expect a complete_journey run with transfer to reuse the instacart selections and C.
    transfer = json.loads(pipeline.run_evaluate(config, "complete_journey", transfer_from="instacart", skip_l3=True).read_text(encoding="utf-8"))
    assert transfer["transfer_from"] == "instacart" and set(transfer["v15"]["primary"]) == PRIMARY
    assert transfer["v15"]["selections"] == full["v15"]["selections"] and transfer["v15"]["lr_repeat_icf"]["C"] == full["v15"]["lr_repeat_icf"]["C"]
    assert transfer["v15"]["selections_source"].endswith("sample_instacart_tuning.json")


def test_evaluate_v15_with_identical_lists_and_perfect_c4(workspace):
    # Tune with the item-CF feature flag on and build all lists of the eval split.
    config = _icf_config(workspace)
    pipeline.run_tune(config, "instacart")
    loaded = _loaded(config)
    icf_bundle = pipeline.load_lr_repeat(config, "instacart", ICF_SUFFIX)
    selections = load_v15_selections(config, "instacart")
    data = pipeline.prepare(config, "instacart", "eval")
    lists, _ = pipeline.build_lists(data["history"], data["pop"], loaded["selections"], config, lr_bundle=loaded["lr_bundle"],
                                    integration=loaded["integration"], icf_bundle=icf_bundle, v15=selections, skip_l3=True)
    lengths = pipeline.history_lengths(data["history"])
    # Replace the R2' list by the R2 list and the C4 list by the C3 list (R2 + E2).
    same = dict(lists, repeat_lri=lists["repeat_lr"].assign(source="repeat_lri"), final_lri_ucf=lists["final_lr_ucf"])
    block = v15.evaluate_v15(same, data, lengths, pipeline.HISTORY_BINS["instacart"], config, icf_bundle, selections)
    # Expect zero differences, zero-width intervals and "non-inferior" for C4 against C3 (H14).
    h14 = block["primary"]["H14_ndcg"]
    assert h14["diff"] == 0.0 and h14["ci"] == {"0.95": [0.0, 0.0], "0.975": [0.0, 0.0]}
    assert h14["decision"] == {"0.95": "non-inferior", "0.975": "non-inferior"}
    assert block["secondary"]["repeat_lri_vs_repeat_lr_repeat_recall"]["diff"] == 0.0
    assert block["secondary"]["final_lri_ucf_vs_final_lr_ucf_recall"]["diff"] == 0.0
    assert all(row["H14_ndcg"] is None or row["H14_ndcg"]["diff"] == 0.0 for row in block["by_history_length"])
    # A margin wider than any drop makes the decision "non-inferior"; a perfect C4 gives diff > 0 and "improvement observed".
    perfect = data["target"][["user_id", "item_id"]].drop_duplicates().copy()
    perfect["rank"] = perfect.groupby("user_id").cumcount() + 1
    perfect["source"] = "final_lri_ucf"
    best = v15.evaluate_v15(dict(lists, final_lri_ucf=perfect[["user_id", "rank", "item_id", "source"]]), data, lengths,
                            pipeline.HISTORY_BINS["instacart"], config, icf_bundle, selections, label="x", delta=1.0)
    assert best["label"] == "x" and best["delta"] == 1.0 and best["primary"]["H14_ndcg"]["rule"] == "non-inferior if the 0.975 lower bound > -1"
    assert best["primary"]["H14_ndcg"]["model_mean"] == pytest.approx(1.0) and best["primary"]["H14_ndcg"]["diff"] > 0
    assert best["primary"]["H15_ndcg"]["decision"]["0.975"] == "improvement observed"
    # Use the popularity list as C4 with margin 0 and expect non-inferiority not established.
    worse = v15.evaluate_v15(dict(lists, final_lri_ucf=lists["popularity"]), data, lengths, pipeline.HISTORY_BINS["instacart"], config,
                             icf_bundle, selections, delta=0.0)
    assert worse["primary"]["H14_ndcg"]["decision"]["0.975"] == "non-inferiority not established"
    # Expect a ValueError when the R2' list is missing.
    with pytest.raises(ValueError, match="missing"):
        v15.evaluate_v15({k: v for k, v in lists.items() if k != "repeat_lri"}, data, lengths, pipeline.HISTORY_BINS["instacart"], config,
                         icf_bundle, selections)
