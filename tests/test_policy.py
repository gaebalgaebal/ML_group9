# *------------------- 2026 Fall Semester Machine Learning Term-project: Team 9 --------------------*
# Code First written at 2026-11-07
# Code Final Edited at 2026-11-13
# Editor: Yejun Kim

'''
Purpose : W17 tests: the PREREG v1.4 exploration-slot policy (nbr.policy) and its wiring into
          the pipeline and the CLI.
Inputs  : Hand-built list frames and synthetic parquet data with varied history lengths
          written to pytest's tmp_path; _config and _frame of tests.test_pipeline.
Outputs : pytest pass/fail results; tuning and evaluation files are written under tmp_path
          only.
Usage   :
    python -m pytest tests/test_policy.py
'''

# ---------------------------------------------------------------------------
# imports
# ---------------------------------------------------------------------------
import json

import numpy as np
import pandas as pd
import pytest

import nbr.pipeline as pipeline
import nbr.policy as policy
from nbr.combine import combine
from nbr.policy import (
    DELTA,
    LEVELS,
    R_SHORT,
    T,
    combine_policy,
    evaluate_policy,
    improvement_decision,
    new_hit_share,
    noninferiority_decision,
    policy_name,
    slot_counts,
)
from tests.test_pipeline import _config, _frame

# ---------------------------------------------------------------------------
# constants
# ---------------------------------------------------------------------------
COLUMNS = ["user_id", "rank", "item_id", "source"]
POLICY_PRIMARY = {"H11_ndcg", "H12_new_hit_share", "H13_new_hit_share"}
POLICY_SECONDARY = {"P_T_vs_final_lr_ucf_recall", "P_T_vs_final_lr_ucf_new_recall", "P_T_vs_final_lr_ucf_repeat_recall",
                    "P_T_vs_final_lr_ucf_new_hit_share_new_users", "usercf_vs_flat_explore_new_recall"}
VARIANTS = {"P_T4_r8", "P_T6_r9", "P_T10_r9"}
VOLATILE = {"timing_seconds", "peak_memory_mib", "process_peak_working_set_mib", "utc_time"}


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _lists(rows, source):
    '''Build a list frame from ``rows`` of (user, rank, item) with the constant ``source``.'''
    frame = pd.DataFrame(rows, columns=["user_id", "rank", "item_id"])
    frame["source"] = source
    return frame[COLUMNS]


def _repeat_explore(users, n_repeat=10, n_explore=10):
    '''Return repeat and explore lists for ``users``, ``n_repeat`` / ``n_explore`` items each.'''
    # Name every item after its user, its list and its rank.
    repeat = _lists([(u, i, f"{u}r{i}") for u in users for i in range(1, n_repeat + 1)], "repeat_lr")
    explore = _lists([(u, i, f"{u}e{i}") for u in users for i in range(1, n_explore + 1)], "usercf")
    return repeat, explore


# ---------------------------------------------------------------------------
# constants fixed by PREREG v1.4
# ---------------------------------------------------------------------------
def test_prereg_constants():
    # Expect the pre-registered constants, the policy list names and the pipeline's bin labels.
    assert (T, DELTA, R_SHORT) == (4, 0.003, 9)
    assert LEVELS == (0.95, 0.9833) and policy.PRIMARY_LEVEL == "0.9833"
    assert policy.K == pipeline.K == 10
    assert policy.T_SENSITIVITY == (6, 10) and policy.R_SHORT_VARIANT == 8
    assert policy_name(4, 9) == "P_T" and policy_name(4, 8) == "P_T4_r8" and policy_name(6, 9) == "P_T6_r9"
    for dataset, edges in pipeline.HISTORY_BINS.items():
        assert policy.bin_labels(edges) == pipeline._bin_labels(edges), dataset


# ---------------------------------------------------------------------------
# lists
# ---------------------------------------------------------------------------
def test_slot_counts_threshold_is_inclusive():
    # Build history lengths below, at and above the threshold.
    lengths = pd.Series({"a": 1, "b": 4, "c": 5, "d": 30})
    counts = slot_counts(lengths, 4, 9, 10)
    # Expect the short slot count for lengths up to and including the threshold.
    assert counts.to_dict() == {"a": 9, "b": 9, "c": 10, "d": 10}
    assert slot_counts(lengths, 0, 9, 10).to_dict() == {"a": 10, "b": 10, "c": 10, "d": 10}
    assert slot_counts(lengths, 30, 8, 10).to_dict() == {"a": 8, "b": 8, "c": 8, "d": 8}


def test_combine_policy_equal_slots_reproduces_combine_exactly():
    # Build lists for six users, one of them with only three repeat candidates.
    users = [f"u{i}" for i in range(6)]
    repeat, explore = _repeat_explore(users)
    repeat = repeat[~((repeat["user_id"] == "u2") & (repeat["rank"] > 3))]  # u2 has only 3 repeat candidates
    lengths = pd.Series([1, 4, 5, 2, 30, 4], index=users)
    # Expect the output of combine when the short and the default slot counts are equal.
    for r in (10, 9, 5, 0):
        pd.testing.assert_frame_equal(combine_policy(repeat, explore, lengths, 4, r, r), combine(repeat, explore, r, 10))
    # C3 reproduction: r_short = r_default = r* gives the C3 list whatever T is.
    pd.testing.assert_frame_equal(combine_policy(repeat, explore, lengths, 1000, 10, 10), combine(repeat, explore, 10, 10))


def test_combine_policy_gives_short_users_one_exploration_slot():
    # Build four users with history lengths 3, 4, 5 and 12 and apply T = 4 with r_short = 9.
    users = ["a", "b", "c", "d"]
    repeat, explore = _repeat_explore(users)
    lengths = pd.Series([3, 4, 5, 12], index=users)
    out = combine_policy(repeat, explore, lengths, 4, 9, 10)
    # Expect one explore item for the two short users and the repeat-only list for the others.
    assert list(out.columns) == COLUMNS and out["rank"].tolist() == [*range(1, 11)] * 4
    assert out["user_id"].tolist() == sorted(out["user_id"].tolist())
    by_user = {u: g for u, g in out.groupby("user_id")}
    for user in ("a", "b"):  # history_len <= T: nine repeat items, then the first explore item
        assert by_user[user]["item_id"].tolist() == [f"{user}r{i}" for i in range(1, 10)] + [f"{user}e1"]
        assert by_user[user]["source"].tolist() == ["repeat_lr"] * 9 + ["usercf"]
    for user in ("c", "d"):  # longer histories: the C3 list (repeat only)
        assert by_user[user]["item_id"].tolist() == [f"{user}r{i}" for i in range(1, 11)]
    reference = combine(repeat, explore, 10, 10)
    pd.testing.assert_frame_equal(out[out["user_id"].isin(["c", "d"])].reset_index(drop=True),
                                  reference[reference["user_id"].isin(["c", "d"])].reset_index(drop=True))
    short = combine(repeat[repeat["user_id"].isin(["a", "b"])], explore[explore["user_id"].isin(["a", "b"])], 9, 10)
    pd.testing.assert_frame_equal(out[out["user_id"].isin(["a", "b"])].reset_index(drop=True), short)
    # r_short = 8: two exploration slots for the short users.
    eight = combine_policy(repeat, explore, lengths, 4, 8, 10)
    assert eight[eight["user_id"] == "a"]["item_id"].tolist()[-2:] == ["ae1", "ae2"]
    assert eight[eight["user_id"] == "c"]["item_id"].tolist() == [f"cr{i}" for i in range(1, 11)]


def test_combine_policy_fills_short_repeat_lists_and_rejects_unknown_users():
    # Build two users with only four repeat candidates each.
    repeat, explore = _repeat_explore(["a", "b"], n_repeat=4)
    lengths = pd.Series([2, 9], index=["a", "b"])
    out = combine_policy(repeat, explore, lengths, 4, 9, 10)
    # Expect explore items to fill the list and ValueError for a missing length, 11 slots or T = -1.
    assert out[out["user_id"] == "a"]["item_id"].tolist() == ["ar1", "ar2", "ar3", "ar4"] + [f"ae{i}" for i in range(1, 7)]
    assert out[out["user_id"] == "b"]["item_id"].tolist() == ["br1", "br2", "br3", "br4"] + [f"be{i}" for i in range(1, 7)]
    with pytest.raises(ValueError, match="history length"):
        combine_policy(repeat, explore, lengths.drop("b"), 4, 9, 10)
    with pytest.raises(ValueError):
        combine_policy(repeat, explore, lengths, 4, 11, 10)
    with pytest.raises(ValueError):
        combine_policy(repeat, explore, lengths, -1, 9, 10)


# ---------------------------------------------------------------------------
# metric and decisions
# ---------------------------------------------------------------------------
def test_new_hit_share_over_all_users_and_new_users():
    # Build lists with one new-target hit in the top 10 and one at rank 11.
    lists = _lists([("a", 1, "x"), ("a", 2, "n1"), ("b", 1, "y"), ("c", 1, "n3"), ("c", 11, "n4")], "final")
    new_target = pd.DataFrame({"user_id": ["a", "b", "c", "c"], "item_id": ["n1", "n2", "n4", "n5"]})
    all_users = pd.Index(["a", "b", "c", "d"])  # d has no new target item
    # Expect 1 for a user with a new-target hit within the cutoff and 0 otherwise.
    share = new_hit_share(lists, new_target, all_users, 10)
    assert share.index.tolist() == ["a", "b", "c", "d"] and share.tolist() == [1.0, 0.0, 0.0, 0.0]  # c's hit is at rank 11
    new_users = pd.Index(["a", "b", "c"])
    assert new_hit_share(lists, new_target, new_users, 10).tolist() == [1.0, 0.0, 0.0]
    assert new_hit_share(lists, new_target, new_users, 11).tolist() == [1.0, 0.0, 1.0]
    assert new_hit_share(lists, new_target, all_users).mean() == pytest.approx(0.25)


def test_decision_wording():
    # Expect "non-inferior" only when the lower bound is above minus delta.
    assert noninferiority_decision([-0.0029, 0.001]) == "non-inferior"
    assert noninferiority_decision([-0.003, 0.001]) == "non-inferiority not established"
    assert noninferiority_decision([-0.01, -0.004]) == "non-inferiority not established"
    assert noninferiority_decision([0.001, 0.002]) == "non-inferior"
    assert noninferiority_decision([-0.0049, 0.0], delta=0.005) == "non-inferior"
    # Expect improvement_decision to give the same wording as pipeline.decision.
    grid = [[-0.05, -0.01], [-0.01, 0.05], [0.0, 0.05], [0.01, 0.05], [-0.05, 0.0]]
    assert [improvement_decision(ci) for ci in grid] == [pipeline.decision(ci) for ci in grid]
    assert improvement_decision([0.01, 0.05]) == "improvement observed" and improvement_decision([-0.05, -0.01]) == "lower"


def test_compare_levels_and_decisions():
    # Compare a series with a shifted copy and with an identical copy.
    config = {"bootstrap": {"n": 200, "seed": 42}}
    a = pd.Series([0.1, 0.2, 0.3, 0.4], index=list("abcd"))
    entry = policy.compare(a, a + 0.5, config, decision=improvement_decision)
    # Expect both levels, and decisions only when a decision function is passed.
    assert set(entry["ci"]) == {"0.95", "0.9833"} and set(entry["decision"]) == {"0.95", "0.9833"}
    assert entry["diff"] == pytest.approx(0.5) and entry["decision"]["0.9833"] == "improvement observed"
    assert "decision" not in policy.compare(a, a + 0.5, config)
    same = policy.compare(a, a.copy(), config, decision=noninferiority_decision)
    assert same["diff"] == 0.0 and same["ci"]["0.9833"] == [0.0, 0.0] and same["decision"]["0.9833"] == "non-inferior"


# ---------------------------------------------------------------------------
# synthetic data with varied history lengths (some users at or below T = 4)
# ---------------------------------------------------------------------------
def _varied(rng, n_users=36, n_items=40, basket_size=4):
    '''Draw history and target rows for ``n_users`` users with 2 to 8 history baskets each.

    ``rng`` is the generator; a basket holds ``basket_size`` of the ``n_items`` items.
    '''
    # Name the items and give them popularity weights that fall with the item index.
    items = [f"i{j:02d}" for j in range(n_items)]
    weights = 1.0 / (np.arange(n_items) + 1.0)
    weights /= weights.sum()
    history, target = [], []
    for u in range(n_users):
        user = f"u{u:02d}"
        n_baskets = 2 + u % 7  # 2..8 history baskets: 16 users with <= 4, 10 with 5-6, 10 with 7-8
        # Draw the history baskets of the user without repeated items inside a basket.
        for s in range(1, n_baskets + 1):
            chosen = rng.choice(items, size=basket_size, replace=False, p=weights)
            history += [(user, s, item) for item in chosen]
        # Build the target basket from two history items and two unseen items.
        seen = sorted({i for uu, _, i in history if uu == user})
        repeat = list(rng.choice(seen, size=2, replace=False))
        unseen = [item for item in items if item not in seen]
        fresh = list(rng.choice(unseen, size=2, replace=False))
        target += [(user, n_baskets + 1, item) for item in repeat + fresh]
    # Return the history rows, the target rows and the item names.
    return history, target, items


def _write_varied(root, dataset, rng):
    '''Write the sample-profile parquet files of ``dataset`` under ``root`` using ``rng``.'''
    # Create the dataset folder.
    folder = root / "sample" / dataset
    folder.mkdir(parents=True, exist_ok=True)
    # Draw the rows and write the same history and target for the fit and eval splits.
    history, target, items = _varied(rng)
    for split in ("fit", "eval"):
        _frame(history, dataset).to_parquet(folder / f"{split}_history.parquet", index=False)
        _frame(target, dataset).to_parquet(folder / f"{split}_target.parquet", index=False)
    # Write an item catalogue with three category columns.
    pd.DataFrame({"item_id": items, "dept": [f"d{j % 2}" for j in range(len(items))], "aisle": [f"a{j % 5}" for j in range(len(items))],
                  "leaf": [f"l{j % 8}" for j in range(len(items))]}).to_parquet(folder / "items.parquet", index=False)


@pytest.fixture
def varied_workspace(tmp_path):
    '''Write both varied synthetic datasets under ``tmp_path`` and return it.'''
    # Seed the generator so that every run writes the same data.
    rng = np.random.default_rng(3)
    _write_varied(tmp_path / "data", "instacart", rng)
    _write_varied(tmp_path / "data", "complete_journey", rng)
    return tmp_path


def _evaluate(config, dataset="instacart", **kwargs):
    '''Run pipeline.evaluate for ``config`` / ``dataset`` with all tuned inputs plus ``kwargs``.'''
    # Load the tuned selections.
    selections = pipeline.load_selections(config, dataset)
    # Evaluate with the LR bundle and the integration selections, with skip_l3 set.
    return pipeline.evaluate(config, dataset, selections, lr_bundle=pipeline.load_lr_repeat(config, dataset),
                             integration=pipeline.load_integration_selections(config, dataset), skip_l3=True, **kwargs)


# ---------------------------------------------------------------------------
# policy block in the pipeline
# ---------------------------------------------------------------------------
def test_policy_block_structure_and_c3_identity(varied_workspace):
    # Tune and evaluate the varied data with the policy block.
    config = _config(varied_workspace)
    pipeline.run_tune(config, "instacart")
    document = _evaluate(config, include_policy=True)
    block = document["policy"]
    # Expect the pre-registered settings, user counts, comparisons, bins and variants.
    assert block["label"] == "smoke (sample)" and block["prereg"] == "v1.4"
    assert (block["T"], block["delta"], block["r_short"], block["k"]) == (4, 0.003, 9, 10)
    assert block["r_default"] == document["integration"]["selections"]["r"]["final_lr_ucf"]
    assert block["levels"] == ["0.95", "0.9833"] and block["primary_level"] == "0.9833"
    assert block["lists"] == {"P_T": f"combine_policy(repeat_lr, usercf, T=4, r_short=9, r_default={block['r_default']})",
                              "C3": "final_lr_ucf", "S": "usercf", "flat": "flat_explore"}
    n = block["n_users"]
    assert n["all"] == document["n_users"]["all"] == 36 and n["new_target"] == document["n_users"]["new_target"]
    assert n["repeat_target"] == document["n_users"]["repeat_target"]
    assert n["short_history"] == 16 and n["short_history_by_T"] == {"4": 16, "6": 26, "10": 36}
    assert set(block["primary"]) == POLICY_PRIMARY and set(block["secondary"]) == POLICY_SECONDARY
    assert set(block["variants"]) == VARIANTS
    for key, entry in block["primary"].items():
        assert set(entry["ci"]) == {"0.95", "0.9833"} and set(entry["decision"]) == {"0.95", "0.9833"}
        assert entry["diff"] == pytest.approx(entry["model_mean"] - entry["comparator_mean"])
        assert entry["ci"]["0.95"][0] <= entry["diff"] <= entry["ci"]["0.95"][1]
        assert entry["ci"]["0.9833"][0] <= entry["ci"]["0.95"][0] and entry["ci"]["0.95"][1] <= entry["ci"]["0.9833"][1]
    h11, h12, h13 = (block["primary"][key] for key in ("H11_ndcg", "H12_new_hit_share", "H13_new_hit_share"))
    assert (h11["comparator"], h11["model"], h11["metric"], h11["users"]) == ("final_lr_ucf", "P_T", "NDCG@10", "all users")
    assert h11["n_users"] == 36 and h11["delta"] == 0.003 and h11["decision"]["0.9833"] == noninferiority_decision(h11["ci"]["0.9833"])
    assert h11["decision"]["0.9833"] in ("non-inferior", "non-inferiority not established")
    assert (h12["comparator"], h12["model"], h12["users"]) == ("final_lr_ucf", "P_T", "all users") and h12["n_users"] == 36
    assert h12["decision"]["0.9833"] == improvement_decision(h12["ci"]["0.9833"])
    assert (h13["comparator"], h13["model"], h13["users"]) == ("flat_explore", "usercf", "users with new target")
    assert h13["n_users"] == n["new_target"] and h13["decision"]["0.95"] == improvement_decision(h13["ci"]["0.95"])
    # C3 numbers in the block are the final_lr_ucf numbers of the main document.
    assert h11["comparator_mean"] == pytest.approx(document["list_metrics"]["final_lr_ucf"]["ndcg"])
    assert block["list_metrics"]["final_lr_ucf"]["recall"] == pytest.approx(document["list_metrics"]["final_lr_ucf"]["recall"])
    # H13 is the new-hit-share version of H8 (same lists, same users); its new-item Recall@10 secondary equals H8 exactly.
    h8 = document["integration"]["primary"]["H8_new_recall"]
    s_recall = block["secondary"]["usercf_vs_flat_explore_new_recall"]
    assert (s_recall["diff"], s_recall["n_users"]) == (h8["diff"], h8["n_users"]) and s_recall["ci"]["0.95"] == h8["ci"]["0.95"]
    for entry in block["secondary"].values():
        assert "decision" not in entry and set(entry["ci"]) == {"0.95", "0.9833"}
    assert [row["bin"] for row in block["by_history_length"]] == ["(0,4]", "(4,6]", "(6,10]", "(10,20]", "(20,inf)"]
    assert [row["n_users"] for row in block["by_history_length"]] == [16, 10, 10, 0, 0]
    assert set(block["by_history_length"][0]) == {"bin", "n_users"} | POLICY_PRIMARY | (POLICY_SECONDARY - {"usercf_vs_flat_explore_new_recall"})
    assert block["by_history_length"][3]["H11_ndcg"] is None
    # Users above T get the C3 list: zero difference in the (4,6] and (6,10] bins, every P_T-vs-C3 difference comes from (0,4].
    for row in block["by_history_length"][1:3]:
        assert row["H11_ndcg"]["diff"] == 0.0 and row["H11_ndcg"]["ci_0.95"] == [0.0, 0.0] and row["H12_new_hit_share"]["diff"] == 0.0
    short = block["by_history_length"][0]["H11_ndcg"]
    assert h11["diff"] == pytest.approx(short["diff"] * short["n_users"] / 36)
    for name, (t_value, r_value) in {"P_T4_r8": (4, 8), "P_T6_r9": (6, 9), "P_T10_r9": (10, 9)}.items():
        variant = block["variants"][name]
        assert (variant["T"], variant["r_short"], variant["r_default"]) == (t_value, r_value, block["r_default"])
        assert variant["n_short_users"] == n["short_history_by_T"][str(t_value)]
        assert set(variant["comparisons"]) == {"ndcg", "new_hit_share", "recall", "new_recall", "repeat_recall", "new_hit_share_new_users"}
        assert all("decision" not in entry and entry["model"] == name for entry in variant["comparisons"].values())
        assert block["list_metrics"][name]["ndcg"] == pytest.approx(variant["comparisons"]["ndcg"]["model_mean"])
    # The T = 10 variant covers every user (all have <= 8 baskets): one exploration slot for everyone.
    assert block["variants"]["P_T10_r9"]["n_short_users"] == 36
    assert "policy_metrics" in document["timing_seconds"] and "policy" in document["peak_memory_mib"]


def test_policy_with_equal_slots_reproduces_c3_and_other_keys_unchanged(varied_workspace):
    # Tune and evaluate once with and once without the policy block.
    config = _config(varied_workspace)
    pipeline.run_tune(config, "instacart")
    with_policy = _evaluate(config, include_policy=True)
    without = _evaluate(config)
    # Expect the same document apart from the policy block and the volatile fields.
    assert without["policy"] is None
    assert set(with_policy) == set(without)

    def stable(value):  # drop timing / memory fields anywhere in the document
        '''Return ``value`` without the VOLATILE keys at any depth of dicts and lists.'''
        if isinstance(value, dict):
            return {k: stable(v) for k, v in value.items() if k not in VOLATILE}
        return [stable(v) for v in value] if isinstance(value, list) else value

    for key in set(without) - {"policy"} - VOLATILE:
        assert stable(with_policy[key]) == stable(without[key]), key
    # r_short = r_default: P_T is C3, every P_T-vs-C3 entry is exactly zero.
    data = pipeline.prepare(config, "instacart", "eval")
    integration = pipeline.load_integration_selections(config, "instacart")
    lists, _ = pipeline.build_lists(data["history"], data["pop"], pipeline.load_selections(config, "instacart"), config,
                                    lr_bundle=pipeline.load_lr_repeat(config, "instacart"), integration=integration, skip_l3=True)
    r_star = int(integration["r"]["final_lr_ucf"])
    lengths = pipeline.history_lengths(data["history"])
    pd.testing.assert_frame_equal(combine_policy(lists["repeat_lr"], lists["usercf"], lengths, T, r_star, r_star), lists["final_lr_ucf"])
    block = evaluate_policy(lists, data, lengths, pipeline.HISTORY_BINS["instacart"], config, r_star, r_short=r_star)
    h11 = block["primary"]["H11_ndcg"]
    assert h11["diff"] == 0.0 and h11["ci"] == {"0.95": [0.0, 0.0], "0.9833": [0.0, 0.0]}
    assert h11["decision"] == {"0.95": "non-inferior", "0.9833": "non-inferior"}
    assert block["primary"]["H12_new_hit_share"]["decision"]["0.9833"] == "no difference established"
    for entry in block["secondary"].values():
        if entry["model"] == "P_T":
            assert entry["diff"] == 0.0
    assert block["list_metrics"]["P_T"] == block["list_metrics"]["final_lr_ucf"]
    # The pre-registered P_T differs from C3 for the short-history users only.
    p_t = combine_policy(lists["repeat_lr"], lists["usercf"], lengths, T, R_SHORT, r_star)
    short_users = set(lengths[lengths <= T].index)
    changed = set(pd.concat([p_t, lists["final_lr_ucf"]]).drop_duplicates(keep=False)["user_id"])
    assert changed <= short_users


def test_policy_requires_lr_and_integration(varied_workspace):
    # Tune and load the selections.
    config = _config(varied_workspace)
    pipeline.run_tune(config, "instacart")
    selections = pipeline.load_selections(config, "instacart")
    # Expect ValueError when the policy block is requested without LR or integration inputs.
    with pytest.raises(ValueError, match="policy block"):
        pipeline.evaluate(config, "instacart", selections, include_policy=True)
    with pytest.raises(ValueError, match="policy block"):
        pipeline.run_evaluate(config, "instacart", include_integration=False, include_policy=True)
    with pytest.raises(ValueError, match="policy block"):
        pipeline.run_evaluate(config, "instacart", include_lr=False, include_policy=True)


def test_run_evaluate_writes_policy_block_and_transfer(varied_workspace):
    # Tune and write the evaluation with the policy block.
    config = _config(varied_workspace)
    pipeline.run_tune(config, "instacart")
    path = pipeline.run_evaluate(config, "instacart", skip_l3=True, include_policy=True)
    document = json.loads(path.read_text(encoding="utf-8"))
    # Expect the policy block only with the flag, also in the transfer evaluation.
    assert set(document["policy"]["primary"]) == POLICY_PRIMARY and document["skipped"] == ["explore_L3"]
    plain = json.loads(pipeline.run_evaluate(config, "instacart", skip_l3=True).read_text(encoding="utf-8"))
    assert plain["policy"] is None
    transfer = json.loads(pipeline.run_evaluate(config, "complete_journey", transfer_from="instacart", skip_l3=True,
                                                include_policy=True).read_text(encoding="utf-8"))
    assert transfer["transfer_from"] == "instacart" and set(transfer["policy"]["primary"]) == POLICY_PRIMARY
    assert transfer["policy"]["r_default"] == document["policy"]["r_default"]


# ---------------------------------------------------------------------------
# command-line flag
# ---------------------------------------------------------------------------
def test_cli_policy_flag(monkeypatch):
    import nbr.__main__ as cli

    # Replace load_config by a stub and run_evaluate by a stub that records its calls.
    calls = []
    monkeypatch.setattr(cli, "load_config", lambda profile: {"profile": profile})
    monkeypatch.setattr(pipeline, "run_evaluate", lambda config, dataset, **kwargs: calls.append((config, dataset, kwargs)) or "out.json")
    cli.main(["evaluate", "--profile", "sample", "--dataset", "instacart", "--skip-l3", "--policy"])
    cli.main(["evaluate", "--profile", "sample", "--dataset", "instacart"])
    # Expect include_policy to follow the --policy flag.
    assert calls[0][2]["include_policy"] is True and calls[0][2]["skip_l3"] is True
    assert calls[1][2]["include_policy"] is False
    assert calls[0][2]["allow_holdout_eval"] is False
