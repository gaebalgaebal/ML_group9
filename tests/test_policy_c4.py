# *------------------- 2026 Fall Semester Machine Learning Term-project: Team 9 --------------------*
# Code First written at 2026-11-07
# Code Final Edited at 2026-11-13
# Editor: Yejun Kim

'''
Purpose : W19 tests (PREREG v1.5.1): the exploration policy with C4, the two-CF final system,
          as the base (nbr.policy base="C4") and the "policy_c4" block.
Inputs  : The varied_workspace fixture and the key sets of tests.test_policy; _icf_config and
          _stable of tests.test_v15.
Outputs : pytest pass/fail results; tuning and evaluation files are written under tmp_path
          only.
Usage   :
    python -m pytest tests/test_policy_c4.py
'''

# ---------------------------------------------------------------------------
# imports
# ---------------------------------------------------------------------------
import json

import pandas as pd
import pytest

import nbr.pipeline as pipeline
import nbr.policy as policy
from nbr.pipeline import ICF_SUFFIX, load_v15_selections, policy_label
from nbr.policy import BASES, LIST_SOURCES, R_SHORT, T, combine_policy, evaluate_policy
from tests.test_policy import POLICY_PRIMARY, POLICY_SECONDARY, VARIANTS, VOLATILE, varied_workspace  # noqa: F401  (fixture re-used)
from tests.test_v15 import _icf_config, _stable


# ---------------------------------------------------------------------------
# base definitions and labels
# ---------------------------------------------------------------------------
def test_bases():
    # Expect the list names of both bases and the label of each profile and role.
    assert LIST_SOURCES is BASES["C3"] and set(BASES) == {"C3", "C4"}
    assert BASES["C3"] == {"C3": "final_lr_ucf", "repeat": "repeat_lr", "explore": "usercf", "flat": "flat_explore"}
    assert BASES["C4"] == {"C3": "final_lri_ucf", "repeat": "repeat_lri", "explore": "usercf", "flat": "flat_explore"}
    assert policy_label({"profile": "holdout4", "role": "final"}) == "secondary replication on holdout4 (PREREG v1.5.1)"
    assert policy_label({"profile": "sample", "role": "smoke"}) == "smoke (sample)"
    assert policy_label({"profile": "holdout3", "role": "final"}) == "final (holdout3)"


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _lists(config, dataset="instacart"):
    '''Build the eval lists of ``dataset`` for ``config``; return data, lists and selections.'''
    # Load the eval split and the tuned selections.
    data = pipeline.prepare(config, dataset, "eval")
    integration = pipeline.load_integration_selections(config, dataset)
    v15 = load_v15_selections(config, dataset)
    # Build all lists with the two logistic-regression bundles, with skip_l3 set.
    lists, _ = pipeline.build_lists(data["history"], data["pop"], pipeline.load_selections(config, dataset), config,
                                    lr_bundle=pipeline.load_lr_repeat(config, dataset), integration=integration,
                                    icf_bundle=pipeline.load_lr_repeat(config, dataset, ICF_SUFFIX), v15=v15, skip_l3=True)
    return data, lists, integration, v15


# ---------------------------------------------------------------------------
# policy block with the C4 base
# ---------------------------------------------------------------------------
def test_base_c4_block_on_synthetic_data(varied_workspace):
    # Tune, build the lists and evaluate the policy with the C4 base and with the default base.
    config = _icf_config(varied_workspace)
    pipeline.run_tune(config, "instacart")
    data, lists, integration, v15 = _lists(config)
    lengths = pipeline.history_lengths(data["history"])
    r4, r3 = int(v15["r"]["final_lri_ucf"]), int(integration["r"]["final_lr_ucf"])
    edges = pipeline.HISTORY_BINS["instacart"]
    block = evaluate_policy(lists, data, lengths, edges, config, r4, base="C4", label="x")
    base = evaluate_policy(lists, data, lengths, edges, config, r3, label="x")
    # Expect the same block layout with the final_lri_ucf and repeat_lri lists in place.
    assert set(block) == set(base) and block["label"] == "x" and block["r_default"] == r4
    assert block["lists"] == {"P_T": f"combine_policy(repeat_lri, usercf, T=4, r_short=9, r_default={r4})", "C3": "final_lri_ucf",
                              "S": "usercf", "flat": "flat_explore"}
    assert set(block["primary"]) == POLICY_PRIMARY and set(block["variants"]) == VARIANTS
    assert set(block["secondary"]) == {key.replace("final_lr_ucf", "final_lri_ucf") for key in POLICY_SECONDARY}
    h11, h12, h13 = (block["primary"][key] for key in ("H11_ndcg", "H12_new_hit_share", "H13_new_hit_share"))
    assert (h11["comparator"], h11["model"]) == ("final_lri_ucf", "P_T") and (h12["comparator"], h12["model"]) == ("final_lri_ucf", "P_T")
    assert (h13["comparator"], h13["model"]) == ("flat_explore", "usercf")
    assert set(h11["ci"]) == {"0.95", "0.9833"} and set(h11["decision"]) == {"0.95", "0.9833"} and h11["delta"] == 0.003
    # C4 numbers in the block are the final_lri_ucf numbers; S vs flat is shared with the base-C3 block.
    assert h11["comparator_mean"] == pytest.approx(float(pipeline.per_user_ndcg(lists["final_lri_ucf"], data["target"], 10).mean()))
    assert block["list_metrics"]["final_lri_ucf"]["ndcg"] == pytest.approx(h11["comparator_mean"])
    assert h13 == base["primary"]["H13_new_hit_share"]
    assert block["secondary"]["usercf_vs_flat_explore_new_recall"] == base["secondary"]["usercf_vs_flat_explore_new_recall"]
    for entry in block["secondary"].values():
        assert "decision" not in entry
    assert all(entry["model"] == name for name, variant in block["variants"].items() for entry in variant["comparisons"].values())
    # P_T(C4) is built from repeat_lri and usercf: r_short = r_default reproduces C4 exactly, every P_T-vs-C4 entry is zero.
    pd.testing.assert_frame_equal(combine_policy(lists["repeat_lri"], lists["usercf"], lengths, T, r4, r4), lists["final_lri_ucf"])
    same = evaluate_policy(lists, data, lengths, edges, config, r4, base="C4", r_short=r4)
    assert same["primary"]["H11_ndcg"]["diff"] == 0.0 and same["primary"]["H11_ndcg"]["ci"] == {"0.95": [0.0, 0.0], "0.9833": [0.0, 0.0]}
    assert same["primary"]["H11_ndcg"]["decision"] == {"0.95": "non-inferior", "0.9833": "non-inferior"}
    assert same["primary"]["H12_new_hit_share"]["diff"] == 0.0 and same["list_metrics"]["P_T"] == same["list_metrics"]["final_lri_ucf"]
    for entry in same["secondary"].values():
        if entry["model"] == "P_T":
            assert entry["diff"] == 0.0
    # The pre-registered P_T(C4) changes the short-history users only, and uses the R2' list (not R2).
    p_t = combine_policy(lists["repeat_lri"], lists["usercf"], lengths, T, R_SHORT, r4)
    changed = set(pd.concat([p_t, lists["final_lri_ucf"]]).drop_duplicates(keep=False)["user_id"])
    assert changed <= set(lengths[lengths <= T].index)
    assert (p_t[p_t["source"] != "usercf"]["source"] == "repeat_lri").all()
    with pytest.raises(ValueError, match="base"):
        evaluate_policy(lists, data, lengths, edges, config, r4, base="C5")


def test_base_c3_is_unchanged_by_the_parameter(varied_workspace):
    # Evaluate the policy with the default base and with the same base named explicitly.
    config = _icf_config(varied_workspace)
    pipeline.run_tune(config, "instacart")
    data, lists, integration, _ = _lists(config)
    lengths = pipeline.history_lengths(data["history"])
    r3 = int(integration["r"]["final_lr_ucf"])
    default = evaluate_policy(lists, data, lengths, pipeline.HISTORY_BINS["instacart"], config, r3, label="x")
    explicit = evaluate_policy(lists, data, lengths, pipeline.HISTORY_BINS["instacart"], config, r3, label="x", base="C3")
    # Expect equal blocks that use the final_lr_ucf and repeat_lr lists.
    assert _stable(default) == _stable(explicit)
    assert default["lists"]["C3"] == "final_lr_ucf" and default["lists"]["P_T"].startswith("combine_policy(repeat_lr, usercf")


# ---------------------------------------------------------------------------
# policy_c4 block written by run_evaluate
# ---------------------------------------------------------------------------
def test_run_evaluate_policy_c4_block(varied_workspace):
    # Tune and write the evaluation with the policy flag.
    config = _icf_config(varied_workspace)
    pipeline.run_tune(config, "instacart")
    with_c4 = json.loads(pipeline.run_evaluate(config, "instacart", skip_l3=True, include_policy=True).read_text(encoding="utf-8"))
    # Expect a policy_c4 block next to the policy block, each with its own lists and r_default.
    assert set(with_c4["policy_c4"]) == set(with_c4["policy"]) and with_c4["policy"]["label"] == with_c4["policy_c4"]["label"] == "smoke (sample)"
    assert with_c4["policy"]["lists"]["C3"] == "final_lr_ucf" and with_c4["policy_c4"]["lists"]["C3"] == "final_lri_ucf"
    assert with_c4["policy_c4"]["r_default"] == with_c4["v15"]["selections"]["r"]["final_lri_ucf"]
    assert with_c4["policy"]["r_default"] == with_c4["integration"]["selections"]["r"]["final_lr_ucf"]
    assert with_c4["policy_c4"]["primary"]["H11_ndcg"]["comparator_mean"] == pytest.approx(with_c4["list_metrics"]["final_lri_ucf"]["ndcg"])
    assert {"policy_metrics", "policy_c4_metrics"} <= set(with_c4["timing_seconds"]) and "policy_c4" in with_c4["peak_memory_mib"]
    # Without --policy: no policy_c4 key; with the flag off: the policy block is unchanged and policy_c4 is absent.
    no_policy = json.loads(pipeline.run_evaluate(config, "instacart", skip_l3=True).read_text(encoding="utf-8"))
    assert no_policy["policy"] is None and "policy_c4" not in no_policy
    plain = _icf_config(varied_workspace)
    plain["lr_repeat"]["itemcf_feature"] = False
    off = json.loads(pipeline.run_evaluate(plain, "instacart", skip_l3=True, include_policy=True).read_text(encoding="utf-8"))
    assert "policy_c4" not in off and "v15" not in off
    assert _stable(off["policy"]) == _stable(with_c4["policy"])
    for key in set(off) - VOLATILE - {"list_metrics"}:
        assert _stable(off[key]) == _stable(with_c4[key]), key
    transfer = json.loads(pipeline.run_evaluate(config, "complete_journey", transfer_from="instacart", skip_l3=True,
                                                include_policy=True).read_text(encoding="utf-8"))
    assert transfer["policy_c4"]["r_default"] == with_c4["policy_c4"]["r_default"] and transfer["policy_c4"]["lists"]["C3"] == "final_lri_ucf"


# ---------------------------------------------------------------------------
# label of the holdout4 profile
# ---------------------------------------------------------------------------
def test_policy_c4_label_for_holdout4_profile(varied_workspace, monkeypatch):
    # Build the lists, then evaluate the policy under a holdout4 profile with the role final.
    config = _icf_config(varied_workspace)
    pipeline.run_tune(config, "instacart")
    data, lists, integration, v15 = _lists(config)
    lengths = pipeline.history_lengths(data["history"])
    holdout = {**config, "profile": "holdout4", "role": "final"}
    block = evaluate_policy(lists, data, lengths, pipeline.HISTORY_BINS["instacart"], holdout, int(v15["r"]["final_lri_ucf"]),
                            label=policy_label(holdout), base="C4")
    # Expect the secondary-replication label and an unchanged policy.PREREG value.
    assert block["label"] == "secondary replication on holdout4 (PREREG v1.5.1)" and policy.PREREG == "v1.4"
