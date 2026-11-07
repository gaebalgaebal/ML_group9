"""Tests for scripts/posthoc_explore_category.py: argument guard, events, category tables, bootstrap, rankings, predictability,
quadrants, mismatch, robustness helpers, an end-to-end run on synthetic data and (when the development data is present) on the sample profile."""

import importlib.util
import json
import math
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import nbr.pipeline as pipeline
from nbr.config import REPO_ROOT


def _load_script():
    path = REPO_ROOT / "scripts" / "posthoc_explore_category.py"
    spec = importlib.util.spec_from_file_location("posthoc_explore_category", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


pec = _load_script()
COLUMNS = ["dataset", "user_id", "basket_id", "basket_seq", "item_id", "coarse_category", "fine_category", "quantity"]
CATEGORIES = {"x": ("dairy", "milk"), "y": ("dairy", "cheese"), "z": ("produce", "apples"), "w": ("produce", "bananas"), "q": ("missing", "missing"),
              "a": ("snacks", "chips"), "n1": ("snacks", "cookies"), "n2": ("dairy", "yogurt"), "m": ("produce", "apples")}


def _frame(rows, dataset="instacart"):
    """rows = (user, seq, item); basket_id = f"{user}-{seq}"; L1 / L2 from CATEGORIES."""
    frame = pd.DataFrame([(dataset, u, f"{u}-{s}", s, i, CATEGORIES[i][0], CATEGORIES[i][1], 1) for u, s, i in rows], columns=COLUMNS)
    frame = frame.astype({"user_id": "string", "basket_id": "string", "basket_seq": "int32", "item_id": "string"})
    frame["L1"], frame["L2"] = frame["coarse_category"], frame["fine_category"]
    return frame


def _lists(rows, source="usercf"):
    frame = pd.DataFrame(rows, columns=["user_id", "rank", "item_id"])
    frame["source"] = source
    frame = frame.astype({"user_id": "string", "item_id": "string", "rank": "int64"})
    frame["L1"] = [CATEGORIES[i][0] for i in frame["item_id"]]
    frame["L2"] = [CATEGORIES[i][1] for i in frame["item_id"]]
    return frame


# ---------------------------------------------------------------------------
# arguments and helpers
# ---------------------------------------------------------------------------
def test_parse_args_defaults_and_guards(tmp_path):
    args = pec.parse_args([])
    assert args.profile == "medium" and args.datasets == ["instacart", "complete_journey"] and args.usercf_k is None
    assert args.reference_tuning_dir.endswith(os.path.join("dev_ML-wt-w8", "reports", "metrics")) and (args.bootstrap_n, args.seed) == (1000, 42)
    for argv in (["--profile", "holdout"], ["--profile", "holdout3"], ["--datasets", "holdout"], ["--usercf-k", "5"], ["--usercf-weighting", "binary"]):
        with pytest.raises(SystemExit):
            pec.parse_args(argv)
    for bad in ("final", "final3", "holdout", "holdout3"):
        with pytest.raises(PermissionError):
            pec.guard_path(tmp_path / "reports" / bad / "x.json")
    assert pec.is_missing("missing") and pec.is_missing(" Missing ") and not pec.is_missing("produce")
    for dataset, edges in pipeline.HISTORY_BINS.items():
        assert pec.bin_labels(edges) == pipeline._bin_labels(edges), dataset


def test_usercf_selection_reads_verifies_and_overrides(tmp_path):
    folder = tmp_path / "metrics"
    folder.mkdir()
    document = {"meta": {"git_commit": "abc"}, "integration": {"selections": {"k": 100, "weighting": "binary"}}}
    (folder / "medium_instacart_tuning.json").write_text(json.dumps(document), encoding="utf-8")
    result = pec.usercf_selection("medium", "instacart", folder, None)
    assert (result["k"], result["weighting"], result["verified"], result["source_commit"]) == (100, "binary", True, "abc")
    assert result["expected"] == [100, "binary"] and len(result["source_sha256"]) == 64
    document["integration"]["selections"]["k"] = 50
    (folder / "medium_instacart_tuning.json").write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(SystemExit, match="expected"):
        pec.usercf_selection("medium", "instacart", folder, None)
    (folder / "sample_instacart_tuning.json").write_text(json.dumps(document), encoding="utf-8")
    assert pec.usercf_selection("sample", "instacart", folder, None)["verified"] is None            # no expectation for sample
    assert pec.usercf_selection("medium", "instacart", None, (7, "log1p")) == {"k": 7, "weighting": "log1p", "source": "override", "verified": None}
    with pytest.raises(PermissionError):
        pec.usercf_selection("medium", "instacart", tmp_path / "final2", None)


# ---------------------------------------------------------------------------
# events and category tables
# ---------------------------------------------------------------------------
def _split():
    history = _frame([("A", 1, "x"), ("A", 1, "y"), ("A", 2, "x"), ("A", 2, "z"), ("A", 3, "z"), ("A", 3, "w"), ("A", 3, "w"),
                      ("B", 1, "a"), ("B", 2, "q"), ("B", 2, "a")])
    target = _frame([("A", 4, "x"), ("A", 4, "n1"), ("B", 3, "a"), ("B", 3, "n2")])
    return history, target


def test_build_events_hand_checked():
    history, target = _split()
    events, rows = pec.build_events(history, target)
    assert len(rows) == 13 and rows.duplicated(["user_id", "basket_seq", "item_id"]).sum() == 0     # w deduplicated
    assert len(events) == 10                                                                          # first baskets A-1 (2 items), B-1 (1) dropped
    by_key = events.set_index(["user_id", "basket_seq", "item_id"])
    assert by_key.loc[("A", 2, "x"), "is_new"] == False and by_key.loc[("A", 2, "z"), "is_new"] == True  # noqa: E712
    assert by_key.loc[("A", 3, "z"), "is_new"] == False and by_key.loc[("A", 3, "w"), "is_new"] == True  # noqa: E712
    assert by_key.loc[("A", 4, "n1"), "is_new"] == True and by_key.loc[("A", 4, "x"), "is_new"] == False  # noqa: E712
    assert by_key.loc[("B", 2, "q"), "is_new"] == True and by_key.loc[("B", 3, "n2"), "is_new"] == True  # noqa: E712
    # category novelty: z (produce/apples) in A-2 is a new item in a category new to A; w (produce/bananas) in A-3: produce known, bananas new
    assert by_key.loc[("A", 2, "z"), "known_L1"] == False and by_key.loc[("A", 3, "w"), "known_L1"] == True  # noqa: E712
    assert by_key.loc[("A", 3, "w"), "known_L2"] == False and by_key.loc[("A", 4, "n1"), "known_L1"] == False  # noqa: E712
    assert by_key.loc[("B", 3, "n2"), "known_L1"] == False and by_key.loc[("A", 4, "x"), "known_L2"] == True  # noqa: E712
    assert events["is_target"].sum() == 4


def test_category_table_values():
    history, target = _split()
    events, rows = pec.build_events(history, target)
    users = pd.Index(["A", "B"])
    table = pec.category_table(events, rows, "L1", users, n_boot=0, seed=0).set_index("category")
    # events from the 2nd basket on: A: x,z | z,w | x,n1 ; B: q,a | a,n2  (10 events, 5 new: z, w, n1, q, n2)
    assert table.loc["dairy", "n_events"] == 3 and table.loc["dairy", "n_new"] == 1 and table.loc["dairy", "rate"] == pytest.approx(1 / 3)
    assert table.loc["produce", "n_events"] == 3 and table.loc["produce", "n_new"] == 2 and table.loc["produce", "rate"] == pytest.approx(2 / 3)
    assert table.loc["snacks", "n_events"] == 3 and table.loc["snacks", "n_new"] == 1
    assert table.loc["missing", "n_new"] == 1 and bool(table.loc["missing", "is_missing"]) is True and not table.loc["produce", "is_missing"]
    total_new, total_events = 5, 10
    assert table["n_new"].sum() == total_new and table["n_events"].sum() == total_events
    assert table.loc["produce", "share_new_events"] == pytest.approx(2 / total_new) and table.loc["produce", "share_events"] == pytest.approx(3 / total_events)
    assert table.loc["produce", "lift"] == pytest.approx((2 / 5) / (3 / 10))
    assert table.loc["produce", "breadth_all"] == 0.5 and table.loc["produce", "n_users_bought"] == 1 and table.loc["produce", "breadth_buyers"] == 1.0
    assert table.loc["dairy", "n_users_bought"] == 2 and table.loc["dairy", "breadth_all"] == 0.5 and table.loc["dairy", "breadth_buyers"] == 0.5
    assert table.loc["dairy", "known_category_share"] == 0.0                              # n2: dairy was new to B
    assert table.loc["produce", "known_category_share"] == pytest.approx(0.5)            # z: produce new to A; w: produce known
    assert table.loc["snacks", "known_category_share"] == 0.0 and math.isnan(table.loc["produce", "rate_lo"])
    rows_json = pec.table_rows(table.reset_index())
    assert rows_json[0]["rate_ci"] is None and isinstance(rows_json[0]["n_new"], int)
    fine = pec.category_table(events, rows, "L2", users, n_boot=0, seed=0).set_index("category")
    assert fine.loc["apples", "n_new"] == 1 and fine.loc["bananas", "known_category_share"] == 0.0 and fine.loc["cookies", "n_new"] == 1


def test_user_category_matrix_and_bootstrap():
    users = pd.Index(["A", "B", "C"])
    categories = pd.Index(["c1", "c2"])
    frame = pd.DataFrame({"user_id": pd.array(["A", "A", "B", "B", "B", "Z"], dtype="string"), "L1": ["c1", "c2", "c1", "c1", "c2", "c1"],
                          "is_new": [True, False, True, False, True, True]})
    den = pec.user_category_matrix(frame, "L1", users, categories)
    num = pec.user_category_matrix(frame, "L1", users, categories, frame["is_new"].to_numpy())
    assert den.tolist() == [[1, 1], [2, 1], [0, 0]] and num.tolist() == [[1, 0], [1, 1], [0, 0]]   # user Z (not in users) ignored
    ci = pec.bootstrap_ratios(num, den, n=200, seed=1)
    assert set(ci) == {"rate_lo", "rate_hi", "lift_lo", "lift_hi"} and ci["rate_lo"].shape == (2,)
    assert 0.0 <= ci["rate_lo"][0] <= 2 / 3 <= ci["rate_hi"][0] <= 1.0
    # constant data -> degenerate CI
    const_num, const_den = np.array([[1.0], [1.0], [1.0]]), np.array([[2.0], [2.0], [2.0]])
    ci = pec.bootstrap_ratios(const_num, const_den, n=50, seed=0)
    assert ci["rate_lo"][0] == pytest.approx(0.5) and ci["rate_hi"][0] == pytest.approx(0.5) and ci["lift_lo"][0] == pytest.approx(1.0)


def test_support_mask_and_rankings():
    table = pd.DataFrame({"category": [f"c{i}" for i in range(12)], "n_new": [500, 300, 250, 220, 210, 205, 150, 100, 90, 80, 10, 5],
                          "n_events": [1000] * 12, "is_missing": [False] * 11 + [True]})
    table["rate"] = table["n_new"] / table["n_events"]
    table["share_new_events"] = table["n_new"] / table["n_new"].sum()
    table["share_events"] = 1 / 12
    table["lift"] = table["share_new_events"] / table["share_events"]
    table["breadth_all"] = np.linspace(0.9, 0.1, 12)
    table["breadth_buyers"] = table["breadth_all"]
    table["n_users_new"] = 10
    table["rate_lo"], table["rate_hi"], table["lift_lo"], table["lift_hi"] = table["rate"] - 0.01, table["rate"] + 0.01, table["lift"] * 0.9, table["lift"] * 1.1
    assert pec.support_mask(table, "coarse", "instacart").all()
    assert pec.support_mask(table, "fine", "instacart").sum() == 6 and pec.support_mask(table, "fine", "complete_journey").sum() == 8
    ranks = pec.rankings(table, pec.support_mask(table, "fine", "complete_journey"))
    assert [r["category"] for r in ranks["volume"]] == [f"c{i}" for i in range(8)]
    assert [r["category"] for r in ranks["lift_bottom"]][:2] == ["c7", "c6"] and ranks["lift_top"][0]["lift_ci"] == pytest.approx([ranks["lift_top"][0]["lift"] * 0.9, ranks["lift_top"][0]["lift"] * 1.1])
    assert ranks["propensity"][0]["rate_ci"] is not None and ranks["breadth"][0]["category"] == "c0" and len(ranks["breadth"]) == 8
    all_ranks = pec.rankings(table, pec.support_mask(table, "coarse", "instacart"))
    assert len(all_ranks["volume"]) == 10 and all_ranks["lift_bottom"][0]["category"] == "c11" and all_ranks["lift_bottom"][0]["is_missing"]


# ---------------------------------------------------------------------------
# predictability, quadrants, mismatch
# ---------------------------------------------------------------------------
def test_predictability_table_quadrants_and_mismatch():
    new_target = _frame([("A", 4, "n1"), ("A", 4, "n2"), ("B", 3, "n2"), ("B", 3, "m"), ("C", 3, "m")])
    e2 = _lists([("A", 1, "n1"), ("A", 2, "m"), ("B", 1, "n2"), ("B", 2, "n1"), ("C", 1, "n1"), ("C", 2, "a")])
    pop_list = _lists([("A", 1, "m"), ("B", 1, "m"), ("C", 1, "m")], "popularity")
    users = pd.Index(["A", "B", "C"])
    table = pec.predictability_table(new_target, e2, pop_list, "L2", users, n_boot=100, seed=0).set_index("category")
    # target new items: cookies (A), yogurt (A, B), apples (B, C); E2 hits: A-n1 (cookies), B-n2 (yogurt); pop hits: B-m, C-m (apples)
    assert table.loc["cookies", "n_target_new"] == 1 and table.loc["cookies", "e2_hit_rate"] == 1.0 and table.loc["cookies", "pop_hit_rate"] == 0.0
    assert table.loc["yogurt", "n_target_new"] == 2 and table.loc["yogurt", "e2_hit_rate"] == 0.5
    assert table.loc["apples", "e2_hit_rate"] == 0.0 and table.loc["apples", "pop_hit_rate"] == 1.0
    assert table.loc["chips", "n_target_new"] == 0 and table.loc["chips", "e2_slots"] == 1 and math.isnan(table.loc["chips", "e2_hit_rate"])
    assert table["e2_slots"].sum() == 6 and table.loc["cookies", "e2_slot_share"] == pytest.approx(3 / 6) and table.loc["cookies", "target_new_share"] == pytest.approx(1 / 5)
    assert table.loc["cookies", "slot_minus_target_share"] == pytest.approx(0.5 - 0.2) and table.loc["cookies", "slot_ratio"] == pytest.approx(2.5)
    assert table.loc["cookies", "e2_lo"] <= 1.0 and not math.isnan(table.loc["yogurt", "e2_lo"])
    rows = pec.table_rows(table.reset_index())
    assert {r["category"]: r["e2_hit_rate_ci"] for r in rows}["chips"] is None
    # quadrants over a small universe: lower the support threshold by monkeypatching the constant
    category = pd.DataFrame({"category": ["cookies", "yogurt", "apples", "chips"], "rate": [0.9, 0.5, 0.2, 0.1], "n_new": [9, 5, 2, 1],
                             "is_missing": [False] * 4})
    original = dict(pec.SUPPORT_MIN_NEW)
    pec.SUPPORT_MIN_NEW["instacart"] = 1
    try:
        q = pec.quadrants(category, table.reset_index(), "instacart")
        assert q["n_categories"] == 3 and q["median_rate"] == 0.5 and q["median_e2_hit_rate"] == 0.5
        assert [r["category"] for r in q["quadrants"]["explored_easy_to_predict"]] == ["cookies"]          # rate 0.9 > 0.5, e2 1.0 > 0.5
        assert [r["category"] for r in q["quadrants"]["rarely_explored_hard_to_predict"]] == ["apples", "yogurt"]   # medians count as low; ties by name
        assert q["quadrants"]["explored_hard_to_predict"] == [] and q["quadrants"]["rarely_explored_easy_to_predict"] == []
        mm = pec.mismatches(table.reset_index(), "fine", "instacart")
        assert mm["over_served"][0]["category"] == "cookies" and mm["over_served"][0]["difference"] == pytest.approx(0.3)
        assert mm["under_served"][0]["category"] == "apples" and mm["under_served"][0]["difference"] == pytest.approx(1 / 6 - 2 / 5)   # one E2 slot (A rank 2)
        assert mm["n_categories"] == 4                                                           # chips kept through the 1 % slot-share rule
    finally:
        pec.SUPPORT_MIN_NEW.update(original)


def test_spearman_and_history_bins():
    eval_table = pd.DataFrame({"category": ["a", "b", "c", "d"], "n_new": [40, 30, 20, 10], "rate": [0.4, 0.3, 0.2, 0.1]})
    fit_table = pd.DataFrame({"category": ["a", "b", "c", "e"], "n_new": [10, 20, 30, 5], "rate": [0.1, 0.2, 0.3, 0.9]})
    result = pec.spearman_against_fit(eval_table, fit_table, pd.Series([True, True, True, True]))
    assert result["n_categories"] == 3 and result["volume"]["spearman"] == pytest.approx(-1.0) and result["propensity"]["spearman"] == pytest.approx(-1.0)
    assert pec.spearman_against_fit(eval_table, fit_table, pd.Series([True, True, False, False]))["volume"]["spearman"] is None
    history, target = _split()
    events, _ = pec.build_events(history, target)
    lengths = pd.Series({"A": 3, "B": 2})
    bins = pec.propensity_by_history_bin(events, lengths, "instacart")
    assert [b["bin"] for b in bins] == ["(0,4]", "(4,6]", "(6,10]", "(10,20]", "(20,inf)"]
    assert bins[0]["n_users"] == 2 and bins[0]["n_events"] == 10 and bins[0]["n_new"] == 5 and bins[1]["categories"] == []
    produce = next(c for c in bins[0]["categories"] if c["category"] == "produce")
    assert produce["rate"] == pytest.approx(2 / 3) and produce["share_new_events"] == pytest.approx(0.4)


# ---------------------------------------------------------------------------
# end-to-end runs
# ---------------------------------------------------------------------------
def _synthetic_dataset(folder, dataset, seed, n_users=200, n_items=40):
    """Users with 3-9 history baskets plus one target basket, favourite items plus random ones, 4 coarse / 10 fine categories, one 'missing' item group."""
    rng = np.random.default_rng(seed)
    coarse = [("missing" if j % 10 == 9 else f"dept{j % 4}") for j in range(n_items)]
    fine = [("missing" if j % 10 == 9 else f"aisle{j % 10}") for j in range(n_items)]
    rows = []
    for user in range(n_users):
        n_baskets = int(rng.integers(3, 10))
        favourites = rng.choice(n_items, size=5, replace=False)
        for seq in range(1, n_baskets + 2):
            items = set(rng.choice(favourites, size=3).tolist()) | set(rng.choice(n_items, size=2).tolist())
            rows.extend((dataset, str(user), f"{user}-{seq}", seq, str(item), coarse[item], fine[item], 1) for item in sorted(items))
    frame = pd.DataFrame(rows, columns=COLUMNS).astype(
        {"user_id": "string", "basket_id": "string", "basket_seq": "int32", "item_id": "string", "quantity": "int16"})
    last = frame.groupby("user_id")["basket_seq"].transform("max")
    folder.mkdir(parents=True, exist_ok=True)
    frame[frame["basket_seq"] == last].to_parquet(folder / "eval_target.parquet", index=False)
    frame[frame["basket_seq"] < last].to_parquet(folder / "eval_history.parquet", index=False)
    frame[frame["basket_seq"] == last - 1].to_parquet(folder / "fit_target.parquet", index=False)
    frame[frame["basket_seq"] < last - 1].to_parquet(folder / "fit_history.parquet", index=False)
    item_ids = [str(j) for j in range(n_items)]
    catalogue = pd.DataFrame({"item_id": item_ids, "coarse_category": coarse, "fine_category": fine, "ingredient_type": [f"t{j % 7}" for j in range(n_items)],
                              "DEPARTMENT": coarse, "commodity": fine})
    catalogue.to_parquet(folder / "items.parquet", index=False)


def _check_document(document, n_users):
    assert document["label"] == pec.LABEL and document["bootstrap"]["level"] == 0.95
    for dataset, entry in document["datasets"].items():
        ev = entry["eval"]
        assert ev["n_users"] == n_users and ev["n_new_events"] <= ev["n_events"] and 0.0 < ev["overall_rate"]["all"] < 1.0
        for level_name in ("coarse", "fine"):
            block = ev["levels"][level_name]
            for event_set, sub in block["sets"].items():
                table = sub["table"]
                assert sum(r["n_new"] for r in table) == (ev["n_new_events"] if event_set == "all" else ev["n_target_new_events"] if event_set == "target"
                                                          else ev["n_new_events"] - ev["n_target_new_events"])
                assert abs(sum(r["share_new_events"] for r in table if r["share_new_events"] is not None) - 1.0) < 1e-9
                for r in table:
                    assert 0.0 <= r["rate"] <= 1.0 and 0.0 <= r["breadth_all"] <= 1.0 and (r["breadth_buyers"] is None or r["breadth_buyers"] <= 1.0)
                    if r["rate_ci"]:
                        assert r["rate_ci"][0] - 1e-12 <= r["rate"] <= r["rate_ci"][1] + 1e-12
                assert set(sub["rankings"]) == {"volume", "propensity", "lift_top", "lift_bottom", "breadth"}
                assert len(sub["rankings"]["volume"]) <= pec.TOP_N and sub["n_pass_support"] <= sub["n_categories"]
            if level_name == "coarse":
                assert block["sets"]["all"]["n_pass_support"] == block["sets"]["all"]["n_categories"]
            assert block["n_missing_label_categories"] >= 0
        pred = ev["predictability"]
        assert 0.0 <= pred["overall_e2_hit_rate"] <= 1.0 and pred["n_e2_slots"] == n_users * pec.K
        for level_name, block in pred["levels"].items():
            table = block["table"]
            assert sum(r["n_target_new"] for r in table) == pred["n_target_new_items"] and sum(r["e2_slots"] for r in table) == pred["n_e2_slots"]
            assert abs(sum(r["e2_slot_share"] for r in table) - 1.0) < 1e-9
            assert set(block["mismatch"]) == {"n_categories", "definition", "over_served", "under_served"}
            for r in block["mismatch"]["over_served"]:
                assert r["difference"] == pytest.approx(r["e2_slot_share"] - r["target_new_share"])
        q = ev["quadrants_fine"]
        assert set(q["quadrants"]) <= {"explored_hard_to_predict", "explored_easy_to_predict", "rarely_explored_hard_to_predict", "rarely_explored_easy_to_predict"}
        assert sum(len(v) for v in q["quadrants"].values()) == q["n_categories"]
        bins = ev["coarse_propensity_by_history_bin"]
        assert [b["bin"] for b in bins] == entry["history_bins"] and sum(b["n_users"] for b in bins) == n_users
        assert entry["fit"]["role"] == "robustness check"
        for level_name in ("coarse", "fine"):
            s = entry["fit"]["levels"][level_name]["spearman_vs_eval"]
            assert s["n_categories"] >= 0 and (s["volume"]["spearman"] is None or -1.0 <= s["volume"]["spearman"] <= 1.0)


def test_end_to_end_on_synthetic_sample(tmp_path, monkeypatch):
    data_dir = tmp_path / "processed"
    for seed, dataset in enumerate(pec.DATASETS):
        _synthetic_dataset(data_dir / "sample" / dataset, dataset, seed)
    monkeypatch.setenv("NBR_DATA_DIR", str(data_dir))
    out = tmp_path / "out"
    json_path = pec.main(["--profile", "sample", "--out-dir", str(out), "--usercf-k", "5", "--usercf-weighting", "binary", "--bootstrap-n", "40", "--seed", "1"])
    document = json.loads(json_path.read_text(encoding="utf-8"))
    assert document["profile"] == "sample" and document["bootstrap"] == {"n": 40, "seed": 1, "level": 0.95}
    assert set(document["datasets"]) == set(pec.DATASETS) and set(document["inputs_sha256"]["instacart"]) == set(pec.INPUT_FILES)
    _check_document(document, 200)
    entry = document["datasets"]["instacart"]
    assert entry["usercf"] == {"k": 5, "weighting": "binary", "source": "override", "verified": None}
    assert entry["eval"]["levels"]["fine"]["sets"]["all"]["n_pass_support"] > 0                      # the lift figure has rows
    assert entry["eval"]["levels"]["coarse"]["n_missing_label_categories"] == 1 and entry["eval"]["levels"]["coarse"]["n_categories"] == 5
    coarse = {r["category"]: r for r in entry["eval"]["levels"]["coarse"]["sets"]["all"]["table"]}
    assert coarse["missing"]["is_missing"] and not coarse["dept0"]["is_missing"]
    for name, figure in document["figures"].items():
        assert (out / figure["png"]).exists() and (out / figure["csv"]).exists()
        assert figure["points"] > 0 or name == "rate_vs_e2_fine"           # the 2 x 2 universe can be empty on small synthetic data
    rate_csv = pd.read_csv(out / "fig_category_rate_coarse.csv")
    assert set(rate_csv["dataset"]) == set(pec.DATASETS) and len(rate_csv[rate_csv["dataset"] == "instacart"]) == 5
    markdown = (out / "explore_category_sample.md").read_text(encoding="utf-8")
    assert pec.LABEL in markdown and "## Instacart" in markdown and "## Complete Journey" in markdown and "2 x 2 placement" in markdown
    assert "missing (missing label)" in markdown and "### Instacart: robustness" in markdown
    for word in ("significant", "novel", "outperform", "causes", "because of"):
        assert word not in markdown.lower()
    assert "confirmatory" not in markdown.replace("not confirmatory", "")


REAL_DATA = Path(os.environ.get("NBR_DATA_DIR", str(REPO_ROOT.parent / "dev_ML" / "data" / "processed"))) / "sample"


@pytest.mark.skipif(not (REAL_DATA / "instacart" / "eval_target.parquet").exists(), reason="development sample data not available")
def test_end_to_end_on_real_sample_profile(tmp_path, monkeypatch):
    monkeypatch.setenv("NBR_DATA_DIR", str(REAL_DATA.parent))
    out = tmp_path / "out"
    json_path = pec.main(["--profile", "sample", "--datasets", "instacart", "--out-dir", str(out), "--usercf-k", "20", "--usercf-weighting", "binary",
                          "--bootstrap-n", "20", "--seed", "1"])
    document = json.loads(json_path.read_text(encoding="utf-8"))
    entry = document["datasets"]["instacart"]
    _check_document(document, entry["eval"]["n_users"])
    assert entry["eval"]["levels"]["coarse"]["n_categories"] == 21 and entry["eval"]["levels"]["fine"]["n_categories"] == 134
    assert entry["eval"]["levels"]["coarse"]["n_missing_label_categories"] == 1
    assert (out / "explore_category_sample.md").exists() and (out / "fig_category_rate_vs_e2_fine.png").exists()
