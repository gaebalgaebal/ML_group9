# *------------------- 2026 Fall Semester Machine Learning Term-project: Team 9 --------------------*
# Code First written at 2026-11-06
# Code Final Edited at 2026-11-13
# Editor: Yejun Kim, Seongwoo Yang

'''
Purpose : Tests for scripts/make_basket_days.py: argument parsing, day derivation, id-mapping
          checks, provenance files.
Inputs  : The script scripts/make_basket_days.py loaded as a module, hand-written order,
          transaction and basket frames, and tiny zip archives and parquet files written to the
          pytest tmp_path folder.
Outputs : None; the tests assert on the parsed arguments, the day frames and statistics, the
          mapping checks, the written basket_days.parquet and the manifest and source files.
Usage   :
    python -m pytest tests/test_make_basket_days.py
'''

# ---------------------------------------------------------------------------
# imports
# ---------------------------------------------------------------------------
import importlib.util
import zipfile

import numpy as np
import pandas as pd
import pytest

from nbr.config import REPO_ROOT


# ---------------------------------------------------------------------------
# script loading
# ---------------------------------------------------------------------------
def _load_script():
    '''Load scripts/make_basket_days.py as a module and return it.'''
    # Build the import spec from the path of the script.
    path = REPO_ROOT / "scripts" / "make_basket_days.py"
    spec = importlib.util.spec_from_file_location("make_basket_days", path)
    # Create the module object and execute the script in it.
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


mbd = _load_script()


# ---------------------------------------------------------------------------
# arguments
# ---------------------------------------------------------------------------
def test_parse_args_defaults():
    # Parse only the two required options.
    args = mbd.parse_args(["--raw-dir", "raw", "--profile", "sample"])
    # Expect both datasets and both flags switched off.
    assert args.raw_dir == "raw" and args.profile == "sample"
    assert args.datasets == ["instacart", "complete_journey"]
    assert not args.overwrite and not args.no_provenance


def test_parse_args_dataset_subset_and_flags():
    # Parse a dataset list with blanks and a repeated name, plus both flags.
    args = mbd.parse_args(["--raw-dir", "raw", "--profile", "medium", "--datasets", " complete_journey, ,complete_journey",
                           "--overwrite", "--no-provenance"])
    # Expect the dataset once and both flags switched on.
    assert args.datasets == ["complete_journey"] and args.overwrite and args.no_provenance


@pytest.mark.parametrize("argv", [
    ["--raw-dir", "raw", "--profile", "holdout"],
    ["--raw-dir", "raw", "--profile", "holdout2"],
    ["--raw-dir", "raw", "--profile", "sample", "--datasets", "holdout"],
    ["--raw-dir", "raw", "--profile", "sample", "--datasets", " , "],
    ["--raw-dir", "raw"],
])
def test_parse_args_rejects(argv):
    # Expect SystemExit for each argument list of the parameter table.
    with pytest.raises(SystemExit):
        mbd.parse_args(argv)


# ---------------------------------------------------------------------------
# Instacart days
# ---------------------------------------------------------------------------
def _orders(rows):
    '''Build a raw Instacart orders frame with typed columns from the tuples in ``rows``.'''
    # Build the four order columns from the tuples.
    frame = pd.DataFrame(rows, columns=["order_id", "user_id", "order_number", "days_since_prior_order"])
    # Cast the ids and the order number to int64 and the gap to float64.
    return frame.astype({"order_id": "int64", "user_id": "int64", "order_number": "int64", "days_since_prior_order": "float64"})


ORDERS = _orders([
    (10, 1, 1, np.nan), (11, 1, 2, 5.0), (12, 1, 3, 30.0), (13, 1, 4, 30.0),
    (20, 2, 1, np.nan),
    (30, 3, 1, np.nan), (31, 3, 2, 0.0),
])


def test_instacart_days_cumulative_sum_and_capped_share():
    # Derive the days from the shuffled orders of three users.
    days, stats = mbd.instacart_days(ORDERS.sample(frac=1, random_state=1))
    # Expect the day to be the running sum of the gaps per user and 2 of 4 gaps at the cap.
    lookup = {(u, b): d for u, b, d in days[["user_id", "basket_id", "day"]].itertuples(index=False)}
    assert lookup == {("1", "10"): 0, ("1", "11"): 5, ("1", "12"): 35, ("1", "13"): 65, ("2", "20"): 0,
                      ("3", "30"): 0, ("3", "31"): 0}
    assert days["day"].dtype == "int64" and list(days.columns) == ["user_id", "basket_id", "day", "order_number"]
    assert stats["orders"] == 7 and stats["users"] == 3 and stats["non_first_orders"] == 4
    assert stats["gaps_equal_cap"] == 2 and stats["share_gaps_equal_cap"] == pytest.approx(0.5)
    assert stats["gap_cap_days"] == 30


@pytest.mark.parametrize("bad", [
    _orders([(10, 1, 1, np.nan), (11, 1, 3, 5.0)]),            # order_number not contiguous
    _orders([(10, 1, 2, 5.0)]),                                  # does not start at 1
    _orders([(10, 1, 1, 3.0), (11, 1, 2, 5.0)]),                # first order with a gap
    _orders([(10, 1, 1, np.nan), (11, 1, 2, np.nan)]),          # later order without a gap
    _orders([(10, 1, 1, np.nan), (10, 2, 1, np.nan)]),          # duplicated order_id
    _orders([(10, 1, 1, np.nan), (11, 1, 2, -1.0)]),            # negative gap
])
def test_instacart_days_rejects_inconsistent_orders(bad):
    # Expect RuntimeError for each inconsistent orders frame of the parameter table.
    with pytest.raises(RuntimeError):
        mbd.instacart_days(bad)


# ---------------------------------------------------------------------------
# Complete Journey days
# ---------------------------------------------------------------------------
def _transactions(rows):
    '''Build a raw transaction frame (household_key, BASKET_ID, DAY) from ``rows``.'''
    # Build the three columns from the tuples.
    frame = pd.DataFrame(rows, columns=["household_key", "BASKET_ID", "DAY"])
    # Cast every column to int64.
    return frame.astype("int64")


def test_complete_journey_days_one_day_per_basket():
    # Derive the days from four transaction rows, two of them in the same basket.
    days, stats = mbd.complete_journey_days(_transactions([(1, 100, 3), (1, 100, 3), (1, 101, 9), (2, 200, 3)]))
    # Expect one row per basket with its day and the matching statistics.
    lookup = {(u, b): d for u, b, d in days.itertuples(index=False)}
    assert lookup == {("1", "100"): 3, ("1", "101"): 9, ("2", "200"): 3}
    assert stats == {"baskets": 3, "households": 2, "baskets_with_several_days": 0, "day_min": 3, "day_max": 9}


def test_complete_journey_days_stops_on_multi_day_basket():
    # Expect RuntimeError for a basket whose rows carry two different days.
    with pytest.raises(RuntimeError):
        mbd.complete_journey_days(_transactions([(1, 100, 3), (1, 100, 4)]))


# ---------------------------------------------------------------------------
# id-mapping check
# ---------------------------------------------------------------------------
def _baskets(rows):
    '''Build a processed basket frame (user_id, basket_id, basket_seq) from ``rows``.'''
    # Build the three columns from the tuples.
    frame = pd.DataFrame(rows, columns=["user_id", "basket_id", "basket_seq"])
    # Cast the basket sequence number to int64.
    return frame.astype({"basket_seq": "int64"})


def test_check_mapping_instacart():
    # Check the processed baskets of users 1 and 3 against the Instacart days.
    days, _ = mbd.instacart_days(ORDERS)
    baskets = _baskets([("1", "10", 1), ("1", "11", 2), ("1", "12", 3), ("1", "13", 4), ("3", "30", 1), ("3", "31", 2)])
    frame, stats = mbd.check_mapping("instacart", baskets, days)
    # Expect all six baskets matched, then RuntimeError for the two inconsistent inputs.
    assert list(frame.columns) == ["user_id", "basket_id", "day"] and len(frame) == 6
    assert stats["processed_baskets"] == 6 and stats["matched_raw_baskets"] == 6 and stats["unmatched_baskets"] == 0
    assert stats["basket_seq_equals_order_number"] == 6 and stats["day_decreasing_steps"] == 0
    assert (stats["day_min"], stats["day_max"]) == (0, 65)
    with pytest.raises(RuntimeError):   # a processed basket without a raw order
        mbd.check_mapping("instacart", _baskets([("1", "10", 1), ("1", "99", 2)]), days)
    with pytest.raises(RuntimeError):   # basket_seq disagrees with order_number
        mbd.check_mapping("instacart", _baskets([("1", "10", 2), ("1", "11", 1)]), days)


def test_check_mapping_complete_journey():
    # Check four processed baskets against the Complete Journey days.
    days, _ = mbd.complete_journey_days(_transactions([(1, 100, 3), (1, 101, 9), (1, 102, 9), (2, 200, 3)]))
    baskets = _baskets([("1", "100", 1), ("1", "101", 2), ("1", "102", 3), ("2", "200", 1)])
    frame, stats = mbd.check_mapping("complete_journey", baskets, days)
    # Expect all four baskets matched, then RuntimeError for the two inconsistent inputs.
    assert len(frame) == 4 and stats["unmatched_baskets"] == 0 and stats["day_decreasing_steps"] == 0
    with pytest.raises(RuntimeError):   # day decreases along basket_seq
        mbd.check_mapping("complete_journey", _baskets([("1", "100", 2), ("1", "101", 1)]), days)
    with pytest.raises(RuntimeError):   # unmatched household
        mbd.check_mapping("complete_journey", _baskets([("9", "100", 1)]), days)


# ---------------------------------------------------------------------------
# processed baskets and the end-to-end run on a tiny archive
# ---------------------------------------------------------------------------
def _write_processed(folder, frames):
    '''Write each entry of ``frames`` (name -> row tuples) to ``folder``/<name>.parquet.'''
    # Create the folder when it does not exist yet.
    folder.mkdir(parents=True, exist_ok=True)
    for name, rows in frames.items():
        # Build the typed four-column frame of this split and write it as parquet.
        frame = pd.DataFrame(rows, columns=["user_id", "basket_id", "basket_seq", "item_id"])
        frame = frame.astype({"user_id": "string", "basket_id": "string", "basket_seq": "int32", "item_id": "string"})
        frame.to_parquet(folder / f"{name}.parquet", index=False)


def test_processed_baskets(tmp_path):
    # Write the four processed frames of one user with four baskets.
    _write_processed(tmp_path, {
        "fit_history": [("1", "10", 1, "a"), ("1", "10", 1, "b"), ("1", "11", 2, "a")],
        "fit_target": [("1", "12", 3, "a")],
        "eval_history": [("1", "10", 1, "a"), ("1", "11", 2, "a"), ("1", "12", 3, "a")],
        "eval_target": [("1", "13", 4, "c")],
    })
    baskets, counts = mbd.processed_baskets(tmp_path)
    # Expect the basket count of each frame and four distinct baskets.
    assert counts == {"fit_history": 2, "fit_target": 1, "eval_history": 3, "eval_target": 1, "distinct": 4}
    assert sorted(baskets["basket_id"]) == ["10", "11", "12", "13"]
    assert baskets["user_id"].dtype == object
    # Expect RuntimeError when a basket appears with two different sequence numbers.
    _write_processed(tmp_path, {"eval_target": [("1", "10", 9, "c")]})   # same basket, other seq
    with pytest.raises(RuntimeError):
        mbd.processed_baskets(tmp_path)


def _instacart_zip(path):
    '''Write a tiny Instacart archive with the orders of users 1, 2 and 9 to ``path``.'''
    with zipfile.ZipFile(path, "w") as archive:
        # Write eight order rows and two empty member files.
        archive.writestr("orders.csv", "order_id,user_id,eval_set,order_number,order_dow,order_hour_of_day,days_since_prior_order\n"
                                       "10,1,prior,1,0,8,\n11,1,prior,2,1,9,5.0\n12,1,prior,3,2,10,30.0\n13,1,train,4,3,11,30.0\n"
                                       "20,2,prior,1,0,8,\n21,2,train,2,0,8,7.0\n"
                                       "90,9,prior,1,0,8,\n91,9,test,2,0,8,30.0\n")
        archive.writestr("order_products__prior.csv", "")
        archive.writestr("products.csv", "")


def _complete_journey_zip(path):
    '''Write a tiny Complete Journey archive with households 1, 2 and 7 to ``path``.'''
    with zipfile.ZipFile(path, "w") as archive:
        # Write five transaction rows and one empty member file.
        archive.writestr("transaction_data.csv", "household_key,BASKET_ID,DAY,PRODUCT_ID,QUANTITY,SALES_VALUE,STORE_ID,RETAIL_DISC,TRANS_TIME,WEEK_NO,COUPON_DISC,COUPON_MATCH_DISC\n"
                                                 "1,100,3,825123,1,1.0,1,0,1000,1,0,0\n1,100,3,831447,1,1.0,1,0,1000,1,0,0\n"
                                                 "1,101,9,825123,1,1.0,1,0,1000,2,0,0\n2,200,3,825123,1,1.0,1,0,1000,1,0,0\n"
                                                 "7,700,5,825123,1,1.0,1,0,1000,1,0,0\n")
        archive.writestr("product.csv", "")


def test_run_dataset_end_to_end(tmp_path):
    # Write both tiny archives and the processed Instacart frames of users 1 and 2.
    raw = tmp_path / "raw"
    (raw / "instacart").mkdir(parents=True)
    (raw / "complete_journey").mkdir(parents=True)
    _instacart_zip(raw / "instacart" / "instacart.zip")
    _complete_journey_zip(raw / "complete_journey" / "archive.zip")
    ic = tmp_path / "sample" / "instacart"
    _write_processed(ic, {
        "fit_history": [("1", "10", 1, "a"), ("1", "11", 2, "a"), ("2", "20", 1, "b")],
        "fit_target": [("1", "12", 3, "a"), ("2", "21", 2, "b")],
        "eval_history": [("1", "10", 1, "a"), ("1", "11", 2, "a"), ("1", "12", 3, "a"), ("2", "20", 1, "b")],
        "eval_target": [("1", "13", 4, "a"), ("2", "21", 2, "b")],
    })
    entry = mbd.run_dataset("instacart", raw, ic, overwrite=False)
    out = pd.read_parquet(ic / "basket_days.parquet")
    # Expect one day per processed basket and the statistics of the returned entry.
    assert list(out.columns) == ["user_id", "basket_id", "day"]
    assert out["day"].dtype == "int64" and str(out["user_id"].dtype) == "string"
    lookup = {(u, b): d for u, b, d in out.itertuples(index=False)}
    assert lookup == {("1", "10"): 0, ("1", "11"): 5, ("1", "12"): 35, ("1", "13"): 65, ("2", "20"): 0, ("2", "21"): 7}
    assert entry["rows"] == 6 and entry["users"] == 2 and entry["archive"] == "instacart.zip"
    assert entry["raw"]["share_gaps_equal_cap"] == pytest.approx(2 / 4)        # selected users 1 and 2
    assert entry["raw_all_users"]["share_gaps_equal_cap"] == pytest.approx(3 / 5)   # user 9 included
    assert entry["mapping_check"]["unmatched_baskets"] == 0 and entry["sha256"] == mbd.sha256_of(ic / "basket_days.parquet")
    # Expect FileExistsError for a second run without overwrite and the same digest with it.
    with pytest.raises(FileExistsError):
        mbd.run_dataset("instacart", raw, ic, overwrite=False)
    assert mbd.run_dataset("instacart", raw, ic, overwrite=True)["sha256"] == entry["sha256"]   # deterministic bytes

    # Write the processed Complete Journey frames of households 1 and 2 and run the dataset.
    cj = tmp_path / "sample" / "complete_journey"
    _write_processed(cj, {
        "fit_history": [("1", "100", 1, "a")], "fit_target": [("1", "101", 2, "a")],
        "eval_history": [("1", "100", 1, "a"), ("1", "101", 2, "a")], "eval_target": [("2", "200", 1, "a")],
    })
    entry = mbd.run_dataset("complete_journey", raw, cj, overwrite=False)
    out = pd.read_parquet(cj / "basket_days.parquet")
    # Expect the raw DAY of each processed basket and the household counts of the entry.
    assert {(u, b): d for u, b, d in out.itertuples(index=False)} == {("1", "100"): 3, ("1", "101"): 9, ("2", "200"): 3}
    assert entry["raw"]["households"] == 2 and entry["raw_all_users"]["households"] == 3
    assert entry["mapping_check"]["mapping"].startswith("basket_id == str(BASKET_ID)")


def test_run_dataset_stops_on_unmatched_basket(tmp_path):
    # Write the archive and processed frames that contain a basket 55 without a raw order.
    raw = tmp_path / "raw"
    raw.mkdir()
    _instacart_zip(raw / "instacart.zip")
    ic = tmp_path / "instacart"
    _write_processed(ic, {name: [("1", "10", 1, "a"), ("1", "55", 2, "a")] for name in mbd.FRAMES})
    # Expect RuntimeError and no output file.
    with pytest.raises(RuntimeError):
        mbd.run_dataset("instacart", raw, ic, overwrite=False)
    assert not (ic / "basket_days.parquet").exists()


# ---------------------------------------------------------------------------
# provenance files
# ---------------------------------------------------------------------------
def test_append_manifest(tmp_path):
    # Write a manifest with one line and append one new entry.
    manifest = tmp_path / "MANIFEST.sha256"
    manifest.write_text("aaa  processed/medium/instacart/items.parquet\n", encoding="utf-8")
    added = mbd.append_manifest(manifest, [("processed/medium/instacart/basket_days.parquet", "bbb")])
    # Expect the new line at the end, no change on a repeat and RuntimeError for another digest.
    assert added == ["bbb  processed/medium/instacart/basket_days.parquet"]
    text = manifest.read_text(encoding="utf-8")
    assert text == "aaa  processed/medium/instacart/items.parquet\nbbb  processed/medium/instacart/basket_days.parquet\n"
    assert mbd.append_manifest(manifest, [("processed/medium/instacart/basket_days.parquet", "bbb")]) == []   # already listed
    assert manifest.read_text(encoding="utf-8") == text
    with pytest.raises(RuntimeError):   # listed with another digest: never silently replaced
        mbd.append_manifest(manifest, [("processed/medium/instacart/basket_days.parquet", "ccc")])


def test_append_source_once(tmp_path):
    # Write a source file with a title line and prepare a block for the sample profile.
    source = tmp_path / "SOURCE.md"
    source.write_text("# Data provenance\n", encoding="utf-8")
    block = "## basket_days: sample (2026-11-06, W14; post-hoc, development data, not confirmatory)\n- x\n"
    # Expect the block to be appended once per profile and skipped on a repeat.
    assert mbd.append_source(source, "sample", block) is True
    assert source.read_text(encoding="utf-8") == "# Data provenance\n\n" + block
    assert mbd.append_source(source, "sample", block) is False
    assert mbd.append_source(source, "medium", block.replace("sample", "medium")) is True
    assert source.read_text(encoding="utf-8").count("## basket_days:") == 2
