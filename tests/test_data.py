import zipfile

import pandas as pd
import pytest

from ml_group9.data import discover_archive, prepare_complete_journey


def test_archive_is_discovered_by_members_not_filename(tmp_path):
    path = tmp_path / "arbitrary-name.zip"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("first.csv", "x\n1\n")
        archive.writestr("second.csv", "y\n2\n")
    assert discover_archive(tmp_path, {"first.csv", "second.csv"}) == path


def test_archive_discovery_rejects_missing_members(tmp_path):
    with pytest.raises(FileNotFoundError):
        discover_archive(tmp_path, {"missing.csv"})


PRODUCT_COLUMNS = ["PRODUCT_ID", "MANUFACTURER", "DEPARTMENT", "BRAND", "COMMODITY_DESC", "SUB_COMMODITY_DESC", "CURR_SIZE_OF_PRODUCT"]
TRANSACTION_COLUMNS = ["household_key", "BASKET_ID", "DAY", "PRODUCT_ID", "QUANTITY", "SALES_VALUE", "STORE_ID", "TRANS_TIME", "WEEK_NO"]


def synthetic_complete_journey(tmp_path):
    products = pd.DataFrame(
        [
            ("1", 1, "GROCERY", "National", "MILK", "FLUID MILK", ""),
            ("2", 1, "GROCERY", "Private", "MILK", "FLUID MILK", ""),
            ("3", 1, "PRODUCE", "National", "ORGANICS FRUIT & VEGETABLES", "ORGANIC FRUIT", ""),
            ("4", 1, "GROCERY", "National", "BREAD", "BREAD", ""),
            ("90", 1, "KIOSK-GAS", "National", "COUPON/MISC ITEMS", "GASOLINE", ""),
            ("91", 1, " ", "National", "", "", ""),
        ],
        columns=PRODUCT_COLUMNS,
    )
    rows = [
        # b1: one valid line plus an unknown product and a blank-department product.
        ("h1", "b1", 1, "1", 1, 1.0, 1, 900, 1),
        ("h1", "b1", 1, "2", 1, 1.0, 1, 900, 1),
        ("h1", "b1", 1, "999", 1, 1.0, 1, 900, 1),
        ("h1", "b1", 1, "91", 1, 1.0, 1, 900, 1),
        # bfuel vanishes after cleaning, so b2 must become basket_seq 2.
        ("h1", "bfuel", 2, "90", 15000, 30.0, 1, 900, 1),
        ("h1", "b2", 3, "1", 0, 0.0, 1, 900, 2),
        ("h1", "b2", 3, "3", 1, 0.0, 1, 900, 2),
        ("h1", "b3", 4, "2", 1, 1.0, 1, 900, 3),
        # Future: a two-item basket precedes the first three-item basket.
        ("h1", "b4", 640, "1", 1, 1.0, 1, 900, 92),
        ("h1", "b4", 640, "2", 1, 1.0, 1, 900, 92),
        ("h1", "b5", 647, "1", 1, 1.0, 1, 900, 93),
        ("h1", "b5", 647, "2", 1, 1.0, 1, 900, 93),
        ("h1", "b5", 647, "4", 1, 1.0, 1, 900, 93),
        ("h1", "b6", 654, "1", 1, 1.0, 1, 900, 94),
        ("h1", "b6", 654, "2", 1, 1.0, 1, 900, 94),
        ("h1", "b6", 654, "3", 1, 1.0, 1, 900, 94),
        # h2 has three history baskets but no future basket with three distinct items.
        ("h2", "c1", 1, "1", 1, 1.0, 1, 900, 1),
        ("h2", "c2", 2, "2", 1, 1.0, 1, 900, 1),
        ("h2", "c3", 3, "3", 1, 1.0, 1, 900, 1),
        ("h2", "c4", 640, "1", 1, 1.0, 1, 900, 92),
        ("h2", "c4", 640, "1", 2, 1.0, 1, 900, 92),
        ("h2", "c4", 640, "2", 1, 1.0, 1, 900, 92),
    ]
    transactions = pd.DataFrame(rows, columns=TRANSACTION_COLUMNS)
    raw = tmp_path / "raw" / "complete_journey"
    raw.mkdir(parents=True)
    with zipfile.ZipFile(raw / "cj.zip", "w") as archive:
        archive.writestr("product.csv", products.to_csv(index=False))
        archive.writestr("transaction_data.csv", transactions.to_csv(index=False))
    return {
        "profile": "test",
        "seed": 42,
        "sample_users": {"complete_journey": None},
        "complete_journey_history_week": 90,
        "complete_journey": {"exclude_departments": ["KIOSK-GAS"], "min_target_items": 3},
        "attributes": {
            "complete_journey": ["private_label", "organic"],
            "private_label_brands": ["Private"],
            "min_with": 1,
            "min_without": 1,
            "distinct_min_n": 1,
        },
        "raw_dir": tmp_path / "raw",
        "processed_dir": tmp_path / "processed",
        "report_dir": tmp_path / "reports",
    }


def test_complete_journey_drops_non_product_rows_before_sequencing(tmp_path):
    config = synthetic_complete_journey(tmp_path)
    result = prepare_complete_journey(config)
    history = pd.read_parquet(tmp_path / "processed" / "complete_journey" / "eval_history.parquet")
    items = pd.read_parquet(tmp_path / "processed" / "complete_journey" / "items.parquet")

    assert result["cleaning"]["removed_unknown_product"] == 1
    assert result["cleaning"]["removed_missing_department"] == 1
    assert result["cleaning"]["removed_excluded_department"] == 1
    assert result["cleaning"]["removed_non_positive_quantity"] == 1
    assert result["cleaning"]["kept_non_positive_sales_value_rows"] == 1
    assert not history["item_id"].isin(["90", "91", "999"]).any()
    assert history["quantity"].min() >= 1
    assert "bfuel" not in set(history["basket_id"])
    h1 = history[history["user_id"].eq("h1")].drop_duplicates("basket_id").set_index("basket_id")["basket_seq"]
    assert h1.to_dict() == {"b1": 1, "b2": 2, "b3": 3}
    assert not items["item_id"].isin(["90", "91"]).any()


def test_complete_journey_target_skips_baskets_below_minimum_size(tmp_path):
    config = synthetic_complete_journey(tmp_path)
    result = prepare_complete_journey(config)
    history = pd.read_parquet(tmp_path / "processed" / "complete_journey" / "eval_history.parquet")
    target = pd.read_parquet(tmp_path / "processed" / "complete_journey" / "eval_target.parquet")

    assert set(target["basket_id"]) == {"b5"}
    assert set(target["item_id"]) == {"1", "2", "4"}
    # h2's only future basket has two distinct items, so the household is dropped entirely.
    assert set(history["user_id"]) == {"h1"}
    assert result["target"] == {
        "min_items": 3,
        "households": 1,
        "basket_size_median": 3.0,
        "basket_size_at_most_2_share": 0.0,
        "week_min": 93,
        "week_median": 93.0,
        "week_max": 93,
    }
    assert result["target_first_future_basket"]["households"] == 2
    assert result["target_first_future_basket"]["basket_size_median"] == 2.0


def test_complete_journey_items_carry_attributes_and_applicable_categories(tmp_path):
    config = synthetic_complete_journey(tmp_path)
    prepare_complete_journey(config)
    base = tmp_path / "processed" / "complete_journey"
    items = pd.read_parquet(base / "items.parquet").set_index("item_id")
    table = pd.read_parquet(base / "attribute_categories.parquet")

    assert items.loc["2", "attr_private_label"] == 1
    assert items.loc["3", "attr_organic"] == 1
    assert list(table.columns) == ["attribute", "fine_category", "n_with", "n_without", "included"]
    fluid_milk = table[table["attribute"].eq("private_label") & table["fine_category"].eq("FLUID MILK")].iloc[0]
    assert (fluid_milk["n_with"], fluid_milk["n_without"], bool(fluid_milk["included"])) == (1, 1, True)
    assert (tmp_path / "reports" / "metrics" / "test_attributes.json").exists()


def fit_split_journey(tmp_path, filter_before_sampling, sample_size):
    products = pd.DataFrame(
        [(str(item), 1, "GROCERY", "National", "MILK", "FLUID MILK", "") for item in range(1, 6)],
        columns=PRODUCT_COLUMNS,
    )
    rows = []

    def basket(household, basket_id, week, items):
        rows.extend((household, basket_id, week * 7, str(item), 1, 1.0, 1, 900, week) for item in items)

    # g1 forms both splits: three baskets by week 10, a 2-item then a 3-item basket in weeks 11-20.
    for week in [1, 2, 3]:
        basket("g1", f"g1-{week}", week, [week])
    basket("g1", "g1-12", 12, [1, 2])
    basket("g1", "g1-14", 14, [1, 2, 3])
    basket("g1", "g1-18", 18, [4])
    basket("g1", "g1-22", 22, [3, 4, 5])
    # g2 has no 3-item basket in weeks 11-20, so it cannot form the fit split.
    for week in [1, 2, 3]:
        basket("g2", f"g2-{week}", week, [week])
    basket("g2", "g2-15", 15, [1])
    basket("g2", "g2-25", 25, [1, 2, 3])
    # g3 has only two baskets by week 10.
    for week in [1, 2, 15]:
        basket("g3", f"g3-{week}", week, [1])
    basket("g3", "g3-16", 16, [1, 2, 3])
    basket("g3", "g3-22", 22, [1, 2, 3])
    raw = tmp_path / "raw" / "complete_journey"
    raw.mkdir(parents=True)
    with zipfile.ZipFile(raw / "cj.zip", "w") as archive:
        archive.writestr("product.csv", products.to_csv(index=False))
        archive.writestr("transaction_data.csv", pd.DataFrame(rows, columns=TRANSACTION_COLUMNS).to_csv(index=False))
    return {
        "profile": "test",
        "seed": 42,
        "sample_users": {"complete_journey": sample_size},
        "complete_journey_history_week": 20,
        "complete_journey": {
            "exclude_departments": [],
            "min_target_items": 3,
            "fit_history_week": 10,
            "filter_before_sampling": filter_before_sampling,
        },
        "attributes": {
            "complete_journey": ["private_label"],
            "private_label_brands": ["Private"],
            "min_with": 1,
            "min_without": 1,
            "distinct_min_n": 1,
        },
        "raw_dir": tmp_path / "raw",
        "processed_dir": tmp_path / "processed",
        "report_dir": tmp_path / "reports",
    }


def test_complete_journey_fit_split_precedes_eval_split_and_needs_both(tmp_path):
    result = prepare_complete_journey(fit_split_journey(tmp_path, filter_before_sampling=False, sample_size=None))
    base = tmp_path / "processed" / "complete_journey"
    frames = {name: pd.read_parquet(base / f"{name}.parquet") for name in ["fit_history", "fit_target", "eval_history", "eval_target"]}

    assert all(set(frame["user_id"]) == {"g1"} for frame in frames.values())
    assert set(frames["fit_target"]["basket_id"]) == {"g1-14"}
    assert set(frames["eval_target"]["basket_id"]) == {"g1-22"}
    assert frames["fit_history"]["basket_seq"].max() < frames["fit_target"]["basket_seq"].min()
    assert frames["eval_history"]["basket_seq"].max() < frames["eval_target"]["basket_seq"].min()
    # The fit target lies inside the eval history window (week 14 <= 20).
    assert "g1-14" in set(frames["eval_history"]["basket_id"])
    assert set(frames["fit_history"]["basket_id"]) == {"g1-1", "g1-2", "g1-3"}
    assert result["fit_target"]["week_min"] == 14 and result["fit_target"]["week_max"] == 14
    assert result["households_with_fit_target"] == 1
    assert result["households_with_eval_target"] == 3
    assert result["households_with_both_splits"] == 1
    assert result["sampled_households"] == 3


def test_complete_journey_filter_before_sampling_draws_only_qualifying_households(tmp_path):
    result = prepare_complete_journey(fit_split_journey(tmp_path, filter_before_sampling=True, sample_size=1))
    assert result["households_spanning_cutoff"] == 3
    assert result["households_in_sampling_pool"] == 1
    assert result["sampled_households"] == 1
    assert result["households_with_both_splits"] == 1
    assert result["users"] == 1
