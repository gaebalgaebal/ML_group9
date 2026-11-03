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
        "complete_journey": {"exclude_departments": ["KIOSK-GAS"]},
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
