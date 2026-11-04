import hashlib

import pandas as pd
import pytest

from ml_group9.ingredient_types import (
    assign_types,
    food_mask,
    import_mapping,
    read_mapping,
    type_attribute_categories,
    type_category_report,
)


VERSION = "2026-11-02.v2.1"
AUTO = ["auto_exact", "auto_fixed", "auto_normalized", "auto_partial"]


def mapping(rows, version=VERSION):
    return pd.DataFrame(
        [(pid, status, ingredient, note, version) for pid, status, ingredient, note in rows],
        columns=["product_id", "mapping_status", "ingredient_id", "mapping_note", "mapping_version"],
    )


def catalogue():
    return pd.DataFrame(
        {
            "item_id": ["1", "2", "3", "4", "5", "6"],
            "coarse_category": ["dairy eggs", "dairy eggs", "dairy eggs", "babies", "household", "dairy eggs"],
            "fine_category": ["milk", "milk", "yogurt", "baby food", "cleaning", "milk"],
        }
    )


def test_read_mapping_requires_one_row_per_product(tmp_path):
    path = tmp_path / "m.parquet"
    mapping([(1, "auto_exact", "i_milk", ""), (1, "auto_partial", "i_cream", "")]).to_parquet(path)
    with pytest.raises(ValueError, match="repeated products"):
        read_mapping(path, VERSION)


def test_version_mismatch_warns(tmp_path):
    path = tmp_path / "m.parquet"
    mapping([(1, "auto_exact", "i_milk", "")], version="2026-10-01.v1").to_parquet(path)
    with pytest.warns(UserWarning, match="mapping_version"):
        _, schema = read_mapping(path, VERSION)
    assert schema["version_matches"] is False


def test_import_copies_and_records_sha256(tmp_path):
    source = tmp_path / "source" / "m.parquet"
    source.parent.mkdir()
    mapping([(1, "auto_exact", "i_milk", "")]).to_parquet(source)
    before = source.read_bytes()
    destination = tmp_path / "raw" / "mapping" / "m.parquet"
    frame, report = import_mapping(source, destination, VERSION)
    expected = hashlib.sha256(before).hexdigest()
    assert report["sha256"] == expected and report["source_sha256"] == expected
    assert destination.read_bytes() == before and source.read_bytes() == before
    assert report["rows"] == 1 and report["mapping_version"] == [VERSION]
    assert report["source_path"] == source.as_posix()


def test_non_auto_items_get_the_aisle_fallback_type():
    table = assign_types(
        catalogue(),
        mapping(
            [
                (1, "auto_exact", "i_milk", ""),
                (2, "review_aisle_conflict", None, ""),
                (3, "auto_partial", "i_yogurt", "processed_form"),
                (4, "review_scope", None, ""),
                (5, "excluded_nonfood", None, ""),
                # Product 6 has no mapping row.
            ]
        ),
        AUTO,
    )
    assert table["ingredient_type"].tolist() == [
        "i_milk", "aisle:milk", "i_yogurt", "aisle:baby food", "aisle:cleaning", "aisle:milk",
    ]
    assert table["type_source"].tolist() == ["mapped", "fallback", "mapped", "fallback", "fallback", "fallback"]
    assert table["mapping_status"].tolist()[-1] == "missing"
    assert table["mapping_note"].tolist()[2] == "processed_form"
    # Re-assigning replaces the columns instead of duplicating them.
    again = assign_types(table, mapping([(1, "auto_exact", "i_milk", "")]), AUTO)
    assert list(again.columns).count("ingredient_type") == 1


def test_food_mask_uses_nonfood_departments():
    assert food_mask(catalogue(), ["household"]).tolist() == [True, True, True, True, False, True]


def type_items(flags):
    # Every row is in aisle "a"; types t1, t2.
    types, values = zip(*flags)
    return pd.DataFrame(
        {
            "item_id": [str(index) for index in range(len(flags))],
            "fine_category": "a",
            "ingredient_type": list(types),
            "attr_fat_reduced": list(values),
        }
    ).astype({"attr_fat_reduced": "int8"})


def test_type_categories_boundary_at_two_items_each():
    items = type_items([("t1", 1), ("t1", 1), ("t1", 0), ("t1", 0), ("t2", 1), ("t2", 0), ("t2", 0)])
    table = type_attribute_categories(items, "fine_category", "ingredient_type", ["fat_reduced"], 2, 2)
    included = dict(zip(table["type"], table["included"]))
    assert included == {"t1": True, "t2": False}
    row = table.set_index("type").loc["t2"]
    assert (row["n_with"], row["n_without"]) == (1, 2)


def test_type_categories_are_nested_in_stage1_keys():
    items = type_items([("t1", 1), ("t1", 1), ("t1", 0), ("t1", 0)])
    items.loc[[1, 3], "fine_category"] = "b"
    table = type_attribute_categories(items, "fine_category", "ingredient_type", ["fat_reduced"], 1, 1)
    assert sorted(zip(table["stage1"], table["type"], table["n_with"])) == [("a", "t1", 1), ("b", "t1", 1)]
    strict = type_attribute_categories(items, "fine_category", "ingredient_type", ["fat_reduced"], 2, 2)
    assert not strict["included"].any()


def test_type_category_report_shares():
    items = type_items([("t1", 1), ("t1", 1), ("t1", 0), ("t1", 0), ("t2", 1), ("t3", 0)])
    table = type_attribute_categories(items, "fine_category", "ingredient_type", ["fat_reduced"], 2, 2)
    history = pd.DataFrame(
        {"user_id": "u", "basket_id": ["b1", "b1", "b2", "b2"], "item_id": ["0", "2", "4", "5"]}
    )
    report = type_category_report(items, table, history, "fine_category", "ingredient_type")["fat_reduced"]
    assert report["included_pairs"] == 1
    assert report["purchase_share_included"] == pytest.approx(0.5)
    assert report["purchase_share_attribute_decides_type"] == pytest.approx(0.25)
    assert report["purchase_share_type_has_no_attribute_item"] == pytest.approx(0.25)
