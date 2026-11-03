import pandas as pd
import pytest

from ml_group9.attributes import (
    TEXT_ATTRIBUTES,
    applicable_categories,
    category_sets,
    complete_journey_attributes,
    instacart_text_diagnostics,
    text_attribute_flags,
)


PERCENT_AISLES = ["milk", "yogurt"]


def flags_for(name, aisle="milk"):
    flags = text_attribute_flags(pd.Series([name]), pd.Series([aisle]), PERCENT_AISLES, [0, 1, 2])
    return {attribute for attribute in TEXT_ATTRIBUTES if flags.loc[0, attribute] == 1}


@pytest.mark.parametrize(
    ("name", "aisle", "expected"),
    [
        ("Organic Low-Fat 1% Milk", "milk", {"fat_reduced", "organic"}),
        ("2% Reduced Fat Milk", "milk", {"fat_reduced"}),
        ("2% Lactose Free Milk", "milk", {"fat_reduced"}),
        ("Whole Milk", "milk", set()),
        ("100% Lactose Free Whole Milk", "milk", set()),
        ("Light Brown Sugar", "baking ingredients", set()),
        ("Lite Coconut Milk", "asian foods", set()),
        ("Diet Cola", "soft drinks", set()),
        ("Unsweetened Almondmilk", "soy lactosefree", {"sugar_reduced"}),
        ("Lightly Salted Baked Snap Pea Crisps", "chips pretzels", set()),
        ("No Salt Added Black Beans", "canned meals beans", {"low_sodium"}),
        ("Gluten Free Whole Grain Bread", "bread", {"gluten_free"}),
        ("Fat Free Half & Half", "cream", {"fat_reduced"}),
        ("Skimmed Coconut Water", "juice nectars", set()),
    ],
)
def test_text_attributes(name, aisle, expected):
    assert flags_for(name, aisle) == expected


def test_percent_fat_only_applies_inside_configured_aisles():
    assert flags_for("Orange Juice 2% Pulp", "juice nectars") == set()
    assert flags_for("Greek Yogurt 0% Plain", "yogurt") == {"fat_reduced"}
    assert flags_for("Greek Yogurt 0 % Plain", "yogurt") == {"fat_reduced"}


def test_two_percent_only_diagnostic_counts_names_without_other_fat_terms():
    items = pd.DataFrame(
        {
            "product_name": ["2% Lactose Free Milk", "2% Reduced Fat Milk", "1% Lowfat Milk", "Diet Cola"],
            "fine_category": ["milk", "milk", "milk", "soft drinks"],
        }
    )
    config = {"attributes": {"percent_fat_aisles": ["milk"]}}
    diagnostics = instacart_text_diagnostics(items, config)
    assert diagnostics["fat_reduced_two_percent_only"] == 1
    assert diagnostics["diet_items_not_used"] == 1


def category_items(n_with, n_without, category):
    flags = [1] * n_with + [0] * n_without
    return pd.DataFrame({"fine_category": [category] * len(flags), "attr_organic": flags})


def test_applicable_categories_threshold_boundaries():
    items = pd.concat(
        [
            category_items(4, 5, "four_with"),
            category_items(5, 5, "five_each"),
            category_items(5, 4, "four_without"),
        ],
        ignore_index=True,
    )
    table = applicable_categories(items, ["organic"], min_with=5, min_without=5)
    included = table.set_index("fine_category")["included"].to_dict()
    assert included == {"five_each": True, "four_with": False, "four_without": False}
    assert table.set_index("fine_category").loc["four_with", "n_with"] == 4
    assert category_sets(table) == {"organic": {"five_each"}}


def test_complete_journey_attributes_use_brand_and_category_text():
    items = pd.DataFrame(
        {
            "item_id": ["1", "2", "3"],
            "BRAND": ["Private", "National", "National"],
            "coarse_category": ["MILK", "ORGANICS FRUIT & VEGETABLES", "BREAD"],
            "fine_category": ["FLUID MILK", "ORGANIC FRUIT", "BREAD:ORGANIC"],
        }
    )
    config = {"attributes": {"complete_journey": ["private_label", "organic"], "private_label_brands": ["Private"]}}
    result = complete_journey_attributes(items, config)
    assert result["attr_private_label"].tolist() == [1, 0, 0]
    assert result["attr_organic"].tolist() == [0, 1, 1]
    assert result["attr_organic"].dtype == "int8"
