import pandas as pd
import pytest

from ml_group9.attributes import TEXT_PATTERNS, _contains
from ml_group9.cj_keyword_audit import (
    HEALTH_ATTRIBUTES,
    Keyword,
    column_counts,
    commodity_structure,
    implicit_history,
    keyword_list,
    keyword_matches,
    keyword_summary,
    match,
    name_columns,
    pattern_report,
)
from ml_group9.hstar import taste_attributes


TASTE = ["chocolate", "vanilla", "strawberry"]
TEXTS = pd.Series(
    ["CHOCOLATE MILK", "CHOCOLATEY", "VANILLA", "FAT FREE YOGURT", "LOW FAT", "LOWFAT", "NON-FAT", "GLUTEN-FREE", ""],
    dtype=object,
)


def test_primary_patterns_are_the_imported_ones():
    report = pattern_report(TASTE)
    for name in HEALTH_ATTRIBUTES:
        assert report["primary"]["health"]["patterns"][name] is TEXT_PATTERNS[name]
        assert match(TEXTS, Keyword("primary", "health", name)).tolist() == _contains(TEXTS, TEXT_PATTERNS[name]).tolist()
    expected = taste_attributes(pd.DataFrame({"product_name": TEXTS}), TASTE)
    for name in TASTE:
        assert match(TEXTS, Keyword("primary", "taste", name)).astype(int).tolist() == expected[f"attr_{name}"].tolist()
    assert "re.escape(name)" in report["primary"]["taste"]["pattern_line"]


def test_word_boundaries_and_layers_stay_separate():
    def hit(text, layer, group, name):
        return bool(match(pd.Series([text], dtype=object), Keyword(layer, group, name)).iloc[0])

    assert hit("CHOCOLATE MILK", "primary", "taste", "chocolate")
    assert not hit("CHOCOLATEY", "primary", "taste", "chocolate")
    assert not hit("VANILLA", "secondary", "taste", "VAN")
    assert hit("VAN CRM SANDWICH", "secondary", "taste", "VAN")
    assert hit("LOW FAT", "primary", "health", "fat_reduced")
    assert not hit("LOW FAT", "secondary", "health", "LOWFAT")
    assert hit("lowfat", "secondary", "health", "LOWFAT")
    # The Instacart pattern low[- ]?fat also covers LOWFAT, which is why layers are never added together.
    assert hit("LOWFAT", "primary", "health", "fat_reduced")
    keys = [keyword.key for keyword in keyword_list(TASTE)]
    assert len(keys) == len(set(keys))


def test_category_matches_expand_to_products_and_transaction_rows():
    products = pd.DataFrame(
        {
            "PRODUCT_ID": ["1", "2", "3", "4", "5"],
            "DEPARTMENT": ["GROCERY", "GROCERY", "DRUG GM", "GROCERY", pd.NA],
            "COMMODITY_DESC": ["MILK", "MILK", "MILK", "MILK", "CANDY"],
            "SUB_COMMODITY_DESC": ["LOWFAT MILK", "LOWFAT MILK", "LOWFAT MILK", "WHOLE MILK", "CHOCOLATE BARS"],
        }
    )
    cleaned = pd.DataFrame({"PRODUCT_ID": ["1", "1", "1", "2", "4", "4"]})
    keywords = [Keyword("secondary", "health", "LOWFAT"), Keyword("primary", "taste", "chocolate")]
    matches = keyword_matches(column_counts(products, cleaned, "SUB_COMMODITY_DESC"), "SUB_COMMODITY_DESC", keywords)
    rows = {
        (row.keyword, row.department): (row.products, row.transaction_rows_reference)
        for row in matches.itertuples(index=False)
    }
    assert rows == {("LOWFAT", "GROCERY"): (2, 4), ("LOWFAT", "DRUG GM"): (1, 0), ("chocolate", "<missing>"): (1, 0)}

    summary = {row["keyword"]: row for row in keyword_summary(matches, keywords, ["SUB_COMMODITY_DESC"], 5, 6)}
    assert summary["LOWFAT"]["unique_strings"] == 1
    assert summary["LOWFAT"]["products"] == 3
    assert summary["LOWFAT"]["product_share"] == pytest.approx(3 / 5)
    assert summary["LOWFAT"]["transaction_rows_reference"] == 4
    assert summary["LOWFAT"]["transaction_row_share_reference"] == pytest.approx(4 / 6)
    assert summary["chocolate"]["products"] == 1
    assert summary["chocolate"]["transaction_rows_reference"] == 0


def test_commodity_applicability_boundary_and_household_counts():
    rows = []
    for commodity, n_with, n_without in [("A", 5, 5), ("B", 4, 5), ("C", 5, 4)]:
        rows += [(commodity, f"{commodity} LOWFAT") for _ in range(n_with)]
        rows += [(commodity, f"{commodity} REGULAR") for _ in range(n_without)]
    products = pd.DataFrame(rows, columns=["COMMODITY_DESC", "SUB_COMMODITY_DESC"])
    products.insert(0, "PRODUCT_ID", [str(i) for i in range(len(products))])
    products["DEPARTMENT"] = "GROCERY"
    flag = match(products["SUB_COMMODITY_DESC"].astype(object), Keyword("secondary", "health", "LOWFAT"))

    in_a = products.loc[products["COMMODITY_DESC"].eq("A"), "PRODUCT_ID"].tolist()
    in_b = products.loc[products["COMMODITY_DESC"].eq("B"), "PRODUCT_ID"].tolist()
    lines = [("u1", f"b{i}", in_a[i]) for i in range(10)]  # 10 purchases inside C_a
    lines += [("u2", f"b{i}", in_a[i]) for i in range(9)] + [("u2", "b0", in_a[0])]  # 9 after deduplication
    lines += [("u3", f"b{i}", in_b[i]) for i in range(9)] + [("u3", "b9", in_b[0])]  # 10, but outside C_a
    cleaned = pd.DataFrame(lines, columns=["household_key", "BASKET_ID", "PRODUCT_ID"])

    result = commodity_structure(products, implicit_history(cleaned), flag, "secondary:LOWFAT", 5, 5, [])
    assert result["commodities_with_matched_sub"] == 3
    assert result["commodities_mixed"] == 3
    assert result["applicable_commodities"] == 1
    assert [row["commodity"] for row in result["applicable_list"]] == ["A"]
    assert result["households_n_c_at_least_10_reference"] == 1


def test_name_columns_flag_only_unexpected_description_columns():
    assert name_columns(["PRODUCT_ID", "COMMODITY_DESC", "SUB_COMMODITY_DESC", "BRAND"]) == []
    assert name_columns(["PRODUCT_ID", "PRODUCT_NAME", "ITEM_DESCRIPTION"]) == ["PRODUCT_NAME", "ITEM_DESCRIPTION"]
