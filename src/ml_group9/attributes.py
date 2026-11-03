"""Product attribute extraction, applicable categories, and history-based preference counts."""

from __future__ import annotations

import re

import numpy as np
import pandas as pd


ATTRIBUTE_NAMES = [
    "fat_reduced",
    "sugar_reduced",
    "gluten_free",
    "low_sodium",
    "organic",
    "private_label",
]
TEXT_ATTRIBUTES = ["fat_reduced", "sugar_reduced", "gluten_free", "low_sodium", "organic"]

# Terms such as "light", "lite", "diet" and "lightly salted" are intentionally absent:
# they describe roast, colour, sweeteners, or a salted product rather than a reduced nutrient.
TEXT_PATTERNS = {
    "fat_reduced": r"\b(?:fat[- ]free|non[- ]?fat|skim|low[- ]?fat|reduced fat)\b",
    "sugar_reduced": r"\b(?:unsweetened|sugar[- ]free|no sugar added|no added sugar|reduced sugar)\b",
    "gluten_free": r"\bgluten[- ]free\b",
    "low_sodium": r"\b(?:low sodium|reduced sodium|no salt added|unsalted)\b",
    "organic": r"\borganic\b",
}
DIET_PATTERN = r"\bdiet\b"


def attribute_column(name: str) -> str:
    return f"attr_{name}"


def _contains(text: pd.Series, pattern: str) -> pd.Series:
    return text.fillna("").str.contains(pattern, case=False, regex=True)


def percent_fat_pattern(levels: list[int]) -> str:
    """Match "0%", "1 %", "2%" without matching "100%" or "1.5%".

    A trailing word boundary is not used because "%" followed by a space has none.
    """
    alternatives = "|".join(re.escape(str(level)) for level in sorted(set(levels)))
    return rf"(?<![\d.])(?:{alternatives})\s?%"


def text_attribute_flags(
    names: pd.Series,
    fine_categories: pd.Series,
    percent_fat_aisles: list[str],
    percent_fat_levels: list[int],
) -> pd.DataFrame:
    """Flag text attributes in product names; percent fat applies only in listed aisles."""
    flags = pd.DataFrame(index=names.index)
    for name in TEXT_ATTRIBUTES:
        flags[name] = _contains(names, TEXT_PATTERNS[name])
    if percent_fat_levels:
        in_aisle = fine_categories.isin(percent_fat_aisles)
        flags["fat_reduced"] |= in_aisle & _contains(names, percent_fat_pattern(percent_fat_levels))
    return flags.astype("int8")


def instacart_attributes(items: pd.DataFrame, config: dict[str, object]) -> pd.DataFrame:
    """Add attr_<name> columns for the configured Instacart attributes."""
    settings = config["attributes"]
    flags = text_attribute_flags(
        items["product_name"],
        items["fine_category"],
        list(settings["percent_fat_aisles"]),
        [int(level) for level in settings["percent_fat_levels"]],
    )
    result = items.copy()
    for name in settings["instacart"]:
        result[attribute_column(name)] = flags[name]
    return result


def complete_journey_attributes(items: pd.DataFrame, config: dict[str, object]) -> pd.DataFrame:
    """Add attr_<name> columns for the configured Complete Journey attributes."""
    settings = config["attributes"]
    available = {
        "private_label": items["BRAND"].isin(list(settings["private_label_brands"])),
        "organic": _contains(items["coarse_category"], r"organic") | _contains(items["fine_category"], r"organic"),
    }
    result = items.copy()
    for name in settings["complete_journey"]:
        if name not in available:
            raise ValueError(f"Attribute {name!r} is not available for Complete Journey")
        result[attribute_column(name)] = available[name].astype("int8")
    return result


def used_attributes(items: pd.DataFrame) -> list[str]:
    """Attributes materialized as attr_<name> columns in an item table."""
    return [name for name in ATTRIBUTE_NAMES if attribute_column(name) in items.columns]


def applicable_categories(
    items: pd.DataFrame, attribute_names: list[str], min_with: int, min_without: int
) -> pd.DataFrame:
    """Fine categories where an attribute is a real choice, using the item list only."""
    fine = items["fine_category"].astype("string").fillna("UNKNOWN")
    rows: list[pd.DataFrame] = []
    for name in attribute_names:
        flag = items[attribute_column(name)].astype(int)
        counts = (
            pd.DataFrame({"fine_category": fine, "flag": flag})
            .groupby("fine_category", observed=True)["flag"]
            .agg(n_with="sum", n_total="size")
            .reset_index()
        )
        counts["n_without"] = counts["n_total"] - counts["n_with"]
        counts["included"] = counts["n_with"].ge(min_with) & counts["n_without"].ge(min_without)
        counts.insert(0, "attribute", name)
        rows.append(counts[["attribute", "fine_category", "n_with", "n_without", "included"]])
    if not rows:
        return pd.DataFrame(columns=["attribute", "fine_category", "n_with", "n_without", "included"])
    table = pd.concat(rows, ignore_index=True)
    table["n_with"] = table["n_with"].astype("int64")
    table["n_without"] = table["n_without"].astype("int64")
    return table


def category_sets(table: pd.DataFrame) -> dict[str, set[str]]:
    """Map each attribute to its applicable fine categories C_a."""
    sets: dict[str, set[str]] = {str(name): set() for name in table["attribute"].unique()}
    for name, fine, included in zip(table["attribute"], table["fine_category"], table["included"]):
        if included:
            sets[str(name)].add(str(fine))
    return sets


def preference_counts(
    history: pd.DataFrame, items: pd.DataFrame, categories: dict[str, set[str]]
) -> pd.DataFrame:
    """Per-user n_C (purchases inside C_a) and n_a (attribute purchases inside C_a).

    Purchases are history rows deduplicated by (user, basket, item); targets are never used.
    """
    implicit = history[["user_id", "basket_id", "item_id"]].astype("string").drop_duplicates()
    item_table = items.copy()
    item_table["item_id"] = item_table["item_id"].astype("string")
    item_table["fine_category"] = item_table["fine_category"].astype("string").fillna("UNKNOWN")
    pieces: list[pd.DataFrame] = []
    for name, applicable in categories.items():
        column = attribute_column(name)
        lookup = item_table[["item_id", "fine_category", column]]
        joined = implicit.merge(lookup, on="item_id", how="inner")
        joined = joined[joined["fine_category"].isin(applicable)]
        counts = (
            joined.groupby("user_id", observed=True)[column]
            .agg(n_a="sum", n_c="size")
            .reset_index()
        )
        counts.insert(1, "attribute", name)
        pieces.append(counts)
    if not pieces:
        return pd.DataFrame(columns=["user_id", "attribute", "n_a", "n_c"])
    result = pd.concat(pieces, ignore_index=True)
    result["n_a"] = result["n_a"].astype("int64")
    result["n_c"] = result["n_c"].astype("int64")
    return result


def population_rates(counts: pd.DataFrame, attribute_names: list[str]) -> dict[str, float]:
    """pi_bar(a) = sum_u n_a / sum_u n_C; 0.5 when no user bought inside C_a."""
    rates: dict[str, float] = {}
    for name in attribute_names:
        rows = counts[counts["attribute"].eq(name)]
        total = int(rows["n_c"].sum())
        rates[name] = float(rows["n_a"].sum()) / total if total else 0.5
    return rates


def attribute_report(
    items: pd.DataFrame,
    history: pd.DataFrame,
    table: pd.DataFrame,
    min_n: int,
) -> dict[str, dict[str, object]]:
    """Coverage of each used attribute in the item list and in history purchases."""
    attribute_names = used_attributes(items)
    categories = category_sets(table)
    counts = preference_counts(history, items, categories)
    rates = population_rates(counts, attribute_names)
    implicit = history[["user_id", "basket_id", "item_id"]].astype("string").drop_duplicates()
    item_table = items.copy()
    item_table["item_id"] = item_table["item_id"].astype("string")
    purchases = implicit.merge(item_table, on="item_id", how="left")
    report: dict[str, dict[str, object]] = {}
    for name in attribute_names:
        column = attribute_column(name)
        rows = counts[counts["attribute"].eq(name)]
        report[name] = {
            "items_with_attribute": int(items[column].sum()),
            "items_total": int(len(items)),
            "applicable_categories": len(categories.get(name, set())),
            "items_in_applicable_categories": int(
                items["fine_category"].astype("string").fillna("UNKNOWN").isin(categories.get(name, set())).sum()
            ),
            "history_purchase_share": float(purchases[column].fillna(0).mean()) if len(purchases) else 0.0,
            "history_purchases_in_applicable_categories": int(rows["n_c"].sum()),
            "pi_bar": rates[name],
            "users_with_n_c_at_least_min": int(rows["n_c"].ge(min_n).sum()),
            "history_users": int(implicit["user_id"].nunique()),
        }
    return report


def instacart_text_diagnostics(items: pd.DataFrame, config: dict[str, object]) -> dict[str, int]:
    """Counts the team needs to decide on percent-fat and diet handling."""
    settings = config["attributes"]
    names = items["product_name"]
    in_aisle = items["fine_category"].isin(list(settings["percent_fat_aisles"]))
    keyword = _contains(names, TEXT_PATTERNS["fat_reduced"])
    two = in_aisle & _contains(names, percent_fat_pattern([2]))
    zero_one = in_aisle & _contains(names, percent_fat_pattern([0, 1]))
    any_percent = in_aisle & _contains(names, percent_fat_pattern([0, 1, 2]))
    return {
        "fat_reduced_keyword_only": int((keyword & ~any_percent).sum()),
        "fat_reduced_keyword_and_percent": int((keyword & any_percent).sum()),
        "fat_reduced_percent_only": int((~keyword & any_percent).sum()),
        "fat_reduced_two_percent_only": int((two & ~keyword & ~zero_one).sum()),
        "diet_items_not_used": int(_contains(names, DIET_PATTERN).sum()),
    }


def complete_journey_text_diagnostics(items: pd.DataFrame) -> dict[str, int]:
    """Instacart name patterns applied to CJ category text; reported, never modelled."""
    text = items["coarse_category"].fillna("").astype(str) + " " + items["fine_category"].fillna("").astype(str)
    result = {f"{name}_category_text": int(_contains(text, TEXT_PATTERNS[name]).sum()) for name in TEXT_ATTRIBUTES}
    result["diet_category_text"] = int(_contains(text, DIET_PATTERN).sum())
    return result
