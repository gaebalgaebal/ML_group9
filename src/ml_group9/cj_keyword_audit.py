"""Keyword audit of Complete Journey category text for health and taste terms.

Complete Journey has no product-name column, so the Instacart name-based attributes cannot be built
for it. This module counts health and taste keywords in the CJ category text columns as descriptive
evidence only; no model, C_a set or H* result depends on it.

Two keyword layers are counted separately and never merged:
- primary: the Instacart definitions, imported from attributes.TEXT_PATTERNS (health) and applied
  through hstar.taste_attributes (taste);
- secondary: CJ spelling variants, matched as whole words ignoring case; prone to false positives.

A category-level match gives every product of that category the same value, which differs from the
product-level Instacart attributes. Transaction counts use the cleaned full period, target weeks
included, and are for reference only.

Input: the CJ ZIP under data/raw/. Output: reports/metrics/cj_keyword_audit.json and
reports/tables/cj_keyword_matches.csv (no profile prefix). Entry point: `python -m ml_group9.cj_keyword_audit`.
"""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import re
import time
import zipfile
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from .attributes import (
    TEXT_PATTERNS,
    _contains,
    applicable_categories,
    attribute_column,
    exclude_nonfood_categories,
    preference_counts,
)
from .config import load_config
from .data import (
    COMPLETE_JOURNEY_MEMBERS,
    _read_complete_journey,
    _read_csv,
    clean_complete_journey,
    discover_archive,
)
from .hstar import taste_attributes
from .pipeline import _write_json


PROFILES = ["sample", "medium", "full"]
HEALTH_ATTRIBUTES = ["fat_reduced", "sugar_reduced", "gluten_free", "low_sodium"]
SECONDARY_TERMS = {
    "health": [
        "LOWFAT", "NONFAT", "NON-FAT", "FAT FREE", "REDUCED FAT", "SKIM", "LITE", "LIGHT", "DIET",
        "SUGAR FREE", "NO SUGAR", "UNSWEETENED", "GLUTEN FREE", "LOW SODIUM", "LOW SALT", "NO SALT",
        "REDUCED SODIUM", "LESS SODIUM",
    ],
    "taste": ["CHOC", "CHOCOLATE", "VANILLA", "VAN", "STRAWBERRY", "STRAWB", "STRWBRY"],
}
EXPECTED_PRODUCT_COLUMNS = [
    "PRODUCT_ID", "MANUFACTURER", "DEPARTMENT", "BRAND", "COMMODITY_DESC", "SUB_COMMODITY_DESC",
    "CURR_SIZE_OF_PRODUCT",
]
MAIN_COLUMNS = ["COMMODITY_DESC", "SUB_COMMODITY_DESC"]
REFERENCE_COLUMNS = ["DEPARTMENT", "BRAND", "CURR_SIZE_OF_PRODUCT"]
# Column names that would mean a product name or description exists, which this audit assumes it does not.
NAME_COLUMN_PATTERN = r"NAME|DESC|TITLE|LABEL"
MISSING = "<missing>"
TOP_STRINGS = 10
MIN_HOUSEHOLD_PURCHASES = 10
CATEGORY_LEVEL_NOTE = (
    "A category-level match assigns the same value to every product in that category; this differs "
    "from the product-level, product-name-based Instacart attributes."
)


@dataclass(frozen=True)
class Keyword:
    layer: str  # "primary" (Instacart definition) or "secondary" (CJ spelling variant)
    group: str  # "health" or "taste"
    name: str

    @property
    def key(self) -> str:
        return f"{self.layer}:{self.name}"


def keyword_list(taste_names: list[str]) -> list[Keyword]:
    primary = [Keyword("primary", "health", name) for name in HEALTH_ATTRIBUTES]
    primary += [Keyword("primary", "taste", name) for name in taste_names]
    secondary = [Keyword("secondary", group, term) for group, terms in SECONDARY_TERMS.items() for term in terms]
    return primary + secondary


def secondary_pattern(term: str) -> str:
    return rf"\b{re.escape(term)}\b"


def match(values: pd.Series, keyword: Keyword) -> pd.Series:
    """Boolean mask of values that contain the keyword under its layer's definition."""
    if keyword.layer == "primary" and keyword.group == "health":
        return _contains(values, TEXT_PATTERNS[keyword.name]).astype(bool)
    if keyword.layer == "primary":
        flags = taste_attributes(pd.DataFrame({"product_name": values}), [keyword.name])
        return flags[attribute_column(keyword.name)].astype(bool)
    return _contains(values, secondary_pattern(keyword.name)).astype(bool)


def taste_pattern_source() -> str:
    """The pattern line of hstar.taste_attributes, verbatim, for the report."""
    lines = [line.strip() for line in inspect.getsource(taste_attributes).splitlines() if "pattern =" in line]
    if len(lines) != 1:
        raise RuntimeError("Could not locate the pattern line in hstar.taste_attributes")
    return lines[0]


def pattern_report(taste_names: list[str]) -> dict[str, object]:
    return {
        "primary": {
            "health": {
                "source": "ml_group9.attributes.TEXT_PATTERNS via attributes._contains (case=False, regex=True)",
                "patterns": {name: TEXT_PATTERNS[name] for name in HEALTH_ATTRIBUTES},
            },
            "taste": {
                "source": "ml_group9.hstar.taste_attributes (text column product_name := category text)",
                "pattern_line": taste_pattern_source(),
                "names": list(taste_names),
            },
        },
        "secondary": {
            group: {term: secondary_pattern(term) for term in terms} for group, terms in SECONDARY_TERMS.items()
        },
        "secondary_flags": "attributes._contains (case=False, regex=True)",
    }


def name_columns(columns: list[str]) -> list[str]:
    """Columns that look like a product name or description, beyond the expected category columns."""
    return [
        column for column in columns
        if column not in EXPECTED_PRODUCT_COLUMNS and re.search(NAME_COLUMN_PATTERN, column, re.IGNORECASE)
    ]


def text_columns(frame: pd.DataFrame) -> list[str]:
    return [column for column in frame.columns if not pd.api.types.is_numeric_dtype(frame[column])]


def column_counts(products: pd.DataFrame, cleaned: pd.DataFrame, column: str) -> pd.DataFrame:
    """Products and cleaned transaction rows per (category string, DEPARTMENT) of one column."""
    table = pd.DataFrame(
        {
            "PRODUCT_ID": products["PRODUCT_ID"].to_numpy(),
            "value": products[column].to_numpy(),
            "department": products["DEPARTMENT"].astype("string").fillna(MISSING).to_numpy(),
        }
    ).dropna(subset=["value"])
    table["value"] = table["value"].astype(str)
    product_counts = table.groupby(["value", "department"]).size().rename("products")
    rows = cleaned[["PRODUCT_ID"]].merge(table, on="PRODUCT_ID", how="inner")
    row_counts = rows.groupby(["value", "department"]).size().rename("transaction_rows")
    counts = pd.concat([product_counts, row_counts], axis=1).fillna(0).astype("int64")
    return counts.reset_index()


def keyword_matches(counts: pd.DataFrame, column: str, keywords: list[Keyword]) -> pd.DataFrame:
    """One row per (keyword, matched category string, DEPARTMENT) of one column."""
    values = pd.Series(sorted(counts["value"].unique()), dtype=object)
    pieces: list[pd.DataFrame] = []
    for keyword in keywords:
        matched = set(values[match(values, keyword).to_numpy()])
        part = counts[counts["value"].isin(matched)]
        pieces.append(
            pd.DataFrame(
                {
                    "column": column,
                    "category_string": part["value"].to_numpy(),
                    "keyword": keyword.name,
                    "layer": keyword.layer,
                    "group": keyword.group,
                    "department": part["department"].to_numpy(),
                    "products": part["products"].to_numpy(),
                    "transaction_rows_reference": part["transaction_rows"].to_numpy(),
                }
            )
        )
    return pd.concat(pieces, ignore_index=True)


def keyword_summary(
    matches: pd.DataFrame, keywords: list[Keyword], columns: list[str], total_products: int, total_rows: int
) -> list[dict[str, object]]:
    """Keyword x column x layer: unique strings, products and transaction rows (with shares)."""
    summary: list[dict[str, object]] = []
    for keyword in keywords:
        for column in columns:
            part = matches[
                matches["column"].eq(column) & matches["layer"].eq(keyword.layer) & matches["keyword"].eq(keyword.name)
            ]
            by_string = (
                part.groupby("category_string")[["products", "transaction_rows_reference"]].sum().reset_index()
                .sort_values(["products", "category_string"], ascending=[False, True])
            )
            products = int(part["products"].sum())
            rows = int(part["transaction_rows_reference"].sum())
            summary.append(
                {
                    "layer": keyword.layer,
                    "group": keyword.group,
                    "keyword": keyword.name,
                    "column": column,
                    "column_role": "main" if column in MAIN_COLUMNS else "reference",
                    "unique_strings": int(len(by_string)),
                    "products": products,
                    "product_share": products / total_products if total_products else 0.0,
                    "transaction_rows_reference": rows,
                    "transaction_row_share_reference": rows / total_rows if total_rows else 0.0,
                    "top_strings": [
                        {"category_string": value, "products": int(n), "transaction_rows_reference": int(r)}
                        for value, n, r in by_string.head(TOP_STRINGS).itertuples(index=False)
                    ],
                }
            )
    return summary


def implicit_history(cleaned: pd.DataFrame) -> pd.DataFrame:
    """Cleaned transactions in the (user, basket, item) form used by preference_counts."""
    history = pd.DataFrame(
        {"user_id": cleaned["household_key"], "basket_id": cleaned["BASKET_ID"], "item_id": cleaned["PRODUCT_ID"]}
    )
    return history.astype("string").drop_duplicates()


def commodity_structure(
    products: pd.DataFrame,
    history: pd.DataFrame,
    flag: pd.Series,
    name: str,
    min_with: int,
    min_without: int,
    nonfood_departments: list[str],
    min_purchases: int = MIN_HOUSEHOLD_PURCHASES,
) -> dict[str, object]:
    """Descriptive structure of a SUB_COMMODITY-level flag inside each COMMODITY_DESC.

    The Instacart C_a rule is applied with COMMODITY_DESC as the category; households are counted by
    n_C (purchases inside C_a, deduplicated by household, basket and product) over the full period.
    """
    pairs = pd.DataFrame(
        {"commodity": products["COMMODITY_DESC"], "sub": products["SUB_COMMODITY_DESC"], "flag": flag.to_numpy()}
    ).dropna(subset=["commodity", "sub"]).drop_duplicates(["commodity", "sub"])
    per_commodity = pairs.groupby("commodity")["flag"].agg(["any", "all"])
    items = pd.DataFrame(
        {
            "item_id": products["PRODUCT_ID"].astype("string").to_numpy(),
            "fine_category": products["COMMODITY_DESC"].to_numpy(),
            "DEPARTMENT": products["DEPARTMENT"].to_numpy(),
            attribute_column(name): flag.astype("int8").to_numpy(),
        }
    )
    table = applicable_categories(items, [name], min_with, min_without)
    annotated = exclude_nonfood_categories(table, items, nonfood_departments)
    applicable = annotated[table["included"].to_numpy()]
    categories = {name: set(applicable["fine_category"].astype(str))}
    counts = preference_counts(history, items, categories)
    return {
        "commodities_with_matched_sub": int(per_commodity["any"].sum()),
        "commodities_mixed": int((per_commodity["any"] & ~per_commodity["all"]).sum()),
        "applicable_commodities": int(len(applicable)),
        "applicable_commodities_food_only": int((~applicable["nonfood"]).sum()),
        f"households_n_c_at_least_{min_purchases}_reference": int(counts["n_c"].ge(min_purchases).sum()),
        "applicable_list": [
            {
                "commodity": str(row.fine_category),
                "n_with": int(row.n_with),
                "n_without": int(row.n_without),
                "department": str(row.department),
                "nonfood": bool(row.nonfood),
            }
            for row in applicable.itertuples(index=False)
        ],
    }


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def member_sha256(archive_path: Path, member: str) -> str:
    digest = hashlib.sha256()
    with zipfile.ZipFile(archive_path) as archive, archive.open(member) as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def schema_report(products: pd.DataFrame, transactions: pd.DataFrame, transaction_header: list[str]) -> dict[str, object]:
    return {
        "product": {
            "rows": int(len(products)),
            "dtypes": {column: str(dtype) for column, dtype in products.dtypes.items()},
            "duplicated_product_ids": int(products["PRODUCT_ID"].duplicated().sum()),
            "text_unique_values": {
                column: int(products[column].nunique(dropna=True))
                for column in text_columns(products) if column != "PRODUCT_ID"
            },
            "missing_expected_columns": [c for c in EXPECTED_PRODUCT_COLUMNS if c not in products.columns],
            "unexpected_columns": [c for c in products.columns if c not in EXPECTED_PRODUCT_COLUMNS],
        },
        "transaction_data": {
            "rows": int(len(transactions)),
            "columns_in_file": transaction_header,
            "dtypes_of_columns_read": {column: str(dtype) for column, dtype in transactions.dtypes.items()},
        },
    }


def run_audit(config: dict[str, object]) -> dict[str, object]:
    start = time.perf_counter()
    raw_dir = Path(config["raw_dir"])
    products, transactions = _read_complete_journey(raw_dir)
    archive_path = discover_archive(raw_dir, COMPLETE_JOURNEY_MEMBERS)
    with zipfile.ZipFile(archive_path) as archive:
        transaction_header = list(_read_csv(archive, "transaction_data.csv", nrows=0).columns)
    schema = schema_report(products, transactions, transaction_header)
    print("Schema:", json.dumps(schema, indent=2), sep="\n")
    found = name_columns(list(products.columns)) + name_columns(transaction_header)
    if found:
        raise SystemExit(f"Product name/description columns exist ({found}); the audit premise does not hold.")

    exclude = list(config["complete_journey"]["exclude_departments"])
    cleaned, cleaning = clean_complete_journey(transactions, products, exclude)
    columns = [c for c in MAIN_COLUMNS + REFERENCE_COLUMNS if c in text_columns(products)]
    keywords = keyword_list(list(config["hierarchical"]["hstar"]["taste_attributes"]))
    matches = pd.concat(
        [keyword_matches(column_counts(products, cleaned, column), column, keywords) for column in columns],
        ignore_index=True,
    ).sort_values(["layer", "group", "keyword", "column", "products", "category_string"],
                  ascending=[True, True, True, True, False, True], ignore_index=True)
    summary = keyword_summary(matches, keywords, columns, len(products), len(cleaned))

    settings = config["attributes"]
    history = implicit_history(cleaned)
    sub_values = pd.Series(products["SUB_COMMODITY_DESC"].fillna("").astype(str).to_numpy(), dtype=object)
    structure: list[dict[str, object]] = []
    for keyword in keywords:
        flag = match(sub_values, keyword) & products["SUB_COMMODITY_DESC"].notna().to_numpy()
        if not flag.any():
            continue
        structure.append(
            {
                "layer": keyword.layer,
                "group": keyword.group,
                "keyword": keyword.name,
                "matched_sub_commodity_strings": int(products.loc[flag.to_numpy(), "SUB_COMMODITY_DESC"].nunique()),
                "matched_products": int(flag.sum()),
                **commodity_structure(
                    products, history, flag, keyword.key, int(settings["min_with"]), int(settings["min_without"]),
                    list(settings["exclude_departments"]["complete_journey"]),
                ),
            }
        )

    report_dir = Path(config["report_dir"])
    table_path = report_dir / "tables" / "cj_keyword_matches.csv"
    table_path.parent.mkdir(parents=True, exist_ok=True)
    matches.to_csv(table_path, index=False, encoding="utf-8")
    config_path = Path(config["project_root"]) / "configs" / f"{config['profile']}.yaml"
    report = {
        "profile": config["profile"],
        "inputs": {
            "archive": {"path": str(archive_path), "sha256": sha256(archive_path)},
            "members": {member: member_sha256(archive_path, member) for member in sorted(COMPLETE_JOURNEY_MEMBERS)},
            "config": {"path": str(config_path), "sha256": sha256(config_path)},
        },
        "schema": schema,
        "product_name_columns": found,
        "cleaning": cleaning,
        "audited_columns": {"main": [c for c in columns if c in MAIN_COLUMNS],
                            "reference": [c for c in columns if c in REFERENCE_COLUMNS]},
        "denominators": {"products_in_product_csv": int(len(products)), "cleaned_transaction_rows": int(len(cleaned))},
        "patterns": pattern_report(list(config["hierarchical"]["hstar"]["taste_attributes"])),
        "keywords": summary,
        "commodity_structure": structure,
        "notes": {
            "category_level": CATEGORY_LEVEL_NOTE,
            "transactions": "Cleaned full period including target weeks; reference only.",
            "layers": "Primary and secondary matches are counted separately and must not be added together.",
            "households": f"n_C >= {MIN_HOUSEHOLD_PURCHASES} inside C_a, deduplicated by household, basket and product.",
        },
        "outputs": {"matches_csv": str(table_path)},
    }
    report["elapsed_seconds"] = time.perf_counter() - start
    _write_json(report_dir / "metrics" / "cj_keyword_audit.json", report)
    return report


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Count health and taste keywords in Complete Journey category text")
    parser.add_argument("--profile", choices=PROFILES, default="medium")
    args = parser.parse_args(argv)
    report = run_audit(load_config(args.profile))
    table = pd.DataFrame(report["keywords"]).drop(columns="top_strings")
    with pd.option_context("display.width", 200, "display.max_rows", None):
        print(table[table["unique_strings"].gt(0)].to_string(index=False))
    print(f"elapsed_seconds={report['elapsed_seconds']:.1f}")


if __name__ == "__main__":
    main()
