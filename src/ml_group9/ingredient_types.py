"""Ingredient-type mapping import and type-level applicable attribute categories C_a^type.

The mapping parquet is produced by the food_recipe_preprocessing_gu project, which is read only; a
copy is kept under data/raw/mapping/. Types are nested in stage-1 keys, so the unit everywhere is
the (stage-1 key, type) pair. C_a^type uses the item list only.
Entry point: `python -m ml_group9 prepare-types`, which adds type columns to
data/processed/<profile>/*/items.parquet and writes reports/metrics/<profile>_types.json.
"""

from __future__ import annotations

import json
import shutil
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from .attributes import attribute_column, item_departments
from .modeling import file_checksum


MAPPING_COLUMNS = ["product_id", "mapping_status", "ingredient_id", "mapping_note", "mapping_version"]
TYPE_COLUMNS = ["ingredient_type", "type_source", "mapping_status", "mapping_note"]
FALLBACK_PREFIX = "aisle:"
# Status given to catalogue items that have no mapping row.
MISSING_STATUS = "missing"


def read_mapping(path: Path, expected_version: str) -> tuple[pd.DataFrame, dict[str, object]]:
    """Read the mapping, require one row per product and warn when mapping_version differs."""
    frame = pd.read_parquet(path)
    missing = [column for column in MAPPING_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(f"Mapping is missing columns {missing}")
    duplicated = frame["product_id"].duplicated(keep=False)
    if duplicated.any():
        examples = frame.loc[duplicated, "product_id"].drop_duplicates().head(10).tolist()
        raise ValueError(f"Mapping has {int(duplicated.sum())} rows for repeated products, e.g. {examples}")
    versions = sorted(str(value) for value in frame["mapping_version"].dropna().unique())
    if versions != [expected_version]:
        warnings.warn(f"mapping_version {versions} differs from expected {expected_version!r}", stacklevel=2)
    schema = {
        "columns": {column: str(dtype) for column, dtype in frame.dtypes.items()},
        "rows": int(len(frame)),
        "mapping_version": versions,
        "version_matches": versions == [expected_version],
    }
    return frame, schema


def import_mapping(source: Path, destination: Path, expected_version: str) -> tuple[pd.DataFrame, dict[str, object]]:
    """Copy the mapping into the project (the source is only read), check sha256, and read the copy.

    When the source is unavailable an existing copy is used and the report says so.
    """
    destination.parent.mkdir(parents=True, exist_ok=True)
    source_sha = file_checksum(source) if source.exists() else None
    if source_sha is None:
        if not destination.exists():
            raise FileNotFoundError(f"Neither the mapping source {source} nor a copy at {destination} exists")
        warnings.warn(f"Mapping source {source} not found; using the existing copy", stacklevel=2)
    elif not destination.exists() or file_checksum(destination) != source_sha:
        shutil.copyfile(source, destination)
    copy_sha = file_checksum(destination)
    if source_sha is not None and copy_sha != source_sha:
        raise RuntimeError("Mapping copy does not match its source")
    frame, schema = read_mapping(destination, expected_version)
    return frame, {
        "source_path": source.as_posix(),
        "source_sha256": source_sha,
        "copy_path": destination.as_posix(),
        "sha256": copy_sha,
        **schema,
    }


def assign_types(items: pd.DataFrame, mapping: pd.DataFrame, auto_statuses: list[str]) -> pd.DataFrame:
    """Add ingredient_type, type_source, mapping_status and mapping_note to the item table.

    Items with an auto status get their ingredient_id; every other item gets "aisle:" + its aisle.
    """
    result = items.drop(columns=[column for column in TYPE_COLUMNS if column in items.columns])
    lookup = mapping[["product_id", "mapping_status", "ingredient_id", "mapping_note"]].assign(
        product_id=mapping["product_id"].astype(str)
    )
    merged = result[["item_id"]].astype({"item_id": str}).merge(
        lookup, left_on="item_id", right_on="product_id", how="left"
    )
    status = merged["mapping_status"].fillna(MISSING_STATUS).astype(str)
    mapped = status.isin(auto_statuses) & merged["ingredient_id"].notna()
    fallback = FALLBACK_PREFIX + result["fine_category"].astype("string").fillna("UNKNOWN").astype(str)
    result["ingredient_type"] = np.where(mapped, merged["ingredient_id"].astype(str), fallback.to_numpy())
    result["type_source"] = np.where(mapped, "mapped", "fallback")
    result["mapping_status"] = status.to_numpy()
    result["mapping_note"] = merged["mapping_note"].fillna("").astype(str).to_numpy()
    return result


def food_mask(items: pd.DataFrame, nonfood_departments: list[str]) -> np.ndarray:
    """Items outside the non-food departments (the scope F of the food recommendation scope)."""
    departments = item_departments(items).reindex(items["item_id"].astype(str).to_numpy())
    return ~departments.isin(nonfood_departments).to_numpy()


def type_attribute_categories(
    items: pd.DataFrame,
    stage1_column: str,
    type_column: str,
    attribute_names: list[str],
    min_with: int,
    min_without: int,
) -> pd.DataFrame:
    """Per (stage-1 key, type) and attribute: item counts with and without it, and inclusion in C_a^type."""
    keys = pd.DataFrame(
        {
            "stage1": items[stage1_column].astype("string").fillna("UNKNOWN").to_numpy(),
            "type": items[type_column].astype("string").fillna("UNKNOWN").to_numpy(),
        }
    )
    rows: list[pd.DataFrame] = []
    for name in attribute_names:
        counts = (
            keys.assign(flag=items[attribute_column(name)].astype(int).to_numpy())
            .groupby(["stage1", "type"], observed=True)["flag"]
            .agg(n_with="sum", n_total="size")
            .reset_index()
        )
        counts["n_without"] = counts["n_total"] - counts["n_with"]
        counts["included"] = counts["n_with"].ge(min_with) & counts["n_without"].ge(min_without)
        counts.insert(0, "attribute", name)
        rows.append(counts[["attribute", "stage1", "type", "n_with", "n_without", "included"]])
    columns = ["attribute", "stage1", "type", "n_with", "n_without", "included"]
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(columns=columns)


def purchase_rows(history: pd.DataFrame, items: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """History purchases deduplicated by (user, basket, item) joined with item columns."""
    implicit = history[["user_id", "basket_id", "item_id"]].astype(str).drop_duplicates()
    table = items.drop_duplicates("item_id").assign(item_id=lambda frame: frame["item_id"].astype(str))
    return implicit.merge(table[["item_id", *columns]], on="item_id", how="inner")


def _distribution(values: pd.Series) -> dict[str, float]:
    quantiles = values.quantile([0.1, 0.25, 0.5, 0.75, 0.9, 0.99]).to_dict()
    return {
        "count": int(len(values)),
        "mean": float(values.mean()) if len(values) else 0.0,
        "max": int(values.max()) if len(values) else 0,
        **{f"q{int(round(level * 100))}": float(value) for level, value in quantiles.items()},
        "singletons": int(values.eq(1).sum()),
    }


def type_report(items: pd.DataFrame, food: np.ndarray, histories: dict[str, pd.DataFrame]) -> dict[str, object]:
    """Type counts, items per type, mapped purchase shares, mapping notes and multi-aisle types (food only)."""
    table = items.loc[food].copy()
    table["pair"] = table["fine_category"].astype(str) + "|" + table["ingredient_type"].astype(str)
    report: dict[str, object] = {"food_items": int(len(table))}
    for source in ["mapped", "fallback"]:
        part = table[table["type_source"].eq(source)]
        report[source] = {
            "items": int(len(part)),
            "types": int(part["ingredient_type"].nunique()),
            "aisle_type_pairs": int(part["pair"].nunique()),
            "items_per_aisle_type_pair": _distribution(part.groupby("pair").size()),
        }
    report["items_per_aisle_type_pair_all"] = _distribution(table.groupby("pair").size())
    report["mapping_status_counts"] = {str(k): int(v) for k, v in table["mapping_status"].value_counts().items()}
    report["mapping_note_counts"] = {
        (str(k) if k else "<none>"): int(v) for k, v in table["mapping_note"].value_counts().items()
    }
    report["mapping_note_counts_mapped"] = {
        (str(k) if k else "<none>"): int(v)
        for k, v in table.loc[table["type_source"].eq("mapped"), "mapping_note"].value_counts().items()
    }
    mapped = table[table["type_source"].eq("mapped")]
    aisles_per_type = mapped.groupby("ingredient_type")["fine_category"].nunique()
    multi = set(aisles_per_type[aisles_per_type.gt(1)].index)
    report["multi_aisle_types"] = {
        "types": int(len(multi)),
        "share_of_mapped_types": float(len(multi) / len(aisles_per_type)) if len(aisles_per_type) else 0.0,
        "aisles_per_type": _distribution(aisles_per_type),
        "items": int(mapped["ingredient_type"].isin(multi).sum()),
    }
    shares: dict[str, object] = {}
    for split, history in histories.items():
        rows = purchase_rows(history, table, ["type_source", "mapping_status", "ingredient_type"])
        shares[split] = {
            "food_purchases": int(len(rows)),
            "mapped_share": float(rows["type_source"].eq("mapped").mean()) if len(rows) else 0.0,
            "fallback_share": float(rows["type_source"].eq("fallback").mean()) if len(rows) else 0.0,
            "multi_aisle_type_share": float(rows["ingredient_type"].isin(multi).mean()) if len(rows) else 0.0,
            "status_share": {str(k): float(v) for k, v in rows["mapping_status"].value_counts(normalize=True).items()},
        }
    report["purchase_shares"] = shares
    return report


def type_category_report(
    items: pd.DataFrame,
    table: pd.DataFrame,
    history: pd.DataFrame,
    stage1_column: str,
    type_column: str,
) -> dict[str, dict[str, object]]:
    """Per attribute: |C_a^type|, its history purchase share, and the share where the attribute decides the type.

    Shares are over food history purchases. "Attribute decides type" means every item of the pair has the
    attribute (all_with) or none has it (all_without); then the attribute step carries no information.
    """
    keyed = items.assign(
        stage1=items[stage1_column].astype("string").fillna("UNKNOWN").astype(str),
        type=items[type_column].astype("string").fillna("UNKNOWN").astype(str),
    )
    rows = purchase_rows(history, keyed, ["stage1", "type"])
    total = len(rows)
    report: dict[str, dict[str, object]] = {}
    for name, group in table.groupby("attribute", sort=False):
        group = group.assign(
            all_with=group["n_without"].eq(0) & group["n_with"].gt(0),
            all_without=group["n_with"].eq(0),
        )
        merged = rows.merge(group, on=["stage1", "type"], how="inner")
        included = group[group["included"]]
        report[str(name)] = {
            "pairs": int(len(group)),
            "included_pairs": int(len(included)),
            "stage1_keys_with_included_pairs": int(included["stage1"].nunique()),
            "items_in_included_pairs": int((included["n_with"] + included["n_without"]).sum()),
            "attribute_items_in_included_pairs": int(included["n_with"].sum()),
            "purchase_share_included": float(merged["included"].sum() / total) if total else 0.0,
            "purchase_share_attribute_decides_type": float(merged["all_with"].sum() / total) if total else 0.0,
            "purchase_share_type_has_no_attribute_item": float(merged["all_without"].sum() / total) if total else 0.0,
            "purchase_share_mixed_not_included": float(
                (~merged["included"] & ~merged["all_with"] & ~merged["all_without"]).sum() / total
            )
            if total
            else 0.0,
            "pairs_attribute_decides_type": int(group["all_with"].sum()),
        }
    return report


def stage_columns(config: dict[str, object], dataset: str) -> tuple[str, str]:
    """Item-table columns of the stage-1 key and the type for a dataset."""
    settings = config["hierarchical"]
    return str(settings["stage1_key"][dataset]), str(settings["type_key"][dataset])


def prepare_types(config: dict[str, object]) -> dict[str, object]:
    """Import the mapping, add type columns to the Instacart items, a commodity column to CJ items,
    and report type coverage and C_a^type for both datasets."""
    settings = config["hierarchical"]
    processed = Path(config["processed_dir"])
    destination = Path(config["project_root"]) / settings["mapping"]["path"]
    mapping, provenance = import_mapping(
        Path(settings["mapping"]["source"]), destination, str(settings["mapping"]["expected_version"])
    )
    report: dict[str, object] = {"profile": config["profile"], "mapping": provenance}
    min_with, min_without = int(settings["type_attr_min_with"]), int(settings["type_attr_min_without"])

    base = processed / "instacart"
    items = assign_types(pd.read_parquet(base / "items.parquet"), mapping, list(settings["mapping"]["auto_statuses"]))
    items.to_parquet(base / "items.parquet", index=False)
    food = food_mask(items, list(config["scope"]["nonfood_departments"]["instacart"]))
    histories = {split: pd.read_parquet(base / f"{split}_history.parquet") for split in ["fit", "eval"]}
    stage1, type_key = stage_columns(config, "instacart")
    table = type_attribute_categories(
        items.loc[food], stage1, type_key, list(settings["attributes"]["instacart"]), min_with, min_without
    )
    report["instacart"] = {
        "types": type_report(items, food, histories),
        "type_attribute_categories": type_category_report(items.loc[food], table, histories["eval"], stage1, type_key),
    }

    base = processed / "complete_journey"
    items = pd.read_parquet(base / "items.parquet")
    # COMMODITY_DESC is stored as coarse_category by prepare; it is exposed under its hierarchy name.
    items["commodity"] = items["coarse_category"]
    items.to_parquet(base / "items.parquet", index=False)
    food = food_mask(items, list(config["scope"]["nonfood_departments"]["complete_journey"]))
    history = pd.read_parquet(base / "eval_history.parquet")
    stage1, type_key = stage_columns(config, "complete_journey")
    table = type_attribute_categories(
        items.loc[food], stage1, type_key, list(settings["attributes"]["complete_journey"]), min_with, min_without
    )
    food_items = items.loc[food]
    pairs = food_items.groupby([stage1, type_key], observed=True).size()
    report["complete_journey"] = {
        "food_items": int(len(food_items)),
        "stage1_keys": int(food_items[stage1].nunique()),
        "types": int(food_items[type_key].nunique()),
        "stage1_type_pairs": int(len(pairs)),
        "items_per_pair": _distribution(pairs),
        "types_in_several_stage1_keys": int(
            food_items.groupby(type_key, observed=True)[stage1].nunique().gt(1).sum()
        ),
        "type_attribute_categories": type_category_report(food_items, table, history, stage1, type_key),
    }
    path = Path(config["report_dir"]) / "metrics" / f"{config['profile']}_types.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report
