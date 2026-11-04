"""Raw archive discovery, normalization, and leakage-safe temporal splits.

Input: the ZIP archives under data/raw/. Output: data/processed/<profile>/<dataset>/*.parquet and
reports/metrics/<profile>_attributes.json. Entry points: `python -m ml_group9 prepare` and `inspect-cj`.
"""

from __future__ import annotations

import json
import zipfile
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from .attributes import (
    applicable_categories,
    attribute_report,
    complete_journey_attributes,
    complete_journey_text_diagnostics,
    instacart_attributes,
    instacart_text_diagnostics,
    used_attributes,
)


COMMON_COLUMNS = [
    "dataset",
    "user_id",
    "basket_id",
    "basket_seq",
    "item_id",
    "coarse_category",
    "fine_category",
    "quantity",
]

INSTACART_MEMBERS = {
    "orders.csv",
    "order_products__prior.csv",
    "order_products__train.csv",
    "products.csv",
    "aisles.csv",
    "departments.csv",
}
COMPLETE_JOURNEY_MEMBERS = {"transaction_data.csv", "product.csv"}


def discover_archive(raw_dir: Path, required_members: set[str]) -> Path:
    """Find one ZIP by its member names instead of relying on its filename."""
    matches: list[Path] = []
    for path in sorted(raw_dir.rglob("*.zip")):
        with zipfile.ZipFile(path) as archive:
            if required_members.issubset(set(archive.namelist())):
                matches.append(path)
    if len(matches) != 1:
        raise FileNotFoundError(
            f"Expected exactly one archive containing {sorted(required_members)}, found {matches}"
        )
    return matches[0]


def _read_csv(archive: zipfile.ZipFile, member: str, **kwargs: object) -> pd.DataFrame:
    with archive.open(member) as stream:
        return pd.read_csv(stream, **kwargs)


def _read_complete_journey(raw_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Read CJ products and the transaction columns used downstream."""
    archive_path = discover_archive(raw_dir, COMPLETE_JOURNEY_MEMBERS)
    with zipfile.ZipFile(archive_path) as archive:
        products = _read_csv(archive, "product.csv", dtype={"PRODUCT_ID": "string"})
        with archive.open("transaction_data.csv") as stream:
            transactions = pd.read_csv(
                stream,
                usecols=[
                    "household_key",
                    "BASKET_ID",
                    "DAY",
                    "PRODUCT_ID",
                    "QUANTITY",
                    "SALES_VALUE",
                    "TRANS_TIME",
                    "WEEK_NO",
                ],
                dtype={"household_key": "string", "BASKET_ID": "string", "PRODUCT_ID": "string"},
            )
    # Blank DEPARTMENT strings are treated as missing.
    products["DEPARTMENT"] = products["DEPARTMENT"].astype("string").str.strip().replace("", pd.NA)
    return products, transactions


def clean_complete_journey(
    transactions: pd.DataFrame, products: pd.DataFrame, exclude_departments: list[str]
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Drop non-product and non-positive-quantity rows before any user or basket logic.

    Reasons are applied in order, so each row is counted under its first failing rule.
    """
    department = transactions["PRODUCT_ID"].map(products.set_index("PRODUCT_ID")["DEPARTMENT"])
    known = transactions["PRODUCT_ID"].isin(products["PRODUCT_ID"])
    rules = [
        ("unknown_product", ~known),
        ("missing_department", known & department.isna()),
        ("excluded_department", department.isin(exclude_departments)),
        ("non_positive_quantity", transactions["QUANTITY"].le(0)),
    ]
    keep = pd.Series(True, index=transactions.index)
    removed: dict[str, int] = {}
    for reason, mask in rules:
        mask = mask.fillna(False).astype(bool)
        removed[reason] = int((keep & mask).sum())
        keep &= ~mask
    cleaned = transactions[keep].copy()
    stats = {
        "raw_rows": int(len(transactions)),
        "kept_rows": int(len(cleaned)),
        **{f"removed_{reason}": count for reason, count in removed.items()},
        "kept_non_positive_sales_value_rows": int(cleaned["SALES_VALUE"].le(0).sum()),
    }
    return cleaned, stats


def first_future_basket(future: pd.DataFrame, min_items: int) -> pd.DataFrame:
    """Each user's earliest future basket with at least min_items distinct items."""
    sizes = future.groupby(["user_id", "basket_seq"], observed=True)["item_id"].nunique()
    qualifying = sizes[sizes.ge(min_items)].reset_index()
    first = qualifying.groupby("user_id", observed=True)["basket_seq"].min().rename("target_seq")
    frame = future.merge(first, left_on="user_id", right_index=True, how="inner")
    return frame[frame["basket_seq"].eq(frame["target_seq"])].drop(columns="target_seq")


def _target_summary(target: pd.DataFrame, min_items: int) -> dict[str, object]:
    baskets = target.groupby("user_id", observed=True).agg(
        size=("item_id", "nunique"), week=("WEEK_NO", "first")
    )
    if baskets.empty:
        return {"min_items": min_items, "households": 0}
    return {
        "min_items": min_items,
        "households": int(len(baskets)),
        "basket_size_median": float(baskets["size"].median()),
        "basket_size_at_most_2_share": float(baskets["size"].le(2).mean()),
        "week_min": int(baskets["week"].min()),
        "week_median": float(baskets["week"].median()),
        "week_max": int(baskets["week"].max()),
    }


def complete_journey_department_summary(config: dict[str, object], top_commodities: int = 3) -> list[dict[str, object]]:
    """Transaction share and top commodities per DEPARTMENT, to choose exclusions."""
    products, transactions = _read_complete_journey(Path(config["raw_dir"]))
    lines = transactions[["PRODUCT_ID"]].merge(
        products[["PRODUCT_ID", "DEPARTMENT", "COMMODITY_DESC"]], on="PRODUCT_ID", how="left"
    )
    lines["DEPARTMENT"] = lines["DEPARTMENT"].fillna("<missing>")
    lines["COMMODITY_DESC"] = lines["COMMODITY_DESC"].fillna("<unknown product>")
    summary: list[dict[str, object]] = []
    for department, group in lines.groupby("DEPARTMENT", observed=True):
        commodities = group["COMMODITY_DESC"].value_counts().head(top_commodities)
        summary.append(
            {
                "department": str(department),
                "rows": int(len(group)),
                "row_share": len(group) / len(lines),
                "top_commodities": {str(name): int(count) for name, count in commodities.items()},
            }
        )
    return sorted(summary, key=lambda row: -row["rows"])


def _sample_ids(ids: Iterable[object], size: int | None, seed: int) -> set[str]:
    values = np.array(sorted({str(value) for value in ids}), dtype=object)
    if size is None or size >= len(values):
        return set(values.tolist())
    rng = np.random.default_rng(seed)
    return set(rng.choice(values, size=size, replace=False).tolist())


def _write_frames(base: Path, frames: dict[str, pd.DataFrame]) -> None:
    base.mkdir(parents=True, exist_ok=True)
    for name, frame in frames.items():
        frame.to_parquet(base / f"{name}.parquet", index=False)


def _attribute_outputs(
    config: dict[str, object], dataset: str, items: pd.DataFrame, history: pd.DataFrame, extra: dict[str, object]
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Build C_a from the item list and merge per-dataset coverage into the profile report."""
    settings = config["attributes"]
    table = applicable_categories(
        items, used_attributes(items), int(settings["min_with"]), int(settings["min_without"])
    )
    payload = {
        "history_split": "eval_history",
        "attributes": attribute_report(items, history, table, int(settings["distinct_min_n"])),
        **extra,
    }
    path = Path(config["report_dir"]) / "metrics" / f"{config['profile']}_attributes.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    report = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    report[dataset] = payload
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return table, payload


def prepare_instacart(config: dict[str, object]) -> dict[str, int]:
    """Prepare Instacart fit and official-train evaluation splits."""
    raw_dir = Path(config["raw_dir"])
    archive_path = discover_archive(raw_dir, INSTACART_MEMBERS)
    sample_size = config["sample_users"]["instacart"]
    seed = int(config["seed"])

    with zipfile.ZipFile(archive_path) as archive:
        orders = _read_csv(archive, "orders.csv", dtype={"order_id": "int64", "user_id": "string"})
        eligible = orders.loc[orders["eval_set"].eq("train"), "user_id"]
        selected = _sample_ids(eligible, sample_size, seed)
        orders = orders[orders["user_id"].isin(selected)].copy()

        products = _read_csv(archive, "products.csv")
        aisles = _read_csv(archive, "aisles.csv")
        departments = _read_csv(archive, "departments.csv")
        items = (
            products.merge(aisles, on="aisle_id", how="left")
            .merge(departments, on="department_id", how="left")
            .rename(
                columns={
                    "product_id": "item_id",
                    "department": "coarse_category",
                    "aisle": "fine_category",
                }
            )[["item_id", "product_name", "coarse_category", "fine_category"]]
        )
        items["item_id"] = items["item_id"].astype("string")
        items = instacart_attributes(items, config)

        prior_order_ids = set(orders.loc[orders["eval_set"].eq("prior"), "order_id"])
        train_order_ids = set(orders.loc[orders["eval_set"].eq("train"), "order_id"])
        prior_parts: list[pd.DataFrame] = []
        train_parts: list[pd.DataFrame] = []
        for member, wanted, sink in [
            ("order_products__prior.csv", prior_order_ids, prior_parts),
            ("order_products__train.csv", train_order_ids, train_parts),
        ]:
            with archive.open(member) as stream:
                for chunk in pd.read_csv(stream, chunksize=1_000_000):
                    part = chunk[chunk["order_id"].isin(wanted)]
                    if not part.empty:
                        sink.append(part[["order_id", "product_id"]].copy())

    prior_lines = pd.concat(prior_parts, ignore_index=True)
    train_lines = pd.concat(train_parts, ignore_index=True)
    order_meta = orders[["order_id", "user_id", "order_number", "eval_set"]]

    def normalize(lines: pd.DataFrame) -> pd.DataFrame:
        normalized_lines = lines.copy()
        normalized_lines["product_id"] = normalized_lines["product_id"].astype("string")
        frame = normalized_lines.merge(order_meta, on="order_id", how="inner").merge(
            items[["item_id", "coarse_category", "fine_category"]],
            left_on="product_id",
            right_on="item_id",
            how="left",
        )
        frame = frame.assign(
            dataset="instacart",
            basket_id=frame["order_id"].astype("string"),
            basket_seq=frame["order_number"].astype("int32"),
            item_id=frame["product_id"],
            quantity=np.int16(1),
        )
        return frame[COMMON_COLUMNS]

    prior = normalize(prior_lines)
    official_target = normalize(train_lines)
    last_prior = prior.groupby("user_id", observed=True)["basket_seq"].transform("max")
    fit_target = prior[last_prior.eq(prior["basket_seq"])].copy()
    fit_history = prior[last_prior.gt(prior["basket_seq"])].copy()
    eligible_users = set(
        fit_history.groupby("user_id", observed=True)["basket_id"].nunique().loc[lambda s: s >= 2].index
    )
    fit_history = fit_history[fit_history["user_id"].isin(eligible_users)]
    fit_target = fit_target[fit_target["user_id"].isin(eligible_users)]

    table, attributes = _attribute_outputs(
        config,
        "instacart",
        items,
        prior,
        {
            "percent_fat_aisles": list(config["attributes"]["percent_fat_aisles"]),
            "percent_fat_levels": list(config["attributes"]["percent_fat_levels"]),
            "text_diagnostics": instacart_text_diagnostics(items, config),
        },
    )
    out = Path(config["processed_dir"]) / "instacart"
    _write_frames(
        out,
        {
            "items": items,
            "attribute_categories": table,
            "fit_history": fit_history,
            "fit_target": fit_target,
            "eval_history": prior,
            "eval_target": official_target,
        },
    )
    return {
        "users": prior["user_id"].nunique(),
        "fit_interactions": len(fit_history),
        "eval_interactions": len(prior),
        "items": len(items),
        "attributes": attributes["attributes"],
    }


def _complete_journey_common(transactions: pd.DataFrame, items: pd.DataFrame) -> pd.DataFrame:
    """Sequence each household's cleaned baskets by time and map them to the common schema."""
    transactions = transactions.sort_values(["household_key", "DAY", "TRANS_TIME", "BASKET_ID"])
    basket_order = transactions[["household_key", "BASKET_ID", "DAY", "TRANS_TIME"]].drop_duplicates()
    basket_order["basket_seq"] = basket_order.groupby("household_key").cumcount().add(1).astype("int32")
    transactions = transactions.merge(basket_order[["household_key", "BASKET_ID", "basket_seq"]], on=["household_key", "BASKET_ID"])
    transactions = transactions.merge(
        items[["item_id", "coarse_category", "fine_category"]],
        left_on="PRODUCT_ID",
        right_on="item_id",
        how="left",
    )
    return transactions.assign(
        dataset="complete_journey",
        user_id=transactions["household_key"].astype("string"),
        basket_id=transactions["BASKET_ID"].astype("string"),
        item_id=transactions["PRODUCT_ID"].astype("string"),
        quantity=transactions["QUANTITY"].astype("int32"),
    )[COMMON_COLUMNS + ["WEEK_NO"]]


def _history_and_target(
    common: pd.DataFrame, history_end: int, target_end: int | None, min_target_items: int
) -> tuple[pd.DataFrame, set[str], pd.DataFrame, pd.DataFrame]:
    """History up to history_end; target = first qualifying basket in (history_end, target_end].

    Users need at least three history baskets. Also returns the first-future-basket reference.
    """
    history = common[common["WEEK_NO"].le(history_end)]
    future = common[common["WEEK_NO"].gt(history_end)]
    if target_end is not None:
        future = future[future["WEEK_NO"].le(target_end)]
    history_users = set(
        history.groupby("user_id", observed=True)["basket_id"].nunique().loc[lambda s: s >= 3].index
    )
    future = future[future["user_id"].isin(history_users)]
    return history, history_users, first_future_basket(future, min_target_items), first_future_basket(future, 1)


def _complete_journey_splits(
    common: pd.DataFrame, cutoff: int, fit_week: int | None, min_target_items: int
) -> dict[str, object]:
    """Eval split (history <= cutoff, target after it) and, if fit_week is set, a fit split
    (history <= fit_week, target in (fit_week, cutoff]). Eligible users can form every configured split."""
    eval_history, eval_history_users, eval_target, eval_reference = _history_and_target(
        common, cutoff, None, min_target_items
    )
    eval_users = set(eval_target["user_id"])
    splits: dict[str, object] = {
        "eval_history": eval_history,
        "eval_history_users": eval_history_users,
        "eval_target": eval_target,
        "eval_reference": eval_reference,
        "eval_users": eval_users,
        "eligible": eval_users,
    }
    if fit_week is not None:
        fit_history, _, fit_target, _ = _history_and_target(common, int(fit_week), cutoff, min_target_items)
        fit_users = set(fit_target["user_id"])
        splits.update(
            {"fit_history": fit_history, "fit_target": fit_target, "fit_users": fit_users, "eligible": eval_users & fit_users}
        )
    return splits


def prepare_complete_journey(config: dict[str, object]) -> dict[str, object]:
    """Prepare the global week-cutoff evaluation split and, when configured, an earlier fit split."""
    cutoff = int(config["complete_journey_history_week"])
    sample_size = config["sample_users"]["complete_journey"]
    seed = int(config["seed"])
    exclude_departments = list(config["complete_journey"]["exclude_departments"])
    min_target_items = int(config["complete_journey"]["min_target_items"])
    fit_week = config["complete_journey"].get("fit_history_week")
    filter_before_sampling = bool(config["complete_journey"].get("filter_before_sampling", False))

    products, transactions = _read_complete_journey(Path(config["raw_dir"]))
    transactions, cleaning = clean_complete_journey(transactions, products, exclude_departments)
    products = products[products["DEPARTMENT"].notna() & ~products["DEPARTMENT"].isin(exclude_departments)]
    items = products.rename(
        columns={
            "PRODUCT_ID": "item_id",
            "COMMODITY_DESC": "coarse_category",
            "SUB_COMMODITY_DESC": "fine_category",
        }
    )[["item_id", "DEPARTMENT", "BRAND", "coarse_category", "fine_category"]]
    items = complete_journey_attributes(items, config)

    user_weeks = transactions.groupby("household_key")["WEEK_NO"].agg(["min", "max"])
    candidates = user_weeks[(user_weeks["min"] <= cutoff) & (user_weeks["max"] > cutoff)].index
    spanning_households = len(candidates)
    if filter_before_sampling:
        # Basket sequences are per household, so qualifying before sampling changes only who is drawn.
        pool = _complete_journey_common(transactions[transactions["household_key"].isin(candidates)], items)
        candidates = _complete_journey_splits(pool, cutoff, fit_week, min_target_items)["eligible"]
    selected = _sample_ids(candidates, sample_size, seed)
    common = _complete_journey_common(transactions[transactions["household_key"].isin(selected)], items)
    splits = _complete_journey_splits(common, cutoff, fit_week, min_target_items)
    eligible = splits["eligible"]
    history = splits["eval_history"]
    history = history[history["user_id"].isin(eligible)][COMMON_COLUMNS]
    target = splits["eval_target"]
    target = target[target["user_id"].isin(eligible)]
    frames = {"eval_history": history, "eval_target": target[COMMON_COLUMNS]}
    fit_summary: dict[str, object] = {}
    if fit_week is not None:
        fit_history = splits["fit_history"]
        fit_target = splits["fit_target"]
        fit_target = fit_target[fit_target["user_id"].isin(eligible)]
        frames["fit_history"] = fit_history[fit_history["user_id"].isin(eligible)][COMMON_COLUMNS]
        frames["fit_target"] = fit_target[COMMON_COLUMNS]
        fit_summary = {
            "fit_history_week": int(fit_week),
            "households_with_fit_target": len(splits["fit_users"]),
            "households_with_eval_target": len(splits["eval_users"]),
            "households_with_both_splits": len(eligible),
            "fit_target": _target_summary(fit_target, min_target_items),
        }

    table, attributes = _attribute_outputs(
        config,
        "complete_journey",
        items,
        history,
        {
            "private_label_brands": list(config["attributes"]["private_label_brands"]),
            "brand_distribution": {
                str(brand): int(count) for brand, count in items["BRAND"].fillna("<missing>").value_counts().items()
            },
            "text_diagnostics": complete_journey_text_diagnostics(items),
        },
    )
    out = Path(config["processed_dir"]) / "complete_journey"
    _write_frames(
        out,
        {
            "items": items,
            "attribute_categories": table,
            **frames,
        },
    )
    return {
        "users": history["user_id"].nunique(),
        "eval_interactions": len(history),
        "targets": len(target),
        "items": len(items),
        "cleaning": cleaning,
        "filter_before_sampling": filter_before_sampling,
        "households_spanning_cutoff": spanning_households,
        "households_in_sampling_pool": len(candidates),
        "sampled_households": len(selected),
        "households_with_history": len(splits["eval_history_users"]),
        "target": _target_summary(target, min_target_items),
        "target_first_future_basket": _target_summary(splits["eval_reference"], 1),
        **fit_summary,
        "attributes": attributes["attributes"],
    }


def prepare(config: dict[str, object], dataset: str) -> dict[str, dict[str, object]]:
    """Prepare one dataset or both datasets."""
    result: dict[str, dict[str, object]] = {}
    if dataset in {"instacart", "all"}:
        result["instacart"] = prepare_instacart(config)
    if dataset in {"complete_journey", "all"}:
        result["complete_journey"] = prepare_complete_journey(config)
    return result


def load_split(config: dict[str, object], dataset: str, split: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load items, history, and target for a prepared split."""
    base = Path(config["processed_dir"]) / dataset
    items = pd.read_parquet(base / "items.parquet")
    history = pd.read_parquet(base / f"{split}_history.parquet")
    target = pd.read_parquet(base / f"{split}_target.parquet")
    return items, history, target
