"""Raw archive discovery, normalization, and leakage-safe temporal splits."""

from __future__ import annotations

import zipfile
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


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

    out = Path(config["processed_dir"]) / "instacart"
    _write_frames(
        out,
        {
            "items": items,
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
    }


def prepare_complete_journey(config: dict[str, object]) -> dict[str, object]:
    """Prepare a global week-cutoff external evaluation split."""
    cutoff = int(config["complete_journey_history_week"])
    sample_size = config["sample_users"]["complete_journey"]
    seed = int(config["seed"])
    exclude_departments = list(config["complete_journey"]["exclude_departments"])

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

    user_weeks = transactions.groupby("household_key")["WEEK_NO"].agg(["min", "max"])
    candidates = user_weeks[(user_weeks["min"] <= cutoff) & (user_weeks["max"] > cutoff)].index
    selected = _sample_ids(candidates, sample_size, seed)
    transactions = transactions[transactions["household_key"].isin(selected)].copy()
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
    common = transactions.assign(
        dataset="complete_journey",
        user_id=transactions["household_key"].astype("string"),
        basket_id=transactions["BASKET_ID"].astype("string"),
        item_id=transactions["PRODUCT_ID"].astype("string"),
        quantity=transactions["QUANTITY"].astype("int32"),
    )[COMMON_COLUMNS + ["WEEK_NO"]]
    history = common[common["WEEK_NO"].le(cutoff)].copy()
    future = common[common["WEEK_NO"].gt(cutoff)].copy()
    first_future = future.groupby("user_id", observed=True)["basket_seq"].transform("min")
    target = future[first_future.eq(future["basket_seq"])].copy()
    eligible_users = set(
        history.groupby("user_id", observed=True)["basket_id"].nunique().loc[lambda s: s >= 3].index
    )
    history = history[history["user_id"].isin(eligible_users)][COMMON_COLUMNS]
    target = target[target["user_id"].isin(eligible_users)][COMMON_COLUMNS]

    out = Path(config["processed_dir"]) / "complete_journey"
    _write_frames(out, {"items": items, "eval_history": history, "eval_target": target})
    return {
        "users": history["user_id"].nunique(),
        "eval_interactions": len(history),
        "targets": len(target),
        "items": len(items),
        "cleaning": cleaning,
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
