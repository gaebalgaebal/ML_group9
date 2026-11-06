"""Data loading (with holdout guard), category-level attachment, and target splitting.

Stored columns of history/target frames: dataset, user_id(str), basket_id(str),
basket_seq(int32), item_id(str), coarse_category, fine_category, quantity.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

LEVEL_NAMES = ("L1", "L2", "L3")
KEY_COLUMNS = ["user_id", "item_id"]


def dataset_dir(config: dict[str, Any], dataset: str) -> Path:
    """Return <data_dir>/<profile>/<dataset>, honouring complete_journey_profile."""
    profile = config["profile"]
    if dataset == "complete_journey" and config.get("complete_journey_profile"):
        profile = config["complete_journey_profile"]
    return Path(config["data_dir"]) / profile / dataset


def load_items(config: dict[str, Any], dataset: str) -> pd.DataFrame:
    """Load the item catalogue (items.parquet) of ``dataset``."""
    return pd.read_parquet(dataset_dir(config, dataset) / "items.parquet")


def load_split(
    config: dict[str, Any],
    dataset: str,
    split: str,
    *,
    allow_holdout_eval: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load (history, target) for ``split`` in {"fit", "eval"}.

    Holdout guard: with role == "final", reading the Instacart eval split is
    refused (PermissionError) unless ``allow_holdout_eval`` is True.
    """
    if (
        config.get("role") == "final"
        and split == "eval"
        and dataset == "instacart"
        and not allow_holdout_eval
    ):
        raise PermissionError(
            "profile role is 'final': the Instacart eval split is read once, "
            "after all selections are fixed; pass allow_holdout_eval=True explicitly"
        )
    folder = dataset_dir(config, dataset)
    history = pd.read_parquet(folder / f"{split}_history.parquet")
    target = pd.read_parquet(folder / f"{split}_target.parquet")
    return history, target


def attach_levels(
    frame: pd.DataFrame, items: pd.DataFrame, levels: dict[str, str]
) -> pd.DataFrame:
    """Return ``frame`` with L1, L2, L3 (str) columns looked up by item_id.

    ``levels`` maps "L1"/"L2"/"L3" to column names of ``items``
    (config["levels"][dataset]). Items missing from ``items`` raise ValueError.
    """
    lookup = items.set_index(items["item_id"].astype(str))
    out = frame.drop(columns=[c for c in LEVEL_NAMES if c in frame.columns]).copy()
    item_ids = out["item_id"].astype(str)
    missing = ~item_ids.isin(lookup.index)
    if missing.any():
        sample = sorted(item_ids[missing].unique())[:5]
        raise ValueError(f"{int(missing.sum())} rows have item_id not in items, e.g. {sample}")
    for level in LEVEL_NAMES:
        column = levels[level]
        values = lookup[column].reindex(item_ids.to_numpy())
        if values.isna().any():
            raise ValueError(f"items column '{column}' has null values for some item_id")
        out[level] = values.astype(str).to_numpy()
    return out


def split_target(
    history: pd.DataFrame, target: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split target rows into repeat (item in I(H_u)) and new (item not in I(H_u)).

    (user_id, item_id) duplicates are dropped first (first occurrence kept), so
    any extra columns of ``target`` (e.g. L1..L3) are preserved.
    """
    deduped = target.drop_duplicates(subset=KEY_COLUMNS).reset_index(drop=True)
    seen = pd.MultiIndex.from_frame(history[KEY_COLUMNS].drop_duplicates())
    keys = pd.MultiIndex.from_frame(deduped[KEY_COLUMNS])
    is_repeat = keys.isin(seen)
    repeat_target = deduped[is_repeat].reset_index(drop=True)
    new_target = deduped[~is_repeat].reset_index(drop=True)
    return repeat_target, new_target
