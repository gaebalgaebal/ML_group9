# *------------------- 2026 Fall Semester Machine Learning Term-project: Team 9 --------------------*
# Code First written at 2026-11-05
# Code Final Edited at 2026-11-13
# Editor: Yejun Kim, Seongwoo Yang

'''
Purpose : Data loading (with holdout guard), category-level attachment, and target splitting.
Inputs  : The merged config (data_dir, profile, role, complete_journey_profile), a dataset
          name and a split name; the parquet files items.parquet, <split>_history.parquet
          and <split>_target.parquet of the dataset folder.
          Stored columns of history/target frames: dataset, user_id(str), basket_id(str),
          basket_seq(int32), item_id(str), coarse_category, fine_category, quantity.
Outputs : load_split returns (history, target); load_items returns the item catalogue;
          attach_levels returns a frame with the added columns L1, L2, L3 (str);
          split_target returns (repeat_target, new_target).
Usage   :
    from nbr.data import attach_levels, load_items, load_split, split_target
    history, target = load_split(config, dataset, split, allow_holdout_eval=allow_holdout_eval)
    items = load_items(config, dataset)
    history = attach_levels(history, items, levels)
    repeat_target, new_target = split_target(history, target)
'''

from __future__ import annotations

# ---------------------------------------------------------------------------
# imports
# ---------------------------------------------------------------------------
from pathlib import Path
from typing import Any

import pandas as pd

# ---------------------------------------------------------------------------
# constants
# ---------------------------------------------------------------------------
LEVEL_NAMES = ("L1", "L2", "L3")
KEY_COLUMNS = ["user_id", "item_id"]


# ---------------------------------------------------------------------------
# data loading
# ---------------------------------------------------------------------------
def dataset_dir(config: dict[str, Any], dataset: str) -> Path:
    """Return <data_dir>/<profile>/<dataset> from ``config`` (data_dir, profile), honouring complete_journey_profile."""
    # Start from the profile of the config.
    profile = config["profile"]
    # Use complete_journey_profile for the complete_journey dataset when the config sets it.
    if dataset == "complete_journey" and config.get("complete_journey_profile"):
        profile = config["complete_journey_profile"]
    # Join the data directory, the profile and the dataset name.
    return Path(config["data_dir"]) / profile / dataset


def load_items(config: dict[str, Any], dataset: str) -> pd.DataFrame:
    """Load the item catalogue (items.parquet) of ``dataset`` from dataset_dir(config, dataset)."""
    return pd.read_parquet(dataset_dir(config, dataset) / "items.parquet")


def load_split(
    config: dict[str, Any],
    dataset: str,
    split: str,
    *,
    allow_holdout_eval: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load (history, target) of ``dataset`` for ``split`` in {"fit", "eval"} from dataset_dir(config, dataset).

    Holdout guard: with role == "final", reading the Instacart eval split is
    refused (PermissionError) unless ``allow_holdout_eval`` is True.
    """
    # Refuse the Instacart eval split of a final-role profile unless it is explicitly allowed.
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
    # Locate the dataset folder.
    folder = dataset_dir(config, dataset)
    # Read the history and target files of the split.
    history = pd.read_parquet(folder / f"{split}_history.parquet")
    target = pd.read_parquet(folder / f"{split}_target.parquet")
    return history, target


# ---------------------------------------------------------------------------
# category levels
# ---------------------------------------------------------------------------
def attach_levels(
    frame: pd.DataFrame, items: pd.DataFrame, levels: dict[str, str]
) -> pd.DataFrame:
    """Return ``frame`` with L1, L2, L3 (str) columns looked up by item_id.

    ``levels`` maps "L1"/"L2"/"L3" to column names of ``items``
    (config["levels"][dataset]). Items missing from ``items`` raise ValueError.
    """
    # Index the item catalogue by item_id as str.
    lookup = items.set_index(items["item_id"].astype(str))
    # Copy the frame without any existing level columns.
    out = frame.drop(columns=[c for c in LEVEL_NAMES if c in frame.columns]).copy()
    # Take the item ids of the frame as str.
    item_ids = out["item_id"].astype(str)
    # Fail when some rows have an item that is not in the catalogue.
    missing = ~item_ids.isin(lookup.index)
    if missing.any():
        sample = sorted(item_ids[missing].unique())[:5]
        raise ValueError(f"{int(missing.sum())} rows have item_id not in items, e.g. {sample}")
    for level in LEVEL_NAMES:
        # Look up the catalogue column of this level for every row.
        column = levels[level]
        values = lookup[column].reindex(item_ids.to_numpy())
        # Fail when the catalogue column has a null value for a looked-up item.
        if values.isna().any():
            raise ValueError(f"items column '{column}' has null values for some item_id")
        # Store the level as a str column.
        out[level] = values.astype(str).to_numpy()
    return out


# ---------------------------------------------------------------------------
# target splitting
# ---------------------------------------------------------------------------
def split_target(
    history: pd.DataFrame, target: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split ``target`` rows into repeat (item in I(H_u) of ``history``) and new (item not in I(H_u)).

    (user_id, item_id) duplicates are dropped first (first occurrence kept), so
    any extra columns of ``target`` (e.g. L1..L3) are preserved.
    """
    # Drop repeated (user, item) rows of the target.
    deduped = target.drop_duplicates(subset=KEY_COLUMNS).reset_index(drop=True)
    # Build the index of the distinct (user, item) pairs of the history.
    seen = pd.MultiIndex.from_frame(history[KEY_COLUMNS].drop_duplicates())
    # Mark the target rows whose (user, item) pair occurs in the history.
    keys = pd.MultiIndex.from_frame(deduped[KEY_COLUMNS])
    is_repeat = keys.isin(seen)
    # Separate the marked rows from the other rows.
    repeat_target = deduped[is_repeat].reset_index(drop=True)
    new_target = deduped[~is_repeat].reset_index(drop=True)
    return repeat_target, new_target
