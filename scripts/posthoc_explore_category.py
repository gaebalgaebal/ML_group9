# *------------------- 2026 Fall Semester Machine Learning Term-project: Team 9 --------------------*
# Code First written at 2026-11-07
# Code Final Edited at 2026-11-13
# Editor: Yejun Kim

'''
Purpose : Post-hoc case study: where new-item exploration happens, by department and product
          group (W18; post-hoc, development data, descriptive, not confirmatory).
          Events: for user u and basket b from u's 2nd basket on (history baskets, then the
          target basket of the split), item i in b (distinct items per basket) is a new-item
          event if i is not in any earlier basket of u, otherwise a repeat event.
          Category names are reported exactly as stored; categories named "missing" are
          flagged, never dropped.
          Per category c (both levels, event sets all = history + target, history, target):
          volume (new-item events, share of all new-item events), propensity (new-item rate =
          new events in c / all events in c), lift (share of new-item events in c / share of
          all events in c), breadth (share of users with >= 1 new-item event in c, over all
          users and over users who bought anything in c in any basket of the split),
          known-category share (share of the new-item events in c whose category c was already
          in the user's earlier baskets).
          Predictability (eval target only): hit rate of the top 10 of the user-based
          collaborative-filtering list (E2 user-CF; k*, weighting* read from the reference
          tuning document and verified) and of flat popularity excluding history top 10 on the
          target new items of c (category-level new-item recall), the share of all E2 top-10
          slots that fall in c next to the share of target new items in c.
          User-level bootstrap (n, seed; 95 % CI) for rate, lift and E2 hit rate.
          Support filter for fine-level rankings: >= 200 new-item events (Instacart) / >= 100
          (Complete Journey) in the event set ranked.
          Rankings: top 10 by volume, propensity, breadth, top and bottom 10 by lift; a 2 x 2
          placement of fine categories by new-item rate (eval, all events) and E2 hit rate
          (above / below the medians); mismatch between E2 slot share and target new-item
          share.
          Robustness: fit-split volume / propensity with Spearman rank correlations against
          the eval split; coarse propensity by T5 history-length bin.
Inputs  : The fit and eval splits of the processed development data of the profile and the
          reference tuning document <profile>_<dataset>_tuning.json in --reference-tuning-dir.
          Category levels: the pipeline's L1 / L2 columns of items.parquet (configs/common.yaml
          ``levels``): Instacart coarse_category = department and fine_category = aisle
          (identical to the processed frame columns), Complete Journey DEPARTMENT and commodity
          (the processed frame columns of Complete Journey hold commodity and sub-commodity
          instead).
          Only the development profiles ``sample`` and ``medium`` are accepted; no file under
          reports/final* or data/processed/holdout* is read.
Outputs : In --out-dir: explore_category_<profile>.json, explore_category_<profile>.md,
          fig_category_rate_coarse.{png,csv}, fig_category_lift_fine.{png,csv},
          fig_category_rate_vs_e2_fine.{png,csv}.
Usage   :
    python scripts/posthoc_explore_category.py [--profile medium]
        [--datasets instacart,complete_journey] [--out-dir reports/posthoc]
        [--reference-tuning-dir PATH] [--usercf-k K --usercf-weighting W] [--bootstrap-n 1000]
        [--seed 42]
'''

from __future__ import annotations

# ---------------------------------------------------------------------------
# imports
# ---------------------------------------------------------------------------
import argparse
import hashlib
import json
import math
import platform
import sys
import time
import warnings
from pathlib import Path
from typing import Any

# Put the src folder of the repository on the import path.
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import matplotlib  # noqa: E402

# Select the non-interactive Agg backend before pyplot is imported.
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402

from nbr.baselines import popularity_list  # noqa: E402
from nbr.config import load_config  # noqa: E402
from nbr.data import attach_levels, dataset_dir, load_items  # noqa: E402
from nbr.io import write_json  # noqa: E402
from nbr.pipeline import HISTORY_BINS, K, history_lengths, prepare  # noqa: E402
from nbr.usercf import ucf_list  # noqa: E402
from nbr.usercf_eval import peak_memory_mib  # noqa: E402

# ---------------------------------------------------------------------------
# constants
# ---------------------------------------------------------------------------
LABEL = "post-hoc, development data, descriptive, not confirmatory"
TASK = "W18 post-hoc case study: where new-item exploration happens, by department and product group"
ALLOWED_PROFILES = ("sample", "medium")
DATASETS = ("instacart", "complete_journey")
NAMES = {"instacart": "Instacart", "complete_journey": "Complete Journey"}
LEVELS = {"coarse": "L1", "fine": "L2"}
LEVEL_NAMES = {"instacart": {"coarse": "department (coarse_category)", "fine": "aisle (fine_category)"},
               "complete_journey": {"coarse": "DEPARTMENT", "fine": "commodity"}}
EVENT_SETS = ("all", "history", "target")
INPUT_FILES = ("fit_history", "fit_target", "eval_history", "eval_target", "items")
SUPPORT_MIN_NEW = {"instacart": 200, "complete_journey": 100}
EXPECTED_USERCF = {"medium": {"instacart": (100, "binary"), "complete_journey": (20, "binary")}}
MISSING_LABEL = "missing"
TOP_N = 10
FADE_MIN_EVENTS = 100                          # F-1: bars of categories with fewer events are drawn faded
COLORS = {"instacart": "#2a78d6", "complete_journey": "#eb6834"}
INK = {"primary": "#0b0b0b", "secondary": "#52514e", "muted": "#898781", "grid": "#e1e0d9", "axis": "#c3c2b7", "surface": "#fcfcfb"}
FIGURES = {"rate_coarse": "fig_category_rate_coarse", "lift_fine": "fig_category_lift_fine", "rate_vs_e2_fine": "fig_category_rate_vs_e2_fine"}


# ---------------------------------------------------------------------------
# arguments and helpers
# ---------------------------------------------------------------------------
def parse_datasets(value: str) -> list[str]:
    '''Split the comma-separated ``value`` into unique dataset names and validate them.'''
    # Collect the stripped, non-empty names once each in the given order.
    names: list[str] = []
    for item in value.split(","):
        item = item.strip()
        if item and item not in names:
            names.append(item)
    # Reject an empty list or a name outside DATASETS.
    unknown = [name for name in names if name not in DATASETS]
    if unknown or not names:
        raise ValueError(f"datasets must be a non-empty subset of {list(DATASETS)}, got {value!r}")
    # Return the validated names.
    return names


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    '''Parse the command line ``argv`` (None = sys.argv) into a Namespace of the run options.'''
    # Declare the options of the script.
    parser = argparse.ArgumentParser(description="Post-hoc category-level new-item exploration case study (development profiles only).")
    parser.add_argument("--profile", default="medium", choices=ALLOWED_PROFILES)
    parser.add_argument("--datasets", default=",".join(DATASETS), type=parse_datasets)
    parser.add_argument("--out-dir", default=str(ROOT / "reports" / "posthoc"))
    parser.add_argument("--reference-tuning-dir", default=str(ROOT.parent / "dev_ML-wt-w8" / "reports" / "metrics"),
                        help="read-only folder with <profile>_<dataset>_tuning.json giving the user-CF (k, weighting)")
    parser.add_argument("--usercf-k", type=int, default=None, help="override the user-CF k (tests); with --usercf-weighting")
    parser.add_argument("--usercf-weighting", default=None, choices=[None, "binary", "log1p"])
    parser.add_argument("--bootstrap-n", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    # Parse the arguments.
    args = parser.parse_args(argv)
    # Require the two user-CF override options to be given together.
    if (args.usercf_k is None) != (args.usercf_weighting is None):
        parser.error("--usercf-k and --usercf-weighting must be given together")
    return args


def guard_path(path: Path) -> Path:
    """Refuse any path with a component starting with 'final' or 'holdout' (confirmatory inputs are never read)."""
    # Check every component of the resolved path.
    for part in path.resolve().parts:
        if part.lower().startswith(("final", "holdout")):
            raise PermissionError(f"refusing to read {path}: confirmatory inputs are never read by this script")
    # Return the path unchanged when it is allowed.
    return path


def sha256_of(path: Path) -> str:
    '''Return the SHA-256 hex digest of the file at ``path``.'''
    # Feed the file to the hash in chunks of 1 MiB.
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    # Return the digest as a hex string.
    return digest.hexdigest()


def bin_labels(edges: list[int]) -> list[tuple[str, float, float]]:
    '''Return (label, lower, upper) per history-length bin given the inner ``edges``.'''
    # Add 0 and infinity as the outer bounds.
    bounds = [0, *edges, math.inf]
    # Label each pair of neighbouring bounds as a half-open interval.
    return [(f"({lo},inf)" if math.isinf(hi) else f"({lo},{hi}]", lo, hi) for lo, hi in zip(bounds[:-1], bounds[1:])]


def is_missing(name: str) -> bool:
    '''Return True when ``name`` equals MISSING_LABEL ignoring case and outer spaces.'''
    return str(name).strip().lower() == MISSING_LABEL


def usercf_selection(profile: str, dataset: str, reference_dir: Path | None, override: tuple[int, str] | None) -> dict[str, Any]:
    """(k, weighting) of the E2 user-CF list: the override, else the reference tuning document (verified against EXPECTED_USERCF)."""
    # Use the override when one is given.
    if override is not None:
        return {"k": int(override[0]), "weighting": str(override[1]), "source": "override", "verified": None}
    # Read the selection from the reference tuning document of the profile and dataset.
    path = guard_path(Path(reference_dir)) / f"{profile}_{dataset}_tuning.json"
    document = json.loads(path.read_text(encoding="utf-8"))
    selections = document["integration"]["selections"]
    k, weighting = int(selections["k"]), str(selections["weighting"])
    # Compare the selection with the expected one when the profile has an expected one.
    expected = EXPECTED_USERCF.get(profile, {}).get(dataset)
    verified = None if expected is None else (k, weighting) == expected
    # Stop when the selection differs from the expected one.
    if verified is False:
        raise SystemExit(f"user-CF selection of {dataset} in {path} is (k = {k}, {weighting}); expected {expected}; stopping")
    # Return the selection with the path, hash and commit of its source document.
    return {"k": k, "weighting": weighting, "source": str(path), "source_sha256": sha256_of(path),
            "source_commit": document.get("meta", {}).get("git_commit"), "expected": list(expected) if expected else None, "verified": verified}


# ---------------------------------------------------------------------------
# events
# ---------------------------------------------------------------------------
def build_events(history: pd.DataFrame, target: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(events, rows): per-item events from each user's 2nd basket on, and all deduplicated rows incl. the first basket.

    ``history`` / ``target`` carry L1 and L2 (prepare output). events columns: user_id, basket_seq, item_id, L1, L2,
    is_target, is_new (item not in any earlier basket of the user), known_L1 / known_L2 (the item's category was in an
    earlier basket of the user). rows = the deduplicated (user, basket, item) rows of history + target with L1, L2, used
    for the "bought anything in c" denominators.
    """
    # Stack the history and target rows and flag the target rows.
    columns = ["user_id", "basket_seq", "item_id", "L1", "L2"]
    frame = pd.concat([history[columns].assign(is_target=False), target[columns].assign(is_target=True)], ignore_index=True)
    # Keep one row per (user, basket, item).
    frame = frame.drop_duplicates(["user_id", "basket_seq", "item_id"]).reset_index(drop=True)
    # Encode the users as integer codes and read the basket sequence numbers.
    user_code, users = pd.factorize(frame["user_id"])
    seq = frame["basket_seq"].to_numpy(dtype="int64")

    def first_seq(column: str) -> np.ndarray:
        '''Return per row the first basket_seq in which the user has the value of ``column``.'''
        # Build one integer key per (user, value) pair.
        code, values = pd.factorize(frame[column])
        key = user_code.astype(np.int64) * len(values) + code.astype(np.int64)
        # Take the smallest basket sequence number of each key for every row.
        return pd.Series(seq).groupby(key).transform("min").to_numpy()

    # Flag a row as new when its basket is the first one of the user with that item.
    frame["is_new"] = seq == first_seq("item_id")
    # Flag a row when its category was in an earlier basket of the user, per level.
    frame["known_L1"] = first_seq("L1") < seq
    frame["known_L2"] = first_seq("L2") < seq
    # Find the first basket of every user.
    first_basket = pd.Series(seq).groupby(user_code).transform("min").to_numpy()
    # Keep the rows after the first basket of the user as events.
    events = frame[seq > first_basket].reset_index(drop=True)
    # Return the events and all deduplicated rows.
    return events, frame


# ---------------------------------------------------------------------------
# user-level bootstrap of ratios of sums
# ---------------------------------------------------------------------------
def user_category_matrix(frame: pd.DataFrame, level: str, users: pd.Index, categories: pd.Index, mask: np.ndarray | None = None) -> np.ndarray:
    """Counts of rows of ``frame`` (optionally only where ``mask``) per (user, category) as a dense users x categories array."""
    # Restrict the frame to the masked rows when a mask is given.
    sub = frame if mask is None else frame[mask]
    # Map the rows to user and category positions.
    user_idx = users.get_indexer(sub["user_id"])
    cat_idx = categories.get_indexer(sub[level])
    # Drop the rows whose user or category is not in the given indexes.
    keep = (user_idx >= 0) & (cat_idx >= 0)
    # Count the rows per (user, category) cell and reshape the counts to a matrix.
    flat = np.bincount(user_idx[keep] * len(categories) + cat_idx[keep], minlength=len(users) * len(categories))
    return flat.reshape(len(users), len(categories)).astype("float64")


def bootstrap_ratios(num: np.ndarray, den: np.ndarray, n: int, seed: int, chunk: int = 100) -> dict[str, np.ndarray]:
    """User-level bootstrap (users resampled with replacement) of rate = sum num / sum den and lift = (sum num_c / sum num) /
    (sum den_c / sum den) per category column; returns the 2.5 / 97.5 percentiles (NaN when a resample has no events)."""
    # Create the seeded generator and the arrays of resampled rates and lifts.
    n_users, n_cat = num.shape
    rng = np.random.default_rng(seed)
    rates = np.empty((n, n_cat))
    lifts = np.empty((n, n_cat))
    done = 0
    while done < n:
        # Draw the resample weights of the next chunk as per-user draw counts.
        size = min(chunk, n - done)
        weights = np.zeros((size, n_users))
        for row in range(size):
            weights[row] = np.bincount(rng.integers(0, n_users, n_users), minlength=n_users)
        # Sum the numerators and denominators of each resample per category.
        sum_num, sum_den = weights @ num, weights @ den
        # Store the rate and the lift of each resample, without division warnings.
        with np.errstate(divide="ignore", invalid="ignore"):
            rates[done:done + size] = sum_num / sum_den
            lifts[done:done + size] = (sum_num / sum_num.sum(axis=1, keepdims=True)) / (sum_den / sum_den.sum(axis=1, keepdims=True))
        done += size
    # Return the 2.5 and 97.5 percentiles per category, ignoring NaN resamples.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)       # all-NaN columns (categories without events) give NaN bounds
        return {"rate_lo": np.nanpercentile(rates, 2.5, axis=0), "rate_hi": np.nanpercentile(rates, 97.5, axis=0),
                "lift_lo": np.nanpercentile(lifts, 2.5, axis=0), "lift_hi": np.nanpercentile(lifts, 97.5, axis=0)}


def _ci(lo: float, hi: float) -> list[float] | None:
    '''Return [``lo``, ``hi``] as floats, or None when either bound is None or NaN.'''
    return None if (lo is None or hi is None or np.isnan(lo) or np.isnan(hi)) else [float(lo), float(hi)]


# ---------------------------------------------------------------------------
# category tables
# ---------------------------------------------------------------------------
def category_table(events: pd.DataFrame, rows: pd.DataFrame, level: str, users: pd.Index, n_boot: int, seed: int) -> pd.DataFrame:
    """Volume, propensity, lift, breadth and known-category share per category of ``level`` for the event frame ``events``.

    ``rows`` (all deduplicated rows incl. first baskets) gives the users who bought anything in c; ``users`` = all users
    of the split (bootstrap population and breadth denominator). Bootstrap CIs (user level) for rate and lift.
    """
    # Count events, new-item events, new-item users and known-category new items per category.
    known = f"known_{level}"
    grouped = events.groupby(level, sort=True)
    table = pd.DataFrame({
        "n_events": grouped.size(),
        "n_new": grouped["is_new"].sum().astype("int64"),
        "n_users_new": events[events["is_new"]].groupby(level)["user_id"].nunique(),
        "n_new_known_category": events[events["is_new"]].groupby(level)[known].sum().astype("int64"),
    }).fillna(0)
    # Count the users with any row in the category, taken from all deduplicated rows.
    table["n_users_bought"] = rows.groupby(level)["user_id"].nunique().reindex(table.index).fillna(0).astype("int64")
    # Store the counts as integers.
    for column in ("n_events", "n_new", "n_users_new", "n_new_known_category"):
        table[column] = table[column].astype("int64")
    # Compute the event shares, the new-item rate and the lift.
    total_events, total_new = int(table["n_events"].sum()), int(table["n_new"].sum())
    table["share_events"] = table["n_events"] / total_events
    table["share_new_events"] = table["n_new"] / total_new if total_new else np.nan
    table["rate"] = table["n_new"] / table["n_events"]
    table["lift"] = table["share_new_events"] / table["share_events"]
    # Compute the breadth over all users and over the buyers, and the known-category share.
    table["breadth_all"] = table["n_users_new"] / len(users)
    with np.errstate(divide="ignore", invalid="ignore"):
        table["breadth_buyers"] = np.where(table["n_users_bought"] > 0, table["n_users_new"] / table["n_users_bought"], np.nan)
        table["known_category_share"] = np.where(table["n_new"] > 0, table["n_new_known_category"] / table["n_new"], np.nan)
    # Flag the categories whose name is the missing label.
    table["is_missing"] = [is_missing(name) for name in table.index]
    # Add the user-level bootstrap bounds of rate and lift, or NaN without a bootstrap.
    categories = pd.Index(table.index)
    if n_boot > 0 and len(categories):
        num = user_category_matrix(events, level, users, categories, events["is_new"].to_numpy())
        den = user_category_matrix(events, level, users, categories)
        ci = bootstrap_ratios(num, den, n_boot, seed)
        for key, values in ci.items():
            table[key] = values
    else:
        for key in ("rate_lo", "rate_hi", "lift_lo", "lift_hi"):
            table[key] = np.nan
    # Return the table with the category as a column.
    table.index.name = "category"
    return table.reset_index()


def table_rows(table: pd.DataFrame) -> list[dict[str, Any]]:
    """JSON rows of a category table (NaN -> None, CIs as [lo, hi])."""
    out = []
    # Convert every record of the table to a JSON row.
    for record in table.to_dict("records"):
        row = {}
        for key, value in record.items():
            # Skip the bound columns, which are written as intervals below.
            if key in ("rate_lo", "rate_hi", "lift_lo", "lift_hi", "e2_lo", "e2_hi"):
                continue
            # Convert NaN to None and numpy scalars to Python scalars.
            if isinstance(value, float) and np.isnan(value):
                value = None
            elif isinstance(value, (np.integer,)):
                value = int(value)
            elif isinstance(value, (np.floating,)):
                value = float(value)
            elif isinstance(value, (np.bool_,)):
                value = bool(value)
            row[key] = value
        # Add the rate and lift intervals when the table has them.
        if "rate_lo" in record:
            row["rate_ci"] = _ci(record["rate_lo"], record["rate_hi"])
            row["lift_ci"] = _ci(record["lift_lo"], record["lift_hi"])
        # Add the interval of the E2 hit rate when the table has it.
        if "e2_lo" in record:
            row["e2_hit_rate_ci"] = _ci(record["e2_lo"], record["e2_hi"])
        out.append(row)
    # Return the JSON rows.
    return out


def support_mask(table: pd.DataFrame, level_name: str, dataset: str, count_column: str = "n_new") -> pd.Series:
    """Coarse level: every category; fine level: categories with >= SUPPORT_MIN_NEW[dataset] events in ``count_column``."""
    # Keep every category at the coarse level.
    if level_name == "coarse":
        return pd.Series(True, index=table.index)
    # Keep the fine categories with enough events in the count column.
    return table[count_column] >= SUPPORT_MIN_NEW[dataset]


def rankings(table: pd.DataFrame, eligible: pd.Series) -> dict[str, list[dict[str, Any]]]:
    """Top-10 lists among ``eligible`` rows: volume (n_new), propensity (rate), lift top and bottom, breadth (breadth_all)."""
    # Restrict the table to the eligible rows.
    sub = table[eligible]

    def pick(column: str, ascending: bool, extra: list[str]) -> list[dict[str, Any]]:
        '''Return the top TOP_N rows by ``column`` (order ``ascending``) with the ``extra`` keys.'''
        # Sort by the column, break ties by more new-item events and keep the first TOP_N rows.
        ordered = sub.sort_values([column, "n_new"], ascending=[ascending, False]).head(TOP_N)
        out = []
        for record in ordered.to_dict("records"):
            # Start the entry with the category name and its missing flag.
            item = {"category": record["category"], "is_missing": bool(record["is_missing"])}
            # Copy the ranked column and the extra columns, as int for counts and float otherwise.
            for key in [column, *extra]:
                value = record.get(key)
                if value is None or (isinstance(value, float) and np.isnan(value)):
                    item[key] = None
                else:
                    item[key] = int(value) if key.startswith("n_") else float(value)
            # Add the bootstrap interval when the ranked column is the rate or the lift.
            if "rate_lo" in record and column in ("rate", "lift"):
                item[f"{column}_ci"] = _ci(record[f"{column}_lo"], record[f"{column}_hi"])
            out.append(item)
        return out

    # Return the five rankings.
    return {
        "volume": pick("n_new", False, ["share_new_events", "rate"]),
        "propensity": pick("rate", False, ["n_new", "n_events"]),
        "lift_top": pick("lift", False, ["n_new", "rate"]),
        "lift_bottom": pick("lift", True, ["n_new", "rate"]),
        "breadth": pick("breadth_all", False, ["breadth_buyers", "n_users_new", "n_new"]),
    }


# ---------------------------------------------------------------------------
# predictability (eval target)
# ---------------------------------------------------------------------------
def list_hits(lists: pd.DataFrame, new_target: pd.DataFrame) -> pd.Series:
    """Boolean per row of ``new_target``: the (user, item) pair appears in the top-K rows of ``lists``."""
    # Take the distinct (user, item) pairs of the top-K rows.
    top = lists.loc[lists["rank"] <= K, ["user_id", "item_id"]].drop_duplicates().assign(hit=True)
    # Left-join the target rows to these pairs.
    merged = new_target[["user_id", "item_id"]].merge(top, on=["user_id", "item_id"], how="left")
    # Return whether each target row found a pair.
    return merged["hit"].notna().to_numpy()


def predictability_table(new_target: pd.DataFrame, e2: pd.DataFrame, pop_list: pd.DataFrame, level: str, users: pd.Index,
                         n_boot: int, seed: int) -> pd.DataFrame:
    """Per category: target new items, E2 and popularity hit rates (category-level new-item recall), E2 slot share and mismatch."""
    # Keep one row per (user, item) of the target new items.
    frame = new_target[["user_id", "item_id", level]].drop_duplicates(["user_id", "item_id"]).reset_index(drop=True)
    # Flag the rows that are in the E2 list and in the popularity list.
    frame["e2_hit"] = list_hits(e2, frame)
    frame["pop_hit"] = list_hits(pop_list, frame)
    # Count the target new items and the hits of both lists per category.
    grouped = frame.groupby(level, sort=True)
    table = pd.DataFrame({"n_target_new": grouped.size(), "e2_hits": grouped["e2_hit"].sum().astype("int64"),
                          "pop_hits": grouped["pop_hit"].sum().astype("int64")})
    # Count the top-K E2 slots per category.
    slots = e2[e2["rank"] <= K]
    slot_counts = slots.groupby(level).size()
    table["e2_slots"] = slot_counts.reindex(table.index).fillna(0).astype("int64")
    # Add the categories that hold E2 slots but have no target new item.
    extra = slot_counts[~slot_counts.index.isin(table.index)]           # categories recommended but never bought as new target items
    if len(extra):
        extra_table = pd.DataFrame({"n_target_new": 0, "e2_hits": 0, "pop_hits": 0, "e2_slots": extra.astype("int64")})
        table = pd.concat([table, extra_table]).sort_index()
    # Compute the share of target new items and the share of E2 slots per category.
    total_new, total_slots = int(table["n_target_new"].sum()), int(len(slots))
    table["target_new_share"] = table["n_target_new"] / total_new if total_new else np.nan
    table["e2_slot_share"] = table["e2_slots"] / total_slots if total_slots else np.nan
    # Compute both hit rates and the ratio of slot share to target share.
    with np.errstate(divide="ignore", invalid="ignore"):
        table["e2_hit_rate"] = np.where(table["n_target_new"] > 0, table["e2_hits"] / table["n_target_new"], np.nan)
        table["pop_hit_rate"] = np.where(table["n_target_new"] > 0, table["pop_hits"] / table["n_target_new"], np.nan)
        table["slot_ratio"] = np.where(table["target_new_share"] > 0, table["e2_slot_share"] / table["target_new_share"], np.nan)
    # Compute the difference of slot share and target share.
    table["slot_minus_target_share"] = table["e2_slot_share"] - table["target_new_share"]
    # Flag the categories whose name is the missing label.
    table["is_missing"] = [is_missing(name) for name in table.index]
    # Add the user-level bootstrap bounds of the E2 hit rate, or NaN without a bootstrap.
    categories = pd.Index(table.index)
    if n_boot > 0 and len(categories):
        num = user_category_matrix(frame, level, users, categories, frame["e2_hit"].to_numpy())
        den = user_category_matrix(frame, level, users, categories)
        ci = bootstrap_ratios(num, den, n_boot, seed)
        table["e2_lo"], table["e2_hi"] = ci["rate_lo"], ci["rate_hi"]
    else:
        table["e2_lo"] = table["e2_hi"] = np.nan
    # Return the table with the category as a column.
    table.index.name = "category"
    return table.reset_index()


def quadrants(category: pd.DataFrame, predict: pd.DataFrame, dataset: str) -> dict[str, Any]:
    """2 x 2 placement of fine categories passing the support filter on target new items: new-item rate (eval, all events)
    above / below its median x E2 hit rate above / below its median (high = strictly above the median)."""
    # Join the predictability table to the rate and new-item count of the category table.
    merged = predict.merge(category[["category", "rate", "n_new"]], on="category", how="inner")
    # Keep the categories that pass the support filter and have both values.
    merged = merged[support_mask(merged, "fine", dataset, "n_target_new") & merged["rate"].notna() & merged["e2_hit_rate"].notna()]
    # Return an empty placement when no category is left.
    if merged.empty:
        return {"n_categories": 0, "median_rate": None, "median_e2_hit_rate": None, "quadrants": {}}
    # Compute both medians and flag the categories strictly above each.
    median_rate, median_e2 = float(merged["rate"].median()), float(merged["e2_hit_rate"].median())
    high_rate, high_e2 = merged["rate"] > median_rate, merged["e2_hit_rate"] > median_e2
    # Define the four cells of the placement.
    names = {"explored_hard_to_predict": high_rate & ~high_e2, "explored_easy_to_predict": high_rate & high_e2,
             "rarely_explored_hard_to_predict": ~high_rate & ~high_e2, "rarely_explored_easy_to_predict": ~high_rate & high_e2}
    # List the categories of each cell, sorted by target new items and then by name.
    out = {}
    for name, mask in names.items():
        sub = merged[mask].sort_values(["n_target_new", "category"], ascending=[False, True])
        out[name] = [{"category": r["category"], "is_missing": bool(r["is_missing"]), "rate": float(r["rate"]), "e2_hit_rate": float(r["e2_hit_rate"]),
                      "n_target_new": int(r["n_target_new"]), "n_new_all": int(r["n_new"])} for r in sub.to_dict("records")]
    # Return the medians, the definition text and the four lists.
    return {"n_categories": int(len(merged)), "median_rate": median_rate, "median_e2_hit_rate": median_e2,
            "definition": "fine categories with >= support new target items; rate = new-item rate over all eval events (history + target); "
                          "high = strictly above the median of the placed categories; lists sorted by target new items", "quadrants": out}


def mismatches(predict: pd.DataFrame, level_name: str, dataset: str) -> dict[str, Any]:
    """Categories where the E2 slot share exceeds the target new-item share most (over-served) and the reverse (under-served)."""
    # Keep the categories passing the support filter or holding at least 1 % of the E2 slots.
    eligible = predict[support_mask(predict, level_name, dataset, "n_target_new") | (predict["e2_slot_share"] >= 0.01)]
    # Drop the categories without a share difference.
    eligible = eligible[eligible["slot_minus_target_share"].notna()]

    def rows(frame: pd.DataFrame) -> list[dict[str, Any]]:
        '''Return the JSON rows (shares, difference, ratio, count, E2 hit rate) of ``frame``.'''
        return [{"category": r["category"], "is_missing": bool(r["is_missing"]), "e2_slot_share": float(r["e2_slot_share"]),
                 "target_new_share": float(r["target_new_share"]), "difference": float(r["slot_minus_target_share"]),
                 "ratio": None if pd.isna(r["slot_ratio"]) else float(r["slot_ratio"]), "n_target_new": int(r["n_target_new"]),
                 "e2_hit_rate": None if pd.isna(r["e2_hit_rate"]) else float(r["e2_hit_rate"])} for r in frame.to_dict("records")]

    # Return the counts and the TOP_N largest differences in each direction.
    return {"n_categories": int(len(eligible)),
            "n_over_served": int((eligible["slot_minus_target_share"] > 0).sum()),
            "n_under_served": int((eligible["slot_minus_target_share"] < 0).sum()),
            "definition": "difference = E2 slot share - target new-item share; categories passing the support filter on target new items "
                          "or holding >= 1 % of the E2 slots",
            "over_served": rows(eligible.sort_values(["slot_minus_target_share", "category"], ascending=[False, True]).head(TOP_N)),
            "under_served": rows(eligible.sort_values(["slot_minus_target_share", "category"], ascending=[True, True]).head(TOP_N))}


# ---------------------------------------------------------------------------
# robustness
# ---------------------------------------------------------------------------
def spearman_against_fit(eval_table: pd.DataFrame, fit_table: pd.DataFrame, eligible: pd.Series) -> dict[str, Any]:
    """Spearman rank correlation of volume (n_new) and propensity (rate) between the eval and fit tables over the eligible eval categories present in both."""
    # Join the eligible eval categories to the fit table on the category name.
    merged = eval_table[eligible].merge(fit_table[["category", "n_new", "rate"]], on="category", suffixes=("_eval", "_fit"))
    # Keep the categories with a rate in both splits.
    merged = merged[merged["rate_eval"].notna() & merged["rate_fit"].notna()]
    out: dict[str, Any] = {"n_categories": int(len(merged))}
    # Compute the Spearman correlation of each measure when at least 3 categories are left.
    for name, column in (("volume", "n_new"), ("propensity", "rate")):
        if len(merged) >= 3:
            result = spearmanr(merged[f"{column}_eval"], merged[f"{column}_fit"])
            out[name] = {"spearman": float(result.statistic), "p_value": float(result.pvalue)}
        else:
            out[name] = {"spearman": None, "p_value": None}
    # Return the category count and both correlations.
    return out


def propensity_by_history_bin(events: pd.DataFrame, lengths: pd.Series, dataset: str, level: str = "L1") -> list[dict[str, Any]]:
    """Coarse new-item rate per T5 history-length bin (users binned by distinct history baskets)."""
    # Look up the history length of the user of every event.
    user_length = events["user_id"].map(lengths)
    out = []
    for name, lo, hi in bin_labels(HISTORY_BINS[dataset]):
        # Select the events of the users whose history length falls in the bin.
        sub = events[(user_length > lo) & (user_length <= hi)]
        # Count the events and new-item events per category in the bin.
        grouped = sub.groupby(level)
        table = pd.DataFrame({"n_events": grouped.size(), "n_new": grouped["is_new"].sum().astype("int64")})
        # Compute the new-item rate and the share of new-item events per category.
        total_new = int(table["n_new"].sum())
        table["rate"] = table["n_new"] / table["n_events"]
        table["share_new_events"] = table["n_new"] / total_new if total_new else np.nan
        # Store the totals of the bin and its per-category rows.
        out.append({"bin": name, "n_users": int(sub["user_id"].nunique()), "n_events": int(len(sub)), "n_new": total_new,
                    "overall_rate": float(sub["is_new"].mean()) if len(sub) else None,
                    "categories": [{"category": c, "n_events": int(r["n_events"]), "n_new": int(r["n_new"]), "rate": float(r["rate"]),
                                    "share_new_events": None if pd.isna(r["share_new_events"]) else float(r["share_new_events"])}
                                   for c, r in table.iterrows()]})
    # Return one entry per bin.
    return out


# ---------------------------------------------------------------------------
# one dataset
# ---------------------------------------------------------------------------
def analyse_dataset(config: dict[str, Any], dataset: str, usercf: dict[str, Any], n_boot: int, seed: int) -> dict[str, Any]:
    '''Run the whole analysis of one ``dataset`` and return its JSON entry.

    ``config`` is the profile configuration, ``usercf`` the (k, weighting) selection of the
    user-based collaborative-filtering list (E2), ``n_boot`` and ``seed`` the bootstrap
    settings.
    '''
    # Start the timing dict and the entry with the fixed settings of the dataset.
    timings: dict[str, float] = {}
    entry: dict[str, Any] = {"label": LABEL, "level_names": LEVEL_NAMES[dataset], "support_min_new_fine": SUPPORT_MIN_NEW[dataset],
                             "usercf": usercf, "history_bins": [name for name, _, _ in bin_labels(HISTORY_BINS[dataset])]}

    # eval split
    t0 = time.perf_counter()
    # Load the eval split and take its users and history lengths.
    data = prepare(config, dataset, "eval")
    history, target, pop = data["history"], data["target"], data["pop"]
    users = pd.Index(target["user_id"].drop_duplicates().sort_values())
    lengths = history_lengths(history)
    timings["eval_load"] = time.perf_counter() - t0
    # Build the events of the eval split.
    t0 = time.perf_counter()
    events, rows = build_events(history, target)
    timings["eval_events"] = time.perf_counter() - t0
    # Store the counts and the overall new-item rate of each event set.
    entry["eval"] = {
        "n_users": int(len(users)),
        "n_rows": int(len(rows)), "n_events": int(len(events)), "n_new_events": int(events["is_new"].sum()),
        "n_target_events": int(events["is_target"].sum()), "n_target_new_events": int((events["is_target"] & events["is_new"]).sum()),
        "overall_rate": {s: float(events.loc[_set_mask(events, s), "is_new"].mean()) for s in EVENT_SETS},
        "levels": {},
    }
    # Build the category table and the rankings of every level and event set.
    event_tables: dict[tuple[str, str], pd.DataFrame] = {}
    t0 = time.perf_counter()
    for level_name, level in LEVELS.items():
        block: dict[str, Any] = {"n_categories": int(events[level].nunique()),
                                 "n_missing_label_categories": int(sum(is_missing(c) for c in events[level].unique())), "sets": {}}
        for event_set in EVENT_SETS:
            table = category_table(events[_set_mask(events, event_set)], rows, level, users, n_boot, seed)
            event_tables[(level_name, event_set)] = table
            eligible = support_mask(table, level_name, dataset)
            block["sets"][event_set] = {"n_categories": int(len(table)), "n_pass_support": int(eligible.sum()),
                                        "table": table_rows(table), "rankings": rankings(table, eligible)}
        entry["eval"]["levels"][level_name] = block
    timings["eval_category_tables"] = time.perf_counter() - t0

    # predictability: E2 and flat popularity lists on the eval history, hits on the target new items
    t0 = time.perf_counter()
    # Build the E2 list from the eval history.
    e2 = ucf_list(history, int(usercf["k"]), str(usercf["weighting"]), K, pop, block_size=int(config["usercf"]["block_size"]))
    timings["e2_list"] = time.perf_counter() - t0
    # Build the popularity list that excludes the history items of each user.
    t0 = time.perf_counter()
    pop_list = popularity_list(history, users, K, pop, exclude_history=True)
    timings["popularity_list"] = time.perf_counter() - t0
    # Attach the category levels of the items table to both lists.
    t0 = time.perf_counter()
    items = load_items(config, dataset)
    levels = config["levels"][dataset]
    e2 = attach_levels(e2, items, levels)
    pop_list = attach_levels(pop_list, items, levels)
    # Store the overall hit rates of both lists on the target new items.
    new_target = data["new_target"]
    entry["eval"]["predictability"] = {
        "n_target_new_items": int(len(new_target.drop_duplicates(["user_id", "item_id"]))),
        "n_e2_slots": int(len(e2[e2["rank"] <= K])),
        "overall_e2_hit_rate": float(list_hits(e2, new_target).mean()),
        "overall_pop_hit_rate": float(list_hits(pop_list, new_target).mean()),
        "definition": "hit rate = target new items of the category that appear in the user's top-10 list / target new items of the category "
                      "(item-level new-item recall per category); slot share = share of all top-10 E2 rows whose item is in the category",
        "levels": {},
    }
    # Build the predictability table and the mismatch lists of every level.
    predict_tables: dict[str, pd.DataFrame] = {}
    for level_name, level in LEVELS.items():
        table = predictability_table(new_target, e2, pop_list, level, users, n_boot, seed)
        predict_tables[level_name] = table
        eligible = support_mask(table, level_name, dataset, "n_target_new")
        ranked = table[eligible].sort_values("n_target_new", ascending=False)
        entry["eval"]["predictability"]["levels"][level_name] = {
            "n_categories": int(len(table)), "n_pass_support_target": int(eligible.sum()), "table": table_rows(table),
            "by_target_volume": [r["category"] for r in ranked.to_dict("records")],
            "mismatch": mismatches(table, level_name, dataset)}
    # Place the fine categories in the 2 x 2 table of new-item rate and E2 hit rate.
    entry["eval"]["quadrants_fine"] = quadrants(event_tables[("fine", "all")], predict_tables["fine"], dataset)
    timings["predictability"] = time.perf_counter() - t0

    # robustness: history-length bins (eval) and the fit split
    t0 = time.perf_counter()
    # Compute the coarse new-item rate per history-length bin on all eval events.
    entry["eval"]["coarse_propensity_by_history_bin"] = propensity_by_history_bin(events[_set_mask(events, "all")], lengths, dataset)
    # Free the eval frames before the fit split is loaded.
    del data, history, target, e2, pop_list, rows
    # Load the fit split and build its events.
    fit = prepare(config, dataset, "fit")
    fit_users = pd.Index(fit["target"]["user_id"].drop_duplicates().sort_values())
    fit_events, fit_rows = build_events(fit["history"], fit["target"])
    # Store the counts and the overall new-item rate of the fit split.
    entry["fit"] = {"role": "robustness check", "n_users": int(len(fit_users)), "n_events": int(len(fit_events)),
                    "n_new_events": int(fit_events["is_new"].sum()), "overall_rate": float(fit_events["is_new"].mean()), "levels": {}}
    # Build the fit tables without a bootstrap and correlate them with the eval tables.
    for level_name, level in LEVELS.items():
        fit_table = category_table(fit_events, fit_rows, level, fit_users, 0, seed)
        eligible_fit = support_mask(fit_table, level_name, dataset)
        eval_table = event_tables[(level_name, "all")]
        entry["fit"]["levels"][level_name] = {
            "n_categories": int(len(fit_table)), "n_pass_support": int(eligible_fit.sum()), "table": table_rows(fit_table),
            "rankings": rankings(fit_table, eligible_fit),
            "spearman_vs_eval": spearman_against_fit(eval_table, fit_table, support_mask(eval_table, level_name, dataset))}
    # Free the fit frames.
    del fit, fit_events, fit_rows
    timings["robustness"] = time.perf_counter() - t0
    # Store the timings and the peak memory and return the entry.
    entry["timing_seconds"] = timings
    entry["peak_memory_mib"] = peak_memory_mib()
    return entry


def _set_mask(events: pd.DataFrame, event_set: str) -> np.ndarray:
    '''Return the boolean row mask of ``events`` for ``event_set`` (all, history or target).'''
    # Select every event.
    if event_set == "all":
        return np.ones(len(events), dtype=bool)
    # Select the events of the history baskets.
    if event_set == "history":
        return ~events["is_target"].to_numpy()
    # Select the events of the target basket.
    return events["is_target"].to_numpy()


# ---------------------------------------------------------------------------
# figures
# ---------------------------------------------------------------------------
def _style() -> None:
    '''Set the matplotlib font, colours and backgrounds used by all figures.'''
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9.5, "axes.edgecolor": INK["axis"], "axes.labelcolor": INK["secondary"],
                         "xtick.color": INK["secondary"], "ytick.color": INK["secondary"], "text.color": INK["primary"],
                         "figure.facecolor": INK["surface"], "savefig.facecolor": INK["surface"], "axes.facecolor": INK["surface"]})


def _label(name: str, missing: bool) -> str:
    '''Return ``name`` as text, marked "(missing label)" when ``missing`` is true.'''
    return f"{name} (missing label)" if missing else str(name)


def figure_rate_coarse(entries: dict[str, dict[str, Any]], out_dir: Path) -> dict[str, Any]:
    """F-1: coarse categories, new-item rate with 95 % CI and volume share (eval, all events), horizontal bars sorted by rate."""
    # Collect one row per dataset and coarse category from the eval tables of all events.
    rows = []
    for dataset, entry in entries.items():
        for r in entry["eval"]["levels"]["coarse"]["sets"]["all"]["table"]:
            rows.append({"dataset": dataset, "category": r["category"], "is_missing": r["is_missing"], "n_events": r["n_events"], "n_new": r["n_new"],
                         "rate": r["rate"], "rate_lo": r["rate_ci"][0] if r["rate_ci"] else None, "rate_hi": r["rate_ci"][1] if r["rate_ci"] else None,
                         "share_new_events": r["share_new_events"]})
    # Build the frame of plotted values and write it to the CSV file.
    frame = pd.DataFrame(rows, columns=["dataset", "category", "is_missing", "n_events", "n_new", "rate", "rate_lo", "rate_hi", "share_new_events"])
    csv_path = out_dir / f"{FIGURES['rate_coarse']}.csv"
    frame.to_csv(csv_path, index=False)
    # Create one panel per dataset, tall enough for the largest number of categories.
    _style()
    n = len(entries)
    heights = [max(3.0, 0.28 * len(frame[frame["dataset"] == d]) + 1.2) for d in entries]
    fig, axes = plt.subplots(1, n, figsize=(6.4 * n, max(heights)), layout="constrained")
    axes = np.atleast_1d(axes)
    for ax, dataset in zip(axes, entries):
        # Sort the categories of the dataset by rate and compute the error-bar lengths.
        sub = frame[frame["dataset"] == dataset].sort_values("rate", ascending=True)
        y = np.arange(len(sub))
        lo = (sub["rate"] - sub["rate_lo"]).fillna(0).to_numpy()
        hi = (sub["rate_hi"] - sub["rate"]).fillna(0).to_numpy()
        # Fade the bars of categories with fewer than FADE_MIN_EVENTS events.
        faded = (sub["n_events"] < FADE_MIN_EVENTS).to_numpy()
        colors = [matplotlib.colors.to_rgba(COLORS[dataset], 0.35 if fade else 1.0) for fade in faded]
        # Draw the horizontal bars with their intervals.
        ax.barh(y, sub["rate"], color=colors, height=0.72, xerr=np.vstack([lo, hi]), ecolor=INK["secondary"], error_kw={"elinewidth": 1.0, "capsize": 2})
        # Take the largest upper bound as the reference of the x range.
        x_max = float(np.nanmax(sub["rate_hi"].fillna(sub["rate"]))) if len(sub) else 1.0
        # Label each bar with its share of new-item events and its event count.
        for yi, (rate, share, n_events) in enumerate(zip(sub["rate"], sub["share_new_events"], sub["n_events"])):
            share_text = "< 0.1 %" if share < 0.001 else f"{share:.1%}"
            ax.annotate(f"{share_text} of new-item events ({n_events:,} events)",
                        (min(rate + (sub['rate_hi'].iloc[yi] - rate if pd.notna(sub['rate_hi'].iloc[yi]) else 0), x_max), yi),
                        xytext=(5, 0), textcoords="offset points", va="center", fontsize=7.5, color=INK["secondary"])
        # Set the tick labels, x range, axis label, title and grid of the panel.
        ax.set_yticks(y)
        ax.set_yticklabels([_label(c, m) for c, m in zip(sub["category"], sub["is_missing"])], fontsize=8)
        ax.set_xlim(0, x_max * 1.45)
        ax.set_xlabel("new-item rate (share of events that are new items), 95 % CI")
        ax.set_title(f"{NAMES[dataset]}: {entries[dataset]['level_names']['coarse']} ({len(sub)} categories)", fontsize=10, color=INK["primary"])
        ax.grid(True, axis="x", color=INK["grid"], lw=0.8)
        ax.grid(False, axis="y")
        # Hide the top and right frame lines.
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    # Add the common title, save the PNG file and close the figure.
    fig.suptitle(f"New-item rate by coarse category, eval split, history + target baskets; label = share of all new-item events; "
                 f"faded bars = fewer than {FADE_MIN_EVENTS} events\n{LABEL}", fontsize=9.5, color=INK["secondary"])
    png_path = out_dir / f"{FIGURES['rate_coarse']}.png"
    fig.savefig(png_path, dpi=200)
    plt.close(fig)
    # Return the file names and the number of plotted rows.
    return {"png": png_path.name, "csv": csv_path.name, "points": int(len(frame))}


def figure_lift_fine(entries: dict[str, dict[str, Any]], out_dir: Path) -> dict[str, Any]:
    """F-2: fine categories passing the support filter, lift (log scale) top 15 and bottom 10 (eval, all events), dots with 95 % CIs."""
    rows = []
    for dataset, entry in entries.items():
        # Take the fine categories of the dataset that pass the support filter and have a lift.
        table = pd.DataFrame(entry["eval"]["levels"]["fine"]["sets"]["all"]["table"])
        eligible = table[support_mask(table, "fine", dataset) & table["lift"].notna()].sort_values("lift", ascending=False)
        # Select the 15 highest and the TOP_N lowest lifts.
        top, bottom = eligible.head(15), eligible.tail(TOP_N)
        # Collect the rows of both groups, skipping bottom rows already in the top group.
        for group, sub in (("top", top), ("bottom", bottom)):
            for r in sub.to_dict("records"):
                if group == "bottom" and r["category"] in set(top["category"]):
                    continue
                rows.append({"dataset": dataset, "group": group, "category": r["category"], "is_missing": r["is_missing"], "n_new": r["n_new"],
                             "lift": r["lift"], "lift_lo": r["lift_ci"][0] if r["lift_ci"] else None, "lift_hi": r["lift_ci"][1] if r["lift_ci"] else None})
    # Build the frame of plotted values and write it to the CSV file.
    frame = pd.DataFrame(rows, columns=["dataset", "group", "category", "is_missing", "n_new", "lift", "lift_lo", "lift_hi"])
    csv_path = out_dir / f"{FIGURES['lift_fine']}.csv"
    frame.to_csv(csv_path, index=False)
    # Create one panel per dataset.
    _style()
    n = len(entries)
    fig, axes = plt.subplots(1, n, figsize=(6.4 * n, 7.2), layout="constrained")
    axes = np.atleast_1d(axes)
    for ax, dataset in zip(axes, entries):
        # Sort the rows of the dataset by lift.
        sub = frame[frame["dataset"] == dataset].sort_values("lift", ascending=True).reset_index(drop=True)
        # Leave only a title in the panel when the dataset has no row.
        if sub.empty:
            ax.set_title(f"{NAMES[dataset]}: no fine category passes the support filter", fontsize=10, color=INK["primary"])
            continue
        # Compute the error-bar lengths.
        y = np.arange(len(sub))
        lo = (sub["lift"] - sub["lift_lo"]).fillna(0).to_numpy()
        hi = (sub["lift_hi"] - sub["lift"]).fillna(0).to_numpy()
        # Draw the reference line at lift 1 and the dots with their intervals.
        ax.axvline(1.0, color=INK["muted"], lw=1, ls="--")
        ax.errorbar(sub["lift"], y, xerr=np.vstack([lo, hi]), fmt="o", color=COLORS[dataset], ecolor=INK["secondary"], elinewidth=1.0, capsize=2,
                    ms=6, mec=INK["surface"], mew=1.2)
        # Draw a line between the bottom group and the top group when both are present.
        n_bottom = int((sub["group"] == "bottom").sum())
        if 0 < n_bottom < len(sub):
            ax.axhline(n_bottom - 0.5, color=INK["grid"], lw=1)
        # Set the log scale, tick labels, axis label, title and grid of the panel.
        ax.set_xscale("log")
        ax.set_yticks(y)
        ax.set_yticklabels([_label(c, m) for c, m in zip(sub["category"], sub["is_missing"])], fontsize=8)
        ax.set_xlabel("lift = share of new-item events / share of all events (log scale), 95 % CI")
        ax.set_title(f"{NAMES[dataset]}: {entries[dataset]['level_names']['fine']}, top 15 and bottom 10 by lift", fontsize=10, color=INK["primary"])
        ax.grid(True, axis="x", which="major", color=INK["grid"], lw=0.8)
        ax.grid(False, axis="y")
        # Hide the top and right frame lines.
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    # Add the common title, save the PNG file and close the figure.
    fig.suptitle(f"Lift of fine categories, eval split, history + target baskets, categories above the support filter; dashed = lift 1\n{LABEL}",
                 fontsize=9.5, color=INK["secondary"])
    png_path = out_dir / f"{FIGURES['lift_fine']}.png"
    fig.savefig(png_path, dpi=200)
    plt.close(fig)
    # Return the file names and the number of plotted rows.
    return {"png": png_path.name, "csv": csv_path.name, "points": int(len(frame))}


def figure_rate_vs_e2(entries: dict[str, dict[str, Any]], out_dir: Path) -> dict[str, Any]:
    """F-3: fine categories of the 2 x 2 placement, new-item rate vs E2 hit rate, point size = volume, median lines, top 8 by volume labelled."""
    # Collect one row per category of the 2 x 2 placement with the medians of its dataset.
    rows = []
    for dataset, entry in entries.items():
        q = entry["eval"]["quadrants_fine"]
        for name, items in q["quadrants"].items():
            for r in items:
                rows.append({"dataset": dataset, "quadrant": name, "category": r["category"], "is_missing": r["is_missing"], "rate": r["rate"],
                             "e2_hit_rate": r["e2_hit_rate"], "n_target_new": r["n_target_new"], "n_new_all": r["n_new_all"],
                             "median_rate": q["median_rate"], "median_e2_hit_rate": q["median_e2_hit_rate"]})
    # Build the frame of plotted values and write it to the CSV file.
    frame = pd.DataFrame(rows, columns=["dataset", "quadrant", "category", "is_missing", "rate", "e2_hit_rate", "n_target_new", "n_new_all",
                                        "median_rate", "median_e2_hit_rate"])
    csv_path = out_dir / f"{FIGURES['rate_vs_e2_fine']}.csv"
    frame.to_csv(csv_path, index=False)
    # Create one panel per dataset.
    _style()
    n = len(entries)
    fig, axes = plt.subplots(1, n, figsize=(6.2 * n, 5.2), layout="constrained")
    axes = np.atleast_1d(axes)
    for ax, dataset in zip(axes, entries):
        # Take the rows of the dataset.
        sub = frame[frame["dataset"] == dataset]
        # Leave only a title in the panel when the dataset has no row.
        if sub.empty:
            ax.set_title(f"{NAMES[dataset]}: no category passes the support filter", fontsize=10)
            continue
        # Scale the point size with the number of target new items.
        size = 20 + 380 * sub["n_target_new"] / sub["n_target_new"].max()
        # Draw the points and the two median lines.
        ax.scatter(sub["rate"], sub["e2_hit_rate"], s=size, color=COLORS[dataset], alpha=0.55, edgecolor=INK["surface"], linewidth=1.0)
        ax.axvline(sub["median_rate"].iloc[0], color=INK["muted"], lw=1, ls="--")
        ax.axhline(sub["median_e2_hit_rate"].iloc[0], color=INK["muted"], lw=1, ls="--")
        # Label the 8 categories with the most target new items.
        labelled = sub.sort_values("n_target_new", ascending=False).head(8).sort_values(["rate", "e2_hit_rate"])
        for i, r in enumerate(labelled.to_dict("records")):
            ax.annotate(_label(r["category"], r["is_missing"]), (r["rate"], r["e2_hit_rate"]), xytext=(6, 4 if i % 2 == 0 else -11),
                        textcoords="offset points", fontsize=7.5, color=INK["primary"])
        # Set the axis labels, axis origins, title and grid of the panel.
        ax.set_xlabel("new-item rate (eval, history + target events)")
        ax.set_ylabel("E2 user-CF top-10 hit rate on target new items")
        ax.set_xlim(left=0)
        ax.set_ylim(bottom=0)
        ax.set_title(f"{NAMES[dataset]}: {entries[dataset]['level_names']['fine']} ({len(sub)} categories; point size = target new items)",
                     fontsize=10, color=INK["primary"])
        ax.grid(True, color=INK["grid"], lw=0.8)
        # Hide the top and right frame lines.
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    # Add the common title, save the PNG file and close the figure.
    fig.suptitle(f"New-item rate vs E2 hit rate of fine categories; dashed = medians; lower right = explored but hard to predict\n{LABEL}",
                 fontsize=9.5, color=INK["secondary"])
    png_path = out_dir / f"{FIGURES['rate_vs_e2_fine']}.png"
    fig.savefig(png_path, dpi=200)
    plt.close(fig)
    # Return the file names and the number of plotted rows.
    return {"png": png_path.name, "csv": csv_path.name, "points": int(len(frame))}


# ---------------------------------------------------------------------------
# markdown
# ---------------------------------------------------------------------------
def _f(value: float | None, digits: int = 3) -> str:
    '''Format ``value`` with ``digits`` decimals; None or NaN gives "-".'''
    return "-" if value is None or (isinstance(value, float) and math.isnan(value)) else f"{value:.{digits}f}"


def _pct(value: float | None, digits: int = 1) -> str:
    '''Format ``value`` as a percentage with ``digits`` decimals; None or NaN gives "-".'''
    return "-" if value is None or (isinstance(value, float) and math.isnan(value)) else f"{100 * value:.{digits}f} %"


def _with_ci(value: float | None, ci: list[float] | None, digits: int = 3) -> str:
    '''Format ``value`` with ``digits`` decimals, followed by the interval ``ci`` if given.'''
    # Print a dash for a missing value.
    if value is None:
        return "-"
    # Append the interval in square brackets when there is one.
    return f"{value:.{digits}f}" + (f" [{ci[0]:.{digits}f}, {ci[1]:.{digits}f}]" if ci else "")


def _name(row: dict[str, Any]) -> str:
    '''Return the display name of the category of ``row``, with its missing-label mark.'''
    return _label(row["category"], row.get("is_missing", False))


def _table(header: list[str], rows: list[list[str]]) -> list[str]:
    '''Return the markdown table lines for ``header`` and ``rows`` plus an empty line.'''
    # Write the header row and the separator row.
    lines = ["| " + " | ".join(header) + " |", "| " + " | ".join("---" for _ in header) + " |"]
    # Write one line per row, replacing "|" inside cells by "/".
    lines.extend("| " + " | ".join(str(cell).replace("|", "/") for cell in row) + " |" for row in rows)
    # Return the lines followed by an empty line.
    return lines + [""]


def render_rankings(block: dict[str, Any]) -> list[str]:
    '''Return the markdown table of the five rankings stored in ``block``.'''
    # Read the rankings of the block.
    ranks = block["rankings"]
    rows = []
    for i in range(TOP_N):
        # Start the table row with its position.
        cells = [str(i + 1)]
        # Add the entry of each ranking at this position, or a dash past the end of its list.
        for key, fmt in (("volume", lambda r: f"{_name(r)} ({r['n_new']:,}; {_pct(r['share_new_events'])})"),
                         ("propensity", lambda r: f"{_name(r)} ({_with_ci(r['rate'], r.get('rate_ci'))}; n {r['n_new']:,})"),
                         ("lift_top", lambda r: f"{_name(r)} ({_with_ci(r['lift'], r.get('lift_ci'), 2)})"),
                         ("lift_bottom", lambda r: f"{_name(r)} ({_with_ci(r['lift'], r.get('lift_ci'), 2)})"),
                         ("breadth", lambda r: f"{_name(r)} ({_pct(r['breadth_all'])} / {_pct(r['breadth_buyers'])})")):
            items = ranks[key]
            cells.append(fmt(items[i]) if i < len(items) else "-")
        rows.append(cells)
    # Return the table with one column per ranking.
    return _table(["#", "volume (new events; share)", "propensity (rate [CI]; new events)", "lift top (lift [CI])", "lift bottom (lift [CI])",
                   "breadth (all users / buyers)"], rows)


def render_dataset(entry: dict[str, Any], dataset: str) -> list[str]:
    '''Return the markdown lines of the section of ``dataset`` built from its ``entry``.'''
    # Write the heading and the summary bullets of the dataset.
    ev = entry["eval"]
    lines = [f"## {NAMES[dataset]}", "",
             f"- Eval split: {ev['n_users']:,} users; {ev['n_rows']:,} distinct (user, basket, item) rows; {ev['n_events']:,} events from the 2nd basket on, "
             f"{ev['n_new_events']:,} new-item events (overall rate {_f(ev['overall_rate']['all'])}; history {_f(ev['overall_rate']['history'])}, "
             f"target {_f(ev['overall_rate']['target'])}); target events {ev['n_target_events']:,} of which new {ev['n_target_new_events']:,}.",
             f"- Levels: coarse = {entry['level_names']['coarse']} ({ev['levels']['coarse']['n_categories']} categories, "
             f"{ev['levels']['coarse']['n_missing_label_categories']} named 'missing'); fine = {entry['level_names']['fine']} "
             f"({ev['levels']['fine']['n_categories']} categories, {ev['levels']['fine']['n_missing_label_categories']} named 'missing'). "
             f"Fine-level support filter >= {entry['support_min_new_fine']} new-item events: passing " + ", ".join(
                 f"{s} {ev['levels']['fine']['sets'][s]['n_pass_support']}" for s in EVENT_SETS)
             + f"; target new items >= {entry['support_min_new_fine']}: {ev['predictability']['levels']['fine']['n_pass_support_target']}.",
             f"- E2 user-CF k = {entry['usercf']['k']}, weighting {entry['usercf']['weighting']} ({'verified against the reference tuning document' if entry['usercf']['verified'] else entry['usercf']['source']}); "
             f"overall item-level new-item hit rate of the top 10: E2 {_f(ev['predictability']['overall_e2_hit_rate'])}, flat popularity "
             f"{_f(ev['predictability']['overall_pop_hit_rate'])} ({ev['predictability']['n_target_new_items']:,} target new items; {ev['predictability']['n_e2_slots']:,} E2 slots).", ""]

    # Write the coarse category table of all eval events, sorted by volume.
    coarse_all = ev["levels"]["coarse"]["sets"]["all"]["table"]
    lines += [f"### {NAMES[dataset]}: coarse categories, eval split, history + target (sorted by volume)", ""]
    rows = []
    for r in sorted(coarse_all, key=lambda r: -r["n_new"]):
        rows.append([_name(r), f"{r['n_events']:,}", f"{r['n_new']:,}", _pct(r["share_new_events"]), _with_ci(r["rate"], r["rate_ci"]),
                     _with_ci(r["lift"], r["lift_ci"], 2), f"{_pct(r['breadth_all'])} / {_pct(r['breadth_buyers'])}", _pct(r["known_category_share"])])
    lines += _table(["category", "events", "new-item events", "share of new events", "rate [95 % CI]", "lift [95 % CI]",
                     "breadth all / buyers", "new item in a known category"], rows)
    # Write the ranking tables of both levels for the history and target event sets.
    for level_name in ("coarse", "fine"):
        for event_set in ("history", "target"):
            block = ev["levels"][level_name]["sets"][event_set]
            lines += [f"### {NAMES[dataset]}: {level_name} rankings, eval {event_set} baskets ({block['n_pass_support']} of {block['n_categories']} categories eligible)", ""]
            lines += render_rankings(block)

    # Write the predictability table and the mismatch table of both levels.
    pred = ev["predictability"]
    for level_name in ("coarse", "fine"):
        block = pred["levels"][level_name]
        # Index the table rows by category and take the names to print.
        table = {r["category"]: r for r in block["table"]}
        names = block["by_target_volume"][: (len(block["by_target_volume"]) if level_name == "coarse" else 15)]
        lines += [f"### {NAMES[dataset]}: predictability of target new items, {level_name} level (eligible categories by target volume"
                  f"{'' if level_name == 'coarse' else ', top 15'})", ""]
        # Add one row per printed category.
        rows = []
        for name in names:
            r = table[name]
            rows.append([_name(r), f"{r['n_target_new']:,}", _pct(r["target_new_share"]), _with_ci(r["e2_hit_rate"], r["e2_hit_rate_ci"]), _f(r["pop_hit_rate"]),
                         _pct(r["e2_slot_share"]), f"{100 * r['slot_minus_target_share']:+.1f} pp", _f(r["slot_ratio"], 2)])
        lines += _table(["category", "target new items", "share of target new items", "E2 hit rate [95 % CI]", "popularity hit rate", "E2 slot share",
                         "slot share - target share", "ratio"], rows)
        # Write the counts of the mismatch and its two top lists side by side.
        mm = block["mismatch"]
        lines += [f"Mismatch ({mm['n_categories']} categories considered; {mm['n_over_served']} with an E2 slot share above their target new-item share, "
                  f"{mm['n_under_served']} below): the largest differences in both directions (rows past the counts are the least negative / least positive):", ""]
        rows = []
        for i in range(TOP_N):
            cells = [str(i + 1)]
            for key in ("over_served", "under_served"):
                items = mm[key]
                cells.append(f"{_name(items[i])} ({_pct(items[i]['e2_slot_share'])} vs {_pct(items[i]['target_new_share'])}; {100 * items[i]['difference']:+.1f} pp)"
                             if i < len(items) else "-")
            rows.append(cells)
        lines += _table(["#", "over-served (slot share vs target share)", "under-served (slot share vs target share)"], rows)

    # Write the four cells of the 2 x 2 placement of the fine categories.
    q = ev["quadrants_fine"]
    lines += [f"### {NAMES[dataset]}: 2 x 2 placement of fine categories ({q['n_categories']} categories; median rate {_f(q['median_rate'])}, median E2 hit rate {_f(q['median_e2_hit_rate'])})", ""]
    for key, title in (("explored_hard_to_predict", "explored but hard to predict (rate above median, E2 hit rate below median)"),
                       ("explored_easy_to_predict", "explored and easier to predict (both above median)"),
                       ("rarely_explored_hard_to_predict", "rarely explored and hard to predict (both below median)"),
                       ("rarely_explored_easy_to_predict", "rarely explored but easier to predict (rate below, E2 above)")):
        items = q["quadrants"].get(key, [])
        lines += [f"- {title}: {len(items)} categories: " + ("; ".join(f"{_name(r)} ({_f(r['rate'])}, {_f(r['e2_hit_rate'])}, n {r['n_target_new']:,})" for r in items) or "none"), ""]

    # Write the Spearman correlations between the fit split and the eval split.
    lines += [f"### {NAMES[dataset]}: robustness", ""]
    rows = []
    for level_name in ("coarse", "fine"):
        s = entry["fit"]["levels"][level_name]["spearman_vs_eval"]
        rows.append([level_name, f"{s['n_categories']}", _f(s["volume"]["spearman"]), _f(s["propensity"]["spearman"])])
    lines += ["Spearman rank correlation of the fit-split and eval-split category values (history + target events; eligible eval categories present in both). "
              "The fit split shares most of its events with the eval split (Instacart: every basket but the last; Complete Journey: an earlier cutoff), "
              "so these correlations check the stability of the category values rather than provide independent evidence:", ""]
    lines += _table(["level", "categories", "volume (new-item events)", "propensity (rate)"], rows)
    # Write the coarse new-item rate per history-length bin, categories sorted by volume.
    bins = ev["coarse_propensity_by_history_bin"]
    categories = sorted({c["category"] for b in bins for c in b["categories"]}, key=lambda name: -next((c["n_new"] for b in bins for c in b["categories"] if c["category"] == name), 0))
    lines += ["Coarse new-item rate by T5 history-length bin (eval, history + target events; categories sorted by volume; '-' = no events in the bin):", ""]
    header = ["category"] + [f"{b['bin']} ({b['n_users']:,} users)" for b in bins]
    rows = [["overall"] + [_f(b["overall_rate"]) for b in bins]]
    for name in categories:
        cells = [_label(name, is_missing(name))]
        for b in bins:
            item = next((c for c in b["categories"] if c["category"] == name), None)
            cells.append(_f(item["rate"]) if item else "-")
        rows.append(cells)
    lines += _table(header, rows)
    # Return the lines of the section.
    return lines


def render_markdown(payload: dict[str, Any]) -> str:
    '''Return the markdown report text built from the JSON ``payload``.'''
    # Write the title, the definitions paragraph and the run-information bullets.
    meta, entries, profile = payload["meta"], payload["datasets"], payload["profile"]
    lines = [f"# Where new-item exploration happens, by department and product group ({profile})", "",
             f"**{LABEL}.** For every user and every basket from the 2nd basket on (eval split: history baskets, then the target basket), an item is a "
             "new-item event when it is not in any earlier basket of the user, otherwise a repeat event (distinct items per basket). Per category: "
             "volume = new-item events and their share of all new-item events; propensity = new-item rate = new-item events / all events in the category; "
             "lift = share of new-item events / share of all events; breadth = share of users with >= 1 new-item event in the category (over all users and "
             "over users who bought anything in the category in any basket of the split); known-category share = share of the new-item events whose category "
             "was already in the user's earlier baskets. Predictability (target basket only): hit rate of the E2 user-CF top 10 and of flat popularity "
             "excluding history on the target new items of the category, the share of E2 slots in the category next to the share of target new items. "
             "CIs are user-level bootstrap percentiles. Rankings and figures describe the development data only; no causal reading is implied.", "",
             f"- Code commit `{meta['git_commit']}`, {meta['utc_time']}; profile `{profile}`; bootstrap n = {payload['bootstrap']['n']}, seed {payload['bootstrap']['seed']}.",
             "- Category levels are the pipeline's L1 / L2 columns from items.parquet (configs/common.yaml levels): Instacart department / aisle "
             "(identical to the processed coarse_category / fine_category columns), Complete Journey DEPARTMENT / commodity (the processed Complete "
             "Journey frame columns hold commodity / sub-commodity, which are not used here). Names are reported as stored; 'missing' labels are flagged.",
             f"- Support filter for fine-level rankings: >= {SUPPORT_MIN_NEW['instacart']} new-item events (Instacart), >= {SUPPORT_MIN_NEW['complete_journey']} "
             "(Complete Journey) in the event set ranked; coarse level unfiltered.",
             f"- Inputs (sha256 in the JSON): {', '.join(f'{d}: {len(payload['inputs_sha256'][d])} files' for d in entries)}.",
             f"- Runtime {payload['runtime_seconds']:.1f} s; peak working set {payload['peak_memory_mib']:.0f} MiB (in-process).", ""]
    # Write the table of the user-CF selections per dataset.
    lines += ["## User-CF selections", ""]
    rows = []
    for d, e in entries.items():
        u = e["usercf"]
        rows.append([NAMES[d], str(u["k"]), u["weighting"], u["source"] if u["source"] == "override" else f"`{u['source']}` (commit `{u.get('source_commit')}`)",
                     "yes" if u["verified"] else ("not checked" if u["verified"] is None else "NO")])
    lines += _table(["dataset", "k", "weighting", "source", "equals the expected medium selection"], rows)
    # Write the section of every dataset.
    for dataset, entry in entries.items():
        lines += render_dataset(entry, dataset)
    # Write the list of figures.
    figures = payload["figures"]
    lines += ["## Figures", "",
              f"- F-1 `{figures['rate_coarse']['png']}` (`{figures['rate_coarse']['csv']}`): coarse categories, new-item rate with 95 % CI, labelled with the share of all new-item events, one panel per dataset.",
              f"- F-2 `{figures['lift_fine']['png']}` (`{figures['lift_fine']['csv']}`): fine categories above the support filter, lift on a log scale, top 15 and bottom 10, one panel per dataset.",
              f"- F-3 `{figures['rate_vs_e2_fine']['png']}` (`{figures['rate_vs_e2_fine']['csv']}`): fine categories of the 2 x 2 placement, new-item rate vs E2 hit rate, point size = target new items, median lines, top 8 by volume labelled.", ""]
    # Write the table of timings and peak memory per dataset.
    lines += ["## Timing and memory", ""]
    rows = []
    for d, e in entries.items():
        t = e["timing_seconds"]
        rows.append([NAMES[d]] + [f"{t[k]:.1f} s" for k in ("eval_load", "eval_events", "eval_category_tables", "e2_list", "popularity_list", "predictability", "robustness")]
                    + [f"{e['peak_memory_mib']:.0f}" if e["peak_memory_mib"] else "-"])
    lines += _table(["dataset", "eval load", "events", "category tables + CIs", "E2 list", "popularity list", "predictability", "robustness", "peak working set (MiB)"], rows)
    # Join the lines into one text.
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def input_digests(config: dict[str, Any], dataset: str) -> dict[str, str]:
    '''Return {file name: SHA-256} of the INPUT_FILES parquet files of ``dataset``.

    ``config`` is the profile configuration that gives the data folder.
    '''
    # Refuse a data folder with confirmatory inputs.
    folder = guard_path(dataset_dir(config, dataset))
    # Hash every input file.
    return {name: sha256_of(folder / f"{name}.parquet") for name in INPUT_FILES}


def main(argv: list[str] | None = None) -> Path:
    '''Run the case study for the options in ``argv`` and return the path of the JSON file.'''
    # Parse the options and reject a profile outside the development profiles.
    args = parse_args(argv)
    if args.profile not in ALLOWED_PROFILES:
        raise ValueError(f"profile must be one of {ALLOWED_PROFILES}")
    # Start the run timer and load the configuration of the profile.
    t_start = time.perf_counter()
    config = load_config(args.profile)
    # Refuse a configuration whose role is "final".
    if config.get("role") == "final":
        raise PermissionError("posthoc_explore_category: profiles with role 'final' are never analysed")
    # Create the output folder.
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    # Take the user-CF override when both override options are given.
    override = (args.usercf_k, args.usercf_weighting) if args.usercf_k is not None else None
    entries: dict[str, dict[str, Any]] = {}
    digests = {}
    for dataset in args.datasets:
        # Select and print the user-CF (k, weighting) of the dataset.
        usercf = usercf_selection(args.profile, dataset, Path(args.reference_tuning_dir), override)
        print(f"[{dataset}] user-CF k = {usercf['k']}, weighting {usercf['weighting']} ({usercf['source']}; verified {usercf['verified']})")
        # Hash the input files and run the analysis of the dataset.
        digests[dataset] = input_digests(config, dataset)
        entries[dataset] = analyse_dataset(config, dataset, usercf, args.bootstrap_n, args.seed)
        # Print a one-line summary of the dataset.
        ev = entries[dataset]["eval"]
        top = ev["levels"]["coarse"]["sets"]["all"]["rankings"]["volume"][:3]
        q = ev["quadrants_fine"]
        print(f"[{dataset}] events {ev['n_events']:,}, new {ev['n_new_events']:,} (rate {ev['overall_rate']['all']:.3f}); top coarse by volume: "
              + ", ".join(f"{r['category']} ({r['share_new_events']:.1%})" for r in top)
              + f"; fine categories passing support (all / target): {ev['levels']['fine']['sets']['all']['n_pass_support']} / "
              f"{ev['predictability']['levels']['fine']['n_pass_support_target']}; 2x2 explored-hard: {len(q['quadrants'].get('explored_hard_to_predict', []))}; "
              f"E2 overall hit rate {ev['predictability']['overall_e2_hit_rate']:.3f}; timings "
              + ", ".join(f"{k} {v:.1f} s" for k, v in entries[dataset]["timing_seconds"].items()))
    # Draw the three figures.
    figures = {"rate_coarse": figure_rate_coarse(entries, out_dir), "lift_fine": figure_lift_fine(entries, out_dir),
               "rate_vs_e2_fine": figure_rate_vs_e2(entries, out_dir)}
    # Assemble the JSON payload with definitions, inputs, results and environment.
    payload = {
        "label": LABEL, "task": TASK, "profile": args.profile, "k": K,
        "definitions": {
            "event": "item of a basket from the user's 2nd basket on (distinct items per basket); new-item event when the item is in no earlier basket of the user",
            "levels": {"coarse": "L1 of configs/common.yaml levels (Instacart coarse_category = department, Complete Journey DEPARTMENT)",
                       "fine": "L2 (Instacart fine_category = aisle, Complete Journey commodity)"},
            "volume": "new-item events in c and their share of all new-item events", "propensity": "new-item events in c / all events in c",
            "lift": "share of new-item events in c / share of all events in c",
            "breadth": "share of users with >= 1 new-item event in c over all users and over users with >= 1 row in c in any basket of the split",
            "known_category_share": "among new-item events in c, share whose category c was already in the user's earlier baskets",
            "predictability": "target new items of c in the user's top-10 E2 (user-CF) or flat popularity list / target new items of c; E2 slot share = "
                              "share of all top-10 E2 rows in c; target new share = share of all target new items in c",
            "bootstrap": "users resampled with replacement; ratio-of-sums statistics; 2.5 / 97.5 percentiles",
            "support_filter": SUPPORT_MIN_NEW, "event_sets": list(EVENT_SETS), "top_n": TOP_N,
        },
        "bootstrap": {"n": args.bootstrap_n, "seed": args.seed, "level": 0.95},
        "inputs_sha256": digests, "data_dir": str(config["data_dir"]),
        "datasets": entries, "figures": figures,
        "environment": {"python": f"Python {platform.python_version()} ({platform.system()})", "pandas": pd.__version__, "numpy": np.__version__,
                        "matplotlib": matplotlib.__version__},
        "runtime_seconds": time.perf_counter() - t_start, "peak_memory_mib": peak_memory_mib(),
    }
    # Write the JSON file and read it back.
    json_path = write_json(out_dir / f"explore_category_{args.profile}.json", payload, config)
    document = json.loads(json_path.read_text(encoding="utf-8"))
    # Render the markdown report from the stored JSON document and write it.
    md_path = out_dir / f"explore_category_{args.profile}.md"
    md_path.write_text(render_markdown(document), encoding="utf-8")
    # Print the output paths, the runtime and the peak memory.
    print(f"wrote {json_path}, {md_path} and {', '.join(f['png'] for f in figures.values())}; runtime {payload['runtime_seconds']:.1f} s; "
          f"peak working set {payload['peak_memory_mib'] or 0:.0f} MiB")
    # Return the path of the JSON file.
    return json_path


# ---------------------------------------------------------------------------
# command-line entry
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    main()
