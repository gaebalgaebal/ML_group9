# *------------------- 2026 Fall Semester Machine Learning Term-project: Team 9 --------------------*
# Code First written at 2026-11-07
# Code Final Edited at 2026-11-13
# Editor: Yejun Kim

'''
Purpose : Post-hoc case study: new-item purchase rates and the value of an exploration slot (W15;
          post-hoc, development data, not confirmatory). Only the development profiles ``sample``
          and ``medium`` are accepted; no file under reports/final* or data/processed/holdout* is
          read.
          A. New-item purchase rates (descriptive): for every user u and every basket b from the
             user's 2nd basket on (history baskets, then the target basket of the split), prior(b) =
             items of u's earlier baskets of the same split (history + target, smaller basket_seq);
             new_share(b) = |b minus prior(b)| / |b| over distinct items, any_new(b) = 1 if |b minus
             prior(b)| >= 1, n_new(b) = |b minus prior(b)|. Basket-weighted means (over baskets) and
             user-weighted means (mean of the per-user means) overall, by basket position (=
             basket_seq; bins 2-4, 5-10, 11-20, 21-50, > 50) and by the T5 history-length bins
             (nbr.pipeline.HISTORY_BINS on the number of distinct history baskets of the split), for
             the history baskets, the history baskets with >= 3 distinct items (the Complete Journey
             target is the first basket with >= 3 items after the cutoff while its history baskets
             carry no size filter) and the target basket. Eval split main, fit split robustness
             check.
          B. Hit probability by rank (eval split): P(the item at rank j of a list is in the
             user's target basket) over all users, for the repeat list of the logistic-regression
             repeat ranker (R2; LR ranker, ranks 1-12) and the explore list of the user-based CF
             explore module (E2; user-CF, ranks 1-3); share of users with an item at the rank and
             the hit rate given an item is present. Paired bootstrap over users of explore rank 1
             minus repeat rank 10 (the slot that changes under r = 9) and of explore ranks 1 + 2
             minus repeat ranks 9 + 10 (r = 8), overall and by history-length bin.
          C. Forced exploration slot (eval split): final lists nbr.combine.combine(repeat,
             explore, r) with r in {10, 9, 8} (explore items fill the last 10 - r slots), repeat in
             {R2, the decayed purchase-frequency repeat module (R1) with gamma 0.9, 0.97, 1.0, the
             dataset's gamma*}, explore = E2 with the selected (k, weighting). Per list: NDCG@10,
             Recall@10, new-item Recall@10, share of users with >= 1 new-item hit in the top 10,
             repeat Recall@10. Paired bootstrap (95 % CI) of r = 9 vs r = 10 and r = 8 vs r = 10 per
             repeat module, overall and by history-length bin. Fit-split check: NDCG@10 by r in
             0..10 per repeat module (R2 scored with the C* model fitted on the LR training users
             and evaluated on the validation users, as in pipeline.tune_integration; R1 on all fit
             users) and the r selected by pipeline.select_r. Reproduction checks: R2 + E2 at r = 10
             against the W8b medium final_lr_ucf eval NDCG@10; R1 gamma 0.97 vs 1.0 repeat Recall@10
             on the Complete Journey eval split; fit curves against the tuning document.
Inputs  : the fit and eval splits of the profile, the tuning document
          reports/metrics/<profile>_<dataset>_tuning.json and the LR artifact
          artifacts/<profile>/<dataset>_lr_repeat.joblib written by ``python -m nbr tune`` (under
          --model-root, default: this repository).
          Setup check (--reference-tuning-dir): the tuning document of this run is compared with the
          reference document of the same name in PATH (the W8b medium run). The selections (gamma,
          gamma_c, alpha, r, LR C, user-CF k and weighting, the re-selected slots) must be equal;
          the user-CF grid rows other than k = 10 (added to the grid after W8b), the stage grids and
          the LR validation rows are compared as well and reported. Any differing selection stops
          the script before the analysis. The LR artifact of this run is checked against the sha256
          recorded in the tuning document.
Outputs : D. Outputs in --out-dir: explore_slot_<profile>.json (all numbers; meta with the code
          commit; inputs_sha256 with the sha256 of every input file),
          explore_slot_<profile>.md (tables),
          fig_new_rate_by_position.{png,csv}, fig_hit_probability_by_rank.{png,csv},
          fig_slot_effects.{png,csv} (each figure drawn from its own CSV).
Usage   :
    python scripts/posthoc_explore_slot.py [--profile medium] [--datasets instacart,complete_journey]
                                           [--out-dir reports/posthoc] [--reference-tuning-dir PATH]
                                           [--model-root PATH] [--bootstrap-n 1000] [--seed 42]
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

from nbr.combine import combine  # noqa: E402
from nbr.config import load_config  # noqa: E402
from nbr.data import dataset_dir, split_target  # noqa: E402
from nbr.io import write_json  # noqa: E402
from nbr.lr_repeat import SOURCE as LR_SOURCE  # noqa: E402
from nbr.lr_repeat import lr_repeat_list, lr_repeat_scores, train_only_bundle  # noqa: E402
from nbr.metrics import paired_bootstrap, per_user_ndcg, per_user_recall  # noqa: E402
from nbr.pipeline import (  # noqa: E402
    HISTORY_BINS,
    K,
    history_lengths,
    load_integration_selections,
    load_lr_repeat,
    load_selections,
    prepare,
    select_r,
    tune_lr_repeat,
    tuning_path,
)
from nbr.repeat import rank_scores, repeat_list  # noqa: E402
from nbr.usercf import ucf_list  # noqa: E402
from nbr.usercf_eval import peak_memory_mib  # noqa: E402

# ---------------------------------------------------------------------------
# constants
# ---------------------------------------------------------------------------
LABEL = "post-hoc, development data, not confirmatory"
TASK = "W15 post-hoc case study: new-item purchase rates and the value of an exploration slot"
ALLOWED_PROFILES = ("sample", "medium")
DATASETS = ("instacart", "complete_journey")
NAMES = {"instacart": "Instacart", "complete_journey": "Complete Journey"}
SPLITS = ("eval", "fit")
INPUT_FILES = ("fit_history", "fit_target", "eval_history", "eval_target", "items")
MIN_ITEMS = 3                                   # Complete Journey target rule: first basket with >= 3 items after the cutoff
POSITION_BINS = [("2-4", 2, 4), ("5-10", 5, 10), ("11-20", 11, 20), ("21-50", 21, 50), (">50", 51, math.inf)]
METRICS = {"mean_new_share": "new_share", "p_any_new": "any_new", "mean_n_new": "n_new", "mean_size": "size"}
BASKET_SETS = {
    "history_all": "history baskets (2nd basket on)",
    "history_min3": f"history baskets with >= {MIN_ITEMS} distinct items (2nd basket on)",
    "target": "target basket",
}
COLUMNS = ["user_id", "basket_id", "basket_seq", "item_id"]
# Part C: slot grid, R1 decay values and the list metrics (name -> description).
R_VALUES = (10, 9, 8)
R_GRID = list(range(0, 11))
R1_GAMMAS = (0.9, 0.97, 1.0)
LIST_METRICS = {
    "ndcg": "NDCG@10 (all users)", "recall": "Recall@10 (all users)",
    "new_recall": "new-item Recall@10 (users with >= 1 new target item)",
    "new_hit_share": "share of users with >= 1 new-item hit in the top 10 (all users)",
    "repeat_recall": "repeat Recall@10 (users with >= 1 repeat target item)",
}
COMPARISONS = {9: "r = 9 vs r = 10", 8: "r = 8 vs r = 10"}
# Part B: ranks examined.
REPEAT_RANKS = 12
EXPLORE_RANKS = 3
# Selections compared with the reference tuning document: name -> key path in the document.
SELECTION_PATHS = {
    "gamma": ("selections", "gamma"),
    "gamma_c": ("selections", "gamma_c"),
    "alpha": ("selections", "alpha"),
    "r": ("selections", "r"),
    "lr_C": ("lr_repeat", "C"),
    "usercf_k": ("integration", "selections", "k"),
    "usercf_weighting": ("integration", "selections", "weighting"),
    "r_final_lr_r": ("integration", "selections", "r", "final_lr_r"),
    "r_final_lr_ucf": ("integration", "selections", "r", "final_lr_ucf"),
    "r_final_star_ucf": ("integration", "selections", "r", "final_star_ucf"),
}
NEW_K = 10                                      # user-CF k added to the grid after the reference run (PREREG v1.3 item 7)
TOLERANCE = 1e-9
# Numbers to reproduce (computed earlier on the same development data; four decimals).
SANITY = {
    ("medium", "instacart"): {"R2_E2_r10_ndcg": 0.4274},
    ("medium", "complete_journey"): {"R2_E2_r10_ndcg": 0.2046, "R1_g0.97_repeat_list_recall": 0.2717, "R1_g1_repeat_list_recall": 0.2497},
}
# Figure colours: categorical slots 1 and 2 of the validated default palette (light surface) and its chrome tokens.
COLORS = {"slot1": "#2a78d6", "slot2": "#eb6834"}
INK = {"primary": "#0b0b0b", "secondary": "#52514e", "muted": "#898781", "grid": "#e1e0d9", "axis": "#c3c2b7", "surface": "#fcfcfb"}
FIGURES = {"new_rate": "fig_new_rate_by_position", "rank_hits": "fig_hit_probability_by_rank", "slots": "fig_slot_effects"}


# ---------------------------------------------------------------------------
# arguments and small helpers
# ---------------------------------------------------------------------------
def parse_datasets(value: str) -> list[str]:
    """Comma-separated dataset names -> list in the given order (blanks and repeats dropped); unknown names raise."""
    # Collect the stripped names in order, skipping blanks and repeats.
    names: list[str] = []
    for item in value.split(","):
        item = item.strip()
        if item and item not in names:
            names.append(item)
    # Reject an empty list or any name outside DATASETS.
    unknown = [name for name in names if name not in DATASETS]
    if unknown or not names:
        raise ValueError(f"datasets must be a non-empty subset of {list(DATASETS)}, got {value!r}")
    return names


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Command line of the script (``argv`` defaults to sys.argv); ``datasets`` is returned as a list."""
    # Build the parser with the profile, dataset, folder, bootstrap and seed options.
    parser = argparse.ArgumentParser(description="Post-hoc new-item purchase rates and exploration-slot case study (development profiles only).")
    parser.add_argument("--profile", default="medium", choices=ALLOWED_PROFILES)
    parser.add_argument("--datasets", default=",".join(DATASETS), type=parse_datasets,
                        help="comma-separated subset of instacart,complete_journey (default: both)")
    parser.add_argument("--out-dir", default=str(ROOT / "reports" / "posthoc"))
    parser.add_argument("--reference-tuning-dir", default=None,
                        help="folder with the reference <profile>_<dataset>_tuning.json files (read-only); "
                             "selections of this run's tuning documents must equal them")
    parser.add_argument("--model-root", default=None,
                        help="folder whose reports/metrics/<profile>_<dataset>_tuning.json and artifacts/<profile>/"
                             "<dataset>_lr_repeat.joblib are read (default: this repository)")
    parser.add_argument("--bootstrap-n", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    # Parse ``argv`` and return the namespace.
    return parser.parse_args(argv)


def guard_path(path: Path) -> Path:
    """Refuse any path with a component starting with 'final' or 'holdout' (confirmatory inputs are never read)."""
    # Raise when any component of the resolved path starts with 'final' or 'holdout'.
    for part in path.resolve().parts:
        if part.lower().startswith(("final", "holdout")):
            raise PermissionError(f"refusing to read {path}: confirmatory inputs are never read by this script")
    # Return the path unchanged.
    return path


def sha256_of(path: Path) -> str:
    '''Return the sha256 hex digest of the file ``path``.'''
    # Start an empty sha256 hash.
    digest = hashlib.sha256()
    # Feed the file to the hash in chunks of 1 MiB.
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    # Return the hex digest.
    return digest.hexdigest()


def bin_labels(edges: list[int]) -> list[tuple[str, float, float]]:
    """(label, lo, hi] bins of nbr.pipeline (upper edges, last bin open): (0,4], (4,6], ..., (20,inf)."""
    # Put 0 before the upper edges and infinity after them.
    bounds = [0, *edges, math.inf]
    # Pair consecutive bounds into labelled (lo, hi] bins, the last one open.
    return [(f"({lo},inf)" if math.isinf(hi) else f"({lo},{hi}]", lo, hi) for lo, hi in zip(bounds[:-1], bounds[1:])]


def compare(a: pd.Series, b: pd.Series, n: int, seed: int) -> dict[str, Any]:
    """Paired bootstrap of mean(b) - mean(a) over users (95 % CI)."""
    # Run the paired bootstrap at the 95 % level.
    result = paired_bootstrap(a, b, n, seed, [0.95])
    # Keep the user count, both means, the difference and the 95 % interval.
    return {"n_users": result["n_users"], "mean_a": result["mean_a"], "mean_b": result["mean_b"],
            "diff": result["diff"], "ci_0.95": result["ci"][0.95]}


def compare_by_bin(a: pd.Series, b: pd.Series, lengths: pd.Series, dataset: str, n: int, seed: int) -> list[dict[str, Any]]:
    """compare() restricted to the users of every T5 history-length bin.

    A bin without a scored user gives a row with only the bin name and n_users = 0.
    """
    # Compare the two series on the users of each history-length bin.
    rows = []
    for name, lo, hi in bin_labels(HISTORY_BINS[dataset]):
        # Take the scored users whose history length lies in (lo, hi].
        users = a.index.intersection(lengths[(lengths > lo) & (lengths <= hi)].index)
        # Store the comparison of the bin, or a zero user count when the bin is empty.
        rows.append({"bin": name, **(compare(a.loc[users], b.loc[users], n, seed) if len(users) else {"n_users": 0})})
    return rows


# ---------------------------------------------------------------------------
# A. per-basket novelty
# ---------------------------------------------------------------------------
def basket_novelty(history: pd.DataFrame, target: pd.DataFrame) -> pd.DataFrame:
    """One row per basket from each user's 2nd basket on: user_id, basket_id, position, is_target, size, n_new, new_share, any_new.

    ``history`` and ``target`` are the frames of one split (columns user_id, basket_id, basket_seq, item_id, ...).
    Items are counted once per basket; an item of basket b is new when no basket of the same user with a smaller
    basket_seq (history and target together) contains it; position = basket_seq. The user's first basket (smallest
    basket_seq) is dropped. Raises when a basket_id carries several basket_seq values or a target basket does not
    come after every history basket of its user.
    """
    # Stack the history and target rows with a flag that marks the target rows.
    frame = pd.concat([history[COLUMNS].assign(is_target=False), target[COLUMNS].assign(is_target=True)], ignore_index=True)
    # Count every item once per basket.
    frame = frame.drop_duplicates(["user_id", "basket_id", "item_id"])
    # List the distinct baskets with their sequence number and target flag.
    baskets = frame[["user_id", "basket_id", "basket_seq", "is_target"]].drop_duplicates()
    # Reject a basket_id with several basket_seq or is_target values within one user.
    if baskets.duplicated(["user_id", "basket_id"]).any():
        raise ValueError("a basket_id carries several basket_seq / is_target values within one user")
    # Reject several basket_id values on one basket_seq within one user.
    if baskets.duplicated(["user_id", "basket_seq"]).any():
        raise ValueError("several basket_id values share one basket_seq within one user")
    # Find the largest history basket_seq of every user.
    max_history = baskets[~baskets["is_target"]].groupby("user_id")["basket_seq"].max()
    # Flag the target baskets that do not come after every history basket of their user.
    targets = baskets[baskets["is_target"]]
    behind = targets["basket_seq"].to_numpy() <= max_history.reindex(targets["user_id"]).fillna(-np.inf).to_numpy()
    # Raise when any target basket is flagged.
    if behind.any():
        raise ValueError(f"{int(behind.sum())} target baskets do not come after every history basket of their user")

    # Encode every (user, item) pair as one integer.
    user_code, _ = pd.factorize(frame["user_id"])
    item_code, items = pd.factorize(frame["item_id"])
    pair = user_code.astype(np.int64) * len(items) + item_code.astype(np.int64)
    # Find the smallest basket_seq at which each pair occurs.
    first_seq = frame["basket_seq"].groupby(pair).transform("min")
    # Mark a row as new when its basket is the first one of the user with that item.
    frame = frame.assign(new=(frame["basket_seq"].to_numpy() == first_seq.to_numpy()))
    # Count the items and the new items of every basket.
    out = (frame.groupby(["user_id", "basket_id", "basket_seq", "is_target"], sort=False)
           .agg(size=("item_id", "size"), n_new=("new", "sum")).reset_index())
    # Drop the first basket of every user.
    first_basket = out.groupby("user_id")["basket_seq"].transform("min")
    out = out[out["basket_seq"] > first_basket].copy()
    # Rename basket_seq to position and cast the count columns to int64.
    out = out.rename(columns={"basket_seq": "position"})
    out["position"] = out["position"].astype("int64")
    out["size"] = out["size"].astype("int64")
    out["n_new"] = out["n_new"].astype("int64")
    # Add the share of new items and the flag for at least one new item.
    out["new_share"] = out["n_new"] / out["size"]
    out["any_new"] = (out["n_new"] >= 1).astype("float64")
    # Return the rows sorted by user and position.
    return out.sort_values(["user_id", "position"]).reset_index(drop=True)


def summarise(baskets: pd.DataFrame) -> dict[str, Any]:
    """Counts plus basket-weighted and user-weighted (mean of per-user means) means of new_share, any_new, n_new, size."""
    # Return empty counts when there is no basket.
    if baskets.empty:
        return {"n_baskets": 0, "n_users": 0, "basket_weighted": None, "user_weighted": None}
    # Average the metric columns within each user.
    columns = list(METRICS.values())
    user_means = baskets.groupby("user_id")[columns].mean()
    # Return the counts with the means over baskets and over the per-user means.
    return {
        "n_baskets": int(len(baskets)),
        "n_users": int(len(user_means)),
        "basket_weighted": {key: float(baskets[column].mean()) for key, column in METRICS.items()},
        "user_weighted": {key: float(user_means[column].mean()) for key, column in METRICS.items()},
    }


def n_new_distribution(baskets: pd.DataFrame) -> dict[str, float] | None:
    """Basket-weighted shares of baskets with n_new = 0, 1, 2 and >= 3."""
    # Return None when there is no basket.
    if baskets.empty:
        return None
    # Compute the shares of baskets with 0, 1, 2 and at least 3 new items.
    n_new = baskets["n_new"]
    return {"0": float((n_new == 0).mean()), "1": float((n_new == 1).mean()), "2": float((n_new == 2).mean()),
            "3+": float((n_new >= 3).mean())}


def by_position(baskets: pd.DataFrame) -> list[dict[str, Any]]:
    """summarise per POSITION_BINS of the basket position."""
    # Summarise the baskets whose position lies in each position bin.
    rows = []
    for name, lo, hi in POSITION_BINS:
        subset = baskets[(baskets["position"] >= lo) & (baskets["position"] <= hi)]
        rows.append({"bin": name, **summarise(subset)})
    return rows


def by_history_length(baskets: pd.DataFrame, lengths: pd.Series, dataset: str) -> list[dict[str, Any]]:
    """summarise per T5 history-length bin (HISTORY_BINS[dataset]) of the basket's user; ``lengths`` maps user_id -> history baskets."""
    # Look up the history length of each basket's user.
    user_length = baskets["user_id"].map(lengths)
    # Raise when a user has no history length.
    if user_length.isna().any():
        raise ValueError("a basket belongs to a user without history baskets")
    # Summarise the baskets of the users in each history-length bin.
    rows = []
    for name, lo, hi in bin_labels(HISTORY_BINS[dataset]):
        subset = baskets[(user_length > lo) & (user_length <= hi)]
        rows.append({"bin": name, **summarise(subset)})
    return rows


def basket_set_block(baskets: pd.DataFrame, lengths: pd.Series, dataset: str, description: str) -> dict[str, Any]:
    '''Summary block of one basket set ``baskets`` labelled ``description``.

    Holds the counts and means, the n_new shares and the tables by position and by history length
    (``lengths`` maps user_id -> history baskets; ``dataset`` selects the bins).
    '''
    return {"description": description, **summarise(baskets), "n_new_distribution": n_new_distribution(baskets),
            "by_position": by_position(baskets), "by_history_length": by_history_length(baskets, lengths, dataset)}


def analyse_novelty(history: pd.DataFrame, target: pd.DataFrame, dataset: str) -> dict[str, Any]:
    """Part A for one split: data checks and the three basket-set blocks (history_all, history_min3, target)."""
    # Start the timer of this part.
    t0 = time.perf_counter()
    # Compute the per-basket new-item rows and the history length of every user.
    baskets = basket_novelty(history, target)
    lengths = history_lengths(history)
    # Split the rows into history baskets and target baskets.
    history_baskets = baskets[~baskets["is_target"]]
    target_baskets = baskets[baskets["is_target"]]
    # Form the three basket sets: all history, history with >= MIN_ITEMS items, target.
    sets = {
        "history_all": history_baskets,
        "history_min3": history_baskets[history_baskets["size"] >= MIN_ITEMS],
        "target": target_baskets,
    }
    # Build the summary block of every basket set.
    blocks = {name: basket_set_block(frame, lengths, dataset, BASKET_SETS[name]) for name, frame in sets.items()}

    # data checks: basket_seq structure, the Complete Journey gap before the target, agreement with nbr.data.split_target
    # List the distinct baskets of history and target together.
    all_baskets = pd.concat([history[["user_id", "basket_seq"]], target[["user_id", "basket_seq"]]]).drop_duplicates()
    # Group the distinct history basket_seq values by user.
    seq = history[["user_id", "basket_seq"]].drop_duplicates().groupby("user_id")["basket_seq"]
    # Compute the share of users whose history starts at 1 and the share with consecutive values.
    starts_at_one = float((seq.min() == 1).mean())
    consecutive = float(((seq.max() - seq.min() + 1) == seq.size()).mean())
    # Compute the gap between the target basket_seq and the last history basket_seq.
    target_seq = target[["user_id", "basket_seq"]].drop_duplicates().groupby("user_id")["basket_seq"].min()
    gap = target_seq - seq.max().reindex(target_seq.index)
    # Take the new target items from nbr.data.split_target.
    _, new_target = split_target(history, target)
    # Count the users whose target basket has at least one new item.
    any_new_users = int(target_baskets.loc[target_baskets["any_new"] > 0, "user_id"].nunique())
    # Count the distinct items of every history basket.
    history_sizes = history.drop_duplicates(["user_id", "basket_id", "item_id"]).groupby(["user_id", "basket_id"]).size()
    # Collect the check values.
    checks = {
        "n_users": int(target["user_id"].nunique()),
        "n_history_baskets": int(len(history[["user_id", "basket_id"]].drop_duplicates())),
        "n_target_baskets": int(len(target[["user_id", "basket_id"]].drop_duplicates())),
        "n_baskets_analysed": int(len(baskets)),
        "n_first_baskets_dropped": int(len(all_baskets) - len(baskets)),
        "history_seq_starts_at_one_share": starts_at_one,
        "history_seq_consecutive_share": consecutive,
        "target_seq_minus_last_history_seq": {str(int(k)): int(v) for k, v in gap.value_counts().sort_index().items()},
        "users_with_skipped_baskets_before_target": int((gap > 1).sum()),
        "history_baskets_min_items_share": float((history_sizes >= MIN_ITEMS).mean()),
        "target_size_min": int(target_baskets["size"].min()) if len(target_baskets) else None,
        "split_target_new_users": int(new_target["user_id"].nunique()),
        "target_any_new_users": any_new_users,
        "matches_split_target": int(new_target["user_id"].nunique()) == any_new_users,
    }
    # Return the checks, the history-length summary, the blocks and the elapsed seconds.
    return {"checks": checks, "history_length_summary": {"min": int(lengths.min()), "median": float(lengths.median()),
                                                           "max": int(lengths.max())},
            "basket_sets": blocks, "seconds": time.perf_counter() - t0}


# ---------------------------------------------------------------------------
# selections, lists and repeat modules
# ---------------------------------------------------------------------------
def model_config(config: dict[str, Any], model_root: str | None) -> dict[str, Any]:
    """``config`` with repo_root replaced by ``model_root`` (tuning documents and LR artifacts are read under it)."""
    return {**config, "repo_root": Path(model_root)} if model_root else config


def repeat_modules(gamma_star: float) -> list[dict[str, Any]]:
    """R2 followed by the R1 modules gamma in R1_GAMMAS plus the dataset's gamma* (merged when equal), R1 sorted by gamma."""
    # Start with the logistic-regression repeat ranker (R2).
    modules = [{"name": "R2", "label": "R2 (LR repeat ranker)", "kind": "R2", "gamma": None, "is_gamma_star": False}]
    # Merge gamma* into the fixed gammas of the decayed purchase-frequency repeat module (R1).
    gammas = sorted(set(R1_GAMMAS) | {float(gamma_star)})
    # Add one R1 module per gamma and mark the one equal to gamma*.
    for gamma in gammas:
        star = math.isclose(gamma, gamma_star)
        modules.append({"name": f"R1_g{gamma:g}", "label": f"R1 gamma {gamma:g}" + (" (gamma*)" if star else ""),
                        "kind": "R1", "gamma": gamma, "is_gamma_star": star})
    return modules


def repeat_lists(history: pd.DataFrame, pop: pd.Series, modules: list[dict[str, Any]], bundle: dict[str, Any],
                 ranks: int = K) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    """Top-K repeat list of every module and the top-``ranks`` R2 list (same scores, longer cut) for part B."""
    lists: dict[str, pd.DataFrame] = {}
    # Score the repeat candidates with the R2 model and rank the top ``ranks`` per user.
    scores = lr_repeat_scores(history, bundle)
    r2_long = rank_scores(scores, pop, ranks, LR_SOURCE)
    # Cut the long R2 list at K for R2 and build the R1 list for every gamma.
    for module in modules:
        if module["kind"] == "R2":
            lists[module["name"]] = r2_long[r2_long["rank"] <= K].reset_index(drop=True)
        else:
            lists[module["name"]] = repeat_list(history, float(module["gamma"]), K, pop)
    return lists, r2_long


# ---------------------------------------------------------------------------
# B. hit probability by rank
# ---------------------------------------------------------------------------
def rank_hits(lists: pd.DataFrame, target: pd.DataFrame, max_rank: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(hit, present): users x ranks 1..max_rank frames of 0/1 floats over the users of ``target``.

    hit[u, j] = 1 when the item at rank j of user u's list is in the user's target basket; present[u, j] = 1 when the
    list has an item at rank j. Users without a list have zeros.
    """
    # List the target users in sorted order.
    users = pd.Index(target["user_id"].drop_duplicates().sort_values())
    # Mark every distinct (user, target item) pair as a hit.
    pairs = target[["user_id", "item_id"]].drop_duplicates().assign(hit=1.0)
    # Keep the list rows up to ``max_rank`` and attach the hit flag, 0 when absent.
    top = lists.loc[lists["rank"] <= max_rank, ["user_id", "rank", "item_id"]]
    top = top.merge(pairs, on=["user_id", "item_id"], how="left")
    top["hit"] = top["hit"].fillna(0.0)
    # Pivot to users x ranks frames for the hits and for the presence of an item.
    columns = list(range(1, max_rank + 1))
    hit = top.pivot(index="user_id", columns="rank", values="hit").reindex(index=users, columns=columns).fillna(0.0)
    present = top.pivot(index="user_id", columns="rank", values="item_id").reindex(index=users, columns=columns).notna().astype("float64")
    # Remove the column axis name from both frames.
    hit.columns.name = present.columns.name = None
    return hit, present


def rank_table(hit: pd.DataFrame, present: pd.DataFrame) -> list[dict[str, Any]]:
    '''Per-rank rows from ``hit`` and ``present``: P(hit), share present, P(hit given present).'''
    # Compute the three rates for every rank; the conditional rate is None without items.
    rows = []
    for rank in hit.columns:
        n_present = float(present[rank].sum())
        rows.append({"rank": int(rank), "p_hit": float(hit[rank].mean()), "share_present": float(present[rank].mean()),
                     "p_hit_given_present": float(hit[rank].sum() / n_present) if n_present else None})
    return rows


def analyse_rank_hits(r2_long: pd.DataFrame, explore: pd.DataFrame, target: pd.DataFrame, lengths: pd.Series, dataset: str,
                      n_boot: int, seed: int) -> dict[str, Any]:
    """Part B: per-rank hit probabilities of R2 (ranks 1..REPEAT_RANKS) and E2 (1..EXPLORE_RANKS) and the slot comparisons."""
    # Start the timer of this part.
    t0 = time.perf_counter()
    # Build the hit and presence frames of the R2 list and of the user-based CF list (E2).
    rep_hit, rep_present = rank_hits(r2_long, target, REPEAT_RANKS)
    exp_hit, exp_present = rank_hits(explore, target, EXPLORE_RANKS)
    # Bootstrap explore minus repeat hits in the slots that change under r = 9 and r = 8.
    comparisons = {
        "slot10_explore1_vs_repeat10": {
            "description": "explore rank 1 minus repeat rank 10 (the slot that changes under r = 9); hits per user",
            "repeat_ranks": [10], "explore_ranks": [1],
            **compare(rep_hit[10], exp_hit[1], n_boot, seed),
            "by_history_length": compare_by_bin(rep_hit[10], exp_hit[1], lengths, dataset, n_boot, seed)},
        "slots9_10_explore12_vs_repeat910": {
            "description": "explore ranks 1 + 2 minus repeat ranks 9 + 10 (the slots that change under r = 8); hits per user",
            "repeat_ranks": [9, 10], "explore_ranks": [1, 2],
            **compare(rep_hit[9] + rep_hit[10], exp_hit[1] + exp_hit[2], n_boot, seed),
            "by_history_length": compare_by_bin(rep_hit[9] + rep_hit[10], exp_hit[1] + exp_hit[2], lengths, dataset, n_boot, seed)},
    }
    # Average the hits at repeat ranks 8-10 and explore ranks 1-3 per history-length bin.
    by_bin = []
    for name, lo, hi in bin_labels(HISTORY_BINS[dataset]):
        users = rep_hit.index.intersection(lengths[(lengths > lo) & (lengths <= hi)].index)
        row: dict[str, Any] = {"bin": name, "n_users": int(len(users))}
        if len(users):
            row["repeat"] = {str(j): float(rep_hit.loc[users, j].mean()) for j in (8, 9, 10)}
            row["explore"] = {str(j): float(exp_hit.loc[users, j].mean()) for j in (1, 2, 3)}
        by_bin.append(row)
    # Return the definition, the per-rank tables, the comparisons and the elapsed seconds.
    return {
        "definition": "P(hit at rank j) = share of all target users whose list has an item at rank j that is in their target basket; "
                      "share_present = share of users with an item at rank j; p_hit_given_present = hits / users with an item",
        "n_users": int(len(rep_hit)),
        "repeat_R2": rank_table(rep_hit, rep_present),
        "explore_E2": rank_table(exp_hit, exp_present),
        "expected_hits_top10": {"repeat_R2_ranks_1_10": float(rep_hit[list(range(1, K + 1))].sum(axis=1).mean()),
                                "explore_E2_ranks_1_3": float(exp_hit.sum(axis=1).mean())},
        "comparisons": comparisons,
        "by_history_length": by_bin,
        "seconds": time.perf_counter() - t0,
    }


# ---------------------------------------------------------------------------
# C. forced exploration slot (eval split)
# ---------------------------------------------------------------------------
def list_metrics(final: pd.DataFrame, data: dict[str, Any], all_users: pd.Index) -> dict[str, pd.Series]:
    """Per-user series of LIST_METRICS for the final list ``final`` against the split ``data`` (prepare output)."""
    # Compute the new-item recall once because two metrics use it.
    new_recall = per_user_recall(final, data["new_target"], K)
    # Return the five per-user metric series.
    return {
        "ndcg": per_user_ndcg(final, data["target"], K),
        "recall": per_user_recall(final, data["target"], K),
        "new_recall": new_recall,
        "new_hit_share": (new_recall > 0).astype("float64").reindex(all_users, fill_value=0.0),
        "repeat_recall": per_user_recall(final, data["repeat_target"], K),
    }


def analyse_slots(data: dict[str, Any], lists: dict[str, pd.DataFrame], explore: pd.DataFrame, modules: list[dict[str, Any]],
                  lengths: pd.Series, dataset: str, n_boot: int, seed: int) -> dict[str, Any]:
    """Part C on the eval split: metrics of every (module, r) list and paired bootstraps of r = 9 / 8 against r = 10."""
    # Prepare the timing record and the sorted target users.
    timings: dict[str, float] = {}
    all_users = pd.Index(data["target"]["user_id"].drop_duplicates().sort_values())
    # Compute the per-user metrics of combine(repeat, explore, r) for every module and r.
    series: dict[str, dict[int, dict[str, pd.Series]]] = {}
    t0 = time.perf_counter()
    for module in modules:
        name = module["name"]
        series[name] = {r: list_metrics(combine(lists[name], explore, r, K), data, all_users) for r in R_VALUES}
    # Record the elapsed seconds of the lists and metrics.
    timings["lists_and_metrics"] = time.perf_counter() - t0

    # Build the table of metric means per module and r, with the recall of the repeat list alone.
    t0 = time.perf_counter()
    table = []
    for module in modules:
        name = module["name"]
        repeat_only = per_user_recall(lists[name], data["repeat_target"], K)
        for r in R_VALUES:
            row: dict[str, Any] = {"module": name, "r": r}
            for metric, values in series[name][r].items():
                row[metric] = float(values.mean())
                row[f"{metric}_n_users"] = int(len(values))
            row["repeat_list_recall"] = float(repeat_only.mean())
            table.append(row)
    # Bootstrap every metric of r = 9 and r = 8 against r = 10, overall and per bin.
    diffs = []
    for module in modules:
        name = module["name"]
        for r in (9, 8):
            row = {"module": name, "comparison": COMPARISONS[r], "r": r, "metrics": {}}
            for metric in LIST_METRICS:
                a, b = series[name][10][metric], series[name][r][metric]
                row["metrics"][metric] = {**compare(a, b, n_boot, seed),
                                          "by_history_length": compare_by_bin(a, b, lengths, dataset, n_boot, seed)}
            diffs.append(row)
    # Record the elapsed seconds of the table and the bootstraps.
    timings["bootstrap"] = time.perf_counter() - t0
    # Return the definitions, the table, the differences and the timings.
    return {"definition": {"lists": "nbr.combine.combine(repeat, explore, r, 10): the first r slots from the repeat list (fewer when the user "
                                    "has fewer repeat candidates), then explore items, then leftover repeat items",
                           "metrics": LIST_METRICS, "diff": "mean(r = 9 or 8) - mean(r = 10) per user, paired bootstrap 95 % CI"},
            "r_values": list(R_VALUES), "n_users": int(len(all_users)), "table": table, "differences": diffs, "timing_seconds": timings}


# ---------------------------------------------------------------------------
# C. fit-split check: NDCG@10 by r and the selected r per module
# ---------------------------------------------------------------------------
def slot_curves(fit: dict[str, Any], config: dict[str, Any], modules: list[dict[str, Any]], k_cf: int, weighting: str,
                saved_bundle: dict[str, Any], tuning_document: dict[str, Any]) -> dict[str, Any]:
    """NDCG@10 of combine(repeat, E2, r) for r in R_GRID on the fit split per module and select_r; R2 as in tune_integration."""
    # Prepare the timing record and unpack the fit split.
    timings: dict[str, float] = {}
    history, target, pop = fit["history"], fit["target"], fit["pop"]
    # Build the E2 explore list on the fit split and time it.
    t0 = time.perf_counter()
    explore = ucf_list(history, k_cf, weighting, K, pop, block_size=int(config["usercf"]["block_size"]))
    timings["explore_list"] = time.perf_counter() - t0

    # Refit the LR ranker and time it.
    t0 = time.perf_counter()
    bundle = tune_lr_repeat(history, target, config)        # deterministic refit: gives the train-only model and the validation users
    timings["lr_refit"] = time.perf_counter() - t0
    # Compare C and the coefficients of the refit with the saved artifact.
    saved_coefficients = saved_bundle["coefficients"]
    coefficient_gap = max(abs(float(bundle["coefficients"][name]) - float(saved_coefficients[name])) for name in bundle["coefficients"])
    lr_consistency = {"C_refit": bundle["C"], "C_saved": saved_bundle["C"], "C_equal": math.isclose(bundle["C"], saved_bundle["C"]),
                      "max_abs_coefficient_difference": coefficient_gap, "n_val_users": int(bundle["n_val_users"])}

    # Read the reference curves of the tuning document.
    t0 = time.perf_counter()
    reference = tuning_document.get("integration", {}).get("r_scores", {})
    # Compute the NDCG@10 curve over r and the selected r for every module.
    curves = []
    for module in modules:
        if module["kind"] == "R2":
            # Score R2 with the train-only model on the LR validation users.
            val_users = pd.Index(bundle["val_users"])
            repeat = lr_repeat_list(history, train_only_bundle(bundle), K, pop, users=val_users)
            explore_m = explore[explore["user_id"].isin(val_users)]
            scored_target, note = target[target["user_id"].isin(val_users)], "LR validation users (train-only model)"
            reference_name = "final_lr_ucf"
        else:
            # Score R1 on all fit users.
            repeat = repeat_list(history, float(module["gamma"]), K, pop)
            explore_m, scored_target, note = explore, target, "all fit users"
            reference_name = "final_star_ucf" if module["is_gamma_star"] else None
        # Compute the mean NDCG@10 of the combined list for every r in R_GRID.
        scores = {r: float(per_user_ndcg(combine(repeat, explore_m, r, K), scored_target, K).mean()) for r in R_GRID}
        # Select r with pipeline.select_r.
        r_star = select_r(scores)
        # Store the curve, the selected r and whether it is below K.
        row = {"module": module["name"], "users": note, "n_users": int(scored_target["user_id"].nunique()),
               "ndcg_by_r": [{"r": r, "ndcg": value} for r, value in scores.items()], "r_star": r_star,
               "r_star_below_10": r_star < K}
        # Compare the curve with the one in the tuning document when it has one.
        if reference_name and reference_name in reference:
            ref = {int(item["r"]): float(item["ndcg"]) for item in reference[reference_name]}
            common = sorted(set(ref) & set(scores))
            gap = max(abs(scores[r] - ref[r]) for r in common) if common else None
            row["tuning_document_curve"] = {"name": reference_name, "max_abs_difference": gap,
                                            "equal": gap is not None and gap < TOLERANCE}
        curves.append(row)
    # Record the elapsed seconds of the curves.
    timings["curves"] = time.perf_counter() - t0
    # Return the definition, the LR consistency check, the curves and the timings.
    return {"definition": "fit-split NDCG@10 of combine(repeat, E2, r), r in 0..10; r* = pipeline.select_r (argmax, ties -> larger r); "
                          "R2 scored with the C* model fitted on the LR training users and evaluated on the validation users "
                          "(pipeline.tune_integration), R1 on all fit users",
            "r_grid": R_GRID, "lr_consistency": lr_consistency, "curves": curves, "timing_seconds": timings}


# ---------------------------------------------------------------------------
# one dataset
# ---------------------------------------------------------------------------
def analyse_dataset(config: dict[str, Any], models: dict[str, Any], dataset: str, n_boot: int, seed: int) -> dict[str, Any]:
    """Parts A, B, C on the eval split and A (robustness) + the C fit-split check on the fit split of ``dataset``."""
    # Load the selections, the LR bundle and the tuning document of the dataset.
    timings: dict[str, float] = {}
    selections = load_selections(models, dataset)
    integration = load_integration_selections(models, dataset)
    bundle = load_lr_repeat(models, dataset)
    tuning_document = json.loads(tuning_path(models, dataset).read_text(encoding="utf-8"))
    # Take gamma*, the user-CF settings and the repeat modules.
    gamma_star = float(selections["gamma"])
    k_cf, weighting = int(integration["k"]), str(integration["weighting"])
    modules = repeat_modules(gamma_star)
    # Start the result entry with the label, the bins, the selections and the modules.
    entry: dict[str, Any] = {
        "label": LABEL,
        "history_bins": [name for name, _, _ in bin_labels(HISTORY_BINS[dataset])],
        "selections": {"gamma": gamma_star, "gamma_c": float(selections["gamma_c"]), "alpha": float(selections["alpha"]),
                       "r": int(selections["r"]), "usercf_k": k_cf, "usercf_weighting": weighting, "lr_C": float(bundle["C"]),
                       "lr_artifact": bundle["artifact"], "lr_sha256": bundle["sha256"], "tuning_document": selections["source"]},
        "modules": modules,
    }

    # eval split: A, lists, C, B
    # Load the eval split and time it.
    t0 = time.perf_counter()
    data = prepare(config, dataset, "eval")
    timings["eval_load"] = time.perf_counter() - t0
    # Compute the history lengths and part A on the eval split.
    lengths = history_lengths(data["history"])
    novelty_eval = analyse_novelty(data["history"], data["target"], dataset)
    # Build the E2 explore list and time it.
    t0 = time.perf_counter()
    explore = ucf_list(data["history"], k_cf, weighting, K, data["pop"], block_size=int(config["usercf"]["block_size"]))
    timings["eval_explore_list"] = time.perf_counter() - t0
    # Build the repeat lists and the long R2 list and time them.
    t0 = time.perf_counter()
    lists, r2_long = repeat_lists(data["history"], data["pop"], modules, bundle, REPEAT_RANKS)
    timings["eval_repeat_lists"] = time.perf_counter() - t0
    # Run part C and part B.
    slots = analyse_slots(data, lists, explore, modules, lengths, dataset, n_boot, seed)
    hits = analyse_rank_hits(r2_long, explore, data["target"], lengths, dataset, n_boot, seed)
    # Store the eval results and free the eval frames.
    entry["eval"] = {"role": "main", "n_users": int(data["target"]["user_id"].nunique()), "novelty": novelty_eval,
                     "rank_hits": hits, "slots": slots}
    del data, lists, r2_long, explore

    # fit split: A robustness check and the slot curves
    # Load the fit split and time it.
    t0 = time.perf_counter()
    fit = prepare(config, dataset, "fit")
    timings["fit_load"] = time.perf_counter() - t0
    # Run part A and the slot curves on the fit split.
    novelty_fit = analyse_novelty(fit["history"], fit["target"], dataset)
    curves = slot_curves(fit, config, modules, k_cf, weighting, bundle, tuning_document)
    # Store the fit results and free the fit frames.
    entry["fit"] = {"role": "robustness check", "n_users": int(fit["target"]["user_id"].nunique()), "novelty": novelty_fit,
                    "slot_curves": curves}
    del fit

    # Add the reproduction check, the timings and the peak memory.
    entry["reproduction_check"] = reproduction_check(config["profile"], dataset, entry)
    entry["timing_seconds"] = timings
    entry["peak_memory_mib"] = peak_memory_mib()
    return entry


def reproduction_check(profile: str, dataset: str, entry: dict[str, Any]) -> dict[str, Any]:
    """Compare the SANITY numbers of (profile, dataset) with the observed eval values at four decimals."""
    # Return an empty result when no number is recorded for (profile, dataset).
    expected = SANITY.get((profile, dataset))
    if not expected:
        return {}
    # Index the eval slot table by (module, r).
    table = {(row["module"], row["r"]): row for row in entry["eval"]["slots"]["table"]}
    # Collect the observed NDCG@10 of R2 at r = 10 and the R1 repeat-list recalls.
    observed = {"R2_E2_r10_ndcg": table[("R2", 10)]["ndcg"]}
    for module in ("R1_g0.97", "R1_g1"):
        if (module, 10) in table:
            observed[f"{module}_repeat_list_recall"] = table[(module, 10)]["repeat_list_recall"]
    # Compare expected and observed values at four decimals.
    checks = {key: {"expected": value, "observed": observed.get(key),
                    "matched": observed.get(key) is not None and abs(round(observed[key], 4) - value) < 1e-9}
              for key, value in expected.items()}
    # Add the flag that every value matched.
    checks["all_matched"] = all(item["matched"] for item in checks.values())
    return checks


# ---------------------------------------------------------------------------
# setup check: tuning documents against the reference
# ---------------------------------------------------------------------------
def _get(document: dict[str, Any], path: tuple[str, ...]) -> Any:
    '''Value at the key ``path`` of the nested dict ``document``; None when a key is missing.'''
    # Walk down the keys and stop with None at a missing key or a non-dict value.
    value: Any = document
    for key in path:
        if not isinstance(value, dict) or key not in value:
            return None
        value = value[key]
    return value


def _equal(a: Any, b: Any) -> bool:
    """Numbers within TOLERANCE (10 == 10.0), everything else by ==; a bool never equals a number."""
    # Treat a bool and a non-bool as different.
    if isinstance(a, bool) != isinstance(b, bool):
        return False
    # Compare two numbers within TOLERANCE.
    if isinstance(a, (int, float)) and isinstance(b, (int, float)) and not isinstance(a, bool):
        return math.isclose(float(a), float(b), rel_tol=0.0, abs_tol=TOLERANCE)
    # Compare everything else with ==.
    return a == b


def _rows_equal(run_rows: list[dict[str, Any]], ref_rows: list[dict[str, Any]], keys: tuple[str, ...], value: str) -> bool:
    """Same key tuples in the same order and ``value`` equal within TOLERANCE (None-safe)."""
    # Return False when a list is missing or the lengths differ.
    if run_rows is None or ref_rows is None or len(run_rows) != len(ref_rows):
        return False
    # Return False at the first row whose keys or value differ.
    for a, b in zip(run_rows, ref_rows):
        if any(not _equal(a.get(key), b.get(key)) for key in keys) or not _equal(a.get(value), b.get(value)):
            return False
    return True


def compare_tuning(run: dict[str, Any], reference: dict[str, Any]) -> dict[str, Any]:
    """Selections of ``run`` against ``reference``; grids compared without the user-CF rows of k = NEW_K."""
    # Compare every selection of SELECTION_PATHS.
    selections = []
    for name, path in SELECTION_PATHS.items():
        a, b = _get(run, path), _get(reference, path)
        selections.append({"quantity": name, "run": a, "reference": b, "equal": _equal(a, b)})
    # Take the user-CF grids without the rows of k = NEW_K.
    run_grid = _get(run, ("integration", "usercf", "grid")) or []
    ref_grid = _get(reference, ("integration", "usercf", "grid")) or []
    run_other = [row for row in run_grid if int(row["k"]) != NEW_K]
    ref_other = [row for row in ref_grid if int(row["k"]) != NEW_K]
    # Compare the three stage grids row by row.
    stage_equal = {
        "stage1_repeat_recall": _rows_equal(run.get("stage1_repeat_recall"), reference.get("stage1_repeat_recall"), ("gamma",), "recall"),
        "stage2_explore_new_recall": _rows_equal(run.get("stage2_explore_new_recall"), reference.get("stage2_explore_new_recall"),
                                                 ("gamma_c", "alpha"), "recall"),
        "stage3_final_ndcg": _rows_equal(run.get("stage3_final_ndcg"), reference.get("stage3_final_ndcg"), ("r",), "ndcg"),
    }
    # Return the selection rows, the grid comparisons, the artifact hashes and the run metadata.
    return {
        "selections": selections,
        "all_selections_equal": all(row["equal"] for row in selections),
        "usercf_grid_excluding_k10_equal": _rows_equal(run_other, ref_other, ("k", "weighting"), "new_recall@10"),
        "usercf_k10_rows_run": [row for row in run_grid if int(row["k"]) == NEW_K],
        "usercf_k10_rows_reference": [row for row in ref_grid if int(row["k"]) == NEW_K],
        "stage_grids_equal": stage_equal,
        "lr_validation_equal": _rows_equal(_get(run, ("lr_repeat", "validation")), _get(reference, ("lr_repeat", "validation")),
                                           ("C",), "val_recall"),
        "lr_artifact_sha256": {"run": _get(run, ("lr_repeat", "sha256")), "reference": _get(reference, ("lr_repeat", "sha256")),
                               "equal": _get(run, ("lr_repeat", "sha256")) == _get(reference, ("lr_repeat", "sha256"))},
        "run_meta": {key: run.get("meta", {}).get(key) for key in ("git_commit", "utc_time")},
        "reference_meta": {key: reference.get("meta", {}).get(key) for key in ("git_commit", "utc_time")},
    }


def check_lr_artifact(run: dict[str, Any]) -> dict[str, Any]:
    """The LR artifact named in the tuning document exists and its sha256 equals the recorded one."""
    # Read the recorded artifact path and sha256 from the tuning document.
    recorded = _get(run, ("lr_repeat", "artifact"))
    expected = _get(run, ("lr_repeat", "sha256"))
    # Hash the artifact when the file exists.
    path = Path(recorded) if recorded else None
    exists = bool(path and path.exists())
    observed = sha256_of(path) if exists else None
    # Return the recorded and observed values and whether they match.
    return {"artifact": recorded, "exists": exists, "sha256_recorded": expected, "sha256_observed": observed,
            "matches": exists and observed == expected}


def setup_check(models: dict[str, Any], datasets: list[str], reference_dir: Path) -> dict[str, Any]:
    """compare_tuning for every dataset (run documents under the repo_root of ``models``); SystemExit when a selection differs."""
    # Refuse a reference folder with a confirmatory path component.
    reference_dir = guard_path(Path(reference_dir))
    # Prepare the result and the list of differing selections.
    out: dict[str, Any] = {"reference_dir": str(reference_dir), "datasets": {}}
    problems = []
    # Compare the tuning document of every dataset with its reference.
    for dataset in datasets:
        # Locate the run document and the reference document of the same name.
        run_path = guard_path(tuning_path(models, dataset))
        ref_path = reference_dir / run_path.name
        # Raise when either document is missing.
        for path in (run_path, ref_path):
            if not path.exists():
                raise FileNotFoundError(f"tuning document missing: {path}")
        # Load both documents and compare them.
        run = json.loads(run_path.read_text(encoding="utf-8"))
        reference = json.loads(ref_path.read_text(encoding="utf-8"))
        result = compare_tuning(run, reference)
        # Add the artifact check and the file record.
        result["lr_artifact"] = check_lr_artifact(run)
        result["files"] = {"run": str(run_path), "reference": str(ref_path), "reference_sha256": sha256_of(ref_path)}
        out["datasets"][dataset] = result
        # Note every selection that differs.
        for row in result["selections"]:
            if not row["equal"]:
                problems.append(f"{dataset}: {row['quantity']} run={row['run']!r} reference={row['reference']!r}")
    # Stop the script when any selection differs.
    out["passed"] = not problems
    if problems:
        raise SystemExit("setup check failed; selections differ from the reference tuning documents:\n  " + "\n  ".join(problems))
    return out


# ---------------------------------------------------------------------------
# figures (each drawn from its own CSV)
# ---------------------------------------------------------------------------
def _style() -> None:
    '''Set the matplotlib font, text, axis and background colours from INK.'''
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.edgecolor": INK["axis"], "axes.labelcolor": INK["secondary"],
                         "xtick.color": INK["secondary"], "ytick.color": INK["secondary"], "text.color": INK["primary"],
                         "figure.facecolor": INK["surface"], "savefig.facecolor": INK["surface"], "axes.facecolor": INK["surface"]})


def _finish_axes(ax: plt.Axes) -> None:
    '''Draw horizontal grid lines only on ``ax`` and hide its top and right spines.'''
    # Show the grid on the y axis only.
    ax.grid(True, axis="y", color=INK["grid"], lw=0.8)
    ax.grid(False, axis="x")
    # Hide the top and right spines.
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def new_rate_rows(entries: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    """F-A numbers: eval history baskets (basket-weighted) P(any_new) and mean new_share by position bin, plus the target basket overall."""
    rows = []
    # Collect the rows of every dataset.
    for dataset, entry in entries.items():
        # Take the basket sets of the eval split.
        sets = entry["eval"]["novelty"]["basket_sets"]
        # Add one row per position bin of the history baskets.
        for row in sets["history_all"]["by_position"]:
            bw = row["basket_weighted"] or {}
            rows.append({"dataset": dataset, "basket_set": "history_all", "position_bin": row["bin"], "n_baskets": row["n_baskets"],
                         "p_any_new": bw.get("p_any_new"), "mean_new_share": bw.get("mean_new_share")})
        # Add one row for the target basket over all positions.
        t = sets["target"]["basket_weighted"] or {}
        rows.append({"dataset": dataset, "basket_set": "target", "position_bin": "all", "n_baskets": sets["target"]["n_baskets"],
                     "p_any_new": t.get("p_any_new"), "mean_new_share": t.get("mean_new_share")})
    return rows


def figure_new_rate(entries: dict[str, dict[str, Any]], out_dir: Path) -> dict[str, Any]:
    '''Write the CSV and the PNG of figure F-A (new-item rate by basket position) to ``out_dir``.

    ``entries`` maps dataset -> result entry; returns the file names and the number of points.
    '''
    # Write the figure numbers to the CSV file.
    rows = new_rate_rows(entries)
    csv_path = out_dir / f"{FIGURES['new_rate']}.csv"
    frame = pd.DataFrame(rows)
    frame.to_csv(csv_path, index=False)
    # Open a figure with one panel per metric.
    _style()
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4), layout="constrained")
    # Use the position bins as the x axis.
    bins = [name for name, _, _ in POSITION_BINS]
    x = np.arange(len(bins))
    # Draw one panel for P(any_new) and one for mean new_share.
    for ax, (metric, title) in zip(axes, (("p_any_new", "P(any_new): share of baskets with >= 1 new item"),
                                          ("mean_new_share", "mean new_share: share of the basket that is new"))):
        for slot, dataset in zip(("slot1", "slot2"), entries):
            # Take the history rows of the dataset in bin order.
            sub = frame[(frame["dataset"] == dataset) & (frame["basket_set"] == "history_all")].set_index("position_bin").reindex(bins)
            y = sub[metric].to_numpy(dtype="float64")
            # Plot the line and label its last point with the dataset name.
            ax.plot(x, y, color=COLORS[slot], lw=2, marker="o", ms=8, mec=INK["surface"], mew=1.5, label=f"{NAMES[dataset]} history baskets")
            ax.annotate(NAMES[dataset], (x[-1], y[-1]), xytext=(8, 0), textcoords="offset points", va="center", fontsize=9,
                        color=INK["primary"])
            # Draw the target-basket value as a dotted horizontal line.
            target = frame[(frame["dataset"] == dataset) & (frame["basket_set"] == "target")][metric].iloc[0]
            ax.axhline(target, color=COLORS[slot], lw=1.2, ls=":", label=f"{NAMES[dataset]} target basket")
        # Set the ticks, limits, labels, title and legend of the panel.
        ax.set_xticks(x)
        ax.set_xticklabels(bins)
        ax.set_xlim(-0.4, len(bins) - 0.3)
        ax.set_ylim(0, 1.02)
        ax.set_xlabel("basket position (basket_seq)")
        ax.set_ylabel(metric.replace("_", " "))
        ax.set_title(title, fontsize=10.5, color=INK["primary"])
        ax.legend(frameon=False, fontsize=8.5, loc="lower left")
        _finish_axes(ax)
    # Add the figure title with the label of the study.
    fig.suptitle("New-item purchase rate of history baskets by basket position (eval split, basket-weighted); dotted = target basket\n"
                 f"{LABEL}", fontsize=9.5, color=INK["secondary"])
    # Save the PNG and close the figure.
    png_path = out_dir / f"{FIGURES['new_rate']}.png"
    fig.savefig(png_path, dpi=200)
    plt.close(fig)
    return {"png": png_path.name, "csv": csv_path.name, "points": int(len(frame))}


def rank_hit_rows(entries: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    '''F-B rows of ``entries``: one per dataset, list (repeat_R2, explore_E2) and eval rank.'''
    rows = []
    # Copy every per-rank row of both lists with the dataset and the list name.
    for dataset, entry in entries.items():
        hits = entry["eval"]["rank_hits"]
        for list_name, key in (("repeat_R2", "repeat_R2"), ("explore_E2", "explore_E2")):
            for row in hits[key]:
                rows.append({"dataset": dataset, "list": list_name, **row})
    return rows


def figure_rank_hits(entries: dict[str, dict[str, Any]], out_dir: Path) -> dict[str, Any]:
    '''Write the CSV and the PNG of figure F-B (hit probability by rank) to ``out_dir``.

    ``entries`` maps dataset -> result entry; returns the file names and the number of points.
    '''
    # Write the figure numbers to the CSV file.
    rows = rank_hit_rows(entries)
    csv_path = out_dir / f"{FIGURES['rank_hits']}.csv"
    frame = pd.DataFrame(rows)
    frame.to_csv(csv_path, index=False)
    # Open a figure with one panel per dataset.
    _style()
    n = len(entries)
    fig, axes = plt.subplots(1, n, figsize=(6.0 * n, 4.4), layout="constrained")
    axes = np.atleast_1d(axes)
    # Draw the panel of every dataset.
    for ax, dataset in zip(axes, entries):
        # Take the rows of the repeat list and of the explore list in rank order.
        rep = frame[(frame["dataset"] == dataset) & (frame["list"] == "repeat_R2")].sort_values("rank")
        exp = frame[(frame["dataset"] == dataset) & (frame["list"] == "explore_E2")].sort_values("rank")
        # Plot the hit probability of both lists against the rank.
        ax.plot(rep["rank"], rep["p_hit"], color=COLORS["slot1"], lw=2, marker="o", ms=8, mec=INK["surface"], mew=1.5,
                label="repeat list R2 (LR ranker), ranks 1-12")
        ax.plot(exp["rank"], exp["p_hit"], color=COLORS["slot2"], lw=2, marker="s", ms=8, mec=INK["surface"], mew=1.5,
                label="explore list E2 (user-CF), ranks 1-3")
        # Annotate R2 rank 10 and E2 rank 1 with their values.
        r10 = float(rep.loc[rep["rank"] == 10, "p_hit"].iloc[0])
        e1 = float(exp.loc[exp["rank"] == 1, "p_hit"].iloc[0])
        ax.annotate(f"R2 rank 10: {r10:.3f}", (10, r10), xytext=(0, 12), textcoords="offset points", ha="center", fontsize=9)
        ax.annotate(f"E2 rank 1: {e1:.3f}", (1, e1), xytext=(12, 6), textcoords="offset points", ha="left", va="bottom", fontsize=9)
        # Set the ticks, labels, limits, title and legend of the panel.
        ax.set_xticks(range(1, REPEAT_RANKS + 1))
        ax.set_xlabel("rank in the list")
        ax.set_ylabel("P(item at the rank is in the target basket)")
        ax.set_ylim(bottom=0)
        ax.margins(y=0.15)
        ax.set_title(f"{NAMES[dataset]} ({entries[dataset]['eval']['rank_hits']['n_users']:,} users)", fontsize=10.5, color=INK["primary"])
        ax.legend(frameon=False, fontsize=8.5, loc="upper right")
        _finish_axes(ax)
    # Add the figure title with the label of the study.
    fig.suptitle("Hit probability by rank, eval split, all users (users without an item at a rank count as no hit); y axes differ per panel\n"
                 f"{LABEL}", fontsize=9.5, color=INK["secondary"])
    # Save the PNG and close the figure.
    png_path = out_dir / f"{FIGURES['rank_hits']}.png"
    fig.savefig(png_path, dpi=200)
    plt.close(fig)
    return {"png": png_path.name, "csv": csv_path.name, "points": int(len(frame))}


def slot_effect_rows(entries: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    '''F-C numbers: per dataset, module and r the difference to r = 10 with its 95 % CI.

    ``entries`` maps dataset -> result entry; the metrics are NDCG@10 and the new-item hit share.
    '''
    rows = []
    for dataset, entry in entries.items():
        # Map the module names to their labels.
        labels = {module["name"]: module["label"] for module in entry["modules"]}
        # Add one row per difference for NDCG@10 and for the new-item hit share.
        for diff in entry["eval"]["slots"]["differences"]:
            for metric in ("ndcg", "new_hit_share"):
                item = diff["metrics"][metric]
                rows.append({"dataset": dataset, "module": diff["module"], "module_label": labels[diff["module"]], "r": diff["r"],
                             "metric": metric, "diff": item["diff"], "ci_lo": item["ci_0.95"][0], "ci_hi": item["ci_0.95"][1],
                             "n_users": item["n_users"]})
    return rows


def figure_slot_effects(entries: dict[str, dict[str, Any]], out_dir: Path) -> dict[str, Any]:
    '''Write the CSV and the PNG of figure F-C (metric change of r = 9 and 8 vs 10) to ``out_dir``.

    ``entries`` maps dataset -> result entry; returns the file names and the number of points.
    '''
    # Write the figure numbers to the CSV file.
    rows = slot_effect_rows(entries)
    csv_path = out_dir / f"{FIGURES['slots']}.csv"
    frame = pd.DataFrame(rows)
    frame.to_csv(csv_path, index=False)
    # Open a grid of panels with one row per metric and one column per dataset.
    _style()
    n = len(entries)
    metrics = (("ndcg", "change in NDCG@10"), ("new_hit_share", "change in share with a new-item hit"))
    fig, axes = plt.subplots(len(metrics), n, figsize=(6.0 * n, 3.3 * len(metrics)), layout="constrained", squeeze=False)
    # Set the colour, marker, fill and legend label of r = 9 and r = 8.
    styles = {9: ("slot1", "o", "full", "r = 9 (one exploration slot)"), 8: ("slot2", "s", "none", "r = 8 (two exploration slots)")}
    # Draw the panels of every dataset.
    for col, dataset in enumerate(entries):
        # Place the modules on the y axis from top to bottom.
        modules = [module["label"] for module in entries[dataset]["modules"]]
        y_base = np.arange(len(modules))[::-1]
        for row_index, (metric, title) in enumerate(metrics):
            # Mark zero with a dashed vertical line.
            ax = axes[row_index, col]
            ax.axvline(0, color=INK["muted"], lw=1, ls="--")
            # Plot the differences of r = 9 and r = 8 with their intervals, offset vertically.
            for r, (slot, marker, fill, label) in styles.items():
                offset = 0.17 if r == 9 else -0.17
                sub = frame[(frame["dataset"] == dataset) & (frame["metric"] == metric) & (frame["r"] == r)]
                sub = sub.set_index("module_label").reindex(modules)
                y = y_base + offset
                err = np.vstack([sub["diff"] - sub["ci_lo"], sub["ci_hi"] - sub["diff"]])
                ax.errorbar(sub["diff"], y, xerr=err, fmt=marker, color=COLORS[slot], ecolor=COLORS[slot], elinewidth=1.6, capsize=3,
                            ms=8, mfc=COLORS[slot] if fill == "full" else INK["surface"], mec=COLORS[slot], mew=1.6, label=label)
            # Set the ticks, limits, label and title of the panel.
            ax.set_yticks(y_base)
            ax.set_yticklabels(modules)
            ax.set_ylim(-0.7, len(modules) - 0.3)
            ax.set_xlabel(f"{title} vs r = 10 (95 % CI)")
            ax.set_title(f"{NAMES[dataset]}: {title}", fontsize=10.5, color=INK["primary"])
            # Show the grid on the x axis only and hide the top and right spines.
            ax.grid(True, axis="x", color=INK["grid"], lw=0.8)
            ax.grid(False, axis="y")
            for side in ("top", "right"):
                ax.spines[side].set_visible(False)
    # Put one shared legend below the panels.
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=2, frameon=False, fontsize=9)
    # Add the figure title with the label of the study.
    fig.suptitle("Forced exploration slots: metric change of r = 9 and r = 8 against r = 10 per repeat module (eval split, paired bootstrap); "
                 f"x axes differ per panel\n{LABEL}", fontsize=9.5, color=INK["secondary"])
    # Save the PNG and close the figure.
    png_path = out_dir / f"{FIGURES['slots']}.png"
    fig.savefig(png_path, dpi=200)
    plt.close(fig)
    return {"png": png_path.name, "csv": csv_path.name, "points": int(len(frame))}


# ---------------------------------------------------------------------------
# markdown
# ---------------------------------------------------------------------------
def _f(value: float | None, digits: int = 3) -> str:
    '''Format ``value`` with ``digits`` decimals; a dash for None.'''
    return "-" if value is None else f"{value:.{digits}f}"


def _d(value: float | None, digits: int = 4) -> str:
    '''Format ``value`` with a sign and ``digits`` decimals; a dash for None.'''
    return "-" if value is None else f"{value:+.{digits}f}"


def _ci(item: dict[str, Any] | None, digits: int = 4) -> str:
    '''Format ``item`` as diff [lo, hi] with signs and ``digits`` decimals; a dash without a CI.'''
    # Return a dash when the item is empty or has no interval.
    if not item or "ci_0.95" not in item:
        return "-"
    # Format the difference and both interval bounds with a sign.
    lo, hi = item["ci_0.95"]
    return f"{item['diff']:+.{digits}f} [{lo:+.{digits}f}, {hi:+.{digits}f}]"


def _pair(block: dict[str, Any], key: str, digits: int = 3) -> str:
    """basket-weighted / user-weighted cell."""
    # Return a dash when the block has no basket.
    if block["basket_weighted"] is None:
        return "-"
    # Join the basket-weighted and the user-weighted value.
    return f"{_f(block['basket_weighted'][key], digits)} / {_f(block['user_weighted'][key], digits)}"


def _table(header: list[str], rows: list[list[str]]) -> list[str]:
    '''Markdown table lines for ``header`` and ``rows``, followed by one empty line.'''
    # Write the header row and the separator row.
    lines = ["| " + " | ".join(header) + " |", "| " + " | ".join("---" for _ in header) + " |"]
    # Add one line per row.
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return lines + [""]


def _bins_table(blocks: dict[str, dict[str, Any]], key: str, bin_header: str) -> list[str]:
    """Per basket set: rows = bins, columns = baskets / users / new_share / P(any_new) / n_new / size."""
    lines = []
    # Write one table per basket set.
    for set_name, block in blocks.items():
        rows = []
        # Build one row per bin.
        for row in block[key]:
            # Start with the bin name and the basket and user counts.
            cells = [row["bin"], f"{row['n_baskets']:,}", f"{row['n_users']:,}"]
            # Show basket-weighted values for the target set and both weightings otherwise.
            if set_name == "target":
                bw = row["basket_weighted"]
                cells += [_f(bw and bw["mean_new_share"]), _f(bw and bw["p_any_new"]), _f(bw and bw["mean_n_new"], 2),
                          _f(bw and bw["mean_size"], 1)]
            else:
                cells += [_pair(row, "mean_new_share"), _pair(row, "p_any_new"), _pair(row, "mean_n_new", 2), _pair(row, "mean_size", 1)]
            rows.append(cells)
        # Write the caption of the basket set and its table.
        weighting = "" if set_name == "target" else " (basket-weighted / user-weighted)"
        lines += [f"{block['description'].capitalize()}{weighting}:", ""]
        lines += _table([bin_header, "baskets", "users", "mean new_share", "P(any_new)", "mean n_new", "mean size"], rows)
    return lines


def render_novelty(entry: dict[str, Any], dataset: str, split: str) -> list[str]:
    '''Markdown lines of part A for ``dataset`` and ``split`` of the result ``entry``.'''
    # Take the checks and the basket sets of the split.
    block = entry[split]["novelty"]
    checks, sets = block["checks"], block["basket_sets"]
    # Write the heading.
    lines = [f"### {NAMES[dataset]}, {split} split ({entry[split]['role']})", ""]
    # Write the bullet lines with the counts and the data checks.
    gap = checks["target_seq_minus_last_history_seq"]
    lines += [f"- Users {checks['n_users']:,}; history baskets {checks['n_history_baskets']:,}; target baskets {checks['n_target_baskets']:,}; "
              f"baskets analysed {checks['n_baskets_analysed']:,} (first baskets dropped {checks['n_first_baskets_dropped']:,}). "
              f"History length (distinct history baskets): min {block['history_length_summary']['min']}, median "
              f"{block['history_length_summary']['median']:g}, max {block['history_length_summary']['max']}.",
              f"- basket_seq starts at 1 for {checks['history_seq_starts_at_one_share']:.0%} of the users and is consecutive for "
              f"{checks['history_seq_consecutive_share']:.0%}; target basket_seq minus last history basket_seq: "
              + ", ".join(f"{k}: {v:,}" for k, v in gap.items())
              + f" ({checks['users_with_skipped_baskets_before_target']:,} users have baskets between history and target that are in neither frame).",
              f"- Share of history baskets with >= {MIN_ITEMS} distinct items {checks['history_baskets_min_items_share']:.3f}; smallest target basket "
              f"{checks['target_size_min']} items. Users with >= 1 new target item: {checks['target_any_new_users']:,} here, "
              f"{checks['split_target_new_users']:,} by nbr.data.split_target ({'equal' if checks['matches_split_target'] else 'NOT equal'}).", ""]
    # Write the overall table: one target row, two weighting rows per history set.
    lines += ["Overall:", ""]
    rows = []
    for name, s in sets.items():
        if s["basket_weighted"] is None:
            rows.append([s["description"], "0", "0", "-", "-", "-", "-", "-"])
            continue
        if name == "target":
            bw = s["basket_weighted"]
            rows.append([s["description"], f"{s['n_baskets']:,}", f"{s['n_users']:,}", "per user", _f(bw["mean_new_share"]),
                         _f(bw["p_any_new"]), _f(bw["mean_n_new"], 2), _f(bw["mean_size"], 1)])
        else:
            for weighting, values in (("basket-weighted", s["basket_weighted"]), ("user-weighted", s["user_weighted"])):
                rows.append([s["description"], f"{s['n_baskets']:,}", f"{s['n_users']:,}", weighting, _f(values["mean_new_share"]),
                             _f(values["p_any_new"]), _f(values["mean_n_new"], 2), _f(values["mean_size"], 1)])
    lines += _table(["basket set", "baskets", "users", "weighting", "mean new_share", "P(any_new)", "mean n_new", "mean size"], rows)
    # Write the distribution of n_new.
    lines += ["Distribution of n_new (basket-weighted shares):", ""]
    rows = [[s["description"]] + ([_f(s["n_new_distribution"][k]) for k in ("0", "1", "2", "3+")] if s["n_new_distribution"] else ["-"] * 4)
            for s in sets.values()]
    lines += _table(["basket set", "n_new = 0", "n_new = 1", "n_new = 2", "n_new >= 3"], rows)
    # Write the tables by basket position.
    lines += ["By basket position (position = basket_seq):", ""]
    lines += _bins_table(sets, "by_position", "position")
    # Write the tables by history-length bin.
    lines += [f"By T5 history-length bin (distinct {split}-history baskets of the user):", ""]
    lines += _bins_table(sets, "by_history_length", "history length")
    return lines


def render_rank_hits(entries: dict[str, dict[str, Any]]) -> list[str]:
    '''Markdown lines of part B for all datasets in ``entries``.'''
    # Write the heading and the definitions.
    lines = ["## B. Hit probability by rank (eval split)", "",
             "P(hit at rank j) = share of all users whose list has an item at rank j that is in their target basket (users without an item at "
             "the rank count as no hit); 'present' = share of users with an item at the rank; 'given present' = hits / users with an item. "
             "R2 = LR repeat ranker (genuine repeat candidates only, so short histories run out of items), E2 = user-CF explore list "
             "(unseen items, padded with popularity; every E2 hit is a new-item hit).", ""]
    # Write the per-rank table of every dataset.
    for dataset, entry in entries.items():
        hits = entry["eval"]["rank_hits"]
        # Index the R2 and E2 rows by rank.
        rep = {row["rank"]: row for row in hits["repeat_R2"]}
        exp = {row["rank"]: row for row in hits["explore_E2"]}
        # Build one row per rank; E2 cells are dashes beyond its ranks.
        rows = []
        for rank in range(1, REPEAT_RANKS + 1):
            r, e = rep[rank], exp.get(rank)
            rows.append([str(rank), _f(r["p_hit"]), _f(r["share_present"]), _f(r["p_hit_given_present"]),
                         _f(e["p_hit"]) if e else "-", _f(e["share_present"]) if e else "-", _f(e["p_hit_given_present"]) if e else "-"])
        # Write the caption with the expected hits per user and the table.
        lines += [f"{NAMES[dataset]} ({hits['n_users']:,} users); expected hits per user: R2 ranks 1-10 {hits['expected_hits_top10']['repeat_R2_ranks_1_10']:.3f}, "
                  f"E2 ranks 1-3 {hits['expected_hits_top10']['explore_E2_ranks_1_3']:.3f}:", ""]
        lines += _table(["rank", "R2 P(hit)", "R2 present", "R2 P(hit given present)", "E2 P(hit)", "E2 present", "E2 P(hit given present)"], rows)
    # Write the table of the two slot comparisons.
    lines += ["Slot comparisons (paired bootstrap over all users; diff = explore hits minus repeat hits in the slots that change):", ""]
    rows = []
    for dataset, entry in entries.items():
        for key, item in entry["eval"]["rank_hits"]["comparisons"].items():
            rows.append([NAMES[dataset], item["description"].split(";")[0], _f(item["mean_a"]), _f(item["mean_b"]), _ci(item), f"{item['n_users']:,}"])
    lines += _table(["dataset", "comparison", "repeat hits (mean)", "explore hits (mean)", "diff [95 % CI]", "users"], rows)
    # Write the table by history-length bin of every dataset.
    lines += ["By T5 history-length bin (P(hit) at the slot-relevant ranks and the two comparisons):", ""]
    for dataset, entry in entries.items():
        hits = entry["eval"]["rank_hits"]
        # Index the two comparisons by bin.
        comp1 = {row["bin"]: row for row in hits["comparisons"]["slot10_explore1_vs_repeat10"]["by_history_length"]}
        comp2 = {row["bin"]: row for row in hits["comparisons"]["slots9_10_explore12_vs_repeat910"]["by_history_length"]}
        # Build one row per bin, with dashes for an empty bin.
        rows = []
        for row in hits["by_history_length"]:
            if row["n_users"] == 0:
                rows.append([row["bin"], "0", "-", "-", "-", "-", "-", "-"])
                continue
            rows.append([row["bin"], f"{row['n_users']:,}", _f(row["repeat"]["9"]), _f(row["repeat"]["10"]), _f(row["explore"]["1"]),
                         _f(row["explore"]["2"]), _ci(comp1[row["bin"]]), _ci(comp2[row["bin"]])])
        # Write the dataset caption and the table.
        lines += [f"{NAMES[dataset]}:", ""]
        lines += _table(["history length", "users", "R2 rank 9", "R2 rank 10", "E2 rank 1", "E2 rank 2", "E2 1 - R2 10 [CI]", "E2 1+2 - R2 9+10 [CI]"], rows)
    return lines


def render_slots(entries: dict[str, dict[str, Any]]) -> list[str]:
    '''Markdown lines of part C (eval split) for all datasets in ``entries``.'''
    # Write the heading and the definitions.
    lines = ["## C. Forced exploration slot (eval split)", "",
             "Final list = nbr.combine.combine(repeat, E2, r): the first r slots from the repeat module (fewer when the user has fewer repeat "
             "candidates; E2 items fill up), the remaining 10 - r slots from E2. Metrics: NDCG@10 and Recall@10 over all users; new-item "
             "Recall@10 over users with >= 1 new target item; share of users (all) with >= 1 new-item hit in the top 10; repeat Recall@10 over "
             "users with >= 1 repeat target item. 'repeat list' = repeat Recall@10 of the repeat module's own list.", ""]
    # Write the metric table of every dataset.
    for dataset, entry in entries.items():
        # Map the module names to their labels.
        labels = {module["name"]: module["label"] for module in entry["modules"]}
        slots = entry["eval"]["slots"]
        # Build one row per module and r.
        rows = [[labels[row["module"]], str(row["r"]), _f(row["ndcg"], 4), _f(row["recall"], 4), _f(row["new_recall"], 4),
                 _f(row["new_hit_share"], 4), _f(row["repeat_recall"], 4), _f(row["repeat_list_recall"], 4)] for row in slots["table"]]
        # Write the caption with the user counts and the table.
        first = slots["table"][0]
        lines += [f"{NAMES[dataset]} ({slots['n_users']:,} users; new-target users {first['new_recall_n_users']:,}, repeat-target users "
                  f"{first['repeat_recall_n_users']:,}):", ""]
        lines += _table(["repeat module", "r", "NDCG@10", "Recall@10", "new-item Recall@10", "share with new hit", "repeat Recall@10", "repeat list"], rows)
    # Write the table of differences against r = 10 of every dataset.
    lines += ["Differences against r = 10 (paired bootstrap, diff [95 % CI]):", ""]
    for dataset, entry in entries.items():
        labels = {module["name"]: module["label"] for module in entry["modules"]}
        rows = []
        for diff in entry["eval"]["slots"]["differences"]:
            rows.append([labels[diff["module"]], diff["comparison"]] + [_ci(diff["metrics"][metric]) for metric in LIST_METRICS])
        lines += [f"{NAMES[dataset]}:", ""]
        lines += _table(["repeat module", "comparison", "NDCG@10", "Recall@10", "new-item Recall@10", "share with new hit", "repeat Recall@10"], rows)
    # Write the difference tables by history-length bin per dataset, metric and r.
    lines += ["By T5 history-length bin (diff [95 % CI]; NDCG@10 and the share of users with >= 1 new-item hit):", ""]
    for dataset, entry in entries.items():
        labels = {module["name"]: module["label"] for module in entry["modules"]}
        for metric, title in (("ndcg", "NDCG@10"), ("new_hit_share", "share with new hit")):
            for r in (9, 8):
                # Take the differences of this r and the bin names.
                diffs = [diff for diff in entry["eval"]["slots"]["differences"] if diff["r"] == r]
                bins = [row["bin"] for row in diffs[0]["metrics"][metric]["by_history_length"]]
                # Build one row per bin with one cell per module.
                rows = []
                for bin_name in bins:
                    cells = [bin_name]
                    users = None
                    for diff in diffs:
                        item = next(row for row in diff["metrics"][metric]["by_history_length"] if row["bin"] == bin_name)
                        users = item["n_users"]
                        cells.append(_ci(item) if item["n_users"] else "-")
                    rows.append([cells[0], f"{users:,}"] + cells[1:])
                # Write the caption and the table.
                lines += [f"{NAMES[dataset]}, {title}, {COMPARISONS[r]}:", ""]
                lines += _table(["history length", "users"] + [labels[diff["module"]] for diff in diffs], rows)
    return lines


def render_curves(entries: dict[str, dict[str, Any]]) -> list[str]:
    '''Markdown lines of the fit-split check of part C for all datasets in ``entries``.'''
    # Write the heading and the definitions.
    lines = ["## C. Fit-split check: NDCG@10 by r and the selected r per repeat module", "",
             "r* = pipeline.select_r (argmax over r in 0..10, ties -> larger r). R2 is scored with the C* model fitted on the LR training users "
             "and evaluated on the validation users (pipeline.tune_integration); R1 modules on all fit users. 'tuning document' = the same curve "
             "recorded by python -m nbr tune (final_lr_ucf for R2, final_star_ucf for gamma*).", ""]
    # Write the LR consistency line and the curve table of every dataset.
    for dataset, entry in entries.items():
        # Take the module labels, the curves and the LR consistency check.
        labels = {module["name"]: module["label"] for module in entry["modules"]}
        curves = entry["fit"]["slot_curves"]
        lr = curves["lr_consistency"]
        # Write the line that compares the refit with the saved artifact.
        lines += [f"{NAMES[dataset]}: LR refit C = {lr['C_refit']:g} (saved artifact C = {lr['C_saved']:g}, {'equal' if lr['C_equal'] else 'NOT equal'}), "
                  f"max |coefficient difference| {lr['max_abs_coefficient_difference']:.2e}; validation users {lr['n_val_users']:,}.", ""]
        # Build one row per module with NDCG@10 by r, the selected r and the document check.
        rows = []
        for curve in curves["curves"]:
            doc = curve.get("tuning_document_curve")
            rows.append([labels[curve["module"]], curve["users"], f"{curve['n_users']:,}"] + [_f(item["ndcg"], 4) for item in curve["ndcg_by_r"]]
                        + [str(curve["r_star"]), "yes" if curve["r_star_below_10"] else "no",
                           (f"{doc['name']}: {'equal' if doc['equal'] else 'NOT equal'}" if doc else "-")])
        # Write the table.
        lines += _table(["repeat module", "users", "n"] + [f"r={r}" for r in curves["r_grid"]] + ["r*", "r* < 10", "tuning document"], rows)
    return lines


def render_reproduction(entries: dict[str, dict[str, Any]]) -> list[str]:
    '''Markdown lines of the reproduction checks for all datasets in ``entries``.'''
    # Write the heading.
    lines = ["## Reproduction checks (eval split, four decimals)", ""]
    # Build one row per checked number.
    rows = []
    for dataset, entry in entries.items():
        for key, item in entry["reproduction_check"].items():
            if isinstance(item, dict):
                rows.append([NAMES[dataset], key, _f(item["expected"], 4), _f(item["observed"], 4), "yes" if item["matched"] else "NO"])
    # Write the table, or a note when the profile has no reference numbers.
    if rows:
        lines += _table(["dataset", "quantity", "expected", "observed", "matched"], rows)
    else:
        lines += ["(no reference numbers for this profile)", ""]
    return lines


def render_setup_check(check: dict[str, Any] | None) -> list[str]:
    '''Markdown lines of the setup check ``check`` (None when it was not run).'''
    # Write the heading.
    lines = ["## Setup check: tuning documents of this run against the W8b reference", ""]
    # Return a note when no reference folder was given.
    if check is None:
        return lines + ["(no --reference-tuning-dir given: not checked)", ""]
    # Write the reference folder and the result.
    lines += [f"- Reference: `{check['reference_dir']}` (read only). Result: {'passed' if check['passed'] else 'FAILED'} (every selection equal).", ""]
    # Write the comparison table of every dataset.
    for dataset, result in check["datasets"].items():
        # Write the caption with the files and commits.
        lines += [f"{NAMES[dataset]} (this run `{result['files']['run']}`, commit `{result['run_meta']['git_commit']}`; reference commit "
                  f"`{result['reference_meta']['git_commit']}`, {result['reference_meta']['utc_time']}):", ""]
        # Build the rows for the selections, the grids and the LR artifact.
        rows = [[row["quantity"], f"{row['run']}", f"{row['reference']}", "yes" if row["equal"] else "NO"] for row in result["selections"]]
        rows.append(["user-CF grid rows other than k = 10 (new_recall@10)", "", "", "yes" if result["usercf_grid_excluding_k10_equal"] else "NO"])
        for name, equal in result["stage_grids_equal"].items():
            rows.append([f"{name} grid", "", "", "yes" if equal else "NO"])
        rows.append(["LR validation rows (val_recall by C)", "", "", "yes" if result["lr_validation_equal"] else "NO"])
        sha = result["lr_artifact_sha256"]
        rows.append(["LR artifact sha256", (sha["run"] or "")[:12], (sha["reference"] or "")[:12], "yes" if sha["equal"] else "no"])
        art = result["lr_artifact"]
        rows.append(["LR artifact file exists with the recorded sha256", "yes" if art["matches"] else "NO", "", ""])
        # Write the table.
        lines += _table(["quantity", "this run", "reference", "equal"], rows)
        # List the k = 10 rows of this run when there are any.
        k10 = result["usercf_k10_rows_run"]
        if k10:
            lines += ["k = 10 rows of this run (absent from the reference grid): "
                      + "; ".join(f"{row['weighting']}: new_recall@10 = {row['new_recall@10']:.4f}" for row in k10), ""]
    return lines


def render_markdown(payload: dict[str, Any]) -> str:
    """Tables of the JSON document ``payload`` (as written by main)."""
    # Unpack the metadata, the dataset entries and the profile.
    meta, entries, profile = payload["meta"], payload["datasets"], payload["profile"]
    # Write the title, the description of the parts and the run facts.
    lines = [f"# New-item purchase rates and the value of an exploration slot ({profile})", "",
             f"**{LABEL}.** A: for every user and every basket from the 2nd basket on, prior(b) = items of the user's earlier baskets of the "
             "same split (history + target); new_share(b) = |b minus prior(b)| / |b| over distinct items, any_new(b) = 1 if at least one item is "
             "new, n_new(b) = number of new items; basket-weighted = mean over baskets, user-weighted = mean of the per-user means; position = "
             "basket_seq; history-length bins = T5 bins of nbr.pipeline.HISTORY_BINS. B: hit probability by rank of the R2 repeat list and the "
             "E2 explore list. C: final lists with r in {10, 9, 8} repeat slots per repeat module, paired bootstraps against r = 10, and the "
             "fit-split NDCG@10 curves by r. The eval split (already used by earlier development runs) is the main split. Descriptive and "
             "exploratory only; no confirmatory claim is made.", "",
             f"- Code commit `{meta['git_commit']}`, {meta['utc_time']}; profile `{profile}`; bootstrap n = {payload['bootstrap']['n']}, seed "
             f"{payload['bootstrap']['seed']}, 95 % CI.",
             f"- Position bins {', '.join(name for name, _, _ in POSITION_BINS)}; history-length bins "
             + "; ".join(f"{NAMES[d]} {', '.join(e['history_bins'])}" for d, e in entries.items()) + ".",
             "- Selections (python -m nbr tune, this run): " + "; ".join(
                 f"{NAMES[d]} gamma* {e['selections']['gamma']:g}, gamma_c {e['selections']['gamma_c']:g}, alpha {e['selections']['alpha']:g}, "
                 f"r* {e['selections']['r']}, user-CF k {e['selections']['usercf_k']} {e['selections']['usercf_weighting']}, LR C {e['selections']['lr_C']:g}"
                 for d, e in entries.items()) + ".",
             f"- Complete Journey note: its target is the first basket with >= {MIN_ITEMS} items after the cutoff while history baskets have no size "
             f"filter, so the A history rows are also given for baskets with >= {MIN_ITEMS} distinct items; baskets between the last history basket "
             "and the target that are in neither frame are not part of prior(b), as in the pipeline.",
             f"- Inputs (sha256 in the JSON): {', '.join(f'{d}: {len(payload['inputs_sha256'][d])} files' for d in entries)}.",
             f"- Runtime {payload['runtime_seconds']:.1f} s; peak working set {payload['peak_memory_mib']:.0f} MiB (in-process).", ""]
    # Add the setup check section.
    lines += render_setup_check(payload["setup_check"])
    # Write the headline table of part A.
    lines += ["## A. New-item purchase rates", "", "Headline (eval split; bw / uw = basket-weighted / user-weighted):", ""]
    rows = []
    for d, e in entries.items():
        sets = e["eval"]["novelty"]["basket_sets"]
        t, h = sets["target"]["basket_weighted"], sets["history_all"]
        rows.append([NAMES[d], f"{sets['target']['n_users']:,}", _f(t and t["mean_new_share"]), _f(t and t["p_any_new"]),
                     _f(t and t["mean_n_new"], 2), _pair(h, "mean_new_share"), _pair(h, "p_any_new"), _pair(h, "mean_n_new", 2)])
    lines += _table(["dataset", "users", "target mean new_share", "target P(any_new)", "target mean n_new",
                     "history new_share (bw / uw)", "history P(any_new) (bw / uw)", "history n_new (bw / uw)"], rows)
    # Add part A for the eval split and then for the fit split.
    for split in SPLITS:
        lines += [f"### {'Eval split (main)' if split == 'eval' else 'Fit split (robustness check)'}", ""]
        for dataset, entry in entries.items():
            lines += render_novelty(entry, dataset, split)
    # Add parts B and C, the fit-split curves and the reproduction checks.
    lines += render_rank_hits(entries)
    lines += render_slots(entries)
    lines += render_curves(entries)
    lines += render_reproduction(entries)
    # List the three figures with their CSV files.
    figures = payload["figures"]
    lines += ["## Figures", "",
              f"- F-A `{figures['new_rate']['png']}` (numbers in `{figures['new_rate']['csv']}`): P(any_new) and mean new_share of the eval history "
              "baskets by basket position (basket-weighted), both datasets; dotted lines = target basket.",
              f"- F-B `{figures['rank_hits']['png']}` (`{figures['rank_hits']['csv']}`): hit probability by rank, R2 ranks 1-12 and E2 ranks 1-3, "
              "one panel per dataset.",
              f"- F-C `{figures['slots']['png']}` (`{figures['slots']['csv']}`): change in NDCG@10 and in the share of users with >= 1 new-item hit "
              "for r = 9 and r = 8 against r = 10 per repeat module with 95 % CIs, one column per dataset.", ""]
    # Write the timing and memory table.
    lines += ["## Timing and memory", ""]
    rows = []
    for d, e in entries.items():
        t = e["timing_seconds"]
        rows.append([NAMES[d], f"{t['eval_load']:.1f} s", f"{e['eval']['novelty']['seconds']:.1f} s", f"{t['eval_explore_list']:.1f} s",
                     f"{t['eval_repeat_lists']:.1f} s", f"{e['eval']['slots']['timing_seconds']['lists_and_metrics']:.1f} s",
                     f"{e['eval']['slots']['timing_seconds']['bootstrap']:.1f} s", f"{e['eval']['rank_hits']['seconds']:.1f} s",
                     f"{t['fit_load']:.1f} s", f"{e['fit']['novelty']['seconds']:.1f} s",
                     f"{sum(e['fit']['slot_curves']['timing_seconds'].values()):.1f} s", f"{e['peak_memory_mib']:.0f}" if e["peak_memory_mib"] else "-"])
    lines += _table(["dataset", "eval load", "A eval", "E2 list", "repeat lists", "C lists + metrics", "C bootstrap", "B", "fit load", "A fit",
                     "C fit curves", "peak working set (MiB)"], rows)
    # Join the lines into one text.
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def input_digests(config: dict[str, Any], dataset: str) -> dict[str, str]:
    '''sha256 of every INPUT_FILES parquet file of ``dataset`` in the data folder of ``config``.'''
    # Refuse a data folder with a confirmatory path component.
    folder = guard_path(dataset_dir(config, dataset))
    # Hash every input file.
    return {name: sha256_of(folder / f"{name}.parquet") for name in INPUT_FILES}


def main(argv: list[str] | None = None) -> Path:
    '''Run the study for the arguments ``argv`` and return the path of the JSON file.

    Runs the setup check and the analysis and writes the JSON, the markdown and the figures.
    '''
    # Parse the arguments and reject a profile outside ALLOWED_PROFILES.
    args = parse_args(argv)
    if args.profile not in ALLOWED_PROFILES:
        raise ValueError(f"profile must be one of {ALLOWED_PROFILES}")
    # Start the run timer and load the profile configuration.
    t_start = time.perf_counter()
    config = load_config(args.profile)
    # Refuse a profile with role 'final'.
    if config.get("role") == "final":
        raise PermissionError("posthoc_explore_slot: profiles with role 'final' are never analysed")
    # Point the model inputs at --model-root and create the output folder.
    models = model_config(config, args.model_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    # Run the setup check when a reference folder is given and print its result.
    check = None
    if args.reference_tuning_dir:
        check = setup_check(models, args.datasets, Path(args.reference_tuning_dir))
        for dataset, result in check["datasets"].items():
            print(f"[{dataset}] setup check passed: selections equal the reference; user-CF grid without k = 10 equal: "
                  f"{result['usercf_grid_excluding_k10_equal']}; LR artifact sha256 equal: {result['lr_artifact_sha256']['equal']}")
    # Analyse every dataset and print its headline numbers.
    entries: dict[str, dict[str, Any]] = {}
    digests = {}
    for dataset in args.datasets:
        # Hash the input files and run the analysis.
        print(f"[{dataset}] analysing (profile {args.profile})")
        digests[dataset] = input_digests(config, dataset)
        entries[dataset] = analyse_dataset(config, models, dataset, args.bootstrap_n, args.seed)
        # Print the part A target rates, the R2 + E2 NDCG@10 by r and the selected r per module.
        entry = entries[dataset]
        target = entry["eval"]["novelty"]["basket_sets"]["target"]["basket_weighted"]
        table = {(row["module"], row["r"]): row for row in entry["eval"]["slots"]["table"]}
        print(f"[{dataset}] A eval target: mean new_share {target['mean_new_share']:.3f}, P(any_new) {target['p_any_new']:.3f}, "
              f"mean n_new {target['mean_n_new']:.2f}; C R2+E2 NDCG@10 r=10 {table[('R2', 10)]['ndcg']:.4f}, r=9 {table[('R2', 9)]['ndcg']:.4f}, "
              f"r=8 {table[('R2', 8)]['ndcg']:.4f}; fit r* per module "
              + ", ".join(f"{c['module']}: {c['r_star']}" for c in entry["fit"]["slot_curves"]["curves"]))
        # Print every reproduction check.
        for key, item in entry["reproduction_check"].items():
            if isinstance(item, dict):
                print(f"[{dataset}] reproduction {key}: expected {item['expected']:.4f} observed {item['observed']:.4f} "
                      f"{'matched' if item['matched'] else 'NOT matched'}")
    # Draw the three figures.
    figures = {"new_rate": figure_new_rate(entries, out_dir), "rank_hits": figure_rank_hits(entries, out_dir),
               "slots": figure_slot_effects(entries, out_dir)}
    # Assemble the JSON document.
    payload = {
        "label": LABEL,
        "task": TASK,
        "profile": args.profile,
        "parts": ["setup check", "A", "B", "C", "D"],
        "definitions": {
            "prior": "items of the user's earlier baskets of the same split (history + target), i.e. baskets with a smaller basket_seq",
            "new_share": "|b minus prior(b)| / |b| over distinct items", "any_new": "1 if |b minus prior(b)| >= 1", "n_new": "|b minus prior(b)|",
            "basket_weighted": "mean over baskets", "user_weighted": "mean over users of the per-user mean",
            "position": "basket_seq of the basket", "history_length": "distinct history baskets of the user in the split (nbr.pipeline.history_lengths)",
            "min_items": MIN_ITEMS, "k": K, "r_values": list(R_VALUES), "r1_gammas": list(R1_GAMMAS),
            "repeat_ranks": REPEAT_RANKS, "explore_ranks": EXPLORE_RANKS,
        },
        "position_bins": [name for name, _, _ in POSITION_BINS],
        "history_bins": {dataset: entries[dataset]["history_bins"] for dataset in entries},
        "bootstrap": {"n": args.bootstrap_n, "seed": args.seed, "level": 0.95},
        "setup_check": check,
        "inputs_sha256": digests,
        "data_dir": str(config["data_dir"]),
        "model_root": str(models["repo_root"]),
        "datasets": entries,
        "figures": figures,
        "environment": {"python": f"Python {platform.python_version()} ({platform.system()})", "pandas": pd.__version__,
                        "numpy": np.__version__, "matplotlib": matplotlib.__version__},
        "runtime_seconds": time.perf_counter() - t_start,
        "peak_memory_mib": peak_memory_mib(),
    }
    # Write the JSON file and read it back.
    json_path = write_json(out_dir / f"explore_slot_{args.profile}.json", payload, config)
    document = json.loads(json_path.read_text(encoding="utf-8"))   # the markdown reflects exactly what is on disk
    # Render the markdown from the read-back document and write it.
    md_path = out_dir / f"explore_slot_{args.profile}.md"
    md_path.write_text(render_markdown(document), encoding="utf-8")
    # Print the written files, the runtime and the peak memory.
    print(f"wrote {json_path}, {md_path} and {', '.join(f['png'] for f in figures.values())}; runtime {payload['runtime_seconds']:.1f} s; "
          f"peak working set {payload['peak_memory_mib'] or 0:.0f} MiB")
    return json_path


# ---------------------------------------------------------------------------
# command-line entry
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    main()
