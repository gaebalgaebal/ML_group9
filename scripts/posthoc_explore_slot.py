"""Post-hoc case study: new-item purchase rates and the value of an exploration slot (W15; post-hoc, development data, not confirmatory).

Usage:
    python scripts/posthoc_explore_slot.py [--profile medium] [--datasets instacart,complete_journey]
                                           [--out-dir reports/posthoc] [--reference-tuning-dir PATH]

Setup check (--reference-tuning-dir): the tuning document of this repository (reports/metrics/<profile>_<dataset>_tuning.json,
written by ``python -m nbr tune``) is compared with the reference document of the same name in PATH (the W8b medium run).
The selections (gamma, gamma_c, alpha, r, LR C, user-CF k and weighting, the re-selected slots) must be equal; the user-CF grid
rows other than k = 10 (added to the grid after W8b), the stage grids and the LR validation rows are compared as well and
reported. Any differing selection stops the script before the analysis. The LR artifact of this run is checked against the
sha256 recorded in the tuning document.

Part A (descriptive): for every user u and every basket b from the user's 2nd basket on (history baskets, then the target
basket of the split), prior(b) = items of u's earlier baskets of the same split (history + target, smaller basket_seq);
new_share(b) = |b minus prior(b)| / |b| over distinct items, any_new(b) = 1 if |b minus prior(b)| >= 1, n_new(b) = |b minus prior(b)|.
Basket-weighted means (over baskets) and user-weighted means (mean of the per-user means) of new_share, any_new and n_new are
reported overall, by basket position (= basket_seq; bins 2-4, 5-10, 11-20, 21-50, > 50) and by the T5 history-length bins
(nbr.pipeline.HISTORY_BINS on the number of distinct history baskets of the split), separately for the history baskets, the
history baskets with >= 3 distinct items (the Complete Journey target is the first basket with >= 3 items after the cutoff
while its history baskets carry no size filter) and the target basket. The eval split is the main split; the fit split is the
robustness check.

Outputs in --out-dir: explore_slot_<profile>.json (all numbers; meta with the code commit and the sha256 of every input file)
and explore_slot_<profile>.md (tables). Only the development profiles ``sample`` and ``medium`` are accepted; no file under
reports/final* or data/processed/holdout* is read.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from nbr.config import load_config, resolve_tuning_dir  # noqa: E402
from nbr.data import dataset_dir, load_split, split_target  # noqa: E402
from nbr.io import write_json  # noqa: E402
from nbr.pipeline import HISTORY_BINS, history_lengths  # noqa: E402
from nbr.usercf_eval import peak_memory_mib  # noqa: E402

LABEL = "post-hoc, development data, not confirmatory"
TASK = "W15 post-hoc case study: new-item purchase rates and the value of an exploration slot"
ALLOWED_PROFILES = ("sample", "medium")
DATASETS = ("instacart", "complete_journey")
NAMES = {"instacart": "Instacart", "complete_journey": "Complete Journey"}
SPLITS = ("eval", "fit")
INPUT_FILES = ("fit_history", "fit_target", "eval_history", "eval_target")
MIN_ITEMS = 3                                   # Complete Journey target rule: first basket with >= 3 items after the cutoff
POSITION_BINS = [("2-4", 2, 4), ("5-10", 5, 10), ("11-20", 11, 20), ("21-50", 21, 50), (">50", 51, math.inf)]
METRICS = {"mean_new_share": "new_share", "p_any_new": "any_new", "mean_n_new": "n_new", "mean_size": "size"}
BASKET_SETS = {
    "history_all": "history baskets (2nd basket on)",
    "history_min3": f"history baskets with >= {MIN_ITEMS} distinct items (2nd basket on)",
    "target": "target basket",
}
COLUMNS = ["user_id", "basket_id", "basket_seq", "item_id"]
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


# ---------------------------------------------------------------------------
# arguments and small helpers
# ---------------------------------------------------------------------------
def parse_datasets(value: str) -> list[str]:
    """Comma-separated dataset names -> list in the given order (blanks and repeats dropped); unknown names raise."""
    names: list[str] = []
    for item in value.split(","):
        item = item.strip()
        if item and item not in names:
            names.append(item)
    unknown = [name for name in names if name not in DATASETS]
    if unknown or not names:
        raise ValueError(f"datasets must be a non-empty subset of {list(DATASETS)}, got {value!r}")
    return names


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Command line of the script (``argv`` defaults to sys.argv); ``datasets`` is returned as a list."""
    parser = argparse.ArgumentParser(description="Post-hoc new-item purchase rates (development profiles only).")
    parser.add_argument("--profile", default="medium", choices=ALLOWED_PROFILES)
    parser.add_argument("--datasets", default=",".join(DATASETS), type=parse_datasets,
                        help="comma-separated subset of instacart,complete_journey (default: both)")
    parser.add_argument("--out-dir", default=str(ROOT / "reports" / "posthoc"))
    parser.add_argument("--reference-tuning-dir", default=None,
                        help="folder with the reference <profile>_<dataset>_tuning.json files (read-only); "
                             "selections of this repository's tuning documents must equal them")
    return parser.parse_args(argv)


def guard_path(path: Path) -> Path:
    """Refuse any path with a component starting with 'final' or 'holdout' (confirmatory inputs are never read)."""
    for part in path.resolve().parts:
        if part.lower().startswith(("final", "holdout")):
            raise PermissionError(f"refusing to read {path}: confirmatory inputs are never read by this script")
    return path


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def bin_labels(edges: list[int]) -> list[tuple[str, float, float]]:
    """(label, lo, hi] bins of nbr.pipeline (upper edges, last bin open): (0,4], (4,6], ..., (20,inf)."""
    bounds = [0, *edges, math.inf]
    return [(f"({lo},inf)" if math.isinf(hi) else f"({lo},{hi}]", lo, hi) for lo, hi in zip(bounds[:-1], bounds[1:])]


# ---------------------------------------------------------------------------
# per-basket novelty
# ---------------------------------------------------------------------------
def basket_novelty(history: pd.DataFrame, target: pd.DataFrame) -> pd.DataFrame:
    """One row per basket from each user's 2nd basket on: user_id, basket_id, position, is_target, size, n_new, new_share, any_new.

    ``history`` and ``target`` are the frames of one split (columns user_id, basket_id, basket_seq, item_id, ...).
    Items are counted once per basket; an item of basket b is new when no basket of the same user with a smaller
    basket_seq (history and target together) contains it; position = basket_seq. The user's first basket (smallest
    basket_seq) is dropped. Raises when a basket_id carries several basket_seq values or a target basket does not
    come after every history basket of its user.
    """
    frame = pd.concat([history[COLUMNS].assign(is_target=False), target[COLUMNS].assign(is_target=True)], ignore_index=True)
    frame = frame.drop_duplicates(["user_id", "basket_id", "item_id"])
    baskets = frame[["user_id", "basket_id", "basket_seq", "is_target"]].drop_duplicates()
    if baskets.duplicated(["user_id", "basket_id"]).any():
        raise ValueError("a basket_id carries several basket_seq / is_target values within one user")
    if baskets.duplicated(["user_id", "basket_seq"]).any():
        raise ValueError("several basket_id values share one basket_seq within one user")
    max_history = baskets[~baskets["is_target"]].groupby("user_id")["basket_seq"].max()
    targets = baskets[baskets["is_target"]]
    behind = targets["basket_seq"].to_numpy() <= max_history.reindex(targets["user_id"]).fillna(-np.inf).to_numpy()
    if behind.any():
        raise ValueError(f"{int(behind.sum())} target baskets do not come after every history basket of their user")

    user_code, _ = pd.factorize(frame["user_id"])
    item_code, items = pd.factorize(frame["item_id"])
    pair = user_code.astype(np.int64) * len(items) + item_code.astype(np.int64)
    first_seq = frame["basket_seq"].groupby(pair).transform("min")
    frame = frame.assign(new=(frame["basket_seq"].to_numpy() == first_seq.to_numpy()))
    out = (frame.groupby(["user_id", "basket_id", "basket_seq", "is_target"], sort=False)
           .agg(size=("item_id", "size"), n_new=("new", "sum")).reset_index())
    first_basket = out.groupby("user_id")["basket_seq"].transform("min")
    out = out[out["basket_seq"] > first_basket].copy()
    out = out.rename(columns={"basket_seq": "position"})
    out["position"] = out["position"].astype("int64")
    out["size"] = out["size"].astype("int64")
    out["n_new"] = out["n_new"].astype("int64")
    out["new_share"] = out["n_new"] / out["size"]
    out["any_new"] = (out["n_new"] >= 1).astype("float64")
    return out.sort_values(["user_id", "position"]).reset_index(drop=True)


# ---------------------------------------------------------------------------
# aggregation
# ---------------------------------------------------------------------------
def summarise(baskets: pd.DataFrame) -> dict[str, Any]:
    """Counts plus basket-weighted and user-weighted (mean of per-user means) means of new_share, any_new, n_new, size."""
    if baskets.empty:
        return {"n_baskets": 0, "n_users": 0, "basket_weighted": None, "user_weighted": None}
    columns = list(METRICS.values())
    user_means = baskets.groupby("user_id")[columns].mean()
    return {
        "n_baskets": int(len(baskets)),
        "n_users": int(len(user_means)),
        "basket_weighted": {key: float(baskets[column].mean()) for key, column in METRICS.items()},
        "user_weighted": {key: float(user_means[column].mean()) for key, column in METRICS.items()},
    }


def n_new_distribution(baskets: pd.DataFrame) -> dict[str, float] | None:
    """Basket-weighted shares of baskets with n_new = 0, 1, 2 and >= 3."""
    if baskets.empty:
        return None
    n_new = baskets["n_new"]
    return {"0": float((n_new == 0).mean()), "1": float((n_new == 1).mean()), "2": float((n_new == 2).mean()),
            "3+": float((n_new >= 3).mean())}


def by_position(baskets: pd.DataFrame) -> list[dict[str, Any]]:
    """summarise per POSITION_BINS of the basket position."""
    rows = []
    for name, lo, hi in POSITION_BINS:
        subset = baskets[(baskets["position"] >= lo) & (baskets["position"] <= hi)]
        rows.append({"bin": name, **summarise(subset)})
    return rows


def by_history_length(baskets: pd.DataFrame, lengths: pd.Series, dataset: str) -> list[dict[str, Any]]:
    """summarise per T5 history-length bin (HISTORY_BINS[dataset]) of the basket's user; ``lengths`` maps user_id -> history baskets."""
    user_length = baskets["user_id"].map(lengths)
    if user_length.isna().any():
        raise ValueError("a basket belongs to a user without history baskets")
    rows = []
    for name, lo, hi in bin_labels(HISTORY_BINS[dataset]):
        subset = baskets[(user_length > lo) & (user_length <= hi)]
        rows.append({"bin": name, **summarise(subset)})
    return rows


def basket_set_block(baskets: pd.DataFrame, lengths: pd.Series, dataset: str, description: str) -> dict[str, Any]:
    return {"description": description, **summarise(baskets), "n_new_distribution": n_new_distribution(baskets),
            "by_position": by_position(baskets), "by_history_length": by_history_length(baskets, lengths, dataset)}


def analyse_split(history: pd.DataFrame, target: pd.DataFrame, dataset: str) -> dict[str, Any]:
    """Part A for one split: data checks and the three basket-set blocks (history_all, history_min3, target)."""
    t0 = time.perf_counter()
    baskets = basket_novelty(history, target)
    lengths = history_lengths(history)
    history_baskets = baskets[~baskets["is_target"]]
    target_baskets = baskets[baskets["is_target"]]
    sets = {
        "history_all": history_baskets,
        "history_min3": history_baskets[history_baskets["size"] >= MIN_ITEMS],
        "target": target_baskets,
    }
    blocks = {name: basket_set_block(frame, lengths, dataset, BASKET_SETS[name]) for name, frame in sets.items()}

    # data checks: basket_seq structure, the Complete Journey gap before the target, agreement with nbr.data.split_target
    all_baskets = pd.concat([history[["user_id", "basket_seq"]], target[["user_id", "basket_seq"]]]).drop_duplicates()
    seq = history[["user_id", "basket_seq"]].drop_duplicates().groupby("user_id")["basket_seq"]
    starts_at_one = float((seq.min() == 1).mean())
    consecutive = float(((seq.max() - seq.min() + 1) == seq.size()).mean())
    target_seq = target[["user_id", "basket_seq"]].drop_duplicates().groupby("user_id")["basket_seq"].min()
    gap = target_seq - seq.max().reindex(target_seq.index)
    _, new_target = split_target(history, target)
    any_new_users = int(target_baskets.loc[target_baskets["any_new"] > 0, "user_id"].nunique())
    history_sizes = history.drop_duplicates(["user_id", "basket_id", "item_id"]).groupby(["user_id", "basket_id"]).size()
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
    return {"checks": checks, "history_length_summary": {"min": int(lengths.min()), "median": float(lengths.median()),
                                                           "max": int(lengths.max())},
            "basket_sets": blocks, "timing_seconds": time.perf_counter() - t0}


def analyse_dataset(config: dict[str, Any], dataset: str) -> dict[str, Any]:
    """Part A on the eval split (main) and the fit split (robustness check) of ``dataset``."""
    entry: dict[str, Any] = {"label": LABEL, "history_bins": [name for name, _, _ in bin_labels(HISTORY_BINS[dataset])]}
    for split in SPLITS:
        t0 = time.perf_counter()
        history, target = load_split(config, dataset, split)
        load_seconds = time.perf_counter() - t0
        entry[split] = analyse_split(history, target, dataset)
        entry[split]["load_seconds"] = load_seconds
        entry[split]["role"] = "main" if split == "eval" else "robustness check"
    entry["peak_memory_mib"] = peak_memory_mib()
    return entry


# ---------------------------------------------------------------------------
# setup check: tuning documents against the reference
# ---------------------------------------------------------------------------
def _get(document: dict[str, Any], path: tuple[str, ...]) -> Any:
    value: Any = document
    for key in path:
        if not isinstance(value, dict) or key not in value:
            return None
        value = value[key]
    return value


def _equal(a: Any, b: Any) -> bool:
    """Numbers within TOLERANCE (10 == 10.0), everything else by ==; a bool never equals a number."""
    if isinstance(a, bool) != isinstance(b, bool):
        return False
    if isinstance(a, (int, float)) and isinstance(b, (int, float)) and not isinstance(a, bool):
        return math.isclose(float(a), float(b), rel_tol=0.0, abs_tol=TOLERANCE)
    return a == b


def _rows_equal(run_rows: list[dict[str, Any]], ref_rows: list[dict[str, Any]], keys: tuple[str, ...], value: str) -> bool:
    """Same key tuples in the same order and ``value`` equal within TOLERANCE (None-safe)."""
    if run_rows is None or ref_rows is None or len(run_rows) != len(ref_rows):
        return False
    for a, b in zip(run_rows, ref_rows):
        if any(not _equal(a.get(key), b.get(key)) for key in keys) or not _equal(a.get(value), b.get(value)):
            return False
    return True


def compare_tuning(run: dict[str, Any], reference: dict[str, Any]) -> dict[str, Any]:
    """Selections of ``run`` against ``reference``; grids compared without the user-CF rows of k = NEW_K."""
    selections = []
    for name, path in SELECTION_PATHS.items():
        a, b = _get(run, path), _get(reference, path)
        selections.append({"quantity": name, "run": a, "reference": b, "equal": _equal(a, b)})
    run_grid = _get(run, ("integration", "usercf", "grid")) or []
    ref_grid = _get(reference, ("integration", "usercf", "grid")) or []
    run_other = [row for row in run_grid if int(row["k"]) != NEW_K]
    ref_other = [row for row in ref_grid if int(row["k"]) != NEW_K]
    stage_equal = {
        "stage1_repeat_recall": _rows_equal(run.get("stage1_repeat_recall"), reference.get("stage1_repeat_recall"), ("gamma",), "recall"),
        "stage2_explore_new_recall": _rows_equal(run.get("stage2_explore_new_recall"), reference.get("stage2_explore_new_recall"),
                                                 ("gamma_c", "alpha"), "recall"),
        "stage3_final_ndcg": _rows_equal(run.get("stage3_final_ndcg"), reference.get("stage3_final_ndcg"), ("r",), "ndcg"),
    }
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
    recorded = _get(run, ("lr_repeat", "artifact"))
    expected = _get(run, ("lr_repeat", "sha256"))
    path = Path(recorded) if recorded else None
    exists = bool(path and path.exists())
    observed = sha256_of(path) if exists else None
    return {"artifact": recorded, "exists": exists, "sha256_recorded": expected, "sha256_observed": observed,
            "matches": exists and observed == expected}


def setup_check(config: dict[str, Any], datasets: list[str], reference_dir: Path) -> dict[str, Any]:
    """compare_tuning for every dataset; raises SystemExit when a selection differs from the reference."""
    reference_dir = guard_path(Path(reference_dir))
    tuning_dir = guard_path(resolve_tuning_dir(config))
    out: dict[str, Any] = {"tuning_dir": str(tuning_dir), "reference_dir": str(reference_dir), "datasets": {}}
    problems = []
    for dataset in datasets:
        name = f"{config['profile']}_{dataset}_tuning.json"
        run_path, ref_path = tuning_dir / name, reference_dir / name
        for path in (run_path, ref_path):
            if not path.exists():
                raise FileNotFoundError(f"tuning document missing: {path}")
        run = json.loads(run_path.read_text(encoding="utf-8"))
        reference = json.loads(ref_path.read_text(encoding="utf-8"))
        result = compare_tuning(run, reference)
        result["lr_artifact"] = check_lr_artifact(run)
        result["files"] = {"run": str(run_path), "reference": str(ref_path), "reference_sha256": sha256_of(ref_path)}
        out["datasets"][dataset] = result
        for row in result["selections"]:
            if not row["equal"]:
                problems.append(f"{dataset}: {row['quantity']} run={row['run']!r} reference={row['reference']!r}")
    out["passed"] = not problems
    if problems:
        raise SystemExit("setup check failed; selections differ from the reference tuning documents:\n  " + "\n  ".join(problems))
    return out


# ---------------------------------------------------------------------------
# markdown
# ---------------------------------------------------------------------------
def _f(value: float | None, digits: int = 3) -> str:
    return "-" if value is None else f"{value:.{digits}f}"


def _pair(block: dict[str, Any], key: str, digits: int = 3) -> str:
    """basket-weighted / user-weighted cell."""
    if block["basket_weighted"] is None:
        return "-"
    return f"{_f(block['basket_weighted'][key], digits)} / {_f(block['user_weighted'][key], digits)}"


def _table(header: list[str], rows: list[list[str]]) -> list[str]:
    lines = ["| " + " | ".join(header) + " |", "| " + " | ".join("---" for _ in header) + " |"]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return lines + [""]


def _bins_table(blocks: dict[str, dict[str, Any]], key: str, bin_header: str) -> list[str]:
    """Rows = bins of the first set, columns = baskets / users / new_share / P(any_new) / n_new / size per basket set."""
    lines = []
    for set_name, block in blocks.items():
        rows = []
        for row in block[key]:
            cells = [row["bin"], f"{row['n_baskets']:,}", f"{row['n_users']:,}"]
            if set_name == "target":
                bw = row["basket_weighted"]
                cells += [_f(bw and bw["mean_new_share"]), _f(bw and bw["p_any_new"]), _f(bw and bw["mean_n_new"], 2),
                          _f(bw and bw["mean_size"], 1)]
            else:
                cells += [_pair(row, "mean_new_share"), _pair(row, "p_any_new"), _pair(row, "mean_n_new", 2), _pair(row, "mean_size", 1)]
            rows.append(cells)
        weighting = "" if set_name == "target" else " (basket-weighted / user-weighted)"
        lines += [f"{block['description'].capitalize()}{weighting}:", ""]
        lines += _table([bin_header, "baskets", "users", "mean new_share", "P(any_new)", "mean n_new", "mean size"], rows)
    return lines


def render_split(entry: dict[str, Any], dataset: str, split: str) -> list[str]:
    block = entry[split]
    checks, sets = block["checks"], block["basket_sets"]
    lines = [f"### {NAMES[dataset]}, {split} split ({block['role']})", ""]
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
    lines += ["Distribution of n_new (basket-weighted shares):", ""]
    rows = [[s["description"]] + ([_f(s["n_new_distribution"][k]) for k in ("0", "1", "2", "3+")] if s["n_new_distribution"] else ["-"] * 4)
            for s in sets.values()]
    lines += _table(["basket set", "n_new = 0", "n_new = 1", "n_new = 2", "n_new >= 3"], rows)
    lines += ["By basket position (position = basket_seq):", ""]
    lines += _bins_table(sets, "by_position", "position")
    lines += [f"By T5 history-length bin (distinct {split}-history baskets of the user):", ""]
    lines += _bins_table(sets, "by_history_length", "history length")
    return lines


def render_setup_check(check: dict[str, Any] | None) -> list[str]:
    lines = ["## Setup check: tuning documents of this run against the W8b reference", ""]
    if check is None:
        return lines + ["(no --reference-tuning-dir given: not checked)", ""]
    lines += [f"- This run: `{check['tuning_dir']}`; reference: `{check['reference_dir']}` (read only). Result: "
              f"{'passed' if check['passed'] else 'FAILED'} (every selection equal).", ""]
    for dataset, result in check["datasets"].items():
        lines += [f"{NAMES[dataset]} (run commit `{result['run_meta']['git_commit']}`, reference commit `{result['reference_meta']['git_commit']}`, "
                  f"reference time {result['reference_meta']['utc_time']}):", ""]
        rows = [[row["quantity"], f"{row['run']}", f"{row['reference']}", "yes" if row["equal"] else "NO"] for row in result["selections"]]
        rows.append(["user-CF grid rows other than k = 10 (new_recall@10)", "", "", "yes" if result["usercf_grid_excluding_k10_equal"] else "NO"])
        for name, equal in result["stage_grids_equal"].items():
            rows.append([f"{name} grid", "", "", "yes" if equal else "NO"])
        rows.append(["LR validation rows (val_recall by C)", "", "", "yes" if result["lr_validation_equal"] else "NO"])
        sha = result["lr_artifact_sha256"]
        rows.append(["LR artifact sha256", (sha["run"] or "")[:12], (sha["reference"] or "")[:12], "yes" if sha["equal"] else "no"])
        art = result["lr_artifact"]
        rows.append(["LR artifact file exists with the recorded sha256", "yes" if art["matches"] else "NO", "", ""])
        lines += _table(["quantity", "this run", "reference", "equal"], rows)
        k10 = result["usercf_k10_rows_run"]
        if k10:
            lines += ["k = 10 rows of this run (absent from the reference grid): "
                      + "; ".join(f"{row['weighting']}: new_recall@10 = {row['new_recall@10']:.4f}" for row in k10), ""]
    return lines


def render_markdown(payload: dict[str, Any]) -> str:
    """Tables of the JSON document ``payload`` (as written by main)."""
    meta, entries, profile = payload["meta"], payload["datasets"], payload["profile"]
    lines = [f"# New-item purchase rates and the value of an exploration slot ({profile})", "",
             f"**{LABEL}.** For every user and every basket from the 2nd basket on, prior(b) = items of the user's earlier baskets of the "
             "same split (history + target); new_share(b) = |b minus prior(b)| / |b| over distinct items, any_new(b) = 1 if at least one item is "
             "new, n_new(b) = number of new items. Basket-weighted = mean over baskets; user-weighted = mean of the per-user means. "
             "Position = basket_seq (the ordinal number of the basket in the user's sequence). History-length bins are the T5 bins of "
             "nbr.pipeline.HISTORY_BINS on the number of distinct history baskets of the split. The target rows describe the basket the "
             "recommendation lists are scored against: P(any_new) is the share of users for whom an exploration slot can hit at all, mean n_new "
             "the number of new items it could hit. Descriptive only; no confirmatory claim is made.", "",
             f"- Code commit `{meta['git_commit']}`, {meta['utc_time']}; profile `{profile}`; splits: eval (main), fit (robustness check).",
             f"- Position bins {', '.join(name for name, _, _ in POSITION_BINS)}; history-length bins "
             + "; ".join(f"{NAMES[d]} {', '.join(e['history_bins'])}" for d, e in entries.items()) + ".",
             f"- Complete Journey note: its target is the first basket with >= {MIN_ITEMS} items after the cutoff while history baskets have no size "
             f"filter, so the history rows are also given for baskets with >= {MIN_ITEMS} distinct items; baskets between the last history basket and "
             "the target that are in neither frame are not part of prior(b), as in the pipeline.",
             f"- Inputs (sha256 in the JSON): {', '.join(f'{d}: {len(payload['inputs_sha256'][d])} files' for d in entries)}.",
             f"- Runtime {payload['runtime_seconds']:.1f} s; peak working set {payload['peak_memory_mib']:.0f} MiB (in-process).", ""]
    lines += render_setup_check(payload["setup_check"])
    lines += ["## A. New-item purchase rates", "", "Headline (eval split; bw / uw = basket-weighted / user-weighted):", ""]
    rows = []
    for d, e in entries.items():
        sets = e["eval"]["basket_sets"]
        t, h = sets["target"]["basket_weighted"], sets["history_all"]
        rows.append([NAMES[d], f"{sets['target']['n_users']:,}", _f(t and t["mean_new_share"]), _f(t and t["p_any_new"]),
                     _f(t and t["mean_n_new"], 2), _pair(h, "mean_new_share"), _pair(h, "p_any_new"), _pair(h, "mean_n_new", 2)])
    lines += _table(["dataset", "users", "target mean new_share", "target P(any_new)", "target mean n_new",
                     "history new_share (bw / uw)", "history P(any_new) (bw / uw)", "history n_new (bw / uw)"], rows)
    for split in SPLITS:
        lines += [f"## {'Eval split (main)' if split == 'eval' else 'Fit split (robustness check)'}", ""]
        for dataset, entry in entries.items():
            lines += render_split(entry, dataset, split)
    lines += ["## Timing and memory", ""]
    lines += _table(["dataset", "eval load", "eval analysis", "fit load", "fit analysis", "peak working set (MiB)"],
                    [[NAMES[d], f"{e['eval']['load_seconds']:.1f} s", f"{e['eval']['timing_seconds']:.1f} s", f"{e['fit']['load_seconds']:.1f} s",
                      f"{e['fit']['timing_seconds']:.1f} s", f"{e['peak_memory_mib']:.0f}" if e["peak_memory_mib"] else "-"] for d, e in entries.items()])
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def input_digests(config: dict[str, Any], dataset: str) -> dict[str, str]:
    folder = guard_path(dataset_dir(config, dataset))
    return {name: sha256_of(folder / f"{name}.parquet") for name in INPUT_FILES}


def main(argv: list[str] | None = None) -> Path:
    args = parse_args(argv)
    if args.profile not in ALLOWED_PROFILES:
        raise ValueError(f"profile must be one of {ALLOWED_PROFILES}")
    t_start = time.perf_counter()
    config = load_config(args.profile)
    if config.get("role") == "final":
        raise PermissionError("posthoc_explore_slot: profiles with role 'final' are never analysed")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    check = None
    if args.reference_tuning_dir:
        check = setup_check(config, args.datasets, Path(args.reference_tuning_dir))
        for dataset, result in check["datasets"].items():
            print(f"[{dataset}] setup check passed: selections equal the reference; user-CF grid without k = 10 equal: "
                  f"{result['usercf_grid_excluding_k10_equal']}; LR artifact sha256 equal: {result['lr_artifact_sha256']['equal']}")
    entries: dict[str, dict[str, Any]] = {}
    digests = {}
    for dataset in args.datasets:
        print(f"[{dataset}] analysing (profile {args.profile})")
        digests[dataset] = input_digests(config, dataset)
        entries[dataset] = analyse_dataset(config, dataset)
        target = entries[dataset]["eval"]["basket_sets"]["target"]["basket_weighted"]
        hist = entries[dataset]["eval"]["basket_sets"]["history_all"]
        print(f"[{dataset}] eval target: mean new_share {target['mean_new_share']:.3f}, P(any_new) {target['p_any_new']:.3f}, "
              f"mean n_new {target['mean_n_new']:.2f}; history baskets: new_share {hist['basket_weighted']['mean_new_share']:.3f} "
              f"(basket-weighted) / {hist['user_weighted']['mean_new_share']:.3f} (user-weighted)")
    payload = {
        "label": LABEL,
        "task": TASK,
        "profile": args.profile,
        "parts": ["setup check", "A"],
        "definitions": {
            "prior": "items of the user's earlier baskets of the same split (history + target), i.e. baskets with a smaller basket_seq",
            "new_share": "|b minus prior(b)| / |b| over distinct items", "any_new": "1 if |b minus prior(b)| >= 1", "n_new": "|b minus prior(b)|",
            "basket_weighted": "mean over baskets", "user_weighted": "mean over users of the per-user mean",
            "position": "basket_seq of the basket", "history_length": "distinct history baskets of the user in the split (nbr.pipeline.history_lengths)",
            "min_items": MIN_ITEMS,
        },
        "position_bins": [name for name, _, _ in POSITION_BINS],
        "history_bins": {dataset: entries[dataset]["history_bins"] for dataset in entries},
        "setup_check": check,
        "inputs_sha256": digests,
        "data_dir": str(config["data_dir"]),
        "datasets": entries,
        "environment": {"python": f"Python {platform.python_version()} ({platform.system()})", "pandas": pd.__version__,
                        "numpy": np.__version__},
        "runtime_seconds": time.perf_counter() - t_start,
        "peak_memory_mib": peak_memory_mib(),
    }
    json_path = write_json(out_dir / f"explore_slot_{args.profile}.json", payload, config)
    document = json.loads(json_path.read_text(encoding="utf-8"))   # the markdown reflects exactly what is on disk
    md_path = out_dir / f"explore_slot_{args.profile}.md"
    md_path.write_text(render_markdown(document), encoding="utf-8")
    print(f"wrote {json_path}, {md_path}; runtime {payload['runtime_seconds']:.1f} s; "
          f"peak working set {payload['peak_memory_mib'] or 0:.0f} MiB")
    return json_path


if __name__ == "__main__":
    main()
