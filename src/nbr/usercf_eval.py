"""Stand-alone development-only user-CF fit selection and eval-split H8 evaluation (W7).

Module screening entry point (stand-alone; ``python -m nbr tune`` reuses tune_history):

    python -m nbr.usercf_eval --profile {sample,medium} --dataset D --stage {tune,evaluate,all} [--tuning-dir DIR]

The base selections (gamma, gamma_c, alpha, r) of evaluate are read from
<tuning-dir>/<profile>_<dataset>_tuning.json (default: NBR_TUNING_DIR or reports/metrics of this
repository, nbr.config.resolve_tuning_dir); holdout profiles and paths are refused.
"""

from __future__ import annotations

import argparse
import ctypes
import json
import math
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd

from nbr.baselines import item_popularity, popularity_list
from nbr.combine import combine
from nbr.config import load_config, resolve_tuning_dir
from nbr.data import attach_levels, dataset_dir, load_items, load_split, split_target
from nbr.explore import explore_list
from nbr.io import write_json
from nbr.itemcf import cf_list, item_neighbors
from nbr.metrics import paired_bootstrap, per_user_ndcg, per_user_recall
from nbr.repeat import repeat_list
from nbr.usercf import _list_from_vectors, user_neighbors, user_vectors

K = 10
DATASETS = ("instacart", "complete_journey")
HISTORY_BINS = {"instacart": [4, 6, 10, 20], "complete_journey": [20, 50, 100]}
TIMING_NOTE = "Other jobs, including W6, may run on the same PC; timings are not isolated benchmarks."


def peak_memory_mib() -> float | None:
    """Process-lifetime peak resident memory, not a per-operation allocation.

    Windows GetProcessMemoryInfo(handle, counters, size) reads OS process
    counters; PeakWorkingSetSize is the maximum resident working set in bytes.
    """
    if os.name != "nt":
        return None

    class Counters(ctypes.Structure):
        _fields_ = [("cb", ctypes.c_ulong), ("PageFaultCount", ctypes.c_ulong)] + [
            (name, ctypes.c_size_t) for name in (
                "PeakWorkingSetSize", "WorkingSetSize", "QuotaPeakPagedPoolUsage",
                "QuotaPagedPoolUsage", "QuotaPeakNonPagedPoolUsage", "QuotaNonPagedPoolUsage",
                "PagefileUsage", "PeakPagefileUsage",
            )
        ]

    counters = Counters()
    counters.cb = ctypes.sizeof(counters)
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    psapi = ctypes.WinDLL("psapi", use_last_error=True)
    kernel.GetCurrentProcess.restype = ctypes.c_void_p
    psapi.GetProcessMemoryInfo.argtypes = [ctypes.c_void_p, ctypes.POINTER(Counters), ctypes.c_ulong]
    ok = psapi.GetProcessMemoryInfo(kernel.GetCurrentProcess(), ctypes.byref(counters), counters.cb)
    return counters.PeakWorkingSetSize / 2**20 if ok else None


def _guard(config: dict, dataset: str) -> None:
    """Reject unsupported profiles/paths before accessing any dataset files."""
    if config.get("profile") not in ("sample", "medium") or dataset not in DATASETS:
        raise ValueError("Only sample/medium and instacart/complete_journey are supported")
    if config.get("role") == "final" or any(
        "holdout" in part.lower() for part in dataset_dir(config, dataset).parts
    ):
        raise ValueError("Holdout access is forbidden in the user-CF development runner")


def settings(profile: str, tuning_dir: str | None = None) -> dict:
    """Runner configuration of ``profile``: load_config (merges the ``usercf`` block of configs/usercf.yaml) plus
    ``selection_source.tuning_dir`` = resolve_tuning_dir(config, tuning_dir), the folder of the base tuning JSON.
    """
    if profile not in ("sample", "medium"):
        raise ValueError("Only sample and medium are supported")
    config = load_config(profile)
    config["selection_source"] = {"tuning_dir": str(resolve_tuning_dir(config, tuning_dir))}
    return config


def output_path(config: dict, dataset: str, stage: str) -> Path:
    """reports/metrics/<profile>_<dataset>_usercf_<stage>.json under the repo_root of ``config``."""
    return Path(config["repo_root"]) / "reports" / "metrics" / f"{config['profile']}_{dataset}_usercf_{stage}.json"


def _prepare(config: dict, dataset: str, split: str) -> tuple:
    _guard(config, dataset)
    history, target = load_split(config, dataset, split)
    items = load_items(config, dataset)
    levels = config["levels"][dataset]
    return attach_levels(history, items, levels), attach_levels(target, items, levels)


def select_parameters(grid: list[dict]) -> dict:
    """Row of ``grid`` (dicts with k, weighting, new_recall@10) with the maximum fit recall; exact ties use smaller k,
    then binary weighting."""
    if not grid or any(not math.isfinite(row["new_recall@10"]) for row in grid):
        raise ValueError("selection requires finite fit scores and a nonempty grid")
    best = min(grid, key=lambda row: (-row["new_recall@10"], row["k"], row["weighting"] != "binary"))
    return {"k": int(best["k"]), "weighting": best["weighting"]}


def tune_history(history: pd.DataFrame, target: pd.DataFrame, config: dict) -> dict:
    """Fit-only (k, weighting) selection over the ``config["usercf"]`` grid on ``history``; ``target`` provides labels for
    recall, never model inputs."""
    started = time.perf_counter()
    _, new_target = split_target(history, target)
    if new_target.empty:
        raise ValueError("fit split has no users with new target items")
    pop = item_popularity(history)
    ks = sorted(set(config["usercf"]["k"]))
    weights = config["usercf"]["weighting"]
    block_size = config["usercf"]["block_size"]
    grid, cache_timings = [], {}
    for weighting in weights:
        tick = time.perf_counter()
        X, _, _ = user_vectors(history, weighting)
        neighbors = user_neighbors(X, max(ks), block_size)
        cache_timings[weighting] = time.perf_counter() - tick
        positions = neighbors.groupby("user_id", sort=False).cumcount()
        for k in ks:
            tick = time.perf_counter()
            lists = _list_from_vectors(history, neighbors[positions < k], X, K, pop, block_size)
            recall = float(per_user_recall(lists, new_target, K).mean())
            row = {"k": k, "weighting": weighting, "new_recall@10": recall,
                   "seconds": time.perf_counter() - tick}
            grid.append(row)
            print(f"fit k={k} weighting={weighting} recall={recall:.9f} seconds={row['seconds']:.2f}", flush=True)
    selected = select_parameters(grid)
    return {
        "split": "fit", "top_k": K, "n_new_target_users": int(new_target["user_id"].nunique()),
        "grid": grid, "selections": selected,
        "at_grid_edge": {"k": selected["k"] in (min(ks), max(ks)),
                         "weighting": "categorical; no ordered boundary"},
        "timing_seconds": {"vectors_and_neighbors_by_weighting": cache_timings,
                           "tuning": time.perf_counter() - started},
    }


def tune(config: dict, dataset: str) -> dict:
    """Fit-split (k, weighting) selection of ``dataset`` under ``config`` (tune_history) with provenance, timing and
    peak memory; returns the tuning document.
    """
    started = time.perf_counter()
    history, target = _prepare(config, dataset, "fit")
    prepare_seconds = time.perf_counter() - started
    document = tune_history(history, target, config)
    document.update(profile=config["profile"], dataset=dataset, label="development (medium)")
    document["timing_seconds"].update(prepare=prepare_seconds, total=time.perf_counter() - started)
    document["process_peak_working_set_mib"] = peak_memory_mib()
    document["timing_note"] = TIMING_NOTE
    return document


def _read_selections(path: Path, required: tuple, profile: str, dataset: str) -> dict:
    if not path.is_file():
        raise FileNotFoundError(f"Required tuning selections are missing: {path}")
    with open(path, encoding="utf-8") as handle:
        document = json.load(handle)
    if document.get("profile") != profile or document.get("dataset") != dataset or document.get("split") != "fit":
        raise ValueError(f"Tuning provenance does not match the requested fit split: {path}")
    selections = document["selections"]
    if any(key not in selections for key in required):
        raise ValueError(f"Incomplete tuning selections: {path}")
    return {key: selections[key] for key in required}


def _decision(ci: list) -> str:
    if ci[0] > 0:
        return "improvement observed"
    if ci[1] < 0:
        return "lower"
    return "no difference established"


def compare(a: pd.Series, b: pd.Series, config: dict, comparator: str,
            model: str, metric: str, primary: bool = False) -> dict:
    """Paired comparison of the per-user series ``a`` (comparator) and ``b`` (model) with the bootstrap settings of
    ``config``: diff = model minus comparator. ``primary`` adds every level and the decision wording (else 95 % only);
    ``comparator``, ``model`` and ``metric`` are recorded as labels."""
    levels = [float(level) for level in config["bootstrap"]["levels"]] if primary else [0.95]
    result = {"comparator": comparator, "model": model, "metric": metric, "n_users": len(a)}
    if a.empty:
        result.update(comparator_mean=None, model_mean=None, diff=None,
                      ci={str(level): None for level in levels})
        if primary:
            result["decision"] = {str(level): "not estimable" for level in levels}
        return result
    boot = config["bootstrap"]
    paired = paired_bootstrap(a, b, int(boot["n"]), int(boot["seed"]), levels)
    result.update(n_users=paired["n_users"], comparator_mean=paired["mean_a"], model_mean=paired["mean_b"],
                  diff=paired["diff"], ci={str(level): paired["ci"][level] for level in levels})
    if primary:
        result["decision"] = {str(level): _decision(paired["ci"][level]) for level in levels}
    return result


def history_breakdown(history: pd.DataFrame, a: pd.Series, b: pd.Series,
                      dataset: str, config: dict) -> list:
    """H8 comparison (``a`` = flat_explore, ``b`` = usercf new-item recall per user) per history-length bin of
    ``dataset`` (HISTORY_BINS; basket counts from ``history``) with the bootstrap settings of ``config``.
    """
    lengths = history[["user_id", "basket_seq"]].drop_duplicates().groupby("user_id").size()
    edges = [0, *HISTORY_BINS[dataset], math.inf]
    rows = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        users = lengths[(lengths > lo) & (lengths <= hi)].index
        eligible = a.index.intersection(users)
        row = compare(a.loc[eligible], b.loc[eligible], config, "flat_explore", "usercf", "new_recall@10", True)
        row.update(bin=f"({lo},{'inf)' if math.isinf(hi) else str(hi) + ']'}",
                   n_history_users=len(users))
        rows.append(row)
    return rows


def diagnostics(lists: dict, pop: pd.Series, neighbors: pd.DataFrame) -> dict:
    """Popularity-rank and neighbour-similarity diagnostics of ``lists`` (usercf, flat_explore, itemcf_flat) and the
    ``neighbors`` frame; popularity rank uses ``pop`` descending, then lexical item ID; counts slots."""
    ranked = pop.rename("pop").rename_axis("item_id").reset_index().sort_values(
        ["pop", "item_id"], ascending=[False, True], kind="mergesort")
    ranks = pd.Series(np.arange(1, len(ranked) + 1), index=ranked["item_id"])
    popularity = {}
    for name in ("usercf", "flat_explore", "itemcf_flat"):
        values = lists[name]["item_id"].map(ranks)
        popularity[name] = {"n_slots": len(values),
                            "top100_fraction": float((values <= 100).mean()) if len(values) else None,
                            "median_popularity_rank": float(values.median()) if len(values) else None}
    sims = neighbors["sim"]
    ucf = lists["usercf"]
    return {"popularity": popularity,
            "popularity_fill_fraction": float((ucf["source"] == "popularity").mean()) if len(ucf) else None,
            "neighbor_similarity": {"n": len(sims), **{
                name: float(sims.quantile(q)) if len(sims) else None
                for name, q in (("median", .5), ("p10", .1), ("p90", .9))}},
            "popularity_rank_tie_rule": "pop desc, item_id asc",
            "fill_denominator": "returned usercf list slots"}


def evaluate(config: dict, dataset: str) -> dict:
    """Eval-split H8 evaluation of ``dataset`` under ``config``: user-CF list with the own (k, weighting) selection,
    comparators and combinations with the base selections, paired bootstrap, breakdown and diagnostics.
    """
    _guard(config, dataset)
    started = time.perf_counter()
    profile = config["profile"]
    base_path = Path(config["selection_source"]["tuning_dir"]) / f"{profile}_{dataset}_tuning.json"
    own_path = output_path(config, dataset, "tuning")
    base = _read_selections(base_path, ("gamma", "gamma_c", "alpha", "r"), profile, dataset)
    own = _read_selections(own_path, ("k", "weighting"), profile, dataset)
    history, target = _prepare(config, dataset, "eval")
    _, new_target = split_target(history, target)
    pop = item_popularity(history)
    users = history["user_id"].drop_duplicates()
    timings = {"prepare": time.perf_counter() - started}
    lists = {}

    def timed(name, function):
        tick = time.perf_counter()
        result = function()
        timings[name] = time.perf_counter() - tick
        print(f"eval {name} seconds={timings[name]:.2f}", flush=True)
        return result

    block_size = config["usercf"]["block_size"]
    X, _, _ = timed("user_vectors", lambda: user_vectors(history, own["weighting"]))
    neighbors = timed("user_neighbors", lambda: user_neighbors(X, own["k"], block_size))
    lists["usercf"] = timed("usercf", lambda: _list_from_vectors(history, neighbors, X, K, pop, block_size))
    del X
    lists["flat_explore"] = timed("flat_explore", lambda: popularity_list(history, users, K, pop, True))
    item_nbr = timed("item_neighbors", lambda: item_neighbors(history, 20, block_size))
    lists["itemcf_flat"] = timed("itemcf_flat", lambda: cf_list(history, item_nbr, K, pop))
    del item_nbr
    lists["explore_star"] = timed("explore_star", lambda: explore_list(
        history, K, base["gamma_c"], base["alpha"], config["explore"]["level"]))
    lists["repeat_star"] = timed("repeat_star", lambda: repeat_list(history, base["gamma"], K, pop))
    lists["final"] = timed("final", lambda: combine(lists["repeat_star"], lists["explore_star"], base["r"], K))
    lists["final_ucf"] = timed("final_ucf", lambda: combine(lists["repeat_star"], lists["usercf"], base["r"], K))
    tick = time.perf_counter()
    new_recall = {name: per_user_recall(frame, new_target, K) for name, frame in lists.items()}
    recalls = {name: per_user_recall(frame, target, K) for name, frame in lists.items()}
    ndcgs = {name: per_user_ndcg(frame, target, K) for name, frame in lists.items()}
    h8 = compare(new_recall["flat_explore"], new_recall["usercf"], config,
                 "flat_explore", "usercf", "new_recall@10", True)
    auxiliary = {f"usercf_vs_{name}": compare(new_recall[name], new_recall["usercf"], config,
                 name, "usercf", "new_recall@10") for name in ("itemcf_flat", "explore_star")}
    for metric, values in (("ndcg@10", ndcgs), ("recall@10", recalls)):
        auxiliary[f"final_ucf_vs_final_{metric}"] = compare(values["final"], values["final_ucf"],
            config, "final", "final_ucf", metric)
    breakdown = history_breakdown(history, new_recall["flat_explore"], new_recall["usercf"], dataset, config)
    diagnostic = diagnostics(lists, pop, neighbors)
    timings["metrics_bootstrap_diagnostics"] = time.perf_counter() - tick
    timings["total"] = time.perf_counter() - started
    return {"profile": profile, "dataset": dataset, "split": "eval", "label": "development (medium)",
            "top_k": K, "selections": own, "base_selections": base,
            "selection_sources": {"usercf": str(own_path), "base": str(base_path)},
            "n_users": {"history": int(history["user_id"].nunique()),
                        "target": int(target["user_id"].nunique()), "new_target": len(new_recall["usercf"])},
            "H8": h8, "auxiliary": auxiliary, "history_length": breakdown, "diagnostics": diagnostic,
            "model_means": {name: {"recall@10": float(recalls[name].mean()),
                                   "ndcg@10": float(ndcgs[name].mean()),
                                   "new_recall@10": float(new_recall[name].mean()) if len(new_target) else None}
                            for name in lists},
            "timing_seconds": timings, "process_peak_working_set_mib": peak_memory_mib(),
            "timing_note": TIMING_NOTE}


def main(argv: list[str] | None = None) -> None:
    """Parse ``argv`` (default: sys.argv) and run the tune / evaluate / all stage; the base tuning JSON must exist
    before any data is read.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=("sample", "medium"), required=True)
    parser.add_argument("--dataset", choices=DATASETS, required=True)
    parser.add_argument("--stage", choices=("tune", "evaluate", "all"), required=True)
    parser.add_argument("--tuning-dir", default=None,
                        help="folder with the base <profile>_<dataset>_tuning.json (default: NBR_TUNING_DIR or <repo>/reports/metrics)")
    args = parser.parse_args(argv)
    config = settings(args.profile, args.tuning_dir)
    _guard(config, args.dataset)
    # For all, fail before expensive work when the read-only baseline is absent.
    if args.stage in ("evaluate", "all"):
        path = Path(config["selection_source"]["tuning_dir"]) / f"{args.profile}_{args.dataset}_tuning.json"
        _read_selections(path, ("gamma", "gamma_c", "alpha", "r"), args.profile, args.dataset)
    if args.stage in ("tune", "all"):
        path = write_json(output_path(config, args.dataset, "tuning"), tune(config, args.dataset), config)
        print(path, flush=True)
    if args.stage in ("evaluate", "all"):
        path = write_json(output_path(config, args.dataset, "eval"), evaluate(config, args.dataset), config)
        print(path, flush=True)


if __name__ == "__main__":
    main()
