"""Fit-split selection (PREREG "Selection"), eval-split evaluation with paired
bootstrap (PREREG "Hypotheses"), and the auxiliary history-length breakdown.

All lists are built from history frames only; targets are used solely to
score them. The holdout guard of nbr.data.load_split applies: the Instacart
eval split of a role == "final" profile is read only with
``allow_holdout_eval=True`` (CLI flag --final).
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Any

import pandas as pd

from nbr.baselines import item_popularity, personal_frequency_list, popularity_list
from nbr.combine import combine
from nbr.data import attach_levels, load_items, load_split, split_target
from nbr.explore import category_profile, category_ranking, explore_list, explore_list_cf, population_category_ranking
from nbr.io import write_json
from nbr.itemcf import cf_list, cf_scores, item_neighbors
from nbr.metrics import paired_bootstrap, per_user_category_hit, per_user_ndcg, per_user_recall
from nbr.repeat import repeat_list

K = 10
# History-length bins (number of distinct eval-history baskets); upper edges, last bin open.
HISTORY_BINS = {"instacart": [4, 6, 10, 20], "complete_journey": [20, 50, 100]}
PRIMARY_MODELS = {"final_cf": "final", "final_r10": "final", "explore_L3": "explore_star",
                  "explore_cf": "explore_star", "itemcf_flat": "explore_star"}


# ---------------------------------------------------------------------------
# data preparation
# ---------------------------------------------------------------------------
def prepare(config: dict[str, Any], dataset: str, split: str, *, allow_holdout_eval: bool = False) -> dict[str, Any]:
    """Load a split, attach L1..L3, split the target, and compute pop from the history."""
    history, target = load_split(config, dataset, split, allow_holdout_eval=allow_holdout_eval)
    items = load_items(config, dataset)
    levels = config["levels"][dataset]
    history = attach_levels(history, items, levels)
    target = attach_levels(target, items, levels)
    repeat_target, new_target = split_target(history, target)
    return {
        "history": history,
        "target": target,
        "repeat_target": repeat_target,
        "new_target": new_target,
        "pop": item_popularity(history),
    }


def _isclose(a: float, b: float) -> bool:
    return math.isclose(a, b, rel_tol=1e-12, abs_tol=1e-12)


def _at_edge(value: float, grid: list[float]) -> bool:
    return _isclose(value, min(grid)) or _isclose(value, max(grid))


# ---------------------------------------------------------------------------
# selection rules (pure functions, testable)
# ---------------------------------------------------------------------------
def select_gamma(scores: dict[float, float]) -> float:
    """argmax repeat Recall@10; ties -> gamma closest to 1."""
    best = max(scores.values())
    tied = [g for g, s in scores.items() if _isclose(s, best)]
    return min(tied, key=lambda g: (abs(1.0 - g), -g))


def select_explore(scores: dict[tuple[float, float], float]) -> tuple[float, float]:
    """argmax new-item Recall@10 over (gamma_c, alpha); ties -> larger alpha, then larger gamma_c."""
    best = max(scores.values())
    tied = [key for key, s in scores.items() if _isclose(s, best)]
    return max(tied, key=lambda key: (key[1], key[0]))


def select_r(scores: dict[int, float]) -> int:
    """argmax final-list NDCG@10; ties -> larger r."""
    best = max(scores.values())
    return max(r for r, s in scores.items() if _isclose(s, best))


# ---------------------------------------------------------------------------
# tuning
# ---------------------------------------------------------------------------
def tune(config: dict[str, Any], dataset: str) -> dict[str, Any]:
    """Run the three-stage selection on the fit split and return the tuning document."""
    timings: dict[str, float] = {}
    t0 = time.perf_counter()
    data = prepare(config, dataset, "fit")
    history, target, pop = data["history"], data["target"], data["pop"]
    level = config["explore"]["level"]
    timings["prepare"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    gammas = [float(g) for g in config["repeat"]["gammas"]]
    stage1 = {}
    for gamma in gammas:
        lists = repeat_list(history, gamma, K, pop)
        stage1[gamma] = float(per_user_recall(lists, data["repeat_target"], K).mean())
    gamma_star = select_gamma(stage1)
    timings["stage1_repeat"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    gamma_cs = [float(g) for g in config["explore"]["gamma_c"]]
    alphas = [float(a) for a in config["explore"]["alphas"]]
    stage2 = {}
    for gamma_c in gamma_cs:
        for alpha in alphas:
            lists = explore_list(history, K, gamma_c, alpha, level)
            stage2[(gamma_c, alpha)] = float(per_user_recall(lists, data["new_target"], K).mean())
    gamma_c_star, alpha_star = select_explore(stage2)
    timings["stage2_explore"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    rs = [int(r) for r in config["combine"]["r"]]
    repeat_star = repeat_list(history, gamma_star, K, pop)
    explore_star = explore_list(history, K, gamma_c_star, alpha_star, level)
    stage3 = {}
    for r in rs:
        final = combine(repeat_star, explore_star, r, K)
        stage3[r] = float(per_user_ndcg(final, target, K).mean())
    r_star = select_r(stage3)
    timings["stage3_combine"] = time.perf_counter() - t0

    return {
        "profile": config["profile"],
        "dataset": dataset,
        "split": "fit",
        "k": K,
        "level": level,
        "n_users": {
            "all": int(target["user_id"].nunique()),
            "repeat_target": int(data["repeat_target"]["user_id"].nunique()),
            "new_target": int(data["new_target"]["user_id"].nunique()),
        },
        "stage1_repeat_recall": [{"gamma": g, "recall": s} for g, s in stage1.items()],
        "stage2_explore_new_recall": [
            {"gamma_c": gc, "alpha": a, "recall": s} for (gc, a), s in stage2.items()
        ],
        "stage3_final_ndcg": [{"r": r, "ndcg": s} for r, s in stage3.items()],
        "selections": {"gamma": gamma_star, "gamma_c": gamma_c_star, "alpha": alpha_star, "r": r_star},
        "at_grid_edge": {
            "gamma": _at_edge(gamma_star, gammas),
            "gamma_c": _at_edge(gamma_c_star, gamma_cs),
            "alpha": _at_edge(alpha_star, alphas),
            "r": r_star in (min(rs), max(rs)),
        },
        "timing_seconds": timings,
    }


def tuning_path(config: dict[str, Any], dataset: str) -> Path:
    return Path(config["repo_root"]) / "reports" / "metrics" / f"{config['profile']}_{dataset}_tuning.json"


def eval_path(config: dict[str, Any], dataset: str, transfer_from: str | None) -> Path:
    suffix = "_eval_transfer.json" if transfer_from else "_eval.json"
    return Path(config["repo_root"]) / "reports" / "metrics" / f"{config['profile']}_{dataset}{suffix}"


def run_tune(config: dict[str, Any], dataset: str) -> Path:
    document = tune(config, dataset)
    return write_json(tuning_path(config, dataset), document, config)


def load_selections(config: dict[str, Any], dataset: str) -> dict[str, Any]:
    """Read the selected values from reports/metrics/<profile>_<dataset>_tuning.json."""
    path = tuning_path(config, dataset)
    with open(path, encoding="utf-8") as handle:
        document = json.load(handle)
    selections = dict(document["selections"])
    selections["source"] = str(path)
    return selections


# ---------------------------------------------------------------------------
# evaluation
# ---------------------------------------------------------------------------
def decision(ci: list[float]) -> str:
    """PREREG decision wording for a confidence interval of (model - comparator)."""
    lo, hi = ci
    if lo > 0:
        return "improvement observed"
    if hi < 0:
        return "lower"
    return "no difference established"


def _compare(a: pd.Series, b: pd.Series, config: dict[str, Any], *, primary: bool) -> dict[str, Any]:
    """paired_bootstrap(comparator a, model b) with decision wording for primary tests."""
    boot = config["bootstrap"]
    levels = [float(level) for level in boot["levels"]]
    result = paired_bootstrap(a, b, int(boot["n"]), int(boot["seed"]), levels)
    out = {
        "n_users": result["n_users"],
        "comparator_mean": result["mean_a"],
        "model_mean": result["mean_b"],
        "diff": result["diff"],
        "ci": {str(level): result["ci"][level] for level in levels},
    }
    if primary:
        out["decision"] = {str(level): decision(result["ci"][level]) for level in levels}
    return out


def build_lists(
    history: pd.DataFrame,
    pop: pd.Series,
    selections: dict[str, Any],
    config: dict[str, Any],
    *,
    include_itemcf: bool = True,
) -> tuple[dict[str, pd.DataFrame], dict[str, float]]:
    """All model lists (k = 10) from the eval history; returns (lists, timings)."""
    level = config["explore"]["level"]
    gamma, gamma_c, alpha, r = (
        float(selections["gamma"]), float(selections["gamma_c"]), float(selections["alpha"]), int(selections["r"])
    )
    users = history["user_id"].drop_duplicates()
    lists: dict[str, pd.DataFrame] = {}
    timings: dict[str, float] = {}

    def timed(name, function):
        t0 = time.perf_counter()
        lists[name] = function()
        timings[name] = time.perf_counter() - t0

    timed("popularity", lambda: popularity_list(history, users, K, pop, exclude_history=False))
    timed("personal_frequency", lambda: personal_frequency_list(history, K, pop))
    timed("repeat_pf", lambda: repeat_list(history, 1.0, K, pop))
    timed("repeat_star", lambda: repeat_list(history, gamma, K, pop))
    timed("flat_explore", lambda: popularity_list(history, users, K, pop, exclude_history=True))
    timed("explore_star", lambda: explore_list(history, K, gamma_c, alpha, level))
    timed("final", lambda: combine(lists["repeat_star"], lists["explore_star"], r, K))
    timed("final_r10", lambda: combine(lists["repeat_star"], lists["explore_star"], K, K))
    timed("explore_L3", lambda: explore_list(history, K, gamma_c, alpha, "L3"))
    if include_itemcf:
        n_neighbors = int(config["explore"]["itemcf"]["neighbors"])
        t0 = time.perf_counter()
        neighbors = item_neighbors(history, n_neighbors)
        cf = cf_scores(history, neighbors)
        timings["itemcf_scores"] = time.perf_counter() - t0
        timed("itemcf_flat", lambda: cf_list(history, neighbors, K, pop))
        timed("explore_cf", lambda: explore_list_cf(history, cf, K, gamma_c, alpha, level))
        timed("final_cf", lambda: combine(lists["repeat_star"], lists["explore_cf"], r, K))
    return lists, timings


def history_lengths(history: pd.DataFrame) -> pd.Series:
    """Number of distinct baskets per user in the history."""
    return history[["user_id", "basket_seq"]].drop_duplicates().groupby("user_id").size()


def _bin_labels(edges: list[int]) -> list[tuple[str, float, float]]:
    bounds = [0] + edges + [math.inf]
    labels = []
    for lo, hi in zip(bounds[:-1], bounds[1:]):
        name = f"({lo},{'inf' if math.isinf(hi) else hi}]" if not math.isinf(hi) else f"({lo},inf)"
        labels.append((name, lo, hi))
    return labels


def by_history_length(
    metrics: dict[str, tuple[pd.Series, pd.Series]], lengths: pd.Series, dataset: str, config: dict[str, Any]
) -> list[dict[str, Any]]:
    """Per history-length bin: user counts and comparator/model means with a 95% CI of the diff.

    ``metrics`` maps a hypothesis name to (comparator, model) per-user Series.
    """
    boot = config["bootstrap"]
    rows = []
    for name, lo, hi in _bin_labels(HISTORY_BINS[dataset]):
        users = lengths[(lengths > lo) & (lengths <= hi)].index
        entry: dict[str, Any] = {"bin": name, "n_users": int(len(users))}
        for key, (a, b) in metrics.items():
            index = a.index.intersection(users)
            if len(index) == 0:
                entry[key] = None
                continue
            result = paired_bootstrap(a.loc[index], b.loc[index], int(boot["n"]), int(boot["seed"]), [0.95])
            entry[key] = {
                "n_users": result["n_users"],
                "comparator_mean": result["mean_a"],
                "model_mean": result["mean_b"],
                "diff": result["diff"],
                "ci_0.95": result["ci"][0.95],
            }
        rows.append(entry)
    return rows


def evaluate(
    config: dict[str, Any],
    dataset: str,
    selections: dict[str, Any],
    *,
    allow_holdout_eval: bool = False,
    transfer_from: str | None = None,
    include_itemcf: bool = True,
) -> dict[str, Any]:
    """Evaluate all models on the eval split with the given (fixed) selections."""
    timings: dict[str, float] = {}
    t0 = time.perf_counter()
    data = prepare(config, dataset, "eval", allow_holdout_eval=allow_holdout_eval)
    history, target = data["history"], data["target"]
    repeat_target, new_target, pop = data["repeat_target"], data["new_target"], data["pop"]
    level = config["explore"]["level"]
    timings["prepare"] = time.perf_counter() - t0

    lists, list_timings = build_lists(history, pop, selections, config, include_itemcf=include_itemcf)
    timings.update({f"list_{name}": seconds for name, seconds in list_timings.items()})

    t0 = time.perf_counter()
    list_metrics = {
        name: {
            "recall": float(per_user_recall(frame, target, K).mean()),
            "ndcg": float(per_user_ndcg(frame, target, K).mean()),
        }
        for name, frame in lists.items()
    }

    def recall_on(name, frame_target):
        return per_user_recall(lists[name], frame_target, K)

    def ndcg_all(name):
        return per_user_ndcg(lists[name], target, K)

    # Primary hypotheses: comparator first, model second (diff = model - comparator).
    theta, p_c = category_profile(history, float(selections["gamma_c"]), float(selections["alpha"]), level)
    new_users = new_target["user_id"].drop_duplicates()
    model_rank = category_ranking(theta, K)
    population_rank = population_category_ranking(p_c, new_users, K)
    primary_series = {
        "H1_repeat_recall": (recall_on("repeat_pf", repeat_target), recall_on("repeat_star", repeat_target)),
        "H2_new_recall": (recall_on("flat_explore", new_target), recall_on("explore_star", new_target)),
        "H3_ndcg": (ndcg_all("personal_frequency"), ndcg_all("final")),
        "H3_recall": (recall_on("personal_frequency", target), recall_on("final", target)),
        "H4_category_hit": (
            per_user_category_hit(population_rank, new_target, K, level=level),
            per_user_category_hit(model_rank, new_target, K, level=level),
        ),
    }
    labels = {
        "H1_repeat_recall": ("repeat_pf", "repeat_star", "repeat Recall@10", "users with repeat target"),
        "H2_new_recall": ("flat_explore", "explore_star", "new-item Recall@10", "users with new target"),
        "H3_ndcg": ("personal_frequency", "final", "NDCG@10", "all users"),
        "H3_recall": ("personal_frequency", "final", "Recall@10", "all users"),
        "H4_category_hit": ("population_category_ranking", "category_ranking", "category_hit@10", "users with new target"),
    }
    primary = {}
    for key, (a, b) in primary_series.items():
        comparator, model, metric, users = labels[key]
        primary[key] = {"comparator": comparator, "model": model, "metric": metric, "users": users,
                        **_compare(a, b, config, primary=True)}

    # H5 (PREREG v1.2, generated on development data): flat item-CF vs flat popularity
    # excluding history, new-item Recall@10, users with >= 1 new target item, 95% CI only.
    h5 = None
    if "itemcf_flat" in lists:
        boot = config["bootstrap"]
        result = paired_bootstrap(recall_on("flat_explore", new_target), recall_on("itemcf_flat", new_target),
                                  int(boot["n"]), int(boot["seed"]), [0.95])
        h5 = {
            "label": "secondary confirmatory; generated on development data",
            "comparator": "flat_explore", "model": "itemcf_flat",
            "metric": "new-item Recall@10", "users": "users with new target",
            "n_users": result["n_users"],
            "comparator_mean": result["mean_a"],
            "model_mean": result["mean_b"],
            "diff": result["diff"],
            "ci": {"0.95": result["ci"][0.95]},
            "decision": {"0.95": decision(result["ci"][0.95])},
        }

    # Secondary comparisons (CI only): variant vs primary model, and final vs popularity.
    secondary = {}
    for variant, reference in PRIMARY_MODELS.items():
        if variant not in lists:
            continue
        if reference == "final":
            pairs = {"ndcg": (ndcg_all("final"), ndcg_all(variant)),
                     "recall": (recall_on("final", target), recall_on(variant, target))}
        else:
            pairs = {"new_recall": (recall_on("explore_star", new_target), recall_on(variant, new_target))}
        for metric, (a, b) in pairs.items():
            secondary[f"{variant}_vs_{reference}_{metric}"] = {
                "comparator": reference, "model": variant, "metric": metric, **_compare(a, b, config, primary=False)
            }
    for metric, (a, b) in {"ndcg": (ndcg_all("popularity"), ndcg_all("final")),
                           "recall": (recall_on("popularity", target), recall_on("final", target))}.items():
        secondary[f"final_vs_popularity_{metric}"] = {
            "comparator": "popularity", "model": "final", "metric": metric, **_compare(a, b, config, primary=False)
        }
    timings["metrics_and_bootstrap"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    lengths = history_lengths(history)
    breakdown = by_history_length(
        {key: primary_series[key] for key in ("H1_repeat_recall", "H2_new_recall", "H3_ndcg", "H3_recall")},
        lengths, dataset, config,
    )
    timings["by_history_length"] = time.perf_counter() - t0

    return {
        "profile": config["profile"],
        "dataset": dataset,
        "split": "eval",
        "k": K,
        "level": level,
        "selections": selections,
        "transfer_from": transfer_from,
        "include_itemcf": include_itemcf,
        "n_users": {
            "all": int(target["user_id"].nunique()),
            "repeat_target": int(repeat_target["user_id"].nunique()),
            "new_target": int(new_target["user_id"].nunique()),
        },
        "list_metrics": list_metrics,
        "primary": primary,
        "H5": h5,
        "secondary": secondary,
        "by_history_length": breakdown,
        "timing_seconds": timings,
    }


def run_evaluate(
    config: dict[str, Any],
    dataset: str,
    *,
    transfer_from: str | None = None,
    allow_holdout_eval: bool = False,
    include_itemcf: bool = True,
) -> Path:
    selections = load_selections(config, transfer_from or dataset)
    document = evaluate(config, dataset, selections, allow_holdout_eval=allow_holdout_eval,
                        transfer_from=transfer_from, include_itemcf=include_itemcf)
    return write_json(eval_path(config, dataset, transfer_from), document, config)
