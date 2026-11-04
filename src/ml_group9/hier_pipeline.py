"""Tuning, evaluation and stage-wise diagnostics of the hierarchical recommender (food scope only).

Parameters are selected on fit splits only. Instacart uses its own fit split; CJ reports transfer
(Instacart parameters and gamma) and refit (CJ fit-split parameters and CJ gamma). This design was
fixed after the repeat-explore results (post-hoc).
Entry points: `python -m ml_group9 tune-hier` and `eval-hier --dataset ...`, which write
reports/metrics/<profile>_hier_tuning.json and <profile>_hier_<dataset>.json.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from .data import load_split
from .health_subgroup import subgroups
from .hierarchical import (
    VARIANTS,
    HierState,
    Params,
    build_hier_state,
    recommend_hier,
    stage_probabilities,
    type_preferences,
)
from .metrics import paired_bootstrap
from .pipeline import _write_json
from .re_pipeline import baseline_scores, candidate_lists, score_lists, tuning_path
from .repeat_explore import PROBABILITY_CLIP


SCOPE = "food"
Lists = dict[str, list[str]]
# alpha used for the "personal frequency only" stage-1 ranking (the alpha -> 0 limit; the prior only breaks ties).
FREQUENCY_ALPHA = 1e-9
# (label, first model, second model, metric) for paired bootstrap CIs on the final lists.
COMPARISONS = [
    ("recall_hier_full_minus_personal_repeat", "hier_full", "personal_repeat", "recall_at_k"),
    ("ndcg_hier_full_minus_personal_repeat", "hier_full", "personal_repeat", "ndcg_at_k"),
    ("ndcg_hier_full_minus_lr_hybrid", "hier_full", "lr_hybrid", "ndcg_at_k"),
] + [
    (f"{metric.split('_at_k')[0]}_hier_full_minus_{variant}", "hier_full", variant, metric)
    for variant in ["hier_noattr", "hier_notype", "hier_noknn"]
    for metric in ["recall_at_k", "ndcg_at_k", "new_item_recall_at_k"]
]


def _log(message: str) -> None:
    print(f"[hier] {message}", file=sys.stderr, flush=True)


def selected_gamma(config: dict[str, object], dataset: str) -> float:
    """gamma of the repeat-explore food scope selected on that dataset's fit split."""
    tuning = json.loads(tuning_path(config).read_text(encoding="utf-8"))
    return float(tuning[dataset]["scopes"][SCOPE]["selected"]["gamma"])


def final_scores(lists: Lists, state: HierState, top_k: int) -> tuple[dict[str, float], pd.DataFrame]:
    """Final-list metrics; new-item and repeat recall are also measured on the final list."""
    return score_lists(lists, lists, lists, state.targets, state.history_items, top_k)


def evaluate_params(state: HierState, variant: str, params: Params, config: dict[str, object]) -> tuple[dict[str, object], pd.DataFrame]:
    started = time.perf_counter()
    result = recommend_hier(state, variant, params, config)
    summary, per_user = final_scores(result.lists, state, int(config["evaluation"]["top_k"]))
    summary.update(
        {
            "candidate_recall": result.candidate_recall,
            "new_item_candidate_recall": result.new_item_candidate_recall,
            "candidates_per_user": result.candidates_per_user,
            "runtime_seconds": time.perf_counter() - started,
        }
    )
    return summary, per_user


def _params_row(params: Params) -> dict[str, float]:
    return {"alpha": params.alpha, "mu": params.mu, "lambda_type": params.lambda_type, "k": params.k}


def tune_state(config: dict[str, object], state: HierState, label: str) -> dict[str, object]:
    """Fixed-order grid on NDCG@K: (alpha, mu) at k = 0 and the initial lambda_t, then k, then lambda_t.

    Ties keep the earlier setting. Repeated settings are evaluated once.
    """
    settings = config["hierarchical"]
    cache: dict[Params, dict[str, object]] = {}
    rows: list[dict[str, object]] = []

    def run(step: str, params: Params) -> dict[str, object]:
        if params not in cache:
            cache[params], _ = evaluate_params(state, "hier_full", params, config)
            _log(f"{label} {step} {_params_row(params)} ndcg={cache[params]['ndcg_at_k']:.4f} "
                 f"({cache[params]['runtime_seconds']:.1f}s)")
        rows.append({"step": step, **_params_row(params), **cache[params]})
        return cache[params]

    def best(candidates: list[Params], step: str) -> Params:
        chosen, score = None, -np.inf
        for params in candidates:
            value = run(step, params)["ndcg_at_k"]
            if value > score:
                chosen, score = params, value
        return chosen

    lambda_initial = float(settings["lambda_type_initial"])
    chosen = best(
        [Params(float(a), float(m), lambda_initial, 0) for a in settings["alphas"] for m in settings["mus"]],
        "1_alpha_mu",
    )
    chosen = best([Params(chosen.alpha, chosen.mu, chosen.lambda_type, int(k)) for k in settings["knn_k"]], "2_k")
    chosen = best([Params(chosen.alpha, chosen.mu, float(value), chosen.k) for value in settings["lambda_type"]], "3_lambda_type")
    return {
        "gamma": state.gamma,
        "users": len(state.targets),
        "excluded_users": state.excluded_users,
        "grid": rows,
        "selected": _params_row(chosen),
    }


def hier_tuning_path(config: dict[str, object]) -> Path:
    return Path(config["report_dir"]) / "metrics" / f"{config['profile']}_hier_tuning.json"


def tune_hier(config: dict[str, object]) -> dict[str, object]:
    """Select parameters on the Instacart fit split and, for the CJ refit, on the CJ fit split."""
    report: dict[str, object] = {
        "profile": config["profile"],
        "selection_split": "fit",
        "scope": SCOPE,
        "selection_metric": "ndcg_at_k",
        "grid_settings": {
            key: config["hierarchical"][key]
            for key in ["alphas", "mus", "knn_k", "lambda_type", "lambda_type_initial", "candidate_cells", "items_per_cell"]
        },
        "design": "post-hoc (fixed after the repeat-explore results)",
    }
    for dataset in ["instacart", "complete_journey"]:
        if not (Path(config["processed_dir"]) / dataset / "fit_history.parquet").exists():
            report[dataset] = {"skipped": "fit split not prepared"}
            continue
        started = time.perf_counter()
        items, history, target = load_split(config, dataset, "fit")
        state = build_hier_state(config, dataset, items, history, target, selected_gamma(config, dataset), ("full",))
        build_seconds = time.perf_counter() - started
        _log(f"{dataset} fit state built in {build_seconds:.1f}s")
        report[dataset] = tune_state(config, state, dataset)
        report[dataset]["runtime_seconds"] = {"build": build_seconds, "total": time.perf_counter() - started}
        del state
    _write_json(hier_tuning_path(config), report)
    return report


def popularity_list(state: HierState, top_k: int) -> Lists:
    """The K most popular food items (history baskets) for every target user."""
    item_ids = state.interactions.items.to_numpy()
    order = np.lexsort((np.arange(state.n_items), -state.popularity))
    top = [str(item_ids[item]) for item in order if state.in_scope[item]][:top_k]
    return {user: top for user in state.targets}


def baseline_lists(
    config: dict[str, object], state: HierState, items: pd.DataFrame, history: pd.DataFrame, target: pd.DataFrame
) -> tuple[dict[str, Lists], dict[str, object]]:
    """Final food lists of popularity, personal repeat and both frozen LR rankers (scores reused unchanged)."""
    top_k = int(config["evaluation"]["top_k"])
    scored, checksums = baseline_scores(config, items, history, target)
    allowed = set(state.interactions.items.to_numpy()[state.in_scope].astype(str))
    lists = {"popularity": popularity_list(state, top_k)}
    for name, column in [("personal_repeat", "score_repeat"), ("lr_hybrid", "score_lr"), ("lr_hybrid_pref", "score_pref_lr")]:
        lists[name] = candidate_lists(scored, column, allowed, top_k)[0]
    return lists, checksums


def _bootstrap(first: pd.Series, second: pd.Series, config: dict[str, object]) -> dict[str, float]:
    settings = config["bootstrap"]
    return paired_bootstrap(first.dropna(), second.dropna(), int(settings["n"]), int(settings["seed"]))


# ---------------------------------------------------------------------------------------------
# Stage-wise metrics


def set_ranking_metrics(ranked: np.ndarray, truth: list[set[int]], ks: list[int]) -> dict[str, dict[str, float]]:
    """Per k: mean recall |top-k & T| / |T|, hit rate 1[top-k & T non-empty] and precision |top-k & T| / k."""
    result: dict[str, dict[str, float]] = {}
    for k in ks:
        recall, hit, precision = [], [], []
        for row, units in zip(ranked, truth):
            if not units:
                continue
            found = len(set(row[:k].tolist()) & units)
            recall.append(found / len(units))
            hit.append(float(found > 0))
            precision.append(found / k)
        result[f"at_{k}"] = {
            "recall": float(np.mean(recall)) if recall else None,
            "hit_rate": float(np.mean(hit)) if hit else None,
            "precision": float(np.mean(precision)) if precision else None,
            "users": len(recall),
        }
    return result


def _top_indices(scores: np.ndarray, width: int) -> np.ndarray:
    """Column indices by descending score (ties by index)."""
    return np.argsort(-scores, axis=1, kind="stable")[:, :width]


def stage_rankings(
    state: HierState, variant: str, params: Params, width: int, block: int
) -> tuple[np.ndarray, np.ndarray]:
    """Top stage-1 keys by theta and top types by theta * tau for every target user row."""
    hierarchy = state.hierarchies[VARIANTS[variant][0]]
    stage1, types = [], []
    for start in range(0, len(state.target_rows), block):
        chunk = state.target_rows[start : start + block]
        theta, tau = stage_probabilities(state, hierarchy, params, chunk)
        stage1.append(_top_indices(theta, width))
        types.append(_top_indices(theta[:, hierarchy.type_stage1] * tau, width))
    return np.concatenate(stage1), np.concatenate(types)


def target_units(state: HierState, units: np.ndarray) -> list[set[int]]:
    """Per target user row (in target_rows order): the set of units of its food target items."""
    position = np.searchsorted(state.target_rows, state.target_user)
    sets: list[set[int]] = [set() for _ in state.target_rows]
    for row, unit in zip(position, units[state.target_item]):
        sets[row].add(int(unit))
    return sets


def auc_score(scores: np.ndarray, labels: np.ndarray, weights: np.ndarray | None = None) -> float | None:
    """(Weighted) ROC AUC with ties counted as one half; None when a class is missing."""
    weights = np.ones(len(scores)) if weights is None else np.asarray(weights, dtype=np.float64)
    unique, inverse = np.unique(scores, return_inverse=True)
    positive = np.bincount(inverse, weights=weights * (labels == 1), minlength=len(unique))
    negative = np.bincount(inverse, weights=weights * (labels == 0), minlength=len(unique))
    total_positive, total_negative = positive.sum(), negative.sum()
    if total_positive == 0 or total_negative == 0:
        return None
    below = np.cumsum(negative) - negative
    return float((positive * (below + 0.5 * negative)).sum() / (total_positive * total_negative))


def log_loss(probability: np.ndarray, labels: np.ndarray) -> float | None:
    if len(labels) == 0:
        return None
    p = np.clip(probability, PROBABILITY_CLIP, 1.0 - PROBABILITY_CLIP)
    return float(-np.mean(labels * np.log(p) + (1 - labels) * np.log(1 - p)))


def bootstrap_auc_difference(
    users: np.ndarray, labels: np.ndarray, first: np.ndarray, second: np.ndarray, resamples: int, seed: int
) -> dict[str, object]:
    """AUC(first) - AUC(second) with a 95% percentile CI from resampling users (rows keep their user)."""
    point_first, point_second = auc_score(first, labels), auc_score(second, labels)
    unique, inverse = np.unique(users, return_inverse=True)
    if point_first is None or point_second is None:
        return {"users": int(len(unique)), "rows": int(len(labels)), "mean_difference": None, "ci_low": None, "ci_high": None}
    rng = np.random.default_rng(seed)
    differences = []
    for _ in range(resamples):
        counts = np.bincount(rng.integers(0, len(unique), len(unique)), minlength=len(unique))
        weights = counts[inverse].astype(np.float64)
        a, b = auc_score(first, labels, weights), auc_score(second, labels, weights)
        if a is not None and b is not None:
            differences.append(a - b)
    low, high = np.percentile(differences, [2.5, 97.5]) if differences else (np.nan, np.nan)
    return {
        "users": int(len(unique)),
        "rows": int(len(labels)),
        "resamples": int(resamples),
        "valid_resamples": len(differences),
        "mean_difference": point_first - point_second,
        "ci_low": float(low),
        "ci_high": float(high),
    }


def attribute_rows(state: HierState, params: Params) -> pd.DataFrame:
    """One row per (target user, food target item, attribute a) with (c, t(i)) in C_a^type."""
    hierarchy = state.hierarchies["full"]
    frames = []
    types = hierarchy.item_type[state.target_item]
    for column, name in enumerate(hierarchy.attributes):
        keep = hierarchy.type_applicable[types, column]
        users, items, kept_types = state.target_user[keep], state.target_item[keep], types[keep]
        pi_type = type_preferences(state, hierarchy, params.lambda_type, users, kept_types)[:, column]
        frames.append(
            pd.DataFrame(
                {
                    "attribute": name,
                    "user": users,
                    "label": hierarchy.item_flag[items, column].astype(np.int64),
                    "new": state.target_new[keep],
                    "pi_user_type": pi_type,
                    "pi_population_type": hierarchy.type_population[kept_types, column],
                    "pi_user_aisle": state.aisle_model.pi[users, column],
                }
            )
        )
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


PREDICTORS = ["pi_user_type", "pi_population_type", "pi_user_aisle"]


def attribute_stage(state: HierState, params: Params, config: dict[str, object]) -> dict[str, object]:
    """AUC and log-loss of P(x_a(i) = 1) for target items in C_a^type: pi_u(a | t) vs pi_bar(a | t) vs pi_u(a).

    Reported for all users and the health-subgroup definitions G_a and S_a, each split into new and repeat
    items, with a user-bootstrap CI of AUC(pi_u(a | t)) - AUC(pi_bar(a | t)).
    """
    frame = attribute_rows(state, params)
    settings = config["health_subgroup"]
    groups = subgroups(
        state.aisle_model,
        int(settings["min_attribute_purchases"]),
        int(settings["strict"]["min_n_c"]),
        float(settings["strict"]["min_gap"]),
    )
    resamples, seed = int(config["bootstrap"]["n"]), int(config["bootstrap"]["seed"])
    result: dict[str, object] = {}
    for column, name in enumerate(state.hierarchies["full"].attributes):
        rows = frame[frame["attribute"].eq(name)]
        entry: dict[str, object] = {}
        for group, mask in [("all", None), ("G", groups.members[:, column]), ("S", groups.strict[:, column])]:
            members = rows if mask is None else rows[mask[rows["user"].to_numpy()]]
            entry[group] = {}
            for part, subset in [("all", members), ("new", members[members["new"]]), ("repeat", members[~members["new"]])]:
                labels = subset["label"].to_numpy()
                entry[group][part] = {
                    "rows": int(len(subset)),
                    "users": int(subset["user"].nunique()),
                    "positive_share": float(labels.mean()) if len(labels) else None,
                    "auc": {p: auc_score(subset[p].to_numpy(), labels) for p in PREDICTORS},
                    "log_loss": {p: log_loss(subset[p].to_numpy(), labels) for p in PREDICTORS},
                    "auc_user_type_minus_population_type": bootstrap_auc_difference(
                        subset["user"].to_numpy(), labels, subset["pi_user_type"].to_numpy(),
                        subset["pi_population_type"].to_numpy(), resamples, seed,
                    ),
                }
        result[name] = entry
    return result


def cell_ranks(state: HierState, params: Params) -> pd.DataFrame:
    """Rank of each food target item inside its own cell by phi (and by popularity alone); ties by item index."""
    hierarchy = state.hierarchies["full"]
    mu = float(params.mu)
    repeat = state.repeat
    rows = []
    for user, item, new in zip(state.target_user, state.target_item, state.target_new):
        cell = hierarchy.item_cell[item]
        members = hierarchy.cell_members[hierarchy.cell_ptr[cell] : hierarchy.cell_ptr[cell + 1]]
        start, stop = repeat.indptr[user], repeat.indptr[user + 1]
        own_items, own_scores = repeat.indices[start:stop], repeat.data[start:stop]
        rep = np.zeros(len(members))
        if len(own_items):
            position = np.minimum(np.searchsorted(own_items, members), len(own_items) - 1)
            rep = np.where(own_items[position] == members, own_scores[position], 0.0)
        score = rep + mu * hierarchy.pop_cell[members]
        mine = int(np.flatnonzero(members == item)[0])
        phi_rank = int(np.sum((score > score[mine]) | ((score == score[mine]) & (members < item))))
        rows.append((int(user), bool(new), len(members), phi_rank, mine))
    return pd.DataFrame(rows, columns=["user", "new", "cell_size", "phi_rank", "popularity_rank"])


def item_stage(state: HierState, params: Params, top: int = 3) -> dict[str, object]:
    """Given the true cell: share of target items ranked in the top `top` of the cell by phi and by popularity."""
    frame = cell_ranks(state, params)
    result: dict[str, object] = {}
    for part, subset in [("all", frame), ("new", frame[frame["new"]]), ("repeat", frame[~frame["new"]])]:
        result[part] = {
            "rows": int(len(subset)),
            f"phi_hit_at_{top}": float(subset["phi_rank"].lt(top).mean()) if len(subset) else None,
            f"popularity_hit_at_{top}": float(subset["popularity_rank"].lt(top).mean()) if len(subset) else None,
            f"cell_size_at_most_{top}_share": float(subset["cell_size"].le(top).mean()) if len(subset) else None,
            "cell_size_median": float(subset["cell_size"].median()) if len(subset) else None,
        }
    return result


def stagewise(state: HierState, params: Params, config: dict[str, object]) -> dict[str, object]:
    """Stage-1 and type rankings (HIER vs k = 0 vs personal frequency), attribute stage, item stage."""
    block = int(config["hierarchical"].get("user_block", 1000))
    hierarchy = state.hierarchies["full"]
    stage1_truth = target_units(state, state.item_stage1)
    type_truth = target_units(state, hierarchy.item_type)
    settings = {
        "hier": params,
        "k0": Params(params.alpha, params.mu, params.lambda_type, 0),
        "personal_frequency": Params(FREQUENCY_ALPHA, params.mu, params.lambda_type, 0),
    }
    stage1: dict[str, object] = {}
    types: dict[str, object] = {}
    for label, setting in settings.items():
        ranked_stage1, ranked_types = stage_rankings(state, "hier_full", setting, 10, block)
        stage1[label] = set_ranking_metrics(ranked_stage1, stage1_truth, [5, 10])
        types[label] = set_ranking_metrics(ranked_types, type_truth, [10])
    return {
        "stage1_keys": int(len(state.stage1_labels)),
        "types": int(hierarchy.n_types),
        "stage1": stage1,
        "type": types,
        "attribute": attribute_stage(state, params, config),
        "item": item_stage(state, params),
    }


# ---------------------------------------------------------------------------------------------
# Evaluation


def eval_hier(config: dict[str, object], dataset: str) -> dict[str, object]:
    """Evaluate baselines, HIER-full and its ablations on the eval split with fit-split parameters."""
    started = time.perf_counter()
    tuning = json.loads(hier_tuning_path(config).read_text(encoding="utf-8"))
    if dataset == "instacart":
        sources = {"instacart": (tuning["instacart"], "instacart")}
    else:
        sources = {"transfer": (tuning["instacart"], "instacart"), "refit": (tuning.get("complete_journey", {}), "complete_journey")}
    top_k = int(config["evaluation"]["top_k"])
    items, history, target = load_split(config, dataset, "eval")
    report: dict[str, object] = {
        "profile": config["profile"],
        "dataset": dataset,
        "scope": SCOPE,
        "design": "post-hoc (fixed after the repeat-explore results)",
        "new_item_recall_definition": "final top-K list against target items not in I_u",
        "sources": {},
        "runtime_seconds": {},
    }
    states: dict[float, HierState] = {}
    baselines: dict[str, Lists] | None = None
    baseline_users: dict[str, pd.DataFrame] = {}
    for source, (tuned, gamma_dataset) in sources.items():
        if "selected" not in tuned:
            report["sources"][source] = {"skipped": "no tuning for this source"}
            continue
        gamma = selected_gamma(config, gamma_dataset)
        if gamma not in states:
            step = time.perf_counter()
            states[gamma] = build_hier_state(config, dataset, items, history, target, gamma)
            report["runtime_seconds"][f"build_gamma_{gamma}"] = time.perf_counter() - step
        state = states[gamma]
        if baselines is None:
            step = time.perf_counter()
            baselines, checksums = baseline_lists(config, state, items, history, target)
            report["frozen_models"] = checksums
            report["frozen_model_verified"] = all(row["sha256_before"] == row["sha256_after"] for row in checksums.values())
            report["history_users"] = state.n_users
            report["target_users"] = len(state.targets)
            report["excluded_users"] = state.excluded_users
            report["baselines"] = {}
            for name, lists in baselines.items():
                report["baselines"][name], baseline_users[name] = final_scores(lists, state, top_k)
            report["runtime_seconds"]["baselines"] = time.perf_counter() - step
        selected = tuned["selected"]
        params = Params(float(selected["alpha"]), float(selected["mu"]), float(selected["lambda_type"]), int(selected["k"]))
        metrics: dict[str, object] = {}
        per_user = dict(baseline_users)
        for variant in VARIANTS:
            metrics[variant], per_user[variant] = evaluate_params(state, variant, params, config)
            _log(f"{dataset} {source} {variant} ndcg={metrics[variant]['ndcg_at_k']:.4f} "
                 f"({metrics[variant]['runtime_seconds']:.1f}s)")
        step = time.perf_counter()
        stages = stagewise(state, params, config)
        stage_seconds = time.perf_counter() - step
        _log(f"{dataset} {source} stage-wise metrics in {stage_seconds:.1f}s")
        report["sources"][source] = {
            "gamma": gamma,
            "gamma_source": gamma_dataset,
            "selected": selected,
            "metrics": metrics,
            "bootstrap_ci": {
                label: _bootstrap(per_user[first][metric], per_user[second][metric], config)
                for label, first, second, metric in COMPARISONS
            },
            "stagewise": stages,
            "stagewise_runtime_seconds": stage_seconds,
        }
    report["runtime_seconds"]["total"] = time.perf_counter() - started
    _write_json(Path(config["report_dir"]) / "metrics" / f"{config['profile']}_hier_{dataset}.json", report)
    return report
