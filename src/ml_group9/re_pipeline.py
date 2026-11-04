"""Tuning and evaluation of the repeat-explore recommender for each recommendation scope.

Entry points: `python -m ml_group9 tune-re` and `eval-re --dataset ...`, which write
reports/metrics/<profile>_re_tuning.json and <profile>_re_<dataset>.json.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse

from .attributes import (
    applicable_categories,
    category_set_sizes,
    category_sets,
    exclude_nonfood_categories,
    used_attributes,
)
from .data import load_split
from .metrics import list_metrics, paired_bootstrap, subset_recall
from .modeling import file_checksum, load_model
from .pipeline import MODEL_VARIANTS, _write_json, model_path
from .recommender import build_feature_frame, score_baselines
from .repeat_explore import (
    Interactions,
    PreferenceModel,
    allocate_slots,
    build_interactions,
    build_preferences,
    cf_scores,
    explore_candidates,
    explore_ranking,
    explore_scores,
    item_fine_categories,
    item_popularity,
    popularity_order,
    preference_candidates,
    repeat_counts,
    repeat_ranking,
    repeat_ratio,
    repeat_scores,
    scope_mask,
    scope_targets,
    seen_item_sets,
    seen_matrix,
    similarity_matrix,
    slot_counts,
    to_lists,
)


Lists = dict[str, list[str]]


@dataclass
class SplitState:
    """Scope-independent statistics of one split, all computed from its history."""

    dataset: str
    items: pd.DataFrame
    interactions: Interactions
    similarity: sparse.csr_matrix
    seen: sparse.csr_matrix
    seen_sets: list[set[int]]
    popularity: np.ndarray
    item_fine: np.ndarray
    preferences: PreferenceModel
    history_items: dict[str, set[str]]
    target_sets: dict[str, set[str]]
    category_sizes: dict[str, dict[str, int]]


@dataclass
class ScopeState:
    mode: str
    in_scope: np.ndarray
    targets: dict[str, set[str]]
    excluded_users: int
    n: np.ndarray
    n_rep: np.ndarray
    preferred: pd.DataFrame


@dataclass
class GammaState:
    repeat_ranked: pd.DataFrame
    candidates: pd.DataFrame


@dataclass
class Recommendation:
    final: Lists
    explore: Lists
    repeat: Lists
    allocation: pd.DataFrame


def build_state(
    config: dict[str, object], dataset: str, items: pd.DataFrame, history: pd.DataFrame, target: pd.DataFrame
) -> SplitState:
    """Index history, item-CF similarity (all departments), and preferences over food-only C_a."""
    interactions = build_interactions(history, items)
    settings = config["attributes"]
    table = applicable_categories(
        items, used_attributes(items), int(settings["min_with"]), int(settings["min_without"])
    )
    food_table = exclude_nonfood_categories(table, items, list(settings["exclude_departments"][dataset]))
    preferences = build_preferences(
        history, items, interactions, category_sets(food_table), float(settings["smoothing"])
    )
    seen = seen_matrix(interactions)
    seen_sets = seen_item_sets(seen)
    item_ids = interactions.items.to_numpy()
    history_items = {
        str(user): {str(item_ids[item]) for item in row} for user, row in zip(interactions.users, seen_sets)
    }
    frame = target.assign(user_id=target["user_id"].astype(str), item_id=target["item_id"].astype(str))
    target_sets = {
        user: set(group) for user, group in frame.groupby("user_id")["item_id"] if user in history_items
    }
    similarity = similarity_matrix(
        history, interactions.items, int(config["item_cf"]["top_k"]), int(config["item_cf"]["block_size"])
    )
    return SplitState(
        dataset=dataset,
        items=items,
        interactions=interactions,
        similarity=similarity,
        seen=seen,
        seen_sets=seen_sets,
        popularity=item_popularity(interactions),
        item_fine=item_fine_categories(items, interactions),
        preferences=preferences,
        history_items=history_items,
        target_sets=target_sets,
        category_sizes={"before_nonfood_exclusion": category_set_sizes(table), "after": category_set_sizes(food_table)},
    )


def build_scope(state: SplitState, mode: str, config: dict[str, object]) -> ScopeState:
    nonfood = list(config["scope"]["nonfood_departments"][state.dataset])
    in_scope = scope_mask(state.items, state.interactions.items, mode, nonfood)
    allowed = None if mode == "all" else set(state.interactions.items.to_numpy()[in_scope].astype(str))
    targets, excluded = scope_targets(state.target_sets, allowed)
    n, n_rep = repeat_counts(state.interactions, in_scope)
    preferred = preference_candidates(
        state.seen_sets, in_scope, state.preferences, state.item_fine, state.popularity, config["explore"]
    )
    return ScopeState(mode, in_scope, targets, excluded, n, n_rep, preferred)


def build_gamma(state: SplitState, scope: ScopeState, gamma: float, config: dict[str, object]) -> GammaState:
    repeat = repeat_scores(state.interactions, gamma)
    cf = cf_scores(repeat, state.similarity, state.seen, scope.in_scope)
    candidates = explore_candidates(
        cf,
        state.seen,
        scope.in_scope,
        state.preferences,
        state.item_fine,
        state.popularity,
        config["explore"],
        int(config["evaluation"]["top_k"]),
        preferred=scope.preferred,
        seen_sets=state.seen_sets,
    )
    return GammaState(repeat_ranking(repeat, scope.in_scope), candidates)


def recommend(
    state: SplitState,
    scope: ScopeState,
    gamma_state: GammaState,
    kappa: float,
    beta: float,
    config: dict[str, object],
    global_r: int | None = None,
) -> Recommendation:
    """Final lists (repeat slots first, then explore slots) plus the two component rankings."""
    top_k = int(config["evaluation"]["top_k"])
    n_users = state.interactions.shape[0]
    top_repeat = gamma_state.repeat_ranked[gamma_state.repeat_ranked["rank"] < top_k]
    available = np.bincount(gamma_state.repeat_ranked["user"].to_numpy(np.int64), minlength=n_users)
    if global_r is None:
        rho, _ = repeat_ratio(scope.n, scope.n_rep, kappa)
        slots = slot_counts(rho, top_k, available)
    else:
        slots = np.minimum(int(global_r), available)
    scores = explore_scores(gamma_state.candidates, beta, float(config["explore"]["epsilon"]))
    ranked = explore_ranking(gamma_state.candidates, scores)
    top_explore = ranked[ranked["rank"] < top_k]
    allocation = allocate_slots(top_repeat, top_explore, slots, top_k)
    users, items = state.interactions.users, state.interactions.items
    return Recommendation(
        final=to_lists(allocation, users, items),
        explore=to_lists(top_explore, users, items, order="rank"),
        repeat=to_lists(top_repeat, users, items, order="rank"),
        allocation=allocation,
    )


def score_lists(
    final: Lists,
    explore: Lists,
    repeat: Lists,
    targets: dict[str, set[str]],
    history_items: dict[str, set[str]],
    top_k: int,
) -> tuple[dict[str, float], pd.DataFrame]:
    """Overall metrics on the final list, new-item recall on the explore list, repeat recall on the repeat list."""
    summary, per_user = list_metrics(final, targets, top_k)
    new_items = {user: truth - history_items.get(user, set()) for user, truth in targets.items()}
    repeat_items = {user: truth & history_items.get(user, set()) for user, truth in targets.items()}
    new_mean, new_values = subset_recall(explore, new_items, top_k)
    repeat_mean, repeat_values = subset_recall(repeat, repeat_items, top_k)
    summary.update(
        {
            "new_item_recall_at_k": new_mean,
            "new_item_users": int(len(new_values)),
            "repeat_recall_at_k": repeat_mean,
            "repeat_users": int(len(repeat_values)),
        }
    )
    per_user = per_user.assign(new_item_recall_at_k=new_values, repeat_recall_at_k=repeat_values)
    return summary, per_user


def _evaluate(state: SplitState, scope: ScopeState, recommendation: Recommendation, top_k: int):
    return score_lists(
        recommendation.final, recommendation.explore, recommendation.repeat, scope.targets, state.history_items, top_k
    )


def tune_split(config: dict[str, object], state: SplitState) -> dict[str, object]:
    """Select (gamma, kappa) by NDCG@K at beta = 0, then beta* by new-item recall, then global r; per scope."""
    grid = config["repeat_explore"]
    top_k = int(config["evaluation"]["top_k"])
    result: dict[str, object] = {"category_sizes": state.category_sizes, "scopes": {}}
    for mode in config["scope"]["modes"]:
        scope = build_scope(state, mode, config)
        rows: list[dict[str, object]] = []
        best: tuple[float, float, float, GammaState] | None = None
        for gamma in grid["gammas"]:
            gamma_state = build_gamma(state, scope, float(gamma), config)
            for kappa in grid["kappas"]:
                summary, _ = _evaluate(state, scope, recommend(state, scope, gamma_state, float(kappa), 0.0, config), top_k)
                rows.append({"gamma": float(gamma), "kappa": float(kappa), **summary})
                if best is None or summary["ndcg_at_k"] > best[0]:
                    best = (summary["ndcg_at_k"], float(gamma), float(kappa), gamma_state)
            del gamma_state
        _, gamma, kappa, gamma_state = best
        beta_rows: list[dict[str, object]] = []
        for beta in grid["betas"]:
            summary, _ = _evaluate(state, scope, recommend(state, scope, gamma_state, kappa, float(beta), config), top_k)
            beta_rows.append({"beta": float(beta), **summary})
        beta_star = max(beta_rows, key=lambda row: row["new_item_recall_at_k"])["beta"]
        r_rows: list[dict[str, object]] = []
        for r in grid["global_r"]:
            recommendation = recommend(state, scope, gamma_state, kappa, beta_star, config, global_r=int(r))
            summary, _ = _evaluate(state, scope, recommendation, top_k)
            r_rows.append({"global_r": int(r), **summary})
        global_r = max(r_rows, key=lambda row: row["ndcg_at_k"])["global_r"]
        result["scopes"][mode] = {
            "users": len(scope.targets),
            "excluded_users": scope.excluded_users,
            "grid_gamma_kappa": rows,
            "grid_beta": beta_rows,
            "grid_global_r": r_rows,
            "selected": {"gamma": gamma, "kappa": kappa, "beta": beta_star, "global_r": global_r},
        }
    return result


def tuning_path(config: dict[str, object]) -> Path:
    return Path(config["report_dir"]) / "metrics" / f"{config['profile']}_re_tuning.json"


def tune_re(config: dict[str, object]) -> dict[str, object]:
    """Tune on the Instacart fit split (transfer) and, when prepared, on the CJ fit split (refit)."""
    report: dict[str, object] = {"profile": config["profile"], "selection_split": "fit"}
    for dataset in ["instacart", "complete_journey"]:
        base = Path(config["processed_dir"]) / dataset
        if not (base / "fit_history.parquet").exists():
            report[dataset] = {"skipped": "fit split not prepared"}
            continue
        items, history, target = load_split(config, dataset, "fit")
        report[dataset] = tune_split(config, build_state(config, dataset, items, history, target))
    _write_json(tuning_path(config), report)
    return report


def _top_lists(frame: pd.DataFrame, column: str, top_k: int) -> Lists:
    ordered = frame.sort_values(["user_id", column, "item_id"], ascending=[True, False, True], kind="mergesort")
    top = ordered.groupby("user_id", sort=False).head(top_k)
    return {str(user): list(group) for user, group in top.groupby("user_id", sort=False)["item_id"]}


def baseline_scores(
    config: dict[str, object], items: pd.DataFrame, history: pd.DataFrame, target: pd.DataFrame
) -> tuple[pd.DataFrame, dict[str, dict[str, str]]]:
    """Candidate scores of personal repeat and both frozen LR rankers, with checksums around scoring."""
    paths = {name: model_path(config, stem) for name, (stem, _, _) in MODEL_VARIANTS.items()}
    before = {name: file_checksum(path) for name, path in paths.items()}
    features, _, _ = build_feature_frame(history, items, target, config)
    scored = score_baselines(features)
    for name, (_, column, _) in MODEL_VARIANTS.items():
        artifact = load_model(paths[name])
        scored[column] = artifact["model"].predict_proba(scored[artifact["feature_columns"]])[:, 1]
    after = {name: file_checksum(path) for name, path in paths.items()}
    if before != after:
        raise RuntimeError("Model artifact changed during evaluation")
    columns = ["user_id", "item_id", "user_item_rate", "score_repeat"] + [
        column for _, column, _ in MODEL_VARIANTS.values()
    ]
    frame = scored[columns].assign(user_id=scored["user_id"].astype(str), item_id=scored["item_id"].astype(str))
    checksums = {name: {"sha256_before": before[name], "sha256_after": after[name]} for name in paths}
    return frame, checksums


def candidate_lists(
    scored: pd.DataFrame, column: str, allowed: set[str] | None, top_k: int
) -> tuple[Lists, Lists, Lists]:
    """Re-rank existing candidates inside F with unchanged scores: final, new-item and repeat lists."""
    frame = scored if allowed is None else scored[scored["item_id"].isin(allowed)]
    seen = frame["user_item_rate"].gt(0)
    return (
        _top_lists(frame, column, top_k),
        _top_lists(frame[~seen], column, top_k),
        _top_lists(frame[seen], column, top_k),
    )


def popularity_lists(state: SplitState, scope: ScopeState, top_k: int) -> tuple[Lists, Lists, Lists]:
    """Global history popularity inside F; new-item and repeat lists use the same order."""
    item_ids = state.interactions.items.to_numpy()
    order = [int(item) for item in popularity_order(state.popularity) if scope.in_scope[item]]
    rank = {item: position for position, item in enumerate(order)}
    final_list = [str(item_ids[item]) for item in order[:top_k]]
    final: Lists = {}
    explore: Lists = {}
    repeat: Lists = {}
    for user in scope.targets:
        row = state.interactions.users.get_loc(user)
        seen_items = state.seen_sets[row]
        final[user] = final_list
        unseen: list[str] = []
        for item in order:
            if item not in seen_items:
                unseen.append(str(item_ids[item]))
                if len(unseen) == top_k:
                    break
        explore[user] = unseen
        own = sorted((item for item in seen_items if item in rank), key=rank.__getitem__)
        repeat[user] = [str(item_ids[item]) for item in own[:top_k]]
    return final, explore, repeat


def re_diagnostics(
    state: SplitState,
    scope: ScopeState,
    gamma_state: GammaState,
    recommendation: Recommendation,
    top_k: int,
) -> dict[str, float]:
    """Candidate coverage of new target items and the make-up of explore slots."""
    users = state.interactions.users
    item_ids = state.interactions.items.to_numpy()
    candidates = gamma_state.candidates
    candidate_pairs = pd.Series(item_ids[candidates["item"].to_numpy(np.int64)]).groupby(
        users.to_numpy()[candidates["user"].to_numpy(np.int64)]
    ).apply(set)
    coverage = []
    for user, truth in scope.targets.items():
        new_items = truth - state.history_items.get(user, set())
        if new_items:
            coverage.append(len(candidate_pairs.get(user, set()) & new_items) / len(new_items))
    allocation = recommendation.allocation
    target_rows = users.get_indexer(list(scope.targets))
    explore_rows = allocation[allocation["slot"].eq("explore")]
    explore_counts = np.bincount(explore_rows["user"].to_numpy(np.int64), minlength=len(users))
    return {
        "new_item_candidate_recall": float(np.mean(coverage)) if coverage else 0.0,
        "explore_slot_share": float((explore_counts[target_rows] / top_k).mean()) if len(target_rows) else 0.0,
        "explore_from_preference_share": float(explore_rows["from_pref"].mean()) if len(explore_rows) else 0.0,
        "explore_zero_cf_share": float(explore_rows["s_cf"].eq(0).mean()) if len(explore_rows) else 0.0,
    }


def attribute_hit_rates(
    state: SplitState, scope: ScopeState, recommendations: dict[str, Recommendation], config: dict[str, object]
) -> dict[str, dict[str, object]]:
    """Per attribute: users with n_C >= distinct_min_n and a new attribute item in the target,
    and the explore-list recall of those items for each recommendation."""
    top_k = int(config["evaluation"]["top_k"])
    min_n = int(config["attributes"]["distinct_min_n"])
    model = state.preferences
    item_ids = state.interactions.items.to_numpy()
    rows = state.interactions.users.get_indexer(list(scope.targets))
    result: dict[str, dict[str, object]] = {}
    for column, name in enumerate(model.attributes):
        mask = (model.item_flag[:, column] == 1) & model.item_applicable[:, column] & scope.in_scope
        attribute_items = set(item_ids[mask].astype(str))
        relevant = {}
        for user, row in zip(scope.targets, rows):
            if model.n_c[row, column] >= min_n:
                new_items = (scope.targets[user] - state.history_items.get(user, set())) & attribute_items
                if new_items:
                    relevant[user] = new_items
        result[name] = {"users": len(relevant)}
        for label, recommendation in recommendations.items():
            result[name][label] = subset_recall(recommendation.explore, relevant, top_k)[0]
    return result


def _bootstrap(first: pd.Series, second: pd.Series, config: dict[str, object]) -> dict[str, float]:
    settings = config["bootstrap"]
    return paired_bootstrap(first.dropna(), second.dropna(), int(settings["n"]), int(settings["seed"]))


def eval_re(config: dict[str, object], dataset: str) -> dict[str, object]:
    """Evaluate baselines and repeat-explore on the eval split for every scope.

    Instacart uses parameters tuned on its own fit split; CJ reports both transfer (Instacart
    parameters) and refit (CJ fit-split parameters). Nothing is selected on the eval split.
    """
    tuning = json.loads(tuning_path(config).read_text(encoding="utf-8"))
    sources = {"instacart": tuning["instacart"]} if dataset == "instacart" else {
        "transfer": tuning["instacart"],
        "refit": tuning.get("complete_journey", {}),
    }
    top_k = int(config["evaluation"]["top_k"])
    items, history, target = load_split(config, dataset, "eval")
    state = build_state(config, dataset, items, history, target)
    scored, checksums = baseline_scores(config, items, history, target)
    report: dict[str, object] = {
        "profile": config["profile"],
        "dataset": dataset,
        "history_users": len(state.interactions.users),
        "target_users": len(state.target_sets),
        "category_sizes": state.category_sizes,
        "frozen_models": checksums,
        "frozen_model_verified": all(row["sha256_before"] == row["sha256_after"] for row in checksums.values()),
        "slots_mode": config["slots"]["mode"],
        "scopes": {},
    }
    for mode in config["scope"]["modes"]:
        scope = build_scope(state, mode, config)
        allowed = None if mode == "all" else set(state.interactions.items.to_numpy()[scope.in_scope].astype(str))
        metrics: dict[str, dict[str, float]] = {}
        per_user: dict[str, pd.DataFrame] = {}
        baselines = {"popularity": popularity_lists(state, scope, top_k)}
        for name, column in [("personal_repeat", "score_repeat"), ("lr_hybrid", "score_lr"), ("lr_hybrid_pref", "score_pref_lr")]:
            baselines[name] = candidate_lists(scored, column, allowed, top_k)
        for name, lists in baselines.items():
            metrics[name], per_user[name] = score_lists(*lists, scope.targets, state.history_items, top_k)
        scope_report: dict[str, object] = {
            "users": len(scope.targets),
            "excluded_users": scope.excluded_users,
            "baselines": metrics,
            "repeat_explore": {},
        }
        gamma_cache: dict[float, GammaState] = {}
        for source, tuned in sources.items():
            if "scopes" not in tuned:
                scope_report["repeat_explore"][source] = {"skipped": "no tuning for this source"}
                continue
            selected = tuned["scopes"][mode]["selected"]
            gamma = float(selected["gamma"])
            if gamma not in gamma_cache:
                gamma_cache[gamma] = build_gamma(state, scope, gamma, config)
            global_r = int(selected["global_r"]) if config["slots"]["mode"] == "global" else None
            recommendations = {
                label: recommend(state, scope, gamma_cache[gamma], float(selected["kappa"]), beta, config, global_r)
                for label, beta in [("re_beta0", 0.0), ("re_beta_star", float(selected["beta"]))]
            }
            re_metrics = {}
            re_users = {}
            for label, recommendation in recommendations.items():
                re_metrics[label], re_users[label] = _evaluate(state, scope, recommendation, top_k)
            star = re_users["re_beta_star"]
            scope_report["repeat_explore"][source] = {
                "selected": selected,
                "metrics": re_metrics,
                "bootstrap_ci": {
                    "new_item_recall_star_minus_beta0": _bootstrap(
                        star["new_item_recall_at_k"], re_users["re_beta0"]["new_item_recall_at_k"], config
                    ),
                    "recall_star_minus_personal_repeat": _bootstrap(
                        star["recall_at_k"], per_user["personal_repeat"]["recall_at_k"], config
                    ),
                    "ndcg_star_minus_personal_repeat": _bootstrap(
                        star["ndcg_at_k"], per_user["personal_repeat"]["ndcg_at_k"], config
                    ),
                    "ndcg_star_minus_lr_hybrid": _bootstrap(
                        star["ndcg_at_k"], per_user["lr_hybrid"]["ndcg_at_k"], config
                    ),
                },
                "diagnostics": {
                    label: re_diagnostics(state, scope, gamma_cache[gamma], recommendation, top_k)
                    for label, recommendation in recommendations.items()
                },
                "attribute_hit_rates": attribute_hit_rates(state, scope, recommendations, config),
            }
        report["scopes"][mode] = scope_report
    _write_json(Path(config["report_dir"]) / "metrics" / f"{config['profile']}_re_{dataset}.json", report)
    return report
