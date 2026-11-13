# *------------------- 2026 Fall Semester Machine Learning Term-project: Team 9 --------------------*
# Code First written at 2026-11-05
# Code Final Edited at 2026-11-13
# Editor: Yejun Kim

'''
Purpose : Fit-split selection (PREREG "Selection"), eval-split evaluation with paired
          bootstrap (PREREG "Hypotheses"), and the auxiliary history-length breakdown.
          All lists are built from history frames only; targets are used solely to
          score them (and, in tune, to label the logistic-regression training rows of
          W6). The holdout guard of nbr.data.load_split applies: the Instacart eval
          split of a role == "final" profile is read only with
          ``allow_holdout_eval=True`` (CLI flag --final).
Inputs  : The profile configuration dict (config) and a dataset name; the history and
          target frames of the fit / eval split (nbr.data.load_split) and the item table
          (nbr.data.load_items). run_evaluate also reads the tuning JSON and the lr_repeat
          joblib artifacts written by run_tune.
Outputs : tune returns the tuning document and run_tune writes it to
          reports/metrics/<profile>_<dataset>_tuning.json; the fitted lr_repeat bundles go to
          artifacts/<profile>/<dataset>_lr_repeat<suffix>.joblib.
          evaluate returns the eval document and run_evaluate writes it to
          reports/metrics/<profile>_<dataset>_eval.json (_eval_transfer.json with transfer_from).
Usage   :
    from nbr.pipeline import run_evaluate, run_tune
    path = run_tune(config, args.dataset)
'''

from __future__ import annotations

# ---------------------------------------------------------------------------
# imports
# ---------------------------------------------------------------------------
import hashlib
import json
import math
import time
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from nbr.baselines import item_popularity, personal_frequency_list, popularity_list
from nbr.combine import combine
from nbr.data import attach_levels, load_items, load_split, split_target
from nbr.explore import category_profile, category_ranking, explore_list, explore_list_cf, population_category_ranking
from nbr.io import write_json
from nbr.itemcf import cf_list, cf_scores, item_neighbors
from nbr.lr_repeat import SOURCE_ICF, build_training, fit_select, lr_repeat_list, train_only_bundle
from nbr.metrics import paired_bootstrap, per_user_category_hit, per_user_ndcg, per_user_recall
from nbr.policy import evaluate_policy
from nbr.repeat import repeat_list
from nbr.usercf import ucf_list
from nbr.usercf_eval import peak_memory_mib
from nbr.usercf_eval import tune_history as tune_usercf
from nbr.v15 import evaluate_v15

# ---------------------------------------------------------------------------
# constants
# ---------------------------------------------------------------------------
K = 10
# History-length bins (number of distinct eval-history baskets); upper edges, last bin open.
HISTORY_BINS = {"instacart": [4, 6, 10, 20], "complete_journey": [20, 50, 100]}
PRIMARY_MODELS = {"final_cf": "final", "final_r10": "final", "explore_L3": "explore_star",
                  "explore_cf": "explore_star", "itemcf_flat": "explore_star"}
# W6: logistic-regression repeat ranker. Secondary comparisons name -> (comparator, model, metric).
LR_SECONDARY = {"repeat_lr_vs_repeat_pf_repeat_recall": ("repeat_pf", "repeat_lr", "repeat_recall"),
                "final_lr_vs_personal_frequency_ndcg": ("personal_frequency", "final_lr", "ndcg")}


# W8: combinations of a repeat list and an explore list.
# Their slot count r is re-selected on the fit split.
# Each entry maps name -> (repeat list, explore list, uses the LR ranker).
# LR combinations are selected on the LR validation users.
# Their model is fitted on the LR training users only (no label leakage).
INTEGRATION_COMBOS = {"final_lr_ucf": ("repeat_lr", "usercf", True),
                      "final_star_ucf": ("repeat_star", "usercf", False),
                      "final_lr_r": ("repeat_lr", "explore_star", True)}
INTEGRATION_SECONDARY = {"final_star_ucf_vs_final_ndcg": ("final", "final_star_ucf", "ndcg"),
                         "final_lr_r_vs_final_lr_ndcg": ("final_lr", "final_lr_r", "ndcg"),
                         "final_lr_ucf_vs_final_ndcg": ("final", "final_lr_ucf", "ndcg"),
                         "final_lr_ucf_vs_final_recall": ("final", "final_lr_ucf", "recall"),
                         "final_lr_ucf_vs_personal_frequency_ndcg": ("personal_frequency", "final_lr_ucf", "ndcg"),
                         "usercf_vs_itemcf_flat_new_recall": ("itemcf_flat", "usercf", "new_recall")}
# PREREG v1.3 primary H7' = C3 (final_lr_ucf) vs C0 (final), NDCG@10 with Recall@10 reported.
# These two comparisons also carry the decision wording at both CI levels.
# This is reporting only; lists and selections are unchanged.
INTEGRATION_DECISION = ("final_lr_ucf_vs_final_ndcg", "final_lr_ucf_vs_final_recall")
# PREREG v1.5 (W19): R2' = R2 + item-CF feature (artifact suffix) and C4 = combine(R2', E2, r) with r re-selected like C3's.
ICF_SUFFIX = "_icf"
V15_COMBOS = {"final_lri_ucf": ("repeat_lri", "usercf")}
V15_PREREG = "v1.5"
# PREREG v1.5.1: label of the policy blocks of the holdout4 run (secondary replication of the v1.4 policy, base C3 and base C4).
POLICY_LABELS = {"holdout4": "secondary replication on holdout4 (PREREG v1.5.1)"}


# ---------------------------------------------------------------------------
# run labels and feature switches
# ---------------------------------------------------------------------------
def policy_label(config: dict[str, Any]) -> str:
    """Label of the "policy" / "policy_c4" blocks: POLICY_LABELS for the holdout4 profile, else run_label(config)."""
    return POLICY_LABELS.get(str(config.get("profile")), run_label(config))


def icf_enabled(config: dict[str, Any]) -> bool:
    """True when ``config`` sets lr_repeat.itemcf_feature (PREREG v1.5: fit R2' and evaluate C4; default false)."""
    return bool((config.get("lr_repeat") or {}).get("itemcf_feature", False))


def icf_neighbor_count(config: dict[str, Any]) -> int:
    """Neighbour count of the icf feature: explore.itemcf.neighbors of ``config`` (20, fixed by PREREG v1.5)."""
    return int(config["explore"]["itemcf"]["neighbors"])


def run_label(config: dict[str, Any]) -> str:
    """Role and profile of a run from ``config`` ("role", "profile"), e.g. "development (medium)"."""
    return f"{config.get('role')} ({config.get('profile')})"


# ---------------------------------------------------------------------------
# data preparation
# ---------------------------------------------------------------------------
def prepare(config: dict[str, Any], dataset: str, split: str, *, allow_holdout_eval: bool = False) -> dict[str, Any]:
    """Load ``split`` of ``dataset`` under ``config``, attach L1..L3, split the target, compute pop.

    ``allow_holdout_eval`` is passed to nbr.data.load_split (holdout guard). Returns a dict with
    history, target, repeat_target, new_target and pop (basket popularity of the history).
    """
    # Load the history and target frames of the split through the holdout guard.
    history, target = load_split(config, dataset, split, allow_holdout_eval=allow_holdout_eval)
    # Load the item table and the category levels configured for the dataset.
    items = load_items(config, dataset)
    levels = config["levels"][dataset]
    # Attach the category level columns to both frames.
    history = attach_levels(history, items, levels)
    target = attach_levels(target, items, levels)
    # Split the target into items already in the history and new items.
    repeat_target, new_target = split_target(history, target)
    # Return the frames together with the basket popularity of the history.
    return {
        "history": history,
        "target": target,
        "repeat_target": repeat_target,
        "new_target": new_target,
        "pop": item_popularity(history),
    }


# ---------------------------------------------------------------------------
# numeric helpers
# ---------------------------------------------------------------------------
def _isclose(a: float, b: float) -> bool:
    '''True when ``a`` and ``b`` are equal within relative and absolute tolerance 1e-12.'''
    return math.isclose(a, b, rel_tol=1e-12, abs_tol=1e-12)


def _at_edge(value: float, grid: list[float]) -> bool:
    '''True when ``value`` equals the smallest or the largest value of ``grid`` (_isclose).'''
    return _isclose(value, min(grid)) or _isclose(value, max(grid))


# ---------------------------------------------------------------------------
# selection rules (pure functions, testable)
# ---------------------------------------------------------------------------
def select_gamma(scores: dict[float, float]) -> float:
    """argmax of ``scores`` (gamma -> repeat Recall@10); ties -> gamma closest to 1."""
    # Find the best score.
    best = max(scores.values())
    # Collect the gammas whose score ties with the best.
    tied = [g for g, s in scores.items() if _isclose(s, best)]
    # Return the tied gamma closest to 1, preferring the larger gamma at equal distance.
    return min(tied, key=lambda g: (abs(1.0 - g), -g))


def select_explore(scores: dict[tuple[float, float], float]) -> tuple[float, float]:
    """argmax of ``scores`` ((gamma_c, alpha) -> new-item Recall@10); ties -> larger alpha, then larger gamma_c."""
    # Find the best score.
    best = max(scores.values())
    # Collect the (gamma_c, alpha) keys whose score ties with the best.
    tied = [key for key, s in scores.items() if _isclose(s, best)]
    # Return the tied key with the largest alpha, then the largest gamma_c.
    return max(tied, key=lambda key: (key[1], key[0]))


def select_r(scores: dict[int, float]) -> int:
    """argmax of ``scores`` (r -> final-list NDCG@10); ties -> larger r."""
    # Find the best score.
    best = max(scores.values())
    # Return the largest r whose score ties with the best.
    return max(r for r, s in scores.items() if _isclose(s, best))


# ---------------------------------------------------------------------------
# tuning
# ---------------------------------------------------------------------------
def tune(config: dict[str, Any], dataset: str) -> dict[str, Any]:
    """Three-stage selection, W6 LR fit and W8 integration on the fit split of ``dataset`` under ``config``; returns the tuning document."""
    # Load the fit split and read the explore level, timing the step.
    timings: dict[str, float] = {}
    t0 = time.perf_counter()
    data = prepare(config, dataset, "fit")
    history, target, pop = data["history"], data["target"], data["pop"]
    level = config["explore"]["level"]
    timings["prepare"] = time.perf_counter() - t0

    # Stage 1 scores the decayed-frequency repeat list (R1) of every gamma by repeat Recall@10.
    t0 = time.perf_counter()
    gammas = [float(g) for g in config["repeat"]["gammas"]]
    stage1 = {}
    for gamma in gammas:
        lists = repeat_list(history, gamma, K, pop)
        stage1[gamma] = float(per_user_recall(lists, data["repeat_target"], K).mean())
    # Select gamma with the tie rule of select_gamma.
    gamma_star = select_gamma(stage1)
    # Record the time and the process peak memory of stage 1.
    timings["stage1_repeat"] = time.perf_counter() - t0
    peak_memory: dict[str, float | None] = {"stage1_repeat": peak_memory_mib()}

    # Stage 2 scores the hierarchical explore list (E1) of every (gamma_c, alpha) pair.
    # The score is the mean new-item Recall@10.
    t0 = time.perf_counter()
    gamma_cs = [float(g) for g in config["explore"]["gamma_c"]]
    alphas = [float(a) for a in config["explore"]["alphas"]]
    stage2 = {}
    for gamma_c in gamma_cs:
        for alpha in alphas:
            lists = explore_list(history, K, gamma_c, alpha, level)
            stage2[(gamma_c, alpha)] = float(per_user_recall(lists, data["new_target"], K).mean())
    # Select (gamma_c, alpha) with the tie rule of select_explore.
    gamma_c_star, alpha_star = select_explore(stage2)
    # Record the time and the peak memory of stage 2.
    timings["stage2_explore"] = time.perf_counter() - t0
    peak_memory["stage2_explore"] = peak_memory_mib()

    # Stage 3 reads the grid of repeat slots r and rebuilds the two selected lists.
    t0 = time.perf_counter()
    rs = [int(r) for r in config["combine"]["r"]]
    repeat_star = repeat_list(history, gamma_star, K, pop)
    explore_star = explore_list(history, K, gamma_c_star, alpha_star, level)
    # Score the combined list of every r by mean NDCG@10 over all fit users.
    stage3 = {}
    for r in rs:
        final = combine(repeat_star, explore_star, r, K)
        stage3[r] = float(per_user_ndcg(final, target, K).mean())
    # Select r with the tie rule of select_r.
    r_star = select_r(stage3)
    # Record the time and the peak memory of stage 3.
    timings["stage3_combine"] = time.perf_counter() - t0
    peak_memory["stage3_combine"] = peak_memory_mib()

    # Fit the logistic-regression repeat ranker (R2) on the fit split and save its artifact.
    t0 = time.perf_counter()
    lr_bundle = tune_lr_repeat(history, target, config)
    lr_document = save_lr_repeat(lr_bundle, config, dataset)
    timings["lr_repeat"] = time.perf_counter() - t0
    peak_memory["lr_repeat"] = peak_memory_mib()

    # PREREG v1.5: R2' (same rows, same validation users, C selected separately), only with lr_repeat.itemcf_feature.
    icf_bundle = icf_document = None
    if icf_enabled(config):
        # Fit and save the ranker with the item-CF similarity feature (R2') under ICF_SUFFIX.
        t0 = time.perf_counter()
        icf_bundle = tune_lr_repeat(history, target, config, icf_neighbors=icf_neighbor_count(config))
        icf_document = save_lr_repeat(icf_bundle, config, dataset, suffix=ICF_SUFFIX)
        timings["lr_repeat_icf"] = time.perf_counter() - t0
        peak_memory["lr_repeat_icf"] = peak_memory_mib()

    # Select the user-CF settings and the slot counts of the integration combinations.
    # The fit-split lists built on the way are kept in fit_lists.
    t0 = time.perf_counter()
    fit_lists: dict[str, pd.DataFrame] = {}
    integration = tune_integration(history, target, pop, lr_bundle,
                                   {"repeat_star": repeat_star, "explore_star": explore_star}, config, keep_lists=fit_lists)
    timings["integration"] = time.perf_counter() - t0
    peak_memory["integration"] = peak_memory_mib()

    # Select the slot count of the combination of R2' and user-based CF (C4) if R2' was fitted.
    v15 = None
    if icf_bundle is not None:
        t0 = time.perf_counter()
        v15 = tune_v15(history, target, pop, lr_bundle, icf_bundle, fit_lists["usercf"], icf_document, config)
        timings["v15"] = time.perf_counter() - t0
        peak_memory["v15"] = peak_memory_mib()

    # Assemble the tuning document from the grids, selections, edge flags and sub-documents.
    document = {
        "profile": config["profile"],
        "dataset": dataset,
        "split": "fit",
        "k": K,
        "level": level,
        "label": run_label(config),
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
        "lr_repeat": lr_document,
        "integration": integration,
        "timing_seconds": timings,
        "peak_memory_mib": peak_memory,
    }
    # Attach the v15 block when it was computed.
    if v15 is not None:  # added only under lr_repeat.itemcf_feature; every other key is unchanged
        document["v15"] = v15
    return document


# ---------------------------------------------------------------------------
# W8: user-CF selection and slot re-selection of the new combinations (fit split only)
# ---------------------------------------------------------------------------
def tune_integration(
    history: pd.DataFrame,
    target: pd.DataFrame,
    pop: pd.Series,
    lr_bundle: dict[str, Any],
    lists: dict[str, pd.DataFrame],
    config: dict[str, Any],
    keep_lists: dict[str, pd.DataFrame] | None = None,
) -> dict[str, Any]:
    """Select (k, weighting) of user-CF by fit new-item Recall@10 and r of each combination by NDCG@10.

    ``history`` / ``target`` are the fit split, ``pop`` its basket popularity, ``lr_bundle`` the
    fit_select result of tune_lr_repeat and ``config`` supplies the usercf, integration.r and
    lr_repeat.C grids. ``lists`` holds the fit-split repeat_star and explore_star lists. For the
    combinations that use the LR ranker, repeat_lr is scored with the C* model
    fitted on the LR training users only and r is chosen on the LR validation
    users (same split and seed as fit_select); the other combination uses all
    fit users. Ties -> larger r (select_r). ``keep_lists``, when given, receives the fit-split
    user-CF list ("usercf") and the validation-user repeat_lr list ("repeat_lr") for reuse (PREREG v1.5).
    """
    # Run the user-CF grid and read the selected neighbour count k and weighting.
    timings: dict[str, float] = {}
    t0 = time.perf_counter()
    usercf = tune_usercf(history, target, config)
    k_star, weighting_star = int(usercf["selections"]["k"]), str(usercf["selections"]["weighting"])
    timings["usercf_grid"] = time.perf_counter() - t0

    # Build the fit-split user-CF list with the selected settings.
    t0 = time.perf_counter()
    block_size = int(config["usercf"]["block_size"])
    usercf_fit = ucf_list(history, k_star, weighting_star, K, pop, block_size=block_size)
    timings["usercf_list"] = time.perf_counter() - t0

    # Build the repeat list of the LR validation users with the train-only model.
    t0 = time.perf_counter()
    val_users = pd.Index(lr_bundle["val_users"])
    repeat_lr_val = lr_repeat_list(history, train_only_bundle(lr_bundle), K, pop, users=val_users)
    timings["repeat_lr_validation"] = time.perf_counter() - t0

    # Hand the two lists to the caller when requested.
    if keep_lists is not None:
        keep_lists["usercf"] = usercf_fit
        keep_lists["repeat_lr"] = repeat_lr_val

    # Collect the input lists, the validation-user target and the grid of r.
    t0 = time.perf_counter()
    inputs = {"repeat_lr": repeat_lr_val, "repeat_star": lists["repeat_star"],
              "explore_star": lists["explore_star"], "usercf": usercf_fit}
    val_target = target[target["user_id"].isin(val_users)]
    r_grid = [int(r) for r in config["integration"]["r"]]
    r_scores, r_selections, r_users = {}, {}, {}
    for name, (repeat_name, explore_name, uses_lr) in INTEGRATION_COMBOS.items():
        # Pick the repeat and explore lists of the combination.
        repeat, explore = inputs[repeat_name], inputs[explore_name]
        # Score LR combinations on the validation users and the other combination on all fit users.
        if uses_lr:
            explore = explore[explore["user_id"].isin(val_users)]
            scored_target, note = val_target, "LR validation users (train-only model)"
        else:
            scored_target, note = target, "all fit users"
        # Compute the mean NDCG@10 of the combined list for every r.
        scores = {r: float(per_user_ndcg(combine(repeat, explore, r, K), scored_target, K).mean()) for r in r_grid}
        # Select r and record the score curve and the scored users.
        r_selections[name] = select_r(scores)
        r_scores[name] = [{"r": r, "ndcg": value} for r, value in scores.items()]
        r_users[name] = {"users": note, "n_users": int(scored_target["user_id"].nunique())}
    timings["slot_grid"] = time.perf_counter() - t0

    # Read the C grid for the edge flags.
    c_grid = [float(c) for c in config["lr_repeat"]["C"]]
    # Return the selections, score curves, edge flags, timings and peak memory.
    return {
        "label": run_label(config),
        "usercf": usercf,
        "lr": {"C": lr_bundle["C"], "C_grid": c_grid, "at_grid_edge": lr_bundle["C"] in (min(c_grid), max(c_grid)),
               "n_train_users": lr_bundle["n_train_users"], "n_val_users": lr_bundle["n_val_users"]},
        "r_grid": r_grid,
        "r_scores": r_scores,
        "r_selection_users": r_users,
        "selections": {"k": k_star, "weighting": weighting_star, "C": lr_bundle["C"], "r": r_selections},
        "at_grid_edge": {"k": bool(usercf["at_grid_edge"]["k"]),
                         "C": lr_bundle["C"] in (min(c_grid), max(c_grid)),
                         "r": {name: r in (min(r_grid), max(r_grid)) for name, r in r_selections.items()}},
        "timing_seconds": timings,
        "process_peak_working_set_mib": peak_memory_mib(),
    }


def load_integration_selections(config: dict[str, Any], dataset: str) -> dict[str, Any]:
    """Read the W8 selections (k, weighting, C, r per combination) of ``dataset`` from the tuning JSON of ``config``'s profile."""
    # Read the tuning JSON of the profile.
    path = tuning_path(config, dataset)
    with open(path, encoding="utf-8") as handle:
        document = json.load(handle)
    # Fail when the document has no integration block.
    if not document.get("integration"):
        raise ValueError(f"{path} has no integration block; rerun tune")
    # Return a copy of the selections with the source path added.
    selections = dict(document["integration"]["selections"])
    selections["source"] = str(path)
    return selections


# ---------------------------------------------------------------------------
# PREREG v1.5 (W19): slot re-selection of C4 = combine(R2', E2, r) on the fit split
# ---------------------------------------------------------------------------
def tune_v15(
    history: pd.DataFrame,
    target: pd.DataFrame,
    pop: pd.Series,
    lr_bundle: dict[str, Any],
    icf_bundle: dict[str, Any],
    usercf_fit: pd.DataFrame,
    icf_document: dict[str, Any],
    config: dict[str, Any],
) -> dict[str, Any]:
    """The "v15" block of the tuning document: R2' (``icf_bundle``, fitted by tune_lr_repeat with icf_neighbors) and
    the slot count r of C4 selected exactly like C3's (tune_integration): repeat_lri scored with the C* model of R2'
    fitted on the LR training users only, NDCG@10 on the LR validation users of ``lr_bundle`` (the same users, checked),
    explore list = the fit-split user-CF list ``usercf_fit`` restricted to those users; ties -> larger r (select_r).
    ``icf_document`` is the save_lr_repeat summary of R2'. C3's selection is not touched.
    """
    # Require that R2' was selected on the same validation users as R2.
    if not np.array_equal(np.asarray(icf_bundle["val_users"]), np.asarray(lr_bundle["val_users"])):
        raise ValueError("tune_v15: R2' must be selected on the validation users of R2")
    # Build the R2' repeat list of the validation users with the train-only model.
    timings: dict[str, float] = {}
    t0 = time.perf_counter()
    val_users = pd.Index(icf_bundle["val_users"])
    repeat_lri_val = lr_repeat_list(history, train_only_bundle(icf_bundle), K, pop, users=val_users, source=SOURCE_ICF)
    timings["repeat_lri_validation"] = time.perf_counter() - t0

    # Restrict the user-CF list and the target to the validation users and read the grid of r.
    t0 = time.perf_counter()
    explore = usercf_fit[usercf_fit["user_id"].isin(val_users)]
    val_target = target[target["user_id"].isin(val_users)]
    r_grid = [int(r) for r in config["integration"]["r"]]
    r_scores, r_selections, r_users = {}, {}, {}
    for name in V15_COMBOS:
        # Compute the mean NDCG@10 of the combined list for every r.
        scores = {r: float(per_user_ndcg(combine(repeat_lri_val, explore, r, K), val_target, K).mean()) for r in r_grid}
        # Select r and record the score curve and the scored users.
        r_selections[name] = select_r(scores)
        r_scores[name] = [{"r": r, "ndcg": value} for r, value in scores.items()]
        r_users[name] = {"users": "LR validation users (train-only model)", "n_users": int(val_target["user_id"].nunique())}
    timings["slot_grid"] = time.perf_counter() - t0

    # Read the C grid for the edge flags.
    c_grid = [float(c) for c in config["lr_repeat"]["C"]]
    # Return the block with the selections, score curves, edge flags, timings and peak memory.
    return {
        "label": run_label(config),
        "prereg": V15_PREREG,
        "icf_neighbors": int(icf_bundle["icf_neighbors"]),
        "lr_repeat_icf": icf_document,
        "lr": {"C": icf_bundle["C"], "C_grid": c_grid, "at_grid_edge": icf_bundle["C"] in (min(c_grid), max(c_grid)),
               "n_train_users": icf_bundle["n_train_users"], "n_val_users": icf_bundle["n_val_users"],
               "same_validation_users_as_R2": True},
        "r_grid": r_grid,
        "r_scores": r_scores,
        "r_selection_users": r_users,
        "selections": {"C": icf_bundle["C"], "r": r_selections},
        "at_grid_edge": {"C": icf_bundle["C"] in (min(c_grid), max(c_grid)),
                         "r": {name: r in (min(r_grid), max(r_grid)) for name, r in r_selections.items()}},
        "timing_seconds": timings,
        "process_peak_working_set_mib": peak_memory_mib(),
    }


def load_v15_selections(config: dict[str, Any], dataset: str) -> dict[str, Any]:
    """Read the PREREG v1.5 selections (C of R2', r of C4) of ``dataset`` from the tuning JSON of ``config``'s profile."""
    # Read the tuning JSON of the profile.
    path = tuning_path(config, dataset)
    with open(path, encoding="utf-8") as handle:
        document = json.load(handle)
    # Fail when the document has no v15 block.
    if not document.get("v15"):
        raise ValueError(f"{path} has no v15 block; rerun tune with lr_repeat.itemcf_feature: true")
    # Return a copy of the selections with the source path added.
    selections = dict(document["v15"]["selections"])
    selections["source"] = str(path)
    return selections


# ---------------------------------------------------------------------------
# W6: logistic-regression repeat ranker (fit split only)
# ---------------------------------------------------------------------------
def tune_lr_repeat(
    history: pd.DataFrame, target: pd.DataFrame, config: dict[str, Any], icf_neighbors: int | None = None
) -> dict[str, Any]:
    """Build the training rows from the fit split (``history``, ``target``) and select C / refit
    (nbr.lr_repeat.fit_select) with the lr_repeat settings of ``config``. With ``icf_neighbors`` (PREREG v1.5, R2')
    the icf feature is appended and the bundle records ``icf_neighbors``."""
    # Read the lr_repeat settings and the decay gammas.
    settings = config["lr_repeat"]
    gammas = [float(g) for g in settings["decay_gammas"]]
    # Build the training rows, timing the step.
    t0 = time.perf_counter()
    X, y, groups = build_training(history, target, gammas, icf_neighbors=icf_neighbors)
    build_seconds = time.perf_counter() - t0
    # Select C on the validation users and refit (fit_select).
    bundle = fit_select(X, y, groups, [float(c) for c in settings["C"]], float(settings["val_fraction"]),
                        int(settings["seed"]), int(settings["max_iter"]), K)
    # Record the decay gammas, the neighbour count when given and the build time in the bundle.
    bundle["decay_gammas"] = gammas
    if icf_neighbors is not None:
        bundle["icf_neighbors"] = int(icf_neighbors)
    bundle["timing_seconds"]["build_training"] = build_seconds
    return bundle


def lr_artifact_path(config: dict[str, Any], dataset: str, suffix: str = "") -> Path:
    """artifacts/<profile>/<dataset>_lr_repeat<suffix>.joblib under the repo_root of ``config`` (``suffix`` "_icf" for R2')."""
    return Path(config["repo_root"]) / "artifacts" / str(config["profile"]) / f"{dataset}_lr_repeat{suffix}.joblib"


def sha256_of(path: Path) -> str:
    """Hex SHA-256 digest of the file at ``path`` (hashlib.sha256 over the bytes, read in 1 MiB blocks)."""
    # Start an empty digest.
    digest = hashlib.sha256()
    # Feed the file to the digest in 1 MiB blocks.
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    # Return the digest as a hex string.
    return digest.hexdigest()


def save_lr_repeat(bundle: dict[str, Any], config: dict[str, Any], dataset: str, suffix: str = "") -> dict[str, Any]:
    """Persist the fitted ``bundle`` of ``dataset`` with joblib.dump (pickle-based) at
    lr_artifact_path(config, dataset, suffix) and return the JSON summary (R2' bundles also store icf_neighbors)."""
    # Resolve the artifact path and create its directory.
    path = lr_artifact_path(config, dataset, suffix)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Keep the bundle entries that are stored in the artifact.
    stored = {key: bundle[key] for key in ("scaler", "model", "C", "features", "decay_gammas", "coefficients",
                                            "intercept")}
    # Store the neighbour count of the icf feature when the bundle has one.
    if "icf_neighbors" in bundle:
        stored["icf_neighbors"] = int(bundle["icf_neighbors"])
    # Write the artifact with joblib.
    joblib.dump(stored, path)
    # Start the summary with the neighbour count when the bundle has one.
    summary = {} if "icf_neighbors" not in bundle else {"icf_neighbors": int(bundle["icf_neighbors"])}
    # Return the summary with the validation table, fit statistics, artifact path and hash.
    return {
        **summary,
        "C_grid": [row["C"] for row in bundle["validation"]],
        "validation": bundle["validation"],
        "C": bundle["C"],
        "coefficients": [{"feature": name, "coef": value} for name, value in bundle["coefficients"].items()],
        "intercept": bundle["intercept"],
        "n_iter_refit": bundle["n_iter_refit"],
        "n_rows": bundle["n_rows"],
        "positive_rate": bundle["positive_rate"],
        "n_train_users": bundle["n_train_users"],
        "n_val_users": bundle["n_val_users"],
        "val_fraction": bundle["val_fraction"],
        "seed": bundle["seed"],
        "max_iter": bundle["max_iter"],
        "decay_gammas": bundle["decay_gammas"],
        "artifact": str(path),
        "sha256": sha256_of(path),
        "timing_seconds": bundle["timing_seconds"],
    }


def load_lr_repeat(config: dict[str, Any], dataset: str, suffix: str = "") -> dict[str, Any]:
    """Load the bundle of ``dataset`` written by save_lr_repeat (artifacts/<profile>/<dataset>_lr_repeat<suffix>.joblib
    under the repo_root of ``config``; ``suffix`` "_icf" for R2')."""
    # Resolve the artifact path and fail when the file is missing.
    path = lr_artifact_path(config, dataset, suffix)
    if not path.exists():
        raise FileNotFoundError(f"missing lr_repeat artifact {path}; run tune first")
    # Load the bundle and add its path and SHA-256 digest.
    bundle = joblib.load(path)
    bundle["artifact"] = str(path)
    bundle["sha256"] = sha256_of(path)
    return bundle


# ---------------------------------------------------------------------------
# report paths, tuning entry and stored selections
# ---------------------------------------------------------------------------
def tuning_path(config: dict[str, Any], dataset: str) -> Path:
    """reports/metrics/<profile>_<dataset>_tuning.json under the repo_root of ``config``."""
    return Path(config["repo_root"]) / "reports" / "metrics" / f"{config['profile']}_{dataset}_tuning.json"


def eval_path(config: dict[str, Any], dataset: str, transfer_from: str | None) -> Path:
    """reports/metrics/<profile>_<dataset>_eval.json (``_eval_transfer.json`` when ``transfer_from`` is set) under the repo_root of ``config``."""
    # Choose the file suffix by whether the selections are transferred from another dataset.
    suffix = "_eval_transfer.json" if transfer_from else "_eval.json"
    return Path(config["repo_root"]) / "reports" / "metrics" / f"{config['profile']}_{dataset}{suffix}"


def run_tune(config: dict[str, Any], dataset: str) -> Path:
    """tune(config, dataset) and write the document to tuning_path(config, dataset); returns the written path."""
    # Run the selection on the fit split.
    document = tune(config, dataset)
    # Write the document and return the written path.
    return write_json(tuning_path(config, dataset), document, config)


def load_selections(config: dict[str, Any], dataset: str) -> dict[str, Any]:
    """Read the selected values of ``dataset`` from reports/metrics/<profile>_<dataset>_tuning.json (profile of ``config``)."""
    # Read the tuning JSON of the profile.
    path = tuning_path(config, dataset)
    with open(path, encoding="utf-8") as handle:
        document = json.load(handle)
    # Return a copy of the selections with the source path added.
    selections = dict(document["selections"])
    selections["source"] = str(path)
    return selections


# ---------------------------------------------------------------------------
# evaluation
# ---------------------------------------------------------------------------
def decision(ci: list[float]) -> str:
    """PREREG decision wording for the confidence interval ``ci`` = [lo, hi] of (model - comparator)."""
    # Unpack the lower and upper bound.
    lo, hi = ci
    # An interval entirely above zero is reported as an improvement.
    if lo > 0:
        return "improvement observed"
    # An interval entirely below zero is reported as lower.
    if hi < 0:
        return "lower"
    # Any interval that contains zero establishes no difference.
    return "no difference established"


def _compare(a: pd.Series, b: pd.Series, config: dict[str, Any], *, primary: bool) -> dict[str, Any]:
    """paired_bootstrap(comparator a, model b) with decision wording for primary tests."""
    # Read the bootstrap settings and the CI levels.
    boot = config["bootstrap"]
    levels = [float(level) for level in boot["levels"]]
    # Run the paired bootstrap over users.
    result = paired_bootstrap(a, b, int(boot["n"]), int(boot["seed"]), levels)
    # Collect the user count, the two means, the difference and one CI per level.
    out = {
        "n_users": result["n_users"],
        "comparator_mean": result["mean_a"],
        "model_mean": result["mean_b"],
        "diff": result["diff"],
        "ci": {str(level): result["ci"][level] for level in levels},
    }
    # Add the decision wording per level for primary tests.
    if primary:
        out["decision"] = {str(level): decision(result["ci"][level]) for level in levels}
    return out


# ---------------------------------------------------------------------------
# model lists
# ---------------------------------------------------------------------------
def build_lists(
    history: pd.DataFrame,
    pop: pd.Series,
    selections: dict[str, Any],
    config: dict[str, Any],
    *,
    include_itemcf: bool = True,
    lr_bundle: dict[str, Any] | None = None,
    integration: dict[str, Any] | None = None,
    skip_l3: bool = False,
    peak_memory: dict[str, float | None] | None = None,
    icf_bundle: dict[str, Any] | None = None,
    v15: dict[str, Any] | None = None,
) -> tuple[dict[str, pd.DataFrame], dict[str, float]]:
    """All model lists (k = 10) from the eval ``history``; returns (lists, timings).

    ``pop`` is the basket popularity of the history, ``selections`` the tuned (gamma, gamma_c,
    alpha, r), ``config`` supplies the explore level, item-CF neighbour count and user-CF block
    size, and ``include_itemcf`` adds the item-CF lists (itemcf_flat, explore_cf, final_cf).

    ``skip_l3`` (W8b) leaves out the explore_L3 list. ``peak_memory``, when given, receives the
    process peak working set (MiB) after each list.

    With ``lr_bundle`` (W6) the lists repeat_lr and final_lr = combine(repeat_lr, explore_star, r*) are added.
    With ``integration`` selections (W8, requires ``lr_bundle``) the user-CF list and the combinations
    final_lr_ucf, final_star_ucf and final_lr_r with their re-selected r are added.

    With ``icf_bundle`` (R2', PREREG v1.5; requires ``integration`` and the ``v15`` tuning selections) the lists
    repeat_lri and final_lri_ucf = combine(repeat_lri, usercf, v15["r"]["final_lri_ucf"]) are added; the item
    neighbours of ``history`` computed for the item-CF lists are shared with the icf feature when the counts agree.
    """
    # Read the explore level and the tuned values.
    level = config["explore"]["level"]
    gamma, gamma_c, alpha, r = (
        float(selections["gamma"]), float(selections["gamma_c"]), float(selections["alpha"]), int(selections["r"])
    )
    # List the users of the history and start the outputs.
    users = history["user_id"].drop_duplicates()
    lists: dict[str, pd.DataFrame] = {}
    timings: dict[str, float] = {}

    def timed(name, function):
        '''Build the list ``name`` with ``function``; record its time and, if asked, peak memory.'''
        # Time the call that builds the list.
        t0 = time.perf_counter()
        lists[name] = function()
        timings[name] = time.perf_counter() - t0
        # Record the process peak working set after the list when requested.
        if peak_memory is not None:
            peak_memory[name] = peak_memory_mib()

    # Build the popularity and personal-frequency baselines.
    timed("popularity", lambda: popularity_list(history, users, K, pop, exclude_history=False))
    timed("personal_frequency", lambda: personal_frequency_list(history, K, pop))
    # Build the personal-frequency repeat list (R5) and the decayed-frequency repeat list.
    timed("repeat_pf", lambda: repeat_list(history, 1.0, K, pop))
    timed("repeat_star", lambda: repeat_list(history, gamma, K, pop))
    # Build the popularity list excluding history (flat_pop) and the hierarchical explore list.
    timed("flat_explore", lambda: popularity_list(history, users, K, pop, exclude_history=True))
    timed("explore_star", lambda: explore_list(history, K, gamma_c, alpha, level))
    # Combine repeat and explore with the selected r and with all k slots given to repeat.
    timed("final", lambda: combine(lists["repeat_star"], lists["explore_star"], r, K))
    timed("final_r10", lambda: combine(lists["repeat_star"], lists["explore_star"], K, K))
    # Build the explore list at level L3 unless it is skipped.
    if not skip_l3:
        timed("explore_L3", lambda: explore_list(history, K, gamma_c, alpha, "L3"))
    # Read the item-CF neighbour count.
    neighbors = None
    n_neighbors = int(config["explore"]["itemcf"]["neighbors"])
    if include_itemcf:
        # Compute the item neighbours and the item-CF scores of the history.
        t0 = time.perf_counter()
        neighbors = item_neighbors(history, n_neighbors)
        cf = cf_scores(history, neighbors)
        timings["itemcf_scores"] = time.perf_counter() - t0
        # Build the flat item-CF list (E3) and the item-CF explore list within category (E4).
        # Combine the repeat list with the item-CF explore list at the selected r.
        timed("itemcf_flat", lambda: cf_list(history, neighbors, K, pop))
        timed("explore_cf", lambda: explore_list_cf(history, cf, K, gamma_c, alpha, level))
        timed("final_cf", lambda: combine(lists["repeat_star"], lists["explore_cf"], r, K))
    if lr_bundle is not None:
        # Build the LR repeat list and combine it with the explore list at the selected r.
        timed("repeat_lr", lambda: lr_repeat_list(history, lr_bundle, K, pop))
        timed("final_lr", lambda: combine(lists["repeat_lr"], lists["explore_star"], r, K))
    if integration is not None:
        # Refuse the integration lists without the LR bundle.
        if lr_bundle is None:
            raise ValueError("integration lists need the lr_repeat bundle")
        # Build the user-based CF list (E2) with the selected k and weighting.
        block_size = int(config["usercf"]["block_size"])
        timed("usercf", lambda: ucf_list(history, int(integration["k"]), str(integration["weighting"]), K, pop,
                                         block_size=block_size))
        for name, (repeat_name, explore_name, _) in INTEGRATION_COMBOS.items():
            # Combine each integration pair with its re-selected r.
            slots = int(integration["r"][name])
            timed(name, lambda rn=repeat_name, en=explore_name, rr=slots: combine(lists[rn], lists[en], rr, K))
    if icf_bundle is not None:
        # Refuse the PREREG v1.5 lists without the integration and v15 selections.
        if integration is None or v15 is None:
            raise ValueError("the PREREG v1.5 lists need the integration selections and the v15 slot selection")
        # Share the item neighbours when the icf feature uses the same neighbour count.
        shared = neighbors if neighbors is not None and int(icf_bundle["icf_neighbors"]) == n_neighbors else None
        # Build the R2' repeat list.
        timed("repeat_lri", lambda: lr_repeat_list(history, icf_bundle, K, pop, source=SOURCE_ICF, neighbors=shared))
        for name, (repeat_name, explore_name) in V15_COMBOS.items():
            # Combine each PREREG v1.5 pair with its re-selected r.
            slots = int(v15["r"][name])
            timed(name, lambda rn=repeat_name, en=explore_name, rr=slots: combine(lists[rn], lists[en], rr, K))
    return lists, timings


# ---------------------------------------------------------------------------
# history-length breakdown
# ---------------------------------------------------------------------------
def history_lengths(history: pd.DataFrame) -> pd.Series:
    """Number of distinct baskets per user in the history."""
    return history[["user_id", "basket_seq"]].drop_duplicates().groupby("user_id").size()


def _bin_labels(edges: list[int]) -> list[tuple[str, float, float]]:
    '''(label, lo, hi) of the bins with upper ``edges``: (0,e1], ... and an open last bin.'''
    # Put 0 before and infinity after the upper edges.
    bounds = [0] + edges + [math.inf]
    labels = []
    for lo, hi in zip(bounds[:-1], bounds[1:]):
        # Name the bin by its bounds and keep the bounds with the label.
        name = f"({lo},inf)" if math.isinf(hi) else f"({lo},{hi}]"
        labels.append((name, lo, hi))
    return labels


def by_history_length(
    metrics: dict[str, tuple[pd.Series, pd.Series]], lengths: pd.Series, dataset: str, config: dict[str, Any]
) -> list[dict[str, Any]]:
    """Per history-length bin: user counts and comparator/model means with a 95% CI of the diff.

    ``metrics`` maps a hypothesis name to (comparator, model) per-user Series; ``lengths`` maps
    user_id to the number of history baskets (history_lengths); ``dataset`` selects HISTORY_BINS
    and ``config`` the bootstrap settings.
    """
    # Read the bootstrap settings.
    boot = config["bootstrap"]
    rows = []
    for name, lo, hi in _bin_labels(HISTORY_BINS[dataset]):
        # Select the users whose history length falls in the bin.
        users = lengths[(lengths > lo) & (lengths <= hi)].index
        entry: dict[str, Any] = {"bin": name, "n_users": int(len(users))}
        for key, (a, b) in metrics.items():
            # Restrict the comparison to the bin's users that have a metric value.
            index = a.index.intersection(users)
            # Record None when the bin has no scored user.
            if len(index) == 0:
                entry[key] = None
                continue
            # Bootstrap the difference at the 95% level and store the means and the CI.
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


# ---------------------------------------------------------------------------
# eval-split evaluation
# ---------------------------------------------------------------------------
def evaluate(
    config: dict[str, Any],
    dataset: str,
    selections: dict[str, Any],
    *,
    allow_holdout_eval: bool = False,
    transfer_from: str | None = None,
    include_itemcf: bool = True,
    lr_bundle: dict[str, Any] | None = None,
    integration: dict[str, Any] | None = None,
    skip_l3: bool = False,
    include_policy: bool = False,
    icf_bundle: dict[str, Any] | None = None,
    v15: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Evaluate all models on the eval split of ``dataset`` under ``config`` with the given (fixed) ``selections``.

    ``allow_holdout_eval`` is passed to the holdout guard (CLI --final); ``transfer_from`` is only
    recorded in the document (the caller loads the selections of that dataset); ``include_itemcf``
    adds the item-CF lists and H5.

    ``skip_l3`` (W8b) leaves out the explore_L3 list and its secondary comparison; the
    document then carries "skipped": ["explore_L3"]. Every other entry is unchanged.

    ``lr_bundle`` (W6) adds repeat_lr / final_lr and the H6, H7 comparisons under the
    "lr_repeat" key; the H1-H5 entries do not depend on it. ``integration`` (W8, the
    selections of tune_integration) adds usercf and the combinations with H8, H9 under
    the "integration" key; H1-H7 do not depend on it.

    ``include_policy`` (W17, CLI --policy; requires ``lr_bundle`` and ``integration``) adds the PREREG v1.4
    exploration-policy block (nbr.policy.evaluate_policy: P_T vs C3, S vs flat popularity; H11-H13) under the
    "policy" key (None otherwise); no other entry depends on it. With the v1.5 lists present it also adds the block
    "policy_c4" (PREREG v1.5.1: the same policy with C4 as the base, same keys, label policy_label(config)).

    ``icf_bundle`` with ``v15`` (W19, PREREG v1.5; requires ``lr_bundle`` and ``integration``) adds the lists repeat_lri
    (R2') and final_lri_ucf (C4) and the "v15" block (nbr.v15.evaluate_v15: H14, H15 and the secondary comparisons at
    the 95 % and 97.5 % levels). Without them the document has no "v15" key and every entry is unchanged.
    """
    # Reject argument combinations that lack a required bundle or selection.
    if include_policy and (lr_bundle is None or integration is None):
        raise ValueError("the policy block needs the lr_repeat bundle and the integration selections (no --no-lr / --no-integration)")
    if (icf_bundle is None) != (v15 is None):
        raise ValueError("the v15 block needs both the R2' bundle and the v15 selections")
    if icf_bundle is not None and (lr_bundle is None or integration is None):
        raise ValueError("the v15 block needs the lr_repeat bundle and the integration selections (no --no-lr / --no-integration)")
    # Load the eval split and read the explore level, timing the step.
    timings: dict[str, float] = {}
    t0 = time.perf_counter()
    data = prepare(config, dataset, "eval", allow_holdout_eval=allow_holdout_eval)
    history, target = data["history"], data["target"]
    repeat_target, new_target, pop = data["repeat_target"], data["new_target"], data["pop"]
    level = config["explore"]["level"]
    timings["prepare"] = time.perf_counter() - t0
    peak_memory: dict[str, float | None] = {"prepare": peak_memory_mib()}

    # Build all model lists and record their timings.
    lists, list_timings = build_lists(history, pop, selections, config, include_itemcf=include_itemcf,
                                      lr_bundle=lr_bundle, integration=integration, skip_l3=skip_l3,
                                      peak_memory=peak_memory, icf_bundle=icf_bundle, v15=v15)
    timings.update({f"list_{name}": seconds for name, seconds in list_timings.items()})

    # Compute the mean Recall@10 and NDCG@10 of every list against the full target.
    t0 = time.perf_counter()
    list_metrics = {
        name: {
            "recall": float(per_user_recall(frame, target, K).mean()),
            "ndcg": float(per_user_ndcg(frame, target, K).mean()),
        }
        for name, frame in lists.items()
    }

    def recall_on(name, frame_target):
        '''Per-user Recall@10 of the list ``name`` against ``frame_target``.'''
        return per_user_recall(lists[name], frame_target, K)

    def ndcg_all(name):
        '''Per-user NDCG@10 of the list ``name`` against the full target.'''
        return per_user_ndcg(lists[name], target, K)

    # Primary hypotheses: comparator first, model second (diff = model - comparator).
    # Rank the categories of every user by theta and of the new-target users by population share.
    theta, p_c = category_profile(history, float(selections["gamma_c"]), float(selections["alpha"]), level)
    new_users = new_target["user_id"].drop_duplicates()
    model_rank = category_ranking(theta, K)
    population_rank = population_category_ranking(p_c, new_users, K)
    # Pair the comparator and model per-user series of each primary hypothesis.
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
    # Name the comparator, model, metric and user group of each primary hypothesis.
    labels = {
        "H1_repeat_recall": ("repeat_pf", "repeat_star", "repeat Recall@10", "users with repeat target"),
        "H2_new_recall": ("flat_explore", "explore_star", "new-item Recall@10", "users with new target"),
        "H3_ndcg": ("personal_frequency", "final", "NDCG@10", "all users"),
        "H3_recall": ("personal_frequency", "final", "Recall@10", "all users"),
        "H4_category_hit": ("population_category_ranking", "category_ranking", "category_hit@10", "users with new target"),
    }
    # Bootstrap each primary pair and add the decision wording.
    primary = {}
    for key, (a, b) in primary_series.items():
        comparator, model, metric, users = labels[key]
        primary[key] = {"comparator": comparator, "model": model, "metric": metric, "users": users,
                        **_compare(a, b, config, primary=True)}

    # H5 (PREREG v1.2, generated on development data) compares flat item-CF with flat popularity.
    # The flat popularity comparator excludes the history.
    # The metric is new-item Recall@10 on users with >= 1 new target item, with the 95% CI only.
    h5 = None
    if "itemcf_flat" in lists:
        # Bootstrap the pair at the 95% level and store the result with its decision.
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
        # Skip variants whose list was not built.
        if variant not in lists:
            continue
        # Compare with final on NDCG@10 and Recall@10, or with explore_star on new-item Recall@10.
        if reference == "final":
            pairs = {"ndcg": (ndcg_all("final"), ndcg_all(variant)),
                     "recall": (recall_on("final", target), recall_on(variant, target))}
        else:
            pairs = {"new_recall": (recall_on("explore_star", new_target), recall_on(variant, new_target))}
        # Bootstrap each pair without decision wording.
        for metric, (a, b) in pairs.items():
            secondary[f"{variant}_vs_{reference}_{metric}"] = {
                "comparator": reference, "model": variant, "metric": metric, **_compare(a, b, config, primary=False)
            }
    # Compare final with popularity on NDCG@10 and Recall@10.
    for metric, (a, b) in {"ndcg": (ndcg_all("popularity"), ndcg_all("final")),
                           "recall": (recall_on("popularity", target), recall_on("final", target))}.items():
        secondary[f"final_vs_popularity_{metric}"] = {
            "comparator": "popularity", "model": "final", "metric": metric, **_compare(a, b, config, primary=False)
        }
    timings["metrics_and_bootstrap"] = time.perf_counter() - t0

    # Break four of the primary comparisons down by history-length bin.
    t0 = time.perf_counter()
    lengths = history_lengths(history)
    breakdown = by_history_length(
        {key: primary_series[key] for key in ("H1_repeat_recall", "H2_new_recall", "H3_ndcg", "H3_recall")},
        lengths, dataset, config,
    )
    timings["by_history_length"] = time.perf_counter() - t0

    # W6: logistic-regression repeat ranker (H6: repeat_lr vs repeat_star; H7: final_lr vs final).
    lr_repeat = None
    if lr_bundle is not None:
        t0 = time.perf_counter()
        # Pair the comparator and model per-user series of H6 and H7.
        lr_series = {
            "H6_repeat_recall": (recall_on("repeat_star", repeat_target), recall_on("repeat_lr", repeat_target)),
            "H7_ndcg": (ndcg_all("final"), ndcg_all("final_lr")),
            "H7_recall": (recall_on("final", target), recall_on("final_lr", target)),
        }
        # Name the comparator, model, metric and user group of each pair.
        lr_labels = {
            "H6_repeat_recall": ("repeat_star", "repeat_lr", "repeat Recall@10", "users with repeat target"),
            "H7_ndcg": ("final", "final_lr", "NDCG@10", "all users"),
            "H7_recall": ("final", "final_lr", "Recall@10", "all users"),
        }
        # Bootstrap each pair and add the decision wording.
        lr_primary = {}
        for key, (a, b) in lr_series.items():
            comparator, model, metric, users = lr_labels[key]
            lr_primary[key] = {"comparator": comparator, "model": model, "metric": metric, "users": users,
                               **_compare(a, b, config, primary=True)}
        # Bootstrap the secondary comparisons on repeat Recall@10 or NDCG@10.
        lr_secondary = {}
        for key, (reference, variant, metric) in LR_SECONDARY.items():
            if metric == "repeat_recall":
                a, b = recall_on(reference, repeat_target), recall_on(variant, repeat_target)
            else:
                a, b = ndcg_all(reference), ndcg_all(variant)
            lr_secondary[key] = {"comparator": reference, "model": variant, "metric": metric,
                                 **_compare(a, b, config, primary=False)}
        # Break the comparisons down by history-length bin.
        lr_breakdown = by_history_length(lr_series, lengths, dataset, config)
        timings["lr_repeat_metrics"] = time.perf_counter() - t0
        # Assemble the lr_repeat block with the artifact identity, C and feature names.
        lr_repeat = {
            "label": run_label(config),
            "artifact": lr_bundle.get("artifact"),
            "sha256": lr_bundle.get("sha256"),
            "C": lr_bundle["C"],
            "features": list(lr_bundle["features"]),
            "primary": lr_primary,
            "secondary": lr_secondary,
            "by_history_length": lr_breakdown,
        }

    # W8: user-CF explore list and the re-selected combinations.
    # H8 compares usercf with flat_explore, and H9 compares final_lr_ucf with final_lr.
    # Secondary comparisons are CI only, except the PREREG v1.3 H7' pair (INTEGRATION_DECISION).
    # The H7' pair adds the decision wording.
    integration_result = None
    if integration is not None:
        t0 = time.perf_counter()
        # Pair the comparator and model per-user series of H8 and H9.
        int_series = {
            "H8_new_recall": (recall_on("flat_explore", new_target), recall_on("usercf", new_target)),
            "H9_ndcg": (ndcg_all("final_lr"), ndcg_all("final_lr_ucf")),
            "H9_recall": (recall_on("final_lr", target), recall_on("final_lr_ucf", target)),
        }
        # Name the comparator, model, metric and user group of each pair.
        int_labels = {
            "H8_new_recall": ("flat_explore", "usercf", "new-item Recall@10", "users with new target"),
            "H9_ndcg": ("final_lr", "final_lr_ucf", "NDCG@10", "all users"),
            "H9_recall": ("final_lr", "final_lr_ucf", "Recall@10", "all users"),
        }
        # Bootstrap each pair and add the decision wording.
        int_primary = {}
        for key, (a, b) in int_series.items():
            comparator, model, metric, users = int_labels[key]
            int_primary[key] = {"comparator": comparator, "model": model, "metric": metric, "users": users,
                                **_compare(a, b, config, primary=True)}
        # Bootstrap the secondary comparisons whose lists exist, choosing the series by metric.
        int_secondary = {}
        for key, (reference, variant, metric) in INTEGRATION_SECONDARY.items():
            if reference not in lists or variant not in lists:
                continue
            if metric == "new_recall":
                a, b = recall_on(reference, new_target), recall_on(variant, new_target)
            elif metric == "recall":
                a, b = recall_on(reference, target), recall_on(variant, target)
            else:
                a, b = ndcg_all(reference), ndcg_all(variant)
            int_secondary[key] = {"comparator": reference, "model": variant, "metric": metric,
                                  **_compare(a, b, config, primary=key in INTEGRATION_DECISION)}
        # Break the comparisons down by history-length bin.
        int_breakdown = by_history_length(int_series, lengths, dataset, config)
        timings["integration_metrics"] = time.perf_counter() - t0
        # Assemble the integration block.
        integration_result = {
            "label": run_label(config),
            "selections": integration,
            "primary": int_primary,
            "secondary": int_secondary,
            "by_history_length": int_breakdown,
            "process_peak_working_set_mib": peak_memory_mib(),
        }

    # W17 (PREREG v1.4): per-user exploration-slot policy P_T vs C3.
    # The separate list S is compared with flat popularity.
    # The hypotheses are H11 (non-inferiority), H12 and H13.
    # Secondary metrics, bins and variants are included.
    # Lists above are unchanged.
    policy_result = policy_c4_result = None
    if include_policy:
        t0 = time.perf_counter()
        # Evaluate the policy on the combination of R2 and user-based CF (C3) with its r.
        policy_result = evaluate_policy(lists, data, lengths, HISTORY_BINS[dataset], config,
                                        int(integration["r"]["final_lr_ucf"]), label=policy_label(config))
        timings["policy_metrics"] = time.perf_counter() - t0
        peak_memory["policy"] = peak_memory_mib()
        if icf_bundle is not None:  # PREREG v1.5.1: the same policy on C4 (secondary replication)
            # Evaluate the same policy with C4 as the base and its r.
            t0 = time.perf_counter()
            policy_c4_result = evaluate_policy(lists, data, lengths, HISTORY_BINS[dataset], config,
                                               int(v15["r"]["final_lri_ucf"]), label=policy_label(config), base="C4")
            timings["policy_c4_metrics"] = time.perf_counter() - t0
            peak_memory["policy_c4"] = peak_memory_mib()

    # W19 (PREREG v1.5): R2' and C4 against C3 / personal frequency (H14 non-inferiority, H15).
    # The secondary comparisons are included.
    # Lists above are unchanged and the key is present only when the feature is enabled.
    v15_result = None
    if icf_bundle is not None:
        t0 = time.perf_counter()
        # Evaluate the v15 block from the lists, the R2' bundle and the v15 selections.
        v15_result = evaluate_v15(lists, data, lengths, HISTORY_BINS[dataset], config, icf_bundle, v15,
                                  label=run_label(config))
        timings["v15_metrics"] = time.perf_counter() - t0
        peak_memory["v15"] = peak_memory_mib()

    # Assemble the eval document.
    document = {
        "profile": config["profile"],
        "dataset": dataset,
        "split": "eval",
        "k": K,
        "level": level,
        "label": run_label(config),
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
        "lr_repeat": lr_repeat,
        "integration": integration_result,
        "policy": policy_result,
        "skipped": ["explore_L3"] if skip_l3 else [],
        "timing_seconds": timings,
        "peak_memory_mib": {**peak_memory, "end": peak_memory_mib()},
    }
    # Attach the v15 and policy_c4 blocks when they were computed.
    if v15_result is not None:
        document["v15"] = v15_result
    if policy_c4_result is not None:
        document["policy_c4"] = policy_c4_result
    return document


# ---------------------------------------------------------------------------
# evaluation entry
# ---------------------------------------------------------------------------
def run_evaluate(
    config: dict[str, Any],
    dataset: str,
    *,
    transfer_from: str | None = None,
    allow_holdout_eval: bool = False,
    include_itemcf: bool = True,
    include_lr: bool = True,
    include_integration: bool = True,
    skip_l3: bool = False,
    include_policy: bool = False,
) -> Path:
    """Evaluate ``dataset`` with the selections of ``transfer_from or dataset`` and write the eval JSON.

    ``config`` is the profile configuration, ``allow_holdout_eval`` the holdout-guard override
    (--final); ``include_itemcf`` / ``include_lr`` / ``include_integration`` add the item-CF lists,
    the lr_repeat artifact (H6, H7) and the W8 selections (H8, H9); ``skip_l3`` leaves out
    explore_L3; ``include_policy`` (--policy) adds the PREREG v1.4 policy block. With lr_repeat.itemcf_feature
    in ``config`` (PREREG v1.5) the R2' artifact and the v15 selections of ``source`` are loaded as well and the
    "v15" block is written. Returns the written path.
    """
    # Take the selections of the transfer dataset when given, else of the evaluated dataset.
    source = transfer_from or dataset
    selections = load_selections(config, source)
    # Load the LR bundle and the integration selections unless they are switched off.
    lr_bundle = load_lr_repeat(config, source) if include_lr else None
    integration = load_integration_selections(config, source) if include_lr and include_integration else None
    # Load the R2' bundle and the v15 selections when integration is on and the icf feature is set.
    with_v15 = integration is not None and icf_enabled(config)
    icf_bundle = load_lr_repeat(config, source, suffix=ICF_SUFFIX) if with_v15 else None
    v15 = load_v15_selections(config, source) if with_v15 else None
    # Evaluate with the loaded selections.
    document = evaluate(config, dataset, selections, allow_holdout_eval=allow_holdout_eval,
                        transfer_from=transfer_from, include_itemcf=include_itemcf, lr_bundle=lr_bundle,
                        integration=integration, skip_l3=skip_l3, include_policy=include_policy,
                        icf_bundle=icf_bundle, v15=v15)
    # Write the eval document and return its path.
    return write_json(eval_path(config, dataset, transfer_from), document, config)
