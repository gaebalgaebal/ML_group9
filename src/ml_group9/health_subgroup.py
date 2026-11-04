"""Health-attribute subgroups: users who bought a health-attribute item, analysed on their own.

Subgroup membership, pi, pi_bar, n_a and n_C all come from one split's history; targets are only used
for metrics. This is a post-hoc subgroup analysis chosen after the repeat-explore results.
Entry points: `python -m ml_group9 tune-hs` and `eval-hs`, which write
reports/metrics/<profile>_hs_tuning.json and <profile>_hs_instacart.json.
"""

from __future__ import annotations

import json
import sys
import time
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse

from .attributes import applicable_categories, category_sets, exclude_nonfood_categories
from .data import load_split
from .metrics import paired_bootstrap, subset_recall
from .pipeline import _write_json
from .re_pipeline import (
    GammaState,
    ScopeState,
    SplitState,
    baseline_scores,
    build_scope,
    build_state,
    candidate_lists,
    popularity_lists,
    recommend,
    score_lists,
    tuning_path,
)
from .repeat_explore import (
    Interactions,
    PreferenceModel,
    build_preferences,
    cf_scores,
    explore_candidates,
    preference_candidates,
    repeat_ranking,
    repeat_scores,
)


DATASET = "instacart"
COMBINED = "combined"
# Key of the union of all health-attribute items in the combined analysis.
ANY_ATTRIBUTE = "any"

Lists = dict[str, list[str]]


# pi - pi_bar is a float difference; a gap that equals min_gap on paper must still reach it.
GAP_TOLERANCE = 1e-9


def health_categories(config: dict[str, object], dataset: str, items: pd.DataFrame) -> dict[str, set[str]]:
    """C_a of each health attribute, with non-food aisles removed (same rule as the repeat-explore state)."""
    settings = config["attributes"]
    names = list(config["health_subgroup"]["attributes"])
    table = applicable_categories(items, names, int(settings["min_with"]), int(settings["min_without"]))
    food_table = exclude_nonfood_categories(table, items, list(settings["exclude_departments"][dataset]))
    sets = category_sets(food_table)
    return {name: sets.get(name, set()) for name in names}


def health_preferences(
    config: dict[str, object],
    dataset: str,
    items: pd.DataFrame,
    history: pd.DataFrame,
    interactions: Interactions,
) -> PreferenceModel:
    """Preference model over the health attributes only; pi_bar uses every history user."""
    categories = health_categories(config, dataset, items)
    return build_preferences(history, items, interactions, categories, float(config["attributes"]["smoothing"]))


def select_attributes(model: PreferenceModel, names: list[str]) -> PreferenceModel:
    """Model restricted to the given attributes; per-attribute statistics are unchanged."""
    columns = [model.attributes.index(name) for name in names]
    return PreferenceModel(
        list(names),
        model.prior[columns],
        model.pi[:, columns],
        model.n_c[:, columns],
        model.item_flag[:, columns],
        model.item_applicable[:, columns],
        None if model.n_a is None else model.n_a[:, columns],
    )


def select_users(model: PreferenceModel, rows: np.ndarray) -> PreferenceModel:
    """Model restricted to the given user rows (local row r is global row rows[r])."""
    return PreferenceModel(
        model.attributes,
        model.prior,
        model.pi[rows],
        model.n_c[rows],
        model.item_flag,
        model.item_applicable,
        None if model.n_a is None else model.n_a[rows],
    )


@dataclass
class Subgroups:
    """users x attributes membership masks aligned to interaction user indices."""

    attributes: list[str]
    members: np.ndarray  # G_a: n_a(u, a) >= min_attribute_purchases
    strict: np.ndarray  # S_a: G_a with n_C >= min_n_c and pi - pi_bar >= min_gap

    @property
    def any_member(self) -> np.ndarray:
        """G_H = union of G_a."""
        return self.members.any(axis=1)

    @property
    def any_strict(self) -> np.ndarray:
        """S_H = union of S_a."""
        return self.strict.any(axis=1)


def subgroups(model: PreferenceModel, min_attribute_purchases: int, min_n_c: int, min_gap: float) -> Subgroups:
    """G_a and S_a from history counts; the strict gap is one-sided (pi above pi_bar)."""
    if model.n_a is None:
        raise ValueError("The preference model has no n_a counts")
    members = model.n_a >= min_attribute_purchases
    gap = model.pi - model.prior
    strict = members & (model.n_c >= min_n_c) & (gap >= min_gap - GAP_TOLERANCE)
    return Subgroups(list(model.attributes), members, strict)


def config_subgroups(model: PreferenceModel, config: dict[str, object]) -> Subgroups:
    settings = config["health_subgroup"]
    strict = settings["strict"]
    return subgroups(
        model, int(settings["min_attribute_purchases"]), int(strict["min_n_c"]), float(strict["min_gap"])
    )


@dataclass
class HealthSplit:
    state: SplitState
    model: PreferenceModel
    groups: Subgroups


def build_health_split(
    config: dict[str, object], dataset: str, items: pd.DataFrame, history: pd.DataFrame, target: pd.DataFrame
) -> HealthSplit:
    """Repeat-explore state plus the health preference model and subgroups, all from history."""
    state = build_state(config, dataset, items, history, target)
    model = health_preferences(config, dataset, items, history, state.interactions)
    return HealthSplit(state, model, config_subgroups(model, config))


def row_gamma_state(
    state: SplitState,
    scope: ScopeState,
    model: PreferenceModel,
    rows: np.ndarray,
    repeat: sparse.csr_matrix,
    config: dict[str, object],
    user_mask: np.ndarray | None = None,
) -> GammaState:
    """Repeat ranking and explore candidates for the given user rows only, in global user indices.

    CF scores, preference items, popular fill, truncation and l_u are all computed per user, so the
    result equals the all-user computation restricted to these rows. Nothing here depends on beta.
    """
    rows = np.sort(np.asarray(rows, dtype=np.int64))
    local_repeat = repeat[rows]
    seen = state.seen[rows]
    seen_sets = [state.seen_sets[row] for row in rows]
    local_model = select_users(model, rows)
    cf = cf_scores(local_repeat, state.similarity, seen, scope.in_scope)
    preferred = preference_candidates(
        seen_sets, scope.in_scope, local_model, state.item_fine, state.popularity, config["explore"]
    )
    candidates = explore_candidates(
        cf,
        seen,
        scope.in_scope,
        local_model,
        state.item_fine,
        state.popularity,
        config["explore"],
        int(config["evaluation"]["top_k"]),
        preferred=preferred,
        seen_sets=seen_sets,
    )
    if user_mask is not None:
        candidates["ell"] = local_model.log_ratio(
            candidates["user"].to_numpy(np.int64), candidates["item"].to_numpy(np.int64), user_mask=user_mask[rows]
        )
    candidates["user"] = rows[candidates["user"].to_numpy(np.int64)]
    ranked = repeat_ranking(local_repeat, scope.in_scope)
    ranked["user"] = rows[ranked["user"].to_numpy(np.int64)]
    return GammaState(ranked, candidates)


def attribute_item_sets(model: PreferenceModel, scope: ScopeState, item_ids: pd.Index) -> dict[str, set[str]]:
    """Items with attr_a = 1 and fine_category in C_a inside the scope F, per attribute."""
    ids = item_ids.to_numpy()
    return {
        name: set(ids[(model.item_flag[:, column] == 1) & model.item_applicable[:, column] & scope.in_scope].astype(str))
        for column, name in enumerate(model.attributes)
    }


def new_item_sets(targets: dict[str, set[str]], history_items: dict[str, set[str]]) -> dict[str, set[str]]:
    """N_u = T_u minus I_u."""
    return {user: truth - history_items.get(user, set()) for user, truth in targets.items()}


def attribute_new_item_recall(
    explore: Lists, new_items: dict[str, set[str]], attribute_items: set[str], top_k: int
) -> tuple[float, pd.Series]:
    """Recall@K of the explore list on N_u^a = N_u intersected with attribute items; empty N_u^a skipped."""
    return subset_recall(explore, {user: items & attribute_items for user, items in new_items.items()}, top_k)


def attribute_shares(
    explore: Lists, new_items: dict[str, set[str]], attribute_items: set[str], top_k: int
) -> pd.DataFrame:
    """Per user with a non-empty N_u: attribute share of the explore top-K and of N_u."""
    rows: list[tuple[str, float, float]] = []
    for user, truth in new_items.items():
        if not truth:
            continue
        top = explore.get(user, [])[:top_k]
        explore_share = len(set(top) & attribute_items) / len(top) if top else np.nan
        rows.append((user, explore_share, len(truth & attribute_items) / len(truth)))
    return pd.DataFrame(rows, columns=["user_id", "explore_share", "target_share"]).set_index("user_id")


def user_scores(
    lists: tuple[Lists, Lists, Lists],
    targets: dict[str, set[str]],
    history_items: dict[str, set[str]],
    attribute_items: dict[str, set[str]],
    top_k: int,
) -> pd.DataFrame:
    """Per-user list metrics plus attribute new-item recall and attribute shares; NaN where undefined."""
    _, per_user = score_lists(*lists, targets, history_items, top_k)
    new_items = new_item_sets(targets, history_items)
    columns: dict[str, pd.Series] = {}
    for name, items in attribute_items.items():
        _, columns[f"attr_new_item_recall_{name}"] = attribute_new_item_recall(lists[1], new_items, items, top_k)
        shares = attribute_shares(lists[1], new_items, items, top_k)
        columns[f"explore_share_{name}"] = shares["explore_share"]
        columns[f"target_share_{name}"] = shares["target_share"]
    return per_user.assign(**columns)


def group_summary(per_user: pd.DataFrame, users: set[str]) -> dict[str, object]:
    """Means over the group's users that have a value; subset recalls also report their user counts."""
    frame = per_user.loc[per_user.index.isin(users)]
    summary: dict[str, object] = {"users": int(len(frame))}
    for column in frame.columns:
        values = frame[column].dropna()
        summary[column] = float(values.mean()) if len(values) else None
        if column.startswith(("new_item_recall", "repeat_recall", "attr_new_item_recall")):
            summary[f"{column}_users"] = int(len(values))
    return summary


def subgroup_bootstrap(
    first: pd.Series, second: pd.Series, users: set[str], resamples: int, seed: int
) -> dict[str, float]:
    """Paired bootstrap of first - second over subgroup users with values in both, sorted by user_id."""
    index = pd.Index(sorted(users), dtype=object)
    first = first.reindex(index).dropna()
    second = second.reindex(index).dropna()
    shared = first.index.intersection(second.index, sort=False)
    return paired_bootstrap(first.loc[shared], second.loc[shared], resamples, seed)


@dataclass
class Analysis:
    """One preference model with its primary subgroup G, strict subgroup S, and attribute item sets."""

    label: str
    model: PreferenceModel
    user_mask: np.ndarray | None
    members: np.ndarray  # users, G_a or G_H
    strict: np.ndarray  # users, S_a or S_H
    attribute_items: dict[str, set[str]]
    primary: str  # key of attribute_items used for the attribute new-item recall


def analyses(split: HealthSplit, item_sets: dict[str, set[str]]) -> list[Analysis]:
    """Single-attribute models on G_a, then the combined model on G_H with per-user attribute masks."""
    groups = split.groups
    result = [
        Analysis(
            name,
            select_attributes(split.model, [name]),
            None,
            groups.members[:, column],
            groups.strict[:, column],
            {name: item_sets[name]},
            name,
        )
        for column, name in enumerate(split.model.attributes)
    ]
    union = set().union(*item_sets.values())
    result.append(
        Analysis(
            COMBINED,
            split.model,
            groups.members,
            groups.any_member,
            groups.any_strict,
            {**item_sets, ANY_ATTRIBUTE: union},
            ANY_ATTRIBUTE,
        )
    )
    return result


def evaluated_users(state: SplitState, scope: ScopeState, mask: np.ndarray) -> set[str]:
    """Subgroup users that have a target inside the scope."""
    return {str(user) for user in state.interactions.users[mask] if str(user) in scope.targets}


def subgroup_sizes(split: HealthSplit, scope: ScopeState, analysis_list: list[Analysis]) -> dict[str, object]:
    """Sizes of G and S per analysis, with medians of n_a, n_C and pi among evaluated users."""
    state, model = split.state, split.model
    new_items = new_item_sets(scope.targets, state.history_items)
    result: dict[str, object] = {}
    for analysis in analysis_list:
        entry: dict[str, object] = {}
        for kind, mask in [("G", analysis.members), ("S", analysis.strict)]:
            group = evaluated_users(state, scope, mask)
            rows = np.sort(state.interactions.users.get_indexer(sorted(group)))
            primary = analysis.attribute_items[analysis.primary]
            medians = {
                name: {
                    statistic: float(np.median(values[rows, column])) if len(rows) else None
                    for statistic, values in [("n_a", model.n_a), ("n_c", model.n_c), ("pi", model.pi)]
                }
                for column, name in enumerate(model.attributes)
                if name in analysis.attribute_items
            }
            entry[kind] = {
                "history_users": int(mask.sum()),
                "evaluated_users": len(group),
                "new_item_users": sum(1 for user in group if new_items[user]),
                "attribute_new_item_users": sum(1 for user in group if new_items[user] & primary),
                "medians": medians,
            }
        result[analysis.label] = entry
    return result


def _announce(split_name: str, mode: str, sizes: dict[str, object]) -> None:
    print(f"[health-subgroup] {split_name} split, scope {mode}: subgroup sizes", file=sys.stderr)
    for label, entry in sizes.items():
        for kind, row in entry.items():
            print(
                f"  {kind}[{label}]: history={row['history_users']} evaluated={row['evaluated_users']} "
                f"N_u={row['new_item_users']} N_u^a={row['attribute_new_item_users']}",
                file=sys.stderr,
                flush=True,
            )


def fixed_parameters(config: dict[str, object], mode: str) -> dict[str, object]:
    """gamma and kappa from the repeat-explore tuning (Instacart, this scope); the slot mode from the config."""
    path = tuning_path(config)
    selected = json.loads(path.read_text(encoding="utf-8"))[DATASET]["scopes"][mode]["selected"]
    slots_mode = str(config["slots"]["mode"])
    try:
        source = path.relative_to(Path(config["project_root"])).as_posix()
    except (KeyError, ValueError):
        source = path.as_posix()
    return {
        "source": source,
        "gamma": float(selected["gamma"]),
        "kappa": float(selected["kappa"]),
        "slots_mode": slots_mode,
        "global_r": int(selected["global_r"]) if slots_mode == "global" else None,
    }


def _analysis_state(
    split: HealthSplit,
    scope: ScopeState,
    analysis: Analysis,
    users: set[str],
    repeat: sparse.csr_matrix,
    config: dict[str, object],
) -> GammaState:
    rows = split.state.interactions.users.get_indexer(sorted(users))
    return row_gamma_state(split.state, scope, analysis.model, rows, repeat, config, analysis.user_mask)


def _re_scores(
    split: HealthSplit,
    scope: ScopeState,
    gamma_state: GammaState,
    analysis: Analysis,
    users: set[str],
    beta: float,
    fixed: dict[str, object],
    config: dict[str, object],
) -> pd.DataFrame:
    recommendation = recommend(split.state, scope, gamma_state, float(fixed["kappa"]), beta, config, fixed["global_r"])
    targets = {user: scope.targets[user] for user in users}
    lists = (recommendation.final, recommendation.explore, recommendation.repeat)
    top_k = int(config["evaluation"]["top_k"])
    return user_scores(lists, targets, split.state.history_items, analysis.attribute_items, top_k)


def tune_scope(
    config: dict[str, object], split: HealthSplit, mode: str, fixed: dict[str, object]
) -> dict[str, object]:
    """beta*_a on G_a (and beta*_H on G_H) by mean new-item recall@K; ties keep the smaller beta."""
    scope = build_scope(split.state, mode, config)
    item_sets = attribute_item_sets(split.model, scope, split.state.interactions.items)
    analysis_list = analyses(split, item_sets)
    sizes = subgroup_sizes(split, scope, analysis_list)
    _announce("fit", mode, sizes)
    repeat = repeat_scores(split.state.interactions, float(fixed["gamma"]))
    report: dict[str, object] = {"fixed": fixed, "subgroup_sizes": sizes, "analyses": {}}
    for analysis in analysis_list:
        users = evaluated_users(split.state, scope, analysis.members)
        if not users:
            report["analyses"][analysis.label] = {"users": 0, "grid_beta": [], "selected_beta": 0.0}
            continue
        gamma_state = _analysis_state(split, scope, analysis, users, repeat, config)
        curve = []
        for beta in sorted(float(value) for value in config["health_subgroup"]["betas"]):
            per_user = _re_scores(split, scope, gamma_state, analysis, users, beta, fixed, config)
            curve.append({"beta": beta, **group_summary(per_user, users)})
        del gamma_state
        best = max(curve, key=lambda row: -np.inf if row["new_item_recall_at_k"] is None else row["new_item_recall_at_k"])
        report["analyses"][analysis.label] = {
            "users": len(users),
            "primary_attribute": analysis.primary,
            "grid_beta": curve,
            "selected_beta": best["beta"],
        }
    return report


def hs_tuning_path(config: dict[str, object]) -> Path:
    return Path(config["report_dir"]) / "metrics" / f"{config['profile']}_hs_tuning.json"


def _split_report(config: dict[str, object], split: HealthSplit, split_name: str) -> dict[str, object]:
    model = split.model
    return {
        "profile": config["profile"],
        "dataset": DATASET,
        "split": split_name,
        "analysis_type": "post-hoc subgroup analysis",
        "attributes": list(model.attributes),
        "category_sizes": {
            name: len(categories)
            for name, categories in health_categories(config, DATASET, split.state.items).items()
        },
        "pi_bar": {name: float(model.prior[column]) for column, name in enumerate(model.attributes)},
        "history_users": int(len(split.state.interactions.users)),
        "scopes": {},
    }


def tune_hs(config: dict[str, object]) -> dict[str, object]:
    """Select beta per analysis on the Instacart fit split with gamma, kappa and slots held fixed."""
    started = time.perf_counter()
    items, history, target = load_split(config, DATASET, "fit")
    split = build_health_split(config, DATASET, items, history, target)
    report = _split_report(config, split, "fit")
    for mode in config["health_subgroup"]["scopes"]:
        report["scopes"][mode] = tune_scope(config, split, mode, fixed_parameters(config, mode))
    report["runtime_seconds"] = time.perf_counter() - started
    _write_json(hs_tuning_path(config), report)
    return report


def _bootstrap(first: pd.Series, second: pd.Series, users: set[str], config: dict[str, object]) -> dict[str, float]:
    settings = config["bootstrap"]
    return subgroup_bootstrap(first, second, users, int(settings["n"]), int(settings["seed"]))


# (label, first model, second model, metric); "re_star" is the combined model at beta*_H.
BASELINE_COMPARISONS = [
    ("recall_re_star_minus_personal_repeat", "re_star", "personal_repeat", "recall_at_k"),
    ("ndcg_re_star_minus_personal_repeat", "re_star", "personal_repeat", "ndcg_at_k"),
    ("ndcg_re_star_minus_lr_hybrid", "re_star", "lr_hybrid", "ndcg_at_k"),
    ("recall_lr_pref_minus_lr_hybrid", "lr_hybrid_pref", "lr_hybrid", "recall_at_k"),
    ("ndcg_lr_pref_minus_lr_hybrid", "lr_hybrid_pref", "lr_hybrid", "ndcg_at_k"),
    ("new_item_recall_lr_pref_minus_lr_hybrid", "lr_hybrid_pref", "lr_hybrid", "new_item_recall_at_k"),
    ("new_item_recall_re_star_minus_popularity", "re_star", "popularity", "new_item_recall_at_k"),
]


def baseline_user_scores(
    config: dict[str, object],
    split: HealthSplit,
    scope: ScopeState,
    scored: pd.DataFrame,
    users: set[str],
    attribute_items: dict[str, set[str]],
) -> dict[str, pd.DataFrame]:
    """Per-user scores of popularity, personal repeat and both frozen LR rankers for the given users."""
    top_k = int(config["evaluation"]["top_k"])
    targets = {user: scope.targets[user] for user in users}
    item_ids = split.state.interactions.items.to_numpy()
    allowed = None if scope.mode == "all" else set(item_ids[scope.in_scope].astype(str))
    lists = {"popularity": popularity_lists(split.state, replace(scope, targets=targets), top_k)}
    subset = scored[scored["user_id"].isin(users)]
    for name, column in [("personal_repeat", "score_repeat"), ("lr_hybrid", "score_lr"), ("lr_hybrid_pref", "score_pref_lr")]:
        lists[name] = candidate_lists(subset, column, allowed, top_k)
    return {
        name: user_scores(value, targets, split.state.history_items, attribute_items, top_k)
        for name, value in lists.items()
    }


def _baseline_comparisons(
    config: dict[str, object],
    split: HealthSplit,
    scope: ScopeState,
    scored: pd.DataFrame,
    analysis: Analysis,
    re_star: pd.DataFrame,
) -> dict[str, object]:
    """Final-list comparisons of the combined model at beta*_H with baselines, on G_H and on each G_a."""
    users = evaluated_users(split.state, scope, analysis.members)
    frames = {"re_star": re_star, **baseline_user_scores(config, split, scope, scored, users, analysis.attribute_items)}
    groups = {"G_H": users}
    for column, name in enumerate(split.model.attributes):
        groups[f"G_{name}"] = evaluated_users(split.state, scope, split.groups.members[:, column])
    result: dict[str, object] = {}
    for label, group in groups.items():
        result[label] = {
            "users": len(group),
            "metrics": {name: group_summary(frame, group) for name, frame in frames.items()},
            "bootstrap_ci": {
                name: _bootstrap(frames[first][metric], frames[second][metric], group, config)
                for name, first, second, metric in BASELINE_COMPARISONS
            },
        }
    return result


def eval_scope(
    config: dict[str, object], split: HealthSplit, mode: str, tuned: dict[str, object], scored: pd.DataFrame
) -> dict[str, object]:
    """Compare beta*, beta_fixed and beta = 0 per analysis on G and S; baselines on G_H and each G_a."""
    settings = config["health_subgroup"]
    fixed = tuned["fixed"]
    beta_fixed = float(settings["beta_fixed"])
    warning = int(settings["min_eval_users_warning"])
    scope = build_scope(split.state, mode, config)
    item_sets = attribute_item_sets(split.model, scope, split.state.interactions.items)
    analysis_list = analyses(split, item_sets)
    sizes = subgroup_sizes(split, scope, analysis_list)
    _announce("eval", mode, sizes)
    repeat = repeat_scores(split.state.interactions, float(fixed["gamma"]))
    report: dict[str, object] = {"fixed": fixed, "subgroup_sizes": sizes, "analyses": {}, "insufficient_sample": []}
    for analysis in analysis_list:
        users = evaluated_users(split.state, scope, analysis.members)
        beta_star = float(tuned["analyses"][analysis.label]["selected_beta"])
        betas = {"beta0": 0.0, "beta_star": beta_star, "beta_fixed": beta_fixed}
        entry: dict[str, object] = {"selected_beta": beta_star, "beta_fixed": beta_fixed, "groups": {}}
        if not users:
            report["analyses"][analysis.label] = {**entry, "skipped": "no evaluated users"}
            continue
        gamma_state = _analysis_state(split, scope, analysis, users, repeat, config)
        by_value = {
            value: _re_scores(split, scope, gamma_state, analysis, users, value, fixed, config)
            for value in sorted(set(betas.values()))
        }
        del gamma_state
        per_user = {label: by_value[value] for label, value in betas.items()}
        attribute_metric = f"attr_new_item_recall_{analysis.primary}"
        for kind, mask in [("G", analysis.members), ("S", analysis.strict)]:
            group = evaluated_users(split.state, scope, mask)
            base = per_user["beta0"]
            attribute_users = int(base.loc[base.index.isin(group), attribute_metric].notna().sum())
            insufficient = attribute_users < warning
            if insufficient:
                report["insufficient_sample"].append(f"{kind}[{analysis.label}]")
            entry["groups"][kind] = {
                "users": len(group),
                "attribute_new_item_users": attribute_users,
                "insufficient_sample": insufficient,
                "metrics": {label: group_summary(frame, group) for label, frame in per_user.items()},
                "bootstrap_ci": {
                    f"{metric}_{label}_minus_beta0": _bootstrap(per_user[label][metric], base[metric], group, config)
                    for label in ["beta_star", "beta_fixed"]
                    for metric in ["new_item_recall_at_k", attribute_metric]
                },
            }
        if analysis.label == COMBINED:
            entry["baselines"] = _baseline_comparisons(config, split, scope, scored, analysis, per_user["beta_star"])
        report["analyses"][analysis.label] = entry
    return report


def eval_hs(config: dict[str, object]) -> dict[str, object]:
    """Evaluate on the Instacart eval split with betas selected on the fit split; nothing is selected here."""
    started = time.perf_counter()
    tuning = json.loads(hs_tuning_path(config).read_text(encoding="utf-8"))
    items, history, target = load_split(config, DATASET, "eval")
    split = build_health_split(config, DATASET, items, history, target)
    scored, checksums = baseline_scores(config, items, history, target)
    report = _split_report(config, split, "eval")
    report["frozen_models"] = checksums
    report["frozen_model_verified"] = all(row["sha256_before"] == row["sha256_after"] for row in checksums.values())
    for mode in config["health_subgroup"]["scopes"]:
        report["scopes"][mode] = eval_scope(config, split, mode, tuning["scopes"][mode], scored)
    report["runtime_seconds"] = time.perf_counter() - started
    _write_json(Path(config["report_dir"]) / "metrics" / f"{config['profile']}_hs_{DATASET}.json", report)
    return report
