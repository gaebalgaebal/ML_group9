"""Health-attribute subgroups: users who bought a health-attribute item, analysed on their own.

Subgroup membership, pi, pi_bar, n_a and n_C all come from one split's history; targets are only used
for metrics. This is a post-hoc subgroup analysis chosen after the repeat-explore results.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .attributes import applicable_categories, category_sets, exclude_nonfood_categories
from .re_pipeline import SplitState, build_state
from .repeat_explore import Interactions, PreferenceModel, build_preferences


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
