"""Post-hoc follow-up of the hierarchical attribute stage (Instacart eval split, food scope, health attributes).

The stage-2 evaluation of hier_pipeline pools target items of many (aisle, type) pairs, so pi_bar(a | t)
carries between-type differences in attribute share. This module asks whether, inside the same type,
the user's attribute preference is associated with the attribute of the target item:
- within-type AUC: AUC per (aisle, type) weighted by its positive x negative pairs (baseline 0.5),
  for pi_u(a), pi_u(a | t) and the unsmoothed n_a / n_C, with user-bootstrap CIs,
- per-type contributions and the AUC without the largest type,
- a split by n_u(t) = 0 (pi_u(a | t) backs off to pi_u(a)) vs n_u(t) > 0,
- an offset logistic model with logit pi_bar(a | t) as offset,
- a diagnostic of attribute combinations without items inside a type.

All probabilities come from history; the target only defines rows and labels.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import optimize

from .config import load_config
from .data import load_split
from .hier_pipeline import PREDICTORS, SCOPE, attribute_rows, auc_score
from .hierarchical import HierState, Hierarchy, Params, build_hier_state, lookup, type_preferences
from .pipeline import _write_json


DATASET = "instacart"
SCORES = {"a": "pi_user_aisle", "b": "pi_user_type", "c": "pi_raw"}
LEVELS = [0.95, 1 - 0.05 / 4]
LOGIT_CLIP = 1e-6
TOP_TYPES = 5
EXAMPLES = 3
DOMINANT_SHARE = 0.5
MIN_POSITIVE_ROWS = 100
MIN_TYPES = 5
DIAGNOSTIC_USERS = 2000
# Type labels of items without a mapped ingredient type fall back to the aisle.
FALLBACK_PREFIX = "aisle:"
REPRODUCTION_TOLERANCE = 1e-9


# ---------------------------------------------------------------------------------------------
# Within-type AUC


class WithinAUC:
    """AUC_within = sum_t n_t+ n_t- AUC_t / sum_t n_t+ n_t-, i.e. the pooled AUC over positive-negative
    pairs of the same type, with ties counted as one half. Rows are binned once by (type, score) so that
    bootstrap weights only need bincounts."""

    def __init__(self, types: np.ndarray, scores: np.ndarray, labels: np.ndarray):
        self.labels = np.asarray(labels, dtype=np.float64)
        types, scores = np.asarray(types, dtype=np.int64), np.asarray(scores, dtype=np.float64)
        self.n_rows = len(types)
        if self.n_rows == 0:
            empty = np.zeros(0, dtype=np.int64)
            self.bin, self.bin_type, self.first_bin, self.type_codes = empty, empty, empty, empty
            return
        order = np.lexsort((scores, types))
        change = np.r_[True, (types[order][1:] != types[order][:-1]) | (scores[order][1:] != scores[order][:-1])]
        bin_sorted = np.cumsum(change) - 1
        self.bin = np.empty(self.n_rows, dtype=np.int64)
        self.bin[order] = bin_sorted
        starts = np.flatnonzero(change)
        bin_types = types[order][starts]
        self.type_codes, self.bin_type = np.unique(bin_types, return_inverse=True)
        type_start = np.flatnonzero(np.r_[True, bin_types[1:] != bin_types[:-1]])
        self.first_bin = type_start[self.bin_type]

    def _parts(self, weights: np.ndarray | None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Per type: weighted positives, negatives and concordant pairs (ties one half)."""
        n_bins, n_types = len(self.bin_type), len(self.type_codes)
        weights = np.ones(self.n_rows) if weights is None else weights
        positive = np.bincount(self.bin, weights=weights * self.labels, minlength=n_bins)
        negative = np.bincount(self.bin, weights=weights * (1.0 - self.labels), minlength=n_bins)
        below = np.cumsum(negative) - negative
        below = below - below[self.first_bin]
        concordant = positive * (below + 0.5 * negative)
        return (
            np.bincount(self.bin_type, weights=positive, minlength=n_types),
            np.bincount(self.bin_type, weights=negative, minlength=n_types),
            np.bincount(self.bin_type, weights=concordant, minlength=n_types),
        )

    def value(self, weights: np.ndarray | None = None) -> float | None:
        if self.n_rows == 0:
            return None
        positive, negative, concordant = self._parts(weights)
        pairs = (positive * negative).sum()
        return float(concordant.sum() / pairs) if pairs > 0 else None

    def per_type(self) -> pd.DataFrame:
        """type, positives, negatives, pairs and AUC_t for types with both classes."""
        if self.n_rows == 0:
            return pd.DataFrame(columns=["type", "positive", "negative", "pairs", "auc"])
        positive, negative, concordant = self._parts(None)
        pairs = positive * negative
        keep = pairs > 0
        return pd.DataFrame(
            {
                "type": self.type_codes[keep],
                "positive": positive[keep].astype(np.int64),
                "negative": negative[keep].astype(np.int64),
                "pairs": pairs[keep],
                "auc": concordant[keep] / pairs[keep],
            }
        )


def within_summary(frame: pd.DataFrame, score: str) -> dict[str, object]:
    """Counts over the types with both classes for one score column (rows with a missing score dropped)."""
    rows = frame[frame[score].notna()]
    estimator = WithinAUC(rows["type"].to_numpy(), rows[score].to_numpy(), rows["label"].to_numpy())
    table = estimator.per_type()
    used = rows[rows["type"].isin(table["type"])]
    return {
        "auc_within": estimator.value(),
        "types": int(len(table)),
        "positive_rows": int(used["label"].sum()),
        "negative_rows": int((1 - used["label"]).sum()),
        "rows_in_single_class_types": int(len(rows) - len(used)),
        "rows_without_score": int(len(frame) - len(rows)),
        "users": int(used["user"].nunique()),
        "pairs": float(table["pairs"].sum()),
    }


def level_ci(values: list[float], level: float) -> tuple[float | None, float | None]:
    if not values:
        return None, None
    tail = (1 - level) / 2 * 100
    low, high = np.percentile(values, [tail, 100 - tail])
    return float(low), float(high)


def user_weights(users: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Row weights from one user bootstrap draw (rows keep their user)."""
    unique, inverse = np.unique(users, return_inverse=True)
    counts = np.bincount(rng.integers(0, len(unique), len(unique)), minlength=len(unique))
    return counts[inverse].astype(np.float64)


def bootstrap_draws(users: np.ndarray, statistics: dict[str, object], resamples: int, seed: int) -> dict[str, list[float]]:
    """statistics[key](weights) -> value or None; one shared user draw per resample for every key."""
    rng = np.random.default_rng(seed)
    draws: dict[str, list[float]] = {key: [] for key in statistics}
    for _ in range(resamples):
        weights = user_weights(users, rng)
        for key, statistic in statistics.items():
            value = statistic(weights)
            if value is not None:
                draws[key].append(value)
    return draws


def _with_ci(value: float | None, draws: list[float]) -> dict[str, object]:
    result: dict[str, object] = {"value": value, "valid_resamples": len(draws)}
    for level in LEVELS:
        low, high = level_ci(draws, level)
        result[f"ci{level * 100:g}_low"], result[f"ci{level * 100:g}_high"] = low, high
    return result


def _minus(a: float | None, b: float | None) -> float | None:
    return a - b if a is not None and b is not None else None


def _subset_statistic(mask: np.ndarray, frame: pd.DataFrame, score: str):
    rows = frame[mask]
    estimator = WithinAUC(rows["type"].to_numpy(), rows[score].to_numpy(), rows["label"].to_numpy())
    return lambda weights: estimator.value(weights[mask])


# ---------------------------------------------------------------------------------------------
# Offset logistic


def logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=np.float64), LOGIT_CLIP, 1.0 - LOGIT_CLIP)
    return np.log(p) - np.log1p(-p)


def fit_offset_logit(
    offset: np.ndarray, x: np.ndarray, labels: np.ndarray, weights: np.ndarray | None = None, start: np.ndarray | None = None
) -> np.ndarray | None:
    """Maximum likelihood (beta0, beta) of logit P(y = 1) = beta0 + offset + beta * x."""
    weights = np.ones(len(labels)) if weights is None else weights
    labels = np.asarray(labels, dtype=np.float64)
    if weights.sum() == 0 or (weights * labels).sum() == 0 or (weights * (1 - labels)).sum() == 0:
        return None

    def objective(params: np.ndarray) -> tuple[float, np.ndarray]:
        z = params[0] + offset + params[1] * x
        nll = (weights * (np.logaddexp(0.0, z) - labels * z)).sum()
        residual = weights * (1.0 / (1.0 + np.exp(-z)) - labels)
        return float(nll), np.array([residual.sum(), (residual * x).sum()])

    result = optimize.minimize(objective, np.zeros(2) if start is None else start, jac=True, method="BFGS")
    return result.x if np.all(np.isfinite(result.x)) else None


def beta_statistic(offset: np.ndarray, x: np.ndarray, labels: np.ndarray, start: np.ndarray):
    """Bootstrap statistic: the weighted MLE of beta, started at the point estimate."""

    def statistic(weights: np.ndarray) -> float | None:
        fit = fit_offset_logit(offset, x, labels, weights, start)
        return None if fit is None else float(fit[1])

    return statistic


def dominant_type(table: pd.DataFrame, share: float = DOMINANT_SHARE) -> pd.Series | None:
    """The type with the most pairs when its share of all pairs exceeds `share` (strictly)."""
    if len(table) == 0 or table["pairs"].sum() == 0:
        return None
    top = table.sort_values(["pairs", "type"], ascending=[False, True], kind="mergesort").iloc[0]
    return top if top["pairs"] / table["pairs"].sum() > share else None


# ---------------------------------------------------------------------------------------------
# Diagnostic: attribute combinations without items


def type_patterns(hierarchy: Hierarchy) -> dict[int, tuple[np.ndarray, set[int]]]:
    """For types with m >= 1 applicable attributes: (applicable columns, attribute patterns present among the
    type's items), a pattern being sum_j bit_j 2^j over the applicable columns in order."""
    result = {}
    for t in np.flatnonzero(hierarchy.type_applicable.any(axis=1)):
        columns = np.flatnonzero(hierarchy.type_applicable[t])
        members = hierarchy.type_members[hierarchy.type_ptr[t] : hierarchy.type_ptr[t + 1]]
        bits = hierarchy.item_bits[members][:, columns].astype(np.int64)
        result[int(t)] = (columns, set((bits @ (1 << np.arange(len(columns)))).tolist()))
    return result


def present_mass(pi: np.ndarray, patterns: set[int]) -> np.ndarray:
    """sum over present patterns x of prod_j pi_j^x_j (1 - pi_j)^(1 - x_j), one value per row of pi."""
    total = np.zeros(len(pi))
    for pattern in patterns:
        bits = (pattern >> np.arange(pi.shape[1])) & 1
        total += np.prod(np.where(bits == 1, pi, 1.0 - pi), axis=1)
    return total


def empty_combinations(state: HierState, hierarchy: Hierarchy, lambda_type: float, users: np.ndarray) -> dict[str, object]:
    patterns = type_patterns(hierarchy)
    gaps = {t: value for t, value in patterns.items() if len(value[1]) < 2 ** len(value[0])}
    purchases = np.bincount(hierarchy.item_type[state.in_scope], weights=state.popularity[state.in_scope], minlength=hierarchy.n_types)
    missing = []
    for t, (columns, present) in gaps.items():
        pi = type_preferences(state, hierarchy, lambda_type, users, np.full(len(users), t, dtype=np.int64))[:, columns]
        missing.append(1.0 - present_mass(pi, present))
    values = np.concatenate(missing) if missing else np.zeros(0)
    return {
        "types_with_attributes": len(patterns),
        "types_with_empty_combination": len(gaps),
        "history_purchase_share_of_those_types": float(purchases[list(gaps)].sum() / purchases.sum()) if gaps and purchases.sum() else 0.0,
        "sample_users": int(len(users)),
        "user_type_pairs": int(len(values)),
        "missing_mass": {
            "mean": float(values.mean()) if len(values) else None,
            "median": float(np.median(values)) if len(values) else None,
            "p90": float(np.percentile(values, 90)) if len(values) else None,
        },
    }


# ---------------------------------------------------------------------------------------------
# Analysis


def scored_rows(state: HierState, params: Params) -> pd.DataFrame:
    """attribute_rows plus item, type, n_u(t), pi_raw = n_a / n_C (NaN when n_C = 0) and pi_bar_a, in the
    same row order."""
    frame = attribute_rows(state, params)
    hierarchy = state.hierarchies["full"]
    model = state.aisle_model
    types = hierarchy.item_type[state.target_item]
    extra = []
    for column, name in enumerate(hierarchy.attributes):
        keep = hierarchy.type_applicable[types, column]
        users, kept_types = state.target_user[keep], types[keep]
        n_c, n_a = model.n_c[users, column], model.n_a[users, column]
        extra.append(
            pd.DataFrame(
                {
                    "item": state.target_item[keep],
                    "type": kept_types,
                    "n_type": lookup(hierarchy.n_type, users, kept_types),
                    "pi_raw": np.divide(n_a, n_c, out=np.full(len(users), np.nan), where=n_c > 0),
                    "pi_bar_a": model.prior[column],
                }
            )
        )
    return pd.concat([frame, pd.concat(extra, ignore_index=True)], axis=1) if extra else frame


def is_fallback(type_key: str) -> bool:
    """A type key "aisle|type" whose type is the aisle fallback ("aisle:<aisle>") rather than a mapped type."""
    return str(type_key).split("|", 1)[-1].startswith(FALLBACK_PREFIX)


def fallback_pair_share(table: pd.DataFrame, fallback: np.ndarray) -> float | None:
    """Share of the within-type positive-negative pairs that fall in fallback types."""
    total = table["pairs"].sum()
    return float(table["pairs"][fallback[table["type"].to_numpy()]].sum() / total) if total else None


def difference_statistic(first: WithinAUC, second: WithinAUC):
    """Bootstrap statistic: AUC_within(first) - AUC_within(second) under the same row weights."""

    def statistic(weights: np.ndarray) -> float | None:
        a, b = first.value(weights), second.value(weights)
        return a - b if a is not None and b is not None else None

    return statistic


def modal_key(hierarchy: Hierarchy, item_keys: np.ndarray, t: int) -> str | None:
    """Most common mapping key among the type's items (readable label of a hashed type id); None for an
    aisle fallback type."""
    if is_fallback(hierarchy.type_labels[t]):
        return None
    members = hierarchy.type_members[hierarchy.type_ptr[t] : hierarchy.type_ptr[t + 1]]
    keys = pd.Series(item_keys[members])
    keys = keys[keys != ""]
    return str(keys.value_counts().sort_index(kind="mergesort").sort_values(ascending=False, kind="mergesort").index[0]) if len(keys) else None


def attribute_analysis(
    rows: pd.DataFrame,
    hierarchy: Hierarchy,
    product_names: np.ndarray,
    item_keys: np.ndarray,
    resamples: int,
    seed: int,
    sensitivity: bool = False,
) -> dict[str, object]:
    """sensitivity adds the mapped-type-only AUC and the paired difference AUC(a) - AUC(b). The extra
    statistics share the bootstrap draws, so every other value is unchanged."""
    users = rows["user"].to_numpy()
    statistics: dict[str, object] = {}
    result: dict[str, object] = {"rows": int(len(rows)), "users": int(rows["user"].nunique())}
    all_rows = np.ones(len(rows), dtype=bool)
    for key, column in SCORES.items():
        statistics[key] = _subset_statistic(all_rows & rows[column].notna().to_numpy(), rows, column)
    fallback = np.array([is_fallback(label) for label in hierarchy.type_labels], dtype=bool)
    mapped = ~fallback[rows["type"].to_numpy()]
    if sensitivity:
        for key in ["a", "c"]:
            statistics[f"mapped_{key}"] = _subset_statistic(mapped & rows[SCORES[key]].notna().to_numpy(), rows, SCORES[key])
        estimators = {
            key: WithinAUC(rows["type"].to_numpy(), rows[SCORES[key]].to_numpy(), rows["label"].to_numpy()) for key in ["a", "b"]
        }
        statistics["a_minus_b"] = difference_statistic(estimators["a"], estimators["b"])
    split = rows["n_type"].to_numpy() > 0
    for label, mask in [("n_type_zero", ~split), ("n_type_positive", split)]:
        statistics[label] = _subset_statistic(mask, rows, SCORES["a"])

    table = WithinAUC(rows["type"].to_numpy(), rows[SCORES["a"]].to_numpy(), rows["label"].to_numpy()).per_type()
    table = table.sort_values(["pairs", "type"], ascending=[False, True], kind="mergesort")
    total_pairs = table["pairs"].sum()
    dominant = dominant_type(table)
    if dominant is not None:
        statistics["without_top_type"] = _subset_statistic(rows["type"].to_numpy() != int(dominant["type"]), rows, SCORES["a"])

    offset = logit(rows["pi_population_type"].to_numpy())
    x = logit(rows["pi_user_aisle"].to_numpy()) - logit(rows["pi_bar_a"].to_numpy())
    labels = rows["label"].to_numpy()
    point = fit_offset_logit(offset, x, labels)
    if point is not None:
        statistics["beta"] = beta_statistic(offset, x, labels, point)

    draws = bootstrap_draws(users, statistics, resamples, seed)
    scores = {}
    for key, column in SCORES.items():
        summary = within_summary(rows, column)
        scores[key] = {"score": column, **summary, **_with_ci(summary["auc_within"], draws[key])}
    result["within_type_auc"] = scores
    if sensitivity:
        mapped_rows = rows[mapped]
        result["within_type_auc_mapped"] = {
            key: {"score": SCORES[key], **summary, **_with_ci(summary["auc_within"], draws[f"mapped_{key}"])}
            for key in ["a", "c"]
            for summary in [within_summary(mapped_rows, SCORES[key])]
        }
        result["fallback_pair_share"] = fallback_pair_share(table, fallback)
        result["diff_a_minus_b"] = _with_ci(_minus(scores["a"]["auc_within"], scores["b"]["auc_within"]), draws["a_minus_b"])

    top = []
    for _, row in table.head(TOP_TYPES).iterrows():
        t = int(row["type"])
        positives = rows[(rows["type"] == t) & (rows["label"] == 1)]["item"].value_counts(sort=False)
        positives = positives.sort_values(ascending=False, kind="mergesort")
        label = str(hierarchy.type_labels[t])
        top.append(
            {
                "type_key": label,
                "mapping_key": modal_key(hierarchy, item_keys, t),
                "aisle": label.split("|", 1)[0],
                "pair_share": float(row["pairs"] / total_pairs),
                "auc_t": float(row["auc"]),
                "positive_rows": int(row["positive"]),
                "negative_rows": int(row["negative"]),
                "positive_examples": [str(product_names[i]) for i in positives.index[:EXAMPLES]],
            }
        )
    result["top_types"] = top
    if dominant is not None:
        mask = rows["type"].to_numpy() != int(dominant["type"])
        summary = within_summary(rows[mask], SCORES["a"])
        result["without_top_type"] = {
            "type_key": str(hierarchy.type_labels[int(dominant["type"])]),
            "mapping_key": modal_key(hierarchy, item_keys, int(dominant["type"])),
            **summary,
            **_with_ci(summary["auc_within"], draws["without_top_type"]),
        }
    else:
        result["without_top_type"] = None

    backoff = {}
    for label, mask in [("n_type_zero", ~split), ("n_type_positive", split)]:
        subset = rows[mask]
        labels_subset = subset["label"].to_numpy()
        summary = within_summary(subset, SCORES["a"])
        backoff[label] = {
            "rows": int(len(subset)),
            "positive_rows": int(labels_subset.sum()),
            "users": int(subset["user"].nunique()),
            "pooled_auc": {p: auc_score(subset[p].to_numpy(), labels_subset) if len(subset) else None for p in PREDICTORS},
            "within_type_auc_a": {**summary, **_with_ci(summary["auc_within"], draws[label])},
        }
    result["backoff_split"] = backoff

    result["offset_logistic"] = {
        "rows": int(len(rows)),
        "beta0": float(point[0]) if point is not None else None,
        "beta": _with_ci(float(point[1]) if point is not None else None, draws.get("beta", [])),
    }
    return result


def mapping_keys(config: dict[str, object], state: HierState) -> np.ndarray:
    """matched_key of each item from the imported mapping (empty when the file or the key is missing)."""
    path = Path(config["project_root"]) / config["hierarchical"]["mapping"]["path"]
    if not path.exists():
        return np.full(state.n_items, "", dtype=object)
    mapping = pd.read_parquet(path, columns=["product_id", "matched_key"])
    keys = mapping.assign(product_id=mapping["product_id"].astype(str)).drop_duplicates("product_id").set_index("product_id")["matched_key"]
    return keys.reindex(state.interactions.items).fillna("").astype(str).to_numpy(dtype=object)


def _reproduce(frame: pd.DataFrame, reference: dict[str, object] | None, attributes: list[str]) -> dict[str, object]:
    if reference is None:
        return {"checked": False}
    stages = reference["sources"][DATASET]["stagewise"]["attribute"]
    values, worst = {}, 0.0
    for name in attributes:
        rows = frame[frame["attribute"].eq(name)]
        for part, subset in [("new", rows[rows["new"]]), ("repeat", rows[~rows["new"]])]:
            expected = stages[name]["all"][part]
            labels = subset["label"].to_numpy()
            for predictor in PREDICTORS:
                value, target = auc_score(subset[predictor].to_numpy(), labels), expected["auc"][predictor]
                diff = abs(value - target) if value is not None and target is not None else (0.0 if value is target else float("inf"))
                values[f"{name}.{part}.{predictor}"] = {"reference": target, "recomputed": value, "abs_diff": diff}
                worst = max(worst, diff)
            if len(subset) != expected["rows"]:
                worst = float("inf")
                values[f"{name}.{part}.rows"] = {"reference": expected["rows"], "recomputed": int(len(subset))}
    return {"checked": True, "tolerance": REPRODUCTION_TOLERANCE, "max_abs_diff": worst, "passed": worst <= REPRODUCTION_TOLERANCE, "values": values}


def _interpretation(attributes: list[str], analyses: dict[str, object]) -> dict[str, object]:
    """Labels from the post-hoc rules fixed before the run (new-item rows)."""

    def conclusion(low: float | None, high: float | None) -> str | None:
        if low is None or high is None:
            return None
        if low > 0.5:
            return "associated"
        if high < 0.5:
            return "opposite direction"
        return "difference not established"

    result: dict[str, object] = {}
    for name in attributes:
        new = analyses[name]["new"]
        main = new["within_type_auc"]["a"]
        if main["positive_rows"] < MIN_POSITIVE_ROWS or main["types"] < MIN_TYPES:
            result[name] = {"main": "undetermined (insufficient sample)"}
            continue
        main95 = conclusion(main["ci95_low"], main["ci95_high"])
        main9875 = conclusion(main["ci98.75_low"], main["ci98.75_high"])
        smoothing = conclusion(new["within_type_auc"]["c"]["ci95_low"], new["within_type_auc"]["c"]["ci95_high"])
        entry: dict[str, object] = {
            "auc_within_a": main["value"],
            "main": main95,
            "main_9875": main9875,
            "changes_at_9875": main9875 != main95,
            "score_c": smoothing,
            "sensitive_to_smoothing": smoothing != main95,
        }
        dropped = new["without_top_type"]
        if dropped is not None:
            entry["depends_on_one_type"] = (
                dropped["type_key"] if dropped["ci95_low"] is not None and dropped["ci95_low"] <= 0.5 <= dropped["ci95_high"] else None
            )
        beta = new["offset_logistic"]["beta"]
        entry["beta_supports"] = bool(beta["ci95_low"] is not None and beta["ci95_low"] > 0)
        result[name] = entry
    result["note"] = (
        "'associated' = after controlling for the ingredient type, the user's attribute preference is associated with "
        "the attribute of new items. Post-hoc and observational; health attributes reported separately."
    )
    return result


def analyze(
    config: dict[str, object],
    items: pd.DataFrame,
    history: pd.DataFrame,
    target: pd.DataFrame,
    params: Params,
    gamma: float,
    reference: dict[str, object] | None = None,
) -> dict[str, object]:
    resamples, seed = int(config["bootstrap"]["n"]), int(config["bootstrap"]["seed"])
    started = time.perf_counter()
    state = build_hier_state(config, DATASET, items, history, target, gamma, ("full",))
    build_seconds = time.perf_counter() - started
    hierarchy = state.hierarchies["full"]
    frame = scored_rows(state, params)
    reproduction = _reproduce(frame, reference, hierarchy.attributes)
    if reproduction["checked"] and not reproduction["passed"]:
        raise RuntimeError(f"Stage-2 AUCs differ from the reference: {json.dumps(reproduction, default=str)}")
    table = items.drop_duplicates("item_id").assign(item_id=lambda f: f["item_id"].astype(str)).set_index("item_id")
    product_names = table["product_name"].reindex(state.interactions.items).fillna("").astype(str).to_numpy()
    item_keys = mapping_keys(config, state)
    analyses = {}
    for name in hierarchy.attributes:
        rows = frame[frame["attribute"].eq(name)]
        analyses[name] = {
            part: attribute_analysis(
                subset.reset_index(drop=True), hierarchy, product_names, item_keys, resamples, seed, sensitivity=part == "new"
            )
            for part, subset in [("new", rows[rows["new"]]), ("repeat", rows[~rows["new"]])]
        }
    sample = np.random.default_rng(seed).choice(state.target_rows, size=min(DIAGNOSTIC_USERS, len(state.target_rows)), replace=False)
    return {
        "reproduction": reproduction,
        "attributes": analyses,
        "diagnostic_empty_combinations": empty_combinations(state, hierarchy, float(params.lambda_type), np.sort(sample)),
        "interpretation": _interpretation(hierarchy.attributes, analyses),
        "history_users": int(state.n_users),
        "target_users": int(len(state.targets)),
        "build_seconds": build_seconds,
    }


def analyze_hier_followup(config: dict[str, object], reference_dir: str | None = None) -> dict[str, object]:
    """Run the follow-up on the Instacart eval split with the selected hierarchical parameters."""
    started = time.perf_counter()
    base = Path(reference_dir) if reference_dir else Path(config["report_dir"]) / "metrics"
    profile = config["profile"]

    def read(name: str) -> dict[str, object]:
        return json.loads((base / f"{profile}_{name}.json").read_text(encoding="utf-8"))

    selected = read("hier_tuning")[DATASET]["selected"]
    params = Params(float(selected["alpha"]), float(selected["mu"]), float(selected["lambda_type"]), int(selected["k"]))
    gamma = float(read("re_tuning")[DATASET]["scopes"][SCOPE]["selected"]["gamma"])
    reference_path = base / f"{profile}_hier_{DATASET}.json"
    reference = read(f"hier_{DATASET}") if reference_path.exists() else None
    items, history, target = load_split(config, DATASET, "eval")
    report = {
        "profile": profile,
        "dataset": DATASET,
        "split": "eval",
        "scope": SCOPE,
        "design": "post-hoc follow-up of the hierarchical attribute stage; health attributes only; observational",
        "reference_dir": base.as_posix(),
        "params": {"alpha": params.alpha, "mu": params.mu, "lambda_type": params.lambda_type, "k": params.k, "gamma": gamma},
        "settings": {
            "bootstrap": config["bootstrap"],
            "type_unit": "(aisle, ingredient type) pair of the full hierarchy",
            "scores": SCORES,
            "ci_levels": LEVELS,
            "offset_logistic": "logit P = beta0 + logit pi_bar(a | t) + beta * (logit pi_u(a) - logit pi_bar_a), clip 1e-6",
            "diagnostic_users": f"{DIAGNOSTIC_USERS} target users drawn with seed {config['bootstrap']['seed']}",
        },
        **analyze(config, items, history, target, params, gamma, reference),
    }
    report["runtime_seconds"] = time.perf_counter() - started
    _write_json(Path(config["report_dir"]) / "metrics" / f"{profile}_hier_followup.json", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Post-hoc within-type attribute analysis of the hierarchical recommender")
    parser.add_argument("--profile", choices=["sample", "medium", "full"], default="sample")
    parser.add_argument(
        "--reference-dir", default=None,
        help="Directory with <profile>_hier_tuning.json, <profile>_re_tuning.json and <profile>_hier_instacart.json (read only)",
    )
    args = parser.parse_args()
    report = analyze_hier_followup(load_config(args.profile), args.reference_dir)
    summary = {key: report[key] for key in ["reproduction", "interpretation", "runtime_seconds"]}
    summary["reproduction"] = {key: value for key, value in summary["reproduction"].items() if key != "values"}
    print(json.dumps(summary, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
