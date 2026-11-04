"""Descriptive analysis of H*: health-attribute purchases are stable over time in a user group, extend
to new items of the same aisles, and do so more than comparable taste attributes and placebo attributes.

Instacart eval split. M1 (stability) and M3 (concentration) use history only; M2 (new-item extension)
uses the target. Placebo attributes are random non-attribute items matched on aisle and popularity
decile. The decision rule in `verdict` was fixed before the results were seen. This is a post-hoc
design, chosen after the repeat-explore results; recommendation metrics play no part in it.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import sparse

from .attributes import attribute_column
from .data import load_split
from .hierarchical import attribute_categories
from .pipeline import _write_json


DATASET = "instacart"
TOP_SHARE = 0.2
# A taste attribute is flagged as differently sized when its in-C_a purchases differ by this factor.
SIZE_RATIO_WARNING = 3.0


def taste_attributes(items: pd.DataFrame, names: list[str]) -> pd.DataFrame:
    """Add attr_<name> = product name contains the word (case-insensitive, word boundaries)."""
    result = items.copy()
    text = result["product_name"].fillna("")
    for name in names:
        pattern = rf"\b{re.escape(name)}\b"
        result[attribute_column(name)] = text.str.contains(pattern, case=False, regex=True).astype("int8")
    return result


@dataclass
class Purchases:
    """users x items purchase counts (history deduplicated by user, basket, item) and target items."""

    users: pd.Index
    items: pd.Index
    history: sparse.csr_matrix
    first: sparse.csr_matrix  # the user's earlier floor(n_b / 2) baskets
    second: sparse.csr_matrix  # the remaining, later baskets
    seen: sparse.csr_matrix  # 1[i in I_u]
    new_target: sparse.csr_matrix  # 1[i in T_u and i not in I_u]
    item_aisle: np.ndarray
    popularity: np.ndarray  # history baskets containing the item


def purchase_data(items: pd.DataFrame, history: pd.DataFrame, target: pd.DataFrame) -> Purchases:
    item_ids = pd.Index(items["item_id"].astype(str).drop_duplicates().to_numpy())
    implicit = history[["user_id", "basket_id", "basket_seq", "item_id"]].copy()
    for column in ["user_id", "basket_id", "item_id"]:
        implicit[column] = implicit[column].astype(str)
    implicit = implicit.drop_duplicates(["user_id", "basket_id", "item_id"])
    implicit = implicit[implicit["item_id"].isin(item_ids)]
    users = pd.Index(np.sort(implicit["user_id"].unique()))
    baskets = implicit[["user_id", "basket_id", "basket_seq"]].drop_duplicates().sort_values(
        ["user_id", "basket_seq", "basket_id"], kind="mergesort"
    )
    baskets["rank"] = baskets.groupby("user_id").cumcount()
    baskets["n_b"] = baskets.groupby("user_id")["basket_id"].transform("size")
    implicit = implicit.merge(baskets[["user_id", "basket_id", "rank", "n_b"]], on=["user_id", "basket_id"])
    rows = users.get_indexer(implicit["user_id"])
    cols = item_ids.get_indexer(implicit["item_id"])
    early = (implicit["rank"] < implicit["n_b"] // 2).to_numpy()
    shape = (len(users), len(item_ids))

    def matrix(mask: np.ndarray) -> sparse.csr_matrix:
        return sparse.csr_matrix((np.ones(int(mask.sum())), (rows[mask], cols[mask])), shape=shape)

    full = matrix(np.ones(len(rows), dtype=bool))
    seen = full.copy()
    seen.data[:] = 1.0
    frame = target[["user_id", "item_id"]].astype(str).drop_duplicates()
    frame = frame[frame["user_id"].isin(users) & frame["item_id"].isin(item_ids)]
    truth = sparse.csr_matrix(
        (np.ones(len(frame)), (users.get_indexer(frame["user_id"]), item_ids.get_indexer(frame["item_id"]))), shape=shape
    )
    new_target = truth - truth.multiply(seen)
    new_target.eliminate_zeros()
    table = items.drop_duplicates("item_id").assign(item_id=lambda f: f["item_id"].astype(str)).set_index("item_id")
    return Purchases(
        users=users,
        items=item_ids,
        history=full,
        first=matrix(early),
        second=matrix(~early),
        seen=seen,
        new_target=sparse.csr_matrix(new_target),
        item_aisle=table["fine_category"].reindex(item_ids).astype("string").fillna("UNKNOWN").astype(str).to_numpy(),
        popularity=np.bincount(cols, minlength=len(item_ids)).astype(np.float64),
    )


def counts(matrix: sparse.csr_matrix, flag: np.ndarray, in_c: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(n_a, n_C) per user: purchases of attribute items inside C_a and all purchases inside C_a."""
    return matrix @ (flag & in_c).astype(np.float64), matrix @ in_c.astype(np.float64)


# ---------------------------------------------------------------------------------------------
# Statistics


def weighted_ranks(values: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Mid-ranks of each value in the sample where observation j appears weights[j] times."""
    unique, inverse = np.unique(values, return_inverse=True)
    mass = np.bincount(inverse, weights=weights, minlength=len(unique))
    below = np.cumsum(mass) - mass
    return (below + (mass + 1.0) / 2.0)[inverse]


def weighted_spearman(x: np.ndarray, y: np.ndarray, weights: np.ndarray | None = None) -> float | None:
    """Spearman rho (Pearson of mid-ranks) of the sample with observation j repeated weights[j] times."""
    weights = np.ones(len(x)) if weights is None else np.asarray(weights, dtype=np.float64)
    if weights.sum() < 3:
        return None
    rx, ry = weighted_ranks(x, weights), weighted_ranks(y, weights)
    total = weights.sum()
    mx, my = (weights * rx).sum() / total, (weights * ry).sum() / total
    cov = (weights * (rx - mx) * (ry - my)).sum()
    var = (weights * (rx - mx) ** 2).sum() * (weights * (ry - my) ** 2).sum()
    return float(cov / np.sqrt(var)) if var > 0 else None


def gini(values: np.ndarray) -> float | None:
    """Gini coefficient of non-negative values: sum_i (2i - n - 1) x_(i) / (n sum x), x ascending."""
    values = np.sort(np.asarray(values, dtype=np.float64))
    n, total = len(values), values.sum()
    if n == 0 or total == 0:
        return None
    ranks = np.arange(1, n + 1)
    return float(((2 * ranks - n - 1) * values).sum() / (n * total))


def top_share(values: np.ndarray, share: float = TOP_SHARE) -> float | None:
    """Share of the total held by the top ceil(share * n) users."""
    values = np.sort(np.asarray(values, dtype=np.float64))[::-1]
    total = values.sum()
    if total == 0:
        return None
    return float(values[: int(np.ceil(share * len(values)))].sum() / total)


def lorenz(values: np.ndarray, points: int = 101) -> dict[str, list[float]]:
    values = np.sort(np.asarray(values, dtype=np.float64))
    cumulative = np.r_[0.0, np.cumsum(values)] / max(values.sum(), 1e-12)
    grid = np.linspace(0, 1, points)
    return {"user_share": grid.tolist(), "purchase_share": np.interp(grid, np.linspace(0, 1, len(cumulative)), cumulative).tolist()}


def percentile_ci(values: list[float]) -> tuple[float | None, float | None]:
    if not values:
        return None, None
    low, high = np.percentile(values, [2.5, 97.5])
    return float(low), float(high)


# ---------------------------------------------------------------------------------------------
# Metrics


@dataclass
class Stability:
    """Per-user halves for M1: eligible user rows and pi(u, a) in each half."""

    rows: np.ndarray
    first: np.ndarray
    second: np.ndarray

    def rho(self, user_weights: np.ndarray | None = None) -> float | None:
        if len(self.rows) == 0:
            return None
        return weighted_spearman(self.first, self.second, None if user_weights is None else user_weights[self.rows])


def stability(data: Purchases, flag: np.ndarray, in_c: np.ndarray, smoothing: float, min_n_c: int) -> Stability:
    """M1 inputs: users with n_C >= min_n_c in both halves; pi = (n_a + lambda pi_bar) / (n_C + lambda),
    with pi_bar from the whole history."""
    n_a, n_c = counts(data.history, flag, in_c)
    prior = float(n_a.sum() / n_c.sum()) if n_c.sum() else 0.5
    a1, c1 = counts(data.first, flag, in_c)
    a2, c2 = counts(data.second, flag, in_c)
    rows = np.flatnonzero((c1 >= min_n_c) & (c2 >= min_n_c))
    return Stability(
        rows,
        (a1[rows] + smoothing * prior) / (c1[rows] + smoothing),
        (a2[rows] + smoothing * prior) / (c2[rows] + smoothing),
    )


def extension_outcomes(data: Purchases, flag: np.ndarray, in_c: np.ndarray) -> dict[str, np.ndarray]:
    """M2 inputs over users with n_C >= 1: history attribute buyer, and whether the target has a new
    attribute item (outcome) or a new non-attribute item (activity control) inside C_a."""
    n_a, n_c = counts(data.history, flag, in_c)
    base = np.flatnonzero(n_c >= 1)
    attribute_new = data.new_target @ (flag & in_c).astype(np.float64) > 0
    control_new = data.new_target @ (~flag & in_c).astype(np.float64) > 0
    return {"has": n_a[base] >= 1, "attribute": attribute_new[base], "control": control_new[base]}


def relative_risk(has: np.ndarray, outcome: np.ndarray, weights: np.ndarray | None = None) -> float | None:
    weights = np.ones(len(has)) if weights is None else weights
    exposed, unexposed = weights[has].sum(), weights[~has].sum()
    if exposed == 0 or unexposed == 0:
        return None
    risk_unexposed = (weights[~has] * outcome[~has]).sum() / unexposed
    if risk_unexposed == 0:
        return None
    return float(((weights[has] * outcome[has]).sum() / exposed) / risk_unexposed)


def extension(outcomes: dict[str, np.ndarray], resamples: int, seed: int) -> dict[str, object]:
    """RR of a new attribute item (and of a new non-attribute item) for history buyers vs non-buyers."""
    has = outcomes["has"]
    rng = np.random.default_rng(seed)
    draws = {"attribute": [], "control": []}
    for _ in range(resamples):
        weights = np.bincount(rng.integers(0, len(has), len(has)), minlength=len(has)).astype(np.float64)
        for key in draws:
            value = relative_risk(has, outcomes[key], weights)
            if value is not None:
                draws[key].append(value)
    result: dict[str, object] = {
        "users": int(len(has)),
        "buyers": int(has.sum()),
        "non_buyers": int((~has).sum()),
    }
    for key in ["attribute", "control"]:
        outcome = outcomes[key]
        low, high = percentile_ci(draws[key])
        result[key] = {
            "rate_buyers": float(outcome[has].mean()) if has.any() else None,
            "rate_non_buyers": float(outcome[~has].mean()) if (~has).any() else None,
            "relative_risk": relative_risk(has, outcome),
            "ci_low": low,
            "ci_high": high,
            "valid_resamples": len(draws[key]),
        }
    return result


def concentration(data: Purchases, flag: np.ndarray, in_c: np.ndarray) -> dict[str, object]:
    """M3 over all history users: top-20% share and Gini of n_a, and the top-20% share of n_C."""
    n_a, n_c = counts(data.history, flag, in_c)
    attribute_top, category_top = top_share(n_a), top_share(n_c)
    return {
        "users": int(len(n_a)),
        "attribute_purchases": float(n_a.sum()),
        "top20_share": attribute_top,
        "gini": gini(n_a),
        "category_top20_share": category_top,
        "top20_share_ratio": attribute_top / category_top if attribute_top is not None and category_top else None,
    }


# ---------------------------------------------------------------------------------------------
# Placebo


def popularity_deciles(popularity: np.ndarray) -> np.ndarray:
    """Decile 0..9 of each item's popularity rank inside the group (ties by position)."""
    n = len(popularity)
    rank = np.empty(n, dtype=np.int64)
    rank[np.lexsort((np.arange(n), popularity))] = np.arange(n)
    return (rank * 10) // max(n, 1)


def placebo_flag(
    data: Purchases, flag: np.ndarray, aisles: set[str], rng: np.random.Generator
) -> tuple[np.ndarray, dict[str, int]]:
    """In each C_a aisle, as many random non-attribute items as there are attribute items, matched per
    popularity decile; a decile short of non-attribute items borrows from the nearest deciles."""
    fake = np.zeros(len(flag), dtype=bool)
    stats = {"items": 0, "same_decile": 0, "nearby_decile": 0, "unmatched": 0}
    for aisle in sorted(aisles):
        members = np.flatnonzero(data.item_aisle == aisle)
        decile = popularity_deciles(data.popularity[members])
        is_attr = flag[members]
        available = np.flatnonzero(~is_attr)
        used = np.zeros(len(members), dtype=bool)
        short: dict[int, int] = {}
        for d in range(10):
            need = int((is_attr & (decile == d)).sum())
            pool = available[(decile[available] == d)]
            take = rng.choice(pool, size=min(need, len(pool)), replace=False) if need and len(pool) else np.zeros(0, dtype=np.int64)
            used[take] = True
            stats["same_decile"] += len(take)
            if need > len(take):
                short[d] = need - len(take)
        for d, need in short.items():
            pool = available[~used[available]]
            if len(pool) == 0:
                stats["unmatched"] += need
                continue
            order = np.lexsort((rng.random(len(pool)), np.abs(decile[pool] - d)))
            take = pool[order[:need]]
            used[take] = True
            stats["nearby_decile"] += len(take)
            stats["unmatched"] += need - len(take)
        fake[members[used]] = True
        stats["items"] += int(is_attr.sum())
    return fake, stats


# ---------------------------------------------------------------------------------------------
# Analysis


def verdict(rho: float | None, placebo_p95: float | None, differences: dict[str, dict[str, object]]) -> dict[str, object]:
    """Pre-registered rule: supported = rho above the placebo 95th percentile and the CI of rho(a) - rho(taste)
    above zero for every taste attribute; partially supported = one of the two; else not supported."""
    above_placebo = rho is not None and placebo_p95 is not None and rho > placebo_p95
    above_taste = bool(differences) and all(
        row["ci_low"] is not None and row["ci_low"] > 0 for row in differences.values()
    )
    met = {"m1_above_placebo_p95": bool(above_placebo), "m1_difference_ci_above_zero_for_all_taste": above_taste}
    if rho is None:
        label = "undetermined"
    elif above_placebo and above_taste:
        label = "supported"
    elif above_placebo or above_taste:
        label = "partially supported"
    else:
        label = "not supported"
    return {"verdict": label, "conditions": met}


def _summary(values: list[float]) -> dict[str, float | None]:
    clean = [value for value in values if value is not None]
    if not clean:
        return {"reps": 0, "mean": None, "p95": None, "min": None, "max": None}
    return {
        "reps": len(clean),
        "mean": float(np.mean(clean)),
        "p95": float(np.percentile(clean, 95)),
        "min": float(np.min(clean)),
        "max": float(np.max(clean)),
    }


def analyze(config: dict[str, object], items: pd.DataFrame, history: pd.DataFrame, target: pd.DataFrame) -> dict[str, object]:
    settings = config["hierarchical"]["hstar"]
    health = list(config["hierarchical"]["attributes"][DATASET])
    taste = list(settings["taste_attributes"])
    smoothing = float(config["attributes"]["smoothing"])
    min_half = int(settings["min_n_c_each_half"])
    min_n = int(config["attributes"]["distinct_min_n"])
    resamples, boot_seed = int(config["bootstrap"]["n"]), int(config["bootstrap"]["seed"])
    items = taste_attributes(items, taste)
    categories = attribute_categories(config, DATASET, items, health + taste)
    data = purchase_data(items, history, target)
    table = items.drop_duplicates("item_id").assign(item_id=lambda f: f["item_id"].astype(str)).set_index("item_id")
    flags = {name: table[attribute_column(name)].reindex(data.items).fillna(0).to_numpy().astype(bool) for name in health + taste}
    in_c = {name: np.isin(data.item_aisle, sorted(categories[name])) for name in health + taste}

    coverage: dict[str, object] = {}
    stabilities: dict[str, Stability] = {}
    for name in health + taste:
        n_a, n_c = counts(data.history, flags[name], in_c[name])
        stabilities[name] = stability(data, flags[name], in_c[name], smoothing, min_half)
        coverage[name] = {
            "group": "health" if name in health else "taste",
            "items_with_attribute": int(flags[name].sum()),
            "applicable_categories": len(categories[name]),
            "attribute_items_in_c_a": int((flags[name] & in_c[name]).sum()),
            "items_in_c_a": int(in_c[name].sum()),
            "history_attribute_purchases_in_c_a": int(n_a.sum()),
            "history_purchases_in_c_a": int(n_c.sum()),
            "users_with_n_c_at_least_min": int((n_c >= min_n).sum()),
            "users_m1_eligible": int(len(stabilities[name].rows)),
        }
    health_size = np.median([coverage[name]["history_attribute_purchases_in_c_a"] for name in health])
    for name in taste:
        size = coverage[name]["history_attribute_purchases_in_c_a"]
        ratio = size / health_size if health_size else None
        coverage[name]["size_ratio_to_health_median"] = ratio
        coverage[name]["size_differs"] = bool(ratio is None or ratio > SIZE_RATIO_WARNING or ratio < 1 / SIZE_RATIO_WARNING)

    rng = np.random.default_rng(boot_seed)
    draws = {name: [] for name in health + taste}
    differences = {(a, b): [] for a in health for b in taste}
    for _ in range(resamples):
        weights = np.bincount(rng.integers(0, len(data.users), len(data.users)), minlength=len(data.users)).astype(np.float64)
        values = {name: stabilities[name].rho(weights) for name in health + taste}
        for name, value in values.items():
            if value is not None:
                draws[name].append(value)
        for a, b in differences:
            if values[a] is not None and values[b] is not None:
                differences[(a, b)].append(values[a] - values[b])

    attributes: dict[str, object] = {}
    lorenz_curves: dict[str, object] = {}
    for name in health + taste:
        rho = stabilities[name].rho()
        low, high = percentile_ci(draws[name])
        attributes[name] = {
            "m1": {"users": int(len(stabilities[name].rows)), "rho": rho, "ci_low": low, "ci_high": high},
            "m2": extension(extension_outcomes(data, flags[name], in_c[name]), resamples, boot_seed),
            "m3": concentration(data, flags[name], in_c[name]),
        }
        lorenz_curves[name] = lorenz(counts(data.history, flags[name], in_c[name])[0])

    placebo: dict[str, object] = {}
    for column, name in enumerate(health):
        m1, m2, m3_top, m3_gini, matching = [], [], [], [], []
        for rep in range(int(settings["placebo_reps"])):
            fake, stats = placebo_flag(data, flags[name], categories[name], np.random.default_rng([int(settings["seed"]), column, rep]))
            matching.append(stats)
            m1.append(stability(data, fake, in_c[name], smoothing, min_half).rho())
            outcomes = extension_outcomes(data, fake, in_c[name])
            m2.append(relative_risk(outcomes["has"], outcomes["attribute"]))
            m3 = concentration(data, fake, in_c[name])
            m3_top.append(m3["top20_share"])
            m3_gini.append(m3["gini"])
        placebo[name] = {
            "m1_rho": _summary(m1),
            "m2_relative_risk": _summary(m2),
            "m3_top20_share": _summary(m3_top),
            "m3_gini": _summary(m3_gini),
            "matching": {key: int(np.mean([row[key] for row in matching])) for key in matching[0]} if matching else {},
        }

    verdicts: dict[str, object] = {}
    for name in health:
        diffs = {}
        for other in taste:
            low, high = percentile_ci(differences[(name, other)])
            a, b = attributes[name]["m1"]["rho"], attributes[other]["m1"]["rho"]
            diffs[other] = {
                "difference": a - b if a is not None and b is not None else None,
                "ci_low": low,
                "ci_high": high,
                "valid_resamples": len(differences[(name, other)]),
            }
        attributes[name]["m1_minus_taste"] = diffs
        verdicts[name] = verdict(attributes[name]["m1"]["rho"], placebo[name]["m1_rho"]["p95"], diffs)
    return {
        "coverage": coverage,
        "attributes": attributes,
        "placebo": placebo,
        "verdicts": verdicts,
        "lorenz": lorenz_curves,
        "history_users": int(len(data.users)),
        "users_with_new_target_items": int((data.new_target.getnnz(axis=1) > 0).sum()),
    }


def _figures(report: dict[str, object], health: list[str], taste: list[str], base: Path) -> dict[str, str]:
    base.mkdir(parents=True, exist_ok=True)
    names = health + taste
    attributes = report["attributes"]
    rho = [attributes[name]["m1"]["rho"] or 0.0 for name in names]
    low = [attributes[name]["m1"]["ci_low"] for name in names]
    high = [attributes[name]["m1"]["ci_high"] for name in names]
    errors = np.array([[r - (l if l is not None else r) for r, l in zip(rho, low)], [(h if h is not None else r) - r for r, h in zip(rho, high)]])
    colors = ["#2f6db5"] * len(health) + ["#c7792b"] * len(taste)
    figure, axis = plt.subplots(figsize=(9, 5))
    positions = np.arange(len(names))
    axis.bar(positions, rho, yerr=errors, color=colors, capsize=4)
    for position, name in enumerate(health):
        summary = report["placebo"][name]["m1_rho"]
        if summary["reps"]:
            axis.plot([position, position], [summary["min"], summary["max"]], color="black", linewidth=6, alpha=0.25)
            axis.scatter([position], [summary["p95"]], color="black", marker="_", s=400, label="placebo p95" if position == 0 else None)
    axis.set_xticks(positions, names, rotation=20)
    axis.set_ylabel("Spearman rho, pi(u, a) first vs second half")
    axis.set_title("M1 stability: health (blue) vs taste (orange); grey = placebo range")
    axis.legend(loc="lower right")
    figure.tight_layout()
    m1_path = base / f"{report['profile']}_hstar_m1.png"
    figure.savefig(m1_path, dpi=160)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(6, 6))
    axis.plot([0, 1], [0, 1], color="grey", linewidth=1)
    for name in names:
        curve = report["lorenz"][name]
        axis.plot(curve["user_share"], curve["purchase_share"], linestyle="-" if name in health else "--", label=name)
    axis.set_xlabel("Cumulative share of users (ascending n_a)")
    axis.set_ylabel("Cumulative share of attribute purchases in C_a")
    axis.set_title("M3 Lorenz curves (history)")
    axis.legend(fontsize=8)
    figure.tight_layout()
    m3_path = base / f"{report['profile']}_hstar_m3_lorenz.png"
    figure.savefig(m3_path, dpi=160)
    plt.close(figure)
    return {"m1": m1_path.as_posix(), "m3_lorenz": m3_path.as_posix()}


def analyze_hstar(config: dict[str, object]) -> dict[str, object]:
    """Run the H* analysis on the Instacart eval split and write the report and two figures."""
    started = time.perf_counter()
    items, history, target = load_split(config, DATASET, "eval")
    report = {
        "profile": config["profile"],
        "dataset": DATASET,
        "split": "eval",
        "design": "post-hoc (fixed after the repeat-explore results); descriptive, separate from recommendation metrics",
        "settings": {
            **config["hierarchical"]["hstar"],
            "smoothing": float(config["attributes"]["smoothing"]),
            "bootstrap": config["bootstrap"],
            "c_a_rule": {"min_with": config["attributes"]["min_with"], "min_without": config["attributes"]["min_without"]},
            "halves": "first floor(n_b / 2) history baskets vs the rest, by basket_seq",
            "m2_population": "users with at least one history purchase inside C_a",
        },
        **analyze(config, items, history, target),
    }
    health = list(config["hierarchical"]["attributes"][DATASET])
    taste = list(config["hierarchical"]["hstar"]["taste_attributes"])
    report["figures"] = _figures(report, health, taste, Path(config["report_dir"]) / "figures")
    report["runtime_seconds"] = time.perf_counter() - started
    _write_json(Path(config["report_dir"]) / "metrics" / f"{config['profile']}_hstar.json", report)
    return {key: value for key, value in report.items() if key != "lorenz"}
