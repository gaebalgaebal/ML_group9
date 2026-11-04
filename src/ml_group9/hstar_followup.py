"""Post-hoc sensitivity analysis for H* (does not change the pre-registered verdict in the H* report).

Follows up on `hstar.analyze` on the Instacart eval split:
- taste-attribute placebos (same procedure as the health placebos),
- placebo-adjusted stability Delta_a = M1(a) - mean placebo M1, with a user bootstrap in which the
  attribute and its placebos share the resampled users,
- placebo extraction checks and an aisle-filtered sensitivity (S1),
- ties, a Bonferroni (98.75%) version of the H* M1 difference CIs, taste-definition checks,
- item concentration of attribute purchases and a drop-the-top-item sensitivity (S2),
- M1 on new items only (M1_new), which removes repeat purchases of the same item by construction.

Everything is computed from history only; the target is never read.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import sparse

from .attributes import attribute_column
from .config import load_config
from .data import load_split
from .hierarchical import attribute_categories
from .hstar import (
    DATASET,
    Purchases,
    Stability,
    _summary,
    counts,
    percentile_ci,
    placebo_flag,
    purchase_data,
    stability,
    taste_attributes,
)
from .pipeline import _write_json


NEW_MIN_N_C = 5
NEW_MIN_N_C_SECONDARY = 10
# S1 keeps aisles with at least this many non-attribute items per attribute item.
S1_RATIO = 2.0
HIGH_EXTRACTION = 0.5
TOP_NAMES = 20
TOP_CONCENTRATION = 5
NEGATION_PATTERNS = ["non[- ]chocolate", "chocolate[- ]free", "no chocolate", "vanilla[- ]free", "strawberry[- ]free"]
BONFERRONI_LEVEL = 1 - 0.05 / 4
TIME_BUDGET_SECONDS = 90 * 60
REDUCED_RESAMPLES = 500
# Resamples timed before projecting the bootstrap runtime.
PROBE_RESAMPLES = 20
REPRODUCTION_TOLERANCE = 1e-9


# ---------------------------------------------------------------------------------------------
# New-item purchases


def new_item_halves(items: pd.DataFrame, history: pd.DataFrame, data: Purchases) -> tuple[sparse.csr_matrix, sparse.csr_matrix]:
    """(first, second) users x items matrices of new-item purchases: (u, b, i) where i is in none of u's
    earlier history baskets. Baskets are ordered and split into halves exactly as in `purchase_data`."""
    implicit = history[["user_id", "basket_id", "basket_seq", "item_id"]].copy()
    for column in ["user_id", "basket_id", "item_id"]:
        implicit[column] = implicit[column].astype(str)
    implicit = implicit.drop_duplicates(["user_id", "basket_id", "item_id"])
    implicit = implicit[implicit["item_id"].isin(data.items)]
    baskets = implicit[["user_id", "basket_id", "basket_seq"]].drop_duplicates().sort_values(
        ["user_id", "basket_seq", "basket_id"], kind="mergesort"
    )
    baskets["rank"] = baskets.groupby("user_id").cumcount()
    baskets["n_b"] = baskets.groupby("user_id")["basket_id"].transform("size")
    implicit = implicit.merge(baskets[["user_id", "basket_id", "rank", "n_b"]], on=["user_id", "basket_id"])
    first_seen = implicit.sort_values(["user_id", "rank"], kind="mergesort").drop_duplicates(["user_id", "item_id"], keep="first")
    rows = data.users.get_indexer(first_seen["user_id"])
    cols = data.items.get_indexer(first_seen["item_id"])
    early = (first_seen["rank"] < first_seen["n_b"] // 2).to_numpy()
    shape = (len(data.users), len(data.items))

    def matrix(mask: np.ndarray) -> sparse.csr_matrix:
        return sparse.csr_matrix((np.ones(int(mask.sum())), (rows[mask], cols[mask])), shape=shape)

    return matrix(early), matrix(~early)


def new_item_data(data: Purchases, first_new: sparse.csr_matrix, second_new: sparse.csr_matrix) -> Purchases:
    """Purchases whose halves hold only new-item purchases; the smoothing prior of `stability` then comes
    from all new-item purchases."""
    return dataclasses.replace(data, history=sparse.csr_matrix(first_new + second_new), first=first_new, second=second_new)


# ---------------------------------------------------------------------------------------------
# Statistics


class RankedHalves:
    """`Stability.rho` with the sorting of each half done once, for repeated bootstrap weights.
    Uses the same operations as `hstar.weighted_spearman`."""

    def __init__(self, halves: Stability):
        self.rows = halves.rows
        _, self.inverse_x = np.unique(halves.first, return_inverse=True)
        _, self.inverse_y = np.unique(halves.second, return_inverse=True)
        self.size_x, self.size_y = int(self.inverse_x.max(initial=-1)) + 1, int(self.inverse_y.max(initial=-1)) + 1

    @staticmethod
    def _ranks(inverse: np.ndarray, size: int, weights: np.ndarray) -> np.ndarray:
        mass = np.bincount(inverse, weights=weights, minlength=size)
        below = np.cumsum(mass) - mass
        return (below + (mass + 1.0) / 2.0)[inverse]

    def rho(self, user_weights: np.ndarray) -> float | None:
        if len(self.rows) == 0:
            return None
        weights = user_weights[self.rows]
        if weights.sum() < 3:
            return None
        rx, ry = self._ranks(self.inverse_x, self.size_x, weights), self._ranks(self.inverse_y, self.size_y, weights)
        total = weights.sum()
        mx, my = (weights * rx).sum() / total, (weights * ry).sum() / total
        cov = (weights * (rx - mx) * (ry - my)).sum()
        var = (weights * (rx - mx) ** 2).sum() * (weights * (ry - my) ** 2).sum()
        return float(cov / np.sqrt(var)) if var > 0 else None


def mean_or_none(values: list[float | None]) -> float | None:
    clean = [value for value in values if value is not None]
    return float(np.mean(clean)) if clean else None


def delta(rho: float | None, placebo_rhos: list[float | None]) -> float | None:
    """Delta = M1(a) - mean of the placebo M1 values (placebos without a value are left out)."""
    placebo_mean = mean_or_none(placebo_rhos)
    return rho - placebo_mean if rho is not None and placebo_mean is not None else None


def level_ci(values: list[float], level: float) -> tuple[float | None, float | None]:
    if not values:
        return None, None
    tail = (1 - level) / 2 * 100
    low, high = np.percentile(values, [tail, 100 - tail])
    return float(low), float(high)


def share_summary(values: list[float | None]) -> dict[str, float | None]:
    clean = [value for value in values if value is not None]
    if not clean:
        return {"reps": 0, "mean": None, "p5": None, "p95": None}
    return {"reps": len(clean), "mean": float(np.mean(clean)), "p5": float(np.percentile(clean, 5)), "p95": float(np.percentile(clean, 95))}


def item_concentration(item_counts: np.ndarray, k: int = TOP_CONCENTRATION) -> dict[str, float | None]:
    """Shares of the top-1 and top-k items and HHI = sum_i s_i^2 over item purchase counts."""
    values = np.sort(np.asarray(item_counts, dtype=np.float64))[::-1]
    total = values.sum()
    if total == 0:
        return {"top1_share": None, f"top{k}_share": None, "hhi": None}
    shares = values / total
    return {"top1_share": float(shares[0]), f"top{k}_share": float(shares[:k].sum()), "hhi": float((shares**2).sum())}


def item_purchases(data: Purchases) -> np.ndarray:
    """History purchases (user, basket) of each item."""
    return np.asarray(data.history.sum(axis=0)).ravel()


def without_top_item(flag: np.ndarray, in_c: np.ndarray, purchases: np.ndarray) -> tuple[np.ndarray, np.ndarray, int | None]:
    """Drop the most purchased attribute item inside C_a from both the attribute set and C_a (S2)."""
    candidates = np.flatnonzero(flag & in_c)
    if len(candidates) == 0:
        return flag, in_c, None
    # Stable choice on ties: the lowest item index among the most purchased.
    top = int(candidates[np.argmax(purchases[candidates])])
    flag, in_c = flag.copy(), in_c.copy()
    flag[top] = False
    in_c[top] = False
    return flag, in_c, top


def aisle_extraction(data: Purchases, flag: np.ndarray, fakes: list[np.ndarray], aisles: set[str]) -> list[dict[str, object]]:
    """Per C_a aisle: attribute and non-attribute item counts, placebo items and extraction ratio
    (placebo items / non-attribute items), averaged over the placebo repetitions."""
    rows = []
    for aisle in sorted(aisles):
        members = data.item_aisle == aisle
        attribute_items, other_items = int((flag & members).sum()), int((~flag & members).sum())
        placebo_items = float(np.mean([(fake & members).sum() for fake in fakes])) if fakes else 0.0
        rows.append(
            {
                "aisle": aisle,
                "attribute_items": attribute_items,
                "non_attribute_items": other_items,
                "placebo_items_mean": placebo_items,
                "extraction_ratio": placebo_items / other_items if other_items else None,
            }
        )
    return rows


def s1_aisles(data: Purchases, flag: np.ndarray, aisles: set[str], ratio: float = S1_RATIO) -> set[str]:
    """C_a' = aisles of C_a with non-attribute items >= ratio x attribute items."""
    kept = set()
    for aisle in aisles:
        members = data.item_aisle == aisle
        if (~flag & members).sum() >= ratio * (flag & members).sum():
            kept.add(aisle)
    return kept


def both_zero_share(data: Purchases, flag: np.ndarray, in_c: np.ndarray, rows: np.ndarray) -> float | None:
    """Share of the eligible users with no attribute purchase inside C_a in either half (raw pi = 0)."""
    if len(rows) == 0:
        return None
    first, _ = counts(data.first, flag, in_c)
    second, _ = counts(data.second, flag, in_c)
    return float(((first[rows] == 0) & (second[rows] == 0)).mean())


def decision(ci_lows: list[float | None]) -> str:
    """Post-hoc comparison rule fixed before the run: every CI strictly above zero."""
    if ci_lows and all(low is not None and low > 0 for low in ci_lows):
        return "higher than all three taste attributes after placebo adjustment"
    return "difference not established"


# ---------------------------------------------------------------------------------------------
# Analysis


def _placebo_column(name: str, health: list[str], taste: list[str]) -> int:
    """Seed column: the health placebos keep the index used by hstar.analyze; taste attributes follow."""
    return (health + taste).index(name)


def analyze(
    config: dict[str, object],
    items: pd.DataFrame,
    history: pd.DataFrame,
    reference: dict[str, object] | None = None,
    time_budget: float = TIME_BUDGET_SECONDS,
) -> dict[str, object]:
    started = time.perf_counter()
    settings = config["hierarchical"]["hstar"]
    health = list(config["hierarchical"]["attributes"][DATASET])
    taste = list(settings["taste_attributes"])
    names = health + taste
    smoothing = float(config["attributes"]["smoothing"])
    min_half = int(settings["min_n_c_each_half"])
    reps = int(settings["placebo_reps"])
    placebo_seed = int(settings["seed"])
    resamples, boot_seed = int(config["bootstrap"]["n"]), int(config["bootstrap"]["seed"])

    items = taste_attributes(items, taste)
    categories = attribute_categories(config, DATASET, items, names)
    # The target is not used: an empty frame keeps purchase_data's history matrices unchanged.
    data = purchase_data(items, history, history.iloc[:0])
    table = items.drop_duplicates("item_id").assign(item_id=lambda f: f["item_id"].astype(str)).set_index("item_id")
    product_names = table["product_name"].reindex(data.items).fillna("").astype(str).to_numpy()
    flags = {name: table[attribute_column(name)].reindex(data.items).fillna(0).to_numpy().astype(bool) for name in names}
    in_c = {name: np.isin(data.item_aisle, sorted(categories[name])) for name in names}
    purchases = item_purchases(data)
    new_data = new_item_data(data, *new_item_halves(items, history, data))

    # Placebo sets: for health attributes the same draws as hstar.analyze.
    fakes: dict[str, list[np.ndarray]] = {}
    unmatched: dict[str, list[int]] = {}
    for name in names:
        column = _placebo_column(name, health, taste)
        fakes[name], unmatched[name] = [], []
        for rep in range(reps):
            fake, stats = placebo_flag(data, flags[name], categories[name], np.random.default_rng([placebo_seed, column, rep]))
            fakes[name].append(fake)
            unmatched[name].append(int(stats["unmatched"]))

    halves = {name: stability(data, flags[name], in_c[name], smoothing, min_half) for name in names}
    placebo_halves = {name: [stability(data, fake, in_c[name], smoothing, min_half) for fake in fakes[name]] for name in names}
    rho = {name: halves[name].rho() for name in names}
    placebo_rho = {name: [s.rho() for s in placebo_halves[name]] for name in names}

    reproduction = _reproduce(reference, health, taste, rho, placebo_rho)
    if reproduction["checked"] and not reproduction["passed"]:
        raise RuntimeError(f"M1 or placebo values differ from the H* report: {json.dumps(reproduction, default=str)}")

    new_halves = {name: stability(new_data, flags[name], in_c[name], smoothing, NEW_MIN_N_C) for name in names}
    new_placebo_halves = {name: [stability(new_data, fake, in_c[name], smoothing, NEW_MIN_N_C) for fake in fakes[name]] for name in names}
    new_rho = {name: new_halves[name].rho() for name in names}
    new_placebo_rho = {name: [s.rho() for s in new_placebo_halves[name]] for name in names}

    boot = _bootstrap(
        data, names, health, taste, halves, placebo_halves, new_halves, new_placebo_halves,
        resamples, boot_seed, time_budget - (time.perf_counter() - started),
    )

    attributes: dict[str, object] = {}
    for name in names:
        same_users = all(np.array_equal(s.rows, halves[name].rows) for s in placebo_halves[name])
        new_same_users = all(np.array_equal(s.rows, new_halves[name].rows) for s in new_placebo_halves[name])
        low, high = percentile_ci(boot["delta"][name])
        new_low, new_high = percentile_ci(boot["delta_new"][name])
        secondary = stability(new_data, flags[name], in_c[name], smoothing, NEW_MIN_N_C_SECONDARY)
        secondary_placebo = [stability(new_data, fake, in_c[name], smoothing, NEW_MIN_N_C_SECONDARY).rho() for fake in fakes[name]]
        attributes[name] = {
            "group": "health" if name in health else "taste",
            "m1": rho[name],
            "users": int(len(halves[name].rows)),
            "placebo_m1": _summary(placebo_rho[name]),
            "placebo_users_same_as_attribute": bool(same_users),
            "placebo_users_range": _range([len(s.rows) for s in placebo_halves[name]]),
            "delta": {"value": delta(rho[name], placebo_rho[name]), "ci_low": low, "ci_high": high, "valid_resamples": len(boot["delta"][name])},
            "ties": {
                "attribute_both_halves_zero_share": both_zero_share(data, flags[name], in_c[name], halves[name].rows),
                "placebo_both_halves_zero_share_mean": mean_or_none(
                    [both_zero_share(data, fake, in_c[name], s.rows) for fake, s in zip(fakes[name], placebo_halves[name])]
                ),
            },
            "new_items": {
                "share_of_c_a_purchases": _new_share(data, new_data, flags[name], in_c[name]),
                "min_n_c_new_5": {
                    "users": int(len(new_halves[name].rows)),
                    "m1_new": new_rho[name],
                    "placebo_m1_new": _summary(new_placebo_rho[name]),
                    "placebo_users_same_as_attribute": bool(new_same_users),
                    "delta_new": {
                        "value": delta(new_rho[name], new_placebo_rho[name]),
                        "ci_low": new_low,
                        "ci_high": new_high,
                        "valid_resamples": len(boot["delta_new"][name]),
                    },
                    "ties": {
                        "attribute_both_halves_zero_share": both_zero_share(new_data, flags[name], in_c[name], new_halves[name].rows),
                        "placebo_both_halves_zero_share_mean": mean_or_none(
                            [both_zero_share(new_data, fake, in_c[name], s.rows) for fake, s in zip(fakes[name], new_placebo_halves[name])]
                        ),
                    },
                },
                "min_n_c_new_10": {
                    "users": int(len(secondary.rows)),
                    "m1_new": secondary.rho(),
                    "placebo_m1_new": _summary(secondary_placebo),
                    "delta_new": delta(secondary.rho(), secondary_placebo),
                },
            },
        }

    delta_differences, new_delta_differences = {}, {}
    for a in health:
        for b in taste:
            key = f"{a}-{b}"
            low, high = percentile_ci(boot["delta_diff"][(a, b)])
            delta_differences[key] = {
                "difference": _minus(attributes[a]["delta"]["value"], attributes[b]["delta"]["value"]),
                "ci_low": low,
                "ci_high": high,
                "valid_resamples": len(boot["delta_diff"][(a, b)]),
            }
            low, high = percentile_ci(boot["delta_new_diff"][(a, b)])
            new_delta_differences[key] = {
                "difference": _minus(
                    attributes[a]["new_items"]["min_n_c_new_5"]["delta_new"]["value"],
                    attributes[b]["new_items"]["min_n_c_new_5"]["delta_new"]["value"],
                ),
                "ci_low": low,
                "ci_high": high,
                "valid_resamples": len(boot["delta_new_diff"][(a, b)]),
            }

    multiplicity = {}
    for a in health:
        rows = {}
        for b in taste:
            draws = boot["m1_diff"][(a, b)]
            low95, high95 = percentile_ci(draws)
            low, high = level_ci(draws, BONFERRONI_LEVEL)
            rows[b] = {"difference": _minus(rho[a], rho[b]), "ci95_low": low95, "ci95_high": high95, "ci9875_low": low, "ci9875_high": high}
        multiplicity[a] = {
            "taste": rows,
            "all_taste_above_zero_95": all(row["ci95_low"] is not None and row["ci95_low"] > 0 for row in rows.values()),
            "all_taste_above_zero_9875": all(row["ci9875_low"] is not None and row["ci9875_low"] > 0 for row in rows.values()),
        }
    if reference is not None and boot["resamples"] == int(reference["settings"]["bootstrap"]["n"]):
        reproduction["m1_difference_ci95"] = _reproduce_differences(reference, multiplicity, health, taste)

    extraction, s1 = {}, {}
    for name in names:
        column = _placebo_column(name, health, taste)
        rows = aisle_extraction(data, flags[name], fakes[name], categories[name])
        aisle_purchases = {row["aisle"]: float(purchases[in_c[name] & (data.item_aisle == row["aisle"])].sum()) for row in rows}
        total = sum(aisle_purchases.values())
        for row in rows:
            row["c_a_purchase_share"] = aisle_purchases[row["aisle"]] / total if total else None
        high = [row for row in rows if row["extraction_ratio"] is not None and row["extraction_ratio"] >= HIGH_EXTRACTION]
        extraction[name] = {
            "aisles": rows,
            "unmatched_per_rep": unmatched[name],
            "unmatched_mean": float(np.mean(unmatched[name])),
            "high_ratio_aisles": len(high),
            "high_ratio_c_a_purchase_share": float(sum(row["c_a_purchase_share"] for row in high)) if total else None,
        }

        kept = s1_aisles(data, flags[name], categories[name])
        kept_in_c = np.isin(data.item_aisle, sorted(kept))
        kept_rho = stability(data, flags[name], kept_in_c, smoothing, min_half)
        kept_placebo = [
            stability(
                data, placebo_flag(data, flags[name], kept, np.random.default_rng([placebo_seed, column, rep]))[0], kept_in_c, smoothing, min_half
            ).rho()
            for rep in range(reps)
        ]
        s1[name] = {
            "aisles_kept": len(kept),
            "aisles_dropped": sorted(categories[name] - kept),
            "users": int(len(kept_rho.rows)),
            "m1": kept_rho.rho(),
            "placebo_m1": _summary(kept_placebo),
            "delta": delta(kept_rho.rho(), kept_placebo),
            "delta_original": attributes[name]["delta"]["value"],
        }

    concentration, s2 = {}, {}
    for name in names:
        inside = flags[name] & in_c[name]
        order = np.flatnonzero(inside)[np.argsort(-purchases[inside], kind="mergesort")]
        placebo_values = [item_concentration(purchases[fake & in_c[name]]) for fake in fakes[name]]
        concentration[name] = {
            **item_concentration(purchases[inside]),
            "placebo": {key: share_summary([row[key] for row in placebo_values]) for key in ["top1_share", "top5_share", "hhi"]},
            "top5_items": [
                {"item_id": data.items[i], "product_name": product_names[i], "history_purchases": int(purchases[i])}
                for i in order[:TOP_CONCENTRATION]
            ],
        }

        flag2, in_c2, top = without_top_item(flags[name], in_c[name], purchases)
        reduced = stability(data, flag2, in_c2, smoothing, min_half)
        reduced_placebo, reduced_users = [], []
        for fake in fakes[name]:
            fake2, fake_in_c2, _ = without_top_item(fake, in_c[name], purchases)
            result = stability(data, fake2, fake_in_c2, smoothing, min_half)
            reduced_placebo.append(result.rho())
            reduced_users.append(len(result.rows))
        value, original = delta(reduced.rho(), reduced_placebo), attributes[name]["delta"]["value"]
        s2[name] = {
            "removed_item": None if top is None else {"item_id": data.items[top], "product_name": product_names[top], "history_purchases": int(purchases[top])},
            "users": int(len(reduced.rows)),
            "users_original": attributes[name]["users"],
            "placebo_users_range": _range(reduced_users),
            "m1": reduced.rho(),
            "m1_original": rho[name],
            "placebo_m1": _summary(reduced_placebo),
            "delta": value,
            "delta_original": original,
            "delta_ratio": value / original if value is not None and original else None,
            "halved_or_less": bool(value is not None and original is not None and original > 0 and value <= original / 2),
        }

    taste_definition = _taste_definition(data, flags, in_c, taste, product_names, purchases, items)
    interpretation = _interpretation(health, taste, attributes, delta_differences, new_delta_differences, s2)
    return {
        "reproduction": reproduction,
        "attributes": attributes,
        "delta_differences": delta_differences,
        "delta_new_differences": new_delta_differences,
        "extraction": extraction,
        "s1": s1,
        "multiplicity": multiplicity,
        "taste_definition": taste_definition,
        "concentration": concentration,
        "s2": s2,
        "bootstrap": {"requested": resamples, "used": boot["resamples"], "reduced": boot["reduced"], "seed": boot_seed, "projected_seconds": boot["projected_seconds"]},
        "interpretation": interpretation,
        "history_users": int(len(data.users)),
    }


def _bootstrap(
    data: Purchases,
    names: list[str],
    health: list[str],
    taste: list[str],
    halves: dict[str, Stability],
    placebo_halves: dict[str, list[Stability]],
    new_halves: dict[str, Stability],
    new_placebo_halves: dict[str, list[Stability]],
    resamples: int,
    seed: int,
    budget: float,
) -> dict[str, object]:
    """User bootstrap: one weight vector per resample shared by every attribute, its placebos and the
    new-item versions (the same draws as hstar.analyze for the same seed)."""
    ranked = {name: RankedHalves(halves[name]) for name in names}
    ranked_placebo = {name: [RankedHalves(s) for s in placebo_halves[name]] for name in names}
    ranked_new = {name: RankedHalves(new_halves[name]) for name in names}
    ranked_new_placebo = {name: [RankedHalves(s) for s in new_placebo_halves[name]] for name in names}
    pairs = [(a, b) for a in health for b in taste]
    draws = {key: {name: [] for name in names} for key in ["delta", "delta_new"]}
    draws.update({key: {pair: [] for pair in pairs} for key in ["m1_diff", "delta_diff", "delta_new_diff"]})
    rng = np.random.default_rng(seed)
    started, target, reduced, projected = time.perf_counter(), resamples, False, None
    done = 0
    while done < target:
        weights = np.bincount(rng.integers(0, len(data.users), len(data.users)), minlength=len(data.users)).astype(np.float64)
        m1 = {name: ranked[name].rho(weights) for name in names}
        d = {name: delta(m1[name], [r.rho(weights) for r in ranked_placebo[name]]) for name in names}
        d_new = {name: delta(ranked_new[name].rho(weights), [r.rho(weights) for r in ranked_new_placebo[name]]) for name in names}
        for name in names:
            if d[name] is not None:
                draws["delta"][name].append(d[name])
            if d_new[name] is not None:
                draws["delta_new"][name].append(d_new[name])
        for a, b in pairs:
            for key, values in [("m1_diff", m1), ("delta_diff", d), ("delta_new_diff", d_new)]:
                if values[a] is not None and values[b] is not None:
                    draws[key][(a, b)].append(values[a] - values[b])
        done += 1
        if done == min(PROBE_RESAMPLES, resamples) and projected is None:
            projected = (time.perf_counter() - started) / done * resamples
            if projected > budget and resamples > REDUCED_RESAMPLES:
                target, reduced = REDUCED_RESAMPLES, True
    return {**draws, "resamples": done, "reduced": reduced, "projected_seconds": projected}


def _new_share(data: Purchases, new_data: Purchases, flag: np.ndarray, in_c: np.ndarray) -> dict[str, dict[str, float | None]]:
    """Per half, new-item purchases / all purchases inside C_a (all items and attribute items), pooled over users."""
    result = {}
    for half in ["first", "second"]:
        new_a, new_c = counts(getattr(new_data, half), flag, in_c)
        all_a, all_c = counts(getattr(data, half), flag, in_c)
        result[half] = {
            "c_a": float(new_c.sum() / all_c.sum()) if all_c.sum() else None,
            "attribute": float(new_a.sum() / all_a.sum()) if all_a.sum() else None,
        }
    return result


def _taste_definition(
    data: Purchases,
    flags: dict[str, np.ndarray],
    in_c: dict[str, np.ndarray],
    taste: list[str],
    product_names: np.ndarray,
    purchases: np.ndarray,
    items: pd.DataFrame,
) -> dict[str, object]:
    top = {}
    for name in taste:
        inside = np.flatnonzero(flags[name] & in_c[name])
        order = inside[np.argsort(-purchases[inside], kind="mergesort")][:TOP_NAMES]
        top[name] = [{"item_id": data.items[i], "product_name": product_names[i], "history_purchases": int(purchases[i])} for i in order]
    catalogue = pd.Series(product_names)
    negation = {}
    for pattern in NEGATION_PATTERNS:
        match = catalogue.str.contains(pattern, case=False, regex=True).to_numpy()
        negation[pattern] = {
            "catalogue_items": int(match.sum()),
            "flagged": {name: int((match & flags[name]).sum()) for name in taste},
            "flagged_in_c_a": {name: int((match & flags[name] & in_c[name]).sum()) for name in taste},
            "history_purchases": int(purchases[match].sum()),
            "product_names": sorted(set(catalogue[match].tolist()))[:TOP_NAMES],
        }
    return {"top_items_in_c_a": top, "negation": negation, "scope": "attribute items inside C_a, history purchases"}


def _interpretation(
    health: list[str],
    taste: list[str],
    attributes: dict[str, object],
    delta_differences: dict[str, object],
    new_delta_differences: dict[str, object],
    s2: dict[str, object],
) -> dict[str, object]:
    """Labels from the post-hoc rules fixed before the run."""
    result: dict[str, object] = {}
    for a in health:
        result[a] = {
            "delta_vs_taste": decision([delta_differences[f"{a}-{b}"]["ci_low"] for b in taste]),
            "delta_new_vs_taste": decision([new_delta_differences[f"{a}-{b}"]["ci_low"] for b in taste]),
        }
    for name in health + taste:
        low = attributes[name]["new_items"]["min_n_c_new_5"]["delta_new"]["ci_low"]
        result.setdefault(name, {})["delta_new_above_zero"] = (
            "new-item attribute share more consistent over time than placebo, excluding repeat purchases"
            if low is not None and low > 0
            else "difference not established once repeat purchases are excluded"
        )
        if s2[name]["halved_or_less"]:
            result[name]["s2"] = "placebo-adjusted stability depends heavily on the top item"
    result["note"] = (
        "Post-hoc; the pre-registered verdicts in the H* report are unchanged. M1_new does not control for type: "
        "new-item consistency mixes type choice and within-type attribute choice (separated in T2). Observational, not causal."
    )
    return result


def _reproduce(
    reference: dict[str, object] | None,
    health: list[str],
    taste: list[str],
    rho: dict[str, float | None],
    placebo_rho: dict[str, list[float | None]],
) -> dict[str, object]:
    if reference is None:
        return {"checked": False}
    rows, worst = {}, 0.0
    for name in health + taste:
        expected = reference["attributes"][name]["m1"]["rho"]
        rows[f"{name}.m1"] = {"reference": expected, "recomputed": rho[name], "abs_diff": abs(expected - rho[name])}
    for name in health:
        summary = _summary(placebo_rho[name])
        for key in ["mean", "p95"]:
            expected = reference["placebo"][name]["m1_rho"][key]
            rows[f"{name}.placebo_{key}"] = {"reference": expected, "recomputed": summary[key], "abs_diff": abs(expected - summary[key])}
    worst = max(row["abs_diff"] for row in rows.values())
    return {"checked": True, "tolerance": REPRODUCTION_TOLERANCE, "max_abs_diff": worst, "passed": worst <= REPRODUCTION_TOLERANCE, "values": rows}


def _reproduce_differences(reference: dict[str, object], multiplicity: dict[str, object], health: list[str], taste: list[str]) -> dict[str, object]:
    worst = 0.0
    for a in health:
        for b in taste:
            expected = reference["attributes"][a]["m1_minus_taste"][b]
            row = multiplicity[a]["taste"][b]
            worst = max(worst, abs(expected["ci_low"] - row["ci95_low"]), abs(expected["ci_high"] - row["ci95_high"]))
    return {"max_abs_diff": worst, "passed": worst <= REPRODUCTION_TOLERANCE}


def _range(values: list[int]) -> dict[str, int]:
    return {"min": int(min(values)), "max": int(max(values))} if values else {"min": 0, "max": 0}


def _minus(a: float | None, b: float | None) -> float | None:
    return a - b if a is not None and b is not None else None


def _figure(report: dict[str, object], health: list[str], taste: list[str], path: Path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    names = health + taste
    rows = report["attributes"]
    series = [
        ("Delta (all purchases)", [rows[name]["delta"] for name in names], None),
        ("Delta_new (new items, n_C_new >= 5)", [rows[name]["new_items"]["min_n_c_new_5"]["delta_new"] for name in names], "//"),
    ]
    colors = ["#2f6db5" if name in health else "#c7792b" for name in names]
    figure, axis = plt.subplots(figsize=(10, 5))
    positions, width = np.arange(len(names)), 0.38
    for offset, (label, values, hatch) in zip([-width / 2, width / 2], series):
        point = [value["value"] if value["value"] is not None else 0.0 for value in values]
        low = [p - (v["ci_low"] if v["ci_low"] is not None else p) for p, v in zip(point, values)]
        high = [(v["ci_high"] if v["ci_high"] is not None else p) - p for p, v in zip(point, values)]
        axis.bar(positions + offset, point, width, yerr=[low, high], color=colors, hatch=hatch, edgecolor="black", linewidth=0.5, capsize=3, label=label)
    axis.axhline(0, color="grey", linewidth=1)
    axis.set_xticks(positions, names, rotation=20)
    axis.set_ylabel("M1 minus mean placebo M1 (Spearman rho)")
    axis.set_title("Placebo-adjusted M1: health (blue) and taste (orange), 95% bootstrap CI")
    axis.legend(loc="upper right")
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)
    return path.as_posix()


def analyze_hstar_followup(config: dict[str, object], reference_path: str | None = None) -> dict[str, object]:
    """Run the follow-up analysis on the Instacart eval split history and write the report and figure."""
    started = time.perf_counter()
    items, history, _ = load_split(config, DATASET, "eval")
    reference = None
    if reference_path:
        with open(reference_path, encoding="utf-8") as stream:
            reference = json.load(stream)
    report = {
        "profile": config["profile"],
        "dataset": DATASET,
        "split": "eval",
        "design": "post-hoc sensitivity analysis of H*; history only; the pre-registered H* verdicts are not changed",
        "reference": reference_path,
        "settings": {
            **config["hierarchical"]["hstar"],
            "smoothing": float(config["attributes"]["smoothing"]),
            "bootstrap": config["bootstrap"],
            "placebo_seed": "default_rng([seed, position in health + taste attributes, rep])",
            "new_item": "first purchase of the item in the user's history; halves as in H*",
            "new_min_n_c_each_half": {"primary": NEW_MIN_N_C, "secondary": NEW_MIN_N_C_SECONDARY},
            "new_pi_smoothing": "same as M1 (lambda = smoothing, prior = pooled new-item share)",
            "s1_rule": f"keep aisles with non-attribute items >= {S1_RATIO} x attribute items",
            "s2_rule": "drop the most purchased attribute (or placebo) item from the set and from C_a",
            "bonferroni_level": BONFERRONI_LEVEL,
        },
        **analyze(config, items, history, reference),
    }
    health = list(config["hierarchical"]["attributes"][DATASET])
    taste = list(config["hierarchical"]["hstar"]["taste_attributes"])
    report["figure"] = _figure(report, health, taste, Path(config["report_dir"]) / "figures" / f"{config['profile']}_hstar_delta.png")
    report["runtime_seconds"] = time.perf_counter() - started
    _write_json(Path(config["report_dir"]) / "metrics" / f"{config['profile']}_hstar_followup.json", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Post-hoc H* follow-up: placebo-adjusted, concentration and new-item stability")
    parser.add_argument("--profile", choices=["sample", "medium", "full"], default="sample")
    parser.add_argument("--reference", default=None, help="H* report (JSON) whose M1 and placebo values must be reproduced")
    args = parser.parse_args()
    report = analyze_hstar_followup(load_config(args.profile), args.reference)
    summary = {key: report[key] for key in ["reproduction", "bootstrap", "interpretation", "runtime_seconds", "figure"]}
    print(json.dumps(summary, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
