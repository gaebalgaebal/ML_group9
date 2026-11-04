"""Post-hoc robustness checks of the placebo-adjusted H* stability (hstar_followup).

On the Instacart eval split history, for every health and taste attribute:
- R1: M1 and M1_new with unsmoothed pi = n_a / n_C (users with n_a = 0 in both halves are then tied),
- R2: buyer decomposition: persistence of buying in C_a across halves (odds ratio) and the stability of
  pi among users who bought in both halves (M1_within),
- R3: bootstrap CIs for the aisle-filtered sensitivity S1 of hstar_followup,
- R4: M1 and Delta with items carrying both a health and a taste attribute removed from the analysis.

The bootstrap draws one user weight vector per resample shared by every attribute, placebo and check.
Nothing here changes the pre-registered H* verdicts or the hstar_followup results.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from .attributes import attribute_column
from .config import load_config
from .data import load_split
from .hierarchical import attribute_categories
from .hstar import DATASET, Purchases, Stability, _summary, counts, percentile_ci, placebo_flag, purchase_data, stability, taste_attributes
from .hstar_followup import (
    NEW_MIN_N_C,
    REPRODUCTION_TOLERANCE,
    RankedHalves,
    _minus,
    _placebo_column,
    delta,
    item_purchases,
    mean_or_none,
    new_item_data,
    new_item_halves,
    s1_aisles,
)
from .pipeline import _write_json


# Aisle label given to removed items so that they fall outside every C_a and every placebo pool.
EXCLUDED_AISLE = "__excluded__"
TOP_OVERLAP = 10
# Families of (attribute, placebos) stability inputs that get a bootstrap Delta.
FAMILIES = ["t1", "t1_new", "raw", "raw_new", "within", "s1", "r4"]


# ---------------------------------------------------------------------------------------------
# Building blocks


def raw_halves(data: Purchases, flag: np.ndarray, in_c: np.ndarray, min_n_c: int) -> Stability:
    """Users with n_C >= min_n_c in both halves and unsmoothed pi = n_a / n_C per half."""
    return stability(data, flag, in_c, 0.0, min_n_c)


def buyer_flags(data: Purchases, flag: np.ndarray, in_c: np.ndarray, rows: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """B_h = 1[n_a,h >= 1] for the given user rows in the first and second half."""
    first, _ = counts(data.first, flag, in_c)
    second, _ = counts(data.second, flag, in_c)
    return first[rows] >= 1, second[rows] >= 1


def cell_counts(b1: np.ndarray, b2: np.ndarray, weights: np.ndarray | None = None) -> np.ndarray:
    """Weighted 2 x 2 table [[n00, n01], [n10, n11]] with rows B_1 and columns B_2."""
    weights = np.ones(len(b1)) if weights is None else weights
    code = 2 * b1.astype(np.int64) + b2.astype(np.int64)
    return np.bincount(code, weights=weights, minlength=4).reshape(2, 2)


def log_odds_ratio(table: np.ndarray) -> float | None:
    """log[odds(B_2 | B_1 = 1) / odds(B_2 | B_1 = 0)]; None when a cell is empty."""
    if (table <= 0).any():
        return None
    return float(np.log(table[1, 1]) - np.log(table[1, 0]) - np.log(table[0, 1]) + np.log(table[0, 0]))


def persistence(b1: np.ndarray, b2: np.ndarray, weights: np.ndarray | None = None) -> dict[str, float | None]:
    table = cell_counts(b1, b2, weights)
    buyers, others = table[1].sum(), table[0].sum()
    log_or = log_odds_ratio(table)
    return {
        "users": float(table.sum()),
        "first_half_buyers": float(buyers),
        "p_b2_given_b1": float(table[1, 1] / buyers) if buyers else None,
        "p_b2_given_not_b1": float(table[0, 1] / others) if others else None,
        "odds_ratio": float(np.exp(log_or)) if log_or is not None else None,
        "log_odds_ratio": log_or,
    }


def within_halves(raw: Stability, b1: np.ndarray, b2: np.ndarray) -> Stability:
    """The raw halves restricted to users who bought the attribute in C_a in both halves."""
    keep = b1 & b2
    return Stability(raw.rows[keep], raw.first[keep], raw.second[keep])


def overlap_items(flags: dict[str, np.ndarray], health: list[str], taste: list[str]) -> np.ndarray:
    """Items carrying at least one health attribute and at least one taste attribute."""
    return np.logical_or.reduce([flags[name] for name in health]) & np.logical_or.reduce([flags[name] for name in taste])


def without_items(data: Purchases, removed: np.ndarray) -> Purchases:
    """Purchases in which removed items belong to no aisle: they leave every C_a (numerator and denominator),
    every placebo pool and every popularity decile. Baskets and halves are unchanged."""
    aisle = data.item_aisle.astype(object).copy()
    aisle[removed] = EXCLUDED_AISLE
    return dataclasses.replace(data, item_aisle=aisle.astype(str))


def placebo_sets(data: Purchases, flag: np.ndarray, aisles: set[str], seed: int, column: int, reps: int) -> list[np.ndarray]:
    return [placebo_flag(data, flag, aisles, np.random.default_rng([seed, column, rep]))[0] for rep in range(reps)]


# ---------------------------------------------------------------------------------------------
# Bootstrap


@dataclasses.dataclass
class Family:
    """An attribute's stability inputs and its placebos' for one check."""

    attribute: Stability
    placebos: list[Stability]

    def rho(self) -> float | None:
        return self.attribute.rho()

    def placebo_rho(self) -> list[float | None]:
        return [s.rho() for s in self.placebos]


def bootstrap(
    n_users: int,
    families: dict[str, dict[str, Family]],
    buyers: dict[str, tuple[tuple[np.ndarray, np.ndarray, np.ndarray], list[tuple[np.ndarray, np.ndarray, np.ndarray]]]],
    pairs: list[tuple[str, str]],
    resamples: int,
    seed: int,
) -> dict[str, object]:
    """Paired user bootstrap. families[f][name] gives Delta draws; buyers[name] = ((rows, b1, b2) of the
    attribute, [(rows, b1, b2) per placebo]) gives draws of log OR(attribute) - mean log OR(placebos)."""
    ranked = {
        family: {name: (RankedHalves(f.attribute), [RankedHalves(s) for s in f.placebos]) for name, f in rows.items()}
        for family, rows in families.items()
    }
    draws = {family: {name: [] for name in rows} for family, rows in families.items()}
    differences = {family: {pair: [] for pair in pairs} for family in families}
    log_or = {name: [] for name in buyers}
    rng = np.random.default_rng(seed)
    for _ in range(resamples):
        weights = np.bincount(rng.integers(0, n_users, n_users), minlength=n_users).astype(np.float64)
        for family, rows in ranked.items():
            values = {name: delta(attribute.rho(weights), [p.rho(weights) for p in placebos]) for name, (attribute, placebos) in rows.items()}
            for name, value in values.items():
                if value is not None:
                    draws[family][name].append(value)
            for a, b in pairs:
                if a in values and b in values and values[a] is not None and values[b] is not None:
                    differences[family][(a, b)].append(values[a] - values[b])
        for name, ((rows_a, b1, b2), placebos) in buyers.items():
            value = log_odds_ratio(cell_counts(b1, b2, weights[rows_a]))
            placebo = mean_or_none([log_odds_ratio(cell_counts(p1, p2, weights[rows_p])) for rows_p, p1, p2 in placebos])
            if value is not None and placebo is not None:
                log_or[name].append(value - placebo)
    return {"delta": draws, "differences": differences, "log_or_difference": log_or, "resamples": resamples}


# ---------------------------------------------------------------------------------------------
# Analysis


def _ci(value: float | None, values: list[float]) -> dict[str, object]:
    low, high = percentile_ci(values)
    return {"value": value, "ci_low": low, "ci_high": high, "valid_resamples": len(values)}


def _family_report(family: Family, draws: list[float]) -> dict[str, object]:
    rho, placebo = family.rho(), family.placebo_rho()
    return {
        "users": int(len(family.attribute.rows)),
        "placebo_users_mean": float(np.mean([len(s.rows) for s in family.placebos])) if family.placebos else None,
        "m1": rho,
        "placebo_m1": _summary(placebo),
        "delta": _ci(delta(rho, placebo), draws),
    }


def _pair_report(families: dict[str, Family], draws: dict[tuple[str, str], list[float]], pairs: list[tuple[str, str]]) -> dict[str, object]:
    result = {}
    for a, b in pairs:
        value = _minus(delta(families[a].rho(), families[a].placebo_rho()), delta(families[b].rho(), families[b].placebo_rho()))
        result[f"{a}-{b}"] = _ci(value, draws[(a, b)])
    return result


def analyze(config: dict[str, object], items: pd.DataFrame, history: pd.DataFrame, reference: dict[str, object] | None = None) -> dict[str, object]:
    settings = config["hierarchical"]["hstar"]
    health = list(config["hierarchical"]["attributes"][DATASET])
    taste = list(settings["taste_attributes"])
    names = health + taste
    pairs = [(a, b) for a in health for b in taste]
    smoothing = float(config["attributes"]["smoothing"])
    min_half = int(settings["min_n_c_each_half"])
    reps, seed = int(settings["placebo_reps"]), int(settings["seed"])
    resamples, boot_seed = int(config["bootstrap"]["n"]), int(config["bootstrap"]["seed"])

    items = taste_attributes(items, taste)
    categories = attribute_categories(config, DATASET, items, names)
    # History only: an empty target leaves the history matrices of purchase_data unchanged.
    data = purchase_data(items, history, history.iloc[:0])
    table = items.drop_duplicates("item_id").assign(item_id=lambda f: f["item_id"].astype(str)).set_index("item_id")
    product_names = table["product_name"].reindex(data.items).fillna("").astype(str).to_numpy()
    flags = {name: table[attribute_column(name)].reindex(data.items).fillna(0).to_numpy().astype(bool) for name in names}
    in_c = {name: np.isin(data.item_aisle, sorted(categories[name])) for name in names}
    new_data = new_item_data(data, *new_item_halves(items, history, data))
    fakes = {name: placebo_sets(data, flags[name], categories[name], seed, _placebo_column(name, health, taste), reps) for name in names}

    families: dict[str, dict[str, Family]] = {family: {} for family in FAMILIES}
    buyers, persistence_rows = {}, {}
    for name in names:
        fake = fakes[name]
        families["t1"][name] = Family(
            stability(data, flags[name], in_c[name], smoothing, min_half), [stability(data, f, in_c[name], smoothing, min_half) for f in fake]
        )
        families["t1_new"][name] = Family(
            stability(new_data, flags[name], in_c[name], smoothing, NEW_MIN_N_C),
            [stability(new_data, f, in_c[name], smoothing, NEW_MIN_N_C) for f in fake],
        )
        raw = Family(raw_halves(data, flags[name], in_c[name], min_half), [raw_halves(data, f, in_c[name], min_half) for f in fake])
        families["raw"][name] = raw
        families["raw_new"][name] = Family(
            raw_halves(new_data, flags[name], in_c[name], NEW_MIN_N_C), [raw_halves(new_data, f, in_c[name], NEW_MIN_N_C) for f in fake]
        )
        b1, b2 = buyer_flags(data, flags[name], in_c[name], raw.attribute.rows)
        placebo_buyers = [(s.rows, *buyer_flags(data, f, in_c[name], s.rows)) for f, s in zip(fake, raw.placebos)]
        buyers[name] = ((raw.attribute.rows, b1, b2), placebo_buyers)
        families["within"][name] = Family(within_halves(raw.attribute, b1, b2), [within_halves(s, p1, p2) for s, (_, p1, p2) in zip(raw.placebos, placebo_buyers)])
        persistence_rows[name] = (persistence(b1, b2), [persistence(p1, p2) for _, p1, p2 in placebo_buyers])

        kept = s1_aisles(data, flags[name], categories[name])
        kept_in_c = np.isin(data.item_aisle, sorted(kept))
        families["s1"][name] = Family(
            stability(data, flags[name], kept_in_c, smoothing, min_half),
            [stability(data, f, kept_in_c, smoothing, min_half) for f in placebo_sets(data, flags[name], kept, seed, _placebo_column(name, health, taste), reps)],
        )

    reproduction = _reproduce(reference, names, families)
    if reproduction["checked"] and not reproduction["passed"]:
        raise RuntimeError(f"Values differ from the hstar_followup report: {json.dumps(reproduction, default=str)}")

    # R4: items with both a health and a taste attribute leave the analysis; C_a, placebo pools and
    # popularity deciles are recomputed without them and the placebos are drawn again with the same seeds.
    overlap = overlap_items(flags, health, taste)
    reduced_items = items[~items["item_id"].astype(str).isin(set(data.items[overlap]))]
    reduced_categories = attribute_categories(config, DATASET, reduced_items, names)
    reduced = without_items(data, overlap)
    for name in names:
        flag = flags[name] & ~overlap
        reduced_in_c = np.isin(reduced.item_aisle, sorted(reduced_categories[name]))
        families["r4"][name] = Family(
            stability(reduced, flag, reduced_in_c, smoothing, min_half),
            [
                stability(reduced, f, reduced_in_c, smoothing, min_half)
                for f in placebo_sets(reduced, flag, reduced_categories[name], seed, _placebo_column(name, health, taste), reps)
            ],
        )

    boot = bootstrap(len(data.users), families, buyers, pairs, resamples, boot_seed)
    if reference is not None and resamples == int(reference["bootstrap"]["used"]):
        reproduction["delta_ci"] = _reproduce_ci(reference, names, families, boot)

    report: dict[str, object] = {"reproduction": reproduction}
    for family, key in [("raw", "r1_raw"), ("raw_new", "r1_raw_new"), ("s1", "r3_s1"), ("r4", "r4_overlap_removed")]:
        report[key] = {
            "attributes": {
                name: {"group": "health" if name in health else "taste", **_family_report(families[family][name], boot["delta"][family][name])}
                for name in names
            },
            "differences": _pair_report(families[family], boot["differences"][family], pairs),
        }
    for name in names:
        report["r1_raw"]["attributes"][name]["ties"] = {
            "attribute_both_halves_zero_share": _zero_share(families["raw"][name].attribute),
            "placebo_both_halves_zero_share_mean": mean_or_none([_zero_share(s) for s in families["raw"][name].placebos]),
        }
        report["r3_s1"]["attributes"][name]["aisles_dropped"] = sorted(categories[name] - s1_aisles(data, flags[name], categories[name]))

    r2 = {}
    for name in names:
        attribute, placebos = persistence_rows[name]
        log_low, log_high = percentile_ci(boot["log_or_difference"][name])
        log_placebo = mean_or_none([p["log_odds_ratio"] for p in placebos])
        r2[name] = {
            "persistence": {
                **attribute,
                "placebo_p_b2_given_b1_mean": mean_or_none([p["p_b2_given_b1"] for p in placebos]),
                "placebo_p_b2_given_not_b1_mean": mean_or_none([p["p_b2_given_not_b1"] for p in placebos]),
                "placebo_odds_ratio": _summary([p["odds_ratio"] for p in placebos]),
                "placebo_log_odds_ratio": _summary([p["log_odds_ratio"] for p in placebos]),
                "log_odds_ratio_minus_placebo": {
                    "value": _minus(attribute["log_odds_ratio"], log_placebo),
                    "ci_low": log_low,
                    "ci_high": log_high,
                    "valid_resamples": len(boot["log_or_difference"][name]),
                },
            },
            "within": _family_report(families["within"][name], boot["delta"]["within"][name]),
        }
    report["r2_buyers"] = r2

    purchases = item_purchases(data)
    order = np.flatnonzero(overlap)[np.argsort(-purchases[overlap], kind="mergesort")]
    report["r4_overlap_removed"]["overlap"] = {
        "items": int(overlap.sum()),
        "items_with_history_purchases": int((overlap & (purchases > 0)).sum()),
        "history_purchase_share": float(purchases[overlap].sum() / purchases.sum()) if purchases.sum() else None,
        "share_of_attribute_purchases_in_c_a": {
            name: float(purchases[overlap & flags[name] & in_c[name]].sum() / purchases[flags[name] & in_c[name]].sum())
            if purchases[flags[name] & in_c[name]].sum()
            else None
            for name in names
        },
        "top_items": [
            {
                "item_id": data.items[i],
                "product_name": product_names[i],
                "history_purchases": int(purchases[i]),
                "attributes": [name for name in names if flags[name][i]],
            }
            for i in order[:TOP_OVERLAP]
        ],
        "c_a_changes": {
            name: {"removed": sorted(categories[name] - reduced_categories[name]), "added": sorted(reduced_categories[name] - categories[name])}
            for name in names
        },
    }
    report["interpretation"] = _interpretation(health, taste, reference, report)
    report["bootstrap"] = {"resamples": boot["resamples"], "seed": boot_seed}
    report["history_users"] = int(len(data.users))
    return report


def _zero_share(halves: Stability) -> float | None:
    """Share of eligible users with raw pi = 0 in both halves (tied at zero)."""
    if len(halves.rows) == 0:
        return None
    return float(((halves.first == 0) & (halves.second == 0)).mean())


def _all_above_zero(differences: dict[str, object], a: str, taste: list[str]) -> bool:
    return all(differences[f"{a}-{b}"]["ci_low"] is not None and differences[f"{a}-{b}"]["ci_low"] > 0 for b in taste)


def _interpretation(health: list[str], taste: list[str], reference: dict[str, object] | None, report: dict[str, object]) -> dict[str, object]:
    """Labels from the post-hoc rules fixed before the run."""
    result: dict[str, object] = {}
    analyses = {
        "t1_delta": reference["delta_differences"] if reference is not None else None,
        "r1_delta_raw": report["r1_raw"]["differences"],
        "r4_delta": report["r4_overlap_removed"]["differences"],
    }
    for a in health:
        above = {key: (_all_above_zero(rows, a, taste) if rows is not None else None) for key, rows in analyses.items()}
        if all(above.values()):
            label = "robust: higher than all three taste attributes after placebo adjustment in T1, R1 and R4"
        else:
            label = "sensitive to the analysis choice: " + ", ".join(key for key, value in above.items() if not value)
        result[a] = {"taste_comparison": label, "all_pairs_above_zero": above}
    for name in health + taste:
        persistence_low = report["r2_buyers"][name]["persistence"]["log_odds_ratio_minus_placebo"]["ci_low"]
        within = report["r2_buyers"][name]["within"]["delta"]
        persists = persistence_low is not None and persistence_low > 0
        within_above = within["ci_low"] is not None and within["ci_low"] > 0
        within_includes_zero = within["ci_low"] is not None and within["ci_high"] is not None and within["ci_low"] <= 0 <= within["ci_high"]
        if persists and within_includes_zero:
            label = "stability comes mainly from the persistence of the buyer / non-buyer split"
        elif persists and within_above:
            label = "both the persistence of the buyer split and within-buyer share stability"
        else:
            label = "neither R2 rule applies"
        result.setdefault(name, {})["r2"] = {
            "label": label,
            "persistence_ci_above_zero": persists,
            "within_ci_above_zero": within_above,
        }
    result["note"] = (
        "Post-hoc robustness checks; the pre-registered H* verdicts and the hstar_followup results are unchanged. "
        "R3 for fat_reduced drops the milk and yogurt aisles, so its population differs. Observational, not causal."
    )
    return result


def _reproduce(reference: dict[str, object] | None, names: list[str], families: dict[str, dict[str, Family]]) -> dict[str, object]:
    if reference is None:
        return {"checked": False}
    rows = {}

    def check(key: str, expected: float | None, value: float | None) -> None:
        diff = abs(expected - value) if expected is not None and value is not None else (0.0 if expected is value else float("inf"))
        rows[key] = {"reference": expected, "recomputed": value, "abs_diff": diff}

    for name in names:
        t1, t1_new, s1 = families["t1"][name], families["t1_new"][name], families["s1"][name]
        row = reference["attributes"][name]
        new = row["new_items"]["min_n_c_new_5"]
        check(f"{name}.m1", row["m1"], t1.rho())
        check(f"{name}.placebo_mean", row["placebo_m1"]["mean"], _summary(t1.placebo_rho())["mean"])
        check(f"{name}.delta", row["delta"]["value"], delta(t1.rho(), t1.placebo_rho()))
        check(f"{name}.m1_new", new["m1_new"], t1_new.rho())
        check(f"{name}.delta_new", new["delta_new"]["value"], delta(t1_new.rho(), t1_new.placebo_rho()))
        check(f"{name}.s1_delta", reference["s1"][name]["delta"], delta(s1.rho(), s1.placebo_rho()))
    worst = max(row["abs_diff"] for row in rows.values())
    return {"checked": True, "tolerance": REPRODUCTION_TOLERANCE, "max_abs_diff": worst, "passed": worst <= REPRODUCTION_TOLERANCE, "values": rows}


def _reproduce_ci(reference: dict[str, object], names: list[str], families: dict[str, dict[str, Family]], boot: dict[str, object]) -> dict[str, object]:
    worst = 0.0
    for name in names:
        row = reference["attributes"][name]
        for family, expected in [("t1", row["delta"]), ("t1_new", row["new_items"]["min_n_c_new_5"]["delta_new"])]:
            low, high = percentile_ci(boot["delta"][family][name])
            worst = max(worst, abs(expected["ci_low"] - low), abs(expected["ci_high"] - high))
    return {"max_abs_diff": worst, "passed": worst <= REPRODUCTION_TOLERANCE}


def analyze_hstar_robust(config: dict[str, object], reference_path: str | None = None) -> dict[str, object]:
    """Run the robustness checks on the Instacart eval split history and write the report."""
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
        "design": "post-hoc robustness checks of placebo-adjusted H* stability; history only",
        "reference": reference_path,
        "settings": {
            **config["hierarchical"]["hstar"],
            "smoothing": float(config["attributes"]["smoothing"]),
            "bootstrap": config["bootstrap"],
            "r1": "pi = n_a / n_C without smoothing; mid-rank Spearman; n_C >= min_n_c_each_half (new items: n_C_new >= 5)",
            "r2": "eligible users of R1; B_h = 1[n_a,h >= 1]; M1_within on users with B_1 = B_2 = 1 (raw pi)",
            "r2_log_or_difference": "log OR(attribute) - mean log OR over placebos with a defined OR",
            "r3": "S1 aisles of hstar_followup (non-attribute items >= 2 x attribute items), smoothed pi",
            "r4": "items with a health and a taste attribute removed from C_a, attribute sets, placebo pools and deciles; halves unchanged",
        },
        **analyze(config, items, history, reference),
    }
    report["runtime_seconds"] = time.perf_counter() - started
    _write_json(Path(config["report_dir"]) / "metrics" / f"{config['profile']}_hstar_robust.json", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Post-hoc robustness checks of placebo-adjusted H* stability")
    parser.add_argument("--profile", choices=["sample", "medium", "full"], default="sample")
    parser.add_argument("--reference", default=None, help="hstar_followup report (JSON) whose values must be reproduced")
    args = parser.parse_args()
    report = analyze_hstar_robust(load_config(args.profile), args.reference)
    summary = {key: report[key] for key in ["reproduction", "interpretation", "bootstrap", "runtime_seconds"]}
    print(json.dumps(summary, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
