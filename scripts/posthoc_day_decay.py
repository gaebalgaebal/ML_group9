"""Post-hoc analysis: day-based recency decay for the repeat scorer (W14; post-hoc, development data, not confirmatory).

Usage:
    python scripts/posthoc_day_decay.py [--profile medium] [--datasets instacart,complete_journey]
                                        [--out-dir reports/posthoc] [--bootstrap-n 1000] [--seed 42]

Metric: repeat Recall@10 on users with >= 1 repeat target item (nbr.metrics.per_user_recall against the
repeat target of a split; mean over those users).
(a) Fit split: day decay (nbr.day_decay) over h in HALF_LIFE_GRID and basket decay (nbr.repeat) over
    gamma in GAMMA_GRID; h* and gamma* maximise fit recall (ties -> larger h / larger gamma).
(b) Eval split, paired bootstrap over users (nbr.metrics.paired_bootstrap, 95 % CI): day(h*) vs gamma = 1;
    day(h*) vs basket(gamma*); basket(gamma*) vs gamma = 1.
(c) Transfer: the h* (gamma*) selected on the other dataset's fit split applied to this dataset's eval
    split; transfer loss = own-selection recall - transferred recall, with a paired bootstrap CI.
(d) Unified scale: basket half-life ln(0.5)/ln(gamma) and its day equivalent using the mean and the
    median inter-basket gap of the fit histories, next to h*.
(e) History-length bins (distinct fit-history baskets): fit recall by h and by gamma.

Outputs in --out-dir: day_decay_<profile>.json (all numbers; meta with the code commit and the sha256 of
every input file), day_decay_<profile>.md (tables), fig_recall_vs_halflife.png + .csv (fit recall vs
half-life in days on a log axis; day decay solid, basket decay converted with the median gap dashed;
one panel per dataset). Only the development profiles ``sample`` and ``medium`` are accepted; no file
under reports/final* or data/processed/holdout* is read.
"""
from __future__ import annotations

import argparse
import hashlib
import math
import platform
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from nbr.baselines import item_popularity  # noqa: E402
from nbr.config import load_config  # noqa: E402
from nbr.data import dataset_dir, load_split, split_target  # noqa: E402
from nbr.day_decay import (  # noqa: E402
    HALF_LIFE_GRID,
    basket_days_path,
    basket_half_life,
    day_decay_list,
    history_basket_days,
    inter_basket_gaps,
    load_basket_days,
)
from nbr.io import write_json  # noqa: E402
from nbr.metrics import paired_bootstrap, per_user_recall  # noqa: E402
from nbr.repeat import repeat_list  # noqa: E402
from nbr.usercf_eval import peak_memory_mib  # noqa: E402

LABEL = "post-hoc, development data, not confirmatory"
K = 10
GAMMA_GRID = [0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.97, 0.98, 0.99, 0.995, 1.0]
HISTORY_BIN_EDGES = [4, 10, 20, 50, 100]
ALLOWED_PROFILES = ("sample", "medium")
DATASETS = ("instacart", "complete_journey")
NAMES = {"instacart": "Instacart", "complete_journey": "Complete Journey"}
INPUT_FILES = ("fit_history", "fit_target", "eval_history", "eval_target")
# Figure colours: categorical slots 1 and 2 of the validated default palette (light surface); neutral ink for text.
COLORS = {"instacart": "#2a78d6", "complete_journey": "#eb6834"}
INK = {"primary": "#1f1f1e", "secondary": "#5c5b57", "grid": "#e4e3df", "surface": "#fcfcfb"}
# Numbers to reproduce (computed earlier on the same development data, four decimals).
SANITY = {("medium", "complete_journey"): {"fit_basket_0.97": 0.2890, "fit_basket_1.0": 0.2766,
                                           "eval_basket_0.97_minus_1.0": 0.0220}}


# ---------------------------------------------------------------------------
# arguments and small helpers
# ---------------------------------------------------------------------------
def parse_datasets(value: str) -> list[str]:
    """Comma-separated dataset names -> list in the given order (blanks and repeats dropped); unknown names raise."""
    names: list[str] = []
    for item in value.split(","):
        item = item.strip()
        if item and item not in names:
            names.append(item)
    unknown = [name for name in names if name not in DATASETS]
    if unknown or not names:
        raise ValueError(f"datasets must be a non-empty subset of {list(DATASETS)}, got {value!r}")
    return names


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Command line of the script (``argv`` defaults to sys.argv); ``datasets`` is returned as a list."""
    parser = argparse.ArgumentParser(description="Post-hoc day-based recency decay analysis (development profiles only).")
    parser.add_argument("--profile", default="medium", choices=ALLOWED_PROFILES)
    parser.add_argument("--datasets", default=",".join(DATASETS), type=parse_datasets,
                        help="comma-separated subset of instacart,complete_journey (default: both)")
    parser.add_argument("--out-dir", default=str(ROOT / "reports" / "posthoc"))
    parser.add_argument("--bootstrap-n", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args(argv)


def jsonable(value: float) -> float | str:
    """inf -> "inf" (strict JSON has no infinity); other values unchanged."""
    return "inf" if isinstance(value, float) and math.isinf(value) else value


def fmt_param(value: float) -> str:
    return "inf" if isinstance(value, float) and math.isinf(value) else f"{value:g}"


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def select_param(scores: dict[float, float]) -> float:
    """argmax of ``scores`` (parameter -> recall); ties (within 1e-12) -> the larger parameter (inf allowed)."""
    best = max(scores.values())
    tied = [param for param, score in scores.items() if math.isclose(score, best, rel_tol=1e-12, abs_tol=1e-12)]
    return max(tied)


def at_edge(value: float, grid: list[float]) -> bool:
    return value == min(grid) or value == max(grid)


def mean_recall(series: dict[float, pd.Series]) -> dict[float, float]:
    return {param: float(values.mean()) for param, values in series.items()}


def compare(a: pd.Series, b: pd.Series, n: int, seed: int) -> dict[str, Any]:
    """Paired bootstrap of mean(b) - mean(a) over users (95 % CI)."""
    result = paired_bootstrap(a, b, n, seed, [0.95])
    return {"n_users": result["n_users"], "mean_a": result["mean_a"], "mean_b": result["mean_b"],
            "diff": result["diff"], "ci_0.95": result["ci"][0.95]}


def history_lengths(history: pd.DataFrame) -> pd.Series:
    """Number of distinct baskets per user of ``history`` (user_id -> count)."""
    return history[["user_id", "basket_id"]].drop_duplicates().groupby("user_id").size()


def bin_labels(edges: list[int]) -> list[tuple[str, float, float]]:
    bounds = [0, *edges, math.inf]
    return [(f"({lo},inf)" if math.isinf(hi) else f"({lo},{hi}]", lo, hi) for lo, hi in zip(bounds[:-1], bounds[1:])]


def by_history_length(day: dict[float, pd.Series], basket: dict[float, pd.Series], lengths: pd.Series) -> list[dict[str, Any]]:
    """Per history-length bin: user count and the mean per-user recall of every h and every gamma (None when empty)."""
    rows = []
    users_scored = next(iter(day.values())).index
    for name, lo, hi in bin_labels(HISTORY_BIN_EDGES):
        users = lengths[(lengths > lo) & (lengths <= hi)].index
        index = users_scored.intersection(users)
        row: dict[str, Any] = {"bin": name, "n_users": int(len(index))}
        row["day"] = [{"h": jsonable(h), "recall": float(s.loc[index].mean()) if len(index) else None} for h, s in day.items()]
        row["basket"] = [{"gamma": g, "recall": float(s.loc[index].mean()) if len(index) else None} for g, s in basket.items()]
        rows.append(row)
    return rows


def unified_scale(gammas: list[float], mean_gap: float, median_gap: float) -> list[dict[str, Any]]:
    """Per gamma: half-life in baskets and its day equivalent with the mean and with the median inter-basket gap."""
    rows = []
    for gamma in gammas:
        hb = basket_half_life(gamma)
        rows.append({"gamma": gamma, "basket_half_life": jsonable(hb),
                     "day_equivalent_mean_gap": jsonable(hb * mean_gap),
                     "day_equivalent_median_gap": jsonable(hb * median_gap)})
    return rows


def gap_summary(gaps: pd.Series, lengths: pd.Series) -> dict[str, Any]:
    nonzero = gaps[gaps > 0]
    return {
        "definition": "day difference between consecutive baskets of a user (fit history, baskets ordered by day); "
                      "same-day baskets give 0",
        "n_gaps": int(len(gaps)),
        "users_with_one_basket": int((lengths == 1).sum()),
        "mean_days": float(gaps.mean()),
        "median_days": float(gaps.median()),
        "share_zero": float((gaps == 0).mean()),
        "mean_days_excluding_zero": float(nonzero.mean()) if len(nonzero) else None,
        "median_days_excluding_zero": float(nonzero.median()) if len(nonzero) else None,
        "p90_days": float(gaps.quantile(0.9)),
    }


# ---------------------------------------------------------------------------
# one dataset
# ---------------------------------------------------------------------------
def analyse_dataset(config: dict[str, Any], dataset: str, n_boot: int, seed: int) -> tuple[dict[str, Any], dict[str, dict[float, pd.Series]]]:
    """Steps (a), (b), (d), (e) for ``dataset``; returns the report entry and the per-user eval series (for (c))."""
    timings: dict[str, float] = {}
    t0 = time.perf_counter()
    fit_history, fit_target = load_split(config, dataset, "fit")
    eval_history, eval_target = load_split(config, dataset, "eval")
    days = load_basket_days(config, dataset)
    fit_repeat, _ = split_target(fit_history, fit_target)
    eval_repeat, _ = split_target(eval_history, eval_target)
    fit_pop, eval_pop = item_popularity(fit_history), item_popularity(eval_history)
    fit_baskets = history_basket_days(fit_history, days)      # raises when a history basket has no day
    eval_baskets = history_basket_days(eval_history, days)
    timings["load_and_check"] = time.perf_counter() - t0

    def day_recall(history: pd.DataFrame, repeat_target: pd.DataFrame, pop: pd.Series, h: float) -> pd.Series:
        return per_user_recall(day_decay_list(history, days, h, K, pop), repeat_target, K)

    def basket_recall(history: pd.DataFrame, repeat_target: pd.DataFrame, pop: pd.Series, gamma: float) -> pd.Series:
        return per_user_recall(repeat_list(history, gamma, K, pop), repeat_target, K)

    t0 = time.perf_counter()
    fit_day = {h: day_recall(fit_history, fit_repeat, fit_pop, h) for h in HALF_LIFE_GRID}
    fit_basket = {g: basket_recall(fit_history, fit_repeat, fit_pop, g) for g in GAMMA_GRID}
    fit_day_mean, fit_basket_mean = mean_recall(fit_day), mean_recall(fit_basket)
    h_star, gamma_star = select_param(fit_day_mean), select_param(fit_basket_mean)
    timings["fit_grids"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    eval_day = {h: day_recall(eval_history, eval_repeat, eval_pop, h) for h in HALF_LIFE_GRID}
    eval_basket = {g: basket_recall(eval_history, eval_repeat, eval_pop, g) for g in GAMMA_GRID}
    timings["eval_grids"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    comparisons = {
        "day_hstar_vs_frequency": {"comparator": "basket decay gamma = 1 (personal frequency)",
                                   "model": f"day decay h* = {fmt_param(h_star)} days",
                                   **compare(eval_basket[1.0], eval_day[h_star], n_boot, seed)},
        "day_hstar_vs_basket_gstar": {"comparator": f"basket decay gamma* = {gamma_star:g}",
                                      "model": f"day decay h* = {fmt_param(h_star)} days",
                                      **compare(eval_basket[gamma_star], eval_day[h_star], n_boot, seed)},
        "basket_gstar_vs_frequency": {"comparator": "basket decay gamma = 1 (personal frequency)",
                                      "model": f"basket decay gamma* = {gamma_star:g}",
                                      **compare(eval_basket[1.0], eval_basket[gamma_star], n_boot, seed)},
        "basket_0.97_vs_frequency": {"comparator": "basket decay gamma = 1 (personal frequency)",
                                     "model": "basket decay gamma = 0.97 (reproduction check)",
                                     **compare(eval_basket[1.0], eval_basket[0.97], n_boot, seed)},
    }
    timings["bootstrap"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    lengths = history_lengths(fit_history)
    gaps = inter_basket_gaps(fit_history, days)
    gap_stats = gap_summary(gaps, lengths)
    mean_gap, median_gap = gap_stats["mean_days"], gap_stats["median_days"]
    timings["gaps_and_bins"] = time.perf_counter() - t0

    entry = {
        "label": LABEL,
        "n_users": {"fit_all": int(fit_target["user_id"].nunique()), "fit_repeat_target": int(fit_repeat["user_id"].nunique()),
                    "eval_all": int(eval_target["user_id"].nunique()), "eval_repeat_target": int(eval_repeat["user_id"].nunique())},
        "n_history_baskets": {"fit": int(len(fit_baskets)), "eval": int(len(eval_baskets))},
        "basket_days": {"fit_day_min": int(fit_baskets["day"].min()), "fit_day_max": int(fit_baskets["day"].max()),
                        "eval_day_min": int(eval_baskets["day"].min()), "eval_day_max": int(eval_baskets["day"].max())},
        "fit": {
            "day": [{"h": jsonable(h), "recall": r} for h, r in fit_day_mean.items()],
            "basket": [{"gamma": g, "basket_half_life": jsonable(basket_half_life(g)), "recall": r} for g, r in fit_basket_mean.items()],
            "h_star": jsonable(h_star),
            "gamma_star": gamma_star,
            "at_grid_edge": {"h": at_edge(h_star, HALF_LIFE_GRID), "gamma": at_edge(gamma_star, GAMMA_GRID)},
            "selection_rule": "max fit repeat Recall@10; ties -> larger h / larger gamma",
        },
        "eval": {
            "day": [{"h": jsonable(h), "recall": r} for h, r in mean_recall(eval_day).items()],
            "basket": [{"gamma": g, "recall": r} for g, r in mean_recall(eval_basket).items()],
            "comparisons": comparisons,
        },
        "gaps": gap_stats,
        "unified_scale": unified_scale(GAMMA_GRID, mean_gap, median_gap),
        "h_star_days": jsonable(h_star),
        "h_star_in_baskets": {"via_mean_gap": jsonable(h_star / mean_gap), "via_median_gap": jsonable(h_star / median_gap)},
        "by_history_length": by_history_length(fit_day, fit_basket, lengths),
        "timing_seconds": timings,
        "peak_memory_mib": peak_memory_mib(),
    }
    return entry, {"day": eval_day, "basket": eval_basket}


# ---------------------------------------------------------------------------
# transfer (c) and the reproduction check
# ---------------------------------------------------------------------------
def transfer(entries: dict[str, dict[str, Any]], series: dict[str, dict[str, dict[float, pd.Series]]],
             n_boot: int, seed: int) -> dict[str, Any]:
    """Apply the h* / gamma* of the other dataset to each dataset's eval split; loss = own - transferred (paired bootstrap)."""
    out: dict[str, Any] = {}
    names = list(entries)
    for dataset in names:
        others = [name for name in names if name != dataset]
        if not others:
            continue
        source = others[0]
        for kind, key, grid in (("day", "h_star", HALF_LIFE_GRID), ("basket", "gamma_star", GAMMA_GRID)):
            own_param = _param(entries[dataset]["fit"][key])
            source_param = _param(entries[source]["fit"][key])
            own, transferred = series[dataset][kind][own_param], series[dataset][kind][source_param]
            result = compare(transferred, own, n_boot, seed)   # diff = mean(own) - mean(transferred) = transfer loss
            out[f"{dataset}_{kind}"] = {
                "dataset": dataset, "scorer": kind, "selected_on": source,
                "own_param": jsonable(own_param), "transferred_param": jsonable(source_param),
                "own_recall": result["mean_b"], "transferred_recall": result["mean_a"],
                "transfer_loss": result["diff"], "ci_0.95": result["ci_0.95"], "n_users": result["n_users"],
            }
    return out


def _param(value: float | str) -> float:
    return math.inf if value == "inf" else float(value)


def reproduction_check(profile: str, entries: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Compare the fit/eval numbers of SANITY (same development data) with the observed ones at four decimals."""
    checks: dict[str, Any] = {}
    for (prof, dataset), expected in SANITY.items():
        if prof != profile or dataset not in entries:
            continue
        entry = entries[dataset]
        fit_basket = {row["gamma"]: row["recall"] for row in entry["fit"]["basket"]}
        observed = {"fit_basket_0.97": fit_basket[0.97], "fit_basket_1.0": fit_basket[1.0],
                    "eval_basket_0.97_minus_1.0": entry["eval"]["comparisons"]["basket_0.97_vs_frequency"]["diff"]}
        checks[dataset] = {key: {"expected": value, "observed": observed[key],
                                 "matched": abs(round(observed[key], 4) - value) < 1e-9} for key, value in expected.items()}
        checks[dataset]["all_matched"] = all(item["matched"] for item in checks[dataset].values() if isinstance(item, dict))
    return checks


# ---------------------------------------------------------------------------
# figure
# ---------------------------------------------------------------------------
def figure_rows(entries: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    """Plotted numbers: fit recall vs half-life in days for the day decay and the median-gap-converted basket decay."""
    rows = []
    for dataset, entry in entries.items():
        median_gap = entry["gaps"]["median_days"]
        for row in entry["fit"]["day"]:
            rows.append({"dataset": dataset, "scorer": "day", "parameter": row["h"], "half_life_days": row["h"],
                         "fit_recall": row["recall"], "selected": row["h"] == entry["fit"]["h_star"]})
        for row in entry["fit"]["basket"]:
            hb = _param(row["basket_half_life"])
            rows.append({"dataset": dataset, "scorer": "basket", "parameter": row["gamma"],
                         "half_life_days": jsonable(hb * median_gap), "fit_recall": row["recall"],
                         "selected": row["gamma"] == entry["fit"]["gamma_star"]})
    return rows


def make_figure(entries: dict[str, dict[str, Any]], out_dir: Path) -> dict[str, str]:
    rows = figure_rows(entries)
    csv_path = out_dir / "fig_recall_vs_halflife.csv"
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.edgecolor": INK["secondary"],
                         "axes.labelcolor": INK["primary"], "xtick.color": INK["secondary"], "ytick.color": INK["secondary"],
                         "text.color": INK["primary"], "figure.facecolor": INK["surface"], "savefig.facecolor": INK["surface"],
                         "axes.facecolor": INK["surface"]})
    n = len(entries)
    fig, axes = plt.subplots(1, n, figsize=(6.2 * n, 4.6), sharex=True, layout="constrained")
    axes = np.atleast_1d(axes)
    for ax, (dataset, entry) in zip(axes, entries.items()):
        color = COLORS[dataset]
        subset = [row for row in rows if row["dataset"] == dataset]
        day = [(row["half_life_days"], row["fit_recall"], row["selected"]) for row in subset
               if row["scorer"] == "day" and row["half_life_days"] != "inf"]
        basket = [(row["half_life_days"], row["fit_recall"], row["selected"]) for row in subset
                  if row["scorer"] == "basket" and row["half_life_days"] != "inf" and row["half_life_days"] > 0]
        frequency = next(row["fit_recall"] for row in subset if row["scorer"] == "day" and row["half_life_days"] == "inf")
        ax.axhline(frequency, color=INK["secondary"], lw=1.2, ls=":", label="personal frequency (h = inf, gamma = 1)")
        ax.plot([x for x, _, _ in day], [y for _, y, _ in day], color=color, lw=2, ls="-", marker="o", ms=6,
                label="day decay, h in days (solid)")
        ax.plot([x for x, _, _ in basket], [y for _, y, _ in basket], color=color, lw=2, ls="--", marker="s", ms=6,
                mfc=INK["surface"], label="basket decay, half-life x median gap (dashed)")
        for x, y, selected in day:
            if selected:
                ax.plot([x], [y], marker="o", ms=13, mfc="none", mec=INK["primary"], mew=1.4, ls="none")
                ax.annotate(f"h* = {x:g} d", (x, y), xytext=(0, 12), textcoords="offset points", ha="center",
                            fontsize=9, color=INK["primary"])
        for x, y, selected in basket:
            if selected:
                gamma = entry["fit"]["gamma_star"]
                ax.plot([x], [y], marker="s", ms=13, mfc="none", mec=INK["primary"], mew=1.4, ls="none")
                ax.annotate(f"gamma* = {gamma:g} (~{x:.0f} d)", (x, y), xytext=(-12, -2), textcoords="offset points",
                            ha="right", va="center", fontsize=9, color=INK["primary"])
        ax.set_xscale("log")
        ax.margins(y=0.18)   # headroom so the h* / gamma* annotations stay clear of the panel title
        ax.set_title(f"{NAMES[dataset]} (median gap {entry['gaps']['median_days']:g} d)", fontsize=11, color=INK["primary"])
        ax.set_xlabel("half-life in days (log scale)")
        ax.set_ylabel("fit repeat Recall@10")
        ax.grid(True, which="major", color=INK["grid"], lw=0.8)
        ax.grid(False, which="minor")
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.legend(frameon=False, fontsize=8.5, loc="best")
    ticks = [1, 3, 7, 14, 30, 60, 90, 180, 365, 730, 1460, 2920]
    xs = [row["half_life_days"] for row in rows if row["half_life_days"] != "inf" and row["half_life_days"] > 0]
    ticks = [t for t in ticks if min(xs) / 1.5 <= t <= max(xs) * 1.5] if xs else []
    for ax in axes:
        if ticks:
            ax.set_xticks(ticks)
            ax.set_xticklabels([str(t) for t in ticks])
        ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    fig.suptitle("Fit repeat Recall@10 vs half-life in days: day decay (solid), basket decay converted with the median "
                 "inter-basket gap (dashed)\ny axes differ per panel; post-hoc, development data, not confirmatory",
                 fontsize=9.5, color=INK["secondary"])
    png_path = out_dir / "fig_recall_vs_halflife.png"
    fig.savefig(png_path, dpi=200)
    plt.close(fig)
    return {"png": png_path.name, "csv": csv_path.name, "points": len(rows)}


# ---------------------------------------------------------------------------
# markdown
# ---------------------------------------------------------------------------
def _f(value: float | None, digits: int = 4, sign: bool = False) -> str:
    if value is None:
        return "-"
    if isinstance(value, str):
        return value
    return f"{value:+.{digits}f}" if sign else f"{value:.{digits}f}"


def _ci(ci: list[float]) -> str:
    return f"[{ci[0]:+.4f}, {ci[1]:+.4f}]"


def _days(value: float | str) -> str:
    return "inf" if value == "inf" else f"{value:.1f}"


def _h(value: float | str) -> str:
    """Grid parameter for tables: "inf" stays, floats print without a trailing .0 (7, 60, 0.97)."""
    return "inf" if value == "inf" else f"{float(value):g}"


def _table(header: list[str], rows: list[list[str]]) -> list[str]:
    lines = ["| " + " | ".join(header) + " |", "| " + " | ".join("---" for _ in header) + " |"]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return lines + [""]


def render_markdown(payload: dict[str, Any]) -> str:
    """Tables of the JSON document ``payload`` (as written by main)."""
    meta, entries, profile = payload["meta"], payload["datasets"], payload["profile"]
    names = list(entries)
    lines = [f"# Day-based recency decay for the repeat scorer ({profile})", "",
             f"**{LABEL}.** Repeat Recall@10 on users with >= 1 repeat target item; day decay score(u, i) = "
             "sum over history baskets containing i of lam ** (d_last(u) - d_b), lam = 2 ** (-1/h), h = half-life in days "
             "(h = inf is personal frequency); basket decay = nbr.repeat with gamma per basket age. Selections are made on "
             "the fit split; the eval split (already used by earlier development runs) carries paired-bootstrap CIs only. "
             "No confirmatory claim is made.", "",
             f"- Code commit `{meta['git_commit']}`, {meta['utc_time']}; profile `{profile}`; bootstrap n = "
             f"{payload['bootstrap']['n']}, seed {payload['bootstrap']['seed']}, {payload['bootstrap']['level']:.0%} CI.",
             f"- Grids: h in days [{', '.join(_h(h) for h in payload['grids']['half_life_days'])}]; gamma {payload['grids']['gamma']}.",
             f"- Inputs (sha256 in the JSON): {', '.join(f'{d}: {len(payload['inputs_sha256'][d])} files' for d in names)}.",
             f"- Runtime {payload['runtime_seconds']:.1f} s; peak working set {payload['peak_memory_mib']:.0f} MiB (in-process).", ""]

    lines += ["## Data and metric", ""]
    lines += _table(["dataset", "fit users (repeat target / all)", "eval users (repeat target / all)", "fit baskets", "eval baskets",
                     "inter-basket gap mean / median (days, fit)", "share of zero gaps"],
                    [[NAMES[d], f"{e['n_users']['fit_repeat_target']:,} / {e['n_users']['fit_all']:,}",
                      f"{e['n_users']['eval_repeat_target']:,} / {e['n_users']['eval_all']:,}",
                      f"{e['n_history_baskets']['fit']:,}", f"{e['n_history_baskets']['eval']:,}",
                      f"{e['gaps']['mean_days']:.2f} / {e['gaps']['median_days']:g}", f"{e['gaps']['share_zero']:.3f}"]
                     for d, e in entries.items()])

    lines += ["## (a) Fit-split grids and selections", "", "Day decay (h in days; * = h*):", ""]
    rows = []
    for i, row in enumerate(entries[names[0]]["fit"]["day"]):
        cells = [_h(row["h"])]
        for d in names:
            r = entries[d]["fit"]["day"][i]
            cells.append(_f(r["recall"]) + (" *" if r["h"] == entries[d]["fit"]["h_star"] else ""))
        rows.append(cells)
    lines += _table(["h (days)"] + [f"{NAMES[d]} fit recall" for d in names], rows)
    lines += ["Basket decay (gamma; half-life in baskets = ln 0.5 / ln gamma; * = gamma*):", ""]
    rows = []
    for i, row in enumerate(entries[names[0]]["fit"]["basket"]):
        cells = [f"{row['gamma']:g}", _days(row["basket_half_life"])]
        for d in names:
            r = entries[d]["fit"]["basket"][i]
            cells.append(_f(r["recall"]) + (" *" if r["gamma"] == entries[d]["fit"]["gamma_star"] else ""))
        rows.append(cells)
    lines += _table(["gamma", "half-life (baskets)"] + [f"{NAMES[d]} fit recall" for d in names], rows)
    lines += _table(["dataset", "h* (days)", "h* at grid edge", "gamma*", "gamma* at grid edge"],
                    [[NAMES[d], _h(e["fit"]["h_star"]), str(e["fit"]["at_grid_edge"]["h"]), f"{e['fit']['gamma_star']:g}",
                      str(e["fit"]["at_grid_edge"]["gamma"])] for d, e in entries.items()])

    lines += ["## (b) Eval-split comparisons (paired bootstrap over users; diff = model - comparator)", ""]
    rows = []
    for d, e in entries.items():
        for key in ("day_hstar_vs_frequency", "day_hstar_vs_basket_gstar", "basket_gstar_vs_frequency"):
            c = e["eval"]["comparisons"][key]
            rows.append([NAMES[d], c["model"], c["comparator"], _f(c["mean_a"]), _f(c["mean_b"]), _f(c["diff"], sign=True),
                         _ci(c["ci_0.95"]), f"{c['n_users']:,}"])
    lines += _table(["dataset", "model", "comparator", "comparator mean", "model mean", "diff", "95 % CI", "users"], rows)
    lines += ["Eval-split recall of every grid point (reference only):", ""]
    rows = [[_h(row["h"])] + [_f(entries[d]["eval"]["day"][i]["recall"]) for d in names]
            for i, row in enumerate(entries[names[0]]["eval"]["day"])]
    lines += _table(["h (days)"] + [f"{NAMES[d]} eval recall" for d in names], rows)
    rows = [[f"{row['gamma']:g}"] + [_f(entries[d]["eval"]["basket"][i]["recall"]) for d in names]
            for i, row in enumerate(entries[names[0]]["eval"]["basket"])]
    lines += _table(["gamma"] + [f"{NAMES[d]} eval recall" for d in names], rows)

    lines += ["## (c) Transfer of the selection between datasets (eval split; loss = own-selection recall - transferred recall)", ""]
    if payload["transfer"]:
        rows = []
        for t in payload["transfer"].values():
            label = "h" if t["scorer"] == "day" else "gamma"
            rows.append([NAMES[t["dataset"]], t["scorer"], f"{label} = {_h(t['own_param'])}", f"{label} = {_h(t['transferred_param'])} (from {NAMES[t['selected_on']]})",
                         _f(t["own_recall"]), _f(t["transferred_recall"]), _f(t["transfer_loss"], sign=True), _ci(t["ci_0.95"]), f"{t['n_users']:,}"])
        lines += _table(["dataset", "scorer", "own selection", "transferred selection", "own recall", "transferred recall",
                         "transfer loss", "95 % CI", "users"], rows)
    else:
        lines += ["(one dataset only: no transfer)", ""]

    lines += ["## (d) Unified scale: basket half-life and its day equivalent", ""]
    for d, e in entries.items():
        lines += [f"{NAMES[d]}: mean gap {e['gaps']['mean_days']:.2f} d, median gap {e['gaps']['median_days']:g} d; "
                  f"h* = {_h(e['h_star_days'])} d = {_days(e['h_star_in_baskets']['via_mean_gap'])} baskets (mean gap) = "
                  f"{_days(e['h_star_in_baskets']['via_median_gap'])} baskets (median gap); gamma* = {e['fit']['gamma_star']:g}.", ""]
        rows = []
        for row in e["unified_scale"]:
            mark = " *" if row["gamma"] == e["fit"]["gamma_star"] else ""
            rows.append([f"{row['gamma']:g}{mark}", _days(row["basket_half_life"]), _days(row["day_equivalent_mean_gap"]),
                         _days(row["day_equivalent_median_gap"])])
        rows.append([f"h* = {_h(e['h_star_days'])} d", _days(e["h_star_in_baskets"]["via_median_gap"]) + " (median gap)",
                     _h(e["h_star_days"]), _h(e["h_star_days"])])
        lines += _table(["gamma (* = gamma*)", "half-life (baskets)", "day equivalent, mean gap", "day equivalent, median gap"], rows)

    lines += ["## (e) History-length bins (distinct fit-history baskets; fit recall)", ""]
    for d, e in entries.items():
        bins = e["by_history_length"]
        hs = [_h(x["h"]) for x in bins[0]["day"]]
        gs = [f"{x['gamma']:g}" for x in bins[0]["basket"]]
        lines += [f"{NAMES[d]}, day decay by h (days):", ""]
        lines += _table(["bin", "users"] + hs, [[b["bin"], f"{b['n_users']:,}"] + [_f(x["recall"]) for x in b["day"]] for b in bins])
        lines += [f"{NAMES[d]}, basket decay by gamma:", ""]
        lines += _table(["bin", "users"] + gs, [[b["bin"], f"{b['n_users']:,}"] + [_f(x["recall"]) for x in b["basket"]] for b in bins])

    lines += ["## Reproduction check", ""]
    if payload["reproduction_check"]:
        rows = []
        for d, checks in payload["reproduction_check"].items():
            for key, item in checks.items():
                if isinstance(item, dict):
                    rows.append([NAMES[d], key, _f(item["expected"]), _f(item["observed"]), "yes" if item["matched"] else "NO"])
        lines += _table(["dataset", "quantity", "expected", "observed", "matched at 4 decimals"], rows)
    else:
        lines += ["(no reference numbers for this profile)", ""]

    lines += ["## Figure", "", f"`{payload['figure']['png']}` (numbers in `{payload['figure']['csv']}`): fit repeat Recall@10 vs "
              "half-life in days (log x); day decay solid, basket decay converted with the median inter-basket gap dashed; "
              "one panel per dataset; dotted line = personal frequency (h = inf, gamma = 1).", ""]
    lines += ["## Timing and memory", ""]
    lines += _table(["dataset", "load + check", "fit grids", "eval grids", "bootstrap", "gaps + bins", "peak working set (MiB)"],
                    [[NAMES[d], *(f"{e['timing_seconds'][k]:.1f} s" for k in ("load_and_check", "fit_grids", "eval_grids", "bootstrap", "gaps_and_bins")),
                      f"{e['peak_memory_mib']:.0f}"] for d, e in entries.items()])
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def input_digests(config: dict[str, Any], dataset: str) -> dict[str, str]:
    folder = dataset_dir(config, dataset)
    files = {name: folder / f"{name}.parquet" for name in INPUT_FILES}
    files["basket_days"] = basket_days_path(config, dataset)
    return {name: sha256_of(path) for name, path in files.items()}


def main(argv: list[str] | None = None) -> Path:
    args = parse_args(argv)
    if args.profile not in ALLOWED_PROFILES:
        raise ValueError(f"profile must be one of {ALLOWED_PROFILES}")
    t_start = time.perf_counter()
    config = load_config(args.profile)
    if config.get("role") == "final":
        raise PermissionError("posthoc_day_decay: profiles with role 'final' are never analysed")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    entries: dict[str, dict[str, Any]] = {}
    series: dict[str, dict[str, dict[float, pd.Series]]] = {}
    digests = {}
    for dataset in args.datasets:
        print(f"[{dataset}] analysing (profile {args.profile})")
        digests[dataset] = input_digests(config, dataset)
        entries[dataset], series[dataset] = analyse_dataset(config, dataset, args.bootstrap_n, args.seed)
        fit = entries[dataset]["fit"]
        print(f"[{dataset}] h* = {fit['h_star']} days, gamma* = {fit['gamma_star']:g}; "
              f"timings {{{', '.join(f'{k}: {v:.1f} s' for k, v in entries[dataset]['timing_seconds'].items())}}}")
    transfers = transfer(entries, series, args.bootstrap_n, args.seed)
    checks = reproduction_check(args.profile, entries)
    for dataset, items in checks.items():
        print(f"[{dataset}] reproduction check: {'matched' if items['all_matched'] else 'NOT matched'} "
              + "; ".join(f"{k} expected {v['expected']:.4f} observed {v['observed']:.4f}" for k, v in items.items() if isinstance(v, dict)))
    figure = make_figure(entries, out_dir)
    payload = {
        "label": LABEL,
        "task": "W14 post-hoc day-based recency decay",
        "profile": args.profile,
        "k": K,
        "metric": "repeat Recall@10 on users with >= 1 repeat target item (nbr.metrics.per_user_recall, mean over users)",
        "grids": {"half_life_days": [jsonable(h) for h in HALF_LIFE_GRID], "gamma": GAMMA_GRID},
        "history_bins": [name for name, _, _ in bin_labels(HISTORY_BIN_EDGES)],
        "bootstrap": {"n": args.bootstrap_n, "seed": args.seed, "level": 0.95},
        "inputs_sha256": digests,
        "data_dir": str(config["data_dir"]),
        "datasets": entries,
        "transfer": transfers,
        "reproduction_check": checks,
        "figure": figure,
        "environment": {"python": f"Python {platform.python_version()} ({platform.system()})", "pandas": pd.__version__,
                        "numpy": np.__version__, "matplotlib": matplotlib.__version__},
        "runtime_seconds": time.perf_counter() - t_start,
        "peak_memory_mib": peak_memory_mib(),
    }
    json_path = write_json(out_dir / f"day_decay_{args.profile}.json", payload, config)
    import json  # local: re-read the written document so the markdown reflects exactly what is on disk

    document = json.loads(json_path.read_text(encoding="utf-8"))
    md_path = out_dir / f"day_decay_{args.profile}.md"
    md_path.write_text(render_markdown(document), encoding="utf-8")
    print(f"wrote {json_path}, {md_path}, {out_dir / figure['png']}, {out_dir / figure['csv']}; "
          f"runtime {payload['runtime_seconds']:.1f} s; peak working set {payload['peak_memory_mib']:.0f} MiB")
    return json_path


if __name__ == "__main__":
    main()
