"""Summary figures and tables for the revised claim H*' (health attributes only).

H*': for each health attribute, purchases are concentrated in a group of users whose attribute
purchasing persists over time (1) and is associated with the attribute choice among first-time
purchases within the same ingredient type (2).

This script only reads existing result JSON files. It does not import or modify any ml_group9
module, so it can run while other analyses are in progress.

Design rules (decided 2026-11-05):
- No placebo baseline is shown or used (placebo sets could not be built where attribute items
  outnumber non-attribute items, e.g. milk).
- Flavor attributes appear only in an appendix figure as context, never as a test.
- Correlations use unsmoothed shares (T1b R1). Their 95% CIs are a Fisher-z approximation
  (SE = 1.06 / sqrt(n - 3)) because the JSON stores no bootstrap CI for the raw correlations.

Usage (from the repository root, after analyze-hstar, ml_group9.hstar_robust and ml_group9.hier_followup):
  python scripts/make_hstarp_figures.py \
    --hstar reports/metrics/medium_hstar.json \
    --robust reports/metrics/medium_hstar_robust.json \
    --hier-followup reports/metrics/medium_hier_followup.json \
    --out-dir reports
  --hier-followup is optional; without it the within-type AUC figure is skipped. Outputs go to
  <out-dir>/figures/<prefix>_*.png, <out-dir>/figures/HSTARP_FIGURES.md and <out-dir>/tables/<prefix>_*.csv
  (--prefix defaults to medium_hstarp).
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HEALTH = ["fat_reduced", "sugar_reduced", "gluten_free", "low_sodium"]
FLAVOR = ["chocolate", "vanilla", "strawberry"]
LABEL = {
    "fat_reduced": "Fat-reduced",
    "sugar_reduced": "Sugar-reduced",
    "gluten_free": "Gluten-free",
    "low_sodium": "Low-sodium",
    "chocolate": "Chocolate",
    "vanilla": "Vanilla",
    "strawberry": "Strawberry",
}

# Validated categorical slots (dataviz reference palette, light surface).
SERIES_1 = "#2a78d6"  # blue
SERIES_2 = "#eb6834"  # orange
LORENZ_COLORS = {"fat_reduced": "#2a78d6", "sugar_reduced": "#eb6834", "gluten_free": "#1baf7a", "low_sodium": "#eda100"}
SURFACE = "#fcfcfb"
INK = "#1f1f1d"
INK_MUTED = "#6b6a63"
GRID = "#e4e3de"
DPI = 200
FIGSIZE = (8.0, 4.5)  # 1600 x 900 px at 200 dpi

plt.rcParams.update(
    {
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "axes.edgecolor": GRID,
        "axes.labelcolor": INK,
        "axes.titlecolor": INK,
        "xtick.color": INK_MUTED,
        "ytick.color": INK_MUTED,
        "text.color": INK,
        "font.size": 9,
        "axes.titlesize": 10.5,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "legend.frameon": False,
    }
)


# ---------------------------------------------------------------------------------------------
# Statistics helpers


def fisher_ci(rho: float, n: int, z_crit: float = 1.959964) -> tuple[float, float]:
    """Approximate CI for a Spearman correlation (Fieller-type SE 1.06 / sqrt(n - 3))."""
    if n <= 3 or abs(rho) >= 1:
        return float("nan"), float("nan")
    z = math.atanh(rho)
    se = 1.06 / math.sqrt(n - 3)
    return math.tanh(z - z_crit * se), math.tanh(z + z_crit * se)


def wilson_ci(p: float, n: float, z_crit: float = 1.959964) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion."""
    if n <= 0:
        return float("nan"), float("nan")
    denom = 1 + z_crit**2 / n
    centre = (p + z_crit**2 / (2 * n)) / denom
    half = z_crit * math.sqrt(p * (1 - p) / n + z_crit**2 / (4 * n**2)) / denom
    return centre - half, centre + half


# ---------------------------------------------------------------------------------------------
# Data extraction


def load(path: str | Path) -> dict:
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def persistence_rows(robust: dict, attributes: list[str]) -> list[dict]:
    """Unsmoothed half-to-half correlations (all purchases, first-time purchases) with approx. CIs."""
    rows = []
    for name in attributes:
        all_ = robust["r1_raw"]["attributes"][name]
        new = robust["r1_raw_new"]["attributes"][name]
        lo, hi = fisher_ci(all_["m1"], int(all_["users"]))
        nlo, nhi = fisher_ci(new["m1"], int(new["users"]))
        rows.append(
            {
                "attribute": name,
                "group": "health" if name in HEALTH else "flavor (context)",
                "rho_all": all_["m1"],
                "rho_all_ci_low": lo,
                "rho_all_ci_high": hi,
                "users_all": int(all_["users"]),
                "rho_first_time": new["m1"],
                "rho_first_time_ci_low": nlo,
                "rho_first_time_ci_high": nhi,
                "users_first_time": int(new["users"]),
                "source": "medium_hstar_robust.json: r1_raw / r1_raw_new .attributes.<a>.m1, .users",
            }
        )
    return rows


def buyer_rows(robust: dict, attributes: list[str]) -> list[dict]:
    """P(buy in 2nd half | bought in 1st half) and P(... | not), Wilson CIs, odds ratio."""
    rows = []
    for name in attributes:
        p = robust["r2_buyers"][name]["persistence"]
        n_users = float(p["users"])
        n_b1 = float(p["first_half_buyers"])
        n_not = n_users - n_b1
        lo1, hi1 = wilson_ci(p["p_b2_given_b1"], n_b1)
        lo0, hi0 = wilson_ci(p["p_b2_given_not_b1"], n_not)
        rows.append(
            {
                "attribute": name,
                "group": "health" if name in HEALTH else "flavor (context)",
                "p_again_if_bought": p["p_b2_given_b1"],
                "p_again_if_bought_ci_low": lo1,
                "p_again_if_bought_ci_high": hi1,
                "first_half_buyers": int(n_b1),
                "p_start_if_not_bought": p["p_b2_given_not_b1"],
                "p_start_if_not_bought_ci_low": lo0,
                "p_start_if_not_bought_ci_high": hi0,
                "first_half_non_buyers": int(n_not),
                "odds_ratio": p["odds_ratio"],
                "source": "medium_hstar_robust.json: r2_buyers.<a>.persistence",
            }
        )
    return rows


def within_rows(hier: dict, attributes: list[str]) -> list[dict]:
    """Within-type AUC on first-time purchase rows: all (aisle, type) cells and mapped types only."""
    rows = []
    for name in attributes:
        new = hier["attributes"][name]["new"]
        scores = new["within_type_auc"]
        mapped = new["within_type_auc_mapped"]
        diff = new["diff_a_minus_b"]
        for subset, block in (("all_cells", scores), ("mapped_types_only", mapped)):
            a = block["a"]
            row = {
                "attribute": name,
                "subset": subset,
                "auc_a": a["value"],
                "auc_a_ci95_low": a["ci95_low"],
                "auc_a_ci95_high": a["ci95_high"],
                "auc_a_ci9875_low": a["ci98.75_low"],
                "auc_a_ci9875_high": a["ci98.75_high"],
                "types": int(a["types"]),
                "positive_rows": int(a["positive_rows"]),
                "negative_rows": int(a["negative_rows"]),
                "pairs": int(a["pairs"]),
                "users": int(a["users"]),
                "auc_c": block["c"]["value"],
                "auc_c_ci95_low": block["c"]["ci95_low"],
                "auc_c_ci95_high": block["c"]["ci95_high"],
                "auc_b": block["b"]["value"] if "b" in block else None,
                "auc_b_ci95_low": block["b"]["ci95_low"] if "b" in block else None,
                "auc_b_ci95_high": block["b"]["ci95_high"] if "b" in block else None,
                "diff_a_minus_b": diff["value"] if subset == "all_cells" else None,
                "diff_a_minus_b_ci95_low": diff["ci95_low"] if subset == "all_cells" else None,
                "diff_a_minus_b_ci95_high": diff["ci95_high"] if subset == "all_cells" else None,
                "fallback_pair_share": new["fallback_pair_share"] if subset == "all_cells" else 0.0,
                "source": "medium_hier_followup.json: attributes.<a>.new.within_type_auc[_mapped], diff_a_minus_b",
            }
            rows.append(row)
    return rows


def lorenz_rows(hstar: dict, attributes: list[str]) -> list[dict]:
    rows = []
    for name in attributes:
        curve = hstar["lorenz"][name]
        m3 = hstar["attributes"][name]["m3"]
        for x, y in zip(curve["user_share"], curve["purchase_share"]):
            rows.append(
                {
                    "attribute": name,
                    "user_share": x,
                    "purchase_share": y,
                    "gini": m3["gini"],
                    "top20_share": m3["top20_share"],
                    "category_top20_share": m3["category_top20_share"],
                    "source": "medium_hstar.json: lorenz.<a>, attributes.<a>.m3",
                }
            )
    return rows


# ---------------------------------------------------------------------------------------------
# Output helpers


def write_csv(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def style_axis(ax) -> None:
    ax.yaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(length=0)


def paired_bars(ax, rows: list[dict], keys: tuple[str, str], labels: tuple[str, str], counts: tuple[str, str]) -> None:
    """Two thin bars per attribute with error bars; a small gap separates the pair."""
    width, gap = 0.34, 0.03
    for idx, row in enumerate(rows):
        for side, (key, color) in enumerate(zip(keys, (SERIES_1, SERIES_2))):
            x = idx + (-1 if side == 0 else 1) * (width / 2 + gap / 2)
            value = row[key]
            ax.bar(x, value, width=width, color=color, label=labels[side] if idx == 0 else None, zorder=2)
            ax.errorbar(
                x,
                value,
                yerr=[[value - row[f"{key}_ci_low"]], [row[f"{key}_ci_high"] - value]],
                fmt="none",
                ecolor=INK,
                elinewidth=1,
                capsize=2.5,
                zorder=3,
            )
    ax.set_xticks(range(len(rows)))
    ax.set_xticklabels(
        [f"{LABEL[r['attribute']]}\nn = {r[counts[0]]:,} / {r[counts[1]]:,}" for r in rows], fontsize=8.5
    )
    style_axis(ax)


# ---------------------------------------------------------------------------------------------
# Figures


def figure_persistence(rows: list[dict], path: Path) -> None:
    fig, ax = plt.subplots(figsize=FIGSIZE)
    paired_bars(ax, rows, ("rho_all", "rho_first_time"), ("All purchases", "First-time purchases only"), ("users_all", "users_first_time"))
    ax.set_ylim(0, 0.85)
    ax.set_ylabel("Spearman ρ, first vs second half of history")
    ax.set_title("Persistence of user-level purchase share, health attributes")
    ax.legend(loc="upper right", ncol=2)
    fig.text(
        0.01,
        0.01,
        "Unsmoothed shares within each attribute's applicable aisles. Eligible users: ≥ 10 purchases (all) or ≥ 5 first-time\n"
        "purchases (first-time) in the applicable aisles in each half; n = all / first-time. "
        "Error bars: approximate 95% CI (Fisher z). No placebo baseline.",
        fontsize=7.5,
        color=INK_MUTED,
    )
    fig.tight_layout(rect=(0, 0.065, 1, 1))
    fig.savefig(path, dpi=DPI)
    plt.close(fig)


def figure_buyers(rows: list[dict], path: Path) -> None:
    fig, ax = plt.subplots(figsize=FIGSIZE)
    paired_bars(
        ax,
        rows,
        ("p_again_if_bought", "p_start_if_not_bought"),
        ("Bought in first half", "Did not buy in first half"),
        ("first_half_buyers", "first_half_non_buyers"),
    )
    for idx, row in enumerate(rows):
        ax.text(idx, row["p_again_if_bought_ci_high"] + 0.04, f"OR {row['odds_ratio']:.1f}", ha="center", fontsize=8, color=INK)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Share buying the attribute in the second half")
    ax.set_title("Second-half purchase by first-half buyer status, health attributes")
    ax.legend(loc="upper right", ncol=2, bbox_to_anchor=(1.0, 1.0))
    fig.text(
        0.01,
        0.01,
        "Users with ≥ 10 purchases in the applicable aisles in each half; n = first-half buyers / non-buyers.\n"
        "OR = odds ratio of buying in the second half, first-half buyers vs non-buyers. Error bars: Wilson 95% CI.",
        fontsize=7.5,
        color=INK_MUTED,
    )
    fig.tight_layout(rect=(0, 0.065, 1, 1))
    fig.savefig(path, dpi=DPI)
    plt.close(fig)


def figure_context(persist: list[dict], buyers: list[dict], path: Path) -> None:
    """Appendix: health and flavor attributes side by side, as context only."""
    fig, axes = plt.subplots(1, 2, figsize=(10.0, 4.5))
    names = [r["attribute"] for r in persist]
    split = len([n for n in names if n in HEALTH]) - 0.5
    panels = [
        (axes[0], persist, ("rho_all", "rho_first_time"), ("All purchases", "First-time purchases only"), "Spearman ρ, first vs second half", (0, 0.85)),
        (axes[1], buyers, ("p_again_if_bought", "p_start_if_not_bought"), ("Bought in first half", "Did not buy in first half"), "Share buying in second half", (0, 1.05)),
    ]
    width, gap = 0.34, 0.03
    for ax, rows, keys, labels, ylabel, ylim in panels:
        for idx, row in enumerate(rows):
            for side, (key, color) in enumerate(zip(keys, (SERIES_1, SERIES_2))):
                x = idx + (-1 if side == 0 else 1) * (width / 2 + gap / 2)
                value = row[key]
                ax.bar(x, value, width=width, color=color, label=labels[side] if idx == 0 else None, zorder=2)
                ax.errorbar(
                    x, value,
                    yerr=[[value - row[f"{key}_ci_low"]], [row[f"{key}_ci_high"] - value]],
                    fmt="none", ecolor=INK, elinewidth=1, capsize=2, zorder=3,
                )
        ax.axvline(split, color=INK_MUTED, linewidth=0.8, linestyle=(0, (3, 3)))
        ax.text(split / 2, ylim[1] * 0.97, "Health", ha="center", va="top", fontsize=8.5, color=INK)
        ax.text(split + (len(rows) - split) / 2, ylim[1] * 0.97, "Flavor (context)", ha="center", va="top", fontsize=8.5, color=INK_MUTED)
        ax.set_xticks(range(len(rows)))
        ax.set_xticklabels([LABEL[r["attribute"]] for r in rows], rotation=25, ha="right", fontsize=8)
        ax.set_ylim(*ylim)
        ax.set_ylabel(ylabel)
        style_axis(ax)
    axes[0].set_title("Persistence of purchase share", pad=24)
    axes[1].set_title("Second-half purchase by first-half status", pad=24)
    for ax in axes:
        ax.legend(loc="lower left", bbox_to_anchor=(0.0, 1.0), ncol=2, fontsize=8, borderaxespad=0.2)
    fig.text(
        0.01, 0.01,
        "Appendix. Flavor attributes are shown as context, not as a comparison test. "
        "CIs: Fisher z (left), Wilson (right).",
        fontsize=7.5, color=INK_MUTED,
    )
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(path, dpi=DPI)
    plt.close(fig)


def figure_within(rows: list[dict], path: Path) -> None:
    """Main, H*' (2): forest plot of within-type AUC (score a), all cells vs mapped types only."""
    fig, ax = plt.subplots(figsize=FIGSIZE)
    names = list(dict.fromkeys(r["attribute"] for r in rows))
    series = [("all_cells", "All (aisle, type) cells", SERIES_1, -0.16), ("mapped_types_only", "Mapped ingredient types only", SERIES_2, 0.16)]
    for subset, label, color, offset in series:
        for idx, name in enumerate(names):
            row = next(r for r in rows if r["attribute"] == name and r["subset"] == subset)
            y = idx + offset
            ax.hlines(y, row["auc_a_ci9875_low"], row["auc_a_ci9875_high"], color=color, linewidth=1.2, zorder=2)
            ax.hlines(y, row["auc_a_ci95_low"], row["auc_a_ci95_high"], color=color, linewidth=3.2, zorder=3)
            ax.plot(row["auc_a"], y, "o", color=color, markersize=6.5, markeredgecolor=SURFACE, markeredgewidth=1.2,
                    zorder=4, label=label if idx == 0 else None)
            ax.text(row["auc_a_ci9875_high"] + 0.006, y, f"{row['auc_a']:.3f}", va="center", fontsize=7.5, color=INK)
    ax.axvline(0.5, color=INK_MUTED, linewidth=1, linestyle=(0, (3, 3)), zorder=1)
    ax.text(0.5, -0.62, "0.5 = no association ", ha="right", va="center", fontsize=7.5, color=INK_MUTED)
    labels = []
    for name in names:
        all_ = next(r for r in rows if r["attribute"] == name and r["subset"] == "all_cells")
        mapped = next(r for r in rows if r["attribute"] == name and r["subset"] == "mapped_types_only")
        labels.append(f"{LABEL[name]}\ntypes {all_['types']} / {mapped['types']}")
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(labels, fontsize=8.5)
    ax.set_ylim(len(names) - 0.4, -0.8)
    ax.set_xlim(0.38, 0.80)
    ax.set_xlabel(r"Within-type AUC of the user's attribute share $\pi_u(a)$")
    ax.set_title("Attribute choice among first-time purchases within the same type, health attributes")
    ax.xaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(length=0)
    ax.legend(loc="lower right", fontsize=8)
    fig.text(
        0.01,
        0.01,
        "Pairs are compared only within the same (aisle, type) cell. Mapped types only: aisle-level fallback cells removed.\n"
        "Thick bars: 95% CI; thin bars: 98.75% CI (4 attributes). User bootstrap, 1,000 resamples. Types = all / mapped.",
        fontsize=7.5,
        color=INK_MUTED,
    )
    fig.tight_layout(rect=(0, 0.065, 1, 1))
    fig.savefig(path, dpi=DPI)
    plt.close(fig)


def figure_lorenz(hstar: dict, path: Path) -> None:
    """Appendix: Lorenz curves of attribute purchases across all evaluation users."""
    fig, ax = plt.subplots(figsize=(6.0, 5.0))
    ax.plot([0, 1], [0, 1], color=INK_MUTED, linewidth=1, linestyle=(0, (3, 3)), label="Equal purchases")
    for name in HEALTH:
        curve = hstar["lorenz"][name]
        gini = hstar["attributes"][name]["m3"]["gini"]
        top20 = hstar["attributes"][name]["m3"]["top20_share"]
        ax.plot(
            curve["user_share"], curve["purchase_share"], color=LORENZ_COLORS[name], linewidth=2,
            label=f"{LABEL[name]} (Gini {gini:.2f}; top 20% of users {top20:.0%})",
        )
    category = [hstar["attributes"][name]["m3"]["category_top20_share"] for name in HEALTH]
    users = {int(hstar["attributes"][name]["m3"]["users"]) for name in HEALTH}
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Cumulative share of users (ascending by attribute purchases)")
    ax.set_ylabel("Cumulative share of attribute purchases")
    ax.set_title("Concentration of health-attribute purchases")
    ax.legend(loc="upper left", fontsize=8)
    style_axis(ax)
    ax.xaxis.grid(True, color=GRID, linewidth=0.8)
    fig.text(
        0.01, 0.01,
        f"Appendix. {', '.join(f'{u:,}' for u in sorted(users))} evaluation users, including non-buyers. Reference: the top 20% of users\n"
        f"hold {min(category):.0%}–{max(category):.0%} of all purchases in each attribute's applicable aisles.",
        fontsize=7, color=INK_MUTED,
    )
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.savefig(path, dpi=DPI)
    plt.close(fig)


# ---------------------------------------------------------------------------------------------


def write_index(out_dir: Path, entries: list[tuple[str, str, str]]) -> None:
    lines = [
        "# Figures for H*' (health attributes; no placebo; flavor as context)",
        "",
        "Generated by scripts/make_hstarp_figures.py from existing result JSON files.",
        "",
        "| Figure | Table | Content |",
        "|---|---|---|",
    ]
    lines += [f"| {fig} | {tab} | {desc} |" for fig, tab, desc in entries]
    lines += [
        "",
        "Not used in the main text (placebo-based): medium_hstar_m1.png, medium_hstar_delta.png.",
        "Within-type AUC figure is written only when --hier-followup is given (T2/T2b JSON).",
    ]
    (out_dir / "figures" / "HSTARP_FIGURES.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--hstar", required=True, help="medium_hstar.json (pre-registered analysis)")
    parser.add_argument("--robust", required=True, help="medium_hstar_robust.json (T1b)")
    parser.add_argument("--out-dir", required=True, help="reports directory; figures/ and tables/ are created inside")
    parser.add_argument("--prefix", default="medium_hstarp")
    parser.add_argument("--hier-followup", default=None, help="medium_hier_followup.json (T2), optional")
    args = parser.parse_args()

    hstar, robust = load(args.hstar), load(args.robust)
    out_dir = Path(args.out_dir)
    (out_dir / "figures").mkdir(parents=True, exist_ok=True)
    (out_dir / "tables").mkdir(parents=True, exist_ok=True)
    p = args.prefix

    persist_h, persist_all = persistence_rows(robust, HEALTH), persistence_rows(robust, HEALTH + FLAVOR)
    buyers_h, buyers_all = buyer_rows(robust, HEALTH), buyer_rows(robust, HEALTH + FLAVOR)

    figure_persistence(persist_h, out_dir / "figures" / f"{p}_persistence.png")
    write_csv(persist_h, out_dir / "tables" / f"{p}_persistence.csv")
    figure_buyers(buyers_h, out_dir / "figures" / f"{p}_buyers.png")
    write_csv(buyers_h, out_dir / "tables" / f"{p}_buyers.csv")
    figure_context(persist_all, buyers_all, out_dir / "figures" / f"{p}_appendix_flavor_context.png")
    write_csv(persist_all, out_dir / "tables" / f"{p}_appendix_persistence_all.csv")
    write_csv(buyers_all, out_dir / "tables" / f"{p}_appendix_buyers_all.csv")
    figure_lorenz(hstar, out_dir / "figures" / f"{p}_appendix_lorenz.png")
    write_csv(lorenz_rows(hstar, HEALTH), out_dir / "tables" / f"{p}_appendix_lorenz.csv")

    entries = [
        (f"{p}_persistence.png", f"{p}_persistence.csv", "Main, H*' (1): half-to-half ρ, all vs first-time purchases"),
        (f"{p}_buyers.png", f"{p}_buyers.csv", "Main, H*' (1): second-half purchase by first-half buyer status, odds ratio"),
        (f"{p}_appendix_flavor_context.png", f"{p}_appendix_*_all.csv", "Appendix: health and flavor side by side (context, not a test)"),
        (f"{p}_appendix_lorenz.png", f"{p}_appendix_lorenz.csv", "Appendix: Lorenz curves (includes non-buyers)"),
    ]
    if args.hier_followup:
        hier = load(args.hier_followup)
        within = within_rows(hier, HEALTH)
        figure_within(within, out_dir / "figures" / f"{p}_within_type_auc.png")
        write_csv(within, out_dir / "tables" / f"{p}_within_type_auc.csv")
        entries.append(
            (f"{p}_within_type_auc.png", f"{p}_within_type_auc.csv",
             "Main, H*' (2): within-type AUC, first-time purchases, all cells vs mapped types only")
        )
    write_index(out_dir, entries)
    print(f"Wrote figures and tables under {out_dir}")


if __name__ == "__main__":
    main()
