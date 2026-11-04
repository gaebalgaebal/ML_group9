"""Report figures (A4, 6.3 in wide, every text element >= 10 pt at 100% insertion).

Usage (from the repository root, after ml_group9.hstar_robust and ml_group9.hier_followup):
  python scripts/make_report_figures.py reports/metrics/medium_hstar_robust.json \
    reports/metrics/medium_hier_followup.json reports/figures
Arguments are positional: hstar_robust JSON, hier_followup JSON (Instacart), output directory.
Writes fig1_persistence.png, fig2_pooled_vs_within.png and fig3_within_type_auc.png.
"""
import json
import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

robust = json.load(open(sys.argv[1]))
hier = json.load(open(sys.argv[2]))
out = Path(sys.argv[3])
out.mkdir(parents=True, exist_ok=True)

HEALTH = ["fat_reduced", "sugar_reduced", "gluten_free", "low_sodium"]
TICK = {"fat_reduced": "Fat", "sugar_reduced": "Sugar", "gluten_free": "Gluten", "low_sodium": "Sodium"}
NAME = {"fat_reduced": "Fat-reduced", "sugar_reduced": "Sugar-reduced", "gluten_free": "Gluten-free", "low_sodium": "Low-sodium"}
C1, C2, C3 = "#2a78d6", "#eb6834", "#1baf7a"
SURFACE, INK, MUTED, GRID = "#ffffff", "#1f1f1d", "#5f5e58", "#e4e3de"
W = 6.3
FS = 10

plt.rcParams.update({
    "font.size": FS, "axes.titlesize": FS, "axes.labelsize": FS, "xtick.labelsize": FS, "ytick.labelsize": FS,
    "legend.fontsize": FS, "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "axes.edgecolor": GRID,
    "axes.labelcolor": INK, "text.color": INK, "xtick.color": INK, "ytick.color": INK,
    "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False,
})


def fisher(r, n):
    z, se = math.atanh(r), 1.06 / math.sqrt(n - 3)
    return math.tanh(z - 1.96 * se), math.tanh(z + 1.96 * se)


def wilson(p, n):
    z = 1.96
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def bars(ax, vals, cis, colors, labels, width=0.36, gap=0.03):
    k = len(vals)
    for s in range(k):
        for i, (v, (lo, hi)) in enumerate(zip(vals[s], cis[s])):
            x = i + (s - (k - 1) / 2) * (width + gap)
            ax.bar(x, v, width=width, color=colors[s], label=labels[s] if i == 0 else None, zorder=2)
            ax.errorbar(x, v, yerr=[[v - lo], [hi - v]], fmt="none", ecolor=INK, elinewidth=1, capsize=2.5, zorder=3)
    ax.set_xticks(range(len(HEALTH)))
    ax.set_xticklabels([TICK[a] for a in HEALTH])
    ax.yaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(length=0)


# ---- Figure 1: persistence (a) rho, (b) buyer persistence
fig, axes = plt.subplots(1, 2, figsize=(W, 2.8))
r_all = [robust["r1_raw"]["attributes"][a] for a in HEALTH]
r_new = [robust["r1_raw_new"]["attributes"][a] for a in HEALTH]
bars(axes[0], [[x["m1"] for x in r_all], [x["m1"] for x in r_new]],
     [[fisher(x["m1"], x["users"]) for x in r_all], [fisher(x["m1"], x["users"]) for x in r_new]],
     [C1, C2], ["All purchases", "First-time only"])
axes[0].set_ylim(0, 1.0)
axes[0].set_ylabel("Spearman ρ, 1st vs 2nd half")
axes[0].set_title("(a) Persistence of purchase share", loc="left")
axes[0].legend(loc="upper right", handlelength=1.2, borderaxespad=0.1)

pers = [robust["r2_buyers"][a]["persistence"] for a in HEALTH]
p1 = [p["p_b2_given_b1"] for p in pers]
p0 = [p["p_b2_given_not_b1"] for p in pers]
n1 = [p["first_half_buyers"] for p in pers]
n0 = [p["users"] - p["first_half_buyers"] for p in pers]
bars(axes[1], [p1, p0], [[wilson(p, n) for p, n in zip(p1, n1)], [wilson(p, n) for p, n in zip(p0, n0)]],
     [C1, C2], ["Bought in 1st half", "Did not buy"])
for i, p in enumerate(pers):
    axes[1].text(i, 0.955, f"OR {p['odds_ratio']:.1f}", ha="center", va="center", fontsize=FS, color=INK)
axes[1].set_ylim(0, 1.25)
axes[1].set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
axes[1].set_ylabel("Share buying in 2nd half")
axes[1].set_title("(b) Buyer persistence", loc="left")
axes[1].legend(loc="upper center", ncol=2, handlelength=1.2, columnspacing=0.8, borderaxespad=0.1)
fig.tight_layout(w_pad=1.2)
fig.savefig(out / "fig1_persistence.png", dpi=300)
plt.close(fig)

# ---- Figure 2: mixed AUC vs within-type AUC (first-time purchases)
rep = hier["reproduction"]["values"]
mixed_pop = [rep[f"{a}.new.pi_population_type"]["reference"] for a in HEALTH]
mixed_user = [rep[f"{a}.new.pi_user_type"]["reference"] for a in HEALTH]
within = [hier["attributes"][a]["new"]["within_type_auc"]["a"]["value"] for a in HEALTH]
fig, ax = plt.subplots(figsize=(W, 2.75))
width, gap = 0.26, 0.02
series = [(mixed_pop, C1, "Pooled: population rate (type)"),
          (mixed_user, C2, "Pooled: user share (type)"),
          (within, C3, "Within type: user share (aisle)")]
for s, (vals, color, label) in enumerate(series):
    for i, v in enumerate(vals):
        x = i + (s - 1) * (width + gap)
        ax.bar(x, v, width=width, color=color, label=label if i == 0 else None, zorder=2)
        ax.text(x, v + 0.012, f"{v:.2f}".lstrip("0"), ha="center", va="bottom", fontsize=FS, color=INK)
ax.axhline(0.5, color=MUTED, linewidth=1, linestyle=(0, (3, 3)), zorder=1)
ax.set_xticks(range(len(HEALTH)))
ax.set_xticklabels([NAME[a] for a in HEALTH])
ax.set_ylim(0.3, 1.03)
ax.set_yticks([0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
ax.set_xlim(-0.55, 3.55)
ax.set_ylabel("AUC, first-time purchases")
ax.yaxis.grid(True, color=GRID, linewidth=0.8)
ax.set_axisbelow(True)
ax.tick_params(length=0)
ax.legend(loc="lower left", bbox_to_anchor=(-0.02, 1.0), ncol=2, handlelength=1.2, borderaxespad=0.2, labelspacing=0.25, columnspacing=1.0)
fig.tight_layout()
fig.savefig(out / "fig2_pooled_vs_within.png", dpi=300)
plt.close(fig)

# ---- Figure 3: forest plot of within-type AUC
fig, ax = plt.subplots(figsize=(W, 2.85))
subsets = [("within_type_auc", "All (aisle, type) cells", C1, -0.17),
           ("within_type_auc_mapped", "Mapped ingredient types only", C2, 0.17)]
for key, label, color, off in subsets:
    for i, a in enumerate(HEALTH):
        r = hier["attributes"][a]["new"][key]["a"]
        y = i + off
        ax.hlines(y, r["ci98.75_low"], r["ci98.75_high"], color=color, linewidth=1.3, zorder=2)
        ax.hlines(y, r["ci95_low"], r["ci95_high"], color=color, linewidth=3.6, zorder=3)
        ax.plot(r["value"], y, "o", color=color, markersize=7, markeredgecolor=SURFACE, markeredgewidth=1.2,
                zorder=4, label=label if i == 0 else None)
        ax.text(r["ci98.75_high"] + 0.007, y, f"{r['value']:.3f}", va="center", fontsize=FS, color=INK)
ax.axvline(0.5, color=MUTED, linewidth=1, linestyle=(0, (3, 3)), zorder=1)
ylab = []
for a in HEALTH:
    t_all = hier["attributes"][a]["new"]["within_type_auc"]["a"]["types"]
    t_map = hier["attributes"][a]["new"]["within_type_auc_mapped"]["a"]["types"]
    ylab.append(f"{NAME[a]}\n({t_all} / {t_map} types)")
ax.set_yticks(range(len(HEALTH)))
ax.set_yticklabels(ylab)
ax.set_ylim(len(HEALTH) - 0.5, -0.5)
ax.set_xlim(0.40, 0.80)
ax.set_xlabel("Within-type AUC of the aisle-level user share; 0.5 = no association")
ax.xaxis.grid(True, color=GRID, linewidth=0.8)
ax.set_axisbelow(True)
ax.tick_params(length=0)
ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2, handlelength=1.0, columnspacing=1.2, borderaxespad=0.2)
fig.tight_layout()
fig.savefig(out / "fig3_within_type_auc.png", dpi=300)
plt.close(fig)
print("ok")
