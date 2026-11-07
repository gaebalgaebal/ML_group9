"""Per-user exploration-slot policy of PREREG v1.4 (W17): the lists P_T and S, the new-item hit-share metric and
their evaluation block.

Definitions (docs/PREREG.md v1.4, items 2, 4 and 5):

- C3 = combine(R2, E2, r*) = the list ``final_lr_ucf`` of nbr.pipeline (r* re-selected on the fit split).
- P_T = C3 with r_u = R_SHORT (9) for users with at most T distinct history baskets of the evaluated split and
  r_u = r* otherwise (one exploration slot for short histories; explore items fill the last 10 - r_u slots).
- S = the separate exploration list: the top 10 of E2 (``usercf``) on its own; comparator = flat popularity
  excluding history items (``flat_explore``).
- new-item hit share of a list = share of users with at least one new target item among the top 10; over all
  users a user without a new target item counts 0.

T = 4 and DELTA = 0.003 are fixed by PREREG v1.4 from the W15 post-hoc study and are not re-selected; the policy
changes no selection. Primary (Instacart holdout3, family of 3, two-sided 98.33 % CI = 1 - 0.05/3; the 95 % CI is
reported alongside): H11 NDCG@10 of P_T minus C3 over all users, "non-inferior" if the lower bound > -DELTA;
H12 new-item hit share of P_T minus C3 over all users and H13 new-item hit share of S minus flat_explore over
users with >= 1 new target item, "improvement observed" if the lower bound > 0, "lower" if the upper bound < 0,
otherwise "no difference established". Secondary (no confirmatory claim): Recall@10, new-item Recall@10 and
repeat Recall@10 of P_T vs C3, history-length bins, the r_short = 8 variant and T in {6, 10}.

The functions never read files or targets other than the frames they receive.
"""

from __future__ import annotations

import math
import time
from typing import Any, Callable

import pandas as pd

from nbr.combine import combine
from nbr.metrics import paired_bootstrap, per_user_ndcg, per_user_recall
from nbr.repeat import LIST_COLUMNS

K = 10
T = 4                                   # history-length threshold (distinct history baskets), PREREG v1.4 item 2
DELTA = 0.003                           # non-inferiority margin of H11 (NDCG@10), PREREG v1.4 item 4
R_SHORT = 9                             # repeat slots for users with history_len <= T (one exploration slot)
R_SHORT_VARIANT = 8                     # secondary variant (two exploration slots)
T_SENSITIVITY = (6, 10)                 # secondary sensitivity values of T
LEVELS = (0.95, 0.9833)                 # CI levels: 95 % and 1 - 0.05/3 (family of 3), four decimals
PRIMARY_LEVEL = "0.9833"
PREREG = "v1.4"
LIST_SOURCES = {"C3": "final_lr_ucf", "repeat": "repeat_lr", "explore": "usercf", "flat": "flat_explore"}
METRICS = {
    "ndcg": ("NDCG@10", "all users"),
    "recall": ("Recall@10", "all users"),
    "new_recall": ("new-item Recall@10", "users with new target"),
    "new_hit_share": ("new-item hit share@10", "all users"),
    "new_hit_share_new_users": ("new-item hit share@10", "users with new target"),
    "repeat_recall": ("repeat Recall@10", "users with repeat target"),
}
SECONDARY_METRICS = ("recall", "new_recall", "repeat_recall", "new_hit_share_new_users")
VARIANT_METRICS = ("ndcg", "new_hit_share", "recall", "new_recall", "repeat_recall", "new_hit_share_new_users")
NON_INFERIOR = "non-inferior"
NOT_ESTABLISHED = "non-inferiority not established"


# ---------------------------------------------------------------------------
# lists
# ---------------------------------------------------------------------------
def slot_counts(lengths: pd.Series, T: int, r_short: int, r_default: int) -> pd.Series:
    """r_u per user of ``lengths`` (user_id -> distinct history baskets): ``r_short`` where lengths <= ``T``, else ``r_default``."""
    counts = pd.Series(r_default, index=lengths.index, dtype="int64", name="r")
    counts[lengths.to_numpy() <= T] = r_short
    return counts


def combine_policy(
    repeat: pd.DataFrame, explore: pd.DataFrame, lengths: pd.Series, T: int, r_short: int, r_default: int, k: int = K
) -> pd.DataFrame:
    """Per-user slot combination: nbr.combine.combine(repeat_u, explore_u, r_u, k) with r_u = slot_counts(lengths, T, r_short, r_default).

    ``lengths`` maps user_id to the number of distinct history baskets (nbr.pipeline.history_lengths of the
    evaluated split); every user of ``repeat`` or ``explore`` must be present (ValueError otherwise). Output
    columns and order as combine (user_id, rank, item_id, source; sorted by user_id, rank). With
    r_short == r_default the result equals combine(repeat, explore, r_default, k) exactly.
    """
    for r in (r_short, r_default):
        if not 0 <= r <= k:
            raise ValueError(f"slot counts must lie in 0..{k}, got r_short={r_short}, r_default={r_default}")
    if T < 0:
        raise ValueError(f"T must be >= 0, got {T}")
    users = pd.Index(repeat["user_id"]).union(pd.Index(explore["user_id"]))
    missing = users.difference(lengths.index)
    if len(missing):
        raise ValueError(f"{len(missing)} list users have no history length, e.g. {sorted(missing)[:5]}")
    counts = slot_counts(lengths, T, r_short, r_default)
    parts = []
    for r in sorted({r_short, r_default}):
        selected = counts.index[counts.to_numpy() == r]
        rep = repeat[repeat["user_id"].isin(selected)]
        exp = explore[explore["user_id"].isin(selected)]
        if rep.empty and exp.empty:
            continue
        parts.append(combine(rep, exp, r, k))
    if not parts:
        return repeat.iloc[0:0][LIST_COLUMNS].reset_index(drop=True)
    out = pd.concat(parts, ignore_index=True)
    out = out.sort_values(["user_id", "rank"], kind="mergesort")
    return out[LIST_COLUMNS].reset_index(drop=True)


def policy_name(threshold: int, r_short: int) -> str:
    """List name of the policy with ``threshold`` T and ``r_short`` repeat slots: "P_T" for the pre-registered pair (T, R_SHORT),
    else "P_T<threshold>_r<r_short>" (e.g. "P_T4_r8", "P_T6_r9")."""
    return "P_T" if (threshold, r_short) == (T, R_SHORT) else f"P_T{threshold}_r{r_short}"


# ---------------------------------------------------------------------------
# metrics
# ---------------------------------------------------------------------------
def new_hit_share(lists: pd.DataFrame, new_target: pd.DataFrame, users: pd.Index, k: int = K) -> pd.Series:
    """1.0 for every user of ``users`` whose top-``k`` rows of ``lists`` contain >= 1 of the user's ``new_target`` items, else 0.0.

    Users of ``users`` without a new target item (absent from ``new_target``) score 0.0; the mean over ``users`` is
    the new-item hit share.
    """
    recall = per_user_recall(lists, new_target, k)
    hits = (recall > 0).astype("float64")
    return hits.reindex(pd.Index(users), fill_value=0.0).rename("new_hit")


def list_series(lists: pd.DataFrame, data: dict[str, Any], all_users: pd.Index, k: int = K) -> dict[str, pd.Series]:
    """Per-user series of every METRICS entry for the final list ``lists`` against ``data`` (target, new_target, repeat_target)."""
    new_recall = per_user_recall(lists, data["new_target"], k)
    hits = (new_recall > 0).astype("float64").rename("new_hit")
    return {
        "ndcg": per_user_ndcg(lists, data["target"], k),
        "recall": per_user_recall(lists, data["target"], k),
        "new_recall": new_recall,
        "new_hit_share": hits.reindex(all_users, fill_value=0.0),
        "new_hit_share_new_users": hits,
        "repeat_recall": per_user_recall(lists, data["repeat_target"], k),
    }


# ---------------------------------------------------------------------------
# decisions and bootstrap entries
# ---------------------------------------------------------------------------
def improvement_decision(ci: list[float]) -> str:
    """PREREG wording for H12 / H13 from ``ci`` = [lo, hi] of (model - comparator): lower bound > 0 "improvement observed",
    upper bound < 0 "lower", otherwise "no difference established" (identical to nbr.pipeline.decision)."""
    lo, hi = ci
    if lo > 0:
        return "improvement observed"
    if hi < 0:
        return "lower"
    return "no difference established"


def noninferiority_decision(ci: list[float], delta: float = DELTA) -> str:
    """PREREG v1.4 wording for H11 from ``ci`` = [lo, hi] of (P_T - C3): "non-inferior" if lo > -``delta``, else
    "non-inferiority not established"."""
    return NON_INFERIOR if ci[0] > -delta else NOT_ESTABLISHED


def compare(a: pd.Series, b: pd.Series, config: dict[str, Any], levels: tuple[float, ...] = LEVELS,
            decision: Callable[[list[float]], str] | None = None) -> dict[str, Any]:
    """paired_bootstrap(comparator ``a``, model ``b``) with the bootstrap n and seed of ``config`` at ``levels``; the
    ``decision`` wording (one per level) is added when given."""
    boot = config["bootstrap"]
    result = paired_bootstrap(a, b, int(boot["n"]), int(boot["seed"]), list(levels))
    out = {
        "n_users": result["n_users"],
        "comparator_mean": result["mean_a"],
        "model_mean": result["mean_b"],
        "diff": result["diff"],
        "ci": {str(level): result["ci"][level] for level in levels},
    }
    if decision is not None:
        out["decision"] = {str(level): decision(result["ci"][level]) for level in levels}
    return out


def bin_labels(edges: list[int]) -> list[tuple[str, float, float]]:
    """(label, lo, hi] bins with upper ``edges`` and an open last bin, as nbr.pipeline._bin_labels: (0,4], ..., (20,inf)."""
    bounds = [0, *edges, math.inf]
    return [(f"({lo},inf)" if math.isinf(hi) else f"({lo},{hi}]", lo, hi) for lo, hi in zip(bounds[:-1], bounds[1:])]


def by_history_length(pairs: dict[str, tuple[pd.Series, pd.Series]], lengths: pd.Series, edges: list[int],
                      config: dict[str, Any]) -> list[dict[str, Any]]:
    """Per bin of ``edges``: user count and, per key of ``pairs`` (comparator, model), the diff with a 95 % CI (None when
    the bin has no scored user), in the layout of nbr.pipeline.by_history_length."""
    boot = config["bootstrap"]
    rows = []
    for name, lo, hi in bin_labels(edges):
        users = lengths[(lengths > lo) & (lengths <= hi)].index
        entry: dict[str, Any] = {"bin": name, "n_users": int(len(users))}
        for key, (a, b) in pairs.items():
            index = a.index.intersection(users)
            if len(index) == 0:
                entry[key] = None
                continue
            result = paired_bootstrap(a.loc[index], b.loc[index], int(boot["n"]), int(boot["seed"]), [0.95])
            entry[key] = {"n_users": result["n_users"], "comparator_mean": result["mean_a"], "model_mean": result["mean_b"],
                          "diff": result["diff"], "ci_0.95": result["ci"][0.95]}
        rows.append(entry)
    return rows


# ---------------------------------------------------------------------------
# evaluation block
# ---------------------------------------------------------------------------
def _entry(comparator: str, model: str, metric: str, a: pd.Series, b: pd.Series, config: dict[str, Any],
           decision: Callable[[list[float]], str] | None = None) -> dict[str, Any]:
    label, users = METRICS[metric]
    return {"comparator": comparator, "model": model, "metric": label, "users": users, **compare(a, b, config, LEVELS, decision)}


def evaluate_policy(
    lists: dict[str, pd.DataFrame],
    data: dict[str, Any],
    lengths: pd.Series,
    edges: list[int],
    config: dict[str, Any],
    r_default: int,
    *,
    label: str = "",
    T: int = T,
    delta: float = DELTA,
    r_short: int = R_SHORT,
) -> dict[str, Any]:
    """The "policy" block of the eval document (PREREG v1.4) from the eval-split ``lists`` of nbr.pipeline.build_lists.

    ``lists`` must hold repeat_lr (R2), usercf (E2), final_lr_ucf (C3 with ``r_default`` = its re-selected r) and
    flat_explore; ``data`` is the prepare() output of the split (target, new_target, repeat_target), ``lengths``
    its history_lengths, ``edges`` the history-length bins of the dataset and ``config`` supplies the bootstrap
    settings. P_T is built here with ``T`` / ``r_short`` (defaults: the pre-registered values); every comparison
    carries CIs at LEVELS; decisions only on the three primary entries. The variants (r_short = 8; T in {6, 10})
    and the secondary metrics are CI only.
    """
    timings: dict[str, float] = {}
    t0 = time.perf_counter()
    c3, repeat, explore, flat = (lists[LIST_SOURCES[name]] for name in ("C3", "repeat", "explore", "flat"))
    all_users = pd.Index(data["target"]["user_id"].drop_duplicates().sort_values())
    new_users = pd.Index(data["new_target"]["user_id"].drop_duplicates().sort_values())
    short = lengths.reindex(all_users)
    if short.isna().any():
        raise ValueError("every target user needs a history length")
    policy_list = combine_policy(repeat, explore, lengths, T, r_short, r_default)
    series = {"C3": list_series(c3, data, all_users), "P_T": list_series(policy_list, data, all_users)}
    explore_series = {
        "S": {"new_recall": per_user_recall(explore, data["new_target"], K),
              "new_hit_share_new_users": new_hit_share(explore, data["new_target"], new_users)},
        "flat": {"new_recall": per_user_recall(flat, data["new_target"], K),
                 "new_hit_share_new_users": new_hit_share(flat, data["new_target"], new_users)},
    }
    timings["lists_and_metrics"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    c3_name, s_name, flat_name = LIST_SOURCES["C3"], LIST_SOURCES["explore"], LIST_SOURCES["flat"]
    primary = {
        "H11_ndcg": {**_entry(c3_name, "P_T", "ndcg", series["C3"]["ndcg"], series["P_T"]["ndcg"], config,
                              lambda ci: noninferiority_decision(ci, delta)),
                     "rule": f"non-inferior if the {PRIMARY_LEVEL} lower bound > -{delta:g}", "delta": delta},
        "H12_new_hit_share": {**_entry(c3_name, "P_T", "new_hit_share", series["C3"]["new_hit_share"], series["P_T"]["new_hit_share"],
                                       config, improvement_decision),
                              "rule": f"improvement observed if the {PRIMARY_LEVEL} lower bound > 0"},
        "H13_new_hit_share": {**_entry(flat_name, s_name, "new_hit_share_new_users", explore_series["flat"]["new_hit_share_new_users"],
                                       explore_series["S"]["new_hit_share_new_users"], config, improvement_decision),
                              "rule": f"improvement observed if the {PRIMARY_LEVEL} lower bound > 0"},
    }
    secondary = {}
    for metric in SECONDARY_METRICS:
        secondary[f"P_T_vs_{c3_name}_{metric}"] = _entry(c3_name, "P_T", metric, series["C3"][metric], series["P_T"][metric], config)
    secondary[f"{s_name}_vs_{flat_name}_new_recall"] = _entry(flat_name, s_name, "new_recall", explore_series["flat"]["new_recall"],
                                                            explore_series["S"]["new_recall"], config)
    timings["primary_and_secondary"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    pairs = {
        "H11_ndcg": (series["C3"]["ndcg"], series["P_T"]["ndcg"]),
        "H12_new_hit_share": (series["C3"]["new_hit_share"], series["P_T"]["new_hit_share"]),
        "H13_new_hit_share": (explore_series["flat"]["new_hit_share_new_users"], explore_series["S"]["new_hit_share_new_users"]),
    }
    for metric in SECONDARY_METRICS:
        pairs[f"P_T_vs_{c3_name}_{metric}"] = (series["C3"][metric], series["P_T"][metric])
    breakdown = by_history_length(pairs, lengths, edges, config)
    timings["by_history_length"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    variants = {}
    variant_metrics: dict[str, dict[str, float]] = {}
    for T_v, r_v in [(T, R_SHORT_VARIANT), *[(t, r_short) for t in T_SENSITIVITY]]:
        name = policy_name(T_v, r_v)
        variant_list = combine_policy(repeat, explore, lengths, T_v, r_v, r_default)
        variant_series = list_series(variant_list, data, all_users)
        variants[name] = {
            "T": T_v, "r_short": r_v, "r_default": r_default,
            "n_short_users": int((short <= T_v).sum()),
            "comparisons": {metric: _entry(c3_name, name, metric, series["C3"][metric], variant_series[metric], config)
                            for metric in VARIANT_METRICS},
        }
        variant_metrics[name] = {metric: float(values.mean()) for metric, values in variant_series.items()}
    timings["variants"] = time.perf_counter() - t0

    list_metrics = {name: {metric: float(values.mean()) for metric, values in block.items()}
                    for name, block in ((c3_name, series["C3"]), ("P_T", series["P_T"]))}
    list_metrics[s_name] = {metric: float(values.mean()) for metric, values in explore_series["S"].items()}
    list_metrics[flat_name] = {metric: float(values.mean()) for metric, values in explore_series["flat"].items()}
    list_metrics.update(variant_metrics)
    return {
        "label": label,
        "prereg": PREREG,
        "T": T,
        "delta": delta,
        "r_short": r_short,
        "r_default": int(r_default),
        "k": K,
        "levels": [str(level) for level in LEVELS],
        "primary_level": PRIMARY_LEVEL,
        "lists": {"P_T": f"combine_policy({LIST_SOURCES['repeat']}, {LIST_SOURCES['explore']}, T={T}, r_short={r_short}, r_default={int(r_default)})",
                  "C3": c3_name, "S": s_name, "flat": flat_name},
        "history_length": "distinct history baskets of the evaluated split (nbr.pipeline.history_lengths)",
        "n_users": {
            "all": int(len(all_users)),
            "new_target": int(len(new_users)),
            "repeat_target": int(data["repeat_target"]["user_id"].nunique()),
            "short_history": int((short <= T).sum()),
            "short_history_by_T": {str(t): int((short <= t).sum()) for t in sorted({T, *T_SENSITIVITY})},
        },
        "list_metrics": list_metrics,
        "primary": primary,
        "secondary": secondary,
        "by_history_length": breakdown,
        "variants": variants,
        "timing_seconds": timings,
    }
