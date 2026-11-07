"""Two-CF final system of PREREG v1.5 (W19): the eval block "v15" with H14, H15 and the secondary comparisons.

Definitions (docs/PREREG.md v1.5, items 4-8):

- R2' = ``repeat_lri``: the LR repeat ranker R2 (``repeat_lr``) with the item-CF similarity feature icf of
  nbr.features_icf appended (13 features; neighbours computed on the history being scored; 20 item neighbours,
  fixed). Its C is selected separately from R2 on the same validation users (nbr.pipeline.tune).
- C4 = ``final_lri_ucf`` = combine(R2', E2 = ``usercf``, r) with r re-selected on the fit split the same way as
  for C3 = ``final_lr_ucf`` (train-only R2' bundle, LR validation users, NDCG@10). C3 (PREREG v1.3) is the comparator.
- Primary (Instacart holdout4, family of 2, two-sided 97.5 % CI = 1 - 0.05/2; the 95 % CI is reported alongside):
  H14 (non-inferiority) NDCG@10 of C4 minus C3 over all users, "non-inferior" if the 97.5 % lower bound > -DELTA
  (DELTA = 0.003, the v1.4 margin), else "non-inferiority not established"; H15 NDCG@10 of C4 minus personal
  frequency over all users, "improvement observed" if the lower bound > 0, "lower" if the upper bound < 0, else
  "no difference established".
- Secondary (no confirmatory claim; CI only): repeat Recall@10 of R2' vs R2 and of R2' vs R1 (users with >= 1 repeat
  target item), Recall@10 of C4 vs C3, the standardized coefficients and intercept of R2', history-length bins of H14.

The functions never read files or targets other than the frames they receive; the lists come from
nbr.pipeline.build_lists. The CI levels are constants of this module (not the profile's bootstrap levels).
"""

from __future__ import annotations

import time
from typing import Any

import pandas as pd

from nbr.metrics import per_user_ndcg, per_user_recall
from nbr.policy import by_history_length, compare, improvement_decision, noninferiority_decision

K = 10
DELTA = 0.003                           # non-inferiority margin of H14 (NDCG@10), PREREG v1.5 item 6 (= the v1.4 margin)
LEVELS = (0.95, 0.975)                  # CI levels: 95 % and 1 - 0.05/2 (family of 2)
PRIMARY_LEVEL = "0.975"
PREREG = "v1.5"
LIST_SOURCES = {"R1": "repeat_star", "R2": "repeat_lr", "R2'": "repeat_lri", "E2": "usercf",
                "C3": "final_lr_ucf", "C4": "final_lri_ucf", "PF": "personal_frequency"}
METRICS = {
    "ndcg": ("NDCG@10", "all users"),
    "recall": ("Recall@10", "all users"),
    "repeat_recall": ("repeat Recall@10", "users with repeat target"),
}
ICF_DEFINITION = ("icf(u, i) = log(1 + sum over j in I(H_u) of sim(j, i)), sim from the top-n cosine item-neighbour lists "
                  "of the binary user x item matrix of the scored history (self excluded)")


def _entry(comparator: str, model: str, metric: str, a: pd.Series, b: pd.Series, config: dict[str, Any],
           decision=None) -> dict[str, Any]:
    label, users = METRICS[metric]
    return {"comparator": comparator, "model": model, "metric": label, "users": users, **compare(a, b, config, LEVELS, decision)}


def bundle_summary(icf_bundle: dict[str, Any]) -> dict[str, Any]:
    """R2' description for the block: artifact, sha256, C, icf_neighbors, feature names, standardized coefficients, intercept."""
    return {
        "artifact": icf_bundle.get("artifact"),
        "sha256": icf_bundle.get("sha256"),
        "C": icf_bundle["C"],
        "icf_neighbors": int(icf_bundle["icf_neighbors"]),
        "features": list(icf_bundle["features"]),
        "coefficients": {name: float(value) for name, value in icf_bundle["coefficients"].items()},
        "intercept": float(icf_bundle["intercept"]),
    }


def evaluate_v15(
    lists: dict[str, pd.DataFrame],
    data: dict[str, Any],
    lengths: pd.Series,
    edges: list[int],
    config: dict[str, Any],
    icf_bundle: dict[str, Any],
    selections: dict[str, Any],
    *,
    label: str = "",
    delta: float = DELTA,
) -> dict[str, Any]:
    """The "v15" block of the eval document from the eval-split ``lists`` of nbr.pipeline.build_lists.

    ``lists`` must hold repeat_star (R1), repeat_lr (R2), repeat_lri (R2'), usercf (E2), final_lr_ucf (C3),
    final_lri_ucf (C4) and personal_frequency; ``data`` is the prepare() output of the split (target,
    repeat_target, new_target), ``lengths`` its history_lengths, ``edges`` the history-length bins of the
    dataset, ``config`` supplies the bootstrap n and seed, ``icf_bundle`` the loaded R2' bundle and
    ``selections`` the v15 tuning selections (C of R2', r of C4). Every comparison carries CIs at LEVELS;
    decisions only on the two primary entries.
    """
    timings: dict[str, float] = {}
    t0 = time.perf_counter()
    names = LIST_SOURCES
    for key in ("R1", "R2", "R2'", "E2", "C3", "C4", "PF"):
        if names[key] not in lists:
            raise ValueError(f"evaluate_v15: the list {names[key]} ({key}) is missing")
    target, repeat_target = data["target"], data["repeat_target"]
    ndcg = {key: per_user_ndcg(lists[names[key]], target, K) for key in ("C3", "C4", "PF")}
    recall = {key: per_user_recall(lists[names[key]], target, K) for key in ("C3", "C4")}
    repeat_recall = {key: per_user_recall(lists[names[key]], repeat_target, K) for key in ("R1", "R2", "R2'")}
    timings["metrics"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    primary = {
        "H14_ndcg": {**_entry(names["C3"], names["C4"], "ndcg", ndcg["C3"], ndcg["C4"], config,
                              lambda ci: noninferiority_decision(ci, delta)),
                     "rule": f"non-inferior if the {PRIMARY_LEVEL} lower bound > -{delta:g}", "delta": delta},
        "H15_ndcg": {**_entry(names["PF"], names["C4"], "ndcg", ndcg["PF"], ndcg["C4"], config, improvement_decision),
                     "rule": f"improvement observed if the {PRIMARY_LEVEL} lower bound > 0"},
    }
    r1, r2, r2p, c3, c4 = names["R1"], names["R2"], names["R2'"], names["C3"], names["C4"]
    secondary = {
        f"{r2p}_vs_{r2}_repeat_recall": _entry(r2, r2p, "repeat_recall", repeat_recall["R2"], repeat_recall["R2'"], config),
        f"{r2p}_vs_{r1}_repeat_recall": _entry(r1, r2p, "repeat_recall", repeat_recall["R1"], repeat_recall["R2'"], config),
        f"{c4}_vs_{c3}_recall": _entry(c3, c4, "recall", recall["C3"], recall["C4"], config),
    }
    timings["primary_and_secondary"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    breakdown = by_history_length({"H14_ndcg": (ndcg["C3"], ndcg["C4"])}, lengths, edges, config)
    timings["by_history_length"] = time.perf_counter() - t0

    all_users = int(target["user_id"].nunique())
    list_metrics = {names[key]: {"ndcg": float(ndcg[key].mean())} for key in ("C3", "C4", "PF")}
    for key in ("C3", "C4"):
        list_metrics[names[key]]["recall"] = float(recall[key].mean())
    for key in ("R1", "R2", "R2'"):
        list_metrics[names[key]] = {"repeat_recall": float(repeat_recall[key].mean())}
    return {
        "label": label,
        "prereg": PREREG,
        "delta": delta,
        "k": K,
        "levels": [str(level) for level in LEVELS],
        "primary_level": PRIMARY_LEVEL,
        "lists": dict(names),
        "icf": {"neighbors": int(icf_bundle["icf_neighbors"]), "definition": ICF_DEFINITION},
        "selections": {key: value for key, value in selections.items() if key != "source"},
        "selections_source": selections.get("source"),
        "lr_repeat_icf": bundle_summary(icf_bundle),
        "n_users": {
            "all": all_users,
            "repeat_target": int(repeat_target["user_id"].nunique()),
            "new_target": int(data["new_target"]["user_id"].nunique()),
        },
        "list_metrics": list_metrics,
        "primary": primary,
        "secondary": secondary,
        "by_history_length": breakdown,
        "timing_seconds": timings,
    }
