# *------------------- 2026 Fall Semester Machine Learning Term-project: Team 9 --------------------*
# Code First written at 2026-11-07
# Code Final Edited at 2026-11-13
# Editor: Yejun Kim

'''
Purpose : Development check of the PREREG v1.4 exploration-policy code on the medium profile
          (W17; not confirmatory). This is a code check on development data, not a selection:
          nothing is changed on its basis.
          Codes: C3 is the combination of the logistic-regression repeat ranker (R2) and the
          user-based CF explore module (E2); P_T is C3 with 9 repeat slots for users with at
          most T history baskets.
          Checks (all must pass; the script exits 1 otherwise):
          1. Selections of the tuning documents equal the reference (gamma, gamma_c, alpha, r,
             LR C, user-CF k and weighting, r per combination).
          2. The LR artifact file has the sha256 recorded in the tuning document, and the eval
             document used that artifact (lr_repeat.sha256).
          3. C3 (final_lr_ucf) NDCG@10 of the eval document rounds to the W8b/W15 value
             (Instacart 0.4274, Complete Journey 0.2046).
          4. Agreement with the W15 part C numbers (R2 + E2, r = 9 or 8 against r = 10): the
             policy's P_T-vs-C3 differences and the variants equal the W15 history-length-bin
             differences weighted by the bin user counts (|gap| <= 1e-9), and the policy's
             first-bin entries equal the W15 first-bin entries (diff and 95 % CI). A comparison
             is checkable only when its T is an upper edge of the dataset's history-length bins
             (Instacart: 4, 6, 10; the Complete Journey bins 20, 50, 100 make the W15 numbers
             non-derivable for T in {4, 6, 10}, recorded as not applicable); the C3 list
             metrics are checked for both.
Inputs  : This repository, read-only: reports/metrics/<profile>_<dataset>_tuning.json and the
          eval documents written by
          ``python -m nbr evaluate --profile <profile> --dataset <dataset> --skip-l3 --policy``
          (reports/metrics/<profile>_<dataset>_eval.json), the LR artifacts
          artifacts/<profile>/<dataset>_lr_repeat.joblib, the reference tuning documents of the
          W8b run (--reference-tuning-dir) and the W15 post-hoc JSON (--w15-json).
          Only the development profiles ``sample`` and ``medium`` are accepted; no path with a
          "final" or "holdout" component is read.
Outputs : <out-dir>/policy_dev_<profile>.json (checks plus the full "policy" blocks of the eval
          documents) and policy_dev_<profile>.md (tables).
Usage   :
    python scripts/check_policy_dev.py [--profile medium] [--datasets instacart,complete_journey]
                                       [--reference-tuning-dir PATH]
                                       [--w15-json reports/posthoc/explore_slot_medium.json]
                                       [--out-dir reports/posthoc]
'''

from __future__ import annotations

# ---------------------------------------------------------------------------
# imports
# ---------------------------------------------------------------------------
import argparse
import hashlib
import json
import math
import platform
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from nbr.io import write_json  # noqa: E402
from nbr.policy import DELTA, LEVELS, PRIMARY_LEVEL, R_SHORT, R_SHORT_VARIANT, T, T_SENSITIVITY  # noqa: E402

# ---------------------------------------------------------------------------
# constants
# ---------------------------------------------------------------------------
LABEL = "development check of the PREREG v1.4 code; not confirmatory"
ALLOWED_PROFILES = ("sample", "medium")
DATASETS = ("instacart", "complete_journey")
NAMES = {"instacart": "Instacart", "complete_journey": "Complete Journey"}
TOLERANCE = 1e-9
# C3 NDCG@10 at four decimals (W8b medium eval, reproduced by W15).
EXPECTED_C3_NDCG = {("medium", "instacart"): 0.4274, ("medium", "complete_journey"): 0.2046}
# Approximate overall effects expected from W15 on Instacart medium (task W17 P1.2), for the report only.
EXPECTED_APPROX = {"instacart": {"ndcg": -0.0013, "new_hit_share": 0.010}}
SELECTION_PATHS = {
    "gamma": ("selections", "gamma"), "gamma_c": ("selections", "gamma_c"), "alpha": ("selections", "alpha"), "r": ("selections", "r"),
    "lr_C": ("lr_repeat", "C"), "usercf_k": ("integration", "selections", "k"), "usercf_weighting": ("integration", "selections", "weighting"),
    "r_final_lr_r": ("integration", "selections", "r", "final_lr_r"), "r_final_lr_ucf": ("integration", "selections", "r", "final_lr_ucf"),
    "r_final_star_ucf": ("integration", "selections", "r", "final_star_ucf"),
}
# policy metric -> W15 part C metric name (nbr.policy.list_series vs posthoc_explore_slot.list_metrics).
W15_METRICS = {"ndcg": "ndcg", "recall": "recall", "new_recall": "new_recall", "new_hit_share": "new_hit_share", "repeat_recall": "repeat_recall"}
PRIMARY_KEYS = ("H11_ndcg", "H12_new_hit_share", "H13_new_hit_share")


# ---------------------------------------------------------------------------
# command-line arguments
# ---------------------------------------------------------------------------
def parse_datasets(value: str) -> list[str]:
    '''Dataset names of the comma-separated ``value``; raises ValueError for an unknown name.'''
    # Split at commas and drop empty parts.
    names = [item.strip() for item in value.split(",") if item.strip()]
    # Remove repeated names and keep the first occurrence of each.
    names = list(dict.fromkeys(names))
    # Reject an empty list and any name outside DATASETS.
    if not names or any(name not in DATASETS for name in names):
        raise ValueError(f"datasets must be a non-empty subset of {list(DATASETS)}, got {value!r}")
    return names


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    '''Parsed command-line options of ``argv`` (default: the arguments of the process).'''
    # Declare the options of the script.
    parser = argparse.ArgumentParser(description="Development check of the PREREG v1.4 policy code (development profiles only).")
    parser.add_argument("--profile", default="medium", choices=ALLOWED_PROFILES)
    parser.add_argument("--datasets", default=",".join(DATASETS), type=parse_datasets)
    parser.add_argument("--reference-tuning-dir", required=True, help="folder with the reference <profile>_<dataset>_tuning.json files (W8b)")
    parser.add_argument("--w15-json", default=str(ROOT / "reports" / "posthoc" / "explore_slot_medium.json"))
    parser.add_argument("--metrics-dir", default=str(ROOT / "reports" / "metrics"))
    parser.add_argument("--out-dir", default=str(ROOT / "reports" / "posthoc"))
    # Parse the given arguments.
    return parser.parse_args(argv)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def guard_path(path: Path) -> Path:
    """Refuse any path with a component starting with 'final' or 'holdout' (confirmatory inputs are never read)."""
    # Inspect every component of the resolved path.
    for part in path.resolve().parts:
        if part.lower().startswith(("final", "holdout")):
            raise PermissionError(f"refusing to read {path}: confirmatory inputs are never read by this script")
    return path


def sha256_of(path: Path) -> str:
    '''Hex sha256 digest of the file at ``path``.'''
    digest = hashlib.sha256()
    # Feed the file to the digest in chunks of 1 MiB.
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _get(document: dict[str, Any], path: tuple[str, ...]) -> Any:
    '''Value of ``document`` under the nested keys of ``path``, or None when a key is missing.'''
    value: Any = document
    for key in path:
        # Stop with None when the current value is not a dict or lacks the key.
        if not isinstance(value, dict) or key not in value:
            return None
        # Step one level down.
        value = value[key]
    return value


def _equal(a: Any, b: Any) -> bool:
    '''Whether ``a`` equals ``b``; two numbers are equal when they differ by at most TOLERANCE.'''
    # Treat a bool and a non-bool as different.
    if isinstance(a, bool) != isinstance(b, bool):
        return False
    # Compare two numbers with the absolute tolerance.
    if isinstance(a, (int, float)) and isinstance(b, (int, float)) and not isinstance(a, bool):
        return math.isclose(float(a), float(b), rel_tol=0.0, abs_tol=TOLERANCE)
    # Compare every other pair of values exactly.
    return a == b


# ---------------------------------------------------------------------------
# setup and reproduction checks
# ---------------------------------------------------------------------------
def compare_selections(run: dict[str, Any], reference: dict[str, Any]) -> dict[str, Any]:
    """Every SELECTION_PATHS value of the ``run`` tuning document against ``reference``; ``all_equal`` summarises."""
    # Build one row per selected quantity with both values and the equality flag.
    rows = [{"quantity": name, "run": _get(run, path), "reference": _get(reference, path),
             "equal": _equal(_get(run, path), _get(reference, path))} for name, path in SELECTION_PATHS.items()]
    # Return the rows, the summary flag and the commit and time of both documents.
    return {"selections": rows, "all_equal": all(row["equal"] for row in rows),
            "run_meta": {k: run.get("meta", {}).get(k) for k in ("git_commit", "utc_time")},
            "reference_meta": {k: reference.get("meta", {}).get(k) for k in ("git_commit", "utc_time")}}


def check_artifact(tuning: dict[str, Any], evaluation: dict[str, Any], artifact: Path) -> dict[str, Any]:
    """The artifact file carries the sha256 of the tuning document and the eval document used the same digest."""
    # Read the digest recorded in the tuning document.
    recorded = _get(tuning, ("lr_repeat", "sha256"))
    # Hash the artifact file when it exists.
    observed = sha256_of(artifact) if artifact.exists() else None
    # Read the digest recorded in the eval document.
    used = _get(evaluation, ("lr_repeat", "sha256"))
    # Report the three digests and whether the file exists and all three agree.
    return {"artifact": str(artifact), "exists": artifact.exists(), "sha256_recorded": recorded, "sha256_observed": observed,
            "sha256_eval": used, "eval_artifact": _get(evaluation, ("lr_repeat", "artifact")),
            "matches": artifact.exists() and observed == recorded == used}


def check_c3(evaluation: dict[str, Any], profile: str, dataset: str) -> dict[str, Any]:
    '''
    C3 NDCG@10 of the ``evaluation`` document against the expected four-decimal value of
    (``profile``, ``dataset``) and against the comparator mean of the policy block.
    '''
    # Look up the expected value, which is None for a profile without one.
    expected = EXPECTED_C3_NDCG.get((profile, dataset))
    # Read the C3 NDCG@10 from the list metrics.
    observed = float(evaluation["list_metrics"]["final_lr_ucf"]["ndcg"])
    # Read the comparator mean of the first primary entry of the policy block.
    policy_c3 = float(evaluation["policy"]["primary"]["H11_ndcg"]["comparator_mean"])
    # Report whether the rounded value matches and whether both sources agree.
    return {"expected": expected, "observed": observed, "policy_comparator_mean": policy_c3,
            "matched": expected is None or abs(round(observed, 4) - expected) < 1e-9,
            "policy_equals_list_metric": abs(observed - policy_c3) <= TOLERANCE}


# ---------------------------------------------------------------------------
# agreement with the W15 numbers
# ---------------------------------------------------------------------------
def w15_bins(w15: dict[str, Any], dataset: str, r: int) -> dict[str, dict[str, Any]]:
    """{metric: {bin: entry}} of the W15 part C differences of R2, r = ``r`` vs r = 10 (entries carry n_users, diff, ci_0.95)."""
    entry = w15["datasets"][dataset]["eval"]["slots"]
    # Select the difference rows of R2 with the requested number of repeat slots.
    rows = [row for row in entry["differences"] if row["module"] == "R2" and int(row["r"]) == r]
    # Fail unless exactly one row matches.
    if len(rows) != 1:
        raise ValueError(f"W15 differences for R2 r = {r} on {dataset}: {len(rows)} rows")
    # Index the history-length-bin entries of every metric by the bin label.
    return {metric: {bin_row["bin"]: bin_row for bin_row in values["by_history_length"]} for metric, values in rows[0]["metrics"].items()}


def expected_from_w15(bins: dict[str, dict[str, Any]], metric: str, short_bins: list[str], n_total: int) -> float:
    """Overall P_T-vs-C3 difference implied by W15: sum over the short bins of diff x n_users / ``n_total`` (0 for empty bins)."""
    total = 0.0
    for name in short_bins:
        row = bins[metric].get(name)
        # Add the bin difference weighted by the bin user count when the bin has users.
        if row and row.get("n_users"):
            total += float(row["diff"]) * int(row["n_users"])
    # Divide the weighted sum by the total number of users.
    return total / n_total


def bin_edges(labels: list[str]) -> list[float]:
    """Finite upper edges of the bin ``labels`` ("(0,4]" -> 4.0; the open last bin has none)."""
    edges = []
    for label in labels:
        # Take the text after the comma without the closing bracket.
        upper = label.rstrip("])").split(",")[1]
        # Keep the edge unless the bin is open.
        if upper != "inf":
            edges.append(float(upper))
    return edges


def short_bins(threshold: int, labels: list[str]) -> list[str]:
    """History-length bins whose upper edge is <= ``threshold`` (users with history_len <= threshold), e.g. T = 6 -> (0,4], (4,6]."""
    return [label for label, upper in zip(labels, [*bin_edges(labels), math.inf]) if upper <= threshold]


def checkable(threshold: int, labels: list[str]) -> bool:
    """The users with history_len <= ``threshold`` form a union of bins exactly when ``threshold`` is a finite bin edge."""
    return float(threshold) in bin_edges(labels)


def check_w15_agreement(policy: dict[str, Any], w15: dict[str, Any], dataset: str) -> dict[str, Any]:
    """Compare the policy block with the W15 part C numbers (same lists, same users, same bootstrap seed)."""
    # Read the bin labels of the policy block.
    labels = [row["bin"] for row in policy["by_history_length"]]
    # Read the W15 bin differences for 9 and for 8 repeat slots.
    nine, eight = w15_bins(w15, dataset, 9), w15_bins(w15, dataset, 8)
    rows = []

    def add(name: str, metric: str, bins: dict[str, dict[str, Any]], threshold: int, entry: dict[str, Any]) -> None:
        '''
        Append to ``rows`` the comparison ``name`` of ``metric``: the difference of the policy
        ``entry`` against the difference implied by the W15 ``bins`` for ``threshold``.
        '''
        # Record the comparison as not applicable when the threshold is not a bin edge.
        if not checkable(threshold, labels):
            rows.append({"comparison": name, "metric": metric, "policy_diff": float(entry["diff"]), "w15_implied_diff": None, "gap": None,
                         "n_users": int(entry["n_users"]), "agree": None,
                         "note": f"not applicable: T = {threshold} is not a bin edge of this dataset ({', '.join(labels)})"})
            return
        # Compute the difference implied by the W15 bins at or below the threshold.
        expected = expected_from_w15(bins, W15_METRICS[metric], short_bins(threshold, labels), int(entry["n_users"]))
        # Record both differences, their gap and whether the gap is within the tolerance.
        rows.append({"comparison": name, "metric": metric, "policy_diff": float(entry["diff"]), "w15_implied_diff": expected,
                     "gap": abs(float(entry["diff"]) - expected), "n_users": int(entry["n_users"]),
                     "agree": abs(float(entry["diff"]) - expected) <= TOLERANCE})

    # Compare the two primary P_T-vs-C3 differences with the W15 numbers for 9 repeat slots.
    add("P_T vs C3 (H11)", "ndcg", nine, T, policy["primary"]["H11_ndcg"])
    add("P_T vs C3 (H12)", "new_hit_share", nine, T, policy["primary"]["H12_new_hit_share"])
    # Compare the secondary P_T-vs-C3 differences.
    for metric in ("recall", "new_recall", "repeat_recall"):
        add("P_T vs C3", metric, nine, T, policy["secondary"][f"P_T_vs_final_lr_ucf_{metric}"])
    # Compare every variant with the W15 numbers of its number of repeat slots and its threshold.
    for name, variant in policy["variants"].items():
        bins = eight if int(variant["r_short"]) == R_SHORT_VARIANT else nine
        for metric in ("ndcg", "new_hit_share", "recall", "new_recall", "repeat_recall"):
            add(f"{name} vs C3", metric, bins, int(variant["T"]), variant["comparisons"][metric])
    # The first bin of the policy block against the W15 first bin (same users, same bootstrap draw) when its edge is T.
    bin_rows = []
    first = labels[0]
    first_applicable = bin_edges(labels)[:1] == [float(policy["T"])]
    policy_first = next(row for row in policy["by_history_length"] if row["bin"] == first)
    for key, metric in (("H11_ndcg", "ndcg"), ("H12_new_hit_share", "new_hit_share"), ("P_T_vs_final_lr_ucf_recall", "recall"),
                        ("P_T_vs_final_lr_ucf_new_recall", "new_recall"), ("P_T_vs_final_lr_ucf_repeat_recall", "repeat_recall")):
        # Record the entry as not applicable when the first bin edge is not T.
        if not first_applicable:
            bin_rows.append({"bin": first, "key": key, "agree": None, "note": f"not applicable: the first bin edge is not T = {policy['T']}"})
            continue
        ours, theirs = policy_first.get(key), nine[metric].get(first)
        # Without users on one side, the entries agree only when both sides have none.
        if ours is None or theirs is None or not theirs.get("n_users"):
            bin_rows.append({"bin": first, "key": key, "agree": ours is None and (theirs is None or not theirs.get("n_users"))})
            continue
        # Take the largest gap over the difference and the two bounds of the 95 % CI.
        gap = max(abs(float(ours["diff"]) - float(theirs["diff"])), abs(ours["ci_0.95"][0] - theirs["ci_0.95"][0]),
                  abs(ours["ci_0.95"][1] - theirs["ci_0.95"][1]))
        # Record both entries and whether the user counts are equal and the gap is within tolerance.
        bin_rows.append({"bin": first, "key": key, "policy": {"n_users": ours["n_users"], "diff": ours["diff"], "ci_0.95": ours["ci_0.95"]},
                         "w15": {"n_users": theirs["n_users"], "diff": theirs["diff"], "ci_0.95": theirs["ci_0.95"]}, "gap": gap,
                         "agree": ours["n_users"] == theirs["n_users"] and gap <= TOLERANCE})
    # C3 list metrics against the W15 r = 10 row of R2.
    table = {(row["module"], int(row["r"])): row for row in w15["datasets"][dataset]["eval"]["slots"]["table"]}
    c3_rows = []
    for metric, w15_metric in W15_METRICS.items():
        ours, theirs = float(policy["list_metrics"]["final_lr_ucf"][metric]), float(table[("R2", 10)][w15_metric])
        c3_rows.append({"metric": metric, "policy_C3": ours, "w15_R2_r10": theirs, "agree": abs(ours - theirs) <= TOLERANCE})
    # Keep the comparisons that could be checked.
    applicable = [r for r in rows + bin_rows if r["agree"] is not None]
    # Return all rows, the counts and the overall agreement flag.
    return {"overall": rows, "first_bin": bin_rows, "c3_list_metrics": c3_rows,
            "n_applicable": len(applicable), "n_not_applicable": len(rows) + len(bin_rows) - len(applicable),
            "all_agree": all(r["agree"] for r in applicable) and all(r["agree"] for r in c3_rows),
            "w15_meta": {k: w15.get("meta", {}).get(k) for k in ("git_commit", "utc_time")}}


# ---------------------------------------------------------------------------
# markdown
# ---------------------------------------------------------------------------
def f4(value: Any) -> str:
    '''Text of ``value`` with four decimals, "-" for None.'''
    return "-" if value is None else f"{float(value):.4f}"


def fd(value: Any) -> str:
    '''Text of ``value`` with a sign and four decimals, "-" for None.'''
    return "-" if value is None else f"{float(value):+.4f}"


def ci(values: Any) -> str:
    '''Text of the interval ``values`` (lower, upper) with signs and four decimals, "-" if empty.'''
    return "-" if not values else f"[{values[0]:+.4f}, {values[1]:+.4f}]"


def entry_row(name: str, entry: dict[str, Any]) -> str:
    '''Markdown table row of the comparison ``entry`` labelled ``name``.'''
    decision = entry.get("decision", {})
    # Format the user count, the lists, the means, the difference, both CIs and the decisions.
    cells = [name, f"{entry['n_users']:,}", f"{entry['comparator']} -> {entry['model']}", f4(entry["comparator_mean"]), f4(entry["model_mean"]),
             fd(entry["diff"]), ci(entry["ci"]["0.95"]), ci(entry["ci"][PRIMARY_LEVEL]),
             f"{decision['0.95']} / {decision[PRIMARY_LEVEL]}" if decision else "(CI only)"]
    # Join the cells into one table row.
    return "| " + " | ".join(cells) + " |"


def render_markdown(report: dict[str, Any]) -> str:
    '''Markdown text of the development-check ``report`` (one section per dataset, then inputs).'''
    profile = report["profile"]
    # Write the title, the description of the check and the run facts.
    lines = [f"# Exploration policy (PREREG v1.4): development check on {profile}", "",
             f"**{LABEL}.** The policy code of W17 (nbr.policy, `python -m nbr evaluate --skip-l3 --policy`) is run on the development "
             f"profile with the pre-registered settings T = {T}, delta = {DELTA:g}, r_short = {R_SHORT} (variants r_short = {R_SHORT_VARIANT} and "
             f"T in {{{', '.join(str(t) for t in T_SENSITIVITY)}}}) and compared with the W8b selections and the W15 post-hoc numbers. "
             "This is a code check, not a selection: nothing is changed on its basis. The holdout3 run (reports/final3) is the confirmatory test.",
             "",
             f"- Code commit `{report['meta']['git_commit']}`, {report['meta']['utc_time']}; profile `{profile}`; "
             f"CI levels {', '.join(report['levels'])} (primary level {PRIMARY_LEVEL} = 1 - 0.05/3); paired bootstrap n = 1000, seed 42.",
             f"- Tuning documents reused from the W15 worktree (python -m nbr tune, commit `{report['tuning_commit']}`), verified against the W8b "
             f"reference `{report['reference_tuning_dir']}`; W15 JSON commit `{report['w15_commit']}`.",
             f"- Expected from W15 (Instacart, approximate): overall delta NDCG@10 about {EXPECTED_APPROX['instacart']['ndcg']:+.4f}, delta new-hit share "
             f"about {EXPECTED_APPROX['instacart']['new_hit_share']:+.3f}.",
             f"- All checks passed: **{'yes' if report['passed'] else 'NO'}**.", ""]
    for dataset in report["datasets"]:
        d = report["checks"][dataset]
        policy = report["policy"][dataset]
        # Write the setup table: one row per selection and one row for the LR artifact.
        lines += [f"## {NAMES[dataset]}", "", "### Setup check (tuning selections vs the W8b reference; LR artifact)", "",
                  "| quantity | this run | reference | equal |", "| --- | --- | --- | --- |"]
        for row in d["selections"]["selections"]:
            lines.append(f"| {row['quantity']} | {row['run']} | {row['reference']} | {'yes' if row['equal'] else 'NO'} |")
        art = d["artifact"]
        lines.append(f"| LR artifact sha256 (tuning / file / eval) | {str(art['sha256_recorded'])[:12]} | {str(art['sha256_observed'])[:12]} / "
                     f"{str(art['sha256_eval'])[:12]} | {'yes' if art['matches'] else 'NO'} |")
        # Write the reproduction table of the C3 NDCG@10.
        c3 = d["c3"]
        lines += ["", "### Reproduction check (C3 = final_lr_ucf NDCG@10, four decimals)", "",
                  "| expected | observed | policy comparator mean | matched |", "| --- | --- | --- | --- |",
                  f"| {c3['expected']} | {c3['observed']:.6f} | {c3['policy_comparator_mean']:.6f} | "
                  f"{'yes' if c3['matched'] and c3['policy_equals_list_metric'] else 'NO'} |"]
        # Write the heading with the user counts and the table of the primary entries.
        n = policy["n_users"]
        lines += ["", f"### Policy block ({n['all']:,} users; {n['new_target']:,} with a new target item, {n['repeat_target']:,} with a repeat "
                  f"target item; {n['short_history']:,} with history_len <= {policy['T']}; by T: "
                  + ", ".join(f"T = {t}: {v:,}" for t, v in n["short_history_by_T"].items()) + f"; r* = {policy['r_default']})", "",
                  "Primary (decision at 95 % / 98.33 %; H11 rule: non-inferior if the 98.33 % lower bound > -0.003):", "",
                  "| hypothesis | n | comparator -> model | comparator mean | model mean | diff | 95 % CI | 98.33 % CI | decision |",
                  "| --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
        for key in PRIMARY_KEYS:
            lines.append(entry_row(f"{key} ({policy['primary'][key]['metric']}, {policy['primary'][key]['users']})", policy["primary"][key]))
        # Write the table of the secondary comparisons and of the variants.
        lines += ["", "Secondary (CI only):", "",
                  "| comparison | n | comparator -> model | comparator mean | model mean | diff | 95 % CI | 98.33 % CI | decision |",
                  "| --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
        for key, entry in policy["secondary"].items():
            lines.append(entry_row(f"{key} ({entry['metric']}, {entry['users']})", entry))
        for name, variant in policy["variants"].items():
            for metric, entry in variant["comparisons"].items():
                lines.append(entry_row(f"{name} (T = {variant['T']}, r_short = {variant['r_short']}; {variant['n_short_users']:,} short users): "
                                       f"{entry['metric']}, {entry['users']}", entry))
        # Write the table by history-length bin, one cell per comparison.
        lines += ["", "By history-length bin (P_T vs C3 and S vs flat; diff [95 % CI]; bins above T are identical lists):", "",
                  "| bin | users | H11 NDCG@10 | H12 new-hit share | H13 S vs flat new-hit share | Recall@10 | new-item Recall@10 | repeat Recall@10 |",
                  "| --- | --- | --- | --- | --- | --- | --- | --- |"]
        for row in policy["by_history_length"]:
            cells = [row["bin"], f"{row['n_users']:,}"]
            for key in ("H11_ndcg", "H12_new_hit_share", "H13_new_hit_share", "P_T_vs_final_lr_ucf_recall", "P_T_vs_final_lr_ucf_new_recall",
                        "P_T_vs_final_lr_ucf_repeat_recall"):
                entry = row.get(key)
                cells.append("-" if entry is None else f"{fd(entry['diff'])} {ci(entry['ci_0.95'])} (n = {entry['n_users']:,})")
            lines.append("| " + " | ".join(cells) + " |")
        # Write the table of the agreement with the W15 numbers over all users.
        agree = d["w15_agreement"]
        lines += ["", f"### Agreement with the W15 part C numbers (R2 + E2; |gap| <= {TOLERANCE:g}): **{'yes' if agree['all_agree'] else 'NO'}** "
                  f"({agree['n_applicable']} checkable comparisons, {agree['n_not_applicable']} not applicable because T is not a bin edge of this dataset)", "",
                  "| comparison | metric | policy diff | W15-implied diff | gap | agree |", "| --- | --- | --- | --- | --- | --- |"]
        for row in agree["overall"]:
            if row["agree"] is None:
                lines.append(f"| {row['comparison']} | {row['metric']} | {row['policy_diff']:+.6f} | - | - | n/a |")
                continue
            lines.append(f"| {row['comparison']} | {row['metric']} | {row['policy_diff']:+.6f} | {row['w15_implied_diff']:+.6f} | {row['gap']:.1e} | "
                         f"{'yes' if row['agree'] else 'NO'} |")
        # Write the table of the first-bin entries against the W15 first-bin entries.
        lines += ["", "First bin (same users and bootstrap draw as W15):", "", "| key | policy diff [95 % CI] (n) | W15 diff [95 % CI] (n) | agree |",
                  "| --- | --- | --- | --- |"]
        for row in agree["first_bin"]:
            # Write both entries, the not-applicable note or the no-users text of the row.
            if "policy" in row:
                lines.append(f"| {row['key']} | {fd(row['policy']['diff'])} {ci(row['policy']['ci_0.95'])} ({row['policy']['n_users']:,}) | "
                             f"{fd(row['w15']['diff'])} {ci(row['w15']['ci_0.95'])} ({row['w15']['n_users']:,}) | {'yes' if row['agree'] else 'NO'} |")
            elif row["agree"] is None:
                lines.append(f"| {row['key']} | - | - | n/a ({row['note']}) |")
            else:
                lines.append(f"| {row['key']} | (no users) | (no users) | {'yes' if row['agree'] else 'NO'} |")
        # Write the line that compares the C3 list metrics with the W15 row.
        lines += ["", "C3 list metrics against the W15 R2 r = 10 row: " + ", ".join(
            f"{r['metric']} {r['policy_C3']:.4f} vs {r['w15_R2_r10']:.4f} ({'equal' if r['agree'] else 'DIFFERENT'})" for r in agree["c3_list_metrics"]), ""]
    # Write the table of the input files with their digests and commits.
    lines += ["## Inputs", "", "| file | sha256 | git_commit |", "| --- | --- | --- |"]
    for item in report["inputs"]:
        lines.append(f"| {item['path']} | {item['sha256']} | {item.get('git_commit') or ''} |")
    lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> None:
    '''
    Run the development check for the options in ``argv`` (default: the command line): read
    the documents, run the checks, write the JSON and Markdown reports and exit with code 1
    when a check fails.
    '''
    # Parse the options and refuse confirmatory input paths.
    args = parse_args(argv)
    metrics_dir = guard_path(Path(args.metrics_dir))
    reference_dir = guard_path(Path(args.reference_tuning_dir))
    w15_path = guard_path(Path(args.w15_json))
    out_dir = Path(args.out_dir)
    # Read the W15 post-hoc JSON and record it as the first input.
    w15 = json.loads(w15_path.read_text(encoding="utf-8"))
    inputs: list[dict[str, Any]] = [{"path": str(w15_path), "sha256": sha256_of(w15_path), "git_commit": _get(w15, ("meta", "git_commit"))}]
    checks: dict[str, Any] = {}
    policies: dict[str, Any] = {}
    tuning_commits, levels = set(), None
    for dataset in args.datasets:
        # Build the guarded paths of the tuning, eval and reference documents and the LR artifact.
        tuning_path = guard_path(metrics_dir / f"{args.profile}_{dataset}_tuning.json")
        eval_path = guard_path(metrics_dir / f"{args.profile}_{dataset}_eval.json")
        reference_path = guard_path(reference_dir / tuning_path.name)
        artifact = guard_path(ROOT / "artifacts" / args.profile / f"{dataset}_lr_repeat.joblib")
        # Read the three JSON documents.
        tuning = json.loads(tuning_path.read_text(encoding="utf-8"))
        evaluation = json.loads(eval_path.read_text(encoding="utf-8"))
        reference = json.loads(reference_path.read_text(encoding="utf-8"))
        # Stop when the eval document has no policy block.
        if not evaluation.get("policy"):
            raise SystemExit(f"{eval_path} has no policy block; run python -m nbr evaluate --profile {args.profile} --dataset {dataset} --skip-l3 --policy")
        policy = evaluation["policy"]
        # Stop when the settings of the policy block differ from the constants of nbr.policy.
        if (policy["T"], policy["delta"], policy["r_short"]) != (T, DELTA, R_SHORT) or policy["levels"] != [str(level) for level in LEVELS]:
            raise SystemExit(f"{eval_path}: policy settings differ from PREREG v1.4: {policy['T']}, {policy['delta']}, {policy['r_short']}, {policy['levels']}")
        # Remember the CI levels and the commit of the tuning document.
        levels = policy["levels"]
        tuning_commits.add(str(_get(tuning, ("meta", "git_commit"))))
        # Run the four checks of the dataset.
        checks[dataset] = {
            "selections": compare_selections(tuning, reference),
            "artifact": check_artifact(tuning, evaluation, artifact),
            "c3": check_c3(evaluation, args.profile, dataset),
            "w15_agreement": check_w15_agreement(policy, w15, dataset),
            "eval_meta": {k: evaluation.get("meta", {}).get(k) for k in ("git_commit", "utc_time")},
        }
        # The dataset passes when every check holds.
        checks[dataset]["passed"] = (checks[dataset]["selections"]["all_equal"] and checks[dataset]["artifact"]["matches"]
                                     and checks[dataset]["c3"]["matched"] and checks[dataset]["c3"]["policy_equals_list_metric"]
                                     and checks[dataset]["w15_agreement"]["all_agree"])
        policies[dataset] = policy
        # Record the path, digest and commit of every input file of the dataset.
        for path, doc in ((tuning_path, tuning), (eval_path, evaluation), (reference_path, reference)):
            inputs.append({"path": str(path), "sha256": sha256_of(path), "git_commit": _get(doc, ("meta", "git_commit"))})
        inputs.append({"path": str(artifact), "sha256": sha256_of(artifact), "git_commit": None})
    # Assemble the report document.
    report = {
        "label": LABEL,
        "task": "W17 P1.2: development check of the PREREG v1.4 exploration-policy code on medium",
        "profile": args.profile,
        "datasets": list(args.datasets),
        "settings": {"T": T, "delta": DELTA, "r_short": R_SHORT, "r_short_variant": R_SHORT_VARIANT, "T_sensitivity": list(T_SENSITIVITY)},
        "levels": levels,
        "primary_level": PRIMARY_LEVEL,
        "expected_approx_from_w15": EXPECTED_APPROX,
        "reference_tuning_dir": str(reference_dir),
        "tuning_commit": ", ".join(sorted(tuning_commits)),
        "w15_commit": _get(w15, ("meta", "git_commit")),
        "checks": checks,
        "passed": all(item["passed"] for item in checks.values()),
        "policy": policies,
        "inputs": inputs,
        "environment": {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__, "platform": platform.platform()},
    }
    # Write the JSON report.
    config = {"repo_root": ROOT, "profile": args.profile}
    json_path = write_json(out_dir / f"policy_dev_{args.profile}.json", report, config)
    # Copy the meta block of the written file into the report.
    written = json.loads(json_path.read_text(encoding="utf-8"))
    report["meta"] = written["meta"]
    # Write the Markdown report.
    md_path = out_dir / f"policy_dev_{args.profile}.md"
    md_path.write_text(render_markdown(report), encoding="utf-8", newline="\n")
    # Print the output paths, the check results and the primary entries of every dataset.
    print(f"wrote {json_path} and {md_path}; passed = {report['passed']}")
    for dataset, item in checks.items():
        print(f"{dataset}: selections equal {item['selections']['all_equal']}, artifact {item['artifact']['matches']}, "
              f"C3 NDCG {item['c3']['observed']:.6f} matched {item['c3']['matched']}, W15 agreement {item['w15_agreement']['all_agree']}")
        for key in PRIMARY_KEYS:
            entry = policies[dataset]["primary"][key]
            print(f"  {key}: diff {entry['diff']:+.6f} 95% {entry['ci']['0.95']} {PRIMARY_LEVEL} {entry['ci'][PRIMARY_LEVEL]} -> {entry['decision'][PRIMARY_LEVEL]}")
    # Exit with code 1 when a check failed.
    if not report["passed"]:
        sys.exit(1)


# ---------------------------------------------------------------------------
# command-line entry
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    main()
