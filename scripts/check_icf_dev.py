# *------------------- 2026 Fall Semester Machine Learning Term-project: Team 9 --------------------*
# Code First written at 2026-11-07
# Code Final Edited at 2026-11-13
# Editor: Yejun Kim

'''
Purpose : Development check of the PREREG v1.5 code (R2' = R2 + icf, C4 = R2' + E2) on the
          medium profile (W19 P1.4; not confirmatory). This is a code check on development
          data, not a selection: nothing is changed on its basis.
          Codes: R1 is the decayed purchase frequency repeat module; R2 is the
          logistic-regression repeat ranker; R2' is R2 with the item-CF similarity feature
          icf; E2 is the user-based CF explore module; C3 is the combination R2 + E2; C4 is
          the combination R2' + E2; H14 is the non-inferiority comparison of C4 vs C3 and H15
          the comparison of C4 vs personal frequency, both on NDCG@10; P_T is a combination
          with 9 repeat slots for users with at most T history baskets; S is the top 10 of E2
          evaluated on its own.
          Checks (all must pass; the script exits 1 otherwise):
          1. The existing selections equal the W8b reference (gamma, gamma_c, alpha, r, LR C,
             user-CF k and weighting, r per combination).
          2. C3 (final_lr_ucf) NDCG@10 of the eval document rounds to the W8b value (Instacart
             0.4274, Complete Journey 0.2046) and equals the reference eval document exactly;
             the v15 H14 comparator mean equals the list metric.
          3. The R2 and R2' artifact files carry the sha256 recorded in the tuning document
             and the eval document used them.
          4. R2' was selected on the validation users of R2 (same count, flag written by
             tune_v15) and has the 12 R2 features plus "icf".
          Reported (no decision taken on it): repeat Recall@10 of R2' vs R2 and vs R1 with the
          expected approximate values of the ICF_feature_check development study (Instacart
          .5757 -> .5759, Complete Journey .2740 -> .2739), the v15 primary entries H14 and
          H15 with their 95 % / 97.5 % intervals, the C of R2 and R2', r of C3 and C4, the
          standardized coefficients of R2', the history-length bins of H14, and (PREREG
          v1.5.1) the exploration policy on C4 ("policy_c4" block: P_T(C4) vs C4, S vs flat
          popularity; 95 % / 98.33 % intervals) next to the policy on C3 ("policy" block),
          with the W15/W17 development values on Instacart as the expected order of magnitude
          (overall delta NDCG@10 about -0.0013, delta new-hit share about +0.010).
Inputs  : Unless --skip-run is given, the script runs the pipeline of ``python -m nbr tune``
          and ``python -m nbr evaluate --skip-l3 --policy`` for every dataset
          (nbr.pipeline.run_tune / run_evaluate) with ``lr_repeat.itemcf_feature`` set to true
          in memory; the profile files are not changed. It then reads
          reports/metrics/<profile>_<dataset>_{tuning,eval}.json, the reference W8b documents
          of --reference-dir (<profile>_<dataset>_{tuning,eval}.json) and the LR artifacts
          artifacts/<profile>/<dataset>_lr_repeat{,_icf}.joblib.
          Only the development profiles ``sample`` and ``medium`` are accepted; no path with a
          "final" or "holdout" component is read.
Outputs : <out-dir>/icf_dev_<profile>.json and icf_dev_<profile>.md labelled "development
          check of the PREREG v1.5 code; not confirmatory".
Usage   :
    python scripts/check_icf_dev.py --reference-dir <W8b reports/metrics> [--profile medium]
                                    [--datasets instacart,complete_journey] [--skip-run]
                                    [--metrics-dir reports/metrics] [--out-dir reports/posthoc]
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
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from nbr.io import write_json  # noqa: E402
from nbr.policy import LEVELS as POLICY_LEVELS  # noqa: E402
from nbr.policy import PRIMARY_LEVEL as POLICY_PRIMARY_LEVEL  # noqa: E402
from nbr.v15 import DELTA, LEVELS, PRIMARY_LEVEL  # noqa: E402

# ---------------------------------------------------------------------------
# constants
# ---------------------------------------------------------------------------
LABEL = "development check of the PREREG v1.5 code; not confirmatory"
ALLOWED_PROFILES = ("sample", "medium")
DATASETS = ("instacart", "complete_journey")
NAMES = {"instacart": "Instacart", "complete_journey": "Complete Journey"}
TOLERANCE = 1e-9
# C3 NDCG@10 at four decimals (W8b medium eval, reproduced by W15 and W17).
EXPECTED_C3_NDCG = {("medium", "instacart"): 0.4274, ("medium", "complete_journey"): 0.2046}
# Approximate repeat Recall@10 (R2 -> R2') expected from the ICF_feature_check development study (task W19 P1.4), for the report only.
EXPECTED_APPROX = {"instacart": {"repeat_lr": 0.5757, "repeat_lri": 0.5759}, "complete_journey": {"repeat_lr": 0.2740, "repeat_lri": 0.2739}}
SELECTION_PATHS = {
    "gamma": ("selections", "gamma"), "gamma_c": ("selections", "gamma_c"), "alpha": ("selections", "alpha"), "r": ("selections", "r"),
    "lr_C": ("lr_repeat", "C"), "usercf_k": ("integration", "selections", "k"), "usercf_weighting": ("integration", "selections", "weighting"),
    "r_final_lr_r": ("integration", "selections", "r", "final_lr_r"), "r_final_lr_ucf": ("integration", "selections", "r", "final_lr_ucf"),
    "r_final_star_ucf": ("integration", "selections", "r", "final_star_ucf"),
}
PRIMARY_KEYS = ("H14_ndcg", "H15_ndcg")
POLICY_KEYS = ("H11_ndcg", "H12_new_hit_share", "H13_new_hit_share")
# Approximate policy effects on Instacart medium expected from W15/W17 (P_T vs C3; the C4 base is expected close to them), report only.
EXPECTED_POLICY_APPROX = {"instacart": {"ndcg": -0.0013, "new_hit_share": 0.010}}
SECONDARY_KEYS = ("repeat_lri_vs_repeat_lr_repeat_recall", "repeat_lri_vs_repeat_star_repeat_recall", "final_lri_ucf_vs_final_lr_ucf_recall")


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
    parser = argparse.ArgumentParser(description="Development check of the PREREG v1.5 two-CF code (development profiles only).")
    parser.add_argument("--profile", default="medium", choices=ALLOWED_PROFILES)
    parser.add_argument("--datasets", default=",".join(DATASETS), type=parse_datasets)
    parser.add_argument("--reference-dir", required=True, help="folder with the W8b <profile>_<dataset>_{tuning,eval}.json files")
    parser.add_argument("--skip-run", action="store_true", help="do not run tune / evaluate; only read the existing documents")
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


def check_artifact(recorded: Any, artifact: Path, used: Any) -> dict[str, Any]:
    """The artifact file carries the sha256 ``recorded`` in the tuning document and the eval document ``used`` the same digest."""
    # Hash the artifact file when it exists.
    observed = sha256_of(artifact) if artifact.exists() else None
    # Report the three digests and whether the file exists and all three agree.
    return {"artifact": str(artifact), "exists": artifact.exists(), "sha256_recorded": recorded, "sha256_observed": observed,
            "sha256_eval": used, "matches": artifact.exists() and observed == recorded == used}


def check_c3(evaluation: dict[str, Any], reference_eval: dict[str, Any] | None, profile: str, dataset: str) -> dict[str, Any]:
    '''
    C3 NDCG@10 of the ``evaluation`` document against the expected four-decimal value of
    (``profile``, ``dataset``), against ``reference_eval`` (skipped when None) and against
    the H14 comparator mean of the v15 block.
    '''
    # Look up the expected value, which is None for a profile without one.
    expected = EXPECTED_C3_NDCG.get((profile, dataset))
    # Read the C3 NDCG@10 from the list metrics.
    observed = float(evaluation["list_metrics"]["final_lr_ucf"]["ndcg"])
    # Read the comparator mean of the H14 entry of the v15 block.
    v15_c3 = float(evaluation["v15"]["primary"]["H14_ndcg"]["comparator_mean"])
    # Read the C3 NDCG@10 of the reference eval document when there is one.
    reference = None if reference_eval is None else float(reference_eval["list_metrics"]["final_lr_ucf"]["ndcg"])
    # Report whether the rounded value matches and whether the sources agree within the tolerance.
    return {"expected": expected, "observed": observed, "reference": reference, "v15_comparator_mean": v15_c3,
            "matched": expected is None or abs(round(observed, 4) - expected) < 1e-9,
            "equals_reference": reference is None or abs(observed - reference) <= TOLERANCE,
            "v15_equals_list_metric": abs(observed - v15_c3) <= TOLERANCE}


def check_r2prime(tuning: dict[str, Any], evaluation: dict[str, Any]) -> dict[str, Any]:
    """R2' setup: same validation users as R2 (count and flag), 13 features = R2 features + icf, C grids equal."""
    block, lr = tuning["v15"], tuning["lr_repeat"]
    # List the feature names of R2 and of R2' from the tuning document.
    features_r2 = [row["feature"] for row in lr["coefficients"]]
    features_r2p = [row["feature"] for row in block["lr_repeat_icf"]["coefficients"]]
    # List the feature names of R2' recorded in the eval document.
    eval_features = evaluation["v15"]["lr_repeat_icf"]["features"]
    # Report the counts, the flag, the features, the C grids and whether all conditions hold.
    return {
        "n_val_users_R2": lr["n_val_users"], "n_val_users_R2prime": block["lr"]["n_val_users"],
        "same_validation_users_flag": bool(block["lr"].get("same_validation_users_as_R2")),
        "features_R2prime": features_r2p, "icf_neighbors": block["icf_neighbors"],
        "C_grid_equal": [float(c) for c in block["lr"]["C_grid"]] == [float(c) for c in lr["C_grid"]],
        "ok": (lr["n_val_users"] == block["lr"]["n_val_users"] and bool(block["lr"].get("same_validation_users_as_R2"))
               and features_r2p == features_r2 + ["icf"] and eval_features == features_r2p
               and [float(c) for c in block["lr"]["C_grid"]] == [float(c) for c in lr["C_grid"]]),
    }


# ---------------------------------------------------------------------------
# running the pipeline (development profiles only)
# ---------------------------------------------------------------------------
def run_pipeline(profile: str, datasets: list[str]) -> dict[str, Any]:
    """tune + evaluate --skip-l3 of every dataset with lr_repeat.itemcf_feature forced to true (in memory)."""
    from nbr.config import load_config
    from nbr.data import dataset_dir
    from nbr.pipeline import run_evaluate, run_tune

    # Load the profile and refuse a final profile or a profile outside ALLOWED_PROFILES.
    config = load_config(profile)
    if config.get("role") == "final" or profile not in ALLOWED_PROFILES:
        raise PermissionError("only the development profiles are run by this script")
    # Refuse confirmatory data folders.
    for dataset in datasets:
        guard_path(dataset_dir(config, dataset))
    # Switch the item-CF feature on in the loaded configuration only.
    config["lr_repeat"] = {**config["lr_repeat"], "itemcf_feature": True}
    record: dict[str, Any] = {"profile": profile, "itemcf_feature": True, "commands": []}
    for dataset in datasets:
        for stage in ("tune", "evaluate"):
            # Run the stage and time it.
            t0 = time.perf_counter()
            if stage == "tune":
                path = run_tune(config, dataset)
            else:
                path = run_evaluate(config, dataset, skip_l3=True, include_policy=True)
            # Record the equivalent command, the written path and the run time.
            record["commands"].append({"stage": stage, "dataset": dataset, "equivalent": f"python -m nbr {stage} --profile {profile} --dataset {dataset}"
                                       + (" --skip-l3 --policy" if stage == "evaluate" else "") + " (lr_repeat.itemcf_feature = true in memory)",
                                       "path": str(path), "seconds": round(time.perf_counter() - t0, 1)})
            print(f"{stage} {profile}/{dataset}: wrote {path} in {record['commands'][-1]['seconds']:.1f}s", flush=True)
    return record


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


def entry_row(name: str, entry: dict[str, Any], level: str = PRIMARY_LEVEL) -> str:
    '''
    Markdown table row of the comparison ``entry`` labelled ``name``; ``level`` is the CI
    level shown next to the 95 % level.
    '''
    decision = entry.get("decision", {})
    # Format the user count, the lists, the means, the difference, both CIs and the decisions.
    cells = [name, f"{entry['n_users']:,}", f"{entry['comparator']} -> {entry['model']}", f4(entry["comparator_mean"]), f4(entry["model_mean"]),
             fd(entry["diff"]), ci(entry["ci"]["0.95"]), ci(entry["ci"][level]),
             f"{decision['0.95']} / {decision[level]}" if decision else "(CI only)"]
    # Join the cells into one table row.
    return "| " + " | ".join(cells) + " |"


def render_markdown(report: dict[str, Any]) -> str:
    '''Markdown text of the development-check ``report`` (one section per dataset, then inputs).'''
    profile = report["profile"]
    # Write the title, the description of the check and the run facts.
    lines = [f"# Two-CF final system (PREREG v1.5): development check on {profile}", "",
             f"**{LABEL}.** The W19 code (nbr.features_icf, nbr.v15; `python -m nbr tune` / `evaluate --skip-l3` with "
             "`lr_repeat.itemcf_feature: true`) is run on the development profile: R2' = R2 + icf (20 item neighbours, C selected separately "
             "on the validation users of R2), C4 = combine(R2', E2, r) with r re-selected like C3's, and the v15 block (H14 non-inferiority "
             f"with delta = {DELTA:g}, H15; CI levels {', '.join(str(level) for level in LEVELS)}). The existing selections are compared with "
             "the W8b reference and C3 must reproduce the W8b NDCG@10. This is a code check, not a selection: nothing is changed on its "
             "basis. The holdout4 run (reports/final4) is the confirmatory test.",
             "",
             f"- Code commit `{report['meta']['git_commit']}`, {report['meta']['utc_time']}; profile `{profile}`; CI levels "
             f"{', '.join(report['levels'])} (primary level {PRIMARY_LEVEL} = 1 - 0.05/2); paired bootstrap n = 1000, seed 42.",
             f"- Pipeline run by this script: **{'yes' if report['run'] else 'no (--skip-run; existing documents read)'}**; tuning document "
             f"commit `{report['tuning_commit']}`; W8b reference `{report['reference_dir']}`.",
             "- Expected from the ICF_feature_check development study (approximate, repeat Recall@10 R2 -> R2'): "
             + "; ".join(f"{NAMES[d]} {v['repeat_lr']:.4f} -> {v['repeat_lri']:.4f}" for d, v in EXPECTED_APPROX.items()) + ".",
             f"- All checks passed: **{'yes' if report['passed'] else 'NO'}**.", ""]
    for dataset in report["datasets"]:
        d = report["checks"][dataset]
        block = report["v15"][dataset]
        tuning_v15 = report["tuning_v15"][dataset]
        # Write the setup table: the selections, the two LR artifacts and the R2' setup.
        lines += [f"## {NAMES[dataset]}", "", "### Setup check (existing selections vs the W8b reference; LR artifacts)", "",
                  "| quantity | this run | reference | equal |", "| --- | --- | --- | --- |"]
        for row in d["selections"]["selections"]:
            lines.append(f"| {row['quantity']} | {row['run']} | {row['reference']} | {'yes' if row['equal'] else 'NO'} |")
        for name, art in (("R2 artifact sha256 (tuning / file / eval)", d["artifact_R2"]), ("R2' artifact sha256 (tuning / file / eval)", d["artifact_R2prime"])):
            lines.append(f"| {name} | {str(art['sha256_recorded'])[:12]} | {str(art['sha256_observed'])[:12]} / {str(art['sha256_eval'])[:12]} | "
                         f"{'yes' if art['matches'] else 'NO'} |")
        r2p = d["r2prime"]
        lines.append(f"| R2' validation users = R2 validation users; 13 features (R2 + icf); icf neighbours | {r2p['n_val_users_R2prime']} / "
                     f"{len(r2p['features_R2prime'])} features / {r2p['icf_neighbors']} | {r2p['n_val_users_R2']} / 13 / 20 | {'yes' if r2p['ok'] else 'NO'} |")
        # Write the reproduction table of the C3 NDCG@10.
        c3 = d["c3"]
        reference_text = "-" if c3["reference"] is None else f"{c3['reference']:.6f}"
        c3_ok = c3["matched"] and c3["equals_reference"] and c3["v15_equals_list_metric"]
        lines += ["", "### Reproduction check (C3 = final_lr_ucf NDCG@10)", "",
                  "| expected (4 decimals) | observed | W8b reference | v15 comparator mean | matched |", "| --- | --- | --- | --- | --- |",
                  f"| {c3['expected']} | {c3['observed']:.6f} | {reference_text} | {c3['v15_comparator_mean']:.6f} | {'yes' if c3_ok else 'NO'} |"]
        # Write the table of the selected C and r values with their grids and selection scores.
        sel = tuning_v15["selections"]
        lines += ["", "### Selections of the v1.5 systems (fit split; development data)", "",
                  "| quantity | value | grid | at grid edge | selection score |", "| --- | --- | --- | --- | --- |",
                  f"| C (R2) | {report['tuning_lr'][dataset]['C']} | {report['tuning_lr'][dataset]['C_grid']} | {report['tuning_lr'][dataset]['at_grid_edge']} | "
                  + "validation repeat Recall@10 by C: " + ", ".join(f"{row['C']:g}: {row['val_recall']:.6f}" for row in report['tuning_lr'][dataset]['validation']) + " |",
                  f"| C (R2') | {sel['C']} | {tuning_v15['lr']['C_grid']} | {tuning_v15['at_grid_edge']['C']} | "
                  + "validation repeat Recall@10 by C: " + ", ".join(f"{row['C']:g}: {row['val_recall']:.6f}" for row in tuning_v15['lr_repeat_icf']['validation']) + " |",
                  f"| r (C3 = final_lr_ucf) | {report['tuning_integration'][dataset]['selections']['r']['final_lr_ucf']} | {report['tuning_integration'][dataset]['r_grid']} | "
                  f"{report['tuning_integration'][dataset]['at_grid_edge']['r']['final_lr_ucf']} | NDCG@10 on {report['tuning_integration'][dataset]['r_selection_users']['final_lr_ucf']['n_users']:,} validation users: "
                  + ", ".join(f"r = {row['r']}: {row['ndcg']:.6f}" for row in report['tuning_integration'][dataset]['r_scores']['final_lr_ucf']) + " |",
                  f"| r (C4 = final_lri_ucf) | {sel['r']['final_lri_ucf']} | {tuning_v15['r_grid']} | {tuning_v15['at_grid_edge']['r']['final_lri_ucf']} | "
                  f"NDCG@10 on {tuning_v15['r_selection_users']['final_lri_ucf']['n_users']:,} validation users: "
                  + ", ".join(f"r = {row['r']}: {row['ndcg']:.6f}" for row in tuning_v15['r_scores']['final_lri_ucf']) + " |"]
        # Write the heading with the user counts and the table of the primary entries.
        n = block["n_users"]
        lines += ["", f"### v15 block ({n['all']:,} users; {n['repeat_target']:,} with a repeat target item, {n['new_target']:,} with a new target item)", "",
                  f"Primary (decision at 95 % / 97.5 %; H14 rule: non-inferior if the 97.5 % lower bound > -{DELTA:g}):", "",
                  "| hypothesis | n | comparator -> model | comparator mean | model mean | diff | 95 % CI | 97.5 % CI | decision |",
                  "| --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
        for key in PRIMARY_KEYS:
            lines.append(entry_row(f"{key} ({block['primary'][key]['metric']}, {block['primary'][key]['users']})", block["primary"][key]))
        # Write the table of the secondary comparisons.
        lines += ["", "Secondary (CI only):", "",
                  "| comparison | n | comparator -> model | comparator mean | model mean | diff | 95 % CI | 97.5 % CI | decision |",
                  "| --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
        for key in SECONDARY_KEYS:
            entry = block["secondary"][key]
            lines.append(entry_row(f"{key} ({entry['metric']}, {entry['users']})", entry))
        # Write the observed repeat Recall@10 of R2 and R2' next to the expected approximate values.
        expected = EXPECTED_APPROX.get(dataset)
        rr = block["secondary"]["repeat_lri_vs_repeat_lr_repeat_recall"]
        if expected:
            lines += ["", f"Repeat Recall@10 R2 -> R2': observed {rr['comparator_mean']:.4f} -> {rr['model_mean']:.4f} "
                      f"(expected approx. {expected['repeat_lr']:.4f} -> {expected['repeat_lri']:.4f} from the ICF_feature_check study)."]
        # Write the table of the standardized coefficients of R2'.
        coef = block["lr_repeat_icf"]
        lines += ["", f"R2' (C = {coef['C']}, intercept {coef['intercept']:+.4f}; standardized coefficients):", "",
                  "| feature | coefficient |", "| --- | --- |"]
        for name, value in coef["coefficients"].items():
            lines.append(f"| {name} | {value:+.4f} |")
        # Write the table of the H14 difference by history-length bin.
        lines += ["", "By history-length bin (H14: C4 vs C3, NDCG@10; diff [95 % CI]):", "", "| bin | users | H14 NDCG@10 |", "| --- | --- | --- |"]
        for row in block["by_history_length"]:
            entry = row.get("H14_ndcg")
            lines.append(f"| {row['bin']} | {row['n_users']:,} | " + ("-" if entry is None else f"{fd(entry['diff'])} {ci(entry['ci_0.95'])} (n = {entry['n_users']:,})") + " |")
        # Write the primary policy entries with C4 as the base and with C3 as the base.
        pc4, pc3 = report["policy_c4"][dataset], report["policy"][dataset]
        lines += ["", f"### Exploration policy on C4 (PREREG v1.5.1, `policy_c4` block; T = {pc4['T']}, r_short = {pc4['r_short']}, r_default = r (C4) = "
                  f"{pc4['r_default']}; {pc4['n_users']['short_history']:,} users with history_len <= {pc4['T']}) next to the policy on C3 "
                  f"(`policy` block; r_default = {pc3['r_default']})", "",
                  "Decision wording of PREREG v1.4 at 95 % / 98.33 % (secondary replication; the confirmatory policy result is H11-H13 on holdout3):", "",
                  "| base | hypothesis | n | comparator -> model | comparator mean | model mean | diff | 95 % CI | 98.33 % CI | decision |",
                  "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
        for base_name, pblock in (("C4", pc4), ("C3", pc3)):
            for key in POLICY_KEYS:
                entry = pblock["primary"][key]
                lines.append(f"| {base_name} " + entry_row(f"{key} ({entry['metric']}, {entry['users']})", entry, POLICY_PRIMARY_LEVEL)[2:])
        # Write the table of the policy variants for both bases.
        lines += ["", "Variants (CI only):", "",
                  "| base | variant | n | comparator -> model | comparator mean | model mean | diff | 95 % CI | 98.33 % CI | decision |",
                  "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
        for base_name, pblock in (("C4", pc4), ("C3", pc3)):
            for name, variant in pblock["variants"].items():
                for metric in ("ndcg", "new_hit_share"):
                    entry = variant["comparisons"][metric]
                    lines.append(f"| {base_name} " + entry_row(f"{name} (T = {variant['T']}, r_short = {variant['r_short']}): {entry['metric']}, {entry['users']}",
                                                               entry, POLICY_PRIMARY_LEVEL)[2:])
        # Write the expected order of magnitude of the policy effects next to the observed values.
        expected = EXPECTED_POLICY_APPROX.get(dataset)
        if expected:
            lines += ["", f"Expected order of magnitude from W15/W17 (P_T vs C3 on Instacart medium): delta NDCG@10 about {expected['ndcg']:+.4f}, delta new-hit "
                      f"share about {expected['new_hit_share']:+.3f}; observed with the C4 base: {pc4['primary']['H11_ndcg']['diff']:+.4f} and "
                      f"{pc4['primary']['H12_new_hit_share']['diff']:+.4f}."]
        lines.append("")
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
    Run the development check for the options in ``argv`` (default: the command line): run
    the pipeline unless --skip-run is given, read the documents, run the checks, write the
    JSON and Markdown reports and exit with code 1 when a check fails.
    '''
    # Parse the options and refuse confirmatory input paths.
    args = parse_args(argv)
    metrics_dir = guard_path(Path(args.metrics_dir))
    reference_dir = guard_path(Path(args.reference_dir))
    out_dir = Path(args.out_dir)
    # Run tune and evaluate for every dataset unless --skip-run is given.
    run_record = None if args.skip_run else run_pipeline(args.profile, args.datasets)
    # Prepare the per-dataset containers of the report.
    inputs: list[dict[str, Any]] = []
    checks: dict[str, Any] = {}
    blocks: dict[str, Any] = {}
    policy_blocks: dict[str, Any] = {}
    policy_c4_blocks: dict[str, Any] = {}
    tuning_v15: dict[str, Any] = {}
    tuning_lr: dict[str, Any] = {}
    tuning_integration: dict[str, Any] = {}
    tuning_commits, levels = set(), None
    for dataset in args.datasets:
        # Build the guarded paths of the documents, the reference documents and both LR artifacts.
        tuning_path = guard_path(metrics_dir / f"{args.profile}_{dataset}_tuning.json")
        eval_path = guard_path(metrics_dir / f"{args.profile}_{dataset}_eval.json")
        reference_tuning_path = guard_path(reference_dir / tuning_path.name)
        reference_eval_path = guard_path(reference_dir / eval_path.name)
        artifact = guard_path(ROOT / "artifacts" / args.profile / f"{dataset}_lr_repeat.joblib")
        artifact_icf = guard_path(ROOT / "artifacts" / args.profile / f"{dataset}_lr_repeat_icf.joblib")
        # Read the JSON documents; the reference eval document is optional.
        tuning = json.loads(tuning_path.read_text(encoding="utf-8"))
        evaluation = json.loads(eval_path.read_text(encoding="utf-8"))
        reference = json.loads(reference_tuning_path.read_text(encoding="utf-8"))
        reference_eval = json.loads(reference_eval_path.read_text(encoding="utf-8")) if reference_eval_path.exists() else None
        # Stop when the tuning or the eval document has no v15 block.
        if not tuning.get("v15") or not evaluation.get("v15"):
            raise SystemExit(f"{tuning_path} / {eval_path}: no v15 block; run with lr_repeat.itemcf_feature true (drop --skip-run)")
        # Stop when the eval document lacks one of the two policy blocks.
        if not evaluation.get("policy") or not evaluation.get("policy_c4"):
            raise SystemExit(f"{eval_path}: no policy / policy_c4 block; run evaluate with --policy (drop --skip-run)")
        # Stop when the settings of a policy block differ from T = 4, delta = 0.003 and r_short = 9.
        for name in ("policy", "policy_c4"):
            pblock = evaluation[name]
            if (pblock["T"], pblock["delta"], pblock["r_short"]) != (4, 0.003, 9) or pblock["levels"] != [str(level) for level in POLICY_LEVELS]:
                raise SystemExit(f"{eval_path}: {name} settings differ from PREREG v1.4: {pblock['T']}, {pblock['delta']}, {pblock['r_short']}, {pblock['levels']}")
        # Stop when the base list of a policy block is not the expected list name.
        if evaluation["policy_c4"]["lists"]["C3"] != "final_lri_ucf" or evaluation["policy"]["lists"]["C3"] != "final_lr_ucf":
            raise SystemExit(f"{eval_path}: unexpected base lists {evaluation['policy']['lists']['C3']} / {evaluation['policy_c4']['lists']['C3']}")
        block = evaluation["v15"]
        # Stop when the settings of the v15 block differ from the constants of nbr.v15.
        if block["delta"] != DELTA or block["levels"] != [str(level) for level in LEVELS]:
            raise SystemExit(f"{eval_path}: v15 settings differ from PREREG v1.5: {block['delta']}, {block['levels']}")
        # Remember the CI levels and the commit of the tuning document.
        levels = block["levels"]
        tuning_commits.add(str(_get(tuning, ("meta", "git_commit"))))
        # Run the checks of the dataset.
        checks[dataset] = {
            "selections": compare_selections(tuning, reference),
            "artifact_R2": check_artifact(_get(tuning, ("lr_repeat", "sha256")), artifact, _get(evaluation, ("lr_repeat", "sha256"))),
            "artifact_R2prime": check_artifact(_get(tuning, ("v15", "lr_repeat_icf", "sha256")), artifact_icf,
                                               _get(evaluation, ("v15", "lr_repeat_icf", "sha256"))),
            "c3": check_c3(evaluation, reference_eval, args.profile, dataset),
            "r2prime": check_r2prime(tuning, evaluation),
            "eval_meta": {k: evaluation.get("meta", {}).get(k) for k in ("git_commit", "utc_time")},
        }
        # The dataset passes when every check holds.
        c3 = checks[dataset]["c3"]
        checks[dataset]["passed"] = (checks[dataset]["selections"]["all_equal"] and checks[dataset]["artifact_R2"]["matches"]
                                     and checks[dataset]["artifact_R2prime"]["matches"] and c3["matched"] and c3["equals_reference"]
                                     and c3["v15_equals_list_metric"] and checks[dataset]["r2prime"]["ok"])
        # Keep the eval blocks and the tuning entries that the report shows.
        blocks[dataset] = block
        policy_blocks[dataset] = evaluation["policy"]
        policy_c4_blocks[dataset] = evaluation["policy_c4"]
        tuning_v15[dataset] = tuning["v15"]
        tuning_lr[dataset] = {k: tuning["lr_repeat"][k] for k in ("C", "C_grid", "validation")}
        tuning_lr[dataset]["at_grid_edge"] = tuning["integration"]["at_grid_edge"]["C"]
        tuning_integration[dataset] = {k: tuning["integration"][k] for k in ("selections", "r_grid", "r_scores", "at_grid_edge", "r_selection_users")}
        # Record the path, digest and commit of every document that was read.
        for path, doc in ((tuning_path, tuning), (eval_path, evaluation), (reference_tuning_path, reference), (reference_eval_path, reference_eval)):
            if doc is not None:
                inputs.append({"path": str(path), "sha256": sha256_of(path), "git_commit": _get(doc, ("meta", "git_commit"))})
        # Record the path and digest of the two LR artifacts.
        for file in (artifact, artifact_icf):
            inputs.append({"path": str(file), "sha256": sha256_of(file) if file.exists() else None, "git_commit": None})
    # Assemble the report document.
    report = {
        "label": LABEL,
        "task": "W19 P1.4: development check of the PREREG v1.5 two-CF code on medium (R2' = R2 + icf, C4 = R2' + E2)",
        "profile": args.profile,
        "datasets": list(args.datasets),
        "settings": {"delta": DELTA, "levels": [str(level) for level in LEVELS], "primary_level": PRIMARY_LEVEL, "icf_neighbors": 20},
        "levels": levels,
        "expected_approx_from_icf_feature_check": EXPECTED_APPROX,
        "expected_policy_approx_from_w15_w17": EXPECTED_POLICY_APPROX,
        "policy_settings": {"T": 4, "delta": 0.003, "r_short": 9, "levels": [str(level) for level in POLICY_LEVELS], "primary_level": POLICY_PRIMARY_LEVEL,
                            "prereg": "v1.5.1 (secondary replication of the v1.4 policy with C4 as the base)"},
        "expected_c3_ndcg": {f"{p}/{d}": v for (p, d), v in EXPECTED_C3_NDCG.items()},
        "reference_dir": str(reference_dir),
        "run": run_record,
        "tuning_commit": ", ".join(sorted(tuning_commits)),
        "checks": checks,
        "passed": all(item["passed"] for item in checks.values()),
        "v15": blocks,
        "policy": policy_blocks,
        "policy_c4": policy_c4_blocks,
        "tuning_v15": tuning_v15,
        "tuning_lr": tuning_lr,
        "tuning_integration": tuning_integration,
        "inputs": inputs,
        "environment": {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__, "platform": platform.platform()},
    }
    # Write the JSON report.
    config = {"repo_root": ROOT, "profile": args.profile}
    json_path = write_json(out_dir / f"icf_dev_{args.profile}.json", report, config)
    # Copy the meta block of the written file into the report.
    written = json.loads(json_path.read_text(encoding="utf-8"))
    report["meta"] = written["meta"]
    # Write the Markdown report.
    md_path = out_dir / f"icf_dev_{args.profile}.md"
    md_path.write_text(render_markdown(report), encoding="utf-8", newline="\n")
    # Print the output paths, the check results and the main entries of every dataset.
    print(f"wrote {json_path} and {md_path}; passed = {report['passed']}")
    for dataset, item in checks.items():
        print(f"{dataset}: selections equal {item['selections']['all_equal']}, artifacts {item['artifact_R2']['matches']} / {item['artifact_R2prime']['matches']}, "
              f"C3 NDCG {item['c3']['observed']:.6f} matched {item['c3']['matched']} (reference equal {item['c3']['equals_reference']}), "
              f"R2' setup {item['r2prime']['ok']}")
        rr = blocks[dataset]["secondary"]["repeat_lri_vs_repeat_lr_repeat_recall"]
        print(f"  repeat Recall@10 R2 -> R2': {rr['comparator_mean']:.4f} -> {rr['model_mean']:.4f} diff {rr['diff']:+.6f} 95% {rr['ci']['0.95']}")
        for key in PRIMARY_KEYS:
            entry = blocks[dataset]["primary"][key]
            print(f"  {key}: diff {entry['diff']:+.6f} 95% {entry['ci']['0.95']} {PRIMARY_LEVEL} {entry['ci'][PRIMARY_LEVEL]} -> {entry['decision'][PRIMARY_LEVEL]}")
        for key in POLICY_KEYS:
            c4, c3 = policy_c4_blocks[dataset]["primary"][key], policy_blocks[dataset]["primary"][key]
            print(f"  policy_c4 {key}: diff {c4['diff']:+.6f} {POLICY_PRIMARY_LEVEL} {c4['ci'][POLICY_PRIMARY_LEVEL]} -> {c4['decision'][POLICY_PRIMARY_LEVEL]} "
                  f"(policy on C3: {c3['diff']:+.6f} -> {c3['decision'][POLICY_PRIMARY_LEVEL]})")
    # Exit with code 1 when a check failed.
    if not report["passed"]:
        sys.exit(1)


# ---------------------------------------------------------------------------
# command-line entry
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    main()
