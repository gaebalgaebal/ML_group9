"""Post-hoc multiplicity tests and direct displacement accounting for health subgroups."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from importlib.metadata import version
from pathlib import Path

import numpy as np
import pandas as pd

from . import health_subgroup as hs
from .config import load_config
from .data import load_split
from .re_pipeline import build_scope, recommend
from .repeat_explore import repeat_scores


WORKTREE = Path("C:/dev/ML_group9-wt-hsf")
REFERENCE_ROOT = Path("C:/dev/ML_group9")
BRANCH = "feat/hs-followup"
SEED = 42
RESAMPLES = 10_000
LEGACY_RESAMPLES = 1_000
FAMILY_SIZE = 8
ALPHA = 0.05
TOLERANCE = 1e-9
COUNT_COLUMNS = ["gained_attr", "lost_attr", "gained_other", "lost_other", "changed_slots"]


def holm_adjust(p_values: np.ndarray | list[float]) -> np.ndarray:
    """Return Holm step-down adjusted p values in the input order."""
    values = np.asarray(p_values, dtype=float)
    if values.ndim != 1 or not np.isfinite(values).all() or ((values < 0) | (values > 1)).any():
        raise ValueError("p values must be a finite one-dimensional array in [0, 1]")
    order = np.argsort(values, kind="stable")
    adjusted = np.minimum(1.0, np.maximum.accumulate((len(values) - np.arange(len(values))) * values[order]))
    result = np.empty_like(values)
    result[order] = adjusted
    return result


def bootstrap_p(means: np.ndarray) -> float:
    """Two-sided inclusive bootstrap-tail p, clipped to [1/B, 1]."""
    values = np.asarray(means, dtype=float)
    if values.ndim != 1 or not len(values) or not np.isfinite(values).all():
        raise ValueError("Bootstrap means must be a non-empty finite vector")
    probability = 2 * min(float(np.mean(values <= 0)), float(np.mean(values >= 0)))
    return float(np.clip(probability, 1 / len(values), 1))


def bootstrap_means(differences: np.ndarray, resamples: int, seed: int) -> np.ndarray:
    """Resample paired user differences in sorted user order, matching the HS RNG stream."""
    values = np.asarray(differences, dtype=float)
    if values.ndim != 1 or not len(values) or not np.isfinite(values).all() or resamples < 1:
        raise ValueError("Differences must be non-empty and finite; resamples must be positive")
    rng = np.random.default_rng(seed)
    means = []
    for start in range(0, resamples, 100):
        draws = rng.integers(0, len(values), size=(min(100, resamples - start), len(values)))
        means.append(values[draws].mean(axis=1))
    return np.concatenate(means)


def percentile_interval(means: np.ndarray, confidence: float) -> dict[str, float]:
    """Return a two-sided percentile interval."""
    if not 0 < confidence < 1:
        raise ValueError("Confidence must lie strictly between zero and one")
    tail = 100 * (1 - confidence) / 2
    low, high = np.percentile(means, [tail, 100 - tail])
    return {"confidence": confidence, "ci_low": float(low), "ci_high": float(high)}


def reproduction_check(
    actual: dict[str, float | None], expected: dict[str, float | None], tolerance: float = TOLERANCE
) -> dict[str, dict[str, float | bool | None]]:
    """Require all matching means within absolute tolerance; reject missing or non-finite means."""
    if actual.keys() != expected.keys() or not actual:
        raise ValueError("Reproduction keys differ or are empty")
    checks = {}
    for key in actual:
        observed, reference = actual[key], expected[key]
        valid = (
            observed is not None and reference is not None
            and np.isfinite(observed) and np.isfinite(reference)
        )
        delta = float(observed - reference) if valid else None
        checks[key] = {
            "actual": observed, "reference": reference, "difference": delta,
            "absolute_difference": abs(delta) if valid else None,
            "passed": bool(valid and abs(delta) <= tolerance),
        }
    if not all(row["passed"] for row in checks.values()):
        raise ValueError("Reproduction mismatch: " + json.dumps(checks, allow_nan=False))
    return checks


def displacement_counts(
    first: list[str], second: list[str], new_items: set[str], attribute_items: set[str], top_k: int = 10
) -> dict[str, int]:
    """Count entering/leaving new truths and entering attribute items in explore top-K."""
    before, after = set(first[:top_k]), set(second[:top_k])
    entered, left = after - before, before - after
    return {
        "gained_attr": len(entered & new_items & attribute_items),
        "lost_attr": len(left & new_items & attribute_items),
        "gained_other": len((entered & new_items) - attribute_items),
        "lost_other": len((left & new_items) - attribute_items),
        "changed_slots": len(entered),
        "entered_attr": len(entered & attribute_items),
        "net": len(entered & new_items) - len(left & new_items),
    }


def summarize_displacement(frame: pd.DataFrame) -> dict[str, object]:
    """Summarize non-empty-N users; net intervals resample paired hit-count differences."""
    eligible = frame.loc[frame["new_items_count"] > 0].sort_values("user_id")
    count = len(eligible)
    sums = {column: int(eligible[column].sum()) for column in COUNT_COLUMNS}
    means = {column: float(eligible[column].mean()) if count else None for column in COUNT_COLUMNS}
    net = eligible["net"].to_numpy(dtype=float)
    interval = percentile_interval(bootstrap_means(net, LEGACY_RESAMPLES, SEED), 0.95) if count else None
    entered_attr = int(eligible["entered_attr"].sum())
    same_user = (eligible["gained_attr"] > 0) & (eligible["lost_other"] > 0)
    return {
        "users": count, "sums": sums, "means": means,
        "net_sum": int(net.sum()), "net_mean": float(net.mean()) if count else None,
        "net_ci_95": interval, "resamples": LEGACY_RESAMPLES, "seed": SEED,
        "changed_slot_attribute_items": entered_attr,
        "changed_slot_attribute_share": entered_attr / sums["changed_slots"] if sums["changed_slots"] else None,
        "changed_slot_share_definition": "sum(|(L1 - L0) intersect A|) / sum(|L1 - L0|)",
        "users_gained_attr_and_lost_other": int(same_user.sum()),
        "lost_other_among_users_gaining_attr": int(eligible.loc[eligible["gained_attr"] > 0, "lost_other"].sum()),
    }


def _read_json(path: Path) -> tuple[dict[str, object], dict[str, str]]:
    data = path.read_bytes()
    return json.loads(data), {"path": str(path.resolve()), "sha256": hashlib.sha256(data).hexdigest()}


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def _check_workspace(config: dict[str, object]) -> None:
    if Path(config["project_root"]).resolve() != WORKTREE.resolve():
        raise RuntimeError(f"Run only from the existing worktree: {WORKTREE}")
    branch = subprocess.check_output(
        ["git", "branch", "--show-current"], cwd=WORKTREE, text=True
    ).strip()
    if branch != BRANCH:
        raise RuntimeError(f"Expected {BRANCH}, found {branch}; no branch will be changed")
    for name in ["raw", "processed"]:
        if (WORKTREE / "data" / name).resolve() != (REFERENCE_ROOT / "data" / name).resolve():
            raise RuntimeError(f"Expected a read-only shared-data junction: {name}")


def _per_user(
    split: hs.HealthSplit, scope: object, analysis: hs.Analysis, users: set[str],
    gamma_state: object, fixed: dict[str, object], config: dict[str, object],
) -> pd.DataFrame:
    """Use unchanged HS recommendation and scoring functions; retain lists and hit accounting."""
    user_index = pd.Index(sorted(users), name="user_id", dtype=object)
    frame = pd.DataFrame(index=user_index)
    targets = {user: scope.targets[user] for user in sorted(users)}
    new_items = hs.new_item_sets(targets, split.state.history_items)
    attribute_items = analysis.attribute_items[analysis.primary]
    top_k = int(config["evaluation"]["top_k"])
    for suffix, beta in [("b0", 0.0), ("bf", float(config["health_subgroup"]["beta_fixed"]))]:
        recommendation = recommend(
            split.state, scope, gamma_state, float(fixed["kappa"]), beta, config, fixed["global_r"]
        )
        scores = hs.user_scores(
            (recommendation.final, recommendation.explore, recommendation.repeat),
            targets, split.state.history_items, analysis.attribute_items, top_k,
        )
        frame[f"new_recall_{suffix}"] = scores["new_item_recall_at_k"]
        frame[f"attr_recall_{suffix}"] = scores[f"attr_new_item_recall_{analysis.primary}"]
        frame[f"explore_top10_{suffix}"] = [
            [str(item) for item in recommendation.explore.get(user, [])[:top_k]] for user in user_index
        ]
        del recommendation, scores
    frame["new_items_count"] = [len(new_items[user]) for user in user_index]
    frame["attr_new_items_count"] = [len(new_items[user] & attribute_items) for user in user_index]
    counts = [
        displacement_counts(row.explore_top10_b0, row.explore_top10_bf, new_items[user], attribute_items, top_k)
        for user, row in frame.iterrows()
    ]
    count_frame = pd.DataFrame(counts, index=user_index, columns=COUNT_COLUMNS + ["entered_attr", "net"])
    frame = frame.join(count_frame)
    for prefix, denominator, net in [
        ("new", frame["new_items_count"], frame["net"]),
        ("attr", frame["attr_new_items_count"], frame["gained_attr"] - frame["lost_attr"]),
    ]:
        eligible = denominator > 0
        observed = frame.loc[eligible, f"{prefix}_recall_bf"] - frame.loc[eligible, f"{prefix}_recall_b0"]
        expected = net[eligible] / denominator[eligible]
        if not np.allclose(observed, expected, atol=1e-12, rtol=0):
            raise RuntimeError(f"Displacement/recall identity failed: {analysis.label}, {prefix}")
    return frame.reset_index()


def _metric_tests(frame: pd.DataFrame, label: str, reference_group: dict[str, object]) -> list[dict[str, object]]:
    results = []
    for short, original in [
        ("new", "new_item_recall_at_k"), ("attr", f"attr_new_item_recall_{label}")
    ]:
        paired = frame.sort_values("user_id").dropna(subset=[f"{short}_recall_b0", f"{short}_recall_bf"])
        differences = (paired[f"{short}_recall_bf"] - paired[f"{short}_recall_b0"]).to_numpy(float)
        if not len(differences):
            results.append({
                "attribute": label, "metric": original, "users": 0, "mean_difference": None,
                "ci_95": None, "bonferroni_ci": None, "bootstrap_p": 1.0,
                "legacy_ci_1000": None, "reference_ci_1000": reference_group["bootstrap_ci"][
                    f"{original}_beta_fixed_minus_beta0"
                ], "legacy_ci_difference": None, "legacy_ci_matches": None,
            })
            continue
        means = bootstrap_means(differences, RESAMPLES, SEED)
        legacy = percentile_interval(means[:LEGACY_RESAMPLES], 0.95)
        old = reference_group["bootstrap_ci"][f"{original}_beta_fixed_minus_beta0"]
        delta = {key: float(legacy[key] - old[key]) for key in ["ci_low", "ci_high"]}
        # Independently call the original HS wrapper to verify alignment and its RNG stream.
        indexed = paired.set_index("user_id")
        original_bootstrap = hs.subgroup_bootstrap(
            indexed[f"{short}_recall_bf"], indexed[f"{short}_recall_b0"],
            set(indexed.index), LEGACY_RESAMPLES, SEED,
        )
        for key in ["ci_low", "ci_high"]:
            if not np.isclose(legacy[key], original_bootstrap[key], atol=1e-15, rtol=0):
                raise RuntimeError(f"Legacy bootstrap mismatch for {label}, {original}")
        results.append({
            "attribute": label, "metric": original, "users": len(paired),
            "mean_difference": float(differences.mean()), "ci_95": percentile_interval(means, 0.95),
            "bonferroni_ci": percentile_interval(means, 1 - ALPHA / FAMILY_SIZE),
            "bootstrap_p": bootstrap_p(means), "legacy_ci_1000": legacy,
            "reference_ci_1000": old, "legacy_ci_difference": delta,
            "legacy_ci_matches": all(abs(value) <= TOLERANCE for value in delta.values()),
        })
    return results


def run_followup(profile: str) -> dict[str, object]:
    """Reproduce fixed-beta HS means before running the prespecified eight-test family."""
    started = time.perf_counter()
    config = load_config(profile)
    _check_workspace(config)
    if int(config["evaluation"]["top_k"]) != 10:
        raise ValueError("This analysis requires explore top-10 lists")
    output = Path(config["report_dir"]) / "metrics"
    main_config = {**config, "report_dir": REFERENCE_ROOT / "reports"}
    reference_path = REFERENCE_ROOT / "reports" / "metrics" / f"{profile}_hs_instacart.json"
    reference, reference_source = _read_json(reference_path)
    tuning, tuning_source = _read_json(hs.hs_tuning_path(main_config))
    _, re_source = _read_json(hs.tuning_path(main_config))
    if (reference["profile"], reference["dataset"], reference["split"]) != (profile, hs.DATASET, "eval"):
        raise ValueError("Reference profile, dataset or split mismatch")
    if list(reference["attributes"]) != list(config["health_subgroup"]["attributes"]):
        raise ValueError("Reference attributes differ from the configuration")
    fixed = tuning["scopes"]["food"]["fixed"]
    current_fixed = hs.fixed_parameters(main_config, "food")
    parameter_keys = ["gamma", "kappa", "slots_mode", "global_r"]
    if fixed != reference["scopes"]["food"]["fixed"] or any(
        fixed[key] != current_fixed[key] for key in parameter_keys
    ):
        raise ValueError("HS/reference/current RE fixed parameters differ")
    # The existing helper renders source as absolute when report_dir is in the main checkout.
    fixed_source = Path(fixed["source"])
    if not fixed_source.is_absolute():
        fixed_source = REFERENCE_ROOT / fixed_source
    if fixed_source.resolve() != Path(re_source["path"]).resolve():
        raise ValueError("HS fixed parameters cite a different RE tuning file")
    if fixed["slots_mode"] != config["slots"]["mode"]:
        raise ValueError("Slot mode differs from the original HS evaluation")
    report = {
        "profile": profile, "dataset": hs.DATASET, "split": "eval", "scope": "food",
        "analysis_type": "post-hoc follow-up; overlapping G_a; no target-based tuning",
        "comparison": "beta_fixed minus beta0", "beta0": 0.0,
        "beta_fixed": float(config["health_subgroup"]["beta_fixed"]), "fixed": fixed,
        "sources": {"hs_reference": reference_source, "hs_tuning": tuning_source, "re_tuning": re_source},
        "methods": {
            "resampling_unit": "user within each metric's non-empty-truth subgroup; sorted string user_id",
            "resamples": RESAMPLES, "seed": SEED, "legacy_resamples": LEGACY_RESAMPLES,
            "family_size": FAMILY_SIZE, "alpha": ALPHA, "reproduction_absolute_tolerance": TOLERANCE,
            "p_definition": "min(1, max(1/B, 2*min(P(d*<=0), P(d*>=0))))",
            "empty_truth": "NaN; excluded only from the relevant metric",
            "displacement_population": "G_a or G_H users with non-empty food N_u",
            "net_definition": "gained_attr-lost_attr+gained_other-lost_other; hit count, not recall",
            "combined_attribute_set": "union of all four in-C_a attribute item sets",
            "decision_rule": "difference confirmed only when the 95% CI excludes zero and Holm rejects",
        },
        "environment": {"python": sys.version, "packages": {
            package: version(package) for package in ["numpy", "pandas", "scipy", "scikit-learn", "pyarrow", "PyYAML"]
        }},
        "reproduction": {}, "multiplicity": [], "displacement": {}, "peruser_files": {},
    }
    for key in ["raw_dir", "processed_dir"]:
        report["sources"][key] = str(Path(config[key]).absolute())
    for path in [
        Path(config["project_root"]) / "configs" / f"{profile}.yaml",
        Path(config["processed_dir"]) / hs.DATASET / "items.parquet",
        Path(config["processed_dir"]) / hs.DATASET / "eval_history.parquet",
        Path(config["processed_dir"]) / hs.DATASET / "eval_target.parquet",
    ]:
        report["sources"].setdefault("input_files", []).append({
            "path": str(path.absolute()), "size_bytes": path.stat().st_size, "mtime_ns": path.stat().st_mtime_ns,
        })
    print(f"[hs-followup] {profile}: loading eval split", flush=True)
    items, history, target = load_split(config, hs.DATASET, "eval")
    split = hs.build_health_split(config, hs.DATASET, items, history, target)
    del items, history, target
    scope = build_scope(split.state, "food", config)
    item_sets = hs.attribute_item_sets(split.model, scope, split.state.interactions.items)
    analysis_list = hs.analyses(split, item_sets)
    if len(analysis_list) - 1 != FAMILY_SIZE // 2:
        raise ValueError("The family must contain exactly four attributes")
    repeat = repeat_scores(split.state.interactions, float(fixed["gamma"]))
    frames = {}
    for analysis in analysis_list:
        analysis_started = time.perf_counter()
        label = analysis.label
        group = reference["scopes"]["food"]["analyses"][label]["groups"]["G"]
        if reference["scopes"]["food"]["analyses"][label]["beta_fixed"] != report["beta_fixed"]:
            raise ValueError(f"Reference beta_fixed differs for {label}")
        users = hs.evaluated_users(split.state, scope, analysis.members)
        if len(users) != group["users"]:
            raise ValueError(f"Evaluated-user count differs for {label}")
        print(f"[hs-followup] {label}: {len(users)} evaluated users", flush=True)
        gamma_state = hs._analysis_state(split, scope, analysis, users, repeat, config)
        frame = _per_user(split, scope, analysis, users, gamma_state, fixed, config)
        del gamma_state
        actual, expected = {}, {}
        for suffix, old_label in [("b0", "beta0"), ("bf", "beta_fixed")]:
            for short, metric in [
                ("new", "new_item_recall_at_k"), ("attr", f"attr_new_item_recall_{analysis.primary}")
            ]:
                values = frame[f"{short}_recall_{suffix}"].dropna()
                key = f"{short}_recall_{suffix}"
                actual[key] = float(values.mean()) if len(values) else None
                expected[key] = group["metrics"][old_label][metric]
                if len(values) != group["metrics"][old_label][f"{metric}_users"]:
                    raise ValueError(f"Metric-user count differs for {label}: {key}")
        try:
            report["reproduction"][label] = reproduction_check(actual, expected)
        except ValueError as error:
            _write_json(output / f"{profile}_hs_followup_reproduction_failure.json", {
                "source": reference_source, "attribute": label, "error": str(error),
                "completed_checks": report["reproduction"],
            })
            raise
        frames[label] = frame
        report["displacement"][label] = summarize_displacement(frame)
        report["displacement"][label]["accounting_identity_verified"] = True
        report["displacement"][label]["runtime_seconds"] = time.perf_counter() - analysis_started
        print(f"[hs-followup] {label}: reproduction and accounting passed", flush=True)
    # No inferential tests or per-user artifacts are published until every reproduction gate passes.
    report["reproduction_passed"] = True
    for analysis in analysis_list:
        label, frame = analysis.label, frames[analysis.label]
        output.mkdir(parents=True, exist_ok=True)
        path = output / f"{profile}_hs_followup_peruser_{label}.parquet"
        frame.to_parquet(path, index=False)
        report["peruser_files"][label] = str(path.absolute())
        if label != hs.COMBINED:
            group = reference["scopes"]["food"]["analyses"][label]["groups"]["G"]
            report["multiplicity"].extend(_metric_tests(frame, label, group))
    tests = report["multiplicity"]
    if len(tests) != FAMILY_SIZE:
        raise RuntimeError("Unexpected multiplicity family size")
    adjusted = holm_adjust([row["bootstrap_p"] for row in tests])
    for row, value in zip(tests, adjusted):
        row["holm_p"] = float(value)
        row["holm_reject"] = bool(value <= ALPHA)
        interval = row["ci_95"]
        row["ci_95_excludes_zero"] = bool(interval and interval["ci_low"] * interval["ci_high"] > 0)
        row["difference_confirmed"] = row["ci_95_excludes_zero"] and row["holm_reject"]
        old = row["reference_ci_1000"]
        row["original_ci_excludes_zero"] = bool(old["ci_low"] * old["ci_high"] > 0)
    report["runtime_seconds"] = time.perf_counter() - started
    result_path = output / f"{profile}_hs_followup.json"
    _write_json(result_path, report)
    print(f"[hs-followup] wrote {result_path}; {report['runtime_seconds']:.3f} seconds", flush=True)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=["sample", "medium"], default="medium")
    args = parser.parse_args()
    run_followup(args.profile)


if __name__ == "__main__":
    main()
