# *------------------- 2026 Fall Semester Machine Learning Term-project: Team 9 --------------------*
# Code First written at 2026-11-05
# Code Final Edited at 2026-11-13
# Editor: Yejun Kim, Seongwoo Yang

'''
Purpose : Build an Instacart confirmatory profile: ``--size`` train-set users disjoint from
          earlier profiles.
          The preprocessing code is the unmodified snapshot of ML_group9 release/term-project
          (vendor/ml_group9_e11b365). This wrapper only (1) loads the medium configuration,
          (2) removes previously used users from the eligible set before the seeded draw,
          (3) adds ingredient-type columns with the same mapping functions, and (4) writes a
          selection report. Complete Journey is not regenerated (all eligible households are
          already in medium).
          Excluded profiles: a profile without a selection report (medium, sample) is read from
          <data_dir>/<profile>/instacart/eval_target.parquet (user_id column only). A profile
          that has a report in reports/holdout_prep, i.e. one built by this script (holdout),
          is never read: its seeded draw is re-run from the eligible users minus its own
          exclusions, and the result must match the user-id digest recorded in the report
          (RuntimeError otherwise, without any fallback to the profile's files).
Inputs  : The raw directory --raw-dir (the Instacart ZIP, found by its member names, and
          mapping/), the vendored configs/medium.yaml, the eval targets and selection reports
          of the excluded profiles (see above) and the medium Instacart items.parquet and
          attribute_categories.parquet.
Outputs : Parquet files under <data_dir>/<name>/instacart, where data_dir follows nbr.config
          (the NBR_DATA_DIR environment variable, else <repo>/data/processed); intermediate
          files under artifacts/<name>; the selection report
          reports/holdout_prep/<name>_selection.json. The defaults
          (--name holdout --exclude medium,sample) are the v1.2 holdout draw.
Usage   :
    python scripts/make_holdout.py --raw-dir <ML_group9/data/raw> [--name holdout]
                                   [--exclude medium,sample]
                                   [--size 20000] [--seed 42] [--expect-remaining N]
'''

from __future__ import annotations

# ---------------------------------------------------------------------------
# imports
# ---------------------------------------------------------------------------
import argparse
import hashlib
import json
import platform
import sys
import time
import zipfile
from pathlib import Path

try:
    import resource  # POSIX only: peak RSS of this process
except ImportError:  # Windows: the peak working set is read instead (nbr.usercf_eval.peak_memory_mib)
    resource = None

# ---------------------------------------------------------------------------
# vendored code path and project imports
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
VENDOR = ROOT / "vendor" / "ml_group9_e11b365"
sys.path.insert(0, str(VENDOR / "src"))
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import yaml  # noqa: E402

from ml_group9 import data as prep  # noqa: E402
from ml_group9.ingredient_types import assign_types, read_mapping  # noqa: E402
from nbr.config import resolve_data_dir  # noqa: E402

# ---------------------------------------------------------------------------
# constants
# ---------------------------------------------------------------------------
DEFAULT_NAME = "holdout"
DEFAULT_EXCLUDE = "medium,sample"
REPORT_DIR = ROOT / "reports" / "holdout_prep"
MEDIUM_SIZE, MEDIUM_SEED = 20000, 42  # the vendored medium draw (sanity check only)
ORIGINAL_SAMPLE_IDS = prep._sample_ids  # the unmodified seeded draw; prepare_instacart is patched around it


# ---------------------------------------------------------------------------
# arguments and paths
# ---------------------------------------------------------------------------
def parse_exclude(value: str) -> list[str]:
    """Comma-separated profile names -> list in the given order (blank entries and repeats dropped)."""
    # Collect the stripped names in input order without blanks and repeats.
    names: list[str] = []
    for item in value.split(","):
        item = item.strip()
        if item and item not in names:
            names.append(item)
    return names


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Command line of the script (``argv`` defaults to sys.argv); ``exclude`` is returned as a list of names."""
    # Declare the options of the script.
    parser = argparse.ArgumentParser(description="Draw an Instacart profile disjoint from earlier profiles.")
    parser.add_argument("--raw-dir", required=True, help="ML_group9/data/raw (instacart.zip and mapping/)")
    parser.add_argument("--name", default=DEFAULT_NAME,
                        help="profile name: <data_dir>/<name>, artifacts/<name>, "
                             "reports/holdout_prep/<name>_selection.json (default: %(default)s)")
    parser.add_argument("--exclude", default=DEFAULT_EXCLUDE,
                        help="comma-separated profiles whose users are removed before the draw (default: %(default)s)")
    parser.add_argument("--size", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--expect-remaining", type=int, default=None,
                        help="stop before the draw (nothing written) unless exactly this many users remain")
    # Parse the given arguments.
    args = parser.parse_args(argv)
    # Reject an empty name and one that is not alphanumeric after removing underscores and hyphens.
    if not args.name or not args.name.replace("_", "").replace("-", "").isalnum():
        parser.error(f"invalid --name {args.name!r}: letters, digits, underscore and hyphen only")
    # Turn the exclusion text into a list and require at least one name.
    args.exclude = parse_exclude(args.exclude)
    if not args.exclude:
        parser.error("--exclude needs at least one profile name")
    return args


def report_path(name: str, report_dir: Path = REPORT_DIR) -> Path:
    """<report_dir>/<name>_selection.json ("holdout" -> holdout_selection.json, unchanged since v1.2)."""
    return Path(report_dir) / f"{name}_selection.json"


def output_dirs(name: str, data_dir: Path) -> tuple[Path, Path]:
    """(processed_dir, artifact_dir) of profile ``name``: <data_dir>/<name> and <repo>/artifacts/<name>."""
    return Path(data_dir) / name, ROOT / "artifacts" / name


# ---------------------------------------------------------------------------
# user sets of earlier profiles
# ---------------------------------------------------------------------------
def ids_digest(ids: set[str]) -> str:
    """SHA-256 of the sorted, newline-joined user ``ids`` (provenance of the drawn user set)."""
    return hashlib.sha256("\n".join(sorted(ids)).encode()).hexdigest()


def profile_users(profile: str, data_dir: Path) -> set[str]:
    """Users of a prepared profile: the user_id column of its Instacart eval target (one target basket per user)."""
    # Build the path of the Instacart eval target of the profile.
    path = Path(data_dir) / profile / "instacart" / "eval_target.parquet"
    # Read only the user_id column and return the ids as a set of strings.
    return set(pd.read_parquet(path, columns=["user_id"])["user_id"].astype(str))


def reconstruct_users(profile: str, eligible: set[str], data_dir: Path, report_dir: Path) -> set[str]:
    """Re-run the seeded draw of ``profile`` (built by this script) and check it against its selection report.

    The draw uses ``eligible`` minus the profile's own exclusions (``excluded_profiles`` of the report; the
    v1.2 report predates that key and used the defaults) with the recorded seed and size. Raises RuntimeError
    when the size or user-id digest differs; the profile's data files are never read.
    """
    # Load the selection report of the profile.
    report = json.loads(report_path(profile, report_dir).read_text(encoding="utf-8"))
    # Take the profile's own exclusions from the report, or the defaults when the key is absent.
    own_exclusions = list(report.get("excluded_profiles", parse_exclude(DEFAULT_EXCLUDE)))
    # Build the union of the users of those excluded profiles.
    excluded = exclusion_users(own_exclusions, eligible, data_dir, report_dir)
    # Repeat the seeded draw with the recorded size and seed.
    drawn = ORIGINAL_SAMPLE_IDS(set(eligible) - excluded, int(report["size"]), int(report["seed"]))
    # Stop when the size or the user-id digest of the draw differs from the report.
    digest = ids_digest(drawn)
    if len(drawn) != int(report["selected_users"]) or digest != report["selected_users_sha256"]:
        raise RuntimeError(
            f"profile '{profile}': the reconstructed draw ({len(drawn)} users, sha256 {digest}) does not match "
            f"its selection report ({report['selected_users']} users, sha256 {report['selected_users_sha256']})")
    return drawn


def exclusion_sets(profiles: list[str], eligible: set[str], data_dir: Path,
                   report_dir: Path = REPORT_DIR) -> dict[str, set[str]]:
    """User set of every profile in ``profiles``: read from its eval target when it has no selection report
    in ``report_dir``, reconstructed from the report (never read) otherwise."""
    sets: dict[str, set[str]] = {}
    # Reconstruct a profile that has a selection report and read the eval target of any other.
    for profile in profiles:
        if report_path(profile, report_dir).exists():
            sets[profile] = reconstruct_users(profile, eligible, data_dir, report_dir)
        else:
            sets[profile] = profile_users(profile, data_dir)
    return sets


def exclusion_users(profiles: list[str], eligible: set[str], data_dir: Path,
                    report_dir: Path = REPORT_DIR) -> set[str]:
    """Union of the user sets of ``profiles`` (see exclusion_sets)."""
    return set().union(*exclusion_sets(profiles, eligible, data_dir, report_dir).values())


def check_remaining(remaining: int, expected: int | None) -> None:
    """Stop (SystemExit) when ``expected`` is given and differs from the ``remaining`` user count."""
    if expected is not None and remaining != expected:
        raise SystemExit(f"{remaining} users remain after the exclusions, expected {expected}; nothing written")


# ---------------------------------------------------------------------------
# peak memory
# ---------------------------------------------------------------------------
def peak_memory_mb() -> tuple[float | None, str | None]:
    """(peak resident memory of this process in MB, counter name): ru_maxrss on POSIX, the peak working set on Windows."""
    # On POSIX, return the ru_maxrss counter of this process divided by 1024.
    if resource is not None:
        return round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 0), "ru_maxrss"
    # Otherwise return the peak working set, or (None, None) when it is unavailable.
    from nbr.usercf_eval import peak_memory_mib  # imported lazily: Windows only
    value = peak_memory_mib()
    return (round(value, 0), "PeakWorkingSetSize") if value is not None else (None, None)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> None:
    """Draw the profile's users with the vendored preprocessing code (excluding earlier profiles) and write the selection report."""
    # Parse the command line and resolve the raw, data and output directories.
    args = parse_args(argv)
    start = time.time()
    raw_dir = Path(args.raw_dir)
    data_dir = resolve_data_dir()
    processed_dir, artifact_dir = output_dirs(args.name, data_dir)

    # Load the vendored medium configuration and overwrite the seed, size, profile and paths.
    config = yaml.safe_load((VENDOR / "configs" / "medium.yaml").read_text(encoding="utf-8"))
    config["seed"] = args.seed
    config["sample_users"]["instacart"] = args.size
    config["profile"] = args.name
    config["project_root"] = ROOT
    config["raw_dir"] = raw_dir
    config["processed_dir"] = processed_dir
    config["artifact_dir"] = artifact_dir
    config["report_dir"] = REPORT_DIR

    # Start the selection log with the profile name, data directory and excluded profiles.
    log: dict[str, object] = {"name": args.name, "data_dir": str(data_dir), "excluded_profiles": list(args.exclude)}

    # Eligible users (public train-set order) from the raw archive; prepare_instacart reads it again for the draw.
    archive = prep.discover_archive(raw_dir, prep.INSTACART_MEMBERS)
    with zipfile.ZipFile(archive) as zf:
        orders = prep._read_csv(zf, "orders.csv", dtype={"order_id": "int64", "user_id": "string"})
    eligible = {str(v) for v in orders.loc[orders["eval_set"].eq("train"), "user_id"]}
    # Release the orders frame.
    del orders

    # Sanity check: the unmodified draw reproduces the medium user set.
    log["medium_draw_reproduced"] = ORIGINAL_SAMPLE_IDS(eligible, MEDIUM_SIZE, MEDIUM_SEED) == profile_users("medium", data_dir)

    # Collect the user sets of the excluded profiles and remove them from the eligible users.
    sets = exclusion_sets(args.exclude, eligible, data_dir, REPORT_DIR)
    excluded = set().union(*sets.values())
    remaining = eligible - excluded
    # Log the eligible, excluded and remaining user counts and the source of each exclusion set.
    log["eligible_train_users"] = len(eligible)
    log["excluded_users"] = len(excluded)
    log["remaining_users"] = len(remaining)
    log["exclusion_sources"] = {
        profile: {"source": "reconstructed_from_report" if report_path(profile).exists() else "eval_target_user_ids",
                  "users": len(ids), "sha256": ids_digest(ids)}
        for profile, ids in sets.items()}
    # Stop before the draw when the remaining count differs from the expected one.
    check_remaining(len(remaining), args.expect_remaining)

    # Define the draw that removes the excluded users before the unmodified seeded draw.
    def sample_excluding(ids, size, seed):
        '''Seeded draw of ``size`` users from ``ids`` minus the excluded users, with ``seed``.'''
        return ORIGINAL_SAMPLE_IDS({str(v) for v in ids} - excluded, size, seed)

    # Run the vendored Instacart preparation with the replaced draw, then restore the original.
    prep._sample_ids = sample_excluding
    try:
        summary = prep.prepare_instacart(config)
    finally:
        prep._sample_ids = ORIGINAL_SAMPLE_IDS

    # Add the ingredient-type columns to the item table with the vendored mapping functions.
    base = processed_dir / "instacart"
    settings = config["hierarchical"]["mapping"]
    mapping, provenance = read_mapping(raw_dir / "mapping" / Path(settings["path"]).name,
                                       str(settings["expected_version"]))
    items = assign_types(pd.read_parquet(base / "items.parquet"), mapping, list(settings["auto_statuses"]))
    items.to_parquet(base / "items.parquet", index=False)

    # Read the selected users and the medium item and category tables for the report checks.
    selected = profile_users(args.name, data_dir)  # user ids only
    medium_base = Path(data_dir) / "medium" / "instacart"
    medium_items = pd.read_parquet(medium_base / "items.parquet")
    medium_cats = pd.read_parquet(medium_base / "attribute_categories.parquet")
    # Read the peak memory of this process.
    peak_mb, peak_source = peak_memory_mb()
    # Log the draw, its overlap with the excluded profiles, the comparisons with medium and the run.
    log.update({
        "vendor_commit": (VENDOR / "COMMIT").read_text().strip(),
        "seed": args.seed, "size": args.size,
        "selected_users": len(selected),
        "overlap_with_used_profiles": len(selected & excluded),
        "overlap_by_profile": {profile: len(selected & ids) for profile, ids in sets.items()},
        "selected_users_sha256": ids_digest(selected),
        "items_equal_medium": bool(items.reset_index(drop=True).equals(medium_items.reset_index(drop=True))),
        "attribute_categories_equal_medium": bool(
            pd.read_parquet(base / "attribute_categories.parquet").equals(medium_cats)),
        "mapping": {k: v for k, v in provenance.items() if k != "path"} if isinstance(provenance, dict) else str(provenance),
        "prepare_summary": {k: v for k, v in summary.items() if k != "attributes"},
        "runtime_s": round(time.time() - start, 1),
        "peak_rss_mb": peak_mb, "peak_memory_source": peak_source,
        "python": platform.python_version(), "pandas": pd.__version__, "numpy": np.__version__,
    })
    # Write the selection report as JSON and print it.
    out = report_path(args.name)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(log, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    print(json.dumps(log, indent=2, ensure_ascii=False, default=str))


# ---------------------------------------------------------------------------
# command-line entry
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    main()
