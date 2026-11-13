# *------------------- 2026 Fall Semester Machine Learning Term-project: Team 9 --------------------*
# Code First written at 2026-11-06
# Code Final Edited at 2026-11-13
# Editor: Yejun Kim, Seongwoo Yang

'''
Purpose : Derive the calendar day of every processed basket (W14; post-hoc, development data,
          not confirmatory).
          Instacart: the day of an order is the cumulative sum of ``days_since_prior_order``
          within the user, ordered by ``order_number`` (first order = 0). The source caps that
          gap at 30 days; the share of capped gaps is reported.
          Complete Journey: ``DAY`` of the (household_key, BASKET_ID) in transaction_data.csv;
          every basket must have exactly one DAY.
          Before a dataset's file is written the script checks how the processed ``basket_id`` maps
          to the raw ids (Instacart: basket_id == str(order_id), user_id == str(user_id),
          basket_seq == order_number; Complete Journey: basket_id == str(BASKET_ID),
          user_id == str(household_key), day non-decreasing in basket_seq) and that every
          (user_id, basket_id) of the fit/eval histories and targets gets exactly one day.
          It stops (RuntimeError) otherwise.
Inputs  : The ZIP archives under --raw-dir (orders.csv of the Instacart archive,
          transaction_data.csv of the Complete Journey archive) and the processed frames
          fit_history, fit_target, eval_history and eval_target of
          <data_dir>/<profile>/<dataset>. Only the development profiles ``sample`` and
          ``medium`` are accepted: holdout profiles are never read.
Outputs : <data_dir>/<profile>/<dataset>/basket_days.parquet (columns user_id, basket_id, day;
          one row per processed basket), where data_dir follows nbr.config (NBR_DATA_DIR, else
          <repo>/data/processed); a provenance block appended to data/SOURCE.md and one line
          per new file appended to data/MANIFEST.sha256; the check report
          reports/posthoc/basket_days_<profile>.json.
Usage   :
    python scripts/make_basket_days.py --raw-dir <ML_group9/data/raw> --profile {sample,medium}
                                       [--datasets instacart,complete_journey] [--overwrite]
                                       [--no-provenance]
'''

from __future__ import annotations

# ---------------------------------------------------------------------------
# imports
# ---------------------------------------------------------------------------
import argparse
import hashlib
import platform
import sys
import time
import zipfile
from datetime import datetime, timezone
from pathlib import Path

# ---------------------------------------------------------------------------
# repository root and project imports
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from nbr.config import resolve_data_dir  # noqa: E402
from nbr.io import git_commit, write_json  # noqa: E402
from nbr.usercf_eval import peak_memory_mib  # noqa: E402

# ---------------------------------------------------------------------------
# constants
# ---------------------------------------------------------------------------
LABEL = "post-hoc, development data, not confirmatory"
ALLOWED_PROFILES = ("sample", "medium")
DATASETS = ("instacart", "complete_journey")
FRAMES = ("fit_history", "fit_target", "eval_history", "eval_target")
INSTACART_MEMBERS = {"orders.csv", "order_products__prior.csv", "products.csv"}
COMPLETE_JOURNEY_MEMBERS = {"transaction_data.csv", "product.csv"}
INSTACART_GAP_CAP = 30
SOURCE_MD = ROOT / "data" / "SOURCE.md"
MANIFEST = ROOT / "data" / "MANIFEST.sha256"
REPORT_DIR = ROOT / "reports" / "posthoc"
KEY = ["user_id", "basket_id"]


# ---------------------------------------------------------------------------
# arguments and paths
# ---------------------------------------------------------------------------
def parse_datasets(value: str) -> list[str]:
    """Comma-separated dataset names -> list in the given order (blanks and repeats dropped); unknown names raise."""
    # Collect the stripped names in input order without blanks and repeats.
    names: list[str] = []
    for item in value.split(","):
        item = item.strip()
        if item and item not in names:
            names.append(item)
    # Reject an empty list and names outside DATASETS.
    unknown = [name for name in names if name not in DATASETS]
    if unknown or not names:
        raise ValueError(f"datasets must be a non-empty subset of {list(DATASETS)}, got {value!r}")
    return names


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Command line of the script (``argv`` defaults to sys.argv); ``datasets`` is returned as a list."""
    # Declare the options of the script.
    parser = argparse.ArgumentParser(description="Derive the calendar day of every processed basket (development profiles).")
    parser.add_argument("--raw-dir", required=True, help="ML_group9/data/raw (instacart.zip and archive.zip)")
    parser.add_argument("--profile", required=True, choices=ALLOWED_PROFILES)
    parser.add_argument("--datasets", default=",".join(DATASETS), type=parse_datasets,
                        help="comma-separated subset of instacart,complete_journey (default: both)")
    parser.add_argument("--overwrite", action="store_true", help="replace an existing basket_days.parquet")
    parser.add_argument("--no-provenance", action="store_true",
                        help="do not append to data/SOURCE.md and data/MANIFEST.sha256")
    # Parse the given arguments.
    return parser.parse_args(argv)


def find_archive(raw_dir: Path, required_members: set[str]) -> Path:
    """The one ZIP under ``raw_dir`` (recursive) whose member names include ``required_members``."""
    # Collect every ZIP below the directory whose members include the required names.
    matches = []
    for path in sorted(Path(raw_dir).rglob("*.zip")):
        with zipfile.ZipFile(path) as archive:
            if required_members.issubset(set(archive.namelist())):
                matches.append(path)
    # Require exactly one such archive.
    if len(matches) != 1:
        raise FileNotFoundError(f"expected exactly one archive containing {sorted(required_members)}, found {matches}")
    return matches[0]


def sha256_of(path: Path) -> str:
    '''SHA-256 hex digest of the file ``path``.'''
    # Start an empty SHA-256 hash.
    digest = hashlib.sha256()
    # Feed the file to the hash in chunks of 1 MiB.
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    # Return the digest as hexadecimal text.
    return digest.hexdigest()


# ---------------------------------------------------------------------------
# processed baskets
# ---------------------------------------------------------------------------
def processed_baskets(folder: Path) -> tuple[pd.DataFrame, dict[str, int]]:
    """Distinct (user_id, basket_id, basket_seq) of the four processed frames of ``folder`` (ids as str).

    Also returns the basket count per frame. A (user_id, basket_id) pair mapping to several
    basket_seq values across frames raises RuntimeError.
    """
    # Read the key and basket_seq columns of each frame and count its distinct baskets.
    parts = []
    counts = {}
    for name in FRAMES:
        frame = pd.read_parquet(folder / f"{name}.parquet", columns=KEY + ["basket_seq"])
        frame = frame.astype({"user_id": str, "basket_id": str, "basket_seq": "int64"}).drop_duplicates()
        counts[name] = int(len(frame))
        parts.append(frame)
    # Stack the frames and keep one row per distinct (user_id, basket_id, basket_seq).
    baskets = pd.concat(parts, ignore_index=True).drop_duplicates().reset_index(drop=True)
    # Stop when one (user_id, basket_id) pair has several basket_seq values.
    if baskets.duplicated(KEY).any():
        raise RuntimeError(f"{folder}: a (user_id, basket_id) pair maps to several basket_seq values")
    # Add the number of distinct baskets to the counts.
    counts["distinct"] = int(len(baskets))
    return baskets, counts


# ---------------------------------------------------------------------------
# raw days
# ---------------------------------------------------------------------------
def instacart_days(orders: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, object]]:
    """Day of every order of ``orders`` (order_id, user_id, order_number, days_since_prior_order).

    Within a user (ordered by order_number) day = cumulative sum of days_since_prior_order with
    the first order at day 0. Checks: order_number is 1..n per user, days_since_prior_order is
    null exactly for the first order. Returns [user_id(str), basket_id(str), day, order_number]
    and statistics including the share of gaps equal to INSTACART_GAP_CAP.
    """
    # Copy the needed columns and order the rows by user and order number.
    frame = orders[["order_id", "user_id", "order_number", "days_since_prior_order"]].copy()
    frame = frame.sort_values(["user_id", "order_number"], kind="mergesort").reset_index(drop=True)
    # Check that order_number runs 1..n within every user.
    position = frame.groupby("user_id", sort=False).cumcount() + 1
    if not (position.to_numpy() == frame["order_number"].to_numpy()).all():
        raise RuntimeError("orders.csv: order_number is not 1..n within every user")
    # Check that no order_id occurs twice.
    if frame.duplicated("order_id").any():
        raise RuntimeError("orders.csv: duplicated order_id")
    # Check that the gap is missing for the first order of a user and present for all others.
    first = frame["order_number"].eq(1)
    gaps = frame["days_since_prior_order"]
    if not gaps[first].isna().all() or gaps[~first].isna().any():
        raise RuntimeError("orders.csv: days_since_prior_order must be null exactly for the first order")
    # Check that the gaps of the non-first orders are finite and not negative.
    non_first = gaps[~first]
    if (non_first < 0).any() or not np.isfinite(non_first).all():
        raise RuntimeError("orders.csv: negative or non-finite days_since_prior_order")
    # Set the day to the rounded cumulative sum of the gaps within the user, first order at 0.
    frame["day"] = gaps.fillna(0.0).groupby(frame["user_id"], sort=False).cumsum().round().astype("int64")
    # Summarise the orders, users and gaps, including the share of gaps equal to the cap.
    stats = {
        "orders": int(len(frame)),
        "users": int(frame["user_id"].nunique()),
        "non_first_orders": int(len(non_first)),
        "gap_cap_days": INSTACART_GAP_CAP,
        "gaps_equal_cap": int(non_first.eq(INSTACART_GAP_CAP).sum()),
        "share_gaps_equal_cap": float(non_first.eq(INSTACART_GAP_CAP).mean()) if len(non_first) else None,
        "gap_mean_days": float(non_first.mean()) if len(non_first) else None,
        "gap_median_days": float(non_first.median()) if len(non_first) else None,
    }
    # Build the output frame with string ids, the day and the order number.
    days = pd.DataFrame({
        "user_id": frame["user_id"].astype(str),
        "basket_id": frame["order_id"].astype(str),
        "day": frame["day"],
        "order_number": frame["order_number"].astype("int64"),
    })
    return days, stats


def complete_journey_days(transactions: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, object]]:
    """DAY per (household_key, BASKET_ID) of ``transactions``; a basket spanning several days raises RuntimeError.

    Returns [user_id(str), basket_id(str), day] and statistics.
    """
    # Keep one row per distinct (household_key, BASKET_ID, DAY).
    rows = transactions[["household_key", "BASKET_ID", "DAY"]].drop_duplicates()
    # Count the distinct DAY values of every basket and stop when a basket has several.
    per_basket = rows.groupby(["household_key", "BASKET_ID"], sort=False)["DAY"].nunique()
    multi = int((per_basket > 1).sum())
    if multi:
        raise RuntimeError(f"transaction_data.csv: {multi} (household_key, BASKET_ID) baskets span several DAY values")
    # Build the output frame with string ids and the integer day.
    days = pd.DataFrame({
        "user_id": rows["household_key"].astype(str).to_numpy(),
        "basket_id": rows["BASKET_ID"].astype(str).to_numpy(),
        "day": rows["DAY"].astype("int64").to_numpy(),
    })
    # Summarise the baskets, the households and the range of days.
    stats = {
        "baskets": int(len(days)),
        "households": int(days["user_id"].nunique()),
        "baskets_with_several_days": multi,
        "day_min": int(days["day"].min()) if len(days) else None,
        "day_max": int(days["day"].max()) if len(days) else None,
    }
    return days, stats


# ---------------------------------------------------------------------------
# id-mapping check
# ---------------------------------------------------------------------------
def check_mapping(dataset: str, baskets: pd.DataFrame, days: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, object]]:
    """Join the processed ``baskets`` (user_id, basket_id, basket_seq) to the raw ``days`` and verify the mapping.

    Every processed basket must match exactly one raw basket (same user_id and basket_id as str);
    Instacart additionally requires basket_seq == order_number; for both datasets the day must be
    non-decreasing in basket_seq within every user. Returns [user_id, basket_id, day] for the
    processed baskets and the check statistics; any failure raises RuntimeError.
    """
    # Stop when the raw days list a (user_id, basket_id) pair twice.
    if days.duplicated(KEY).any():
        raise RuntimeError(f"{dataset}: raw days have duplicated (user_id, basket_id)")
    # Attach the raw day to every processed basket and count the baskets without one.
    merged = baskets.merge(days, on=KEY, how="left", validate="one_to_one")
    missing = int(merged["day"].isna().sum())
    # Record the basket, user, matched and unmatched counts.
    stats: dict[str, object] = {
        "processed_baskets": int(len(baskets)),
        "processed_users": int(baskets["user_id"].nunique()),
        "matched_raw_baskets": int(len(baskets) - missing),
        "unmatched_baskets": missing,
    }
    # Stop with a few examples when a processed basket has no raw day.
    if missing:
        sample = merged.loc[merged["day"].isna(), KEY].head(5).to_dict("records")
        raise RuntimeError(f"{dataset}: {missing} processed baskets have no raw day, e.g. {sample}")
    # Store the day as an integer.
    merged["day"] = merged["day"].astype("int64")
    # For Instacart, require basket_seq to equal order_number for every basket.
    if dataset == "instacart":
        equal = int(merged["basket_seq"].eq(merged["order_number"]).sum())
        stats["basket_seq_equals_order_number"] = equal
        if equal != len(merged):
            raise RuntimeError(f"instacart: basket_seq != order_number for {len(merged) - equal} baskets")
    # Order the baskets within each user and count the steps where the day decreases.
    ordered = merged.sort_values(["user_id", "basket_seq"], kind="mergesort")
    steps = ordered.groupby("user_id", sort=False)["day"].diff().dropna()
    decreasing = int((steps < 0).sum())
    stats["day_decreasing_steps"] = decreasing
    # Stop when the day decreases anywhere.
    if decreasing:
        raise RuntimeError(f"{dataset}: day decreases with basket_seq in {decreasing} consecutive basket pairs")
    # Record the verified id mapping and the range of days.
    stats["mapping"] = {
        "instacart": "basket_id == str(order_id), user_id == str(user_id), basket_seq == order_number",
        "complete_journey": "basket_id == str(BASKET_ID), user_id == str(household_key)",
    }[dataset]
    stats["day_min"] = int(merged["day"].min())
    stats["day_max"] = int(merged["day"].max())
    # Return the key columns with the day, and the statistics.
    return merged[KEY + ["day"]], stats


# ---------------------------------------------------------------------------
# raw readers
# ---------------------------------------------------------------------------
def read_instacart_orders(archive_path: Path) -> pd.DataFrame:
    '''Read the needed columns of orders.csv from the Instacart ZIP ``archive_path``.'''
    # Open orders.csv in the archive and parse the listed columns, four of them with fixed dtypes.
    with zipfile.ZipFile(archive_path) as archive, archive.open("orders.csv") as stream:
        return pd.read_csv(
            stream,
            usecols=["order_id", "user_id", "eval_set", "order_number", "days_since_prior_order"],
            dtype={"order_id": "int64", "user_id": "int64", "eval_set": "string", "order_number": "int64"},
        )


def read_complete_journey_transactions(archive_path: Path) -> pd.DataFrame:
    '''Read three columns of transaction_data.csv from the Complete Journey ZIP ``archive_path``.'''
    # Open transaction_data.csv inside the archive and parse the three columns as int64.
    with zipfile.ZipFile(archive_path) as archive, archive.open("transaction_data.csv") as stream:
        return pd.read_csv(
            stream,
            usecols=["household_key", "BASKET_ID", "DAY"],
            dtype={"household_key": "int64", "BASKET_ID": "int64", "DAY": "int64"},
        )


# ---------------------------------------------------------------------------
# provenance
# ---------------------------------------------------------------------------
def append_manifest(manifest: Path, entries: list[tuple[str, str]]) -> list[str]:
    """Append ``<sha256>  <relative path>`` lines for paths not yet listed; a listed path with another digest raises."""
    # Read the digest of every path already listed in the manifest.
    existing = {}
    if manifest.exists():
        for line in manifest.read_text(encoding="utf-8").splitlines():
            if line.strip():
                digest, _, rel = line.partition("  ")
                existing[rel.strip()] = digest.strip()
    # Keep the entries of unlisted paths and stop when a listed path has another digest.
    added = []
    for rel, digest in entries:
        if rel in existing:
            if existing[rel] != digest:
                raise RuntimeError(f"{manifest}: {rel} is listed with digest {existing[rel]}, new file has {digest}")
            continue
        added.append(f"{digest}  {rel}")
    # Append the new lines to the manifest, after a newline when the text lacks a final one.
    if added:
        text = manifest.read_text(encoding="utf-8") if manifest.exists() else ""
        if text and not text.endswith("\n"):
            text += "\n"
        manifest.write_text(text + "\n".join(added) + "\n", encoding="utf-8")
    return added


def provenance_block(profile: str, report: dict[str, object], command: str) -> str:
    """Markdown block for data/SOURCE.md describing the basket-day files of ``profile``."""
    # Start the block with a heading that carries the profile, the report date and the label.
    date = report["utc_time"][:10]
    lines = [f"## basket_days: {profile} ({date}, W14; {LABEL})"]
    # Add the line that lists the files with their row counts and the report path.
    files = ", ".join(f"`{entry['manifest_path']}` ({entry['rows']:,} rows)" for entry in report["datasets"].values())
    lines.append(f"- Files: {files}; columns user_id, basket_id, day; one row per (user_id, basket_id) of the "
                 f"fit/eval histories and targets. Checksums in `MANIFEST.sha256`. Report: `{report['report_path']}`.")
    # Add one line per dataset with the day rule and the result of the id-mapping check.
    for dataset, entry in report["datasets"].items():
        raw, check = entry["raw"], entry["mapping_check"]
        if dataset == "instacart":
            lines.append(
                f"- Instacart: day = cumulative sum of `days_since_prior_order` within user by `order_number` "
                f"(first order = 0), from `orders.csv` of `{entry['archive']}`. The source caps the gap at "
                f"{raw['gap_cap_days']} days: {100 * raw['share_gaps_equal_cap']:.2f} % of the selected users' "
                f"{raw['non_first_orders']:,} non-first orders have the capped gap "
                f"({100 * entry['raw_all_users']['share_gaps_equal_cap']:.2f} % over all orders of the file). "
                f"Id mapping verified ({check['mapping']}) for all {check['processed_baskets']:,} baskets of "
                f"{check['processed_users']:,} users; {check['unmatched_baskets']} unmatched; day non-decreasing in basket_seq.")
        else:
            lines.append(
                f"- Complete Journey: `DAY` of the (`household_key`, `BASKET_ID`) in `transaction_data.csv` of "
                f"`{entry['archive']}`; {raw['baskets_with_several_days']} baskets of the selected households span "
                f"several days (checked over all households: {entry['raw_all_users']['baskets_with_several_days']}). "
                f"Id mapping verified ({check['mapping']}) for all {check['processed_baskets']:,} baskets of "
                f"{check['processed_users']:,} households; {check['unmatched_baskets']} unmatched; day non-decreasing in basket_seq.")
    # Add the command, the code commit, the package versions, the runtime and the peak memory.
    env = report["environment"]
    lines.append(f"- Command: `{command}`; code commit {report['git_commit']}; {env['python']}, pandas {env['pandas']}, "
                 f"numpy {env['numpy']}, pyarrow {env['pyarrow']}; runtime {report['runtime_seconds']:.1f} s, "
                 f"peak working set {report['peak_memory_mib']:.0f} MiB (in-process).")
    # Join the lines into one text that ends with a newline.
    return "\n".join(lines) + "\n"


def append_source(source: Path, profile: str, block: str) -> bool:
    """Append ``block`` to SOURCE.md unless a block for ``profile`` is already present (returns whether it was added)."""
    # Read the current text of the file, or use an empty text when it does not exist.
    text = source.read_text(encoding="utf-8") if source.exists() else ""
    # Leave the file unchanged when it already has a block for this profile.
    if f"## basket_days: {profile} (" in text:
        return False
    # Append the block after one blank line.
    if text and not text.endswith("\n"):
        text += "\n"
    source.write_text(text + "\n" + block, encoding="utf-8")
    return True


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def run_dataset(dataset: str, raw_dir: Path, folder: Path, overwrite: bool) -> dict[str, object]:
    """Derive, check and write the basket days of one dataset; returns the report entry."""
    # Refuse to replace an existing output file unless overwrite is set.
    out = folder / "basket_days.parquet"
    if out.exists() and not overwrite:
        raise FileExistsError(f"{out} exists; pass --overwrite to replace it")
    # Load the processed baskets and the set of their users.
    baskets, counts = processed_baskets(folder)
    users = set(baskets["user_id"])
    # Derive the raw days twice: for all users (statistics only) and for the selected users.
    if dataset == "instacart":
        archive = find_archive(raw_dir, INSTACART_MEMBERS)
        orders = read_instacart_orders(archive)
        _, stats_all = instacart_days(orders)
        selected = orders[orders["user_id"].astype(str).isin(users)]
        days, stats = instacart_days(selected)
        member = "orders.csv"
    else:
        archive = find_archive(raw_dir, COMPLETE_JOURNEY_MEMBERS)
        transactions = read_complete_journey_transactions(archive)
        _, stats_all = complete_journey_days(transactions)
        selected = transactions[transactions["household_key"].astype(str).isin(users)]
        days, stats = complete_journey_days(selected)
        member = "transaction_data.csv"
    # Verify the id mapping and get the day of every processed basket.
    frame, check = check_mapping(dataset, baskets, days)
    # Sort the rows and fix the column dtypes.
    frame = frame.sort_values(["user_id", "day", "basket_id"], kind="mergesort").reset_index(drop=True)
    frame = frame.astype({"user_id": "string", "basket_id": "string", "day": "int64"})
    # Write the Parquet file, creating the folder when needed.
    out.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(out, index=False)
    # Return the report entry of this dataset, including the SHA-256 of the written file.
    return {
        "archive": archive.name,
        "raw_member": member,
        "processed_frames": counts,
        "raw": stats,
        "raw_all_users": stats_all,
        "mapping_check": check,
        "output": str(out),
        "rows": int(len(frame)),
        "users": int(frame["user_id"].nunique()),
        "sha256": sha256_of(out),
    }


def main(argv: list[str] | None = None) -> None:
    '''Derive the basket days of every dataset named in ``argv`` and write the files and reports.'''
    # Parse the command line and check the profile once more.
    args = parse_args(argv)
    if args.profile not in ALLOWED_PROFILES:  # choices already restrict this; kept as an explicit guard
        raise ValueError(f"profile must be one of {ALLOWED_PROFILES}")
    # Start the timer and resolve the data and raw directories.
    t0 = time.perf_counter()
    data_dir = resolve_data_dir()
    raw_dir = Path(args.raw_dir)
    # Rebuild the text of the command line for the provenance records.
    command = f"python scripts/make_basket_days.py --raw-dir {args.raw_dir} --profile {args.profile}"
    if args.datasets != list(DATASETS):
        command += f" --datasets {','.join(args.datasets)}"
    # Process each requested dataset and print the result of its mapping check.
    entries = {}
    for dataset in args.datasets:
        # Derive, check and write the basket days of this dataset.
        folder = data_dir / args.profile / dataset
        print(f"[{dataset}] reading processed baskets from {folder}")
        entries[dataset] = run_dataset(dataset, raw_dir, folder, args.overwrite)
        # Record the path under which the file is listed in the manifest.
        rel = f"processed/{args.profile}/{dataset}/basket_days.parquet"
        entries[dataset]["manifest_path"] = rel
        # Print the verified mapping and the matched, unmatched and written counts.
        check = entries[dataset]["mapping_check"]
        print(f"[{dataset}] {check['mapping']}; {check['processed_baskets']:,} baskets matched, "
              f"{check['unmatched_baskets']} unmatched; wrote {entries[dataset]['rows']:,} rows -> {entries[dataset]['output']}")
        # For Instacart, also print the share of gaps equal to the cap.
        if dataset == "instacart":
            raw = entries[dataset]["raw"]
            print(f"[{dataset}] share of gaps equal to {raw['gap_cap_days']} days: {raw['share_gaps_equal_cap']:.4f} "
                  f"({raw['gaps_equal_cap']:,} of {raw['non_first_orders']:,} non-first orders)")
    # Assemble the report with the command, commit, time, dataset entries and environment.
    report_path = REPORT_DIR / f"basket_days_{args.profile}.json"
    report = {
        "label": LABEL,
        "task": "W14 basket days",
        "profile": args.profile,
        "data_dir": str(data_dir),
        "raw_dir": str(raw_dir),
        "command": command,
        "git_commit": git_commit(ROOT),
        "utc_time": datetime.now(timezone.utc).isoformat(),
        "datasets": entries,
        "report_path": str(report_path.relative_to(ROOT)).replace("\\", "/"),
        "environment": {"python": f"Python {platform.python_version()} ({platform.system()})",
                        "pandas": pd.__version__, "numpy": np.__version__,
                        "pyarrow": __import__("pyarrow").__version__},
        "runtime_seconds": time.perf_counter() - t0,
        "peak_memory_mib": peak_memory_mib(),
    }
    # Unless disabled, append the checksums to the manifest and the block to SOURCE.md.
    if not args.no_provenance:
        added = append_manifest(MANIFEST, [(e["manifest_path"], e["sha256"]) for e in entries.values()])
        report["manifest_lines_added"] = added
        block = provenance_block(args.profile, report, command)
        report["source_md_appended"] = append_source(SOURCE_MD, args.profile, block)
        print(f"MANIFEST.sha256: {len(added)} line(s) added; SOURCE.md appended: {report['source_md_appended']}")
    # Write the JSON report and print its path, the runtime and the peak memory.
    write_json(report_path, report, {"repo_root": ROOT, "profile": args.profile})
    print(f"report -> {report_path}; runtime {report['runtime_seconds']:.1f} s; peak working set {report['peak_memory_mib']:.0f} MiB")


# ---------------------------------------------------------------------------
# command-line entry
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    main()
