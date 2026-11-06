"""Build the Instacart holdout profile: 20,000 train-set users disjoint from the medium and sample profiles.

The preprocessing code is the unmodified snapshot of ML_group9 release/term-project (vendor/ml_group9_e11b365).
This wrapper only (1) loads the medium configuration, (2) removes previously used users from the eligible set
before the seeded draw, (3) adds ingredient-type columns with the same mapping functions, and (4) writes a
selection report. Complete Journey is not regenerated (all eligible households are already in medium).

Usage:
    python scripts/make_holdout.py --raw-dir <ML_group9/data/raw> [--size 20000] [--seed 42]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
import time
from pathlib import Path

try:
    import resource  # POSIX only: peak RSS of this process
except ImportError:  # Windows: peak_rss_mb is recorded as null; the extraction itself is unchanged
    resource = None

ROOT = Path(__file__).resolve().parents[1]
VENDOR = ROOT / "vendor" / "ml_group9_e11b365"
sys.path.insert(0, str(VENDOR / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import yaml  # noqa: E402

from ml_group9 import data as prep  # noqa: E402
from ml_group9.ingredient_types import assign_types, read_mapping  # noqa: E402

USED_PROFILES = ["medium", "sample"]


def used_users() -> set[str]:
    """Users of earlier profiles (every selected user has exactly one eval target basket)."""
    users: set[str] = set()
    for profile in USED_PROFILES:
        path = ROOT / "data" / "processed" / profile / "instacart" / "eval_target.parquet"
        users |= set(pd.read_parquet(path, columns=["user_id"])["user_id"].astype(str))
    return users


def ids_digest(ids: set[str]) -> str:
    """SHA-256 of the sorted, newline-joined user ``ids`` (provenance of the drawn user set)."""
    return hashlib.sha256("\n".join(sorted(ids)).encode()).hexdigest()


def main() -> None:
    """Draw the holdout users with the vendored preprocessing code (excluding earlier profiles) and write the selection report."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", required=True)
    parser.add_argument("--size", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    start = time.time()

    config = yaml.safe_load((VENDOR / "configs" / "medium.yaml").read_text(encoding="utf-8"))
    config["seed"] = args.seed
    config["sample_users"]["instacart"] = args.size
    config["profile"] = "holdout"
    config["project_root"] = ROOT
    config["raw_dir"] = Path(args.raw_dir)
    config["processed_dir"] = ROOT / "data" / "processed" / "holdout"
    config["artifact_dir"] = ROOT / "artifacts" / "holdout"
    config["report_dir"] = ROOT / "reports" / "holdout_prep"

    excluded = used_users()
    original_sample_ids = prep._sample_ids
    log: dict[str, object] = {}

    # Sanity check: the unmodified draw reproduces the medium user set.
    import zipfile
    archive = prep.discover_archive(Path(args.raw_dir), prep.INSTACART_MEMBERS)
    with zipfile.ZipFile(archive) as zf:
        orders = prep._read_csv(zf, "orders.csv", dtype={"order_id": "int64", "user_id": "string"})
    eligible = orders.loc[orders["eval_set"].eq("train"), "user_id"]
    medium_users = set(pd.read_parquet(ROOT / "data/processed/medium/instacart/eval_target.parquet", columns=["user_id"])["user_id"].astype(str))
    log["medium_draw_reproduced"] = original_sample_ids(eligible, 20000, 42) == medium_users

    def sample_excluding(ids, size, seed):
        remaining = {str(v) for v in ids} - excluded
        log["eligible_train_users"] = len({str(v) for v in ids})
        log["excluded_users"] = len(excluded)
        log["remaining_users"] = len(remaining)
        return original_sample_ids(remaining, size, seed)

    prep._sample_ids = sample_excluding
    summary = prep.prepare_instacart(config)
    prep._sample_ids = original_sample_ids

    base = Path(config["processed_dir"]) / "instacart"
    settings = config["hierarchical"]["mapping"]
    mapping, provenance = read_mapping(Path(args.raw_dir) / "mapping" / Path(settings["path"]).name,
                                       str(settings["expected_version"]))
    items = assign_types(pd.read_parquet(base / "items.parquet"), mapping, list(settings["auto_statuses"]))
    items.to_parquet(base / "items.parquet", index=False)

    selected = set(pd.read_parquet(base / "eval_target.parquet", columns=["user_id"])["user_id"].astype(str))
    medium_items = pd.read_parquet(ROOT / "data/processed/medium/instacart/items.parquet")
    medium_cats = pd.read_parquet(ROOT / "data/processed/medium/instacart/attribute_categories.parquet")
    log.update({
        "vendor_commit": (VENDOR / "COMMIT").read_text().strip(),
        "seed": args.seed, "size": args.size,
        "selected_users": len(selected),
        "overlap_with_used_profiles": len(selected & excluded),
        "selected_users_sha256": ids_digest(selected),
        "items_equal_medium": bool(items.reset_index(drop=True).equals(medium_items.reset_index(drop=True))),
        "attribute_categories_equal_medium": bool(
            pd.read_parquet(base / "attribute_categories.parquet").equals(medium_cats)),
        "mapping": {k: v for k, v in provenance.items() if k != "path"} if isinstance(provenance, dict) else str(provenance),
        "prepare_summary": {k: v for k, v in summary.items() if k != "attributes"},
        "runtime_s": round(time.time() - start, 1),
        "peak_rss_mb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 0) if resource is not None else None,
        "python": platform.python_version(), "pandas": pd.__version__, "numpy": np.__version__,
    })
    out = Path(config["report_dir"]) / "holdout_selection.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(log, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    print(json.dumps(log, indent=2, ensure_ascii=False, default=str))


if __name__ == "__main__":
    main()
