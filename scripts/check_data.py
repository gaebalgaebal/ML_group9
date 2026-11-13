# *------------------- 2026 Fall Semester Machine Learning Term-project: Team 9 --------------------*
# Code First written at 2026-11-05
# Code Final Edited at 2026-11-13
# Editor: Yejun Kim, Seongwoo Yang

'''
Purpose : Integrity and descriptive checks for the copied processed data (history/target splits).
Inputs  : The profile name (first command-line argument, default medium) and the Parquet files
          under <data_dir>/<profile>/<dataset>/: items.parquet and, for the fit and eval splits,
          <split>_history.parquet and <split>_target.parquet.
          The data root follows nbr.config.load_config: the NBR_DATA_DIR environment variable
          wins over configs/common.yaml:data_dir (relative to the repository root).
          Only history and target Parquet files are read; nothing is modified.
Outputs : reports/data_check/<profile>.json with the item-table sizes and the per-split
          statistics of every dataset found; the same JSON text is printed.
Usage   :
    python scripts/check_data.py <profile>
    (reads <data_dir>/<profile>/, writes reports/data_check/<profile>.json)
'''

# ---------------------------------------------------------------------------
# imports
# ---------------------------------------------------------------------------
import json
import sys
from pathlib import Path

import pandas as pd

# ---------------------------------------------------------------------------
# repository paths
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from nbr.config import load_config  # noqa: E402

# ---------------------------------------------------------------------------
# constants
# ---------------------------------------------------------------------------
# Hierarchy levels per dataset: L1 -> L2 -> L3 -> item.
LEVELS = {
    "instacart": ["coarse_category", "fine_category", "ingredient_type"],  # department, aisle, ingredient type (aisle fallback)
    "complete_journey": ["DEPARTMENT", "commodity", "fine_category"],      # department, commodity, sub-commodity
}


# ---------------------------------------------------------------------------
# data check
# ---------------------------------------------------------------------------
def check(profile: str, dataset: str, data_dir: Path) -> dict:
    """Item-table sizes and per-split integrity/descriptive statistics of ``dataset`` under <data_dir>/<profile>."""
    # Read the item table of the dataset.
    base = Path(data_dir) / profile / dataset
    items = pd.read_parquet(base / "items.parquet")
    # Record the number of items.
    out = {"items": len(items)}
    # Count the distinct values of every hierarchy level.
    lv = LEVELS[dataset]
    out["level_sizes"] = {c: int(items[c].nunique()) for c in lv}
    # Count the items per type source when the item table has that column.
    if "type_source" in items:
        out["l3_type_source_items"] = items["type_source"].value_counts().to_dict()
    # Build the lookup from item_id to the hierarchy level columns.
    lookup = items.set_index("item_id")[lv]
    for split in ["fit", "eval"]:
        # Read the history and the target rows of the split.
        h = pd.read_parquet(base / f"{split}_history.parquet")
        t = pd.read_parquet(base / f"{split}_target.parquet")
        # Compute the row counts, the user count and the integrity statistics of the split.
        s = {
            "history_rows": len(h), "target_rows": len(t),
            "users": int(h.user_id.nunique()),
            "history_baskets_per_user_median": float(h.groupby("user_id").basket_id.nunique().median()),
            "target_after_history_all": bool(
                (t.groupby("user_id").basket_seq.first() > h.groupby("user_id").basket_seq.max()).all()),
            "history_items_in_item_table": float(h.item_id.isin(items.item_id).mean()),
        }
        # Compute the share of history rows whose item has the type source "mapped".
        if "type_source" in items:
            s["history_rows_l3_mapped"] = float(h.item_id.map(items.set_index("item_id").type_source).eq("mapped").mean())
        # Flag every target row by whether the user bought the item in the history.
        hp = h[["user_id", "item_id"]].drop_duplicates()
        t = t.merge(hp.assign(repeat=True), on=["user_id", "item_id"], how="left")
        t["repeat"] = t["repeat"].eq(True)
        # Compute the share of repeat target rows.
        s["target_repeat_row_share"] = float(t.repeat.mean())
        # Compute the share of target rows whose item occurs in no history row.
        s["target_rows_unseen_globally"] = float((~t.item_id.isin(h.item_id)).mean())
        # Compute the share of users with at least one non-repeat target item.
        s["users_with_new_item"] = float(t.loc[~t.repeat, "user_id"].nunique() / t.user_id.nunique())
        # For new (non-repeat) target items: was the item's category at each level bought by the user before?
        new = t.loc[~t.repeat, ["user_id", "item_id"]].join(lookup, on="item_id")
        hh = h[["user_id", "item_id"]].join(lookup, on="item_id")
        for c in lv:
            # List the (user, category) pairs of the history at this level.
            seen = hh[["user_id", c]].drop_duplicates().assign(seen=True)
            # Compute the share of new target rows whose category the user bought before.
            m = new.merge(seen, on=["user_id", c], how="left")
            s[f"new_item_rows_in_seen_{c}"] = float(m.seen.eq(True).mean())
        # Store the statistics of the split.
        out[split] = s
    return out


# ---------------------------------------------------------------------------
# command-line entry
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # Read the profile name from the command line, with medium as the default.
    profile = sys.argv[1] if len(sys.argv) > 1 else "medium"
    # Take the data root from the profile configuration.
    data_dir = Path(load_config(profile)["data_dir"])
    # Check every dataset whose folder exists under the profile.
    res = {ds: check(profile, ds, data_dir) for ds in LEVELS if (data_dir / profile / ds).exists()}
    # Write the result as JSON and print it.
    dst = ROOT / "reports" / "data_check" / f"{profile}.json"
    dst.write_text(json.dumps(res, indent=2, ensure_ascii=False))
    print(json.dumps(res, indent=2, ensure_ascii=False))
