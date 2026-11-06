"""Integrity and descriptive checks for the copied processed data (history/target splits).

Usage: python scripts/check_data.py <profile>   (reads <data_dir>/<profile>/, writes reports/data_check/<profile>.json)
The data root follows nbr.config.load_config: the NBR_DATA_DIR environment variable wins over
configs/common.yaml:data_dir (relative to the repository root). Only history and target Parquet files
are read; nothing is modified.
"""
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from nbr.config import load_config  # noqa: E402

# Hierarchy levels per dataset: L1 -> L2 -> L3 -> item.
LEVELS = {
    "instacart": ["coarse_category", "fine_category", "ingredient_type"],  # department, aisle, ingredient type (aisle fallback)
    "complete_journey": ["DEPARTMENT", "commodity", "fine_category"],      # department, commodity, sub-commodity
}


def check(profile: str, dataset: str, data_dir: Path) -> dict:
    """Item-table sizes and per-split integrity/descriptive statistics of ``dataset`` under <data_dir>/<profile>."""
    base = Path(data_dir) / profile / dataset
    items = pd.read_parquet(base / "items.parquet")
    out = {"items": len(items)}
    lv = LEVELS[dataset]
    out["level_sizes"] = {c: int(items[c].nunique()) for c in lv}
    if "type_source" in items:
        out["l3_type_source_items"] = items["type_source"].value_counts().to_dict()
    lookup = items.set_index("item_id")[lv]
    for split in ["fit", "eval"]:
        h = pd.read_parquet(base / f"{split}_history.parquet")
        t = pd.read_parquet(base / f"{split}_target.parquet")
        s = {
            "history_rows": len(h), "target_rows": len(t),
            "users": int(h.user_id.nunique()),
            "history_baskets_per_user_median": float(h.groupby("user_id").basket_id.nunique().median()),
            "target_after_history_all": bool(
                (t.groupby("user_id").basket_seq.first() > h.groupby("user_id").basket_seq.max()).all()),
            "history_items_in_item_table": float(h.item_id.isin(items.item_id).mean()),
        }
        if "type_source" in items:
            s["history_rows_l3_mapped"] = float(h.item_id.map(items.set_index("item_id").type_source).eq("mapped").mean())
        hp = h[["user_id", "item_id"]].drop_duplicates()
        t = t.merge(hp.assign(repeat=True), on=["user_id", "item_id"], how="left")
        t["repeat"] = t["repeat"].eq(True)
        s["target_repeat_row_share"] = float(t.repeat.mean())
        s["target_rows_unseen_globally"] = float((~t.item_id.isin(h.item_id)).mean())
        s["users_with_new_item"] = float(t.loc[~t.repeat, "user_id"].nunique() / t.user_id.nunique())
        # For new (non-repeat) target items: was the item's category at each level bought by the user before?
        new = t.loc[~t.repeat, ["user_id", "item_id"]].join(lookup, on="item_id")
        hh = h[["user_id", "item_id"]].join(lookup, on="item_id")
        for c in lv:
            seen = hh[["user_id", c]].drop_duplicates().assign(seen=True)
            m = new.merge(seen, on=["user_id", c], how="left")
            s[f"new_item_rows_in_seen_{c}"] = float(m.seen.eq(True).mean())
        out[split] = s
    return out


if __name__ == "__main__":
    profile = sys.argv[1] if len(sys.argv) > 1 else "medium"
    data_dir = Path(load_config(profile)["data_dir"])
    res = {ds: check(profile, ds, data_dir) for ds in LEVELS if (data_dir / profile / ds).exists()}
    dst = ROOT / "reports" / "data_check" / f"{profile}.json"
    dst.write_text(json.dumps(res, indent=2, ensure_ascii=False))
    print(json.dumps(res, indent=2, ensure_ascii=False))
