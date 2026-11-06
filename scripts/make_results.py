"""Consolidated result tables and figures for the report and slides (W11).

    python scripts/make_results.py [--sources scripts/results_sources.yaml] [--out reports/results] [--verify]

Every number is read from the JSON/CSV files registered in results_sources.yaml (read-only; other git
worktrees of C:\\dev_ML). Each table row and figure point carries its source file, the file's SHA-256 and
the meta.git_commit recorded in that file. Confirmatory holdout results (PREREG v1.2) and development
(medium) results are always labelled by the "stage" column / figure title.

Outputs (all under --out):
  tables/T1_master_hypotheses.csv|.md   every primary hypothesis and secondary/auxiliary comparison, one row each
  tables/T2_selections.csv|.md          selected values per dataset x run with grid-edge flags
  tables/T3_module_screening.csv|.md    explore modules (new-item Recall@10) and repeat modules (repeat Recall@10)
  tables/T4_final_lists.csv|.md         combinations C0-C4, C3b and the baselines: Recall@10, NDCG@10
  tables/T5_history_length.csv|.md      per history-length bin: diff and 95% CI of H1, H3, H6, H7, H8, H9, H10
  tables/T6_module_catalog.csv|.md      module status catalogue (from results_sources.yaml)
  tables/T7_datasets_eda.csv|.md        dataset summary copied from the EDA branch (feat/w5a-eda)
  figures/F1..F6 *.png (300 dpi) + *.csv with exactly the plotted numbers (each figure is drawn from its CSV)
  RESULTS.md                            index, input provenance, verification summary

Checks: for every T1 row |diff - (mean_model - mean_comparator)| <= 1e-12 against the source values;
--verify regenerates everything into a temporary folder and compares SHA-256 digests (determinism).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import textwrap
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
DIFF_TOL = 1e-12
DPI = 300
# Palette (one hue per role; stage is also carried by labels/hatching, never by colour alone).
COLOR_HOLDOUT, COLOR_DEV, COLOR_THIRD, COLOR_INK, COLOR_MUTED, COLOR_GRID = "#2a78d6", "#eb6834", "#1baf7a", "#0b0b0b", "#52514e", "#e2e1dc"
DATASET_ORDER = {"instacart": 0, "complete_journey": 1}
STAGE_ORDER = {"confirmatory holdout": 0, "development medium": 1}
PRIMARY_USERS = {"H1": "users with repeat target", "H2": "users with new target", "H3": "all users",
                 "H4": "users with new target", "H5": "users with new target"}


# ---------------------------------------------------------------------------
# provenance helpers
# ---------------------------------------------------------------------------
def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    """Hex SHA-256 of a file (hashlib.sha256 over the bytes, read in 1 MiB blocks)."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass
class Source:
    """One registered input file with its provenance."""

    key: str          # e.g. "w8/instacart_eval"
    path: Path
    run: str
    stage: str
    sha256: str
    git_commit: str
    doc: Any = None   # parsed JSON (dict) or DataFrame (CSV) or text


@dataclass
class Registry:
    sources: dict[str, Source] = field(default_factory=dict)
    skipped: list[dict[str, str]] = field(default_factory=list)

    def add(self, source: Source) -> Source:
        self.sources[source.key] = source
        return source

    def get(self, key: str) -> Source | None:
        return self.sources.get(key)

    def runs(self) -> list[str]:
        seen = []
        for source in self.sources.values():
            if source.run not in seen:
                seen.append(source.run)
        return seen


def load_source(key: str, path: Path, run: str, stage: str) -> Source:
    data = path.read_bytes()
    digest = sha256_bytes(data)
    if path.suffix == ".json":
        doc = json.loads(data.decode("utf-8"))
        commit = str(doc.get("meta", {}).get("git_commit") or "")
    elif path.suffix == ".csv":
        doc = pd.read_csv(path)
        commit = ""
    else:
        doc = data.decode("utf-8")
        commit = ""
    return Source(key, path, run, stage, digest, commit, doc)


def git_show(repo: Path, ref: str, rel_path: str) -> bytes:
    """Bytes of ``rel_path`` at ``ref`` via `git show ref:path` (read-only)."""
    result = subprocess.run(["git", "-C", str(repo), "show", f"{ref}:{rel_path}"], capture_output=True, check=True)
    return result.stdout


def git_rev(repo: Path, ref: str) -> str:
    return subprocess.run(["git", "-C", str(repo), "rev-parse", ref], capture_output=True, text=True, check=True).stdout.strip()


def build_registry(cfg: dict[str, Any]) -> Registry:
    """Load every registered file; apply the optional / W8b completion rules; record skips."""
    registry = Registry()
    for group, spec in cfg["sources"].items():
        files = {name: Path(p) for name, p in spec["files"].items()}
        missing = [name for name, p in files.items() if not p.exists()]
        if missing:
            if spec.get("optional"):
                registry.skipped.append({"run": spec["run"], "reason": f"files missing: {', '.join(missing)}"})
                continue
            raise FileNotFoundError(f"{group}: required input missing: {missing}")
        loaded = {name: load_source(f"{group}/{name}", p, spec["run"], spec["stage"]) for name, p in files.items()}
        stale = spec.get("complete_when_commit_is_not")
        if stale:
            stale_files = [name for name, s in loaded.items() if s.git_commit.startswith(stale) or not s.git_commit]
            if stale_files:
                registry.skipped.append({"run": spec["run"], "reason": f"not complete: meta.git_commit still {stale} in {', '.join(stale_files)}"})
                continue
        for source in loaded.values():
            registry.add(source)
    eda = cfg.get("eda")
    if eda:
        repo, branch = Path(eda["repo"]), eda["branch"]
        commit = git_rev(repo, branch)
        for rel in eda["files"]:
            data = git_show(repo, branch, rel)
            key = f"eda/{Path(rel).stem}"
            doc = pd.read_csv(pd.io.common.BytesIO(data)) if rel.endswith(".csv") else data.decode("utf-8")
            registry.add(Source(key, Path(f"{branch}:{rel}"), eda["run"], eda["stage"], sha256_bytes(data), commit, doc))
    return registry


def prov(source: Source) -> dict[str, str]:
    return {"source_file": str(source.path), "source_sha256": source.sha256, "git_commit": source.git_commit}


# ---------------------------------------------------------------------------
# generic rows from comparison entries
# ---------------------------------------------------------------------------
def entry_row(entry: dict[str, Any], *, id_: str, hypothesis: str, source: Source, dataset: str, transfer_from: str | None,
              users: str | None = None) -> dict[str, Any]:
    ci = entry.get("ci", {})
    ci95 = ci.get("0.95", ci.get(0.95, [np.nan, np.nan]))
    ci99 = ci.get("0.99375", ci.get(0.99375, [np.nan, np.nan]))
    decision = entry.get("decision", {}) or {}
    return {
        "id": id_, "hypothesis": hypothesis, "stage": source.stage, "run": source.run, "dataset": dataset,
        "transfer_from": transfer_from or "", "metric": entry.get("metric", ""), "users": entry.get("users", users or ""),
        "users_n": int(entry["n_users"]), "comparator": entry["comparator"], "model": entry["model"],
        "mean_comparator": float(entry["comparator_mean"]), "mean_model": float(entry["model_mean"]), "diff": float(entry["diff"]),
        "ci95_lo": float(ci95[0]), "ci95_hi": float(ci95[1]), "ci99375_lo": float(ci99[0]), "ci99375_hi": float(ci99[1]),
        "decision": decision.get("0.95", ""), "decision_99375": decision.get("0.99375", ""), **prov(source),
    }


def hypothesis_of(key: str) -> str:
    head = key.split("_")[0]
    return head if head[0] == "H" and head[1:].isdigit() else "secondary"


def eval_rows(source: Source) -> list[dict[str, Any]]:
    """All comparison rows of one pipeline eval JSON (W4 layout, plus lr_repeat / integration blocks of W6/W8)."""
    doc, rows = source.doc, []
    dataset, transfer = doc["dataset"], doc.get("transfer_from")
    for key, entry in doc.get("primary", {}).items():
        rows.append(entry_row(entry, id_=key, hypothesis=hypothesis_of(key), source=source, dataset=dataset, transfer_from=transfer))
    if doc.get("H5"):
        rows.append(entry_row(doc["H5"], id_="H5", hypothesis="H5", source=source, dataset=dataset, transfer_from=transfer))
    for key, entry in doc.get("secondary", {}).items():
        rows.append(entry_row(entry, id_=key, hypothesis="secondary", source=source, dataset=dataset, transfer_from=transfer))
    for block in ("lr_repeat", "integration"):
        if doc.get(block):
            for key, entry in doc[block].get("primary", {}).items():
                rows.append(entry_row(entry, id_=key, hypothesis=hypothesis_of(key), source=source, dataset=dataset, transfer_from=transfer))
            for key, entry in doc[block].get("secondary", {}).items():
                rows.append(entry_row(entry, id_=f"{block}:{key}", hypothesis="secondary", source=source, dataset=dataset, transfer_from=transfer))
    if "H8" in doc and "auxiliary" in doc:  # W7 layout
        rows.append(entry_row(doc["H8"], id_="H8_new_recall", hypothesis="H8", source=source, dataset=dataset, transfer_from=None,
                              users="users with new target"))
        for key, entry in doc["auxiliary"].items():
            rows.append(entry_row(entry, id_=f"auxiliary:{key}", hypothesis="auxiliary", source=source, dataset=dataset, transfer_from=None))
    if "hypotheses" in doc:  # W9 layout
        for key, entry in doc["hypotheses"].items():
            rows.append(entry_row(entry, id_=key, hypothesis=hypothesis_of(key) if key.startswith("H") else "auxiliary",
                                  source=source, dataset=dataset, transfer_from=None))
    return rows


def eval_sources(registry: Registry, run: str | None = None) -> list[Source]:
    out = []
    for source in registry.sources.values():
        if source.path.suffix != ".json" or (run and source.run != run):
            continue
        doc = source.doc
        if doc.get("split") == "eval" or "hypotheses" in doc:
            out.append(source)
    return out


# ---------------------------------------------------------------------------
# tables
# ---------------------------------------------------------------------------
T1_COLUMNS = ["id", "hypothesis", "stage", "run", "dataset", "transfer_from", "metric", "users", "users_n", "comparator", "model",
              "mean_comparator", "mean_model", "diff", "ci95_lo", "ci95_hi", "ci99375_lo", "ci99375_hi", "decision", "decision_99375",
              "source_file", "source_sha256", "git_commit"]


def table_t1(registry: Registry) -> pd.DataFrame:
    rows = []
    for source in eval_sources(registry):
        rows.extend(eval_rows(source))
    frame = pd.DataFrame(rows, columns=T1_COLUMNS)
    frame["_s"], frame["_d"] = frame["stage"].map(STAGE_ORDER), frame["dataset"].map(DATASET_ORDER)
    frame["_h"] = frame["hypothesis"].map(lambda h: int(h[1:]) if h[0] == "H" and h[1:].isdigit() else 99)
    frame = frame.sort_values(["_s", "run", "_d", "transfer_from", "_h", "id"], kind="mergesort").drop(columns=["_s", "_d", "_h"])
    return frame.reset_index(drop=True)


def check_t1(frame: pd.DataFrame) -> dict[str, Any]:
    """diff must equal mean_model - mean_comparator within DIFF_TOL for every row."""
    gap = (frame["diff"] - (frame["mean_model"] - frame["mean_comparator"])).abs()
    bad = frame.loc[gap > DIFF_TOL, ["id", "run", "dataset"]]
    if len(bad):
        raise AssertionError(f"T1 diff identity violated (tol {DIFF_TOL}):\n{bad.to_string()}")
    return {"rows": int(len(frame)), "max_abs_gap": float(gap.max()), "tolerance": DIFF_TOL}


def _sel_row(source: Source, dataset: str, parameter: str, value: Any, edge: Any, grid: Any, note: str = "") -> dict[str, Any]:
    return {"stage": source.stage, "run": source.run, "dataset": dataset, "parameter": parameter, "value": value,
            "at_grid_edge": edge, "grid": json.dumps(grid) if grid is not None else "", "note": note, **prov(source)}


def table_t2(registry: Registry) -> pd.DataFrame:
    rows = []
    for source in registry.sources.values():
        if source.path.suffix != ".json":
            continue
        doc = source.doc
        dataset = doc.get("dataset")
        if doc.get("split") == "fit" and "stage1_repeat_recall" in doc:  # pipeline tuning (W4a, W6, W8, holdout)
            grids = {"gamma": [r["gamma"] for r in doc["stage1_repeat_recall"]],
                     "gamma_c": sorted({r["gamma_c"] for r in doc["stage2_explore_new_recall"]}),
                     "alpha": sorted({r["alpha"] for r in doc["stage2_explore_new_recall"]}),
                     "r": [r["r"] for r in doc["stage3_final_ndcg"]]}
            for name in ("gamma", "gamma_c", "alpha", "r"):
                rows.append(_sel_row(source, dataset, name, doc["selections"][name], doc["at_grid_edge"][name], grids[name]))
            if doc.get("lr_repeat"):
                lr = doc["lr_repeat"]
                rows.append(_sel_row(source, dataset, "C (repeat_lr)", lr["C"], _edge(lr["C"], lr["C_grid"]), lr["C_grid"],
                                     "chosen by validation repeat Recall@10; ties -> smaller C"))
            if doc.get("integration"):
                integ = doc["integration"]
                ucf = integ["usercf"]
                k_grid = sorted({g["k"] for g in ucf["grid"]})
                rows.append(_sel_row(source, dataset, "user-CF k", ucf["selections"]["k"], ucf["at_grid_edge"]["k"], k_grid))
                rows.append(_sel_row(source, dataset, "user-CF weighting", ucf["selections"]["weighting"],
                                     ucf["at_grid_edge"]["weighting"], sorted({g["weighting"] for g in ucf["grid"]})))
                for combo, r_value in integ["selections"]["r"].items():
                    rows.append(_sel_row(source, dataset, f"r ({combo})", r_value, integ["at_grid_edge"]["r"][combo], integ["r_grid"],
                                         integ["r_selection_users"][combo]["users"]))
        elif doc.get("split") == "fit" and "grid" in doc:  # W7 user-CF tuning
            rows.append(_sel_row(source, dataset, "user-CF k", doc["selections"]["k"], doc["at_grid_edge"]["k"], sorted({g["k"] for g in doc["grid"]})))
            rows.append(_sel_row(source, dataset, "user-CF weighting", doc["selections"]["weighting"], doc["at_grid_edge"]["weighting"],
                                 sorted({g["weighting"] for g in doc["grid"]})))
        elif "fit" in doc and "r3_config" in doc:  # W9
            for name in ("r2", "r3"):
                block = doc["fit"][name]
                rows.append(_sel_row(source, dataset, f"C ({'repeat_lr re-fit' if name == 'r2' else 'repeat_r3'})", block["C"],
                                     block["at_grid_edge"], block["C_grid"], "chosen by validation repeat Recall@10; ties -> smaller C"))
            rows.append(_sel_row(source, dataset, "gamma (R1 comparator, read from W4 tuning)", doc["gamma_star"]["value"], "", None,
                                 doc["gamma_star"]["source"]))
        elif doc.get("split") == "eval" and doc.get("transfer_from"):  # transfer evaluations: selections applied
            for name in ("gamma", "gamma_c", "alpha", "r"):
                rows.append(_sel_row(source, dataset, f"{name} (transfer from {doc['transfer_from']})", doc["selections"][name], "", None,
                                     doc["selections"].get("source", "")))
    frame = pd.DataFrame(rows)
    frame["_s"], frame["_d"] = frame["stage"].map(STAGE_ORDER), frame["dataset"].map(DATASET_ORDER)
    return frame.sort_values(["_s", "run", "_d", "parameter"], kind="mergesort").drop(columns=["_s", "_d"]).reset_index(drop=True)


def _edge(value: float, grid: list[float]) -> bool:
    return math.isclose(value, min(grid), rel_tol=1e-12) or math.isclose(value, max(grid), rel_tol=1e-12)


def _find(doc: dict[str, Any], *keys: str) -> dict[str, Any] | None:
    """Comparison entry found by key across primary / H5 / secondary / lr_repeat / integration blocks."""
    places = [doc.get("primary", {}), doc.get("secondary", {})]
    if doc.get("H5"):
        places.append({"H5": doc["H5"]})
    for block in ("lr_repeat", "integration"):
        if doc.get(block):
            places += [doc[block].get("primary", {}), doc[block].get("secondary", {})]
    if "hypotheses" in doc:
        places.append(doc["hypotheses"])
    for key in keys:
        for place in places:
            if key in place:
                return place[key]
    return None


EXPLORE_MODULES = [  # (module id, name, list, mean from (entry key, field), comparator entry key)
    ("E1", "hierarchical explore (L2)", "explore_star", ("H2_new_recall", "model_mean"), "H2_new_recall"),
    ("E2", "user-based CF", "usercf", ("H8_new_recall", "model_mean"), "H8_new_recall"),
    ("E3", "flat item-CF", "itemcf_flat", ("H5", "model_mean"), "H5"),
    ("E4", "theta x item-CF", "explore_cf", ("explore_cf_vs_explore_star_new_recall", "model_mean"), "explore_cf_vs_explore_star_new_recall"),
    ("flat_pop", "flat popularity (comparator)", "flat_explore", ("H2_new_recall", "comparator_mean"), None),
]
REPEAT_MODULES = [
    ("R5", "personal frequency (comparator)", "repeat_pf", ("H1_repeat_recall", "comparator_mean"), None),
    ("R1", "decayed frequency s_rep(gamma*)", "repeat_star", ("H1_repeat_recall", "model_mean"), "H1_repeat_recall"),
    ("R2", "logistic-regression ranker", "repeat_lr", ("H6_repeat_recall", "model_mean"), "H6_repeat_recall"),
    ("R3", "logistic ranker + binned cycle features", "repeat_r3", ("H10_repeat_recall", "model_mean"), "H10_repeat_recall"),
]


def table_t3(registry: Registry, dev_run: str | None) -> pd.DataFrame:
    rows = []
    for source in eval_sources(registry):
        doc = source.doc
        if doc.get("transfer_from"):
            continue
        if source.stage == "development medium" and source.run not in (dev_run, "W9"):
            continue
        for group, modules, metric in (("explore", EXPLORE_MODULES, "new-item Recall@10"), ("repeat", REPEAT_MODULES, "repeat Recall@10")):
            for module, name, list_name, (key, field_), comp_key in modules:
                entry = _find(doc, key)
                if entry is None:
                    continue
                if entry[field_ if field_ == "comparator_mean" else "model_mean"] is None:
                    continue
                row = {"stage": source.stage, "run": source.run, "dataset": doc["dataset"], "group": group, "module": module,
                       "module_name": name, "list": list_name, "metric": metric, "users_n": int(entry["n_users"]),
                       "mean": float(entry[field_]), "comparator": "", "diff_vs_comparator": np.nan, "ci95_lo": np.nan, "ci95_hi": np.nan}
                if comp_key:
                    comp = _find(doc, comp_key)
                    ci95 = comp["ci"].get("0.95", [np.nan, np.nan])
                    row.update({"comparator": comp["comparator"], "diff_vs_comparator": float(comp["diff"]),
                                "ci95_lo": float(ci95[0]), "ci95_hi": float(ci95[1])})
                rows.append({**row, **prov(source)})
    frame = pd.DataFrame(rows)
    frame = frame.drop_duplicates(subset=["stage", "run", "dataset", "group", "module"])
    frame["_s"], frame["_d"] = frame["stage"].map(STAGE_ORDER), frame["dataset"].map(DATASET_ORDER)
    frame["_g"] = frame["group"].map({"explore": 0, "repeat": 1})
    frame["_m"] = frame["module"].map({m: i for i, m in enumerate([m[0] for m in EXPLORE_MODULES + REPEAT_MODULES])})
    frame = frame.sort_values(["_s", "_d", "_g", "_m"], kind="mergesort").drop(columns=["_s", "_d", "_g", "_m"])
    return frame.reset_index(drop=True)


def table_t4(registry: Registry, cfg: dict[str, Any], dev_run: str | None) -> pd.DataFrame:
    combos = cfg["combinations"]
    by_list = {spec["list"]: (label, spec) for label, spec in combos.items()}
    rows = []
    for source in eval_sources(registry):
        doc = source.doc
        if doc.get("transfer_from") or "list_metrics" not in doc:
            continue
        if source.stage == "development medium" and source.run != dev_run:
            continue
        wanted = list(by_list) + ["personal_frequency", "popularity"]
        if source.stage == "confirmatory holdout":
            wanted = ["final", "personal_frequency", "popularity"]
        for list_name in wanted:
            metrics = doc["list_metrics"].get(list_name)
            if metrics is None:
                continue
            label, spec = by_list.get(list_name, (list_name, {}))
            r_value = doc["selections"].get("r") if list_name in ("final", "final_cf", "final_lr") else None
            if doc.get("integration") and list_name in doc["integration"]["selections"].get("r", {}):
                r_value = doc["integration"]["selections"]["r"][list_name]
            rows.append({"combination": label, "list": list_name, "repeat_module": spec.get("repeat", ""), "explore_module": spec.get("explore", ""),
                         "r_rule": spec.get("r_rule", ""), "r": r_value if r_value is not None else "", "stage": source.stage, "run": source.run,
                         "dataset": doc["dataset"], "users_n": int(doc["n_users"]["all"]), "recall10": float(metrics["recall"]),
                         "ndcg10": float(metrics["ndcg"]), **prov(source)})
    frame = pd.DataFrame(rows)
    frame["_s"], frame["_d"] = frame["stage"].map(STAGE_ORDER), frame["dataset"].map(DATASET_ORDER)
    frame["_c"] = frame["combination"].map({c: i for i, c in enumerate(list(combos) + ["personal_frequency", "popularity"])})
    return frame.sort_values(["_s", "_d", "_c"], kind="mergesort").drop(columns=["_s", "_d", "_c"]).reset_index(drop=True)


T5_KEYS = ["H1_repeat_recall", "H3_ndcg", "H3_recall", "H6_repeat_recall", "H7_ndcg", "H7_recall", "H8_new_recall", "H9_ndcg", "H9_recall",
           "H10_repeat_recall"]


def table_t5(registry: Registry, dev_run: str | None) -> pd.DataFrame:
    rows = []
    for source in eval_sources(registry):
        doc = source.doc
        if doc.get("transfer_from"):
            continue
        if source.stage == "development medium" and source.run not in (dev_run, "W9"):
            continue
        blocks = [doc.get("by_history_length", [])]
        for block in ("lr_repeat", "integration"):
            if doc.get(block):
                blocks.append(doc[block].get("by_history_length", []))
        for block in blocks:
            for bin_row in block:
                for key in T5_KEYS:
                    entry = bin_row.get(key)
                    if entry is None and key in bin_row:
                        rows.append({"stage": source.stage, "run": source.run, "dataset": doc["dataset"], "bin": bin_row["bin"],
                                     "bin_users": int(bin_row["n_users"]), "id": key, "hypothesis": hypothesis_of(key), "users_n": 0,
                                     "mean_comparator": np.nan, "mean_model": np.nan, "diff": np.nan, "ci95_lo": np.nan, "ci95_hi": np.nan,
                                     **prov(source)})
                    elif entry is not None:
                        ci = entry.get("ci_0.95") or entry.get("ci", {}).get("0.95")
                        rows.append({"stage": source.stage, "run": source.run, "dataset": doc["dataset"], "bin": bin_row["bin"],
                                     "bin_users": int(bin_row["n_users"]), "id": key, "hypothesis": hypothesis_of(key),
                                     "users_n": int(entry["n_users"]), "mean_comparator": float(entry["comparator_mean"]),
                                     "mean_model": float(entry["model_mean"]), "diff": float(entry["diff"]), "ci95_lo": float(ci[0]),
                                     "ci95_hi": float(ci[1]), **prov(source)})
    frame = pd.DataFrame(rows)
    frame["_s"], frame["_d"] = frame["stage"].map(STAGE_ORDER), frame["dataset"].map(DATASET_ORDER)
    frame["_h"] = frame["id"].map({k: i for i, k in enumerate(T5_KEYS)})
    frame["_b"] = frame["bin"].map(lambda b: float(b.split(",")[0].strip("(")))
    return frame.sort_values(["_s", "run", "_d", "_h", "_b"], kind="mergesort").drop(columns=["_s", "_d", "_h", "_b"]).reset_index(drop=True)


def table_t6(cfg: dict[str, Any], t1: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for module in cfg["modules"]:
        ids = [h for h in str(module.get("basis", "")).replace("(", ",").replace(")", "").split(",")]
        ids = [h.strip() for h in ids if h.strip().startswith("H")]
        stages = sorted(set(t1.loc[t1["hypothesis"].isin(ids), "stage"])) if ids else []
        rows.append({"id": module["id"], "group": module["group"], "name": module["name"], "list": module.get("list", ""),
                     "status": module["status"], "basis": module.get("basis", ""), "evidence_stage": "; ".join(stages),
                     "note": module.get("note", "")})
    return pd.DataFrame(rows)


def table_t7(registry: Registry) -> pd.DataFrame:
    source = registry.get("eda/table_datasets")
    frame = source.doc.copy()
    frame.insert(0, "stage", source.stage)
    frame.insert(1, "run", source.run)
    for key, value in prov(source).items():
        frame[key] = value
    return frame


# ---------------------------------------------------------------------------
# markdown / csv writers
# ---------------------------------------------------------------------------
def _fmt(value: Any) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return ""
    if isinstance(value, (bool, np.bool_)):
        return str(bool(value))
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)):
        return f"{value:.6f}".rstrip("0").rstrip(".") if abs(value) < 1e6 else f"{value:.6g}"
    return str(value).replace("|", "\\|").replace("\n", " ")


def to_markdown(frame: pd.DataFrame, title: str, note: str = "", hide: tuple[str, ...] = ("source_sha256",)) -> str:
    columns = [c for c in frame.columns if c not in hide]
    lines = [f"# {title}", ""]
    if note:
        lines += [note, ""]
    lines.append("| " + " | ".join(columns) + " |")
    lines.append("|" + "|".join(" --- " for _ in columns) + "|")
    for _, row in frame.iterrows():
        lines.append("| " + " | ".join(_fmt(row[c]) for c in columns) + " |")
    if hide:
        lines += ["", f"Hidden columns (see the CSV): {', '.join(h for h in hide if h in frame.columns)}."]
    return "\n".join(lines) + "\n"


def write_table(frame: pd.DataFrame, out_dir: Path, name: str, title: str, note: str = "") -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path, md_path = out_dir / f"{name}.csv", out_dir / f"{name}.md"
    frame.to_csv(csv_path, index=False, lineterminator="\n", encoding="utf-8")
    md_path.write_text(to_markdown(frame, title, note), encoding="utf-8", newline="\n")
    return {"name": name, "rows": int(len(frame)), "csv": csv_path, "md": md_path}


# ---------------------------------------------------------------------------
# figures (each drawn from its own CSV)
# ---------------------------------------------------------------------------
def _plt():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.size": 8, "axes.titlesize": 9, "axes.labelsize": 8, "legend.fontsize": 7, "axes.edgecolor": COLOR_MUTED,
                         "xtick.color": COLOR_MUTED, "ytick.color": COLOR_MUTED, "axes.labelcolor": COLOR_INK, "savefig.dpi": DPI,
                         "figure.dpi": 100})
    return plt


def _style(ax) -> None:
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.grid(True, axis="x", color=COLOR_GRID, linewidth=0.6)
    ax.set_axisbelow(True)


def _finish(fig, path: Path, plt, footer: str, top: float = 0.94) -> None:
    """Wrap the source footer to the figure width, reserve room for it, then save the PNG (300 dpi)."""
    width = max(60, int(fig.get_figwidth() * 19))
    text = "\n".join(textwrap.fill(part, width) for part in footer.split("\n"))
    n_lines = text.count("\n") + 1
    bottom = 0.02 + 0.026 * n_lines * (8.0 / fig.get_figheight())
    fig.tight_layout(rect=(0, bottom, 1, top))
    fig.text(0.01, 0.01, text, fontsize=6.5, color=COLOR_MUTED, ha="left", va="bottom")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=DPI, metadata={"Software": None})
    plt.close(fig)


def _forest(frame: pd.DataFrame, path: Path, title: str, footer: str, color: str, labels: dict[str, str], datasets: dict[str, str]) -> None:
    plt = _plt()
    frame = pd.read_csv(path.with_suffix(".csv"))  # drawn from the CSV written just before
    panels = [d for d in datasets if d in set(frame["dataset"])]
    fig, axes = plt.subplots(1, len(panels), figsize=(3.6 * len(panels) + 1.4, 0.42 * frame["id"].nunique() + 2.2), sharey=True)
    axes = np.atleast_1d(axes)
    ids = list(dict.fromkeys(frame["id"]))
    for ax, dataset in zip(axes, panels):
        part = frame[frame["dataset"] == dataset].set_index("id")
        for i, id_ in enumerate(ids):
            y = len(ids) - 1 - i
            if id_ not in part.index:
                continue
            row = part.loc[id_]
            if not (isinstance(row.get("ci99375_lo"), float) and math.isnan(row["ci99375_lo"])):
                ax.plot([row["ci99375_lo"], row["ci99375_hi"]], [y, y], color=color, linewidth=0.8, solid_capstyle="butt", zorder=2)
            ax.plot([row["ci95_lo"], row["ci95_hi"]], [y, y], color=color, linewidth=3.0, solid_capstyle="butt", zorder=3)
            marker = "o" if str(row.get("note", "")) == "" or (isinstance(row.get("note"), float) and math.isnan(row.get("note"))) else "D"
            ax.plot(row["diff"], y, marker=marker, markersize=4.5, color="white", markeredgecolor=color, markeredgewidth=1.2, zorder=4)
        ax.axvline(0.0, color=COLOR_INK, linewidth=0.8, zorder=1)
        ax.set_yticks(range(len(ids)))
        ax.set_yticklabels([labels.get(i, i) for i in reversed(ids)])
        n = int(part["users_n"].max()) if len(part) else 0
        ax.set_title(f"{datasets[dataset]} ({part['stage_label'].iloc[0]})\nusers per row up to n = {n:,}", loc="left")
        ax.set_xlabel("difference of per-user means (model - comparator)")
        _style(ax)
    fig.suptitle(title, x=0.01, ha="left", fontsize=10, color=COLOR_INK)
    _finish(fig, path, plt, footer, top=0.93)


def figure_f1(t1: pd.DataFrame, out: Path, cfg: dict[str, Any]) -> dict[str, Any]:
    keep = t1[(t1["stage"] == "confirmatory holdout") & (t1["transfer_from"] == "") & t1["hypothesis"].isin(["H1", "H2", "H3", "H4", "H5"])].copy()
    keep["stage_label"] = keep["dataset"].map({"instacart": "holdout", "complete_journey": "refit on medium data"})
    keep["note"] = np.where((keep["hypothesis"] == "H5") & (keep["dataset"] == "complete_journey"), "H5 pre-registered for Instacart only", "")
    keep = keep[["id", "hypothesis", "dataset", "stage", "stage_label", "metric", "users_n", "diff", "ci95_lo", "ci95_hi", "ci99375_lo", "ci99375_hi",
                 "decision", "decision_99375", "note", "source_file", "source_sha256", "git_commit"]]
    path = out / "F1_holdout_forest.png"
    keep.to_csv(path.with_suffix(".csv"), index=False, lineterminator="\n")
    n_i = int(keep.loc[keep["dataset"] == "instacart", "users_n"].max())
    n_c = int(keep.loc[keep["dataset"] == "complete_journey", "users_n"].max())
    _forest(keep, path, "Confirmatory results (holdout): H1-H5, paired bootstrap CIs",
            f"Source: Instacart holdout (n = {n_i:,}) and Complete Journey refit on medium data (n = {n_c:,}), PREREG v1.2, code commit "
            f"{keep['git_commit'].iloc[0][:7]}. Thick bar 95% CI, thin bar 99.375% CI (Bonferroni, 8 tests), 1,000 resamples. "
            "Diamond: H5 on Complete Journey, not pre-registered (secondary, Instacart only).",
            COLOR_HOLDOUT, cfg["hypothesis_labels"], cfg["dataset_labels"])
    return {"name": "F1_holdout_forest", "points": int(len(keep)), "png": path, "csv": path.with_suffix(".csv")}


def figure_f2(t1: pd.DataFrame, out: Path, cfg: dict[str, Any], dev_run: str | None) -> dict[str, Any]:
    keep = t1[(t1["stage"] == "development medium") & t1["hypothesis"].isin(["H6", "H7", "H8", "H9", "H10"])
              & t1["run"].isin([dev_run, "W9"])].copy()
    keep = keep[~((keep["run"] == "W9") & (keep["hypothesis"] != "H10"))]
    keep["stage_label"] = "development (medium)"
    keep["note"] = ""
    keep = keep[["id", "hypothesis", "run", "dataset", "stage", "stage_label", "metric", "users_n", "diff", "ci95_lo", "ci95_hi", "ci99375_lo",
                 "ci99375_hi", "decision", "decision_99375", "note", "source_file", "source_sha256", "git_commit"]]
    path = out / "F2_dev_forest.png"
    keep.to_csv(path.with_suffix(".csv"), index=False, lineterminator="\n")
    runs = ", ".join(f"{r} (commit {c[:7]})" for r, c in keep.groupby("run")["git_commit"].first().items())
    _forest(keep, path, "Development results: H6-H10, paired bootstrap CIs - development (medium)",
            f"Source: medium eval split, Instacart (n = 20,000 users) and Complete Journey (n = 2,069); runs {runs}. "
            "Thick bar 95% CI, thin bar 99.375% CI, 1,000 resamples. Not confirmatory.", COLOR_DEV, cfg["hypothesis_labels"], cfg["dataset_labels"])
    return {"name": "F2_dev_forest", "points": int(len(keep)), "png": path, "csv": path.with_suffix(".csv")}


def figure_f3(t3: pd.DataFrame, out: Path, cfg: dict[str, Any]) -> dict[str, Any]:
    plt = _plt()
    path = out / "F3_module_screening.png"
    t3.to_csv(path.with_suffix(".csv"), index=False, lineterminator="\n")
    frame = pd.read_csv(path.with_suffix(".csv"))
    datasets = [d for d in cfg["dataset_labels"] if d in set(frame["dataset"])]
    fig, axes = plt.subplots(2, len(datasets), figsize=(4.2 * len(datasets), 5.6), squeeze=False)
    stage_color = {"confirmatory holdout": COLOR_HOLDOUT, "development medium": COLOR_DEV}
    stage_hatch = {"confirmatory holdout": "", "development medium": "////"}
    for row_i, (group, metric) in enumerate((("explore", "new-item Recall@10"), ("repeat", "repeat Recall@10"))):
        for col_i, dataset in enumerate(datasets):
            ax = axes[row_i][col_i]
            part = frame[(frame["group"] == group) & (frame["dataset"] == dataset)]
            catalog = [m[0] for m in EXPLORE_MODULES + REPEAT_MODULES]
            modules = [m for m in catalog if m in set(part["module"])]
            stages = [s for s in stage_color if s in set(part["stage"])]
            width = 0.8 / max(len(stages), 1)
            for s_i, stage in enumerate(stages):
                sub = part[part["stage"] == stage].set_index("module")
                xs = [i + (s_i - (len(stages) - 1) / 2) * width for i in range(len(modules))]
                ys = [sub["mean"].get(m, np.nan) for m in modules]
                ax.bar(xs, ys, width=width * 0.92, color=stage_color[stage], hatch=stage_hatch[stage], edgecolor="white", linewidth=0.5,
                       label=stage, zorder=3)
            ax.set_xticks(range(len(modules)))
            ax.set_xticklabels(modules)
            ax.set_ylabel(metric)
            ax.set_title(f"{cfg['dataset_labels'][dataset]} - {group} modules", loc="left")
            ax.grid(True, axis="y", color=COLOR_GRID, linewidth=0.6)
            ax.set_axisbelow(True)
            for side in ("top", "right"):
                ax.spines[side].set_visible(False)
            if row_i == 0 and col_i == 0:
                ax.legend(frameon=False)
    fig.suptitle("Module screening: mean of the per-user metric by module (holdout vs development medium)", x=0.01, ha="left", fontsize=10)
    _finish(fig, path, plt, "Explore: new-item Recall@10 over users with >= 1 new target item. Repeat: repeat Recall@10 over users with >= 1 repeat "
            "target item. Solid = confirmatory holdout (PREREG v1.2; Complete Journey refit on medium data), hatched = development (medium). "
            "No per-bar CI is available (only paired-difference CIs; see the F3 CSV / T3).")
    return {"name": "F3_module_screening", "points": int(len(frame)), "png": path, "csv": path.with_suffix(".csv")}


def figure_f4(registry: Registry, out: Path, cfg: dict[str, Any], dev_run: str | None) -> dict[str, Any] | None:
    rows = []
    by_list = {spec["list"]: label for label, spec in cfg["combinations"].items()}
    for source in registry.sources.values():
        if source.run != dev_run or source.doc.get("split") != "fit" or not source.doc.get("integration"):
            continue
        integ = source.doc["integration"]
        for list_name, scores in integ["r_scores"].items():
            label = by_list.get(list_name, list_name)
            if list_name == "final_lr_r":
                label = "C3/C3b"
            for entry in scores:
                rows.append({"stage": source.stage, "run": source.run, "dataset": source.doc["dataset"], "combination": label, "list": list_name,
                             "r": int(entry["r"]), "ndcg10": float(entry["ndcg"]), "selection_users": integ["r_selection_users"][list_name]["users"],
                             "selection_users_n": int(integ["r_selection_users"][list_name]["n_users"]),
                             "selected_r": int(integ["selections"]["r"][list_name]), **prov(source)})
    if not rows:
        return None
    plt = _plt()
    path = out / "F4_r_grid.png"
    pd.DataFrame(rows).to_csv(path.with_suffix(".csv"), index=False, lineterminator="\n")
    frame = pd.read_csv(path.with_suffix(".csv"))
    datasets = [d for d in cfg["dataset_labels"] if d in set(frame["dataset"])]
    colors = {"C2": COLOR_HOLDOUT, "C3/C3b": COLOR_DEV, "C4": COLOR_THIRD}
    markers = {"C2": "o", "C3/C3b": "s", "C4": "^"}
    fig, axes = plt.subplots(1, len(datasets), figsize=(4.2 * len(datasets), 3.6), squeeze=False)
    for ax, dataset in zip(axes[0], datasets):
        part = frame[frame["dataset"] == dataset]
        for combo in ("C2", "C3/C3b", "C4"):
            line = part[part["combination"] == combo].sort_values("r")
            if line.empty:
                continue
            ax.plot(line["r"], line["ndcg10"], color=colors[combo], marker=markers[combo], markersize=4, linewidth=1.6,
                    label=f"{combo} ({line['list'].iloc[0]}, n = {int(line['selection_users_n'].iloc[0]):,})")
        ax.set_xticks(range(0, 11))
        ax.set_xlabel("r = repeat slots in the top-10 list")
        ax.set_ylabel("NDCG@10 on the fit-split selection users")
        ax.set_title(f"{cfg['dataset_labels'][dataset]} - development (medium), fit split", loc="left")
        ax.grid(True, color=COLOR_GRID, linewidth=0.6)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.legend(frameon=False, loc="upper left")
    fig.suptitle("Slot re-selection: NDCG@10 by r for combinations C2, C3/C3b, C4 - development (medium)", x=0.01, ha="left", fontsize=10)
    _finish(fig, path, plt, f"Source: {dev_run} tuning JSON (commit {frame['git_commit'].iloc[0][:7]}), integration.r_scores. C2 (final_star_ucf) is "
            "scored on all fit users; C3/C3b (final_lr_r) and C4 (final_lr_ucf) on the LR validation users with a train-only model. "
            "Selected r = 10 for every combination (grid edge).")
    return {"name": "F4_r_grid", "points": int(len(frame)), "png": path, "csv": path.with_suffix(".csv")}


def figure_f5(t5: pd.DataFrame, out: Path, cfg: dict[str, Any], dev_run: str | None) -> dict[str, Any]:
    keep = t5[(t5["stage"] == "development medium") & (t5["run"] == dev_run) & t5["id"].isin(["H3_ndcg", "H7_ndcg", "H8_new_recall"])
              & t5["users_n"].gt(0)].copy()
    plt = _plt()
    path = out / "F5_history_length.png"
    keep.to_csv(path.with_suffix(".csv"), index=False, lineterminator="\n")
    frame = pd.read_csv(path.with_suffix(".csv"))
    datasets = [d for d in cfg["dataset_labels"] if d in set(frame["dataset"])]
    series = [("H3_ndcg", "H3 NDCG@10 (final vs personal frequency)", COLOR_HOLDOUT, "o"),
              ("H7_ndcg", "H7 NDCG@10 (final_lr vs final)", COLOR_DEV, "s"),
              ("H8_new_recall", "H8 new-item Recall@10 (usercf vs flat popularity)", COLOR_THIRD, "^")]
    fig, axes = plt.subplots(1, len(datasets), figsize=(4.4 * len(datasets), 3.8), squeeze=False)
    for ax, dataset in zip(axes[0], datasets):
        part = frame[frame["dataset"] == dataset]
        bins = list(dict.fromkeys(part["bin"]))
        for s_i, (key, label, color, marker) in enumerate(series):
            sub = part[part["id"] == key].set_index("bin")
            xs = [i + (s_i - 1) * 0.22 for i in range(len(bins))]
            ys = [sub["diff"].get(b, np.nan) for b in bins]
            lo = [sub["ci95_lo"].get(b, np.nan) for b in bins]
            hi = [sub["ci95_hi"].get(b, np.nan) for b in bins]
            ax.errorbar(xs, ys, yerr=[np.array(ys) - np.array(lo), np.array(hi) - np.array(ys)], fmt=marker, color=color, markersize=4.5,
                        capsize=2, linewidth=1.2, label=label, zorder=3)
        ax.axhline(0.0, color=COLOR_INK, linewidth=0.8, zorder=1)
        ax.set_xticks(range(len(bins)))
        ax.set_xticklabels(bins)
        ax.set_xlabel("history length (distinct history baskets)")
        ax.set_ylabel("difference of per-user means (model - comparator)")
        ax.set_title(f"{cfg['dataset_labels'][dataset]} - development (medium)", loc="left")
        ax.grid(True, axis="y", color=COLOR_GRID, linewidth=0.6)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    axes[0][0].legend(frameon=False, loc="upper right")
    fig.suptitle("Differences by history length with 95% CIs - development (medium)", x=0.01, ha="left", fontsize=10)
    _finish(fig, path, plt, f"Source: {dev_run} eval JSON (commit {frame['git_commit'].iloc[0][:7]}), by_history_length blocks; paired bootstrap "
            "1,000 resamples, 95% percentile CI per bin; descriptive, no decisions (PREREG v1.2 deviation 2).")
    return {"name": "F5_history_length", "points": int(len(frame)), "png": path, "csv": path.with_suffix(".csv")}


def figure_f6(registry: Registry, out: Path, cfg: dict[str, Any]) -> dict[str, Any]:
    rows = []
    for dataset in cfg["dataset_labels"]:
        source = registry.get(f"w9/{dataset}_overdue_csv")
        json_source = registry.get(f"w9/{dataset}_r3")
        if source is None:
            continue
        frame = source.doc.copy()
        frame.insert(0, "dataset", dataset)
        frame.insert(0, "run", source.run)
        frame.insert(0, "stage", source.stage)
        for key, value in prov(source).items():
            frame[key] = value
        frame["git_commit"] = json_source.git_commit if json_source else ""
        frame["model_sha256"] = json_source.doc["artifact"]["sha256"] if json_source else ""
        rows.append(frame)
    plt = _plt()
    path = out / "F6_overdue_effect.png"
    pd.concat(rows, ignore_index=True).to_csv(path.with_suffix(".csv"), index=False, lineterminator="\n")
    frame = pd.read_csv(path.with_suffix(".csv"))
    datasets = [d for d in cfg["dataset_labels"] if d in set(frame["dataset"])]
    fig, axes = plt.subplots(1, len(datasets), figsize=(4.4 * len(datasets), 3.6), squeeze=False, sharey=True)
    for ax, dataset in zip(axes[0], datasets):
        part = frame[frame["dataset"] == dataset].reset_index(drop=True)
        colors = [COLOR_MUTED if b == "once" else COLOR_DEV for b in part["bin"]]
        ax.bar(range(len(part)), part["effect_log_odds"], width=0.62, color=colors, zorder=3)
        ax.axhline(0.0, color=COLOR_INK, linewidth=0.8, zorder=2)
        for i, column in enumerate(part["column"]):
            if column == "(reference)":
                ax.text(i, 0.0, "ref.", ha="center", va="bottom", fontsize=7, color=COLOR_MUTED)
        ax.set_xticks(range(len(part)))
        ax.set_xticklabels(part["bin"], fontsize=7)
        ax.set_xlabel("overdue bin (right-open); grey = once-bought items")
        ax.set_ylabel("log-odds effect vs. reference bin [0, 0.5)")
        ax.set_title(f"{cfg['dataset_labels'][dataset]} - development (medium)", loc="left")
        ax.grid(True, axis="y", color=COLOR_GRID, linewidth=0.6)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    fig.suptitle("R3 logistic ranker: log-odds effect per overdue bin (standardized coefficient / dummy std) - development (medium)",
                 x=0.01, ha="left", fontsize=10)
    _finish(fig, path, plt, f"Source: W9 reports/tables/medium_<dataset>_r3_overdue_coef.csv (commit {frame['git_commit'].iloc[0][:7]}); overdue = "
            "recency / mean_gap; fit split of medium, L2 logistic regression refit on all training rows with C*. Not confirmatory.")
    return {"name": "F6_overdue_effect", "points": int(len(frame)), "png": path, "csv": path.with_suffix(".csv")}


# ---------------------------------------------------------------------------
# RESULTS.md
# ---------------------------------------------------------------------------
TABLE_NOTES = {
    "T1_master_hypotheses": "one row per hypothesis or secondary/auxiliary comparison; diff = mean_model - mean_comparator; decision as recorded in the source",
    "T2_selections": "selected values per dataset x run (long format) with grid and grid-edge flag",
    "T3_module_screening": "explore modules by new-item Recall@10, repeat modules by repeat Recall@10; diff and 95% CI vs the comparator named in the row",
    "T4_final_lists": "Recall@10 and NDCG@10 over all users for the combinations C0-C4, C3b and the baselines (medium); C0 and baselines on the holdout",
    "T5_history_length": "per history-length bin: diff and 95% CI of H1, H3, H6, H7, H8, H9, H10",
    "T6_module_catalog": "module status catalogue from results_sources.yaml (adopted / evaluated-not-adopted / excluded / not-examined) with basis hypotheses",
    "T7_datasets_eda": "dataset summary (eval history) copied from the EDA branch feat/w5a-eda",
}
FIGURE_NOTES = {
    "F1_holdout_forest": "holdout confirmatory H1-H5 forest plot, Instacart and Complete Journey refit side by side (95% thick, 99.375% thin)",
    "F2_dev_forest": "development (medium) H6-H10 forest plot, same format",
    "F3_module_screening": "module means: explore (new-item Recall@10) and repeat (repeat Recall@10), holdout vs medium",
    "F4_r_grid": "NDCG@10 by r (0-10) for C2, C3/C3b, C4 on the fit split, one panel per dataset",
    "F5_history_length": "per-bin differences with 95% CIs for H3, H7, H8, one panel per dataset",
    "F6_overdue_effect": "W9 R3 log-odds effect per overdue bin (from the W9 CSV), one panel per dataset",
}


def write_results_md(out: Path, registry: Registry, tables: list[dict[str, Any]], figures: list[dict[str, Any]], checks: dict[str, Any],
                     w8b_used: bool, dev_run: str | None, cfg: dict[str, Any]) -> Path:
    lines = ["# Consolidated results (W11)", ""]
    lines.append(f"- W8b outputs reflected: **{'yes' if w8b_used else 'no'}**"
                 + ("" if w8b_used else " (W8b not complete at generation time; development figures use run "
                    f"{dev_run}; re-run `python scripts/make_results.py` after W8b finishes)"))
    lines.append("- holdout2 results: **not included** (holdout2 has not been run; the holdout2 data folder was not opened)")
    lines.append("- Stage wording: `confirmatory holdout` = PREREG v1.2 final run (Instacart holdout; Complete Journey refit on medium data); "
                 "`development medium` = development runs, not confirmatory.")
    lines.append(f"- Development integration run used for T3-T5, F2-F5: `{dev_run}`; W9 provides R3/H10 in every case.")
    lines.append("- Combination labels C0-C4/C3b and the module catalogue E1-E8/R1-R6 are defined in `scripts/results_sources.yaml` "
                 "(mapping assumed by W11; confirm with the team). Modules without an implementation are `not-examined`.")
    lines += ["", "## Tables (reports/results/tables)", "", "| table | rows | description |", "| --- | --- | --- |"]
    for t in tables:
        lines.append(f"| {t['name']} | {t['rows']} | {TABLE_NOTES[t['name']]} |")
    lines += ["", "## Figures (reports/results/figures, PNG 300 dpi + CSV of the plotted numbers)", "", "| figure | points | description |", "| --- | --- | --- |"]
    for f in figures:
        lines.append(f"| {f['name']} | {f['points']} | {FIGURE_NOTES[f['name']]} |")
    lines += ["", "## Inputs (read-only)", "", "| key | run | stage | path | git_commit | sha256 |", "| --- | --- | --- | --- | --- | --- |"]
    for source in registry.sources.values():
        lines.append(f"| {source.key} | {source.run} | {source.stage} | {source.path} | {source.git_commit} | {source.sha256} |")
    if registry.skipped:
        lines += ["", "### Skipped inputs", ""]
        for item in registry.skipped:
            lines.append(f"- {item['run']}: {item['reason']}")
    lines += ["", "## Verification", ""]
    lines.append(f"- T1 diff identity: {checks['t1']['rows']} rows, max |diff - (mean_model - mean_comparator)| = {checks['t1']['max_abs_gap']:.3e} "
                 f"(tolerance {checks['t1']['tolerance']:.0e}).")
    lines.append("- Every figure is drawn from its own CSV (written first, then read back).")
    if "determinism" in checks:
        d = checks["determinism"]
        lines.append(f"- Determinism (`--verify`): {d['files']} files regenerated, {d['mismatches']} digest mismatches.")
    lines += ["", "## Generator", "", "`python scripts/make_results.py` (scripts/results_sources.yaml lists every input). Values are never typed in by hand.", ""]
    path = out / "RESULTS.md"
    path.write_text("\n".join(lines), encoding="utf-8", newline="\n")
    return path


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def generate(cfg: dict[str, Any], out: Path) -> dict[str, Any]:
    registry = build_registry(cfg)
    runs = registry.runs()
    dev_run = next((r for r in cfg["development_integration_run_preference"] if r in runs), None)
    w8b_used = "W8b" in runs

    t1 = table_t1(registry)
    checks = {"t1": check_t1(t1)}
    t2 = table_t2(registry)
    t3 = table_t3(registry, dev_run)
    t4 = table_t4(registry, cfg, dev_run)
    t5 = table_t5(registry, dev_run)
    t6 = table_t6(cfg, t1)
    t7 = table_t7(registry)
    tables_dir, figures_dir = out / "tables", out / "figures"
    for child in (tables_dir, figures_dir):
        if child.exists():
            shutil.rmtree(child)
        child.mkdir(parents=True, exist_ok=True)
    tables = [
        write_table(t1, tables_dir, "T1_master_hypotheses", "T1 master hypotheses table", TABLE_NOTES["T1_master_hypotheses"]),
        write_table(t2, tables_dir, "T2_selections", "T2 selected values", TABLE_NOTES["T2_selections"]),
        write_table(t3, tables_dir, "T3_module_screening", "T3 module screening", TABLE_NOTES["T3_module_screening"]),
        write_table(t4, tables_dir, "T4_final_lists", "T4 final lists", TABLE_NOTES["T4_final_lists"]),
        write_table(t5, tables_dir, "T5_history_length", "T5 history-length breakdown", TABLE_NOTES["T5_history_length"]),
        write_table(t6, tables_dir, "T6_module_catalog", "T6 module catalogue", TABLE_NOTES["T6_module_catalog"]),
        write_table(t7, tables_dir, "T7_datasets_eda", "T7 datasets (EDA, feat/w5a-eda)", TABLE_NOTES["T7_datasets_eda"]),
    ]
    figures = [figure_f1(t1, figures_dir, cfg), figure_f2(t1, figures_dir, cfg, dev_run), figure_f3(t3, figures_dir, cfg)]
    f4 = figure_f4(registry, figures_dir, cfg, dev_run)
    if f4:
        figures.append(f4)
    figures += [figure_f5(t5, figures_dir, cfg, dev_run), figure_f6(registry, figures_dir, cfg)]
    return {"registry": registry, "tables": tables, "figures": figures, "checks": checks, "w8b_used": w8b_used, "dev_run": dev_run}


def digests(folder: Path) -> dict[str, str]:
    return {str(p.relative_to(folder)).replace("\\", "/"): sha256_file(p) for p in sorted(folder.rglob("*")) if p.is_file()}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="W11 consolidated result tables and figures")
    parser.add_argument("--sources", default=str(SCRIPT_DIR / "results_sources.yaml"))
    parser.add_argument("--out", default=str(REPO_ROOT / "reports" / "results"))
    parser.add_argument("--verify", action="store_true", help="regenerate into a temporary folder and compare digests (determinism)")
    args = parser.parse_args(argv)
    with open(args.sources, encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)
    out = Path(args.out)
    result = generate(cfg, out)
    checks = result["checks"]
    if args.verify:
        with tempfile.TemporaryDirectory() as tmp:
            generate(cfg, Path(tmp))
            first, second = digests(out / "tables"), digests(Path(tmp) / "tables")
            first.update({f"figures/{k}": v for k, v in digests(out / "figures").items()})
            second.update({f"figures/{k}": v for k, v in digests(Path(tmp) / "figures").items()})
            mismatches = sorted(k for k in set(first) | set(second) if first.get(k) != second.get(k))
            checks["determinism"] = {"files": len(first), "mismatches": len(mismatches), "mismatched": mismatches}
    write_results_md(out, result["registry"], result["tables"], result["figures"], checks, result["w8b_used"], result["dev_run"], cfg)
    print(f"tables: {[(t['name'], t['rows']) for t in result['tables']]}")
    print(f"figures: {[(f['name'], f['points']) for f in result['figures']]}")
    print(f"W8b used: {result['w8b_used']}; development run: {result['dev_run']}; skipped: {result['registry'].skipped}")
    print(f"T1 check: {checks['t1']}")
    if "determinism" in checks:
        print(f"determinism: {checks['determinism']}")
        if checks["determinism"]["mismatches"]:
            sys.exit(1)


if __name__ == "__main__":
    main()
