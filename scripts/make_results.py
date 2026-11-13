# *------------------- 2026 Fall Semester Machine Learning Term-project: Team 9 --------------------*
# Code First written at 2026-11-06
# Code Final Edited at 2026-11-13
# Editor: Yejun Kim

'''
Purpose : Consolidated result tables and figures for the report and slides (W11). Every number is
          read from the JSON/CSV files registered in results_sources.yaml (read-only; other git
          worktrees of C:\\dev_ML). Each table row and figure point carries its source file, the
          file's SHA-256 and the meta.git_commit recorded in that file. Confirmatory holdout
          results (PREREG v1.2) and development (medium) results are always labelled by the
          "stage" column / figure title; the PREREG v1.3 run on holdout2 (reports/final2) is the
          stage "confirmatory holdout2", the PREREG v1.4 exploration-policy test on holdout3
          (reports/final3, W17) is the stage "confirmatory holdout3", the PREREG v1.5 two-CF
          final run on holdout4 (reports/final4, W19) is the stage "confirmatory holdout4", and
          the T1 columns prereg / tier carry the pre-registered classification of every row
          (results_sources.yaml: classification).
          Checks: for every T1 row |diff - (mean_model - mean_comparator)| <= 1e-12 against the
          source values; --verify regenerates everything into a temporary folder and compares
          SHA-256 digests (determinism).
          Codes used below: H1-H15 = hypothesis ids of docs/PREREG.md and the development tasks;
          R1 = decayed-purchase-frequency repeat ranker; R2 = logistic-regression repeat ranker;
          R2' = R2 with the item-CF similarity feature; E2 = user-based CF explore module;
          E4 = theta x item-CF explore module; C0-C4, C3b = slot combinations of a repeat and an
          explore module (C3 = R2 + E2, C4 = R2' + E2); P_T = C3 with the PREREG v1.4 slot policy.
Inputs  : The registry file given by --sources (scripts/results_sources.yaml) and the JSON/CSV
          files registered in it. The policy block of an eval JSON ("policy": H11-H13 with 95 %
          and 98.33 % CIs, secondary comparisons, variants, bins) is read like the other blocks;
          its intervals fill the T1 columns ci9833_lo / ci9833_hi / decision_9833. The v15 block
          ("v15": H14 non-inferiority and H15 with 95 % and 97.5 % CIs, the R2'-vs-R2 / R2'-vs-R1
          repeat Recall@10 and C4-vs-C3 Recall@10 secondaries, bins of H14) fills ci975_lo /
          ci975_hi / decision_975; in the holdout4 run every other block carries the 97.5 % level
          as well (profile bootstrap levels).
Outputs : All under --out (reports/results):
          tables/T1_master_hypotheses.csv|.md
              every primary hypothesis and secondary/auxiliary comparison, one row each
              (prereg, tier)
          tables/T2_selections.csv|.md
              selected values per dataset x run with grid-edge flags
          tables/T3_module_screening.csv|.md
              explore modules (new-item Recall@10) and repeat modules (repeat Recall@10)
          tables/T4_final_lists.csv|.md
              combinations C0-C4, C3b, P_T, reference E4 + R1 and the baselines: Recall@10,
              NDCG@10
          tables/T5_history_length.csv|.md
              per history-length bin: diff and 95% CI of H1, H3, H6, H7, H8, H9, H10, H11, H12,
              H13, H14
          tables/T6_module_catalog.csv|.md
              module status catalogue (from results_sources.yaml)
          tables/T7_datasets_eda.csv|.md
              dataset summary copied from the EDA branch (feat/w5a-eda)
          figures/F1..F9 *.png (300 dpi) + *.csv with exactly the plotted numbers (each figure is
              drawn from its CSV)
          RESULTS.md
              index, input provenance, verification summary
Usage   :
    python scripts/make_results.py [--sources scripts/results_sources.yaml] [--out reports/results] [--verify]
'''

from __future__ import annotations

# ---------------------------------------------------------------------------
# imports
# ---------------------------------------------------------------------------
import argparse
import hashlib
import json
import math
import os
import re
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


# ---------------------------------------------------------------------------
# constants
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
ENV_PATTERN = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-([^}]*))?\}")
DIFF_TOL = 1e-12
DPI = 300
# Palette (one hue per role; stage is also carried by labels/hatching, never by colour alone).
COLOR_HOLDOUT, COLOR_DEV, COLOR_THIRD, COLOR_INK, COLOR_MUTED, COLOR_GRID = "#2a78d6", "#eb6834", "#1baf7a", "#0b0b0b", "#52514e", "#e2e1dc"
DATASET_ORDER = {"instacart": 0, "complete_journey": 1}
STAGE_ORDER = {"confirmatory holdout4": 0, "confirmatory holdout3": 1, "confirmatory holdout2": 2, "confirmatory holdout": 3,
               "development medium": 4}
POLICY_C3 = "final_lr_ucf"   # comparator list of the policy comparisons (C3)
V15_SECONDARY = ("repeat_lri_vs_repeat_lr_repeat_recall", "repeat_lri_vs_repeat_star_repeat_recall", "final_lri_ucf_vs_final_lr_ucf_recall")
V15_C4 = "final_lri_ucf"     # base list of the policy_c4 block (PREREG v1.5.1)
PRIMARY_USERS = {"H1": "users with repeat target", "H2": "users with new target", "H3": "all users",
                 "H4": "users with new target", "H5": "users with new target"}


# ---------------------------------------------------------------------------
# provenance helpers
# ---------------------------------------------------------------------------
def sha256_bytes(data: bytes) -> str:
    """Hex SHA-256 of ``data``."""
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    """Hex SHA-256 of a file (hashlib.sha256 over the bytes, read in 1 MiB blocks)."""
    # Start an empty SHA-256 digest.
    digest = hashlib.sha256()
    # Feed the file to the digest in blocks of 1 MiB.
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    # Return the digest as a hex string.
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
    """Registered input files by key, plus the optional inputs that were skipped (with the reason)."""

    sources: dict[str, Source] = field(default_factory=dict)
    skipped: list[dict[str, str]] = field(default_factory=list)

    def add(self, source: Source) -> Source:
        """Register ``source`` under its key and return it."""
        self.sources[source.key] = source
        return source

    def get(self, key: str) -> Source | None:
        """The Source registered under ``key``, or None."""
        return self.sources.get(key)

    def runs(self) -> list[str]:
        """Run names in registration order (holdout, W4a, ...)."""
        # Collect each run name once, in the order of first appearance.
        seen = []
        for source in self.sources.values():
            if source.run not in seen:
                seen.append(source.run)
        return seen


def resolve_input(text: str) -> tuple[Path, Path]:
    """(path to read, path to print) of a registered input: ``${VAR}`` / ``${VAR:-default}`` are expanded from the
    environment and ``~`` from the home folder; a relative path is taken from the repository root. The printed
    path is relative to the repository root (no machine-specific prefix in the generated tables)."""
    # Expand the environment variables and the home folder in the text.
    expanded = os.path.expanduser(ENV_PATTERN.sub(lambda m: os.environ.get(m.group(1)) or (m.group(2) or ""), text))
    # Build the path and take a relative one from the repository root.
    path = Path(expanded)
    if not path.is_absolute():
        path = REPO_ROOT / path
    # Return the path to read and the same path relative to the repository root.
    return path, Path(os.path.relpath(path, REPO_ROOT))


def load_source(key: str, path: Path, run: str, stage: str, label: Path | None = None) -> Source:
    """Read the file at ``path`` (JSON, CSV or text) into a Source ``key`` of ``run`` / ``stage``; ``label`` is the path
    recorded in the tables (default: ``path``)."""
    # Read the bytes of the file and hash them.
    data = path.read_bytes()
    digest = sha256_bytes(data)
    # Parse a JSON file and take the commit recorded in its meta block.
    if path.suffix == ".json":
        doc = json.loads(data.decode("utf-8"))
        commit = str(doc.get("meta", {}).get("git_commit") or "")
    # Parse a CSV file into a DataFrame, without a commit.
    elif path.suffix == ".csv":
        doc = pd.read_csv(path)
        commit = ""
    # Keep any other file as text, without a commit.
    else:
        doc = data.decode("utf-8")
        commit = ""
    # Return the Source, with the label path when one is given.
    return Source(key, label if label is not None else path, run, stage, digest, commit, doc)


def git_show(repo: Path, ref: str, rel_path: str) -> bytes:
    """Bytes of ``rel_path`` at ``ref`` via `git show ref:path` (read-only)."""
    # Run git show for the file at the ref and return its standard output.
    result = subprocess.run(["git", "-C", str(repo), "show", f"{ref}:{rel_path}"], capture_output=True, check=True)
    return result.stdout


def git_rev(repo: Path, ref: str) -> str:
    """Full commit hash of ``ref`` in ``repo`` (`git rev-parse`)."""
    return subprocess.run(["git", "-C", str(repo), "rev-parse", ref], capture_output=True, text=True, check=True).stdout.strip()


def build_registry(cfg: dict[str, Any]) -> Registry:
    """Load every registered file; apply the optional / W8b completion rules; record skips."""
    # Start an empty registry.
    registry = Registry()
    for group, spec in cfg["sources"].items():
        # Resolve the read path and the printed path of every file of the group.
        resolved = {name: resolve_input(str(p)) for name, p in spec["files"].items()}
        files = {name: path for name, (path, _) in resolved.items()}
        # List the files of the group that do not exist.
        missing = [name for name, p in files.items() if not p.exists()]
        # Skip an optional group with missing files and fail for a required one.
        if missing:
            if spec.get("optional"):
                registry.skipped.append({"run": spec["run"], "reason": f"files missing: {', '.join(missing)}"})
                continue
            raise FileNotFoundError(f"{group}: required input missing: {missing}")
        # Load every file of the group as a Source.
        loaded = {name: load_source(f"{group}/{name}", p, spec["run"], spec["stage"], resolved[name][1])
                  for name, p in files.items()}
        # Skip the group when a file still carries the commit prefix that marks it as not complete.
        stale = spec.get("complete_when_commit_is_not")
        if stale:
            stale_files = [name for name, s in loaded.items() if s.git_commit.startswith(stale) or not s.git_commit]
            if stale_files:
                registry.skipped.append({"run": spec["run"], "reason": f"not complete: meta.git_commit still {stale} in {', '.join(stale_files)}"})
                continue
        # Register the loaded sources of the group.
        for source in loaded.values():
            registry.add(source)
    # Take the optional EDA block of the configuration.
    eda = cfg.get("eda")
    if eda:
        # Resolve the EDA repository and the commit of its branch.
        repo, branch = resolve_input(str(eda["repo"]))[0], eda["branch"]
        commit = git_rev(repo, branch)
        # Register every EDA file, read with git show, as a Source.
        for rel in eda["files"]:
            data = git_show(repo, branch, rel)
            key = f"eda/{Path(rel).stem}"
            doc = pd.read_csv(pd.io.common.BytesIO(data)) if rel.endswith(".csv") else data.decode("utf-8")
            registry.add(Source(key, Path(f"{branch}:{rel}"), eda["run"], eda["stage"], sha256_bytes(data), commit, doc))
    # Return the filled registry.
    return registry


def prov(source: Source) -> dict[str, str]:
    """Provenance columns (source_file, source_sha256, git_commit) of ``source``."""
    return {"source_file": source.path.as_posix(), "source_sha256": source.sha256, "git_commit": source.git_commit}


# ---------------------------------------------------------------------------
# generic rows from comparison entries
# ---------------------------------------------------------------------------
def entry_row(entry: dict[str, Any], *, id_: str, hypothesis: str, source: Source, dataset: str, transfer_from: str | None,
              users: str | None = None) -> dict[str, Any]:
    """One T1 row from a comparison ``entry`` of an eval JSON: ``id_`` (entry key), ``hypothesis`` label, provenance of
    ``source``, ``dataset``, ``transfer_from`` and the ``users`` label (fallback when the entry carries none).
    """
    # Read the interval of each level, with NaN bounds when the level is absent.
    ci = entry.get("ci", {})
    ci95 = ci.get("0.95", ci.get(0.95, [np.nan, np.nan]))
    ci99 = ci.get("0.99375", ci.get(0.99375, [np.nan, np.nan]))
    ci98 = ci.get("0.9833", ci.get(0.9833, [np.nan, np.nan]))
    ci975 = ci.get("0.975", ci.get(0.975, [np.nan, np.nan]))
    # Read the decisions recorded per level.
    decision = entry.get("decision", {}) or {}
    # Return the row with the identity, mean, interval, decision and provenance columns.
    return {
        "id": id_, "hypothesis": hypothesis, "stage": source.stage, "run": source.run, "dataset": dataset,
        "transfer_from": transfer_from or "", "metric": entry.get("metric", ""), "users": entry.get("users", users or ""),
        "users_n": int(entry["n_users"]), "comparator": entry["comparator"], "model": entry["model"],
        "mean_comparator": float(entry["comparator_mean"]), "mean_model": float(entry["model_mean"]), "diff": float(entry["diff"]),
        "ci95_lo": float(ci95[0]), "ci95_hi": float(ci95[1]), "ci99375_lo": float(ci99[0]), "ci99375_hi": float(ci99[1]),
        "ci9833_lo": float(ci98[0]), "ci9833_hi": float(ci98[1]), "ci975_lo": float(ci975[0]), "ci975_hi": float(ci975[1]),
        "decision": decision.get("0.95", ""), "decision_99375": decision.get("0.99375", ""), "decision_9833": decision.get("0.9833", ""),
        "decision_975": decision.get("0.975", ""),
        **prov(source),
    }


def hypothesis_of(key: str) -> str:
    """"H<n>" when the entry ``key`` starts with a hypothesis id, else "secondary"."""
    # Take the part of the key before the first underscore.
    head = key.split("_")[0]
    # Return it when it is H followed by digits, else the label secondary.
    return head if head[0] == "H" and head[1:].isdigit() else "secondary"


def eval_rows(source: Source) -> list[dict[str, Any]]:
    """All comparison rows of one pipeline eval JSON (W4 layout, plus lr_repeat / integration blocks of W6/W8)."""
    # Take the parsed document, its dataset and its transfer origin.
    doc, rows = source.doc, []
    dataset, transfer = doc["dataset"], doc.get("transfer_from")
    # Add the rows of the primary block.
    for key, entry in doc.get("primary", {}).items():
        rows.append(entry_row(entry, id_=key, hypothesis=hypothesis_of(key), source=source, dataset=dataset, transfer_from=transfer))
    # Add the row of the top-level "H5" entry when present.
    if doc.get("H5"):
        rows.append(entry_row(doc["H5"], id_="H5", hypothesis="H5", source=source, dataset=dataset, transfer_from=transfer))
    # Add the rows of the secondary block.
    for key, entry in doc.get("secondary", {}).items():
        rows.append(entry_row(entry, id_=key, hypothesis="secondary", source=source, dataset=dataset, transfer_from=transfer))
    # Add the primary and secondary rows of the lr_repeat and integration blocks.
    for block in ("lr_repeat", "integration"):
        if doc.get(block):
            for key, entry in doc[block].get("primary", {}).items():
                rows.append(entry_row(entry, id_=key, hypothesis=hypothesis_of(key), source=source, dataset=dataset, transfer_from=transfer))
            for key, entry in doc[block].get("secondary", {}).items():
                rows.append(entry_row(entry, id_=f"{block}:{key}", hypothesis="secondary", source=source, dataset=dataset, transfer_from=transfer))
    # Add the primary, secondary and variant rows of the policy block.
    if doc.get("policy"):  # W17 layout (PREREG v1.4): primary H11-H13, secondary, variants (r_short = 8, T = 6 / 10)
        block = doc["policy"]
        for key, entry in block.get("primary", {}).items():
            rows.append(entry_row(entry, id_=key, hypothesis=hypothesis_of(key), source=source, dataset=dataset, transfer_from=transfer))
        for key, entry in block.get("secondary", {}).items():
            rows.append(entry_row(entry, id_=f"policy:{key}", hypothesis="secondary", source=source, dataset=dataset, transfer_from=transfer))
        for name, variant in block.get("variants", {}).items():
            for metric, entry in variant.get("comparisons", {}).items():
                rows.append(entry_row(entry, id_=f"policy:{name}_vs_{POLICY_C3}_{metric}", hypothesis="secondary", source=source,
                                      dataset=dataset, transfer_from=transfer))
    # Add the rows of the policy_c4 block, each id prefixed with the block name.
    if doc.get("policy_c4"):  # PREREG v1.5.1: the v1.4 policy with C4 as the base (secondary replication); every id is prefixed
        block = doc["policy_c4"]
        for key, entry in block.get("primary", {}).items():
            rows.append(entry_row(entry, id_=f"policy_c4:{key}", hypothesis=hypothesis_of(key), source=source, dataset=dataset, transfer_from=transfer))
        for key, entry in block.get("secondary", {}).items():
            rows.append(entry_row(entry, id_=f"policy_c4:{key}", hypothesis="secondary", source=source, dataset=dataset, transfer_from=transfer))
        for name, variant in block.get("variants", {}).items():
            for metric, entry in variant.get("comparisons", {}).items():
                rows.append(entry_row(entry, id_=f"policy_c4:{name}_vs_{V15_C4}_{metric}", hypothesis="secondary", source=source,
                                      dataset=dataset, transfer_from=transfer))
    # Add the primary and secondary rows of the v15 block.
    if doc.get("v15"):  # W19 layout (PREREG v1.5): primary H14 (non-inferiority), H15; secondary R2'/C4 comparisons
        block = doc["v15"]
        for key, entry in block.get("primary", {}).items():
            rows.append(entry_row(entry, id_=key, hypothesis=hypothesis_of(key), source=source, dataset=dataset, transfer_from=transfer))
        for key, entry in block.get("secondary", {}).items():
            rows.append(entry_row(entry, id_=f"v15:{key}", hypothesis="secondary", source=source, dataset=dataset, transfer_from=transfer))
    # Add the "H8" row and the auxiliary rows of the stand-alone user-CF layout.
    if "H8" in doc and "auxiliary" in doc:  # W7 layout
        rows.append(entry_row(doc["H8"], id_="H8_new_recall", hypothesis="H8", source=source, dataset=dataset, transfer_from=None,
                              users="users with new target"))
        for key, entry in doc["auxiliary"].items():
            rows.append(entry_row(entry, id_=f"auxiliary:{key}", hypothesis="auxiliary", source=source, dataset=dataset, transfer_from=None))
    # Add the rows of the hypotheses block.
    if "hypotheses" in doc:  # W9 layout
        for key, entry in doc["hypotheses"].items():
            rows.append(entry_row(entry, id_=key, hypothesis=hypothesis_of(key) if key.startswith("H") else "auxiliary",
                                  source=source, dataset=dataset, transfer_from=None))
    # Return the collected rows.
    return rows


def eval_sources(registry: Registry, run: str | None = None) -> list[Source]:
    """Eval JSON sources of ``registry`` (optionally only those of ``run``) in registration order."""
    out = []
    # Walk the registered sources in order.
    for source in registry.sources.values():
        # Skip non-JSON sources and sources of another run.
        if source.path.suffix != ".json" or (run and source.run != run):
            continue
        # Keep the documents of the eval split and those with a hypotheses block.
        doc = source.doc
        if doc.get("split") == "eval" or "hypotheses" in doc:
            out.append(source)
    # Return the kept sources.
    return out


# ---------------------------------------------------------------------------
# tables
# ---------------------------------------------------------------------------
T1_COLUMNS = ["id", "hypothesis", "prereg", "tier", "stage", "run", "dataset", "transfer_from", "metric", "users", "users_n", "comparator", "model",
              "mean_comparator", "mean_model", "diff", "ci95_lo", "ci95_hi", "ci99375_lo", "ci99375_hi", "ci9833_lo", "ci9833_hi",
              "ci975_lo", "ci975_hi", "decision", "decision_99375", "decision_9833", "decision_975", "current", "superseded_reason",
              "source_file", "source_sha256", "git_commit"]


def metric_key(metric: str) -> str:
    """Normalised metric name used to match the same comparison across runs."""
    # Lower-case the metric name and drop the @10 suffix.
    text = str(metric).lower().replace("@10", "").strip()
    # Map the name to the first matching metric family, with recall as the fallback.
    if "category" in text:
        return "category_hit"
    if "hit share" in text or "hit_share" in text:
        return "new_hit_share"
    if "new" in text:
        return "new_recall"
    if "repeat" in text:
        return "repeat_recall"
    if "ndcg" in text:
        return "ndcg"
    return "recall"


def flag_current(frame: pd.DataFrame, cfg: dict[str, Any]) -> pd.DataFrame:
    """Mark the row of the highest-priority run per (stage, dataset, transfer, comparator, model, metric) as current.

    Priority comes from cfg["run_priority"] (smaller = newer); every other row of the group gets
    superseded_reason = cfg["supersession_notes"][run] + the current run.
    """
    # Read the run priorities, the supersession notes and the list aliases.
    priority, notes, aliases = cfg["run_priority"], cfg.get("supersession_notes", {}), cfg.get("list_aliases", {})
    # Work on a copy of the frame.
    frame = frame.copy()
    # Build the group key columns with aliased list names and normalised metric names.
    key = pd.DataFrame({
        "stage": frame["stage"], "dataset": frame["dataset"], "transfer_from": frame["transfer_from"].fillna(""),
        "comparator": frame["comparator"].map(lambda x: aliases.get(x, x)), "model": frame["model"].map(lambda x: aliases.get(x, x)),
        "metric": frame["metric"].map(metric_key),
    })
    # Join the key columns into one string per row.
    frame["_key"] = key.astype(str).agg("|".join, axis=1)
    # Look up the priority of each row's run and fail when a run has none.
    frame["_prio"] = frame["run"].map(priority)
    if frame["_prio"].isna().any():
        raise ValueError(f"run_priority missing for runs {sorted(frame.loc[frame['_prio'].isna(), 'run'].unique())}")
    # Mark the rows whose priority equals the smallest priority of their group.
    best = frame.groupby("_key")["_prio"].transform("min")
    frame["current"] = frame["_prio"] == best
    # Find the run of the current row of each group.
    current_run = frame[frame["current"]].groupby("_key")["run"].first()
    # Write the supersession reason of every row that is not current.
    frame["superseded_reason"] = [
        "" if is_current else f"{notes.get(run, 'later run available')}; current row: {current_run[key_]}"
        for is_current, run, key_ in zip(frame["current"], frame["run"], frame["_key"])
    ]
    # Return the frame without the helper columns.
    return frame.drop(columns=["_key", "_prio"])


def hypothesis_order(label: str) -> float:
    """Sort key of a T1 hypothesis label: H<n> -> n, H<n>' -> n + 0.5, anything else -> 99."""
    # Match the label against H<n> with an optional prime.
    match = re.fullmatch(r"H([0-9]+)('?)", str(label))
    # Return n, plus 0.5 with a prime, or 99 without a match.
    return int(match.group(1)) + (0.5 if match.group(2) else 0.0) if match else 99.0


def classify(frame: pd.DataFrame, cfg: dict[str, Any]) -> pd.DataFrame:
    """Add the T1 columns ``prereg`` and ``tier`` from cfg["classification"] and apply its per-stage id renames.

    The classification is explicit per stage (results_sources.yaml) and independent of the JSON block names: a rule
    marks the rows of its stage whose id is in ``ids`` and dataset in ``datasets`` and which are not transfer rows;
    rows matched by no rule get the stage's ``default_tier``. ``rename`` replaces the T1 id and hypothesis label of
    an entry (the C3-vs-C0 comparisons of the v1.3 stage become H7prime_ndcg / H7prime_recall, hypothesis H7').
    Every stage present in ``frame`` must be configured.
    """
    # Fail when a stage of the frame has no classification.
    spec = cfg["classification"]
    missing = sorted(set(frame["stage"]) - set(spec))
    if missing:
        raise ValueError(f"classification missing for stages {missing}")
    # Work on a copy with empty prereg and tier columns.
    frame = frame.copy()
    frame["prereg"], frame["tier"] = "", ""
    for stage, rules in spec.items():
        # Select the rows of the stage.
        mask = frame["stage"] == stage
        # Apply the id and hypothesis renames of the stage.
        for old, new in (rules.get("rename") or {}).items():
            hit = mask & (frame["id"] == old)
            frame.loc[hit, "id"], frame.loc[hit, "hypothesis"] = new["id"], new["hypothesis"]
        # Set the prereg label and the default tier of the stage.
        frame.loc[mask, "prereg"] = str(rules.get("prereg") or "")
        frame.loc[mask, "tier"] = str(rules["default_tier"])
        # Set the tier of the non-transfer rows matched by each rule.
        for rule in rules.get("rules") or []:
            hit = (mask & frame["id"].isin(rule["ids"]) & frame["dataset"].isin(rule["datasets"])
                   & (frame["transfer_from"].fillna("") == ""))
            frame.loc[hit, "tier"] = str(rule["tier"])
    # Return the classified frame.
    return frame


def table_t1(registry: Registry, cfg: dict[str, Any]) -> pd.DataFrame:
    """T1 master hypotheses table from every eval source of ``registry``, with the current / superseded flags and the
    prereg / tier classification of ``cfg``."""
    # Collect the comparison rows of every eval source.
    rows = []
    for source in eval_sources(registry):
        rows.extend(eval_rows(source))
    # Flag the current rows, classify them and keep the T1 columns.
    frame = pd.DataFrame(rows)
    frame = classify(flag_current(frame, cfg), cfg)[T1_COLUMNS]
    # Add sort keys for stage, dataset and hypothesis.
    frame["_s"], frame["_d"] = frame["stage"].map(STAGE_ORDER), frame["dataset"].map(DATASET_ORDER)
    frame["_h"] = frame["hypothesis"].map(hypothesis_order)
    # Sort with a stable sort and drop the sort keys.
    frame = frame.sort_values(["_s", "run", "_d", "transfer_from", "_h", "id"], kind="mergesort").drop(columns=["_s", "_d", "_h"])
    # Return the table with a fresh row index.
    return frame.reset_index(drop=True)


def check_t1(frame: pd.DataFrame) -> dict[str, Any]:
    """diff must equal mean_model - mean_comparator within DIFF_TOL for every row."""
    # Compute the absolute gap between diff and the difference of the two means.
    gap = (frame["diff"] - (frame["mean_model"] - frame["mean_comparator"])).abs()
    # Fail when a row exceeds the tolerance.
    bad = frame.loc[gap > DIFF_TOL, ["id", "run", "dataset"]]
    if len(bad):
        raise AssertionError(f"T1 diff identity violated (tol {DIFF_TOL}):\n{bad.to_string()}")
    # Return the row count, the largest gap, the tolerance and the number of current rows.
    return {"rows": int(len(frame)), "max_abs_gap": float(gap.max()), "tolerance": DIFF_TOL, "current_rows": int(frame["current"].sum())}


def _sel_row(source: Source, dataset: str, parameter: str, value: Any, edge: Any, grid: Any, note: str = "") -> dict[str, Any]:
    '''One T2 row: ``parameter`` of ``dataset`` with the selected ``value``, the grid-edge flag
    ``edge``, the ``grid`` as JSON text, the ``note`` and the provenance of ``source``.'''
    return {"stage": source.stage, "run": source.run, "dataset": dataset, "parameter": parameter, "value": value,
            "at_grid_edge": edge, "grid": json.dumps(grid) if grid is not None else "", "note": note, **prov(source)}


def table_t2(registry: Registry) -> pd.DataFrame:
    """T2 selected values (long format) from every tuning source of ``registry``."""
    rows = []
    # Walk the registered JSON sources.
    for source in registry.sources.values():
        if source.path.suffix != ".json":
            continue
        # Take the parsed document and its dataset.
        doc = source.doc
        dataset = doc.get("dataset")
        # Add the selections of a pipeline tuning file.
        if doc.get("split") == "fit" and "stage1_repeat_recall" in doc:  # pipeline tuning (W4a, W6, W8, holdout)
            # Rebuild the grid of each parameter from the three tuning stages.
            grids = {"gamma": [r["gamma"] for r in doc["stage1_repeat_recall"]],
                     "gamma_c": sorted({r["gamma_c"] for r in doc["stage2_explore_new_recall"]}),
                     "alpha": sorted({r["alpha"] for r in doc["stage2_explore_new_recall"]}),
                     "r": [r["r"] for r in doc["stage3_final_ndcg"]]}
            # Add one row per selected pipeline parameter.
            for name in ("gamma", "gamma_c", "alpha", "r"):
                rows.append(_sel_row(source, dataset, name, doc["selections"][name], doc["at_grid_edge"][name], grids[name]))
            # Add the regularisation strength C of the logistic-regression repeat ranker.
            if doc.get("lr_repeat"):
                lr = doc["lr_repeat"]
                rows.append(_sel_row(source, dataset, "C (repeat_lr)", lr["C"], _edge(lr["C"], lr["C_grid"]), lr["C_grid"],
                                     "chosen by validation repeat Recall@10; ties -> smaller C"))
            # Add the user-CF selections and the re-selected r of each combination.
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
            # Add the C and r selections of the v15 block.
            if doc.get("v15"):  # PREREG v1.5: C of R2' (repeat_lri) and r of C4 (final_lri_ucf)
                block = doc["v15"]
                c_grid = block["lr"]["C_grid"]
                rows.append(_sel_row(source, dataset, "C (repeat_lri)", block["selections"]["C"], _edge(block["selections"]["C"], c_grid), c_grid,
                                     "chosen by validation repeat Recall@10 on the validation users of repeat_lr; ties -> smaller C"))
                for combo, r_value in block["selections"]["r"].items():
                    rows.append(_sel_row(source, dataset, f"r ({combo})", r_value, block["at_grid_edge"]["r"][combo], block["r_grid"],
                                         block["r_selection_users"][combo]["users"]))
        # Add the k and weighting selections of a stand-alone user-CF tuning file.
        elif doc.get("split") == "fit" and "grid" in doc:  # W7 user-CF tuning
            rows.append(_sel_row(source, dataset, "user-CF k", doc["selections"]["k"], doc["at_grid_edge"]["k"], sorted({g["k"] for g in doc["grid"]})))
            rows.append(_sel_row(source, dataset, "user-CF weighting", doc["selections"]["weighting"], doc["at_grid_edge"]["weighting"],
                                 sorted({g["weighting"] for g in doc["grid"]})))
        # Add the C of both re-fitted rankers and the gamma read from the pipeline tuning.
        elif "fit" in doc and "r3_config" in doc:  # W9
            for name in ("r2", "r3"):
                block = doc["fit"][name]
                rows.append(_sel_row(source, dataset, f"C ({'repeat_lr re-fit' if name == 'r2' else 'repeat_r3'})", block["C"],
                                     block["at_grid_edge"], block["C_grid"], "chosen by validation repeat Recall@10; ties -> smaller C"))
            rows.append(_sel_row(source, dataset, "gamma (R1 comparator, read from W4 tuning)", doc["gamma_star"]["value"], "", None,
                                 f"read from {Path(doc['gamma_star']['source']).name}"))  # file name only: no machine-specific path
        # Add the selections that a transfer evaluation applied.
        elif doc.get("split") == "eval" and doc.get("transfer_from"):  # transfer evaluations: selections applied
            for name in ("gamma", "gamma_c", "alpha", "r"):
                rows.append(_sel_row(source, dataset, f"{name} (transfer from {doc['transfer_from']})", doc["selections"][name], "", None,
                                     f"read from {Path(doc['selections']['source']).name}" if doc["selections"].get("source") else ""))
    # Build the frame and add sort keys for stage and dataset.
    frame = pd.DataFrame(rows)
    frame["_s"], frame["_d"] = frame["stage"].map(STAGE_ORDER), frame["dataset"].map(DATASET_ORDER)
    # Sort with a stable sort, drop the sort keys and return the table.
    return frame.sort_values(["_s", "run", "_d", "parameter"], kind="mergesort").drop(columns=["_s", "_d"]).reset_index(drop=True)


def _edge(value: float, grid: list[float]) -> bool:
    '''True when ``value`` equals the smallest or the largest value of ``grid``.'''
    return math.isclose(value, min(grid), rel_tol=1e-12) or math.isclose(value, max(grid), rel_tol=1e-12)


def _find(doc: dict[str, Any], *keys: str) -> dict[str, Any] | None:
    """Comparison entry found by key across primary / H5 / secondary / lr_repeat / integration blocks."""
    # List the blocks to search, in order.
    places = [doc.get("primary", {}), doc.get("secondary", {})]
    if doc.get("H5"):
        places.append({"H5": doc["H5"]})
    for block in ("lr_repeat", "integration", "v15"):
        if doc.get(block):
            places += [doc[block].get("primary", {}), doc[block].get("secondary", {})]
    if "hypotheses" in doc:
        places.append(doc["hypotheses"])
    # Return the first entry found, trying the keys in the order given.
    for key in keys:
        for place in places:
            if key in place:
                return place[key]
    # Return None when no key is found.
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
    ("R2'", "logistic-regression ranker + item-CF feature", "repeat_lri", ("repeat_lri_vs_repeat_lr_repeat_recall", "model_mean"),
     "repeat_lri_vs_repeat_lr_repeat_recall"),
    ("R3", "logistic ranker + binned cycle features", "repeat_r3", ("H10_repeat_recall", "model_mean"), "H10_repeat_recall"),
]


def table_t3(registry: Registry, dev_run: str | None) -> pd.DataFrame:
    """T3 module screening: explore modules by new-item Recall@10 and repeat modules by repeat Recall@10; development
    rows come from ``dev_run``.
    """
    rows = []
    # Walk the eval sources.
    for source in eval_sources(registry):
        doc = source.doc
        # Skip transfer evaluations.
        if doc.get("transfer_from"):
            continue
        # Keep only the chosen development run and the run "W9" among the development sources.
        if source.stage == "development medium" and source.run not in (dev_run, "W9"):
            continue
        # Go through the explore modules and then the repeat modules.
        for group, modules, metric in (("explore", EXPLORE_MODULES, "new-item Recall@10"), ("repeat", REPEAT_MODULES, "repeat Recall@10")):
            for module, name, list_name, (key, field_), comp_key in modules:
                # Skip a module whose entry or mean is absent from the document.
                entry = _find(doc, key)
                if entry is None:
                    continue
                if entry[field_ if field_ == "comparator_mean" else "model_mean"] is None:
                    continue
                # Start the row with the module mean and empty comparator columns.
                row = {"stage": source.stage, "run": source.run, "dataset": doc["dataset"], "group": group, "module": module,
                       "module_name": name, "list": list_name, "metric": metric, "users_n": int(entry["n_users"]),
                       "mean": float(entry[field_]), "comparator": "", "diff_vs_comparator": np.nan, "ci95_lo": np.nan, "ci95_hi": np.nan}
                # Fill the difference and its 95 % interval from the comparator entry.
                if comp_key:
                    comp = _find(doc, comp_key)
                    ci95 = comp["ci"].get("0.95", [np.nan, np.nan])
                    row.update({"comparator": comp["comparator"], "diff_vs_comparator": float(comp["diff"]),
                                "ci95_lo": float(ci95[0]), "ci95_hi": float(ci95[1])})
                # Append the row with the provenance columns.
                rows.append({**row, **prov(source)})
    # Keep one row per stage, run, dataset, group and module.
    frame = pd.DataFrame(rows)
    frame = frame.drop_duplicates(subset=["stage", "run", "dataset", "group", "module"])
    # Add sort keys for stage, dataset, group and module order.
    frame["_s"], frame["_d"] = frame["stage"].map(STAGE_ORDER), frame["dataset"].map(DATASET_ORDER)
    frame["_g"] = frame["group"].map({"explore": 0, "repeat": 1})
    frame["_m"] = frame["module"].map({m: i for i, m in enumerate([m[0] for m in EXPLORE_MODULES + REPEAT_MODULES])})
    # Sort with a stable sort and drop the sort keys.
    frame = frame.sort_values(["_s", "_d", "_g", "_m"], kind="mergesort").drop(columns=["_s", "_d", "_g", "_m"])
    # Return the table with a fresh row index.
    return frame.reset_index(drop=True)


def table_t4(registry: Registry, cfg: dict[str, Any], dev_run: str | None) -> pd.DataFrame:
    """T4 Recall@10 / NDCG@10 of the final lists and baselines per dataset (combination labels of ``cfg``; development
    rows from ``dev_run``).
    """
    # Map each list name to the first combination label that uses it.
    combos = cfg["combinations"]
    by_list: dict[str, tuple[str, dict[str, Any]]] = {}
    for label, spec in combos.items():  # first label per list name wins (P_T = the base-C3 policy list; P_T(C4) is handled below)
        if spec["list"] and spec["list"] not in by_list:
            by_list[spec["list"]] = (label, spec)
    rows = []
    # Walk the eval sources.
    for source in eval_sources(registry):
        doc = source.doc
        # Skip transfer evaluations and documents without list metrics.
        if doc.get("transfer_from") or "list_metrics" not in doc:
            continue
        # Keep only the chosen development run among the development sources.
        if source.stage == "development medium" and source.run != dev_run:
            continue
        # Choose the lists to report, with a fixed set for the first holdout stage.
        wanted = [l for l in by_list if l] + ["personal_frequency", "popularity"]
        if source.stage == "confirmatory holdout":
            wanted = ["final", "final_cf", "personal_frequency", "popularity"]
        for list_name in wanted:
            # Skip a list without metrics in the document.
            metrics = doc["list_metrics"].get(list_name)
            if metrics is None:
                continue
            # Look up the combination label and its specification.
            label, spec = by_list.get(list_name, (list_name, {}))
            # Take r from the block that selected it for this list.
            r_value = doc["selections"].get("r") if list_name in ("final", "final_cf", "final_lr") else None
            if doc.get("integration") and list_name in doc["integration"]["selections"].get("r", {}):
                r_value = doc["integration"]["selections"]["r"][list_name]
            if doc.get("v15") and list_name in doc["v15"]["selections"].get("r", {}):  # C4 (PREREG v1.5)
                r_value = doc["v15"]["selections"]["r"][list_name]
            # Append the row of the list.
            rows.append({"combination": label, "list": list_name, "repeat_module": spec.get("repeat", ""), "explore_module": spec.get("explore", ""),
                         "r_rule": spec.get("r_rule", ""), "r": r_value if r_value is not None else "", "stage": source.stage, "run": source.run,
                         "dataset": doc["dataset"], "users_n": int(doc["n_users"]["all"]), "recall10": float(metrics["recall"]),
                         "ndcg10": float(metrics["ndcg"]), "note": spec.get("note", ""), **prov(source)})
        # Append the row of the policy list of the policy block.
        if doc.get("policy") and "P_T" in by_list:  # W17: the policy list P_T (its C3 row is the final_lr_ucf row above)
            block, (label, spec) = doc["policy"], by_list["P_T"]
            metrics = block["list_metrics"]["P_T"]
            rows.append({"combination": label, "list": "P_T", "repeat_module": spec.get("repeat", ""), "explore_module": spec.get("explore", ""),
                         "r_rule": spec.get("r_rule", ""), "r": int(block["r_default"]), "stage": source.stage, "run": source.run,
                         "dataset": doc["dataset"], "users_n": int(block["n_users"]["all"]), "recall10": float(metrics["recall"]),
                         "ndcg10": float(metrics["ndcg"]),
                         "note": f"{spec.get('note', '')}; r_u = {block['r_short']} for {block['n_users']['short_history']} users with "
                                 f"history_len <= {block['T']}, r* = {block['r_default']} otherwise", **prov(source)})
        # Append the row of the policy list of the policy_c4 block.
        if doc.get("policy_c4") and "P_T(C4)" in combos:  # PREREG v1.5.1: the policy list on C4 (its C4 row is the final_lri_ucf row above)
            block, (label, spec) = doc["policy_c4"], ("P_T(C4)", combos["P_T(C4)"])
            metrics = block["list_metrics"]["P_T"]
            rows.append({"combination": label, "list": "P_T", "repeat_module": spec.get("repeat", ""), "explore_module": spec.get("explore", ""),
                         "r_rule": spec.get("r_rule", ""), "r": int(block["r_default"]), "stage": source.stage, "run": source.run,
                         "dataset": doc["dataset"], "users_n": int(block["n_users"]["all"]), "recall10": float(metrics["recall"]),
                         "ndcg10": float(metrics["ndcg"]),
                         "note": f"{spec.get('note', '')}; r_u = {block['r_short']} for {block['n_users']['short_history']} users with "
                                 f"history_len <= {block['T']}, r (C4) = {block['r_default']} otherwise", **prov(source)})
        # Append one empty row per combination without a list.
        if source.stage == "development medium":  # planned combinations without a list (e.g. C4): one empty row per dataset
            for label, spec in combos.items():
                if not spec["list"]:
                    rows.append({"combination": label, "list": "", "repeat_module": spec.get("repeat", ""), "explore_module": spec.get("explore", ""),
                                 "r_rule": "", "r": "", "stage": source.stage, "run": "", "dataset": doc["dataset"], "users_n": "",
                                 "recall10": np.nan, "ndcg10": np.nan, "note": f"{spec.get('status', 'not run')}: {spec.get('note', '')}",
                                 "source_file": "", "source_sha256": "", "git_commit": ""})
    # Build the frame and write r as integer text.
    frame = pd.DataFrame(rows)
    frame["r"] = frame["r"].map(lambda v: "" if v == "" or v is None else str(int(v)))
    # Add sort keys for stage, dataset and combination order.
    frame["_s"], frame["_d"] = frame["stage"].map(STAGE_ORDER), frame["dataset"].map(DATASET_ORDER)
    frame["_c"] = frame["combination"].map({c: i for i, c in enumerate(list(combos) + ["personal_frequency", "popularity"])})
    # Sort with a stable sort, drop the sort keys and return the table.
    return frame.sort_values(["_s", "_d", "_c"], kind="mergesort").drop(columns=["_s", "_d", "_c"]).reset_index(drop=True)


T5_KEYS = ["H1_repeat_recall", "H3_ndcg", "H3_recall", "H6_repeat_recall", "H7_ndcg", "H7_recall", "H8_new_recall", "H9_ndcg", "H9_recall",
           "H10_repeat_recall", "H11_ndcg", "H12_new_hit_share", "H13_new_hit_share", "H14_ndcg"]


def table_t5(registry: Registry, dev_run: str | None) -> pd.DataFrame:
    """T5 per history-length bin differences with 95% CIs (development rows from ``dev_run``)."""
    rows = []
    # Walk the eval sources.
    for source in eval_sources(registry):
        doc = source.doc
        # Skip transfer evaluations.
        if doc.get("transfer_from"):
            continue
        # Keep only the chosen development run and the run "W9" among the development sources.
        if source.stage == "development medium" and source.run not in (dev_run, "W9"):
            continue
        # Collect the history-length blocks of the document and of its sub-blocks.
        blocks = [doc.get("by_history_length", [])]
        for block in ("lr_repeat", "integration", "policy", "v15"):
            if doc.get(block):
                blocks.append(doc[block].get("by_history_length", []))
        # Go through every bin of every block and every key of the table.
        for block in blocks:
            for bin_row in block:
                for key in T5_KEYS:
                    entry = bin_row.get(key)
                    # Write a row without values when the key is present with no entry.
                    if entry is None and key in bin_row:
                        rows.append({"stage": source.stage, "run": source.run, "dataset": doc["dataset"], "bin": bin_row["bin"],
                                     "bin_users": int(bin_row["n_users"]), "id": key, "hypothesis": hypothesis_of(key), "users_n": 0,
                                     "mean_comparator": np.nan, "mean_model": np.nan, "diff": np.nan, "ci95_lo": np.nan, "ci95_hi": np.nan,
                                     **prov(source)})
                    # Write the difference and its 95 % interval when the entry exists.
                    elif entry is not None:
                        ci = entry.get("ci_0.95") or entry.get("ci", {}).get("0.95")
                        rows.append({"stage": source.stage, "run": source.run, "dataset": doc["dataset"], "bin": bin_row["bin"],
                                     "bin_users": int(bin_row["n_users"]), "id": key, "hypothesis": hypothesis_of(key),
                                     "users_n": int(entry["n_users"]), "mean_comparator": float(entry["comparator_mean"]),
                                     "mean_model": float(entry["model_mean"]), "diff": float(entry["diff"]), "ci95_lo": float(ci[0]),
                                     "ci95_hi": float(ci[1]), **prov(source)})
    # Build the frame and add sort keys for stage, dataset, key order and bin start.
    frame = pd.DataFrame(rows)
    frame["_s"], frame["_d"] = frame["stage"].map(STAGE_ORDER), frame["dataset"].map(DATASET_ORDER)
    frame["_h"] = frame["id"].map({k: i for i, k in enumerate(T5_KEYS)})
    frame["_b"] = frame["bin"].map(lambda b: float(b.split(",")[0].strip("(")))
    # Sort with a stable sort, drop the sort keys and return the table.
    return frame.sort_values(["_s", "run", "_d", "_h", "_b"], kind="mergesort").drop(columns=["_s", "_d", "_h", "_b"]).reset_index(drop=True)


def table_t6(cfg: dict[str, Any], t1: pd.DataFrame) -> pd.DataFrame:
    """T6 module and combination catalogue from ``cfg`` (modules, combinations) with the basis rows of ``t1``."""
    rows = []
    # Start from the module entries of the configuration.
    entries = list(cfg["modules"])
    # Add one entry per combination, named by its explore module, repeat module and r rule.
    for label, spec in cfg["combinations"].items():
        name = f"{spec['explore']} + {spec['repeat']}" + (f", {spec['r_rule']}" if spec.get("r_rule") else "")
        entries.append({"id": label, "group": "combination", "name": name, "list": spec.get("list", ""), "status": spec["status"],
                        "basis": spec.get("basis", ""), "note": spec.get("note", "")})
    for module in entries:
        # Find the hypothesis ids named in the basis text.
        ids = re.findall(r"H[0-9]+'?", str(module.get("basis", "")))  # H<n> and H<n>' (H7' = C3 vs C0 on holdout2)
        # List the stages of the T1 rows of those hypotheses.
        stages = sorted(set(t1.loc[t1["hypothesis"].isin(ids), "stage"])) if ids else []
        # Append the catalogue row.
        rows.append({"id": module["id"], "group": module["group"], "name": module["name"], "list": module.get("list", ""),
                     "status": module["status"], "basis": module.get("basis", ""), "evidence_stage": "; ".join(stages),
                     "note": module.get("note", "")})
    # Return the catalogue as a DataFrame.
    return pd.DataFrame(rows)


def table_t7(registry: Registry) -> pd.DataFrame:
    """T7 dataset summary copied from the EDA table registered in ``registry`` (key eda/table_datasets)."""
    # Copy the EDA dataset table.
    source = registry.get("eda/table_datasets")
    frame = source.doc.copy()
    # Put the stage and run columns first.
    frame.insert(0, "stage", source.stage)
    frame.insert(1, "run", source.run)
    # Add the provenance columns.
    for key, value in prov(source).items():
        frame[key] = value
    # Return the table.
    return frame


# ---------------------------------------------------------------------------
# markdown / csv writers
# ---------------------------------------------------------------------------
def _fmt(value: Any) -> str:
    '''Text of ``value`` for a Markdown cell: empty for None / NaN, plain booleans and integers,
    floats with up to six decimals, other values with pipes escaped and newlines replaced.'''
    # Write None and NaN as an empty cell.
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return ""
    # Write a boolean as True or False.
    if isinstance(value, (bool, np.bool_)):
        return str(bool(value))
    # Write an integer without decimals.
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    # Write a float with six decimals and no trailing zeros, or with six digits from 1e6 on.
    if isinstance(value, (float, np.floating)):
        return f"{value:.6f}".rstrip("0").rstrip(".") if abs(value) < 1e6 else f"{value:.6g}"
    # Escape pipes and replace newlines in any other value.
    return str(value).replace("|", "\\|").replace("\n", " ")


def to_markdown(frame: pd.DataFrame, title: str, note: str = "", hide: tuple[str, ...] = ("source_sha256",)) -> str:
    """Markdown table of ``frame`` headed by ``title`` and ``note``; the columns in ``hide`` are left out."""
    # Keep the columns that are not hidden.
    columns = [c for c in frame.columns if c not in hide]
    # Start with the title and the optional note.
    lines = [f"# {title}", ""]
    if note:
        lines += [note, ""]
    # Add the header row and the separator row.
    lines.append("| " + " | ".join(columns) + " |")
    lines.append("|" + "|".join(" --- " for _ in columns) + "|")
    # Add one row per record.
    for _, row in frame.iterrows():
        lines.append("| " + " | ".join(_fmt(row[c]) for c in columns) + " |")
    # Name the hidden columns below the table.
    if hide:
        lines += ["", f"Hidden columns (see the CSV): {', '.join(h for h in hide if h in frame.columns)}."]
    # Return the lines as one text ending with a newline.
    return "\n".join(lines) + "\n"


def write_table(frame: pd.DataFrame, out_dir: Path, name: str, title: str, note: str = "") -> dict[str, Any]:
    """Write ``frame`` as <out_dir>/<name>.csv and .md (``title``, ``note``); returns the table record (name, rows, path)."""
    # Create the output folder.
    out_dir.mkdir(parents=True, exist_ok=True)
    # Write the CSV and the Markdown file with LF line ends.
    csv_path, md_path = out_dir / f"{name}.csv", out_dir / f"{name}.md"
    frame.to_csv(csv_path, index=False, lineterminator="\n", encoding="utf-8")
    md_path.write_text(to_markdown(frame, title, note), encoding="utf-8", newline="\n")
    # Return the table record.
    return {"name": name, "rows": int(len(frame)), "csv": csv_path, "md": md_path}


# ---------------------------------------------------------------------------
# figures (each drawn from its own CSV)
# ---------------------------------------------------------------------------
def _plt():
    '''Import matplotlib with the Agg backend, set the shared rcParams and return pyplot.'''
    # Select the non-interactive Agg backend before importing pyplot.
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Set the font sizes, the colours and the resolution used by every figure.
    plt.rcParams.update({"font.size": 8, "axes.titlesize": 9, "axes.labelsize": 8, "legend.fontsize": 7, "axes.edgecolor": COLOR_MUTED,
                         "xtick.color": COLOR_MUTED, "ytick.color": COLOR_MUTED, "axes.labelcolor": COLOR_INK, "savefig.dpi": DPI,
                         "figure.dpi": 100})
    # Return the pyplot module.
    return plt


def _style(ax) -> None:
    '''Hide the top and right spines of ``ax`` and draw the x-axis grid behind the marks.'''
    # Hide the top and right spines.
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    # Draw the x-axis grid below the plotted marks.
    ax.grid(True, axis="x", color=COLOR_GRID, linewidth=0.6)
    ax.set_axisbelow(True)


def _finish(fig, path: Path, plt, footer: str, top: float = 0.94) -> None:
    """Wrap the source footer to the figure width, reserve room for it, then save the PNG (300 dpi)."""
    # Wrap each footer line to a width that grows with the figure width.
    width = max(60, int(fig.get_figwidth() * 19))
    text = "\n".join(textwrap.fill(part, width) for part in footer.split("\n"))
    # Reserve a bottom margin in proportion to the number of footer lines.
    n_lines = text.count("\n") + 1
    bottom = 0.02 + 0.026 * n_lines * (8.0 / fig.get_figheight())
    # Lay out the axes inside the remaining area.
    fig.tight_layout(rect=(0, bottom, 1, top))
    # Write the footer in the lower-left corner.
    fig.text(0.01, 0.01, text, fontsize=6.5, color=COLOR_MUTED, ha="left", va="bottom")
    # Create the folder and save the PNG without the software metadata entry.
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=DPI, metadata={"Software": None})
    # Close the figure.
    plt.close(fig)


def _forest(frame: pd.DataFrame, path: Path, title: str, footer: str, color: str, labels: dict[str, str], datasets: dict[str, str],
            panel_column: str = "dataset", thin: tuple[str, str] = ("ci99375_lo", "ci99375_hi"),
            margins: dict[str, float] | None = None) -> None:
    """Forest plot of ``frame`` (one panel per value of ``panel_column`` listed in ``datasets``: value -> panel title) into
    ``path``; the numbers are re-read from the CSV next to ``path``. ``thin`` names the CSV columns of the thin bar
    (default: the 99.375 % CI); ``margins`` maps a row id to an x value drawn as a dashed vertical segment on that row
    (the -delta non-inferiority margin of H11)."""
    # Load pyplot and re-read the plotted numbers from the CSV.
    plt = _plt()
    frame = pd.read_csv(path.with_suffix(".csv"))  # drawn from the CSV written just before
    # Keep the panels that have rows, in the configured order.
    panels = [d for d in datasets if d in set(frame[panel_column])]
    # Size the figure by the number of panels, the longest label and the number of rows.
    longest = max(len(labels.get(i, i)) for i in dict.fromkeys(frame["id"]))
    fig, axes = plt.subplots(1, len(panels), figsize=(3.9 * len(panels) + 0.07 * longest + 0.6, 0.42 * frame["id"].nunique() + 2.2), sharey=True)
    # Make the axes an array and list the row ids in order of first appearance.
    axes = np.atleast_1d(axes)
    ids = list(dict.fromkeys(frame["id"]))
    for ax, dataset in zip(axes, panels):
        # Select the rows of the panel.
        part = frame[frame[panel_column] == dataset].set_index("id")
        for i, id_ in enumerate(ids):
            # Place the id on its row, counted from the top, and skip it when the panel lacks it.
            y = len(ids) - 1 - i
            if id_ not in part.index:
                continue
            row = part.loc[id_]
            # Draw the thin interval when its lower bound is present.
            if not (isinstance(row.get(thin[0]), float) and math.isnan(row[thin[0]])):
                ax.plot([row[thin[0]], row[thin[1]]], [y, y], color=color, linewidth=0.8, solid_capstyle="butt", zorder=2)
            # Draw the thick 95 % interval.
            ax.plot([row["ci95_lo"], row["ci95_hi"]], [y, y], color=color, linewidth=3.0, solid_capstyle="butt", zorder=3)
            # Draw the dashed margin segment of the row when one is given.
            if margins and id_ in margins:
                ax.plot([margins[id_], margins[id_]], [y - 0.42, y + 0.42], color=COLOR_INK, linewidth=1.0, linestyle=(0, (3, 2)), zorder=5)
            # Draw a circle for a row without a note and a diamond otherwise.
            marker = "o" if str(row.get("note", "")) == "" or (isinstance(row.get("note"), float) and math.isnan(row.get("note"))) else "D"
            ax.plot(row["diff"], y, marker=marker, markersize=4.5, color="white", markeredgecolor=color, markeredgewidth=1.2, zorder=4)
        # Draw the zero line.
        ax.axvline(0.0, color=COLOR_INK, linewidth=0.8, zorder=1)
        # Label the rows from top to bottom.
        ax.set_yticks(range(len(ids)))
        ax.set_yticklabels([labels.get(i, i) for i in reversed(ids)])
        # Put the panel name, the stage label and the largest user count in the title.
        n = int(part["users_n"].max()) if len(part) else 0
        ax.set_title(f"{datasets[dataset]} ({part['stage_label'].iloc[0]})\nusers per row up to n = {n:,}", loc="left")
        # Label the x axis and apply the shared axis style.
        ax.set_xlabel("diff = model - comparator (per-user means)")
        _style(ax)
    # Add the figure title and save the figure with the footer.
    fig.suptitle(title, x=0.01, ha="left", fontsize=10, color=COLOR_INK)
    _finish(fig, path, plt, footer, top=0.93)


def figure_f1(t1: pd.DataFrame, out: Path, cfg: dict[str, Any]) -> dict[str, Any]:
    """F1 holdout forest plot (H1-H5) from ``t1`` into ``out``, labels from ``cfg``; returns the figure record."""
    # Keep the non-transfer rows of the first holdout stage for the hypotheses H1 to H5.
    keep = t1[(t1["stage"] == "confirmatory holdout") & (t1["transfer_from"] == "") & t1["hypothesis"].isin(["H1", "H2", "H3", "H4", "H5"])].copy()
    # Add the stage label of each panel and the note of the row that was not pre-registered.
    keep["stage_label"] = keep["dataset"].map({"instacart": "holdout", "complete_journey": "refit on medium data"})
    keep["note"] = np.where((keep["hypothesis"] == "H5") & (keep["dataset"] == "complete_journey"), "H5 pre-registered for Instacart only", "")
    # Keep the plotted columns and the provenance columns.
    keep = keep[["id", "hypothesis", "dataset", "stage", "stage_label", "metric", "users_n", "diff", "ci95_lo", "ci95_hi", "ci99375_lo", "ci99375_hi",
                 "decision", "decision_99375", "note", "source_file", "source_sha256", "git_commit"]]
    # Write the CSV next to the PNG path.
    path = out / "F1_holdout_forest.png"
    keep.to_csv(path.with_suffix(".csv"), index=False, lineterminator="\n")
    # Take the largest user count of each dataset for the footer.
    n_i = int(keep.loc[keep["dataset"] == "instacart", "users_n"].max())
    n_c = int(keep.loc[keep["dataset"] == "complete_journey", "users_n"].max())
    # Draw the forest plot from the CSV.
    _forest(keep, path, "Confirmatory results (holdout): H1-H5, paired bootstrap CIs",
            f"Source: Instacart holdout (n = {n_i:,}) and Complete Journey refit on medium data (n = {n_c:,}), PREREG v1.2, code commit "
            f"{keep['git_commit'].iloc[0][:7]}. Thick bar 95% CI, thin bar 99.375% CI (Bonferroni, 8 tests), 1,000 resamples. "
            "Diamond: H5 on Complete Journey, not pre-registered (secondary, Instacart only).",
            COLOR_HOLDOUT, cfg["hypothesis_labels"], cfg["dataset_labels"])
    # Return the figure record.
    return {"name": "F1_holdout_forest", "points": int(len(keep)), "png": path, "csv": path.with_suffix(".csv")}


def figure_f7(t1: pd.DataFrame, out: Path, cfg: dict[str, Any]) -> dict[str, Any]:
    """F7 holdout2 forest plot from ``t1`` into ``out`` (labels from ``cfg``): PREREG v1.3 primary H6, H7' (NDCG, Recall), H8 and
    the H1-H5 replication; panels Instacart holdout2, Complete Journey refit and Complete Journey transfer; returns the record."""
    # List the row ids in plotting order.
    ids = ["H6_repeat_recall", "H7prime_ndcg", "H7prime_recall", "H8_new_recall",
           "H1_repeat_recall", "H2_new_recall", "H3_ndcg", "H3_recall", "H4_category_hit", "H5"]
    # Keep the rows of the holdout2 stage with those ids.
    keep = t1[(t1["stage"] == "confirmatory holdout2") & t1["id"].isin(ids)].copy()
    # Assign each row to the Instacart, refit or transfer panel.
    transfer = keep["transfer_from"].fillna("") != ""
    keep["panel"] = np.where(keep["dataset"] == "instacart", "instacart",
                             np.where(transfer, "complete_journey_transfer", "complete_journey_refit"))
    # Add the stage label of each panel and the note of the non-primary rows.
    keep["stage_label"] = keep["panel"].map({"instacart": "holdout2", "complete_journey_refit": "refit on medium data",
                                             "complete_journey_transfer": "Instacart holdout2 selections transferred"})
    keep["note"] = np.where(keep["tier"] == "primary", "", "secondary under PREREG v1.3")
    # Order the rows by id position and panel with a stable sort.
    keep["_i"] = keep["id"].map({id_: i for i, id_ in enumerate(ids)})
    keep = keep.sort_values(["_i", "panel"], kind="mergesort").drop(columns="_i")
    # Keep the plotted columns and the provenance columns.
    keep = keep[["id", "hypothesis", "prereg", "tier", "panel", "dataset", "transfer_from", "stage", "stage_label", "metric", "users_n", "diff",
                 "ci95_lo", "ci95_hi", "ci99375_lo", "ci99375_hi", "decision", "decision_99375", "note", "source_file", "source_sha256",
                 "git_commit"]]
    # Write the CSV next to the PNG path.
    path = out / "F7_holdout2_forest.png"
    keep.to_csv(path.with_suffix(".csv"), index=False, lineterminator="\n")
    # Take the largest user count of the Instacart panel and of the other panels for the footer.
    n_i = int(keep.loc[keep["panel"] == "instacart", "users_n"].max())
    n_c = int(keep.loc[keep["panel"] != "instacart", "users_n"].max())
    # Draw the forest plot from the CSV with one panel per panel value.
    _forest(keep, path, "Confirmatory results (holdout2, PREREG v1.3): primary H6, H7', H8 and the H1-H5 replication, paired bootstrap CIs",
            f"Source: Instacart holdout2 (n = {n_i:,}), Complete Journey refit on medium data and Complete Journey with the Instacart holdout2 "
            f"selections transferred (n = {n_c:,}), PREREG v1.3, code commit {keep['git_commit'].iloc[0][:7]}. Thick bar 95% CI, thin bar "
            "99.375% CI (the multiplicity-adjusted interval of the family of 3), 1,000 resamples. Circle: primary (Instacart holdout2: H6, "
            "H7' NDCG, H7' Recall reported, H8). Diamond: secondary under PREREG v1.3 (H1-H5 replication; every Complete Journey row).",
            COLOR_THIRD, cfg["hypothesis_labels"],
            {"instacart": cfg["dataset_labels"]["instacart"], "complete_journey_refit": cfg["dataset_labels"]["complete_journey"],
             "complete_journey_transfer": cfg["dataset_labels"]["complete_journey"]}, panel_column="panel")
    # Return the figure record.
    return {"name": "F7_holdout2_forest", "points": int(len(keep)), "png": path, "csv": path.with_suffix(".csv")}


def figure_f8(t1: pd.DataFrame, out: Path, cfg: dict[str, Any]) -> dict[str, Any] | None:
    """F8 holdout3 forest plot (F7 style) from ``t1`` into ``out``: PREREG v1.4 primary H11 (non-inferiority, margin -delta drawn on its
    row), H12, H13, the secondary P_T-vs-C3 metrics and the H6, H7', H8 replication; panels Instacart holdout3, Complete Journey refit
    and transfer. Thin bar = 98.33 % CI (policy rows) or 99.375 % CI (replication rows). None when the stage is absent."""
    # List the secondary policy ids and the row ids in plotting order.
    secondary_policy = [f"policy:P_T_vs_{POLICY_C3}_{m}" for m in ("recall", "new_recall", "repeat_recall")]
    ids = ["H11_ndcg", "H12_new_hit_share", "H13_new_hit_share", *secondary_policy,
           "H6_repeat_recall", "H7prime_ndcg", "H7prime_recall", "H8_new_recall"]
    # Keep the rows of the holdout3 stage with those ids and return None when there are none.
    keep = t1[(t1["stage"] == "confirmatory holdout3") & t1["id"].isin(ids)].copy()
    if keep.empty:
        return None
    # Assign each row to the Instacart, refit or transfer panel.
    transfer = keep["transfer_from"].fillna("") != ""
    keep["panel"] = np.where(keep["dataset"] == "instacart", "instacart",
                             np.where(transfer, "complete_journey_transfer", "complete_journey_refit"))
    # Add the stage label of each panel and the note of the non-primary rows.
    keep["stage_label"] = keep["panel"].map({"instacart": "holdout3", "complete_journey_refit": "refit on medium data",
                                             "complete_journey_transfer": "Instacart holdout3 selections transferred"})
    keep["note"] = np.where(keep["tier"] == "primary", "", "secondary under PREREG v1.4")
    # Use the 98.33 % interval and decision where present, else the 99.375 % ones.
    has_9833 = keep["ci9833_lo"].notna()
    keep["ci_adj_lo"] = np.where(has_9833, keep["ci9833_lo"], keep["ci99375_lo"])
    keep["ci_adj_hi"] = np.where(has_9833, keep["ci9833_hi"], keep["ci99375_hi"])
    keep["adj_level"] = np.where(has_9833, "0.9833", "0.99375")
    keep["decision_adj"] = np.where(has_9833, keep["decision_9833"], keep["decision_99375"])
    # Order the rows by id position and panel with a stable sort.
    keep["_i"] = keep["id"].map({id_: i for i, id_ in enumerate(ids)})
    keep = keep.sort_values(["_i", "panel"], kind="mergesort").drop(columns="_i")
    # Keep the plotted columns and the provenance columns.
    keep = keep[["id", "hypothesis", "prereg", "tier", "panel", "dataset", "transfer_from", "stage", "stage_label", "metric", "users_n", "diff",
                 "ci95_lo", "ci95_hi", "ci_adj_lo", "ci_adj_hi", "adj_level", "decision", "decision_adj", "note", "source_file",
                 "source_sha256", "git_commit"]]
    # Write the CSV next to the PNG path.
    path = out / "F8_holdout3_forest.png"
    keep.to_csv(path.with_suffix(".csv"), index=False, lineterminator="\n")
    # Take the largest user count of the Instacart panel and of the other panels, 0 without rows.
    n_i = int(keep.loc[keep["panel"] == "instacart", "users_n"].max())
    n_c = int(keep.loc[keep["panel"] != "instacart", "users_n"].max()) if (keep["panel"] != "instacart").any() else 0
    # Add the labels of the secondary policy rows.
    labels = {**cfg["hypothesis_labels"],
              f"policy:P_T_vs_{POLICY_C3}_recall": "P_T vs C3, Recall@10 (secondary)",
              f"policy:P_T_vs_{POLICY_C3}_new_recall": "P_T vs C3, new-item Recall@10 (secondary)",
              f"policy:P_T_vs_{POLICY_C3}_repeat_recall": "P_T vs C3, repeat Recall@10 (secondary)"}
    # Read the non-inferiority margin from the configuration.
    delta = float(cfg.get("policy_delta", 0.003))  # PREREG v1.4 non-inferiority margin of H11
    # Draw the forest plot with the adjusted interval as the thin bar and the margin on its row.
    _forest(keep, path, "Exploration policy (holdout3, PREREG v1.4): primary H11 (non-inferiority), H12, H13; secondary P_T-vs-C3 metrics "
            "and the H6, H7', H8 replication, paired bootstrap CIs",
            f"Source: Instacart holdout3 (n = {n_i:,}), Complete Journey refit on medium data and Complete Journey with the Instacart holdout3 "
            f"selections transferred (n = {n_c:,}), PREREG v1.4, code commit {keep['git_commit'].iloc[0][:7]}. Thick bar 95% CI; thin bar "
            "98.33% CI (= 1 - 0.05/3, the multiplicity-adjusted interval of the family of 3) for the policy rows and 99.375% CI for the "
            f"H6, H7', H8 replication rows; 1,000 resamples. Dashed tick on the H11 row: the non-inferiority margin -{delta:g} (H11 is "
            "non-inferior when the 98.33% lower bound lies right of it). Circle: primary (Instacart holdout3: H11, H12, H13). Diamond: "
            "secondary under PREREG v1.4 (P_T-vs-C3 Recall@10 / new-item Recall@10 / repeat Recall@10, the replication rows, every "
            "Complete Journey row).",
            COLOR_HOLDOUT, labels,
            {"instacart": cfg["dataset_labels"]["instacart"], "complete_journey_refit": cfg["dataset_labels"]["complete_journey"],
             "complete_journey_transfer": cfg["dataset_labels"]["complete_journey"]}, panel_column="panel",
            thin=("ci_adj_lo", "ci_adj_hi"), margins={"H11_ndcg": -delta})
    # Return the figure record.
    return {"name": "F8_holdout3_forest", "points": int(len(keep)), "png": path, "csv": path.with_suffix(".csv")}


def figure_f9(t1: pd.DataFrame, out: Path, cfg: dict[str, Any]) -> dict[str, Any] | None:
    """F9 holdout4 forest plot (F7 style) from ``t1`` into ``out``: PREREG v1.5 primary H14 (non-inferiority, margin -delta drawn on its
    row) and H15, the secondary R2'-vs-R2 / R2'-vs-R1 repeat Recall@10 and C4-vs-C3 Recall@10, and the H6, H7', H8 replication; panels
    Instacart holdout4, Complete Journey refit and transfer. Thin bar = 97.5 % CI (every row of this stage). None when the stage is absent."""
    # List the secondary v15 ids and the row ids in plotting order.
    secondary_v15 = [f"v15:{key}" for key in V15_SECONDARY]
    ids = ["H14_ndcg", "H15_ndcg", *secondary_v15, "H6_repeat_recall", "H7prime_ndcg", "H7prime_recall", "H8_new_recall"]
    # Keep the rows of the holdout4 stage with those ids and return None when there are none.
    keep = t1[(t1["stage"] == "confirmatory holdout4") & t1["id"].isin(ids)].copy()
    if keep.empty:
        return None
    # Assign each row to the Instacart, refit or transfer panel.
    transfer = keep["transfer_from"].fillna("") != ""
    keep["panel"] = np.where(keep["dataset"] == "instacart", "instacart",
                             np.where(transfer, "complete_journey_transfer", "complete_journey_refit"))
    # Add the stage label of each panel and the note of the non-primary rows.
    keep["stage_label"] = keep["panel"].map({"instacart": "holdout4", "complete_journey_refit": "refit on medium data",
                                             "complete_journey_transfer": "Instacart holdout4 selections transferred"})
    keep["note"] = np.where(keep["tier"] == "primary", "", "secondary under PREREG v1.5")
    # Order the rows by id position and panel with a stable sort.
    keep["_i"] = keep["id"].map({id_: i for i, id_ in enumerate(ids)})
    keep = keep.sort_values(["_i", "panel"], kind="mergesort").drop(columns="_i")
    # Keep the plotted columns and the provenance columns.
    keep = keep[["id", "hypothesis", "prereg", "tier", "panel", "dataset", "transfer_from", "stage", "stage_label", "metric", "users_n", "diff",
                 "ci95_lo", "ci95_hi", "ci975_lo", "ci975_hi", "decision", "decision_975", "note", "source_file", "source_sha256", "git_commit"]]
    # Write the CSV next to the PNG path.
    path = out / "F9_holdout4_forest.png"
    keep.to_csv(path.with_suffix(".csv"), index=False, lineterminator="\n")
    # Take the largest user count of the Instacart panel and of the other panels, 0 without rows.
    n_i = int(keep.loc[keep["panel"] == "instacart", "users_n"].max())
    n_c = int(keep.loc[keep["panel"] != "instacart", "users_n"].max()) if (keep["panel"] != "instacart").any() else 0
    # Add the labels of the secondary v15 rows.
    labels = {**cfg["hypothesis_labels"],
              "v15:repeat_lri_vs_repeat_lr_repeat_recall": "R2' vs R2, repeat Recall@10 (secondary)",
              "v15:repeat_lri_vs_repeat_star_repeat_recall": "R2' vs R1, repeat Recall@10 (secondary)",
              "v15:final_lri_ucf_vs_final_lr_ucf_recall": "C4 vs C3, Recall@10 (secondary)"}
    # Read the non-inferiority margin from the configuration.
    delta = float(cfg.get("v15_delta", 0.003))  # PREREG v1.5 non-inferiority margin of H14 (the v1.4 margin)
    # Draw the forest plot with the 97.5 % interval as the thin bar and the margin on its row.
    _forest(keep, path, "Two-CF final system (holdout4, PREREG v1.5): primary H14 (non-inferiority), H15; secondary R2' and C4 comparisons "
            "and the H6, H7', H8 replication, paired bootstrap CIs",
            f"Source: Instacart holdout4 (n = {n_i:,}), Complete Journey refit on medium data and Complete Journey with the Instacart holdout4 "
            f"selections transferred (n = {n_c:,}), PREREG v1.5, code commit {keep['git_commit'].iloc[0][:7]}. Thick bar 95% CI; thin bar "
            "97.5% CI (= 1 - 0.05/2, the multiplicity-adjusted interval of the family of 2; the profile's bootstrap levels, so every row of "
            f"this stage carries it); 1,000 resamples. Dashed tick on the H14 row: the non-inferiority margin -{delta:g} (H14 is non-inferior "
            "when the 97.5% lower bound lies right of it). Circle: primary (Instacart holdout4: H14, H15). Diamond: secondary under PREREG v1.5 "
            "(R2' vs R2 and R2' vs R1 repeat Recall@10, C4 vs C3 Recall@10, the replication rows, every Complete Journey row).",
            COLOR_THIRD, labels,
            {"instacart": cfg["dataset_labels"]["instacart"], "complete_journey_refit": cfg["dataset_labels"]["complete_journey"],
             "complete_journey_transfer": cfg["dataset_labels"]["complete_journey"]}, panel_column="panel",
            thin=("ci975_lo", "ci975_hi"), margins={"H14_ndcg": -delta})
    # Return the figure record.
    return {"name": "F9_holdout4_forest", "points": int(len(keep)), "png": path, "csv": path.with_suffix(".csv")}


def figure_f2(t1: pd.DataFrame, out: Path, cfg: dict[str, Any], dev_run: str | None) -> dict[str, Any]:
    """F2 development forest plot (H6-H10) from the ``dev_run`` rows of ``t1`` into ``out``; returns the figure record."""
    # Keep the development rows of H6 to H10 from the chosen run and the run "W9".
    keep = t1[(t1["stage"] == "development medium") & t1["hypothesis"].isin(["H6", "H7", "H8", "H9", "H10"])
              & t1["run"].isin([dev_run, "W9"])].copy()
    # Drop the rows of the run "W9" that do not belong to H10.
    keep = keep[~((keep["run"] == "W9") & (keep["hypothesis"] != "H10"))]
    # Set one stage label and an empty note for every row.
    keep["stage_label"] = "development (medium)"
    keep["note"] = ""
    # Keep the plotted columns and the provenance columns.
    keep = keep[["id", "hypothesis", "run", "dataset", "stage", "stage_label", "metric", "users_n", "diff", "ci95_lo", "ci95_hi", "ci99375_lo",
                 "ci99375_hi", "decision", "decision_99375", "note", "source_file", "source_sha256", "git_commit"]]
    # Write the CSV next to the PNG path.
    path = out / "F2_dev_forest.png"
    keep.to_csv(path.with_suffix(".csv"), index=False, lineterminator="\n")
    # List each run with its short commit for the footer.
    runs = ", ".join(f"{r} (commit {c[:7]})" for r, c in keep.groupby("run")["git_commit"].first().items())
    # Draw the forest plot from the CSV.
    _forest(keep, path, "Development results: H6-H10, paired bootstrap CIs - development (medium)",
            f"Source: medium eval split, Instacart (n = 20,000 users) and Complete Journey (n = 2,069); runs {runs}. "
            "Thick bar 95% CI, thin bar 99.375% CI, 1,000 resamples. Not confirmatory.", COLOR_DEV, cfg["hypothesis_labels"], cfg["dataset_labels"])
    # Return the figure record.
    return {"name": "F2_dev_forest", "points": int(len(keep)), "png": path, "csv": path.with_suffix(".csv")}


def figure_f3(t3: pd.DataFrame, out: Path, cfg: dict[str, Any]) -> dict[str, Any]:
    """F3 module screening means from ``t3`` into ``out`` (labels from ``cfg``); returns the figure record."""
    # Load pyplot, write the CSV and read it back.
    plt = _plt()
    path = out / "F3_module_screening.png"
    t3.to_csv(path.with_suffix(".csv"), index=False, lineterminator="\n")
    frame = pd.read_csv(path.with_suffix(".csv"))
    # Keep the datasets present in the data, in the configured order.
    datasets = [d for d in cfg["dataset_labels"] if d in set(frame["dataset"])]
    # Create one row of panels per module group and one column per dataset.
    fig, axes = plt.subplots(2, len(datasets), figsize=(4.2 * len(datasets), 5.6), squeeze=False)
    # Give each plotted stage a colour and a hatch.
    stage_color = {"confirmatory holdout2": COLOR_THIRD, "confirmatory holdout": COLOR_HOLDOUT, "development medium": COLOR_DEV}
    stage_hatch = {"confirmatory holdout2": "", "confirmatory holdout": "", "development medium": "////"}
    for row_i, (group, metric) in enumerate((("explore", "new-item Recall@10"), ("repeat", "repeat Recall@10"))):
        for col_i, dataset in enumerate(datasets):
            # Select the rows of the group and the dataset.
            ax = axes[row_i][col_i]
            part = frame[(frame["group"] == group) & (frame["dataset"] == dataset)]
            # Keep the plotted stages that occur in the panel.
            stages = [s for s in stage_color if s in set(part["stage"])]
            part = part[part["stage"].isin(stages)]  # only the plotted stages define the x axis (no empty ticks for holdout3/holdout4-only modules)
            # Order the modules of the panel as in the module lists.
            catalog = [m[0] for m in EXPLORE_MODULES + REPEAT_MODULES]
            modules = [m for m in catalog if m in set(part["module"])]
            # Draw one bar per module and stage, with the stages side by side.
            width = 0.8 / max(len(stages), 1)
            for s_i, stage in enumerate(stages):
                sub = part[part["stage"] == stage].set_index("module")
                xs = [i + (s_i - (len(stages) - 1) / 2) * width for i in range(len(modules))]
                ys = [sub["mean"].get(m, np.nan) for m in modules]
                ax.bar(xs, ys, width=width * 0.92, color=stage_color[stage], hatch=stage_hatch[stage], edgecolor="white", linewidth=0.5,
                       label=stage, zorder=3)
            # Label the axes and the panel.
            ax.set_xticks(range(len(modules)))
            ax.set_xticklabels(modules)
            ax.set_ylabel(metric)
            ax.set_title(f"{cfg['dataset_labels'][dataset]} - {group} modules", loc="left")
            # Draw the y-axis grid behind the bars and hide the top and right spines.
            ax.grid(True, axis="y", color=COLOR_GRID, linewidth=0.6)
            ax.set_axisbelow(True)
            for side in ("top", "right"):
                ax.spines[side].set_visible(False)
            # Show the legend in the first panel only.
            if row_i == 0 and col_i == 0:
                ax.legend(frameon=False)
    # Add the figure title and save the figure with the footer.
    fig.suptitle("Module screening: mean of the per-user metric by module (holdout2, holdout vs development medium)", x=0.01, ha="left",
                 fontsize=10)
    _finish(fig, path, plt, "Explore: new-item Recall@10 over users with >= 1 new target item. Repeat: repeat Recall@10 over users with >= 1 repeat "
            "target item. Solid = confirmatory (holdout2 = PREREG v1.3, holdout = PREREG v1.2; Complete Journey refit on medium data), hatched = "
            "development (medium). "
            "Error bars are not shown; 95% CIs refer to differences from the comparator (see T3).")
    # Return the figure record.
    return {"name": "F3_module_screening", "points": int(len(frame)), "png": path, "csv": path.with_suffix(".csv")}


def figure_f4(registry: Registry, out: Path, cfg: dict[str, Any], dev_run: str | None) -> dict[str, Any] | None:
    """F4 NDCG@10 by r for C2, C3, C3b from the ``dev_run`` tuning sources of ``registry`` into ``out``; None when absent."""
    rows = []
    # Map each list name to its combination label.
    by_list = {spec["list"]: label for label, spec in cfg["combinations"].items() if spec["list"]}
    # Collect NDCG@10 per r from the integration block of the chosen run's tuning files.
    for source in registry.sources.values():
        if source.run != dev_run or source.doc.get("split") != "fit" or not source.doc.get("integration"):
            continue
        integ = source.doc["integration"]
        for list_name, scores in integ["r_scores"].items():
            label = by_list.get(list_name, list_name)
            for entry in scores:
                rows.append({"stage": source.stage, "run": source.run, "dataset": source.doc["dataset"], "combination": label, "list": list_name,
                             "r": int(entry["r"]), "ndcg10": float(entry["ndcg"]), "selection_users": integ["r_selection_users"][list_name]["users"],
                             "selection_users_n": int(integ["r_selection_users"][list_name]["n_users"]),
                             "selected_r": int(integ["selections"]["r"][list_name]), **prov(source)})
    # Return None when no such file exists.
    if not rows:
        return None
    # Load pyplot, write the CSV and read it back.
    plt = _plt()
    path = out / "F4_r_grid.png"
    pd.DataFrame(rows).to_csv(path.with_suffix(".csv"), index=False, lineterminator="\n")
    frame = pd.read_csv(path.with_suffix(".csv"))
    # Keep the datasets present in the data, in the configured order.
    datasets = [d for d in cfg["dataset_labels"] if d in set(frame["dataset"])]
    # Give each combination a colour and a marker.
    colors = {"C2": COLOR_HOLDOUT, "C3": COLOR_DEV, "C3b": COLOR_THIRD}
    markers = {"C2": "o", "C3": "s", "C3b": "^"}
    # Create one panel per dataset.
    fig, axes = plt.subplots(1, len(datasets), figsize=(4.2 * len(datasets), 3.6), squeeze=False)
    for ax, dataset in zip(axes[0], datasets):
        # Select the rows of the dataset.
        part = frame[frame["dataset"] == dataset]
        # Draw one line per combination that has rows.
        for combo in ("C2", "C3", "C3b"):
            line = part[part["combination"] == combo].sort_values("r")
            if line.empty:
                continue
            ax.plot(line["r"], line["ndcg10"], color=colors[combo], marker=markers[combo], markersize=4, linewidth=1.6,
                    label=f"{combo} ({line['list'].iloc[0]}, n = {int(line['selection_users_n'].iloc[0]):,})")
        # Label the axes and the panel.
        ax.set_xticks(range(0, 11))
        ax.set_xlabel("r = repeat slots in the top-10 list")
        ax.set_ylabel("NDCG@10 on the fit-split selection users")
        ax.set_title(f"{cfg['dataset_labels'][dataset]} - development (medium), fit split", loc="left")
        # Draw the grid behind the lines, hide the top and right spines and add the legend.
        ax.grid(True, color=COLOR_GRID, linewidth=0.6)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.legend(frameon=False, loc="upper left")
    # Add the figure title and save the figure with the footer.
    fig.suptitle("Slot re-selection: NDCG@10 by r for combinations C2, C3, C3b - development (medium)", x=0.01, ha="left", fontsize=10)
    _finish(fig, path, plt, f"Source: {dev_run} tuning JSON (commit {frame['git_commit'].iloc[0][:7]}), integration.r_scores. "
            "C2 (final_lr_r, E1 + R2) and C3 (final_lr_ucf, E2 + R2) are scored on the LR validation users with a train-only model; "
            "C3b (final_star_ucf, E2 + R1) on all fit users. Selected r = 10 for every combination (grid edge).")
    # Return the figure record.
    return {"name": "F4_r_grid", "points": int(len(frame)), "png": path, "csv": path.with_suffix(".csv")}


def figure_f5(t5: pd.DataFrame, out: Path, cfg: dict[str, Any], dev_run: str | None) -> dict[str, Any]:
    """F5 per-bin differences (H3, H7, H8) from ``t5`` into ``out`` (``dev_run`` rows for H7/H8); returns the figure record."""
    # Keep the rows of the chosen development run for the three plotted keys with users.
    keep = t5[(t5["stage"] == "development medium") & (t5["run"] == dev_run) & t5["id"].isin(["H3_ndcg", "H7_ndcg", "H8_new_recall"])
              & t5["users_n"].gt(0)].copy()
    # Load pyplot, write the CSV and read it back.
    plt = _plt()
    path = out / "F5_history_length.png"
    keep.to_csv(path.with_suffix(".csv"), index=False, lineterminator="\n")
    frame = pd.read_csv(path.with_suffix(".csv"))
    # Keep the datasets present in the data, in the configured order.
    datasets = [d for d in cfg["dataset_labels"] if d in set(frame["dataset"])]
    # Define the key, label, colour and marker of each series.
    series = [("H3_ndcg", "H3 NDCG@10 (C0 final vs personal frequency)", COLOR_HOLDOUT, "o"),
              ("H7_ndcg", "H7 NDCG@10 (C1 final_lr vs C0 final)", COLOR_DEV, "s"),
              ("H8_new_recall", "H8 new-item Recall@10 (E2 usercf vs flat popularity)", COLOR_THIRD, "^")]
    # Create one panel per dataset.
    fig, axes = plt.subplots(1, len(datasets), figsize=(4.4 * len(datasets), 3.8), squeeze=False)
    for ax, dataset in zip(axes[0], datasets):
        # Select the rows of the dataset and list its bins in order of first appearance.
        part = frame[frame["dataset"] == dataset]
        bins = list(dict.fromkeys(part["bin"]))
        # Draw each series with its 95 % interval, shifted sideways per series.
        for s_i, (key, label, color, marker) in enumerate(series):
            sub = part[part["id"] == key].set_index("bin")
            xs = [i + (s_i - 1) * 0.22 for i in range(len(bins))]
            ys = [sub["diff"].get(b, np.nan) for b in bins]
            lo = [sub["ci95_lo"].get(b, np.nan) for b in bins]
            hi = [sub["ci95_hi"].get(b, np.nan) for b in bins]
            ax.errorbar(xs, ys, yerr=[np.array(ys) - np.array(lo), np.array(hi) - np.array(ys)], fmt=marker, color=color, markersize=4.5,
                        capsize=2, linewidth=1.2, label=label, zorder=3)
        # Draw the zero line.
        ax.axhline(0.0, color=COLOR_INK, linewidth=0.8, zorder=1)
        # Label the axes and the panel.
        ax.set_xticks(range(len(bins)))
        ax.set_xticklabels(bins)
        ax.set_xlabel("history length (distinct history baskets)")
        ax.set_ylabel("difference of per-user means (model - comparator)")
        ax.set_title(f"{cfg['dataset_labels'][dataset]} - development (medium)", loc="left")
        # Draw the y-axis grid behind the marks and hide the top and right spines.
        ax.grid(True, axis="y", color=COLOR_GRID, linewidth=0.6)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    # Add the legend to the first panel and the figure title, then save the figure with the footer.
    axes[0][0].legend(frameon=False, loc="upper right")
    fig.suptitle("Differences by history length with 95% CIs - development (medium)", x=0.01, ha="left", fontsize=10)
    _finish(fig, path, plt, f"Source: {dev_run} eval JSON (commit {frame['git_commit'].iloc[0][:7]}), by_history_length blocks; paired bootstrap "
            "1,000 resamples, 95% percentile CI per bin; descriptive, no decisions (PREREG v1.2 deviation 2).")
    # Return the figure record.
    return {"name": "F5_history_length", "points": int(len(frame)), "png": path, "csv": path.with_suffix(".csv")}


def figure_f6(registry: Registry, out: Path, cfg: dict[str, Any]) -> dict[str, Any]:
    """F6 W9 overdue-bin log-odds effects from the R3 CSV sources of ``registry`` into ``out``; returns the figure record."""
    rows = []
    for dataset in cfg["dataset_labels"]:
        # Take the overdue-coefficient CSV and the model JSON of the dataset.
        source = registry.get(f"w9/{dataset}_overdue_csv")
        json_source = registry.get(f"w9/{dataset}_r3")
        # Skip a dataset without the CSV.
        if source is None:
            continue
        # Copy the table and put the stage, run and dataset columns first.
        frame = source.doc.copy()
        frame.insert(0, "dataset", dataset)
        frame.insert(0, "run", source.run)
        frame.insert(0, "stage", source.stage)
        # Add the provenance columns, then the commit and the model digest of the JSON.
        for key, value in prov(source).items():
            frame[key] = value
        frame["git_commit"] = json_source.git_commit if json_source else ""
        frame["model_sha256"] = json_source.doc["artifact"]["sha256"] if json_source else ""
        # Collect the table of the dataset.
        rows.append(frame)
    # Load pyplot, write the stacked CSV and read it back.
    plt = _plt()
    path = out / "F6_overdue_effect.png"
    pd.concat(rows, ignore_index=True).to_csv(path.with_suffix(".csv"), index=False, lineterminator="\n")
    frame = pd.read_csv(path.with_suffix(".csv"))
    # Keep the datasets present in the data, in the configured order.
    datasets = [d for d in cfg["dataset_labels"] if d in set(frame["dataset"])]
    # Create one panel per dataset with a shared y axis.
    fig, axes = plt.subplots(1, len(datasets), figsize=(4.4 * len(datasets), 3.6), squeeze=False, sharey=True)
    for ax, dataset in zip(axes[0], datasets):
        # Select the rows of the dataset.
        part = frame[frame["dataset"] == dataset].reset_index(drop=True)
        # Draw the bars, grey for the once bin, and the zero line.
        colors = [COLOR_MUTED if b == "once" else COLOR_DEV for b in part["bin"]]
        ax.bar(range(len(part)), part["effect_log_odds"], width=0.62, color=colors, zorder=3)
        ax.axhline(0.0, color=COLOR_INK, linewidth=0.8, zorder=2)
        # Mark the reference bin with a text label.
        for i, column in enumerate(part["column"]):
            if column == "(reference)":
                ax.text(i, 0.0, "ref.", ha="center", va="bottom", fontsize=7, color=COLOR_MUTED)
        # Label the axes and the panel.
        ax.set_xticks(range(len(part)))
        ax.set_xticklabels(part["bin"], fontsize=7)
        ax.set_xlabel("overdue bin (right-open); grey = once-bought items")
        ax.set_ylabel("log-odds effect vs. reference bin [0, 0.5)")
        ax.set_title(f"{cfg['dataset_labels'][dataset]} - development (medium)", loc="left")
        # Draw the y-axis grid behind the bars and hide the top and right spines.
        ax.grid(True, axis="y", color=COLOR_GRID, linewidth=0.6)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    # Add the figure title and save the figure with the footer.
    fig.suptitle("R3 logistic ranker: log-odds effect per overdue bin (standardized coefficient / dummy std) - development (medium)",
                 x=0.01, ha="left", fontsize=10)
    _finish(fig, path, plt, f"Source: W9 reports/tables/medium_<dataset>_r3_overdue_coef.csv (commit {frame['git_commit'].iloc[0][:7]}); overdue = "
            "recency / mean_gap; fit split of medium, L2 logistic regression refit on all training rows with C*. Not confirmatory.")
    # Return the figure record.
    return {"name": "F6_overdue_effect", "points": int(len(frame)), "png": path, "csv": path.with_suffix(".csv")}


# ---------------------------------------------------------------------------
# RESULTS.md
# ---------------------------------------------------------------------------
TABLE_NOTES = {
    "T1_master_hypotheses": "one row per hypothesis or secondary/auxiliary comparison; diff = mean_model - mean_comparator; decision as recorded in the source; prereg / tier = pre-registered classification (results_sources.yaml); current = newest run for that comparison (use only these rows)",
    "T2_selections": "selected values per dataset x run (long format) with grid and grid-edge flag",
    "T3_module_screening": "explore modules by new-item Recall@10, repeat modules by repeat Recall@10; diff and 95% CI vs the comparator named in the row",
    "T4_final_lists": "Recall@10 and NDCG@10 over all users for the combinations C0-C3, C3b, C4 (holdout4, PREREG v1.5), the policy lists P_T (holdout3 and holdout4, base C3) and P_T(C4) (holdout4, PREREG v1.5.1), the reference E4 + R1 and the baselines (holdout4, holdout3, holdout2 and medium); C0, E4 + R1 and baselines on the holdout (v1.2)",
    "T5_history_length": "per history-length bin: diff and 95% CI of H1, H3, H6, H7, H8, H9, H10, (holdout3, PREREG v1.4) H11, H12, H13 and (holdout4, PREREG v1.5) H14",
    "T6_module_catalog": "module and combination status catalogue (adopted / evaluated-not-adopted / not-examined / not run) with basis hypotheses",
    "T7_datasets_eda": "dataset summary (eval history) copied from the EDA branch feat/w5a-eda",
}
FIGURE_NOTES = {
    "F1_holdout_forest": "holdout confirmatory H1-H5 forest plot, Instacart and Complete Journey refit side by side (95% thick, 99.375% thin)",
    "F2_dev_forest": "development (medium) H6-H10 forest plot, same format",
    "F3_module_screening": "module means: explore (new-item Recall@10) and repeat (repeat Recall@10), holdout2 and holdout vs medium",
    "F4_r_grid": "NDCG@10 by r (0-10) for C2, C3, C3b on the fit split, one panel per dataset",
    "F5_history_length": "per-bin differences with 95% CIs for H3, H7, H8, one panel per dataset",
    "F6_overdue_effect": "W9 R3 log-odds effect per overdue bin (from the W9 CSV), one panel per dataset",
    "F7_holdout2_forest": "holdout2 confirmatory forest plot (PREREG v1.3): primary H6, H7' (NDCG, Recall), H8 and the H1-H5 replication; Instacart holdout2, Complete Journey refit and transfer side by side (95% thick, 99.375% thin)",
    "F8_holdout3_forest": "holdout3 confirmatory forest plot (PREREG v1.4): primary H11 (non-inferiority; dashed tick = margin -0.003), H12, H13, the secondary P_T-vs-C3 metrics and the H6, H7', H8 replication; Instacart holdout3, Complete Journey refit and transfer side by side (95% thick; 98.33% thin for policy rows, 99.375% thin for replication rows)",
    "F9_holdout4_forest": "holdout4 confirmatory forest plot (PREREG v1.5): primary H14 (non-inferiority; dashed tick = margin -0.003), H15, the secondary R2'-vs-R2 / R2'-vs-R1 repeat Recall@10 and C4-vs-C3 Recall@10 and the H6, H7', H8 replication; Instacart holdout4, Complete Journey refit and transfer side by side (95% thick, 97.5% thin)",
}


def write_results_md(out: Path, registry: Registry, tables: list[dict[str, Any]], figures: list[dict[str, Any]], checks: dict[str, Any],
                     w8b_used: bool, dev_run: str | None, cfg: dict[str, Any]) -> Path:
    """Write <out>/RESULTS.md: header flags (``w8b_used``, ``dev_run``, combination labels of ``cfg``), the ``tables`` and
    ``figures`` records, the inputs of ``registry`` and the ``checks``.
    """
    # Start with the title line.
    lines = ["# Consolidated results (W11; updated in W11b with the PREREG v1.3 run on holdout2, in W17 with the PREREG v1.4 "
             "exploration-policy test on holdout3 and in W19 with the PREREG v1.5 two-CF final run on holdout4)", ""]
    # State whether the W8b outputs are reflected.
    lines.append(f"- W8b outputs reflected: **{'yes' if w8b_used else 'no'}**"
                 + ("" if w8b_used else " (W8b not complete at generation time; development figures use run "
                    f"{dev_run}; re-run `python scripts/make_results.py` after W8b finishes)"))
    # State whether the holdout4 results are included.
    holdout4_used = any(source.run == "holdout4" for source in registry.sources.values())
    lines.append("- holdout4 results: **included** (stage `confirmatory holdout4`, PREREG v1.5 two-CF final run, reports/final4; "
                 "primary H14 (non-inferiority, delta = 0.003) and H15 on Instacart holdout4 with 97.5 % intervals in the T1 columns "
                 "`ci975_lo` / `ci975_hi` / `decision_975` (the profile's bootstrap levels, so every row of that stage carries them); "
                 "everything else of that stage is secondary)"
                 if holdout4_used else "- holdout4 results: **not included** (reports/final4 not registered)")
    # State whether the holdout3 results are included.
    holdout3_used = any(source.run == "holdout3" for source in registry.sources.values())
    lines.append("- holdout3 results: **included** (stage `confirmatory holdout3`, PREREG v1.4 exploration-policy test, reports/final3; "
                 "primary H11 (non-inferiority, delta = 0.003), H12, H13 on Instacart holdout3 with 98.33 % intervals in the T1 columns "
                 "`ci9833_lo` / `ci9833_hi` / `decision_9833`; everything else of that stage is secondary)"
                 if holdout3_used else "- holdout3 results: **not included** (reports/final3 not registered)")
    # State whether the holdout2 results are included.
    holdout2_used = any(source.run == "holdout2" for source in registry.sources.values())
    lines.append("- holdout2 results: **included** (stage `confirmatory holdout2`, PREREG v1.3 final run, reports/final2; T1 columns "
                 "`prereg` and `tier` carry the pre-registered classification, see `classification` in results_sources.yaml)"
                 if holdout2_used else "- holdout2 results: **not included** (reports/final2 not registered)")
    # Explain the stage names.
    lines.append("- Stage wording: `confirmatory holdout4` = PREREG v1.5 two-CF final run (Instacart holdout4; Complete Journey refit on "
                 "medium data, not independent of development); "
                 "`confirmatory holdout3` = PREREG v1.4 exploration-policy test (Instacart holdout3; Complete Journey refit "
                 "on medium data with the same policy, not independent of development); `confirmatory holdout2` = PREREG v1.3 final run "
                 "(Instacart holdout2; Complete Journey refit on medium data, not independent of development); `confirmatory holdout` = "
                 "PREREG v1.2 final run (Instacart holdout; Complete Journey refit on medium data); `development medium` = development runs, "
                 "not confirmatory.")
    # Explain the tier column.
    lines.append("- Tier (T1 `tier`): primary = pre-registered primary test of that stage (v1.5: H14, H15 on Instacart holdout4; "
                 "v1.4: H11, H12, H13 on Instacart holdout3; "
                 "v1.3: H6, H7' NDCG, H7' Recall, H8 on Instacart "
                 "holdout2; v1.2: H1-H4 on Instacart holdout and the Complete Journey refit); secondary-confirmatory = H5 (v1.2); secondary = every "
                 "other confirmatory-stage row (H9, H7, the H1-H5 replication, the H6/H7'/H8 replication, the policy variants on holdout3, "
                 "the R2'/C4 secondaries and the policy replication (base C3 and base C4, ids policy_c4:*) on holdout4, transfer, "
                 "CI-only comparisons); development = development rows.")
    # Name the development integration run.
    lines.append(f"- Development integration run used for T3-T5, F2-F5: `{dev_run}`; W9 provides R3/H10 in every case.")
    # Add the change note of the tables.
    lines.append("- Change note (W19, PREREG v1.5 / v1.5.1): structural changes to the existing tables are limited to T1 (new columns "
                 "`ci975_lo`, `ci975_hi`, `decision_975`, empty outside the holdout4 stage), T4 (the two empty `C4` placeholder rows "
                 "\"not run: E3 + R2 not implemented\" of the development stage are no longer produced because C4 now denotes the evaluated "
                 "list `final_lri_ucf`; the new rows C4, P_T and P_T(C4) of the holdout4 stage are added) and T6 (new rows R2' and P_T(C4); "
                 "`evidence_stage` of the existing modules gains `confirmatory holdout4` where their hypotheses were replicated; the notes of "
                 "E3, R2 and C4 updated). C4 (PREREG v1.5) = R2' + E2; the earlier C4 placeholder (E3 + R2) was never evaluated.")
    # List the combination labels.
    lines.append("- Combination labels (fixed by the team): C0 = E1 + R1, r = 10 (`final`); C1 = E1 + R2, r = 10 (`final_lr`); "
                 "C2 = E1 + R2, r re-selected (`final_lr_r`); C3 = E2 + R2, r re-selected (`final_lr_ucf`; H9 = C3 vs C1); "
                 "C3b = E2 + R1, r re-selected (`final_star_ucf`); C4 = E2 + R2' (R2 with the item-CF similarity feature), r re-selected "
                 "(`final_lri_ucf`, holdout4, PREREG v1.5); reference E4 + R1, r = 10 (`final_cf`); "
                 "P_T = C3 with the PREREG v1.4 slot policy, r_u = 9 for users with at most 4 history baskets (`P_T`, holdout3; repeated on "
                 "holdout4 as a secondary reference); P_T(C4) = C4 with the same policy (`P_T` of the policy_c4 block, holdout4, PREREG v1.5.1). "
                 "Module codes and statuses are listed in T6 (`scripts/results_sources.yaml`).")
    # State the rule for current rows.
    lines.append("- **For the slides and the report use only T1 rows with `current = true`.** A row is current when it comes from the "
                 "newest run for the same stage, dataset, comparator, model and metric (holdout4, holdout3, holdout2 and holdout are separate confirmatory stages; "
                 "development: W8b > W9 > W8 > W7, W6 > W4a); "
                 "`superseded_reason` explains every other row.")
    # Add the table index.
    lines += ["", "## Tables (reports/results/tables)", "", "| table | rows | description |", "| --- | --- | --- |"]
    for t in tables:
        lines.append(f"| {t['name']} | {t['rows']} | {TABLE_NOTES[t['name']]} |")
    # Add the figure index.
    lines += ["", "## Figures (reports/results/figures, PNG 300 dpi + CSV of the plotted numbers)", "", "| figure | points | description |", "| --- | --- | --- |"]
    for f in figures:
        lines.append(f"| {f['name']} | {f['points']} | {FIGURE_NOTES[f['name']]} |")
    # Add the input list with commit and digest.
    lines += ["", "## Inputs (read-only)", "", "| key | run | stage | path | git_commit | sha256 |", "| --- | --- | --- | --- | --- | --- |"]
    for source in registry.sources.values():
        lines.append(f"| {source.key} | {source.run} | {source.stage} | {source.path.as_posix()} | {source.git_commit} | {source.sha256} |")
    # Add the skipped inputs when there are any.
    if registry.skipped:
        lines += ["", "### Skipped inputs", ""]
        for item in registry.skipped:
            lines.append(f"- {item['run']}: {item['reason']}")
    # Add the verification summary.
    lines += ["", "## Verification", ""]
    lines.append(f"- T1 diff identity: {checks['t1']['rows']} rows, max |diff - (mean_model - mean_comparator)| = {checks['t1']['max_abs_gap']:.3e} "
                 f"(tolerance {checks['t1']['tolerance']:.0e}); current rows: {checks['t1']['current_rows']}.")
    lines.append("- Every figure is drawn from its own CSV (written first, then read back).")
    # Add the determinism result when it was computed.
    if "determinism" in checks:
        d = checks["determinism"]
        lines.append(f"- Determinism (`--verify`): {d['files']} files regenerated, {d['mismatches']} digest mismatches.")
    # Add the generator line.
    lines += ["", "## Generator", "", "`python scripts/make_results.py` (scripts/results_sources.yaml lists every input). Values are never typed in by hand.", ""]
    # Write the file with LF line ends and return its path.
    path = out / "RESULTS.md"
    path.write_text("\n".join(lines), encoding="utf-8", newline="\n")
    return path


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def generate(cfg: dict[str, Any], out: Path) -> dict[str, Any]:
    """Build the registry from ``cfg``, write every table and figure under ``out`` and return the records and checks."""
    # Load the registered inputs and list their runs.
    registry = build_registry(cfg)
    runs = registry.runs()
    # Choose the first preferred development integration run that is present.
    dev_run = next((r for r in cfg["development_integration_run_preference"] if r in runs), None)
    # Record whether the run "W8b" is present.
    w8b_used = "W8b" in runs

    # Build the seven tables and check the T1 diff identity.
    t1 = table_t1(registry, cfg)
    checks = {"t1": check_t1(t1)}
    t2 = table_t2(registry)
    t3 = table_t3(registry, dev_run)
    t4 = table_t4(registry, cfg, dev_run)
    t5 = table_t5(registry, dev_run)
    t6 = table_t6(cfg, t1)
    t7 = table_t7(registry)
    # Empty and re-create the tables and figures folders.
    tables_dir, figures_dir = out / "tables", out / "figures"
    for child in (tables_dir, figures_dir):
        if child.exists():
            shutil.rmtree(child)
        child.mkdir(parents=True, exist_ok=True)
    # Write every table as CSV and Markdown.
    tables = [
        write_table(t1, tables_dir, "T1_master_hypotheses", "T1 master hypotheses table", TABLE_NOTES["T1_master_hypotheses"]),
        write_table(t2, tables_dir, "T2_selections", "T2 selected values", TABLE_NOTES["T2_selections"]),
        write_table(t3, tables_dir, "T3_module_screening", "T3 module screening", TABLE_NOTES["T3_module_screening"]),
        write_table(t4, tables_dir, "T4_final_lists", "T4 final lists", TABLE_NOTES["T4_final_lists"]),
        write_table(t5, tables_dir, "T5_history_length", "T5 history-length breakdown", TABLE_NOTES["T5_history_length"]),
        write_table(t6, tables_dir, "T6_module_catalog", "T6 module catalogue", TABLE_NOTES["T6_module_catalog"]),
        write_table(t7, tables_dir, "T7_datasets_eda", "T7 datasets (EDA, feat/w5a-eda)", TABLE_NOTES["T7_datasets_eda"]),
    ]
    # Draw the figures, leaving out those that return None.
    figures = [figure_f1(t1, figures_dir, cfg), figure_f7(t1, figures_dir, cfg)]
    f8 = figure_f8(t1, figures_dir, cfg)
    if f8:
        figures.append(f8)
    f9 = figure_f9(t1, figures_dir, cfg)
    if f9:
        figures.append(f9)
    figures += [figure_f2(t1, figures_dir, cfg, dev_run), figure_f3(t3, figures_dir, cfg)]
    f4 = figure_f4(registry, figures_dir, cfg, dev_run)
    if f4:
        figures.append(f4)
    figures += [figure_f5(t5, figures_dir, cfg, dev_run), figure_f6(registry, figures_dir, cfg)]
    # Return the registry, the records, the checks and the run flags.
    return {"registry": registry, "tables": tables, "figures": figures, "checks": checks, "w8b_used": w8b_used, "dev_run": dev_run}


def digests(folder: Path) -> dict[str, str]:
    """{relative path: sha256} of every file under ``folder``."""
    return {str(p.relative_to(folder)).replace("\\", "/"): sha256_file(p) for p in sorted(folder.rglob("*")) if p.is_file()}


def main(argv: list[str] | None = None) -> None:
    """Parse ``argv`` (default: sys.argv), generate the results and, with --verify, regenerate into a temporary folder and
    compare the digests.
    """
    # Define and parse the command-line arguments.
    parser = argparse.ArgumentParser(description="W11 consolidated result tables and figures")
    parser.add_argument("--sources", default=str(SCRIPT_DIR / "results_sources.yaml"))
    parser.add_argument("--out", default=str(REPO_ROOT / "reports" / "results"))
    parser.add_argument("--verify", action="store_true", help="regenerate into a temporary folder and compare digests (determinism)")
    args = parser.parse_args(argv)
    # Load the source registry configuration.
    with open(args.sources, encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)
    # Generate the results into the output folder.
    out = Path(args.out)
    result = generate(cfg, out)
    checks = result["checks"]
    # With --verify, generate again into a temporary folder and compare the digests.
    if args.verify:
        with tempfile.TemporaryDirectory() as tmp:
            generate(cfg, Path(tmp))
            first, second = digests(out / "tables"), digests(Path(tmp) / "tables")
            first.update({f"figures/{k}": v for k, v in digests(out / "figures").items()})
            second.update({f"figures/{k}": v for k, v in digests(Path(tmp) / "figures").items()})
            mismatches = sorted(k for k in set(first) | set(second) if first.get(k) != second.get(k))
            checks["determinism"] = {"files": len(first), "mismatches": len(mismatches), "mismatched": mismatches}
    # Write RESULTS.md.
    write_results_md(out, result["registry"], result["tables"], result["figures"], checks, result["w8b_used"], result["dev_run"], cfg)
    # Print the summary of tables, figures, runs and checks.
    print(f"tables: {[(t['name'], t['rows']) for t in result['tables']]}")
    print(f"figures: {[(f['name'], f['points']) for f in result['figures']]}")
    print(f"W8b used: {result['w8b_used']}; development run: {result['dev_run']}; skipped: {result['registry'].skipped}")
    print(f"T1 check: {checks['t1']}")
    # Print the determinism result and exit with status 1 on a mismatch.
    if "determinism" in checks:
        print(f"determinism: {checks['determinism']}")
        if checks["determinism"]["mismatches"]:
            sys.exit(1)


# ---------------------------------------------------------------------------
# command-line entry
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    main()
