"""R3 module screening: binned cycle features vs the R2 logistic repeat ranker (W9, H10).

    python -m nbr.r3_eval --profile P --dataset D [--gamma-dir DIR] [--r3-config FILE]

Development profiles only (sample, medium): a profile with role == "final" is
refused. Steps:

  fit split   R2 = nbr.lr_repeat.build_training / fit_select re-run on the C grid of
              configs/r3.yaml; R3 = the same procedure on nbr.features_cycle.cycle_features
              (identical rows and user split: same seed). C is chosen by repeat Recall@10
              on the validation users (ties -> smaller C); both are refit on the whole
              fit split with their C*.
  eval split  k = 10 repeat lists from the eval history: R1 = repeat_star with gamma*
              read from <gamma-dir>/<profile>_<dataset>_tuning.json (W4, read-only),
              R2 = repeat_lr (re-fit), R3 = repeat_r3. Paired bootstrap of repeat
              Recall@10 over users with >= 1 repeat target item: H10 = R3 vs R2,
              auxiliary R3 vs R1 and R2 vs R1 (W6 H6 check), and the three per
              history-length bin (bins of nbr.pipeline).

Outputs: reports/metrics/<profile>_<dataset>_r3.json, reports/tables/<profile>_<dataset>_r3_overdue_coef.csv
and ..._r3_coefficients.csv, reports/figures/<profile>_<dataset>_r3_overdue_coef.png and the model
artifacts/<profile>/<dataset>_r3.joblib (R3 bundle; the R2 re-fit under key "r2_refit").
Feature and scoring functions read only the history frames they are given.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import yaml

from nbr.config import CONFIG_DIR, load_config
from nbr.features import DEFAULT_DECAY_GAMMAS
from nbr.features_cycle import BIN_GROUPS, ONCE_COLUMN, bin_columns, bin_edges, bin_labels, cycle_feature_names, cycle_features
from nbr.io import write_json
from nbr.lr_repeat import SOURCE as R2_SOURCE
from nbr.lr_repeat import build_training, fit_select, lr_repeat_list, score_frame
from nbr.metrics import paired_bootstrap, per_user_recall
from nbr.pipeline import HISTORY_BINS, by_history_length, decision, history_lengths, prepare, run_label, sha256_of
from nbr.repeat import PAIR_KEY, rank_scores, repeat_list

K = 10
R1_SOURCE = "repeat_star"
R3_SOURCE = "repeat_r3"
R3_CONFIG_PATH = CONFIG_DIR / "r3.yaml"
# gamma* is taken read-only from the W4 tuning files (never recomputed here).
DEFAULT_GAMMA_DIR = Path(r"C:\dev_ML-wt-w4\reports\metrics")
BOOTSTRAP = {"n": 1000, "seed": 42, "levels": [0.95, 0.99375]}
HYPOTHESES = {  # name -> (comparator list, model list)
    "H10_repeat_recall": (R2_SOURCE, R3_SOURCE),
    "aux_r3_vs_r1_repeat_recall": (R1_SOURCE, R3_SOURCE),
    "aux_r2_vs_r1_repeat_recall": (R1_SOURCE, R2_SOURCE),
}
BIN_KEYS = tuple(key for _, key, _ in BIN_GROUPS)


# ---------------------------------------------------------------------------
# configuration and inputs
# ---------------------------------------------------------------------------
def load_r3_config(path: Path = R3_CONFIG_PATH) -> dict[str, Any]:
    """The "r3" mapping of configs/r3.yaml (yaml.safe_load: plain YAML -> dict)."""
    with open(path, encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle)
    cfg = dict(loaded["r3"])
    for key in BIN_KEYS:
        bin_edges(cfg, key)  # validate early
    return cfg


def load_gamma_star(profile: str, dataset: str, gamma_dir: Path = DEFAULT_GAMMA_DIR) -> tuple[float, Path]:
    """(gamma*, path) from <gamma_dir>/<profile>_<dataset>_tuning.json; missing file -> FileNotFoundError."""
    path = Path(gamma_dir) / f"{profile}_{dataset}_tuning.json"
    if not path.exists():
        raise FileNotFoundError(f"gamma* source not found: {path} (W4 tuning file required, not recomputed)")
    with open(path, encoding="utf-8") as handle:
        document = json.load(handle)
    return float(document["selections"]["gamma"]), path


# ---------------------------------------------------------------------------
# training rows and fitting (fit split)
# ---------------------------------------------------------------------------
def build_training_r3(
    fit_history: pd.DataFrame,
    fit_target: pd.DataFrame,
    cfg: dict[str, Any],
    decay_gammas: Sequence[float] = DEFAULT_DECAY_GAMMAS,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """R3 features X (MultiIndex (user_id, item_id)), labels y, groups (user_id): same rows as build_training."""
    features = cycle_features(fit_history, cfg, decay_gammas)
    keys = pd.MultiIndex.from_frame(features[PAIR_KEY])
    positives = pd.MultiIndex.from_frame(fit_target[PAIR_KEY].drop_duplicates())
    y = keys.isin(positives).astype("int64")
    X = features.set_index(keys)[cycle_feature_names(cfg, decay_gammas)]
    groups = features["user_id"].to_numpy()
    return X, y, groups


def _at_edge(value: float, grid: Sequence[float]) -> bool:
    return math.isclose(value, min(grid), rel_tol=1e-12) or math.isclose(value, max(grid), rel_tol=1e-12)


def fit_r2_r3(
    history: pd.DataFrame,
    target: pd.DataFrame,
    cfg: dict[str, Any],
    decay_gammas: Sequence[float] = DEFAULT_DECAY_GAMMAS,
    k: int = K,
) -> dict[str, Any]:
    """Fit R2 (repeat_features) and R3 (cycle_features) with identical rows, labels, C grid and user split.

    Returns {"r2": bundle, "r3": bundle, "C_grid", "timing_seconds"}; each bundle is the
    nbr.lr_repeat.fit_select result plus decay_gammas (R3 also carries "bins").
    """
    grid = [float(c) for c in cfg["C"]]
    val_fraction, seed, max_iter = float(cfg["val_fraction"]), int(cfg["seed"]), int(cfg["max_iter"])
    timings: dict[str, float] = {}

    t0 = time.perf_counter()
    X2, y2, groups2 = build_training(history, target, decay_gammas)
    timings["build_training_r2"] = time.perf_counter() - t0
    t0 = time.perf_counter()
    X3, y3, groups3 = build_training_r3(history, target, cfg, decay_gammas)
    timings["build_training_r3"] = time.perf_counter() - t0
    if not X2.index.equals(X3.index) or not np.array_equal(y2, y3) or not np.array_equal(groups2, groups3):
        raise RuntimeError("R2 and R3 training rows differ; the user split would not be comparable")

    bundles = {}
    for name, X, y, groups in (("r2", X2, y2, groups2), ("r3", X3, y3, groups3)):
        t0 = time.perf_counter()
        bundle = fit_select(X, y, groups, grid, val_fraction, seed, max_iter, k)
        timings[f"fit_select_{name}"] = time.perf_counter() - t0
        bundle["decay_gammas"] = [float(g) for g in decay_gammas]
        bundle["at_grid_edge"] = _at_edge(bundle["C"], grid)
        bundles[name] = bundle
    bundles["r3"]["bins"] = {key: [float(e) for e in cfg[key]] for key in BIN_KEYS}
    return {"r2": bundles["r2"], "r3": bundles["r3"], "C_grid": grid, "timing_seconds": timings}


# ---------------------------------------------------------------------------
# scoring (eval split)
# ---------------------------------------------------------------------------
def r3_scores(history: pd.DataFrame, bundle: dict[str, Any]) -> pd.DataFrame:
    """[user_id, item_id, score] for every pair of ``history`` under the fitted R3 bundle."""
    features = cycle_features(history, bundle["bins"], bundle["decay_gammas"])
    X = features.set_index(pd.MultiIndex.from_frame(features[PAIR_KEY]))[bundle["features"]]
    return score_frame(X, bundle["scaler"], bundle["model"])


def r3_list(history: pd.DataFrame, bundle: dict[str, Any], k: int, pop: pd.Series) -> pd.DataFrame:
    """Top-k repurchase candidates per user ranked by R3 (source = "repeat_r3"), contract tie rule."""
    return rank_scores(r3_scores(history, bundle), pop, k, R3_SOURCE)


# ---------------------------------------------------------------------------
# hypotheses
# ---------------------------------------------------------------------------
def compare(a: pd.Series, b: pd.Series, comparator: str, model: str, bootstrap: dict[str, Any] = BOOTSTRAP) -> dict[str, Any]:
    """paired_bootstrap(comparator a, model b): diff = mean(b) - mean(a) with CI and PREREG decision wording."""
    levels = [float(level) for level in bootstrap["levels"]]
    result = paired_bootstrap(a, b, int(bootstrap["n"]), int(bootstrap["seed"]), levels)
    return {
        "comparator": comparator,
        "model": model,
        "metric": "repeat Recall@10",
        "users": "users with repeat target",
        "n_users": result["n_users"],
        "comparator_mean": result["mean_a"],
        "model_mean": result["mean_b"],
        "diff": result["diff"],
        "ci": {str(level): result["ci"][level] for level in levels},
        "decision": {str(level): decision(result["ci"][level]) for level in levels},
    }


# ---------------------------------------------------------------------------
# interpretation: coefficient tables and the overdue-bin figure
# ---------------------------------------------------------------------------
def coefficient_table(r3: dict[str, Any], r2: dict[str, Any] | None = None) -> pd.DataFrame:
    """All standardized R3 coefficients with the training mean/std of each column (R2 coefficient where shared).

    scaler.mean_ / scaler.scale_ are the per-column mean and standard deviation that
    StandardScaler subtracted and divided by, so coef / scale is the log-odds effect
    of one raw unit (for a dummy: of being in the bin instead of not).
    """
    scaler = r3["scaler"]
    rows = []
    for j, name in enumerate(r3["features"]):
        coef = float(r3["coefficients"][name])
        scale = float(scaler.scale_[j])
        rows.append({
            "feature": name,
            "r3_coef_std": coef,
            "train_mean": float(scaler.mean_[j]),
            "train_std": scale,
            "r3_effect_per_unit": coef / scale,
            "r2_coef_std": float(r2["coefficients"][name]) if r2 is not None and name in r2["coefficients"] else np.nan,
        })
    rows.append({"feature": "intercept", "r3_coef_std": float(r3["intercept"]), "train_mean": np.nan, "train_std": np.nan,
                 "r3_effect_per_unit": np.nan, "r2_coef_std": float(r2["intercept"]) if r2 is not None else np.nan})
    return pd.DataFrame(rows)


def overdue_table(r3: dict[str, Any]) -> pd.DataFrame:
    """Coefficient per overdue bin: the reference bin (0 by construction), the ob_* dummies, and ob_once.

    effect_log_odds = coef_std / train_std is the log-odds shift of a pair in the bin relative to
    the reference bin [edges[0], edges[1]) with every other feature held fixed.
    """
    edges = r3["bins"]["overdue_bins"]
    labels = bin_labels(edges)
    columns = [None] + bin_columns("ob", edges)
    scaler, features = r3["scaler"], list(r3["features"])
    rows = []
    for label, column in zip(labels, columns):
        if column is None:
            rows.append({"bin": label, "column": "(reference)", "coef_std": 0.0, "train_mean": np.nan,
                         "train_std": np.nan, "effect_log_odds": 0.0})
            continue
        j = features.index(column)
        coef, scale = float(r3["coefficients"][column]), float(scaler.scale_[j])
        rows.append({"bin": label, "column": column, "coef_std": coef, "train_mean": float(scaler.mean_[j]),
                     "train_std": scale, "effect_log_odds": coef / scale})
    j = features.index(ONCE_COLUMN)
    coef, scale = float(r3["coefficients"][ONCE_COLUMN]), float(scaler.scale_[j])
    rows.append({"bin": "once", "column": ONCE_COLUMN, "coef_std": coef, "train_mean": float(scaler.mean_[j]),
                 "train_std": scale, "effect_log_odds": coef / scale})
    return pd.DataFrame(rows)


def plot_overdue(table: pd.DataFrame, path: Path, title: str) -> Path:
    """Bar chart of effect_log_odds per overdue bin (English labels), written as PNG with matplotlib (Agg)."""
    import matplotlib

    matplotlib.use("Agg")  # file-only backend; no display needed
    import matplotlib.pyplot as plt

    accent, once_color, ink, muted = "#2a78d6", "#9a9994", "#0b0b0b", "#52514e"
    x = np.arange(len(table))
    values = table["effect_log_odds"].to_numpy(dtype="float64")
    colors = [once_color if b == "once" else accent for b in table["bin"]]
    fig, ax = plt.subplots(figsize=(7.5, 4.2), dpi=150)
    ax.bar(x, values, width=0.62, color=colors, edgecolor="none", zorder=3)
    ax.axhline(0.0, color=ink, linewidth=1.0, zorder=2)
    ref = table.index[table["column"] == "(reference)"]
    for i in ref:
        ax.text(x[i], 0.0, "ref.", ha="center", va="bottom", fontsize=8, color=muted)
    ax.set_xticks(x)
    ax.set_xticklabels(table["bin"].tolist(), fontsize=8)
    ax.set_xlabel("overdue = recency / mean_gap (right-open bin); grey = once-bought items (ob_once)", fontsize=8, color=muted)
    ax.set_ylabel("log-odds effect vs. reference bin", fontsize=9)
    ax.set_title(title, fontsize=10, color=ink, loc="left")
    ax.yaxis.grid(True, color="#e2e1dc", linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.spines["left"].set_color(muted)
    ax.spines["bottom"].set_visible(False)
    ax.tick_params(axis="both", colors=muted, labelsize=8)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# artifacts and process information
# ---------------------------------------------------------------------------
def artifact_path(config: dict[str, Any], dataset: str) -> Path:
    return Path(config["repo_root"]) / "artifacts" / str(config["profile"]) / f"{dataset}_r3.joblib"


def _stored(bundle: dict[str, Any]) -> dict[str, Any]:
    keys = ("scaler", "model", "C", "features", "decay_gammas", "coefficients", "intercept", "bins")
    return {key: bundle[key] for key in keys if key in bundle}


def save_models(fitted: dict[str, Any], config: dict[str, Any], dataset: str) -> tuple[Path, str]:
    """joblib.dump (pickle-based) of the R3 bundle with the R2 re-fit under "r2_refit"; returns (path, sha256)."""
    path = artifact_path(config, dataset)
    path.parent.mkdir(parents=True, exist_ok=True)
    stored = _stored(fitted["r3"])
    stored["r2_refit"] = _stored(fitted["r2"])
    joblib.dump(stored, path)
    return path, sha256_of(path)


def load_r3(config: dict[str, Any], dataset: str) -> dict[str, Any]:
    """Load the bundle written by save_models (artifacts/<profile>/<dataset>_r3.joblib)."""
    path = artifact_path(config, dataset)
    if not path.exists():
        raise FileNotFoundError(f"missing R3 artifact {path}")
    return joblib.load(path)


def peak_memory_mb() -> float | None:
    """Peak resident memory of this process in MiB, or None when it cannot be read.

    Windows: ctypes call psapi.GetProcessMemoryInfo(process handle, &counters, size) whose
    PeakWorkingSetSize field is the peak working set in bytes. POSIX: resource.getrusage(
    RUSAGE_SELF).ru_maxrss (KiB on Linux, bytes on macOS).
    """
    try:
        if sys.platform == "win32":
            import ctypes
            from ctypes import wintypes

            class Counters(ctypes.Structure):
                _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD),
                            ("PeakWorkingSetSize", ctypes.c_size_t), ("WorkingSetSize", ctypes.c_size_t),
                            ("QuotaPeakPagedPoolUsage", ctypes.c_size_t), ("QuotaPagedPoolUsage", ctypes.c_size_t),
                            ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t), ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                            ("PagefileUsage", ctypes.c_size_t), ("PeakPagefileUsage", ctypes.c_size_t)]

            counters = Counters()
            counters.cb = ctypes.sizeof(Counters)
            kernel32 = ctypes.windll.kernel32
            kernel32.GetCurrentProcess.restype = wintypes.HANDLE
            query = ctypes.windll.psapi.GetProcessMemoryInfo
            query.argtypes = [wintypes.HANDLE, ctypes.POINTER(Counters), wintypes.DWORD]
            query.restype = wintypes.BOOL
            if not query(kernel32.GetCurrentProcess(), ctypes.byref(counters), counters.cb):
                return None
            return counters.PeakWorkingSetSize / (1024 * 1024)
        import resource

        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return rss / 1024 if sys.platform.startswith("linux") else rss / (1024 * 1024)
    except Exception:  # noqa: BLE001 - best effort, never fails the run
        return None


# ---------------------------------------------------------------------------
# full run
# ---------------------------------------------------------------------------
def _validation_rows(bundle: dict[str, Any]) -> dict[str, Any]:
    return {
        "C_grid": [row["C"] for row in bundle["validation"]],
        "validation": bundle["validation"],
        "C": bundle["C"],
        "at_grid_edge": bundle["at_grid_edge"],
        "n_iter_refit": bundle["n_iter_refit"],
        "features": bundle["features"],
        "coefficients": [{"feature": name, "coef": value} for name, value in bundle["coefficients"].items()],
        "intercept": bundle["intercept"],
        "timing_seconds": bundle["timing_seconds"],
    }


def run(config: dict[str, Any], cfg: dict[str, Any], dataset: str, gamma_dir: Path = DEFAULT_GAMMA_DIR) -> dict[str, Any]:
    """Fit R2/R3 on the fit split, evaluate R1/R2/R3 on the eval split, write figure/tables/model; return the document."""
    if config.get("role") == "final":
        raise PermissionError("r3_eval is a development tool: holdout (role == 'final') profiles are not evaluated")
    gammas = [float(g) for g in config["lr_repeat"]["decay_gammas"]]
    repo = Path(config["repo_root"])
    profile = str(config["profile"])
    stem = f"{profile}_{dataset}_r3"
    timings: dict[str, float] = {}
    t_all = time.perf_counter()

    gamma_star, gamma_path = load_gamma_star(profile, dataset, gamma_dir)

    t0 = time.perf_counter()
    fit = prepare(config, dataset, "fit")
    timings["prepare_fit"] = time.perf_counter() - t0
    fitted = fit_r2_r3(fit["history"], fit["target"], cfg, gammas, K)
    timings.update({f"fit_{key}": value for key, value in fitted["timing_seconds"].items()})
    t0 = time.perf_counter()
    model_path, model_sha = save_models(fitted, config, dataset)
    timings["save_models"] = time.perf_counter() - t0
    del fit

    t0 = time.perf_counter()
    ev = prepare(config, dataset, "eval")
    history, repeat_target, pop = ev["history"], ev["repeat_target"], ev["pop"]
    timings["prepare_eval"] = time.perf_counter() - t0

    lists: dict[str, pd.DataFrame] = {}
    for name, function in (
        (R1_SOURCE, lambda: repeat_list(history, gamma_star, K, pop)),
        (R2_SOURCE, lambda: lr_repeat_list(history, fitted["r2"], K, pop)),
        (R3_SOURCE, lambda: r3_list(history, fitted["r3"], K, pop)),
    ):
        t0 = time.perf_counter()
        lists[name] = function()
        timings[f"list_{name}"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    recall = {name: per_user_recall(frame, repeat_target, K) for name, frame in lists.items()}
    hypotheses = {name: compare(recall[a], recall[b], a, b) for name, (a, b) in HYPOTHESES.items()}
    series = {name: (recall[a], recall[b]) for name, (a, b) in HYPOTHESES.items()}
    breakdown = by_history_length(series, history_lengths(history), dataset, config)
    timings["metrics_and_bootstrap"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    coefficients = coefficient_table(fitted["r3"], fitted["r2"])
    overdue = overdue_table(fitted["r3"])
    tables_dir = repo / "reports" / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    coefficients_csv = tables_dir / f"{stem}_coefficients.csv"
    overdue_csv = tables_dir / f"{stem}_overdue_coef.csv"
    coefficients.to_csv(coefficients_csv, index=False)
    overdue.to_csv(overdue_csv, index=False)
    figure = plot_overdue(overdue, repo / "reports" / "figures" / f"{stem}_overdue_coef.png",
                          f"R3 coefficient by overdue bin - {dataset}, {run_label(config)}")
    timings["interpretation"] = time.perf_counter() - t0
    timings["total"] = time.perf_counter() - t_all

    return {
        "task": "W9 R3 module screening (binned cycle features)",
        "profile": profile,
        "dataset": dataset,
        "k": K,
        "label": run_label(config),
        "r3_config": {key: cfg[key] for key in cfg},
        "decay_gammas": gammas,
        "gamma_star": {"value": gamma_star, "source": str(gamma_path)},
        "lists": {R1_SOURCE: "repeat_star: decayed frequency with gamma* (R1)",
                  R2_SOURCE: "repeat_lr: R2 logistic ranker re-fit on the r3 C grid",
                  R3_SOURCE: "repeat_r3: logistic ranker on R2 + binned cycle features"},
        "n_users": {"eval_all": int(ev["target"]["user_id"].nunique()),
                    "eval_repeat_target": int(repeat_target["user_id"].nunique())},
        "fit": {
            "n_rows": fitted["r3"]["n_rows"],
            "positive_rate": fitted["r3"]["positive_rate"],
            "n_train_users": fitted["r3"]["n_train_users"],
            "n_val_users": fitted["r3"]["n_val_users"],
            "val_fraction": fitted["r3"]["val_fraction"],
            "seed": fitted["r3"]["seed"],
            "max_iter": fitted["r3"]["max_iter"],
            "same_user_split": fitted["r2"]["n_val_users"] == fitted["r3"]["n_val_users"],
            "r2": _validation_rows(fitted["r2"]),
            "r3": _validation_rows(fitted["r3"]),
        },
        "list_metrics": {name: {"repeat_recall": float(values.mean())} for name, values in recall.items()},
        "hypotheses": hypotheses,
        "by_history_length": breakdown,
        "history_bins": HISTORY_BINS[dataset],
        "bootstrap": BOOTSTRAP,
        "overdue_coefficients": overdue.to_dict(orient="records"),
        "notes": {"ob_once_equals_once": True,  # ob_once is identical to the R2 feature `once`; L2 shares the weight
                  "effect_log_odds": "coef_std / train_std = log-odds shift of the bin vs the reference bin"},
        "outputs": {"figure": str(figure), "overdue_table": str(overdue_csv), "coefficient_table": str(coefficients_csv)},
        "artifact": {"path": str(model_path), "sha256": model_sha},
        "timing_seconds": timings,
        "peak_memory_mb": peak_memory_mb(),
    }


def report_path(config: dict[str, Any], dataset: str) -> Path:
    return Path(config["repo_root"]) / "reports" / "metrics" / f"{config['profile']}_{dataset}_r3.json"


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="python -m nbr.r3_eval", description="W9 R3 screening (H10)")
    parser.add_argument("--profile", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--gamma-dir", default=str(DEFAULT_GAMMA_DIR), help="folder with <profile>_<dataset>_tuning.json")
    parser.add_argument("--r3-config", default=str(R3_CONFIG_PATH))
    args = parser.parse_args(argv)
    config = load_config(args.profile)
    cfg = load_r3_config(Path(args.r3_config))
    t0 = time.perf_counter()
    document = run(config, cfg, args.dataset, Path(args.gamma_dir))
    path = write_json(report_path(config, args.dataset), document, config)
    h10 = document["hypotheses"]["H10_repeat_recall"]
    print(f"r3_eval {args.profile}/{args.dataset}: C*(R2)={document['fit']['r2']['C']} C*(R3)={document['fit']['r3']['C']} "
          f"H10 diff={h10['diff']:.5f} ci95={h10['ci']['0.95']} -> {h10['decision']['0.95']}; "
          f"wrote {path} in {time.perf_counter() - t0:.1f}s")


if __name__ == "__main__":
    main()
