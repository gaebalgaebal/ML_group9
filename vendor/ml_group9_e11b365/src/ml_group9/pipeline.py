"""End-to-end training, evaluation, reporting, and orchestration.

Entry points: `python -m ml_group9 train`, `evaluate --dataset ...` and `run`. Outputs: models and
predictions under artifacts/<profile>/, reports/metrics/<profile>_training.json,
<profile>_<dataset>.json and reports/figures/<profile>_<dataset>.png.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from .attributes import distinct_preference_users
from .data import load_split, prepare
from .metrics import candidate_recall, ranking_metrics
from .modeling import file_checksum, load_model, save_model, train_ranker
from .recommender import FEATURE_COLUMNS_BASE, FEATURE_COLUMNS_PREF, build_feature_frame, score_baselines


def _json_ready(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    return value


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        json.dump(_json_ready(payload), stream, ensure_ascii=False, indent=2)


# Report name -> (artifact stem, score column, feature columns). Both share seed and alpha candidates.
MODEL_VARIANTS = {
    "lr_hybrid": ("hybrid_lr", "score_lr", FEATURE_COLUMNS_BASE),
    "lr_hybrid_pref": ("hybrid_lr_pref", "score_pref_lr", FEATURE_COLUMNS_PREF),
}


def model_path(config: dict[str, object], stem: str = "hybrid_lr") -> Path:
    return Path(config["artifact_dir"]) / "models" / f"{stem}.joblib"


def train(config: dict[str, object]) -> dict[str, object]:
    """Train both rankers only on the leakage-safe Instacart fit split."""
    items, history, target = load_split(config, "instacart", "fit")
    features, target_sets, _ = build_feature_frame(history, items, target, config)
    models: dict[str, dict[str, object]] = {}
    for name, (stem, _, columns) in MODEL_VARIANTS.items():
        artifact, alpha_scores = train_ranker(features, target_sets, config, columns)
        models[name] = {
            "selected_alpha": artifact["alpha"],
            "alpha_ndcg_at_10": alpha_scores,
            "model_sha256": save_model(artifact, model_path(config, stem)),
            "feature_columns": columns,
        }
    feature_path = Path(config["artifact_dir"]) / "training_candidates.parquet"
    feature_path.parent.mkdir(parents=True, exist_ok=True)
    features.to_parquet(feature_path, index=False)
    summary = {
        "profile": config["profile"],
        "users": features["user_id"].nunique(),
        "candidates": len(features),
        "positive_candidates": int(features["label"].sum()),
        "candidate_recall": candidate_recall(features, target_sets),
        "pref_applicable_share": float(features["pref_applicable"].mean()),
        "models": models,
    }
    _write_json(Path(config["report_dir"]) / "metrics" / f"{config['profile']}_training.json", summary)
    return summary


def _save_metric_plot(metrics: dict[str, dict[str, float]], path: Path) -> None:
    rows = [
        {"model": model, "metric": metric, "value": value}
        for model, values in metrics.items()
        for metric, value in values.items()
        if metric in {"precision_at_k", "recall_at_k", "f1_at_k", "ndcg_at_k"}
    ]
    plot_data = pd.DataFrame(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(11, 6))
    sns.barplot(data=plot_data, x="model", y="value", hue="metric")
    plt.title("Top-10 recommendation metrics")
    plt.xlabel("Model")
    plt.ylabel("Macro average")
    plt.xticks(rotation=20)
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def evaluate(config: dict[str, object], dataset: str) -> dict[str, object]:
    """Evaluate all baselines and both frozen LR rankers on one dataset."""
    paths = {name: model_path(config, stem) for name, (stem, _, _) in MODEL_VARIANTS.items()}
    checksums_before = {name: file_checksum(path) for name, path in paths.items()}
    items, history, target = load_split(config, dataset, "eval")
    features, target_sets, context = build_feature_frame(history, items, target, config)
    scored = score_baselines(features)
    for name, (_, column, _) in MODEL_VARIANTS.items():
        artifact = load_model(paths[name])
        scored[column] = artifact["model"].predict_proba(scored[artifact["feature_columns"]])[:, 1]

    score_columns = {
        "popularity": "score_popularity",
        "personal_repeat": "score_repeat",
        "item_cf": "score_item_cf",
        "content": "score_content",
        "weighted_hybrid": "score_weighted",
        **{name: column for name, (_, column, _) in MODEL_VARIANTS.items()},
    }
    top_k = int(config["evaluation"]["top_k"])
    metrics = {
        name: ranking_metrics(scored, target_sets, column, top_k, len(items))
        for name, column in score_columns.items()
    }
    settings = config["attributes"]
    distinct_users = distinct_preference_users(
        context.attribute_counts,
        context.attribute_preferences,
        context.attribute_prior,
        int(settings["distinct_min_n"]),
        float(settings["distinct_min_gap"]),
    )
    subgroup = scored[scored["user_id"].astype(str).isin(distinct_users)]
    subgroup_targets = {user: truth for user, truth in target_sets.items() if user in distinct_users}
    subgroup_metrics = {
        name: ranking_metrics(subgroup, subgroup_targets, score_columns[name], top_k, len(items))
        for name in MODEL_VARIANTS
    }
    checksums_after = {name: file_checksum(path) for name, path in paths.items()}
    if checksums_before != checksums_after:
        raise RuntimeError("Model artifact changed during evaluation")

    predictions = (
        scored.sort_values(["user_id", "score_lr", "item_id"], ascending=[True, False, True])
        .groupby("user_id", observed=True)
        .head(top_k)[["user_id", "item_id", "score_lr", "score_pref_lr", "label"]]
    )
    prediction_path = Path(config["artifact_dir"]) / "predictions" / f"{dataset}_top{top_k}.parquet"
    prediction_path.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_parquet(prediction_path, index=False)

    summary: dict[str, object] = {
        "profile": config["profile"],
        "dataset": dataset,
        "history_users": history["user_id"].nunique(),
        "target_users": len(target_sets),
        "candidate_rows": len(features),
        "candidate_recall": candidate_recall(features, target_sets),
        "model_sha256_before": checksums_before["lr_hybrid"],
        "model_sha256_after": checksums_after["lr_hybrid"],
        "frozen_model_verified": checksums_before == checksums_after,
        "frozen_models": {
            name: {"sha256_before": checksums_before[name], "sha256_after": checksums_after[name]}
            for name in MODEL_VARIANTS
        },
        "pref_applicable_share": float(features["pref_applicable"].mean()),
        "metrics": metrics,
        "distinct_preference_users": {
            "definition": {
                "min_n_c": int(settings["distinct_min_n"]),
                "min_abs_pi_minus_pi_bar": float(settings["distinct_min_gap"]),
                "attributes": sorted(context.attribute_prior),
            },
            "users": len(subgroup_targets),
            "metrics": subgroup_metrics,
        },
    }
    report_stem = f"{config['profile']}_{dataset}"
    _write_json(Path(config["report_dir"]) / "metrics" / f"{report_stem}.json", summary)
    _save_metric_plot(metrics, Path(config["report_dir"]) / "figures" / f"{report_stem}.png")
    return summary


def run(config: dict[str, object]) -> dict[str, object]:
    """Run preparation, training, and both evaluations."""
    return {
        "prepare": prepare(config, "all"),
        "train": train(config),
        "instacart": evaluate(config, "instacart"),
        "complete_journey": evaluate(config, "complete_journey"),
    }
