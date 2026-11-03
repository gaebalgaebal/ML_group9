"""End-to-end training, evaluation, reporting, and orchestration."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from .data import load_split, prepare
from .metrics import candidate_recall, ranking_metrics
from .modeling import file_checksum, load_model, save_model, train_ranker
from .recommender import FEATURE_COLUMNS, build_feature_frame, score_baselines


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


def model_path(config: dict[str, object]) -> Path:
    return Path(config["artifact_dir"]) / "models" / "hybrid_lr.joblib"


def train(config: dict[str, object]) -> dict[str, object]:
    """Train the ranker only on the leakage-safe Instacart fit split."""
    items, history, target = load_split(config, "instacart", "fit")
    features, target_sets, _ = build_feature_frame(history, items, target, config)
    artifact, alpha_scores = train_ranker(features, target_sets, config)
    checksum = save_model(artifact, model_path(config))
    feature_path = Path(config["artifact_dir"]) / "training_candidates.parquet"
    feature_path.parent.mkdir(parents=True, exist_ok=True)
    features.to_parquet(feature_path, index=False)
    summary = {
        "profile": config["profile"],
        "users": features["user_id"].nunique(),
        "candidates": len(features),
        "positive_candidates": int(features["label"].sum()),
        "candidate_recall": candidate_recall(features, target_sets),
        "selected_alpha": artifact["alpha"],
        "alpha_ndcg_at_10": alpha_scores,
        "model_sha256": checksum,
        "feature_columns": FEATURE_COLUMNS,
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
    """Evaluate all baselines and the frozen LR ranker on one dataset."""
    path = model_path(config)
    checksum_before = file_checksum(path)
    artifact = load_model(path)
    items, history, target = load_split(config, dataset, "eval")
    features, target_sets, _ = build_feature_frame(history, items, target, config)
    scored = score_baselines(features)
    scored["score_lr"] = artifact["model"].predict_proba(scored[artifact["feature_columns"]])[:, 1]

    score_columns = {
        "popularity": "score_popularity",
        "personal_repeat": "score_repeat",
        "item_cf": "score_item_cf",
        "content": "score_content",
        "weighted_hybrid": "score_weighted",
        "lr_hybrid": "score_lr",
    }
    top_k = int(config["evaluation"]["top_k"])
    metrics = {
        name: ranking_metrics(scored, target_sets, column, top_k, len(items))
        for name, column in score_columns.items()
    }
    checksum_after = file_checksum(path)
    if checksum_before != checksum_after:
        raise RuntimeError("Model artifact changed during evaluation")

    predictions = (
        scored.sort_values(["user_id", "score_lr", "item_id"], ascending=[True, False, True])
        .groupby("user_id", observed=True)
        .head(top_k)[["user_id", "item_id", "score_lr", "label"]]
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
        "model_sha256_before": checksum_before,
        "model_sha256_after": checksum_after,
        "frozen_model_verified": checksum_before == checksum_after,
        "metrics": metrics,
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
