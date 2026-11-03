import json
from pathlib import Path

import pandas as pd
import pytest


ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed" / "sample"


@pytest.mark.integration
def test_real_sample_splits_are_strictly_temporal():
    required = [
        PROCESSED / "instacart" / "fit_history.parquet",
        PROCESSED / "instacart" / "fit_target.parquet",
        PROCESSED / "complete_journey" / "eval_history.parquet",
        PROCESSED / "complete_journey" / "eval_target.parquet",
    ]
    if not all(path.exists() for path in required):
        pytest.skip("Run the sample prepare command first")
    for history_path, target_path in [(required[0], required[1]), (required[2], required[3])]:
        history = pd.read_parquet(history_path, columns=["user_id", "basket_seq"])
        target = pd.read_parquet(target_path, columns=["user_id", "basket_seq"])
        history_max = history.groupby("user_id")["basket_seq"].max()
        target_min = target.groupby("user_id")["basket_seq"].min()
        assert (target_min > history_max).all()


@pytest.mark.integration
def test_external_evaluation_kept_model_frozen():
    report = ROOT / "reports" / "metrics" / "sample_complete_journey.json"
    if not report.exists():
        pytest.skip("Run Complete Journey sample evaluation first")
    payload = json.loads(report.read_text(encoding="utf-8"))
    assert payload["frozen_model_verified"] is True
    assert payload["model_sha256_before"] == payload["model_sha256_after"]
