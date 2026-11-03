import pandas as pd

from ml_group9.metrics import candidate_recall, ranking_metrics


def test_ranking_metrics_match_hand_calculation():
    frame = pd.DataFrame({"user_id": ["u1", "u1", "u1"], "item_id": ["a", "b", "c"], "score": [0.9, 0.8, 0.7]})
    truth = {"u1": {"a", "c"}}
    metrics = ranking_metrics(frame, truth, "score", top_k=2, catalog_size=3)
    assert metrics["precision_at_k"] == 0.5
    assert metrics["recall_at_k"] == 0.5
    assert metrics["catalog_coverage"] == 2 / 3
    assert candidate_recall(frame, truth) == 1.0
