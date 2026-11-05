"""Tests for nbr.lr_repeat: labels, user-level split, seed reproducibility, tie rule, signal recovery."""

import numpy as np
import pandas as pd
import pytest

from nbr.baselines import item_popularity
from nbr.lr_repeat import build_training, fit_select, lr_repeat_list, select_c, split_users
from nbr.metrics import per_user_recall
from nbr.repeat import rank_scores

GAMMAS = (0.5, 0.8, 1.0)


def _frame(rows):
    return pd.DataFrame(rows, columns=["user_id", "basket_seq", "item_id"]).astype({"basket_seq": "int32"})


def _signal_data(rng, n_users=60, n_items=30, n_baskets=6, basket_size=5):
    """Each basket is a fresh random draw; the target basket repeats the most recent basket exactly."""
    items = [f"i{j:02d}" for j in range(n_items)]
    history, target = [], []
    for u in range(n_users):
        user = f"u{u:02d}"
        last = None
        for s in range(1, n_baskets + 1):
            last = list(rng.choice(items, size=basket_size, replace=False))
            history += [(user, s, item) for item in last]
        target += [(user, n_baskets + 1, item) for item in last]
    return _frame(history), _frame(target)


def test_build_training_labels():
    history = _frame([("u", 1, "A"), ("u", 1, "B"), ("u", 2, "A"), ("v", 1, "A"), ("v", 2, "C")])
    target = _frame([("u", 3, "A"), ("u", 3, "Z"), ("v", 3, "C"), ("v", 3, "C")])
    X, y, groups = build_training(history, target, GAMMAS)
    assert list(X.index.names) == ["user_id", "item_id"]
    labels = dict(zip(X.index, y))
    assert labels == {("u", "A"): 1, ("u", "B"): 0, ("v", "A"): 0, ("v", "C"): 1}
    assert list(groups) == ["u", "u", "v", "v"]
    assert (X.dtypes == "float64").all()


def test_split_users_is_disjoint_and_seeded():
    users = np.array([f"u{i}" for i in range(25)] * 3)
    train, val = split_users(users, 0.2, 42)
    assert len(val) == 5 and len(train) == 20
    assert not set(train) & set(val) and set(train) | set(val) == set(users)
    train2, val2 = split_users(users, 0.2, 42)
    assert list(val2) == list(val)
    assert list(split_users(users, 0.2, 7)[1]) != list(val)


def test_select_c_ties_prefer_smaller():
    assert select_c({0.01: 0.3, 0.1: 0.3, 1.0: 0.2}) == 0.01
    assert select_c({0.01: 0.2, 0.1: 0.3, 1.0: 0.3 + 1e-15}) == 0.1
    assert select_c({0.01: 0.2, 0.1: 0.3, 1.0: 0.31}) == 1.0


def test_fit_select_split_reproducibility_and_outputs():
    rng = np.random.default_rng(3)
    history, target = _signal_data(rng)
    X, y, groups = build_training(history, target, GAMMAS)
    bundle = fit_select(X, y, groups, [0.1, 1.0], 0.2, 42, max_iter=500)
    assert bundle["n_train_users"] == 48 and bundle["n_val_users"] == 12
    assert bundle["n_rows"] == len(X) and 0 < bundle["positive_rate"] < 1
    assert [row["C"] for row in bundle["validation"]] == [0.1, 1.0]
    assert bundle["C"] in (0.1, 1.0)
    assert set(bundle["coefficients"]) == set(X.columns)
    again = fit_select(X, y, groups, [0.1, 1.0], 0.2, 42, max_iter=500)
    assert again["C"] == bundle["C"]
    assert again["validation"][0]["val_recall"] == bundle["validation"][0]["val_recall"]
    np.testing.assert_allclose(again["model"].coef_, bundle["model"].coef_)


def test_fit_select_validation_users_are_not_trained_on(monkeypatch):
    import nbr.lr_repeat as module

    rng = np.random.default_rng(4)
    history, target = _signal_data(rng, n_users=20)
    X, y, groups = build_training(history, target, GAMMAS)
    seen = []
    original = module._fit

    def spy(X_part, y_part, C, max_iter, seed):
        seen.append(set(X_part.index.get_level_values("user_id")))
        return original(X_part, y_part, C, max_iter, seed)

    monkeypatch.setattr(module, "_fit", spy)
    bundle = fit_select(X, y, groups, [1.0], 0.25, 42, max_iter=300)
    train_users, val_users = split_users(groups, 0.25, 42)
    assert seen[0] == set(train_users) and not seen[0] & set(val_users)  # selection fit: training users only
    assert seen[1] == set(groups)  # refit on everything
    assert bundle["n_val_users"] == len(val_users) == 5


def test_lr_repeat_beats_random_ranking_on_signal_data():
    rng = np.random.default_rng(5)
    history, target = _signal_data(rng)
    X, y, groups = build_training(history, target, GAMMAS)
    bundle = fit_select(X, y, groups, [0.1, 1.0, 10.0], 0.2, 42, max_iter=500)
    bundle["decay_gammas"] = list(GAMMAS)
    eval_history, eval_target = _signal_data(np.random.default_rng(6))
    pop = item_popularity(eval_history)
    lists = lr_repeat_list(eval_history, bundle, 10, pop)
    assert list(lists.columns) == ["user_id", "rank", "item_id", "source"]
    assert (lists["source"] == "repeat_lr").all()
    assert not lists.duplicated(["user_id", "item_id"]).any()
    assert lists.groupby("user_id")["rank"].max().le(10).all()
    lr_recall = per_user_recall(lists, eval_target, 10).mean()

    pairs = eval_history[["user_id", "item_id"]].drop_duplicates().reset_index(drop=True)
    pairs["score"] = np.random.default_rng(0).random(len(pairs))
    random_recall = per_user_recall(rank_scores(pairs, pop, 10, "random"), eval_target, 10).mean()
    assert lr_recall > random_recall + 0.2
    # the model should rely on recency: the most recent basket is the target
    assert bundle["coefficients"]["recency"] < 0


def test_fit_select_rejects_single_class():
    history = _frame([("u", 1, "A"), ("u", 2, "B"), ("v", 1, "A"), ("v", 2, "C")])
    X, y, groups = build_training(history, _frame([("u", 3, "Z")]), GAMMAS)
    with pytest.raises(ValueError):
        fit_select(X, y, groups, [1.0], 0.5, 42)
