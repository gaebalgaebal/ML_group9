# *------------------- 2026 Fall Semester Machine Learning Term-project: Team 9 --------------------*
# Code First written at 2026-11-05
# Code Final Edited at 2026-11-13
# Editor: Yejun Kim, Bonjun Gu

'''
Purpose : Tests for nbr.lr_repeat: labels, user-level split, seed reproducibility, tie rule,
          signal recovery.
Inputs  : Small hand-written history and target frames (columns user_id, basket_seq, item_id)
          and seeded random data in which the target basket repeats the most recent basket.
Outputs : None; the tests assert on the training frames, the user split, the selected C, the
          fit_select bundle and the lists returned by nbr.lr_repeat.
Usage   :
    python -m pytest tests/test_lr_repeat.py
'''

# ---------------------------------------------------------------------------
# imports
# ---------------------------------------------------------------------------
import numpy as np
import pandas as pd
import pytest

from nbr.baselines import item_popularity
from nbr.lr_repeat import build_training, fit_select, lr_repeat_list, select_c, split_users
from nbr.metrics import per_user_recall
from nbr.repeat import rank_scores

# ---------------------------------------------------------------------------
# constants
# ---------------------------------------------------------------------------
GAMMAS = (0.5, 0.8, 1.0)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _frame(rows):
    '''Build a basket frame (user_id, basket_seq, item_id) from the tuples in ``rows``.'''
    return pd.DataFrame(rows, columns=["user_id", "basket_seq", "item_id"]).astype({"basket_seq": "int32"})


def _signal_data(rng, n_users=60, n_items=30, n_baskets=6, basket_size=5):
    """Each basket is a fresh random draw; the target basket repeats the most recent basket exactly."""
    # Name the items i00, i01, ... up to the item count.
    items = [f"i{j:02d}" for j in range(n_items)]
    history, target = [], []
    for u in range(n_users):
        user = f"u{u:02d}"
        last = None
        for s in range(1, n_baskets + 1):
            # Draw the items of this basket without replacement and add them to the history.
            last = list(rng.choice(items, size=basket_size, replace=False))
            history += [(user, s, item) for item in last]
        # Use the items of the last basket as the user's target basket.
        target += [(user, n_baskets + 1, item) for item in last]
    return _frame(history), _frame(target)


# ---------------------------------------------------------------------------
# training rows, user split and C tie rule
# ---------------------------------------------------------------------------
def test_build_training_labels():
    # Build a two-user history and a target with an unseen item and a repeated row.
    history = _frame([("u", 1, "A"), ("u", 1, "B"), ("u", 2, "A"), ("v", 1, "A"), ("v", 2, "C")])
    target = _frame([("u", 3, "A"), ("u", 3, "Z"), ("v", 3, "C"), ("v", 3, "C")])
    X, y, groups = build_training(history, target, GAMMAS)
    # Expect one row per history pair, labelled 1 when the pair is in the target.
    assert list(X.index.names) == ["user_id", "item_id"]
    labels = dict(zip(X.index, y))
    assert labels == {("u", "A"): 1, ("u", "B"): 0, ("v", "A"): 0, ("v", "C"): 1}
    assert list(groups) == ["u", "u", "v", "v"]
    assert (X.dtypes == "float64").all()


def test_split_users_is_disjoint_and_seeded():
    # Split 25 users, each listed three times, with a validation share of 0.2 and seed 42.
    users = np.array([f"u{i}" for i in range(25)] * 3)
    train, val = split_users(users, 0.2, 42)
    # Expect 20 training and 5 validation users that are disjoint and cover all users.
    assert len(val) == 5 and len(train) == 20
    assert not set(train) & set(val) and set(train) | set(val) == set(users)
    # Expect the same validation users for the same seed and other ones for another seed.
    train2, val2 = split_users(users, 0.2, 42)
    assert list(val2) == list(val)
    assert list(split_users(users, 0.2, 7)[1]) != list(val)


def test_select_c_ties_prefer_smaller():
    # Expect the smallest C among the values tied for the best score.
    assert select_c({0.01: 0.3, 0.1: 0.3, 1.0: 0.2}) == 0.01
    assert select_c({0.01: 0.2, 0.1: 0.3, 1.0: 0.3 + 1e-15}) == 0.1
    assert select_c({0.01: 0.2, 0.1: 0.3, 1.0: 0.31}) == 1.0


# ---------------------------------------------------------------------------
# fit and selection
# ---------------------------------------------------------------------------
def test_fit_select_split_reproducibility_and_outputs():
    # Fit and select C on seeded signal data of 60 users.
    rng = np.random.default_rng(3)
    history, target = _signal_data(rng)
    X, y, groups = build_training(history, target, GAMMAS)
    bundle = fit_select(X, y, groups, [0.1, 1.0], 0.2, 42, max_iter=500)
    # Expect a 48/12 user split, one validation row per C and one coefficient per feature.
    assert bundle["n_train_users"] == 48 and bundle["n_val_users"] == 12
    assert bundle["n_rows"] == len(X) and 0 < bundle["positive_rate"] < 1
    assert [row["C"] for row in bundle["validation"]] == [0.1, 1.0]
    assert bundle["C"] in (0.1, 1.0)
    assert set(bundle["coefficients"]) == set(X.columns)
    # Expect a second call with the same arguments to give the same C, recall and coefficients.
    again = fit_select(X, y, groups, [0.1, 1.0], 0.2, 42, max_iter=500)
    assert again["C"] == bundle["C"]
    assert again["validation"][0]["val_recall"] == bundle["validation"][0]["val_recall"]
    np.testing.assert_allclose(again["model"].coef_, bundle["model"].coef_)


def test_fit_select_validation_users_are_not_trained_on(monkeypatch):
    import nbr.lr_repeat as module

    # Build the training rows of seeded signal data of 20 users.
    rng = np.random.default_rng(4)
    history, target = _signal_data(rng, n_users=20)
    X, y, groups = build_training(history, target, GAMMAS)
    seen = []
    original = module._fit

    def spy(X_part, y_part, C, max_iter, seed):
        '''Record the users of ``X_part`` and call the original fit with the same arguments.'''
        # Store the set of users in the rows of this fit.
        seen.append(set(X_part.index.get_level_values("user_id")))
        # Fit the model with the unpatched function.
        return original(X_part, y_part, C, max_iter, seed)

    # Run fit_select for one C with the fit function replaced by the recording wrapper.
    monkeypatch.setattr(module, "_fit", spy)
    bundle = fit_select(X, y, groups, [1.0], 0.25, 42, max_iter=300)
    train_users, val_users = split_users(groups, 0.25, 42)
    # Expect the first fit on the training users only and the second fit on all users.
    assert seen[0] == set(train_users) and not seen[0] & set(val_users)  # selection fit: training users only
    assert seen[1] == set(groups)  # refit on everything
    assert bundle["n_val_users"] == len(val_users) == 5


# ---------------------------------------------------------------------------
# signal recovery and rejected input
# ---------------------------------------------------------------------------
def test_lr_repeat_beats_random_ranking_on_signal_data():
    # Fit the model on seeded signal data.
    rng = np.random.default_rng(5)
    history, target = _signal_data(rng)
    X, y, groups = build_training(history, target, GAMMAS)
    bundle = fit_select(X, y, groups, [0.1, 1.0, 10.0], 0.2, 42, max_iter=500)
    bundle["decay_gammas"] = list(GAMMAS)
    # Rank a second signal data set drawn with another seed.
    eval_history, eval_target = _signal_data(np.random.default_rng(6))
    pop = item_popularity(eval_history)
    lists = lr_repeat_list(eval_history, bundle, 10, pop)
    # Expect list columns, the source repeat_lr, no repeated item and ranks up to 10.
    assert list(lists.columns) == ["user_id", "rank", "item_id", "source"]
    assert (lists["source"] == "repeat_lr").all()
    assert not lists.duplicated(["user_id", "item_id"]).any()
    assert lists.groupby("user_id")["rank"].max().le(10).all()
    lr_recall = per_user_recall(lists, eval_target, 10).mean()

    # Rank the same history pairs by seeded random scores.
    pairs = eval_history[["user_id", "item_id"]].drop_duplicates().reset_index(drop=True)
    pairs["score"] = np.random.default_rng(0).random(len(pairs))
    random_recall = per_user_recall(rank_scores(pairs, pop, 10, "random"), eval_target, 10).mean()
    # Expect the model's mean Recall@10 to exceed the random one by more than 0.2.
    assert lr_recall > random_recall + 0.2
    # the model should rely on recency: the most recent basket is the target
    assert bundle["coefficients"]["recency"] < 0


def test_fit_select_rejects_single_class():
    # Build training rows whose target holds only an item that no user bought before.
    history = _frame([("u", 1, "A"), ("u", 2, "B"), ("v", 1, "A"), ("v", 2, "C")])
    X, y, groups = build_training(history, _frame([("u", 3, "Z")]), GAMMAS)
    # Expect ValueError because every label is 0.
    with pytest.raises(ValueError):
        fit_select(X, y, groups, [1.0], 0.5, 42)
