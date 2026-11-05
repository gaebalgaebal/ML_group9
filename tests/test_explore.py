"""Tests for nbr.explore on synthetic baskets (PREREG v1.1 score)."""

import inspect
import math

import numpy as np
import pandas as pd
import pytest

from nbr.baselines import item_popularity
from nbr.explore import (
    category_profile,
    category_ranking,
    explore_list,
    explore_list_cf,
    population_category_ranking,
)

COLUMNS = ["user_id", "basket_seq", "item_id", "L2"]


def _history(rows):
    return pd.DataFrame(rows, columns=COLUMNS).astype({"basket_seq": "int32"})


# Two users, three categories (x, y, z). Basket contents (distinct items):
#   u1: seq 1 -> {a(x), b(x), c(y)}   age 1
#       seq 2 -> {a(x), d(z)}         age 0  (a listed twice)
#   u2: seq 1 -> {c(y), e(y)}         age 0
HISTORY = _history([
    ("u1", 1, "a", "x"), ("u1", 1, "b", "x"), ("u1", 1, "c", "y"),
    ("u1", 2, "a", "x"), ("u1", 2, "a", "x"), ("u1", 2, "d", "z"),
    ("u2", 1, "c", "y"), ("u2", 1, "e", "y"),
])
# pop: a=2, b=1, c=2, d=1, e=1 (total 7); pop(x)=3, pop(y)=3, pop(z)=1


def _theta(frame):
    return {(u, c): t for u, c, t in frame.itertuples(index=False)}


def test_category_profile_hand_computed():
    theta, p_c = category_profile(HISTORY, gamma_c=0.5, alpha=2.0)
    assert p_c.to_dict() == pytest.approx({"x": 3 / 7, "y": 3 / 7, "z": 1 / 7})
    assert list(theta.columns) == ["user_id", "category", "theta"]
    assert len(theta) == 6  # every user x category pair
    values = _theta(theta)
    # u1: weights 0.5 (seq 1) and 1 (seq 2). n(x) = 0.5*2 + 1*1 = 2, n(y) = 0.5, n(z) = 1, total 3.5
    assert values[("u1", "x")] == pytest.approx((2 + 2 * 3 / 7) / (3.5 + 2))
    assert values[("u1", "y")] == pytest.approx((0.5 + 2 * 3 / 7) / (3.5 + 2))
    assert values[("u1", "z")] == pytest.approx((1 + 2 * 1 / 7) / (3.5 + 2))
    # u2: one basket, n(y) = 2, total 2
    assert values[("u2", "x")] == pytest.approx((0 + 2 * 3 / 7) / (2 + 2))
    assert values[("u2", "y")] == pytest.approx((2 + 2 * 3 / 7) / (2 + 2))
    assert values[("u2", "z")] == pytest.approx((0 + 2 * 1 / 7) / (2 + 2))
    assert all(t > 0 for t in values.values())


def test_category_profile_alpha_inf_is_exactly_p():
    theta, p_c = category_profile(HISTORY, gamma_c=0.9, alpha=float("inf"))
    merged = theta.merge(p_c.reset_index(), on="category")
    assert (merged["theta"].to_numpy() == merged["p"].to_numpy()).all()


def _brute_force(history, k, gamma_c, alpha):
    """Full scoring of every unbought history item with the same lift formula and tie key."""
    theta, p_c = category_profile(history, gamma_c, alpha)
    pop = item_popularity(history)
    total = float(pop.sum())
    items = history[["item_id", "L2"]].drop_duplicates("item_id").rename(columns={"L2": "category"})
    lift = theta.merge(p_c.reset_index(), on="category")
    lift["lift"] = lift["theta"] / lift["p"]
    bought = set(map(tuple, history[["user_id", "item_id"]].drop_duplicates().to_numpy()))
    rows = []
    for user in sorted(history["user_id"].unique()):
        scored = []
        for item, category in items.itertuples(index=False):
            if (user, item) in bought:
                continue
            lift_value = float(lift[(lift["user_id"] == user) & (lift["category"] == category)]["lift"].iloc[0])
            score = lift_value * (float(pop[item]) / total)
            scored.append((-score, -float(pop[item]), item))
        for rank, (_, _, item) in enumerate(sorted(scored)[:k], start=1):
            rows.append((user, rank, item, "explore"))
    return pd.DataFrame(rows, columns=["user_id", "rank", "item_id", "source"])


def test_explore_list_finite_alpha_hand_computed():
    # u2 (alpha = 1, gamma_c = 1): theta(x) = 3/7 / 3, theta(y) = (2 + 3/7) / 3, theta(z) = 1/7 / 3.
    # Unbought: a(x, pop 2), b(x, 1), d(z, 1). scores = theta/p * pop/7:
    #   a: (1/3) * 2/7, b: (1/3) * 1/7, d: (1/3) * 1/7 -> a, then b before d by item_id.
    lists = explore_list(HISTORY, k=3, gamma_c=1.0, alpha=1.0)
    u2 = lists[lists["user_id"] == "u2"]
    assert list(u2["item_id"]) == ["a", "b", "d"]
    assert list(u2["rank"]) == [1, 2, 3]
    assert (lists["source"] == "explore").all()
    # u1 (unbought: e(y, pop 1)): theta(y)/p(y) = ((1 + 3/7) / (4 + 1)) / (3/7)
    u1 = lists[lists["user_id"] == "u1"]
    assert list(u1["item_id"]) == ["e"]
    pd.testing.assert_frame_equal(lists, _brute_force(HISTORY, 3, 1.0, 1.0))


def test_alpha_inf_equals_global_popularity_excluding_history():
    history = _random_history(np.random.default_rng(11), n_users=12, n_items=30, n_cats=4)
    pop = item_popularity(history)
    bought = history[["user_id", "item_id"]].drop_duplicates()
    rows = []
    for user in sorted(history["user_id"].unique()):
        seen = set(bought[bought["user_id"] == user]["item_id"])
        ranked = sorted(((-pop[i], i) for i in pop.index if i not in seen))
        rows += [(user, r, item, "explore") for r, (_, item) in enumerate(ranked[:5], start=1)]
    expected = pd.DataFrame(rows, columns=["user_id", "rank", "item_id", "source"])
    pd.testing.assert_frame_equal(explore_list(history, 5, 0.7, float("inf")), expected)


def test_alpha_inf_prefers_popular_item_of_other_category():
    # User bought a (pop 3, category x); unbought b (pop 1, same category x) and c (pop 2, category y).
    history = _history([
        ("u", 1, "a", "x"),
        ("v1", 1, "a", "x"), ("v1", 2, "a", "x"), ("v1", 2, "c", "y"),
        ("v2", 1, "b", "x"), ("v2", 1, "c", "y"),
    ])
    assert item_popularity(history).to_dict() == {"a": 3, "b": 1, "c": 2}
    lists = explore_list(history, 2, 1.0, float("inf"))
    assert list(lists[lists["user_id"] == "u"]["item_id"]) == ["c", "b"]


def test_bought_items_excluded_and_fully_bought_category_skipped():
    history = _history([("u", 1, "a", "x"), ("u", 2, "b", "x"), ("w", 1, "c", "y"), ("w", 1, "a", "x")])
    lists = explore_list(history, 5, 1.0, 1.0)
    u = lists[lists["user_id"] == "u"]
    assert list(u["item_id"]) == ["c"]  # category x fully bought -> only c
    w = lists[lists["user_id"] == "w"]
    assert list(w["item_id"]) == ["b"]


def test_tie_breaking():
    # u bought nothing in y. Unbought: b, c in x? -> design: equal scores, pop then item_id.
    history = _history([
        ("u", 1, "a", "x"),
        ("v", 1, "b", "x"), ("v", 1, "c", "x"), ("v", 1, "d", "x"), ("v", 2, "d", "x"),
    ])
    # pop: a=1, b=1, c=1, d=2. For u, same category x -> score proportional to pop: d, then b, c by id.
    lists = explore_list(history, 3, 1.0, 1.0)
    assert list(lists[lists["user_id"] == "u"]["item_id"]) == ["d", "b", "c"]


def _random_history(rng, n_users, n_items, n_cats, max_baskets=6, max_items=5):
    items = [f"i{j:03d}" for j in range(n_items)]
    categories = {item: f"c{rng.integers(n_cats)}" for item in items}
    rows = []
    for u in range(n_users):
        for b in range(int(rng.integers(1, max_baskets + 1))):
            # Skewed item choice so that popularity ties and gaps both occur.
            chosen = rng.choice(items, size=int(rng.integers(1, max_items + 1)), p=_skew(n_items, rng))
            rows += [(f"u{u:03d}", b + 1, item, categories[item]) for item in chosen]
    return _history(rows)


def _skew(n_items, rng):
    weights = 1.0 / (np.arange(n_items) + 1.0)
    return weights / weights.sum()


@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4, 5, 6])
def test_pruning_matches_brute_force(seed):
    rng = np.random.default_rng(seed)
    history = _random_history(rng, n_users=10, n_items=40, n_cats=5)
    for k, gamma_c, alpha in [(3, 0.8, 1.0), (5, 0.5, 10.0), (7, 1.0, 0.0), (4, 0.9, float("inf"))]:
        expected = _brute_force(history, k, gamma_c, alpha)
        pd.testing.assert_frame_equal(explore_list(history, k, gamma_c, alpha), expected)


def test_category_rankings():
    theta, p_c = category_profile(HISTORY, 1.0, 1.0)
    ranking = category_ranking(theta, 2)
    assert list(ranking.columns) == ["user_id", "category", "rank"]
    u1 = ranking[ranking["user_id"] == "u1"]
    # u1 theta (gamma_c = 1, alpha = 1): x = (2 + 3/7) / 4 > y = (1 + 3/7) / 4 > z = (1 + 1/7) / 4
    assert list(u1["category"]) == ["x", "y"] and list(u1["rank"]) == [1, 2]
    u2 = ranking[ranking["user_id"] == "u2"]
    assert list(u2["category"]) == ["y", "x"] and list(u2["rank"]) == [1, 2]

    population = population_category_ranking(p_c, ["u2", "u1"], 2)
    assert list(population["user_id"]) == ["u1", "u1", "u2", "u2"]
    assert list(population["category"]) == ["x", "y", "x", "y"]  # p ties -> category asc
    assert list(population["rank"]) == [1, 2, 1, 2]


def test_explore_list_cf_hand_computed():
    cf = pd.DataFrame({
        "user_id": ["u2", "u2", "u2", "u2"],
        "item_id": ["a", "b", "d", "zz"],
        "score": [1.0, 3.0, 2.0, 0.0],
    })
    theta, _ = category_profile(HISTORY, 1.0, 1.0)
    values = _theta(theta)
    lists = explore_list_cf(HISTORY, cf, 3, 1.0, 1.0)
    u2 = lists[lists["user_id"] == "u2"]
    assert (lists["source"] == "explore_cf").all()
    # x: a (1/4), b (3/4); z: d (2/2 = 1). Scores: b = theta_x * 0.75, a = theta_x * 0.25, d = theta_z * 1.
    scores = {"a": values[("u2", "x")] * 0.25, "b": values[("u2", "x")] * 0.75, "d": values[("u2", "z")]}
    expected = sorted(scores, key=lambda i: -scores[i])
    assert list(u2["item_id"]) == expected
    assert "zz" not in set(lists["item_id"])


def test_deterministic_and_input_untouched():
    history = _random_history(np.random.default_rng(3), 8, 25, 3)
    original = history.copy()
    first = explore_list(history, 4, 0.8, 10.0)
    second = explore_list(history.sample(frac=1, random_state=1), 4, 0.8, 10.0)
    pd.testing.assert_frame_equal(first, second)
    pd.testing.assert_frame_equal(history, original)
    assert not first.duplicated(["user_id", "item_id"]).any()
    assert first.equals(first.sort_values(["user_id", "rank"]).reset_index(drop=True))


def test_parameter_validation():
    with pytest.raises(ValueError):
        explore_list(HISTORY, 0, 1.0, 1.0)
    with pytest.raises(ValueError):
        explore_list(HISTORY, 3, 1.5, 1.0)
    with pytest.raises(ValueError):
        explore_list(HISTORY, 3, 1.0, -1.0)
    with pytest.raises(ValueError):
        category_profile(HISTORY, 1.0, math.nan)


def test_no_target_argument():
    for function in (category_profile, explore_list, explore_list_cf, category_ranking,
                     population_category_ranking):
        assert "target" not in inspect.signature(function).parameters
