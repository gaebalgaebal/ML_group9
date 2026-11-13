# *------------------- 2026 Fall Semester Machine Learning Term-project: Team 9 --------------------*
# Code First written at 2026-11-05
# Code Final Edited at 2026-11-13
# Editor: Yejun Kim

'''
Purpose : Tests for nbr.explore on synthetic baskets (PREREG v1.1 score).
Inputs  : A hand-written history frame (columns user_id, basket_seq, item_id, L2), random
          histories drawn from seeded numpy generators and a hand-written item-CF score frame.
Outputs : pytest pass/fail results; no file is read or written.
Usage   :
    python -m pytest tests/test_explore.py
'''

# ---------------------------------------------------------------------------
# imports
# ---------------------------------------------------------------------------
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

# ---------------------------------------------------------------------------
# constants
# ---------------------------------------------------------------------------
COLUMNS = ["user_id", "basket_seq", "item_id", "L2"]


# ---------------------------------------------------------------------------
# fixtures and helpers
# ---------------------------------------------------------------------------
def _history(rows):
    '''Build a history frame with the columns of COLUMNS from ``rows``.'''
    return pd.DataFrame(rows, columns=COLUMNS).astype({"basket_seq": "int32"})


# Two users and three categories (x, y, z); the basket contents below are distinct items.
# User u1, seq 1 (age 1) holds a(x), b(x) and c(y).
# User u1, seq 2 (age 0) holds a(x) and d(z), with a listed twice.
# User u2, seq 1 (age 0) holds c(y) and e(y).
HISTORY = _history([
    ("u1", 1, "a", "x"), ("u1", 1, "b", "x"), ("u1", 1, "c", "y"),
    ("u1", 2, "a", "x"), ("u1", 2, "a", "x"), ("u1", 2, "d", "z"),
    ("u2", 1, "c", "y"), ("u2", 1, "e", "y"),
])
# pop: a=2, b=1, c=2, d=1, e=1 (total 7); pop(x)=3, pop(y)=3, pop(z)=1


def _theta(frame):
    '''Map (user_id, category) to theta for each row of the profile frame ``frame``.'''
    return {(u, c): t for u, c, t in frame.itertuples(index=False)}


# ---------------------------------------------------------------------------
# category profile
# ---------------------------------------------------------------------------
def test_category_profile_hand_computed():
    # Compute the category profile with category decay 0.5 and smoothing weight 2.
    theta, p_c = category_profile(HISTORY, gamma_c=0.5, alpha=2.0)
    # The population shares and every theta value equal the hand-computed fractions.
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
    # Compute the category profile with an infinite smoothing weight.
    theta, p_c = category_profile(HISTORY, gamma_c=0.9, alpha=float("inf"))
    merged = theta.merge(p_c.reset_index(), on="category")
    # Every theta value equals the population share of its category exactly.
    assert (merged["theta"].to_numpy() == merged["p"].to_numpy()).all()


# ---------------------------------------------------------------------------
# reference implementation
# ---------------------------------------------------------------------------
def _brute_force(history, k, gamma_c, alpha):
    """Full scoring of every unbought history item with the same lift formula and tie key."""
    # Compute the category profile, the item popularity and its total.
    theta, p_c = category_profile(history, gamma_c, alpha)
    pop = item_popularity(history)
    total = float(pop.sum())
    # Map every item to its category.
    items = history[["item_id", "L2"]].drop_duplicates("item_id").rename(columns={"L2": "category"})
    # Compute the lift theta / p of every user and category.
    lift = theta.merge(p_c.reset_index(), on="category")
    lift["lift"] = lift["theta"] / lift["p"]
    # Collect the (user, item) pairs of the history.
    bought = set(map(tuple, history[["user_id", "item_id"]].drop_duplicates().to_numpy()))
    rows = []
    for user in sorted(history["user_id"].unique()):
        scored = []
        for item, category in items.itertuples(index=False):
            # Skip the items the user bought.
            if (user, item) in bought:
                continue
            # Score the item by the category lift times its popularity share.
            lift_value = float(lift[(lift["user_id"] == user) & (lift["category"] == category)]["lift"].iloc[0])
            score = lift_value * (float(pop[item]) / total)
            scored.append((-score, -float(pop[item]), item))
        # Keep the k best items by score, then popularity, then item id.
        for rank, (_, _, item) in enumerate(sorted(scored)[:k], start=1):
            rows.append((user, rank, item, "explore"))
    # Return the lists in the list frame layout.
    return pd.DataFrame(rows, columns=["user_id", "rank", "item_id", "source"])


# ---------------------------------------------------------------------------
# explore list
# ---------------------------------------------------------------------------
def test_explore_list_finite_alpha_hand_computed():
    # u2 (alpha = 1,gamma_c = 1): theta(x) = 3/7 / 3, theta(y) = (2 + 3/7) / 3, theta(z) = 1/7 / 3.
    # The unbought items of u2 are a(x, pop 2), b(x, 1), d(z, 1), with scores = theta/p * pop/7.
    # Scores a: (1/3) * 2/7, b: (1/3) * 1/7, d: (1/3) * 1/7 give a, then b before d by item_id.
    lists = explore_list(HISTORY, k=3, gamma_c=1.0, alpha=1.0)
    u2 = lists[lists["user_id"] == "u2"]
    # The lists of u2 and u1 hold the hand-computed items and equal the reference lists.
    assert list(u2["item_id"]) == ["a", "b", "d"]
    assert list(u2["rank"]) == [1, 2, 3]
    assert (lists["source"] == "explore").all()
    # u1 (unbought: e(y, pop 1)): theta(y)/p(y) = ((1 + 3/7) / (4 + 1)) / (3/7)
    u1 = lists[lists["user_id"] == "u1"]
    assert list(u1["item_id"]) == ["e"]
    pd.testing.assert_frame_equal(lists, _brute_force(HISTORY, 3, 1.0, 1.0))


def test_alpha_inf_equals_global_popularity_excluding_history():
    # Draw a random history and rank each user's unbought items by popularity, then item id.
    history = _random_history(np.random.default_rng(11), n_users=12, n_items=30, n_cats=4)
    pop = item_popularity(history)
    bought = history[["user_id", "item_id"]].drop_duplicates()
    rows = []
    for user in sorted(history["user_id"].unique()):
        seen = set(bought[bought["user_id"] == user]["item_id"])
        ranked = sorted(((-pop[i], i) for i in pop.index if i not in seen))
        rows += [(user, r, item, "explore") for r, (_, item) in enumerate(ranked[:5], start=1)]
    expected = pd.DataFrame(rows, columns=["user_id", "rank", "item_id", "source"])
    # The explore list with an infinite smoothing weight equals these popularity lists.
    pd.testing.assert_frame_equal(explore_list(history, 5, 0.7, float("inf")), expected)


def test_alpha_inf_prefers_popular_item_of_other_category():
    # User bought a (pop 3, category x); unbought b (pop 1, same category x) and c (pop 2, category y).
    history = _history([
        ("u", 1, "a", "x"),
        ("v1", 1, "a", "x"), ("v1", 2, "a", "x"), ("v1", 2, "c", "y"),
        ("v2", 1, "b", "x"), ("v2", 1, "c", "y"),
    ])
    # With an infinite smoothing weight the more popular item c ranks above b.
    assert item_popularity(history).to_dict() == {"a": 3, "b": 1, "c": 2}
    lists = explore_list(history, 2, 1.0, float("inf"))
    assert list(lists[lists["user_id"] == "u"]["item_id"]) == ["c", "b"]


def test_bought_items_excluded_and_fully_bought_category_skipped():
    # Build a history in which user u bought every item of category x.
    history = _history([("u", 1, "a", "x"), ("u", 2, "b", "x"), ("w", 1, "c", "y"), ("w", 1, "a", "x")])
    lists = explore_list(history, 5, 1.0, 1.0)
    # Each user's list holds only the items the user did not buy.
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


# ---------------------------------------------------------------------------
# random history helpers
# ---------------------------------------------------------------------------
def _random_history(rng, n_users, n_items, n_cats, max_baskets=6, max_items=5):
    '''Draw a random history from ``rng`` with ``n_users``, ``n_items`` and ``n_cats``.

    Each user gets 1 to ``max_baskets`` baskets of 1 to ``max_items`` drawn items.
    '''
    # Name the items and assign each one a random category.
    items = [f"i{j:03d}" for j in range(n_items)]
    categories = {item: f"c{rng.integers(n_cats)}" for item in items}
    rows = []
    for u in range(n_users):
        for b in range(int(rng.integers(1, max_baskets + 1))):
            # Skewed item choice so that popularity ties and gaps both occur.
            chosen = rng.choice(items, size=int(rng.integers(1, max_items + 1)), p=_skew(n_items, rng))
            rows += [(f"u{u:03d}", b + 1, item, categories[item]) for item in chosen]
    # Return the rows as a history frame.
    return _history(rows)


def _skew(n_items, rng):
    '''Return ``n_items`` probabilities proportional to 1 / (position + 1); ``rng`` is unused.'''
    weights = 1.0 / (np.arange(n_items) + 1.0)
    return weights / weights.sum()


# ---------------------------------------------------------------------------
# comparison with the reference implementation
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4, 5, 6])
def test_pruning_matches_brute_force(seed):
    # Draw a random history from the seed.
    rng = np.random.default_rng(seed)
    history = _random_history(rng, n_users=10, n_items=40, n_cats=5)
    # For four parameter sets the explore list equals the reference list.
    for k, gamma_c, alpha in [(3, 0.8, 1.0), (5, 0.5, 10.0), (7, 1.0, 0.0), (4, 0.9, float("inf"))]:
        expected = _brute_force(history, k, gamma_c, alpha)
        pd.testing.assert_frame_equal(explore_list(history, k, gamma_c, alpha), expected)


# ---------------------------------------------------------------------------
# category rankings
# ---------------------------------------------------------------------------
def test_category_rankings():
    # Rank the two best categories of each user by theta.
    theta, p_c = category_profile(HISTORY, 1.0, 1.0)
    ranking = category_ranking(theta, 2)
    # Each user's ranking lists the two categories with the highest theta.
    assert list(ranking.columns) == ["user_id", "category", "rank"]
    u1 = ranking[ranking["user_id"] == "u1"]
    # u1 theta (gamma_c = 1, alpha = 1): x = (2 + 3/7) / 4 > y = (1 + 3/7) / 4 > z = (1 + 1/7) / 4
    assert list(u1["category"]) == ["x", "y"] and list(u1["rank"]) == [1, 2]
    u2 = ranking[ranking["user_id"] == "u2"]
    assert list(u2["category"]) == ["y", "x"] and list(u2["rank"]) == [1, 2]

    # Rank the two best categories by population share for both users.
    population = population_category_ranking(p_c, ["u2", "u1"], 2)
    # Both users get the same two categories, sorted by user.
    assert list(population["user_id"]) == ["u1", "u1", "u2", "u2"]
    assert list(population["category"]) == ["x", "y", "x", "y"]  # p ties -> category asc
    assert list(population["rank"]) == [1, 2, 1, 2]


# ---------------------------------------------------------------------------
# explore list with item-CF scores
# ---------------------------------------------------------------------------
def test_explore_list_cf_hand_computed():
    # Build item-CF scores of u2 for three unbought items and an item outside the history.
    cf = pd.DataFrame({
        "user_id": ["u2", "u2", "u2", "u2"],
        "item_id": ["a", "b", "d", "zz"],
        "score": [1.0, 3.0, 2.0, 0.0],
    })
    theta, _ = category_profile(HISTORY, 1.0, 1.0)
    values = _theta(theta)
    lists = explore_list_cf(HISTORY, cf, 3, 1.0, 1.0)
    u2 = lists[lists["user_id"] == "u2"]
    # The list of u2 follows theta times the within-category share of the item-CF score.
    assert (lists["source"] == "explore_cf").all()
    # x: a (1/4), b (3/4); z: d (2/2 = 1). Scores: b = theta_x * 0.75, a = theta_x * 0.25, d = theta_z * 1.
    scores = {"a": values[("u2", "x")] * 0.25, "b": values[("u2", "x")] * 0.75, "d": values[("u2", "z")]}
    expected = sorted(scores, key=lambda i: -scores[i])
    assert list(u2["item_id"]) == expected
    assert "zz" not in set(lists["item_id"])


# ---------------------------------------------------------------------------
# determinism, validation and signatures
# ---------------------------------------------------------------------------
def test_deterministic_and_input_untouched():
    # Draw a random history and keep a copy of it.
    history = _random_history(np.random.default_rng(3), 8, 25, 3)
    original = history.copy()
    # Build the list from the history and from a shuffled copy of the history.
    first = explore_list(history, 4, 0.8, 10.0)
    second = explore_list(history.sample(frac=1, random_state=1), 4, 0.8, 10.0)
    # Both lists are equal, the input is unchanged and the list is sorted without duplicates.
    pd.testing.assert_frame_equal(first, second)
    pd.testing.assert_frame_equal(history, original)
    assert not first.duplicated(["user_id", "item_id"]).any()
    assert first.equals(first.sort_values(["user_id", "rank"]).reset_index(drop=True))


def test_parameter_validation():
    # A list length of 0, a decay above 1 and a negative or NaN smoothing weight raise ValueError.
    with pytest.raises(ValueError):
        explore_list(HISTORY, 0, 1.0, 1.0)
    with pytest.raises(ValueError):
        explore_list(HISTORY, 3, 1.5, 1.0)
    with pytest.raises(ValueError):
        explore_list(HISTORY, 3, 1.0, -1.0)
    with pytest.raises(ValueError):
        category_profile(HISTORY, 1.0, math.nan)


def test_no_target_argument():
    # No explore function has a parameter named target.
    for function in (category_profile, explore_list, explore_list_cf, category_ranking,
                     population_category_ranking):
        assert "target" not in inspect.signature(function).parameters
