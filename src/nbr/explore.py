# *------------------- 2026 Fall Semester Machine Learning Term-project: Team 9 --------------------*
# Code First written at 2026-11-05
# Code Final Edited at 2026-11-13
# Editor: Yejun Kim

'''
Purpose : Hierarchical exploration (E1): category profile (theta), category-to-item explore
          list, category rankings for H4, and the item-CF flavoured explore variant (E4).
          H4 compares the top-10 L2 categories by theta_u with the top-10 by p(c).
          Basket ages, item popularity and the tie rule for lists come from
          nbr.repeat and nbr.baselines so that every component shares one definition.
Inputs  : A history frame. Required history columns: user_id, basket_seq, item_id and the
          category column (``level``, default "L2"). Functions only use the frames they
          receive; no file reads and no target frames.
Outputs : Category profile frames [user_id, category, theta], list frames [user_id, rank,
          item_id, source] and category rankings [user_id, category, rank].
          All results are deterministic.
Usage   :
    from nbr.explore import category_ranking, explore_list
    explore_star = explore_list(history, K, gamma_c_star, alpha_star, level)
    model_rank = category_ranking(theta, K)
'''

from __future__ import annotations

# ---------------------------------------------------------------------------
# imports
# ---------------------------------------------------------------------------
import math
from collections.abc import Iterable

import numpy as np
import pandas as pd

from nbr.baselines import item_popularity
from nbr.repeat import basket_ages, rank_scores

# ---------------------------------------------------------------------------
# constants
# ---------------------------------------------------------------------------
BASKET_KEY = ["user_id", "basket_seq"]
PAIR_KEY = ["user_id", "item_id"]
LIST_COLUMNS = ["user_id", "rank", "item_id", "source"]
CATEGORY_RANK_COLUMNS = ["user_id", "category", "rank"]


# ---------------------------------------------------------------------------
# shared helpers
# ---------------------------------------------------------------------------
def _check_params(k: int, gamma_c: float, alpha: float) -> None:
    '''Raise ValueError unless ``k`` >= 1, ``gamma_c`` lies in [0, 1] and ``alpha`` >= 0.'''
    # Reject a list length below 1.
    if k < 1:
        raise ValueError(f"k must be >= 1, got {k}")
    # Reject a decay outside [0, 1].
    if not 0.0 <= gamma_c <= 1.0:
        raise ValueError(f"gamma_c must lie in [0, 1], got {gamma_c}")
    # Reject a negative or NaN smoothing weight.
    if not (alpha >= 0.0):  # also rejects NaN; alpha = inf is allowed
        raise ValueError(f"alpha must be >= 0 (inf allowed), got {alpha}")


def _catalog(history: pd.DataFrame, level: str) -> pd.DataFrame:
    """Items seen in history: [item_id, category, pop, order, cat_size].

    ``order`` ranks items inside their category by pop desc, item_id asc
    (0 = most popular). An item's category is its first category in history.
    """
    # Count the item popularity of the history.
    pop = item_popularity(history)
    # Take the first category of each item in the history as a string.
    items = history[["item_id", level]].drop_duplicates("item_id").rename(columns={level: "category"})
    items["category"] = items["category"].astype(str)
    # Join the popularity to the items.
    catalog = items.merge(pop.reset_index(), on="item_id")
    # Order the items by category, popularity descending and item ID ascending.
    catalog = catalog.sort_values(["category", "pop", "item_id"], ascending=[True, False, True], kind="mergesort")
    # Number the items inside each category from 0 in that order.
    catalog["order"] = catalog.groupby("category", sort=False).cumcount().astype("int64")
    # Store the number of items of each category.
    catalog["cat_size"] = catalog.groupby("category", sort=False)["item_id"].transform("size").astype("int64")
    # Return the catalogue with a fresh row index.
    return catalog.reset_index(drop=True)


# ---------------------------------------------------------------------------
# category profile
# ---------------------------------------------------------------------------
def category_profile(
    history: pd.DataFrame, gamma_c: float, alpha: float, level: str = "L2"
) -> tuple[pd.DataFrame, pd.Series]:
    """Smoothed, decayed category profile theta_u(c) of every user of ``history`` and the population share p(c);
    categories are the column ``level``.

    theta_u(c) = (sum_b gamma_c^age(b) n_b(c) + alpha p(c)) / (sum_b gamma_c^age(b) n_b + alpha)
    where n_b(c) is the number of distinct items of category c in basket b and
    n_b the number of distinct items in b. p(c) = sum_{i in c} pop(i) / sum_i pop(i).
    alpha = inf gives theta_u(c) = p(c) exactly (no numerical approximation).

    Returns (theta, p_c): theta has one row for EVERY (user, category) pair
    (users x categories rows, e.g. 20,000 x 134 for Instacart medium; the
    frame is a few hundred MB at most and its size is reported by callers),
    columns [user_id, category(str), theta(float64)], sorted by user_id,
    category; p_c is a float64 Series named "p" indexed by category.
    """
    # Validate the decay and the smoothing weight.
    _check_params(1, gamma_c, alpha)
    # Build the item catalogue and the total popularity.
    catalog = _catalog(history, level)
    total_pop = float(catalog["pop"].sum())
    # Compute the population share p(c) of every category.
    p_c = (catalog.groupby("category", sort=True)["pop"].sum() / total_pop).astype("float64")
    p_c.name = "p"
    p_c.index.name = "category"

    # Keep one row per (basket, item) with its category as a string.
    events = history[BASKET_KEY + ["item_id", level]].drop_duplicates(BASKET_KEY + ["item_id"])
    events = events.rename(columns={level: "category"})
    events["category"] = events["category"].astype(str)
    # Attach the basket ages and the decay weight gamma_c^age.
    events = events.merge(basket_ages(history), on=BASKET_KEY, how="left")
    events["weight"] = np.power(float(gamma_c), events["age"].to_numpy(dtype="float64"))
    # Sum the weights per (user, category) and per user.
    numerator = events.groupby(["user_id", "category"], sort=True)["weight"].sum().rename("num")
    denominator = events.groupby("user_id", sort=True)["weight"].sum().rename("den")

    # Build one row for every (user, category) pair.
    users = pd.DataFrame({"user_id": denominator.index})
    categories = pd.DataFrame({"category": p_c.index})
    theta = users.merge(categories, how="cross")
    # Join the weighted counts, with zero for pairs without a purchase.
    theta = theta.merge(numerator.reset_index(), on=["user_id", "category"], how="left")
    theta["num"] = theta["num"].fillna(0.0)
    # Join the user totals and the population shares.
    theta = theta.merge(denominator.reset_index(), on="user_id", how="left")
    theta = theta.merge(p_c.reset_index(), on="category", how="left")
    # Use p(c) itself for infinite alpha and the smoothed ratio otherwise.
    if math.isinf(alpha):
        theta["theta"] = theta["p"].to_numpy(dtype="float64")
    else:
        theta["theta"] = (theta["num"] + alpha * theta["p"]) / (theta["den"] + alpha)
    # Sort by user and category and return theta together with p(c).
    theta = theta.sort_values(["user_id", "category"], kind="mergesort").reset_index(drop=True)
    return theta[["user_id", "category", "theta"]].astype({"theta": "float64"}), p_c


# ---------------------------------------------------------------------------
# explore list with exact pruning
# ---------------------------------------------------------------------------
def _unbought_top(bought: pd.DataFrame, pairs: pd.DataFrame, k: int) -> pd.DataFrame:
    """For each (user, category) pair, the first k within-category orders not bought by the user.

    ``bought`` has [user_id, category, order] (distinct items the user bought,
    with their within-category order); ``pairs`` has [user_id, category,
    cat_size]. The bought orders of a pair split 0..cat_size-1 into gaps of
    unbought orders; each gap is expanded vectorially (numpy.repeat +
    numpy.arange) until k orders are collected. Output: [user_id, category, order].
    """
    # Keep the bought orders of the requested (user, category) pairs, sorted by order.
    keys = ["user_id", "category"]
    owned = bought.merge(pairs[keys], on=keys, how="inner")
    owned = owned.sort_values(keys + ["order"], kind="mergesort")
    # Number the bought orders of each pair and look up the previous bought order.
    owned["m"] = owned.groupby(keys, sort=False).cumcount()
    owned["prev"] = owned.groupby(keys, sort=False)["order"].shift(1).fillna(-1).astype("int64")
    # Build the gaps of unbought orders that end before each bought order.
    inner_gaps = pd.DataFrame({
        "user_id": owned["user_id"].to_numpy(),
        "category": owned["category"].to_numpy(),
        "start": (owned["prev"] + 1).to_numpy(),
        "end": (owned["order"] - 1).to_numpy(),
        "before": (owned["prev"] + 1 - owned["m"]).to_numpy(),
    })
    # Find the last bought order and the bought count of each pair.
    last = owned.groupby(keys, sort=False).agg(last=("order", "max"), count=("order", "size")).reset_index()
    # Join them to all requested pairs, with defaults for pairs without a purchase.
    tail = pairs.merge(last, on=keys, how="left")
    tail["last"] = tail["last"].fillna(-1).astype("int64")
    tail["count"] = tail["count"].fillna(0).astype("int64")
    # Build the gap from the last bought order to the end of each category.
    tail_gaps = pd.DataFrame({
        "user_id": tail["user_id"].to_numpy(),
        "category": tail["category"].to_numpy(),
        "start": (tail["last"] + 1).to_numpy(),
        "end": (tail["cat_size"] - 1).to_numpy(),
        "before": (tail["last"] + 1 - tail["count"]).to_numpy(),
    })
    # Keep the non-empty gaps that start before k unbought orders are collected.
    gaps = pd.concat([inner_gaps, tail_gaps], ignore_index=True)
    gaps = gaps[(gaps["start"] <= gaps["end"]) & (gaps["before"] < k)]
    # Compute how many orders to take from each gap.
    take = np.minimum(gaps["end"] - gaps["start"] + 1, k - gaps["before"]).to_numpy(dtype="int64")
    # Repeat each gap index once per taken order and compute the offset inside the gap.
    repeat_index = np.repeat(np.arange(len(gaps)), take)
    offsets = np.arange(int(take.sum())) - np.repeat(np.cumsum(take) - take, take)
    # Return one row per taken order.
    return pd.DataFrame({
        "user_id": gaps["user_id"].to_numpy()[repeat_index],
        "category": gaps["category"].to_numpy()[repeat_index],
        "order": gaps["start"].to_numpy()[repeat_index] + offsets,
    })


def explore_list(
    history: pd.DataFrame, k: int, gamma_c: float, alpha: float, level: str = "L2"
) -> pd.DataFrame:
    """Exact top-k explore list per user (source = "explore") over items outside I(H_u); ``gamma_c`` and ``alpha``
    parametrise category_profile and ``level`` is the category column.

    Score (PREREG v1.1): s_exp(u, i) = theta_u(c) * pop(i) / pop(c), with
    pop(c) = sum of pop over all items of c. It is computed in the equivalent
    lift form (theta_u(c) / p(c)) * (pop(i) / sum_j pop(j)), so that with
    alpha = inf (theta = p) the scores equal plain popularity exactly and the
    list coincides with the H2 comparator (popularity excluding history).
    Ties: score desc, pop desc, item_id asc.

    Pruning argument. Within one category the score is pop(i) times a
    constant that depends only on (u, c), so the best unbought item of each
    category is its most popular unbought item, and the within-category order
    of candidates equals the global tie order. Rank the categories of a user
    by the full tie key of their best unbought item and keep the top k; every
    kept category contributes its k best unbought items. If an item i of a
    category outside the top k were in the user's global top k, the best
    items of the k kept categories would all rank strictly above i (their key
    is at least the key of c's best item, which is at least i's key, and the
    order is total), leaving no room for i. Hence the exact top k lies inside
    the k x k candidates. Categories with no unbought item are skipped.

    Items that never occur in ``history`` have pop = 0 and no category here;
    they can never be recommended by this function.
    """
    # Validate the parameters.
    _check_params(k, gamma_c, alpha)
    # Build the item catalogue and the total popularity.
    catalog = _catalog(history, level)
    total_pop = float(catalog["pop"].sum())
    # Compute the category profile and the lift theta / p of every (user, category) pair.
    theta, p_c = category_profile(history, gamma_c, alpha, level)
    theta = theta.merge(p_c.reset_index(), on="category", how="left")
    theta["lift"] = theta["theta"] / theta["p"]

    # Collect the category sizes, the bought items with their orders and all pairs.
    cat_size = catalog.drop_duplicates("category")[["category", "cat_size"]]
    bought = history[PAIR_KEY].drop_duplicates().merge(catalog[["item_id", "category", "order"]], on="item_id")
    pairs = theta[["user_id", "category"]].merge(cat_size, on="category", how="left")

    def _score(frame: pd.DataFrame) -> pd.DataFrame:
        '''Add item_id, pop, lift and score to ``frame`` ([user_id, category, order]).'''
        # Look up the item and its popularity by category and order.
        frame = frame.merge(catalog[["category", "order", "item_id", "pop"]], on=["category", "order"], how="left")
        # Join the lift of the (user, category) pair.
        frame = frame.merge(theta[["user_id", "category", "lift"]], on=["user_id", "category"], how="left")
        # Compute the score as the lift times the popularity share of the item.
        frame["score"] = frame["lift"].to_numpy() * (frame["pop"].to_numpy() / total_pop)
        return frame

    # Stage 1: best unbought item per (user, category) -> top-k categories per user.
    best = _score(_unbought_top(bought, pairs, 1))
    # Order the categories of each user by the tie key of their best unbought item.
    best = best.sort_values(
        ["user_id", "score", "pop", "item_id"], ascending=[True, False, False, True], kind="mergesort"
    )
    # Number the categories of each user and keep the first k with their sizes.
    best["cat_rank"] = best.groupby("user_id", sort=False).cumcount()
    selected = best[best["cat_rank"] < k][["user_id", "category"]].merge(cat_size, on="category", how="left")

    # Stage 2: k best unbought items of each selected category -> exact top k.
    candidates = _score(_unbought_top(bought, selected, k))
    # Rank the candidates into the top-k list with the popularity tie rule.
    pop = catalog.set_index("item_id")["pop"]
    return rank_scores(candidates[["user_id", "item_id", "score"]], pop, k, "explore")


# ---------------------------------------------------------------------------
# category rankings (H4)
# ---------------------------------------------------------------------------
def category_ranking(theta: pd.DataFrame, k: int) -> pd.DataFrame:
    """Top-k categories per user by theta desc, then category asc -> [user_id, category, rank]."""
    # Reject a list length below 1.
    if k < 1:
        raise ValueError(f"k must be >= 1, got {k}")
    # Order by user, theta descending and category ascending.
    ordered = theta.sort_values(["user_id", "theta", "category"], ascending=[True, False, True], kind="mergesort")
    # Number the categories of each user from 1 and keep ranks up to k.
    ordered["rank"] = ordered.groupby("user_id", sort=False).cumcount() + 1
    ordered = ordered[ordered["rank"] <= k]
    return ordered[CATEGORY_RANK_COLUMNS].reset_index(drop=True)


def population_category_ranking(p_c: pd.Series, users: Iterable[str], k: int) -> pd.DataFrame:
    """Top-k categories of ``p_c`` (category -> share) by p desc, then category asc, replicated for every user in ``users``."""
    if k < 1:
        # Reject a list length below 1.
        raise ValueError(f"k must be >= 1, got {k}")
    # Turn the shares into a frame with string categories.
    top = p_c.rename("p").rename_axis("category").reset_index()
    top["category"] = top["category"].astype(str)
    # Keep the k categories with the highest share, ties by category ascending.
    top = top.sort_values(["p", "category"], ascending=[False, True], kind="mergesort").head(k)
    # Number them from 1.
    top["rank"] = np.arange(1, len(top) + 1)
    # Build the sorted frame of distinct users.
    user_frame = pd.DataFrame({"user_id": pd.unique(pd.Series(list(users), dtype="object"))})
    user_frame = user_frame.sort_values("user_id", kind="mergesort")
    # Give every user the same ranked categories.
    result = user_frame.merge(top[["category", "rank"]], how="cross")
    return result[CATEGORY_RANK_COLUMNS].reset_index(drop=True)


# ---------------------------------------------------------------------------
# item-CF flavoured explore (secondary variant)
# ---------------------------------------------------------------------------
def explore_list_cf(
    history: pd.DataFrame, cf: pd.DataFrame, k: int, gamma_c: float, alpha: float, level: str = "L2"
) -> pd.DataFrame:
    """Top-``k`` explore list per user of ``history`` (source = "explore_cf") using item-CF scores inside each
    ``level`` category; ``gamma_c`` and ``alpha`` parametrise category_profile.

    s(u, i) = theta_u(c) * cf(u, i) / sum_{j in c, cf(u, j) > 0} cf(u, j), over
    unbought items with cf(u, i) > 0 (``cf`` is the score frame of
    nbr.itemcf.cf_scores, which already excludes I(H_u)). The normalisation
    runs over the user's positive-cf items of the category, so unlike
    explore_list this variant has no alpha = inf limit that coincides with a
    comparator list. Ties: score desc, pop desc, item_id asc.
    """
    # Validate the parameters.
    _check_params(k, gamma_c, alpha)
    # Build the item catalogue and the category profile.
    catalog = _catalog(history, level)
    theta, _ = category_profile(history, gamma_c, alpha, level)
    # Keep the positive item-CF scores and join category and popularity.
    positive = cf[cf["score"] > 0][PAIR_KEY + ["score"]].rename(columns={"score": "cf"})
    positive = positive.merge(catalog[["item_id", "category", "pop"]], on="item_id", how="inner")
    # Sum the positive scores of each (user, category) pair.
    positive["norm"] = positive.groupby(["user_id", "category"], sort=False)["cf"].transform("sum")
    # Join theta and compute theta times the normalised item-CF score.
    positive = positive.merge(theta, on=["user_id", "category"], how="left")
    positive["score"] = positive["theta"] * positive["cf"] / positive["norm"]
    # Rank the scores into the top-k list with the popularity tie rule.
    pop = catalog.set_index("item_id")["pop"]
    return rank_scores(positive[["user_id", "item_id", "score"]], pop, k, "explore_cf")
