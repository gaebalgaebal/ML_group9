import numpy as np
import pandas as pd
import pytest
from scipy import stats

from ml_group9.hstar import (
    concentration,
    extension_outcomes,
    gini,
    placebo_flag,
    popularity_deciles,
    purchase_data,
    relative_risk,
    stability,
    taste_attributes,
    top_share,
    verdict,
    weighted_spearman,
)


def test_taste_flags_use_word_boundaries_and_ignore_case():
    items = pd.DataFrame({"item_id": ["1", "2", "3", "4"], "product_name": ["Dark CHOCOLATE Bar", "Chocolatey Puffs", "vanilla bean", None]})
    flagged = taste_attributes(items, ["chocolate", "vanilla"])
    assert flagged["attr_chocolate"].tolist() == [1, 0, 0, 0]
    assert flagged["attr_vanilla"].tolist() == [0, 0, 1, 0]


def test_spearman_matches_scipy_and_weights_equal_duplication():
    rng = np.random.default_rng(0)
    x = rng.integers(0, 5, 40).astype(float)
    y = x + rng.integers(0, 3, 40)
    assert weighted_spearman(x, y) == pytest.approx(stats.spearmanr(x, y).statistic)
    weights = rng.integers(0, 3, 40)
    repeated = np.repeat(np.arange(40), weights)
    assert weighted_spearman(x, y, weights) == pytest.approx(stats.spearmanr(x[repeated], y[repeated]).statistic)


def test_gini_and_top_share_by_hand():
    assert gini(np.array([0, 0, 1, 3])) == pytest.approx(10 / 16)
    assert gini(np.array([2, 2, 2])) == pytest.approx(0.0)
    assert top_share(np.array([5, 1, 1, 1, 2])) == pytest.approx(0.5)  # top ceil(0.2 * 5) = 1 user
    assert gini(np.zeros(3)) is None


def test_relative_risk_by_hand():
    has = np.array([True, True, False, False, False, False])
    outcome = np.array([True, False, True, False, False, False])
    assert relative_risk(has, outcome) == pytest.approx(0.5 / 0.25)
    assert relative_risk(has, np.array([True, False, False, False, False, False])) is None


COLUMNS = ["user_id", "basket_id", "basket_seq", "item_id"]


def catalogue():
    # Aisle "a": attribute items a1, a2; others o1..o4. Aisle "b": attribute item b1, others p1, p2.
    return pd.DataFrame(
        {
            "item_id": ["a1", "a2", "o1", "o2", "o3", "o4", "b1", "p1", "p2"],
            "fine_category": ["a"] * 6 + ["b"] * 3,
            "product_name": ["x"] * 9,
        }
    )


def history():
    rows = []
    # u1: 4 baskets; early half mostly attribute, later half mixed.
    for seq, basket in enumerate([["a1", "o1"], ["a1", "a2"], ["a1", "o2"], ["o1", "o3"]], start=1):
        rows += [("u1", f"u1b{seq}", seq, item) for item in basket]
    for seq, basket in enumerate([["o1"], ["o2", "o4"], ["o1"]], start=1):
        rows += [("u2", f"u2b{seq}", seq, item) for item in basket]
    rows += [("u3", "u3b1", 1, "p1"), ("u3", "u3b2", 2, "b1")]
    return pd.DataFrame(rows, columns=COLUMNS)


def target(extra=()):
    rows = [("u1", "t1", 9, "a2"), ("u2", "t2", 9, "a1"), ("u3", "t3", 9, "p2")] + list(extra)
    return pd.DataFrame(rows, columns=COLUMNS)


def flags(data):
    flag = np.isin(data.items, ["a1", "a2", "b1"])
    in_c = np.isin(data.item_aisle, ["a", "b"])
    return flag, in_c


def test_halves_split_each_users_baskets_in_time_order():
    data = purchase_data(catalogue(), history(), target())
    row = data.users.get_loc("u1")
    first = dict(zip(data.items[data.first[row].indices], data.first[row].data))
    second = dict(zip(data.items[data.second[row].indices], data.second[row].data))
    assert first == {"a1": 2.0, "o1": 1.0, "a2": 1.0}
    assert second == {"a1": 1.0, "o2": 1.0, "o1": 1.0, "o3": 1.0}
    # u2 has three baskets: floor(3 / 2) = 1 early basket.
    row = data.users.get_loc("u2")
    assert data.first[row].sum() == 1 and data.second[row].sum() == 3


def test_stability_pi_by_hand():
    data = purchase_data(catalogue(), history(), target())
    flag, in_c = flags(data)
    result = stability(data, flag, in_c, smoothing=2.0, min_n_c=1)
    # n_a: u1 4 (a1 x3, a2), u3 1 (b1); n_C: u1 8, u2 4, u3 2.
    prior = 5 / 14
    row = list(data.users).index("u1")
    position = list(result.rows).index(row)
    assert result.first[position] == pytest.approx((3 + 2 * prior) / (4 + 2))
    assert result.second[position] == pytest.approx((1 + 2 * prior) / (4 + 2))


def test_extension_outcomes_by_hand():
    data = purchase_data(catalogue(), history(), target())
    flag, in_c = flags(data)
    outcomes = extension_outcomes(data, flag, in_c)
    # Users with n_C >= 1: u1 (buyer, a2 already bought: no new attribute item), u2 (non-buyer, new a1),
    # u3 (buyer, new p2 is a non-attribute item).
    assert outcomes["has"].tolist() == [True, False, True]
    assert outcomes["attribute"].tolist() == [False, True, False]
    assert outcomes["control"].tolist() == [False, False, True]


def test_concentration_by_hand():
    data = purchase_data(catalogue(), history(), target())
    flag, in_c = flags(data)
    result = concentration(data, flag, in_c)
    # n_a per user: u1 4, u2 0, u3 1; top ceil(0.6) = 1 user.
    assert result["top20_share"] == pytest.approx(4 / 5)
    assert result["gini"] == pytest.approx(gini(np.array([4, 0, 1])))
    assert result["category_top20_share"] == pytest.approx(8 / 14)


def test_m1_and_m3_ignore_target_only_purchases():
    base = purchase_data(catalogue(), history(), target())
    leaked = purchase_data(catalogue(), history(), target([("u2", "t2", 9, "a2"), ("u3", "t3", 9, "b1"), ("u1", "t1", 9, "o4")]))
    flag, in_c = flags(base)
    a, b = stability(base, flag, in_c, 2.0, 1), stability(leaked, flag, in_c, 2.0, 1)
    assert np.array_equal(a.rows, b.rows) and np.array_equal(a.first, b.first) and np.array_equal(a.second, b.second)
    assert concentration(base, flag, in_c) == concentration(leaked, flag, in_c)


def placebo_data():
    items = pd.DataFrame(
        {"item_id": [f"i{n}" for n in range(40)], "fine_category": ["a"] * 30 + ["b"] * 10, "product_name": "x"}
    )
    rng = np.random.default_rng(5)
    rows = []
    for n in range(40):
        for basket in range(int(rng.integers(0, 12))):
            rows.append(("u", f"b{n}_{basket}", basket, f"i{n}"))
    history_frame = pd.DataFrame(rows, columns=COLUMNS)
    data = purchase_data(items, history_frame, history_frame.iloc[:0])
    flag = np.zeros(40, dtype=bool)
    flag[[0, 3, 7, 11, 19, 25, 31, 33]] = True
    return data, flag


def test_placebo_matches_aisle_decile_and_count_and_is_reproducible():
    data, flag = placebo_data()
    fake, stats_ = placebo_flag(data, flag, {"a", "b"}, np.random.default_rng([42, 0, 0]))
    again, _ = placebo_flag(data, flag, {"a", "b"}, np.random.default_rng([42, 0, 0]))
    other, _ = placebo_flag(data, flag, {"a", "b"}, np.random.default_rng([42, 0, 1]))
    assert np.array_equal(fake, again) and not np.array_equal(fake, other)
    assert not (fake & flag).any()
    for aisle in ["a", "b"]:
        members = data.item_aisle == aisle
        assert (fake & members).sum() == (flag & members).sum()
        decile = np.full(len(flag), -1)
        decile[members] = popularity_deciles(data.popularity[members])
        if stats_["nearby_decile"] == 0:
            for d in range(10):
                assert (fake & members & (decile == d)).sum() == (flag & members & (decile == d)).sum()
    assert stats_["items"] == flag.sum() and stats_["unmatched"] == 0


def test_verdict_rule():
    diffs = {"chocolate": {"ci_low": 0.01}, "vanilla": {"ci_low": 0.02}}
    assert verdict(0.7, 0.6, diffs)["verdict"] == "supported"
    assert verdict(0.7, 0.8, diffs)["verdict"] == "partially supported"
    assert verdict(0.7, 0.6, {**diffs, "strawberry": {"ci_low": -0.01}})["verdict"] == "partially supported"
    assert verdict(0.5, 0.6, {"chocolate": {"ci_low": -0.1}})["verdict"] == "not supported"
    assert verdict(None, 0.6, diffs)["verdict"] == "undetermined"
