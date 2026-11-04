import numpy as np
import pandas as pd
import pytest

from ml_group9.hstar import Stability, counts, placebo_flag, purchase_data, stability, weighted_spearman
from ml_group9.hstar_followup import (
    RankedHalves,
    _bootstrap,
    aisle_extraction,
    delta,
    item_concentration,
    item_purchases,
    level_ci,
    new_item_data,
    new_item_halves,
    s1_aisles,
    without_top_item,
)


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
    return np.isin(data.items, ["a1", "a2", "b1"]), np.isin(data.item_aisle, ["a", "b"])


def row_items(matrix, data, user):
    row = matrix[data.users.get_loc(user)]
    return dict(zip(data.items[row.indices], row.data))


def test_delta_by_hand():
    assert delta(0.7, [0.5, 0.6, None]) == pytest.approx(0.7 - 0.55)
    assert delta(None, [0.5]) is None and delta(0.7, [None]) is None


def test_ranked_halves_match_weighted_spearman():
    rng = np.random.default_rng(1)
    halves = Stability(np.arange(30), rng.integers(0, 6, 30) / 5, rng.integers(0, 6, 30) / 5)
    weights = rng.integers(0, 3, 40).astype(float)
    assert RankedHalves(halves).rho(weights) == pytest.approx(weighted_spearman(halves.first, halves.second, weights[:30]))


def bootstrap_inputs(seed=0):
    rng = np.random.default_rng(seed)
    rows = np.arange(0, 40, 2)
    attribute = Stability(rows, rng.random(20), rng.random(20))
    other = Stability(rows, rng.random(20), rng.random(20))
    return attribute, other


class Users:
    users = pd.Index([f"u{n}" for n in range(40)])


def run_bootstrap(attribute, other, placebo, seed=42, resamples=30):
    halves = {"h": attribute, "t": other}
    placebos = {"h": placebo, "t": [other]}
    return _bootstrap(Users(), ["h", "t"], ["h"], ["t"], halves, placebos, halves, placebos, resamples, seed, 1e9)


def test_bootstrap_uses_the_same_resampled_users_for_attribute_and_placebo():
    attribute, other = bootstrap_inputs()
    # A placebo identical to the attribute gives Delta* = 0 only if both see the same weights.
    result = run_bootstrap(attribute, other, [attribute, attribute])
    assert len(result["delta"]["h"]) == 30
    assert np.allclose(result["delta"]["h"], 0.0, atol=1e-12)
    # With a different placebo, Delta* equals the attribute rho minus the placebo rho under the same draw.
    result = run_bootstrap(attribute, other, [other])
    rng = np.random.default_rng(42)
    for value in result["delta"]["h"][:5]:
        weights = np.bincount(rng.integers(0, 40, 40), minlength=40).astype(float)[attribute.rows]
        expected = weighted_spearman(attribute.first, attribute.second, weights) - weighted_spearman(other.first, other.second, weights)
        assert value == pytest.approx(expected)


def test_bootstrap_is_reproducible_by_seed():
    attribute, other = bootstrap_inputs()
    a = run_bootstrap(attribute, other, [other])
    b = run_bootstrap(attribute, other, [other])
    c = run_bootstrap(attribute, other, [other], seed=7)
    assert a["delta"]["h"] == b["delta"]["h"] and a["m1_diff"] == b["m1_diff"]
    assert a["delta"]["h"] != c["delta"]["h"]


def test_level_ci_widens_with_level():
    values = list(np.linspace(0, 1, 1001))
    assert level_ci(values, 0.95) == pytest.approx((0.025, 0.975))
    assert level_ci(values, 1 - 0.05 / 4) == pytest.approx((0.00625, 0.99375))


def extraction_data():
    # Aisle "a": 2 attribute + 4 other items; aisle "b": 3 attribute + 2 other items;
    # aisle "c": 2 attribute + 3 other items (one short of 2x).
    ids = [f"a{n}" for n in range(6)] + [f"b{n}" for n in range(5)] + [f"c{n}" for n in range(5)]
    items = pd.DataFrame({"item_id": ids, "fine_category": ["a"] * 6 + ["b"] * 5 + ["c"] * 5, "product_name": "x"})
    rows = [("u", f"k{n}", n, item) for n, item in enumerate(ids)]
    data = purchase_data(items, pd.DataFrame(rows, columns=COLUMNS), pd.DataFrame(columns=COLUMNS))
    flag = np.isin(data.items, ["a0", "a1", "b0", "b1", "b2", "c0", "c1"])
    return data, flag


def test_extraction_ratio_and_unmatched_by_hand():
    data, flag = extraction_data()
    fake, stats = placebo_flag(data, flag, {"a", "b"}, np.random.default_rng(0))
    rows = {row["aisle"]: row for row in aisle_extraction(data, flag, [fake], {"a", "b"})}
    assert rows["a"]["placebo_items_mean"] == 2 and rows["a"]["extraction_ratio"] == pytest.approx(2 / 4)
    assert rows["b"]["placebo_items_mean"] == 2 and rows["b"]["extraction_ratio"] == pytest.approx(1.0)
    assert stats["unmatched"] == 1


def test_s1_keeps_aisles_with_exactly_twice_as_many_other_items():
    data, flag = extraction_data()
    assert s1_aisles(data, flag, {"a", "b", "c"}) == {"a"}


def test_item_concentration_by_hand():
    result = item_concentration(np.array([1, 6, 0, 3]))
    assert result["top1_share"] == pytest.approx(0.6)
    assert result["top5_share"] == pytest.approx(1.0)
    assert result["hhi"] == pytest.approx(0.36 + 0.09 + 0.01)
    assert item_concentration(np.array([2, 1, 1, 1, 1, 1, 1]), k=5)["top5_share"] == pytest.approx(6 / 8)
    assert item_concentration(np.zeros(3))["hhi"] is None


def test_s2_drops_the_top_item_from_numerator_and_denominator():
    data = purchase_data(catalogue(), history(), target())
    flag, in_c = flags(data)
    purchases = item_purchases(data)
    flag2, in_c2, top = without_top_item(flag, in_c, purchases)
    assert data.items[top] == "a1"  # bought in three u1 baskets
    n_a, n_c = counts(data.history, flag, in_c)
    n_a2, n_c2 = counts(data.history, flag2, in_c2)
    row = data.users.get_loc("u1")
    assert (n_a[row], n_c[row]) == (4, 8)
    assert (n_a2[row], n_c2[row]) == (1, 5)
    assert flag[top] and in_c[top]  # inputs are not modified


def test_new_items_exclude_earlier_baskets_across_and_within_halves():
    data = purchase_data(catalogue(), history(), target())
    first, second = new_item_halves(catalogue(), history(), data)
    # u1 first half: b1 {a1, o1}, b2 {a1, a2}; the second a1 is a repeat inside the same half.
    assert row_items(first, data, "u1") == {"a1": 1.0, "o1": 1.0, "a2": 1.0}
    # u1 second half: b3 {a1, o2}, b4 {o1, o3}; a1 and o1 were bought in the first half.
    assert row_items(second, data, "u1") == {"o2": 1.0, "o3": 1.0}
    # u2: b1 {o1} | b2 {o2, o4}, b3 {o1}; o1 in b3 repeats b1.
    assert row_items(first, data, "u2") == {"o1": 1.0}
    assert row_items(second, data, "u2") == {"o2": 1.0, "o4": 1.0}


def threshold_data(n_first):
    # u: 2 baskets; first basket holds n_first new items of aisle "a", second basket 5 different ones.
    items = pd.DataFrame({"item_id": [f"i{n}" for n in range(12)], "fine_category": "a", "product_name": "x"})
    rows = [("u", "b1", 1, f"i{n}") for n in range(n_first)] + [("u", "b2", 2, f"i{n}") for n in range(6, 11)]
    rows += [("v", "c1", 1, "i0"), ("v", "c2", 2, "i1"), ("w", "d1", 1, "i2"), ("w", "d2", 2, "i3")]
    frame = pd.DataFrame(rows, columns=COLUMNS)
    data = purchase_data(items, frame, frame.iloc[:0])
    new = new_item_data(data, *new_item_halves(items, frame, data))
    flag = np.isin(data.items, ["i0", "i6"])
    return data, new, flag, np.ones(len(data.items), dtype=bool)


def test_new_item_threshold_boundary():
    data, new, flag, in_c = threshold_data(5)
    assert stability(new, flag, in_c, 10.0, 5).rows.tolist() == [data.users.get_loc("u")]
    data, new, flag, in_c = threshold_data(4)
    assert len(stability(new, flag, in_c, 10.0, 5).rows) == 0


def test_history_only_results_ignore_target_only_purchases():
    extra = [("u2", "t2", 9, "a2"), ("u3", "t3", 9, "b1"), ("u1", "t1", 9, "o4"), ("u1", "t1", 9, "a1")]
    base, leaked = purchase_data(catalogue(), history(), target()), purchase_data(catalogue(), history(), target(extra))
    flag, in_c = flags(base)
    results = []
    for data in [base, leaked]:
        new = new_item_data(data, *new_item_halves(catalogue(), history(), data))
        fake, _ = placebo_flag(data, flag, {"a", "b"}, np.random.default_rng([42, 0, 0]))
        m1, placebo = stability(data, flag, in_c, 2.0, 1), stability(data, fake, in_c, 2.0, 1)
        m1_new, placebo_new = stability(new, flag, in_c, 2.0, 1), stability(new, fake, in_c, 2.0, 1)
        results.append(
            {
                "m1": (m1.rows.tolist(), m1.first.tolist(), m1.second.tolist()),
                "m1_new": (m1_new.rows.tolist(), m1_new.first.tolist(), m1_new.second.tolist()),
                "delta": delta(m1.rho(), [placebo.rho()]),
                "delta_new": delta(m1_new.rho(), [placebo_new.rho()]),
                "concentration": item_concentration(item_purchases(data)[flag & in_c]),
                "fake": fake.tolist(),
            }
        )
    assert results[0] == results[1]
