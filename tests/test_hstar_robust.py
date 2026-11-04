import numpy as np
import pandas as pd
import pytest

from ml_group9.hstar import Stability, counts, placebo_flag, purchase_data, stability
from ml_group9.hstar_robust import (
    EXCLUDED_AISLE,
    Family,
    bootstrap,
    buyer_flags,
    cell_counts,
    log_odds_ratio,
    overlap_items,
    persistence,
    raw_halves,
    within_halves,
    without_items,
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
    # u4: no attribute purchase in either half, with a different n_C from u2.
    for seq, basket in enumerate([["o1", "o2"], ["o3", "o4"]], start=1):
        rows += [("u4", f"u4b{seq}", seq, item) for item in basket]
    return pd.DataFrame(rows, columns=COLUMNS)


def target(extra=()):
    rows = [("u1", "t1", 9, "a2"), ("u2", "t2", 9, "a1"), ("u3", "t3", 9, "p2")] + list(extra)
    return pd.DataFrame(rows, columns=COLUMNS)


def flags(data):
    return np.isin(data.items, ["a1", "a2", "b1"]), np.isin(data.item_aisle, ["a", "b"])


def by_user(halves, data):
    return {data.users[row]: (first, second) for row, first, second in zip(halves.rows, halves.first, halves.second)}


def test_raw_pi_by_hand():
    data = purchase_data(catalogue(), history(), target())
    flag, in_c = flags(data)
    values = by_user(raw_halves(data, flag, in_c, 1), data)
    # u1: first {a1, o1, a1, a2} -> 3 / 4, second {a1, o2, o1, o3} -> 1 / 4; u3: 0 / 1 then 1 / 1.
    assert values == {"u1": (0.75, 0.25), "u2": (0.0, 0.0), "u3": (0.0, 1.0), "u4": (0.0, 0.0)}


def test_users_without_attribute_purchases_are_tied_only_without_smoothing():
    data = purchase_data(catalogue(), history(), target())
    flag, in_c = flags(data)
    raw = by_user(raw_halves(data, flag, in_c, 1), data)
    smoothed = by_user(stability(data, flag, in_c, 10.0, 1), data)
    assert raw["u2"] == raw["u4"]
    assert smoothed["u2"] != smoothed["u4"]  # n_C 1 / 3 for u2 vs 2 / 2 for u4


def test_mid_rank_spearman_by_hand():
    # x ranks [1.5, 1.5, 3, 4], y ranks [1.5, 1.5, 4, 3]: cov 3.5, variances 4.5.
    halves = Stability(np.arange(4), np.array([0.0, 0.0, 0.5, 1.0]), np.array([0.0, 0.0, 1.0, 0.5]))
    assert halves.rho() == pytest.approx(3.5 / 4.5)


def test_buyer_flags_and_odds_ratio_by_hand():
    data = purchase_data(catalogue(), history(), target())
    flag, in_c = flags(data)
    raw = raw_halves(data, flag, in_c, 1)
    b1, b2 = buyer_flags(data, flag, in_c, raw.rows)
    assert dict(zip(data.users[raw.rows], zip(b1, b2))) == {"u1": (True, True), "u2": (False, False), "u3": (False, True), "u4": (False, False)}
    b1 = np.array([1, 1, 1, 1, 0, 0, 0, 0], dtype=bool)
    b2 = np.array([1, 1, 1, 0, 1, 0, 0, 0], dtype=bool)
    assert cell_counts(b1, b2).tolist() == [[3, 1], [1, 3]]
    result = persistence(b1, b2)
    assert result["p_b2_given_b1"] == pytest.approx(3 / 4) and result["p_b2_given_not_b1"] == pytest.approx(1 / 4)
    assert result["odds_ratio"] == pytest.approx((3 / 1) / (1 / 3))
    assert log_odds_ratio(cell_counts(b1, b1)) is None  # empty off-diagonal cells


def test_within_keeps_only_users_buying_in_both_halves():
    data = purchase_data(catalogue(), history(), target())
    flag, in_c = flags(data)
    raw = raw_halves(data, flag, in_c, 1)
    within = within_halves(raw, *buyer_flags(data, flag, in_c, raw.rows))
    assert by_user(within, data) == {"u1": (0.75, 0.25)}


def test_overlap_items_leave_numerator_denominator_and_placebo_pool():
    data = purchase_data(catalogue(), history(), target())
    health = {"h": np.isin(data.items, ["a1", "a2"])}
    taste = {"t": np.isin(data.items, ["a1", "o3"])}
    overlap = overlap_items({**health, **taste}, ["h"], ["t"])
    assert data.items[overlap].tolist() == ["a1"]
    reduced = without_items(data, overlap)
    assert reduced.item_aisle[overlap].tolist() == [EXCLUDED_AISLE]
    flag = health["h"] & ~overlap
    in_c = np.isin(reduced.item_aisle, ["a"])
    n_a, n_c = counts(reduced.history, flag, in_c)
    row = data.users.get_loc("u1")
    # u1 bought a1 three times: without it, n_a = 1 (a2) and n_C = 8 - 3 = 5.
    assert (n_a[row], n_c[row]) == (1, 5)
    for rep in range(30):
        fake, _ = placebo_flag(reduced, flag, {"a"}, np.random.default_rng([42, 0, rep]))
        assert not fake[overlap].any() and fake.sum() == 1
    # Baskets and halves are untouched.
    assert (reduced.first != data.first).nnz == 0 and (reduced.second != data.second).nnz == 0


def bootstrap_inputs():
    rng = np.random.default_rng(0)
    rows = np.arange(0, 40, 2)
    attribute = Stability(rows, rng.random(20), rng.random(20))
    other = Stability(rows, rng.random(20), rng.random(20))
    b1, b2 = rng.random(20) < 0.5, rng.random(20) < 0.5
    return attribute, other, (rows, b1, b2)


def run_bootstrap(placebo, buyer_placebo, seed=42):
    attribute, other, buyers = bootstrap_inputs()
    families = {"f": {"h": Family(attribute, placebo), "t": Family(other, [other])}}
    return bootstrap(40, families, {"h": (buyers, buyer_placebo)}, [("h", "t")], 25, seed)


def test_bootstrap_pairs_attribute_and_placebos_and_is_reproducible():
    attribute, other, buyers = bootstrap_inputs()
    same = run_bootstrap([attribute], [buyers])
    assert np.allclose(same["delta"]["f"]["h"], 0.0, atol=1e-12)
    assert np.allclose(same["log_or_difference"]["h"], 0.0, atol=1e-12)
    a, b = run_bootstrap([other], [buyers]), run_bootstrap([other], [buyers])
    c = run_bootstrap([other], [buyers], seed=7)
    assert a["delta"] == b["delta"] and a["differences"] == b["differences"]
    assert a["delta"]["f"]["h"] != c["delta"]["f"]["h"]


def test_results_ignore_target_only_purchases():
    extra = [("u2", "t2", 9, "a2"), ("u4", "t4", 9, "b1"), ("u1", "t1", 9, "o4")]
    base, leaked = purchase_data(catalogue(), history(), target()), purchase_data(catalogue(), history(), target(extra))
    results = []
    for data in [base, leaked]:
        flag, in_c = flags(data)
        raw = raw_halves(data, flag, in_c, 1)
        b1, b2 = buyer_flags(data, flag, in_c, raw.rows)
        overlap = np.isin(data.items, ["a1"])
        reduced = without_items(data, overlap)
        r4 = stability(reduced, flag & ~overlap, np.isin(reduced.item_aisle, ["a", "b"]), 10.0, 1)
        results.append(
            {
                "raw": by_user(raw, data),
                "persistence": persistence(b1, b2),
                "within": by_user(within_halves(raw, b1, b2), data),
                "r4": by_user(r4, data),
            }
        )
    assert results[0] == results[1]
