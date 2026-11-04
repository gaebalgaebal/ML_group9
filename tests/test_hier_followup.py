import itertools

import numpy as np
import pandas as pd
import pytest

from ml_group9.hier_followup import (
    DATASETS,
    WithinAUC,
    attribute_analysis,
    bootstrap_draws,
    difference_statistic,
    dominant_type,
    fallback_pair_share,
    fit_offset_logit,
    is_fallback,
    item_labels,
    present_mass,
    scored_rows,
    type_patterns,
    within_summary,
)
from ml_group9.hierarchical import Params

from test_hierarchical import items, state, target


PARAMS = Params(alpha=3.0, mu=1.0, lambda_type=2.0, k=0)


def two_types():
    # Type 0: positives 0.8, 0.5; negatives 0.5, 0.2 -> (1 + 1 + 0.5 + 1) / 4.
    # Type 1: positive 0.1; negatives 0.9, 0.1 -> (0 + 0.5) / 2. Type 2 has positives only.
    types = np.array([0, 0, 0, 0, 1, 1, 1, 2, 2])
    scores = np.array([0.8, 0.5, 0.5, 0.2, 0.1, 0.9, 0.1, 0.3, 0.7])
    labels = np.array([1, 1, 0, 0, 1, 0, 0, 1, 1])
    return types, scores, labels


def test_within_type_auc_by_hand_with_ties():
    estimator = WithinAUC(*two_types())
    assert estimator.value() == pytest.approx((4 * 0.875 + 2 * 0.25) / 6)
    table = estimator.per_type()
    assert table["type"].tolist() == [0, 1]  # the single-class type 2 does not contribute
    assert table["auc"].tolist() == pytest.approx([0.875, 0.25])
    assert table["pairs"].tolist() == [4, 2]


def test_within_type_auc_equals_pooled_auc_over_same_type_pairs():
    rng = np.random.default_rng(3)
    types, scores, labels = rng.integers(0, 4, 60), rng.integers(0, 5, 60) / 4, rng.integers(0, 2, 60)
    weights = rng.integers(0, 3, 60).astype(float)
    concordant = pairs = 0.0
    for i, j in itertools.product(range(60), range(60)):
        if types[i] == types[j] and labels[i] == 1 and labels[j] == 0:
            w = weights[i] * weights[j]
            pairs += w
            concordant += w * (1.0 if scores[i] > scores[j] else 0.5 if scores[i] == scores[j] else 0.0)
    assert WithinAUC(types, scores, labels).value(weights) == pytest.approx(concordant / pairs)


def test_within_summary_drops_rows_without_score_and_single_class_types():
    types, scores, labels = two_types()
    frame = pd.DataFrame({"type": types, "label": labels, "user": np.arange(9), "pi_raw": scores})
    frame.loc[0, "pi_raw"] = np.nan  # a type-0 positive without n_C
    summary = within_summary(frame, "pi_raw")
    assert summary["rows_without_score"] == 1 and summary["rows_in_single_class_types"] == 2
    assert summary["types"] == 2 and summary["positive_rows"] == 2 and summary["negative_rows"] == 4
    assert summary["pairs"] == 1 * 2 + 1 * 2
    assert summary["auc_within"] == pytest.approx((0.5 + 1.0 + 0.5) / 4)


def test_pi_raw_is_missing_when_the_user_has_no_purchase_in_c_a():
    # u3 bought only bread (outside fat_reduced's C_a); u4 bought m4 once.
    extra = [("u3", "t3", 3, "m2"), ("u4", "t4", 3, "m1")]
    hier_state = state(target_frame=target(extra))
    frame = scored_rows(hier_state, PARAMS)
    users = hier_state.interactions.users[frame["user"]]
    raw = dict(zip(users, frame["pi_raw"]))
    assert np.isnan(raw["u3"]) and raw["u4"] == 0.0
    assert raw["u1"] == pytest.approx(3 / 3)  # u1: m1 twice and m2 in milk, all fat_reduced
    assert frame["n_type"][users == "u3"].tolist() == [0.0]


def test_dominant_type_needs_more_than_half_of_the_pairs():
    assert dominant_type(pd.DataFrame({"type": [0, 1], "pairs": [5.0, 5.0]})) is None
    top = dominant_type(pd.DataFrame({"type": [0, 1], "pairs": [4.0, 6.0]}))
    assert top is not None and top["type"] == 1


def test_offset_logistic_recovers_known_beta():
    rng = np.random.default_rng(0)
    offset, x = rng.normal(0, 1, 40000), rng.normal(0, 1, 40000)
    labels = rng.random(40000) < 1 / (1 + np.exp(-(-0.3 + offset + 0.8 * x)))
    beta0, beta = fit_offset_logit(offset, x, labels.astype(float))
    assert beta0 == pytest.approx(-0.3, abs=0.05) and beta == pytest.approx(0.8, abs=0.05)


def test_missing_mass_by_hand():
    pi = np.array([[0.2, 0.3]])
    # Present patterns 0 (none), 1 (first) and 3 (both); missing pattern 2 = (0, 1) with mass 0.8 * 0.3.
    assert 1.0 - present_mass(pi, {0, 1, 3})[0] == pytest.approx(0.8 * 0.3)
    assert present_mass(pi, {0, 1, 2, 3})[0] == pytest.approx(1.0)
    hierarchy = state().hierarchies["full"]
    patterns = type_patterns(hierarchy)
    milk = int(np.flatnonzero(hierarchy.type_labels == "milk|i_milk")[0])
    assert patterns[milk][1] == {0, 1} and len(patterns) == 1


def test_fallback_types_are_excluded_by_hand():
    types, scores, labels = two_types()
    labels_by_type = np.array(["milk|i_abc", "soup|aisle:soup", "milk|i_def"], dtype=object)
    fallback = np.array([is_fallback(label) for label in labels_by_type])
    assert fallback.tolist() == [False, True, False]
    table = WithinAUC(types, scores, labels).per_type()
    assert fallback_pair_share(table, fallback) == pytest.approx(2 / 6)
    mapped = ~fallback[types]
    # Only type 0 is mapped and has both classes.
    assert WithinAUC(types[mapped], scores[mapped], labels[mapped]).value() == pytest.approx(0.875)


def test_paired_difference_is_reproducible_by_seed():
    types, scores, labels = two_types()
    first, second = WithinAUC(types, scores, labels), WithinAUC(types, scores[::-1].copy(), labels)
    users = np.array([0, 0, 1, 2, 3, 3, 4, 5, 5])
    statistic = {"diff": difference_statistic(first, second), "a": first.value, "b": second.value}
    a, b = bootstrap_draws(users, statistic, 30, 42), bootstrap_draws(users, statistic, 30, 42)
    assert a == b and a != bootstrap_draws(users, statistic, 30, 7)
    # Same weights per resample: each draw of the difference equals the difference of the draws.
    assert len(a["diff"]) == len(a["a"]) == len(a["b"])
    assert np.allclose(a["diff"], np.array(a["a"]) - np.array(a["b"]))


def test_dataset_option_selects_source_output_and_blocks():
    instacart, cj = DATASETS["instacart"], DATASETS["complete_journey"]
    assert (instacart.reference_source, instacart.output, instacart.mapped_types, instacart.mapping_labels) == (
        "instacart", "hier_followup", True, True,
    )
    assert (cj.reference_source, cj.output, cj.mapped_types, cj.mapping_labels) == (
        "refit", "hier_followup_complete_journey", False, False,
    )


def test_item_labels_fall_back_to_item_ids_without_product_names():
    hier_state = state()
    labels = item_labels(items(), hier_state)  # the synthetic catalogue has no product_name column
    assert labels.tolist() == hier_state.interactions.items.astype(str).tolist()
    named = items().assign(product_name=lambda f: "name " + f["item_id"])
    assert item_labels(named, hier_state)[0] == "name " + str(hier_state.interactions.items[0])


def test_optional_blocks_leave_the_other_outputs_unchanged():
    extra = [("u1", "t1", 3, "m4"), ("u3", "t3", 3, "m2"), ("u4", "t4", 3, "m1"), ("u2", "t2", 3, "m2")]
    hier_state = state(target_frame=target(extra))
    rows = scored_rows(hier_state, PARAMS)
    hierarchy = hier_state.hierarchies["full"]
    names = item_labels(items(), hier_state)
    keys = np.full(hier_state.n_items, "", dtype=object)
    base = attribute_analysis(rows, hierarchy, names, keys, 40, 42)
    full = attribute_analysis(rows, hierarchy, names, keys, 40, 42, mapped_types=True, paired_difference=True)
    cj = attribute_analysis(rows, hierarchy, names, keys, 40, 42, paired_difference=True)
    assert set(full) - set(base) == {"within_type_auc_mapped", "fallback_pair_share", "diff_a_minus_b"}
    assert set(cj) - set(base) == {"diff_a_minus_b"}
    for key in base:
        assert full[key] == base[key] and cj[key] == base[key]
    assert cj["diff_a_minus_b"] == full["diff_a_minus_b"]


def test_bootstrap_is_reproducible_by_seed():
    types, scores, labels = two_types()
    estimator = WithinAUC(types, scores, labels)
    users = np.array([0, 0, 1, 2, 3, 3, 4, 5, 5])
    statistic = {"auc": estimator.value}
    a, b = bootstrap_draws(users, statistic, 30, 42), bootstrap_draws(users, statistic, 30, 42)
    assert a == b and a != bootstrap_draws(users, statistic, 30, 7)


def test_scores_ignore_target_only_purchases():
    base = scored_rows(state(), PARAMS)
    leaked = scored_rows(state(target_frame=target([("u1", "t1", 3, "m4"), ("u4", "t4", 3, "c2")])), PARAMS)
    key = ["user", "item"]
    merged = base.merge(leaked, on=key, suffixes=("", "_leaked"))
    assert len(merged) == len(base)
    for column in ["pi_user_aisle", "pi_user_type", "pi_population_type", "pi_raw", "n_type"]:
        assert np.allclose(merged[column], merged[f"{column}_leaked"], equal_nan=True)
