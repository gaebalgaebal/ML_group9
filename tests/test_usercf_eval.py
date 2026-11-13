# *------------------- 2026 Fall Semester Machine Learning Term-project: Team 9 --------------------*
# Code First written at 2026-11-05
# Code Final Edited at 2026-11-13
# Editor: Yejun Kim

'''
Purpose : Tests of nbr.usercf_eval, the stand-alone runner of user-based collaborative
          filtering (E2): fit/eval separation, comparison direction, CLI guards and runner
          integration.
Inputs  : Small purchase histories and targets built in this file, a configuration dict rooted
          at pytest's tmp_path and the monkeypatch fixture.
Outputs : pytest pass/fail results; files are written only below tmp_path.
Usage   :
    python -m pytest tests/test_usercf_eval.py
'''

# ---------------------------------------------------------------------------
# imports
# ---------------------------------------------------------------------------
import copy
import json

import pandas as pd
import pytest

import nbr.usercf_eval as runner
from nbr.baselines import item_popularity
from nbr.data import split_target
from nbr.io import write_json
from nbr.metrics import per_user_recall
from nbr.usercf import ucf_list, ucf_scores, user_neighbors, user_vectors


# ---------------------------------------------------------------------------
# fixtures and helpers
# ---------------------------------------------------------------------------
@pytest.fixture
def frames():
    '''Return a (history, target) pair of four users in two groups with disjoint items.'''
    # Give users a and b items i00-i09 and users c and d items i10-i19, one basket each.
    rows = [(u, 1, f"i{i:02d}") for u in ("a", "b", "c", "d")
            for i in ({"a": range(0, 6), "b": range(0, 10),
                       "c": range(10, 16), "d": range(10, 20)}[u])]
    h = pd.DataFrame(rows, columns=["user_id", "basket_seq", "item_id"])
    # Derive the L2 category from the tens digit of the item number.
    h["L2"] = h["item_id"].str[1:].astype(int).floordiv(10).astype(str)
    # Give every user one target item that is not in the user's history.
    t = pd.DataFrame([("a", "i06"), ("b", "i10"), ("c", "i16"), ("d", "i00")],
                     columns=["user_id", "item_id"])
    return h, t


@pytest.fixture
def config(tmp_path):
    '''Return a run configuration of the sample profile with all paths below ``tmp_path``.'''
    return {"profile": "sample", "role": "smoke", "repo_root": tmp_path, "data_dir": tmp_path / "data",
            "usercf": {"k": [2, 1], "weighting": ["log1p", "binary"], "block_size": 2},
            "selection_source": {"tuning_dir": str(tmp_path / "base")},
            "explore": {"level": "L2"}, "bootstrap": {"n": 100, "seed": 42, "levels": [.95, .99375]}}


# ---------------------------------------------------------------------------
# parameter selection on the fit split
# ---------------------------------------------------------------------------
def test_selection_exact_ties():
    # Build a grid in which every (k, weighting) pair has the same score.
    grid = [{"k": k, "weighting": w, "new_recall@10": .1}
            for k in [100, 20, 50] for w in ["log1p", "binary"]]
    # Expect the tie to resolve to the smallest k and the weighting "binary".
    assert runner.select_parameters(grid) == {"k": 20, "weighting": "binary"}
    # Raise one score and expect that pair to be selected.
    grid[0]["new_recall@10"] = .2
    assert runner.select_parameters(grid) == {"k": 100, "weighting": "log1p"}
    # Expect a ValueError for an empty grid.
    with pytest.raises(ValueError):
        runner.select_parameters([])


def test_fit_grid_matches_public_lists_and_is_deterministic(frames, config):
    # Tune on the fixture history and target.
    h, t = frames
    document = runner.tune_history(h, t, config)
    _, new = split_target(h, t)
    # Expect every grid score to equal the new-item recall of the list built by ucf_list.
    for row in document["grid"]:
        lists = ucf_list(h, row["k"], row["weighting"], 10, item_popularity(h))
        assert row["new_recall@10"] == float(per_user_recall(lists, new, 10).mean())
    # Expect the same selections and scores when the history rows are shuffled.
    again = runner.tune_history(h.sample(frac=1, random_state=42), t, config)
    assert document["selections"] == again["selections"]
    assert [r["new_recall@10"] for r in document["grid"]] == [r["new_recall@10"] for r in again["grid"]]


# ---------------------------------------------------------------------------
# comparison direction and decisions
# ---------------------------------------------------------------------------
def test_h8_keys_direction_decisions_and_seed(config):
    # Compare a per-user series with a copy shifted up by 0.1 as the primary hypothesis.
    a = pd.Series([0., .2, .4], index=["a", "b", "c"])
    b = a + .1
    result = runner.compare(a, b, config, "flat_explore", "usercf", "new_recall@10", True)
    # Expect model minus comparator, both interval levels and "improvement observed".
    assert result["n_users"] == 3
    assert result["comparator"] == "flat_explore" and result["model"] == "usercf"
    assert result["diff"] == pytest.approx(.1)
    assert set(result["ci"]) == {"0.95", "0.99375"}
    assert set(result["decision"].values()) == {"improvement observed"}
    # Expect an identical result from a second call with the same seed.
    assert result == runner.compare(a, b, config, "flat_explore", "usercf", "new_recall@10", True)
    # Expect "no difference established" unless both interval bounds lie on one side of zero.
    assert runner._decision([-.1, .1]) == "no difference established"
    assert runner._decision([-.1, 0]) == "no difference established"
    assert runner._decision([0, .1]) == "no difference established"
    assert runner._decision([-.2, -.1]) == "lower"
    # Expect only the 0.95 interval for a comparison that is not the primary one.
    aux = runner.compare(a, b, config, "final", "final_ucf", "ndcg@10")
    assert set(aux["ci"]) == {"0.95"}


# ---------------------------------------------------------------------------
# fit/eval separation
# ---------------------------------------------------------------------------
def _save_selections(config):
    '''Write a fit-split tuning file with base selections to the tuning_dir of ``config``.'''
    # Build the path of the tuning file of the instacart data set.
    profile, dataset = config["profile"], "instacart"
    path = runner.Path(config["selection_source"]["tuning_dir"]) / f"{profile}_{dataset}_tuning.json"
    # Write the document with fixed selections.
    write_json(path, {"profile": profile, "dataset": dataset, "split": "fit",
                     "selections": {"gamma": .8, "gamma_c": .9, "alpha": 10, "r": 5}}, config)


def test_eval_target_only_purchases_do_not_change_model_or_selections(frames, config, monkeypatch):
    # Tune on the fit data and store the tuning document and the base selections.
    h, fit_target = frames
    original = runner.tune_history(h, fit_target, config)
    document = {**original, "profile": "sample", "dataset": "instacart"}
    write_json(runner.output_path(config, "instacart", "tuning"), document, config)
    _save_selections(config)
    # Evaluate once with the data loader replaced by the fixture frames.
    target = fit_target.copy()
    monkeypatch.setattr(runner, "_prepare", lambda *args: (h, target))
    first = runner.evaluate(config, "instacart")
    X, _, _ = user_vectors(h, original["selections"]["weighting"])
    nbr = user_neighbors(X, original["selections"]["k"])
    scores = ucf_scores(h, nbr, X)
    # Add a purchase that occurs only in the eval target and evaluate again.
    target = pd.concat([target, pd.DataFrame([("a", "target_only_purchase")], columns=target.columns)])
    second = runner.evaluate(config, "instacart")
    # Expect unchanged selections, diagnostics, neighbours and scores.
    assert first["selections"] == second["selections"] == original["selections"]
    assert first["diagnostics"] == second["diagnostics"]
    X2, _, items2 = user_vectors(h, original["selections"]["weighting"])
    nbr2 = user_neighbors(X2, original["selections"]["k"])
    assert "target_only_purchase" not in items2
    pd.testing.assert_frame_equal(nbr, nbr2)
    pd.testing.assert_frame_equal(scores, ucf_scores(h, nbr2, X2))
    # Eval labels may change metrics, but cannot enter the fit-only selection API.
    assert first["H8"]["metric"] == "new_recall@10"
    assert len(first["auxiliary"]) == 4
    assert len(first["model_means"]) == 7
    assert first["label"] == "development (medium)"


def test_missing_base_selection_stops_before_data_read(config, monkeypatch):
    # Replace the data loader by a function that fails the test when it is called.
    monkeypatch.setattr(runner, "_prepare", lambda *args: pytest.fail("data must not be read"))
    # Expect a FileNotFoundError for the missing tuning file without any data read.
    with pytest.raises(FileNotFoundError, match="Required tuning"):
        runner.evaluate(config, "instacart")


# ---------------------------------------------------------------------------
# profile, path and provenance guards
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("profile", ["holdout", "final", "../holdout"])
def test_forbidden_profile_before_config_or_data_read(profile, monkeypatch):
    # Replace the config loader by a function that fails the test when it is called.
    monkeypatch.setattr(runner, "load_config", lambda *args: pytest.fail("config must not be read"))
    # Expect a ValueError from the settings function and an exit from the command line.
    with pytest.raises(ValueError):
        runner.settings(profile)
    with pytest.raises(SystemExit):
        runner.main(["--profile", profile, "--dataset", "instacart", "--stage", "all"])


def test_holdout_path_guard_and_selection_provenance(config, monkeypatch):
    # Point the Complete Journey profile of a config copy to "holdout".
    forbidden = copy.deepcopy(config)
    forbidden["complete_journey_profile"] = "holdout"
    # Expect the guard to reject that configuration.
    with pytest.raises(ValueError, match="Holdout"):
        runner._guard(forbidden, "complete_journey")
    # Store base selections and change the split of the stored file to "eval".
    _save_selections(config)
    path = runner.Path(config["selection_source"]["tuning_dir"]) / "sample_instacart_tuning.json"
    doc = json.loads(path.read_text(encoding="utf-8"))
    doc["split"] = "eval"
    path.write_text(json.dumps(doc), encoding="utf-8")
    # Expect a provenance error when the selections are read from that file.
    with pytest.raises(ValueError, match="provenance"):
        runner._read_selections(path, ("gamma",), "sample", "instacart")


# ---------------------------------------------------------------------------
# history-length breakdown, fit labels and diagnostics
# ---------------------------------------------------------------------------
def test_history_bins_exact_basket_counts_and_empty_bins(config):
    # Build five users with 4, 6, 10, 20 and 21 baskets plus one repeated row.
    h = pd.DataFrame([(str(n), seq, "a") for n in [4, 6, 10, 20, 21] for seq in range(n)],
                     columns=["user_id", "basket_seq", "item_id"])
    h = pd.concat([h, h.iloc[:1]])
    a = pd.Series(0., index=["4", "6", "10", "20", "21"])
    # Expect one user and a positive difference in each of the five instacart bins.
    rows = runner.history_breakdown(h, a, a + .1, "instacart", config)
    assert [row["bin"] for row in rows] == ["(0,4]", "(4,6]", "(6,10]", "(10,20]", "(20,inf)"]
    assert all(row["n_users"] == 1 and row["diff"] > 0 for row in rows)
    # Expect four complete_journey bins without a difference when no user has a metric value.
    cj = runner.history_breakdown(h, a.iloc[:0], a.iloc[:0], "complete_journey", config)
    assert len(cj) == 4 and all(row["diff"] is None for row in cj)


def test_no_new_fit_labels_is_explicit_error(frames, config):
    # Use the history items themselves as the target, so that no target item is new.
    h, _ = frames
    # Expect a ValueError that reports that there are no users.
    with pytest.raises(ValueError, match="no users"):
        runner.tune_history(h, h[["user_id", "item_id"]], config)


def test_diagnostics_slot_denominators_and_popularity_ranks():
    # Build a 120-item popularity, a two-slot list and two neighbour rows.
    pop = pd.Series({f"i{i:03d}": 200 - i for i in range(120)}, name="pop")
    lists = pd.DataFrame([("a", 1, "i000", "usercf"), ("a", 2, "i110", "popularity")],
                         columns=["user_id", "rank", "item_id", "source"])
    neighbors = pd.DataFrame([("a", "b", .2), ("b", "a", .8)],
                             columns=["user_id", "neighbor_id", "sim"])
    result = runner.diagnostics({name: lists for name in ("usercf", "flat_explore", "itemcf_flat")},
                                pop, neighbors)
    # Expect per-slot shares, the median popularity rank and the similarity quantiles.
    assert result["popularity"]["usercf"] == {
        "n_slots": 2, "top100_fraction": .5, "median_popularity_rank": 56.}
    assert result["popularity_fill_fraction"] == .5
    assert result["neighbor_similarity"]["median"] == pytest.approx(.5)
    assert result["neighbor_similarity"]["p10"] == pytest.approx(.26)
    assert result["neighbor_similarity"]["p90"] == pytest.approx(.74)


# ---------------------------------------------------------------------------
# command-line run
# ---------------------------------------------------------------------------
def test_cli_all_writes_fit_then_eval_with_fixed_selection(frames, config, monkeypatch):
    # Store base selections and replace the settings and the data loader by the fixtures.
    _save_selections(config)
    monkeypatch.setattr(runner, "settings", lambda profile, tuning_dir=None: config)
    monkeypatch.setattr(runner, "_prepare", lambda *args: frames)
    # Run the command line with the stage "all".
    runner.main(["--profile", "sample", "--dataset", "instacart", "--stage", "all"])
    # Expect a fit tuning file and an eval file that hold the same selections.
    tuning = json.loads(runner.output_path(config, "instacart", "tuning").read_text(encoding="utf-8"))
    evaluation = json.loads(runner.output_path(config, "instacart", "eval").read_text(encoding="utf-8"))
    assert tuning["split"] == "fit" and evaluation["split"] == "eval"
    assert tuning["selections"] == evaluation["selections"]
    assert "git_commit" in tuning["meta"]
