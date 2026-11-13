# *------------------- 2026 Fall Semester Machine Learning Term-project: Team 9 --------------------*
# Code First written at 2026-11-03
# Code Final Edited at 2026-11-13
# Editor: Yejun Kim, Seongwoo Yang

'''
Purpose : Tests for nbr.data and the check command on synthetic parquet files.
Inputs  : Synthetic history, target and items frames built in this file and written as
          parquet files under the pytest tmp_path directory.
Outputs : pytest pass/fail results; parquet files are written only under tmp_path.
Usage   :
    python -m pytest tests/test_data.py
'''

# ---------------------------------------------------------------------------
# imports
# ---------------------------------------------------------------------------
import pandas as pd
import pytest

from nbr import __main__ as cli
from nbr.data import attach_levels, dataset_dir, load_items, load_split, split_target

# ---------------------------------------------------------------------------
# constants
# ---------------------------------------------------------------------------
LEVELS = {"L1": "dept", "L2": "aisle", "L3": "ingredient"}
COLUMNS = ["dataset", "user_id", "basket_id", "basket_seq", "item_id",
           "coarse_category", "fine_category", "quantity"]


# ---------------------------------------------------------------------------
# fixtures and helpers
# ---------------------------------------------------------------------------
def _frame(rows):
    '''Build a basket frame with the columns of COLUMNS from ``rows``.'''
    return pd.DataFrame(rows, columns=COLUMNS).astype({"basket_seq": "int32"})


def _items():
    '''Return an items frame of three items with their dept, aisle and ingredient columns.'''
    return pd.DataFrame(
        {
            "item_id": ["i1", "i2", "i3"],
            "dept": ["d1", "d1", "d2"],
            "aisle": ["a1", "a2", "a3"],
            "ingredient": ["g1", "g2", "g3"],
        }
    )


def _write_dataset(root, profile, dataset):
    '''Write the parquet files of ``dataset`` under ``root`` / ``profile`` / ``dataset``.'''
    # Create the dataset folder.
    folder = root / profile / dataset
    folder.mkdir(parents=True)
    # Build a history of three rows and a target of three rows for two users.
    history = _frame([[dataset, "u1", "b1", 1, "i1", "c", "f", 1],
                      [dataset, "u1", "b2", 2, "i2", "c", "f", 1],
                      [dataset, "u2", "b1", 1, "i2", "c", "f", 1]])
    target = _frame([[dataset, "u1", "b3", 3, "i1", "c", "f", 1],
                     [dataset, "u1", "b3", 3, "i3", "c", "f", 1],
                     [dataset, "u2", "b2", 2, "i2", "c", "f", 1]])
    # Write the same history and target for the fit and the eval split.
    for split in ("fit", "eval"):
        history.to_parquet(folder / f"{split}_history.parquet", index=False)
        target.to_parquet(folder / f"{split}_target.parquet", index=False)
    # Write the items file.
    _items().to_parquet(folder / "items.parquet", index=False)


@pytest.fixture
def data_root(tmp_path):
    '''Write three synthetic profile/dataset folders under ``tmp_path`` and return it.'''
    _write_dataset(tmp_path, "holdout", "instacart")
    _write_dataset(tmp_path, "medium", "complete_journey")
    _write_dataset(tmp_path, "sample", "instacart")
    return tmp_path


def _config(data_root, profile, role, cj_profile=None):
    '''Build a config dict from ``data_root``, ``profile``, ``role`` and ``cj_profile``.'''
    config = {"profile": profile, "role": role, "data_dir": data_root}
    # Add the Complete Journey profile only when one is given.
    if cj_profile:
        config["complete_journey_profile"] = cj_profile
    return config


# ---------------------------------------------------------------------------
# holdout guard
# ---------------------------------------------------------------------------
def test_holdout_guard_blocks_instacart_eval(data_root):
    # Build a config with the holdout profile and the final role.
    config = _config(data_root, "holdout", "final", "medium")
    # Loading the Instacart eval split raises PermissionError, the fit split loads.
    with pytest.raises(PermissionError):
        load_split(config, "instacart", "eval")
    history, target = load_split(config, "instacart", "fit")
    assert len(history) == 3 and len(target) == 3


def test_holdout_guard_explicit_allow(data_root):
    # Load the Instacart eval split of the holdout profile with allow_holdout_eval=True.
    config = _config(data_root, "holdout", "final", "medium")
    history, target = load_split(config, "instacart", "eval", allow_holdout_eval=True)
    # The target is returned with its three rows.
    assert len(target) == 3


def test_holdout_guard_not_applied_to_cj_or_smoke(data_root):
    # Build a config with the holdout profile and the final role.
    final = _config(data_root, "holdout", "final", "medium")
    # Complete Journey is read from the medium folder and its eval split loads without error.
    assert dataset_dir(final, "complete_journey") == data_root / "medium" / "complete_journey"
    load_split(final, "complete_journey", "eval")
    # Build a config with the sample profile and the smoke role.
    smoke = _config(data_root, "sample", "smoke")
    # The Instacart eval split loads without error.
    load_split(smoke, "instacart", "eval")


# ---------------------------------------------------------------------------
# items and category levels
# ---------------------------------------------------------------------------
def test_load_items(data_root):
    # Load the items file of the sample profile.
    items = load_items(_config(data_root, "sample", "smoke"), "instacart")
    # The frame holds the three written items in order.
    assert list(items["item_id"]) == ["i1", "i2", "i3"]


def test_attach_levels():
    # Build a frame of two items and keep a copy of it.
    frame = _frame([["x", "u1", "b1", 1, "i1", "c", "f", 1],
                    ["x", "u1", "b1", 1, "i3", "c", "f", 1]])
    original = frame.copy()
    # Attach the three category levels from the items frame.
    out = attach_levels(frame, _items(), LEVELS)
    # The output gains string columns L1, L2 and L3 with the item categories.
    pd.testing.assert_frame_equal(frame, original)  # input untouched
    assert list(out["L1"]) == ["d1", "d2"]
    assert list(out["L2"]) == ["a1", "a3"]
    assert list(out["L3"]) == ["g1", "g3"]
    assert all(isinstance(v, str) for v in out["L2"])
    assert list(out.columns) == COLUMNS + ["L1", "L2", "L3"]


def test_attach_levels_missing_item_raises():
    # Build a frame with an item that is not in the items frame.
    frame = _frame([["x", "u1", "b1", 1, "i9", "c", "f", 1]])
    # Attaching the levels raises ValueError.
    with pytest.raises(ValueError):
        attach_levels(frame, _items(), LEVELS)


# ---------------------------------------------------------------------------
# target split
# ---------------------------------------------------------------------------
def test_split_target():
    # Build a history of one item per user and a target with repeat, duplicate and new rows.
    history = _frame([["x", "u1", "b1", 1, "i1", "c", "f", 1],
                      ["x", "u2", "b1", 1, "i2", "c", "f", 1]])
    target = _frame([["x", "u1", "b2", 2, "i1", "c", "f", 1],   # repeat
                     ["x", "u1", "b2", 2, "i1", "c", "f", 2],   # duplicate -> dropped
                     ["x", "u1", "b2", 2, "i2", "c", "f", 1],   # new (i2 seen by u2 only)
                     ["x", "u2", "b2", 2, "i3", "c", "f", 1]])  # new
    target = attach_levels(target, _items(), LEVELS)
    # Split the target into repeat rows and new rows.
    repeat, new = split_target(history, target)
    # One repeat row and two new rows remain, with the columns of the target.
    assert repeat[["user_id", "item_id"]].values.tolist() == [["u1", "i1"]]
    assert new[["user_id", "item_id"]].values.tolist() == [["u1", "i2"], ["u2", "i3"]]
    assert list(repeat.columns) == list(target.columns)
    assert list(new["L2"]) == ["a2", "a3"]
    assert len(repeat) + len(new) == 3


# ---------------------------------------------------------------------------
# check command
# ---------------------------------------------------------------------------
def test_check_sample_and_holdout(monkeypatch, capsys, data_root):
    # Replace load_config of the command-line module by a config over the synthetic data.
    monkeypatch.setattr(cli, "load_config", lambda profile: {
        **_config(data_root, profile, "smoke" if profile == "sample" else "final", "medium"),
        "datasets": ["instacart"],
    })
    # Run the check command on the sample profile.
    cli.main(["check", "--profile", "sample"])
    out = capsys.readouterr().out.splitlines()
    # The output reports the counts of the fit split and a line for the eval split.
    assert out[1].startswith("instacart/fit:")
    assert "users=2" in out[1] and "history_rows=3" in out[1] and "target_rows=3" in out[1]
    assert "users_with_repeat_target=2" in out[1] and "users_with_new_target=1" in out[1]
    assert out[2].startswith("instacart/eval:")

    # Run the check command on the holdout profile.
    cli.main(["check", "--profile", "holdout"])
    out = capsys.readouterr().out.splitlines()
    # The eval split is reported as skipped by the holdout guard.
    assert out[1].startswith("instacart/fit:")
    assert out[2] == "instacart/eval: skipped (holdout guard)"
