"""Tests for nbr.data and the check command on synthetic parquet files."""

import pandas as pd
import pytest

from nbr import __main__ as cli
from nbr.data import attach_levels, dataset_dir, load_items, load_split, split_target

LEVELS = {"L1": "dept", "L2": "aisle", "L3": "ingredient"}
COLUMNS = ["dataset", "user_id", "basket_id", "basket_seq", "item_id",
           "coarse_category", "fine_category", "quantity"]


def _frame(rows):
    return pd.DataFrame(rows, columns=COLUMNS).astype({"basket_seq": "int32"})


def _items():
    return pd.DataFrame(
        {
            "item_id": ["i1", "i2", "i3"],
            "dept": ["d1", "d1", "d2"],
            "aisle": ["a1", "a2", "a3"],
            "ingredient": ["g1", "g2", "g3"],
        }
    )


def _write_dataset(root, profile, dataset):
    folder = root / profile / dataset
    folder.mkdir(parents=True)
    history = _frame([[dataset, "u1", "b1", 1, "i1", "c", "f", 1],
                      [dataset, "u1", "b2", 2, "i2", "c", "f", 1],
                      [dataset, "u2", "b1", 1, "i2", "c", "f", 1]])
    target = _frame([[dataset, "u1", "b3", 3, "i1", "c", "f", 1],
                     [dataset, "u1", "b3", 3, "i3", "c", "f", 1],
                     [dataset, "u2", "b2", 2, "i2", "c", "f", 1]])
    for split in ("fit", "eval"):
        history.to_parquet(folder / f"{split}_history.parquet", index=False)
        target.to_parquet(folder / f"{split}_target.parquet", index=False)
    _items().to_parquet(folder / "items.parquet", index=False)


@pytest.fixture
def data_root(tmp_path):
    _write_dataset(tmp_path, "holdout", "instacart")
    _write_dataset(tmp_path, "medium", "complete_journey")
    _write_dataset(tmp_path, "sample", "instacart")
    return tmp_path


def _config(data_root, profile, role, cj_profile=None):
    config = {"profile": profile, "role": role, "data_dir": data_root}
    if cj_profile:
        config["complete_journey_profile"] = cj_profile
    return config


def test_holdout_guard_blocks_instacart_eval(data_root):
    config = _config(data_root, "holdout", "final", "medium")
    with pytest.raises(PermissionError):
        load_split(config, "instacart", "eval")
    history, target = load_split(config, "instacart", "fit")
    assert len(history) == 3 and len(target) == 3


def test_holdout_guard_explicit_allow(data_root):
    config = _config(data_root, "holdout", "final", "medium")
    history, target = load_split(config, "instacart", "eval", allow_holdout_eval=True)
    assert len(target) == 3


def test_holdout_guard_not_applied_to_cj_or_smoke(data_root):
    final = _config(data_root, "holdout", "final", "medium")
    assert dataset_dir(final, "complete_journey") == data_root / "medium" / "complete_journey"
    load_split(final, "complete_journey", "eval")
    smoke = _config(data_root, "sample", "smoke")
    load_split(smoke, "instacart", "eval")


def test_load_items(data_root):
    items = load_items(_config(data_root, "sample", "smoke"), "instacart")
    assert list(items["item_id"]) == ["i1", "i2", "i3"]


def test_attach_levels():
    frame = _frame([["x", "u1", "b1", 1, "i1", "c", "f", 1],
                    ["x", "u1", "b1", 1, "i3", "c", "f", 1]])
    original = frame.copy()
    out = attach_levels(frame, _items(), LEVELS)
    pd.testing.assert_frame_equal(frame, original)  # input untouched
    assert list(out["L1"]) == ["d1", "d2"]
    assert list(out["L2"]) == ["a1", "a3"]
    assert list(out["L3"]) == ["g1", "g3"]
    assert all(isinstance(v, str) for v in out["L2"])
    assert list(out.columns) == COLUMNS + ["L1", "L2", "L3"]


def test_attach_levels_missing_item_raises():
    frame = _frame([["x", "u1", "b1", 1, "i9", "c", "f", 1]])
    with pytest.raises(ValueError):
        attach_levels(frame, _items(), LEVELS)


def test_split_target():
    history = _frame([["x", "u1", "b1", 1, "i1", "c", "f", 1],
                      ["x", "u2", "b1", 1, "i2", "c", "f", 1]])
    target = _frame([["x", "u1", "b2", 2, "i1", "c", "f", 1],   # repeat
                     ["x", "u1", "b2", 2, "i1", "c", "f", 2],   # duplicate -> dropped
                     ["x", "u1", "b2", 2, "i2", "c", "f", 1],   # new (i2 seen by u2 only)
                     ["x", "u2", "b2", 2, "i3", "c", "f", 1]])  # new
    target = attach_levels(target, _items(), LEVELS)
    repeat, new = split_target(history, target)
    assert repeat[["user_id", "item_id"]].values.tolist() == [["u1", "i1"]]
    assert new[["user_id", "item_id"]].values.tolist() == [["u1", "i2"], ["u2", "i3"]]
    assert list(repeat.columns) == list(target.columns)
    assert list(new["L2"]) == ["a2", "a3"]
    assert len(repeat) + len(new) == 3


def test_check_sample_and_holdout(monkeypatch, capsys, data_root):
    monkeypatch.setattr(cli, "load_config", lambda profile: {
        **_config(data_root, profile, "smoke" if profile == "sample" else "final", "medium"),
        "datasets": ["instacart"],
    })
    cli.main(["check", "--profile", "sample"])
    out = capsys.readouterr().out.splitlines()
    assert out[1].startswith("instacart/fit:")
    assert "users=2" in out[1] and "history_rows=3" in out[1] and "target_rows=3" in out[1]
    assert "users_with_repeat_target=2" in out[1] and "users_with_new_target=1" in out[1]
    assert out[2].startswith("instacart/eval:")

    cli.main(["check", "--profile", "holdout"])
    out = capsys.readouterr().out.splitlines()
    assert out[1].startswith("instacart/fit:")
    assert out[2] == "instacart/eval: skipped (holdout guard)"
