# *------------------- 2026 Fall Semester Machine Learning Term-project: Team 9 --------------------*
# Code First written at 2026-11-05
# Code Final Edited at 2026-11-13
# Editor: Yejun Kim

'''
Purpose : Tests for nbr.config and nbr.io.
Inputs  : The profile files read by nbr.config.load_config (sample, holdout, holdout3,
          holdout4), the NBR_DATA_DIR environment variable set through monkeypatch and the
          pytest tmp_path directory.
Outputs : pytest pass/fail results; JSON files are written only under tmp_path.
Usage   :
    python -m pytest tests/test_config.py
'''

# ---------------------------------------------------------------------------
# imports
# ---------------------------------------------------------------------------
import json
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from nbr.config import REPO_ROOT, load_config, resolve_data_dir
from nbr.io import write_json


# ---------------------------------------------------------------------------
# profile loading
# ---------------------------------------------------------------------------
def test_profile_overrides_common(monkeypatch, tmp_path):
    # Clear the data directory variable and load the sample profile from another directory.
    monkeypatch.delenv("NBR_DATA_DIR", raising=False)
    monkeypatch.chdir(tmp_path)  # repo_root must not depend on the working directory
    config = load_config("sample")
    # The config holds the profile values, the common values and paths under the repo root.
    assert config["profile"] == "sample"
    assert config["role"] == "smoke"
    assert config["datasets"] == ["instacart", "complete_journey"]
    assert config["top_k"] == 10  # from common.yaml
    assert config["levels"]["instacart"]["L2"] == "fine_category"
    assert config["repo_root"] == REPO_ROOT
    assert config["data_dir"] == REPO_ROOT / "data" / "processed"


def test_env_override_data_dir(monkeypatch, tmp_path):
    # Point the data directory variable at tmp_path and load the sample profile.
    monkeypatch.setenv("NBR_DATA_DIR", str(tmp_path))
    config = load_config("sample")
    # The data directory of the config is tmp_path.
    assert config["data_dir"] == Path(tmp_path)


def test_holdout_profile_fields(monkeypatch):
    # Clear the data directory variable and load the holdout profile.
    monkeypatch.delenv("NBR_DATA_DIR", raising=False)
    config = load_config("holdout")
    # The role is final and the Complete Journey profile is medium.
    assert config["role"] == "final"
    assert config["complete_journey_profile"] == "medium"


@pytest.mark.parametrize("profile", ["", "..", "../sample", "a\\b"])
def test_invalid_profile_name(profile):
    # An empty profile name or one with path parts raises ValueError.
    with pytest.raises(ValueError):
        load_config(profile)


def test_unknown_profile():
    # A profile name without a profile file raises FileNotFoundError.
    with pytest.raises(FileNotFoundError):
        load_config("no_such_profile")


# ---------------------------------------------------------------------------
# write_json metadata
# ---------------------------------------------------------------------------
def test_write_json_meta(monkeypatch, tmp_path):
    def fake_run(command, **kwargs):
        '''Check that ``command`` is the git HEAD query and return a fixed commit id.'''
        assert command == ["git", "rev-parse", "HEAD"]
        return subprocess.CompletedProcess(command, 0, stdout="abc123\n")

    # Replace the git call and write a payload with a numpy float and a non-ASCII label.
    monkeypatch.setattr("nbr.io.subprocess.run", fake_run)
    payload = {"value": np.float64(0.5), "label": "결과"}
    path = write_json(tmp_path / "out" / "r.json", payload, {"repo_root": tmp_path, "profile": "sample"})
    document = json.loads(Path(path).read_text(encoding="utf-8"))
    # The file holds the payload and a meta block with commit, profile, time and versions.
    assert document["value"] == 0.5 and document["label"] == "결과"
    meta = document["meta"]
    assert meta["git_commit"] == "abc123"
    assert meta["profile"] == "sample"
    assert meta["utc_time"].endswith("+00:00")
    assert meta["numpy_version"] == np.__version__
    assert meta["pandas_version"] == pd.__version__
    assert meta["python_version"]


def test_write_json_git_failure(monkeypatch, tmp_path):
    def fail(*args, **kwargs):
        '''Raise FileNotFoundError for any arguments, as when git is not installed.'''
        raise FileNotFoundError("git")

    # Replace the git call by one that fails and write an empty payload.
    monkeypatch.setattr("nbr.io.subprocess.run", fail)
    path = write_json(tmp_path / "r.json", {}, {"repo_root": tmp_path, "profile": "sample"})
    # The file is written with a null git_commit.
    assert json.loads(Path(path).read_text(encoding="utf-8"))["meta"]["git_commit"] is None


# ---------------------------------------------------------------------------
# data directory resolution
# ---------------------------------------------------------------------------
def test_resolve_data_dir(monkeypatch, tmp_path):
    # Clear the data directory variable.
    monkeypatch.delenv("NBR_DATA_DIR", raising=False)
    # Without the variable the directory is data/processed or the config value under the repo root.
    assert resolve_data_dir() == REPO_ROOT / "data" / "processed"
    assert resolve_data_dir({"data_dir": "elsewhere"}) == REPO_ROOT / "elsewhere"
    # Point the data directory variable at tmp_path.
    monkeypatch.setenv("NBR_DATA_DIR", str(tmp_path))
    # With the variable set the directory is tmp_path whatever the config says.
    assert resolve_data_dir() == Path(tmp_path)
    assert resolve_data_dir({"data_dir": "elsewhere"}) == Path(tmp_path)


# ---------------------------------------------------------------------------
# holdout4 profile
# ---------------------------------------------------------------------------
def test_holdout4_profile_overrides_blocks_in_full(monkeypatch):
    """PREREG v1.5: holdout4 = holdout3 plus lr_repeat.itemcf_feature and the 97.5 % level; the shallow merge keeps every other key."""
    # Clear the data directory variable and load the holdout4 and holdout3 profiles.
    monkeypatch.delenv("NBR_DATA_DIR", raising=False)
    config, previous = load_config("holdout4"), load_config("holdout3")
    # Only the lr_repeat and bootstrap blocks differ between the two profiles.
    assert config["role"] == "final" and config["datasets"] == ["instacart"] and config["complete_journey_profile"] == "medium"
    assert config["lr_repeat"] == {**previous["lr_repeat"], "itemcf_feature": True}
    assert previous["lr_repeat"]["itemcf_feature"] is False
    assert config["bootstrap"] == {"n": 1000, "seed": 42, "levels": [0.95, 0.975]}
    assert previous["bootstrap"]["levels"] == [0.95, 0.99375]
    for key in set(previous) - {"profile", "lr_repeat", "bootstrap"}:
        assert config[key] == previous[key], key
    assert config["explore"]["itemcf"]["neighbors"] == 20 and config["usercf"]["k"] == [10, 20, 50, 100, 200, 400]
