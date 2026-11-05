"""Tests for nbr.config and nbr.io."""

import json
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from nbr.config import REPO_ROOT, load_config
from nbr.io import write_json


def test_profile_overrides_common(monkeypatch, tmp_path):
    monkeypatch.delenv("NBR_DATA_DIR", raising=False)
    monkeypatch.chdir(tmp_path)  # repo_root must not depend on the working directory
    config = load_config("sample")
    assert config["profile"] == "sample"
    assert config["role"] == "smoke"
    assert config["datasets"] == ["instacart", "complete_journey"]
    assert config["top_k"] == 10  # from common.yaml
    assert config["levels"]["instacart"]["L2"] == "fine_category"
    assert config["repo_root"] == REPO_ROOT
    assert config["data_dir"] == REPO_ROOT / "data" / "processed"


def test_env_override_data_dir(monkeypatch, tmp_path):
    monkeypatch.setenv("NBR_DATA_DIR", str(tmp_path))
    config = load_config("sample")
    assert config["data_dir"] == Path(tmp_path)


def test_holdout_profile_fields(monkeypatch):
    monkeypatch.delenv("NBR_DATA_DIR", raising=False)
    config = load_config("holdout")
    assert config["role"] == "final"
    assert config["complete_journey_profile"] == "medium"


@pytest.mark.parametrize("profile", ["", "..", "../sample", "a\\b"])
def test_invalid_profile_name(profile):
    with pytest.raises(ValueError):
        load_config(profile)


def test_unknown_profile():
    with pytest.raises(FileNotFoundError):
        load_config("no_such_profile")


def test_write_json_meta(monkeypatch, tmp_path):
    def fake_run(command, **kwargs):
        assert command == ["git", "rev-parse", "HEAD"]
        return subprocess.CompletedProcess(command, 0, stdout="abc123\n")

    monkeypatch.setattr("nbr.io.subprocess.run", fake_run)
    payload = {"value": np.float64(0.5), "label": "결과"}
    path = write_json(tmp_path / "out" / "r.json", payload, {"repo_root": tmp_path, "profile": "sample"})
    document = json.loads(Path(path).read_text(encoding="utf-8"))
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
        raise FileNotFoundError("git")

    monkeypatch.setattr("nbr.io.subprocess.run", fail)
    path = write_json(tmp_path / "r.json", {}, {"repo_root": tmp_path, "profile": "sample"})
    assert json.loads(Path(path).read_text(encoding="utf-8"))["meta"]["git_commit"] is None
