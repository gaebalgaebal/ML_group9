"""Configuration loading: configs/common.yaml overlaid with configs/<profile>.yaml."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml

# repo_root is derived from this file: <repo>/src/nbr/config.py -> <repo>
REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = REPO_ROOT / "configs"


def _read_yaml(path: Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle)
    return dict(loaded or {})


def load_config(profile: str) -> dict[str, Any]:
    """Return the merged config for ``profile``.

    Keys in configs/<profile>.yaml override the same top-level keys in
    configs/common.yaml (shallow merge). The result also carries
    ``repo_root`` (Path) and ``data_dir`` (Path): the NBR_DATA_DIR environment
    variable wins over ``common.data_dir`` (resolved relative to repo_root).
    """
    if not profile or not profile.replace("_", "").replace("-", "").isalnum():
        raise ValueError(f"invalid profile name {profile!r}")
    common = _read_yaml(CONFIG_DIR / "common.yaml")
    profile_path = CONFIG_DIR / f"{profile}.yaml"
    if not profile_path.exists():
        raise FileNotFoundError(f"unknown profile '{profile}': {profile_path} not found")
    config = {**common, **_read_yaml(profile_path)}
    config.setdefault("profile", profile)

    env_dir = os.environ.get("NBR_DATA_DIR")
    if env_dir:
        data_dir = Path(env_dir)
    else:
        data_dir = REPO_ROOT / config["data_dir"]
    config["repo_root"] = REPO_ROOT
    config["data_dir"] = data_dir
    return config
