# *------------------- 2026 Fall Semester Machine Learning Term-project: Team 9 --------------------*
# Code First written at 2026-11-05
# Code Final Edited at 2026-11-13
# Editor: Yejun Kim

'''
Purpose : Configuration loading: configs/common.yaml overlaid with configs/<profile>.yaml.
          Also resolves the processed-data root and the folder of the tuning JSON files.
Inputs  : A profile name, the YAML files configs/common.yaml, configs/<profile>.yaml and
          configs/usercf.yaml, and the environment variables NBR_DATA_DIR and NBR_TUNING_DIR.
Outputs : load_config returns the merged config dict with the added keys profile, repo_root
          and data_dir; resolve_data_dir and resolve_tuning_dir return a Path.
Usage   :
    from nbr.config import load_config
    config = load_config(args.profile)
'''

from __future__ import annotations

# ---------------------------------------------------------------------------
# imports
# ---------------------------------------------------------------------------
import os
from pathlib import Path
from typing import Any

import yaml

# ---------------------------------------------------------------------------
# constants
# ---------------------------------------------------------------------------
# repo_root is derived from this file: <repo>/src/nbr/config.py -> <repo>
REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = REPO_ROOT / "configs"


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _read_yaml(path: Path) -> dict[str, Any]:
    '''Read the YAML file ``path`` and return its top-level mapping (empty dict if empty).'''
    # Parse the file as UTF-8 YAML.
    with open(path, encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle)
    # Return a dict, using an empty one when the file has no content.
    return dict(loaded or {})


# ---------------------------------------------------------------------------
# config loading
# ---------------------------------------------------------------------------
def load_config(profile: str) -> dict[str, Any]:
    """Return the merged config for ``profile``.

    Keys in configs/<profile>.yaml override the same top-level keys in
    configs/common.yaml (shallow merge); the ``usercf`` block of
    configs/usercf.yaml is added when neither file defines one. The result also carries
    ``repo_root`` (Path) and ``data_dir`` (Path, see resolve_data_dir: the NBR_DATA_DIR
    environment variable wins over ``common.data_dir``, resolved relative to repo_root).
    """
    # Reject a profile name with characters other than letters, digits, "_" and "-".
    if not profile or not profile.replace("_", "").replace("-", "").isalnum():
        raise ValueError(f"invalid profile name {profile!r}")
    # Read the common settings.
    common = _read_yaml(CONFIG_DIR / "common.yaml")
    # Locate the profile file and fail when it does not exist.
    profile_path = CONFIG_DIR / f"{profile}.yaml"
    if not profile_path.exists():
        raise FileNotFoundError(f"unknown profile '{profile}': {profile_path} not found")
    # Overlay the profile keys on the common keys.
    config = {**common, **_read_yaml(profile_path)}
    # Record the profile name unless the files already set one.
    config.setdefault("profile", profile)
    # Add the usercf block of configs/usercf.yaml unless one is already defined.
    usercf_path = CONFIG_DIR / "usercf.yaml"
    if usercf_path.exists():  # W7/W8: user-CF grid lives in its own file; only its "usercf" block is merged
        config.setdefault("usercf", _read_yaml(usercf_path).get("usercf"))

    # Add the repository root and the resolved data directory.
    config["repo_root"] = REPO_ROOT
    config["data_dir"] = resolve_data_dir(config)
    return config


# ---------------------------------------------------------------------------
# path resolution
# ---------------------------------------------------------------------------
def resolve_data_dir(settings: dict[str, Any] | None = None) -> Path:
    """Processed-data root shared by every profile (the parent of <profile>/<dataset>/*.parquet).

    The NBR_DATA_DIR environment variable wins; otherwise ``settings["data_dir"]`` (default: the
    ``data_dir`` of configs/common.yaml) is resolved relative to the repository root.
    """
    # Use the NBR_DATA_DIR environment variable when it is set.
    env_dir = os.environ.get("NBR_DATA_DIR")
    if env_dir:
        return Path(env_dir)
    # Fall back to the common settings when no settings are given.
    if settings is None:
        settings = _read_yaml(CONFIG_DIR / "common.yaml")
    # Resolve the configured data_dir relative to the repository root.
    return REPO_ROOT / settings["data_dir"]


def resolve_tuning_dir(config: dict[str, Any], override: str | Path | None = None) -> Path:
    """Folder of the <profile>_<dataset>_tuning.json files read by the stand-alone runners.

    ``override`` (a CLI argument) wins, then the NBR_TUNING_DIR environment variable, then
    <repo_root>/reports/metrics of ``config`` (where ``python -m nbr tune`` writes). Paths are
    used as given (a relative path is relative to the current working directory).
    """
    # Use the override when it is given.
    if override:
        return Path(override)
    # Use the NBR_TUNING_DIR environment variable when it is set.
    env_dir = os.environ.get("NBR_TUNING_DIR")
    if env_dir:
        return Path(env_dir)
    # Default to reports/metrics under the repository root of the config.
    return Path(config["repo_root"]) / "reports" / "metrics"
