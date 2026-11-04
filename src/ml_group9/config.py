"""Configuration and project-path helpers.

load_config(profile) reads configs/<profile>.yaml and adds raw_dir, processed_dir, artifact_dir and
report_dir under the project root.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def load_config(profile: str) -> dict[str, Any]:
    """Load a named YAML profile and attach stable project paths."""
    path = PROJECT_ROOT / "configs" / f"{profile}.yaml"
    if not path.exists():
        raise FileNotFoundError(f"Unknown profile: {profile!r} ({path})")
    with path.open(encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    config["profile"] = profile
    config["project_root"] = PROJECT_ROOT
    config["raw_dir"] = PROJECT_ROOT / "data" / "raw"
    config["processed_dir"] = PROJECT_ROOT / "data" / "processed" / profile
    config["artifact_dir"] = PROJECT_ROOT / "artifacts" / profile
    config["report_dir"] = PROJECT_ROOT / "reports"
    return config
