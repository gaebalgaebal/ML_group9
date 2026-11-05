"""JSON output with provenance metadata."""

from __future__ import annotations

import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def git_commit(repo_root: Path | None = None) -> str | None:
    """Return the output of `git rev-parse HEAD` in ``repo_root`` (None on failure)."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root,
            capture_output=True,
            text=True,
            check=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip() or None


def _default(obj: Any) -> Any:
    """JSON fallback for numpy scalars, arrays, and Paths."""
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(f"not JSON serialisable: {type(obj).__name__}")


def write_json(path: str | Path, payload: dict[str, Any], config: dict[str, Any]) -> Path:
    """Write ``payload`` plus a ``meta`` block (commit, UTC time, profile, versions) as UTF-8 JSON."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    meta = {
        "git_commit": git_commit(config.get("repo_root")),
        "utc_time": datetime.now(timezone.utc).isoformat(),
        "profile": config.get("profile"),
        "python_version": sys.version.split()[0],
        "numpy_version": np.__version__,
        "pandas_version": pd.__version__,
    }
    document = {"meta": meta, **payload}
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(document, handle, ensure_ascii=False, indent=2, default=_default)
    return path
