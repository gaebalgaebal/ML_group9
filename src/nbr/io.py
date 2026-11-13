# *------------------- 2026 Fall Semester Machine Learning Term-project: Team 9 --------------------*
# Code First written at 2026-11-05
# Code Final Edited at 2026-11-13
# Editor: Yejun Kim

'''
Purpose : JSON output with provenance metadata.
Inputs  : An output path, a payload dict and the run config (keys repo_root and profile are
          read).
Outputs : A UTF-8 JSON file holding a ``meta`` block (git commit, UTC time, profile, Python,
          numpy and pandas versions) followed by the payload; write_json returns the path.
Usage   :
    from nbr.io import write_json
    json_path = write_json(out_dir / f"icf_dev_{args.profile}.json", report, config)
'''

from __future__ import annotations

# ---------------------------------------------------------------------------
# imports
# ---------------------------------------------------------------------------
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def git_commit(repo_root: Path | None = None) -> str | None:
    """Return the output of `git rev-parse HEAD` in ``repo_root`` (None on failure)."""
    # Run git in the repository root and capture its text output.
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root,
            capture_output=True,
            text=True,
            check=True,
        )
    # Return None when git cannot be started or exits with an error.
    except (OSError, subprocess.CalledProcessError):
        return None
    # Return the stripped output, or None when it is empty.
    return result.stdout.strip() or None


def _default(obj: Any) -> Any:
    """JSON fallback for numpy scalars, arrays, and Paths."""
    # Convert a numpy scalar to the matching Python scalar.
    if isinstance(obj, np.generic):
        return obj.item()
    # Convert a numpy array to a nested list.
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    # Convert a Path to its string form.
    if isinstance(obj, Path):
        return str(obj)
    # Reject every other type.
    raise TypeError(f"not JSON serialisable: {type(obj).__name__}")


# ---------------------------------------------------------------------------
# json output
# ---------------------------------------------------------------------------
def write_json(path: str | Path, payload: dict[str, Any], config: dict[str, Any]) -> Path:
    """Write ``payload`` plus a ``meta`` block (commit of ``config``'s repo_root, UTC time, profile, versions) to ``path`` as UTF-8 JSON."""
    # Create the parent directory of the output path if it is missing.
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Collect the provenance metadata of this run.
    meta = {
        "git_commit": git_commit(config.get("repo_root")),
        "utc_time": datetime.now(timezone.utc).isoformat(),
        "profile": config.get("profile"),
        "python_version": sys.version.split()[0],
        "numpy_version": np.__version__,
        "pandas_version": pd.__version__,
    }
    # Put the metadata in front of the payload keys.
    document = {"meta": meta, **payload}
    # Write the document as indented JSON without ASCII escaping.
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(document, handle, ensure_ascii=False, indent=2, default=_default)
    return path
