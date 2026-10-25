"""Project folders, resolved from this file's location so they work from any working directory
(scripts/, notebooks/, tests/). With an editable install (pip install -e .) this file lives at
<repo>/src/c2c/paths.py, so the repo root is two folders up from its parent.
"""

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
RAW = REPO_ROOT / "data" / "raw"
PROCESSED = REPO_ROOT / "data" / "processed"
FIGURES = REPO_ROOT / "results" / "figures"
