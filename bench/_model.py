"""Default --model-path for the bench scripts: the same lookup as `clef serve` (never downloads)."""

from __future__ import annotations

import os
import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1] / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))


def default_model_path() -> str:
    """CLEF_MODEL_PATH > ~/models/clef-flash > the pinned snapshot in the Hugging Face cache."""
    env = os.environ.get("CLEF_MODEL_PATH")
    if env:
        return env
    try:
        from clef_server.config import Config
        from clef_server.paths import resolve_model_path

        return str(resolve_model_path(Config()))
    except Exception:  # package or weights missing: keep the old default, the script reports it
        return str(Path.home() / "models" / "clef-flash")
