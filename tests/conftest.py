"""Make the flat `server/` modules importable as top-level modules (config, schemas, engine, ...)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "server"))
