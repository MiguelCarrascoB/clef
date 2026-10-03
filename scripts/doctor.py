"""Thin shim: `python scripts/doctor.py [--no-gpu] [--smoke] [--json]` == `clef doctor ...`."""

from __future__ import annotations

import sys
from pathlib import Path

try:
    from clef_server.doctor import main
except ImportError:  # running from a checkout without `pip install -e .`
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    from clef_server.doctor import main

if __name__ == "__main__":
    sys.exit(main())
