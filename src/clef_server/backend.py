"""Device backends (placeholder until the backend workstream lands; see docs/ARCHITECTURE.md)."""

from __future__ import annotations

import os
from collections.abc import MutableMapping


def prepare_environment(environ: MutableMapping[str, str] = os.environ) -> list[str]:
    return []
