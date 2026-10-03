"""Saved-classifier store: one JSON file per classifier under `<state_dir>/classifiers/`.

Paths are only ever built from a validated name; writes are atomic (temp file in the same dir + `os.replace`).
"""

from __future__ import annotations

import contextlib
import json
import logging
import os
import re
import tempfile
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

log = logging.getLogger("clef")

NAME_RE = re.compile(r"^[a-z0-9][a-z0-9_-]{0,63}$")


class InvalidName(ValueError):
    """Classifier name does not match NAME_RE (-> HTTP 400)."""


class TooManyClassifiers(RuntimeError):
    """max_classifiers reached (-> HTTP 409)."""


def validate_name(name: Any) -> str:
    if not isinstance(name, str) or not NAME_RE.fullmatch(name):
        raise InvalidName("classifier name must match ^[a-z0-9][a-z0-9_-]{0,63}$")
    return name


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


class ClassifierStore:
    def __init__(self, directory: str | Path, max_classifiers: int = 1000):
        self.dir = Path(directory)
        self.max_classifiers = max_classifiers
        self._lock = threading.Lock()

    def _path(self, name: str) -> Path:
        validate_name(name)  # validate first, only then build a path
        path = self.dir / f"{name}.json"
        if path.resolve().parent != self.dir.resolve():  # belt and braces
            raise InvalidName("invalid classifier name")
        return path

    def _read(self, path: Path) -> dict[str, Any] | None:
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return None
        except (OSError, ValueError) as exc:
            log.warning("skipping unreadable classifier file %s: %s", path.name, exc)
            return None
        if not isinstance(data, dict) or data.get("name") != path.stem:
            log.warning("skipping malformed classifier file %s", path.name)
            return None
        return data

    def get(self, name: str) -> dict[str, Any] | None:
        return self._read(self._path(name))

    def list(self) -> list[dict[str, Any]]:
        if not self.dir.is_dir():
            return []
        out = []
        for path in sorted(self.dir.glob("*.json")):
            if not NAME_RE.fullmatch(path.stem):
                continue
            data = self._read(path)
            if data is not None:
                out.append(data)
        return out

    def _count(self) -> int:
        return sum(1 for p in self.dir.glob("*.json") if NAME_RE.fullmatch(p.stem))

    def put(self, name: str, definition: dict[str, Any]) -> dict[str, Any]:
        """Create or overwrite; `created_at` survives an overwrite. Returns the stored document."""
        path = self._path(name)
        with self._lock:
            self.dir.mkdir(parents=True, exist_ok=True)
            existing = self._read(path) if path.exists() else None
            if existing is None and self._count() >= self.max_classifiers:
                raise TooManyClassifiers(f"too many saved classifiers (max {self.max_classifiers})")
            now = _now()
            doc = {
                **{k: v for k, v in definition.items() if k not in ("name", "created_at", "updated_at")},
                "name": name,
                "created_at": (existing or {}).get("created_at") or now,
                "updated_at": now,
            }
            fd, tmp = tempfile.mkstemp(dir=self.dir, prefix=f".{name}.", suffix=".tmp")
            try:
                with os.fdopen(fd, "w", encoding="utf-8") as fh:
                    json.dump(doc, fh, ensure_ascii=False, indent=2)
                    fh.flush()
                    os.fsync(fh.fileno())
                os.replace(tmp, path)
            except BaseException:
                with contextlib.suppress(OSError):
                    os.unlink(tmp)
                raise
            return doc

    def delete(self, name: str) -> bool:
        path = self._path(name)
        with self._lock:
            try:
                path.unlink()
            except FileNotFoundError:
                return False
            return True
