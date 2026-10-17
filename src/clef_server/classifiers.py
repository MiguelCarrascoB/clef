"""Saved-classifier store: versioned definitions with an owner, in the database (``classifiers`` +
``classifier_versions``).

Every PUT appends a version (1, 2, ...; never reused while the name exists); a definition equal to the latest
is a no-op. ``get`` returns the latest unless a version is asked for. The v3.1 JSON files
(``<state_dir>/classifiers/*.json``) are imported once at startup (``import_json_dir``).
"""

from __future__ import annotations

import json
import logging
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pydantic import BaseModel
from sqlalchemy import delete, func, insert, select, update
from sqlalchemy.engine import Connection

from . import dbschema as t
from .db import Database, audit

log = logging.getLogger("clef")

NAME_RE = re.compile(r"^[a-z0-9][a-z0-9_-]{0,63}$")
_SERVER_KEYS = ("name", "created_at", "updated_at", "version", "owner", "model")  # never part of a definition
_C, _V = t.classifiers, t.classifier_versions


class InvalidName(ValueError):
    """Classifier name does not match NAME_RE (-> HTTP 400)."""


class TooManyClassifiers(RuntimeError):
    """max_classifiers reached (-> HTTP 409)."""


def validate_name(name: Any) -> str:
    if not isinstance(name, str) or not NAME_RE.fullmatch(name):
        raise InvalidName("classifier name must match ^[a-z0-9][a-z0-9_-]{0,63}$")
    return name


def _iso(ts: float) -> str:
    return datetime.fromtimestamp(ts, timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def _parse_iso(raw: Any) -> float | None:
    if not isinstance(raw, str):
        return None
    try:
        return datetime.strptime(raw, "%Y-%m-%dT%H:%M:%S.%fZ").replace(tzinfo=timezone.utc).timestamp()
    except ValueError:
        return None


def _definition(definition: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in definition.items() if k not in _SERVER_KEYS}


def _dumps(definition: dict[str, Any]) -> str:
    return json.dumps(definition, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


class ClassifierVersion(BaseModel):
    version: int
    created_at: str
    created_by: str | None = None
    model: str | None = None


class ClassifierVersions(BaseModel):
    name: str
    versions: list[ClassifierVersion]


class ClassifierStore:
    """``get`` / ``list`` / ``put`` / ``delete`` keep v3.1 shapes (+ ``version``, ``owner``, ``model``)."""

    def __init__(self, db: Database, max_classifiers: int = 1000):
        self.db = db
        self.max_classifiers = max_classifiers

    # ---- reads
    def _doc(self, head: Any, ver: Any) -> dict[str, Any]:
        return {
            **json.loads(ver.definition),
            "name": head.name,
            "created_at": _iso(head.created_at),
            "updated_at": _iso(ver.created_at),
            "version": ver.version,
            "owner": head.owner,
            "model": ver.model,
        }

    def _get(self, conn: Connection, name: str, version: int | None) -> dict[str, Any] | None:
        head = conn.execute(select(_C).where(_C.c.name == name)).first()
        if head is None:
            return None
        ver = conn.execute(
            select(_V).where(
                _V.c.name == name, _V.c.version == (head.latest_version if version is None else version)
            )
        ).first()
        return None if ver is None else self._doc(head, ver)

    def get(self, name: str, version: int | None = None) -> dict[str, Any] | None:
        """The latest definition (or ``version``); None for an unknown name or version."""
        validate_name(name)
        with self.db.read() as conn:
            return self._get(conn, name, version)

    def list(self) -> list[dict[str, Any]]:
        """The latest version of every classifier, by name."""
        with self.db.read() as conn:
            rows = conn.execute(
                select(_C, _V.c.version, _V.c.definition, _V.c.model, _V.c.created_at.label("v_created"))
                .join(_V, (_V.c.name == _C.c.name) & (_V.c.version == _C.c.latest_version))
                .order_by(_C.c.name)
            ).all()
        out = []
        for r in rows:
            doc = {**json.loads(r.definition), "name": r.name, "created_at": _iso(r.created_at)}
            doc.update(updated_at=_iso(r.v_created), version=r.version, owner=r.owner, model=r.model)
            out.append(doc)
        return out

    def versions(self, name: str) -> list[dict[str, Any]] | None:
        """``[{version, created_at, created_by, model}]`` newest first; None for an unknown name."""
        validate_name(name)
        with self.db.read() as conn:
            if conn.execute(select(_C.c.name).where(_C.c.name == name)).first() is None:
                return None
            rows = conn.execute(
                select(_V.c.version, _V.c.created_at, _V.c.created_by, _V.c.model)
                .where(_V.c.name == name)
                .order_by(_V.c.version.desc())
            ).all()
        return [
            {
                "version": r.version,
                "created_at": _iso(r.created_at),
                "created_by": r.created_by,
                "model": r.model,
            }
            for r in rows
        ]

    def owner(self, name: str) -> str | None:
        validate_name(name)
        with self.db.read() as conn:
            return conn.execute(select(_C.c.owner).where(_C.c.name == name)).scalar()

    def owner_of(self, name: str) -> tuple[bool, str | None]:
        """``(exists, owner)``: owner-less (auth-off) is ``(True, None)``, a new name is ``(False, None)``."""
        validate_name(name)
        with self.db.read() as conn:
            row = conn.execute(select(_C.c.owner).where(_C.c.name == name)).first()
        return (False, None) if row is None else (True, row[0])

    # ---- writes
    def put(
        self,
        name: str,
        definition: dict[str, Any],
        owner: str | None = None,
        created_by: str | None = None,
        model: str | None = None,
        *,
        actor: str | None = None,
        ip: str | None = None,
        request_id: str | None = None,
    ) -> dict[str, Any]:
        """Append version n+1 (the first PUT creates the name, owned by ``owner``). A definition equal to the
        latest is a no-op that returns the latest. Writes ``classifier.put`` to the audit log."""
        validate_name(name)
        body = _definition(definition)
        with self.db.tx() as conn:
            head = conn.execute(select(_C).where(_C.c.name == name).with_for_update()).first()
            now = time.time()
            if head is None:
                self.db.serialize(conn, "classifiers.count")
                n = conn.execute(select(func.count()).select_from(_C)).scalar_one()
                if n >= self.max_classifiers:
                    raise TooManyClassifiers(f"too many saved classifiers (max {self.max_classifiers})")
                version = 1
                conn.execute(
                    insert(_C).values(
                        name=name, owner=owner, latest_version=1, created_at=now, updated_at=now
                    )
                )
            else:
                latest = conn.execute(
                    select(_V.c.definition).where(_V.c.name == name, _V.c.version == head.latest_version)
                ).scalar()
                if latest is not None and _dumps(json.loads(latest)) == _dumps(body):
                    return self._get(conn, name, None) or {}  # idempotent PUT
                version = head.latest_version + 1
                conn.execute(
                    update(_C).where(_C.c.name == name).values(latest_version=version, updated_at=now)
                )
            conn.execute(
                insert(_V).values(
                    name=name,
                    version=version,
                    definition=_dumps(body),
                    model=model,
                    created_at=now,
                    created_by=created_by,
                )
            )
            audit(
                conn,
                actor or created_by or "anonymous",
                "classifier.put",
                name,
                {"version": version},
                ip=ip,
                request_id=request_id,
            )
            return self._get(conn, name, None) or {}

    def delete(
        self, name: str, *, actor: str | None = None, ip: str | None = None, request_id: str | None = None
    ) -> bool:
        """Remove the name and every version (jobs keep their snapshot). Audited as ``classifier.delete``."""
        validate_name(name)
        with self.db.tx() as conn:
            conn.execute(delete(_V).where(_V.c.name == name))
            if conn.execute(delete(_C).where(_C.c.name == name)).rowcount != 1:
                return False
            audit(conn, actor or "anonymous", "classifier.delete", name, ip=ip, request_id=request_id)
            return True


# ------------------------------------------------------------------ one-time import of the v3.1 JSON files


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        log.warning("classifiers: skipping unreadable file %s: %s", path.name, exc)
        return None
    if not isinstance(data, dict) or data.get("name") != path.stem:
        log.warning("classifiers: skipping malformed file %s", path.name)
        return None
    from pydantic import ValidationError

    from .classify import ClassifierDef

    try:
        ClassifierDef.model_validate(_definition(data))
    except ValidationError as exc:
        log.warning("classifiers: skipping invalid definition %s: %s", path.name, exc.errors()[0].get("msg"))
        return None
    return data


def import_json_dir(db: Database, directory: Path) -> int:
    """Once per database (``clef_meta.classifiers_imported``): every readable ``*.json`` becomes version 1
    (owner NULL, ``created_by`` "import"); the directory is then renamed ``classifiers.imported/``."""
    if db.meta("classifiers_imported") == "1":
        return 0
    paths = (
        sorted(p for p in directory.glob("*.json") if NAME_RE.fullmatch(p.stem)) if directory.is_dir() else []
    )
    imported = 0
    with db.tx() as conn:
        for path in paths:
            data = _read_json(path)
            if data is None:
                continue
            name = path.stem
            if conn.execute(select(_C.c.name).where(_C.c.name == name)).first() is not None:
                log.warning(
                    "classifiers: %s already exists in the database; not importing %s", name, path.name
                )
                continue
            now = time.time()
            created = _parse_iso(data.get("created_at")) or now
            updated = _parse_iso(data.get("updated_at")) or created
            conn.execute(
                insert(_C).values(
                    name=name, owner=None, latest_version=1, created_at=created, updated_at=updated
                )
            )
            conn.execute(
                insert(_V).values(
                    name=name,
                    version=1,
                    definition=_dumps(_definition(data)),
                    model=None,
                    created_at=updated,
                    created_by="import",
                )
            )
            audit(conn, "import", "classifier.put", name, {"version": 1, "source": path.name})
            imported += 1
        db.set_meta(conn, "classifiers_imported", "1")
    if directory.is_dir():
        target = directory.with_name("classifiers.imported")
        if target.exists():
            target = directory.with_name(f"classifiers.imported.{int(time.time())}")
        try:
            directory.rename(target)
            log.info(
                "classifiers: imported %d of %d JSON file(s); moved them to %s", imported, len(paths), target
            )
        except OSError as exc:
            log.warning(
                "classifiers: imported %d file(s) but could not rename %s: %s", imported, directory, exc
            )
    return imported


def export_file(doc: dict[str, Any]) -> dict[str, Any]:
    """The v3.1 file format (+ ``version``) for ``clef classifiers export``."""
    return {k: v for k, v in doc.items() if k not in ("owner", "model")}


__all__ = [
    "ClassifierStore",
    "ClassifierVersions",
    "InvalidName",
    "NAME_RE",
    "TooManyClassifiers",
    "import_json_dir",
    "validate_name",
]
