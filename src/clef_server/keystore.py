"""Principals, hashed API keys, the in-memory key cache and daily record quotas.

See docs/ARCHITECTURE.md ("Keys and principals"). Shared by the server and ``clef keys`` (which works on the
database directly, with the server off). Only ``sha256(key)`` is stored; a secret is shown once, at create /
rotate. The request path never touches the database: keys come from :class:`KeyCache`, usage from
:class:`Quotas` (in memory, flushed every few seconds).
"""

from __future__ import annotations

import asyncio
import getpass
import hashlib
import json
import logging
import secrets
import threading
import time
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any

from sqlalchemy import case, insert, select, update
from sqlalchemy.engine import Connection

from . import dbschema as t
from .config import KEY_NAME_RE
from .db import Database, audit

log = logging.getLogger("clef")

KEY_PREFIX = "clef_live_"
_ID_CHARS = "abcdefghijklmnopqrstuvwxyz234567"
_ID_LEN = 12
_KEY_LEN = len(KEY_PREFIX) + _ID_LEN + 1 + 43  # 66
MAX_ACTIVE_KEYS = 5
DEFAULT_GRACE_S = 24 * 3600
KEYS_GEN = "keys_gen"  # clef_meta counter bumped by every write to api_principals / api_keys

ALL_SCOPES = frozenset({"classify", "jobs", "classifiers", "ops", "admin", "urgent"})


# ------------------------------------------------------------------ key format


def generate_key() -> tuple[str, str]:
    """``(key_id, key)``: ``clef_live_<12 chars [a-z2-7]>_<43 chars token_urlsafe(32)>``."""
    key_id = "".join(secrets.choice(_ID_CHARS) for _ in range(_ID_LEN))
    return key_id, f"{KEY_PREFIX}{key_id}_{secrets.token_urlsafe(32)}"


def parse_key(key: str) -> str | None:
    """The public id of a ``clef_live_`` key, by fixed offsets (the secret may contain ``_``), else None."""
    if len(key) != _KEY_LEN or not key.startswith(KEY_PREFIX):
        return None
    start = len(KEY_PREFIX)
    key_id = key[start : start + _ID_LEN]
    if key[start + _ID_LEN] != "_" or any(c not in _ID_CHARS for c in key_id):
        return None
    return key_id


def hash_key(key: str) -> str:
    return hashlib.sha256(key.encode()).hexdigest()


def display_key(key_id: str) -> str:
    return f"{KEY_PREFIX}{key_id}_…"


def parse_scopes(scopes: Any) -> frozenset[str]:
    """A list or a comma-separated string -> a validated scope set (KeystoreError 400 otherwise)."""
    items = scopes.split(",") if isinstance(scopes, str) else list(scopes or [])
    out = frozenset(str(s).strip() for s in items if str(s).strip())
    unknown = sorted(out - ALL_SCOPES)
    if unknown:
        raise KeystoreError(400, f"unknown scope {unknown[0]!r} (scopes: {', '.join(sorted(ALL_SCOPES))})")
    if not out:
        raise KeystoreError(400, "scopes: give at least one scope")
    return out


# ------------------------------------------------------------------ types


class KeystoreError(Exception):
    """A refused key operation; ``status`` is the HTTP status the admin API answers with."""

    def __init__(self, status: int, message: str):
        super().__init__(message)
        self.status = status


@dataclass(frozen=True)
class Principal:
    name: str
    source: str  # "env" | "db"
    scopes: frozenset[str]  # env: ALL_SCOPES
    daily_quota: int | None  # env: None
    rate_limit: int | None  # None = CLEF_RATE_LIMIT
    key_id: str | None  # env: None
    expires_at: float | None

    def has(self, scope: str) -> bool:
        return "admin" in self.scopes or scope in self.scopes


@dataclass(frozen=True)
class KeyRow:
    id: str
    principal: str
    secret_sha256: str
    expires_at: float | None
    revoked_at: float | None


@dataclass(frozen=True)
class PrincipalRow:
    name: str
    scopes: frozenset[str]
    daily_quota: int | None
    rate_limit: int | None
    disabled_at: float | None


@dataclass(frozen=True)
class Actor:
    """Who a write is audited as: a principal name (``cli:<os user>`` for the CLI)."""

    name: str
    ip: str | None = None
    request_id: str | None = None


def cli_actor() -> Actor:
    try:
        user = getpass.getuser()
    except Exception:  # no passwd entry (some containers)
        user = "unknown"
    return Actor(f"cli:{user}")


def env_principal(name: str) -> Principal:
    """Env keys are full admin with no quota and the global rate limit (break-glass + bootstrap)."""
    return Principal(name, "env", ALL_SCOPES, None, None, None, None)


def key_status(row: dict[str, Any], now: float) -> str:
    if row["revoked_at"] is not None:
        return "revoked"
    exp = row["expires_at"]
    if exp is None:
        return "active"
    return "expired" if exp <= now else "expiring"


def _split_scopes(raw: str) -> frozenset[str]:
    return frozenset(s for s in raw.split(",") if s)


# ------------------------------------------------------------------ store


class KeyStore:
    """CRUD on ``api_principals`` / ``api_keys`` (every write bumps ``keys_gen`` and writes an audit row)."""

    def __init__(self, db: Database):
        self.db = db

    # ---- reads
    def load_all(self) -> tuple[dict[str, PrincipalRow], dict[str, KeyRow]]:
        with self.db.read() as conn:
            prows = conn.execute(select(t.api_principals)).all()
            krows = conn.execute(select(t.api_keys)).all()
        principals = {
            r.name: PrincipalRow(r.name, _split_scopes(r.scopes), r.daily_quota, r.rate_limit, r.disabled_at)
            for r in prows
        }
        keys = {r.id: KeyRow(r.id, r.principal, r.secret_sha256, r.expires_at, r.revoked_at) for r in krows}
        return principals, keys

    def generation(self) -> str | None:
        return self.db.meta(KEYS_GEN)

    def list_principals(self, include_disabled: bool = False) -> list[dict[str, Any]]:
        now = time.time()
        with self.db.read() as conn:
            where = [] if include_disabled else [t.api_principals.c.disabled_at.is_(None)]
            prows = conn.execute(
                select(t.api_principals).where(*where).order_by(t.api_principals.c.name)
            ).all()
            keys = self._keys_by_principal(conn)
        return [self._view(r, keys.get(r.name, []), now) for r in prows]

    def get(self, name: str) -> dict[str, Any] | None:
        with self.db.read() as conn:
            return self._get(conn, name)

    def _get(self, conn: Connection, name: str) -> dict[str, Any] | None:
        row = conn.execute(select(t.api_principals).where(t.api_principals.c.name == name)).first()
        if row is None:
            return None
        keys = self._keys_by_principal(conn, name)
        return self._view(row, keys.get(name, []), time.time())

    @staticmethod
    def _keys_by_principal(conn: Connection, name: str | None = None) -> dict[str, list[Any]]:
        q = select(t.api_keys).order_by(t.api_keys.c.created_at)
        if name is not None:
            q = q.where(t.api_keys.c.principal == name)
        out: dict[str, list[Any]] = defaultdict(list)
        for r in conn.execute(q):
            out[r.principal].append(r)
        return out

    @staticmethod
    def _view(row: Any, keys: list[Any], now: float) -> dict[str, Any]:
        kviews = []
        for k in keys:
            d = {
                "id": k.id,
                "display": display_key(k.id),
                "created_at": k.created_at,
                "expires_at": k.expires_at,
                "revoked_at": k.revoked_at,
                "last_used_at": k.last_used_at,
            }
            d["status"] = key_status(d, now)
            kviews.append(d)
        return {
            "name": row.name,
            "source": "db",
            "scopes": sorted(_split_scopes(row.scopes)),
            "daily_quota": row.daily_quota,
            "rate_limit": row.rate_limit,
            "note": row.note,
            "created_at": row.created_at,
            "created_by": row.created_by,
            "disabled_at": row.disabled_at,
            "keys": kviews,
        }

    # ---- writes
    def create_principal(
        self,
        name: str,
        scopes: Any,
        *,
        actor: Actor,
        daily_quota: int | None = None,
        rate_limit: int | None = None,
        note: str | None = None,
        expires_at: float | None = None,
        env_names: frozenset[str] | set[str] = frozenset(),
    ) -> tuple[dict[str, Any], str]:
        """A new principal and its first key: ``(view, secret)``. KeystoreError on a bad or taken name."""
        if not KEY_NAME_RE.match(name or ""):
            raise KeystoreError(400, "name: letters, digits, _ . -, 1 to 64 characters")
        scope_set = parse_scopes(scopes)
        _check_numbers(daily_quota, rate_limit)
        if name in env_names:
            raise KeystoreError(409, f"{name!r} is the name of an env key (CLEF_API_KEY / CLEF_API_KEYS)")
        key_id, secret = generate_key()
        now = time.time()
        with self.db.tx() as conn:
            self.db.serialize(conn, "keys")
            found = conn.execute(
                select(t.api_principals.c.disabled_at).where(t.api_principals.c.name == name)
            ).first()
            if found is not None:
                what = "was disabled (names are never reused)" if found.disabled_at else "already exists"
                raise KeystoreError(409, f"principal {name!r} {what}")
            conn.execute(
                insert(t.api_principals).values(
                    name=name,
                    scopes=",".join(sorted(scope_set)),
                    daily_quota=daily_quota,
                    rate_limit=rate_limit,
                    note=note,
                    created_at=now,
                    created_by=actor.name,
                )
            )
            self._insert_key(conn, key_id, name, secret, now, actor.name, expires_at)
            self.db.bump(conn, KEYS_GEN)
            audit(
                conn,
                actor.name,
                "key.create",
                name,
                {
                    "key_id": key_id,
                    "scopes": sorted(scope_set),
                    "daily_quota": daily_quota,
                    "rate_limit": rate_limit,
                    "expires_at": expires_at,
                },
                actor.ip,
                actor.request_id,
            )
            return self._get(conn, name) or {}, secret

    @staticmethod
    def _insert_key(
        conn: Connection,
        key_id: str,
        principal: str,
        secret: str,
        now: float,
        created_by: str,
        expires_at: float | None,
    ) -> None:
        conn.execute(
            insert(t.api_keys).values(
                id=key_id,
                principal=principal,
                secret_sha256=hash_key(secret),
                created_at=now,
                created_by=created_by,
                expires_at=expires_at,
            )
        )

    def update_principal(self, name: str, changes: dict[str, Any], *, actor: Actor) -> dict[str, Any]:
        """Apply scopes, daily_quota, rate_limit, note; a key present with None clears it (not scopes)."""
        values: dict[str, Any] = {}
        if "scopes" in changes:
            values["scopes"] = ",".join(sorted(parse_scopes(changes["scopes"])))
        if "daily_quota" in changes or "rate_limit" in changes:
            _check_numbers(changes.get("daily_quota"), changes.get("rate_limit"))
        for k in ("daily_quota", "rate_limit", "note"):
            if k in changes:
                values[k] = changes[k]
        with self.db.tx() as conn:
            self.db.serialize(conn, "keys")
            row = self._principal_row(conn, name)
            if row.disabled_at is not None:
                raise KeystoreError(409, f"principal {name!r} is disabled")
            if values:
                conn.execute(update(t.api_principals).where(t.api_principals.c.name == name).values(**values))
                self.db.bump(conn, KEYS_GEN)
            detail = {k: (v.split(",") if k == "scopes" else v) for k, v in values.items()}
            audit(conn, actor.name, "principal.update", name, detail, actor.ip, actor.request_id)
            return self._get(conn, name) or {}

    @staticmethod
    def _principal_row(conn: Connection, name: str) -> Any:
        row = conn.execute(select(t.api_principals).where(t.api_principals.c.name == name)).first()
        if row is None:
            raise KeystoreError(404, f"principal {name!r} not found")
        return row

    def rotate(
        self,
        name: str,
        *,
        actor: Actor,
        grace_s: float = DEFAULT_GRACE_S,
        expires_at: float | None = None,
    ) -> tuple[dict[str, Any], str]:
        """A new key; the principal's other active keys expire at ``min(current, now + grace_s)``."""
        if grace_s < 0:
            raise KeystoreError(400, "grace_s must be >= 0")
        key_id, secret = generate_key()
        now = time.time()
        with self.db.tx() as conn:
            self.db.serialize(conn, "keys")
            row = self._principal_row(conn, name)
            if row.disabled_at is not None:
                raise KeystoreError(409, f"principal {name!r} is disabled")
            active = [
                k
                for k in conn.execute(select(t.api_keys).where(t.api_keys.c.principal == name))
                if key_status({"revoked_at": k.revoked_at, "expires_at": k.expires_at}, now)
                in ("active", "expiring")
            ]
            if len(active) >= MAX_ACTIVE_KEYS:
                raise KeystoreError(
                    409, f"principal {name!r} already has {MAX_ACTIVE_KEYS} active keys; revoke one first"
                )
            cutoff = now + grace_s
            for k in active:
                if k.expires_at is None or k.expires_at > cutoff:
                    conn.execute(update(t.api_keys).where(t.api_keys.c.id == k.id).values(expires_at=cutoff))
            self._insert_key(conn, key_id, name, secret, now, actor.name, expires_at)
            self.db.bump(conn, KEYS_GEN)
            audit(
                conn,
                actor.name,
                "key.rotate",
                name,
                {"key_id": key_id, "grace_s": grace_s, "rotated": [k.id for k in active]},
                actor.ip,
                actor.request_id,
            )
            return self._get(conn, name) or {}, secret

    def revoke(self, name: str, *, actor: Actor, key_id: str | None = None) -> dict[str, Any]:
        """One key (``key_id``), or the whole principal: disabled, every key revoked. Idempotent."""
        now = time.time()
        with self.db.tx() as conn:
            self.db.serialize(conn, "keys")
            row = self._principal_row(conn, name)
            if key_id is not None:
                key = conn.execute(
                    select(t.api_keys).where(t.api_keys.c.id == key_id, t.api_keys.c.principal == name)
                ).first()
                if key is None:
                    raise KeystoreError(404, f"key {key_id!r} not found for principal {name!r}")
                if key.revoked_at is None:
                    conn.execute(update(t.api_keys).where(t.api_keys.c.id == key_id).values(revoked_at=now))
                    self.db.bump(conn, KEYS_GEN)
                    audit(
                        conn, actor.name, "key.revoke", name, {"key_id": key_id}, actor.ip, actor.request_id
                    )
            elif row.disabled_at is None:
                ids = list(
                    conn.execute(
                        select(t.api_keys.c.id).where(
                            t.api_keys.c.principal == name, t.api_keys.c.revoked_at.is_(None)
                        )
                    ).scalars()
                )
                conn.execute(
                    update(t.api_keys)
                    .where(t.api_keys.c.principal == name, t.api_keys.c.revoked_at.is_(None))
                    .values(revoked_at=now)
                )
                conn.execute(
                    update(t.api_principals).where(t.api_principals.c.name == name).values(disabled_at=now)
                )
                self.db.bump(conn, KEYS_GEN)
                audit(
                    conn, actor.name, "principal.disable", name, {"revoked": ids}, actor.ip, actor.request_id
                )
            return self._get(conn, name) or {}

    def touch(self, used: dict[str, float]) -> None:
        """``last_used_at`` flush: no ``keys_gen`` bump (it would reload every cache each time)."""
        if not used:
            return
        with self.db.tx() as conn:
            for key_id, ts in used.items():
                conn.execute(update(t.api_keys).where(t.api_keys.c.id == key_id).values(last_used_at=ts))

    def audit_entries(
        self, limit: int = 100, before: int | None = None, action: str | None = None
    ) -> tuple[list[dict[str, Any]], int | None]:
        """Newest first. ``(entries, next_before)``; ``next_before`` is None on the last page."""
        a = t.audit_log
        q = select(a).order_by(a.c.id.desc()).limit(limit + 1)
        if before is not None:
            q = q.where(a.c.id < before)
        if action:
            escaped = action.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
            q = q.where(a.c.action.like(escaped + "%", escape="\\"))
        with self.db.read() as conn:
            rows = conn.execute(q).all()
        entries = [
            {
                "id": r.id,
                "ts": r.ts,
                "actor": r.actor,
                "action": r.action,
                "target": r.target,
                "detail": json.loads(r.detail) if r.detail else None,
                "ip": r.ip,
                "request_id": r.request_id,
            }
            for r in rows[:limit]
        ]
        return entries, (entries[-1]["id"] if len(rows) > limit else None)


def _check_numbers(daily_quota: int | None, rate_limit: int | None) -> None:
    if daily_quota is not None and daily_quota < 1:
        raise KeystoreError(400, "daily_quota must be >= 1 (omit it for unlimited)")
    if rate_limit is not None and rate_limit < 1:
        raise KeystoreError(400, "rate_limit must be >= 1 (omit it for the global limit)")


# ------------------------------------------------------------------ key cache


class KeyCache:
    """Every key and principal in memory; the auth path does no database I/O.

    ``reload()`` is a full reload (a few hundred rows). ``run()`` polls ``clef_meta.keys_gen`` and reloads
    when it changed (CLI writes apply within ``refresh_s``; the admin API calls ``reload()`` itself, so its
    changes apply at once). A failed reload keeps the last good state.
    """

    FULL_RELOAD_S = 60.0

    def __init__(self, db: Database, refresh_s: float = 10.0):
        self.db, self.refresh_s = db, refresh_s
        self.store = KeyStore(db)
        self._keys: dict[str, tuple[KeyRow, PrincipalRow]] = {}
        self._any = False  # sticky: once a principal existed, auth never turns off by itself
        self._gen: str | None = None
        self._touched: dict[str, float] = {}
        self.names: frozenset[str] = frozenset()  # every principal name (disabled too)

    def get(self, key_id: str) -> tuple[KeyRow, PrincipalRow] | None:
        return self._keys.get(key_id)

    def any_principal(self) -> bool:
        return self._any

    def touch(self, key_id: str) -> None:
        self._touched[key_id] = time.time()

    def reload(self) -> None:
        gen = self.store.generation()
        principals, keys = self.store.load_all()
        self._keys = {k.id: (k, principals[k.principal]) for k in keys.values() if k.principal in principals}
        self._any = self._any or bool(principals)
        self.names = frozenset(principals)
        self._gen = gen

    def flush_used(self) -> None:
        used, self._touched = self._touched, {}
        try:
            self.store.touch(used)
        except Exception:
            log.warning("keys: could not flush last_used_at", exc_info=True)
            for k, v in used.items():
                self._touched.setdefault(k, v)

    async def run(self) -> None:
        last_full = time.monotonic()
        while True:
            await asyncio.sleep(self.refresh_s)
            try:
                gen = await asyncio.to_thread(self.store.generation)
                if gen != self._gen or time.monotonic() - last_full >= self.FULL_RELOAD_S:
                    await asyncio.to_thread(self.reload)
                    last_full = time.monotonic()
                await asyncio.to_thread(self.flush_used)
            except asyncio.CancelledError:
                raise
            except Exception:
                log.warning("keys: reload failed; keeping the last good state", exc_info=True)


# ------------------------------------------------------------------ quotas


class QuotaExceeded(Exception):
    def __init__(self, limit: int, remaining: int, retry_after: int):
        super().__init__(f"daily quota exceeded ({limit} records/day)")
        self.limit, self.remaining, self.retry_after = limit, remaining, retry_after


@dataclass(frozen=True)
class Reservation:
    principal: str
    day: str
    n: int


class Quotas:
    """Daily record counters per principal and UTC day. In memory and authoritative (one server per DB).

    ``base`` = what the database holds, ``delta`` = changes not flushed yet; ``flush()`` (every 5 s and at
    shutdown) upserts the deltas, so a crash loses at most 5 s of counts. DB principals without a quota are
    counted too, never refused. Env principals and auth off: no reservation (``reserve`` returns None).
    """

    FLUSH_S = 5.0

    def __init__(self, db: Database, clock: Callable[[], float] = time.time):
        self.db, self.clock = db, clock
        self._lock = threading.Lock()
        self._base: dict[tuple[str, str], list[int]] = {}  # (principal, day) -> [records, requests, rejected]
        self._delta: dict[tuple[str, str], list[int]] = defaultdict(lambda: [0, 0, 0])

    # ---- time
    def today(self) -> str:
        return datetime.fromtimestamp(self.clock(), timezone.utc).strftime("%Y-%m-%d")

    def retry_after(self) -> int:
        now = datetime.fromtimestamp(self.clock(), timezone.utc)
        midnight = (now + timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
        return max(1, int((midnight - now).total_seconds()) + 1)

    # ---- counters
    def _used(self, key: tuple[str, str]) -> int:
        base = self._base.get(key, (0,))[0]
        delta = self._delta[key][0] if key in self._delta else 0
        return max(0, base + delta)

    def used(self, name: str, day: str | None = None) -> int:
        with self._lock:
            return self._used((name, day or self.today()))

    def reserve(self, principal: Principal | None, n: int) -> Reservation | None:
        if principal is None or principal.source != "db":
            return None
        key = (principal.name, self.today())
        with self._lock:
            row = self._delta[key]
            row[1] += 1
            limit = principal.daily_quota
            if limit is not None:
                used = self._used(key)
                if used + n > limit:
                    row[2] += 1
                    raise QuotaExceeded(limit, max(0, limit - used), self.retry_after())
            row[0] += n
        return Reservation(principal.name, key[1], n)

    def refund(self, reservation: Reservation, n: int) -> None:
        n = min(n, reservation.n)
        if n > 0:
            with self._lock:
                self._delta[(reservation.principal, reservation.day)][0] -= n

    def reject(self, name: str) -> None:
        with self._lock:
            self._delta[(name, self.today())][2] += 1

    def remaining(self, principal: Principal | None) -> tuple[int, int] | None:
        if principal is None or principal.source != "db" or principal.daily_quota is None:
            return None
        return principal.daily_quota, max(0, principal.daily_quota - self.used(principal.name))

    # ---- persistence
    def load(self) -> None:
        """Today's rows from the database (called at startup)."""
        with self.db.read() as conn:
            rows = conn.execute(select(t.api_usage).where(t.api_usage.c.day == self.today())).all()
        with self._lock:
            for r in rows:
                self._base[(r.principal, r.day)] = [r.records, r.requests, r.rejected]

    def flush(self) -> None:
        with self._lock:
            snap = {k: list(v) for k, v in self._delta.items() if any(v)}
        if not snap:
            return
        usage = t.api_usage
        with self.db.tx() as conn:  # per row: UPDATE, else INSERT (few rows; one writer, so no race)
            for (name, day), (records, requests, rejected) in snap.items():
                grown = usage.c.records + records
                done = conn.execute(
                    update(usage)
                    .where(usage.c.principal == name, usage.c.day == day)
                    .values(
                        records=case((grown < 0, 0), else_=grown),  # a refund never goes below zero
                        requests=usage.c.requests + requests,
                        rejected=usage.c.rejected + rejected,
                    )
                )
                if done.rowcount == 0:
                    conn.execute(
                        insert(usage).values(
                            principal=name,
                            day=day,
                            records=max(0, records),
                            requests=requests,
                            rejected=rejected,
                        )
                    )
        with self._lock:  # committed: move what was flushed from delta to base
            for key, v in snap.items():
                d, b = self._delta[key], self._base.setdefault(key, [0, 0, 0])
                for i in range(3):
                    d[i] -= v[i]
                    b[i] += v[i]
                b[0] = max(0, b[0])

    def history(self, name: str, days: int) -> list[dict[str, Any]]:
        """The last ``days`` UTC days, oldest first, zero-filled, with unflushed counts merged in."""
        end = datetime.fromtimestamp(self.clock(), timezone.utc).date()
        wanted = [(end - timedelta(days=i)).isoformat() for i in range(days - 1, -1, -1)]
        with self.db.read() as conn:
            rows = conn.execute(
                select(t.api_usage).where(t.api_usage.c.principal == name, t.api_usage.c.day.in_(wanted))
            ).all()
        stored = {r.day: [r.records, r.requests, r.rejected] for r in rows}
        out = []
        with self._lock:
            for day in wanted:
                key = (name, day)
                base = self._base[key] if key in self._base else stored.get(day, [0, 0, 0])
                d = self._delta.get(key, [0, 0, 0])
                records, requests, rejected = (max(0, base[i] + d[i]) for i in range(3))
                out.append({"day": day, "records": records, "requests": requests, "rejected": rejected})
        return out

    async def run(self) -> None:
        while True:
            await asyncio.sleep(self.FLUSH_S)
            try:
                await asyncio.to_thread(self.flush)
            except asyncio.CancelledError:
                raise
            except Exception:
                log.warning("quotas: could not flush usage (will retry)", exc_info=True)


__all__ = [
    "ALL_SCOPES",
    "Actor",
    "KeyCache",
    "KeyStore",
    "KeystoreError",
    "Principal",
    "QuotaExceeded",
    "Quotas",
    "Reservation",
]
