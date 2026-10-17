"""The database layer: engine, connections, transactions, migrations and URL handling.

SQLite (default, ``<state_dir>/clef.db``) or Postgres (``CLEF_DATABASE_URL``), through SQLAlchemy Core 2.0
(sync; every call runs in a worker thread) and Alembic (``migrations/``; startup runs ``upgrade heads``).
One clef server per database (D3): on Postgres the server holds an advisory lock for its lifetime.

Stores (``JobStore``, ``ClassifierStore``, ...) take a :class:`Database` and use ``tx()`` / ``read()``; no SQL
lives outside the stores and this module. See docs/ARCHITECTURE.md (Storage).
"""

from __future__ import annotations

import contextlib
import getpass
import importlib.util
import json
import logging
import os
import sqlite3
import threading
import time
import zlib
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from sqlalchemy import create_engine, event, func, insert, inspect, select, update
from sqlalchemy.engine import Connection, Engine, make_url
from sqlalchemy.exc import ArgumentError, OperationalError
from sqlalchemy.exc import DatabaseError as SADatabaseError
from sqlalchemy.pool import QueuePool

from . import dbschema as t

log = logging.getLogger("clef")

PG_LOCK_KEY = 1667032422  # pg_try_advisory_lock key held by the one server on a database
MIGRATIONS_DIR = Path(__file__).with_name("migrations")
_NEEDS_PSYCOPG = (
    "CLEF_DATABASE_URL is Postgres but psycopg is not installed: pip install 'clef-local[postgres]'"
)
_LEGACY_SCHEMA = """
CREATE TABLE IF NOT EXISTS items (
    job_id TEXT NOT NULL REFERENCES jobs(id) ON DELETE CASCADE,
    idx INTEGER NOT NULL,
    data TEXT NOT NULL,
    PRIMARY KEY (job_id, idx)
) WITHOUT ROWID;
CREATE INDEX IF NOT EXISTS jobs_status ON jobs(status, seq);
CREATE UNIQUE INDEX IF NOT EXISTS jobs_idem ON jobs(COALESCE(owner, ''), idem_key) WHERE idem_key IS NOT NULL;
"""


class DatabaseError(RuntimeError):
    """Bad URL, missing driver, second server on the same DB, failed upgrade (-> 503 / startup failure)."""


def is_unavailable(exc: BaseException) -> bool:
    """The errors that mean "database unavailable" (HTTP 503) rather than a bug."""
    return isinstance(exc, (DatabaseError, OperationalError))


# ------------------------------------------------------------------ URLs


def normalise_url(url: str) -> str:
    """``postgres://`` / ``postgresql://`` -> ``postgresql+psycopg://``; anything else is returned as is."""
    url = url.strip()
    for prefix in ("postgres://", "postgresql://"):
        if url.startswith(prefix):
            return "postgresql+psycopg://" + url[len(prefix) :]
    return url


def sqlite_url(path: str | Path) -> str:
    return "sqlite:///" + Path(path).expanduser().absolute().as_posix()


def database_url(cfg: Any) -> str:
    """``CLEF_DATABASE_URL`` (normalised), else ``sqlite:///<state_dir>/clef.db``."""
    from .paths import database_path

    raw = getattr(cfg, "database_url", None)
    return normalise_url(raw) if raw else sqlite_url(database_path(cfg))


def redact_url(url: str) -> str:
    """``postgresql+psycopg://user:***@host:5432/clef``: safe to log and to show in /health."""
    try:
        return make_url(url).render_as_string(hide_password=True)
    except Exception:
        return url.split("://", 1)[0] + "://<unparsable>" if "://" in url else "<unparsable>"


def code_version() -> str:
    """The head revision(s) this code expects, "+"-joined (e.g. "0002")."""
    from alembic.script import ScriptDirectory

    return "+".join(sorted(ScriptDirectory(str(MIGRATIONS_DIR)).get_heads()))


def known_revisions() -> set[str]:
    from alembic.script import ScriptDirectory

    return {s.revision for s in ScriptDirectory(str(MIGRATIONS_DIR)).walk_revisions()}


# ------------------------------------------------------------------ the database


class Database:
    """One SQLAlchemy engine plus the transaction discipline the stores rely on.

    ``tx()`` is the write transaction: on SQLite it holds a process RLock and runs ``BEGIN IMMEDIATE`` (one
    writer at a time, as JobStore always did); on Postgres it is a plain ``BEGIN``. Nested ``tx()`` /
    ``read()`` calls on the same thread join the open transaction (same connection: they see its writes).
    """

    def __init__(self, url: str, pool_size: int = 5) -> None:
        self.url = normalise_url(url)
        self.display_url = redact_url(self.url)
        try:
            parsed = make_url(self.url)
        except ArgumentError as exc:
            raise DatabaseError(f"CLEF_DATABASE_URL is not a valid database URL: {self.display_url}") from exc
        self.dialect = parsed.get_backend_name()
        if pool_size < 1:
            raise DatabaseError("CLEF_DATABASE_POOL_SIZE must be >= 1")
        self.path: Path | None = None
        self._lock = threading.RLock()
        self._local = threading.local()
        self._pg_lock: Connection | None = None
        self._version: str | None = None
        if self.dialect == "sqlite":
            if not parsed.database or parsed.database == ":memory:":
                raise DatabaseError("SQLite needs a file path (sqlite:///path/to/clef.db), not :memory:")
            self.path = Path(parsed.database)
            self.path.parent.mkdir(parents=True, exist_ok=True)
            if not self.path.exists():
                with contextlib.suppress(FileExistsError):  # private by default: payloads hold user data
                    os.close(os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600))
            self.engine: Engine = create_engine(
                self.url,
                poolclass=QueuePool,
                pool_size=pool_size,
                max_overflow=pool_size,
                connect_args={"timeout": 30, "check_same_thread": False},
            )
            self._sqlite_events()
        elif self.dialect == "postgresql":
            if parsed.get_driver_name() == "psycopg" and importlib.util.find_spec("psycopg") is None:
                raise DatabaseError(_NEEDS_PSYCOPG)
            try:
                self.engine = create_engine(
                    self.url, pool_size=pool_size, max_overflow=pool_size, pool_pre_ping=True
                )
            except ImportError as exc:
                raise DatabaseError(_NEEDS_PSYCOPG) from exc
        else:
            raise DatabaseError(f"unsupported database {self.dialect!r} (use sqlite:/// or postgresql://)")

    def _sqlite_events(self) -> None:
        @event.listens_for(self.engine, "connect")
        def _connect(dbapi_conn: Any, _record: Any) -> None:
            dbapi_conn.isolation_level = None  # we emit BEGIN ourselves (below)
            cur = dbapi_conn.cursor()
            with contextlib.suppress(sqlite3.Error):
                cur.execute("PRAGMA journal_mode=WAL")
            cur.execute("PRAGMA synchronous=NORMAL")
            cur.execute("PRAGMA foreign_keys=ON")
            cur.execute("PRAGMA busy_timeout=30000")
            cur.close()

        @event.listens_for(self.engine, "begin")
        def _begin(conn: Connection) -> None:
            write = conn.get_execution_options().get("clef_write", False)
            conn.exec_driver_sql("BEGIN IMMEDIATE" if write else "BEGIN")

    # ---- transactions
    @contextlib.contextmanager
    def tx(self) -> Iterator[Connection]:
        """Write transaction: everything inside commits together, or nothing does."""
        current = getattr(self._local, "conn", None)
        if current is not None:
            yield current
            return
        lock = self._lock if self.dialect == "sqlite" else contextlib.nullcontext()
        with lock, self.engine.connect() as conn:
            conn.execution_options(clef_write=True)
            with conn.begin():
                self._local.conn = conn
                try:
                    yield conn
                finally:
                    self._local.conn = None

    @contextlib.contextmanager
    def read(self) -> Iterator[Connection]:
        """Read-only transaction (no write lock); joins an open ``tx()`` on this thread."""
        current = getattr(self._local, "conn", None)
        if current is not None:
            yield current
            return
        with self.engine.connect() as conn, conn.begin():
            self._local.conn = conn
            try:
                yield conn
            finally:
                self._local.conn = None

    def serialize(self, conn: Connection, name: str) -> None:
        """Inside ``tx()``: serialise check-then-write sections (counts, limits) across connections.

        SQLite's ``BEGIN IMMEDIATE`` already does; Postgres takes a transaction-scoped advisory lock.
        """
        if self.dialect == "postgresql":
            conn.execute(select(func.pg_advisory_xact_lock(zlib.crc32(name.encode()) & 0x7FFFFFFF)))

    # ---- migrations
    def _alembic_config(self, conn: Connection) -> Any:
        from alembic.config import Config as AlembicConfig

        acfg = AlembicConfig()
        acfg.set_main_option("script_location", str(MIGRATIONS_DIR))
        acfg.attributes["connection"] = conn
        return acfg

    def upgrade(self) -> None:
        """``alembic upgrade heads`` in one transaction; a legacy jobs.db is stamped ``0001`` first."""
        from alembic import command

        try:
            with self.tx() as conn:
                tables = set(inspect(conn).get_table_names())
                if "jobs" in tables and "alembic_version" not in tables:
                    self._legacy_baseline(conn)
                    command.stamp(self._alembic_config(conn), "0001")
                command.upgrade(self._alembic_config(conn), "heads")
        except DatabaseError:
            raise
        except SADatabaseError as exc:
            raise DatabaseError(f"database upgrade failed on {self.display_url}: {exc.orig}") from exc
        self._version = None
        self._version = self.schema_version()

    def _legacy_baseline(self, conn: Connection) -> None:
        """A v3.1 jobs.db -> exactly the 0001 shape (the idem_key ALTER, then whatever else is missing)."""
        if self.dialect != "sqlite":
            raise DatabaseError("a 'jobs' table without alembic_version is only expected in a SQLite jobs.db")
        cols = {c["name"] for c in inspect(conn).get_columns("jobs")}
        if "idem_key" not in cols:  # database created before Idempotency-Key support
            conn.exec_driver_sql("ALTER TABLE jobs ADD COLUMN idem_key TEXT")
        for stmt in _LEGACY_SCHEMA.split(";"):
            if stmt.strip():
                conn.exec_driver_sql(stmt)
        log.info("database: legacy jobs schema found; stamped as revision 0001")

    def schema_version(self) -> str:
        """The database's alembic revision(s), "+"-joined; "" when it was never migrated."""
        if self._version is not None:
            return self._version
        from alembic.runtime.migration import MigrationContext

        with self.read() as conn:
            heads = MigrationContext.configure(conn).get_current_heads()
        return "+".join(sorted(heads))

    def info(self) -> dict[str, Any]:
        """For /health: no I/O (the version is cached by ``upgrade``)."""
        return {"dialect": self.dialect, "schema_version": self._version, "url": self.display_url}

    # ---- clef_meta
    def bump(self, conn: Connection, key: str) -> int:
        """Inside the caller's tx: the ``clef_meta`` counter ``key`` += 1; returns the new value."""
        self.serialize(conn, f"meta:{key}")
        row = conn.execute(select(t.clef_meta.c.v).where(t.clef_meta.c.k == key)).first()
        value = (int(row.v) if row else 0) + 1
        if row is None:
            conn.execute(insert(t.clef_meta).values(k=key, v=str(value)))
        else:
            conn.execute(update(t.clef_meta).where(t.clef_meta.c.k == key).values(v=str(value)))
        return value

    def set_meta(self, conn: Connection, key: str, value: str) -> None:
        if conn.execute(select(t.clef_meta.c.k).where(t.clef_meta.c.k == key)).first() is None:
            conn.execute(insert(t.clef_meta).values(k=key, v=value))
        else:
            conn.execute(update(t.clef_meta).where(t.clef_meta.c.k == key).values(v=value))

    def meta(self, key: str) -> str | None:
        with self.read() as conn:
            row = conn.execute(select(t.clef_meta.c.v).where(t.clef_meta.c.k == key)).first()
        return None if row is None else row.v

    # ---- one server per database
    def acquire_server_lock(self) -> None:
        """Postgres: hold ``pg_try_advisory_lock`` for the process lifetime (SQLite: nothing to do)."""
        if self.dialect != "postgresql" or self._pg_lock is not None:
            return
        try:
            conn = self.engine.connect().execution_options(isolation_level="AUTOCOMMIT")
        except OperationalError as exc:
            raise DatabaseError(f"cannot connect to {self.display_url}: {exc.orig}") from exc
        if not conn.execute(select(func.pg_try_advisory_lock(PG_LOCK_KEY))).scalar():
            conn.close()
            raise DatabaseError("another clef server holds this database (one server per database)")
        self._pg_lock = conn

    def close(self) -> None:
        """Release the server lock and the pool (idempotent; the engine reconnects if used again)."""
        if self._pg_lock is not None:
            with contextlib.suppress(Exception):
                self._pg_lock.close()
            self._pg_lock = None
        self.engine.dispose()


# ------------------------------------------------------------------ startup


def migrate_legacy_file(cfg: Any) -> None:
    """``<state_dir>/jobs.db`` -> ``clef.db`` once (only with the default URL; see ARCHITECTURE Storage)."""
    from .paths import database_path, jobs_db

    new, old = database_path(cfg), jobs_db(cfg)
    if not old.exists():
        return
    if new.exists():
        log.warning(
            "database: both %s and %s exist; using %s (jobs.db is left alone)", new, old.name, new.name
        )
        return
    con = sqlite3.connect(str(old))
    try:
        con.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    finally:
        con.close()
    os.replace(old, new)
    for suffix in ("-wal", "-shm"):
        with contextlib.suppress(FileNotFoundError):
            os.unlink(f"{old}{suffix}")
    log.info("database: renamed %s to %s (one-time migration)", old, new)


def open_database(cfg: Any) -> Database:
    """The server's database: legacy rename -> Database -> PG lock -> upgrade -> classifier JSON import."""
    if not getattr(cfg, "database_url", None):
        migrate_legacy_file(cfg)
    db = Database(database_url(cfg), int(getattr(cfg, "database_pool_size", 5) or 5))
    try:
        db.acquire_server_lock()
        db.upgrade()
        from .classifiers import import_json_dir
        from .paths import state_dir

        import_json_dir(db, state_dir(cfg) / "classifiers")
    except BaseException:
        db.close()
        raise
    log.info("database: %s (schema %s)", db.display_url, db.schema_version())
    return db


def audit(
    conn: Connection,
    actor: str,
    action: str,
    target: str,
    detail: dict[str, Any] | None = None,
    ip: str | None = None,
    request_id: str | None = None,
) -> None:
    """One ``audit_log`` row in the caller's transaction (never a secret, never request content)."""
    conn.execute(
        insert(t.audit_log).values(
            ts=time.time(),
            actor=actor,
            action=action,
            target=target,
            detail=None if detail is None else json.dumps(detail, separators=(",", ":"), default=str),
            ip=ip,
            request_id=request_id,
        )
    )


# ------------------------------------------------------------------ CLI helpers (clef db / backup / doctor)


def peek_version(url: str) -> str | None:
    """The schema version without creating or changing anything: None when a SQLite file does not exist,
    "" when the database was never migrated, "legacy" for a v3.1 jobs schema without alembic_version."""
    url = normalise_url(url)
    parsed = make_url(url)
    if parsed.get_backend_name() == "sqlite":
        path = Path(parsed.database or "")
        if not path.is_file():
            return None
        con = sqlite3.connect(f"file:{path.as_posix()}?mode=ro", uri=True, timeout=5)
        try:
            names = {r[0] for r in con.execute("SELECT name FROM sqlite_master WHERE type='table'")}
            if "alembic_version" not in names:
                return "legacy" if "jobs" in names else ""
            return "+".join(sorted(r[0] for r in con.execute("SELECT version_num FROM alembic_version")))
        finally:
            con.close()
    db = Database(url, pool_size=1)
    try:
        with db.read() as conn:
            if "alembic_version" not in inspect(conn).get_table_names():
                return "legacy" if "jobs" in inspect(conn).get_table_names() else ""
        return db.schema_version()
    except OperationalError as exc:
        raise DatabaseError(f"cannot connect to {db.display_url}: {exc.orig}") from exc
    finally:
        db.close()


def version_state(current: str | None) -> str:
    """ "current" | "behind" (the server upgrades it) | "ahead" (newer than this code) | "missing"."""
    if current is None:
        return "missing"
    if current == code_version():
        return "current"
    if current in ("", "legacy"):
        return "behind"
    known = known_revisions()
    return "behind" if all(rev in known for rev in current.split("+")) else "ahead"


def row_counts(db: Database) -> dict[str, int]:
    with db.read() as conn:
        return {
            tbl.name: conn.execute(select(func.count()).select_from(tbl)).scalar_one()
            for tbl in t.metadata.sorted_tables
        }


def copy_database(source: Database, target: Database, batch: int = 1000) -> dict[str, int]:
    """Every table in FK order, ``batch`` rows at a time, in ONE target transaction (all or nothing).

    The target is upgraded first and must be empty. Postgres identity sequences are reset to MAX(id).
    """
    if source.schema_version() != code_version():
        raise DatabaseError(
            f"the source is at schema {source.schema_version() or 'none'}, clef expects {code_version()}:"
            " run `clef db upgrade` against it first (or start the server on it once)"
        )
    target.upgrade()
    busy = {name: n for name, n in row_counts(target).items() if n}
    if busy:
        raise DatabaseError(f"the target is not empty ({', '.join(f'{k}: {v}' for k, v in busy.items())})")
    copied: dict[str, int] = {}
    with target.tx() as dst:
        for tbl in t.metadata.sorted_tables:
            n = 0
            with source.read() as src:
                result = src.execution_options(yield_per=batch).execute(select(tbl))
                for part in result.partitions(batch):
                    dst.execute(insert(tbl), [dict(r._mapping) for r in part])
                    n += len(part)
            copied[tbl.name] = n
        if target.dialect == "postgresql":
            for tbl in t.metadata.sorted_tables:
                for col in tbl.columns:
                    if col.identity is not None:
                        dst.execute(
                            select(
                                func.setval(
                                    func.pg_get_serial_sequence(tbl.name, col.name),
                                    func.coalesce(func.max(col), 1),
                                    func.max(col).isnot(None),
                                )
                            ).select_from(tbl)
                        )
    return copied


def backup_sqlite(path: Path, out: Path) -> None:
    """The sqlite3 online backup API: a consistent copy while the server keeps writing. ``out`` is 0600."""
    out.parent.mkdir(parents=True, exist_ok=True)
    os.close(os.open(out, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600))
    src = sqlite3.connect(f"file:{path.as_posix()}?mode=ro", uri=True, timeout=30)
    try:
        dst = sqlite3.connect(str(out))
        try:
            src.backup(dst)
        finally:
            dst.close()
    finally:
        src.close()


def cli_actor() -> str:
    """Audit actor for CLI writes: ``cli:<os user>``."""
    try:
        return f"cli:{getpass.getuser()}"
    except Exception:
        return "cli:unknown"


__all__ = [
    "Database",
    "DatabaseError",
    "audit",
    "code_version",
    "database_url",
    "is_unavailable",
    "open_database",
    "redact_url",
]
