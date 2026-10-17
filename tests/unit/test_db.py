"""Storage layer (db.py, dbschema.py, migrations): URLs, transactions, migrations, legacy files, server lock,
claim races. Runs on SQLite, and on Postgres too when CLEF_TEST_DATABASE_URL is set (conftest.py)."""

from __future__ import annotations

import os
import sqlite3
import threading
from pathlib import Path

import pytest
from sqlalchemy import inspect, text

from clef_server import db as db_mod
from clef_server.config import Config
from clef_server.db import (
    Database,
    DatabaseError,
    audit,
    code_version,
    database_url,
    open_database,
    redact_url,
)
from clef_server.dbschema import metadata
from clef_server.jobs import JobStore

pytestmark = pytest.mark.postgres


@pytest.fixture
def cfg(tmp_path) -> Config:
    return Config(state_dir=str(tmp_path / "state"), api_key=None)


@pytest.fixture
def db(cfg):
    database = Database(database_url(cfg))
    database.upgrade()
    yield database
    database.close()


# ------------------------------------------------------------------ URLs


def test_default_url_and_normalisation(tmp_path, monkeypatch, sqlite_only):
    cfg = Config(state_dir=str(tmp_path))
    assert database_url(cfg) == "sqlite:///" + (tmp_path / "clef.db").as_posix()
    for raw in ("postgres://u:p@h/clef", "postgresql://u:p@h/clef", " postgresql://u:p@h/clef "):
        assert (
            database_url(Config(state_dir=str(tmp_path), database_url=raw))
            == "postgresql+psycopg://u:p@h/clef"
        )
    keep = "postgresql+psycopg://u:p@h:5432/clef"
    assert database_url(Config(state_dir=str(tmp_path), database_url=keep)) == keep
    monkeypatch.setenv("CLEF_DATABASE_URL", "postgres://x:y@db/clef")
    monkeypatch.setenv("CLEF_DATABASE_POOL_SIZE", "7")
    env_cfg = Config(state_dir=str(tmp_path))
    assert env_cfg.database_url == "postgres://x:y@db/clef" and env_cfg.database_pool_size == 7


def test_redact_url_hides_the_password():
    assert (
        redact_url("postgresql+psycopg://clef:s3cret@db:5432/clef")
        == "postgresql+psycopg://clef:***@db:5432/clef"
    )
    assert redact_url("sqlite:////tmp/x.db") == "sqlite:////tmp/x.db"
    assert "s3cret" not in redact_url("::not a url s3cret@")
    db = Database("postgresql://clef:s3cret@localhost:1/clef")
    assert "s3cret" not in db.display_url and "s3cret" not in str(db.info())


def test_bad_urls_are_database_errors(tmp_path, monkeypatch):
    with pytest.raises(DatabaseError, match="not a valid database URL"):
        Database("not a url")
    with pytest.raises(DatabaseError, match="unsupported database"):
        Database("mysql://u:p@h/clef")
    with pytest.raises(DatabaseError, match="file path"):
        Database("sqlite://")
    with pytest.raises(DatabaseError, match="POOL_SIZE"):
        Database(f"sqlite:///{tmp_path}/x.db", pool_size=0)
    monkeypatch.setattr(db_mod.importlib.util, "find_spec", lambda name: None)
    with pytest.raises(DatabaseError, match=r"pip install 'clef-local\[postgres\]'"):
        Database("postgresql://u:p@h/clef")


@pytest.mark.skipif(os.name != "posix", reason="POSIX permissions")
def test_new_sqlite_file_is_private(tmp_path, sqlite_only):
    Database(f"sqlite:///{tmp_path}/sub/clef.db").close()
    assert (tmp_path / "sub" / "clef.db").stat().st_mode & 0o777 == 0o600


# ------------------------------------------------------------------ transactions


def _count(db: Database) -> int:
    with db.read() as conn:
        return conn.execute(text("SELECT COUNT(*) FROM clef_meta")).scalar_one()


def test_tx_commits_or_rolls_back_as_one(db):
    with pytest.raises(RuntimeError), db.tx() as conn:
        conn.execute(text("INSERT INTO clef_meta (k, v) VALUES ('a', '1')"))
        raise RuntimeError("boom")
    assert _count(db) == 0
    with db.tx() as conn:
        conn.execute(text("INSERT INTO clef_meta (k, v) VALUES ('a', '1')"))
        with db.tx() as inner:  # nested: the same connection, sees the outer write
            assert inner is conn
            assert inner.execute(text("SELECT v FROM clef_meta WHERE k='a'")).scalar() == "1"
        with db.read() as rd:
            assert rd is conn
    assert _count(db) == 1


def test_bump_meta_and_audit(db):
    with db.tx() as conn:
        assert db.bump(conn, "keys_gen") == 1
        assert db.bump(conn, "keys_gen") == 2
        audit(conn, "cli:me", "classifier.put", "x", {"version": 1}, ip="127.0.0.1", request_id="r1")
    assert db.meta("keys_gen") == "2" and db.meta("missing") is None
    with db.read() as conn:
        row = conn.execute(text("SELECT actor, action, target, detail, ip, request_id FROM audit_log")).one()
    assert tuple(row) == ("cli:me", "classifier.put", "x", '{"version":1}', "127.0.0.1", "r1")


def test_concurrent_writers_are_serialised(db):
    errors: list[Exception] = []

    def work() -> None:
        try:
            for _ in range(20):
                with db.tx() as conn:
                    db.bump(conn, "n")
        except Exception as exc:  # pragma: no cover - the assertion below shows it
            errors.append(exc)

    threads = [threading.Thread(target=work) for _ in range(4)]
    for th in threads:
        th.start()
    for th in threads:
        th.join()
    assert not errors and db.meta("n") == "80"


# ------------------------------------------------------------------ migrations


def test_upgrade_is_idempotent_and_reports_the_version(db):
    assert db.schema_version() == code_version() == "0002"
    db.upgrade()
    assert db.info()["schema_version"] == "0002" and db.info()["dialect"] == db.dialect


def test_metadata_matches_the_migrated_database(db):
    """dbschema.py mirrors the migrations: same tables, columns, nullability, keys and indexes."""
    with db.read() as conn:
        insp = inspect(conn)
        tables = set(insp.get_table_names()) - {"alembic_version"}
        assert tables == set(metadata.tables)
        for table in metadata.sorted_tables:
            got = {c["name"]: c for c in insp.get_columns(table.name)}
            assert list(got) == [c.name for c in table.columns], table.name
            for col in table.columns:
                assert bool(got[col.name]["nullable"]) == bool(col.nullable), (table.name, col.name)
                assert type(got[col.name]["type"]).__name__ != "NullType", (table.name, col.name)
            pk = insp.get_pk_constraint(table.name)["constrained_columns"]
            assert pk == [c.name for c in table.primary_key.columns], table.name
            fks = {
                (tuple(f["constrained_columns"]), f["referred_table"])
                for f in insp.get_foreign_keys(table.name)
            }
            want = {
                (tuple(c.name for c in fk.columns), fk.referred_table.name)
                for fk in table.foreign_key_constraints
            }
            assert fks == want, table.name
        # the inspector skips expression indexes (jobs_idem): ask the catalog
        if db.dialect == "sqlite":
            sql = "SELECT name FROM sqlite_master WHERE type='index'"
        else:
            sql = "SELECT indexname FROM pg_indexes WHERE schemaname = current_schema()"
        indexes = set(conn.execute(text(sql)).scalars())
    assert {i.name for table in metadata.sorted_tables for i in table.indexes} <= indexes


def test_legacy_jobs_db_is_renamed_stamped_and_upgraded(cfg, caplog, sqlite_only):
    state = Path(cfg.state_dir)
    state.mkdir(parents=True)
    legacy = sqlite3.connect(str(state / "jobs.db"))
    legacy.execute("PRAGMA journal_mode=WAL")
    legacy.executescript(
        "CREATE TABLE jobs (seq INTEGER PRIMARY KEY AUTOINCREMENT, id TEXT NOT NULL UNIQUE,"
        " kind TEXT NOT NULL, status TEXT NOT NULL, owner TEXT, payload TEXT NOT NULL,"
        " metadata TEXT, webhook TEXT,"
        " deliveries TEXT NOT NULL DEFAULT '{}', total INTEGER, done INTEGER NOT NULL DEFAULT 0,"
        " failed INTEGER NOT NULL DEFAULT 0, created_at REAL NOT NULL, started_at REAL, finished_at REAL,"
        " run_started_at REAL, run_base_done INTEGER NOT NULL DEFAULT 0, resumes INTEGER NOT NULL DEFAULT 0,"
        " cancel_requested INTEGER NOT NULL DEFAULT 0, error TEXT, result TEXT, idem_key TEXT);"
        "CREATE TABLE items (job_id TEXT NOT NULL REFERENCES jobs(id) ON DELETE CASCADE,"
        " idx INTEGER NOT NULL, data TEXT NOT NULL, PRIMARY KEY (job_id, idx)) WITHOUT ROWID;"
        "INSERT INTO jobs (id, kind, status, payload, created_at)"
        " VALUES ('job_a', 'classify', 'succeeded', '{}', 1);"
        "INSERT INTO items VALUES ('job_a', 0, '{\"label\": \"x\"}');"
    )
    legacy.commit()  # left open: the WAL still holds the rows
    (state / "classifiers").mkdir()
    (state / "classifiers" / "tri.json").write_text(
        '{"name": "tri", "kind": "classify", "labels": ["a", "b"]}'
    )
    with caplog.at_level("INFO", logger="clef"):
        database = open_database(cfg)
    legacy.close()
    assert not (state / "jobs.db").exists() and not (state / "jobs.db-wal").exists()
    assert any("renamed" in r.getMessage() for r in caplog.records)
    store = JobStore(database)
    assert store.get("job_a")["status"] == "succeeded" and store.items("job_a", 0, 10) == [
        (0, '{"label": "x"}')
    ]
    assert database.schema_version() == "0002"
    from clef_server.classifiers import ClassifierStore

    assert ClassifierStore(database).get("tri")["version"] == 1
    assert (state / "classifiers.imported" / "tri.json").exists()
    database.close()
    again = open_database(cfg)  # second start: nothing to rename, nothing to import
    assert again.schema_version() == "0002"
    again.close()


def test_both_files_present_keeps_clef_db_and_warns(cfg, caplog, sqlite_only):
    state = Path(cfg.state_dir)
    state.mkdir(parents=True)
    sqlite3.connect(str(state / "jobs.db")).close()
    open_database(cfg).close()  # creates clef.db (no jobs.db rename: the file is empty, renamed once)
    sqlite3.connect(str(state / "jobs.db")).close()
    with caplog.at_level("WARNING", logger="clef"):
        open_database(cfg).close()
    assert (state / "jobs.db").exists() and any("both" in r.getMessage() for r in caplog.records)


# ------------------------------------------------------------------ one server per database (Postgres)


def test_second_server_on_the_same_postgres_database_is_refused(cfg, db_backend):
    if db_backend != "postgres":
        pytest.skip("advisory lock: Postgres only")
    first = open_database(cfg)
    with pytest.raises(DatabaseError, match="one server per database"):
        open_database(cfg)
    first.close()
    open_database(cfg).close()  # released with the first server


def test_claim_next_hands_each_job_to_exactly_one_claimer(db):
    store = JobStore(db)
    ids = {store.create("classify", None, {}, None, None, 0)["id"] for _ in range(40)}
    claimed: list[str] = []
    lock = threading.Lock()

    def work() -> None:
        while True:
            row = store.claim_next()
            if row is None:
                return
            with lock:
                claimed.append(row["id"])

    threads = [threading.Thread(target=work) for _ in range(4)]
    for th in threads:
        th.start()
    for th in threads:
        th.join()
    assert sorted(claimed) == sorted(ids)
    assert all(store.get(j)["status"] == "running" for j in ids)


def test_claim_next_prefers_interrupted_jobs_and_reports_where_it_came_from(db):
    store = JobStore(db)
    first = store.create("classify", None, {}, None, None, 0)["id"]
    second = store.create("classify", None, {}, None, None, 0)["id"]
    assert store.claim(second) and store.park(second) is None  # parked: interrupted
    store.add_items(second, [{"i": 0}])
    row = store.claim_next()
    assert row["id"] == second and row["claimed_from"] == "interrupted" and row["run_base_done"] == 1
    row = store.claim_next()
    assert row["id"] == first and row["claimed_from"] == "queued" and row["status"] == "running"
    assert store.claim_next() is None and not store.claim(first)


def test_jobs_model_is_stored(db):
    store = JobStore(db)
    row = store.create("classify", None, {}, None, None, 0, model="clef-flash@abc")
    assert row["model"] == "clef-flash@abc" and row["priority"] == 1 and row["tokens"] == 0
