"""Storage layer (db.py, dbschema.py, migrations): URLs, transactions, migrations, legacy files, server lock,
claim races. Runs on SQLite, and on Postgres too when CLEF_TEST_DATABASE_URL is set (conftest.py)."""

from __future__ import annotations

import json
import os
import sqlite3
import tempfile
import threading
from pathlib import Path

import pytest
from sqlalchemy import inspect, text
from test_api import FakeEngine
from test_api import make as make_api
from test_jobs import LABELS, open_store, submit, wait
from test_jobs import make as make_jobs

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
    sqlite_url,
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


# ------------------------------------------------------------------ HTTP: versions, owner, /health, 503


class UnknownModelError(LookupError):
    pass


class ModelEngine(FakeEngine):
    """A FakeEngine with C2's hooks: serves clef-flash@<rev>, 404s any other model."""

    def __init__(self, rev: str = "abc1234") -> None:
        super().__init__()
        self.rev = rev

    def model_ref(self) -> str:
        return f"clef-flash@{self.rev}"

    def check_model(self, requested: str | None) -> str:
        if requested in (None, "", "clef-flash"):
            return "clef-flash"
        raise UnknownModelError(f"model {requested!r} is not served here (this server serves 'clef-flash')")


CLS = {"kind": "classify", "labels": ["billing", "technical"]}


def test_classifier_versions_over_http():
    client, _, _ = make_api()
    with client:
        first = client.put("/v1/classifiers/tri", json=CLS).json()
        assert first["version"] == 1 and first["owner"] is None and first["model"] is None
        assert client.put("/v1/classifiers/tri", json=CLS).json()["version"] == 1  # idempotent
        second = client.put("/v1/classifiers/tri", json={**CLS, "labels": ["a", "b", "c"]}).json()
        assert second["version"] == 2 and second["created_at"] == first["created_at"]
        assert client.get("/v1/classifiers/tri").json()["labels"] == ["a", "b", "c"]
        assert client.get("/v1/classifiers/tri", params={"version": 1}).json()["labels"] == CLS["labels"]
        assert client.get("/v1/classifiers/tri/versions/1").json()["version"] == 1
        versions = client.get("/v1/classifiers/tri/versions").json()
        assert versions["name"] == "tri" and [v["version"] for v in versions["versions"]] == [2, 1]
        for url in (
            "/v1/classifiers/tri/versions/9",
            "/v1/classifiers/nope/versions",
            "/v1/classifiers/nope",
        ):
            assert client.get(url).status_code == 404, url
        assert "no version 9" in client.get("/v1/classifiers/tri?version=9").json()["detail"]
        old = client.post("/v1/classifiers/tri", json={"input": "x", "version": 1}).json()
        assert set(old["scores"]) == set(CLS["labels"])
        new = client.post("/v1/classifiers/tri/batch", json={"inputs": ["x"]}).json()
        assert set(new["results"][0]["scores"]) == {"a", "b", "c"}
        assert client.post("/v1/classifiers/tri", json={"input": "x", "version": 7}).status_code == 404


def test_classifier_owner_is_enforced_for_non_admin_principals():
    from test_keys import add_key

    client, _, app = make_api(api_keys_raw="root:kr")  # an env key is admin
    a = {"X-API-Key": add_key(app, "alice", "classify,classifiers")}
    b = {"X-API-Key": add_key(app, "bob", "classify,classifiers")}
    root = {"X-API-Key": "kr"}
    changed = {**CLS, "labels": ["x", "y"]}
    with client:
        assert client.put("/v1/classifiers/tri", json=CLS, headers=a).json()["owner"] == "alice"
        r = client.put("/v1/classifiers/tri", json=changed, headers=b)
        assert r.status_code == 403 and "alice" not in r.json()["detail"]  # only admins may learn the owner
        assert client.delete("/v1/classifiers/tri", headers=b).status_code == 403
        assert client.post("/v1/classifiers/tri", json={"input": "x"}, headers=b).status_code == 200
        assert client.put("/v1/classifiers/tri", json=changed, headers=a).json()["version"] == 2
        r = client.put("/v1/classifiers/tri", json=CLS, headers=root)  # an admin may
        assert r.status_code == 200 and r.json()["owner"] == "alice" and r.json()["version"] == 3
        assert client.delete("/v1/classifiers/tri", headers=root).status_code == 200


def test_unowned_classifiers_are_admin_only_to_change_once_auth_is_on():
    from test_keys import add_key

    client, _, app = make_api()
    headers = {"X-API-Key": add_key(app, "alice", "classify,classifiers")}
    with client:
        app.state.ctx.store.put("open", CLS)  # e.g. imported from a v3.1 JSON file: owner NULL
        changed = {**CLS, "labels": ["x", "y"]}
        assert client.put("/v1/classifiers/open", json=changed, headers=headers).status_code == 403
        assert client.delete("/v1/classifiers/open", headers=headers).status_code == 403
        assert client.post("/v1/classifiers/open", json={"input": "x"}, headers=headers).status_code == 200


def test_classifier_audit_rows_carry_actor_ip_and_request_id():
    client, _, app = make_api(api_keys_raw="alice:ka")
    with client:
        client.put("/v1/classifiers/tri", json=CLS, headers={"X-API-Key": "ka", "X-Request-ID": "req-1"})
        client.delete("/v1/classifiers/tri", headers={"X-API-Key": "ka"})
        with app.state.ctx.db.read() as conn:
            sql = "SELECT actor, action, target, ip, request_id, detail FROM audit_log ORDER BY id"
            rows = conn.execute(text(sql)).all()
    assert [tuple(r[:3]) for r in rows] == [
        ("alice", "classifier.put", "tri"),
        ("alice", "classifier.delete", "tri"),
    ]
    assert rows[0].request_id == "req-1" and rows[0].ip and rows[0].detail == '{"version":1}'


def test_health_reports_the_database():
    client, _, _ = make_api()
    with client:
        db_info = client.get("/health").json()["db"]
    assert db_info["schema_version"] == "0002" and db_info["dialect"] in ("sqlite", "postgresql")
    assert "***" in db_info["url"] or db_info["url"].startswith("sqlite")


def test_database_errors_are_503(monkeypatch):
    from sqlalchemy.exc import OperationalError

    client, _, app = make_api()

    def down(*a, **k):
        raise OperationalError("SELECT 1", {}, Exception("connection refused"))

    with client:
        monkeypatch.setattr(app.state.ctx.store, "list", down)
        monkeypatch.setattr(app.state.ctx.store, "owner_of", down)
        monkeypatch.setattr(app.state.ctx.extra["jobs"].store, "list", down)
        for r in (
            client.get("/v1/classifiers"),
            client.put("/v1/classifiers/x", json=CLS),
            client.get("/v1/jobs"),
        ):
            assert r.status_code == 503 and r.json()["detail"] == "database unavailable", r.text


# ------------------------------------------------------------------ jobs: model ref, model check, versions


def test_jobs_store_the_model_ref_and_404_an_unknown_model():
    client, _, app = make_jobs(ModelEngine())
    with client:
        r = submit(client, payload={"inputs": ["a"], "labels": LABELS, "model": "gpt-4"})
        assert r.status_code == 404 and "not served here" in r.json()["detail"]
        req = {"state": "x", "questions": {"q": {"type": "noul"}}, "model": "other"}
        assert submit(client, "systemone", {"requests": [req]}).status_code == 404
        jid = submit(client, payload={"inputs": ["a"], "labels": LABELS}).json()["id"]
        assert wait(client, jid)["status"] == "succeeded"
        assert app.state.ctx.extra["jobs"].store.get(jid)["model"] == "clef-flash@abc1234"
        stored = json.loads(app.state.ctx.extra["jobs"].store.payload(jid))
        assert "model" not in stored  # the payload default stays "the served model"


def test_a_job_submitted_for_another_model_fails_at_run_time():
    state = tempfile.mkdtemp(prefix="clef-db-")
    store = open_store(Config(state_dir=state))
    payload = {"inputs": ["a"], "labels": LABELS}
    job = store.create("classify", None, payload, None, None, 10, model="clef-flash@old0000")
    legacy = store.create("classify", None, payload, None, None, 10)  # model NULL: a row from v3.1
    store.close()
    client, _, _ = make_jobs(ModelEngine(rev="new1111"), state_dir=state)
    with client:
        failed = wait(client, job["id"])
        assert failed["status"] == "failed"
        assert (
            failed["error"]
            == "job was submitted for clef-flash@old0000, this server serves clef-flash@new1111"
        )
        assert wait(client, legacy["id"])["status"] == "succeeded"  # rows from before v3.2 still run


def test_job_snapshots_record_the_classifier_version():
    client, _, app = make_jobs()
    store = app.state.ctx.extra["jobs"].store
    with client:
        client.put("/v1/classifiers/tri", json={"kind": "classify", "labels": LABELS})
        client.put("/v1/classifiers/tri", json={"kind": "classify", "labels": ["x", "y"]})
        latest = submit(client, payload={"inputs": ["a"], "classifier": "tri"}).json()["id"]
        pinned = submit(client, payload={"inputs": ["a"], "classifier": "tri", "version": 1}).json()["id"]
        assert submit(client, payload={"inputs": ["a"], "classifier": "tri", "version": 5}).status_code == 400
        assert submit(client, payload={"inputs": ["a"], "labels": LABELS, "version": 1}).status_code == 400
        a, b = json.loads(store.payload(latest)), json.loads(store.payload(pinned))
        assert a["classifier_version"] == 2 and a["labels"] == ["x", "y"] and "version" not in a
        assert b["classifier_version"] == 1 and b["labels"] == LABELS
        assert wait(client, pinned)["status"] == "succeeded"  # the snapshot re-validates on the runner
        rows = [{"input": "a", "gold": "technical"}, {"input": "b", "gold": "billing"}]
        ev = submit(client, "evaluate", {"classifier": "tri", "version": 1, "rows": rows}).json()["id"]
        snap = json.loads(store.payload(ev))
        assert snap["classifier_version"] == 1 and snap["labels"] == LABELS and "version" not in snap
        assert wait(client, ev)["status"] == "succeeded"
        live = client.post("/v1/evaluate", json={"classifier": "tri", "version": 1, "rows": rows})
        assert live.status_code == 200 and set(live.json()["labels"]) == set(LABELS)
        bad = client.post("/v1/evaluate", json={"classifier": "tri", "version": 9, "rows": rows})
        assert bad.status_code == 404


# ------------------------------------------------------------------ CLI: db, backup, classifiers export

LEGACY_JOBS = (
    "CREATE TABLE jobs (seq INTEGER PRIMARY KEY AUTOINCREMENT, id TEXT NOT NULL UNIQUE, kind TEXT NOT NULL,"
    " status TEXT NOT NULL, owner TEXT, payload TEXT NOT NULL, metadata TEXT, webhook TEXT,"
    " deliveries TEXT NOT NULL DEFAULT '{}', total INTEGER, done INTEGER NOT NULL DEFAULT 0,"
    " failed INTEGER NOT NULL DEFAULT 0, created_at REAL NOT NULL, started_at REAL, finished_at REAL,"
    " run_started_at REAL, run_base_done INTEGER NOT NULL DEFAULT 0, resumes INTEGER NOT NULL DEFAULT 0,"
    " cancel_requested INTEGER NOT NULL DEFAULT 0, error TEXT, result TEXT)"
)


def run_cli(*argv: str) -> int:
    from clef_server import cli

    return cli.main(list(argv))


def test_db_current_and_upgrade(cfg, monkeypatch, capsys, db_backend):
    monkeypatch.setenv("CLEF_STATE_DIR", cfg.state_dir)
    code = run_cli("db", "current")
    out = capsys.readouterr().out
    if db_backend == "sqlite":
        assert code == 1 and "no database yet" in out and not Path(cfg.state_dir, "clef.db").exists()
    else:
        assert code == 1 and "(empty)" in out and "***" in out
    assert run_cli("db", "upgrade") == 0 and "-> 0002" in capsys.readouterr().out
    assert (
        run_cli("db", "current") == 0 and "0002 (this clef expects 0002: current)" in capsys.readouterr().out
    )


def test_db_upgrade_renames_a_legacy_jobs_db(cfg, monkeypatch, capsys, sqlite_only):
    monkeypatch.setenv("CLEF_STATE_DIR", cfg.state_dir)
    Path(cfg.state_dir).mkdir(parents=True)
    con = sqlite3.connect(str(Path(cfg.state_dir) / "jobs.db"))
    con.execute(LEGACY_JOBS)
    con.commit()
    con.close()
    assert run_cli("db", "current") == 1 and "no database yet" in capsys.readouterr().out
    assert run_cli("db", "upgrade") == 0
    assert not (Path(cfg.state_dir) / "jobs.db").exists()
    assert run_cli("db", "current") == 0


def _seed(url: str) -> None:
    from clef_server.classifiers import ClassifierStore

    database = Database(url)
    database.upgrade()
    store = JobStore(database)
    for i in range(3):
        jid = store.create("classify", "alice", {"n": i}, {"m": i}, None, 0, idem_key=f"k{i}")["id"]
        store.claim(jid)
        store.add_items(jid, [{"index": 0, "label": "a"}, {"index": 1, "error": "x"}])
        store.finish(jid, "succeeded", result={"ok": 1})
    ClassifierStore(database).put("tri", CLS, owner="alice")
    with database.tx() as conn:
        audit(conn, "cli:me", "key.create", "bob")
    database.close()


def test_db_copy_moves_every_row_and_resets_identities(tmp_path, cfg, monkeypatch, capsys, db_backend):
    from clef_server import cli
    from clef_server.classifiers import ClassifierStore

    monkeypatch.setattr(cli, "livez", lambda url: False)
    src = f"sqlite:///{tmp_path}/src.db"
    _seed(src)
    dst = database_url(cfg)  # the backend under test (SQLite file or the per-test Postgres database)
    assert run_cli("db", "copy", "--from", src, "--to", dst, "--yes") == 0
    out = capsys.readouterr().out
    assert "jobs: 3 row(s)" in out and "items: 6 row(s)" in out and "copied" in out
    target = Database(dst)
    store = JobStore(target)
    rows, total = store.list(None, False, None, None, 10, 0)
    assert total == 3 and [r["result"] for r in rows] == [{"ok": 1}] * 3
    assert store.items(rows[0]["id"], 0, 10)[1][1] == '{"index":1,"error":"x"}'
    assert store.create("classify", "alice", {}, None, None, 0, idem_key="k1")["replayed"]
    assert store.create("classify", None, {}, None, None, 0)["seq"] == 4  # identities continue
    with target.tx() as conn:
        audit(conn, "cli:me", "key.revoke", "bob")
        ids = conn.execute(text("SELECT id FROM audit_log ORDER BY id")).scalars().all()
    assert len(ids) == len(set(ids)) == 3
    assert ClassifierStore(target).get("tri")["owner"] == "alice"
    target.close()
    assert run_cli("db", "copy", "--from", src, "--to", dst, "--yes") == 1  # the target is not empty
    assert "not empty" in capsys.readouterr().err


def test_db_copy_refuses_while_a_server_answers(tmp_path, monkeypatch, capsys):
    from clef_server import cli

    monkeypatch.setattr(cli, "livez", lambda url: True)
    a, b = f"sqlite:///{tmp_path}/a.db", f"sqlite:///{tmp_path}/b.db"
    assert run_cli("db", "copy", "--from", a, "--to", b, "-y") == 1
    assert "stop it first" in capsys.readouterr().err


def test_db_copy_needs_an_upgraded_source(tmp_path, monkeypatch, capsys):
    from clef_server import cli

    monkeypatch.setattr(cli, "livez", lambda url: False)
    con = sqlite3.connect(str(tmp_path / "old.db"))
    con.execute(LEGACY_JOBS)
    con.commit()
    con.close()
    assert (
        run_cli(
            "db", "copy", "--from", f"sqlite:///{tmp_path}/old.db", "--to", f"sqlite:///{tmp_path}/n.db", "-y"
        )
        == 1
    )
    assert "clef db upgrade" in capsys.readouterr().err


def test_backup_sqlite_and_postgres_hint(cfg, monkeypatch, capsys, db_backend):
    monkeypatch.setenv("CLEF_STATE_DIR", cfg.state_dir)
    if db_backend == "postgres":
        assert run_cli("backup") == 2
        err = capsys.readouterr().err
        assert "pg_dump" in err and "***" in err
        return
    assert run_cli("backup") == 1  # nothing to back up yet
    _seed(database_url(cfg))
    out_path = Path(cfg.state_dir) / "copy.db"
    assert run_cli("backup", "--out", str(out_path)) == 0
    if os.name == "posix":
        assert out_path.stat().st_mode & 0o777 == 0o600
    copy = Database(sqlite_url(out_path))
    assert copy.schema_version() == "0002" and JobStore(copy).list(None, False, None, None, 10, 0)[1] == 3
    copy.close()
    assert run_cli("backup", "--out", str(out_path)) == 1  # never overwrites
    assert run_cli("backup") == 0
    assert list((Path(cfg.state_dir) / "backups").glob("clef-*Z.db"))


def test_classifiers_export(cfg, tmp_path, monkeypatch, db_backend):
    monkeypatch.setenv("CLEF_STATE_DIR", cfg.state_dir)
    out = tmp_path / "exported"
    if db_backend == "sqlite":
        assert run_cli("classifiers", "export", "--out", str(out)) == 1  # no database yet
    _seed(database_url(cfg))
    assert run_cli("classifiers", "export", "--out", str(out)) == 0
    doc = json.loads((out / "tri.json").read_text())
    assert (
        doc["name"] == "tri" and doc["version"] == 1 and doc["labels"] == CLS["labels"] and "owner" not in doc
    )
    assert run_cli("classifiers", "export", "--out", str(out), "--name", "nope") == 1
    assert run_cli("classifiers", "export", "--out", str(out), "--name", "Bad Name") == 1


def test_db_commands_need_the_server_extra(monkeypatch, capsys):
    from clef_server import cli

    monkeypatch.setattr(cli, "_db_module", lambda: None)
    for argv in (("db", "current"), ("backup",), ("classifiers", "export")):
        assert run_cli(*argv) == 2
        assert "pip install 'clef-local[server]'" in capsys.readouterr().err


# ------------------------------------------------------------------ doctor


def _doctor_db(cfg) -> dict:
    from clef_server import doctor

    r = doctor.Report(echo=False)
    doctor.check_database(r, cfg)
    return r.items[-1]


def test_doctor_database_check(cfg, sqlite_only):
    from clef_server import doctor

    item = _doctor_db(cfg)
    assert item["level"] == doctor.OK and "does not exist yet" in item["detail"]
    assert not Path(cfg.state_dir, "clef.db").exists()  # read-only: nothing created
    database = Database(database_url(cfg))
    database.upgrade()
    assert _doctor_db(cfg)["level"] == doctor.OK and "schema 0002" in _doctor_db(cfg)["detail"]
    with database.tx() as conn:
        conn.execute(text("UPDATE alembic_version SET version_num='0001'"))
    assert _doctor_db(cfg)["level"] == doctor.FAIL and "clef db upgrade" in _doctor_db(cfg)["detail"]
    with database.tx() as conn:
        conn.execute(text("UPDATE alembic_version SET version_num='0099_future'"))
    assert _doctor_db(cfg)["level"] == doctor.FAIL and "newer than this clef" in _doctor_db(cfg)["detail"]
    database.close()
    bad = Config(state_dir=cfg.state_dir, database_url="postgresql://u:s3cret@127.0.0.1:1/clef")
    item = _doctor_db(bad)
    assert item["level"] == doctor.FAIL and "s3cret" not in item["detail"]
