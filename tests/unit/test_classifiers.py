"""Saved-classifier store: CRUD, versions, owner, name validation, limits, audit, the v3.1 JSON import."""

from __future__ import annotations

import json
import threading

import pytest
from sqlalchemy import text

from clef_server.classifiers import (
    ClassifierStore,
    InvalidName,
    TooManyClassifiers,
    export_file,
    import_json_dir,
    validate_name,
)
from clef_server.config import Config
from clef_server.db import Database, database_url

pytestmark = pytest.mark.postgres  # also run on Postgres when CLEF_TEST_DATABASE_URL is set (conftest.py)

DEF = {"kind": "classify", "labels": ["a", "b"], "multi_label": False}


@pytest.fixture
def db(tmp_path):
    database = Database(database_url(Config(state_dir=str(tmp_path / "state"))))
    database.upgrade()
    yield database
    database.close()


@pytest.fixture
def store(db):
    return ClassifierStore(db, max_classifiers=3)


def audit_rows(db: Database) -> list[tuple[str, str, str]]:
    with db.read() as conn:
        rows = conn.execute(text("SELECT actor, action, target FROM audit_log ORDER BY id"))
        return [tuple(r) for r in rows]


def test_put_get_list_delete(store):
    doc = store.put("support-triage", DEF, owner="alice", created_by="alice", model="clef-flash@abc")
    assert doc["name"] == "support-triage" and doc["labels"] == ["a", "b"]
    assert doc["created_at"].endswith("Z") and doc["updated_at"].endswith("Z")
    assert doc["version"] == 1 and doc["owner"] == "alice" and doc["model"] == "clef-flash@abc"
    assert store.get("support-triage") == doc
    store.put("alpha", DEF)
    assert [d["name"] for d in store.list()] == ["alpha", "support-triage"]
    assert store.list()[1] == doc
    assert store.delete("alpha") is True and store.delete("alpha") is False
    assert store.get("alpha") is None


def test_new_definition_appends_a_version_and_keeps_created_at(store):
    first = store.put("x", DEF)
    second = store.put("x", {**DEF, "labels": ["c", "d"], "created_at": "1999", "name": "evil", "version": 9})
    assert second["created_at"] == first["created_at"] and second["name"] == "x"
    assert second["updated_at"] >= first["updated_at"] and second["labels"] == ["c", "d"]
    assert second["version"] == 2 and len(store.list()) == 1
    assert store.get("x", 1)["labels"] == ["a", "b"] and store.get("x", 2) == second
    assert store.get("x", 3) is None and store.get("nope", 1) is None
    versions = store.versions("x")
    assert [v["version"] for v in versions] == [2, 1] and versions[0]["created_by"] is None
    assert store.versions("nope") is None


def test_same_definition_is_an_idempotent_put(store, db):
    first = store.put("x", DEF, created_by="alice")
    again = store.put("x", {**DEF, "name": "x", "updated_at": "later"}, created_by="bob")
    assert again == first and [v["version"] for v in store.versions("x")] == [1]
    assert audit_rows(db) == [("alice", "classifier.put", "x")]  # the no-op is not audited


def test_owner_is_set_by_the_first_put_only(store):
    store.put("x", DEF, owner="alice")
    store.put("x", {**DEF, "labels": ["q", "r"]}, owner="bob")
    assert store.owner("x") == "alice" and store.get("x")["owner"] == "alice"
    store.put("open", DEF)
    assert store.owner("open") is None and store.owner("missing") is None


def test_versions_restart_after_delete_and_audit_is_written(store, db):
    store.put("x", DEF, created_by="alice")
    store.put("x", {**DEF, "labels": ["c", "d"]}, created_by="alice")
    assert store.delete("x", actor="bob")
    assert store.put("x", DEF)["version"] == 1  # versions are never reused while the name exists
    assert audit_rows(db) == [
        ("alice", "classifier.put", "x"),
        ("alice", "classifier.put", "x"),
        ("bob", "classifier.delete", "x"),
        ("anonymous", "classifier.put", "x"),
    ]


@pytest.mark.parametrize(
    "name",
    [
        "",
        "A",
        "Upper",
        "-start",
        "_start",
        "has space",
        "a/b",
        "a\\b",
        "../x",
        "..",
        ".",
        "x.json",
        "a" * 65,
        "café",
        "a\x00b",
        "a\n",
        "C:evil",
    ],
)
def test_invalid_names(store, name):
    with pytest.raises(InvalidName):
        validate_name(name)
    for op in (
        lambda: store.get(name),
        lambda: store.delete(name),
        lambda: store.put(name, DEF),
        lambda: store.versions(name),
        lambda: store.owner(name),
    ):
        with pytest.raises(InvalidName):
            op()
    assert store.list() == []


def test_valid_names(store):
    for name in ("a", "0", "a-b_c", "a" * 64, "9lives"):
        assert validate_name(name) == name


def test_max_classifiers_counts_names(store):
    for n in ("a", "b", "c"):
        store.put(n, DEF)
    with pytest.raises(TooManyClassifiers):
        store.put("d", DEF)
    store.put("a", {**DEF, "labels": ["q", "r"]})  # a new version is fine at the cap
    store.delete("b")
    store.put("d", DEF)


def test_concurrent_puts_respect_cap_and_never_reuse_a_version(db):
    store = ClassifierStore(db, max_classifiers=6)
    store.put("shared", DEF)
    errors: list[Exception] = []

    def work(i: int) -> None:
        try:
            store.put(f"n{i}", DEF)
            store.put("shared", {**DEF, "labels": [f"l{i}", "z"]})
        except TooManyClassifiers as exc:
            errors.append(exc)

    threads = [threading.Thread(target=work, args=(i,)) for i in range(10)]
    for th in threads:
        th.start()
    for th in threads:
        th.join()
    assert len(store.list()) == 6 and len(errors) == 5
    versions = [v["version"] for v in store.versions("shared") or []]
    assert versions == list(range(len(versions), 0, -1)) and len(versions) > 1  # gap-free, never reused


def test_concurrent_first_puts_of_a_new_name_get_distinct_versions(db):
    store = ClassifierStore(db, max_classifiers=50)
    for trial in range(5):
        name = f"fresh{trial}"
        barrier = threading.Barrier(4)
        errors: list[BaseException] = []
        got: list[int] = []

        def first_put(i: int, name=name, barrier=barrier, got=got, errors=errors) -> None:
            barrier.wait()
            try:
                got.append(store.put(name, {**DEF, "labels": [f"l{i}", "z"]}, owner=f"u{i}")["version"])
            except BaseException as exc:  # an IntegrityError here was the bug: SELECT FOR UPDATE on no row
                errors.append(exc)

        threads = [threading.Thread(target=first_put, args=(i,)) for i in range(4)]
        for th in threads:
            th.start()
        for th in threads:
            th.join()
        assert errors == [] and sorted(got) == [1, 2, 3, 4]
        assert store.owner(name) in {f"u{i}" for i in range(4)}


def test_put_runs_the_owner_check_inside_its_transaction(store):
    seen: list[tuple[bool, str | None]] = []

    def allow(exists: bool, owner: str | None) -> None:
        seen.append((exists, owner))

    store.put("t", DEF, owner="alice", check_owner=allow)
    store.put("t", {**DEF, "labels": ["c", "d"]}, owner="bob", check_owner=allow)
    assert seen == [(False, None), (True, "alice")]

    def deny(exists: bool, owner: str | None) -> None:
        raise PermissionError(owner)

    with pytest.raises(PermissionError):
        store.put("t", {**DEF, "labels": ["e", "f"]}, owner="bob", check_owner=deny)
    assert store.get("t")["version"] == 2  # nothing was written


def test_json_files_are_imported_once(tmp_path, db, caplog):
    folder = tmp_path / "state" / "classifiers"
    folder.mkdir(parents=True)
    good = {**DEF, "name": "good", "created_at": "2026-01-02T03:04:05.000000Z"}
    good["updated_at"] = "2026-02-03T04:05:06.000000Z"
    (folder / "good.json").write_text(json.dumps(good))
    (folder / "lvl.json").write_text(json.dumps({"name": "lvl", "kind": "score", "levels": ["lo", "hi"]}))
    (folder / "broken.json").write_text("{not json")
    (folder / "wrongname.json").write_text(json.dumps({**DEF, "name": "other"}))
    (folder / "list.json").write_text("[1, 2]")
    (folder / "invalid.json").write_text(json.dumps({"name": "invalid", "kind": "classify"}))
    (folder / "Bad Name.json").write_text("{}")
    with caplog.at_level("WARNING", logger="clef"):
        assert import_json_dir(db, folder) == 2
    assert any("broken.json" in r.getMessage() for r in caplog.records)
    assert not folder.exists() and (folder.parent / "classifiers.imported" / "broken.json").exists()
    store = ClassifierStore(db)
    doc = store.get("good")
    assert doc["created_at"] == good["created_at"] and doc["updated_at"] == good["updated_at"]
    assert doc["version"] == 1 and doc["owner"] is None and doc["labels"] == ["a", "b"]
    assert store.versions("good")[0]["created_by"] == "import"
    assert [d["name"] for d in store.list()] == ["good", "lvl"]
    assert db.meta("classifiers_imported") == "1"
    folder.mkdir()  # a later directory is never imported again
    (folder / "late.json").write_text(json.dumps({**DEF, "name": "late"}))
    assert import_json_dir(db, folder) == 0 and store.get("late") is None


def test_import_without_a_directory_only_sets_the_flag(tmp_path, db):
    assert import_json_dir(db, tmp_path / "missing") == 0
    assert db.meta("classifiers_imported") == "1"


def test_export_file_is_the_v31_format_plus_version(store):
    doc = store.put("x", DEF, owner="alice", model="m@1")
    out = export_file(doc)
    assert out["version"] == 1 and out["name"] == "x" and "owner" not in out and "model" not in out
    assert set(out) == {*DEF, "name", "created_at", "updated_at", "version"}
