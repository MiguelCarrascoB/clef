"""Saved-classifier store: CRUD, name validation, path traversal, corrupt files, limits."""

from __future__ import annotations

import json
import threading

import pytest

from clef_server.classifiers import ClassifierStore, InvalidName, TooManyClassifiers, validate_name

DEF = {"kind": "classify", "labels": ["a", "b"], "multi_label": False}


@pytest.fixture
def store(tmp_path):
    return ClassifierStore(tmp_path / "classifiers", max_classifiers=3)


def test_put_get_list_delete(store):
    doc = store.put("support-triage", DEF)
    assert doc["name"] == "support-triage" and doc["labels"] == ["a", "b"]
    assert doc["created_at"].endswith("Z") and doc["updated_at"].endswith("Z")
    assert store.get("support-triage") == doc
    store.put("alpha", DEF)
    assert [d["name"] for d in store.list()] == ["alpha", "support-triage"]
    assert store.delete("alpha") is True and store.delete("alpha") is False
    assert store.get("alpha") is None
    assert (store.dir / "support-triage.json").is_file()


def test_overwrite_keeps_created_at(store):
    first = store.put("x", DEF)
    second = store.put("x", {**DEF, "labels": ["c", "d"], "created_at": "1999", "name": "evil"})
    assert second["created_at"] == first["created_at"] and second["name"] == "x"
    assert second["updated_at"] >= first["updated_at"] and second["labels"] == ["c", "d"]
    assert len(store.list()) == 1


def test_no_temp_files_left(store):
    store.put("x", DEF)
    store.put("x", DEF)
    assert sorted(p.name for p in store.dir.iterdir()) == ["x.json"]


def test_atomic_write_failure_keeps_old_and_cleans_up(store, monkeypatch):
    store.put("x", DEF)
    import clef_server.classifiers as mod

    def boom(*a, **k):
        raise OSError("disk full")

    monkeypatch.setattr(mod.os, "replace", boom)
    with pytest.raises(OSError):
        store.put("x", {**DEF, "labels": ["z", "y"]})
    monkeypatch.undo()
    assert store.get("x")["labels"] == ["a", "b"]
    assert sorted(p.name for p in store.dir.iterdir()) == ["x.json"]


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
    for op in (lambda: store.get(name), lambda: store.delete(name), lambda: store.put(name, DEF)):
        with pytest.raises(InvalidName):
            op()
    assert not list(store.dir.parent.rglob("*.json"))


def test_valid_names(store):
    for name in ("a", "0", "a-b_c", "a" * 64, "9lives"):
        assert validate_name(name) == name


def test_traversal_never_touches_outside(tmp_path):
    outside = tmp_path / "secret.json"
    outside.write_text("{}")
    store = ClassifierStore(tmp_path / "classifiers")
    for name in ("../secret", "..%2fsecret", "../../secret", "/etc/passwd"):
        with pytest.raises(InvalidName):
            store.delete(name)
    assert outside.exists()


def test_corrupt_files_skipped_and_logged(store, caplog):
    store.put("good", DEF)
    (store.dir / "broken.json").write_text("{not json")
    (store.dir / "wrongname.json").write_text(json.dumps({"name": "other", "kind": "classify"}))
    (store.dir / "list.json").write_text("[1, 2]")
    (store.dir / "Bad Name.json").write_text("{}")
    with caplog.at_level("WARNING", logger="clef"):
        names = [d["name"] for d in store.list()]
    assert names == ["good"]
    assert any("broken.json" in r.getMessage() for r in caplog.records)
    assert store.get("broken") is None


def test_max_classifiers(store):
    for n in ("a", "b", "c"):
        store.put(n, DEF)
    with pytest.raises(TooManyClassifiers):
        store.put("d", DEF)
    store.put("a", {**DEF, "labels": ["q", "r"]})  # overwrite is fine at the cap
    store.delete("b")
    store.put("d", DEF)


def test_list_before_dir_exists(tmp_path):
    assert ClassifierStore(tmp_path / "nope").list() == []


def test_concurrent_puts_respect_cap(tmp_path):
    store = ClassifierStore(tmp_path / "c", max_classifiers=5)
    errors = []

    def work(i):
        try:
            store.put(f"n{i}", DEF)
        except TooManyClassifiers:
            errors.append(i)

    threads = [threading.Thread(target=work, args=(i,)) for i in range(12)]
    [t.start() for t in threads]
    [t.join() for t in threads]
    assert len(store.list()) == 5 and len(errors) == 7
