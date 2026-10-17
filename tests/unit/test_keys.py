"""Keys and principals (D1): key format, store, cache, quotas, auth, scopes, admin API, CLI.

The default-deny test walks the routes: a route without a scope dependency fails it.
"""

from __future__ import annotations

import asyncio
import json
import os
import tempfile
import time
from datetime import datetime, timezone
from typing import Any

import pytest
from sqlalchemy import text
from test_api import FakeEngine, make
from test_jobs import LABELS, SlowEngine, submit, wait

from clef_server import keystore as ks
from clef_server.config import Config
from clef_server.db import Database, database_url
from clef_server.engine import GpuOutOfMemory, Prepared
from clef_server.keystore import Actor, KeyCache, KeyStore, KeystoreError, Principal, Quotas

CLASSIFY = {"input": "refund please", "labels": ["billing", "technical"]}


def add_key(app: Any, name: str, scopes: str = "classify,jobs,classifiers,ops", **kw: Any) -> str:
    """A DB principal on a test app (the cache is reloaded at once); returns its secret."""
    ctx = app.state.ctx
    _, secret = KeyStore(ctx.db).create_principal(name, scopes, actor=Actor("test"), **kw)
    ctx.keys.reload()
    return secret


def hdr(secret: str) -> dict[str, str]:
    return {"X-API-Key": secret}


@pytest.fixture
def db(tmp_path):
    database = Database(database_url(Config(state_dir=str(tmp_path / "state"), api_key=None)))
    database.upgrade()
    yield database
    database.close()


# ------------------------------------------------------------------ key format


def test_key_format_and_parsing():
    key_id, key = ks.generate_key()
    assert len(key) == 66 and key.startswith("clef_live_") and key[22] == "_"
    assert len(key_id) == 12 and set(key_id) <= set("abcdefghijklmnopqrstuvwxyz234567")
    assert ks.parse_key(key) == key_id
    assert ks.hash_key(key) != key and len(ks.hash_key(key)) == 64
    assert ks.display_key(key_id) == f"clef_live_{key_id}_…"
    for bad in ("", "clef_live_", key[:-1], key + "x", "x" + key[1:], key[:22] + "-" + key[23:]):
        assert ks.parse_key(bad) is None
    # the secret may contain "_" and "-": parsing is by offsets, never by splitting
    assert ks.parse_key(f"clef_live_{key_id}_" + "_-" * 21 + "a") == key_id


def test_scopes_parse_and_admin_implies_all():
    assert ks.parse_scopes("classify, jobs") == {"classify", "jobs"}
    for bad in ("nope", "", []):
        with pytest.raises(KeystoreError):
            ks.parse_scopes(bad)
    p = Principal("a", "db", frozenset({"admin"}), None, None, None, None)
    assert p.has("jobs") and p.has("urgent")
    assert not Principal("b", "db", frozenset({"jobs"}), None, None, None, None).has("classify")


# ------------------------------------------------------------------ store


@pytest.mark.postgres
def test_create_stores_only_the_hash_and_audits(db):
    store = KeyStore(db)
    row, secret = store.create_principal("ci", "classify,jobs", actor=Actor("cli:me"), daily_quota=50)
    assert row["scopes"] == ["classify", "jobs"] and row["daily_quota"] == 50 and len(row["keys"]) == 1
    with db.read() as conn:
        stored = conn.execute(text("SELECT secret_sha256 FROM api_keys")).scalar()
        dump = " ".join(
            str(v)
            for table in ("api_keys", "api_principals", "audit_log")
            for r in conn.execute(text(f"SELECT * FROM {table}"))
            for v in r
        )
    assert stored == ks.hash_key(secret) and secret not in dump
    entries, _ = store.audit_entries()
    assert entries[0]["action"] == "key.create" and entries[0]["actor"] == "cli:me"
    assert entries[0]["target"] == "ci" and secret not in json.dumps(entries[0])


@pytest.mark.postgres
def test_names_are_validated_unique_and_never_reused(db):
    store = KeyStore(db)
    a = Actor("t")
    for bad in ("", "has space", "x" * 65):
        with pytest.raises(KeystoreError) as e:
            store.create_principal(bad, "jobs", actor=a)
        assert e.value.status == 400
    store.create_principal("ci", "jobs", actor=a)
    with pytest.raises(KeystoreError) as e:
        store.create_principal("ci", "jobs", actor=a)
    assert e.value.status == 409
    store.revoke("ci", actor=a)
    with pytest.raises(KeystoreError) as e:
        store.create_principal("ci", "jobs", actor=a)
    assert e.value.status == 409 and "never reused" in str(e.value)
    with pytest.raises(KeystoreError) as e:
        store.create_principal("root", "jobs", actor=a, env_names={"root"})
    assert e.value.status == 409 and "env key" in str(e.value)


@pytest.mark.postgres
def test_rotation_grace_and_the_five_key_limit(db):
    store = KeyStore(db)
    a = Actor("t")
    store.create_principal("ci", "jobs", actor=a)
    t0 = time.time()
    row, _ = store.rotate("ci", actor=a, grace_s=3600)
    old, new = row["keys"]
    assert new["status"] == "active" and old["status"] == "expiring"
    assert t0 + 3500 < old["expires_at"] < t0 + 3700
    # a rotation never extends a key that already expires sooner than the grace
    store.rotate("ci", actor=a, grace_s=10**6)
    keys = store.get("ci")["keys"]
    assert keys[0]["expires_at"] < t0 + 3700
    assert [k["expires_at"] is None for k in keys] == [False, False, True]
    store.rotate("ci", actor=a, grace_s=3600)
    store.rotate("ci", actor=a, grace_s=3600)
    with pytest.raises(KeystoreError) as e:
        store.rotate("ci", actor=a, grace_s=3600)
    assert e.value.status == 409 and "5 active keys" in str(e.value)
    store.revoke("ci", actor=a, key_id=store.get("ci")["keys"][0]["id"])
    store.rotate("ci", actor=a, grace_s=0)  # a slot is free again; grace 0 expires the others now


@pytest.mark.postgres
def test_update_clear_and_disable(db):
    store = KeyStore(db)
    a = Actor("t")
    store.create_principal("ci", "jobs", actor=a, daily_quota=5, rate_limit=7, note="n")
    row = store.update_principal("ci", {"daily_quota": None, "scopes": ["classify"]}, actor=a)
    assert row["daily_quota"] is None and row["rate_limit"] == 7 and row["scopes"] == ["classify"]
    with pytest.raises(KeystoreError):
        store.update_principal("ci", {"scopes": ["bogus"]}, actor=a)
    with pytest.raises(KeystoreError) as e:
        store.update_principal("nope", {"note": "x"}, actor=a)
    assert e.value.status == 404
    row = store.revoke("ci", actor=a)
    assert row["disabled_at"] and {k["status"] for k in row["keys"]} == {"revoked"}
    assert store.revoke("ci", actor=a)["disabled_at"]  # idempotent: no second audit row
    actions = [x["action"] for x in store.audit_entries()[0]]
    assert actions == ["principal.disable", "principal.update", "key.create"]
    with pytest.raises(KeystoreError):
        store.revoke("ci", actor=a, key_id="nope")
    assert [p["name"] for p in store.list_principals()] == []
    assert [p["name"] for p in store.list_principals(include_disabled=True)] == ["ci"]


@pytest.mark.postgres
def test_audit_paging_and_prefix_filter(db):
    store = KeyStore(db)
    a = Actor("t")
    for i in range(5):
        store.create_principal(f"p{i}", "jobs", actor=a)
    store.update_principal("p0", {"note": "x"}, actor=a)
    first, nxt = store.audit_entries(limit=4)
    assert len(first) == 4 and nxt == first[-1]["id"]
    rest, nxt2 = store.audit_entries(limit=4, before=nxt)
    assert len(rest) == 2 and nxt2 is None
    assert {e["action"] for e in store.audit_entries(action="key.")[0]} == {"key.create"}
    assert [e["action"] for e in store.audit_entries(action="principal")[0]] == ["principal.update"]


# ------------------------------------------------------------------ cache and last_used


@pytest.mark.postgres
def test_cache_reload_and_generation_polling(db):
    store, cache = KeyStore(db), KeyCache(db, refresh_s=0.05)
    cache.reload()
    assert not cache.any_principal()
    _, secret = store.create_principal("ci", "jobs", actor=Actor("t"))
    assert cache.get(ks.parse_key(secret)) is None  # a CLI write is not seen until the next poll
    assert store.generation() != cache._gen

    async def poll() -> None:
        task = asyncio.create_task(cache.run())
        await asyncio.sleep(0.3)
        task.cancel()

    asyncio.run(poll())
    assert cache.get(ks.parse_key(secret)) is not None and cache.any_principal()


@pytest.mark.postgres
def test_last_used_flush_does_not_bump_the_generation(db):
    store, cache = KeyStore(db), KeyCache(db)
    _, secret = store.create_principal("ci", "jobs", actor=Actor("t"))
    cache.reload()
    gen = store.generation()
    cache.touch(ks.parse_key(secret))
    cache.flush_used()
    assert store.generation() == gen
    assert store.get("ci")["keys"][0]["last_used_at"] is not None


@pytest.mark.postgres
def test_a_failing_reload_keeps_the_last_good_state(db):
    store, cache = KeyStore(db), KeyCache(db)
    _, secret = store.create_principal("ci", "jobs", actor=Actor("t"))
    cache.reload()

    def broken() -> None:
        raise RuntimeError("database down")

    store_load = cache.store.load_all
    cache.store.load_all = broken  # type: ignore[method-assign]
    with pytest.raises(RuntimeError):
        cache.reload()
    cache.store.load_all = store_load  # type: ignore[method-assign]
    assert cache.get(ks.parse_key(secret)) is not None and cache.any_principal()


# ------------------------------------------------------------------ quotas (unit)


def principal(quota: int | None = 10, source: str = "db") -> Principal:
    return Principal("ci", source, frozenset({"classify"}), quota, None, "k", None)


@pytest.mark.postgres
def test_quota_reserve_refund_and_exceeded(db):
    q = Quotas(db)
    p = principal(10)
    r = q.reserve(p, 6)
    assert r and q.used("ci") == 6 and q.remaining(p) == (10, 4)
    with pytest.raises(ks.QuotaExceeded) as e:
        q.reserve(p, 5)
    assert (e.value.limit, e.value.remaining) == (10, 4) and 0 < e.value.retry_after <= 86401
    q.refund(r, 4)
    assert q.used("ci") == 2
    q.refund(r, 100)  # never more than the reservation, never below zero
    assert q.used("ci") == 0
    assert q.reserve(None, 5) is None and q.reserve(principal(source="env"), 5) is None
    assert q.reserve(principal(None), 10**6) is not None  # no quota: counted, never refused
    assert q.remaining(principal(None)) is None


@pytest.mark.postgres
def test_quota_day_rollover_and_retry_after(db):
    now = [datetime(2026, 10, 17, 23, 59, 0, tzinfo=timezone.utc).timestamp()]
    q = Quotas(db, clock=lambda: now[0])
    p = principal(5)
    r = q.reserve(p, 5)
    with pytest.raises(ks.QuotaExceeded) as e:
        q.reserve(p, 1)
    assert 60 <= e.value.retry_after <= 62  # seconds to the next UTC midnight
    now[0] += 120  # past midnight: a fresh day
    assert q.used("ci") == 0 and q.reserve(p, 5)
    q.refund(r, 5)  # a refund goes to the day it was charged to
    assert q.used("ci", "2026-10-17") == 0 and q.used("ci", "2026-10-18") == 5


@pytest.mark.postgres
def test_quota_flush_upserts_and_history_merges_unflushed(db):
    q = Quotas(db)
    p = principal(100)
    q.reserve(p, 7)
    with pytest.raises(ks.QuotaExceeded):
        q.reserve(p, 500)
    assert q.history("ci", 2)[-1] == {"day": q.today(), "records": 7, "requests": 2, "rejected": 1}
    q.flush()
    q.reserve(p, 3)
    q.flush()
    q.flush()  # nothing new: a no-op
    with db.read() as conn:
        row = conn.execute(text("SELECT records, requests, rejected FROM api_usage")).one()
    assert tuple(row) == (10, 3, 1)
    fresh = Quotas(db)
    fresh.load()  # a restart picks today's counts up from the database
    assert fresh.used("ci") == 10
    hist = fresh.history("ci", 3)
    assert [d["records"] for d in hist] == [0, 0, 10] and hist[0]["day"] < hist[-1]["day"]


@pytest.mark.postgres
def test_quota_refund_below_flushed_base_is_clamped(db):
    q = Quotas(db)
    r = q.reserve(principal(10), 2)
    q.flush()
    q.refund(r, 2)
    q.flush()
    with db.read() as conn:
        assert conn.execute(text("SELECT records FROM api_usage")).scalar() == 0


# ------------------------------------------------------------------ auth over HTTP


def test_auth_is_off_with_no_env_key_and_no_db_key():
    client, _, _ = make()
    with client:
        assert client.post("/v1/classify", json=CLASSIFY).status_code == 200
        who = client.get("/v1/whoami").json()
        assert who["auth"] is False and who["name"] is None and who["scopes"] is None


def test_a_db_key_turns_auth_on_and_revoking_the_last_never_turns_it_off():
    client, _, app = make()
    secret = add_key(app, "ci", "classify")
    with client:
        r = client.post("/v1/classify", json=CLASSIFY)
        assert r.status_code == 401 and r.headers["WWW-Authenticate"] == "Bearer"
        assert client.post("/v1/classify", json=CLASSIFY, headers=hdr(secret)).status_code == 200
        bearer = {"Authorization": f"Bearer {secret}"}
        assert client.post("/v1/classify", json=CLASSIFY, headers=bearer).status_code == 200
        KeyStore(app.state.ctx.db).revoke("ci", actor=Actor("t"))
        app.state.ctx.keys.reload()
        assert client.post("/v1/classify", json=CLASSIFY, headers=hdr(secret)).status_code == 401
        assert client.post("/v1/classify", json=CLASSIFY).status_code == 401  # still on
        assert client.get("/health").json()["limits"]["auth_required"] is True


def test_wrong_expired_revoked_and_disabled_keys_all_look_the_same():
    client, _, app = make(api_key="root-key")
    store = KeyStore(app.state.ctx.db)
    good = add_key(app, "good", "classify")
    exp = add_key(app, "exp", "classify")
    gone = add_key(app, "gone", "classify")
    off = add_key(app, "off", "classify")
    with app.state.ctx.db.tx() as conn:
        conn.execute(
            text("UPDATE api_keys SET expires_at = :t WHERE id = :i"),
            {"t": time.time() - 1, "i": ks.parse_key(exp)},
        )
    store.revoke("gone", actor=Actor("t"), key_id=ks.parse_key(gone))
    store.revoke("off", actor=Actor("t"))
    app.state.ctx.keys.reload()
    keys = {"good": good, "exp": exp, "gone": gone, "off": off, "wrong": good[:-3] + "xyz", "env": "root-key"}
    with client:
        got = {
            n: client.post("/v1/classify", json=CLASSIFY, headers=hdr(k)).status_code for n, k in keys.items()
        }
    assert got == {"good": 200, "exp": 401, "gone": 401, "off": 401, "wrong": 401, "env": 200}


def test_expiry_is_checked_at_request_time():
    client, _, app = make()
    secret = add_key(app, "soon", "classify", expires_at=time.time() + 0.4)
    with client:
        assert client.post("/v1/classify", json=CLASSIFY, headers=hdr(secret)).status_code == 200
        time.sleep(0.6)  # the cache was not reloaded: the check uses the clock
        assert client.post("/v1/classify", json=CLASSIFY, headers=hdr(secret)).status_code == 401


def test_env_keys_are_full_admin_without_quota():
    client, _, _ = make(api_keys_raw="root:kr")
    with client:
        who = client.get("/v1/whoami", headers=hdr("kr")).json()
        assert who["name"] == "root" and who["source"] == "env" and set(who["scopes"]) == set(ks.ALL_SCOPES)
        assert who["daily_quota"] is None
        r = client.post("/v1/classify", json=CLASSIFY, headers=hdr("kr"))
        assert r.status_code == 200 and "x-quota-limit" not in r.headers
        assert client.get("/v1/admin/keys", headers=hdr("kr")).status_code == 200


# ------------------------------------------------------------------ scopes


def test_scopes_gate_each_route_group():
    client, _, app = make(api_key="root")
    classify, jobs, mgmt, ops = (hdr(add_key(app, n, s)) for n, s in (
        ("c", "classify"), ("j", "jobs"), ("m", "classifiers"), ("o", "ops")))  # fmt: skip
    body = {"kind": "classify", "payload": {"inputs": ["a"], "labels": LABELS}}
    put = {"kind": "classify", "labels": ["a", "b"]}
    with client:
        assert client.post("/v1/classify", json=CLASSIFY, headers=classify).status_code == 200
        assert client.post("/v1/classify", json=CLASSIFY, headers=jobs).status_code == 403
        denied = client.post("/v1/classify", json=CLASSIFY, headers=ops)
        assert "lacks scope 'classify'" in denied.json()["detail"]
        assert client.get("/v1/classifiers", headers=classify).status_code == 200
        assert client.get("/v1/classifiers", headers=mgmt).status_code == 403
        assert client.put("/v1/classifiers/x", json=put, headers=classify).status_code == 403
        assert client.put("/v1/classifiers/x", json=put, headers=mgmt).status_code == 200
        assert client.delete("/v1/classifiers/x", headers=classify).status_code == 403
        assert client.post("/v1/jobs", json=body, headers=classify).status_code == 403
        assert client.post("/v1/jobs", json=body, headers=jobs).status_code == 202
        assert client.get("/v1/jobs", headers=jobs).status_code == 200
        assert client.get("/v1/stats", headers=jobs).status_code == 403
        assert client.get("/v1/stats", headers=ops).status_code == 200
        assert client.get("/v1/log", headers=classify).status_code == 403
        assert client.get("/v1/models", headers=classify).status_code == 200
        assert client.get("/v1/models", headers=ops).status_code == 403
        assert client.post("/hf/models/x", json={"inputs": "a"}, headers=ops).status_code == 403
        assert client.post("/v1/evaluate", json={}, headers=ops).status_code == 403
        for h in (classify, jobs, mgmt, ops):
            assert client.get("/v1/admin/keys", headers=h).status_code == 403
            assert client.get("/v1/whoami", headers=h).status_code == 200  # any valid key
        assert client.get("/health").status_code == 200  # open routes stay open


def test_403_uses_the_openai_error_shape_on_openai_routes():
    client, _, app = make(api_key="root")
    ops = hdr(add_key(app, "ops", "ops"))
    with client:
        r = client.post("/v1/chat/completions", json={"messages": []}, headers=ops)
    assert r.status_code == 403
    err = r.json()["error"]
    assert err["code"] == "permission_denied" and "lacks scope" in err["message"]


def test_admin_routes_always_need_an_admin_key():
    client, _, app = make()  # auth off
    with client:
        r = client.get("/v1/admin/keys")
        assert r.status_code == 401 and "clef keys create" in r.json()["detail"]
        assert client.get("/v1/admin/audit").status_code == 401
        add_key(app, "adm", "admin")
        assert client.get("/v1/admin/keys").status_code == 401  # now: a key is required


# ------------------------------------------------------------------ quotas over HTTP


@pytest.mark.postgres
def test_quota_headers_429_and_retry_after():
    client, _, app = make(api_key="root")
    secret = add_key(app, "ci", "classify", daily_quota=3)
    batch = {"inputs": ["a", "b"], "labels": ["x", "y"]}
    with client:
        r = client.post("/v1/classify/batch", json=batch, headers=hdr(secret))
        assert r.status_code == 200
        assert (r.headers["X-Quota-Limit"], r.headers["X-Quota-Remaining"]) == ("3", "1")
        r = client.post("/v1/classify/batch", json=batch, headers=hdr(secret))  # 2 > 1 left
        assert r.status_code == 429 and r.json()["detail"] == "daily quota exceeded (3 records/day)"
        assert 0 < int(r.headers["Retry-After"]) <= 86401 and r.headers["X-Quota-Remaining"] == "1"
        last = client.post("/v1/classify", json=CLASSIFY, headers=hdr(secret))
        assert last.headers["X-Quota-Remaining"] == "0"
        who = client.get("/v1/whoami", headers=hdr(secret)).json()
        assert who["daily_quota"] == 3 and who["used_today"] == 3
        usage = client.get("/v1/admin/keys/ci/usage?days=2", headers=hdr("root")).json()["days"][-1]
        assert usage["records"] == 3 and usage["rejected"] == 1 and usage["requests"] == 3


def test_openai_routes_render_insufficient_quota():
    client, _, app = make(api_key="root")
    secret = add_key(app, "ci", "classify", daily_quota=1)
    schema = {
        "type": "object",
        "properties": {"team": {"type": "string", "enum": ["a", "b"]}},
        "required": ["team"],
    }
    body = {
        "model": "clef-flash",
        "messages": [{"role": "user", "content": "x"}],
        "response_format": {"type": "json_schema", "json_schema": {"name": "t", "schema": schema}},
    }
    with client:
        client.post("/v1/classify", json=CLASSIFY, headers=hdr(secret))
        r = client.post("/v1/chat/completions", json=body, headers=hdr(secret))
    assert r.status_code == 429
    err = r.json()["error"]
    assert err["code"] == "insufficient_quota" and err["type"] == "insufficient_quota"


def test_failed_requests_are_not_charged():
    eng = FakeEngine()
    client, _, app = make(eng, api_key="root")
    secret = add_key(app, "ci", "classify", daily_quota=10)
    with client:
        eng.raises = GpuOutOfMemory("oom")
        assert client.post("/v1/classify", json=CLASSIFY, headers=hdr(secret)).status_code == 503
        assert app.state.ctx.quotas.used("ci") == 0
        bad = client.post("/v1/classify", json={"input": "x", "labels": []}, headers=hdr(secret))
        assert bad.status_code == 400 and app.state.ctx.quotas.used("ci") == 0
        eng.raises = None
        ok = client.post("/v1/classify", json=CLASSIFY, headers=hdr(secret))
        assert ok.status_code == 200 and ok.headers["X-Quota-Remaining"] == "9"


class HitEngine(FakeEngine):
    """C2's two-step path with D3's cache: record 0 is a hit."""

    async def prepare(self, records, cache=True):
        n = len(records)
        first = records[0]
        hit = {
            "model": first["model"],
            "answers": {q: self.answer(q, spec) for q, spec in first["questions"].items()},
            "usage": {"input_tokens": 1},
            "timing": {},
        }
        return Prepared(records, [None] * n, {0: hit} if cache else {}, [None] * n)

    async def submit(self, prepared, priority="background"):
        out = await self.decide(prepared.records, priority)
        return [prepared.hits.get(i, r) for i, r in enumerate(out)]


@pytest.mark.postgres
def test_cache_hits_are_not_charged():
    client, _, app = make(HitEngine(), api_key="root")
    secret = add_key(app, "ci", "classify", daily_quota=10)
    batch = {"inputs": ["a", "b", "c"], "labels": ["x", "y"]}
    with client:
        r = client.post("/v1/classify/batch", json=batch, headers=hdr(secret))
        assert r.status_code == 200 and r.headers["X-Quota-Remaining"] == "8"  # 3 records, 1 hit: 2 charged
        assert app.state.ctx.quotas.used("ci") == 2


def test_db_principal_without_quota_is_counted_but_not_limited():
    client, _, app = make(api_key="root")
    secret = add_key(app, "free", "classify")
    with client:
        for _ in range(3):
            r = client.post("/v1/classify", json=CLASSIFY, headers=hdr(secret))
        assert r.status_code == 200 and "x-quota-limit" not in r.headers
        assert app.state.ctx.quotas.used("free") == 3


@pytest.mark.postgres
def test_usage_is_flushed_at_shutdown():
    client, _, app = make(api_key="root")
    secret = add_key(app, "ci", "classify", daily_quota=10)
    url = app.state.ctx.db.url
    with client:
        client.post("/v1/classify", json=CLASSIFY, headers=hdr(secret))
    other = Database(url)  # the app's own connection is closed at shutdown
    try:
        with other.read() as conn:
            assert conn.execute(text("SELECT records FROM api_usage")).scalar() == 1
    finally:
        other.close()


# ------------------------------------------------------------------ rate limit per principal


def test_principal_rate_limit_applies_even_without_a_global_limit():
    client, _, app = make(api_key="root")
    secret = add_key(app, "slow", "classify", rate_limit=2)
    with client:
        codes = [
            client.post("/v1/classify", json=CLASSIFY, headers=hdr(secret)).status_code for _ in range(3)
        ]
        assert codes == [200, 200, 429]
        root = [client.post("/v1/classify", json=CLASSIFY, headers=hdr("root")).status_code for _ in range(5)]
        assert set(root) == {200}


# ------------------------------------------------------------------ jobs: ownership, quota, audit


@pytest.mark.postgres
def test_job_quota_reserve_refund_replay_and_cancel():
    client, _, app = make(SlowEngine(delay=0.15), api_key="root")
    secret = add_key(app, "ci", "classify,jobs", daily_quota=50)
    h = hdr(secret)
    quotas = app.state.ctx.quotas
    with client:
        r = submit(client, headers={**h, "Idempotency-Key": "k1"})
        assert r.status_code == 202 and r.headers["X-Quota-Remaining"] == "47"
        again = submit(client, headers={**h, "Idempotency-Key": "k1"})
        assert again.headers["Idempotent-Replay"] == "true" and quotas.used("ci") == 3  # a replay is free
        wait(client, r.json()["id"], headers=h)
        big = {"inputs": [f"t{i}" for i in range(40)], "labels": LABELS}
        r = submit(client, payload=big, headers=h)
        assert quotas.used("ci") == 43
        over = submit(client, payload=big, headers=h)
        assert over.status_code == 429 and over.json()["detail"] == "daily quota exceeded (50 records/day)"
        assert "Retry-After" in over.headers
        jid = r.json()["id"]
        time.sleep(0.4)
        assert client.post(f"/v1/jobs/{jid}/cancel", headers=h).status_code == 200
        done = wait(client, jid, headers=h)
        assert done["status"] == "cancelled" and 0 < done["progress"]["done"] < 40
        for _ in range(100):  # the refund follows the status change by a moment
            if quotas.used("ci") == 3 + done["progress"]["done"]:
                break
            time.sleep(0.02)
        assert quotas.used("ci") == 3 + done["progress"]["done"]  # the unprocessed part came back
        with app.state.ctx.db.read() as conn:
            kept = conn.execute(text("SELECT quota_reserved, done FROM jobs WHERE id = :i"), {"i": jid}).one()
        assert kept.quota_reserved == kept.done


@pytest.mark.postgres
def test_settle_quota_refunds_once_and_only_for_failed_or_cancelled_jobs(db):
    from clef_server.jobs import JobStore

    store = JobStore(db)
    day = "2026-10-17"
    jobs = {
        status: store.create("classify", "ci", {}, None, None, 0, None, None, day, 10)["id"]
        for status in ("cancelled", "failed", "succeeded", "queued")
    }
    with db.tx() as conn:
        for status, jid in jobs.items():
            if status != "queued":
                conn.execute(
                    text("UPDATE jobs SET status = :s, done = 4 WHERE id = :i"), {"s": status, "i": jid}
                )
    assert sorted(store.settle_quota()) == [("ci", day, 6), ("ci", day, 6)]
    assert store.settle_quota() == []  # settled: quota_reserved now equals what was charged (done)
    assert store.settle_quota(jobs["cancelled"]) == []


def test_job_quota_is_refunded_when_the_job_cannot_be_created():
    client, _, app = make(SlowEngine(delay=0.5), api_key="root", max_jobs=1)
    secret = add_key(app, "ci", "classify,jobs", daily_quota=50)
    with client:
        assert submit(client, headers=hdr(secret)).status_code == 202
        full = submit(client, headers=hdr(secret))
        assert full.status_code == 409  # JobsFull: nothing queued, nothing charged
        assert app.state.ctx.quotas.used("ci") == 3


def test_admins_see_every_job_and_others_only_their_own():
    client, _, app = make(api_key="root")
    a, b = hdr(add_key(app, "alice", "jobs,classify")), hdr(add_key(app, "bob", "jobs"))
    adm = hdr(add_key(app, "boss", "admin"))
    with client:
        jid = submit(client, headers=a).json()["id"]
        wait(client, jid, headers=a)
        assert client.get(f"/v1/jobs/{jid}", headers=b).status_code == 404
        assert client.get("/v1/jobs", headers=b).json()["total"] == 0
        assert client.get(f"/v1/jobs/{jid}", headers=adm).status_code == 200
        assert client.get("/v1/jobs", headers=adm).json()["total"] == 1
        assert client.get(f"/v1/jobs/{jid}/results", headers=hdr("root")).status_code == 200
        assert client.delete(f"/v1/jobs/{jid}", headers=adm).status_code == 200
        entries = KeyStore(app.state.ctx.db).audit_entries(action="job.")[0]
    assert [(e["actor"], e["action"], e["target"]) for e in entries] == [("boss", "job.delete", jid)]


def test_job_cancel_is_audited():
    client, _, app = make(SlowEngine(delay=0.3), api_key="root")
    h = hdr(add_key(app, "alice", "jobs"))
    payload = {"inputs": [f"t{i}" for i in range(30)], "labels": LABELS}
    with client:
        jid = submit(client, payload=payload, headers=h).json()["id"]
        client.post(f"/v1/jobs/{jid}/cancel", headers=h)
        wait(client, jid, headers=h)
        entries = KeyStore(app.state.ctx.db).audit_entries(action="job.")[0]
    assert entries[0]["action"] == "job.cancel" and entries[0]["actor"] == "alice"
    assert entries[0]["detail"]["owner"] == "alice"


# ------------------------------------------------------------------ admin API


@pytest.mark.postgres
def test_admin_api_lifecycle():
    client, _, app = make(api_key="root")
    root = hdr("root")
    new = {
        "name": "ci",
        "scopes": ["classify", "jobs"],
        "daily_quota": 100,
        "expires_in_days": 30,
        "note": "n",
    }
    with client:
        r = client.post("/v1/admin/keys", json=new, headers=root)
        assert r.status_code == 201 and r.headers["Cache-Control"] == "no-store"
        body = r.json()
        secret = body["key"]
        assert body["name"] == "ci" and body["keys"][0]["display"].endswith("_…") and body["used_today"] == 0
        assert body["keys"][0]["expires_at"] > time.time() + 29 * 86400
        # the new key works at once (the write reloaded the cache)
        assert client.post("/v1/classify", json=CLASSIFY, headers=hdr(secret)).status_code == 200
        listed = client.get("/v1/admin/keys", headers=root).json()["principals"]
        assert [(p["name"], p["source"]) for p in listed] == [("default", "env"), ("ci", "db")]
        assert listed[0]["keys"] == [] and secret not in json.dumps(listed)
        assert listed[1]["used_today"] == 1
        patch = {"daily_quota": None, "scopes": ["classify"]}
        r = client.patch("/v1/admin/keys/ci", json=patch, headers=root)
        assert r.status_code == 200 and r.json()["daily_quota"] is None and r.json()["scopes"] == ["classify"]
        job = {"kind": "classify", "payload": {}}
        assert client.post("/v1/jobs", json=job, headers=hdr(secret)).status_code == 403
        rot = client.post("/v1/admin/keys/ci/rotate", json={"grace_s": 60}, headers=root)
        assert rot.status_code == 201 and rot.json()["key"] != secret
        assert client.post("/v1/classify", json=CLASSIFY, headers=hdr(secret)).status_code == 200  # in grace
        assert [k["status"] for k in rot.json()["keys"]] == ["expiring", "active"]
        first_id = rot.json()["keys"][0]["id"]
        revoke = client.post("/v1/admin/keys/ci/revoke", json={"key_id": first_id}, headers=root)
        assert revoke.status_code == 200
        assert client.post("/v1/classify", json=CLASSIFY, headers=hdr(secret)).status_code == 401  # at once
        newest = hdr(rot.json()["key"])
        assert client.post("/v1/classify", json=CLASSIFY, headers=newest).status_code == 200
        gone = client.post("/v1/admin/keys/ci/revoke", headers=root)
        assert gone.json()["disabled_at"]
        assert client.post("/v1/classify", json=CLASSIFY, headers=newest).status_code == 401
        assert len(client.get("/v1/admin/keys", headers=root).json()["principals"]) == 1
        assert len(client.get("/v1/admin/keys?include_disabled=true", headers=root).json()["principals"]) == 2
        audit = client.get("/v1/admin/audit?action=key.", headers=root).json()
        assert [e["action"] for e in audit["entries"]] == ["key.revoke", "key.rotate", "key.create"]
        assert audit["entries"][0]["actor"] == "default" and audit["next_before"] is None
        page = client.get("/v1/admin/audit?limit=1", headers=root).json()
        assert len(page["entries"]) == 1 and page["next_before"] == page["entries"][0]["id"]
        assert secret not in json.dumps(audit)


@pytest.mark.postgres
def test_admin_api_errors_and_guards():
    client, _, app = make(api_keys_raw="root:kr")
    root = hdr("kr")
    jobs = ["jobs"]
    with client:
        c = client.post
        assert c("/v1/admin/keys", json={"name": "a b", "scopes": jobs}, headers=root).status_code == 400
        assert c("/v1/admin/keys", json={"name": "a", "scopes": ["bogus"]}, headers=root).status_code == 400
        assert (
            c("/v1/admin/keys", json={"name": "a", "scopes": jobs, "x": 1}, headers=root).status_code == 400
        )
        assert c("/v1/admin/keys", json={"name": "root", "scopes": jobs}, headers=root).status_code == 409
        adm = c("/v1/admin/keys", json={"name": "adm", "scopes": ["admin"]}, headers=root).json()["key"]
        assert c("/v1/admin/keys", json={"name": "adm", "scopes": ["admin"]}, headers=root).status_code == 409
        assert client.patch("/v1/admin/keys/root", json={"note": "x"}, headers=root).status_code == 403
        assert c("/v1/admin/keys/root/rotate", headers=root).status_code == 403
        assert c("/v1/admin/keys/root/revoke", headers=root).status_code == 403
        assert client.patch("/v1/admin/keys/nope", json={"note": "x"}, headers=root).status_code == 404
        assert c("/v1/admin/keys/adm/revoke", json={"key_id": "nope"}, headers=root).status_code == 404
        assert client.get("/v1/admin/keys/nope/usage", headers=root).status_code == 404
        # a DB admin cannot lock itself out without ?force=true
        me = hdr(adm)
        assert client.patch("/v1/admin/keys/adm", json={"scopes": jobs}, headers=me).status_code == 403
        assert c("/v1/admin/keys/adm/revoke", headers=me).status_code == 403
        only = client.get("/v1/admin/keys", headers=me).json()["principals"][-1]["keys"][0]["id"]
        assert c("/v1/admin/keys/adm/revoke", json={"key_id": only}, headers=me).status_code == 403
        assert c("/v1/admin/keys/adm/rotate", headers=me).status_code == 201  # a second key
        assert c("/v1/admin/keys/adm/revoke", json={"key_id": only}, headers=me).status_code == 200
        assert c("/v1/admin/keys", json={"name": "five", "scopes": jobs}, headers=root).status_code == 201
        codes = [c("/v1/admin/keys/five/rotate", headers=root).status_code for _ in range(5)]
        assert codes == [201, 201, 201, 201, 409]  # 5 active keys at most


def test_admin_routes_are_rate_limited():
    client, _, _ = make(api_key="root", rate_limit=2)
    with client:
        codes = [client.get("/v1/admin/keys", headers=hdr("root")).status_code for _ in range(3)]
    assert codes == [200, 200, 429]


def clash_app(scopes: str = "classify,jobs,classifiers"):
    """A DB principal 'ci' that already exists when an env key 'ci' is added later (same database)."""
    sd = tempfile.mkdtemp(prefix="clef-clash-")
    _, _, app = make(state_dir=sd)
    secret = add_key(app, "ci", scopes)
    app.state.ctx.db.close()
    client, _, app2 = make(state_dir=sd, api_keys_raw="ci:envsecret,root:rootsecret")
    return client, app2, secret


def test_env_name_clash_is_warned(caplog):
    with caplog.at_level("WARNING", logger="clef"):
        clash_app()
    msgs = [r.getMessage() for r in caplog.records]
    assert any(
        "both an env key and a DB principal" in m and "refused" in m and "clef keys revoke" in m for m in msgs
    )
    assert not any("env key wins" in m for m in msgs)


@pytest.mark.postgres
def test_db_key_named_like_an_env_key_is_refused_and_revocable():
    client, _, secret = clash_app()
    env, root = hdr("envsecret"), hdr("rootsecret")
    with client:
        jid = submit(client, headers=env).json()["id"]
        assert (
            client.put("/v1/classifiers/secretclf", json={"labels": ["a", "b"]}, headers=env).status_code
            == 200
        )
        # the DB key must not inherit the env principal's ownership or its rate-limit bucket
        assert client.get("/v1/whoami", headers=hdr(secret)).status_code == 401
        assert client.get(f"/v1/jobs/{jid}", headers=hdr(secret)).status_code == 401
        assert client.delete("/v1/classifiers/secretclf", headers=hdr(secret)).status_code == 401
        assert client.post("/v1/classify", json=CLASSIFY, headers=hdr(secret)).status_code == 401
        assert client.get("/v1/whoami", headers=env).json()["source"] == "env"
        # an admin can still act on the clashing DB row
        assert client.post("/v1/admin/keys/ci/rotate", headers=root).status_code == 201
        assert client.patch("/v1/admin/keys/ci", json={"note": "x"}, headers=root).status_code == 200
        assert client.post("/v1/admin/keys/ci/revoke", json={}, headers=root).status_code == 200
        # a pure env key has no DB row: still read-only
        assert client.post("/v1/admin/keys/root/revoke", json={}, headers=root).status_code == 403


@pytest.mark.postgres
def test_several_env_keys_are_all_admin_and_the_server_says_so(caplog):
    """D10: CLEF_API_KEYS=alice:..,bob:.. keeps working as before the D1 upgrade, but every env key is a full
    admin. That is deliberate and loud: a startup warning, and the keys see each other's work."""
    with caplog.at_level("WARNING", logger="clef"):
        client, _, _ = make(api_keys_raw="alice:ka,bob:kb")
    assert any(
        "2 env keys" in r.getMessage() and "clef keys create" in r.getMessage() for r in caplog.records
    )
    a, b = hdr("ka"), hdr("kb")
    with client:
        jid = submit(client, headers=a).json()["id"]
        wait(client, jid, headers=a)
        assert client.get(f"/v1/jobs/{jid}", headers=b).status_code == 200
        assert client.get(f"/v1/jobs/{jid}/results", headers=b).status_code == 200
        assert client.get("/v1/jobs", headers=b).json()["total"] == 1
        assert client.put("/v1/classifiers/mine", json={"labels": ["a", "b"]}, headers=a).status_code == 200
        assert client.put("/v1/classifiers/mine", json={"labels": ["c", "d"]}, headers=b).status_code == 200
        assert client.delete("/v1/classifiers/mine", headers=b).status_code == 200
        assert client.get("/v1/admin/audit", headers=b).status_code == 200
        assert client.delete(f"/v1/jobs/{jid}", headers=b).status_code == 200


def test_a_single_env_key_gets_no_multi_key_warning(caplog):
    with caplog.at_level("WARNING", logger="clef"):
        make(api_keys_raw="alice:ka")
    assert not any("env keys" in r.getMessage() for r in caplog.records)


@pytest.mark.postgres
def test_owner_less_rows_from_the_auth_off_era_are_admin_only_once_auth_is_on():
    client, _, app = make(api_keys_raw="root:kr")  # root is only here to be an admin later
    state = app.state.cfg.state_dir
    app.state.ctx.db.close()
    off, _, app_off = make(state_dir=state)  # auth was off: everything created has no owner
    with off:
        jid = submit(off).json()["id"]
        wait(off, jid)
        assert off.put("/v1/classifiers/old", json={"labels": ["a", "b"]}).status_code == 200
    app_off.state.ctx.db.close()
    client, _, app = make(state_dir=state, api_keys_raw="root:kr")
    mallory = hdr(add_key(app, "mallory", "classify,jobs,classifiers"))
    root = hdr("kr")
    with client:
        # jobs: invisible to non-admins (404), so neither readable nor modifiable
        assert client.get(f"/v1/jobs/{jid}/results", headers=mallory).status_code == 404
        assert client.get(f"/v1/jobs/{jid}", headers=mallory).status_code == 404
        assert client.post(f"/v1/jobs/{jid}/cancel", headers=mallory).status_code == 404
        assert client.delete(f"/v1/jobs/{jid}", headers=mallory).status_code == 404
        assert client.get("/v1/jobs", headers=mallory).json()["total"] == 0
        assert client.get(f"/v1/jobs/{jid}/results", headers=root).status_code == 200
        # classifiers: running stays allowed, changing and deleting are admin-only
        body = {"labels": ["x", "y"]}
        assert client.put("/v1/classifiers/old", json=body, headers=mallory).status_code == 403
        assert client.delete("/v1/classifiers/old", headers=mallory).status_code == 403
        assert client.post("/v1/classifiers/old", json={"input": "hi"}, headers=mallory).status_code == 200
        assert client.get("/v1/classifiers/old", headers=mallory).status_code == 200
        assert client.put("/v1/classifiers/old", json=body, headers=root).status_code == 200
        assert client.delete("/v1/classifiers/old", headers=root).status_code == 200
        assert client.delete(f"/v1/jobs/{jid}", headers=root).status_code == 200


def test_owner_403_does_not_name_the_owner():
    client, _, app = make(api_keys_raw="root:kr")
    alice, bob = hdr(add_key(app, "alice")), hdr(add_key(app, "bob"))
    with client:
        assert client.put("/v1/classifiers/c", json={"labels": ["a", "b"]}, headers=alice).status_code == 200
        r = client.put("/v1/classifiers/c", json={"labels": ["x", "y"]}, headers=bob)
        assert r.status_code == 403 and "alice" not in r.text and "owner or an admin" in r.text


# ------------------------------------------------------------------ default-deny


def _scope_marks(dependant: Any) -> list[tuple[str, ...]]:
    found = []
    for dep in dependant.dependencies:
        mark = getattr(dep.call, "clef_scope", None)
        if mark is not None:
            found.append(tuple(mark))
        found += _scope_marks(dep)
    return found


def api_routes(app: Any) -> list[Any]:
    """Routes with a dependant. Newer FastAPI keeps included routers lazy (effective route contexts)."""
    out = []
    for route in app.routes:
        if hasattr(route, "effective_route_contexts"):
            out += list(route.effective_route_contexts())
        else:
            out.append(route)
    return [r for r in out if getattr(r, "dependant", None) is not None]


def test_every_api_route_has_a_scope_dependency():
    _, _, app = make()
    routes = api_routes(app)
    guarded = [r for r in routes if r.path.startswith(("/v1/", "/hf/", "/models/"))]
    missing = [f"{sorted(r.methods)} {r.path}" for r in guarded if not _scope_marks(r.dependant)]
    assert len(guarded) > 30
    assert not missing, f"routes without ctx.scope(...) (default deny): {missing}"
    whoami = next(r for r in guarded if r.path == "/v1/whoami")
    assert _scope_marks(whoami.dependant) == [()]  # explicit: any valid key
    admin_routes = [r for r in guarded if r.path.startswith("/v1/admin")]
    assert len(admin_routes) == 7 and all(("admin",) in _scope_marks(r.dependant) for r in admin_routes)
    open_paths = {r.path for r in routes if r not in guarded}
    assert {"/health", "/livez", "/readyz", "/schema-example"} <= open_paths


# ------------------------------------------------------------------ CLI (direct DB, server off)


def run_cli(*argv: str) -> int:
    from clef_server.cli import main

    return main(["keys", *argv])


def test_cli_bootstrap_on_a_fresh_state_dir(capsys):
    assert run_cli("list") == 0
    assert "no principals" in capsys.readouterr().out
    assert run_cli("create", "boss", "--scope", "admin", "--note", "first") == 0
    out = capsys.readouterr().out
    secret = next(line.split(": ", 1)[1] for line in out.splitlines() if line.startswith("key: "))
    assert ks.parse_key(secret) and "auth is now required for every /v1 route" in out
    assert (
        run_cli("create", "ci", "--scope", "classify,jobs", "--quota", "100", "--expires", "30d", "--json")
        == 0
    )
    created = json.loads(capsys.readouterr().out)
    assert created["key"].startswith("clef_live_") and created["daily_quota"] == 100
    assert created["keys"][0]["expires_at"] > time.time() + 29 * 86400
    assert run_cli("update", "ci", "--quota", "none", "--rate-limit", "30", "--scope", "jobs") == 0
    capsys.readouterr()
    assert run_cli("rotate", "ci", "--grace", "2h", "--json") == 0
    rotated = json.loads(capsys.readouterr().out)
    assert rotated["daily_quota"] is None and rotated["rate_limit"] == 30 and rotated["scopes"] == ["jobs"]
    assert [k["status"] for k in rotated["keys"]] == ["expiring", "active"]
    assert run_cli("list", "--json") == 0
    names = [p["name"] for p in json.loads(capsys.readouterr().out)["principals"]]
    assert names == ["boss", "ci"]
    assert run_cli("revoke", "ci") == 0
    capsys.readouterr()
    assert run_cli("list") == 0
    assert "ci" not in capsys.readouterr().out  # disabled principals are hidden...
    assert run_cli("list", "--all") == 0
    assert "ci (disabled)" in capsys.readouterr().out  # ...unless asked for
    assert run_cli("usage", "ci", "--days", "2") == 0
    capsys.readouterr()
    # the server sees the CLI's writes (a fresh app = a fresh cache)
    client, _, _ = make(state_dir=os.environ["CLEF_STATE_DIR"])
    with client:
        assert client.get("/v1/whoami").status_code == 401  # a DB principal exists: auth is on
        assert client.get("/v1/whoami", headers=hdr(secret)).json()["name"] == "boss"


def test_cli_errors(capsys):
    assert run_cli("create", "x", "--scope", "nope") == 1
    assert "unknown scope" in capsys.readouterr().err
    assert run_cli("create", "x", "--scope", "jobs") == 0
    assert run_cli("create", "x", "--scope", "jobs") == 1
    assert "already exists" in capsys.readouterr().err
    assert run_cli("update", "x") == 1
    assert run_cli("revoke", "nope") == 1
    assert run_cli("usage", "nope") == 1
    assert run_cli("create", "e", "--scope", "jobs", "--expires", "soon") == 1
    capsys.readouterr()


def test_cli_env_keys_are_listed_and_protected(monkeypatch, capsys):
    monkeypatch.setenv("CLEF_API_KEYS", "root:abc")
    assert run_cli("list") == 0
    assert "root" in capsys.readouterr().out
    for cmd in (["rotate", "root"], ["revoke", "root"], ["update", "root", "--note", "x"]):
        assert run_cli(*cmd) == 1
        assert "env key" in capsys.readouterr().err
    assert run_cli("create", "root", "--scope", "jobs") == 1
    assert "env key" in capsys.readouterr().err


def test_cli_can_revoke_a_db_principal_that_clashes_with_an_env_key(monkeypatch, capsys):
    assert run_cli("create", "ci", "--scope", "jobs") == 0
    capsys.readouterr()
    monkeypatch.setenv("CLEF_API_KEYS", "ci:abc")  # the env grew a key with the same name
    assert run_cli("revoke", "ci") == 0
    assert "disabled" in capsys.readouterr().out


def test_cli_expiry_parsing():
    from clef_server.cli import _parse_duration, _parse_expiry

    assert (
        _parse_duration("24h") == 86400
        and _parse_duration("90d") == 90 * 86400
        and _parse_duration("30") == 30
    )
    end = _parse_expiry("2027-01-31")
    assert datetime.fromtimestamp(end, timezone.utc) == datetime(2027, 2, 1, tzinfo=timezone.utc)
    assert abs(_parse_expiry("1d") - (time.time() + 86400)) < 5
    for bad in ("soon", "2027-13-01"):
        with pytest.raises(ValueError):
            _parse_expiry(bad)
