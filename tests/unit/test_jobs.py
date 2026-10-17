"""Async job API: lifecycle, validation, fairness, cancel, resume, ownership, retention (FakeEngine only)."""

from __future__ import annotations

import asyncio
import csv
import io
import json
import sqlite3
import tempfile
import time
from typing import Any

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import text
from sqlalchemy.exc import DBAPIError
from test_api import FakeEngine

from clef_server.config import Config
from clef_server.db import Database, database_url, open_database
from clef_server.engine import InputTooLarge
from clef_server.jobs import JobStore, public_job
from clef_server.main import create_app
from clef_server.paths import jobs_db

pytestmark = pytest.mark.postgres  # also run on Postgres when CLEF_TEST_DATABASE_URL is set (conftest.py)

LABELS = ["billing", "technical"]


def open_store(cfg: Config) -> JobStore:
    """The job store on cfg's database, as a second process would open it (no server lock)."""
    db = Database(database_url(cfg))
    db.upgrade()
    return JobStore(db)


def hold_runner(monkeypatch: pytest.MonkeyPatch, store: JobStore) -> list[bool]:
    """Keep the runner from admitting jobs a test sets up by hand; `.clear()` releases it."""
    hold, real = [True], store.runnable
    monkeypatch.setattr(store, "runnable", lambda: [] if hold else real())
    return hold


def sql(store: JobStore, stmt: str, **params: Any) -> list[Any]:
    """Raw SQL for test set-up (the store's own connection discipline)."""
    with store.db.tx() as conn:
        res = conn.execute(text(stmt), params)
        return res.all() if res.returns_rows else []


class SlowEngine(FakeEngine):
    """Serialises decide() calls (like the single GPU worker) and takes ``delay`` s per call."""

    def __init__(self, delay: float = 0.0) -> None:
        super().__init__()
        self.delay = delay
        self._lock: asyncio.Lock | None = None
        self.sizes: list[int] = []

    async def decide(
        self, records: list[dict[str, Any]], priority: str = "interactive"
    ) -> list[dict[str, Any]]:
        if self._lock is None:
            self._lock = asyncio.Lock()  # fair (FIFO) like the engine queue
        async with self._lock:
            if any("TOOLONG" in json.dumps(r["state"]) for r in records):
                raise InputTooLarge("input is too long (99999 tokens)")
            self.sizes.append(len(records))
            await asyncio.sleep(self.delay)
            return await super().decide(records)


def make(engine: FakeEngine | None = None, **cfg_kw: Any):
    eng = engine or SlowEngine()
    cfg_kw.setdefault("state_dir", tempfile.mkdtemp(prefix="clef-jobs-"))
    app = create_app(Config(**{"api_key": None, **cfg_kw}), eng)
    return TestClient(app, raise_server_exceptions=False), eng, app


def submit(client: TestClient, kind: str = "classify", payload: dict | None = None, headers=None, **body_kw):
    body = {"kind": kind, "payload": payload or {"inputs": ["a", "b", "c"], "labels": LABELS}, **body_kw}
    return client.post("/v1/jobs", json=body, headers=headers)


def wait(client: TestClient, job_id: str, until=("succeeded", "failed", "cancelled"), timeout=15.0, **kw):
    t0 = time.monotonic()
    while True:
        job = client.get(f"/v1/jobs/{job_id}", **kw).json()
        if job["status"] in until:
            return job
        assert time.monotonic() - t0 < timeout, f"timeout waiting for {until}: {job}"
        time.sleep(0.02)


def all_rows(client: TestClient, job_id: str) -> list[dict]:
    r = client.get(f"/v1/jobs/{job_id}/results", params={"format": "ndjson"})
    assert r.status_code == 200
    return [json.loads(line) for line in r.text.splitlines()]


# ------------------------------------------------------------------ lifecycle


def test_classify_job_lifecycle():
    client, _eng, _ = make()
    with client:
        r = submit(client, payload={"inputs": ["a", "b", "c"], "labels": LABELS}, metadata={"batch": 7})
        assert r.status_code == 202, r.text
        job = r.json()
        assert r.headers["Location"] == f"/v1/jobs/{job['id']}"
        assert job["status"] in ("queued", "running", "succeeded") and job["kind"] == "classify"
        done = wait(client, job["id"])
        assert done["status"] == "succeeded", done
        assert done["progress"] == {"done": 3, "total": 3, "failed": 0, "percent": 100.0}
        assert done["metadata"] == {"batch": 7} and done["error"] is None
        assert done["result"] == {"items": 3, "ok": 3, "errors": 0, "by_label": {"technical": 3}}
        assert done["started_at"] and done["finished_at"] and done["duration_s"] >= 0
        page = client.get(f"/v1/jobs/{job['id']}/results").json()
        assert page["count"] == 3 and page["total"] == 3 and page["next_offset"] is None
        row = page["items"][0]
        assert row["index"] == 0 and row["label"] == "technical" and set(row["scores"]) == set(LABELS)
        assert [x["index"] for x in page["items"]] == [0, 1, 2]


def test_results_paging_ndjson_and_csv():
    client, _eng, _ = make()
    with client:
        jid = submit(
            client, payload={"inputs": [f"t{i}" for i in range(5)] + ["TOOLONG"], "labels": LABELS}
        ).json()["id"]
        wait(client, jid)
        p1 = client.get(f"/v1/jobs/{jid}/results", params={"limit": 4}).json()
        assert p1["count"] == 4 and p1["next_offset"] == 4
        p2 = client.get(f"/v1/jobs/{jid}/results", params={"offset": 4, "limit": 4}).json()
        assert p2["count"] == 2 and p2["next_offset"] is None
        assert [r["index"] for r in p1["items"] + p2["items"]] == list(range(6))
        rows = all_rows(client, jid)
        assert len(rows) == 6 and "error" in rows[5] and "too long" in rows[5]["error"]
        sub = client.get(f"/v1/jobs/{jid}/results", params={"format": "ndjson", "offset": 2, "limit": 2})
        assert [json.loads(x)["index"] for x in sub.text.splitlines()] == [2, 3]
        r = client.get(f"/v1/jobs/{jid}/results", params={"format": "csv"})
        assert (
            r.headers["content-type"].startswith("text/csv") and r.headers["X-Clef-Job-Status"] == "succeeded"
        )
        table = list(csv.DictReader(io.StringIO(r.text)))
        assert len(table) == 6 and list(table[0])[0] == "index" and list(table[0])[-1] == "error"
        assert table[0]["label"] == "technical" and table[0]["scores.billing"] != "" and table[5]["error"]
        assert table[5]["label"] == ""


def test_per_item_failure_does_not_fail_job():
    client, eng, _ = make()
    with client:
        inputs = ["ok1", "TOOLONG", "ok2", "ok3", "ok4", "TOOLONG too", "ok5", "ok6", "ok7"]
        jid = submit(client, payload={"inputs": inputs, "labels": LABELS}).json()["id"]
        job = wait(client, jid)
        assert job["status"] == "succeeded"
        assert job["progress"]["failed"] == 2 and job["result"]["errors"] == 2 and job["result"]["ok"] == 7
        rows = all_rows(client, jid)
        assert [("error" in r) for r in rows] == [x.startswith("TOOLONG") for x in inputs]
        assert max(eng.sizes) > 1  # healthy records were still batched together


def test_score_systemone_and_saved_classifier_kinds():
    client, _eng, _ = make()
    with client:
        sc = submit(client, "score", {"inputs": ["x", "y"], "levels": ["low", "high"]}).json()["id"]
        job = wait(client, sc)
        assert job["status"] == "succeeded" and job["result"]["by_level"] == {"high": 2}
        assert all_rows(client, sc)[0]["level"] == "high"

        one = {"state": "t", "questions": {"o": {"type": "noul"}}}
        so = submit(client, "systemone", {"requests": [one, one, one]}).json()["id"]
        job = wait(client, so)
        assert job["result"] == {"items": 3, "ok": 3, "errors": 0}
        assert "o" in all_rows(client, so)[1]["answers"]

        client.put("/v1/classifiers/triage", json={"kind": "classify", "labels": LABELS, "multi_label": True})
        c = submit(client, "classify", {"inputs": ["a"], "classifier": "triage"}).json()["id"]
        wait(client, c)
        row = all_rows(client, c)[0]
        assert "labels" in row and row["threshold"] == 0.5  # multi-label settings come from the classifier
        client.put("/v1/classifiers/lvl", json={"kind": "score", "levels": ["a", "b"]})
        assert submit(client, "classify", {"inputs": ["a"], "classifier": "lvl"}).status_code == 400
        assert submit(client, "score", {"inputs": ["a"], "classifier": "lvl"}).status_code == 202


@pytest.mark.parametrize(
    ("kind", "payload", "needle"),
    [
        ("nope", {}, "unknown job kind"),
        ("classify", {"inputs": [], "labels": LABELS}, "payload.inputs: inputs must contain at least one"),
        ("classify", {"inputs": ["a"]}, "payload.labels: give exactly one of labels or classifier"),
        ("classify", {"inputs": ["a"], "labels": LABELS, "classifier": "x"}, "exactly one"),
        ("classify", {"inputs": ["a"], "classifier": "missing"}, "payload.classifier: 'missing' not found"),
        ("classify", {"inputs": ["a"], "labels": ["only"]}, "payload.labels: at least 2 labels"),
        ("classify", {"inputs": ["a"], "labels": LABELS, "threshold": 0.5}, "threshold: only allowed"),
        ("classify", {"inputs": ["a"], "labels": LABELS, "bogus": 1}, "payload.bogus: unknown field"),
        ("classify", {"inputs": list("abcd"), "labels": LABELS}, "too many items (max 3"),
        ("score", {"inputs": ["a"], "levels": ["one"]}, "payload.levels: at least 2 levels"),
        ("systemone", {"requests": [{"state": "x", "questions": {}}]}, "payload.requests[0].questions"),
    ],
)
def test_submit_validation(kind, payload, needle):
    client, _eng, _ = make(max_job_items=3)
    with client:
        r = client.post("/v1/jobs", json={"kind": kind, "payload": payload})
        assert r.status_code == 400, r.text
        assert needle in r.json()["detail"] and "request_id" in r.json()


def test_submit_body_validation_and_metadata_cap():
    client, _eng, _ = make()
    with client:
        assert client.post("/v1/jobs", json={"payload": {}}).status_code == 400
        assert client.post("/v1/jobs", content=b"nope").status_code == 400
        big = {
            "kind": "classify",
            "payload": {"inputs": ["a"], "labels": LABELS},
            "metadata": {"x": "y" * 20000},
        }
        r = client.post("/v1/jobs", json=big)
        assert r.status_code == 400 and "metadata" in r.json()["detail"]


def test_max_jobs_409_and_list_filter():
    eng = SlowEngine(delay=0.3)
    client, _, _ = make(eng, max_jobs=2)
    with client:
        ids = [submit(client).json()["id"] for _ in range(2)]
        r = submit(client)
        assert r.status_code == 409 and r.headers["Retry-After"]
        listing = client.get("/v1/jobs").json()
        assert [j["id"] for j in listing["jobs"]] == ids[::-1] and listing["total"] == 2  # newest first
        assert client.get("/v1/jobs", params={"status": "succeeded"}).json()["total"] == 0
        assert client.get("/v1/jobs", params={"limit": 1, "offset": 1}).json()["jobs"][0]["id"] == ids[0]
        assert client.get("/v1/jobs", params={"status": "bogus"}).status_code == 400
        for i in ids:
            wait(client, i)
        assert submit(client).status_code == 202  # capacity frees up when jobs finish
        assert client.get("/v1/jobs", params={"status": "succeeded"}).json()["total"] >= 2


def test_fifo_one_at_a_time_with_a_single_slot():
    eng = SlowEngine(delay=0.05)
    client, _, _ = make(eng, job_active=1)  # the default (4) lets jobs advance in turns: test_scheduler.py
    with client:
        a = submit(client, payload={"inputs": list("abcdefgh") * 3, "labels": LABELS}).json()["id"]
        b = submit(client).json()["id"]
        wait(client, a, until=("running", "succeeded"))
        assert client.get(f"/v1/jobs/{b}").json()["status"] in ("queued",)  # not started in parallel
        ja, jb = wait(client, a), wait(client, b)
        assert ja["finished_at"] <= jb["started_at"]


# ------------------------------------------------------------------ fairness


def test_interactive_requests_interleave_with_a_running_job():
    chunk = 0.3
    eng = SlowEngine(delay=chunk)
    client, _, _ = make(eng, max_microbatch=8)
    with client:
        jid = submit(client, payload={"inputs": [f"t{i}" for i in range(80)], "labels": LABELS}).json()["id"]
        wait(client, jid, until=("running",))
        wait_progress = time.monotonic()
        while client.get(f"/v1/jobs/{jid}").json()["progress"]["done"] < 8:
            assert time.monotonic() - wait_progress < 10
            time.sleep(0.01)
        latencies = []
        for _ in range(3):
            t0 = time.perf_counter()
            r = client.post("/v1/classify", json={"input": "hi", "labels": LABELS})
            latencies.append(time.perf_counter() - t0)
            assert r.status_code == 200
            time.sleep(0.11)
        # one engine call in flight at a time: an interactive call waits for the current chunk, then goes
        # (the whole job is 10 chunks = 3 s; without interleaving the call would wait for it)
        assert max(latencies) < 3 * chunk, latencies  # <= current chunk + its own, plus jitter
        job = wait(client, jid)
        assert job["status"] == "succeeded" and job["progress"]["done"] == 80
        sizes = [s for s in eng.sizes if s > 1]
        assert sizes and max(sizes) <= 8  # the job never sends more than one micro-batch at a time
        print(f"fairness: interactive latencies {[round(x, 3) for x in latencies]} s, chunk {chunk} s")


# ------------------------------------------------------------------ cancel / delete


def test_cancel_running_job_between_chunks():
    eng = SlowEngine(delay=0.1)
    client, _, _ = make(eng)
    with client:
        jid = submit(client, payload={"inputs": [f"t{i}" for i in range(80)], "labels": LABELS}).json()["id"]
        while client.get(f"/v1/jobs/{jid}").json()["progress"]["done"] < 8:
            time.sleep(0.01)
        r = client.post(f"/v1/jobs/{jid}/cancel")
        assert r.status_code == 200 and r.json()["cancel_requested"] is True
        job = wait(client, jid)
        assert job["status"] == "cancelled" and 8 <= job["progress"]["done"] < 80
        assert len(all_rows(client, jid)) == job["progress"]["done"]  # partial results are kept
        assert client.post(f"/v1/jobs/{jid}/cancel").status_code == 409
        calls = len(eng.sizes)
        time.sleep(0.4)
        assert len(eng.sizes) == calls  # nothing runs after the cancel took effect


def test_cancel_queued_job_and_delete_rules():
    eng = SlowEngine(delay=0.1)
    client, _, _ = make(eng, job_active=1)
    with client:
        a = submit(client, payload={"inputs": [f"t{i}" for i in range(40)], "labels": LABELS}).json()["id"]
        b = submit(client).json()["id"]
        r = client.post(f"/v1/jobs/{b}/cancel")
        assert r.status_code == 200 and r.json()["status"] == "cancelled"
        assert client.delete(f"/v1/jobs/{a}").status_code == 409  # active: cancel first
        client.post(f"/v1/jobs/{a}/cancel")
        wait(client, a)
        assert client.delete(f"/v1/jobs/{a}").json() == {"deleted": a}
        assert client.get(f"/v1/jobs/{a}").status_code == 404
        assert client.get(f"/v1/jobs/{a}/results").status_code == 404
        assert client.delete(f"/v1/jobs/{a}").status_code == 404
        assert client.post("/v1/jobs/nope/cancel").status_code == 404


# ------------------------------------------------------------------ engine availability


def test_engine_not_ready_waits_instead_of_failing():
    eng = SlowEngine()
    eng.status = "loading"
    client, _, _ = make(eng)
    with client:
        jid = submit(client).json()["id"]
        time.sleep(0.8)
        assert client.get(f"/v1/jobs/{jid}").json()["status"] == "running"
        eng.status = "ready"
        assert wait(client, jid)["status"] == "succeeded"


def test_engine_gone_for_too_long_fails_the_job():
    eng = SlowEngine()
    eng.status = "error"
    eng.error = "weights missing"
    client, _, _ = make(eng, job_engine_wait_s=0.6)
    with client:
        job = wait(client, submit(client).json()["id"])
        assert job["status"] == "failed" and "engine not available" in job["error"]


def test_systemic_error_fails_job_without_leaking():
    eng = SlowEngine()
    eng.raises = RuntimeError("secret internal detail")
    client, _, _ = make(eng)
    with client:
        job = wait(client, submit(client).json()["id"])
        assert job["status"] == "failed" and job["error"] == "internal server error"


# ------------------------------------------------------------------ pluggable kinds


def test_pluggable_kind_contract_and_failure():
    client, _eng, app = make()
    seen: dict[str, Any] = {}

    def validate(payload: dict) -> int:
        if not isinstance(payload.get("n"), int):
            raise ValueError("n: must be an integer")
        return payload["n"]

    async def run(ctx, n, job):
        seen["resumed"], seen["from"] = job.resumed, job.resume_from
        job.set_total(n)
        for i in range(n):
            if job.cancelled:
                break
            await job.add_items([{"i": i, **({"error": "odd"} if i % 2 else {})}])
        return {"sum": sum(range(n))}

    async def boom(ctx, n, job):
        raise RuntimeError("boom")

    with client:
        kinds = app.state.ctx.extra["job_kinds"]
        kinds["evaluate"] = {"validate": validate, "run": run}
        kinds["boom"] = {"validate": lambda p: p, "run": boom}
        r = client.post("/v1/jobs", json={"kind": "evaluate", "payload": {"n": "x"}})
        assert r.status_code == 400 and r.json()["detail"] == "payload.n: must be an integer"
        jid = client.post("/v1/jobs", json={"kind": "evaluate", "payload": {"n": 5}}).json()["id"]
        job = wait(client, jid)
        assert job["status"] == "succeeded" and job["result"] == {"sum": 10}
        assert job["progress"] == {"done": 5, "total": 5, "failed": 2, "percent": 100.0}
        assert [r["i"] for r in all_rows(client, jid)] == [0, 1, 2, 3, 4] and seen == {
            "resumed": False,
            "from": 0,
        }
        bad = wait(client, client.post("/v1/jobs", json={"kind": "boom", "payload": {}}).json()["id"])
        assert bad["status"] == "failed" and bad["error"] == "internal server error"


def test_builtin_kinds_do_not_clobber_registered_ones():
    eng = SlowEngine()
    app = create_app(Config(api_key=None, state_dir=tempfile.mkdtemp(prefix="clef-jobs-")), eng)
    assert {"classify", "score", "systemone"} <= set(app.state.ctx.extra["job_kinds"])
    mine = {"validate": lambda p: p, "run": None}
    ctx_kinds = app.state.ctx.extra["job_kinds"]
    ctx_kinds.setdefault("classify", mine)
    assert ctx_kinds["classify"] is not mine


# ------------------------------------------------------------------ persistence


def test_restart_resumes_from_last_persisted_row():
    state = tempfile.mkdtemp(prefix="clef-jobs-")
    n = 48
    slow = SlowEngine(delay=0.1)
    client, _, _ = make(slow, state_dir=state, max_microbatch=8)
    with client:
        jid = submit(client, payload={"inputs": [f"t{i}" for i in range(n)], "labels": LABELS}).json()["id"]
        while client.get(f"/v1/jobs/{jid}").json()["progress"]["done"] < 16:
            time.sleep(0.01)
    # graceful shutdown: the job is parked as interrupted, rows persisted so far are kept
    store = open_store(Config(state_dir=state))
    parked = store.get(jid)
    store.close()
    # a clean stop is not a crash: it does not count toward CLEF_JOB_MAX_RESUMES
    assert parked["status"] == "interrupted" and 16 <= parked["done"] < n and parked["resumes"] == 0
    persisted = parked["done"]

    fast = SlowEngine()
    client2, _, _ = make(fast, state_dir=state, max_microbatch=8)
    with client2:
        job = wait(client2, jid)
        assert job["status"] == "succeeded" and job["resumes"] == 0
        assert job["progress"]["done"] == n and job["result"]["items"] == n and job["result"]["ok"] == n
        rows = all_rows(client2, jid)
        assert [r["index"] for r in rows] == list(range(n))  # no gap, no duplicate
        assert sum(fast.sizes) == n - persisted  # only the missing items were re-run


def test_hard_crash_running_job_becomes_interrupted_then_resumes():
    state = tempfile.mkdtemp(prefix="clef-jobs-")
    cfg = Config(state_dir=state, api_key=None)
    store = open_store(cfg)
    row = store.create("classify", None, {"inputs": ["a", "b", "c", "d"], "labels": LABELS}, None, None, 10)
    assert store.claim(row["id"])
    store.add_items(row["id"], [{"index": 0, "label": "billing"}, {"index": 1, "label": "billing"}])
    store.close()  # process "dies" with the job running
    client, eng, _ = make(SlowEngine(), state_dir=state)
    with client:
        job = wait(client, row["id"])
        assert job["status"] == "succeeded" and job["resumes"] == 1 and job["progress"]["done"] == 4
        rows = all_rows(client, row["id"])
        assert [r["index"] for r in rows] == [0, 1, 2, 3] and rows[0]["label"] == "billing"
        assert rows[2]["label"] == "technical" and sum(eng.sizes) == 2


def test_non_resumable_kind_restarts_from_scratch():
    state = tempfile.mkdtemp(prefix="clef-jobs-")
    cfg = Config(state_dir=state, api_key=None)
    store = open_store(cfg)
    row = store.create("evaluate", None, {"n": 3}, None, None, 10)
    store.claim(row["id"])
    store.add_items(row["id"], [{"i": 0}])
    store.close()
    calls: list[int] = []

    async def run(ctx, n, job):
        calls.append(job.resume_from)
        for i in range(n):
            await job.add_items([{"i": i}])

    client, _, app = make(state_dir=state)
    app.state.ctx.extra["job_kinds"]["evaluate"] = {"validate": lambda p: p["n"], "run": run}
    with client:
        job = wait(client, row["id"])
        assert job["status"] == "succeeded" and calls == [0]
        assert [r["i"] for r in all_rows(client, row["id"])] == [0, 1, 2]


def test_unknown_kind_on_resume_fails_cleanly():
    state = tempfile.mkdtemp(prefix="clef-jobs-")
    store = open_store(Config(state_dir=state))
    row = store.create("gone", None, {}, None, None, 10)
    store.close()
    client, _, _ = make(state_dir=state)
    with client:
        job = wait(client, row["id"])
        assert job["status"] == "failed" and "unknown job kind" in job["error"]


def test_retention_purges_old_finished_jobs_only():
    state = tempfile.mkdtemp(prefix="clef-jobs-")
    store = open_store(Config(state_dir=state))
    old = store.create("classify", None, {}, None, None, 10)["id"]
    fresh = store.create("classify", None, {}, None, None, 10)["id"]
    queued = store.create("classify", None, {}, None, None, 10)["id"]
    for jid in (old, fresh):
        store.claim(jid)
        store.add_items(jid, [{"x": 1}])
        store.finish(jid, "succeeded")
    sql(store, "UPDATE jobs SET finished_at=:t WHERE id=:id", t=time.time() - 3 * 3600, id=old)
    assert store.purge(time.time() - 2 * 3600) == 1
    assert store.get(old) is None and store.get(fresh) and store.get(queued)
    [(n,)] = sql(store, "SELECT COUNT(*) FROM items WHERE job_id=:id", id=old)
    assert n == 0  # results go with the job
    store.close()

    state2 = tempfile.mkdtemp(prefix="clef-jobs-")
    s2 = open_store(Config(state_dir=state2))
    j = s2.create("classify", None, {}, None, None, 10)["id"]
    s2.claim(j)
    s2.finish(j, "failed", error="x")
    sql(s2, "UPDATE jobs SET finished_at=:t", t=time.time() - 10 * 3600)
    s2.close()
    client, _, _ = make(state_dir=state2, job_ttl_hours=1)
    with client:
        assert client.get(f"/v1/jobs/{j}").status_code == 404  # purged at startup


# ------------------------------------------------------------------ ownership / auth


def test_ownership_with_api_keys():
    from test_keys import add_key

    client, _, app = make(api_keys_raw="root:kr")  # env keys are admins: they see every job (D1)
    a, b = {"X-API-Key": add_key(app, "alice")}, {"X-API-Key": add_key(app, "bob")}
    with client:
        assert client.post("/v1/jobs", json={"kind": "classify", "payload": {}}).status_code == 401
        jid = client.post(
            "/v1/jobs", json={"kind": "classify", "payload": {"inputs": ["a"], "labels": LABELS}}, headers=a
        ).json()["id"]
        wait(client, jid, headers=a)
        assert client.get(f"/v1/jobs/{jid}", headers=b).status_code == 404
        assert client.get(f"/v1/jobs/{jid}/results", headers=b).status_code == 404
        assert client.post(f"/v1/jobs/{jid}/cancel", headers=b).status_code == 404
        assert client.delete(f"/v1/jobs/{jid}", headers=b).status_code == 404
        assert client.get("/v1/jobs", headers=b).json()["total"] == 0
        assert client.get("/v1/jobs", headers=a).json()["total"] == 1
        assert client.get(f"/v1/jobs/{jid}", headers=a).status_code == 200
        admin = {"X-API-Key": "kr"}
        assert client.get(f"/v1/jobs/{jid}", headers=admin).status_code == 200
        assert client.get("/v1/jobs", headers=admin).json()["total"] == 1
        assert client.delete(f"/v1/jobs/{jid}", headers=a).status_code == 200


def test_public_job_hides_secrets_and_payload():
    store = open_store(Config(state_dir=tempfile.mkdtemp(prefix="clef-jobs-")))
    hook = {"url": "https://h.example.com/x", "secret": "s3cret"}
    row = store.create("classify", "alice", {"inputs": ["private"]}, None, hook, 10)
    text = json.dumps(public_job(row))
    assert "s3cret" not in text and "private" not in text and "alice" not in text
    assert json.loads(text)["webhook"]["has_secret"] is True
    store.close()


# ------------------------------------------------------------------ cross-feature: evaluation as a job kind


def test_evaluate_job_kind_runs_through_the_jobs_runner():
    """evaluation.py registers 'evaluate' in ctx.extra['job_kinds']; the runner must pick it up end to end."""
    client, _eng, _ = make()
    rows = [{"input": f"t{i}", "gold": "technical"} for i in range(3)] + [{"input": "inv", "gold": "billing"}]
    with client:
        bad_rows = [{"input": "x", "gold": "nope"}]
        bad = submit(client, kind="evaluate", payload={"labels": LABELS, "rows": bad_rows})
        assert bad.status_code == 400 and "rows[0].gold" in bad.json()["detail"]
        r = submit(client, kind="evaluate", payload={"labels": LABELS, "rows": rows})
        assert r.status_code == 202, r.text
        done = wait(client, r.json()["id"])
        assert done["status"] == "succeeded", done
        assert done["progress"]["done"] == 4 and done["progress"]["total"] == 4
        assert done["result"]["n"] == 4 and done["result"]["accuracy"] == pytest.approx(0.75)
        items = all_rows(client, done["id"])
        assert [x["index"] for x in items] == [0, 1, 2, 3]
        assert [x["correct"] for x in items] == [True, True, True, False]


def test_evaluate_job_isolates_bad_rows_and_validates_classifier_at_submit():
    client, _eng, _ = make()
    rows = [{"input": f"t{i}", "gold": "technical"} for i in range(5)]
    rows[3]["input"] = "TOOLONG"
    with client:
        missing = submit(client, kind="evaluate", payload={"classifier": "nope", "rows": rows})
        assert missing.status_code == 400 and "nope" in missing.json()["detail"]
        client.put("/v1/classifiers/dept", json={"labels": LABELS})
        bad = {"classifier": "dept", "rows": [{"input": "x", "gold": "q"}]}
        bad_gold = submit(client, kind="evaluate", payload=bad)
        assert bad_gold.status_code == 400 and "rows[0].gold" in bad_gold.json()["detail"]
        r = submit(client, kind="evaluate", payload={"classifier": "dept", "rows": rows})
        assert r.status_code == 202, r.text
        done = wait(client, r.json()["id"])
        assert done["status"] == "succeeded", done
        assert done["result"]["n"] == 4 and done["result"]["n_errors"] == 1
        assert done["result"]["classifier"] == "dept"
        items = all_rows(client, done["id"])
        assert [("error" in x) for x in items] == [False, False, False, True, False]


def test_csv_export_neutralises_formulas():
    from clef_server.jobs import _csv_safe

    row = _csv_safe({"input": "=HYPERLINK(1)", "a": "+1", "b": "-x", "c": "@SUM", "ok": "bill", "n": -1.5})
    assert row == {"input": "'=HYPERLINK(1)", "a": "'+1", "b": "'-x", "c": "'@SUM", "ok": "bill", "n": -1.5}


# ------------------------------------------------------------------ review fixes: failures, breaker, warnings


def test_all_items_failing_aborts_the_job_instead_of_succeeding():
    client, eng, _ = make()
    with client:
        jid = submit(client, payload={"inputs": ["TOOLONG"] * 40, "labels": LABELS}).json()["id"]
        job = wait(client, jid)
        assert job["status"] == "failed", job
        assert "first 16 items all failed" in job["error"] and "too long" in job["error"]
        assert job["progress"]["done"] == 16 and job["progress"]["failed"] == 16  # stopped, not 40 bisections
        assert job["warnings"] and "16 of 16" in job["warnings"][0]
        assert len(all_rows(client, jid)) == 16  # the rows that were produced stay readable


def test_small_job_where_every_item_failed_is_failed_not_succeeded():
    client, _, _ = make()
    with client:
        payload = {"inputs": ["TOOLONG", "TOOLONG b"], "labels": LABELS}
        job = wait(client, submit(client, payload=payload).json()["id"])
        assert job["status"] == "failed" and job["error"].startswith("all 2 item(s) failed")
        assert job["result"]["errors"] == 2  # the summary is still there


def test_partial_failure_stays_succeeded_but_carries_a_warning():
    client, _, _ = make()
    with client:
        payload = {"inputs": ["a", "TOOLONG", "b"], "labels": LABELS}
        job = wait(client, submit(client, payload=payload).json()["id"])
        assert job["status"] == "succeeded" and job["progress"]["failed"] == 1
        assert len(job["warnings"]) == 1 and "1 of 3 item(s) failed" in job["warnings"][0]
        clean = wait(client, submit(client).json()["id"])
        assert clean["warnings"] == []


def test_same_error_on_most_items_aborts_after_the_sample():
    client, _, _ = make()
    inputs = ["ok" if i % 20 == 0 else "TOOLONG" for i in range(120)]  # 95 % fail alike
    with client:
        job = wait(client, submit(client, payload={"inputs": inputs, "labels": LABELS}).json()["id"])
        assert job["status"] == "failed" and "with the same error" in job["error"]
        assert 50 <= job["progress"]["done"] < 120


def test_failed_job_is_logged_with_its_id_and_error_text_stays_generic(caplog):
    eng = SlowEngine()
    eng.raises = RuntimeError("secret internal detail")
    client, _, _ = make(eng)
    with client, caplog.at_level("WARNING", logger="clef"):
        jid = submit(client).json()["id"]
        job = wait(client, jid)
    assert job["error"] == "internal server error"
    msgs = [r.getMessage() for r in caplog.records]
    assert any(f"job {jid}" in m and "failed: internal server error" in m for m in msgs)
    assert any(r.exc_info for r in caplog.records)  # the traceback is logged (server side only)


# ------------------------------------------------------------------ review fixes: crash loops, poison pills


def test_recover_gives_up_after_max_resumes():
    store = open_store(Config(state_dir=tempfile.mkdtemp(prefix="clef-jobs-")))
    row = store.create("classify", None, {}, None, None, 10)
    store.claim(row["id"])
    assert store.recover(3) == (1, [], []) and store.get(row["id"])["status"] == "interrupted"
    for _ in range(2):  # crashes 2 and 3 are still resumed
        store.claim(row["id"])
        assert store.recover(3) == (1, [], [])
    assert store.get(row["id"])["resumes"] == 3
    store.claim(row["id"])
    assert store.recover(3) == (0, [row["id"]], [])  # the 4th interruption: give up
    got = store.get(row["id"])
    assert (
        got["status"] == "failed" and got["error"] == "interrupted 4 times, giving up" and got["finished_at"]
    )
    assert store.recover(3) == (0, [], [])  # a finished job is left alone
    unlimited = store.create("classify", None, {}, None, None, 10)["id"]
    for _ in range(6):
        store.claim(unlimited)
        assert store.recover(0) == (1, [], [])
    store.close()


def test_crash_looping_job_is_failed_at_startup_and_logged(caplog):
    state = tempfile.mkdtemp(prefix="clef-jobs-")
    store = open_store(Config(state_dir=state))
    poison = store.create("classify", None, {"inputs": ["a"], "labels": LABELS}, None, None, 10)["id"]
    sql(store, "UPDATE jobs SET resumes=3 WHERE id=:id", id=poison)
    store.claim(poison)
    healthy = store.create("classify", None, {"inputs": ["a"], "labels": LABELS}, None, None, 10)["id"]
    store.close()
    client, _, _ = make(state_dir=state, job_max_resumes=3)
    with client, caplog.at_level("WARNING", logger="clef"):
        job = wait(client, poison)
        assert job["status"] == "failed" and job["error"] == "interrupted 4 times, giving up"
        assert wait(client, healthy)["status"] == "succeeded"  # not starved by the poison job
    assert any(poison in r.getMessage() and r.levelname == "WARNING" for r in caplog.records)


def test_poison_pill_job_is_failed_after_a_few_runner_errors(monkeypatch):
    monkeypatch.setattr("clef_server.jobs.RUNNER_RETRY_PAUSE_S", 0.01)
    client, _, app = make()
    runner = app.state.ctx.extra["jobs"]
    real = runner.store.payload
    calls: list[str] = []
    poison: dict[str, Any] = {}

    def payload(job_id: str) -> str:
        if job_id == poison.get("id"):
            calls.append(job_id)
            raise RuntimeError("disk exploded")  # not a validation error
        return real(job_id)

    monkeypatch.setattr(runner.store, "payload", payload)
    with client:
        poison.update(submit(client).json())
        job = wait(client, poison["id"])
        assert job["status"] == "failed" and "could not be run" in job["error"] and "disk" not in job["error"]
        assert 3 <= len(calls) <= 4  # retried a few times, not forever
        assert wait(client, submit(client).json()["id"])["status"] == "succeeded"  # the queue moves on


def test_cancel_arriving_right_after_the_claim_is_not_lost(monkeypatch):
    eng = SlowEngine(delay=0.05)
    client, _, app = make(eng)
    runner = app.state.ctx.extra["jobs"]
    real = runner.store.claim

    def claim(job_id: str) -> dict | None:
        row = real(job_id)
        if row is not None:
            # exactly what request_cancel does for a running job, landing right after the claim
            runner.cancel_ids.add(row["id"])
            runner.store.flag_cancel(row["id"])
        return row

    monkeypatch.setattr(runner.store, "claim", claim)
    with client:
        payload = {"inputs": [f"t{i}" for i in range(40)], "labels": LABELS}
        job = wait(client, submit(client, payload=payload).json()["id"])
        assert job["status"] == "cancelled" and job["progress"]["done"] < 40


def test_retention_runs_on_its_own_timer_while_a_job_is_running(monkeypatch):
    monkeypatch.setattr("clef_server.jobs.PURGE_INTERVAL_S", 0.05)
    eng = SlowEngine(delay=0.05)
    client, _, app = make(eng)
    runner = app.state.ctx.extra["jobs"]
    calls: list[float] = []
    real = runner._purge
    monkeypatch.setattr(runner, "_purge", lambda: (calls.append(time.monotonic()), real())[1])
    with client:
        jid = submit(client, payload={"inputs": [f"t{i}" for i in range(80)], "labels": LABELS}).json()["id"]
        wait(client, jid, until=("running",))
        before = len(calls)
        assert wait(client, jid)["status"] == "succeeded"
        assert len(calls) - before >= 2, calls  # purged repeatedly while the runner was busy


def test_purge_keeps_jobs_whose_terminal_webhook_is_still_pending():
    store = open_store(Config(state_dir=tempfile.mkdtemp(prefix="clef-jobs-")))
    hook = {"url": "https://hooks.example.com/h"}
    ids = {}
    for state in ("pending", "failed", "delivered"):
        jid = store.create("classify", None, {}, None, hook, 10)["id"]
        store.claim(jid)
        store.finish(jid, "succeeded")
        store.set_delivery(jid, "job.succeeded", {"id": "d", "status": state, "attempts": 1})
        ids[state] = jid
    plain = store.create("classify", None, {}, None, None, 10)["id"]
    store.claim(plain)
    store.finish(plain, "succeeded")
    assert store.purge(time.time() + 10) == 3
    assert store.get(ids["pending"]) and not store.get(ids["failed"]) and not store.get(plain)
    store.close()


# ------------------------------------------------------------------ review fixes: CSV


def test_csv_neutralises_strings_only_never_numbers():
    from clef_server.jobs import flatten

    row = {
        "index": 3, "confidence": -0.5, "n": -2, "label": "=cmd()", "labels": ["@a", "b"], "ok": True,
        "x": None, "scores": {"a": -0.25, "b": "+SUM"}, "neg_list": [-1, 2], "empty": "",
    }  # fmt: skip
    flat = flatten(row, csv_safe=True)
    assert flat["confidence"] == "-0.5" and flat["n"] == "-2" and flat["scores.a"] == "-0.25"
    assert flat["neg_list"] == "-1|2"
    assert flat["label"] == "'=cmd()" and flat["scores.b"] == "'+SUM" and flat["labels"] == "'@a|b"
    assert flat["ok"] == "True" and flat["x"] == "" and flat["empty"] == ""
    assert flatten(row)["label"] == "=cmd()"  # plain flatten (column discovery) leaves text alone


def test_csv_export_of_a_job_with_negative_numbers_is_not_quoted(monkeypatch):
    client, _, app = make()
    store = app.state.ctx.extra["jobs"].store
    hold_runner(monkeypatch, store)
    with client:
        jid = store.create("custom", None, {}, None, None, 10)["id"]
        store.claim(jid)
        store.add_items(jid, [{"index": 0, "score": -0.5, "note": "-bad"}])
        store.finish(jid, "succeeded")
        table = list(csv.DictReader(io.StringIO(client.get(f"/v1/jobs/{jid}/results?format=csv").text)))
        assert table[0]["score"] == "-0.5" and table[0]["note"] == "'-bad"


@pytest.mark.asyncio
async def test_csv_export_is_capped_at_the_snapshot_so_columns_match_rows():
    from clef_server.jobs import _csv

    store = open_store(Config(state_dir=tempfile.mkdtemp(prefix="clef-jobs-")))
    jid = store.create("custom", None, {}, None, None, 10)["id"]
    store.claim(jid)
    store.add_items(jid, [{"index": 0, "a": 1}, {"index": 1, "a": 2}])
    stream = _csv(store, jid, 0, 2)  # 2 = the job's `done` when the request arrived
    header = await anext(stream)
    store.add_items(jid, [{"index": 2, "a": 3, "late": "x"}])  # a row lands between the two passes
    rest = [chunk async for chunk in stream]
    assert (header + "".join(rest)).splitlines() == ["index,a", "0,1", "1,2"]
    store.close()


def test_running_job_exports_stop_at_the_rows_present_when_requested(monkeypatch):
    client, _, app = make()
    store = app.state.ctx.extra["jobs"].store
    hold_runner(monkeypatch, store)
    with client:
        jid = store.create("custom", None, {}, None, None, 10)["id"]
        store.claim(jid)
        store.add_items(jid, [{"index": 0}, {"index": 1}, {"index": 2}])
        r = client.get(f"/v1/jobs/{jid}/results", params={"format": "csv", "limit": 2})
        assert [row["index"] for row in csv.DictReader(io.StringIO(r.text))] == ["0", "1"]
        r = client.get(f"/v1/jobs/{jid}/results", params={"format": "ndjson", "offset": 1})
        assert len(r.text.splitlines()) == 2


# ------------------------------------------------------------------ review fixes: snapshot, idempotency


def test_saved_classifier_is_snapshotted_at_submit():
    eng = SlowEngine(delay=0.2)
    client, _, app = make(eng)
    store = app.state.ctx.extra["jobs"].store
    with client:
        client.put(
            "/v1/classifiers/triage", json={"kind": "classify", "labels": LABELS, "instructions": "v1"}
        )
        blocker = submit(client, payload={"inputs": [f"t{i}" for i in range(16)], "labels": LABELS})
        jid = submit(client, payload={"inputs": ["a", "b"], "classifier": "triage"}).json()["id"]
        client.put(
            "/v1/classifiers/triage", json={"kind": "classify", "labels": ["x", "y", "z"]}
        )  # edited later
        stored = json.loads(store.payload(jid))
        assert "classifier" not in stored and stored["labels"] == LABELS and stored["instructions"] == "v1"
        assert stored["snapshot_of"] == "triage" and stored["multi_label"] is False
        wait(client, blocker.json()["id"])
        assert wait(client, jid)["status"] == "succeeded"
        assert set(all_rows(client, jid)[0]["scores"]) == set(LABELS)  # v1 labels, not x / y / z
        client.put("/v1/classifiers/lvl", json={"kind": "score", "levels": ["lo", "hi"]})
        sj = submit(client, "score", {"inputs": ["a"], "classifier": "lvl"}).json()["id"]
        stored = json.loads(store.payload(sj))
        assert (
            stored["levels"] == ["lo", "hi"] and "classifier" not in stored and stored["snapshot_of"] == "lvl"
        )
        assert wait(client, sj)["result"]["by_level"] == {"hi": 1}
        client.delete("/v1/classifiers/lvl")  # the queued / finished job no longer depends on it


def test_idempotency_key_returns_the_same_job():
    client, _, _ = make(api_keys_raw="alice:ka,bob:kb")
    a, b = {"X-API-Key": "ka"}, {"X-API-Key": "kb"}
    with client:
        h = {**a, "Idempotency-Key": "batch-2026-10-04"}
        first = submit(client, headers=h)
        again = submit(client, headers=h)
        assert first.status_code == 202 and again.status_code == 202
        assert again.json()["id"] == first.json()["id"] and again.headers["Idempotent-Replay"] == "true"
        assert "Idempotent-Replay" not in first.headers
        assert again.headers["Location"] == f"/v1/jobs/{first.json()['id']}"
        assert client.get("/v1/jobs", headers=a).json()["total"] == 1
        other = submit(client, headers={**b, "Idempotency-Key": "batch-2026-10-04"})  # keys are per owner
        assert other.json()["id"] != first.json()["id"]
        assert submit(client, headers={**a, "Idempotency-Key": "other"}).json()["id"] != first.json()["id"]
        assert submit(client, headers={**a, "Idempotency-Key": "x" * 129}).status_code == 400
        wait(client, first.json()["id"], headers=a)
        assert submit(client, headers=h).json()["id"] == first.json()["id"]  # also after it finished


def test_old_database_without_idem_key_is_migrated(sqlite_only):
    cfg = Config(state_dir=tempfile.mkdtemp(prefix="clef-jobs-"))
    path = jobs_db(cfg)  # a v3.1 jobs.db: renamed to clef.db, stamped 0001, upgraded
    db = sqlite3.connect(str(path))
    db.executescript(
        "CREATE TABLE jobs (seq INTEGER PRIMARY KEY AUTOINCREMENT, id TEXT NOT NULL UNIQUE,"
        " kind TEXT NOT NULL,"
        " status TEXT NOT NULL, owner TEXT, payload TEXT NOT NULL, metadata TEXT, webhook TEXT,"
        " deliveries TEXT NOT NULL DEFAULT '{}', total INTEGER, done INTEGER NOT NULL DEFAULT 0,"
        " failed INTEGER NOT NULL DEFAULT 0, created_at REAL NOT NULL, started_at REAL, finished_at REAL,"
        " run_started_at REAL, run_base_done INTEGER NOT NULL DEFAULT 0, resumes INTEGER NOT NULL DEFAULT 0,"
        " cancel_requested INTEGER NOT NULL DEFAULT 0, error TEXT, result TEXT);"
        "INSERT INTO jobs (id, kind, status, payload, created_at)"
        " VALUES ('job_old', 'classify', 'queued', '{}', 1);"
    )
    db.commit()
    db.close()
    store = JobStore(open_database(cfg))
    assert not path.exists() and (path.parent / "clef.db").exists()
    assert store.get("job_old")["status"] == "queued"
    assert store.create("classify", None, {}, None, None, 10, "k")["id"] != "job_old"
    assert store.create("classify", None, {}, None, None, 10, "k").get("replayed")
    store.close()


def test_evaluate_job_snapshots_the_saved_classifier_at_submit():
    client, _eng, _ = make()
    with client:
        assert client.put("/v1/classifiers/tri", json={"kind": "classify", "labels": LABELS}).status_code in (
            200,
            201,
        )
        rows = [{"input": "a", "gold": "technical"}, {"input": "b", "gold": "billing"}]
        r = submit(client, kind="evaluate", payload={"classifier": "tri", "rows": rows})
        assert r.status_code == 202, r.text
        assert client.delete("/v1/classifiers/tri").status_code == 200  # gone before/while the job runs
        done = wait(client, r.json()["id"])
        assert done["status"] == "succeeded", done
        assert done["result"]["n"] == 2 and done["result"]["classifier"] == "tri"


# ------------------------------------------------------------------ crash safety: cancel, events, clean stops

HOOK = {"url": "https://hooks.example.com/h"}


def _running(store: JobStore, cancel: bool = False, hook: dict | None = HOOK, n: int = 2) -> str:
    payload = {"inputs": [f"t{i}" for i in range(n)], "labels": LABELS}
    jid = store.create("classify", None, payload, None, hook, 10)["id"]
    assert store.claim(jid)
    if cancel:
        store.flag_cancel(jid)
    return jid


def test_recover_cancels_a_job_whose_cancel_was_requested_before_the_crash():
    store = open_store(Config(state_dir=tempfile.mkdtemp(prefix="clef-jobs-")))
    doomed, plain = _running(store, cancel=True), _running(store)
    assert store.recover(3) == (1, [], [doomed])
    got = store.get(doomed)
    assert got["status"] == "cancelled" and got["finished_at"] and got["resumes"] == 0 and got["total"] == 0
    rec = got["deliveries"]["job.cancelled"]
    assert rec["status"] == "pending" and rec["attempts"] == 0 and rec["id"]
    assert store.get(plain)["status"] == "interrupted" and store.get(plain)["resumes"] == 1
    poison = _running(store, cancel=True)  # the user's cancel wins over giving up
    sql(store, "UPDATE jobs SET resumes=3 WHERE id=:id", id=poison)
    assert store.recover(3) == (0, [], [poison]) and store.get(poison)["status"] == "cancelled"
    given_up = _running(store)
    sql(store, "UPDATE jobs SET resumes=3 WHERE id=:id", id=given_up)
    assert store.recover(3) == (0, [given_up], [])
    assert store.get(given_up)["deliveries"]["job.failed"]["status"] == "pending"  # written by recover itself
    store.close()


def test_cancel_requested_before_a_crash_ends_cancelled_with_its_webhook_after_restart():
    from test_webhooks import Recorder, delivery, hook_cfg, with_receiver

    state = tempfile.mkdtemp(prefix="clef-jobs-")
    store = open_store(Config(state_dir=state))
    jid = _running(store, cancel=True, n=4)
    store.add_items(jid, [{"index": 0, "label": "billing"}])
    store.close()  # the process "dies" right after the cancel was flagged
    eng = SlowEngine()
    client, _, app = make(eng, state_dir=state, **hook_cfg())
    rec = Recorder(200)

    async def swap() -> None:
        with_receiver(app, rec)

    app.state.ctx.on_startup.insert(0, swap)
    with client:
        d = delivery(client, jid, "job.cancelled")
        job = client.get(f"/v1/jobs/{jid}").json()
        assert job["status"] == "cancelled" and job["progress"]["done"] == 1 and job["resumes"] == 0
        assert d["status"] == "delivered"
        assert [r.headers["X-Clef-Event"] for r in rec.requests] == ["job.cancelled"]
        assert sum(eng.sizes) == 0  # not resumed


def test_terminal_status_and_pending_event_are_one_transaction(sqlite_only):
    _, _, app = make()
    runner = app.state.ctx.extra["jobs"]
    store = runner.store
    running = _running(store)
    queued = store.create("classify", None, {}, None, HOOK, 10)["id"]
    crashed = _running(store, cancel=True)
    sql(
        store,
        "CREATE TRIGGER boom BEFORE UPDATE OF deliveries ON jobs BEGIN SELECT RAISE(ABORT, 'boom'); END",
    )
    with pytest.raises(DBAPIError):
        runner._finish_sync(running, "succeeded", None, None)
    with pytest.raises(DBAPIError):
        runner._cancel_queued_sync(queued)
    with pytest.raises(DBAPIError):
        store.recover(3)
    with pytest.raises(DBAPIError):
        store.park(crashed)
    # nothing half-written: no status changed without its event
    assert [store.get(j)["status"] for j in (running, queued, crashed)] == ["running", "queued", "running"]
    sql(store, "DROP TRIGGER boom")
    assert runner._finish_sync(running, "succeeded", None, None) == (True, "job.succeeded")
    assert runner._cancel_queued_sync(queued) == (True, "job.cancelled")
    assert store.get(running)["deliveries"]["job.succeeded"]["status"] == "pending"
    assert store.get(queued)["deliveries"]["job.cancelled"]["status"] == "pending"
    assert runner._finish_sync(running, "failed", None, None) == (False, None)  # already finished
    store.close()


def test_park_is_a_clean_stop_not_a_resume_and_honours_a_pending_cancel():
    store = open_store(Config(state_dir=tempfile.mkdtemp(prefix="clef-jobs-")))
    plain, cancel = _running(store), _running(store, cancel=True)
    queued = store.create("classify", None, {}, None, HOOK, 10)["id"]
    assert store.park(plain) is None
    assert store.get(plain)["status"] == "interrupted" and store.get(plain)["resumes"] == 0
    assert store.park(cancel) == "job.cancelled"
    got = store.get(cancel)
    assert got["status"] == "cancelled" and got["deliveries"]["job.cancelled"]["status"] == "pending"
    other = _running(store, hook=None)
    assert (
        store.park(other, cancel=True) is None and store.get(other)["status"] == "cancelled"
    )  # in-memory flag
    assert store.park(queued) is None and store.get(queued)["status"] == "queued"  # only running jobs
    store.close()


def test_clean_stop_parks_a_job_left_running_outside_the_kind(monkeypatch):
    client, _, app = make()
    store = app.state.ctx.extra["jobs"].store
    state = app.state.ctx.cfg.state_dir
    hold_runner(monkeypatch, store)
    with client:
        jid = _running(store, hook=None)  # e.g. stopped between claim_next and the kind's first await
    reopened = open_store(Config(state_dir=state))
    got = reopened.get(jid)
    reopened.close()
    assert got["status"] == "interrupted" and got["resumes"] == 0


def test_clean_restarts_do_not_count_toward_max_resumes():
    state = tempfile.mkdtemp(prefix="clef-jobs-")
    n = 48
    store = open_store(Config(state_dir=state))
    payload = {"inputs": [f"t{i}" for i in range(n)], "labels": LABELS}
    jid = store.create("classify", None, payload, None, None, 10)["id"]
    sql(store, "UPDATE jobs SET resumes=2 WHERE id=:id", id=jid)  # two real crashes already
    store.close()
    client, _, _ = make(SlowEngine(delay=0.05), state_dir=state, max_microbatch=8, job_max_resumes=3)
    with client:  # a clean stop mid-job
        while client.get(f"/v1/jobs/{jid}").json()["progress"]["done"] < 8:
            time.sleep(0.01)
    store = open_store(Config(state_dir=state))
    assert store.get(jid)["status"] == "interrupted" and store.get(jid)["resumes"] == 2
    sql(store, "UPDATE jobs SET status='running' WHERE id=:id", id=jid)  # then a third real crash
    store.close()
    client2, _, _ = make(SlowEngine(), state_dir=state, max_microbatch=8, job_max_resumes=3)
    with client2:
        job = wait(client2, jid)
        assert job["status"] == "succeeded" and job["resumes"] == 3, job
        assert [r["index"] for r in all_rows(client2, jid)] == list(range(n))


def test_runner_error_is_not_counted_as_a_crash(monkeypatch):
    monkeypatch.setattr("clef_server.jobs.RUNNER_RETRY_PAUSE_S", 0.01)
    client, _, app = make()
    runner = app.state.ctx.extra["jobs"]
    real = runner.store.claim
    boom = [True]

    def claim(job_id: str) -> dict | None:
        row = real(job_id)
        if row is not None and (boom.pop() if boom else False):
            raise RuntimeError("transient")  # the runner fails with the job left 'running'
        return row

    monkeypatch.setattr(runner.store, "claim", claim)
    with client:
        job = wait(client, submit(client).json()["id"])
        assert job["status"] == "succeeded" and job["resumes"] == 0, job


def test_flag_cancel_on_a_job_parked_meanwhile_cancels_it_at_once():
    store = open_store(Config(state_dir=tempfile.mkdtemp(prefix="clef-jobs-")))
    running = _running(store)
    assert store.flag_cancel(running) == ("running", None)
    assert store.get(running)["cancel_requested"] is True and store.get(running)["status"] == "running"
    parked = _running(store)
    assert store.park(parked) is None  # a runner error parked it before the cancel landed
    assert store.flag_cancel(parked) == ("cancelled", "job.cancelled")
    got = store.get(parked)
    assert got["status"] == "cancelled" and got["finished_at"]
    assert got["deliveries"]["job.cancelled"]["status"] == "pending"
    store.finish(running, "succeeded")
    assert store.claim_next() is None  # the parked job is never picked up again
    assert store.flag_cancel(parked) == (None, None)  # finished: nothing to do
    store.close()


def test_cancel_is_not_lost_when_park_wins_the_race(monkeypatch):
    client, eng, app = make()
    runner = app.state.ctx.extra["jobs"]
    store = runner.store
    sent: list[tuple[str, str]] = []
    monkeypatch.setattr(runner.dispatcher, "enqueue", lambda job_id, event: sent.append((job_id, event)))
    hold = hold_runner(monkeypatch, store)
    with client:
        payload = {"inputs": ["a", "b"], "labels": LABELS}
        jid = store.create("classify", None, payload, None, HOOK, 10)["id"]
        assert store.claim(jid)  # 'running' rows are not picked up by the runner loop
        hold.clear()
        real = store.cancel_queued

        def cancel_queued(job_id: str) -> tuple[bool, str | None]:
            out = real(job_id)  # False: the job is running
            store.park(job_id)  # ... and the runner-error path parks it before the cancel is flagged
            return out

        monkeypatch.setattr(store, "cancel_queued", cancel_queued)
        r = client.post(f"/v1/jobs/{jid}/cancel")
        assert r.status_code == 200 and r.json()["status"] == "cancelled"
        assert sent == [(jid, "job.cancelled")]
        runner.poke()
        time.sleep(0.1)
        got = client.get(f"/v1/jobs/{jid}").json()
        assert got["status"] == "cancelled" and got["resumes"] == 0
        assert sum(eng.sizes) == 0  # never run
