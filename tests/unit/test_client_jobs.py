"""Python client: async job methods against a scripted fake server (sync + async)."""

import json

import httpx
import pytest

from clef_client import AsyncClefClient, ClefClient, ClefError, Job, JobItem, JobList, JobNotFinished
from clef_client.types import Classification, ScoreResult, parse_job


def job_body(status="queued", done=0, total=None, **extra):
    return {
        "id": "job_1",
        "kind": "classify",
        "status": status,
        "progress": {"done": done, "total": total, "failed": 0, "percent": None},
        "eta_s": None,
        "created_at": "2026-01-01T00:00:00Z",
        "started_at": None,
        "finished_at": None,
        "error": None,
        "result": None,
        "metadata": None,
        "webhook": None,
        **extra,
    }


ROWS = [
    {"index": 0, "label": "billing", "confidence": 0.9, "scores": {"billing": 0.9, "technical": 0.1}},
    {"index": 1, "error": "input is too long"},
    {"index": 2, "labels": ["a", "b"], "scores": {"a": 0.8, "b": 0.6}, "threshold": 0.5},
]


class Server:
    def __init__(self, polls=None):
        self.seen: list[tuple[str, str, dict | None, dict]] = []
        self.polls = list(polls or [])

    def __call__(self, req: httpx.Request) -> httpx.Response:
        body = json.loads(req.content) if req.content else None
        params = dict(req.url.params)
        self.seen.append((req.method, req.url.path, body, params))
        p = req.url.path
        if (req.method, p) == ("POST", "/v1/jobs"):
            return httpx.Response(202, json=job_body(metadata=body.get("metadata")))
        if (req.method, p) == ("GET", "/v1/jobs"):
            return httpx.Response(200, json={"jobs": [job_body(), job_body(status="succeeded")], "total": 2})
        if (req.method, p) == ("GET", "/v1/jobs/job_1"):
            return httpx.Response(200, json=self.polls.pop(0) if len(self.polls) > 1 else self.polls[0])
        if p == "/v1/jobs/job_1/cancel":
            return httpx.Response(200, json=job_body(status="cancelled"))
        if (req.method, p) == ("DELETE", "/v1/jobs/job_1"):
            return httpx.Response(200, json={"deleted": "job_1"})
        if p == "/v1/jobs/job_1/results":
            if params.get("format"):
                return httpx.Response(200, content=b'{"index":0}\n{"index":1}\n')
            off, lim = int(params["offset"]), int(params["limit"])
            items = ROWS[off : off + lim]
            nxt = off + len(items)
            return httpx.Response(
                200,
                json={
                    "id": "job_1",
                    "kind": "classify",
                    "items": items,
                    "next_offset": nxt if nxt < 3 else None,
                },
            )
        if p == "/v1/jobs/missing":
            return httpx.Response(404, json={"detail": "job 'missing' not found", "request_id": "r"})
        return httpx.Response(404, json={"detail": "nope", "request_id": "x"})


def sync(server):
    return ClefClient(transport=httpx.MockTransport(server), backoff=0)


def test_submit_and_classify_job_bodies():
    s = Server()
    c = sync(s)
    job = c.submit_job(
        "classify", {"inputs": ["a"], "labels": ["x", "y"]}, webhook="https://h.example/x", metadata={"m": 1}
    )
    assert isinstance(job, Job) and job.id == "job_1" and job.status == "queued" and not job.finished
    assert s.seen[0][2] == {
        "kind": "classify",
        "payload": {"inputs": ["a"], "labels": ["x", "y"]},
        "webhook": {"url": "https://h.example/x"},
        "metadata": {"m": 1},
    }
    c.classify_job(
        ["a", "b"],
        ["x", "y"],
        instructions="why",
        multi_label=True,
        threshold=0.4,
        webhook={"url": "u", "secret": "s"},
    )
    assert s.seen[1][2]["payload"] == {
        "labels": ["x", "y"],
        "instructions": "why",
        "multi_label": True,
        "threshold": 0.4,
        "model": "clef-flash",
        "inputs": ["a", "b"],
    }
    assert s.seen[1][2]["webhook"] == {"url": "u", "secret": "s"}
    c.classify_job(["a"], classifier="triage")
    assert s.seen[2][2]["payload"] == {"classifier": "triage", "model": "clef-flash", "inputs": ["a"]}


def test_job_listing_cancel_delete():
    s = Server(polls=[job_body(status="running", done=5, total=10, eta_s=2.5)])
    c = sync(s)
    job = c.job("job_1")
    assert (job.status, job.done, job.total, job.eta_s) == ("running", 5, 10, 2.5) and job.raw[
        "id"
    ] == "job_1"
    assert [j.status for j in c.jobs(status="running", limit=10, offset=20)] == ["queued", "succeeded"]
    assert s.seen[-1][3] == {"status": "running", "limit": "10", "offset": "20"}
    assert c.cancel_job("job_1").status == "cancelled"
    assert c.delete_job("job_1") == {"deleted": "job_1"}
    with pytest.raises(ClefError) as err:
        c.job("missing")
    assert err.value.status == 404


def test_wait_job_polls_until_finished(monkeypatch):
    monkeypatch.setattr("clef_client._sync.time.sleep", lambda s: None)
    done = job_body(status="succeeded", done=3, total=3, result={"items": 3})
    s = Server(polls=[job_body(status="queued"), job_body(status="running", done=1, total=3), done])
    seen: list[int] = []
    job = sync(s).wait_job("job_1", poll=0, on_progress=lambda j: seen.append(j.done))
    assert job.ok and job.finished and job.result == {"items": 3} and seen == [0, 1, 3]


def test_wait_job_returns_failed_job_and_times_out():
    failed = job_body(status="failed", error="engine gone")
    job = sync(Server(polls=[failed])).wait_job("job_1", poll=0)
    assert job.finished and not job.ok and job.error == "engine gone"
    with pytest.raises(TimeoutError, match="still running"):
        sync(Server(polls=[job_body(status="running", done=1, total=9)])).wait_job(
            "job_1", timeout=0.05, poll=0.1
        )


def test_job_results_pages_and_types():
    s = Server()
    items = list(sync(s).job_results("job_1", page_size=2))
    assert [i.index for i in items] == [0, 1, 2]
    assert isinstance(items[0], JobItem) and isinstance(items[0].result, Classification)
    assert items[0].result.label == "billing" and items[0].ok
    assert items[1].error == "input is too long" and items[1].result is None and not items[1].ok
    assert items[2].result.multi_label and items[2].result.labels == ["a", "b"]
    reqs = [x for x in s.seen if x[1].endswith("/results")]
    assert [r[3] for r in reqs] == [{"offset": "0", "limit": "2"}, {"offset": "2", "limit": "2"}]


def test_job_results_score_kind():
    def handler(req):
        row = {
            "index": 0,
            "score": 1.0,
            "level": "high",
            "level_index": 1,
            "distribution": {"low": 0.1, "high": 0.9},
        }
        return httpx.Response(200, json={"kind": "score", "items": [row], "next_offset": None})

    (item,) = ClefClient(transport=httpx.MockTransport(handler)).job_results("job_1")
    assert isinstance(item.result, ScoreResult) and item.result.level == "high"


def test_save_job_results(tmp_path):
    out = sync(Server()).save_job_results("job_1", tmp_path / "r.ndjson")
    assert out.read_bytes() == b'{"index":0}\n{"index":1}\n'
    with pytest.raises(ClefError):
        sync(Server()).save_job_results("missing", tmp_path / "x")


def test_job_id_is_url_quoted():
    raw: list[bytes] = []

    def handler(req: httpx.Request) -> httpx.Response:
        raw.append(req.url.raw_path)
        return httpx.Response(404, json={"detail": "x"})

    with pytest.raises(ClefError):
        ClefClient(transport=httpx.MockTransport(handler)).job("a/../b")
    assert raw == [b"/v1/jobs/a%2F..%2Fb"]


# ------------------------------------------------------------------ async


@pytest.mark.asyncio
async def test_async_client_job_flow(monkeypatch):
    async def nosleep(_):
        return None

    monkeypatch.setattr("clef_client._async.asyncio.sleep", nosleep)
    done = job_body(status="succeeded", done=3, total=3)
    s = Server(polls=[job_body(status="running", done=1, total=3), done])
    async with AsyncClefClient(transport=httpx.MockTransport(s), backoff=0) as c:
        job = await c.classify_job(["a", "b", "c"], ["x", "y"], webhook="https://h.example/x")
        assert job.id == "job_1" and s.seen[0][2]["webhook"] == {"url": "https://h.example/x"}
        assert (await c.wait_job("job_1", poll=0)).ok
        assert [j.status for j in await c.jobs()] == ["queued", "succeeded"]
        assert [i.index async for i in c.job_results("job_1", page_size=2)] == [0, 1, 2]
        assert (await c.cancel_job("job_1")).status == "cancelled"
        assert await c.delete_job("job_1") == {"deleted": "job_1"}
        with pytest.raises(ClefError):
            await c.job("missing")
        with pytest.raises(TimeoutError):
            await AsyncClefClient(
                transport=httpx.MockTransport(Server(polls=[job_body(status="running")]))
            ).wait_job("job_1", timeout=0.0, poll=0.1)


@pytest.mark.asyncio
async def test_async_save_job_results(tmp_path):
    async with AsyncClefClient(transport=httpx.MockTransport(Server())) as c:
        out = await c.save_job_results("job_1", tmp_path / "r.csv", format="csv")
        assert out.read_bytes().startswith(b'{"index":0}')
        with pytest.raises(ClefError):
            await c.save_job_results("missing", tmp_path / "x")


# ------------------------------------------------------------------ review fixes: partial results, totals


class Scripted:
    """Answers GET results / POST jobs from a script; records request headers."""

    def __init__(self, status="succeeded", pages=None, body=b'{"index":0}\n'):
        self.status, self.pages, self.body = status, list(pages or []), body
        self.requests: list[httpx.Request] = []

    def __call__(self, req: httpx.Request) -> httpx.Response:
        self.requests.append(req)
        p = req.url.path
        if p == "/v1/jobs" and req.method == "POST":
            return httpx.Response(202, json=job_body())
        if p.endswith("/webhook/redeliver"):
            return httpx.Response(200, json=job_body(status="succeeded"))
        if p.endswith("/results") and req.url.params.get("format"):
            return httpx.Response(200, content=self.body, headers={"X-Clef-Job-Status": self.status})
        if p.endswith("/results"):
            return httpx.Response(200, json=self.pages.pop(0) if len(self.pages) > 1 else self.pages[0])
        return httpx.Response(404, json={"detail": "nope"})


def test_jobs_page_carries_total_and_stays_a_list():
    body = {"jobs": [job_body(), job_body(status="succeeded")], "total": 7, "limit": 2, "offset": 4}
    c = ClefClient(transport=httpx.MockTransport(lambda r: httpx.Response(200, json=body)))
    page = c.jobs(limit=2, offset=4)
    assert isinstance(page, list) and isinstance(page, JobList) and len(page) == 2
    assert [j.status for j in page] == ["queued", "succeeded"]
    assert (page.total, page.limit, page.offset, page.has_more) == (7, 2, 4, True)
    assert page == list(page) and page[0].id == "job_1"


def test_job_exposes_warnings_and_has_errors_while_ok_keeps_its_meaning():
    body = job_body(status="succeeded", done=3, total=3, warnings=["1 of 3 item(s) failed"])
    body["progress"]["failed"] = 1
    job = parse_job(body)
    assert job.ok and job.has_errors and job.failed == 1 and job.warnings == ["1 of 3 item(s) failed"]
    clean = parse_job(job_body(status="succeeded"))
    assert clean.ok and not clean.has_errors and clean.warnings == []
    assert not parse_job(job_body(status="failed")).ok


def test_save_job_results_refuses_a_running_job_and_writes_nothing(tmp_path):
    c = ClefClient(transport=httpx.MockTransport(Scripted(status="running")))
    dest = tmp_path / "r.ndjson"
    with pytest.raises(JobNotFinished) as err:
        c.save_job_results("job_1", dest)
    assert (
        isinstance(err.value, ClefError) and err.value.status == 409 and "still running" in err.value.detail
    )
    assert not dest.exists() and not (tmp_path / "r.ndjson.part").exists()
    out = c.save_job_results("job_1", dest, require_finished=False)  # explicit partial export
    assert out.read_bytes() == b'{"index":0}\n'
    for terminal in ("succeeded", "failed", "cancelled"):
        ok = ClefClient(transport=httpx.MockTransport(Scripted(status=terminal)))
        assert ok.save_job_results("job_1", tmp_path / f"{terminal}.ndjson").exists()


def test_save_job_results_is_atomic(tmp_path):
    class Boom(httpx.SyncByteStream):
        def __iter__(self):
            yield b"partial"
            raise httpx.ReadError("connection reset")

    def handler(req):
        return httpx.Response(200, stream=Boom(), headers={"X-Clef-Job-Status": "succeeded"})

    dest = tmp_path / "r.csv"
    dest.write_bytes(b"previous good export")
    with pytest.raises(httpx.ReadError):
        ClefClient(transport=httpx.MockTransport(handler)).save_job_results("job_1", dest, format="csv")
    assert dest.read_bytes() == b"previous good export"  # not truncated
    assert not (tmp_path / "r.csv.part").exists()


def test_job_results_wait_keeps_polling_until_the_job_is_finished(monkeypatch):
    naps: list[float] = []
    monkeypatch.setattr("clef_client._sync.time.sleep", naps.append)
    pages = [
        {"kind": "classify", "items": [{"index": 0, "label": "a", "scores": {"a": 1.0}}], "next_offset": 1},
        {"kind": "classify", "items": [], "next_offset": 1},  # running, nothing new yet
        {"kind": "classify", "items": [{"index": 1, "error": "x"}], "next_offset": 2},
        {"kind": "classify", "items": [], "next_offset": None},  # finished
    ]
    c = ClefClient(transport=httpx.MockTransport(Scripted(pages=pages)))
    assert [i.index for i in c.job_results("job_1", wait=True, poll=0.5)] == [0, 1]
    assert naps == [0.5]
    partial = ClefClient(transport=httpx.MockTransport(Scripted(pages=pages[:1] + [pages[1]])))
    assert [i.index for i in partial.job_results("job_1")] == [0]  # default: what is stored now


def test_idempotency_key_header_and_redeliver():
    s = Scripted()
    c = ClefClient(transport=httpx.MockTransport(s), backoff=0)
    c.submit_job("classify", {"inputs": ["a"], "labels": ["x", "y"]}, idempotency_key="batch-1")
    c.classify_job(["a"], ["x", "y"], idempotency_key="batch-2")
    c.classify_job(["a"], ["x", "y"])
    assert [r.headers.get("Idempotency-Key") for r in s.requests] == ["batch-1", "batch-2", None]
    assert c.redeliver_webhook("job_1").status == "succeeded"
    c.redeliver_webhook("job_1", "job.failed")
    assert s.requests[-2].method == "POST" and s.requests[-2].url.path == "/v1/jobs/job_1/webhook/redeliver"
    assert s.requests[-2].url.query == b"" and s.requests[-1].url.params["event"] == "job.failed"


@pytest.mark.asyncio
async def test_async_partial_results_total_and_redeliver(tmp_path, monkeypatch):
    async def nosleep(_):
        return None

    monkeypatch.setattr("clef_client._async.asyncio.sleep", nosleep)
    body = {"jobs": [job_body()], "total": 3, "limit": 1, "offset": 0}
    pages = [
        {"kind": "classify", "items": [], "next_offset": 0},
        {"kind": "classify", "items": [{"index": 0, "error": "x"}], "next_offset": None},
    ]

    scripted = Scripted(status="running", pages=pages)

    def handler(req):
        if req.url.path == "/v1/jobs" and req.method == "GET":
            return httpx.Response(200, json=body)
        return scripted(req)

    async with AsyncClefClient(transport=httpx.MockTransport(handler), backoff=0) as c:
        page = await c.jobs(limit=1)
        assert page.total == 3 and page.has_more and isinstance(page, list)
        assert [i.index async for i in c.job_results("job_1", wait=True)] == [0]
        with pytest.raises(JobNotFinished):
            await c.save_job_results("job_1", tmp_path / "x.csv", format="csv")
        assert not (tmp_path / "x.csv").exists() and not (tmp_path / "x.csv.part").exists()
        assert (await c.save_job_results("job_1", tmp_path / "y.csv", "csv", require_finished=False)).exists()
        assert (await c.redeliver_webhook("job_1")).status == "succeeded"
        assert (await c.submit_job("classify", {}, idempotency_key="k")).id == "job_1"
