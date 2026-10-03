"""Python client: async job methods against a scripted fake server (sync + async)."""

import json

import httpx
import pytest

from clef_client import AsyncClefClient, ClefClient, ClefError, Job, JobItem
from clef_client.types import Classification, ScoreResult


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
