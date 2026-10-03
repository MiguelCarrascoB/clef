import json

import httpx
import pytest

from clef_client import AsyncClefClient, Classification, ClefClient, ClefError, ScoreResult

SINGLE = {
    "model": "clef-flash",
    "multi_label": False,
    "label": "technical",
    "confidence": 0.96,
    "scores": {"billing": 0.04, "technical": 0.96},
    "usage": {"input_tokens": 10},
    "timing": {"total_ms": 5},
    "request_id": "r1",
}
MULTI = {
    "multi_label": True,
    "labels": ["technical", "billing"],
    "scores": {"billing": 0.6, "technical": 0.9, "account": 0.1},
    "threshold": 0.5,
    "request_id": "r2",
}
SCORE = {
    "score": 1.78,
    "level": "high",
    "level_index": 2,
    "confidence": 0.86,
    "distribution": {"low": 0.07, "medium": 0.07, "high": 0.86},
    "usage": {},
    "timing": {},
    "request_id": "r3",
}


def make_handler(seen):
    def handler(req: httpx.Request) -> httpx.Response:
        body = json.loads(req.content) if req.content else None
        seen.append((req.method, req.url.path, body))
        p = req.url.path
        if p == "/v1/classify":
            return httpx.Response(200, json=MULTI if body.get("multi_label") else SINGLE)
        if p == "/v1/classify/batch":
            res = [
                {**SINGLE, "label": f"l{i}", "scores": {f"l{i}": 0.9}} for i, _ in enumerate(body["inputs"])
            ]
            for r in res:
                r.pop("request_id")
            return httpx.Response(200, json={"batch_ms": 1.0, "results": res, "request_id": "rb"})
        if p == "/v1/score":
            return httpx.Response(200, json=SCORE)
        if p == "/v1/classifiers" and req.method == "GET":
            return httpx.Response(200, json={"classifiers": [{"name": "a"}]})
        if p == "/v1/classifiers/support-triage":
            if req.method == "PUT":
                return httpx.Response(200, json={**body, "name": "support-triage"})
            if req.method == "GET":
                return httpx.Response(200, json={"name": "support-triage", "kind": "classify"})
            if req.method == "DELETE":
                return httpx.Response(200, json={"deleted": "support-triage"})
            return httpx.Response(200, json={**SINGLE, "classifier": "support-triage"})
        if p == "/v1/classifiers/support-triage/batch":
            results = [{**SINGLE, "classifier": "support-triage"}] * len(body["inputs"])
            return httpx.Response(200, json={"results": results, "request_id": "rb"})
        if p == "/v1/classifiers/sc":
            return httpx.Response(200, json={**SCORE, "classifier": "sc"})
        return httpx.Response(404, json={"detail": "missing", "request_id": "x"})

    return handler


def client(seen=None, **kw):
    seen = [] if seen is None else seen
    kw.setdefault("backoff", 0)
    return ClefClient(transport=httpx.MockTransport(make_handler(seen)), **kw), seen


def test_classify_single():
    c, seen = client()
    r = c.classify("Checkout is down", ["billing", "technical"], instructions="why?")
    assert isinstance(r, Classification)
    assert (r.label, r.labels, r.confidence, r.multi_label) == ("technical", ["technical"], 0.96, False)
    assert r.scores["billing"] == 0.04 and r.request_id == "r1" and r.raw["model"] == "clef-flash"
    assert seen[0][2] == {
        "input": "Checkout is down",
        "labels": ["billing", "technical"],
        "instructions": "why?",
        "model": "clef-flash",
    }


def test_classify_multi():
    c, seen = client()
    r = c.classify("x", {"billing": "pay"}, multi_label=True, threshold=0.5, images=["data:a"])
    assert r.labels == ["technical", "billing"] and r.label == "technical" and r.confidence == 0.9
    assert r.threshold == 0.5 and r.multi_label
    assert seen[0][2]["multi_label"] is True and seen[0][2]["images"] == ["data:a"]


def test_multi_label_empty():
    def h(req):
        return httpx.Response(200, json={"multi_label": True, "labels": [], "scores": {"a": 0.1}})

    r = ClefClient(transport=httpx.MockTransport(h)).classify("x", ["a"], multi_label=True)
    assert r.label is None and r.confidence is None and r.labels == []


def test_classify_many_order_and_chunks():
    c, seen = client()
    out = c.classify_many(["a", "b", "c"], ["l0", "l1"])
    assert [r.label for r in out] == ["l0", "l1", "l2"] and out[0].request_id == "rb"
    assert len(seen) == 1 and seen[0][1] == "/v1/classify/batch"
    seen.clear()
    out = c.classify_many(["a", "b", "c"], ["l0", "l1"], chunk_size=2)
    assert len(seen) == 2 and len(out) == 3


def test_score():
    c, seen = client()
    r = c.score("x", ["low", "medium", "high"])
    assert isinstance(r, ScoreResult) and r.level == "high" and r.level_index == 2
    assert r.distribution["high"] == 0.86 and seen[0][2]["levels"] == ["low", "medium", "high"]


def test_classifier_handle_crud():
    c, seen = client()
    h = c.classifier("support-triage")
    saved = h.save(labels=["billing", "technical"], multi_label=False, description="d")
    assert saved["kind"] == "classify" and seen[-1][0] == "PUT"
    assert h.get()["name"] == "support-triage"
    r = h.classify("x")
    assert r.classifier == "support-triage" and r.label == "technical"
    many = h.classify_many(["a", "b"])
    assert len(many) == 2 and many[0].classifier == "support-triage"
    assert c.list_classifiers() == [{"name": "a"}]
    assert h.delete() == {"deleted": "support-triage"}
    assert c.save_classifier("support-triage", levels=["a", "b"])["kind"] == "score"
    assert isinstance(c.classifier("sc").classify("x"), ScoreResult)


@pytest.mark.parametrize("status", [503, 429])
def test_retries_then_success(status):
    calls = []

    def h(req):
        calls.append(1)
        if len(calls) < 3:
            return httpx.Response(status, json={"detail": "busy"}, headers={"Retry-After": "0"})
        return httpx.Response(200, json=SINGLE)

    c = ClefClient(transport=httpx.MockTransport(h), backoff=0)
    assert c.classify("x", ["a", "b"]).label == "technical" and len(calls) == 3


def test_retries_exhausted_and_retry_after():
    calls = []

    def h(req):
        calls.append(1)
        return httpx.Response(429, json={"detail": "slow", "request_id": "q"}, headers={"Retry-After": "0"})

    c = ClefClient(transport=httpx.MockTransport(h), retries=2, backoff=0)
    with pytest.raises(ClefError) as ei:
        c.health()
    assert len(calls) == 3 and ei.value.status == 429 and ei.value.retry_after == 0.0
    assert ei.value.request_id == "q"


def test_retry_after_honoured(monkeypatch):
    sleeps = []
    monkeypatch.setattr("clef_client._sync.time.sleep", sleeps.append)
    n = []

    def h(req):
        n.append(1)
        if len(n) == 1:
            return httpx.Response(429, json={"detail": "x"}, headers={"Retry-After": "7"})
        return httpx.Response(200, json={"ok": 1})

    ClefClient(transport=httpx.MockTransport(h)).health()
    assert sleeps == [7.0]


def test_backoff_exponential(monkeypatch):
    sleeps = []
    monkeypatch.setattr("clef_client._sync.time.sleep", sleeps.append)
    c = ClefClient(
        transport=httpx.MockTransport(lambda r: httpx.Response(503, json={"detail": "loading"})),
        retries=3,
        backoff=1.0,
    )
    with pytest.raises(ClefError):
        c.health()
    assert len(sleeps) == 3
    for i, s in enumerate(sleeps):
        assert 0.5 * 2**i <= s <= 2**i


def test_no_retry_on_400():
    calls = []

    def h(req):
        calls.append(1)
        return httpx.Response(400, json={"detail": "labels: need 2", "request_id": "z"})

    c = ClefClient(transport=httpx.MockTransport(h), backoff=0)
    with pytest.raises(ClefError) as ei:
        c.classify("x", ["a"])
    assert len(calls) == 1 and ei.value.status == 400 and "labels" in str(ei.value)


def test_connect_error_retried():
    calls = []

    def h(req):
        calls.append(1)
        if len(calls) < 2:
            raise httpx.ConnectError("refused")
        return httpx.Response(200, json={"ok": 1})

    assert ClefClient(transport=httpx.MockTransport(h), backoff=0).health() == {"ok": 1}
    assert len(calls) == 2


def test_env_fallbacks(monkeypatch):
    monkeypatch.setenv("CLEF_URL", "http://example.test:1234/")
    monkeypatch.setenv("CLEF_API_KEY", "envkey")
    seen = []

    def h(req):
        seen.append(req)
        return httpx.Response(200, json={})

    ClefClient(transport=httpx.MockTransport(h)).health()
    assert str(seen[0].url) == "http://example.test:1234/health"
    assert seen[0].headers["x-api-key"] == "envkey"
    ClefClient(base_url="http://other.test", api_key="arg", transport=httpx.MockTransport(h)).health()
    assert seen[1].url.host == "other.test" and seen[1].headers["x-api-key"] == "arg"


@pytest.mark.asyncio
async def test_async_parity():
    seen = []
    async with AsyncClefClient(transport=httpx.MockTransport(make_handler(seen)), backoff=0) as c:
        assert (await c.classify("x", ["a", "b"])).label == "technical"
        assert (await c.classify("x", ["a", "b"], multi_label=True)).labels == ["technical", "billing"]
        out = await c.classify_many(["a", "b", "c"], ["l0", "l1"], chunk_size=2)
        assert [r.label for r in out] == ["l0", "l1", "l0"]  # mock restarts per chunk
        assert (await c.score("x", ["low", "high"])).level == "high"
        h = c.classifier("support-triage")
        assert (await h.save(labels=["a", "b"]))["kind"] == "classify"
        assert (await h.classify("x")).classifier == "support-triage"
        assert len(await h.classify_many(["a", "b"])) == 2
        assert (await h.get())["name"] == "support-triage"
        assert await c.list_classifiers() == [{"name": "a"}]
        assert await h.delete() == {"deleted": "support-triage"}


@pytest.mark.asyncio
async def test_async_retry_and_errors(monkeypatch):
    calls = []

    async def nosleep(_):
        return None

    monkeypatch.setattr("clef_client._async.asyncio.sleep", nosleep)

    def h(req):
        calls.append(1)
        if len(calls) < 3:
            return httpx.Response(503, json={"detail": "loading"})
        if len(calls) == 3:
            return httpx.Response(200, json=SINGLE)
        return httpx.Response(400, json={"detail": "bad"})

    c = AsyncClefClient(transport=httpx.MockTransport(h))
    assert (await c.classify("x", ["a", "b"])).label == "technical" and len(calls) == 3
    with pytest.raises(ClefError):
        await c.classify("x", ["a", "b"])
    assert len(calls) == 4
