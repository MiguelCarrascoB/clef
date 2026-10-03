"""HTTP API tests with a FakeEngine (no GPU, no model)."""

from __future__ import annotations

import asyncio
import base64
import io
import json
import tempfile
import time
from typing import Any

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from clef_server.config import Config
from clef_server.engine import EngineNotReady, GpuOutOfMemory, InputTooLarge
from clef_server.main import create_app, sse_events

GOOD = {
    "state": "Checkout errors",
    "questions": {
        "dept": {"type": "choice", "criteria": {"billing": "pay", "technical": "bugs"}},
        "urg": {"type": "score", "criteria": ["low", "high"]},
        "out": {"type": "noul"},
    },
}


class FakeEngine:
    def __init__(self) -> None:
        self.status, self.error, self.load_seconds = "ready", None, 1.5
        self.calls: list[list[dict[str, Any]]] = []
        self.raises: Exception | None = None
        self.started = self.stopped = False
        self.samples = 0
        self.noul: dict[str, float] = {}

    def start(self) -> None:
        self.started = True

    def shutdown(self) -> None:
        self.stopped = True

    def info(self) -> dict[str, Any]:
        return {
            "backend": "cuda",
            "device": "cuda",
            "dtype": "bfloat16",
            "model_path": "/m",
            "torch": "x",
            "transformers": "y",
            "gpu": {
                "available": True,
                "name": "GPU",
                "memory_kind": "vram",
                "vram_total_gb": 24.0,
                "vram_allocated_gb": 1.0,
                "vram_reserved_gb": 2.0,
            },
            "telemetry": {"available": True, "source": "nvml", "util_pct": 12.0},
        }

    def sample(self) -> dict[str, Any]:
        self.samples += 1
        return {"mem_used_gb": 1.0, "mem_total_gb": 24.0, "gpu_util_pct": 5.0}

    def answer(self, qid: str, q: dict[str, Any]) -> dict[str, Any]:
        """Deterministic: choice/score weight i+1 (last option wins); noul = self.noul[qid] (default 0.5)."""
        if q["type"] == "choice":
            opts = list(q["criteria"])
            tot = sum(range(1, len(opts) + 1))
            probs = {o: (i + 1) / tot for i, o in enumerate(opts)}
            return {
                "type": "choice",
                "choice": opts[-1],
                "confidence": probs[opts[-1]],
                "probabilities": probs,
            }
        if q["type"] == "score":
            n = len(q["criteria"])
            tot = sum(range(1, n + 1))
            probs = {str(i): (i + 1) / tot for i in range(n)}
            return {
                "type": "score",
                "score": sum(i * p for i, p in enumerate(probs.values())),
                "confidence": probs[str(n - 1)],
                "legend": {str(i): c for i, c in enumerate(q["criteria"])},
                "probabilities": probs,
            }
        return {"type": "noul", "noul": self.noul.get(qid, 0.5)}

    async def decide(self, records: list[dict[str, Any]]) -> list[dict[str, Any]]:
        self.calls.append(records)
        if self.raises:
            raise self.raises
        return [
            {
                "model": r["model"],
                "answers": {q: self.answer(q, spec) for q, spec in r["questions"].items()},
                "usage": {"input_tokens": 10, "output_tokens": 0},
                "timing": {"queue_ms": 1.0, "forward_ms": 2.0, "batch_size": len(records)},
            }
            for r in records
        ]


def make(engine: FakeEngine | None = None, **cfg_kw: Any):
    eng = engine or FakeEngine()
    cfg_kw.setdefault("state_dir", tempfile.mkdtemp(prefix="clef-test-"))
    app = create_app(Config(**{"api_key": None, **cfg_kw}), eng)
    return TestClient(app, raise_server_exceptions=False), eng, app


@pytest.fixture
def api():
    client, eng, app = make()
    with client:
        yield client, eng, app


def png_url(w: int = 20, h: int = 10) -> str:
    buf = io.BytesIO()
    Image.new("RGB", (w, h), "blue").save(buf, "PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


# ------------------------------------------------------------------ happy path / basics


def test_systemone_ok_and_lifespan(api):
    client, eng, _ = api
    assert eng.started
    r = client.post("/v1/systemone", json=GOOD)
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["model"] == "clef-flash" and set(body["answers"]) == {"dept", "urg", "out"}
    assert body["usage"]["input_tokens"] == 10
    assert body["timing"]["total_ms"] >= 0 and body["timing"]["batch_size"] == 1
    rec = eng.calls[0][0]
    assert rec["images"] is None and rec["videos"] is None and rec["model"] == "clef-flash"
    assert "X-Request-ID" in r.headers


def test_shutdown_called():
    client, eng, _ = make()
    with client:
        pass
    assert eng.stopped


def test_batch_ok_and_order(api):
    client, eng, _ = api
    items = [{**GOOD, "model": f"m{i}"} for i in range(3)]
    r = client.post("/v1/batch", json={"batch": items})
    assert r.status_code == 200
    body = r.json()
    assert [x["model"] for x in body["results"]] == ["m0", "m1", "m2"] and body["batch_ms"] >= 0
    assert len(eng.calls[0]) == 3


def test_batch_media_allowed(api):
    client, eng, _ = api
    r = client.post("/v1/batch", json={"batch": [{**GOOD, "images": [png_url()]}, GOOD]})
    assert r.status_code == 200
    assert len(eng.calls[0][0]["images"]) == 1 and eng.calls[0][1]["images"] is None


def test_schema_example_livez_openapi(api):
    client, _, _ = api
    assert client.get("/livez").json() == {"ok": True}
    assert "text_json" in client.get("/schema-example").json()
    spec = client.get("/openapi.json").json()
    assert "SystemOneRequest" in spec["components"]["schemas"] and "/v1/batch" in spec["paths"]


# ------------------------------------------------------------------ validation


@pytest.mark.parametrize(
    "patch,expect",
    [
        (
            {"questions": {"u": {"type": "score", "criteria": []}}},
            "questions.u: criteria must be a non-empty list",
        ),  # noqa: E501
        (
            {"questions": {"d": {"type": "choice", "criteria": {}}}},
            "questions.d: criteria must be a non-empty object",
        ),  # noqa: E501
        ({"questions": {"n": {"type": "noul", "criteria": {"maybe": "x"}}}}, "questions.n: criteria"),
        ({"questions": {}}, "at least one question"),
        ({"questions": {"": {"type": "noul"}}}, "question ids must be non-empty"),
        ({"media_kwargs": {"max_pixels": 1}}, "media_kwargs: unknown field"),
        ({"model": 5}, "model:"),
        ({"model": None}, "model:"),
        ({"images": ["x"] * 9}, "too many images"),
    ],
)
def test_validation_400(api, patch, expect):
    client, eng, _ = api
    r = client.post("/v1/systemone", json={**GOOD, **patch})
    assert r.status_code == 400, r.text
    body = r.json()
    assert expect in body["detail"] and body["request_id"] == r.headers["X-Request-ID"]
    assert not eng.calls


def test_missing_state_and_bad_json(api):
    client, _, _ = api
    r = client.post("/v1/systemone", json={"questions": GOOD["questions"]})
    assert r.status_code == 400 and "state: field is required" in r.json()["detail"]
    r = client.post("/v1/systemone", content=b"{nope", headers={"content-type": "application/json"})
    assert r.status_code == 400 and "not valid JSON" in r.json()["detail"]


def test_batch_validation_names_index(api):
    client, _, _ = api
    bad = {**GOOD, "questions": {"urgency": {"type": "score", "criteria": []}}}
    r = client.post("/v1/batch", json={"batch": [GOOD, GOOD, GOOD, bad]})
    assert r.status_code == 400
    assert r.json()["detail"] == "batch[3].questions.urgency: criteria must be a non-empty list"
    assert client.post("/v1/batch", json={"batch": []}).status_code == 400


def test_batch_too_large():
    client, eng, _ = make(max_batch=2)
    with client:
        r = client.post("/v1/batch", json={"batch": [GOOD] * 3})
    assert r.status_code == 400 and "max 2" in r.json()["detail"] and not eng.calls


def test_max_questions():
    client, _, _ = make(max_questions=2)
    with client:
        r = client.post("/v1/systemone", json=GOOD)
    assert r.status_code == 400 and "too many questions" in r.json()["detail"]


def test_model_default_on_every_route(api):
    client, eng, _ = api
    client.post("/v1/batch", json={"batch": [GOOD]})
    assert eng.calls[0][0]["model"] == "clef-flash"


# ------------------------------------------------------------------ auth


def test_auth_off_allows(api):
    client, _, _ = api
    assert client.get("/v1/stats").status_code == 200


def test_auth_on():
    client, eng, _ = make(api_key="s3cret")
    with client:
        assert client.post("/v1/systemone", json=GOOD).status_code == 401
        r = client.get("/v1/stats", headers={"X-API-Key": "wrong"})
        assert r.status_code == 401 and r.json()["request_id"]
        assert client.post("/v1/systemone", json=GOOD, headers={"X-API-Key": "s3cret"}).status_code == 200
        assert (
            client.post("/v1/systemone", json=GOOD, headers={"Authorization": "Bearer s3cret"}).status_code
            == 200
        )
        assert client.get("/v1/log?key=s3cret").status_code == 401  # ?key= only for /v1/events
        for open_path in ("/health", "/livez", "/schema-example", "/docs"):
            assert client.get(open_path).status_code == 200, open_path
    assert len(eng.calls) == 2


# ------------------------------------------------------------------ body limit


def test_body_limit_content_length():
    client, eng, _ = make(max_body_mb=1)
    with client:
        r = client.post(
            "/v1/systemone", content=b"x" * (2 * 1024 * 1024), headers={"content-type": "application/json"}
        )
        assert r.status_code == 413 and "limit" in r.json()["detail"]
        assert r.headers["X-Request-ID"] and not eng.calls


def test_body_limit_chunked():
    client, _, _ = make(max_body_mb=1)

    def gen():
        for _ in range(3):
            yield b"x" * (512 * 1024)

    with client:
        r = client.post("/v1/systemone", content=gen(), headers={"content-type": "application/json"})
    assert r.status_code == 413


# ------------------------------------------------------------------ error mapping


@pytest.mark.parametrize(
    "exc,status",
    [
        (EngineNotReady("model is loading"), 503),
        (GpuOutOfMemory("oom"), 503),
        (InputTooLarge("too big"), 413),
    ],
)
def test_engine_error_mapping(exc, status):
    eng = FakeEngine()
    eng.raises = exc
    client, _, _ = make(eng)
    with client:
        for path, body in (("/v1/systemone", GOOD), ("/v1/batch", {"batch": [GOOD]})):
            r = client.post(path, json=body)
            assert r.status_code == status, r.text
            assert r.json()["request_id"] and r.json()["detail"]


def test_500_is_generic_and_logged(caplog):
    eng = FakeEngine()
    eng.raises = RuntimeError("secret boom detail")
    client, _, _ = make(eng)
    with client, caplog.at_level("ERROR"):
        r = client.post("/v1/systemone", json=GOOD)
    assert r.status_code == 500
    text = r.text
    assert "Traceback" not in text and "secret boom" not in text and "RuntimeError" not in text
    assert (
        r.json()["detail"] == "internal server error" and r.headers["X-Request-ID"] == r.json()["request_id"]
    )  # noqa: E501
    assert any("secret boom" in (rec.exc_text or "") or rec.exc_info for rec in caplog.records)


def test_media_error_is_400(api):
    client, eng, _ = api
    r = client.post("/v1/systemone", json={**GOOD, "images": ["data:image/png;base64,@@@"]})
    assert r.status_code == 400 and "images[0]" in r.json()["detail"] and not eng.calls
    r = client.post("/v1/batch", json={"batch": [GOOD, {**GOOD, "images": ["https://example.com/a.png"]}]})
    assert (
        r.status_code == 400
        and "batch[1].images[0]" in r.json()["detail"]
        and "disabled" in r.json()["detail"]
    )  # noqa: E501


def test_not_found_uses_error_shape(api):
    client, _, _ = api
    r = client.get("/v1/nope")
    assert r.status_code == 404 and "request_id" in r.json()


def test_early_503_when_loading_skips_decode():
    eng = FakeEngine()
    eng.status, eng.error = "loading", None
    client, _, _ = make(eng)
    with client:
        r = client.post("/v1/systemone", json=GOOD)
    assert r.status_code == 503 and not eng.calls


# ------------------------------------------------------------------ health


def test_health_ready_warming_and_down():
    eng = FakeEngine()
    client, _, _ = make(eng)
    with client:
        r = client.get("/health")
        h = r.json()
        assert r.status_code == 200 and h["ready"] and h["status"] == "ready" and h["version"]
        assert h["gpu"]["name"] == "GPU" and h["load_seconds"] == 1.5 and "limits" in h and h["uptime_s"] >= 0
        eng.status = "warming"
        assert client.get("/health").status_code == 200
        eng.status, eng.error = "loading", None
        r = client.get("/health")
        assert r.status_code == 503 and r.json()["ready"] is False
        eng.status, eng.error = "error", "no gpu"
        r = client.get("/health")
        assert r.status_code == 503 and r.json()["error"] == "no gpu"


# ------------------------------------------------------------------ media in one record


def test_five_images_one_record(api):
    client, eng, _ = api
    r = client.post("/v1/systemone", json={**GOOD, "images": [png_url() for _ in range(5)]})
    assert r.status_code == 200, r.text
    assert len(eng.calls) == 1 and len(eng.calls[0]) == 1
    assert len(eng.calls[0][0]["images"]) == 5 and eng.calls[0][0]["videos"] is None


def test_image_downscaled_before_engine():
    client, eng, _ = make(max_pixels=500)
    with client:
        r = client.post("/v1/systemone", json={**GOOD, "images": [png_url(200, 100)]})
    assert r.status_code == 200
    w, h = eng.calls[0][0]["images"][0].size
    assert w * h <= 500


def test_video_frame_cap(tmp_path):
    import imageio.v3 as iio
    import numpy as np

    path = tmp_path / "v.mp4"
    frames = np.random.default_rng(1).integers(0, 255, (40, 48, 64, 3), dtype=np.uint8)
    iio.imwrite(str(path), frames, fps=10, plugin="FFMPEG")
    url = "data:video/mp4;base64," + base64.b64encode(path.read_bytes()).decode()
    client, eng, _ = make(max_frames=5, video_fps=5.0)
    with client:
        r = client.post("/v1/systemone", json={**GOOD, "images": [png_url()], "videos": [url]})
    assert r.status_code == 200, r.text
    rec = eng.calls[0][0]
    assert len(eng.calls[0]) == 1 and len(rec["images"]) == 1 and len(rec["videos"]) == 1
    assert rec["videos"][0].shape[0] == 5 and rec["videos"][0].dtype.name == "uint8"


def test_url_fetch_disabled_and_ssrf_via_api():
    client, eng, _ = make()
    with client:
        r = client.post("/v1/systemone", json={**GOOD, "images": ["http://127.0.0.1:8910/health"]})
        assert r.status_code == 400 and "disabled" in r.json()["detail"]
    client, eng, _ = make(allow_url_fetch=True)
    with client:
        for url in ("http://127.0.0.1/x", "http://169.254.169.254/latest/meta-data/"):
            r = client.post("/v1/systemone", json={**GOOD, "images": [url]})
            assert r.status_code == 400 and "non-public" in r.json()["detail"]
    assert not eng.calls


# ------------------------------------------------------------------ stats / log / request id


def test_request_id_header_roundtrip(api):
    client, _, _ = api
    r = client.get("/livez", headers={"X-Request-ID": "abc-123"})
    assert r.headers["X-Request-ID"] == "abc-123"
    r = client.get("/livez", headers={"X-Request-ID": "bad id with spaces!"})
    assert r.headers["X-Request-ID"] != "bad id with spaces!" and len(r.headers["X-Request-ID"]) == 32


def test_stats_and_log_record_requests():
    client, eng, _ = make(log_state=True)
    with client:
        client.post("/v1/systemone", json={**GOOD, "images": [png_url()]})
        client.post("/v1/systemone", json={**GOOD, "questions": {}})
        eng.raises = EngineNotReady("loading")
        client.post("/v1/systemone", json=GOOD)
        entries = client.get("/v1/log").json()["entries"]
        assert [e["status"] for e in entries] == [200, 400, 503]
        ok = entries[0]
        assert ok["endpoint"] == "/v1/systemone" and ok["input_tokens"] == 10 and ok["n_records"] == 1
        assert ok["n_questions"] == 3 and ok["media"] == {"images": 1, "videos": 0}
        assert ok["state_preview"] and len(ok["state_preview"]) <= 200 and ok["ms"] >= 0
        assert entries[1]["error"] and entries[2]["error"] == "loading"
        assert [e["id"] for e in client.get(f"/v1/log?since={entries[0]['id']}").json()["entries"]] == [2, 3]
        s = client.get("/v1/stats").json()
        assert s["total"] == 3 and s["errors"] == 2 and s["status"] == "ready" and s["in_flight"] == 0
        assert s["gpu"]["vram_total_gb"] == 24.0 and len(s["latency_ms"]["hist"]["counts"]) == 11
        assert s["tokens"]["in_total"] == 10


def test_state_preview_off_by_default(api):
    client, _, _ = api
    client.post("/v1/systemone", json=GOOD)
    assert client.get("/v1/log").json()["entries"][0]["state_preview"] is None


def test_413_and_401_are_recorded():
    client, _, _ = make(max_body_mb=1, api_key="k")
    with client:
        client.post("/v1/systemone", json=GOOD)
        client.post("/v1/systemone", content=b"x" * (2 * 1024 * 1024), headers={"X-API-Key": "k"})
        entries = client.get("/v1/log", headers={"X-API-Key": "k"}).json()["entries"]
    assert [e["status"] for e in entries] == [401, 413]


# ------------------------------------------------------------------ SSE


def test_sse_event_stream_and_disconnect():
    async def main() -> list[str]:
        client, eng, app = make()
        stats = app.state.stats
        calls = 0

        async def disconnected() -> bool:
            nonlocal calls
            calls += 1
            if calls == 2:
                stats.record_request({"endpoint": "/v1/systemone", "status": 200, "ms": 3.0})
            return calls > 6

        out = []
        async for chunk in sse_events(stats, lambda: stats.snapshot(), disconnected, interval=0.2):
            out.append(chunk)
        assert stats._subs == []  # unsubscribed after disconnect
        return out

    chunks = asyncio.run(main())
    assert chunks[0].startswith("event: stats\ndata: ")
    logs = [c for c in chunks if c.startswith("event: log")]
    assert len(logs) == 1 and json.loads(logs[0].split("data: ", 1)[1])["endpoint"] == "/v1/systemone"
    assert len([c for c in chunks if c.startswith("event: stats")]) >= 2


def test_events_route_requires_key_param():
    client, _, _ = make(api_key="k")
    with client:
        assert client.get("/v1/events").status_code == 401


# ================================================================== v3: classification API

LABELS = ["billing", "technical"]


def test_classify_single_matches_systemone_record(api):
    client, eng, _ = api
    r = client.post(
        "/v1/classify", json={"input": "Checkout is down", "labels": LABELS, "instructions": "Which team?"}
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["multi_label"] is False and body["label"] == "technical"
    assert body["confidence"] == pytest.approx(2 / 3) and body["scores"]["billing"] == pytest.approx(1 / 3)
    assert body["usage"]["input_tokens"] == 10 and body["timing"]["total_ms"] >= 0
    assert body["request_id"] == r.headers["X-Request-ID"] and "labels" not in body
    rec = eng.calls[0][0]
    assert rec["state"] == "Checkout is down" and rec["model"] == "clef-flash"
    assert rec["questions"] == {
        "label": {
            "type": "choice",
            "instructions": "Which team?",
            "criteria": {"billing": "billing", "technical": "technical"},
        }
    }


def test_classify_multi_label(api):
    client, eng, _ = api
    eng.noul = {"billing": 0.2, "technical": 0.9, "urgent": 0.6}
    r = client.post(
        "/v1/classify",
        json={
            "input": {"ticket": "x"},
            "labels": {"billing": "Payments", "technical": "technical", "urgent": "Time sensitive"},
            "multi_label": True,
            "instructions": "Tag it.",
        },
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["multi_label"] is True and body["labels"] == ["technical", "urgent"]
    assert body["threshold"] == 0.5 and "label" not in body and "confidence" not in body
    assert body["scores"] == {"billing": 0.2, "technical": 0.9, "urgent": 0.6}
    qs = eng.calls[0][0]["questions"]
    assert qs["billing"] == {
        "type": "noul",
        "instructions": 'Tag it. Does the label "billing" apply?',
        "criteria": {"true": "Payments"},
    }
    assert "criteria" not in qs["technical"] and eng.calls[0][0]["state"] == {"ticket": "x"}
    r = client.post(
        "/v1/classify", json={"input": "x", "labels": ["technical"], "multi_label": True, "threshold": 0.95}
    )
    assert r.json()["labels"] == [] and r.json()["threshold"] == 0.95


def test_classify_default_threshold_from_config():
    client, eng, _ = make(classify_threshold=0.8)
    eng.noul = {"a": 0.7, "b": 0.9}
    with client:
        r = client.post("/v1/classify", json={"input": "x", "labels": ["a", "b"], "multi_label": True})
    assert r.json()["labels"] == ["b"] and r.json()["threshold"] == 0.8


@pytest.mark.parametrize(
    "patch,expect",
    [
        ({"labels": ["a"]}, "at least 2 labels"),
        ({"labels": ["a", "a"]}, "labels must be unique"),
        ({"labels": ["a", ""]}, "non-empty"),
        ({"labels": []}, "at least 2 labels"),
        ({"labels": ["a", "b", "c"]}, "max 2"),
        ({"labels": ["a", "b"], "threshold": 0.3}, "threshold"),
        ({"labels": ["a", "b"], "multi_label": True, "threshold": 2}, "threshold"),
        ({"labels": ["a", "b"], "extra": 1}, "extra: unknown field"),
        ({"labels": "ab"}, "labels"),
    ],
)
def test_classify_validation(patch, expect):
    client, eng, _ = make(max_labels=2)
    with client:
        r = client.post("/v1/classify", json={"input": "x", **patch})
        assert r.status_code == 400 and expect in r.json()["detail"], r.text
        r = client.post("/v1/classify", json={"labels": ["a", "b"]})
        assert r.status_code == 400 and "input: field is required" in r.json()["detail"]
    assert not eng.calls


def test_classify_media_goes_through_load_media(api):
    client, eng, _ = api
    r = client.post("/v1/classify", json={"input": "see", "labels": LABELS, "images": [png_url()]})
    assert r.status_code == 200
    assert len(eng.calls[0][0]["images"]) == 1
    bad = {"input": "see", "labels": LABELS, "images": ["data:image/png;base64,@@@"]}
    r = client.post("/v1/classify", json=bad)
    assert r.status_code == 400 and "images[0]" in r.json()["detail"]


def test_classify_batch_is_one_decide_in_order(api):
    client, eng, _ = api
    r = client.post("/v1/classify/batch", json={"inputs": ["one", {"k": 2}, "three"], "labels": LABELS})
    assert r.status_code == 200, r.text
    body = r.json()
    assert len(eng.calls) == 1 and [rec["state"] for rec in eng.calls[0]] == ["one", {"k": 2}, "three"]
    assert len(body["results"]) == 3 and body["batch_ms"] >= 0 and body["request_id"]
    assert all(x["label"] == "technical" and "request_id" not in x for x in body["results"])
    assert body["results"][0]["timing"]["batch_size"] == 3
    assert body["results"][0]["usage"]["input_tokens"] == 10


def test_classify_batch_limits_and_empty():
    client, eng, _ = make(max_batch=2)
    with client:
        r = client.post("/v1/classify/batch", json={"inputs": ["a", "b", "c"], "labels": LABELS})
        assert r.status_code == 400 and "max 2" in r.json()["detail"]
        assert client.post("/v1/classify/batch", json={"inputs": [], "labels": LABELS}).status_code == 400
        r = client.post("/v1/classify/batch", json={"inputs": ["a"], "labels": LABELS, "images": []})
        assert r.status_code == 400  # no media in batch
    assert not eng.calls


def test_classify_batch_error_names_index():
    client, eng, _ = make(max_questions=2)
    with client:
        body = {"inputs": ["a"], "labels": ["a", "b", "c"], "multi_label": True}
        r = client.post("/v1/classify/batch", json=body)
    assert r.status_code == 400 and r.json()["detail"].startswith("inputs[0].questions")


def test_score_route(api):
    client, eng, _ = api
    r = client.post(
        "/v1/score",
        json={"input": "urgent!", "levels": ["low", "medium", "high"], "instructions": "How urgent?"},
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["level"] == "high" and body["level_index"] == 2
    assert body["distribution"] == {
        "low": pytest.approx(1 / 6),
        "medium": pytest.approx(2 / 6),
        "high": pytest.approx(3 / 6),
    }
    assert body["score"] == pytest.approx(2 / 6 + 2 * 3 / 6) and body["request_id"]
    assert eng.calls[0][0]["questions"] == {
        "score": {"type": "score", "criteria": ["low", "medium", "high"], "instructions": "How urgent?"}
    }
    assert client.post("/v1/score", json={"input": "x", "levels": ["only"]}).status_code == 400
    assert client.post("/v1/score", json={"input": "x", "levels": ["a", "a"]}).status_code == 400


def test_classify_engine_errors_map():
    eng = FakeEngine()
    eng.raises = EngineNotReady("loading")
    client, _, _ = make(eng)
    with client:
        for path, body in (
            ("/v1/classify", {"input": "x", "labels": LABELS}),
            ("/v1/classify/batch", {"inputs": ["x"], "labels": LABELS}),
            ("/v1/score", {"input": "x", "levels": ["a", "b"]}),
        ):
            assert client.post(path, json=body).status_code == 503


def test_classify_logged_and_counted():
    client, _, _ = make()
    with client:
        client.post("/v1/classify", json={"input": "x", "labels": LABELS})
        client.post("/v1/classify/batch", json={"inputs": ["x", "y"], "labels": LABELS})
        client.post("/v1/score", json={"input": "x", "levels": ["a", "b"]})
        client.put("/v1/classifiers/tri", json={"kind": "classify", "labels": LABELS})
        client.post("/v1/classifiers/tri", json={"input": "x"})
        client.post("/v1/classifiers/tri/batch", json={"inputs": ["x"]})
        entries = client.get("/v1/log").json()["entries"]
    assert [e["endpoint"] for e in entries] == [
        "/v1/classify",
        "/v1/classify/batch",
        "/v1/score",
        "/v1/classifiers/tri",
        "/v1/classifiers/tri/batch",
    ]
    assert [e["n_records"] for e in entries] == [1, 2, 1, 1, 1]
    assert all(e["status"] == 200 for e in entries)


# ------------------------------------------------------------------ saved classifiers over HTTP


def test_classifier_crud_and_run(api):
    client, eng, _ = api
    labels = {"billing": "Payments", "technical": "Bugs"}
    d = {"kind": "classify", "labels": labels, "instructions": "Team?"}
    r = client.put("/v1/classifiers/support-triage", json={**d, "description": "d"})
    assert r.status_code == 200, r.text
    doc = r.json()
    assert doc["name"] == "support-triage" and doc["created_at"] and doc["updated_at"]
    assert doc["description"] == "d"
    r2 = client.put("/v1/classifiers/support-triage", json=d)
    assert r2.json()["created_at"] == doc["created_at"] and r2.json()["description"] is None
    assert client.get("/v1/classifiers/support-triage").json()["labels"] == labels
    client.put("/v1/classifiers/alpha", json={"kind": "score", "levels": ["lo", "hi"]})
    listed = client.get("/v1/classifiers").json()["classifiers"]
    assert [c["name"] for c in listed] == ["alpha", "support-triage"]

    r = client.post("/v1/classifiers/support-triage", json={"input": "boom"})
    assert r.status_code == 200, r.text
    assert r.json()["classifier"] == "support-triage" and r.json()["label"] == "technical"
    q = eng.calls[-1][0]["questions"]["label"]
    assert q["criteria"] == labels and q["instructions"] == "Team?"
    r = client.post("/v1/classifiers/alpha", json={"input": "boom"})
    assert r.json()["classifier"] == "alpha" and r.json()["level"] == "hi"
    r = client.post("/v1/classifiers/support-triage/batch", json={"inputs": ["a", "b"]})
    assert r.status_code == 200 and r.json()["classifier"] == "support-triage"
    assert len(r.json()["results"]) == 2

    assert client.delete("/v1/classifiers/alpha").json() == {"deleted": "alpha"}
    assert client.delete("/v1/classifiers/alpha").status_code == 404
    assert client.get("/v1/classifiers/alpha").status_code == 404
    r = client.post("/v1/classifiers/alpha", json={"input": "x"})
    assert r.status_code == 404 and r.json()["request_id"]


def test_classifier_multi_label_threshold_precedence(api):
    client, eng, _ = api
    eng.noul = {"a": 0.6, "b": 0.8}
    multi = {"kind": "classify", "labels": ["a", "b"], "multi_label": True, "threshold": 0.7}
    client.put("/v1/classifiers/m", json=multi)
    assert client.post("/v1/classifiers/m", json={"input": "x"}).json()["labels"] == ["b"]
    r = client.post("/v1/classifiers/m", json={"input": "x", "threshold": 0.5})
    assert r.json()["labels"] == ["b", "a"]
    client.put("/v1/classifiers/s", json={"kind": "classify", "labels": ["a", "b"]})
    assert client.post("/v1/classifiers/s", json={"input": "x", "threshold": 0.5}).status_code == 400
    client.put("/v1/classifiers/sc", json={"kind": "score", "levels": ["a", "b"]})
    assert client.post("/v1/classifiers/sc/batch", json={"inputs": ["x"]}).status_code == 400


@pytest.mark.parametrize("name", ["Upper", "-x", "has space", "x" * 65, "a.b", "a%00b"])
def test_classifier_bad_names_400(api, name):
    client, _, _ = api
    body = {"kind": "classify", "labels": LABELS}
    assert client.put(f"/v1/classifiers/{name}", json=body).status_code == 400
    assert client.get(f"/v1/classifiers/{name}").status_code in (400, 404)
    assert client.delete(f"/v1/classifiers/{name}").status_code in (400, 404)
    assert client.get("/v1/classifiers").json()["classifiers"] == []


def test_classifier_path_traversal_attempts(api):
    client, _, app = api
    body = {"kind": "classify", "labels": LABELS}
    for path in ("/v1/classifiers/..%2f..%2fevil", "/v1/classifiers/%2e%2e", "/v1/classifiers/..%5cevil"):
        r = client.put(path, json=body)
        assert r.status_code in (400, 404, 405), (path, r.status_code)
    from pathlib import Path

    root = Path(app.state.cfg.state_dir)
    assert not list(root.rglob("evil*"))
    assert client.get("/v1/classifiers").json()["classifiers"] == []


@pytest.mark.parametrize(
    "body,expect",
    [
        ({"kind": "classify"}, "labels"),
        ({"kind": "classify", "labels": ["a"]}, "at least 2"),
        ({"kind": "classify", "labels": ["a", "b"], "threshold": 0.4}, "threshold"),
        ({"kind": "score", "levels": ["a"]}, "at least 2"),
        ({"kind": "score"}, "levels"),
        ({"kind": "wat", "labels": ["a", "b"]}, "kind"),
        ({"kind": "classify", "labels": ["a", "b"], "zzz": 1}, "zzz"),
    ],
)
def test_classifier_put_validation(api, body, expect):
    client, _, _ = api
    r = client.put("/v1/classifiers/x", json=body)
    assert r.status_code == 400 and expect in r.json()["detail"], r.text
    assert client.get("/v1/classifiers").json()["classifiers"] == []


def test_classifier_max_409():
    client, _, _ = make(max_classifiers=1)
    body = {"kind": "classify", "labels": LABELS}
    with client:
        assert client.put("/v1/classifiers/a", json=body).status_code == 200
        r = client.put("/v1/classifiers/b", json=body)
        assert r.status_code == 409 and r.json()["request_id"]
        assert client.put("/v1/classifiers/a", json=body).status_code == 200


def test_classifier_corrupt_file_skipped_in_list(api):
    client, _, app = api
    client.put("/v1/classifiers/ok", json={"kind": "classify", "labels": LABELS})
    from clef_server.paths import classifiers_dir

    (classifiers_dir(app.state.cfg) / "bad.json").write_text("{nope")
    assert [c["name"] for c in client.get("/v1/classifiers").json()["classifiers"]] == ["ok"]
    assert client.get("/v1/classifiers/bad").status_code == 404


def test_new_routes_in_openapi(api):
    client, _, _ = api
    spec = client.get("/openapi.json").json()
    for path in (
        "/v1/classify",
        "/v1/classify/batch",
        "/v1/score",
        "/v1/classifiers",
        "/v1/classifiers/{name}",
        "/v1/classifiers/{name}/batch",
        "/v1/stats/timeseries",
    ):
        assert path in spec["paths"], path
    assert {"ClassifyRequest", "ClassifyResponse", "ScoreResponse", "ClassifierDef"} <= set(
        spec["components"]["schemas"]
    )


# ------------------------------------------------------------------ keys

KEYS = "web:key-web,batch:key-batch"
CLS = {"input": "x", "labels": LABELS}


def test_multi_key_auth_and_key_in_log():
    client, _, _ = make(api_keys_raw=KEYS)
    with client:
        assert client.post("/v1/systemone", json=GOOD).status_code == 401
        assert client.post("/v1/systemone", json=GOOD, headers={"X-API-Key": "nope"}).status_code == 401
        r = client.post("/v1/systemone", json=GOOD, headers={"X-API-Key": "key-web"})
        assert r.status_code == 200
        r = client.post("/v1/classify", json=CLS, headers={"Authorization": "Bearer key-batch"})
        assert r.status_code == 200
        entries = client.get("/v1/log", headers={"X-API-Key": "key-web"}).json()["entries"]
    assert [(e["status"], e["key"]) for e in entries] == [
        (401, None),
        (401, None),
        (200, "web"),
        (200, "batch"),
    ]
    assert "key-web" not in json.dumps(entries) and "key-batch" not in json.dumps(entries)


def test_api_key_and_api_keys_combined():
    client, _, _ = make(api_key="solo", api_keys_raw="other:o")
    with client:
        for k in ("solo", "o"):
            assert client.post("/v1/systemone", json=GOOD, headers={"X-API-Key": k}).status_code == 200
        log = client.get("/v1/log", headers={"X-API-Key": "o"}).json()["entries"]
    assert [e["key"] for e in log] == ["default", "other"]


def test_key_query_param_only_on_events():
    client, _, _ = make(api_keys_raw=KEYS)
    with client:
        assert client.get("/v1/stats?key=key-web").status_code == 401
        assert client.post("/v1/systemone?key=key-web", json=GOOD).status_code == 401
        assert client.get("/v1/classifiers?key=key-web").status_code == 401


def test_auth_off_key_is_null(api):
    client, _, _ = api
    client.post("/v1/systemone", json=GOOD)
    assert client.get("/v1/log").json()["entries"][0]["key"] is None


def test_new_routes_need_auth():
    client, _, _ = make(api_keys_raw=KEYS)
    with client:
        for method, path in (
            ("get", "/v1/classifiers"),
            ("put", "/v1/classifiers/x"),
            ("delete", "/v1/classifiers/x"),
            ("post", "/v1/classifiers/x"),
            ("post", "/v1/score"),
            ("get", "/v1/stats/timeseries"),
        ):
            assert getattr(client, method)(path).status_code == 401, path


# ------------------------------------------------------------------ rate limit


def test_rate_limit_429_retry_after_and_not_counted():
    client, eng, _ = make(rate_limit=2)
    with client:
        assert client.post("/v1/systemone", json=GOOD).status_code == 200
        assert client.post("/v1/classify", json=CLS).status_code == 200
        r = client.post("/v1/systemone", json=GOOD)
        assert r.status_code == 429
        assert r.json()["detail"] == "rate limit exceeded (2/min)" and r.json()["request_id"]
        ra = r.headers["Retry-After"]
        assert ra.isdigit() and 1 <= int(ra) <= 60
        assert client.post("/v1/batch", json={"batch": [GOOD]}).status_code == 429
        assert len(eng.calls) == 2
        entries = client.get("/v1/log").json()["entries"]
    assert [e["status"] for e in entries] == [200, 200, 429, 429]


def test_rate_limit_only_inference_routes():
    client, _, _ = make(rate_limit=1)
    cdef = {"kind": "classify", "labels": LABELS}
    with client:
        assert client.post("/v1/systemone", json=GOOD).status_code == 200
        assert client.post("/v1/systemone", json=GOOD).status_code == 429
        for _ in range(5):
            assert client.get("/v1/stats").status_code == 200
            assert client.get("/health").status_code == 200
            assert client.get("/v1/classifiers").status_code == 200
            assert client.put("/v1/classifiers/x", json=cdef).status_code == 200
        assert client.post("/v1/classifiers/x", json={"input": "a"}).status_code == 429
        assert client.post("/v1/classifiers/x/batch", json={"inputs": ["a"]}).status_code == 429


def test_rate_limit_is_per_key():
    client, _, _ = make(rate_limit=1, api_keys_raw=KEYS)
    with client:
        a, b = {"X-API-Key": "key-web"}, {"X-API-Key": "key-batch"}
        assert client.post("/v1/systemone", json=GOOD, headers=a).status_code == 200
        assert client.post("/v1/systemone", json=GOOD, headers=a).status_code == 429
        assert client.post("/v1/systemone", json=GOOD, headers=b).status_code == 200
        assert client.post("/v1/systemone", json=GOOD, headers=b).status_code == 429
        # unauthenticated requests never consume a key's budget
        assert client.post("/v1/systemone", json=GOOD).status_code == 401


def test_rate_limit_off_by_default(api):
    client, _, _ = api
    assert all(client.post("/v1/systemone", json=GOOD).status_code == 200 for _ in range(10))


# ------------------------------------------------------------------ CORS / bind warning


def test_cors_off_and_on():
    client, _, _ = make()
    with client:
        r = client.get("/livez", headers={"Origin": "https://a.example"})
        assert "access-control-allow-origin" not in r.headers
    client, _, _ = make(cors_origins=("https://a.example",), api_key="k")
    with client:
        r = client.options(
            "/v1/classify",
            headers={
                "Origin": "https://a.example",
                "Access-Control-Request-Method": "POST",
                "Access-Control-Request-Headers": "x-api-key",
            },
        )
        assert r.status_code == 200 and r.headers["access-control-allow-origin"] == "https://a.example"
        r = client.get("/v1/stats", headers={"Origin": "https://a.example"})
        assert r.status_code == 401 and r.headers["access-control-allow-origin"] == "https://a.example"
        assert "X-Request-ID" in r.headers["access-control-expose-headers"]


def test_run_warns_when_exposed_without_key(monkeypatch, caplog):
    import clef_server.main as main_mod

    monkeypatch.setattr(main_mod.uvicorn, "run", lambda *a, **k: None)
    monkeypatch.setattr(main_mod, "create_app", lambda cfg: None)
    with caplog.at_level("WARNING", logger="clef"):
        main_mod.run(Config(host="0.0.0.0"))
    assert any("NO API key" in r.getMessage() for r in caplog.records)
    caplog.clear()
    with caplog.at_level("WARNING", logger="clef"):
        main_mod.run(Config(host="0.0.0.0", api_key="k"))
        main_mod.run(Config(host="127.0.0.1"))
    assert not caplog.records


# ------------------------------------------------------------------ stats / time series / events


def test_sampler_runs_and_feeds_stats():
    eng = FakeEngine()
    client, _, _ = make(eng, sample_interval_s=0.2)
    with client:
        deadline = time.time() + 5
        while eng.samples < 2 and time.time() < deadline:
            time.sleep(0.05)
        ts = client.get("/v1/stats/timeseries?window_s=10&step_s=1").json()
    assert eng.samples >= 2
    assert any(v == 1.0 for v in ts["mem_used_gb"]) and any(v == 5.0 for v in ts["gpu_util_pct"])
    assert any(v is not None for v in ts["queue_depth"])


def test_sampler_tolerates_engine_without_sample():
    class Bare(FakeEngine):
        sample = None  # type: ignore[assignment]

    client, _, _ = make(Bare(), sample_interval_s=0.2)
    with client:
        time.sleep(0.5)
        assert client.get("/livez").status_code == 200


def test_timeseries_shape_and_validation(api):
    client, _, _ = api
    client.post("/v1/systemone", json=GOOD)
    r = client.get("/v1/stats/timeseries?window_s=300&step_s=5")
    assert r.status_code == 200, r.text
    ts = r.json()
    assert ts["window_s"] == 300 and ts["step_s"] == 5
    cols = ("t", "rps", "p50_ms", "p95_ms", "error_rate", "avg_batch", "queue_depth", "padding_ratio")
    for col in (*cols, "mem_used_gb", "gpu_util_pct", "gpu_temp_c", "gpu_power_w"):
        assert len(ts[col]) == 60, col
    assert sum(ts["rps"]) > 0
    assert client.get("/v1/stats/timeseries").json()["step_s"] == 5  # default step
    for q in (
        "window_s=5",
        "window_s=3601",
        "window_s=abc",
        "window_s=300&step_s=0",
        "window_s=300&step_s=301",
        "window_s=3600&step_s=1",
        "window_s=300&step_s=-2",
    ):
        r = client.get(f"/v1/stats/timeseries?{q}")
        assert r.status_code == 400 and r.json()["detail"] and r.json()["request_id"], q


def test_stats_body_has_new_fields(api):
    client, _, _ = api
    s = client.get("/v1/stats").json()
    assert s["status"] == "ready" and s["gpu"]["memory_kind"] == "vram"
    assert s["telemetry"]["source"] == "nvml"
    h = client.get("/health").json()
    assert h["backend"] == "cuda" and h["telemetry"]["available"] is True and h["version"].startswith("3.")


def test_stats_by_key_and_endpoint():
    client, _, _ = make(api_keys_raw=KEYS)
    hdr = {"X-API-Key": "key-web"}
    with client:
        client.post("/v1/systemone", json=GOOD, headers=hdr)
        client.post("/v1/classify", json=CLS, headers=hdr)
        s = client.get("/v1/stats", headers=hdr).json()
    assert s["by_key"] == {"web": 2}
    assert s["by_endpoint"] == {"/v1/systemone": 1, "/v1/classify": 1}


def test_sse_point_in_stats_events():
    async def main() -> list[str]:
        _, _, app = make()
        stats = app.state.stats
        calls = 0

        async def disconnected() -> bool:
            nonlocal calls
            calls += 1
            return calls > 2

        def status() -> dict[str, Any]:
            return {**stats.snapshot(), "point": stats.latest_point(5)}

        return [c async for c in sse_events(stats, status, disconnected, interval=0.1)]

    chunks = asyncio.run(main())
    first = json.loads(chunks[0].split("data: ", 1)[1])
    assert first["point"]["step_s"] == 5 and first["point"]["t"] % 5 == 0 and "rps" in first["point"]


def test_events_route_includes_point_and_validates_step(monkeypatch):
    import clef_server.main as main_mod

    captured = {}

    async def fake_sse(stats, status_fn, is_disconnected, interval=2.0):
        captured["body"] = status_fn()
        yield "event: stats\ndata: {}\n\n"

    monkeypatch.setattr(main_mod, "sse_events", fake_sse)
    client, _, _ = make()
    with client:
        r = client.get("/v1/events?step_s=15")
        assert r.status_code == 200 and "event: stats" in r.text
        assert captured["body"]["point"]["step_s"] == 15 and captured["body"]["status"] == "ready"
        assert client.get("/v1/events?step_s=0").status_code == 400
        assert client.get("/v1/events?step_s=61").status_code == 400
