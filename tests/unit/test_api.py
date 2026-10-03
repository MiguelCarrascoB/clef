"""HTTP API tests with a FakeEngine (no GPU, no model)."""

from __future__ import annotations

import asyncio
import base64
import io
import json
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

    def start(self) -> None:
        self.started = True

    def shutdown(self) -> None:
        self.stopped = True

    def info(self) -> dict[str, Any]:
        return {
            "device": "cuda",
            "dtype": "bfloat16",
            "model_path": "/m",
            "torch": "x",
            "transformers": "y",
            "gpu": {
                "available": True,
                "name": "GPU",
                "vram_total_gb": 24.0,
                "vram_allocated_gb": 1.0,
                "vram_reserved_gb": 2.0,
            },
        }

    async def decide(self, records: list[dict[str, Any]]) -> list[dict[str, Any]]:
        self.calls.append(records)
        if self.raises:
            raise self.raises
        return [
            {
                "model": r["model"],
                "answers": {q: {"type": "noul", "noul": 0.5} for q in r["questions"]},
                "usage": {"input_tokens": 10, "output_tokens": 0},
                "timing": {"queue_ms": 1.0, "forward_ms": 2.0, "batch_size": len(records)},
            }
            for r in records
        ]


def make(engine: FakeEngine | None = None, **cfg_kw: Any):
    eng = engine or FakeEngine()
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
