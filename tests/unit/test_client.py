import base64
import json

import httpx
import pytest

from clef_client import (
    AsyncClefClient,
    ClefClient,
    ClefError,
    image_to_data_url,
    video_to_data_url,
)

Q = {"outage": {"type": "noul", "instructions": "Is a service down?"}}


def _handler(seen: list[httpx.Request]):
    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        path = request.url.path
        if path == "/v1/systemone":
            body = json.loads(request.content)
            if body["state"] == "bad":
                return httpx.Response(
                    400,
                    json={"detail": "questions: required", "request_id": "abc"},
                    headers={"X-Request-ID": "abc"},
                )
            return httpx.Response(200, json={"model": body["model"], "answers": {}, "echo": body})
        if path == "/v1/batch":
            return httpx.Response(
                200, json={"batch_ms": 1.0, "results": json.loads(request.content)["batch"]}
            )
        if path == "/health":
            return httpx.Response(200, json={"status": "ready"})
        if path == "/v1/stats":
            return httpx.Response(401, text="nope")
        return httpx.Response(404, json={"detail": "not found"})

    return handler


def test_decide_builds_request_and_sends_key():
    seen: list[httpx.Request] = []
    c = ClefClient(api_key="k", transport=httpx.MockTransport(_handler(seen)))
    out = c.decide("hi", Q, images=["data:image/png;base64,AA=="])
    assert out["echo"]["model"] == "clef-flash"
    assert out["echo"]["images"] == ["data:image/png;base64,AA=="]
    assert "videos" not in out["echo"]
    assert seen[0].headers["x-api-key"] == "k"


def test_error_carries_status_detail_request_id():
    c = ClefClient(transport=httpx.MockTransport(_handler([])))
    with pytest.raises(ClefError) as ei:
        c.decide("bad", Q)
    assert ei.value.status == 400
    assert ei.value.detail == "questions: required"
    assert ei.value.request_id == "abc"


def test_non_json_error_body():
    c = ClefClient(transport=httpx.MockTransport(_handler([])))
    with pytest.raises(ClefError) as ei:
        c.stats()
    assert ei.value.status == 401 and ei.value.detail == "nope"


def test_batch_and_health():
    c = ClefClient(transport=httpx.MockTransport(_handler([])))
    recs = [{"model": "clef-flash", "state": "a", "questions": Q}]
    assert c.batch(recs)["results"] == recs
    assert c.health()["status"] == "ready"


def test_transport_error_wrapped():
    def boom(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused")

    c = ClefClient(transport=httpx.MockTransport(boom))
    with pytest.raises(ClefError) as ei:
        c.health()
    assert ei.value.status is None


@pytest.mark.asyncio
async def test_async_client():
    seen: list[httpx.Request] = []
    async with AsyncClefClient(api_key="k", transport=httpx.MockTransport(_handler(seen))) as c:
        out = await c.decide("hi", Q, videos=["data:video/mp4;base64,AA=="])
        assert out["echo"]["videos"] == ["data:video/mp4;base64,AA=="]
        assert (await c.health())["status"] == "ready"
        with pytest.raises(ClefError):
            await c.decide("bad", Q)


def test_image_helpers(tmp_path):
    from PIL import Image

    img = Image.new("RGB", (4, 4), (255, 0, 0))
    url = image_to_data_url(img)
    assert url.startswith("data:image/png;base64,")
    raw = base64.b64decode(url.split(",", 1)[1])
    assert raw[:4] == b"\x89PNG"
    assert image_to_data_url(raw).startswith("data:image/png;base64,")
    p = tmp_path / "x.jpg"
    img.save(p)
    assert image_to_data_url(p).startswith("data:image/jpeg;base64,")
    v = tmp_path / "v.mp4"
    v.write_bytes(b"\x00\x01")
    assert video_to_data_url(v) == "data:video/mp4;base64,AAE="
    with pytest.raises(TypeError):
        image_to_data_url(123)
