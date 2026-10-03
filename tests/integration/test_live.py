"""Integration tests against a LIVE clef server (needs the GPU). Run: pytest tests/integration -m gpu

Env: CLEF_URL (default http://127.0.0.1:8910), CLEF_API_KEY (if the server requires auth).
All tests are skipped when the server is unreachable.
"""

from __future__ import annotations

import io
import os

import httpx
import pytest

from clef_client import image_to_data_url

pytestmark = pytest.mark.gpu

BASE = os.environ.get("CLEF_URL", "http://127.0.0.1:8910").rstrip("/")
KEY = os.environ.get("CLEF_API_KEY")

TICKET = "Our checkout started returning errors and orders are blocked."
TEXT_Q = {
    "department": {
        "type": "choice",
        "instructions": "Which team should handle the message?",
        "criteria": {"billing": "Payments or invoices", "technical": "Bugs or outages"},
    },
    "urgency": {"type": "score", "criteria": ["Can wait", "This week", "Today"]},
    "outage": {"type": "noul", "instructions": "Is a service down?"},
}


@pytest.fixture(scope="module")
def http():
    headers = {"X-API-Key": KEY} if KEY else {}
    with httpx.Client(base_url=BASE, headers=headers, timeout=600) as c:
        try:
            c.get("/livez", timeout=3).raise_for_status()
            status = c.get("/health", timeout=5).json().get("status")
        except (httpx.HTTPError, ValueError):
            pytest.skip(f"no clef server at {BASE}")
        if status not in ("ready", "warming"):
            pytest.skip(f"server not ready: {status}")
        yield c


def decide(http, **body):
    body.setdefault("model", "clef-flash")
    return http.post("/v1/systemone", json=body)


def solid(color, size=224):
    from PIL import Image

    return Image.new("RGB", (size, size), color)


def test_health_fields(http):
    h = http.get("/health").json()
    for key in ("ready", "status", "version", "device", "dtype", "model_path", "torch", "gpu", "limits"):
        assert key in h, key
    assert h["status"] in ("ready", "warming")
    assert h["gpu"]["available"] is True
    assert h["version"].startswith("2.")


def test_text_decision_shape(http):
    r = decide(http, state=TICKET, questions=TEXT_Q)
    assert r.status_code == 200, r.text
    out = r.json()
    assert r.headers.get("x-request-id")
    assert out["usage"]["input_tokens"] > 0 and out["usage"]["output_tokens"] == 0
    a = out["answers"]
    assert set(a) == set(TEXT_Q)
    for qid in ("department", "urgency"):
        probs = a[qid]["probabilities"]
        assert sum(probs.values()) == pytest.approx(1.0, abs=0.01), (qid, probs)
    assert a["department"]["type"] == "choice"
    assert a["department"]["choice"] == "technical"  # README example
    assert 0.0 <= a["urgency"]["score"] <= 2.0
    assert 0.0 <= a["outage"]["noul"] <= 1.0
    assert a["outage"]["noul"] > 0.5
    t = out["timing"]
    assert t["forward_ms"] > 0 and t["batch_size"] >= 1


def test_batch_matches_single(http):
    states = [
        TICKET,
        "Customers are double-charged on invoices since the payment update.",
        "Wordmark in the footer is 2px off-center.",
    ]
    questions = {k: TEXT_Q[k] for k in ("department", "outage")}
    singles = [decide(http, state=s, questions=questions).json() for s in states]
    r = http.post(
        "/v1/batch",
        json={"batch": [{"model": "clef-flash", "state": s, "questions": questions} for s in states]},
    )
    assert r.status_code == 200, r.text
    results = r.json()["results"]
    assert len(results) == len(states)
    for single, batched in zip(singles, results, strict=False):
        for qid in questions:
            s, b = single["answers"][qid], batched["answers"][qid]
            if "probabilities" in s:
                for opt, p in s["probabilities"].items():
                    assert b["probabilities"][opt] == pytest.approx(p, abs=2e-2)
            else:
                assert b["noul"] == pytest.approx(s["noul"], abs=2e-2)


@pytest.mark.parametrize(
    "body",
    [
        {"state": "x"},  # no questions
        {"state": "x", "questions": {}},  # empty questions
        {"state": "x", "questions": {"q": {"type": "bogus"}}},  # bad type
        {"state": "x", "questions": {"q": {"type": "choice", "criteria": {}}}},  # empty criteria
        {"state": "x", "questions": {"q": {"type": "score", "criteria": "abc"}}},  # wrong criteria shape
        {
            "state": "x",
            "questions": {"q": {"type": "noul"}},
            "media_kwargs": {"max_pixels": 1},
        },  # not settable
        {"state": "x", "questions": {"q": {"type": "noul"}}, "images": ["not-a-data-url"]},
    ],
)
def test_validation_400(http, body):
    r = decide(http, **body)
    assert r.status_code in (400, 422), r.text
    j = r.json()
    assert isinstance(j["detail"], str) and j.get("request_id")


def test_batch_validation_names_index(http):
    good = {"model": "clef-flash", "state": "x", "questions": {"q": {"type": "noul"}}}
    bad = {"model": "clef-flash", "state": "x", "questions": {"q": {"type": "score", "criteria": []}}}
    r = http.post("/v1/batch", json={"batch": [good, bad]})
    assert r.status_code in (400, 422)
    assert "batch[1]" in r.json()["detail"]


def test_image_red_square(http):
    r = decide(
        http,
        state="A picture is attached.",
        images=[image_to_data_url(solid("red"))],
        questions={"red": {"type": "noul", "instructions": "Is the image mostly red?"}},
    )
    assert r.status_code == 200, r.text
    assert r.json()["answers"]["red"]["noul"] > 0.5


def test_multi_image_not_sharded(http):
    """v1 split media across records and answered from the last shard only. Only the FIRST image is red."""
    imgs = [solid("red")] + [solid("blue") for _ in range(4)]
    r = decide(
        http,
        state="Five images are attached.",
        images=[image_to_data_url(i) for i in imgs],
        questions={"red": {"type": "noul", "instructions": "Does any image show a red square?"}},
    )
    assert r.status_code == 200, r.text
    assert r.json()["answers"]["red"]["noul"] > 0.5


def test_video_red_ball(http):
    np = pytest.importorskip("numpy")
    iio = pytest.importorskip("imageio.v3")
    frames = []
    for i in range(8):  # red block moving across a dark background (same as the old test_video.sh)
        img = np.zeros((320, 320, 3), dtype=np.uint8)
        img[:, :, 0] = 20
        x = 20 + i * 35
        img[130:190, x : x + 60, 0] = 255
        frames.append(img)
    buf = io.BytesIO()
    try:
        iio.imwrite(buf, np.stack(frames), extension=".mp4", fps=4)  # 8 frames @ 4 fps = 2 s
    except Exception as exc:  # no ffmpeg backend
        pytest.skip(f"cannot encode mp4: {exc!r}")
    import base64

    url = "data:video/mp4;base64," + base64.b64encode(buf.getvalue()).decode()
    r = decide(
        http,
        state="A short video clip is attached.",
        videos=[url],
        questions={
            "has_ball": {"type": "noul", "instructions": "Does a moving red object appear in the video?"},
            "ball_color": {
                "type": "choice",
                "instructions": "Which color is the moving object?",
                "criteria": {"red": "Red", "green": "Green", "blue": "Blue"},
            },
        },
    )
    assert r.status_code == 200, r.text
    a = r.json()["answers"]
    assert a["has_ball"]["noul"] > 0.5
    assert a["ball_color"]["choice"] == "red"
