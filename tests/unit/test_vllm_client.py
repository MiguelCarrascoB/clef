# ruff: noqa: E501
"""bench/vllm/vllm_client.py maps clef records to vLLM's /v1/systemone and back (no GPU, no server)."""

from __future__ import annotations

import copy
import importlib.util
import sys
from pathlib import Path

import pytest

_DIR = Path(__file__).resolve().parents[2] / "bench" / "vllm"
spec = importlib.util.spec_from_file_location("vllm_client", _DIR / "vllm_client.py")
vc = importlib.util.module_from_spec(spec)
sys.modules["vllm_client"] = vc
spec.loader.exec_module(vc)

QUESTIONS = {
    "team": {
        "type": "choice",
        "instructions": "Which team?",
        "criteria": {"billing": "pay", "shipping": "mail"},
    },
    "urgent": {"type": "noul", "instructions": "Within the hour?"},
    "tone": {"type": "score", "criteria": ["calm", "annoyed", "furious"]},
}
# Shape of the response in vLLM's docs/serving/online_serving/structured_decisions.md (PR #61308)
RESPONSE = {
    "id": "decision-1",
    "object": "structured_decision",
    "model": "clef-flash",
    "answers": {
        "team": {
            "type": "choice",
            "choice": "billing",
            "probabilities": {"billing": 0.97, "shipping": 0.03},
            "confidence": 0.9,
        },
        "urgent": {
            "type": "noul",
            "noul": 0.02,
            "probabilities": {"yes": 0.02, "no": 0.98},
            "confidence": 0.95,
        },
        "tone": {
            "type": "score",
            "score": 0.6,
            "legend": {"0": "calm", "1": "annoyed", "2": "furious"},
            "probabilities": {"0": 0.7, "1": 0.2, "2": 0.1},
            "confidence": 0.65,
        },
    },
    "usage": {"input_tokens": 142, "output_tokens": 0},
}


def test_to_request_drops_id_and_unset_fields():
    rec = {"id": "r1", "state": {"t": "x"}, "questions": QUESTIONS}
    body = vc.to_request(rec)
    assert body["model"] == "clef-flash" and "id" not in body and "instructions" not in body
    assert body["questions"]["urgent"] == {
        "type": "noul",
        "instructions": "Within the hour?",
    }  # no criteria key
    assert body["questions"]["tone"] == {"type": "score", "criteria": ["calm", "annoyed", "furious"]}
    assert "images" not in body


def test_to_request_rejects_fields_vllm_would_400():
    rec = {"state": "s", "questions": {"q": {"type": "noul", "extra": 1}}}
    with pytest.raises(vc.ShapeError, match="extra"):
        vc.to_request(rec)


def test_to_request_images_become_data_urls():
    pil = pytest.importorskip("PIL.Image")
    rec = {"state": "s", "questions": QUESTIONS, "images": [pil.new("RGB", (4, 4)), "https://x/y.png"]}
    imgs = vc.to_request(rec)["images"]
    assert imgs[0].startswith("data:image/png;base64,") and imgs[1] == "https://x/y.png"


def test_flatten_maps_to_parity_keys():
    flat = vc.flatten_answers(RESPONSE, QUESTIONS)
    assert flat["team.billing"] == 0.97
    assert flat["tone.2"] == 0.1  # score levels by index
    assert flat["urgent.true"] == 0.02 and flat["urgent.false"] == pytest.approx(0.98)  # not yes/no
    assert set(flat) == {
        "team.billing",
        "team.shipping",
        "urgent.true",
        "urgent.false",
        "tone.0",
        "tone.1",
        "tone.2",
    }


def test_flatten_accepts_clefs_own_shape():
    resp = {"model": "clef-flash", "answers": copy.deepcopy(RESPONSE["answers"])}
    del resp["answers"]["urgent"]["probabilities"]  # the release answers noul with `noul` only
    assert vc.flatten_answers(resp, QUESTIONS)["urgent.true"] == 0.02


@pytest.mark.parametrize(
    ("mutate", "match"),
    [
        (lambda r: r.pop("answers"), "no `answers`"),
        (lambda r: r["answers"].pop("tone"), "answered"),
        (lambda r: r["answers"]["team"].update(type="score"), "expected an answer of type"),
        (lambda r: r["answers"]["team"]["probabilities"].update(other=0.0), "probabilities keys"),
        (lambda r: r["answers"]["tone"]["probabilities"].pop("2"), "probabilities keys"),
        (
            lambda r: r["answers"]["tone"].update(probabilities={"a": 0.5, "b": 0.3, "c": 0.2}),
            "probabilities keys",
        ),
        (lambda r: r["answers"]["team"]["probabilities"].update(billing=0.5), "sum to"),
        (lambda r: r["answers"]["team"].update(choice="nope"), "choice"),
        (lambda r: r["answers"]["urgent"].update(noul=None), "not a probability"),
        (lambda r: r["answers"]["urgent"].update(noul=float("nan")), "not a probability"),
        (lambda r: r["answers"]["team"]["probabilities"].update(billing="0.97"), "not a probability"),
    ],
)
def test_shape_mismatch_fails_loudly(mutate, match):
    resp = copy.deepcopy(RESPONSE)
    mutate(resp)
    with pytest.raises(vc.ShapeError, match=match):
        vc.flatten_answers(resp, QUESTIONS)


def test_check_response_is_a_value_error_for_httpbench():
    body = {"questions": QUESTIONS}
    vc.check_response(body, RESPONSE)
    with pytest.raises(ValueError):
        vc.check_response(body, {"answers": {}})


def test_httpbench_counts_shape_errors():
    """httpbench.run_level(check=...) turns a ShapeError into an error, never a silent success."""
    import asyncio

    import httpx

    from clef_server import httpbench

    def handler(request):
        return httpx.Response(200, json={"answers": {}})

    async def go():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler), base_url="http://t") as c:
            return await httpbench.run_level(c, 2, 4, False, vc.check_response)

    r = asyncio.run(go())
    assert r["ok"] == 0 and sum(r["errors"].values()) == 4
    assert next(iter(r["errors"])).startswith("shape:")
