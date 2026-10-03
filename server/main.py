"""Clef-flash server - SystemOne decision API served locally on AMD ROCm (RX 7900 XTX).

Endpoints:
  POST /v1/systemone    Jev / SystemOne decision API: {model, state, questions, images?, videos?}
                        -> {model, answers, usage}. Every question returns calibrated probabilities.
  POST /v1/batch        {"batch": [<systemone request>, ...]} -> batched one-forward-pass decisions
  GET  /health          Model + GPU status (503 until the model is loaded)
  GET  /schema-example  Copy-paste ready request bodies

State can be any text or JSON. Images/videos are data: URLs or http(s) URLs.
Run:  source ~/venvs/clef/bin/activate && export HSA_ENABLE_DXG_DETECTION=1 && python server.py
"""
from __future__ import annotations

import base64
import io
import os
import sys
import time
import traceback
import urllib.request
from pathlib import Path
from typing import Any

import numpy as np
import torch
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from PIL import Image

MODEL_PATH = os.environ.get("CLEF_MODEL_PATH", str(Path.home() / "models" / "clef-flash"))
DEVICE = os.environ.get("CLEF_DEVICE", "cuda")  # "cuda" (ROCm GPU) or "cpu"
DTYPE = getattr(torch, os.environ.get("CLEF_DTYPE", "bfloat16"), torch.bfloat16)
PORT = int(os.environ.get("CLEF_PORT", "8910"))
MAX_SHARD_IMAGES = 4  # images per record in a request (vision tokens grow fast)

app = FastAPI(title="clef-flash local", version="1.0")
S: dict[str, Any] = {"ready": False, "error": None}


def engine() -> None:
    if "model" in S:
        return
    if S["error"]:
        raise RuntimeError(S["error"])
    t0 = time.time()
    sys.path.insert(0, MODEL_PATH)
    from joint_schema_model import (  # type: ignore
        collate_records,
        encode_record,
        load_release_model,
        systemone,
        systemone_answer,
    )

    model, processor = load_release_model(MODEL_PATH, device=DEVICE, dtype=DTYPE)
    model.eval()
    S.update(
        model=model, processor=processor, encode=encode_record, collate=collate_records,
        systemone=systemone, answer_of=systemone_answer, load_seconds=round(time.time() - t0, 1),
    )
    S["ready"] = True


@app.on_event("startup")
def startup() -> None:
    try:
        engine()
    except Exception as exc:
        traceback.print_exc()
        S["error"] = f"{exc}"


def _image(data_or_url: str) -> Image.Image:
    if data_or_url.startswith("http"):
        with urllib.request.urlopen(data_or_url, timeout=30) as res:  # nosec: user-supplied demo server
            payload = res.read()
    else:
        payload = base64.b64decode(data_or_url.split(",", 1)[-1])
    return Image.open(io.BytesIO(payload)).convert("RGB")


def _frames(data_or_url: str, fps: float = 2.0) -> np.ndarray:
    """Base64/URL video (mp4, webm, mov...) -> sampled frame array (T, H, W, 3) uint8."""
    import imageio.v3 as iio

    if data_or_url.startswith("http"):
        with urllib.request.urlopen(data_or_url, timeout=30) as res:
            payload = res.read()
    else:
        payload = base64.b64decode(data_or_url.split(",", 1)[-1])
    buf = io.BytesIO(payload)
    frames = iio.imread(buf, extension=".mp4", fps=fps)  # type: ignore[arg-type]
    if frames.ndim != 4 or frames.shape[-1] != 3:
        raise ValueError(f"unexpected frame shape {frames.shape}")
    return np.asarray(frames, dtype=np.uint8)


def _split_record(request: dict[str, Any]) -> list[dict[str, Any]]:
    """One record per <=MAX_SHARD_IMAGES images + one per video (mixed-media batching is valid,
    but shard big media sets so each record's vision-token budget stays bounded)."""
    records: list[dict[str, Any]] = []
    images = request.get("images") or []
    videos = request.get("videos") or []
    chunks = [images[i:i + MAX_SHARD_IMAGES] for i in range(0, len(images), MAX_SHARD_IMAGES)] or [[]]
    for chunk in chunks:
        record = {k: v for k, v in request.items() if k not in ("images", "videos")}
        record["images"] = [_image(x) for x in chunk] or None
        records.append(record)
    for video in videos:
        record = {k: v for k, v in request.items() if k not in ("images", "videos")}
        record["images"] = None
        record["videos"] = [_frames(video)]
        records.append(record)
    if not records:
        raise ValueError("request carries no images or videos")
    return records


def _sanitize(request: dict[str, Any]) -> dict[str, Any]:
    """Accept both raw PIL-capable records and a SystemOne body; validate fields."""
    if not isinstance(request, dict):
        raise ValueError("request must be an object")
    request = dict(request)
    if "state" not in request:
        raise ValueError("model and state are required")
    questions = request.get("questions")
    if not isinstance(questions, dict) or not questions:
        raise ValueError("at least one question is required")
    for qid, question in questions.items():
        if not isinstance(question, dict) or question.get("type") not in ("noul", "choice", "score"):
            raise ValueError(f"{qid}: type must be noul, choice, or score")
        if question.get("type") != "noul" and not question.get("criteria"):
            raise ValueError(f"{qid}: criteria must not be empty")
    return request


def _systemone_text(request: dict[str, Any]) -> dict[str, Any]:
    return S["systemone"](S["model"], S["processor"], request)


def _systemone_media(request: dict[str, Any]) -> dict[str, Any]:
    answers: dict[str, Any] = {}
    usage = {"input_tokens": 0, "output_tokens": 0}
    for part in _split_record(request):
        part.setdefault("model", request.get("model", "clef-flash"))
        response = S["systemone"](S["model"], S["processor"], part)
        answers.update(response["answers"])
        usage["input_tokens"] += response["usage"]["input_tokens"]
    return {"model": request.get("model", "clef-flash"), "answers": answers, "usage": usage}


def _handle(request: dict[str, Any]) -> dict[str, Any]:
    request = _sanitize(request)
    if request.get("images") or request.get("videos"):
        return _systemone_media(request)
    return _systemone_text(request)


@app.post("/v1/systemone")
def v1_systemone(request: dict[str, Any]) -> JSONResponse:
    if not S["ready"]:
        raise HTTPException(503, S["error"] or "model loading")
    try:
        return JSONResponse(_handle(request))
    except (ValueError, KeyError) as exc:
        raise HTTPException(400, str(exc))
    except Exception as exc:
        raise HTTPException(500, f"{exc}\n{traceback.format_exc()}")


@app.post("/v1/batch")
def v1_batch(payload: dict[str, Any]) -> JSONResponse:
    if not S["ready"]:
        raise HTTPException(503, S["error"] or "model loading")
    try:
        records = payload.get("batch")
        if not isinstance(records, list) or not records:
            raise ValueError("payload must be {\"batch\": [<request>, ...]}")
        requests = [_sanitize(r) for r in records]
        if any(r.get("images") or r.get("videos") for r in requests):
            raise ValueError("/v1/batch is text/JSON only; use /v1/systemone per media record")
        tokenizer = S["processor"].tokenizer
        encoded = [S["encode"](tokenizer, r, processor=S["processor"]) for r in requests]
        device = next(S["model"].parameters()).device
        batch = S["collate"](encoded, tokenizer.pad_token_id, device)
        t0 = time.perf_counter()
        with torch.inference_mode():
            logits_batch = S["model"](batch)
        torch.cuda.synchronize() if DEVICE == "cuda" else None
        elapsed_ms = (time.perf_counter() - t0) * 1000
        out = []
        for request, enc, logits in zip(requests, encoded, logits_batch):
            answers = {
                q.question_id: S["answer_of"](
                    request["questions"][q.question_id],
                    dict(zip(q.option_ids, ql.float().softmax(-1).tolist())),
                )
                for q, ql in zip(enc.questions, logits)
            }
            out.append({
                "model": request.get("model", "clef-flash"),
                "answers": answers,
                "usage": {"input_tokens": len(enc.input_ids), "output_tokens": 0},
            })
        return JSONResponse({
            "batch_ms": round(elapsed_ms, 1),
            "results": out,
        })
    except (ValueError, KeyError) as exc:
        raise HTTPException(400, str(exc))
    except Exception as exc:
        raise HTTPException(500, f"{exc}\n{traceback.format_exc()}")


@app.get("/health")
def health() -> JSONResponse:
    gpu = {"available": torch.cuda.is_available(), "device": None, "vram_total_gb": None}
    if gpu["available"]:
        prop = torch.cuda.get_device_properties(0)
        gpu["device"] = prop.name
        gpu["vram_total_gb"] = round(prop.total_memory / 1e9, 1)
        gpu["vram_allocated_gb"] = round(torch.cuda.memory_allocated() / 1e9, 1)
    return JSONResponse({
        "ready": S["ready"], "device": DEVICE, "gpu": gpu,
        "load_seconds": S.get("load_seconds"), "error": S["error"],
    }, status_code=200 if S["ready"] else 503)


@app.get("/schema-example")
def schema_example() -> dict[str, Any]:
    return {
        "text_json": {
            "model": "clef-flash",
            "state": {"ticket": {"text": "Checkout errors, orders blocked.", "customers_affected": 1200}},
            "questions": {
                "department": {
                    "type": "choice", "instructions": "Which team should handle the message?",
                    "criteria": {"billing": "Payments or invoices", "technical": "Bugs or outages"}},
                "urgency": {"type": "score", "criteria": ["Can wait", "This week", "Today"]},
                "outage": {"type": "noul", "instructions": "Is a service down?"},
            },
        },
        "media": {
            "model": "clef-flash",
            "state": "Review the attached receipt.",
            "images": ["data:image/jpeg;base64,<base64 bytes of the image>"],
            "videos": ["data:video/mp4;base64,<base64 bytes of the video>"],
            "questions": {"legible": {"type": "noul", "instructions": "Is the total legible?"}},
        },
        "batch": {"batch": ["<2..N systemone request objects>"]},
        "note": "clef returns CALIBRATED PROBABILITIES for every option in one forward pass - no text generation.",
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=PORT)
