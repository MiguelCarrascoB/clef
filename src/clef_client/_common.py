"""Shared, transport-free helpers: media encoding, request building, error parsing, retry policy."""

from __future__ import annotations

import base64
import dataclasses
import io
import mimetypes
import os
import random
from pathlib import Path
from typing import Any
from urllib.parse import quote

import httpx

from .types import Classification, ScoreResult, parse_result

DEFAULT_URL = "http://127.0.0.1:8910"
DEFAULT_TIMEOUT = httpx.Timeout(600.0, connect=5.0)  # cold media forwards can take a while
MAX_RETRY_AFTER = 60.0


class ClefError(Exception):
    """Non-2xx response (or transport failure, with status=None) from the clef server."""

    def __init__(
        self,
        status: int | None,
        detail: str,
        request_id: str | None = None,
        retry_after: float | None = None,
    ):
        self.status = status
        self.detail = detail
        self.request_id = request_id
        self.retry_after = retry_after
        super().__init__(f"[{status}] {detail}" + (f" (request_id={request_id})" if request_id else ""))


def _data_url(raw: bytes, mime: str) -> str:
    return f"data:{mime};base64,{base64.b64encode(raw).decode()}"


def image_to_data_url(image: str | Path | bytes | Any, fmt: str = "PNG") -> str:
    """Encode a path, raw bytes or PIL image as a data URL."""
    if isinstance(image, (str, Path)):
        path = Path(image)
        mime = mimetypes.guess_type(path.name)[0] or "image/png"
        return _data_url(path.read_bytes(), mime)
    if isinstance(image, (bytes, bytearray)):
        raw = bytes(image)
        mime = (
            "image/png" if raw[:4] == b"\x89PNG" else "image/jpeg" if raw[:2] == b"\xff\xd8" else "image/png"
        )
        return _data_url(raw, mime)
    if hasattr(image, "save"):  # PIL.Image.Image
        buf = io.BytesIO()
        image.save(buf, format=fmt)
        return _data_url(buf.getvalue(), f"image/{fmt.lower()}")
    raise TypeError(f"unsupported image type: {type(image).__name__}")


def video_to_data_url(path: str | Path) -> str:
    p = Path(path)
    return _data_url(p.read_bytes(), mimetypes.guess_type(p.name)[0] or "video/mp4")


def resolve_config(base_url: str | None, api_key: str | None) -> tuple[str, str | None]:
    url = base_url or os.environ.get("CLEF_URL") or DEFAULT_URL
    return url.rstrip("/"), api_key or os.environ.get("CLEF_API_KEY") or None


def headers(api_key: str | None) -> dict[str, str]:
    return {"X-API-Key": api_key} if api_key else {}


def record(
    state: Any, questions: dict[str, Any], images: list[str] | None, videos: list[str] | None, model: str
) -> dict[str, Any]:
    rec: dict[str, Any] = {"model": model, "state": state, "questions": questions}
    if images:
        rec["images"] = list(images)
    if videos:
        rec["videos"] = list(videos)
    return rec


def _parse_retry_after(value: str | None) -> float | None:
    try:
        return max(0.0, float(value)) if value is not None else None
    except ValueError:
        return None


def error_from(resp: httpx.Response) -> ClefError:
    request_id = resp.headers.get("X-Request-ID")
    try:
        body = resp.json()
        detail = body.get("detail", resp.text) if isinstance(body, dict) else resp.text
        request_id = (body.get("request_id") if isinstance(body, dict) else None) or request_id
    except ValueError:
        detail = resp.text
    if not isinstance(detail, str):
        detail = str(detail)
    return ClefError(
        resp.status_code, detail, request_id, _parse_retry_after(resp.headers.get("Retry-After"))
    )


def retry_delay(err: ClefError, attempt: int, backoff: float) -> float:
    """Seconds before retry number ``attempt`` (0-based): Retry-After if sent, else exp backoff + jitter."""
    if err.retry_after is not None:
        return min(err.retry_after, MAX_RETRY_AFTER)
    return backoff * (2**attempt) * (0.5 + random.random() / 2)


def should_retry(status: int | None) -> bool:
    """503 (loading / OOM), 429, and connect failures (status None). Never other 4xx."""
    return status in (None, 429, 503)


def _clean(d: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in d.items() if v is not None and v != []}


def classify_body(
    input: Any,
    labels: Any,
    instructions: str | None,
    multi_label: bool,
    threshold: float | None,
    images: list[str] | None,
    videos: list[str] | None,
    model: str | None,
) -> dict[str, Any]:
    return _clean(
        {
            "input": input,
            "labels": labels,
            "instructions": instructions,
            "multi_label": True if multi_label else None,
            "threshold": threshold,
            "images": images,
            "videos": videos,
            "model": model,
        }
    )


def classify_many_body(
    inputs: list[Any],
    labels: Any,
    instructions: str | None,
    multi_label: bool,
    threshold: float | None,
    model: str | None,
) -> dict[str, Any]:
    body = _clean(
        {
            "labels": labels,
            "instructions": instructions,
            "multi_label": True if multi_label else None,
            "threshold": threshold,
            "model": model,
        }
    )
    body["inputs"] = list(inputs)
    return body


def score_body(
    input: Any,
    levels: list[str],
    instructions: str | None,
    images: list[str] | None,
    videos: list[str] | None,
    model: str | None,
) -> dict[str, Any]:
    return _clean(
        {
            "input": input,
            "levels": levels,
            "instructions": instructions,
            "images": images,
            "videos": videos,
            "model": model,
        }
    )


def chunks(items: list[Any], size: int | None) -> list[list[Any]]:
    items = list(items)
    if not size or size <= 0 or len(items) <= size:
        return [items]
    return [items[i : i + size] for i in range(0, len(items), size)]


def parse_batch(body: dict[str, Any], classifier: str | None = None) -> list[Any]:
    rid = body.get("request_id")
    out: list[Classification | ScoreResult] = []
    for r in body.get("results", []):
        r = dict(r)
        if classifier and "classifier" not in r:
            r["classifier"] = classifier
        res = parse_result(r)
        if rid and res.request_id is None:
            res = dataclasses.replace(res, request_id=rid)
        out.append(res)
    return out


def classifier_def(
    kind: str | None,
    labels: Any,
    levels: list[str] | None,
    instructions: str | None,
    multi_label: bool | None,
    threshold: float | None,
    description: str | None,
) -> dict[str, Any]:
    if kind is None:
        kind = "score" if levels is not None and labels is None else "classify"
    return _clean(
        {
            "kind": kind,
            "labels": labels,
            "levels": levels,
            "instructions": instructions,
            "multi_label": multi_label,
            "threshold": threshold,
            "description": description,
        }
    )


def classifier_call_body(input: Any, images: Any, videos: Any, threshold: float | None) -> dict[str, Any]:
    return _clean({"input": input, "images": images, "videos": videos, "threshold": threshold})


# -- async jobs --
def job_body(
    kind: str, payload: dict[str, Any], webhook: str | dict[str, Any] | None, metadata: dict[str, Any] | None
) -> dict[str, Any]:
    """``webhook`` is a URL or ``{"url", "secret", "events"}``."""
    hook = {"url": webhook} if isinstance(webhook, str) else webhook
    return _clean({"kind": kind, "payload": payload, "webhook": hook, "metadata": metadata})


def classify_job_payload(
    inputs: list[Any],
    labels: Any,
    classifier: str | None,
    instructions: str | None,
    multi_label: bool,
    threshold: float | None,
    model: str | None,
) -> dict[str, Any]:
    payload = _clean(
        {
            "labels": labels,
            "classifier": classifier,
            "instructions": instructions,
            "multi_label": True if multi_label else None,
            "threshold": threshold,
            "model": model,
        }
    )
    payload["inputs"] = list(inputs)
    return payload


def job_path(job_id: str, suffix: str = "") -> str:
    return f"/v1/jobs/{quote(job_id, safe='')}{suffix}"


def job_list_params(status: str | None, kind: str | None, limit: int, offset: int) -> dict[str, Any]:
    return _clean({"status": status, "kind": kind, "limit": limit, "offset": offset or None})


def stream_error(resp: httpx.Response) -> ClefError:
    resp.read()
    return error_from(resp)
