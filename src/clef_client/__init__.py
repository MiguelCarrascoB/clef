"""Small sync/async Python client for the local clef-flash server.

from clef_client import ClefClient
c = ClefClient()
out = c.decide("Checkout is down", {"outage": {"type": "noul", "instructions": "Is a service down?"}})
out["answers"]["outage"]["noul"]
"""

from __future__ import annotations

import base64
import io
import mimetypes
from pathlib import Path
from typing import Any

import httpx

__all__ = [
    "AsyncClefClient",
    "ClefClient",
    "ClefError",
    "image_to_data_url",
    "video_to_data_url",
]

DEFAULT_URL = "http://127.0.0.1:8910"
DEFAULT_TIMEOUT = httpx.Timeout(600.0, connect=5.0)  # cold media forwards can take a while


class ClefError(Exception):
    """Non-2xx response (or transport failure, with status=None) from the clef server."""

    def __init__(self, status: int | None, detail: str, request_id: str | None = None):
        self.status = status
        self.detail = detail
        self.request_id = request_id
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


def _record(
    state: Any, questions: dict[str, Any], images: list[str] | None, videos: list[str] | None, model: str
) -> dict[str, Any]:
    rec: dict[str, Any] = {"model": model, "state": state, "questions": questions}
    if images:
        rec["images"] = list(images)
    if videos:
        rec["videos"] = list(videos)
    return rec


def _headers(api_key: str | None) -> dict[str, str]:
    return {"X-API-Key": api_key} if api_key else {}


def _raise_for(resp: httpx.Response) -> dict[str, Any]:
    request_id = resp.headers.get("X-Request-ID")
    if resp.is_success:
        return resp.json()
    try:
        body = resp.json()
        detail = body.get("detail", resp.text) if isinstance(body, dict) else resp.text
        request_id = (body.get("request_id") if isinstance(body, dict) else None) or request_id
    except ValueError:
        detail = resp.text
    if not isinstance(detail, str):
        detail = str(detail)
    raise ClefError(resp.status_code, detail, request_id)


class ClefClient:
    def __init__(
        self,
        base_url: str = DEFAULT_URL,
        api_key: str | None = None,
        timeout: float | httpx.Timeout | None = None,
        transport: httpx.BaseTransport | None = None,
    ):
        self._http = httpx.Client(
            base_url=base_url.rstrip("/"),
            headers=_headers(api_key),
            timeout=DEFAULT_TIMEOUT if timeout is None else timeout,
            transport=transport,
        )

    def _request(self, method: str, path: str, **kw: Any) -> dict[str, Any]:
        try:
            return _raise_for(self._http.request(method, path, **kw))
        except httpx.TransportError as exc:
            raise ClefError(None, f"transport error: {exc!r}") from exc

    def decide(
        self,
        state: Any,
        questions: dict[str, Any],
        images: list[str] | None = None,
        videos: list[str] | None = None,
        model: str = "clef-flash",
    ) -> dict[str, Any]:
        return self._request("POST", "/v1/systemone", json=_record(state, questions, images, videos, model))

    def batch(self, records: list[dict[str, Any]]) -> dict[str, Any]:
        return self._request("POST", "/v1/batch", json={"batch": records})

    def health(self) -> dict[str, Any]:
        return self._request("GET", "/health")

    def stats(self) -> dict[str, Any]:
        return self._request("GET", "/v1/stats")

    def close(self) -> None:
        self._http.close()

    def __enter__(self) -> ClefClient:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


class AsyncClefClient:
    def __init__(
        self,
        base_url: str = DEFAULT_URL,
        api_key: str | None = None,
        timeout: float | httpx.Timeout | None = None,
        transport: httpx.AsyncBaseTransport | None = None,
    ):
        self._http = httpx.AsyncClient(
            base_url=base_url.rstrip("/"),
            headers=_headers(api_key),
            timeout=DEFAULT_TIMEOUT if timeout is None else timeout,
            transport=transport,
        )

    async def _request(self, method: str, path: str, **kw: Any) -> dict[str, Any]:
        try:
            return _raise_for(await self._http.request(method, path, **kw))
        except httpx.TransportError as exc:
            raise ClefError(None, f"transport error: {exc!r}") from exc

    async def decide(
        self,
        state: Any,
        questions: dict[str, Any],
        images: list[str] | None = None,
        videos: list[str] | None = None,
        model: str = "clef-flash",
    ) -> dict[str, Any]:
        return await self._request(
            "POST", "/v1/systemone", json=_record(state, questions, images, videos, model)
        )

    async def batch(self, records: list[dict[str, Any]]) -> dict[str, Any]:
        return await self._request("POST", "/v1/batch", json={"batch": records})

    async def health(self) -> dict[str, Any]:
        return await self._request("GET", "/health")

    async def stats(self) -> dict[str, Any]:
        return await self._request("GET", "/v1/stats")

    async def aclose(self) -> None:
        await self._http.aclose()

    async def __aenter__(self) -> AsyncClefClient:
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.aclose()
