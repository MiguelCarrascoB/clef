"""Asynchronous client (mirror of ``_sync``)."""

from __future__ import annotations

import asyncio
from typing import Any
from urllib.parse import quote

import httpx

from . import _common as c
from .types import Classification, ScoreResult, parse_classification, parse_result, parse_score

DEFAULT_MODEL = "clef-flash"


class AsyncClefClient:
    """Async client. Retries 503, 429 (Retry-After) and connect errors."""

    def __init__(
        self,
        base_url: str | None = None,
        api_key: str | None = None,
        timeout: float | httpx.Timeout | None = None,
        retries: int = 3,
        backoff: float = 0.5,
        transport: httpx.AsyncBaseTransport | None = None,
    ):
        url, key = c.resolve_config(base_url, api_key)
        self.retries = max(0, retries)
        self.backoff = backoff
        self._http = httpx.AsyncClient(
            base_url=url,
            headers=c.headers(key),
            timeout=c.DEFAULT_TIMEOUT if timeout is None else timeout,
            transport=transport,
        )

    async def _request(self, method: str, path: str, **kw: Any) -> dict[str, Any]:
        attempt = 0
        while True:
            try:
                resp = await self._http.request(method, path, **kw)
                if resp.is_success:
                    return resp.json()
                err = c.error_from(resp)
            except (httpx.ConnectError, httpx.ConnectTimeout) as exc:
                err = c.ClefError(None, f"transport error: {exc!r}")
                err.__cause__ = exc
            except httpx.TransportError as exc:
                raise c.ClefError(None, f"transport error: {exc!r}") from exc
            if attempt >= self.retries or not c.should_retry(err.status):
                raise err
            await asyncio.sleep(c.retry_delay(err, attempt, self.backoff))
            attempt += 1

    # -- v2 API --
    async def decide(
        self,
        state: Any,
        questions: dict[str, Any],
        images: list[str] | None = None,
        videos: list[str] | None = None,
        model: str = DEFAULT_MODEL,
    ) -> dict[str, Any]:
        return await self._request(
            "POST", "/v1/systemone", json=c.record(state, questions, images, videos, model)
        )

    async def batch(self, records: list[dict[str, Any]]) -> dict[str, Any]:
        return await self._request("POST", "/v1/batch", json={"batch": records})

    async def health(self) -> dict[str, Any]:
        return await self._request("GET", "/health")

    async def stats(self) -> dict[str, Any]:
        return await self._request("GET", "/v1/stats")

    # -- classification --
    async def classify(
        self,
        input: Any,
        labels: list[str] | dict[str, str],
        *,
        instructions: str | None = None,
        multi_label: bool = False,
        threshold: float | None = None,
        images: list[str] | None = None,
        videos: list[str] | None = None,
        model: str | None = DEFAULT_MODEL,
    ) -> Classification:
        body = c.classify_body(input, labels, instructions, multi_label, threshold, images, videos, model)
        return parse_classification(await self._request("POST", "/v1/classify", json=body))

    async def classify_many(
        self,
        inputs: list[Any],
        labels: list[str] | dict[str, str],
        *,
        instructions: str | None = None,
        multi_label: bool = False,
        threshold: float | None = None,
        model: str | None = DEFAULT_MODEL,
        chunk_size: int | None = None,
    ) -> list[Classification]:
        out: list[Classification] = []
        for part in c.chunks(inputs, chunk_size):
            body = c.classify_many_body(part, labels, instructions, multi_label, threshold, model)
            out.extend(c.parse_batch(await self._request("POST", "/v1/classify/batch", json=body)))
        return out

    async def score(
        self,
        input: Any,
        levels: list[str],
        *,
        instructions: str | None = None,
        images: list[str] | None = None,
        videos: list[str] | None = None,
        model: str | None = DEFAULT_MODEL,
    ) -> ScoreResult:
        body = c.score_body(input, levels, instructions, images, videos, model)
        return parse_score(await self._request("POST", "/v1/score", json=body))

    # -- saved classifiers --
    def classifier(self, name: str) -> AsyncClassifier:
        return AsyncClassifier(self, name)

    async def list_classifiers(self) -> list[dict[str, Any]]:
        return (await self._request("GET", "/v1/classifiers")).get("classifiers", [])

    async def save_classifier(
        self,
        name: str,
        *,
        labels: Any = None,
        levels: list[str] | None = None,
        kind: str | None = None,
        instructions: str | None = None,
        multi_label: bool | None = None,
        threshold: float | None = None,
        description: str | None = None,
    ) -> dict[str, Any]:
        body = c.classifier_def(kind, labels, levels, instructions, multi_label, threshold, description)
        return await self._request("PUT", f"/v1/classifiers/{quote(name, safe='')}", json=body)

    async def get_classifier(self, name: str) -> dict[str, Any]:
        return await self._request("GET", f"/v1/classifiers/{quote(name, safe='')}")

    async def delete_classifier(self, name: str) -> dict[str, Any]:
        return await self._request("DELETE", f"/v1/classifiers/{quote(name, safe='')}")

    async def aclose(self) -> None:
        await self._http.aclose()

    async def __aenter__(self) -> AsyncClefClient:
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.aclose()


class AsyncClassifier:
    """Handle to a saved classifier on the server (async)."""

    def __init__(self, client: AsyncClefClient, name: str):
        self.client = client
        self.name = name
        self._path = f"/v1/classifiers/{quote(name, safe='')}"

    async def classify(
        self,
        input: Any,
        *,
        images: list[str] | None = None,
        videos: list[str] | None = None,
        threshold: float | None = None,
    ) -> Classification | ScoreResult:
        body = c.classifier_call_body(input, images, videos, threshold)
        return parse_result(await self.client._request("POST", self._path, json=body))

    async def classify_many(self, inputs: list[Any], *, chunk_size: int | None = None) -> list[Any]:
        out: list[Any] = []
        for part in c.chunks(inputs, chunk_size):
            body = await self.client._request("POST", self._path + "/batch", json={"inputs": part})
            out.extend(c.parse_batch(body, self.name))
        return out

    async def save(self, **kw: Any) -> dict[str, Any]:
        return await self.client.save_classifier(self.name, **kw)

    async def get(self) -> dict[str, Any]:
        return await self.client.get_classifier(self.name)

    async def delete(self) -> dict[str, Any]:
        return await self.client.delete_classifier(self.name)
