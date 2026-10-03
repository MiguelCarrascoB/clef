"""Asynchronous client (mirror of ``_sync``)."""

from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncIterator, Callable
from pathlib import Path
from typing import Any
from urllib.parse import quote

import httpx

from . import _common as c
from .types import (
    Classification,
    Job,
    JobItem,
    ScoreResult,
    parse_classification,
    parse_job,
    parse_job_item,
    parse_result,
    parse_score,
)

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

    # -- async jobs (large batches; see docs/jobs.md) --
    async def submit_job(
        self,
        kind: str,
        payload: dict[str, Any],
        *,
        webhook: str | dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> Job:
        """Queue a job and return at once. ``webhook``: a URL, or ``{"url", "secret", "events"}``."""
        body = c.job_body(kind, payload, webhook, metadata)
        return parse_job(await self._request("POST", "/v1/jobs", json=body))

    async def classify_job(
        self,
        inputs: list[Any],
        labels: list[str] | dict[str, str] | None = None,
        *,
        classifier: str | None = None,
        instructions: str | None = None,
        multi_label: bool = False,
        threshold: float | None = None,
        model: str | None = DEFAULT_MODEL,
        webhook: str | dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> Job:
        """``classify_many`` without holding a connection open: labels or a saved ``classifier`` name."""
        payload = c.classify_job_payload(
            inputs, labels, classifier, instructions, multi_label, threshold, model
        )
        return await self.submit_job("classify", payload, webhook=webhook, metadata=metadata)

    async def job(self, job_id: str) -> Job:
        return parse_job(await self._request("GET", c.job_path(job_id)))

    async def jobs(
        self, *, status: str | None = None, kind: str | None = None, limit: int = 50, offset: int = 0
    ) -> list[Job]:
        """One page of jobs, newest first."""
        params = c.job_list_params(status, kind, limit, offset)
        body = await self._request("GET", "/v1/jobs", params=params)
        return [parse_job(j) for j in body.get("jobs", [])]

    async def cancel_job(self, job_id: str) -> Job:
        return parse_job(await self._request("POST", c.job_path(job_id, "/cancel")))

    async def delete_job(self, job_id: str) -> dict[str, Any]:
        return await self._request("DELETE", c.job_path(job_id))

    async def wait_job(
        self,
        job_id: str,
        timeout: float | None = None,
        poll: float = 1.0,
        on_progress: Callable[[Job], None] | None = None,
    ) -> Job:
        """Poll until succeeded / failed / cancelled (check ``job.ok``); TimeoutError after ``timeout`` s."""
        deadline = None if timeout is None else time.monotonic() + timeout
        while True:
            job = await self.job(job_id)
            if on_progress:
                on_progress(job)
            if job.finished:
                return job
            if deadline is not None and time.monotonic() + poll > deadline:
                raise TimeoutError(
                    f"job {job_id} still {job.status} after {timeout} s ({job.done}/{job.total})"
                )
            await asyncio.sleep(poll)

    async def job_results(
        self, job_id: str, *, page_size: int = 500, offset: int = 0
    ) -> AsyncIterator[JobItem]:
        """Every result row currently stored, in input order, fetched page by page."""
        while True:
            page = await self._request(
                "GET", c.job_path(job_id, "/results"), params={"offset": offset, "limit": page_size}
            )
            for row in page.get("items", []):
                yield parse_job_item(row, page.get("kind", ""))
            offset = page.get("next_offset")
            if offset is None or not page.get("items"):
                return

    async def save_job_results(self, job_id: str, path: str | Path, format: str = "ndjson") -> Path:
        """Stream the whole result (``ndjson`` or ``csv``) to a file without loading it in memory."""
        dest = Path(path)
        async with self._http.stream(
            "GET", c.job_path(job_id, "/results"), params={"format": format}
        ) as resp:
            if not resp.is_success:
                await resp.aread()
                raise c.error_from(resp)
            with dest.open("wb") as fh:
                async for chunk in resp.aiter_bytes():
                    fh.write(chunk)
        return dest

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
