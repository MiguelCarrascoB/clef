"""Synchronous client."""

from __future__ import annotations

import time
from typing import Any
from urllib.parse import quote

import httpx

from . import _common as c
from .types import Classification, ScoreResult, parse_classification, parse_result, parse_score

DEFAULT_MODEL = "clef-flash"


class ClefClient:
    """Sync client. Retries 503, 429 (Retry-After) and connect errors."""

    def __init__(
        self,
        base_url: str | None = None,
        api_key: str | None = None,
        timeout: float | httpx.Timeout | None = None,
        retries: int = 3,
        backoff: float = 0.5,
        transport: httpx.BaseTransport | None = None,
    ):
        url, key = c.resolve_config(base_url, api_key)
        self.retries = max(0, retries)
        self.backoff = backoff
        self._http = httpx.Client(
            base_url=url,
            headers=c.headers(key),
            timeout=c.DEFAULT_TIMEOUT if timeout is None else timeout,
            transport=transport,
        )

    def _request(self, method: str, path: str, **kw: Any) -> dict[str, Any]:
        attempt = 0
        while True:
            try:
                resp = self._http.request(method, path, **kw)
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
            time.sleep(c.retry_delay(err, attempt, self.backoff))
            attempt += 1

    # -- v2 API --
    def decide(
        self,
        state: Any,
        questions: dict[str, Any],
        images: list[str] | None = None,
        videos: list[str] | None = None,
        model: str = DEFAULT_MODEL,
    ) -> dict[str, Any]:
        return self._request("POST", "/v1/systemone", json=c.record(state, questions, images, videos, model))

    def batch(self, records: list[dict[str, Any]]) -> dict[str, Any]:
        return self._request("POST", "/v1/batch", json={"batch": records})

    def health(self) -> dict[str, Any]:
        return self._request("GET", "/health")

    def stats(self) -> dict[str, Any]:
        return self._request("GET", "/v1/stats")

    # -- classification --
    def classify(
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
        return parse_classification(self._request("POST", "/v1/classify", json=body))

    def classify_many(
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
            out.extend(c.parse_batch(self._request("POST", "/v1/classify/batch", json=body)))
        return out

    def score(
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
        return parse_score(self._request("POST", "/v1/score", json=body))

    # -- saved classifiers --
    def classifier(self, name: str) -> Classifier:
        return Classifier(self, name)

    def list_classifiers(self) -> list[dict[str, Any]]:
        return self._request("GET", "/v1/classifiers").get("classifiers", [])

    def save_classifier(
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
        return self._request("PUT", f"/v1/classifiers/{quote(name, safe='')}", json=body)

    def get_classifier(self, name: str) -> dict[str, Any]:
        return self._request("GET", f"/v1/classifiers/{quote(name, safe='')}")

    def delete_classifier(self, name: str) -> dict[str, Any]:
        return self._request("DELETE", f"/v1/classifiers/{quote(name, safe='')}")

    def close(self) -> None:
        self._http.close()

    def __enter__(self) -> ClefClient:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


class Classifier:
    """Handle to a saved classifier on the server."""

    def __init__(self, client: ClefClient, name: str):
        self.client = client
        self.name = name
        self._path = f"/v1/classifiers/{quote(name, safe='')}"

    def classify(
        self,
        input: Any,
        *,
        images: list[str] | None = None,
        videos: list[str] | None = None,
        threshold: float | None = None,
    ) -> Classification | ScoreResult:
        body = c.classifier_call_body(input, images, videos, threshold)
        return parse_result(self.client._request("POST", self._path, json=body))

    def classify_many(self, inputs: list[Any], *, chunk_size: int | None = None) -> list[Any]:
        out: list[Any] = []
        for part in c.chunks(inputs, chunk_size):
            body = self.client._request("POST", self._path + "/batch", json={"inputs": part})
            out.extend(c.parse_batch(body, self.name))
        return out

    def save(self, **kw: Any) -> dict[str, Any]:
        return self.client.save_classifier(self.name, **kw)

    def get(self) -> dict[str, Any]:
        return self.client.get_classifier(self.name)

    def delete(self) -> dict[str, Any]:
        return self.client.delete_classifier(self.name)
