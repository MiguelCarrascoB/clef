"""One OpenAI-compatible chat client over httpx (no SDK), for the router's label proposals.

It speaks ``POST {base_url}chat/completions`` with a JSON-schema ``response_format``, so Gemini's OpenAI
endpoint (the default), ollama, LM Studio and OpenAI all work by changing the URL, model and key. The key
travels only in the ``Authorization`` header: it is never logged, never put in an error and not in ``repr``.
See docs/ARCHITECTURE.md "Review queue and router" and docs/router.md.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import time
from dataclasses import dataclass
from typing import Any, Protocol
from urllib.parse import urlsplit

import httpx

from .config import Config

log = logging.getLogger("clef")

RETRY_STATUS = frozenset({429, 500, 502, 503, 504})
MAX_RETRY_AFTER_S = 30.0
_FENCE = re.compile(r"^\s*```(?:json)?\s*(.*?)\s*```\s*$", re.DOTALL)


class LLMError(RuntimeError):
    """Transport, HTTP status, timeout or a response that is not valid JSON for the schema."""


@dataclass(frozen=True)
class LLMResult:
    data: dict[str, Any]  # the parsed JSON, already validated against the schema
    model: str
    input_tokens: int | None
    output_tokens: int | None
    latency_ms: float


class LLMClient(Protocol):
    base_url: str
    model: str

    async def complete_json(
        self, *, system: str, user: str, schema: dict[str, Any], name: str, max_tokens: int = 2048
    ) -> LLMResult: ...


# ------------------------------------------------------------------ response parsing


def parse_json_object(text: str) -> dict[str, Any]:
    """The JSON object in a model's reply: the whole text, a fenced block, or else the first object in it."""
    text = text.strip()
    candidates = [text]
    fenced = _FENCE.match(text)
    if fenced:
        candidates.append(fenced.group(1))
    for cand in candidates:
        try:
            value = json.loads(cand)
        except ValueError:
            continue
        if isinstance(value, dict):
            return value
    decoder = json.JSONDecoder()
    for m in re.finditer(r"\{", text):  # tolerant parse: prose before / after the object
        try:
            value, _ = decoder.raw_decode(text, m.start())
        except ValueError:
            continue
        if isinstance(value, dict):
            return value
    raise LLMError("the response is not a JSON object")


_TYPES: dict[str, Any] = {
    "object": dict,
    "array": list,
    "string": str,
    "integer": int,
    "number": (int, float),
    "boolean": bool,
}


def validate_schema(value: Any, schema: dict[str, Any], path: str = "$") -> None:
    """The small JSON-schema subset the router uses (type, properties, required, additionalProperties, items,
    minItems, maxItems, enum); raises ``LLMError`` naming the first violation."""
    want = schema.get("type")
    if want is not None:
        ok = isinstance(value, _TYPES[want]) and not (
            want in ("integer", "number") and isinstance(value, bool)
        )
        if not ok:
            raise LLMError(f"{path}: expected {want}")
    if "enum" in schema and value not in schema["enum"]:
        raise LLMError(f"{path}: not one of {schema['enum']}")
    if isinstance(value, dict):
        props = schema.get("properties", {})
        for key in schema.get("required", []):
            if key not in value:
                raise LLMError(f"{path}.{key}: missing")
        for key, sub in value.items():
            if key in props:
                validate_schema(sub, props[key], f"{path}.{key}")
            elif schema.get("additionalProperties") is False:
                raise LLMError(f"{path}.{key}: unexpected property")
    elif isinstance(value, list):
        if len(value) < schema.get("minItems", 0) or len(value) > schema.get("maxItems", 1 << 30):
            raise LLMError(f"{path}: wrong number of items ({len(value)})")
        for i, sub in enumerate(value):
            if "items" in schema:
                validate_schema(sub, schema["items"], f"{path}[{i}]")


# ------------------------------------------------------------------ the client


class OpenAICompatClient:
    """``complete_json`` with 2 retries (429 / 5xx / transport errors, honouring Retry-After) and a tolerant
    JSON parse. Pass ``http`` to supply the httpx client (tests use a MockTransport)."""

    def __init__(
        self,
        base_url: str,
        api_key: str,
        model: str,
        timeout_s: float = 60.0,
        http: httpx.AsyncClient | None = None,
        retries: int = 2,
        backoff_s: float = 1.0,
    ) -> None:
        self.base_url = base_url.rstrip("/") + "/"
        self.model = model
        self.retries, self.backoff_s = retries, backoff_s
        self._key = api_key
        self._http = http or httpx.AsyncClient(timeout=timeout_s)
        self._own_http = http is None

    def __repr__(self) -> str:  # the key never shows up in a log line or a traceback
        return f"OpenAICompatClient(base_url={self.base_url!r}, model={self.model!r})"

    @property
    def host(self) -> str:
        return urlsplit(self.base_url).netloc

    async def aclose(self) -> None:
        if self._own_http:
            await self._http.aclose()

    def _delay(self, attempt: int, resp: httpx.Response | None) -> float:
        retry_after = resp.headers.get("Retry-After") if resp is not None else None
        try:
            if retry_after is not None:
                return min(max(float(retry_after), 0.0), MAX_RETRY_AFTER_S)
        except ValueError:
            pass
        return min(self.backoff_s * 2**attempt, MAX_RETRY_AFTER_S)

    async def complete_json(
        self, *, system: str, user: str, schema: dict[str, Any], name: str, max_tokens: int = 2048
    ) -> LLMResult:
        body = {
            "model": self.model,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            "response_format": {
                "type": "json_schema",
                "json_schema": {"name": name, "schema": schema, "strict": True},
            },
            "max_tokens": max_tokens,
        }
        headers = {"Authorization": f"Bearer {self._key}", "Content-Type": "application/json"}
        t0 = time.perf_counter()
        resp: httpx.Response | None = None
        for attempt in range(self.retries + 1):
            resp = None
            try:
                resp = await self._http.post(self.base_url + "chat/completions", json=body, headers=headers)
            except httpx.TimeoutException as exc:
                failure = f"timeout ({type(exc).__name__})"
            except httpx.TransportError as exc:
                failure = f"transport error ({type(exc).__name__})"
            else:
                if resp.status_code == 200:
                    break
                failure = f"HTTP {resp.status_code}: {resp.text[:200]!r}"
                if resp.status_code not in RETRY_STATUS:
                    raise LLMError(f"LLM request failed: {failure}")
            if attempt >= self.retries:
                raise LLMError(f"LLM request failed after {attempt + 1} attempts: {failure}")
            log.warning("llm: %s; retrying (%d/%d)", failure, attempt + 1, self.retries)
            await asyncio.sleep(self._delay(attempt, resp))
        assert resp is not None
        try:
            payload = resp.json()
            content = payload["choices"][0]["message"]["content"]
        except (ValueError, KeyError, IndexError, TypeError) as exc:
            raise LLMError("LLM response has no message content") from exc
        if not isinstance(content, str):
            raise LLMError("LLM response has no message content")
        data = parse_json_object(content)
        validate_schema(data, schema)
        usage = payload.get("usage") if isinstance(payload, dict) else None
        usage = usage if isinstance(usage, dict) else {}
        return LLMResult(
            data=data,
            model=str(payload.get("model") or self.model),
            input_tokens=_int_or_none(usage.get("prompt_tokens")),
            output_tokens=_int_or_none(usage.get("completion_tokens")),
            latency_ms=round((time.perf_counter() - t0) * 1000, 1),
        )


def _int_or_none(v: Any) -> int | None:
    return v if isinstance(v, int) and not isinstance(v, bool) else None


def client_from_config(cfg: Config) -> LLMClient | None:
    """None (the router's LLM is off) unless CLEF_LLM_API_KEY or GEMINI_API_KEY is set in the process env."""
    if not cfg.llm_api_key:
        return None
    return OpenAICompatClient(cfg.llm_base_url, cfg.llm_api_key, cfg.llm_model, cfg.llm_timeout_s)


__all__ = [
    "LLMClient",
    "LLMError",
    "LLMResult",
    "OpenAICompatClient",
    "client_from_config",
    "parse_json_object",
    "validate_schema",
]
