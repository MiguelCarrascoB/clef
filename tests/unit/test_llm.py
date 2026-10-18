"""The OpenAI-compatible LLM client over a mocked httpx transport (no network)."""

from __future__ import annotations

import json
import logging

import httpx
import pytest

from clef_server.config import DEFAULT_LLM_BASE_URL, DEFAULT_LLM_MODEL, Config
from clef_server.llm import (
    LLMError,
    OpenAICompatClient,
    client_from_config,
    parse_json_object,
    validate_schema,
)

SCHEMA = {
    "type": "object",
    "properties": {"labels": {"type": "array", "items": {"type": "string"}, "maxItems": 3}},
    "required": ["labels"],
    "additionalProperties": False,
}


def reply(content, status=200, headers=None, usage=True):
    body = {"model": "m-served", "choices": [{"message": {"role": "assistant", "content": content}}]}
    if usage:
        body["usage"] = {"prompt_tokens": 11, "completion_tokens": 7}
    return httpx.Response(status, json=body, headers=headers)


def make(handler, **kw):
    http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    return OpenAICompatClient("https://llm.example/v1", "sk-secret-123", "m", http=http, backoff_s=0, **kw)


async def call(client):
    return await client.complete_json(system="sys", user="usr", schema=SCHEMA, name="out")


@pytest.mark.asyncio
async def test_success_request_shape_and_usage():
    seen = []

    def handler(req):
        seen.append(req)
        return reply('{"labels": ["a", "b"]}')

    res = await call(make(handler))
    assert res.data == {"labels": ["a", "b"]} and res.model == "m-served"
    assert (res.input_tokens, res.output_tokens) == (11, 7) and res.latency_ms >= 0
    req = seen[0]
    assert str(req.url) == "https://llm.example/v1/chat/completions"
    assert req.headers["authorization"] == "Bearer sk-secret-123"
    body = json.loads(req.content)
    assert body["model"] == "m" and body["max_tokens"] == 2048
    assert [m["role"] for m in body["messages"]] == ["system", "user"]
    assert body["response_format"]["json_schema"] == {"name": "out", "schema": SCHEMA, "strict": True}


@pytest.mark.asyncio
async def test_retries_429_then_succeeds_honouring_retry_after(monkeypatch):
    sleeps = []

    async def fake_sleep(s):
        sleeps.append(s)

    monkeypatch.setattr("clef_server.llm.asyncio.sleep", fake_sleep)
    answers = [reply("", 429, {"Retry-After": "3"}), reply("", 503), reply('{"labels": []}')]
    res = await call(make(lambda req: answers.pop(0)))
    assert res.data == {"labels": []} and not answers
    assert sleeps == [3.0, 0.0]  # Retry-After first, then the (zero) test backoff


@pytest.mark.asyncio
async def test_gives_up_after_bounded_retries():
    n = []

    def handler(req):
        n.append(1)
        return reply("", 500)

    with pytest.raises(LLMError, match="after 3 attempts"):
        await call(make(handler))
    assert len(n) == 3


@pytest.mark.asyncio
async def test_client_error_is_not_retried_and_never_leaks_the_key():
    n = []

    def handler(req):
        n.append(1)
        return httpx.Response(401, text="bad key")

    client = make(handler)
    with pytest.raises(LLMError, match="HTTP 401") as exc:
        await call(client)
    assert len(n) == 1 and "sk-secret-123" not in str(exc.value)
    assert "sk-secret-123" not in repr(client)


@pytest.mark.asyncio
async def test_timeout_and_transport_errors_retry():
    calls = []

    def handler(req):
        calls.append(1)
        if len(calls) == 1:
            raise httpx.ReadTimeout("slow", request=req)
        if len(calls) == 2:
            raise httpx.ConnectError("down", request=req)
        return reply('{"labels": ["x"]}')

    assert (await call(make(handler))).data == {"labels": ["x"]}
    assert len(calls) == 3


@pytest.mark.asyncio
async def test_tolerant_parse_of_fenced_or_chatty_json():
    for content in ('```json\n{"labels": ["a"]}\n```', 'Sure! Here you go: {"labels": ["a"]} Hope it helps.'):
        assert (await call(make(lambda req, c=content: reply(c)))).data == {"labels": ["a"]}


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "content",
    ["not json at all", '["a"]', '{"labels": "a"}', '{"labels": ["a", "b", "c", "d"]}', '{"other": 1}', None],
)
async def test_malformed_or_schema_invalid_json_is_an_llm_error(content):
    with pytest.raises(LLMError):
        await call(make(lambda req: reply(content)))


@pytest.mark.asyncio
async def test_missing_choices_is_an_llm_error():
    with pytest.raises(LLMError, match="no message content"):
        await call(make(lambda req: httpx.Response(200, json={"unexpected": True})))


@pytest.mark.asyncio
async def test_missing_usage_is_none():
    res = await call(make(lambda req: reply('{"labels": []}', usage=False)))
    assert res.input_tokens is None and res.output_tokens is None


def test_parse_json_object_and_validate_schema_edges():
    assert parse_json_object(' {"a": 1} ') == {"a": 1}
    assert parse_json_object('Sure: {"a": 1} done') == {"a": 1}  # prose around one object
    assert parse_json_object('{"a": 1}\nDone.') == {"a": 1}  # prose after it (ollama, LM Studio)
    assert parse_json_object('```json\n{"a": 1}\n```\nHope this helps') == {"a": 1}
    for cut in ('{"proposals": [{"name": "x"}, {"na', '```json\n{"proposals": [{"name": "x"}, {"na'):
        with pytest.raises(LLMError, match="cut off"):  # never an inner object of a truncated reply
            parse_json_object(cut)
    with pytest.raises(LLMError):
        validate_schema({"labels": [1]}, SCHEMA)
    with pytest.raises(LLMError):
        validate_schema({"labels": [], "extra": 1}, SCHEMA)
    validate_schema({"labels": ["a"]}, SCHEMA)
    with pytest.raises(LLMError):
        validate_schema(True, {"type": "integer"})


def test_client_from_config_is_off_without_a_key(monkeypatch):
    assert client_from_config(Config()) is None
    monkeypatch.setenv("GEMINI_API_KEY", "g-key")
    client = client_from_config(Config())
    assert client.base_url == DEFAULT_LLM_BASE_URL and client.model == DEFAULT_LLM_MODEL
    assert "g-key" not in repr(client) and "g-key" not in repr(Config())
    monkeypatch.setenv("CLEF_LLM_API_KEY", "ollama")
    monkeypatch.setenv("CLEF_LLM_BASE_URL", "http://localhost:11434/v1")
    monkeypatch.setenv("CLEF_LLM_MODEL", "llama3")
    cfg = Config()
    assert cfg.llm_api_key == "ollama"  # CLEF_LLM_API_KEY wins over GEMINI_API_KEY
    client = client_from_config(cfg)
    assert client.base_url == "http://localhost:11434/v1/" and client.model == "llama3"


def test_key_is_not_logged_on_retry(caplog, monkeypatch):
    import asyncio

    async def nosleep(s):
        return None

    monkeypatch.setattr("clef_server.llm.asyncio.sleep", nosleep)
    answers = [reply("", 500), reply('{"labels": []}')]
    with caplog.at_level(logging.DEBUG):
        asyncio.run(call(make(lambda req: answers.pop(0))))
    assert "sk-secret-123" not in caplog.text


@pytest.mark.asyncio
async def test_a_reply_cut_off_at_max_tokens_is_an_error():
    def handler(req):
        r = reply('{"labels": ["a"')
        body = json.loads(r.content)
        body["choices"][0]["finish_reason"] = "length"
        return httpx.Response(200, json=body)

    with pytest.raises(LLMError, match="max_tokens=2048"):
        await call(make(handler))
