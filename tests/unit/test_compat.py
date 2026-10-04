"""OpenAI / Hugging Face compat endpoints with a FakeEngine (no GPU).

FakeEngine: choice/score put weight i+1 on option i (the LAST option wins); noul answers
`eng.noul[qid]` (default 0.5).
"""

from __future__ import annotations

import json
import math
import socket
import threading
import time
from typing import Any

import pytest

from clef_server import compat_schema as cs
from tests.unit.test_api import make, png_url

TEAM = {"type": "string", "enum": ["billing", "technical", "account"], "description": "Who handles it?"}
SCHEMA = {
    "type": "object",
    "properties": {
        "team": TEAM,
        "outage": {"type": "boolean", "description": "Is a service down?"},
        "urgency": {"type": "integer", "enum": [1, 2, 3]},
    },
    "required": ["team", "outage", "urgency"],
    "additionalProperties": False,
}


def rf(schema: dict[str, Any] = SCHEMA) -> dict[str, Any]:
    return {"type": "json_schema", "json_schema": {"name": "triage", "schema": schema, "strict": True}}


def chat(**kw: Any) -> dict[str, Any]:
    body: dict[str, Any] = {
        "model": "clef-flash",
        "messages": [{"role": "user", "content": "Checkout is down"}],
        "response_format": rf(),
    }
    body.update(kw)
    return body


@pytest.fixture
def api():
    client, eng, app = make()
    with client:
        yield client, eng, app


def post(client, body, **kw):
    return client.post("/v1/chat/completions", json=body, **kw)


# ------------------------------------------------------------------ /v1/models


def test_models_list_and_get(api):
    client, _, _ = api
    body = client.get("/v1/models").json()
    assert body["object"] == "list" and [m["id"] for m in body["data"]] == ["clef-flash"]
    assert body["data"][0]["object"] == "model"
    assert client.get("/v1/models/clef-flash").json()["id"] == "clef-flash"
    r = client.get("/v1/models/gpt-4o")
    assert r.status_code == 404 and r.json()["error"]["code"] == "model_not_found"


# ------------------------------------------------------------------ chat completions: happy path


def test_chat_structured_output(api):
    client, eng, _ = api
    r = post(client, chat())
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["object"] == "chat.completion" and body["id"].startswith("chatcmpl-")
    ch = body["choices"][0]
    assert ch["finish_reason"] == "stop" and ch["message"]["role"] == "assistant"
    assert json.loads(ch["message"]["content"]) == {"team": "account", "outage": True, "urgency": 3}
    assert body["usage"] == {"prompt_tokens": 10, "completion_tokens": 0, "total_tokens": 10}
    q = body["clef"]["questions"]
    assert q["team"]["type"] == "choice" and math.isclose(sum(q["team"]["probabilities"].values()), 1)
    assert q["outage"]["probabilities"] == {"true": 0.5, "false": 0.5}
    assert set(q["urgency"]["probabilities"]) == {"1", "2", "3"}
    assert body["clef"]["request_id"] == r.headers["x-request-id"]
    # translation: one record, three typed questions, descriptions -> instructions, state verbatim
    rec = eng.calls[0][0]
    assert rec["state"] == "Checkout is down"
    assert rec["questions"]["team"] == {
        "type": "choice",
        "instructions": "Who handles it?",
        "criteria": {"billing": "billing", "technical": "technical", "account": "account"},
    }
    assert rec["questions"]["outage"] == {"type": "noul", "instructions": "Is a service down?"}
    assert rec["questions"]["urgency"] == {"type": "score", "criteria": ["1", "2", "3"]}


def test_chat_noul_respects_probability(api):
    client, eng, _ = api
    eng.noul["outage"] = 0.2
    body = post(client, chat()).json()
    assert json.loads(body["choices"][0]["message"]["content"])["outage"] is False
    assert math.isclose(body["clef"]["questions"]["outage"]["probabilities"]["false"], 0.8)


def test_chat_transcript_and_roles(api):
    client, eng, _ = api
    msgs = [
        {"role": "system", "content": "You triage tickets."},
        {
            "role": "user",
            "content": [{"type": "text", "text": "Hi"}, {"type": "text", "text": "Checkout down"}],
        },
        {"role": "assistant", "content": "Noted."},
        {"role": "user", "content": "Still down"},
    ]
    assert post(client, chat(messages=msgs)).status_code == 200
    assert eng.calls[0][0]["state"] == (
        "system: You triage tickets.\n\nuser: Hi\nCheckout down\n\nassistant: Noted.\n\nuser: Still down"
    )


def test_chat_images_and_videos(api):
    client, eng, _ = api
    url = png_url()
    msgs = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "what is this?"},
                {"type": "image_url", "image_url": {"url": url, "detail": "low"}},
            ],
        }
    ]
    assert post(client, chat(messages=msgs)).status_code == 200
    rec = eng.calls[0][0]
    assert rec["state"] == "what is this?" and len(rec["images"]) == 1


def test_chat_http_image_rejected_unless_fetch_allowed(api):
    client, _, _ = api
    msgs = [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": "http://x.test/a.png"}}]}]
    r = post(client, chat(messages=msgs))
    assert r.status_code == 400 and r.json()["error"]["type"] == "invalid_request_error"


def test_chat_ignores_generation_params(api):
    client, _, _ = api
    r = post(client, chat(temperature=0.2, max_tokens=5, top_p=0.9, seed=1, stop=["x"], user="u", n=1))
    assert r.status_code == 200


def test_chat_reports_ignored_fields(api):
    client, _, _ = api
    r = post(client, chat(model="gpt-4o", temperature=0.2, max_tokens=5, seed=None, stop=["x"], n=1))
    assert r.json()["clef"]["ignored"] == ["model", "temperature", "max_tokens", "stop"]
    assert post(client, chat()).json()["clef"]["ignored"] == []


def test_chat_array_multilabel(api):
    client, eng, _ = api
    eng.noul.update({"tags:bug": 0.9, "tags:ui": 0.1, "tags:perf": 0.7})
    schema = {
        "type": "object",
        "properties": {
            "tags": {
                "type": "array",
                "description": "Which apply?",
                "items": {"type": "string", "enum": ["bug", "ui", "perf"]},
            }
        },
    }
    body = post(client, chat(response_format=rf(schema))).json()
    assert json.loads(body["choices"][0]["message"]["content"]) == {"tags": ["bug", "perf"]}
    qs = eng.calls[0][0]["questions"]
    assert set(qs) == {"tags:bug", "tags:ui", "tags:perf"}
    assert qs["tags:ui"]["instructions"] == 'Which apply? Does the label "ui" apply?'
    assert body["clef"]["questions"]["tags"]["probabilities"] == {"bug": 0.9, "ui": 0.1, "perf": 0.7}


# ------------------------------------------------------------------ schema mapping


def plan(schema: dict[str, Any], **kw: Any) -> cs.Plan:
    return cs.build_plan(schema, "schema", kw.pop("max_labels", 64), **kw)


def obj(**props: Any) -> dict[str, Any]:
    return {"type": "object", "properties": props}


def test_pydantic_style_refs_and_optional():
    schema = {
        "$defs": {"Team": {"enum": ["a", "b"], "type": "string", "title": "Team"}},
        "type": "object",
        "properties": {
            "team": {"$ref": "#/$defs/Team", "description": "who"},
            "opt": {"anyOf": [{"$ref": "#/$defs/Team"}, {"type": "null"}]},
            "wrapped": {"allOf": [{"$ref": "#/$defs/Team"}], "description": "w"},
            "flag": {"anyOf": [{"type": "boolean"}, {"type": "null"}]},
            "lit": {"enum": ["x", "y"], "type": "string"},
        },
    }
    p = plan(schema)
    assert [(x.name, x.kind) for x in p.props] == [
        ("team", "choice"),
        ("opt", "choice"),
        ("wrapped", "choice"),
        ("flag", "bool"),
        ("lit", "choice"),
    ]
    assert p.questions["team"]["instructions"] == "who"
    assert p.questions["wrapped"]["instructions"] == "w"


def test_described_enum_via_oneof_and_hint():
    p = plan(
        obj(
            kind={
                "oneOf": [
                    {"const": "bug", "description": "Something is broken"},
                    {"const": "ask", "description": "A question"},
                ]
            },
            lvl={
                "type": "string",
                "enum": ["low", "high"],
                "x-clef-descriptions": {"low": "Can wait", "high": "Now"},
            },
        )
    )
    assert p.questions["kind"]["criteria"] == {"bug": "Something is broken", "ask": "A question"}
    assert p.questions["lvl"]["criteria"] == {"low": "Can wait", "high": "Now"}


def test_scores_integer_enum_range_and_ordinal_strings():
    p = plan(
        obj(
            a={"type": "integer", "enum": [3, 1, 2]},
            b={"type": "integer", "minimum": 0, "maximum": 2, "x-clef-descriptions": {"0": "none"}},
            c={"type": "string", "enum": ["low", "mid", "high"], "x-clef-ordinal": True},
        )
    )
    assert p.questions["a"] == {"type": "score", "criteria": ["1", "2", "3"]}
    assert p.questions["b"]["criteria"] == ["none", "1", "2"]
    assert p.questions["c"] == {"type": "score", "criteria": ["low", "mid", "high"]}
    assert [x.options for x in p.props] == [[1, 2, 3], [0, 1, 2], ["low", "mid", "high"]]


def test_score_values_decode_to_integers_and_strings():
    p = plan(
        obj(
            a={"type": "integer", "enum": [10, 20]},
            c={"type": "string", "enum": ["low", "high"], "x-clef-ordinal": True},
        )
    )
    answers = {
        "a": {"probabilities": {"0": 0.3, "1": 0.7}, "confidence": 0.7},
        "c": {"probabilities": {"0": 0.9, "1": 0.1}},
    }
    out = cs.decode(p, answers, 0.5)
    assert out.values == {"a": 20, "c": "low"}
    assert out.detail["a"]["probabilities"] == {"10": 0.3, "20": 0.7}


@pytest.mark.parametrize(
    "schema, needle",
    [
        ("nope", "JSON schema object"),
        ({"type": "object", "properties": {}}, "at least one property"),
        ({"type": "string"}, "at least one property"),
        (obj(x={"type": "string"}), "unsupported property type"),
        (obj(x={"type": "number"}), "unsupported property type"),
        (obj(x={"type": "object", "properties": {}}), "unsupported property type"),
        (obj(x={"type": "string", "enum": ["only"]}), "at least 2 options"),
        (obj(x={"enum": ["a", 1]}), "all strings or all integers"),
        (obj(x={"type": "array", "items": {"type": "string"}}), "enum of strings"),
        (obj(x={"$ref": "#/$defs/missing"}), "cannot resolve"),
        (obj(x={"$ref": "http://evil/x"}), "local"),
        (obj(x={"anyOf": [{"type": "string"}, {"type": "boolean"}]}), "unsupported anyOf"),
        (obj(x={"type": "integer", "minimum": 0, "maximum": 1000}), "integer range"),
    ],
)
def test_unsupported_schemas(schema, needle):
    with pytest.raises(cs.SchemaError, match=needle):
        plan(schema)


def test_too_many_options_and_property_collision():
    with pytest.raises(cs.SchemaError, match="too many options"):
        plan(obj(x={"enum": ["a", "b", "c"]}), max_labels=2)
    with pytest.raises(cs.SchemaError, match="collides"):
        plan(obj(**{"t:a": {"type": "boolean"}, "t": {"type": "array", "items": {"enum": ["a"]}}}))


def test_recursive_ref_is_rejected():
    schema = {"$defs": {"A": {"$ref": "#/$defs/A"}}, **obj(x={"$ref": "#/$defs/A"})}
    with pytest.raises(cs.SchemaError, match="too deeply"):
        plan(schema)


def test_messages_errors():
    for bad in (None, [], "hi", [1], [{"content": "x"}], [{"role": "system", "content": "only system"}]):
        with pytest.raises(cs.SchemaError):
            cs.messages_to_state(bad)
    with pytest.raises(cs.SchemaError, match="unsupported content part"):
        cs.messages_to_state([{"role": "user", "content": [{"type": "input_audio", "input_audio": {}}]}])


def test_tool_call_history_rendered():
    state, _, _ = cs.messages_to_state(
        [
            {"role": "user", "content": "weather?"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [{"function": {"name": "w", "arguments": "{}"}}],
            },
            {"role": "tool", "content": "sunny"},
        ]
    )
    assert state == "user: weather?\n\nassistant: [called w({})]\n\ntool: sunny"


# ------------------------------------------------------------------ tools


def tool(name: str = "route", params: dict[str, Any] | None = None) -> dict[str, Any]:
    return {
        "type": "function",
        "function": {"name": name, "description": "Route the ticket.", "parameters": params or SCHEMA},
    }


def test_forced_tool_returns_tool_calls(api):
    client, eng, _ = api
    body = chat(
        tools=[tool()], tool_choice={"type": "function", "function": {"name": "route"}}, response_format=None
    )
    body.pop("response_format")
    out = post(client, body).json()
    ch = out["choices"][0]
    assert ch["finish_reason"] == "stop" and ch["message"]["content"] is None
    call = ch["message"]["tool_calls"][0]
    assert (
        call["type"] == "function" and call["function"]["name"] == "route" and call["id"].startswith("call_")
    )
    assert json.loads(call["function"]["arguments"]) == {"team": "account", "outage": True, "urgency": 3}
    # the tool description becomes the question preamble
    assert eng.calls[0][0]["questions"]["team"]["instructions"].startswith("Route the ticket. Who handles")


def test_tool_choice_required_and_auto_single_tool(api):
    client, _, _ = api
    for choice in ("required", "auto", None):
        body = chat(tools=[tool()], tool_choice=choice)
        body.pop("response_format")
        body = {k: v for k, v in body.items() if v is not None}
        out = post(client, body).json()
        assert out["choices"][0]["finish_reason"] == "tool_calls"


def test_tool_choice_errors(api):
    client, _, _ = api
    two = [tool("a"), tool("b")]
    base = chat()
    base.pop("response_format")
    cases = [
        ({"tools": two}, "tool_choice"),
        ({"tools": two, "tool_choice": "required"}, "tool_choice"),
        (
            {"tools": [tool()], "tool_choice": {"type": "function", "function": {"name": "zzz"}}},
            "tool_choice",
        ),
        ({"tools": [tool()], "tool_choice": "banana"}, "tool_choice"),
        ({"tools": [{"type": "retrieval"}]}, "tools"),
        ({"tools": "x"}, "tools"),
        ({"tools": [tool()], "tool_choice": "none"}, "response_format"),
    ]
    for extra, param in cases:
        r = post(client, {**base, **extra})
        assert r.status_code == 400, (extra, r.text)
        assert r.json()["error"]["param"] == param, (extra, r.text)


def test_forced_tool_wins_over_response_format(api):
    client, _, _ = api
    other = obj(only={"type": "boolean"})
    out = post(
        client,
        chat(
            response_format=rf(other),
            tools=[tool()],
            tool_choice={"type": "function", "function": {"name": "route"}},
        ),
    ).json()
    assert "team" in json.loads(out["choices"][0]["message"]["tool_calls"][0]["function"]["arguments"])


# ------------------------------------------------------------------ logprobs


def test_logprobs(api):
    client, _, _ = api
    out = post(client, chat(logprobs=True, top_logprobs=2)).json()
    lp = out["choices"][0]["logprobs"]
    assert lp["refusal"] is None
    first = lp["content"][0]
    assert first["token"] == "account" and math.isclose(first["logprob"], math.log(3 / 6))
    assert [t["token"] for t in first["top_logprobs"]] == ["account", "technical"]
    assert math.isclose(first["top_logprobs"][1]["logprob"], math.log(2 / 6))
    assert first["bytes"] == list(b"account")
    outage = lp["content"][1]
    assert outage["token"] == "true" and {t["token"] for t in outage["top_logprobs"]} == {"true", "false"}


def test_logprobs_without_top_has_empty_top_and_off_by_default(api):
    client, _, _ = api
    lp = post(client, chat(logprobs=True)).json()["choices"][0]["logprobs"]
    assert all(e["top_logprobs"] == [] for e in lp["content"])
    assert post(client, chat()).json()["choices"][0]["logprobs"] is None


def test_logprob_floor_for_zero_probability():
    assert cs._log(0.0) == cs.LOGPROB_FLOOR


def test_top_logprobs_validation(api):
    client, _, _ = api
    assert post(client, chat(top_logprobs=2)).status_code == 400  # needs logprobs: true
    assert post(client, chat(logprobs=True, top_logprobs=500)).status_code == 400
    assert post(client, chat(logprobs=True, top_logprobs="x")).status_code == 400


# ------------------------------------------------------------------ streaming


def sse_events(text: str) -> list[Any]:
    out = []
    for block in text.strip().split("\n\n"):
        assert block.startswith("data: "), block
        data = block[6:]
        out.append(data if data == "[DONE]" else json.loads(data))
    return out


def test_stream_chunks(api):
    client, _, _ = api
    r = post(client, chat(stream=True, logprobs=True, top_logprobs=1))
    assert r.status_code == 200 and r.headers["content-type"].startswith("text/event-stream")
    ev = sse_events(r.text)
    assert ev[-1] == "[DONE]"
    assert all(e["object"] == "chat.completion.chunk" and e["id"] == ev[0]["id"] for e in ev[:-1])
    assert ev[0]["choices"][0]["delta"] == {"role": "assistant", "content": ""}
    content = "".join(e["choices"][0]["delta"].get("content") or "" for e in ev[:-1] if e["choices"])
    assert json.loads(content) == {"team": "account", "outage": True, "urgency": 3}
    assert ev[1]["choices"][0]["logprobs"]["content"][0]["token"] == "account"
    assert ev[-2]["choices"][0]["finish_reason"] == "stop" and ev[-2]["choices"][0]["delta"] == {}
    assert all("usage" not in e for e in ev[:-1])


def test_stream_include_usage_and_tools(api):
    client, _, _ = api
    ev = sse_events(post(client, chat(stream=True, stream_options={"include_usage": True})).text)
    assert ev[-2]["choices"] == [] and ev[-2]["usage"]["prompt_tokens"] == 10
    body = chat(stream=True, tools=[tool()], tool_choice="required")
    body.pop("response_format")
    ev = sse_events(post(client, body).text)
    call = ev[1]["choices"][0]["delta"]["tool_calls"][0]
    assert call["index"] == 0 and call["function"]["name"] == "route"
    assert ev[-2]["choices"][0]["finish_reason"] == "tool_calls"


def test_stream_errors_are_plain_json_not_sse(api):
    client, _, _ = api
    r = post(client, chat(stream=True, response_format=None))
    assert r.status_code == 400 and r.headers["content-type"].startswith("application/json")


# ------------------------------------------------------------------ errors (OpenAI shape)


def err(r) -> dict[str, Any]:
    assert set(r.json()) == {"error"}
    e = r.json()["error"]
    assert set(e) == {"message", "type", "param", "code"}
    return e


def test_missing_schema_is_clear_400(api):
    client, _, _ = api
    for rfv in (None, {"type": "json_object"}, {"type": "text"}):
        r = post(client, {**chat(), "response_format": rfv})
        e = err(r)
        assert r.status_code == 400 and e["type"] == "invalid_request_error" and e["code"] == "missing_schema"
        assert "free text" in e["message"] and "enum" in e["message"]


def test_bad_schema_names_the_param(api):
    client, _, _ = api
    r = post(client, chat(response_format=rf(obj(x={"type": "string"}))))
    e = err(r)
    assert r.status_code == 400
    assert e["param"] == "response_format.json_schema.schema.properties.x"
    assert "free text" in e["message"].lower() or "unsupported" in e["message"]


def test_n_greater_than_one_rejected(api):
    client, _, _ = api
    r = post(client, chat(n=2))
    assert r.status_code == 400 and err(r)["param"] == "n"


def test_body_errors(api):
    client, _, _ = api
    r = client.post("/v1/chat/completions", content=b"{nope", headers={"content-type": "application/json"})
    assert r.status_code == 400 and "valid JSON" in err(r)["message"]
    r = client.post("/v1/chat/completions", json=[1])
    assert r.status_code == 400
    r = post(client, chat(messages=[]))
    assert r.status_code == 400 and err(r)["param"] == "messages"
    r = post(client, {"response_format": rf()})
    assert r.status_code == 400 and err(r)["param"] == "messages"


def test_too_many_questions_is_400(api):
    client, _, _ = api
    props = {f"q{i}": {"type": "boolean"} for i in range(80)}
    r = post(client, chat(response_format=rf(obj(**props))))
    assert r.status_code == 400 and "too many questions" in err(r)["message"]


def test_engine_errors_mapped():
    from clef_server.engine import EngineNotReady

    client, eng, _ = make()
    with client:
        eng.raises = EngineNotReady("model is loading")
        r = post(client, chat())
        assert r.status_code == 503 and err(r)["type"] == "server_error"
        eng.raises = RuntimeError("boom")
        r = post(client, chat())
        assert r.status_code == 500 and "boom" not in r.text


# ------------------------------------------------------------------ auth, rate limit, stats


def test_auth_and_openai_error_shape():
    client, _, _ = make(api_key="sekret")
    with client:
        r = post(client, chat())
        assert r.status_code == 401 and err(r)["code"] == "invalid_api_key"
        assert r.headers["www-authenticate"] == "Bearer"
        ok = post(client, chat(), headers={"Authorization": "Bearer sekret"})
        assert ok.status_code == 200
        assert post(client, chat(), headers={"X-API-Key": "sekret"}).status_code == 200
        assert client.get("/v1/models").status_code == 401
        assert client.get("/v1/models", headers={"Authorization": "Bearer sekret"}).status_code == 200
        hf = client.post("/hf/models/x", json={"inputs": "a"})
        assert hf.status_code == 401 and set(hf.json()) == {"error"}


def test_rate_limit_applies_to_all_compat_routes():
    client, _, _ = make(rate_limit=2)
    with client:
        assert post(client, chat()).status_code == 200
        hf = {"inputs": "x", "parameters": {"candidate_labels": ["a", "b"]}}
        assert client.post("/hf/models/m", json=hf).status_code == 200
        r = post(client, chat())
        assert r.status_code == 429 and err(r)["type"] == "rate_limit_error" and "retry-after" in r.headers
        r = client.post("/models/m", json=hf)
        assert r.status_code == 429 and set(r.json()) == {"error"}


def test_requests_are_logged_and_counted(api):
    client, _, _ = api
    post(client, chat())
    client.post("/hf/models/m", json={"inputs": "x", "parameters": {"candidate_labels": "a,b"}})
    post(client, chat(n=2))
    log = client.get("/v1/log").json()["entries"]
    paths = [(e["endpoint"], e["status"]) for e in log]
    assert ("/v1/chat/completions", 200) in paths and ("/hf/models/m", 200) in paths
    assert ("/v1/chat/completions", 400) in paths
    first = next(e for e in log if e["endpoint"] == "/v1/chat/completions")
    assert first["n_questions"] == 3 and first["input_tokens"] == 10


# ------------------------------------------------------------------ Hugging Face


def hf(client, inputs: Any, labels: Any = ("billing", "technical"), url="/hf/models/clef-flash", **params):
    body = {"inputs": inputs, "parameters": {"candidate_labels": labels, **params}}
    return client.post(url, json=body)


def test_hf_single_input_sorted_and_normalised(api):
    client, eng, _ = api
    r = hf(client, "Checkout is down")
    assert r.status_code == 200, r.text
    b = r.json()
    assert b["sequence"] == "Checkout is down" and b["labels"] == ["technical", "billing"]
    assert math.isclose(b["scores"][0], 2 / 3) and math.isclose(sum(b["scores"]), 1)
    q = eng.calls[0][0]["questions"]["label"]
    assert q["type"] == "choice" and q["criteria"] == {"billing": "billing", "technical": "technical"}


def test_hf_aliases_and_model_ids_with_slashes(api):
    client, _, _ = api
    for url in ("/hf/models/facebook/bart-large-mnli", "/models/facebook/bart-large-mnli", "/hf/models/x"):
        assert hf(client, "hello", url=url).status_code == 200, url


def test_hf_comma_string_labels_dedupe(api):
    client, eng, _ = api
    b = hf(client, "t", "a, b ,a,,c").json()
    assert sorted(b["labels"]) == ["a", "b", "c"]
    assert list(eng.calls[0][0]["questions"]["label"]["criteria"]) == ["a", "b", "c"]


def test_hf_list_input_is_one_batch(api):
    client, eng, _ = api
    b = hf(client, ["one", "two", "three"]).json()
    assert [x["sequence"] for x in b] == ["one", "two", "three"] and all("labels" in x for x in b)
    assert len(eng.calls) == 1 and len(eng.calls[0]) == 3


def test_hf_multi_label(api):
    client, eng, _ = api
    eng.noul.update({"billing": 0.9, "technical": 0.2, "account": 0.6})
    b = hf(client, "text", ["billing", "technical", "account"], multi_label=True).json()
    assert b["labels"] == ["billing", "account", "technical"] and b["scores"] == [0.9, 0.6, 0.2]
    assert {q["type"] for q in eng.calls[0][0]["questions"].values()} == {"noul"}


def test_hf_hypothesis_template_becomes_descriptions(api):
    client, eng, _ = api
    assert hf(client, "t", hypothesis_template="This is about {}.").status_code == 200
    crit = eng.calls[0][0]["questions"]["label"]["criteria"]
    assert crit == {"billing": "This is about billing.", "technical": "This is about technical."}
    r = hf(client, "t", hypothesis_template="no placeholder")
    assert r.status_code == 400 and "{}" in r.json()["error"]


def test_hf_format_selection(api):
    client, _, _ = api
    body = {"inputs": "t", "parameters": {"candidate_labels": ["a", "b"]}}
    new = {"user-agent": "unknown/None; hf_hub/1.33.0; python/3.12"}
    old = {"user-agent": "unknown/None; hf_hub/0.24.0; python/3.10"}
    r = client.post("/hf/models/m", json=body, headers=new).json()
    assert r == [{"label": "b", "score": pytest.approx(2 / 3)}, {"label": "a", "score": pytest.approx(1 / 3)}]
    assert "labels" in client.post("/hf/models/m", json=body, headers=old).json()
    assert "labels" in client.post("/hf/models/m", json=body).json()
    assert isinstance(client.post("/hf/models/m?format=list", json=body).json(), list)
    assert "labels" in client.post("/hf/models/m?format=classic", json=body, headers=new).json()
    assert client.post("/hf/models/m?format=x", json=body).status_code == 400


@pytest.mark.parametrize(
    "body, needle",
    [
        ({}, "inputs"),
        ({"inputs": ""}, "inputs"),
        ({"inputs": []}, "inputs"),
        ({"inputs": [1]}, "inputs"),
        ({"inputs": "x"}, "candidate_labels"),
        ({"inputs": "x", "parameters": {"candidate_labels": []}}, "candidate_labels"),
        ({"inputs": "x", "parameters": {"candidate_labels": 5}}, "candidate_labels"),
        ({"inputs": "x", "parameters": {"candidate_labels": ["a"]}}, "at least 2 labels"),
        ({"inputs": "x", "parameters": {"candidate_labels": "a,b", "multi_label": "yes"}}, "multi_label"),
        ({"inputs": "x", "parameters": "nope"}, "parameters"),
    ],
)
def test_hf_validation_errors(api, body, needle):
    client, _, _ = api
    r = client.post("/hf/models/m", json=body)
    assert r.status_code == 400 and set(r.json()) == {"error"} and needle in r.json()["error"]


def test_hf_batch_limit_and_bad_json(api):
    client, _, _ = api
    r = hf(client, ["x"] * 1000)
    assert r.status_code == 400 and "too many inputs" in r.json()["error"]
    r = client.post("/hf/models/m", content=b"{", headers={"content-type": "application/json"})
    assert r.status_code == 400 and "JSON" in r.json()["error"]


def test_console_and_other_routes_unaffected(api):
    client, _, _ = api
    assert client.get("/livez").json() == {"ok": True}
    assert client.get("/models/x").status_code in (404, 405)  # GET is not a compat route


# ------------------------------------------------------------------ real client libraries over HTTP


@pytest.fixture(scope="module")
def live():
    uvicorn = pytest.importorskip("uvicorn")
    client, eng, app = make()
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="error"))
    t = threading.Thread(target=server.run, daemon=True)
    t.start()
    for _ in range(100):
        if server.started:
            break
        time.sleep(0.05)
    assert server.started
    yield f"http://127.0.0.1:{port}", eng
    server.should_exit = True
    t.join(5)


def test_openai_sdk_create_stream_and_parse(live):
    openai = pytest.importorskip("openai")
    pydantic = pytest.importorskip("pydantic")
    from enum import Enum
    from typing import Literal

    base, eng = live
    client = openai.OpenAI(base_url=f"{base}/v1", api_key="unused", max_retries=0)
    assert [m.id for m in client.models.list().data] == ["clef-flash"]
    assert client.models.retrieve("clef-flash").id == "clef-flash"

    r = client.chat.completions.create(
        model="clef-flash",
        messages=[{"role": "user", "content": "Checkout is down"}],
        response_format=rf(),
        logprobs=True,
        top_logprobs=2,
    )
    assert json.loads(r.choices[0].message.content)["team"] == "account"
    assert r.usage.prompt_tokens == 10 and r.choices[0].logprobs.content[0].top_logprobs[0].token == "account"

    class Urgency(str, Enum):
        low = "low"
        high = "high"

    class Triage(pydantic.BaseModel):
        team: Literal["billing", "technical", "account"]
        urgency: Urgency
        outage: bool

    parsed = client.beta.chat.completions.parse(
        model="clef-flash",
        messages=[{"role": "system", "content": "Triage."}, {"role": "user", "content": "Checkout is down"}],
        response_format=Triage,
    )
    assert parsed.choices[0].message.parsed == Triage(team="account", urgency=Urgency.high, outage=True)
    assert "system: Triage." in eng.calls[-1][0]["state"]

    chunks = list(
        client.chat.completions.create(
            model="clef-flash",
            messages=[{"role": "user", "content": "x"}],
            response_format=rf(),
            stream=True,
            stream_options={"include_usage": True},
        )
    )
    text = "".join(c.choices[0].delta.content or "" for c in chunks if c.choices)
    assert json.loads(text)["outage"] is True and chunks[-1].usage.prompt_tokens == 10

    t = client.chat.completions.create(
        model="clef-flash",
        messages=[{"role": "user", "content": "x"}],
        tools=[tool()],
        tool_choice={"type": "function", "function": {"name": "route"}},
    )
    assert json.loads(t.choices[0].message.tool_calls[0].function.arguments)["team"] == "account"

    with pytest.raises(openai.BadRequestError) as ei:
        client.chat.completions.create(model="clef-flash", messages=[{"role": "user", "content": "x"}])
    assert "free text" in str(ei.value)
    with pytest.raises(openai.NotFoundError):
        client.models.retrieve("gpt-4o")


def test_huggingface_hub_inference_client(live):
    hub = pytest.importorskip("huggingface_hub")
    base, _ = live
    client = hub.InferenceClient(model=f"{base}/hf/models/clef-flash")
    out = client.zero_shot_classification("Checkout is down", ["billing", "technical"])
    assert [o.label for o in out] == ["technical", "billing"] and math.isclose(out[0].score, 2 / 3)
    multi = client.zero_shot_classification(
        "x", ["a", "b", "c"], multi_label=True, hypothesis_template="It is {}."
    )
    assert {o.label for o in multi} == {"a", "b", "c"} and all(o.score == 0.5 for o in multi)
