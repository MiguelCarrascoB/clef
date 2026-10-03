"""OpenAI-compatible routes: GET /v1/models[/{id}] and POST /v1/chat/completions (structured decisions)."""

from __future__ import annotations

import json
import time
import uuid
from collections.abc import Iterator
from typing import Any

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse, StreamingResponse

from .appctx import AppContext
from .classify import resolve_threshold
from .compat_errors import openai_error, read_json_object, route_class
from .compat_schema import Decoded, Plan, SchemaError, build_plan, decode, messages_to_state
from .schemas import DEFAULT_MODEL, Question, SystemOneRequest

OWNER = "cloudflare"
MAX_TOP_LOGPROBS = 64
NO_SCHEMA = (
    "clef decides between options; it does not generate free text. Send a schema whose properties are "
    "decisions, e.g. response_format={'type': 'json_schema', 'json_schema': {'name': 'triage', 'schema': "
    "{'type': 'object', 'properties': {'team': {'type': 'string', 'enum': ['billing', 'technical']}}}}}, "
    "or force one function with tool_choice. See docs/openai-compat.md."
)


def _model_obj(created: int) -> dict[str, Any]:
    return {"id": DEFAULT_MODEL, "object": "model", "created": created, "owned_by": OWNER}


def _int_param(body: dict[str, Any], key: str) -> int | None:
    v = body.get(key)
    if v is None:
        return None
    if isinstance(v, bool) or not isinstance(v, int):
        raise SchemaError(f"{key} must be an integer", key)
    return v


def _target(body: dict[str, Any], max_labels: int) -> tuple[Plan, str | None, bool]:
    """Pick the schema to decide on: a forced tool wins, then response_format, then a lone tool.

    Returns (plan, tool name or None, tool_choice was a named function).
    """
    tools = body.get("tools")
    choice = body.get("tool_choice")
    tool: dict[str, Any] | None = None
    named = False
    if tools is not None and (not isinstance(tools, list) or not all(isinstance(t, dict) for t in tools)):
        raise SchemaError("tools must be an array of tool objects", "tools")
    fns = [t for t in tools or [] if t.get("type") == "function" and isinstance(t.get("function"), dict)]
    if tools and not fns:
        raise SchemaError("only tools of type 'function' are supported", "tools")
    if isinstance(choice, dict):
        name = (choice.get("function") or {}).get("name") if choice.get("type") == "function" else None
        if not isinstance(name, str):
            raise SchemaError(
                "tool_choice must be 'auto', 'required', 'none' or a named function", "tool_choice"
            )
        tool = next((t for t in fns if t["function"].get("name") == name), None)
        if tool is None:
            raise SchemaError(f"tool_choice names {name!r}, which is not in tools", "tool_choice")
        named = True
    elif choice not in (None, "auto", "required", "none"):
        raise SchemaError("tool_choice must be 'auto', 'required', 'none' or a named function", "tool_choice")
    elif choice == "required" and fns:
        if len(fns) > 1:
            raise SchemaError(
                "clef cannot pick which function to call; force one with "
                "tool_choice={'type': 'function', 'function': {'name': ...}}",
                "tool_choice",
            )
        tool = fns[0]

    rf = body.get("response_format")
    if tool is None and isinstance(rf, dict) and rf.get("type") == "json_schema":
        js = rf.get("json_schema")
        if not isinstance(js, dict) or "schema" not in js:
            raise SchemaError("response_format.json_schema.schema is required", "response_format.json_schema")
        return build_plan(js["schema"], "response_format.json_schema.schema", max_labels), None, False
    if tool is None and choice != "none" and len(fns) == 1:
        tool = fns[0]
    if tool is None:
        if rf is not None and not (isinstance(rf, dict) and rf.get("type") in ("text", "json_object", None)):
            raise SchemaError("response_format.type must be 'json_schema' for clef", "response_format.type")
        if len(fns) > 1 and choice != "none":
            raise SchemaError(
                "clef cannot pick which function to call; force one with "
                "tool_choice={'type': 'function', 'function': {'name': ...}}",
                "tool_choice",
            )
        raise SchemaError(NO_SCHEMA, "response_format", "missing_schema")
    fn = tool["function"]
    where = f"tools[{fn.get('name', '?')}].function.parameters"
    plan = build_plan(fn.get("parameters") or {}, where, max_labels, fn.get("description"))
    return plan, str(fn.get("name") or "function"), named


def _usage(result: dict[str, Any]) -> dict[str, int]:
    u = result.get("usage") or {}
    p, c = int(u.get("input_tokens", 0)), int(u.get("output_tokens", 0))
    return {"prompt_tokens": p, "completion_tokens": c, "total_tokens": p + c}


def _sse(obj: Any) -> str:
    return f"data: {json.dumps(obj, ensure_ascii=False) if obj != '[DONE]' else '[DONE]'}\n\n"


def router(ctx: AppContext) -> APIRouter:
    r = APIRouter(
        prefix="/v1", dependencies=ctx.auth, responses=ctx.errors, route_class=route_class(ctx, openai_error)
    )
    created_at = int(time.time())

    @r.get("/models", summary="List models (OpenAI format)")
    async def list_models() -> dict[str, Any]:
        return {"object": "list", "data": [_model_obj(created_at)]}

    @r.get("/models/{model_id:path}", summary="Retrieve a model (OpenAI format)")
    async def get_model(model_id: str, request: Request) -> Any:
        if model_id != DEFAULT_MODEL:
            return openai_error(
                request,
                404,
                f"The model '{model_id}' does not exist (clef serves '{DEFAULT_MODEL}')",
                "model",
                "model_not_found",
                {},
            )
        return _model_obj(created_at)

    @r.post(
        "/chat/completions",
        summary="Chat completions as structured decisions (OpenAI format)",
        dependencies=ctx.limited,
    )
    async def chat_completions(request: Request) -> Any:
        body = await read_json_object(request)
        if _int_param(body, "n") not in (None, 1):
            raise SchemaError(
                "n > 1 is not supported: clef returns one deterministic decision", "n", "unsupported_value"
            )
        top = _int_param(body, "top_logprobs")
        if top is not None and not 0 <= top <= MAX_TOP_LOGPROBS:
            raise SchemaError(f"top_logprobs must be between 0 and {MAX_TOP_LOGPROBS}", "top_logprobs")
        if top is not None and body.get("logprobs") is not True:
            raise SchemaError("top_logprobs requires logprobs: true", "top_logprobs")
        stream = body.get("stream") is True
        opts = body.get("stream_options")
        include_usage = stream and isinstance(opts, dict) and opts.get("include_usage") is True

        plan, tool_name, named = _target(body, ctx.cfg.max_labels)
        state, images, videos = messages_to_state(body.get("messages"))
        try:
            sreq = SystemOneRequest(
                model=DEFAULT_MODEL,
                state=state,
                questions={k: Question(**q) for k, q in plan.questions.items()},
                images=images or None,
                videos=videos or None,
            )
        except ValueError as exc:  # pydantic: message already names the field
            raise SchemaError(str(exc), None) from exc
        t0 = time.perf_counter()
        results = await ctx.infer(request, [sreq], batch=False)
        res = results[0]
        out = decode(plan, res["answers"], resolve_threshold(None, ctx.cfg))

        cid = f"chatcmpl-{uuid.uuid4().hex[:24]}"
        message, finish = _message(out, tool_name, named)
        logprobs = out.logprobs(plan, top or 0) if body.get("logprobs") is True else None
        usage = _usage(res)
        extra = {
            "questions": out.detail,
            "timing": {**(res.get("timing") or {}), "total_ms": round((time.perf_counter() - t0) * 1000, 1)},
            "request_id": request.state.request_id,
        }
        if stream:
            return StreamingResponse(
                _stream(cid, created_at, message, finish, logprobs, usage if include_usage else None, extra),
                media_type="text/event-stream",
                headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
            )
        return JSONResponse(
            {
                "id": cid,
                "object": "chat.completion",
                "created": int(time.time()),
                "model": DEFAULT_MODEL,
                "choices": [{"index": 0, "message": message, "logprobs": logprobs, "finish_reason": finish}],
                "usage": usage,
                "system_fingerprint": None,
                "clef": extra,
            }
        )

    return r


def _message(out: Decoded, tool_name: str | None, named: bool) -> tuple[dict[str, Any], str]:
    if tool_name is None:
        return {"role": "assistant", "content": out.content(), "refusal": None}, "stop"
    call = {
        "id": f"call_{uuid.uuid4().hex[:24]}",
        "type": "function",
        "function": {"name": tool_name, "arguments": out.content()},
    }
    # OpenAI reports "stop" when the call was forced by name, "tool_calls" otherwise
    return {"role": "assistant", "content": None, "refusal": None, "tool_calls": [call]}, (
        "stop" if named else "tool_calls"
    )


def _stream(
    cid: str,
    created: int,
    message: dict[str, Any],
    finish: str,
    logprobs: dict[str, Any] | None,
    usage: dict[str, int] | None,
    extra: dict[str, Any],
) -> Iterator[str]:
    """The decision is computed in one pass, so the stream is: role, the whole content, finish, [DONE]."""

    def chunk(delta: dict[str, Any], fin: str | None = None, lp: Any = None, **more: Any) -> str:
        choices = [{"index": 0, "delta": delta, "logprobs": lp, "finish_reason": fin}] if delta or fin else []
        return _sse(
            {
                "id": cid,
                "object": "chat.completion.chunk",
                "created": created,
                "model": DEFAULT_MODEL,
                "choices": choices,
                **more,
            }
        )

    yield chunk({"role": "assistant", "content": "" if message.get("content") is not None else None})
    if message.get("tool_calls"):
        call = message["tool_calls"][0]
        yield chunk({"tool_calls": [{"index": 0, **call}]})
    else:
        yield chunk({"content": message["content"]}, lp=logprobs)
    yield chunk({}, finish, clef=extra)
    if usage is not None:
        yield chunk({}, usage=usage, clef=extra)
    yield _sse("[DONE]")
