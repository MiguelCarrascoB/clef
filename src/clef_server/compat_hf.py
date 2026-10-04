"""Hugging Face zero-shot-classification compatible route: POST /hf/models/{id} (and /models/{id})."""

from __future__ import annotations

import re
from typing import Any

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from .appctx import AppContext
from .compat_errors import hf_error, read_json_object, route_class
from .compat_schema import SchemaError
from .schemas import DEFAULT_MODEL

# huggingface_hub >= 1.0 parses the response as [{label, score}]; older clients and the classic Inference API
# use {sequence, labels, scores}. The classic shape is the default; hub >= 1 is detected by its User-Agent.
_NEW_HUB = re.compile(r"hf_hub/(\d+)\.")


def parse_labels(raw: Any) -> list[str]:
    if isinstance(raw, str):
        raw = raw.split(",")
    if not isinstance(raw, list) or not all(isinstance(x, str) for x in raw):
        raise SchemaError(
            "parameters.candidate_labels must be a list of strings or a comma-separated string",
            "parameters.candidate_labels",
        )
    labels = list(dict.fromkeys(x.strip() for x in raw if x.strip()))  # de-duplicated, order kept
    if not labels:
        raise SchemaError("parameters.candidate_labels is required", "parameters.candidate_labels")
    return labels


def label_map(labels: list[str], template: Any) -> list[str] | dict[str, str]:
    """With a hypothesis_template the filled template becomes each label's description."""
    if template is None:
        return labels
    if not isinstance(template, str) or "{}" not in template:
        raise SchemaError(
            "parameters.hypothesis_template must be a string containing '{}'",
            "parameters.hypothesis_template",
        )
    return {name: template.replace("{}", name) for name in labels}


def shape(sequence: str, scores: dict[str, float], as_list: bool) -> Any:
    ranked = sorted(scores.items(), key=lambda kv: -kv[1])  # stable on ties
    if as_list:
        return [{"label": k, "score": v} for k, v in ranked]
    return {"sequence": sequence, "labels": [k for k, _ in ranked], "scores": [v for _, v in ranked]}


def wants_list(request: Request) -> bool:
    fmt = request.query_params.get("format")
    if fmt not in (None, "classic", "list"):
        raise SchemaError("format must be 'classic' or 'list'", "format")
    if fmt is not None:
        return fmt == "list"
    m = _NEW_HUB.search(request.headers.get("user-agent", ""))
    return m is not None and int(m.group(1)) >= 1


def router(ctx: AppContext) -> APIRouter:
    r = APIRouter(dependencies=ctx.auth, route_class=route_class(ctx, hf_error))

    async def zero_shot(model_id: str, request: Request) -> Any:
        # model_id is deliberately ignored: any id is served by clef-flash (documented in openai-compat.md)
        body = await read_json_object(request)
        inputs = body.get("inputs")
        many = isinstance(inputs, list)
        texts = inputs if many else [inputs]
        if not texts or not all(isinstance(t, str) and t.strip() for t in texts):
            raise SchemaError("inputs must be a non-empty string or a non-empty list of strings", "inputs")
        params = body.get("parameters") or {}
        if not isinstance(params, dict):
            raise SchemaError("parameters must be an object", "parameters")
        labels = label_map(parse_labels(params.get("candidate_labels")), params.get("hypothesis_template"))
        multi = params.get("multi_label", False)
        if not isinstance(multi, bool):
            raise SchemaError("parameters.multi_label must be a boolean", "parameters.multi_label")
        as_list = wants_list(request)

        common: dict[str, Any] = {
            "instructions": None,
            "multi_label": multi,
            "threshold": None,
            "model": DEFAULT_MODEL,
        }
        if not many:
            res = await ctx.run_classify(
                request, state=texts[0], labels=labels, images=None, videos=None, **common
            )
            return JSONResponse(shape(texts[0], res["scores"], as_list))
        res = await ctx.run_classify_batch(request, inputs=texts, labels=labels, **common)
        items = zip(texts, res["results"], strict=True)
        return JSONResponse([shape(t, x["scores"], as_list) for t, x in items])

    summary = "Zero-shot classification (Hugging Face Inference API format)"
    r.add_api_route(
        "/hf/models/{model_id:path}", zero_shot, methods=["POST"], summary=summary, dependencies=ctx.limited
    )
    r.add_api_route(
        "/models/{model_id:path}",
        zero_shot,
        methods=["POST"],
        include_in_schema=False,
        dependencies=ctx.limited,
    )
    return r
