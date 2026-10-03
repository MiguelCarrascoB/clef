"""Error bodies for the compat routes: OpenAI `{"error": {...}}` and Hugging Face `{"error": "..."}`.

The rest of the API reports `{detail, request_id}`. SDKs surface their own error shape much better, so the
compat routers use a route class that turns every failure (auth, rate limit, validation, engine) into the
shape the client library expects. The HTTP status codes are the same as everywhere else.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any

from fastapi import Request, Response
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from fastapi.routing import APIRoute
from starlette.exceptions import HTTPException as StarletteHTTPException

from .appctx import AppContext
from .compat_schema import SchemaError
from .schemas import format_errors

# format(request, status, message, param, code, headers) -> response
Formatter = Callable[[Request, int, str, str | None, str | None, dict[str, str]], JSONResponse]

_OPENAI_TYPES = {401: "invalid_request_error", 404: "invalid_request_error", 429: "rate_limit_error"}
_OPENAI_CODES = {401: "invalid_api_key", 429: "rate_limit_exceeded"}


def openai_error(
    request: Request, status: int, message: str, param: str | None, code: str | None, headers: dict[str, str]
) -> JSONResponse:
    request.state.error = message
    etype = _OPENAI_TYPES.get(status, "server_error" if status >= 500 else "invalid_request_error")
    body = {
        "error": {
            "message": message,
            "type": etype,
            "param": param,
            "code": code or _OPENAI_CODES.get(status),
        }
    }
    return JSONResponse(body, status_code=status, headers=headers)


def hf_error(
    request: Request, status: int, message: str, param: str | None, code: str | None, headers: dict[str, str]
) -> JSONResponse:
    request.state.error = message
    return JSONResponse({"error": message}, status_code=status, headers=headers)


def route_class(ctx: AppContext, fmt: Formatter) -> type[APIRoute]:
    """An APIRoute whose failures are rendered by `fmt` (dependencies run inside the handler, so auth too)."""

    class CompatRoute(APIRoute):
        def get_route_handler(self) -> Callable[[Request], Awaitable[Response]]:
            inner = super().get_route_handler()

            async def handler(request: Request) -> Response:
                try:
                    return await inner(request)
                except SchemaError as exc:
                    return fmt(request, 400, str(exc), exc.param, exc.code, {})
                except RequestValidationError as exc:
                    return fmt(request, 400, format_errors(list(exc.errors())), None, None, {})
                except StarletteHTTPException as exc:
                    return fmt(request, exc.status_code, str(exc.detail), None, None, dict(exc.headers or {}))
                except Exception as exc:
                    err = ctx.map_exception(exc)  # ApiError passes through; the rest is mapped / logged
                    return fmt(request, err.status, err.detail, None, None, dict(err.headers))  # type: ignore[attr-defined]

            return handler

    return CompatRoute


async def read_json_object(request: Request) -> dict[str, Any]:
    """Request body as a JSON object, or SchemaError (-> 400 in the client's own error shape)."""
    try:
        body = await request.json()
    except ValueError as exc:
        raise SchemaError("request body is not valid JSON", None) from exc
    if not isinstance(body, dict):
        raise SchemaError("request body must be a JSON object", None)
    return body
