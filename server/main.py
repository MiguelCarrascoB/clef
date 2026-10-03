"""Clef-flash server v2 - SystemOne decision API served locally (AMD ROCm / WSL2).

See docs/ARCHITECTURE.md for the API contract. Run:  python server/main.py   (single worker, one GPU thread).

  POST /v1/systemone  {model?, state, questions, images?, videos?} -> {model, answers, usage, timing}
  POST /v1/batch      {"batch": [<systemone request>, ...]}        -> {batch_ms, results}
  GET  /health /livez /v1/stats /v1/log /v1/events /schema-example   console at /
"""

from __future__ import annotations

import asyncio
import hmac
import json
import logging
import re
import time
import uuid
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager, nullcontext
from pathlib import Path
from typing import Any

import config as config_mod
import uvicorn
from config import VERSION, Config
from engine import Engine, EngineNotReady, GpuOutOfMemory, InputTooLarge
from fastapi import APIRouter, Depends, FastAPI, HTTPException, Query, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from media import MediaError, load_media
from schemas import (
    BatchRequest,
    BatchResponse,
    ErrorBody,
    SystemOneRequest,
    SystemOneResponse,
    check_batch_size,
    check_limits,
    format_errors,
)
from starlette.concurrency import run_in_threadpool
from starlette.exceptions import HTTPException as StarletteHTTPException
from stats import Stats

log = logging.getLogger("clef")
TRACKED = {"/v1/systemone", "/v1/batch"}
SSE_STATS_INTERVAL_S = 2.0
_RID_OK = re.compile(r"^[A-Za-z0-9._-]{1,64}$")
ERRORS: dict[int | str, dict[str, Any]] = {
    c: {"model": ErrorBody, "description": d}
    for c, d in {
        400: "Invalid input",
        401: "Missing/invalid API key",
        413: "Request too large",
        503: "Model not ready or GPU out of memory",
        500: "Internal error",
    }.items()
}


def setup_logging() -> None:
    root = logging.getLogger()
    if not any(getattr(h, "_clef", False) for h in root.handlers):
        handler = logging.StreamHandler()
        handler._clef = True  # type: ignore[attr-defined]
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)-7s %(name)s %(message)s"))
        root.addHandler(handler)
    if root.level in (logging.NOTSET, logging.WARNING):
        root.setLevel(logging.INFO)


class ApiError(Exception):
    def __init__(self, status: int, detail: str, headers: dict[str, str] | None = None):
        super().__init__(detail)
        self.status, self.detail, self.headers = status, detail, headers or {}


class BodyTooLarge(HTTPException):
    """Raised from the streamed-body guard; an HTTPException so FastAPI's body parsing does not turn it into 400."""  # noqa: E501

    def __init__(self, limit_mb: int = 0):
        super().__init__(413, f"request body exceeds {limit_mb} MB limit")


def _rid(request: Request) -> str:
    return getattr(request.state, "request_id", "-")


def _error(request: Request, status: int, detail: str, headers: dict[str, str] | None = None) -> JSONResponse:
    request.state.error = detail
    return JSONResponse({"detail": detail, "request_id": _rid(request)}, status_code=status, headers=headers)


def map_exception(exc: Exception) -> ApiError:
    """Engine/media/validation exceptions -> contract HTTP errors (never leaks a traceback)."""
    if isinstance(exc, ApiError):
        return exc
    if isinstance(exc, GpuOutOfMemory):
        return ApiError(503, str(exc) or "GPU out of memory, retry shortly")
    if isinstance(exc, EngineNotReady):
        return ApiError(503, str(exc) or "model is not ready")
    if isinstance(exc, InputTooLarge):
        return ApiError(413, str(exc) or "input too large")
    if isinstance(exc, (MediaError, ValueError)):
        return ApiError(400, str(exc))
    log.error("unhandled error", exc_info=exc)
    return ApiError(500, "internal server error")


# ------------------------------------------------------------------ ASGI middleware


class GuardMiddleware:
    """Request id, body-size limit (Content-Length + streamed), and request recording for inference routes."""

    def __init__(self, app: Any, cfg: Config, stats: Stats):
        self.app, self.cfg, self.stats = app, cfg, stats
        self.limit = cfg.max_body_mb * 1024 * 1024

    async def __call__(self, scope: dict[str, Any], receive: Callable, send: Callable) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        headers = {k.decode("latin1").lower(): v.decode("latin1") for k, v in scope["headers"]}
        rid = headers.get("x-request-id", "")
        rid = rid if _RID_OK.match(rid) else uuid.uuid4().hex
        state = scope.setdefault("state", {})
        state.update(request_id=rid, t0=time.perf_counter(), rec={}, error=None)
        tracked = scope["method"] == "POST" and scope["path"] in TRACKED
        status = 500
        received = 0

        async def send_wrap(msg: dict[str, Any]) -> None:
            nonlocal status
            if msg["type"] == "http.response.start":
                status = msg["status"]
                msg = {**msg, "headers": [*msg.get("headers", []), (b"x-request-id", rid.encode())]}
            await send(msg)

        async def receive_wrap() -> dict[str, Any]:
            nonlocal received
            msg = await receive()
            if msg["type"] == "http.request":
                received += len(msg.get("body", b""))
                if received > self.limit:
                    raise BodyTooLarge(self.cfg.max_body_mb)
            return msg

        try:
            with self.stats.in_flight() if tracked else nullcontext():
                declared = headers.get("content-length", "")
                if declared.isdigit() and int(declared) > self.limit:
                    state["error"] = msg_ = f"request body exceeds {self.cfg.max_body_mb} MB limit"
                    resp = JSONResponse({"detail": msg_, "request_id": rid}, status_code=413)
                    await resp(scope, receive, send_wrap)
                else:
                    await self.app(scope, receive_wrap, send_wrap)
        finally:
            if tracked:
                self._record(scope["path"], status, state)

    def _record(self, path: str, status: int, state: dict[str, Any]) -> None:
        rec = state["rec"]
        self.stats.record_request(
            {
                "endpoint": path,
                "status": status,
                "ms": round((time.perf_counter() - state["t0"]) * 1000, 1),
                "input_tokens": rec.get("input_tokens", 0),
                "n_records": rec.get("n_records", 0),
                "n_questions": rec.get("n_questions", 0),
                "media": rec.get("media", {"images": 0, "videos": 0}),
                "error": state.get("error"),
                "state_preview": rec.get("state_preview"),
            }
        )


# ------------------------------------------------------------------ SSE


async def sse_events(
    stats: Stats,
    status_fn: Callable[[], dict[str, Any]],
    is_disconnected: Callable[[], Awaitable[bool]],
    interval: float = SSE_STATS_INTERVAL_S,
) -> AsyncIterator[str]:
    """`event: log` per request + `event: stats` every `interval` s; stops when the client goes away."""

    def frame(event: str, data: Any) -> str:
        return f"event: {event}\ndata: {json.dumps(data)}\n\n"

    sub = stats.subscribe()
    task: asyncio.Task[dict[str, Any]] | None = None
    try:
        yield frame("stats", status_fn())
        next_stats = time.monotonic() + interval
        while not await is_disconnected():
            if task is None:
                task = asyncio.ensure_future(sub.__anext__())
            wait = max(0.0, min(next_stats - time.monotonic(), 1.0))
            done, _ = await asyncio.wait({task}, timeout=wait)
            if done:
                entry = task.result()
                task = None
                yield frame("log", entry)
            if time.monotonic() >= next_stats:
                next_stats = time.monotonic() + interval
                yield frame("stats", status_fn())
    finally:
        if task is not None:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
        await sub.aclose()


# ------------------------------------------------------------------ app factory


def create_app(cfg: Config | None = None, engine: Any | None = None) -> FastAPI:
    setup_logging()
    cfg = cfg or config_mod.load()
    stats = Stats(cfg.log_buffer)
    eng = engine if engine is not None else Engine(cfg, stats)
    started = time.time()

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        log.info("starting clef server %s (model load runs in background)", VERSION)
        eng.start()
        try:
            yield
        finally:
            eng.shutdown()

    app = FastAPI(
        title="clef-flash local",
        version=VERSION,
        lifespan=lifespan,
        description=(
            "SystemOne decision API: calibrated probabilities for choice / score / noul questions over any text or "  # noqa: E501
            "JSON state (+ images and videos), in one forward pass. Errors are `{detail, request_id}`."
        ),
    )
    app.state.cfg, app.state.stats, app.state.engine = cfg, stats, eng
    app.add_middleware(GuardMiddleware, cfg=cfg, stats=stats)

    # ---- errors
    @app.exception_handler(ApiError)
    async def _api_error(request: Request, exc: ApiError) -> JSONResponse:
        return _error(request, exc.status, exc.detail, exc.headers)

    @app.exception_handler(RequestValidationError)
    async def _invalid(request: Request, exc: RequestValidationError) -> JSONResponse:
        return _error(request, 400, format_errors(list(exc.errors())))

    @app.exception_handler(StarletteHTTPException)
    async def _http(request: Request, exc: StarletteHTTPException) -> JSONResponse:
        return _error(request, exc.status_code, str(exc.detail), dict(exc.headers or {}))

    @app.exception_handler(Exception)
    async def _unhandled(request: Request, exc: Exception) -> JSONResponse:
        log.error("unhandled error [%s]", _rid(request), exc_info=exc)
        resp = _error(request, 500, "internal server error")
        resp.headers["X-Request-ID"] = _rid(request)  # runs outside GuardMiddleware
        return resp

    # ---- auth
    async def require_key(request: Request) -> None:
        if cfg.api_key is None:
            return
        given = request.headers.get("x-api-key")
        auth = request.headers.get("authorization", "")
        if not given and auth.lower().startswith("bearer "):
            given = auth[7:].strip()
        if not given and request.url.path == "/v1/events":
            given = request.query_params.get("key")
        if not given or not hmac.compare_digest(given.encode(), cfg.api_key.encode()):
            raise ApiError(401, "missing or invalid API key", {"WWW-Authenticate": "Bearer"})

    v1 = APIRouter(prefix="/v1", dependencies=[Depends(require_key)], responses=ERRORS)

    # ---- inference
    def gpu_ready() -> None:
        if eng.status in ("loading", "error"):
            raise EngineNotReady(eng.error or "model is loading")

    async def infer(request: Request, reqs: list[SystemOneRequest], batch: bool) -> list[dict[str, Any]]:
        rec = request.state.rec
        rec["n_records"] = len(reqs)
        rec["n_questions"] = sum(len(r.questions) for r in reqs)
        rec["media"] = {
            "images": sum(len(r.images or []) for r in reqs),
            "videos": sum(len(r.videos or []) for r in reqs),
        }
        if cfg.log_state:
            rec["state_preview"] = json.dumps(reqs[0].state, default=str, ensure_ascii=False)[:200]
        for i, r in enumerate(reqs):
            check_limits(r, cfg, f"batch[{i}]." if batch else "")
        gpu_ready()

        def build() -> list[dict[str, Any]]:
            records = []
            for i, r in enumerate(reqs):
                try:
                    images, videos = load_media(r.images, r.videos, cfg)
                except MediaError as exc:
                    raise MediaError(f"batch[{i}].{exc}" if batch else str(exc)) from exc
                records.append(r.to_record(images, videos))
            return records

        records = await run_in_threadpool(build)
        results = await eng.decide(records)
        rec["input_tokens"] = sum(int(r.get("usage", {}).get("input_tokens", 0)) for r in results)
        return results

    def with_total(res: dict[str, Any], total_ms: float) -> dict[str, Any]:
        return {**res, "timing": {**res.get("timing", {}), "total_ms": round(total_ms, 1)}}

    @v1.post("/systemone", response_model=SystemOneResponse, summary="Decide one request")
    async def systemone(body: SystemOneRequest, request: Request) -> dict[str, Any]:
        t0 = time.perf_counter()
        try:
            results = await infer(request, [body], batch=False)
        except Exception as exc:
            raise map_exception(exc) from exc
        return with_total(results[0], (time.perf_counter() - t0) * 1000)

    @v1.post("/batch", response_model=BatchResponse, summary="Decide many requests (one micro-batched pass)")
    async def batch(body: BatchRequest, request: Request) -> dict[str, Any]:
        t0 = time.perf_counter()
        try:
            check_batch_size(body, cfg)
            results = await infer(request, body.batch, batch=True)
        except Exception as exc:
            raise map_exception(exc) from exc
        ms = (time.perf_counter() - t0) * 1000
        return {"batch_ms": round(ms, 1), "results": [with_total(r, ms) for r in results]}

    # ---- observability
    def info() -> dict[str, Any]:
        try:
            return dict(eng.info())
        except Exception:
            log.exception("engine.info() failed")
            return {}

    def stats_body(window_s: int = 300) -> dict[str, Any]:
        body = stats.snapshot(window_s)
        gpu = info().get("gpu") or {}
        body["status"] = eng.status
        body["gpu"] = {k: gpu.get(k) for k in ("vram_allocated_gb", "vram_reserved_gb", "vram_total_gb")}
        return body

    @v1.get("/stats", summary="Rolling latency / throughput / GPU stats")
    async def v1_stats(window_s: int = Query(300, ge=5, le=3600)) -> dict[str, Any]:
        return stats_body(window_s)

    @v1.get("/log", summary="Recent request log (newest last)")
    async def v1_log(
        limit: int = Query(100, ge=1, le=1000), since: int | None = Query(None, ge=0)
    ) -> dict[str, Any]:
        return {"entries": stats.log(limit, since)}

    @v1.get("/events", summary="Server-sent events: request log + stats every 2 s")
    async def v1_events(request: Request) -> StreamingResponse:
        return StreamingResponse(
            sse_events(stats, stats_body, request.is_disconnected),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    app.include_router(v1)

    @app.get("/health", summary="Model + GPU status (503 until ready/warming)")
    async def health() -> JSONResponse:
        status = eng.status
        ok = status in ("ready", "warming")
        body = {
            "ready": ok,
            "status": status,
            "version": VERSION,
            **info(),
            "load_seconds": eng.load_seconds,
            "uptime_s": int(time.time() - started),
            "error": eng.error,
            "limits": cfg.public_limits(),
        }
        return JSONResponse(body, status_code=200 if ok else 503)

    @app.get("/livez", summary="Process liveness")
    async def livez() -> dict[str, bool]:
        return {"ok": True}

    @app.get("/schema-example", summary="Copy-paste ready request bodies")
    async def schema_example() -> dict[str, Any]:
        return {
            "text_json": {
                "model": "clef-flash",
                "state": {"ticket": {"text": "Checkout errors, orders blocked.", "customers_affected": 1200}},
                "questions": {
                    "department": {
                        "type": "choice",
                        "instructions": "Which team should handle the message?",
                        "criteria": {"billing": "Payments or invoices", "technical": "Bugs or outages"},
                    },
                    "urgency": {"type": "score", "criteria": ["Can wait", "This week", "Today"]},
                    "outage": {"type": "noul", "instructions": "Is a service down?"},
                },
            },
            "media": {
                "model": "clef-flash",
                "state": "Review the attached receipt.",
                "images": ["data:image/jpeg;base64,<base64 bytes of the image>"],
                "videos": ["data:video/mp4;base64,<base64 bytes of the video>"],
                "questions": {"legible": {"type": "noul", "instructions": "Is the total legible?"}},
                "note": (
                    f"All media of a request goes into ONE record. Limits: {cfg.max_images} images, "
                    f"{cfg.max_videos} videos; images > {cfg.max_pixels} px are downscaled; videos sampled at "  # noqa: E501
                    f"{cfg.video_fps} fps, max {cfg.max_frames} frames. http(s) URLs only with CLEF_ALLOW_URL_FETCH=1."  # noqa: E501
                ),
            },
            "batch": {"batch": ["<1..N systemone request objects, media allowed>"]},
            "auth": "If CLEF_API_KEY is set send X-API-Key: <key> or Authorization: Bearer <key> on /v1/*.",
            "note": "clef returns CALIBRATED PROBABILITIES for every option in one forward pass - no text generation.",  # noqa: E501
        }

    # console: must be the LAST route so API routes win
    static = Path(__file__).parent / "static"
    if static.is_dir():
        app.mount("/", StaticFiles(directory=static, html=True), name="console")
    return app


app = create_app()

if __name__ == "__main__":
    _cfg: Config = app.state.cfg
    uvicorn.run(app, host=_cfg.host, port=_cfg.port, workers=1)
