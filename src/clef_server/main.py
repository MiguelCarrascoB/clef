"""Clef-flash server v3 - SystemOne decision + classification API served locally (CUDA, ROCm, MPS or CPU).

See docs/ARCHITECTURE.md for the API contract. Run: clef serve (one uvicorn worker, one GPU thread).

  POST /v1/systemone  {model?, state, questions, images?, videos?} -> {model, answers, usage, timing}
  POST /v1/batch      {"batch": [<systemone request>, ...]}        -> {batch_ms, results}
  POST /v1/classify[/batch], /v1/score, /v1/classifiers/{name}[/batch]; PUT|GET|DELETE /v1/classifiers[/n]
  GET  /health /livez /readyz /v1/stats /v1/stats/timeseries /v1/log /v1/events /schema-example   console at /
"""

from __future__ import annotations

import asyncio
import functools
import importlib
import json
import logging
import re
import signal
import sys
import threading
import time
import uuid
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager, nullcontext, suppress
from pathlib import Path
from typing import Any

import uvicorn
from fastapi import APIRouter, Depends, FastAPI, HTTPException, Query, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from sqlalchemy.exc import OperationalError
from starlette.concurrency import run_in_threadpool
from starlette.exceptions import HTTPException as StarletteHTTPException

from . import config as config_mod
from .appctx import AppContext
from .classifiers import ClassifierStore, ClassifierVersions, InvalidName, TooManyClassifiers, validate_name
from .classify import (
    ClassifierBatchRequest,
    ClassifierDef,
    ClassifierDeleted,
    ClassifierInfo,
    ClassifierList,
    ClassifierRunRequest,
    ClassifyBatchRequest,
    ClassifyBatchResponse,
    ClassifyRequest,
    ClassifyResponse,
    ScoreRequest,
    ScoreResponse,
    check_label_count,
    check_levels_count,
    classify_result,
    classify_to_systemone,
    resolve_threshold,
    score_result,
    score_to_systemone,
)
from .config import VERSION, Config
from .db import DatabaseError, is_unavailable, open_database
from .engine import Engine, EngineNotReady, GpuOutOfMemory, InputTooLarge, Prepared
from .keystore import KeyCache, QuotaExceeded, Quotas
from .logs import install_access_filter, request_id_var, setup_logging, uvicorn_log_config
from .media import MediaError, load_media
from .models import UnknownModel
from .models.registry import served_model
from .schemas import (
    BatchRequest,
    BatchResponse,
    ErrorBody,
    SystemOneRequest,
    SystemOneResponse,
    check_batch_size,
    check_limits,
    format_errors,
)
from .security import (
    Authenticator,
    KeyAuth,
    RateLimiter,
    bind_warning,
    client_identity,
    install_cors,
    is_tracked,
)
from .stats import Stats

log = logging.getLogger("clef")
SSE_STATS_INTERVAL_S = 2.0
_RID_OK = re.compile(r"^[A-Za-z0-9._-]{1,64}$")
ERRORS: dict[int | str, dict[str, Any]] = {
    c: {"model": ErrorBody, "description": d}
    for c, d in {
        400: "Invalid input",
        401: "Missing/invalid API key",
        403: "The key lacks the route's scope, or the classifier belongs to another key",
        404: "Unknown classifier, or a model this server does not serve",
        409: "Too many saved classifiers",
        413: "Request too large",
        429: "Rate limit or daily quota exceeded (see Retry-After)",
        503: "Model not ready or GPU out of memory",
        500: "Internal error",
    }.items()
}
# Feature modules: each exposes router(ctx: AppContext) -> APIRouter (see appctx.py). One line each.
FEATURES: list[str] = ["evaluation", "compat", "batches", "jobs", "admin", "router"]


class ApiError(Exception):
    def __init__(
        self, status: int, detail: str, headers: dict[str, str] | None = None, code: str | None = None
    ):
        super().__init__(detail)
        self.status, self.detail, self.headers = status, detail, headers or {}
        self.code = code  # a client-visible error code (the OpenAI routes render it), e.g. insufficient_quota


class BodyTooLarge(HTTPException):
    """Raised from the streamed-body guard; an HTTPException so FastAPI's body parsing does not turn it into 400."""  # noqa: E501

    def __init__(self, limit_mb: int = 0):
        super().__init__(413, f"request body exceeds {limit_mb} MB limit")


def _rid(request: Request) -> str:
    return getattr(request.state, "request_id", "-")


def _error(request: Request, status: int, detail: str, headers: dict[str, str] | None = None) -> JSONResponse:
    request.state.error = detail
    return JSONResponse({"detail": detail, "request_id": _rid(request)}, status_code=status, headers=headers)


def wants_no_cache(request: Request) -> bool:
    """`Cache-Control: no-cache` (the token anywhere in the header, any case): skip the cache lookup."""
    header = request.headers.get("cache-control") or ""
    return any(part.strip().split("=")[0].strip().lower() == "no-cache" for part in header.split(","))


def map_exception(exc: Exception) -> ApiError:
    """Engine/media/validation exceptions -> contract HTTP errors (never leaks a traceback)."""
    if isinstance(exc, ApiError):
        return exc
    if isinstance(exc, UnknownModel):
        return ApiError(404, str(exc))
    if isinstance(exc, GpuOutOfMemory):
        return ApiError(503, str(exc) or "GPU out of memory, retry shortly")
    if isinstance(exc, EngineNotReady):
        return ApiError(503, str(exc) or "model is not ready")
    if isinstance(exc, InputTooLarge):
        return ApiError(413, str(exc) or "input too large")
    if isinstance(exc, TooManyClassifiers):
        return ApiError(409, str(exc))
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
        self.file_limit = cfg.max_file_mb * 1024 * 1024  # F1: POST /v1/files only

    def limit_for(self, scope: dict[str, Any]) -> tuple[int, int]:
        """(bytes, MB) allowed for this request: the file limit for the upload route, else the body limit."""
        if scope["method"] == "POST" and scope["path"] == "/v1/files":
            return self.file_limit, self.cfg.max_file_mb
        return self.limit, self.cfg.max_body_mb

    async def __call__(self, scope: dict[str, Any], receive: Callable, send: Callable) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        headers = {k.decode("latin1").lower(): v.decode("latin1") for k, v in scope["headers"]}
        rid = headers.get("x-request-id", "")
        rid = rid if _RID_OK.match(rid) else uuid.uuid4().hex
        state = scope.setdefault("state", {})
        state.update(request_id=rid, t0=time.perf_counter(), rec={}, error=None)
        rid_token = request_id_var.set(rid)  # log records (incl. uvicorn's access line) carry it in json mode
        tracked = is_tracked(scope["method"], scope["path"])
        status = 500
        received = 0
        limit, limit_mb = self.limit_for(scope)

        async def send_wrap(msg: dict[str, Any]) -> None:
            nonlocal status
            if msg["type"] == "http.response.start":
                status = msg["status"]
                extra = [(b"x-request-id", rid.encode())]
                quota = state.get("quota")  # (limit, remaining) of a quota'd principal: infer / job submit
                if quota:
                    extra.append((b"x-quota-limit", str(quota[0]).encode()))
                    extra.append((b"x-quota-remaining", str(quota[1]).encode()))
                msg = {**msg, "headers": [*msg.get("headers", []), *extra]}
            await send(msg)

        async def receive_wrap() -> dict[str, Any]:
            nonlocal received
            msg = await receive()
            if msg["type"] == "http.request":
                received += len(msg.get("body", b""))
                if received > limit:
                    raise BodyTooLarge(limit_mb)
            return msg

        try:
            with self.stats.in_flight() if tracked else nullcontext():
                declared = headers.get("content-length", "")
                if declared.isdigit() and int(declared) > limit:
                    state["error"] = msg_ = f"request body exceeds {limit_mb} MB limit"
                    if scope["path"] == "/v1/files":  # the OpenAI SDKs parse this error shape
                        err = {"message": msg_, "type": "invalid_request_error", "param": None, "code": None}
                        resp = JSONResponse({"error": err}, status_code=413)
                    else:
                        resp = JSONResponse({"detail": msg_, "request_id": rid}, status_code=413)
                    await resp(scope, receive, send_wrap)
                else:
                    await self.app(scope, receive_wrap, send_wrap)
        finally:
            request_id_var.reset(rid_token)
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
                "key": state.get("key_name"),
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
    cfg = cfg or config_mod.load()
    setup_logging(cfg.log_format)
    stats = Stats(cfg.log_buffer)
    eng = engine if engine is not None else Engine(cfg, stats)
    started = time.time()
    limiter = RateLimiter(cfg.rate_limit)
    db = open_database(cfg)  # the one Database of this process (ctx.db); closed at the end of the lifespan
    store = ClassifierStore(db, cfg.max_classifiers)
    # D1: keys and quotas load now, so auth is right from the first request; their tasks start in the lifespan
    keycache = KeyCache(db)
    keycache.reload()
    quotas = Quotas(db)
    quotas.load()
    auth = Authenticator(cfg.api_keys(), keycache)

    async def sampler() -> None:
        """Feed the time series every sample_interval_s: engine.sample() (in a thread) + queue depth."""
        interval = max(0.2, float(cfg.sample_interval_s))
        sample_fn = getattr(eng, "sample", None)  # test engines may not have it
        while True:
            try:
                gauges: dict[str, Any] = {}
                if callable(sample_fn):
                    gauges = dict(await asyncio.to_thread(sample_fn) or {})
                gauges["queue_depth"] = stats.queue_depth
                stats.sample(gauges)
            except asyncio.CancelledError:
                raise
            except Exception:
                log.debug("gauge sampling failed", exc_info=True)
            await asyncio.sleep(interval)

    @asynccontextmanager
    async def lifespan(app_: FastAPI) -> AsyncIterator[None]:
        install_access_filter()  # after uvicorn's dictConfig; covers launchers other than run()
        log.info("starting clef server %s (model load runs in background)", VERSION)
        for line in (bind_warning(cfg, auth.enabled) or "").splitlines():  # needs the DB: DB keys count
            log.warning(line)
        app_.state.draining = False
        eng.start()
        db.start_heartbeat()  # Postgres: exit (status 3) if the one-server lock is ever lost
        task = asyncio.create_task(sampler())
        for hook in ctx.on_startup:
            await hook()
        try:
            yield
        finally:
            app_.state.draining = True  # other launchers; under run() DrainingServer set it already
            for hook in reversed(ctx.on_shutdown):
                try:
                    await hook()
                except Exception:
                    log.exception("shutdown hook failed")
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
            eng.shutdown()
            await asyncio.to_thread(db.close)

    app = FastAPI(
        title="clef local",
        version=VERSION,
        lifespan=lifespan,
        description=(
            "SystemOne decision API: calibrated probabilities for choice / score / noul questions over any text or "  # noqa: E501
            "JSON state (+ images and videos), in one forward pass, plus a classification layer (classify, score, "  # noqa: E501
            "saved classifiers). Errors are `{detail, request_id}`."
        ),
    )
    app.state.cfg, app.state.stats, app.state.engine = cfg, stats, eng
    app.state.draining = False
    app.add_middleware(GuardMiddleware, cfg=cfg, stats=stats)
    install_cors(app, cfg)  # added last = outermost, so preflights never reach auth

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

    @app.exception_handler(DatabaseError)
    @app.exception_handler(OperationalError)
    async def _db_unavailable(request: Request, exc: Exception) -> JSONResponse:
        log.error("database unavailable [%s]: %s", _rid(request), exc)
        return _error(request, 503, "database unavailable", {"Retry-After": "5"})

    @app.exception_handler(Exception)
    async def _unhandled(request: Request, exc: Exception) -> JSONResponse:
        log.error("unhandled error [%s]", _rid(request), exc_info=exc)
        resp = _error(request, 500, "internal server error")
        resp.headers["X-Request-ID"] = _rid(request)  # runs outside GuardMiddleware
        return resp

    # ---- auth
    async def require_key(request: Request) -> None:
        if not auth.enabled:
            request.state.key_name = request.state.principal = None
            return
        principal = auth.resolve(KeyAuth.extract(request))
        if principal is None:
            raise ApiError(401, "missing or invalid API key", {"WWW-Authenticate": "Bearer"})
        request.state.principal = principal
        request.state.key_name = principal.name

    def scope(*names: str) -> list[Any]:
        """Dependencies for a route: a valid key (this runs ``require_key``) with every scope in ``names``.

        No names = any valid key. ``admin`` routes need a key even when auth is off. The ``clef_scope``
        attribute is the marker the default-deny test looks for.
        """

        async def check_scope(request: Request, _: None = Depends(require_key)) -> None:
            principal = request.state.principal
            if principal is None:
                if "admin" in names:
                    raise ApiError(
                        401,
                        "admin routes need a key: set CLEF_API_KEY or run clef keys create",
                        {"WWW-Authenticate": "Bearer"},
                    )
                return
            for name in names:
                if not principal.has(name):
                    raise ApiError(403, f"key {principal.name!r} lacks scope {name!r}")

        check_scope.clef_scope = names  # type: ignore[attr-defined]
        return [Depends(check_scope)]

    async def rate_limit(request: Request) -> None:
        principal = getattr(request.state, "principal", None)
        limit = principal.rate_limit if principal is not None and principal.rate_limit else cfg.rate_limit
        if limit <= 0:
            return
        retry = limiter.check(client_identity(request, getattr(request.state, "key_name", None)), limit)
        if retry:
            raise ApiError(429, f"rate limit exceeded ({limit}/min)", {"Retry-After": str(retry)})

    limited = [Depends(rate_limit)]
    classify_scope = scope("classify")
    manage_scope = scope("classifiers")
    ops_scope = scope("ops")
    v1 = APIRouter(prefix="/v1", dependencies=[Depends(require_key)], responses=ERRORS)

    # ---- inference
    def gpu_ready() -> None:
        if eng.status in ("loading", "error"):
            raise EngineNotReady(eng.error or "model is loading")

    async def infer(
        request: Request, reqs: list[SystemOneRequest], batch: bool, label: str = "batch"
    ) -> list[dict[str, Any]]:
        """The one inference path: limits, media load (threadpool), ONE engine.decide() for all records."""
        rec = request.state.rec
        principal = getattr(request.state, "principal", None)
        try:  # reserved up front (429 before any work); cache hits are handed back below
            reservation = quotas.reserve(principal, len(reqs))
        except QuotaExceeded as exc:
            request.state.quota = (exc.limit, exc.remaining)
            raise ApiError(
                429, str(exc), {"Retry-After": str(exc.retry_after)}, code="insufficient_quota"
            ) from exc
        charged = len(reqs)
        request.state.quota = quotas.remaining(principal)
        rec["n_records"] = len(reqs)
        rec["n_questions"] = sum(len(r.questions) for r in reqs)
        rec["media"] = {
            "images": sum(len(r.images or []) for r in reqs),
            "videos": sum(len(r.videos or []) for r in reqs),
        }
        if cfg.log_state:
            rec["state_preview"] = json.dumps(reqs[0].state, default=str, ensure_ascii=False)[:200]
        cache = not wants_no_cache(request)
        try:
            prepared = await prepare(reqs, batch=batch, label=label, cache=cache)
            if reservation is not None and prepared.hits:  # hits cost no GPU time: not charged
                quotas.refund(reservation, len(prepared.hits))
                charged -= len(prepared.hits)
            results = await decide_prepared(prepared, priority="interactive")
        except BaseException:
            if reservation is not None:  # the request failed: charge nothing
                quotas.refund(reservation, charged)
            raise
        finally:
            request.state.quota = quotas.remaining(principal)
        rec["input_tokens"] = sum(int(r.get("usage", {}).get("input_tokens", 0)) for r in results)
        return results

    async def decide(
        reqs: list[SystemOneRequest],
        batch: bool = True,
        label: str = "batch",
        priority: str = "background",
        cache: bool = True,
    ) -> list[dict[str, Any]]:
        """Limits, media load (threadpool), ONE engine call. No request needed (background work).

        ``priority`` picks the engine lane: background by default (jobs), interactive from ``infer``.
        = decide_prepared(await prepare(...)).
        """
        prepared = await prepare(reqs, batch=batch, label=label, cache=cache)
        return await decide_prepared(prepared, priority=priority)

    async def prepare(
        reqs: list[SystemOneRequest], batch: bool = True, label: str = "batch", cache: bool = True
    ) -> Prepared:
        """Limits, the model check (404 before any inference), media load (threadpool) and engine.prepare
        (capabilities, cache lookup, tokenizing in threads). Nothing is queued until decide_prepared."""
        ids = []
        for i, r in enumerate(reqs):
            check_limits(r, cfg, f"{label}[{i}]." if batch else "")
            ids.append(served_model(eng, r.model))
        gpu_ready()
        validate = getattr(eng, "validate", None)  # capabilities, before any media is decoded (fakes: none)
        if callable(validate):
            for i, r in enumerate(reqs):
                try:
                    validate({**r.to_record(r.images, r.videos), "model": ids[i]})
                except ValueError as exc:
                    if not batch:
                        raise
                    raise ValueError(f"{label}[{i}].{exc}") from exc

        def build() -> list[dict[str, Any]]:
            records = []
            for i, r in enumerate(reqs):
                try:
                    images, videos = load_media(r.images, r.videos, cfg)
                except MediaError as exc:
                    raise MediaError(f"{label}[{i}].{exc}" if batch else str(exc)) from exc
                records.append({**r.to_record(images, videos), "model": ids[i]})
            return records

        records = await run_in_threadpool(build)
        if callable(getattr(eng, "prepare", None)):
            return await eng.prepare(records, cache=cache)
        return Prepared(records, [None] * len(records), {}, [None] * len(records))  # test engines

    async def decide_prepared(prepared: Prepared, priority: str = "background") -> list[dict[str, Any]]:
        """Queue prepared records in the engine lane `priority` and await their results (input order)."""
        if callable(getattr(eng, "submit", None)):
            return await eng.submit(prepared, priority=priority)
        return await eng.decide(prepared.records, priority=priority)  # test engines

    def with_total(res: dict[str, Any], total_ms: float) -> dict[str, Any]:
        return {**res, "timing": {**res.get("timing", {}), "total_ms": round(total_ms, 1)}}

    @v1.post(
        "/systemone",
        response_model=SystemOneResponse,
        summary="Decide one request",
        dependencies=[*classify_scope, *limited],
    )
    async def systemone(body: SystemOneRequest, request: Request) -> dict[str, Any]:
        t0 = time.perf_counter()
        try:
            results = await infer(request, [body], batch=False)
        except Exception as exc:
            raise map_exception(exc) from exc
        return with_total(results[0], (time.perf_counter() - t0) * 1000)

    @v1.post(
        "/batch",
        response_model=BatchResponse,
        summary="Decide many requests (one micro-batched pass)",
        dependencies=[*classify_scope, *limited],
    )
    async def batch(body: BatchRequest, request: Request) -> dict[str, Any]:
        t0 = time.perf_counter()
        try:
            check_batch_size(body, cfg)
            results = await infer(request, body.batch, batch=True)
        except Exception as exc:
            raise map_exception(exc) from exc
        ms = (time.perf_counter() - t0) * 1000
        return {"batch_ms": round(ms, 1), "results": [with_total(r, ms) for r in results]}

    # ---- classification (thin layer over the same infer path)
    def usage_timing(res: dict[str, Any], total_ms: float) -> dict[str, Any]:
        out = with_total(res, total_ms)
        return {"usage": out.get("usage", {}), "timing": out["timing"]}

    def classify_one(
        res: dict[str, Any], labels: Any, multi: bool, thr: float | None, model: str | None, ms: float
    ) -> dict[str, Any]:
        served = res.get("model") or model  # the engine reports the served id
        return {**classify_result(res["answers"], labels, multi, thr, served), **usage_timing(res, ms)}

    async def run_classify(
        request: Request,
        *,
        state: Any,
        labels: Any,
        instructions: str | None,
        multi_label: bool,
        threshold: float | None,
        model: str | None,
        images: list[str] | None,
        videos: list[str] | None,
        classifier: str | None = None,
    ) -> dict[str, Any]:
        t0 = time.perf_counter()
        try:
            check_label_count(labels, multi_label, cfg.max_labels)
            thr = resolve_threshold(threshold, cfg) if multi_label else None
            sreq = classify_to_systemone(state, labels, instructions, multi_label, model, images, videos)
            results = await infer(request, [sreq], batch=False)
        except Exception as exc:
            raise map_exception(exc) from exc
        out = classify_one(results[0], labels, multi_label, thr, model, (time.perf_counter() - t0) * 1000)
        out["request_id"] = _rid(request)
        if classifier:
            out["classifier"] = classifier
        return out

    async def run_classify_batch(
        request: Request,
        *,
        inputs: list[Any],
        labels: Any,
        instructions: str | None,
        multi_label: bool,
        threshold: float | None,
        model: str | None,
        classifier: str | None = None,
    ) -> dict[str, Any]:
        t0 = time.perf_counter()
        try:
            if len(inputs) > cfg.max_batch:
                raise ValueError(f"inputs: too many inputs (max {cfg.max_batch})")
            check_label_count(labels, multi_label, cfg.max_labels)
            thr = resolve_threshold(threshold, cfg) if multi_label else None
            sreqs = [classify_to_systemone(x, labels, instructions, multi_label, model) for x in inputs]
            results = await infer(request, sreqs, batch=True, label="inputs")  # ONE engine.decide()
        except Exception as exc:
            raise map_exception(exc) from exc
        ms = (time.perf_counter() - t0) * 1000
        out: dict[str, Any] = {
            "batch_ms": round(ms, 1),
            "results": [classify_one(r, labels, multi_label, thr, model, ms) for r in results],
            "request_id": _rid(request),
        }
        if classifier:
            out["classifier"] = classifier
        return out

    async def run_score(
        request: Request,
        *,
        state: Any,
        levels: list[str],
        instructions: str | None,
        model: str | None,
        images: list[str] | None,
        videos: list[str] | None,
        classifier: str | None = None,
    ) -> dict[str, Any]:
        t0 = time.perf_counter()
        try:
            check_levels_count(levels, cfg.max_labels)
            sreq = score_to_systemone(state, levels, instructions, model, images, videos)
            results = await infer(request, [sreq], batch=False)
        except Exception as exc:
            raise map_exception(exc) from exc
        out = {
            **score_result(results[0]["answers"], levels, results[0].get("model") or model),
            **usage_timing(results[0], (time.perf_counter() - t0) * 1000),
            "request_id": _rid(request),
        }
        if classifier:
            out["classifier"] = classifier
        return out

    @v1.post(
        "/classify",
        response_model=ClassifyResponse,
        response_model_exclude_none=True,
        summary="Classify one input (single or multi-label)",
        dependencies=[*classify_scope, *limited],
    )
    async def classify(body: ClassifyRequest, request: Request) -> dict[str, Any]:
        return await run_classify(
            request,
            state=body.input,
            labels=body.labels,
            instructions=body.instructions,
            multi_label=body.multi_label,
            threshold=body.threshold,
            model=body.model,
            images=body.images,
            videos=body.videos,
        )

    @v1.post(
        "/classify/batch",
        response_model=ClassifyBatchResponse,
        response_model_exclude_none=True,
        summary="Classify many inputs in one micro-batched pass",
        dependencies=[*classify_scope, *limited],
    )
    async def classify_batch(body: ClassifyBatchRequest, request: Request) -> dict[str, Any]:
        return await run_classify_batch(
            request,
            inputs=body.inputs,
            labels=body.labels,
            instructions=body.instructions,
            multi_label=body.multi_label,
            threshold=body.threshold,
            model=body.model,
        )

    @v1.post(
        "/score",
        response_model=ScoreResponse,
        response_model_exclude_none=True,
        summary="Place one input on an ordinal scale",
        dependencies=[*classify_scope, *limited],
    )
    async def score(body: ScoreRequest, request: Request) -> dict[str, Any]:
        return await run_score(
            request,
            state=body.input,
            levels=body.levels,
            instructions=body.instructions,
            model=body.model,
            images=body.images,
            videos=body.videos,
        )

    # ---- saved classifiers
    def checked_name(name: str) -> str:
        try:
            return validate_name(name)
        except InvalidName as exc:
            raise ApiError(400, str(exc)) from exc

    async def load_classifier(name: str, version: int | None = None) -> dict[str, Any]:
        checked_name(name)
        doc = await run_in_threadpool(store.get, name, version)
        if doc is None:
            if version is not None and await run_in_threadpool(store.get, name) is not None:
                raise ApiError(404, f"classifier {name!r} has no version {version}")
            raise ApiError(404, f"classifier {name!r} not found")
        return doc

    def store_error(exc: Exception) -> Exception:
        if is_unavailable(exc):
            log.error("database unavailable: %s", exc)
            return ApiError(503, "database unavailable", {"Retry-After": "5"})
        return map_exception(exc)

    def is_admin(request: Request) -> bool:
        """The owner check's override (decision 2): an admin principal; auth off (no principal) is admin."""
        principal = getattr(request.state, "principal", None)
        return True if principal is None else bool(principal.has("admin"))

    def owner_check(request: Request, name: str) -> Callable[[bool, str | None], None]:
        """403 unless the caller owns ``name`` or is admin (auth off counts as admin). A classifier nobody
        owns (created while auth was off) is admin-only to change once auth is on. The owner is never named:
        admins can read it from the classifier itself and never get this error. Runs inside put's transaction
        (atomic with the append) or, for DELETE, on ``store.owner_of``."""
        caller = getattr(request.state, "key_name", None)

        def check(exists: bool, owner: str | None) -> None:
            if exists and owner != caller and not is_admin(request):
                raise ApiError(
                    403, f"classifier {name!r} is not yours; only its owner or an admin may change it"
                )

        return check

    async def check_owner(request: Request, name: str) -> None:
        owner_check(request, name)(*await run_in_threadpool(store.owner_of, name))

    owner_errors: dict[int | str, dict[str, Any]] = {
        403: {"model": ErrorBody, "description": "Not the classifier's owner"}
    }

    @v1.put(
        "/classifiers/{name}",
        response_model=ClassifierInfo,
        summary="Create a saved classifier or add a version",
        responses=owner_errors,
        dependencies=manage_scope,
    )
    async def put_classifier(name: str, body: ClassifierDef, request: Request) -> dict[str, Any]:
        checked_name(name)
        caller = getattr(request.state, "key_name", None)
        try:
            if body.kind == "classify":
                check_label_count(body.labels or [], body.multi_label, cfg.max_labels)
            else:
                check_levels_count(body.levels or [], cfg.max_labels)
            return await run_in_threadpool(
                functools.partial(
                    store.put,
                    name,
                    # review_threshold joins the stored definition only when set, so an unchanged re-PUT of an
                    # older classifier stays a no-op
                    body.model_dump(
                        exclude=None if body.review_threshold is not None else {"review_threshold"}
                    ),
                    owner=caller,
                    created_by=caller,
                    model=ctx.model_ref(),
                    check_owner=owner_check(request, name),  # in put's transaction: atomic with the append
                    actor=caller or "anonymous",
                    ip=request.client.host if request.client else None,
                    request_id=_rid(request),
                )
            )
        except Exception as exc:
            raise store_error(exc) from exc

    @v1.get(
        "/classifiers",
        response_model=ClassifierList,
        summary="List saved classifiers",
        dependencies=classify_scope,
    )
    async def list_classifiers() -> dict[str, Any]:
        return {"classifiers": await run_in_threadpool(store.list)}

    @v1.get(
        "/classifiers/{name}",
        response_model=ClassifierInfo,
        summary="Get a saved classifier",
        dependencies=classify_scope,
    )
    async def get_classifier(
        name: str, version: int | None = Query(None, ge=1, description="default: the latest")
    ) -> dict[str, Any]:
        return await load_classifier(name, version)

    @v1.get(
        "/classifiers/{name}/versions",
        response_model=ClassifierVersions,
        summary="Versions of a saved classifier (newest first)",
        dependencies=classify_scope,
    )
    async def classifier_versions(name: str) -> dict[str, Any]:
        checked_name(name)
        versions = await run_in_threadpool(store.versions, name)
        if versions is None:
            raise ApiError(404, f"classifier {name!r} not found")
        return {"name": name, "versions": versions}

    @v1.get(
        "/classifiers/{name}/versions/{version}",
        response_model=ClassifierInfo,
        summary="One version of a saved classifier",
        dependencies=classify_scope,
    )
    async def classifier_version(name: str, version: int) -> dict[str, Any]:
        return await load_classifier(name, version)

    @v1.delete(
        "/classifiers/{name}",
        response_model=ClassifierDeleted,
        summary="Delete a saved classifier (every version)",
        responses=owner_errors,
        dependencies=manage_scope,
    )
    async def delete_classifier(name: str, request: Request) -> dict[str, str]:
        checked_name(name)
        await check_owner(request, name)
        deleted = await run_in_threadpool(
            functools.partial(
                store.delete,
                name,
                actor=getattr(request.state, "key_name", None) or "anonymous",
                ip=request.client.host if request.client else None,
                request_id=_rid(request),
            )
        )
        if not deleted:
            raise ApiError(404, f"classifier {name!r} not found")
        return {"deleted": name}

    async def capture_review(name: str, doc: dict[str, Any], inputs: list[Any], results: list[Any]) -> None:
        """E2: single-label results below review_threshold become Review items (router.py)."""
        capture = ctx.extra.get("review_capture")  # set by the router feature; ctx exists by request time
        if capture is not None and doc.get("review_threshold") is not None:
            await capture(name, doc, inputs, results)

    def run_threshold(doc: dict[str, Any], given: float | None) -> float | None:
        if not doc.get("multi_label"):
            if given is not None:
                raise ApiError(400, "threshold: only allowed for multi-label classifiers")
            return None
        return given if given is not None else doc.get("threshold")

    @v1.post(
        "/classifiers/{name}",
        response_model=ClassifyResponse | ScoreResponse,
        response_model_exclude_none=True,
        summary="Run a saved classifier on one input",
        dependencies=[*classify_scope, *limited],
    )
    async def run_classifier(name: str, body: ClassifierRunRequest, request: Request) -> dict[str, Any]:
        doc = await load_classifier(name, body.version)
        common = {"state": body.input, "images": body.images, "videos": body.videos, "classifier": name}
        if doc["kind"] == "score":
            if body.threshold is not None:
                raise ApiError(400, "threshold: not allowed for score classifiers")
            return await run_score(
                request,
                levels=doc["levels"],
                instructions=doc.get("instructions"),
                model=body.model,
                **common,
            )
        out = await run_classify(
            request,
            labels=doc["labels"],
            instructions=doc.get("instructions"),
            multi_label=bool(doc.get("multi_label")),
            threshold=run_threshold(doc, body.threshold),
            model=body.model,
            **common,
        )
        if not body.images and not body.videos:  # E2: results below review_threshold go to the Review queue
            await capture_review(name, doc, [body.input], [out])
        return out

    @v1.post(
        "/classifiers/{name}/batch",
        response_model=ClassifyBatchResponse,
        response_model_exclude_none=True,
        summary="Run a saved classifier on many inputs (one micro-batched pass)",
        dependencies=[*classify_scope, *limited],
    )
    async def run_classifier_batch(
        name: str, body: ClassifierBatchRequest, request: Request
    ) -> dict[str, Any]:
        doc = await load_classifier(name, body.version)
        if doc["kind"] == "score":
            raise ApiError(400, "batch is only supported for classify classifiers")
        out = await run_classify_batch(
            request,
            inputs=body.inputs,
            labels=doc["labels"],
            instructions=doc.get("instructions"),
            multi_label=bool(doc.get("multi_label")),
            threshold=run_threshold(doc, body.threshold),
            model=body.model,
            classifier=name,
        )
        await capture_review(name, doc, body.inputs, out["results"])
        return out

    # ---- observability
    def model_info() -> dict[str, Any]:
        """The served model (engine.model_info(): id, adapter, revision, ref, capabilities, distribution)."""
        fn = getattr(eng, "model_info", None)
        if callable(fn):
            try:
                return dict(fn())
            except Exception:
                log.exception("engine.model_info() failed")
        model_id = served_model(eng, None)
        return {"id": model_id, "adapter": None, "revision": None, "ref": f"{model_id}@local"}

    def info() -> dict[str, Any]:
        try:
            return dict(eng.info())
        except Exception:
            log.exception("engine.info() failed")
            return {}

    def stats_body(window_s: int = 300) -> dict[str, Any]:
        body = stats.snapshot(window_s)
        meta = info()
        gpu = meta.get("gpu") or {}
        body["status"] = eng.status
        body["gpu"] = {
            k: gpu.get(k) for k in ("vram_allocated_gb", "vram_reserved_gb", "vram_total_gb", "memory_kind")
        }
        body["telemetry"] = meta.get("telemetry")
        return body

    @v1.get("/stats", summary="Rolling latency / throughput / GPU stats", dependencies=ops_scope)
    async def v1_stats(window_s: int = Query(300, ge=5, le=3600)) -> dict[str, Any]:
        return stats_body(window_s)

    @v1.get(
        "/stats/timeseries", summary="Columnar time series for charts (oldest first)", dependencies=ops_scope
    )
    async def v1_timeseries(
        window_s: int = Query(300, ge=10, le=3600), step_s: int | None = Query(None, ge=1)
    ) -> dict[str, Any]:
        try:
            return stats.timeseries(window_s, step_s)
        except ValueError as exc:
            raise ApiError(400, str(exc)) from exc

    @v1.get("/log", summary="Recent request log (newest last)", dependencies=ops_scope)
    async def v1_log(
        limit: int = Query(100, ge=1, le=1000), since: int | None = Query(None, ge=0)
    ) -> dict[str, Any]:
        return {"entries": stats.log(limit, since)}

    @v1.get("/events", summary="Server-sent events: request log + stats every 2 s", dependencies=ops_scope)
    async def v1_events(request: Request, step_s: int = Query(5, ge=1, le=60)) -> StreamingResponse:
        def status() -> dict[str, Any]:
            return {**stats_body(), "point": stats.latest_point(step_s)}

        return StreamingResponse(
            sse_events(stats, status, request.is_disconnected),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    app.include_router(v1)

    ctx = AppContext(
        cfg=cfg,
        stats=stats,
        engine=eng,
        store=store,
        auth=[Depends(require_key)],
        limited=limited,
        errors=ERRORS,
        infer=infer,
        decide=decide,
        map_exception=map_exception,
        api_error=ApiError,
        request_id=_rid,
        run_classify=run_classify,
        run_classify_batch=run_classify_batch,
        run_score=run_score,
        db=db,
        prepare=prepare,
        decide_prepared=decide_prepared,
        scope=scope,
        quotas=quotas,
        auth_enabled=lambda: auth.enabled,
        keys=keycache,
        env_key_names=auth.env_names,
    )
    app.state.ctx = ctx
    for clash in sorted(auth.env_names & keycache.names):
        log.warning(
            "key name %r is both an env key and a DB principal: the DB keys of %r are refused (401) so they "
            "cannot inherit the env key's ownership; run `clef keys revoke %s` or rename the env key",
            clash,
            clash,
            clash,
        )
    if len(auth.env_names) > 1:
        log.warning(
            "%d env keys (%s) are configured: every env key is a full admin (sees and deletes every job, "
            "edits every classifier, mints keys, reads the audit log). For separate tenants use "
            "`clef keys create` per tenant and keep a single env key for operators.",
            len(auth.env_names),
            ", ".join(sorted(auth.env_names)),
        )

    async def start_keys() -> None:
        """Key cache reload + quota flush tasks (a hook must return, so they run as tasks)."""
        ctx.extra["key_tasks"] = [
            asyncio.create_task(keycache.run(), name="clef-keys"),
            asyncio.create_task(quotas.run(), name="clef-quotas"),
        ]

    async def stop_keys() -> None:
        tasks = ctx.extra.pop("key_tasks", [])
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        # waits for a periodic flush still running in a thread (Quotas.flush is serialised)
        await asyncio.to_thread(quotas.flush)
        await asyncio.to_thread(keycache.flush_used)

    ctx.on_startup.append(start_keys)  # first, so jobs' recover() can refund into loaded counters
    ctx.on_shutdown.append(stop_keys)  # reversed order: runs after the features' shutdown hooks
    for name in FEATURES:
        module = importlib.import_module(f".{name}", __package__)
        app.include_router(module.router(ctx))

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
            "limits": {**cfg.public_limits(), "auth_required": auth.enabled},
            "db": db.info(),
            "model": model_info(),
        }
        return JSONResponse(body, status_code=200 if ok else 503)

    @app.get("/livez", summary="Process liveness")
    async def livez() -> dict[str, bool]:
        return {"ok": True}

    @app.get("/readyz", summary="Readiness (503 until the model is ready, and while draining)")
    async def readyz() -> JSONResponse:
        status = "draining" if app.state.draining else eng.status
        if status == "ready":
            return JSONResponse({"ready": True})
        return JSONResponse({"ready": False, "status": status, "error": eng.error}, status_code=503)

    @app.get("/schema-example", summary="Copy-paste ready request bodies")
    async def schema_example() -> dict[str, Any]:
        return {
            "text_json": {
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
            "model": (
                f"Omit `model` to use the served model ({served_model(eng, None)!r}, see GET /v1/models); "
                "a model this server does not serve is 404."
            ),
            "note": "clef returns CALIBRATED PROBABILITIES for every option in one forward pass - no text generation.",  # noqa: E501
        }

    # console: must be the LAST route so API routes win
    static = Path(__file__).parent / "static"
    if static.is_dir():
        app.mount("/", StaticFiles(directory=static, html=True), name="console")
    return app


class DrainingServer(uvicorn.Server):
    """uvicorn Server with a drain: after the FIRST exit signal it keeps serving for ``drain_s`` seconds.

    uvicorn closes its sockets and connections before the lifespan shutdown runs, so the lifespan's own
    ``draining = True`` is never seen by a client. Here the first SIGTERM / Ctrl-C sets ``app.state.draining``
    (``/readyz`` answers 503 "draining" while requests are still served) and starts a timer; when it ends,
    uvicorn's own ``handle_exit`` starts the shutdown. A second signal during the drain exits at once, and
    later ones keep uvicorn's semantics (a Ctrl-C during the shutdown forces it). ``drain_s = 0``: no drain.
    """

    def __init__(
        self,
        config: uvicorn.Config,
        app: Any,
        drain_s: float = 0.0,
        timer: Callable[..., Any] = threading.Timer,
    ) -> None:
        super().__init__(config)
        self._app = app
        self.drain_s = drain_s
        self._timer = timer
        self._drain_timer: Any = None

    def handle_exit(self, sig: int, frame: Any) -> None:
        if self._drain_timer is not None:  # second signal during the drain: stop waiting
            self._drain_timer.cancel()
            self._drain_timer = None
            super().handle_exit(sig, frame)
            return
        if self.should_exit:  # already shutting down: uvicorn decides (force exit on Ctrl-C)
            super().handle_exit(sig, frame)
            return
        if getattr(self._app, "state", None) is not None:
            self._app.state.draining = True
        if self.drain_s <= 0:
            super().handle_exit(sig, frame)
            return
        try:
            name = signal.Signals(sig).name
        except ValueError:
            name = str(sig)
        log.info(
            "%s: draining for %.1fs (/readyz 503), then shutting down; signal again to stop now",
            name,
            self.drain_s,
        )
        self._drain_timer = self._timer(self.drain_s, self._drain_done, (sig,))
        self._drain_timer.daemon = True
        self._drain_timer.start()

    def _drain_done(self, sig: int) -> None:
        if self._drain_timer is None:  # a second signal got there first
            return
        self._drain_timer = None
        super().handle_exit(sig, None)


def run(cfg: Config | None = None) -> None:
    """Serve in the foreground: ONE uvicorn worker (one model per process); drains CLEF_DRAIN_S on stop."""
    cfg = cfg or config_mod.load()
    # log_config: redact ?key= in the access log and apply CLEF_LOG_FORMAT to uvicorn's own loggers
    log_config = uvicorn_log_config(cfg.log_format)
    try:
        app = create_app(cfg)
    except DatabaseError as exc:  # C1: bad URL, no driver, a second server, failed upgrade
        log.error("cannot start: %s", exc)
        sys.exit(3)
    config = uvicorn.Config(app, host=cfg.host, port=cfg.port, workers=1, log_config=log_config)
    server = DrainingServer(config, app, cfg.drain_s)
    # the tail of uvicorn.run(): the re-raised Ctrl-C after a clean exit is not an error; a failed start is 3
    with suppress(KeyboardInterrupt):
        server.run()
    if not server.started:
        sys.exit(3)  # uvicorn's STARTUP_FAILURE


if __name__ == "__main__":
    run()
