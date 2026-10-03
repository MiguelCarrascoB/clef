"""Clef-flash server v3 - SystemOne decision + classification API served locally (CUDA, ROCm, MPS or CPU).

See docs/ARCHITECTURE.md for the API contract. Run: clef serve (one uvicorn worker, one GPU thread).

  POST /v1/systemone  {model?, state, questions, images?, videos?} -> {model, answers, usage, timing}
  POST /v1/batch      {"batch": [<systemone request>, ...]}        -> {batch_ms, results}
  POST /v1/classify[/batch], /v1/score, /v1/classifiers/{name}[/batch]; PUT|GET|DELETE /v1/classifiers[/n]
  GET  /health /livez /v1/stats /v1/stats/timeseries /v1/log /v1/events /schema-example   console at /
"""

from __future__ import annotations

import asyncio
import importlib
import json
import logging
import re
import time
import uuid
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager, nullcontext
from pathlib import Path
from typing import Any

import uvicorn
from fastapi import APIRouter, Depends, FastAPI, HTTPException, Query, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from starlette.concurrency import run_in_threadpool
from starlette.exceptions import HTTPException as StarletteHTTPException

from . import config as config_mod
from .appctx import AppContext
from .classifiers import ClassifierStore, InvalidName, TooManyClassifiers, validate_name
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
from .engine import Engine, EngineNotReady, GpuOutOfMemory, InputTooLarge
from .media import MediaError, load_media
from .paths import classifiers_dir
from .schemas import (
    DEFAULT_MODEL,
    BatchRequest,
    BatchResponse,
    ErrorBody,
    SystemOneRequest,
    SystemOneResponse,
    check_batch_size,
    check_limits,
    format_errors,
)
from .security import KeyAuth, RateLimiter, bind_warning, client_identity, install_cors, is_tracked
from .stats import Stats

log = logging.getLogger("clef")
SSE_STATS_INTERVAL_S = 2.0
_RID_OK = re.compile(r"^[A-Za-z0-9._-]{1,64}$")
ERRORS: dict[int | str, dict[str, Any]] = {
    c: {"model": ErrorBody, "description": d}
    for c, d in {
        400: "Invalid input",
        401: "Missing/invalid API key",
        404: "Unknown classifier",
        409: "Too many saved classifiers",
        413: "Request too large",
        429: "Rate limit exceeded (see Retry-After)",
        503: "Model not ready or GPU out of memory",
        500: "Internal error",
    }.items()
}
# Feature modules: each exposes router(ctx: AppContext) -> APIRouter (see appctx.py). One line each.
FEATURES: list[str] = []


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

    async def __call__(self, scope: dict[str, Any], receive: Callable, send: Callable) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        headers = {k.decode("latin1").lower(): v.decode("latin1") for k, v in scope["headers"]}
        rid = headers.get("x-request-id", "")
        rid = rid if _RID_OK.match(rid) else uuid.uuid4().hex
        state = scope.setdefault("state", {})
        state.update(request_id=rid, t0=time.perf_counter(), rec={}, error=None)
        tracked = is_tracked(scope["method"], scope["path"])
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
    setup_logging()
    cfg = cfg or config_mod.load()
    stats = Stats(cfg.log_buffer)
    eng = engine if engine is not None else Engine(cfg, stats)
    started = time.time()
    auth = KeyAuth(cfg.api_keys())
    limiter = RateLimiter(cfg.rate_limit)
    store = ClassifierStore(classifiers_dir(cfg), cfg.max_classifiers)

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
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        log.info("starting clef server %s (model load runs in background)", VERSION)
        eng.start()
        task = asyncio.create_task(sampler())
        for hook in ctx.on_startup:
            await hook()
        try:
            yield
        finally:
            for hook in reversed(ctx.on_shutdown):
                try:
                    await hook()
                except Exception:
                    log.exception("shutdown hook failed")
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
            eng.shutdown()

    app = FastAPI(
        title="clef-flash local",
        version=VERSION,
        lifespan=lifespan,
        description=(
            "SystemOne decision API: calibrated probabilities for choice / score / noul questions over any text or "  # noqa: E501
            "JSON state (+ images and videos), in one forward pass, plus a classification layer (classify, score, "  # noqa: E501
            "saved classifiers). Errors are `{detail, request_id}`."
        ),
    )
    app.state.cfg, app.state.stats, app.state.engine = cfg, stats, eng
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

    @app.exception_handler(Exception)
    async def _unhandled(request: Request, exc: Exception) -> JSONResponse:
        log.error("unhandled error [%s]", _rid(request), exc_info=exc)
        resp = _error(request, 500, "internal server error")
        resp.headers["X-Request-ID"] = _rid(request)  # runs outside GuardMiddleware
        return resp

    # ---- auth
    async def require_key(request: Request) -> None:
        if not auth.enabled:
            request.state.key_name = None
            return
        name = auth.authenticate(KeyAuth.extract(request))
        if name is None:
            raise ApiError(401, "missing or invalid API key", {"WWW-Authenticate": "Bearer"})
        request.state.key_name = name

    async def rate_limit(request: Request) -> None:
        if cfg.rate_limit <= 0:
            return
        retry = limiter.check(client_identity(request, getattr(request.state, "key_name", None)))
        if retry:
            raise ApiError(429, f"rate limit exceeded ({cfg.rate_limit}/min)", {"Retry-After": str(retry)})

    limited = [Depends(rate_limit)]
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
        rec["n_records"] = len(reqs)
        rec["n_questions"] = sum(len(r.questions) for r in reqs)
        rec["media"] = {
            "images": sum(len(r.images or []) for r in reqs),
            "videos": sum(len(r.videos or []) for r in reqs),
        }
        if cfg.log_state:
            rec["state_preview"] = json.dumps(reqs[0].state, default=str, ensure_ascii=False)[:200]
        results = await decide(reqs, batch=batch, label=label)
        rec["input_tokens"] = sum(int(r.get("usage", {}).get("input_tokens", 0)) for r in results)
        return results

    async def decide(
        reqs: list[SystemOneRequest], batch: bool = True, label: str = "batch"
    ) -> list[dict[str, Any]]:
        """Limits, media load (threadpool), ONE engine.decide(). No request needed (background work)."""
        for i, r in enumerate(reqs):
            check_limits(r, cfg, f"{label}[{i}]." if batch else "")
        gpu_ready()

        def build() -> list[dict[str, Any]]:
            records = []
            for i, r in enumerate(reqs):
                try:
                    images, videos = load_media(r.images, r.videos, cfg)
                except MediaError as exc:
                    raise MediaError(f"{label}[{i}].{exc}" if batch else str(exc)) from exc
                records.append(r.to_record(images, videos))
            return records

        records = await run_in_threadpool(build)
        return await eng.decide(records)

    def with_total(res: dict[str, Any], total_ms: float) -> dict[str, Any]:
        return {**res, "timing": {**res.get("timing", {}), "total_ms": round(total_ms, 1)}}

    @v1.post(
        "/systemone", response_model=SystemOneResponse, summary="Decide one request", dependencies=limited
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
        dependencies=limited,
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
        res: dict[str, Any], labels: Any, multi: bool, thr: float | None, model: str, ms: float
    ) -> dict[str, Any]:
        return {**classify_result(res["answers"], labels, multi, thr, model), **usage_timing(res, ms)}

    async def run_classify(
        request: Request,
        *,
        state: Any,
        labels: Any,
        instructions: str | None,
        multi_label: bool,
        threshold: float | None,
        model: str,
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
        model: str,
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
        model: str,
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
            **score_result(results[0]["answers"], levels, model),
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
        dependencies=limited,
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
        dependencies=limited,
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
        dependencies=limited,
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

    async def load_classifier(name: str) -> dict[str, Any]:
        checked_name(name)
        doc = await run_in_threadpool(store.get, name)
        if doc is None:
            raise ApiError(404, f"classifier {name!r} not found")
        return doc

    @v1.put(
        "/classifiers/{name}", response_model=ClassifierInfo, summary="Create or replace a saved classifier"
    )
    async def put_classifier(name: str, body: ClassifierDef) -> dict[str, Any]:
        checked_name(name)
        try:
            if body.kind == "classify":
                check_label_count(body.labels or [], body.multi_label, cfg.max_labels)
            else:
                check_levels_count(body.levels or [], cfg.max_labels)
            return await run_in_threadpool(store.put, name, body.model_dump())
        except Exception as exc:
            raise map_exception(exc) from exc

    @v1.get("/classifiers", response_model=ClassifierList, summary="List saved classifiers")
    async def list_classifiers() -> dict[str, Any]:
        return {"classifiers": await run_in_threadpool(store.list)}

    @v1.get("/classifiers/{name}", response_model=ClassifierInfo, summary="Get a saved classifier")
    async def get_classifier(name: str) -> dict[str, Any]:
        return await load_classifier(name)

    @v1.delete("/classifiers/{name}", response_model=ClassifierDeleted, summary="Delete a saved classifier")
    async def delete_classifier(name: str) -> dict[str, str]:
        checked_name(name)
        if not await run_in_threadpool(store.delete, name):
            raise ApiError(404, f"classifier {name!r} not found")
        return {"deleted": name}

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
        dependencies=limited,
    )
    async def run_classifier(name: str, body: ClassifierRunRequest, request: Request) -> dict[str, Any]:
        doc = await load_classifier(name)
        common = {"state": body.input, "images": body.images, "videos": body.videos, "classifier": name}
        if doc["kind"] == "score":
            if body.threshold is not None:
                raise ApiError(400, "threshold: not allowed for score classifiers")
            return await run_score(
                request,
                levels=doc["levels"],
                instructions=doc.get("instructions"),
                model=DEFAULT_MODEL,
                **common,
            )
        return await run_classify(
            request,
            labels=doc["labels"],
            instructions=doc.get("instructions"),
            multi_label=bool(doc.get("multi_label")),
            threshold=run_threshold(doc, body.threshold),
            model=DEFAULT_MODEL,
            **common,
        )

    @v1.post(
        "/classifiers/{name}/batch",
        response_model=ClassifyBatchResponse,
        response_model_exclude_none=True,
        summary="Run a saved classifier on many inputs (one micro-batched pass)",
        dependencies=limited,
    )
    async def run_classifier_batch(
        name: str, body: ClassifierBatchRequest, request: Request
    ) -> dict[str, Any]:
        doc = await load_classifier(name)
        if doc["kind"] == "score":
            raise ApiError(400, "batch is only supported for classify classifiers")
        return await run_classify_batch(
            request,
            inputs=body.inputs,
            labels=doc["labels"],
            instructions=doc.get("instructions"),
            multi_label=bool(doc.get("multi_label")),
            threshold=run_threshold(doc, body.threshold),
            model=DEFAULT_MODEL,
            classifier=name,
        )

    # ---- observability
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

    @v1.get("/stats", summary="Rolling latency / throughput / GPU stats")
    async def v1_stats(window_s: int = Query(300, ge=5, le=3600)) -> dict[str, Any]:
        return stats_body(window_s)

    @v1.get("/stats/timeseries", summary="Columnar time series for charts (oldest first)")
    async def v1_timeseries(
        window_s: int = Query(300, ge=10, le=3600), step_s: int | None = Query(None, ge=1)
    ) -> dict[str, Any]:
        try:
            return stats.timeseries(window_s, step_s)
        except ValueError as exc:
            raise ApiError(400, str(exc)) from exc

    @v1.get("/log", summary="Recent request log (newest last)")
    async def v1_log(
        limit: int = Query(100, ge=1, le=1000), since: int | None = Query(None, ge=0)
    ) -> dict[str, Any]:
        return {"entries": stats.log(limit, since)}

    @v1.get("/events", summary="Server-sent events: request log + stats every 2 s")
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
    )
    app.state.ctx = ctx
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


def run(cfg: Config | None = None) -> None:
    """Serve in the foreground: ONE uvicorn worker (one model per process)."""
    cfg = cfg or config_mod.load()
    warning = bind_warning(cfg)
    if warning:
        setup_logging()
        for line in warning.splitlines():
            log.warning(line)
    uvicorn.run(create_app(cfg), host=cfg.host, port=cfg.port, workers=1)


if __name__ == "__main__":
    run()
