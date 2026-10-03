"""What a feature module gets from create_app(): config, engine, stats and the shared inference helpers.

A feature module exposes ``router(ctx: AppContext) -> APIRouter`` and is listed in ``main.FEATURES``.
Routes that call the model go through ``ctx.infer`` (with a request: logged, counted in stats) or
``ctx.decide`` (no request, for background work). Long-lived tasks register async callables in
``ctx.on_startup`` / ``ctx.on_shutdown``; they run inside the app lifespan after the engine starts and
before it shuts down.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any

from fastapi import Request

from .classifiers import ClassifierStore
from .config import Config
from .schemas import SystemOneRequest
from .stats import Stats

Hook = Callable[[], Awaitable[None]]


@dataclass
class AppContext:
    cfg: Config
    stats: Stats
    engine: Any
    store: ClassifierStore
    # FastAPI dependencies: `auth` goes on every /v1 router, `limited` on inference routes.
    auth: list[Any]
    limited: list[Any]
    errors: dict[int | str, dict[str, Any]]
    # infer(request, reqs, batch, label="batch") -> results; the request path (logged, stats, limits).
    infer: Callable[..., Awaitable[list[dict[str, Any]]]]
    # decide(reqs, label="batch") -> results; limits + media + ONE engine.decide(), no request needed.
    decide: Callable[..., Awaitable[list[dict[str, Any]]]]
    # map_exception(exc) -> ApiError (raise it `from exc`).
    # api_error(status, detail, headers=None) -> ApiError.
    map_exception: Callable[[Exception], Exception]
    api_error: Callable[..., Exception]
    request_id: Callable[[Request], str]
    # The classification routes' implementations (same signatures as in main.create_app).
    run_classify: Callable[..., Awaitable[dict[str, Any]]]
    run_classify_batch: Callable[..., Awaitable[dict[str, Any]]]
    run_score: Callable[..., Awaitable[dict[str, Any]]]
    on_startup: list[Hook] = field(default_factory=list)
    on_shutdown: list[Hook] = field(default_factory=list)
    extra: dict[str, Any] = field(default_factory=dict)  # features may publish objects to each other here


__all__ = ["AppContext", "Hook", "SystemOneRequest"]
