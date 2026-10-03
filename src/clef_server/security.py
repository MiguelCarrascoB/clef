"""Multi-key auth, per-key sliding-window rate limiting and CORS setup."""

from __future__ import annotations

import hmac
import math
import re
import threading
import time
from collections import deque
from collections.abc import Callable
from typing import Any

from fastapi import FastAPI, Request
from starlette.middleware.cors import CORSMiddleware

from .config import Config

TRACKED_EXACT = frozenset({"/v1/systemone", "/v1/batch", "/v1/classify", "/v1/classify/batch", "/v1/score"})
_CLASSIFIER_RUN = re.compile(r"^/v1/classifiers/[^/]+(/batch)?$")


def is_tracked(method: str, path: str) -> bool:
    """POST inference routes: logged, counted in stats and rate limited."""
    if method != "POST":
        return False
    return path in TRACKED_EXACT or bool(_CLASSIFIER_RUN.match(path))


class KeyAuth:
    """name -> key table; comparison is constant-time and walks every key (no early exit)."""

    def __init__(self, keys: dict[str, str]):
        self._keys = [(name, key.encode()) for name, key in keys.items()]

    @property
    def enabled(self) -> bool:
        return bool(self._keys)

    def authenticate(self, given: str | None) -> str | None:
        """Name of the matching key, or None."""
        if not given:
            return None
        data = given.encode()
        found: str | None = None
        for name, key in self._keys:
            if hmac.compare_digest(data, key) and found is None:
                found = name
        return found

    @staticmethod
    def extract(request: Request) -> str | None:
        given = request.headers.get("x-api-key")
        auth = request.headers.get("authorization", "")
        if not given and auth.lower().startswith("bearer "):
            given = auth[7:].strip()
        if not given and request.url.path == "/v1/events":  # EventSource cannot set headers
            given = request.query_params.get("key")
        return given or None


class RateLimiter:
    """Sliding-window limiter: `limit` hits per `window` seconds per identity. Thread-safe."""

    def __init__(self, limit: int, window: float = 60.0, clock: Callable[[], float] = time.monotonic):
        self.limit, self.window, self.clock = limit, window, clock
        self._hits: dict[str, deque[float]] = {}
        self._lock = threading.Lock()
        self._next_prune = clock() + window

    def check(self, identity: str) -> int:
        """Record a hit: 0 when allowed, else Retry-After seconds (>= 1). Rejections are not recorded."""
        if self.limit <= 0:
            return 0
        now = self.clock()
        with self._lock:
            if now >= self._next_prune:
                self._prune(now)
            dq = self._hits.get(identity)
            if dq is None:
                dq = self._hits[identity] = deque()
            cutoff = now - self.window
            while dq and dq[0] <= cutoff:
                dq.popleft()
            if len(dq) >= self.limit:
                return max(1, math.ceil(dq[0] + self.window - now))
            dq.append(now)
            return 0

    def _prune(self, now: float) -> None:
        cutoff = now - self.window
        for ident in [i for i, dq in self._hits.items() if not dq or dq[-1] <= cutoff]:
            del self._hits[ident]
        self._next_prune = now + self.window

    def identities(self) -> int:
        with self._lock:
            return len(self._hits)


def client_identity(request: Request, key_name: str | None) -> str:
    if key_name is not None:
        return f"key:{key_name}"
    return f"ip:{request.client.host if request.client else '-'}"


def install_cors(app: FastAPI, cfg: Config) -> None:
    """CORS is off unless CLEF_CORS_ORIGINS is set. Call AFTER the other middleware so it is outermost."""
    if not cfg.cors_origins:
        return
    app.add_middleware(
        CORSMiddleware,
        allow_origins=list(cfg.cors_origins),
        allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
        allow_headers=["*"],
        expose_headers=["X-Request-ID", "Retry-After"],
        allow_credentials=False,
    )


def bind_warning(cfg: Config) -> str | None:
    """Multi-line warning when listening beyond loopback without any API key."""
    if cfg.is_loopback or cfg.auth_required:
        return None
    bar = "!" * 72
    return "\n".join(
        [
            bar,
            f"WARNING: clef is listening on {cfg.host}:{cfg.port} (not loopback) with NO API key.",
            "Anyone who can reach this port can run inference on your GPU and manage classifiers.",
            "Set CLEF_API_KEY (or CLEF_API_KEYS=name:key,...) or bind to 127.0.0.1.",
            bar,
        ]
    )


def key_name_of(request: Request) -> Any:
    return getattr(request.state, "key_name", None)
