"""Auth (env keys + hashed DB keys), per-principal sliding-window rate limiting and CORS setup."""

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
from .keystore import KeyCache, Principal, env_principal, hash_key, key_status, parse_key

TRACKED_EXACT = frozenset(
    {"/v1/systemone", "/v1/batch", "/v1/classify", "/v1/classify/batch", "/v1/score", "/v1/evaluate"}
)
_CLASSIFIER_RUN = re.compile(r"^/v1/classifiers/[^/]+(/batch)?$")
# compat routes (OpenAI chat completions, Hugging Face zero-shot): see compat.py
_COMPAT_RUN = re.compile(r"^(/v1/chat/completions|(/hf)?/models/.+)$")


def is_tracked(method: str, path: str) -> bool:
    """POST inference routes: logged, counted in stats and rate limited."""
    if method != "POST":
        return False
    if _COMPAT_RUN.match(path):
        return True
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


class Authenticator:
    """Who is calling: env keys (``name:key``, full admin) and the hashed keys of the DB principals.

    Auth is on when an env key exists OR a DB principal row ever existed (``any_principal()`` is sticky and
    counts disabled principals), so revoking the last DB key never switches auth off. No I/O (memory).
    """

    def __init__(self, env_keys: dict[str, str], cache: KeyCache | None = None):
        self._env = KeyAuth(env_keys)
        self.env_names = frozenset(env_keys)
        self._env_principals = {name: env_principal(name) for name in env_keys}
        self.cache = cache

    @property
    def enabled(self) -> bool:
        return self._env.enabled or (self.cache is not None and self.cache.any_principal())

    def resolve(self, token: str | None) -> Principal | None:
        """The principal for a key, or None (401): revoked, expired, disabled, unknown all look alike."""
        if not token:
            return None
        key_id = parse_key(token) if self.cache is not None else None
        if key_id is not None:
            hit = self.cache.get(key_id)  # type: ignore[union-attr]
            if hit is not None:
                key, principal = hit
                if not hmac.compare_digest(hash_key(token), key.secret_sha256):
                    return None
                if (
                    key_status({"revoked_at": key.revoked_at, "expires_at": key.expires_at}, time.time())
                    in (
                        "revoked",
                        "expired",
                    )
                    or principal.disabled_at is not None
                ):
                    return None
                self.cache.touch(key_id)  # type: ignore[union-attr]
                return Principal(
                    principal.name,
                    "db",
                    principal.scopes,
                    principal.daily_quota,
                    principal.rate_limit,
                    key_id,
                    key.expires_at,
                )
        name = self._env.authenticate(token)  # constant-time walk over every env key
        return self._env_principals[name] if name is not None else None


class RateLimiter:
    """Sliding-window limiter: `limit` hits per `window` seconds per identity. Thread-safe."""

    def __init__(self, limit: int, window: float = 60.0, clock: Callable[[], float] = time.monotonic):
        self.limit, self.window, self.clock = limit, window, clock
        self._hits: dict[str, deque[float]] = {}
        self._lock = threading.Lock()
        self._next_prune = clock() + window

    def check(self, identity: str, limit: int | None = None) -> int:
        """Record a hit: 0 when allowed, else Retry-After seconds (>= 1). Rejections are not recorded.

        ``limit`` overrides the default for this identity (a principal's own ``rate_limit``)."""
        limit = self.limit if limit is None else limit
        if limit <= 0:
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
            if len(dq) >= limit:
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
        allow_methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"],
        allow_headers=["*"],
        expose_headers=["X-Request-ID", "Retry-After", "X-Quota-Limit", "X-Quota-Remaining"],
        allow_credentials=False,
    )


def bind_warning(cfg: Config, auth_enabled: bool | None = None) -> str | None:
    """Multi-line warning when listening beyond loopback without any API key (env or DB)."""
    if auth_enabled is None:
        auth_enabled = cfg.auth_required
    if cfg.is_loopback or auth_enabled:
        return None
    bar = "!" * 72
    return "\n".join(
        [
            bar,
            f"WARNING: clef is listening on {cfg.host}:{cfg.port} (not loopback) with NO API key.",
            "Anyone who can reach this port can run inference on your GPU and manage classifiers.",
            "Set CLEF_API_KEY (or CLEF_API_KEYS=name:key,...), run `clef keys create`, or bind to 127.0.0.1.",
            bar,
        ]
    )


def key_name_of(request: Request) -> Any:
    return getattr(request.state, "key_name", None)
