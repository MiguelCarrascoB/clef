"""Unit tests for security.py: key table, sliding-window limiter, route tracking, CORS, bind warning."""

from __future__ import annotations

import threading

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from clef_server.config import Config
from clef_server.security import KeyAuth, RateLimiter, bind_warning, install_cors, is_tracked


class Clock:
    def __init__(self) -> None:
        self.t = 1000.0

    def __call__(self) -> float:
        return self.t


# ------------------------------------------------------------------ keys


def test_keyauth_names_and_misses():
    auth = KeyAuth({"web": "k-web", "batch": "k-batch"})
    assert auth.enabled
    assert auth.authenticate("k-web") == "web"
    assert auth.authenticate("k-batch") == "batch"
    assert auth.authenticate("k-web ") is None
    assert auth.authenticate("") is None and auth.authenticate(None) is None
    assert not KeyAuth({}).enabled


def test_keyauth_compares_every_key(monkeypatch):
    import clef_server.security as sec

    calls = []
    real = sec.hmac.compare_digest
    monkeypatch.setattr(sec.hmac, "compare_digest", lambda a, b: calls.append(1) or real(a, b))
    auth = KeyAuth({"a": "1", "b": "2", "c": "3"})
    assert auth.authenticate("1") == "a"
    assert len(calls) == 3  # no early exit on the first match


def test_keyauth_unicode_key():
    assert KeyAuth({"u": "clé"}).authenticate("clé") == "u"


# ------------------------------------------------------------------ rate limiter


def test_limiter_blocks_over_limit_and_recovers():
    clock = Clock()
    rl = RateLimiter(3, 60.0, clock)
    assert [rl.check("a") for _ in range(3)] == [0, 0, 0]
    assert rl.check("a") == 60
    clock.t += 10
    assert rl.check("a") == 50  # Retry-After counts down to the oldest hit leaving the window
    clock.t += 50.1
    assert rl.check("a") == 0


def test_limiter_rejections_do_not_extend_the_window():
    clock = Clock()
    rl = RateLimiter(1, 60.0, clock)
    assert rl.check("a") == 0
    for _ in range(5):
        clock.t += 10
        assert rl.check("a") > 0
    clock.t = 1060.5
    assert rl.check("a") == 0


def test_limiter_identities_are_isolated():
    rl = RateLimiter(1, 60.0, Clock())
    assert rl.check("a") == 0 and rl.check("b") == 0
    assert rl.check("a") > 0 and rl.check("b") > 0


def test_limiter_retry_after_is_int_ge_1():
    clock = Clock()
    rl = RateLimiter(1, 60.0, clock)
    rl.check("a")
    clock.t += 59.9
    ra = rl.check("a")
    assert isinstance(ra, int) and ra == 1


def test_limiter_disabled_and_pruning():
    assert RateLimiter(0).check("a") == 0
    clock = Clock()
    rl = RateLimiter(5, 60.0, clock)
    for i in range(100):
        rl.check(f"ip{i}")
    assert rl.identities() == 100
    clock.t += 61
    rl.check("fresh")  # triggers the sweep
    assert rl.identities() == 1


def test_limiter_thread_safe():
    rl = RateLimiter(50, 60.0)
    allowed = []

    def work() -> None:
        for _ in range(40):
            if rl.check("shared") == 0:
                allowed.append(1)

    threads = [threading.Thread(target=work) for _ in range(8)]
    [t.start() for t in threads]
    [t.join() for t in threads]
    assert len(allowed) == 50


# ------------------------------------------------------------------ tracking


@pytest.mark.parametrize(
    "method,path,expect",
    [
        ("POST", "/v1/systemone", True),
        ("POST", "/v1/batch", True),
        ("POST", "/v1/classify", True),
        ("POST", "/v1/classify/batch", True),
        ("POST", "/v1/score", True),
        ("POST", "/v1/classifiers/support-triage", True),
        ("POST", "/v1/classifiers/support-triage/batch", True),
        ("PUT", "/v1/classifiers/support-triage", False),
        ("GET", "/v1/classifiers", False),
        ("POST", "/v1/classifiers", False),
        ("POST", "/v1/classifiers/a/b/c", False),
        ("GET", "/v1/stats", False),
        ("POST", "/health", False),
    ],
)
def test_is_tracked(method, path, expect):
    assert is_tracked(method, path) is expect


# ------------------------------------------------------------------ CORS


def _cors_client(origins: tuple[str, ...]) -> TestClient:
    app = FastAPI()

    @app.get("/x")
    def x() -> dict[str, int]:
        return {"a": 1}

    install_cors(app, Config(cors_origins=origins))
    return TestClient(app)


def test_cors_off_by_default():
    r = _cors_client(()).get("/x", headers={"Origin": "https://a.example"})
    assert "access-control-allow-origin" not in r.headers


def test_cors_allowed_origin_and_preflight():
    c = _cors_client(("https://a.example",))
    r = c.get("/x", headers={"Origin": "https://a.example"})
    assert r.headers["access-control-allow-origin"] == "https://a.example"
    assert "access-control-allow-credentials" not in r.headers
    r = c.options(
        "/x",
        headers={
            "Origin": "https://a.example",
            "Access-Control-Request-Method": "PUT",
            "Access-Control-Request-Headers": "x-api-key,content-type",
        },
    )
    assert r.status_code == 200 and "PUT" in r.headers["access-control-allow-methods"]
    assert "x-api-key" in r.headers["access-control-allow-headers"].lower()
    r = c.get("/x", headers={"Origin": "https://evil.example"})
    assert "access-control-allow-origin" not in r.headers


def test_cors_wildcard():
    r = _cors_client(("*",)).get("/x", headers={"Origin": "https://any.example"})
    assert r.headers["access-control-allow-origin"] == "*"


# ------------------------------------------------------------------ bind warning


def test_bind_warning():
    assert bind_warning(Config(host="127.0.0.1")) is None
    assert bind_warning(Config(host="localhost")) is None
    assert bind_warning(Config(host="0.0.0.0", api_key="k")) is None
    assert bind_warning(Config(host="0.0.0.0", api_keys_raw="a:b")) is None
    text = bind_warning(Config(host="0.0.0.0"))
    assert text and "WARNING" in text and "CLEF_API_KEY" in text and text.count("\n") >= 3
