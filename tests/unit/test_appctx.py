"""Feature modules plug into create_app() through main.FEATURES and get an AppContext."""

from __future__ import annotations

import sys
import tempfile
import types

from fastapi import APIRouter
from fastapi.testclient import TestClient

from clef_server import main
from clef_server.appctx import AppContext
from clef_server.config import Config
from clef_server.schemas import SystemOneRequest
from tests.unit.test_api import GOOD, FakeEngine


def _feature(events: list[str]) -> types.ModuleType:
    mod = types.ModuleType("clef_server._test_feature")

    def router(ctx: AppContext) -> APIRouter:
        r = APIRouter(prefix="/v1", dependencies=ctx.auth)

        async def up() -> None:
            events.append("startup")

        async def down() -> None:
            events.append("shutdown")

        ctx.on_startup.append(up)
        ctx.on_shutdown.append(down)

        @r.post("/_feature")
        async def feature() -> dict:
            results = await ctx.decide([SystemOneRequest(**GOOD)])
            return {"choice": results[0]["answers"]["dept"]["choice"]}

        return r

    mod.router = router  # type: ignore[attr-defined]
    return mod


def test_feature_router_and_hooks(monkeypatch) -> None:
    events: list[str] = []
    monkeypatch.setitem(sys.modules, "clef_server._test_feature", _feature(events))
    monkeypatch.setattr(main, "FEATURES", ["_test_feature"])
    with tempfile.TemporaryDirectory() as d:
        app = main.create_app(Config(api_key=None, state_dir=d), engine=FakeEngine())
        assert isinstance(app.state.ctx, AppContext)
        with TestClient(app) as c:
            assert events == ["startup"]
            r = c.post("/v1/_feature")
            assert r.status_code == 200, r.text
            assert r.json() == {"choice": "technical"}
        assert events == ["startup", "shutdown"]


def test_feature_routes_require_key(monkeypatch) -> None:
    monkeypatch.setitem(sys.modules, "clef_server._test_feature", _feature([]))
    monkeypatch.setattr(main, "FEATURES", ["_test_feature"])
    with tempfile.TemporaryDirectory() as d:
        cfg = Config(api_key="s3cret", api_keys_raw=None, state_dir=d)
        with TestClient(main.create_app(cfg, engine=FakeEngine())) as c:
            assert c.post("/v1/_feature").status_code == 401
            assert c.post("/v1/_feature", headers={"X-API-Key": "s3cret"}).status_code == 200
