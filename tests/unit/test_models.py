"""Decision model interface (models/): registry, CLEF_MODEL resolution, Engine.check_model / prepare / submit,
the unknown-model 404 on every route, Cache-Control: no-cache, /v1/models and /health `model`. CPU only."""

from __future__ import annotations

import dataclasses
import json
import logging
import sys
import tempfile
from types import SimpleNamespace
from typing import Any

import pytest
from fastapi.testclient import TestClient

from clef_server import backend as bk
from clef_server import cli, doctor
from clef_server.config import MODEL_REPO, MODEL_REVISION, Config
from clef_server.engine import Engine, Prepared, _import_model_module
from clef_server.main import create_app
from clef_server.models import Capabilities, ModelSpec, UnknownModel, registry
from clef_server.models.clef_release import CLEF_REPO, CLEF_REVISION, ClefReleaseAdapter
from clef_server.offload import DEFAULT_LAYOUT, OffloadPlan, split_sizes
from tests.unit.test_engine import FakeModel, FakeStats, expected, make_cfg, make_loader, rec, wait_for

FLASH_SHA = MODEL_REVISION


# ---------------------------------------------------------------- registry
class FakeEP:
    def __init__(self, name: str, value: str, obj: Any = None, dist: str = "toy-pkg 1.0", fail: bool = False):
        self.name, self.value, self._obj, self._fail = name, value, obj, fail
        pkg, _, ver = dist.partition(" ")
        self.dist = SimpleNamespace(metadata={"Name": pkg}, version=ver)

    def load(self) -> Any:
        if self._fail:
            raise ImportError("no module named toy")
        return self._obj


class ToyAdapter(ClefReleaseAdapter):
    name = "toy"

    @classmethod
    def catalog(cls) -> list[ModelSpec]:
        return [
            ModelSpec("toy-1", cls.name, "acme/toy-1", "a" * 40, 1.0),
            ModelSpec("clef-flash", cls.name, "acme/fake-flash", "b" * 40, 1.0),  # duplicate id
        ]


BUILTIN_EP = FakeEP("clef-release", registry.BUILTIN["clef-release"], ClefReleaseAdapter, "clef-local 3.1.0")


@pytest.fixture
def entry_points(monkeypatch):
    """Replace the installed clef.models entry points with `eps` (registry cache cleared both ways)."""

    def use(eps: list[FakeEP]) -> None:
        monkeypatch.setattr(registry.importlib.metadata, "entry_points", lambda group: list(eps))
        registry.cache_clear()

    yield use
    registry.cache_clear()


def test_builtin_registers_through_the_entry_point_group():
    registry.cache_clear()
    assert registry.adapters()["clef-release"] is ClefReleaseAdapter
    cat = registry.catalog()
    assert cat["clef-flash"].ref == f"clef-flash@{FLASH_SHA}"
    assert cat["clef"].repo == CLEF_REPO and cat["clef"].revision == CLEF_REVISION


def test_third_party_adapter_and_duplicate_ids(entry_points, caplog):
    entry_points([FakeEP("toy", "toy_pkg:ToyAdapter", ToyAdapter), BUILTIN_EP])
    with caplog.at_level(logging.WARNING, logger="clef.models"):
        cat = registry.catalog()
    assert cat["toy-1"].adapter == "toy" and registry.adapter_origin("toy") == "toy-pkg 1.0"
    assert cat["clef-flash"].adapter == "clef-release"  # the built-in wins the duplicate id
    assert any("'clef-flash' from adapter 'toy' ignored" in r.getMessage() for r in caplog.records)
    spec = registry.resolve_spec(dataclasses.replace(Config(), model="toy-1", model_path=None))
    assert spec.ref == "toy-1@" + "a" * 40 and isinstance(registry.create(Config(), spec), ToyAdapter)


def test_broken_entry_point_is_skipped_and_builtin_falls_back(entry_points, caplog):
    entry_points([FakeEP("toy", "toy_pkg:ToyAdapter", fail=True)])
    with caplog.at_level(logging.WARNING, logger="clef.models"):
        found = registry.adapters()
    assert "toy" not in found and found["clef-release"] is ClefReleaseAdapter
    assert registry.adapter_origin("clef-release") == "built-in (no entry point)"
    messages = " ".join(r.getMessage() for r in caplog.records)
    assert "failed to import" in messages and "not installed" in messages


def test_third_party_cannot_take_a_builtin_name(entry_points):
    entry_points([FakeEP("clef-release", "evil:Adapter", ToyAdapter), BUILTIN_EP])
    assert registry.adapters()["clef-release"] is ClefReleaseAdapter


def test_third_party_cannot_take_a_builtin_name_even_without_the_builtin_entry_point(entry_points, caplog):
    entry_points([FakeEP("clef-release", "evil:Adapter", ToyAdapter, "evil-pkg 6.6")])  # source checkout
    with caplog.at_level(logging.WARNING, logger="clef.models"):
        assert registry.adapters()["clef-release"] is ClefReleaseAdapter
    assert registry.adapter_origin("clef-release") == "built-in (no entry point)"
    assert any(
        "'clef-release' from evil-pkg 6.6 ignored" in r.getMessage() and "built-in" in r.getMessage()
        for r in caplog.records
    )


# ---------------------------------------------------------------- CLEF_MODEL and the aliases
def cfg_of(**kw: Any) -> Config:
    base = {"model": None, "model_repo": None, "model_revision": None, "model_path": None}
    return dataclasses.replace(Config(), **{**base, **kw})


def test_default_is_clef_flash_at_the_pin():
    spec = registry.resolve_spec(cfg_of())
    assert (spec.id, spec.repo, spec.revision) == ("clef-flash", MODEL_REPO, MODEL_REVISION)


@pytest.mark.parametrize(
    "kw, ref",
    [
        ({"model": "clef"}, f"clef@{CLEF_REVISION}"),
        ({"model": "clef@abc1234"}, "clef@abc1234"),
        ({"model": "clef", "model_revision": "abc1234"}, "clef@abc1234"),
        ({"model": "clef@abc1234", "model_revision": "abc1234"}, "clef@abc1234"),
        ({"model_repo": CLEF_REPO}, f"clef@{CLEF_REVISION}"),  # an unset CLEF_MODEL + the 27B's repo
        ({"model_path": "/somewhere"}, "clef-flash@local"),
        ({"model_path": "/somewhere", "model_revision": "abc1234"}, "clef-flash@abc1234"),
        ({"model": "clef", "model_path": "/somewhere"}, "clef@local"),
    ],
)
def test_resolution_rules(kw, ref):
    assert registry.resolve_spec(cfg_of(**kw)).ref == ref


def test_repo_alias_overrides_the_entry_repo():
    spec = registry.resolve_spec(cfg_of(model_repo="me/my-flash-fork"))
    assert spec.id == "clef-flash" and spec.repo == "me/my-flash-fork" and spec.revision == MODEL_REVISION


def test_unknown_model_lists_the_known_ids():
    with pytest.raises(
        UnknownModel, match=r"CLEF_MODEL='gpt-4o' is not a known model \(known: clef, clef-flash"
    ):
        registry.resolve_spec(cfg_of(model="gpt-4o"))


@pytest.mark.parametrize("value", ["@abc", "clef@"])
def test_bad_clef_model_syntax(value):
    with pytest.raises(ValueError, match="CLEF_MODEL"):
        cfg_of(model=value)


def test_revision_alias_conflict_is_a_config_error(monkeypatch):
    monkeypatch.setenv("CLEF_MODEL", "clef@abc1234")
    monkeypatch.setenv("CLEF_MODEL_REVISION", "def5678")
    with pytest.raises(ValueError, match="disagree"):
        Config()


def test_env_is_read(monkeypatch):
    monkeypatch.setenv("CLEF_MODEL", "clef")
    monkeypatch.delenv("CLEF_MODEL_REPO", raising=False)
    monkeypatch.delenv("CLEF_MODEL_REVISION", raising=False)
    cfg = Config()
    assert cfg.model == "clef" and cfg.model_repo is None and cfg.model_revision is None


def test_27b_is_pinned_and_flagged_for_24gb():
    clef = {s.id: s for s in ClefReleaseAdapter.catalog()}["clef"]
    assert len(clef.revision) == 40 and all(c in "0123456789abcdef" for c in clef.revision)
    assert clef.options["fits_24gb"] == "nf4" and clef.weights_gb > 24


# ---------------------------------------------------------------- engine identity
def engine(**kw: Any) -> Engine:
    return Engine(make_cfg(**kw), FakeStats(), loader=make_loader(FakeModel()))


@pytest.mark.parametrize(
    "requested",
    [None, "", "clef-flash", f"clef-flash@{FLASH_SHA}", f"clef-flash@{FLASH_SHA[:7]}", MODEL_REPO],
)
def test_check_model_accepts_the_served_names(requested):
    eng = engine(model_path=None)
    assert eng.check_model(requested) == "clef-flash"
    assert eng.model_ref() == f"clef-flash@{FLASH_SHA}"


@pytest.mark.parametrize(
    "requested", ["clef", f"clef-flash@{FLASH_SHA[:6]}", "clef-flash@0000000", "gpt-4o", "Cloudflare/clef"]
)
def test_check_model_rejects_the_rest(requested):
    with pytest.raises(UnknownModel) as ei:
        engine(model_path=None).check_model(requested)
    assert str(ei.value) == f"model {requested!r} is not served here (this server serves 'clef-flash')"


def test_a_local_dir_has_no_revision_to_match():
    eng = engine()  # make_cfg sets model_path
    assert eng.model_ref() == "clef-flash@local"
    with pytest.raises(UnknownModel):
        eng.check_model(f"clef-flash@{FLASH_SHA}")
    assert eng.check_model("clef-flash@local") == "clef-flash"  # the ref /health and jobs.model report


def test_check_model_accepts_its_own_ref():
    eng = engine(model_path=None)
    assert eng.check_model(eng.model_ref()) == "clef-flash"


def test_unknown_clef_model_is_a_load_error_not_a_crash():
    eng = Engine(make_cfg(model="nope"), FakeStats())
    assert eng.spec.id == "nope" and eng.model_info()["id"] == "nope"
    eng.start()
    assert wait_for(lambda: eng.status == "error")
    assert "CLEF_MODEL='nope' is not a known model" in eng.error and "clef-flash" in eng.error
    eng.shutdown()


@pytest.fixture
def ready():
    made: list[Engine] = []

    def _make(model: FakeModel | None = None, adapter: Any = None, **kw: Any) -> tuple[Engine, FakeModel]:
        model = model or FakeModel()
        if adapter is not None:
            eng = Engine(make_cfg(**kw), FakeStats(), model=adapter)
        else:
            eng = Engine(make_cfg(**kw), FakeStats(), loader=make_loader(model))
        made.append(eng)
        eng.start()
        assert wait_for(lambda: eng.status == "ready"), eng.error
        return eng, model

    yield _make
    for e in made:
        e.shutdown()


@pytest.mark.asyncio
async def test_prepare_then_submit_matches_decide(ready):
    eng, model = ready()
    prepared = await eng.prepare([rec("a b"), rec("c")])
    assert model.calls == []  # prepare never queues
    assert [e.length for e in prepared.encoded] == [5, 4] and all(isinstance(k, str) for k in prepared.keys)
    assert all(r["model"] == "clef-flash" for r in prepared.records)
    out = await eng.submit(prepared, priority="background")
    assert [o["answers"]["d"]["probs"] for o in out] == [expected("a b"), expected("c")]
    assert {o["model"] for o in out} == {"clef-flash"}


@pytest.mark.asyncio
async def test_submit_merges_hits_without_a_forward(ready):
    eng, model = ready()
    prepared = await eng.prepare([rec("a"), rec("b b"), rec("c")])
    cached = {"model": "clef-flash", "answers": {"d": "cached"}, "usage": {}, "timing": {"cached": True}}
    prepared.hits[1] = cached
    out = await eng.submit(prepared)
    assert out[1] is cached and out[0]["answers"]["d"]["probs"] == expected("a")
    assert sum(n for n, _, _ in model.calls) == 2
    part = prepared.slice(1, 3)
    assert part.hits == {0: cached} and len(part) == 2 and part.records[1]["state"] == "c"


@pytest.mark.asyncio
async def test_unknown_record_model_fails_before_encoding(ready):
    eng, model = ready()
    with pytest.raises(UnknownModel):
        await eng.decide([rec("a"), {**rec("b"), "model": "gpt-4o"}])
    assert model.calls == []


@pytest.mark.asyncio
async def test_capabilities_are_checked(ready):
    adapter = ClefReleaseAdapter.from_module(
        make_loader(FakeModel()), registry.catalog()["clef-flash"], make_cfg()
    )
    adapter.capabilities = dataclasses.replace(
        ClefReleaseAdapter.capabilities, images=False, question_types=frozenset({"choice"}), max_options=2
    )
    eng, _ = ready(adapter=adapter)
    with pytest.raises(ValueError, match="images: model 'clef-flash' does not accept images"):
        await eng.decide([rec("a", media=True)])
    with pytest.raises(ValueError, match="too many options"):
        await eng.decide([rec("a")])  # three options
    noul = {"state": "a", "questions": {"n": {"type": "noul"}}}
    with pytest.raises(ValueError, match=r"questions\.n\.type"):
        await eng.decide([noul])


def test_info_and_model_info(ready):
    eng, _ = ready()
    info = eng.model_info()
    assert (
        info["id"] == "clef-flash" and info["adapter"] == "clef-release" and info["ref"] == "clef-flash@local"
    )
    caps = info["capabilities"]
    assert caps["images"] is True and caps["question_types"] == ["choice", "noul", "score"]
    assert caps["optional_kernels"] == ["fla", "causal_conv1d"]
    json.dumps(info)  # JSON-serialisable


def test_fast_path_lists_only_the_model_kernels():
    eng = engine()
    eng._injected_model = SimpleNamespace(capabilities=Capabilities())  # no optional kernels
    fake = SimpleNamespace(fast_path=lambda: {"causal_conv1d": True, "fla": True, "expected": True})
    assert eng._fast_path(fake) == {"expected": False}


def test_release_module_imports_under_its_own_name(tmp_path):
    (tmp_path / "joint_schema_model.py").write_text(
        "from dataclasses import dataclass\n\n"
        "@dataclass(frozen=True)\nclass EncodedRecord:\n    input_ids: tuple\n"
    )
    mod = _import_model_module(tmp_path, "clef_release_test1234")
    try:
        assert mod.__name__ == "clef_release_test1234" and mod.EncodedRecord((1,)).input_ids == (1,)
        assert _import_model_module(tmp_path, "clef_release_test1234") is mod
        assert "joint_schema_model" not in sys.modules or sys.modules["joint_schema_model"] is not mod
    finally:
        sys.modules.pop("clef_release_test1234", None)


# ---------------------------------------------------------------- HTTP: 404, /v1/models, /health, no-cache
def systemone_answer(question: dict[str, Any], probs: dict[str, float]) -> dict[str, Any]:
    """The release's answer shapes (what the classify / score layer reads)."""
    if question["type"] == "noul":
        return {"type": "noul", "noul": round(probs.get("true", 0.5), 4)}
    best = max(probs, key=probs.__getitem__)
    return {"type": question["type"], "choice": best, "confidence": probs[best], "probabilities": probs}


def api_loader(model: FakeModel):
    base = make_loader(model)

    def loader(cfg: Any) -> Any:
        return SimpleNamespace(**{**vars(base(cfg)), "systemone_answer": systemone_answer})

    return loader


@pytest.fixture
def http():
    model = FakeModel()
    cfg = make_cfg(model_path=None, state_dir=tempfile.mkdtemp(prefix="clef-test-"), api_key=None)
    eng = Engine(cfg, FakeStats(), loader=api_loader(model))
    with TestClient(create_app(cfg, eng), raise_server_exceptions=False) as client:
        assert wait_for(lambda: eng.status == "ready")
        yield client, eng, model


Q = {"d": {"type": "choice", "criteria": {"a": "x", "b": "y"}}}
NOT_SERVED = "model 'gpt-4o' is not served here (this server serves 'clef-flash')"


def test_batch_capability_errors_name_the_item_before_media_decode(http):
    client, _, model = http
    many = {"d": {"type": "choice", "criteria": {f"o{i}": "x" for i in range(65)}}}
    bad = {"state": "s", "questions": many, "images": ["data:image/png;base64,not-an-image"]}
    r = client.post("/v1/batch", json={"batch": [{"state": "s", "questions": Q}, bad]})
    assert r.status_code == 400, r.text
    assert r.json()["detail"].startswith("batch[1].questions.d.criteria: too many options (max 64)")
    assert model.calls == []


def _routes(client: TestClient) -> list[tuple[str, dict[str, Any]]]:
    assert client.put("/v1/classifiers/triage", json={"labels": ["a", "b"]}).status_code == 200
    m = {"model": "gpt-4o"}
    return [
        ("/v1/systemone", {"state": "x", "questions": Q, **m}),
        ("/v1/batch", {"batch": [{"state": "x", "questions": Q}, {"state": "y", "questions": Q, **m}]}),
        ("/v1/classify", {"input": "x", "labels": ["a", "b"], **m}),
        ("/v1/classify/batch", {"inputs": ["x"], "labels": ["a", "b"], **m}),
        ("/v1/score", {"input": "x", "levels": ["lo", "hi"], **m}),
        ("/v1/classifiers/triage", {"input": "x", **m}),
        ("/v1/classifiers/triage/batch", {"inputs": ["x"], **m}),
        ("/v1/evaluate", {"labels": ["a", "b"], "rows": [{"input": "x", "gold": "a"}], **m}),
    ]


def test_unknown_model_is_404_on_every_route_before_any_forward(http):
    client, _, model = http
    for path, body in _routes(client):
        r = client.post(path, json=body)
        assert r.status_code == 404, (path, r.text)
        assert r.json()["detail"] == NOT_SERVED, path
    r = client.post(
        "/v1/chat/completions", json={"model": "gpt-4o", "messages": [{"role": "user", "content": "x"}]}
    )
    assert r.status_code == 404 and r.json()["error"]["code"] == "model_not_found"
    assert model.calls == []


def test_served_names_and_the_hf_path_id(http):
    client, _, model = http
    for name in (None, "clef-flash", MODEL_REPO, f"clef-flash@{FLASH_SHA[:8]}"):
        body = {"state": "a b", "questions": Q, **({"model": name} if name else {})}
        r = client.post("/v1/systemone", json=body)
        assert r.status_code == 200 and r.json()["model"] == "clef-flash", name
    r = client.post("/v1/classify", json={"input": "x", "labels": ["a", "b"], "model": MODEL_REPO})
    assert r.status_code == 200 and r.json()["model"] == "clef-flash"
    hf = {"inputs": "x", "parameters": {"candidate_labels": ["a", "b"]}}
    assert client.post("/hf/models/facebook/bart-large-mnli", json=hf).status_code == 200  # path id ignored


def test_models_routes_and_health(http):
    client, eng, _ = http
    data = client.get("/v1/models").json()["data"]
    assert [m["id"] for m in data] == ["clef-flash"] and data[0]["owned_by"] == "cloudflare"
    clef = data[0]["clef"]
    assert clef["adapter"] == "clef-release" and clef["ref"] == f"clef-flash@{FLASH_SHA}"
    assert clef["status"] == "ready" and clef["capabilities"]["images"] is True
    for name in ("clef-flash", f"clef-flash@{FLASH_SHA[:7]}", MODEL_REPO):
        assert client.get(f"/v1/models/{name}").json()["id"] == "clef-flash"
    r = client.get("/v1/models/clef")
    assert r.status_code == 404 and r.json()["error"]["param"] == "model"
    health = client.get("/health").json()
    assert health["model"]["id"] == "clef-flash" and health["model"]["ref"] == eng.model_ref()
    assert health["model"]["distribution"]


def test_cache_control_no_cache_reaches_engine_prepare(http, monkeypatch):
    client, eng, _ = http
    seen: list[bool] = []
    orig = eng.prepare

    async def spy(records, cache=True):
        seen.append(cache)
        return await orig(records, cache=cache)

    monkeypatch.setattr(eng, "prepare", spy)
    body = {"state": "a", "questions": Q}
    client.post("/v1/systemone", json=body)
    client.post("/v1/systemone", json=body, headers={"Cache-Control": "max-age=0, No-Cache"})
    client.post(
        "/v1/classify", json={"input": "x", "labels": ["a", "b"]}, headers={"cache-control": "no-cache"}
    )
    client.post("/v1/systemone", json=body, headers={"Cache-Control": "no-store"})
    assert seen == [True, False, False, True]


@pytest.mark.asyncio
async def test_ctx_prepare_and_decide_prepared(http):
    client, eng, model = http
    from clef_server.schemas import SystemOneRequest

    ctx = client.app.state.ctx
    prepared = await ctx.prepare([SystemOneRequest(state="a b", questions=Q)], batch=False)
    assert isinstance(prepared, Prepared) and model.calls == []
    out = await ctx.decide_prepared(prepared)
    assert out[0]["model"] == "clef-flash" and len(model.calls) == 1


# ---------------------------------------------------------------- backend / offload / doctor / cli
def test_tunableop_file_is_per_model(monkeypatch):
    monkeypatch.setattr(bk, "_installed_torch_version", lambda: "2.11.0+rocm7.2")
    monkeypatch.setattr(bk, "_is_wsl", lambda: False)
    env = {"CLEF_TUNABLEOP": "1", "CLEF_MODEL": "clef"}
    bk.prepare_environment(env)
    path = env["PYTORCH_TUNABLEOP_FILENAME"]
    assert path.endswith(f"tunableop/clef@{CLEF_REVISION[:8]}/tunableop_%d.csv")
    assert registry.model_slug({}) == f"clef-flash@{FLASH_SHA[:8]}"
    assert registry.model_slug({"CLEF_MODEL": "clef-flash@deadbeef99"}) == "clef-flash@deadbeef"
    assert registry.model_slug({"CLEF_MODEL_PATH": "/x"}) == "clef-flash@local"


def test_model_slug_follows_the_repo_rule_of_resolve_spec():
    env = {"CLEF_MODEL_REPO": "Cloudflare/clef"}  # CLEF_MODEL unset: the catalog entry with that repo
    assert registry.resolve_spec(make_cfg(model_path=None, model_repo="Cloudflare/clef")).id == "clef"
    assert registry.model_slug(env) == f"clef@{CLEF_REVISION[:8]}"
    assert registry.model_slug({"CLEF_MODEL_REPO": "someone/fork"}) == f"clef-flash@{FLASH_SHA[:8]}"
    assert registry.model_slug({**env, "CLEF_MODEL": "clef-flash"}) == f"clef-flash@{FLASH_SHA[:8]}"


def test_offload_layout_names():
    plan = OffloadPlan(2, (1,), (10, 10), 10, 10, 10)
    assert plan.device_map("cuda:0") == {
        "model.visual": "cuda:0",
        "model.language_model.norm": "cuda:0",
        "model.language_model.rotary_emb": "cuda:0",
        "model.language_model.embed_tokens": "cpu",
        "lm_head": "cpu",
        "model.language_model.layers.0": "cuda:0",
        "model.language_model.layers.1": "cpu",
    }
    other = dataclasses.replace(DEFAULT_LAYOUT, layers_prefix="backbone.blocks.", device_prefixes=("vis",))
    assert plan.device_map("mps", layout=other)["backbone.blocks.1"] == "cpu"
    sizes = {"backbone.blocks.0.w": 4, "backbone.blocks.1.w": 6, "lm_head.weight": 3, "vis.w": 1}
    assert split_sizes(sizes, layout=other) == ([4, 6], 3, 1, 0)


def test_doctor_model_checks():
    r = doctor.Report(echo=False)
    assert doctor.check_model(r, cfg_of(model="nope")) is None
    assert r.items[-1]["level"] == doctor.FAIL and "not a known model" in r.items[-1]["detail"]
    r = doctor.Report(echo=False)
    doctor.check_model(r, cfg_of(model="clef"))
    fit = {i["name"]: i for i in r.items}["model fit"]
    assert fit["level"] == doctor.WARN and "only with CLEF_QUANT=nf4" in fit["detail"]
    r = doctor.Report(echo=False)
    doctor.check_model(r, dataclasses.replace(cfg_of(model="clef"), quant="nf4"))
    assert [i["name"] for i in r.items] == ["model"]


def test_cli_models_and_download_model(monkeypatch, capsys, tmp_path):
    for var in ("CLEF_MODEL", "CLEF_MODEL_PATH", "CLEF_MODEL_REPO", "CLEF_MODEL_REVISION"):
        monkeypatch.delenv(var, raising=False)
    assert cli.main(["models", "--json"]) == 0
    out = json.loads(capsys.readouterr().out)
    rows = {m["id"]: m for m in out["models"]}
    assert rows["clef-flash"]["selected"] and not rows["clef"]["selected"]
    assert rows["clef"]["revision"] == CLEF_REVISION and out["adapters"][0]["name"] == "clef-release"
    seen: dict[str, Any] = {}

    def fake(repo, **kw):
        seen.update(repo=repo, **kw)
        return str(tmp_path)

    monkeypatch.setattr(cli, "free_gb", lambda p: 500.0)
    monkeypatch.setitem(sys.modules, "huggingface_hub", SimpleNamespace(snapshot_download=fake))
    assert cli.main(["download", "--yes", "--model", "clef"]) == 0
    assert seen == {"repo": CLEF_REPO, "revision": CLEF_REVISION}
    assert "CLEF_MODEL=clef clef serve" in capsys.readouterr().out
    monkeypatch.setattr(cli, "free_gb", lambda p: 40.0)  # enough for clef-flash, not for the 27B
    assert cli.main(["download", "--yes", "--model", "clef"]) == 1
    assert "at least 57 GB" in capsys.readouterr().err
    assert cli.main(["download", "--yes", "--model", "gpt-4o"]) == 2


def test_a_minimal_third_party_adapter_constructs():
    """Python 3.10 replaces a Protocol's __init__; subclasses that define none must still take (spec, cfg)."""
    from clef_server.models import LocalTensorAdapter

    class Minimal(LocalTensorAdapter):
        name = "minimal"
        capabilities = Capabilities()

    spec = ModelSpec("m", "minimal", None, None, 1.0)
    adapter = Minimal(spec, Config())
    assert adapter.spec is spec and adapter.info()["ref"] == "m@local"
    adapter.validate({"questions": {"q": {"type": "noul"}}})
