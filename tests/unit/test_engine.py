"""Engine tests: CPU only, fake model/loader, no weights."""

from __future__ import annotations

import asyncio
import dataclasses
import importlib.util
import os
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from clef_server.backend import BackendError
from clef_server.config import Config
from clef_server.engine import (
    Engine,
    EngineNotReady,
    GpuOutOfMemory,
    InputTooLarge,
    LoadInfo,
    bucket_for,
    pad_batch_right,
)

REAL_MODEL = (
    Path(os.environ.get("CLEF_MODEL_PATH", Path.home() / "models" / "clef-flash")) / "joint_schema_model.py"
)


# ---------------------------------------------------------------- fakes
@dataclasses.dataclass(frozen=True)
class FQ:
    question_id: str
    option_ids: tuple


@dataclasses.dataclass(frozen=True)
class FEnc:
    input_ids: tuple
    questions: tuple
    record_id: str = "x"
    media: dict | None = None


class FakeStats:
    def __init__(self):
        self.forwards = []
        self.depths = []

    def record_forward(self, n_records, n_tokens, padded_tokens, ms):
        self.forwards.append((n_records, n_tokens, padded_tokens, ms))

    def set_queue_depth(self, n):
        self.depths.append(n)


class FakeModel:
    def __init__(self):
        self.calls = []  # (batch_size, seq_len, has_media)
        self.gate: threading.Event | None = None
        self.entered = threading.Event()

    def eval(self):
        return self

    def __call__(self, batch):
        recs = batch["records"]
        self.calls.append((len(recs), batch["input_ids"].shape[1], any(r.media for r in recs)))
        self.entered.set()
        if self.gate is not None:
            self.gate.wait(5)
        out = []
        for i, rec in enumerate(recs):
            ids = batch["input_ids"][i][batch["attention_mask"][i].bool()]
            if 10_000 in ids.tolist():
                raise torch.cuda.OutOfMemoryError("fake oom")
            s = float(ids.sum())
            out.append(
                [
                    torch.tensor([(s * (k + 1) + j) % 5 for k in range(len(q.option_ids))])
                    for j, q in enumerate(rec.questions)
                ]
            )
        return out


def fake_encode(tokenizer, record, max_length=16384, max_state_tokens=None, processor=None):
    state = record["state"]
    if state == "TOO_BIG":
        raise ValueError("schema requires 99999 tokens before state; maximum is 16")
    if state == "BAD":
        raise ValueError("other problem")
    ids = [len(w) + 1 for w in str(state).split()]
    if state == "OOM":
        ids = [10_000]
    ids = [1, 2, 3] + ids  # fixed prefix
    media = {"token_offset": 1} if record.get("images") else None
    qs = tuple(
        FQ(str(k), tuple(sorted(q["criteria"])) if q["type"] == "choice" else ("false", "true"))
        for k, q in record["questions"].items()
    )
    return FEnc(tuple(ids), qs, media=media)


def fake_collate(records, pad_token_id, device):
    n = max(len(r.input_ids) for r in records)
    ids = torch.full((len(records), n), pad_token_id, dtype=torch.long)
    mask = torch.zeros((len(records), n), dtype=torch.long)
    for i, r in enumerate(records):
        ids[i, : len(r.input_ids)] = torch.tensor(r.input_ids)
        mask[i, : len(r.input_ids)] = 1
    return {"input_ids": ids, "attention_mask": mask, "records": records, "media": {}}


def fake_answer(question, probs):
    return {"type": question["type"], "probs": {k: round(v, 4) for k, v in probs.items()}}


def make_loader(model: FakeModel):
    def loader(cfg):
        tok = SimpleNamespace(pad_token_id=0)
        proc = SimpleNamespace(tokenizer=tok)
        return SimpleNamespace(
            load_release_model=lambda path, device, dtype: (model, proc),
            encode_record=fake_encode,
            collate_records=fake_collate,
            systemone_answer=fake_answer,
        )

    return loader


def make_cfg(**kw):
    base = dict(
        device="cpu",
        dtype="float32",
        warmup=False,
        batch_window_ms=100.0,
        max_microbatch=8,
        buckets=(8, 16, 32, 64),
        pad_to_bucket=True,
        model_path="/nowhere",
        max_tokens=1000,
        preflight=False,
    )
    base.update(kw)
    return dataclasses.replace(Config(), **base)


def wait_for(pred, timeout=5.0):
    t0 = time.time()
    while time.time() - t0 < timeout:
        if pred():
            return True
        time.sleep(0.005)
    return False


def rec(state, media=False):
    r = {
        "model": "clef-flash",
        "state": state,
        "questions": {"d": {"type": "choice", "criteria": {"a": "x", "b": "y", "c": "z"}}},
    }
    if media:
        r["images"] = ["img"]
    return r


def expected(state):
    ids = [1, 2, 3] + [len(w) + 1 for w in state.split()]
    s = float(sum(ids))
    logits = torch.tensor([(s * (k + 1)) % 5 for k in range(3)])
    probs = dict(zip(("a", "b", "c"), logits.softmax(-1).tolist(), strict=True))
    return fake_answer({"type": "choice"}, probs)["probs"]


@pytest.fixture
def started():
    created = []

    def _make(cfg=None, model=None):
        model = model or FakeModel()
        eng = Engine(cfg or make_cfg(), FakeStats(), loader=make_loader(model))
        created.append(eng)
        eng.start()
        assert wait_for(lambda: eng.status == "ready")
        return eng, model

    yield _make
    for e in created:
        e.shutdown()


# ---------------------------------------------------------------- pure helpers
def test_bucket_rounding():
    b = (128, 256, 512)
    assert bucket_for(1, b) == 128
    assert bucket_for(128, b) == 128
    assert bucket_for(129, b) == 256
    assert bucket_for(512, b) == 512
    assert bucket_for(513, b) == 1024
    assert bucket_for(1024, b) == 1024
    assert bucket_for(1025, b) == 1536


def test_pad_batch_right():
    batch = fake_collate([FEnc((5, 6, 7), ()), FEnc((9,), ())], 0, "cpu")
    out = pad_batch_right(batch, 6, 0)
    assert out["input_ids"].tolist() == [[5, 6, 7, 0, 0, 0], [9, 0, 0, 0, 0, 0]]
    assert out["attention_mask"].tolist() == [[1, 1, 1, 0, 0, 0], [1, 0, 0, 0, 0, 0]]
    assert pad_batch_right(batch, 3, 0) is batch
    with pytest.raises(ValueError):
        pad_batch_right(batch, 2, 0)


@pytest.mark.skipif(not REAL_MODEL.exists(), reason="real joint_schema_model.py not reachable")
def test_collate_matches_original_plus_right_padding():
    spec = importlib.util.spec_from_file_location("joint_schema_model_real", REAL_MODEL)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    mk = lambda ids: mod.EncodedRecord(tuple(ids), (), "r")  # noqa: E731
    recs = [mk(range(1, 6)), mk(range(1, 3)), mk(range(7, 16))]
    orig = mod.collate_records(recs, 151, torch.device("cpu"))
    padded = pad_batch_right(orig, bucket_for(9, (16, 32)), 151)
    assert padded["input_ids"].shape == (3, 16)
    assert torch.equal(padded["input_ids"][:, :9], orig["input_ids"])
    assert torch.equal(padded["attention_mask"][:, :9], orig["attention_mask"])
    assert (padded["input_ids"][:, 9:] == 151).all() and (padded["attention_mask"][:, 9:] == 0).all()
    assert padded["attention_mask"].sum(1).tolist() == [5, 2, 9]  # head slices by this
    assert padded["records"] is recs


# ---------------------------------------------------------------- lifecycle
@pytest.mark.asyncio
async def test_not_ready_before_start():
    eng = Engine(make_cfg(), FakeStats(), loader=make_loader(FakeModel()))
    with pytest.raises(EngineNotReady):
        await eng.decide([rec("hello")])


@pytest.mark.asyncio
async def test_load_error_status():
    def bad_loader(cfg):
        raise RuntimeError("boom")

    eng = Engine(make_cfg(), FakeStats(), loader=bad_loader)
    eng.start()
    assert wait_for(lambda: eng.status == "error")
    assert "boom" in eng.error
    with pytest.raises(EngineNotReady):
        await eng.decide([rec("hello")])
    eng.shutdown()


def test_unknown_dtype_is_rejected_by_config():
    with pytest.raises(ValueError, match="CLEF_DTYPE"):
        make_cfg(dtype="bfloat17")


class FakeBackend:
    """Backend stand-in recording the calls the engine makes."""

    name = "rocm"
    device = torch.device("cpu")

    def __init__(self, free_gb=100.0, name="rocm"):
        self.name = name
        self.syncs = 0
        self.cache_clears = 0
        self.telemetry_enabled = True
        self.free_gb = free_gb
        self.telemetry_calls = 0

    @property
    def has_discrete_memory(self):
        return self.name in ("cuda", "rocm")

    def apply_memory_cap(self, cap_gb):
        self.cap = cap_gb
        return True

    def synchronize(self):
        self.syncs += 1

    def empty_cache(self):
        self.cache_clears += 1

    def is_oom(self, exc):
        return isinstance(exc, torch.cuda.OutOfMemoryError)

    def device_name(self):
        return "Fake GPU 9000"

    def memory(self):
        return {
            "kind": "vram",
            "total_gb": 24.0,
            "allocated_gb": 10.0,
            "reserved_gb": 11.0,
            "free_gb": self.free_gb,
        }

    def telemetry(self):
        self.telemetry_calls += 1
        return {
            "available": True,
            "source": "amdsmi",
            "util_pct": 55.0,
            "temp_c": 61.0,
            "power_w": 200.0,
            "mem_used_gb": 10.0,
            "mem_total_gb": 24.0,
        }

    def resolve_dtype(self, requested):
        return torch.float32, "fake fallback"

    def versions(self):
        return {"torch": "9.9", "cuda": None, "hip": "7.2", "driver": None, "macos": None}

    def fast_path(self):
        return {"causal_conv1d": False, "fla": True, "expected": True}


def test_info_before_backend_detected():
    eng = Engine(make_cfg(), FakeStats(), loader=make_loader(FakeModel()))
    info = eng.info()
    assert info["backend"] is None and info["gpu"]["available"] is False
    assert info["telemetry"]["available"] is False and info["warnings"] == []
    assert eng.sample()["mem_used_gb"] is None


def test_info_and_sample_shapes_with_fake_backend(started):
    fake = FakeBackend()
    eng = Engine(make_cfg(), FakeStats(), loader=make_loader(FakeModel()), backend=fake)
    eng.start()
    assert wait_for(lambda: eng.status == "ready")
    try:
        assert eng.info()["telemetry"]["available"] is False  # nothing sampled yet
        gauges = eng.sample()
        assert gauges == {
            "mem_used_gb": 10.0,
            "mem_total_gb": 24.0,
            "gpu_util_pct": 55.0,
            "gpu_temp_c": 61.0,
            "gpu_power_w": 200.0,
        }
        info = eng.info()
        assert info["backend"] == "rocm" and info["device"] == "cpu" and info["dtype"] == "float32"
        assert info["quant"] == "none" and info["model_path"] == "/nowhere"
        assert info["gpu"] == {
            "available": True,
            "name": "Fake GPU 9000",
            "memory_kind": "vram",
            "vram_total_gb": 24.0,
            "vram_allocated_gb": 10.0,
            "vram_reserved_gb": 11.0,
            "vram_free_gb": 100.0,
        }
        assert info["telemetry"]["source"] == "amdsmi" and info["telemetry"]["util_pct"] == 55.0
        assert info["fast_path"] == {"causal_conv1d": False, "fla": True, "expected": True}
        assert info["versions"]["hip"] == "7.2"
        assert info["warnings"] == ["fake fallback"]
        calls = fake.telemetry_calls
        eng.info()
        assert fake.telemetry_calls == calls  # info() never polls telemetry
    finally:
        eng.shutdown()


def test_telemetry_off_is_not_polled():
    fake = FakeBackend()
    eng = Engine(make_cfg(telemetry=False), FakeStats(), loader=make_loader(FakeModel()), backend=fake)
    eng.start()
    assert wait_for(lambda: eng.status == "ready")
    assert eng.sample()["gpu_util_pct"] is None and fake.telemetry_calls == 0
    eng.shutdown()


def test_forward_uses_backend_sync_and_oom_clears_cache():
    fake = FakeBackend()
    eng = Engine(make_cfg(), FakeStats(), loader=make_loader(FakeModel()), backend=fake)
    eng.start()
    assert wait_for(lambda: eng.status == "ready")

    async def go():
        await eng.decide([rec("a b")])
        assert fake.syncs == 1
        with pytest.raises(GpuOutOfMemory):
            await eng.decide([rec("OOM")])
        assert fake.cache_clears == 1 and eng.status == "ready"

    asyncio.run(go())
    eng.shutdown()


def test_backend_error_is_status_error(monkeypatch):
    def boom(requested):
        raise BackendError("CLEF_DEVICE=cuda is not available: no NVIDIA GPU visible")

    monkeypatch.setattr("clef_server.engine.backend_mod.detect", boom)
    eng = Engine(make_cfg(), FakeStats(), loader=make_loader(FakeModel()))
    eng.start()
    assert wait_for(lambda: eng.status == "error")
    assert eng.error == "CLEF_DEVICE=cuda is not available: no NVIDIA GPU visible"
    eng.shutdown()


def test_nf4_without_gpu_bitsandbytes_is_status_error():
    eng = Engine(
        make_cfg(quant="nf4"), FakeStats(), loader=make_loader(FakeModel()), backend=FakeBackend(name="mps")
    )
    eng.start()
    assert wait_for(lambda: eng.status == "error")
    assert "needs an NVIDIA or AMD GPU" in eng.error and "mps" in eng.error
    eng.shutdown()


def test_preflight_failure_is_status_error():
    eng = Engine(
        make_cfg(preflight=True),
        FakeStats(),
        loader=make_loader(FakeModel()),
        backend=FakeBackend(free_gb=2.0),
    )
    eng.start()
    assert wait_for(lambda: eng.status == "error")
    assert "not enough" in eng.error and "\n" not in eng.error
    eng.shutdown()


def test_missing_model_is_status_error_with_hint(tmp_path, monkeypatch):
    monkeypatch.setattr("clef_server.paths.legacy_model_dir", lambda: tmp_path / "none")
    cfg = make_cfg(model_path=str(tmp_path / "missing"))
    eng = Engine(cfg, FakeStats(), backend=FakeBackend())  # default loader resolves the path
    eng.start()
    assert wait_for(lambda: eng.status == "error")
    assert "CLEF_MODEL_PATH" in eng.error
    eng.shutdown()


def test_shutdown_joins(started):
    eng, _ = started()
    assert eng._thread.is_alive()
    eng.shutdown()
    assert not eng._thread.is_alive()


def test_info_on_cpu(started):
    eng, _ = started()
    info = eng.info()
    assert info["backend"] == "cpu" and info["device"] == "cpu" and info["gpu"]["available"] is False
    assert info["torch"] == torch.__version__
    assert eng.load_seconds is not None


def test_warmup_covers_buckets_and_sizes(started):
    cfg = make_cfg(warmup=True, buckets=(16, 32, 64), max_microbatch=4, batch_window_ms=1.0)
    eng, model = started(cfg)
    assert eng.warmup_seconds is not None
    assert {(n, length) for n, length, _ in model.calls} == {(n, b) for n in (1, 4) for b in (16, 32, 64)}


# ---------------------------------------------------------------- serving
@pytest.mark.asyncio
async def test_single_decide_result_shape(started):
    eng, _ = started()
    (res,) = await eng.decide([rec("alpha beta")])
    assert res["model"] == "clef-flash"
    assert res["usage"] == {"input_tokens": 5, "output_tokens": 0}
    assert res["timing"]["batch_size"] == 1
    assert res["timing"]["queue_ms"] >= 0 and res["timing"]["forward_ms"] >= 0
    assert res["answers"]["d"]["probs"] == expected("alpha beta")


@pytest.mark.asyncio
async def test_no_bucket_padding_by_default(started):
    eng, model = started(cfg=make_cfg(pad_to_bucket=False, pad_multiple=1))
    states = ["a", "bb cc", "ddd"]
    await asyncio.gather(*(eng.decide([rec(s)]) for s in states))
    assert model.calls == [(3, 5, False)]  # one forward, padded only to the longest ("bb cc" = 3 + 2)


@pytest.mark.asyncio
async def test_concurrent_decides_coalesce_and_route_correctly(started):
    eng, model = started()
    states = ["a", "bb cc", "ddd", "e f g", "hhhh"]
    outs = await asyncio.gather(*(eng.decide([rec(s)]) for s in states))
    assert len(model.calls) == 1
    assert model.calls[0] == (5, 8, False)  # one forward, padded to bucket 8
    for s, out in zip(states, outs, strict=True):
        assert out[0]["answers"]["d"]["probs"] == expected(s)
        assert out[0]["timing"]["batch_size"] == 5
    n_rec, n_tok, padded, _ = eng.stats.forwards[0]
    assert n_rec == 5 and padded == 40 and n_tok == sum(3 + len(s.split()) for s in states)


@pytest.mark.asyncio
async def test_multi_record_order_preserved(started):
    eng, _ = started()
    states = ["x", "y y y y y y y y y y y", "zz zz", "w w w w w w w w w w w w w w w w w w w w"]
    out = await eng.decide([rec(s) for s in states])
    assert [o["answers"]["d"]["probs"] for o in out] == [expected(s) for s in states]
    assert [o["usage"]["input_tokens"] for o in out] == [3 + len(s.split()) for s in states]


@pytest.mark.asyncio
async def test_different_buckets_run_as_separate_forwards(started):
    eng, model = started()
    short, long = "a b", " ".join(["w"] * 20)  # 5 -> bucket 8, 23 -> bucket 32
    out = await eng.decide([rec(long), rec(short)])
    assert sorted((n, length) for n, length, _ in model.calls) == [(1, 8), (1, 32)]
    assert out[0]["answers"]["d"]["probs"] == expected(long)
    assert out[1]["answers"]["d"]["probs"] == expected(short)


@pytest.mark.asyncio
async def test_media_never_batched(started):
    eng, model = started()
    reqs = [rec("a"), rec("b b", media=True), rec("c c c"), rec("d", media=True), rec("e e")]
    out = await eng.decide(reqs)
    for n, _, has_media in model.calls:
        if has_media:
            assert n == 1
    assert sum(n for n, _, _ in model.calls) == 5
    assert sorted(n for n, _, m in model.calls if not m) == [3]
    assert [o["answers"]["d"]["probs"] for o in out] == [
        expected(s) for s in ["a", "b b", "c c c", "d", "e e"]
    ]
    # media is not bucketed: exact length
    media_lens = sorted(length for _, length, m in model.calls if m)
    assert media_lens == [4, 5]


@pytest.mark.asyncio
async def test_input_too_large_mapping(started):
    eng, model = started()
    with pytest.raises(InputTooLarge):
        await eng.decide([rec("TOO_BIG")])
    with pytest.raises(ValueError) as ei:
        await eng.decide([rec("BAD")])
    assert not isinstance(ei.value, InputTooLarge)
    assert model.calls == []


@pytest.mark.asyncio
async def test_oom_fails_group_but_worker_survives(started):
    eng, model = started()
    ok = " ".join(["w"] * 5)  # bucket 8
    oom_task = asyncio.create_task(eng.decide([rec("OOM")]))
    # OOM record: length 4 -> bucket 8 too; give it its own group by length bucket
    await asyncio.sleep(0)
    results = await asyncio.gather(oom_task, return_exceptions=True)
    assert isinstance(results[0], GpuOutOfMemory)
    out = await eng.decide([rec(ok)])
    assert out[0]["answers"]["d"]["probs"] == expected(ok)
    assert eng._thread.is_alive() and eng.status == "ready"


@pytest.mark.asyncio
async def test_oom_in_one_group_does_not_fail_other_group(started):
    eng, _ = started()
    long = " ".join(["w"] * 20)  # bucket 32; the OOM record (len 4) is bucket 8
    res = await asyncio.gather(eng.decide([rec("OOM")]), eng.decide([rec(long)]), return_exceptions=True)
    assert isinstance(res[0], GpuOutOfMemory)
    assert res[1][0]["answers"]["d"]["probs"] == expected(long)


@pytest.mark.asyncio
async def test_other_exception_fails_only_that_group(started):
    eng, model = started()
    calls = {"n": 0}
    orig = model.__class__.__call__

    def flaky(self, batch):
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("kernel exploded")
        return orig(self, batch)

    model.__class__ = type("Flaky", (FakeModel,), {"__call__": flaky})
    with pytest.raises(RuntimeError, match="kernel exploded"):
        await eng.decide([rec("a")])
    out = await eng.decide([rec("a")])
    assert out[0]["answers"]["d"]["probs"] == expected("a")


@pytest.mark.asyncio
async def test_queue_depth_reported(started):
    eng, model = started(make_cfg(batch_window_ms=0.0))
    model.gate = threading.Event()
    t1 = asyncio.create_task(eng.decide([rec("a")]))
    assert await asyncio.to_thread(model.entered.wait, 5)  # worker is now blocked in forward
    t2 = asyncio.create_task(eng.decide([rec("b"), rec("c")]))
    assert await asyncio.to_thread(wait_for, lambda: max(eng.stats.depths, default=0) >= 2)
    model.gate.set()
    await asyncio.gather(t1, t2)
    assert await asyncio.to_thread(wait_for, lambda: eng.stats.depths[-1] == 0)


@pytest.mark.asyncio
async def test_decide_empty_and_after_shutdown(started):
    eng, _ = started()
    assert await eng.decide([]) == []
    eng.shutdown()
    with pytest.raises(EngineNotReady):
        await eng.decide([rec("a")])


@pytest.mark.asyncio
async def test_pads_to_multiple_by_default(started):
    eng, model = started(cfg=make_cfg(pad_to_bucket=False, pad_multiple=4))
    await asyncio.gather(*(eng.decide([rec(s)]) for s in ["a", "bb cc"]))
    assert model.calls == [(2, 8, False)]  # longest is 5 tokens -> next multiple of 4
    assert eng.padded_length(8) == 8 and eng.padded_length(9) == 12


def test_memory_cap_is_applied_before_load():
    backend = FakeBackend()
    eng = Engine(
        make_cfg(max_device_memory_gb=14.0), FakeStats(), loader=make_loader(FakeModel()), backend=backend
    )
    eng.start()
    assert wait_for(lambda: eng.status in ("ready", "warming"))
    assert backend.cap == 14.0
    eng.shutdown()


def test_offload_on_unified_memory_is_a_warning_not_an_error():
    eng = Engine(
        make_cfg(offload="cpu"),
        FakeStats(),
        loader=make_loader(FakeModel()),
        backend=FakeBackend(name="mps"),
    )
    eng.start()
    assert wait_for(lambda: eng.status in ("ready", "warming"))
    assert any("no effect" in w for w in eng.info()["warnings"])
    eng.shutdown()


def test_info_memory_block_standard_and_offload(started):
    eng = Engine(make_cfg(), FakeStats(), loader=make_loader(FakeModel()), backend=FakeBackend())
    mem = eng.info()["memory"]  # before any load
    assert mem["mode"] == "standard" and mem["offload"] == "none" and mem["max_device_memory_gb"] is None
    assert mem["layers_on_host"] == 0 and mem["embeddings_on_host"] is False and mem["quant_backend"] is None
    cfg = make_cfg(offload="cpu", max_device_memory_gb=15.0, quant="int8")
    eng = Engine(cfg, FakeStats(), loader=make_loader(FakeModel()), backend=FakeBackend())
    eng._load_info = LoadInfo(
        mode="offload", streamed_layers=6, n_layers=32, host_weights_gb=7.5, device_weights_gb=11.0
    )
    mem = eng.info()["memory"]
    assert mem["offload"] == "cpu" and mem["mode"] == "offload" and mem["max_device_memory_gb"] == 15.0
    assert mem["embeddings_on_host"] is True and mem["layers_on_host"] == 6 and mem["layers_total"] == 32
    assert mem["quant"] == "int8" and mem["quant_backend"] == "torchao"  # FakeBackend is rocm: not bnb
    assert mem["host_weights_gb"] == 7.5 and mem["device_weights_gb"] == 11.0
