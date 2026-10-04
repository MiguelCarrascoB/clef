"""Doctor tests: CPU only, no weights, no GPU."""

from __future__ import annotations

import json

import pytest
import torch

from clef_server import backend as bk
from clef_server import doctor
from clef_server.config import Config
from clef_server.paths import ModelNotFound


@pytest.fixture(autouse=True)
def quiet_port(monkeypatch):
    monkeypatch.setattr(doctor, "_healthy_server", lambda port: None)


def test_no_gpu_exits_zero_and_skips(capsys):
    code = doctor.main(["--no-gpu"])
    out = capsys.readouterr().out
    assert "skipped (--no-gpu)" in out
    # only an unrelated process on the port or missing core deps may fail; neither is the case in CI
    assert code in (0, 1)
    if code == 1:
        assert "FAIL" in out


def test_no_gpu_does_not_touch_backend_or_weights(monkeypatch):
    def boom(*a, **k):
        raise AssertionError("must not be called with --no-gpu")

    monkeypatch.setattr(doctor, "check_backend", boom)
    monkeypatch.setattr(doctor, "check_weights_and_disk", boom)
    monkeypatch.setattr(doctor, "run_smoke", boom)
    r = doctor.run_checks(no_gpu=True, smoke=True, echo=False)
    assert any(i["name"] == "smoke" and i["level"] == doctor.SKIP for i in r.items)


def test_json_output(capsys):
    code = doctor.main(["--no-gpu", "--json"])
    payload = json.loads(capsys.readouterr().out)
    assert payload["ok"] is (code == 0)
    assert {"ok", "warnings", "failures"} == set(payload["summary"])
    assert all(set(c) == {"level", "name", "detail"} for c in payload["checks"])


def test_bad_config_is_a_failure(monkeypatch, capsys):
    monkeypatch.setenv("CLEF_DTYPE", "bfloat17")
    assert doctor.main(["--no-gpu"]) == 1
    assert "CLEF_DTYPE" in capsys.readouterr().out


def test_port_taken_by_foreign_process_fails(monkeypatch):
    class Sock:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def settimeout(self, t):
            pass

        def connect_ex(self, addr):
            return 0

    monkeypatch.setattr(doctor.socket, "socket", lambda: Sock())
    r = doctor.Report(echo=False)
    assert doctor.check_port(r, Config()) is False
    assert r.items[-1]["level"] == doctor.FAIL


def test_healthy_server_is_ok(monkeypatch):
    monkeypatch.setattr(doctor, "_healthy_server", lambda port: {"status": "ready"})
    r = doctor.Report(echo=False)
    assert doctor.check_port(r, Config()) is True and r.items[-1]["level"] == doctor.OK


def test_backend_check_on_cpu_warns_not_fails(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)
    monkeypatch.setattr(doctor.shutil, "which", lambda n: None)
    monkeypatch.setattr(bk, "_is_wsl", lambda: False)
    # host RAM must not matter: the memory preflight would FAIL on small CI runners
    monkeypatch.setattr(bk, "_cpu_memory_bytes", lambda: (2**30, 64 * 2**30, 48 * 2**30))
    r = doctor.Report(echo=False)
    b = doctor.check_backend(r, Config(device="cpu"))
    assert b is not None and b.name == "cpu"
    by_name = {i["name"]: i for i in r.items}
    assert by_name["backend"]["level"] == doctor.WARN
    assert r.ok


def test_backend_check_unavailable_explicit_device_fails(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    r = doctor.Report(echo=False)
    assert doctor.check_backend(r, Config(device="cuda")) is None
    assert r.items[-1]["level"] == doctor.FAIL and "CLEF_DEVICE=cuda" in r.items[-1]["detail"]


def test_backend_check_quant_on_cpu_fails(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(doctor.shutil, "which", lambda n: None)
    r = doctor.Report(echo=False)
    doctor.check_backend(r, Config(device="cpu", quant="nf4"))
    fails = [i for i in r.items if i["level"] == doctor.FAIL]
    assert any(i["name"] == "quantization" and "NVIDIA" in i["detail"] for i in fails)


def test_rocm_wsl_requires_dxg_env(monkeypatch):
    monkeypatch.setattr(bk, "_is_wsl", lambda: True)
    monkeypatch.delenv("HSA_ENABLE_DXG_DETECTION", raising=False)
    r = doctor.Report(echo=False)
    doctor.check_env(r, "rocm")
    assert any(i["name"] == "env HSA_ENABLE_DXG_DETECTION" and i["level"] == doctor.FAIL for i in r.items)
    monkeypatch.setenv("HSA_ENABLE_DXG_DETECTION", "1")
    r = doctor.Report(echo=False)
    doctor.check_env(r, "rocm")
    assert any(i["name"] == "env HSA_ENABLE_DXG_DETECTION" and i["level"] == doctor.OK for i in r.items)


def test_missing_weights_fail_and_check_disk(monkeypatch, tmp_path):
    def missing(cfg):
        raise ModelNotFound("model weights not found; run `clef download`")

    monkeypatch.setattr(doctor, "resolve_model_path", missing)
    monkeypatch.setenv("HF_HOME", str(tmp_path))
    r = doctor.Report(echo=False)
    doctor.check_weights_and_disk(r, Config())
    levels = {i["name"]: i["level"] for i in r.items}
    assert levels["model weights"] == doctor.FAIL and "disk space" in levels


def test_disk_space_threshold(monkeypatch, tmp_path):
    monkeypatch.setenv("HF_HOME", str(tmp_path))
    gb = 1024**3
    for free_gb, level in ((5, doctor.FAIL), (100, doctor.OK)):
        monkeypatch.setattr(
            doctor.shutil, "disk_usage", lambda p, f=free_gb: type("U", (), {"free": f * gb})()
        )
        r = doctor.Report(echo=False)
        doctor.check_disk(r, Config())
        assert r.items[-1]["level"] == level


def test_model_files_ok(monkeypatch, tmp_path):
    for f in doctor.MODEL_FILES:
        (tmp_path / f).write_bytes(b"x")
    (tmp_path / "model-00001-of-00001.safetensors").write_bytes(b"x" * 2_000_000)
    monkeypatch.setattr(doctor, "resolve_model_path", lambda cfg: tmp_path)
    r = doctor.Report(echo=False)
    doctor.check_weights_and_disk(r, Config())
    assert r.items[-1]["level"] == doctor.OK
    (tmp_path / "model-00001-of-00001.safetensors").write_bytes(b"version https://git-lfs")
    r = doctor.Report(echo=False)
    doctor.check_weights_and_disk(r, Config())
    assert r.items[-1]["level"] == doctor.FAIL and "LFS" in r.items[-1]["detail"]


def test_smoke_skipped_when_server_holds_gpu(monkeypatch):
    monkeypatch.setattr(doctor, "_healthy_server", lambda port: {"status": "ready"})
    monkeypatch.setattr(doctor, "check_backend", lambda r, cfg, server_up=False: None)
    monkeypatch.setattr(doctor, "check_weights_and_disk", lambda r, cfg: None)
    called = []
    monkeypatch.setattr(doctor, "run_smoke", lambda r, cfg: called.append(1))
    r = doctor.run_checks(smoke=True, echo=False)
    assert not called and any(i["name"] == "smoke" and i["level"] == doctor.WARN for i in r.items)


def _smoke_loader(overdue_p: float):
    """A fake joint_schema_model whose logits make the smoke record come out `overdue` / large=true."""
    import dataclasses
    from types import SimpleNamespace

    @dataclasses.dataclass(frozen=True)
    class Q:
        question_id: str
        option_ids: tuple

    @dataclasses.dataclass(frozen=True)
    class Enc:
        input_ids: tuple
        questions: tuple
        media: dict | None = None

    class Model:
        def eval(self):
            return self

        def __call__(self, batch):
            hi = torch.log(torch.tensor(overdue_p / (1 - overdue_p) * 2))
            return [[torch.tensor([0.0, hi, 0.0]), torch.tensor([0.0, 3.0])] for _ in batch["records"]]

    def encode(tok, record, max_length=0, processor=None):
        qs = (Q("status", ("paid", "overdue", "draft")), Q("large", ("false", "true")))
        return Enc((1, 2, 3), qs)

    def collate(records, pad, device):
        n = len(records)
        return {
            "input_ids": torch.ones((n, 3), dtype=torch.long),
            "attention_mask": torch.ones((n, 3), dtype=torch.long),
            "records": records,
            "media": {},
        }

    def answer(question, probs):
        if question["type"] == "noul":
            return {"type": "noul", "noul": probs["true"]}
        best = max(probs, key=probs.get)
        return {"type": "choice", "choice": best, "confidence": probs[best], "probabilities": dict(probs)}

    def loader(cfg):
        proc = SimpleNamespace(tokenizer=SimpleNamespace(pad_token_id=0))
        return SimpleNamespace(
            load_release_model=lambda path, device, dtype: (Model(), proc),
            encode_record=encode,
            collate_records=collate,
            systemone_answer=answer,
        )

    return loader


@pytest.mark.parametrize("overdue_p, level", [(0.9, doctor.OK), (0.2, doctor.FAIL)])
def test_smoke_runs_engine(monkeypatch, overdue_p, level):
    """run_smoke drives a real Engine; the model is stubbed so it runs on CPU in milliseconds."""
    from clef_server import engine as engine_mod

    real_init = engine_mod.Engine.__init__

    def patched(self, cfg, stats, loader=None, backend=None):
        real_init(self, cfg, stats, loader=_smoke_loader(overdue_p))

    monkeypatch.setattr(engine_mod.Engine, "__init__", patched)
    r = doctor.Report(echo=False)
    doctor.run_smoke(r, Config(device="cpu", dtype="float32", preflight=False), timeout=20)
    by_name = {i["name"]: i for i in r.items}
    assert by_name["smoke load"]["level"] == doctor.OK
    assert by_name["smoke probabilities"]["level"] == level
    assert "smoke latency" in by_name


def test_preflight_skipped_when_server_holds_the_gpu(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)
    monkeypatch.setattr(bk, "_is_wsl", lambda: False)
    monkeypatch.setattr(
        bk, "preflight", lambda *a, **k: (_ for _ in ()).throw(bk.PreflightError("no memory"))
    )
    r = doctor.Report(echo=False)
    doctor.check_backend(r, Config(device="cpu"), server_up=True)
    by_name = {i["name"]: i for i in r.items}
    assert by_name["memory preflight"]["level"] == doctor.SKIP and r.ok


def test_doctor_warns_on_lossy_quant_combo(monkeypatch):
    b = bk.Backend("rocm", torch.device("cpu"))
    monkeypatch.setattr(doctor.backend_mod, "detect", lambda d: b)
    monkeypatch.setattr(doctor.backend_mod, "quantization_config", lambda *a, **k: None)
    monkeypatch.setattr(doctor.backend_mod, "quant_method", lambda *a, **k: "bnb")
    r = doctor.Report(echo=False)
    doctor.check_backend(r, Config(device="rocm", quant="nf4"), server_up=True)
    q = next(i for i in r.items if i["name"] == "quantization")
    assert q["level"] == doctor.WARN and "lossy" in q["detail"]
