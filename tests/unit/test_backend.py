"""Backend tests: CPU only. Device availability is faked by monkeypatching torch; no GPU is ever touched."""

from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
import types
from pathlib import Path

import pytest
import torch

from clef_server import backend as bk
from clef_server.backend import Backend, BackendError, PreflightError


# ---------------------------------------------------------------- helpers
def fake_platform(monkeypatch, *, cuda=False, hip=None, mps=False, count=1, bf16=True):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: cuda)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: count)
    monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda *a, **k: bf16)
    monkeypatch.setattr(torch.version, "hip", hip, raising=False)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: mps)


def with_memory(b: Backend, free, total=24.0, kind="vram") -> Backend:
    b.memory = lambda: {
        "kind": kind,
        "total_gb": total,
        "allocated_gb": 0.0,
        "reserved_gb": 0.0,
        "free_gb": free,
    }
    b.device_name = lambda: "Test Device"
    return b


# ---------------------------------------------------------------- import hygiene
def test_module_imports_without_torch():
    code = (
        "import sys; sys.modules['torch'] = None\n"
        "import clef_server.backend as b\n"
        "print(b.prepare_environment({}))"
    )
    src = str(Path(bk.__file__).resolve().parents[1])
    env = {**os.environ, "PYTHONPATH": src}
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env)
    assert out.returncode == 0, out.stderr


# ---------------------------------------------------------------- prepare_environment
def test_prepare_environment_rocm_on_wsl(monkeypatch):
    monkeypatch.setattr(bk, "_installed_torch_version", lambda: "2.11.0+rocm7.2")
    monkeypatch.setattr(bk, "_is_wsl", lambda: True)
    monkeypatch.setattr(sys, "platform", "linux")
    env: dict = {}
    set_names = bk.prepare_environment(env)
    assert env == {"TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL": "1", "HSA_ENABLE_DXG_DETECTION": "1"}
    assert set(set_names) == set(env)


def test_prepare_environment_rocm_native_linux_has_no_dxg(monkeypatch):
    monkeypatch.setattr(bk, "_installed_torch_version", lambda: "2.11.0+rocm7.2")
    monkeypatch.setattr(bk, "_is_wsl", lambda: False)
    monkeypatch.setattr(sys, "platform", "linux")
    env: dict = {}
    bk.prepare_environment(env)
    assert "HSA_ENABLE_DXG_DETECTION" not in env and "PYTORCH_TUNABLEOP_ENABLED" not in env


def test_prepare_environment_tunableop_opt_in(monkeypatch):
    monkeypatch.setattr(bk, "_installed_torch_version", lambda: "2.11.0+rocm7.2")
    monkeypatch.setattr(bk, "_is_wsl", lambda: False)
    env = {"CLEF_TUNABLEOP": "1"}
    bk.prepare_environment(env)
    assert env["PYTORCH_TUNABLEOP_ENABLED"] == "1" and env["PYTORCH_TUNABLEOP_FILENAME"].endswith(
        "tunableop_%d.csv"
    )


@pytest.mark.parametrize("version", ["2.11.0+cu128", "2.11.0", "2.11.0+cpu", None])
def test_prepare_environment_never_sets_rocm_vars_elsewhere(monkeypatch, version):
    monkeypatch.setattr(bk, "_installed_torch_version", lambda: version)
    monkeypatch.setattr(bk, "_is_wsl", lambda: True)  # even on WSL
    monkeypatch.setattr(sys, "platform", "linux")
    env: dict = {}
    assert bk.prepare_environment(env) == [] and env == {}


def test_prepare_environment_macos_and_setdefault(monkeypatch):
    monkeypatch.setattr(bk, "_installed_torch_version", lambda: "2.11.0")
    monkeypatch.setattr(sys, "platform", "darwin")
    env: dict = {}
    assert (
        bk.prepare_environment(env) == ["PYTORCH_ENABLE_MPS_FALLBACK"]
        and env["PYTORCH_ENABLE_MPS_FALLBACK"] == "1"
    )
    env = {"PYTORCH_ENABLE_MPS_FALLBACK": "0"}
    assert bk.prepare_environment(env) == [] and env["PYTORCH_ENABLE_MPS_FALLBACK"] == "0"


def test_prepare_environment_does_not_set_offline(monkeypatch):
    monkeypatch.setattr(bk, "_installed_torch_version", lambda: "2.11.0+rocm7.2")
    env: dict = {}
    bk.prepare_environment(env)
    assert "HF_HUB_OFFLINE" not in env


# ---------------------------------------------------------------- detect
def test_detect_auto_prefers_cuda_then_rocm_then_mps_then_cpu(monkeypatch):
    fake_platform(monkeypatch, cuda=True, hip=None, mps=True)
    b = bk.detect("auto")
    assert (b.name, str(b.device)) == ("cuda", "cuda:0")
    fake_platform(monkeypatch, cuda=True, hip="7.2", mps=True)
    b = bk.detect("auto")
    assert (b.name, str(b.device)) == ("rocm", "cuda:0")
    fake_platform(monkeypatch, mps=True)
    assert bk.detect("auto").name == "mps"
    fake_platform(monkeypatch)
    b = bk.detect("auto")
    assert (b.name, str(b.device)) == ("cpu", "cpu")


def test_detect_explicit_index(monkeypatch):
    fake_platform(monkeypatch, cuda=True, count=2)
    assert str(bk.detect("cuda:1").device) == "cuda:1" and bk.detect("cuda:1").index == 1
    with pytest.raises(BackendError, match="only 2"):
        bk.detect("cuda:2")
    fake_platform(monkeypatch, cuda=True, hip="7.2", count=2)
    assert bk.detect("rocm:1").name == "rocm"


def test_detect_explicit_unavailable_has_hints(monkeypatch):
    fake_platform(monkeypatch, cuda=True, hip="7.2")
    with pytest.raises(BackendError, match="ROCm build"):
        bk.detect("cuda")
    fake_platform(monkeypatch, cuda=True, hip=None)
    with pytest.raises(BackendError, match="not a ROCm build"):
        bk.detect("rocm")
    fake_platform(monkeypatch, cuda=False, hip="7.2")
    monkeypatch.setattr(bk, "_is_wsl", lambda: True)
    with pytest.raises(BackendError, match="HSA_ENABLE_DXG_DETECTION"):
        bk.detect("rocm")
    fake_platform(monkeypatch)
    with pytest.raises(BackendError, match="mps"):
        bk.detect("mps")
    assert bk.detect("cpu").name == "cpu"


@pytest.mark.parametrize("bad", ["tpu", "cpu:1", "cuda:x", "auto:0"])
def test_detect_rejects_garbage(bad):
    with pytest.raises(BackendError):
        bk.detect(bad)


# ---------------------------------------------------------------- dtype rules
def test_dtype_cuda(monkeypatch):
    fake_platform(monkeypatch, cuda=True, bf16=True)
    b = bk.detect("cuda")
    assert b.resolve_dtype("auto") == (torch.bfloat16, None)
    assert b.resolve_dtype("float16") == (torch.float16, None)
    assert b.resolve_dtype("float32") == (torch.float32, None)
    fake_platform(monkeypatch, cuda=True, bf16=False)
    dtype, warn = b.resolve_dtype("auto")
    assert dtype == torch.float16 and "bfloat16" in warn
    dtype, warn = b.resolve_dtype("bfloat16")
    assert dtype == torch.float16 and warn


def test_dtype_rocm_is_bf16(monkeypatch):
    fake_platform(monkeypatch, cuda=True, hip="7.2")
    assert bk.detect("rocm").resolve_dtype("auto") == (torch.bfloat16, None)


def test_dtype_mps_needs_macos_14(monkeypatch):
    fake_platform(monkeypatch, mps=True)
    b = bk.detect("mps")
    monkeypatch.setattr(bk, "_macos_version", lambda: (14, 2))
    assert b.resolve_dtype("auto") == (torch.bfloat16, None)
    monkeypatch.setattr(bk, "_macos_version", lambda: (13, 6))
    dtype, warn = b.resolve_dtype("auto")
    assert dtype == torch.float16 and "macOS 14" in warn
    assert b.resolve_dtype("float16") == (torch.float16, None)


def test_dtype_cpu(monkeypatch):
    b = Backend("cpu", torch.device("cpu"))
    monkeypatch.setattr(b, "_bf16_supported", lambda: True)
    assert b.resolve_dtype("auto") == (torch.bfloat16, None)
    monkeypatch.setattr(b, "_bf16_supported", lambda: False)
    dtype, warn = b.resolve_dtype("auto")
    assert dtype == torch.float32 and warn
    dtype, warn = b.resolve_dtype("bfloat16")
    assert dtype == torch.float32 and warn
    dtype, warn = b.resolve_dtype("float16")
    assert dtype == torch.float32 and warn
    assert b.resolve_dtype("float32") == (torch.float32, None)


def test_dtype_unknown():
    with pytest.raises(BackendError, match="unknown dtype"):
        Backend("cpu", torch.device("cpu")).resolve_dtype("int4")


# ---------------------------------------------------------------- preflight
def test_preflight_error_warning_and_ok():
    b = with_memory(Backend("cuda", torch.device("cuda", 0)), free=10.0)
    with pytest.raises(PreflightError) as ei:
        bk.preflight(b, torch.bfloat16, "none", 19.0)
    msg = str(ei.value)
    assert "need ~19.0 GB" in msg and "10.0 GB free" in msg and "CLEF_QUANT=int8" in msg and "\n" not in msg
    with_memory(b, free=20.0)
    warns = bk.preflight(b, torch.bfloat16, "none", 19.0)
    assert len(warns) == 1 and "low" in warns[0]
    with_memory(b, free=30.0)
    assert bk.preflight(b, torch.bfloat16, "none", 19.0) == []


def test_preflight_scales_with_dtype_and_quant():
    b = with_memory(Backend("cuda", torch.device("cuda", 0)), free=30.0, total=32.0)
    with pytest.raises(PreflightError, match="float32"):  # fp32 doubles the weights
        bk.preflight(b, torch.float32, "none", 19.0)
    assert bk.preflight(b, torch.bfloat16, "nf4", 19.0) == []
    with_memory(b, free=12.0)
    assert bk.preflight(b, torch.bfloat16, "nf4", 19.0) == []  # ~6 GB needed
    assert bk.preflight(b, torch.bfloat16, "int8", 19.0)  # ~10.5 GB + headroom > 12 -> warning


def test_preflight_unknown_memory_passes():
    b = with_memory(Backend("cpu", torch.device("cpu")), free=None, total=None, kind="system")
    assert bk.preflight(b, torch.float32, "none", 19.0) == []


def test_preflight_mps_message():
    b = with_memory(Backend("mps", torch.device("mps")), free=8.0, total=16.0, kind="unified")
    with pytest.raises(PreflightError, match="unified memory"):
        bk.preflight(b, torch.bfloat16, "none", 19.0)


# ---------------------------------------------------------------- quantization
def test_quant_none_is_none():
    assert bk.quantization_config(Backend("cpu", torch.device("cpu")), "none", torch.float32) is None


@pytest.mark.parametrize("name", ["mps", "cpu"])
def test_nf4_refused_without_gpu_bitsandbytes(name):
    b = Backend(name, torch.device("cpu"))
    with pytest.raises(BackendError, match="needs an NVIDIA or AMD GPU"):
        bk.quantization_config(b, "nf4", torch.bfloat16)


def test_quant_method_rules():
    cuda, rocm, mps = (Backend(n, torch.device("cpu")) for n in ("cuda", "rocm", "mps"))
    assert bk.quant_method(cuda, "int8") == "bnb" and bk.quant_method(cuda, "nf4") == "bnb"
    assert bk.quant_method(rocm, "int8") == "torchao" and bk.quant_method(mps, "int8") == "torchao"
    assert bk.quant_method(rocm, "nf4") == "bnb"
    assert bk.quant_method(cuda, "int8", "torchao") == "torchao"
    assert bk.quant_method(rocm, "int8", "bnb") == "bnb"
    with pytest.raises(BackendError, match="only available with bitsandbytes"):
        bk.quant_method(cuda, "nf4", "torchao")
    with pytest.raises(BackendError, match="CLEF_QUANT_BACKEND"):
        bk.quant_method(cuda, "int8", "gguf")


def test_quant_torchao_needs_package(monkeypatch):
    monkeypatch.setattr(bk, "_has_module", lambda name: False)
    with pytest.raises(BackendError, match="pip install torchao"):
        bk.quantization_config(Backend("rocm", torch.device("cuda", 0)), "int8", torch.bfloat16)


def test_quant_torchao_builds_config(monkeypatch):
    pytest.importorskip("transformers")
    pytest.importorskip("torchao")
    qc = bk.quantization_config(Backend("rocm", torch.device("cuda", 0)), "int8", torch.bfloat16)
    assert type(qc).__name__ == "TorchAoConfig" and "lm_head" in qc.modules_to_not_convert


def test_quant_cuda_needs_bitsandbytes(monkeypatch):
    b = Backend("cuda", torch.device("cuda", 0))
    monkeypatch.setattr(bk, "_has_module", lambda name: False)
    with pytest.raises(BackendError, match="pip install bitsandbytes"):
        bk.quantization_config(b, "nf4", torch.bfloat16)


def test_quant_cuda_builds_config(monkeypatch):
    pytest.importorskip("transformers")
    b = Backend("cuda", torch.device("cuda", 0))
    monkeypatch.setattr(bk, "_has_module", lambda name: True)
    int8 = bk.quantization_config(b, "int8", torch.bfloat16)
    assert int8.load_in_8bit is True
    nf4 = bk.quantization_config(b, "nf4", torch.bfloat16)
    assert nf4.load_in_4bit is True and nf4.bnb_4bit_quant_type == "nf4"


def test_quant_unknown():
    with pytest.raises(BackendError, match="unknown CLEF_QUANT"):
        bk.quantization_config(Backend("cuda", torch.device("cuda", 0)), "fp4", torch.bfloat16)


# ---------------------------------------------------------------- offload / cap / advice
def test_preflight_cap_limits_available_memory():
    b = with_memory(Backend("cuda", torch.device("cuda", 0)), free=24.0, total=24.0)
    with pytest.raises(PreflightError, match="the cap allows 16.0 GB") as ei:
        bk.preflight(b, torch.bfloat16, "none", 19.0, cap_gb=16.0)
    assert "CLEF_OFFLOAD=cpu" in str(ei.value) and "raise CLEF_MAX_DEVICE_MEMORY_GB" in str(ei.value)
    assert bk.preflight(b, torch.bfloat16, "int8", 19.0, cap_gb=16.0) == []


def test_preflight_offload_defers_to_the_planner():
    b = with_memory(Backend("rocm", torch.device("cuda", 0)), free=8.0, total=24.0)
    assert bk.preflight(b, torch.bfloat16, "none", 19.0, cap_gb=7.0, offload="cpu") == []
    warn = bk.preflight(b, torch.bfloat16, "none", 19.0, cap_gb=12.0, offload="cpu")[0]
    assert "only 8.0 GB" in warn and "plan uses the 8.0 GB" in warn
    # offload is meaningless on unified memory: the normal check applies
    m = with_memory(Backend("mps", torch.device("mps")), free=8.0, total=16.0, kind="unified")
    with pytest.raises(PreflightError):
        bk.preflight(m, torch.bfloat16, "none", 19.0, offload="cpu")


def test_recommend_memory_setting():
    big = with_memory(Backend("cuda", torch.device("cuda", 0)), free=30.0, total=32.0)
    assert bk.recommend_memory_setting(big) is None
    mid = with_memory(Backend("rocm", torch.device("cuda", 0)), free=16.0, total=16.0)
    assert "CLEF_OFFLOAD=cpu CLEF_MAX_DEVICE_MEMORY_GB=15" in bk.recommend_memory_setting(mid)
    twelve = with_memory(Backend("cuda", torch.device("cuda", 0)), free=11.5, total=12.0)
    assert "CLEF_QUANT=int8 CLEF_OFFLOAD=cpu" in bk.recommend_memory_setting(twelve)
    eight = with_memory(Backend("cuda", torch.device("cuda", 0)), free=7.8, total=8.0)
    advice = bk.recommend_memory_setting(eight)
    assert "CLEF_OFFLOAD=cpu CLEF_MAX_DEVICE_MEMORY_GB=7" in advice and "3.5x slower" in advice
    tiny = with_memory(Backend("cuda", torch.device("cuda", 0)), free=5.0, total=6.0)
    assert "below" in bk.recommend_memory_setting(tiny)
    mac = with_memory(Backend("mps", torch.device("mps")), free=18.0, total=24.0, kind="unified")
    assert "CLEF_QUANT=int8" in bk.recommend_memory_setting(mac)


def test_apply_memory_cap_only_on_discrete_memory():
    assert Backend("cpu", torch.device("cpu")).apply_memory_cap(8.0) is False
    assert Backend("mps", torch.device("mps")).apply_memory_cap(8.0) is False
    assert Backend("cuda", torch.device("cuda", 0)).apply_memory_cap(0.0) is False


def test_pin_host_is_identity_without_discrete_memory():
    t = torch.zeros(2)
    assert Backend("cpu", torch.device("cpu")).pin_host(t) is t


# ---------------------------------------------------------------- OOM / memory / identity
def test_is_oom():
    b = Backend("cpu", torch.device("cpu"))
    assert b.is_oom(torch.cuda.OutOfMemoryError("x"))
    assert b.is_oom(MemoryError())
    assert not b.is_oom(RuntimeError("out of memory"))  # only MPS reports OOM that way
    assert not b.is_oom(ValueError("nope"))
    mps = Backend("mps", torch.device("mps"))
    assert mps.is_oom(RuntimeError("MPS backend out of memory (MPS allocated: 9 GB)"))
    assert not mps.is_oom(RuntimeError("something else"))


def test_cpu_memory_and_device_name():
    b = Backend("cpu", torch.device("cpu"))
    mem = b.memory()
    assert (
        set(mem) == {"kind", "total_gb", "allocated_gb", "reserved_gb", "free_gb"} and mem["kind"] == "system"
    )
    assert isinstance(b.device_name(), str) and b.device_name()
    b.synchronize()
    b.empty_cache()  # no-ops on cpu


def test_memory_never_raises(monkeypatch):
    b = Backend("cuda", torch.device("cuda", 0))
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda d: (_ for _ in ()).throw(RuntimeError("no gpu")))
    mem = b.memory()
    assert mem["kind"] == "vram" and mem["total_gb"] is None


def test_cuda_memory_uses_torch(monkeypatch):
    gb = 1024**3
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda d: (5 * gb, 24 * gb))
    monkeypatch.setattr(torch.cuda, "memory_allocated", lambda d: 17 * gb)
    monkeypatch.setattr(torch.cuda, "memory_reserved", lambda d: 18 * gb)
    mem = Backend("rocm", torch.device("cuda", 0)).memory()
    assert mem == {
        "kind": "vram",
        "total_gb": 24.0,
        "allocated_gb": 17.0,
        "reserved_gb": 18.0,
        "free_gb": 5.0,
    }


def test_mps_memory_is_unified(monkeypatch):
    gb = 1024**3
    monkeypatch.setattr(torch.mps, "current_allocated_memory", lambda: 2 * gb)
    monkeypatch.setattr(torch.mps, "driver_allocated_memory", lambda: 3 * gb)
    monkeypatch.setattr(torch.mps, "recommended_max_memory", lambda: 12 * gb)
    mem = Backend("mps", torch.device("mps")).memory()
    assert mem["kind"] == "unified" and mem["total_gb"] == 12.0 and mem["free_gb"] == 9.0


def test_versions_and_fast_path(monkeypatch):
    b = Backend("rocm", torch.device("cuda", 0))
    monkeypatch.setattr(torch.version, "hip", "7.2", raising=False)
    v = b.versions()
    assert set(v) == {"torch", "cuda", "hip", "driver", "macos"} and v["hip"] == "7.2" and v["driver"] is None
    monkeypatch.setattr(bk, "_has_module", lambda n: n == "fla")
    assert b.fast_path() == {"causal_conv1d": False, "fla": True, "expected": True}
    assert Backend("mps", torch.device("mps")).fast_path()["expected"] is False
    assert Backend("cpu", torch.device("cpu")).fast_path()["expected"] is False


# ---------------------------------------------------------------- telemetry
@pytest.mark.parametrize("name", ["cpu", "mps"])
def test_telemetry_unavailable_off_gpu(name):
    t = Backend(name, torch.device("cpu")).telemetry()
    assert t["available"] is False and t["source"] is None and t["util_pct"] is None


def test_telemetry_unavailable_without_pynvml(monkeypatch):
    monkeypatch.setitem(sys.modules, "pynvml", None)  # import raises ImportError
    b = Backend("cuda", torch.device("cuda", 0))
    assert b.telemetry()["available"] is False
    assert b.telemetry()["available"] is False  # stays off, still no raise


def test_telemetry_unavailable_without_amd_tools(monkeypatch):
    monkeypatch.setitem(sys.modules, "amdsmi", None)
    monkeypatch.setattr(bk, "_is_wsl", lambda: True)
    monkeypatch.setattr(bk.shutil, "which", lambda n: None)
    t = Backend("rocm", torch.device("cuda", 0)).telemetry()
    assert t == {**bk._empty_telemetry()}


def test_telemetry_disabled_flag():
    b = Backend("cuda", torch.device("cuda", 0), telemetry_enabled=False)
    assert b.telemetry()["available"] is False


def test_telemetry_nvml(monkeypatch):
    gb = 1024**3
    fake = types.SimpleNamespace(
        NVML_TEMPERATURE_GPU=0,
        nvmlInit=lambda: None,
        nvmlDeviceGetHandleByIndex=lambda i: f"h{i}",
        nvmlDeviceGetUtilizationRates=lambda h: types.SimpleNamespace(gpu=42),
        nvmlDeviceGetMemoryInfo=lambda h: types.SimpleNamespace(used=3 * gb, total=24 * gb),
        nvmlDeviceGetTemperature=lambda h, kind: 65,
        nvmlDeviceGetPowerUsage=lambda h: 250_000,
    )
    monkeypatch.setitem(sys.modules, "pynvml", fake)
    t = Backend("cuda", torch.device("cuda", 0)).telemetry()
    assert t == {
        "available": True,
        "source": "nvml",
        "util_pct": 42.0,
        "temp_c": 65.0,
        "power_w": 250.0,
        "mem_used_gb": 3.0,
        "mem_total_gb": 24.0,
    }


def test_telemetry_nvml_failure_turns_off(monkeypatch):
    def bad(i):
        raise RuntimeError("nvml shared library not found")

    monkeypatch.setitem(
        sys.modules, "pynvml", types.SimpleNamespace(nvmlInit=lambda: None, nvmlDeviceGetHandleByIndex=bad)
    )
    assert Backend("cuda", torch.device("cuda", 0)).telemetry()["available"] is False


def test_parse_rocm_smi():
    raw = json.dumps(
        {
            "card0": {
                "GPU use (%)": "37",
                "Temperature (Sensor edge) (C)": "51.0",
                "Average Graphics Package Power (W)": "180.5",
                "VRAM Total Memory (B)": str(24 * 1024**3),
                "VRAM Total Used Memory (B)": str(2 * 1024**3),
            }
        }
    )
    d = bk._parse_rocm_smi(raw)
    assert (
        d["source"] == "rocm-smi" and d["util_pct"] == 37.0 and d["temp_c"] == 51.0 and d["power_w"] == 180.5
    )
    assert d["mem_used_gb"] == 2.0 and d["mem_total_gb"] == 24.0
    assert bk._parse_rocm_smi("not json") is None
    assert bk._parse_rocm_smi("{}") is None


def test_rocm_smi_results_are_cached(monkeypatch):
    calls = []
    out = json.dumps({"card0": {"GPU use (%)": "10"}})
    monkeypatch.setitem(sys.modules, "amdsmi", None)
    monkeypatch.setattr(bk, "_is_wsl", lambda: False)
    monkeypatch.setattr(bk.shutil, "which", lambda n: "/usr/bin/rocm-smi")
    monkeypatch.setattr(bk, "_run", lambda cmd, timeout=3.0: calls.append(cmd) or out)
    b = Backend("rocm", torch.device("cuda", 0))
    assert b.telemetry()["source"] == "rocm-smi" and b.telemetry()["util_pct"] == 10.0
    assert len(calls) == 1


# ---------------------------------------------------------------- bench/parity.py helpers
def _load_parity():
    path = Path(__file__).resolve().parents[2] / "bench" / "parity.py"
    spec = importlib.util.spec_from_file_location("clef_parity", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_parity_eval_set_and_compare():
    parity = _load_parity()
    recs = parity.build_eval_set(50)
    assert len(recs) == 50 and recs == parity.build_eval_set(50)  # deterministic
    ref = [{"q.a": 0.7, "q.b": 0.3}, {"q.a": 0.2, "q.b": 0.8}]
    cand = [{"q.a": 0.65, "q.b": 0.35}, {"q.a": 0.6, "q.b": 0.4}]
    out = parity.compare(ref, cand)
    assert out["max_abs_diff"] == pytest.approx(0.4) and out["top1_flips"] == 1 and out["questions"] == 2
    with pytest.raises(ValueError):
        parity.compare([{"q.a": 1.0}], [{"q.b": 1.0}])


def test_telemetry_amdsmi_drops_impossible_vram_usage(monkeypatch):
    # Measured on the 7900 XTX under WSL: amdsmi reports more VRAM used than the card has.
    fake = types.SimpleNamespace(
        amdsmi_init=lambda: None,
        amdsmi_get_processor_handles=lambda: ["h0"],
        amdsmi_get_gpu_activity=lambda h: {"gfx_activity": 97},
        AmdSmiTemperatureType=types.SimpleNamespace(EDGE=0),
        AmdSmiTemperatureMetric=types.SimpleNamespace(CURRENT=0),
        amdsmi_get_temp_metric=lambda h, t, m: 72,
        amdsmi_get_power_info=lambda h: {"average_socket_power": "N/A"},
        amdsmi_get_gpu_vram_usage=lambda h: {"vram_total": 24560, "vram_used": 863707},
    )
    monkeypatch.setitem(sys.modules, "amdsmi", fake)
    t = Backend("rocm", torch.device("cuda", 0)).telemetry()
    assert t["available"] is True and t["source"] == "amdsmi" and t["util_pct"] == 97
    assert t["mem_used_gb"] is None and t["mem_total_gb"] == 23.98 and t["power_w"] is None


def test_telemetry_recovers_after_transient_amdsmi_error(monkeypatch):
    calls = {"n": 0}

    def activity(h):
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("AMDSMI_STATUS_BUSY")
        return {"gfx_activity": 50}

    fake = types.SimpleNamespace(
        amdsmi_init=lambda: None,
        amdsmi_get_processor_handles=lambda: ["h0"],
        amdsmi_get_gpu_activity=activity,
    )
    monkeypatch.setitem(sys.modules, "amdsmi", fake)
    monkeypatch.setattr(bk, "_is_wsl", lambda: True)
    b = Backend("rocm", torch.device("cuda", 0))
    assert b.telemetry()["available"] is False  # transient failure -> back off, not off for good
    monkeypatch.setattr(bk, "TELEMETRY_RETRY_S", 0.0)
    b._cache["tel_off_until"] = 0.0
    assert b.telemetry()["util_pct"] == 50


def test_telemetry_off_for_good_without_amdsmi_on_wsl(monkeypatch):
    monkeypatch.setitem(sys.modules, "amdsmi", None)
    monkeypatch.setattr(bk, "_is_wsl", lambda: True)
    b = Backend("rocm", torch.device("cuda", 0))
    b.telemetry()
    assert b._cache["tel_off_until"] == float("inf")


def test_offload_cap_below_floor_refused_on_wsl_warned_elsewhere(monkeypatch):
    b = with_memory(Backend("rocm", torch.device("cuda", 0)), free=24.0, total=24.0)
    monkeypatch.setattr(bk, "_is_wsl", lambda: True)
    with pytest.raises(PreflightError, match="segfaulted"):
        bk.preflight(b, torch.bfloat16, "none", 19.0, cap_gb=6.0, offload="cpu")
    monkeypatch.setattr(bk, "_is_wsl", lambda: False)
    assert "floor" in bk.preflight(b, torch.bfloat16, "none", 19.0, cap_gb=6.0, offload="cpu")[0]
    assert bk.preflight(b, torch.bfloat16, "int8", 19.0, cap_gb=6.0, offload="cpu") == []


def test_apply_memory_cap_failure_reasons(monkeypatch):
    cpu = Backend("mps", torch.device("mps"))
    assert cpu.apply_memory_cap(8.0) is False and "shares one memory pool" in cpu.cap_failure()
    b = Backend("cuda", torch.device("cuda", 0))

    def boom(*a, **k):
        raise RuntimeError("nope")

    monkeypatch.setattr(torch.cuda, "get_device_properties", boom)
    assert b.apply_memory_cap(8.0) is False and "nope" in b.cap_failure()
    b.reset_load_state()
    assert b.cap_failure() is None


def test_pin_failure_is_recorded_and_warned_once(caplog):
    b = Backend("cuda", torch.device("cuda", 0))

    class T:
        def pin_memory(self):
            raise RuntimeError("cannot pin")

    t = T()
    with caplog.at_level("WARNING", logger="clef.backend"):
        assert b.pin_host(t) is t and b.pin_host(t) is t
    assert len([r for r in caplog.records if "pin_memory failed" in r.message]) == 1
    assert "cannot pin" in b.pin_failure()
    b.reset_load_state()
    assert b.pin_failure() is None


def test_quant_caveats():
    rocm, mps, cuda = (Backend(n, torch.device("cpu")) for n in ("rocm", "mps", "cuda"))
    assert "lossy" in bk.quant_caveat(rocm, "nf4")
    assert "not validated" in bk.quant_caveat(mps, "int8")
    assert bk.quant_caveat(cuda, "nf4") is None and bk.quant_caveat(rocm, "int8") is None
    assert bk.quant_caveat(rocm, "none") is None


def test_recommend_int8_says_12_gb():
    mac = with_memory(Backend("mps", torch.device("mps")), free=18.0, total=24.0, kind="unified")
    assert "about 12 GB" in bk.recommend_memory_setting(mac)
    cpu = with_memory(Backend("cpu", torch.device("cpu")), free=8.0, total=16.0, kind="system")
    assert "about 12 GB" in bk.recommend_memory_setting(cpu)


def test_memory_bench_sampler_failure_is_unavailable_not_zero():
    path = Path(__file__).resolve().parents[2] / "bench" / "memory_bench.py"
    sys.path.insert(0, str(path.parent))
    spec = importlib.util.spec_from_file_location("clef_memory_bench", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    s = types.SimpleNamespace(samples=0, peak_used=0.0, baseline=None, error="RuntimeError: no cuda")
    m = mod.sampler_metrics(s)
    assert m["peak_device_used_gb"] is None and m["device_baseline_gb"] is None
    assert "no cuda" in m["device_sampler_error"]
    s = types.SimpleNamespace(samples=3, peak_used=9.123, baseline=1.0, error=None)
    assert mod.sampler_metrics(s) == {"peak_device_used_gb": 9.12, "device_baseline_gb": 1.0}


def test_blank_env_values_mean_unset(monkeypatch):
    """WSLENV forwards Windows variables that are unset as empty strings; they must not break config."""
    from clef_server.config import Config

    for name in (
        "CLEF_QUANT",
        "CLEF_PORT",
        "CLEF_RATE_LIMIT",
        "CLEF_TELEMETRY",
        "CLEF_MAX_DEVICE_MEMORY_GB",
        "CLEF_BUCKETS",
    ):
        monkeypatch.setenv(name, " ")
    cfg = Config()
    assert cfg.quant == "none" and cfg.port == 8910 and cfg.rate_limit == 0 and cfg.telemetry is True
    assert cfg.max_device_memory_gb == 0 and cfg.buckets[0] == 128
