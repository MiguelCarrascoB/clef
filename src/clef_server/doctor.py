"""`clef doctor`: environment checks for the detected backend. Exit 0 = ok/warnings, 1 = at least one FAIL.

    clef doctor [--no-gpu] [--smoke] [--json]

--no-gpu skips everything that needs a GPU or the weights (CI smoke). --smoke loads the model on the detected
backend, runs one fixed record, checks the probabilities sum to 1 and prints the latency.
"""

from __future__ import annotations

import argparse
import importlib
import importlib.metadata
import json
import os
import re
import shutil
import socket
import sys
import time
import urllib.request
from pathlib import Path
from typing import Any

from . import backend as backend_mod
from .config import VERSION, Config
from .paths import MODEL_MARKER, WEIGHTS_GB, ModelNotFound, resolve_model_path

OK, WARN, FAIL, SKIP = "OK", "WARN", "FAIL", "SKIP"
DISK_MARGIN_GB = 2.0
MODEL_FILES = [
    "config.json",
    "joint_head.safetensors",
    "joint_head_config.json",
    MODEL_MARKER,
    "model.safetensors.index.json",
    "tokenizer.json",
    "processor_config.json",
]
REQUIRED_IMPORTS = ("fastapi", "uvicorn", "pydantic", "PIL", "httpx")
ML_IMPORTS = ("torch", "transformers", "safetensors", "huggingface_hub")
SMOKE_RECORD = {
    "model": "clef-flash",
    "state": {"invoice": {"vendor": "Acme", "total": 1250.0, "currency": "USD", "status": "overdue"}},
    "questions": {
        "status": {
            "type": "choice",
            "instructions": "What is the invoice status?",
            "criteria": {"paid": "Invoice is paid.", "overdue": "Invoice is past due.", "draft": "Not sent."},
        },
        "large": {"type": "noul", "instructions": "Is the total above 1000 USD?"},
    },
}


class Report:
    def __init__(self, echo: bool = True) -> None:
        self.items: list[dict[str, str]] = []
        self.echo = echo

    def add(self, level: str, name: str, detail: str = "") -> None:
        self.items.append({"level": level, "name": name, "detail": detail})
        if self.echo:
            print(f"[{level:4}] {name}" + (f": {detail}" if detail else ""), flush=True)

    def count(self, level: str) -> int:
        return sum(1 for i in self.items if i["level"] == level)

    @property
    def ok(self) -> bool:
        return self.count(FAIL) == 0


# ---------------------------------------------------------------------------------------------- checks
def check_python(r: Report) -> None:
    v = sys.version_info
    if v < (3, 10):
        r.add(FAIL, "python", f"{sys.version.split()[0]} (need >= 3.10)")
    else:
        r.add(OK, "python", f"{sys.version.split()[0]} on {sys.platform}")


def check_config(r: Report) -> Config | None:
    try:
        cfg = Config()
    except Exception as exc:
        r.add(FAIL, "config", str(exc))
        return None
    r.add(
        OK,
        "config",
        f"device={cfg.device} dtype={cfg.dtype} quant={cfg.quant} host={cfg.host}:{cfg.port} v{VERSION}",
    )
    return cfg


def check_imports(r: Report, no_gpu: bool) -> None:
    for mod in REQUIRED_IMPORTS:
        try:
            m = importlib.import_module(mod)
            r.add(OK, f"import {mod}", str(getattr(m, "__version__", "")))
        except Exception as exc:
            r.add(FAIL, f"import {mod}", repr(exc))
    for mod in ML_IMPORTS:
        try:
            v = importlib.metadata.version(mod.replace("_", "-"))
            r.add(OK, f"package {mod}", v)
        except importlib.metadata.PackageNotFoundError:
            level = WARN if no_gpu else FAIL
            r.add(level, f"package {mod}", "not installed" + (" (skipped with --no-gpu)" if no_gpu else ""))


def _lock_name() -> str:
    """requirements/<lock>.txt matching the installed torch build (a checkout only)."""
    try:
        torch_version = importlib.metadata.version("torch")
    except importlib.metadata.PackageNotFoundError:
        torch_version = ""
    if "+rocm" in torch_version:
        return "rocm"
    if "+cu" in torch_version:
        return "cuda"
    if sys.platform == "darwin":
        return "macos"
    return "cpu"


def check_pins(r: Report) -> None:
    name = _lock_name()
    req = Path(__file__).resolve().parents[2] / "requirements" / f"{name}.txt"
    if not req.exists():
        r.add(SKIP, "pins", f"requirements/{name}.txt not found (installed package, not a checkout)")
        return
    bad = []
    for line in req.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if ";" in line:  # platform-specific pin; not worth evaluating markers here
            continue
        m = re.match(r"^([A-Za-z0-9_.\-]+)==([^\s#]+)", line)
        if not m:
            continue
        pkg, want = m.groups()
        try:
            have = importlib.metadata.version(pkg)
        except importlib.metadata.PackageNotFoundError:
            bad.append(f"{pkg} missing (want {want})")
            continue
        if have != want:
            bad.append(f"{pkg} {have} != {want}")
    if bad:
        r.add(
            WARN,
            "pins",
            f"vs requirements/{name}.txt: " + "; ".join(bad[:8]) + (" ..." if len(bad) > 8 else ""),
        )
    else:
        r.add(OK, "pins", f"installed versions match requirements/{name}.txt")


def check_env(r: Report, backend_name: str | None) -> None:
    wsl = backend_mod._is_wsl()
    if backend_name == "rocm":
        if wsl:
            got = os.environ.get("HSA_ENABLE_DXG_DETECTION")
            r.add(OK if got == "1" else FAIL, "env HSA_ENABLE_DXG_DETECTION", got or "unset (WSL needs =1)")
        got = os.environ.get("TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL")
        r.add(
            OK if got else WARN,
            "env TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL",
            got or "unset (slower attention)",
        )
    elif backend_name == "mps":
        got = os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK")
        r.add(OK if got == "1" else WARN, "env PYTORCH_ENABLE_MPS_FALLBACK", got or "unset")


def _rocm_checks(r: Report, torch: Any) -> None:
    r.add(OK, "torch build", f"{torch.__version__} (HIP {torch.version.hip})")
    if backend_mod._is_wsl():
        r.add(OK, "wsl /dev/dxg", "present (ROCm over DirectX)")
    else:
        kfd = os.path.exists("/dev/kfd")
        r.add(OK if kfd else FAIL, "/dev/kfd", "present" if kfd else "missing: amdgpu/ROCm driver not loaded")
        try:
            import grp

            names = {grp.getgrgid(g).gr_name for g in os.getgroups()}
            ok = bool(names & {"render", "video"})
            r.add(
                OK if ok else WARN,
                "render group",
                "member" if ok else "user is not in the render/video group",
            )
        except Exception:
            pass


def _cuda_checks(r: Report, torch: Any, b: backend_mod.Backend) -> None:
    v = b.versions()
    r.add(OK, "torch build", f"{torch.__version__} (CUDA {v['cuda']})")
    r.add(
        OK if v["driver"] else WARN,
        "nvidia driver",
        v["driver"] or "unknown (pynvml / nvidia-smi unavailable)",
    )
    try:
        cap = torch.cuda.get_device_capability(b.device)
        r.add(OK, "compute capability", f"{cap[0]}.{cap[1]}")
    except Exception:
        pass


def _mps_checks(r: Report, torch: Any) -> None:
    r.add(OK, "torch build", f"{torch.__version__} (MPS built={torch.backends.mps.is_built()})")
    ver = backend_mod._macos_version()
    if ver is None:
        return
    text = ".".join(map(str, ver))
    if ver >= (14,):
        r.add(OK, "macOS", f"{text} (bfloat16 supported)")
    else:
        r.add(WARN, "macOS", f"{text}: bfloat16 on MPS needs macOS 14+, float16 will be used")


def check_backend(r: Report, cfg: Config, server_up: bool = False) -> backend_mod.Backend | None:
    try:
        import torch
    except Exception as exc:
        r.add(FAIL, "torch import", repr(exc))
        return None
    try:
        b = backend_mod.detect(cfg.device)
    except backend_mod.BackendError as exc:
        r.add(FAIL, "backend", str(exc))
        return None
    if b.name == "cpu":
        hints = [s for s in ("nvidia-smi", "rocm-smi") if shutil.which(s)]
        if backend_mod._is_wsl():
            hints.append("/dev/dxg")
        if hints and cfg.device == "auto":
            r.add(
                WARN,
                "backend",
                f"cpu, but {', '.join(hints)} found: torch {torch.__version__} cannot see the GPU",
            )
        else:
            r.add(WARN, "backend", "cpu (no GPU found): the 9B model will be very slow")
    else:
        r.add(OK, "backend", f"{b.name} -> {b.device_name()} ({b.device})")
    if b.name == "rocm":
        _rocm_checks(r, torch)
    elif b.name == "cuda":
        _cuda_checks(r, torch, b)
    elif b.name == "mps":
        _mps_checks(r, torch)
    check_env(r, b.name)

    if b.name != "cpu":
        try:
            dev = b.device
            x = torch.randn(1024, 1024, device=dev, dtype=torch.float16)
            y = float((x @ x).float().mean())
            b.synchronize()
            r.add(OK, "device matmul", f"fp16 1024x1024 ok ({y:.3f})")
        except Exception as exc:
            r.add(FAIL, "device matmul", repr(exc))

    try:
        dtype, warn = b.resolve_dtype(cfg.dtype)
        r.add(WARN if warn else OK, "dtype", warn or str(dtype).replace("torch.", ""))
    except backend_mod.BackendError as exc:
        r.add(FAIL, "dtype", str(exc))
        return b
    try:
        backend_mod.quantization_config(b, cfg.quant, dtype)
        if cfg.quant != "none":
            r.add(OK, "quantization", cfg.quant)
    except backend_mod.BackendError as exc:
        r.add(FAIL, "quantization", str(exc))

    fp = b.fast_path()
    if fp["expected"]:
        needed = {"fla": "flash-linear-attention"}
        if b.name == "cuda":
            needed["causal_conv1d"] = "causal-conv1d"
        missing = [label for key, label in needed.items() if not fp[key]]
        if missing:
            r.add(WARN, "fast path", f"missing {', '.join(missing)} (the torch fallback is slower)")
        else:
            r.add(OK, "fast path", "linear-attention kernels available")
    else:
        r.add(OK, "fast path", "torch fallback is expected on this backend")

    mem = b.memory()
    if mem["free_gb"] is not None:
        r.add(OK, f"free {mem['kind']} memory", f"{mem['free_gb']:.1f} / {mem['total_gb']:.1f} GB")
    if server_up:  # the running server already holds the weights; free memory says nothing about a new load
        r.add(SKIP, "memory preflight", "model already loaded by the running clef server")
        return b
    try:
        warns = backend_mod.preflight(b, dtype, cfg.quant, WEIGHTS_GB)
        for w in warns:
            r.add(WARN, "memory preflight", w)
        if not warns:
            r.add(OK, "memory preflight", f"enough memory for {cfg.dtype}/{cfg.quant}")
    except backend_mod.PreflightError as exc:
        r.add(FAIL, "memory preflight", str(exc))
    return b


def check_weights_and_disk(r: Report, cfg: Config) -> None:
    try:
        path = resolve_model_path(cfg)
    except ModelNotFound as exc:
        r.add(FAIL, "model weights", str(exc))
        check_disk(r, cfg)
        return
    missing = [f for f in MODEL_FILES if not (path / f).exists()]
    shards = sorted(path.glob("model-*.safetensors"))
    tiny = [s.name for s in shards if s.stat().st_size < 1_000_000]  # git-lfs pointer files
    if missing:
        r.add(FAIL, "model files", f"{path}: missing {', '.join(missing)}")
    elif not shards or tiny:
        r.add(FAIL, "model shards", f"{len(shards)} shard(s); LFS pointers instead of weights: {tiny}")
    else:
        gb = sum(s.stat().st_size for s in shards) / 1024**3
        r.add(OK, "model files", f"{path} ({len(shards)} shards, {gb:.1f} GB)")


def check_disk(r: Report, cfg: Config) -> None:
    target = Path(os.environ.get("HF_HOME") or Path.home() / ".cache" / "huggingface")
    while not target.exists() and target != target.parent:
        target = target.parent
    free = shutil.disk_usage(target).free / 1024**3
    need = WEIGHTS_GB + DISK_MARGIN_GB
    r.add(
        OK if free >= need else FAIL,
        "disk space",
        f"{free:.1f} GB free at {target}; the download needs ~{need:.0f} GB",
    )


def _healthy_server(port: int) -> dict | None:
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=3) as resp:
            return json.load(resp)
    except Exception:
        return None


def check_port(r: Report, cfg: Config) -> bool:
    """Returns True when a healthy clef server answers (it holds the GPU)."""
    body = _healthy_server(cfg.port)
    if body is not None:
        r.add(OK, "server", f"healthy on :{cfg.port}, status={body.get('status')} (it holds the GPU)")
        return True
    with socket.socket() as s:
        s.settimeout(1)
        if s.connect_ex(("127.0.0.1", cfg.port)) == 0:
            r.add(FAIL, "port", f":{cfg.port} is taken by something that is not a healthy clef server")
        else:
            r.add(OK, "port", f":{cfg.port} free on {cfg.host}")
    return False


# ---------------------------------------------------------------------------------------------- smoke
class _NullStats:
    def record_forward(self, *a: Any, **k: Any) -> None: ...

    def set_queue_depth(self, n: int) -> None: ...


def run_smoke(r: Report, cfg: Config, timeout: float = 900.0) -> None:
    import asyncio
    import dataclasses

    from .engine import Engine

    eng = Engine(dataclasses.replace(cfg, warmup=False), _NullStats())
    t0 = time.perf_counter()
    eng.start()
    try:
        while eng.status == "loading" and time.perf_counter() - t0 < timeout:
            time.sleep(0.2)
        if eng.status != "ready":
            r.add(FAIL, "smoke load", eng.error or f"engine status {eng.status} after {timeout:.0f}s")
            return
        info = eng.info()
        r.add(OK, "smoke load", f"{info['backend']} {info['dtype']} in {eng.load_seconds}s")

        async def go() -> tuple[dict, dict]:
            first = (await eng.decide([SMOKE_RECORD]))[0]
            second = (await eng.decide([SMOKE_RECORD]))[0]
            return first, second

        first, second = asyncio.run(go())
        probs = first["answers"]["status"]["probabilities"]
        total = sum(probs.values())
        ok = abs(total - 1.0) < 1e-2 and first["answers"]["status"]["choice"] == "overdue"
        r.add(
            OK if ok else FAIL,
            "smoke probabilities",
            f"sum={total:.4f} choice={first['answers']['status']['choice']} "
            f"large={first['answers']['large']['noul']}",
        )
        r.add(
            OK,
            "smoke latency",
            f"first {first['timing']['forward_ms']:.1f} ms, warm {second['timing']['forward_ms']:.1f} ms",
        )
    except Exception as exc:
        r.add(FAIL, "smoke", f"{type(exc).__name__}: {exc}")
    finally:
        eng.shutdown()


# ---------------------------------------------------------------------------------------------- entry
def run_checks(no_gpu: bool = False, smoke: bool = False, echo: bool = True) -> Report:
    r = Report(echo=echo)
    check_python(r)
    cfg = check_config(r)
    check_imports(r, no_gpu)
    check_pins(r)
    if cfg is None:
        return r
    server_up = check_port(r, cfg)
    if no_gpu:
        r.add(SKIP, "gpu / weights / disk", "skipped (--no-gpu)")
        if smoke:
            r.add(SKIP, "smoke", "skipped (--no-gpu)")
        return r
    check_backend(r, cfg, server_up=server_up)
    check_weights_and_disk(r, cfg)
    if smoke:
        if server_up:
            r.add(
                WARN, "smoke", f"skipped: a clef server on :{cfg.port} already holds the GPU (stop it first)"
            )
        elif r.ok:
            run_smoke(r, cfg)
        else:
            r.add(SKIP, "smoke", "skipped: fix the failures above first")
    return r


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="clef doctor", description=__doc__.split("\n\n")[0])
    ap.add_argument("--no-gpu", action="store_true", help="skip GPU, weights and disk checks (CI)")
    ap.add_argument(
        "--smoke", action="store_true", help="load the model, run one record, check probabilities"
    )
    ap.add_argument("--json", action="store_true", help="print the result as JSON")
    args = ap.parse_args(argv)
    if not args.json:
        print(f"clef doctor v{VERSION} - python {sys.version.split()[0]}")
    r = run_checks(no_gpu=args.no_gpu, smoke=args.smoke, echo=not args.json)
    fails, warns = r.count(FAIL), r.count(WARN)
    if args.json:
        print(
            json.dumps(
                {
                    "ok": r.ok,
                    "summary": {"ok": r.count(OK), "warnings": warns, "failures": fails},
                    "checks": r.items,
                },
                indent=2,
            )
        )
    else:
        print(f"\n{r.count(OK)} ok, {warns} warnings, {fails} failures")
    return 0 if r.ok else 1


if __name__ == "__main__":
    sys.exit(main())
