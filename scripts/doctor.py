"""Preflight for the clef stack: GPU, versions, model files, env vars, port. Never loads model weights.

Exit code 0 = all required checks pass (warnings allowed), 1 = at least one failure.
"""

from __future__ import annotations

import importlib
import json
import os
import re
import socket
import sys
import urllib.request
from importlib import metadata
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODEL_FILES = [
    "config.json",
    "joint_head.safetensors",
    "joint_head_config.json",
    "joint_schema_model.py",
    "model.safetensors.index.json",
    "tokenizer.json",
    "processor_config.json",
]
ENV_REQUIRED = {"HSA_ENABLE_DXG_DETECTION": "1", "HF_HUB_OFFLINE": "1"}
ENV_RECOMMENDED = ["HSA_OVERRIDE_GFX_VERSION", "PYTORCH_HIP_ALLOC_CONF", "TRITON_CACHE_DIR"]

results: list[tuple[str, str, str]] = []  # (level, name, detail)


def record(level: str, name: str, detail: str = "") -> None:
    results.append((level, name, detail))
    print(f"[{level:4}] {name}" + (f": {detail}" if detail else ""), flush=True)


def check_pins() -> None:
    req = ROOT / "requirements" / "server.txt"
    if not req.exists():
        record("WARN", "pins", "requirements/server.txt not found")
        return
    bad = []
    for line in req.read_text().splitlines():
        m = re.match(r"^([A-Za-z0-9_.\-]+)==([^\s#]+)", line.strip())
        if not m:
            continue
        name, want = m.groups()
        try:
            have = metadata.version(name)
        except metadata.PackageNotFoundError:
            bad.append(f"{name} missing (want {want})")
            continue
        if have != want:
            bad.append(f"{name} {have} != {want}")
    if bad:
        record("WARN", "pins", "; ".join(bad[:8]) + (" ..." if len(bad) > 8 else ""))
    else:
        record("OK", "pins", "installed versions match requirements/server.txt")


def check_gpu() -> None:
    try:
        import torch
    except Exception as exc:
        record("FAIL", "torch import", repr(exc))
        return
    hip = getattr(torch.version, "hip", None)
    if hip:
        record("OK", "torch build", f"{torch.__version__} (HIP {hip})")
    else:
        record("FAIL", "torch build", f"{torch.__version__} is not a ROCm build")
    if not torch.cuda.is_available():
        record("FAIL", "cuda available", "False (is HSA_ENABLE_DXG_DETECTION=1 set? Windows driver current?)")
        return
    props = torch.cuda.get_device_properties(0)
    record(
        "OK", "gpu", f"{props.name}, {props.total_memory / 1e9:.1f} GB, {torch.cuda.device_count()} device(s)"
    )
    try:
        x = torch.randn(2048, 2048, device="cuda", dtype=torch.bfloat16)
        y = float((x @ x).float().mean())
        torch.cuda.synchronize()
        record("OK", "gpu matmul", f"bf16 2048x2048 ok ({y:.3f})")
    except Exception as exc:
        record("FAIL", "gpu matmul", repr(exc))
    free, total = torch.cuda.mem_get_info()
    level = "OK" if free > 22e9 else "WARN"
    record(level, "free VRAM", f"{free / 1e9:.1f} / {total / 1e9:.1f} GB (model needs ~19.6 GB)")


def check_imports() -> None:
    for mod in ("transformers", "safetensors", "PIL", "fastapi", "uvicorn", "pydantic", "imageio", "fla"):
        try:
            m = importlib.import_module(mod)
            record("OK", f"import {mod}", str(getattr(m, "__version__", "")))
        except Exception as exc:
            record("FAIL" if mod != "fla" else "WARN", f"import {mod}", repr(exc))


def check_model() -> None:
    path = Path(os.environ.get("CLEF_MODEL_PATH", Path.home() / "models" / "clef-flash"))
    if not path.is_dir():
        record("FAIL", "model dir", f"{path} missing (run scripts/wsl_setup_user.sh)")
        return
    missing = [f for f in MODEL_FILES if not (path / f).exists()]
    shards = sorted(path.glob("model-*.safetensors"))
    tiny = [s.name for s in shards if s.stat().st_size < 1_000_000]  # git-lfs pointer files
    if missing:
        record("FAIL", "model files", "missing: " + ", ".join(missing))
    elif not shards or tiny:
        record("FAIL", "model shards", f"{len(shards)} shard(s); LFS pointers instead of weights: {tiny}")
    else:
        gb = sum(s.stat().st_size for s in shards) / 1e9
        record("OK", "model files", f"{path} ({len(shards)} shards, {gb:.1f} GB)")


def check_env() -> None:
    for key, want in ENV_REQUIRED.items():
        got = os.environ.get(key)
        record("OK" if got == want else "FAIL", f"env {key}", got or "unset (source scripts/env.sh)")
    for key in ENV_RECOMMENDED:
        got = os.environ.get(key)
        record("OK" if got else "WARN", f"env {key}", got or "unset")


def check_port() -> None:
    host = os.environ.get("CLEF_HOST", "127.0.0.1")
    port = int(os.environ.get("CLEF_PORT", "8910"))
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=3) as r:
            body = json.load(r)
        record("OK", "server", f"healthy on :{port}, status={body.get('status')} (server holds the GPU)")
        return
    except Exception:
        pass
    with socket.socket() as s:
        s.settimeout(1)
        if s.connect_ex(("127.0.0.1", port)) == 0:
            record("FAIL", "port", f":{port} is taken by something that is not a healthy clef server")
        else:
            record("OK", "port", f":{port} free on {host}")


def main() -> int:
    print(f"clef doctor - python {sys.version.split()[0]}, repo {ROOT}")
    check_env()
    check_imports()
    check_pins()
    check_gpu()
    check_model()
    check_port()
    fails = [r for r in results if r[0] == "FAIL"]
    warns = [r for r in results if r[0] == "WARN"]
    print(f"\n{len(results) - len(fails) - len(warns)} ok, {len(warns)} warnings, {len(fails)} failures")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
