"""Try enabling the Qwen3.5 linear-attention fast path on ROCm:
1) flash-linear-attention (Triton kernels - should work on ROCm)
2) causal-conv1d (compiled CUDA extension - builds via HIP on ROCm; may take minutes)
Prints verdicts; never fails the caller.
"""
import subprocess
import sys


def try_install(pkg: str, extra_args: list[str] | None = None) -> bool:
    args = [sys.executable, "-m", "pip", "install", "-q"]
    if extra_args:
        args += extra_args
    args.append("--no-build-isolation")
    args.append(pkg)
    print("installing:", pkg, flush=True)
    proc = subprocess.run(args, capture_output=True, text=True)
    tail = (proc.stdout + proc.stderr)[-600:]
    print(tail, flush=True)
    return proc.returncode == 0


def can_import(name: str) -> bool:
    proc = subprocess.run(
        [sys.executable, "-c", f"import {name}"],
        capture_output=True, text=True,
        env={"PATH": "/usr/bin:/bin:~/venvs/clef/bin"},
    )
    return proc.returncode == 0


ok_fla = try_install("flash-linear-attention")
print("fla installed:", ok_fla and can_import("fla"), flush=True)

# causal-conv1d: prebuilt wheels only for CUDA; on ROCm build from source via HIP
ok_cc = try_install("causal-conv1d")
print("causal-conv1d installed:", ok_cc and can_import("causal_conv1d"), flush=True)
