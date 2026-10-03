"""torch.profiler view of one warm single-record forward and one batch-8 forward. Loads the model.

    python bench/profile_forward.py          # prints top-20 kernels + time groups; trace -> bench/out/*.json

Open the chrome traces at chrome://tracing or https://ui.perfetto.dev. Refuses to run if a server holds the GPU.
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import sys
import time
import urllib.request
from pathlib import Path

_HERE = Path(__file__).resolve().parent

_spec = importlib.util.spec_from_file_location("clef_bench", _HERE / "bench.py")
clef_bench = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(clef_bench)
add_backend_args, load_model, make_records = (
    clef_bench.add_backend_args,
    clef_bench.load_model,
    clef_bench.make_records,
)

# heuristic kernel-name groups, first match wins
GROUPS = [
    ("attention", ("attn", "attention", "flash", "sdpa", "softmax", "aotriton")),
    (
        "fla(linear attn)",
        ("fla", "chunk", "gated_delta", "delta_rule", "recurrent", "fwd_prepare", "wy_fast", "l2norm"),
    ),
    ("conv", ("conv",)),
    (
        "GEMM",
        ("gemm", "cijk", "matmul", "mm_", "wmma", "hipblas", "rocblas", "cublas", "addmm", "bmm", "tensile"),
    ),
]


def group_of(name: str) -> str:
    low = name.lower()
    for group, keys in GROUPS:
        if any(k in low for k in keys):
            return group
    return "head/other (elementwise, norm, memcpy, ...)"


def server_running(port: int) -> bool:
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/livez", timeout=2):
            return True
    except Exception:
        return False


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--model-path", default=os.environ.get("CLEF_MODEL_PATH", str(Path.home() / "models" / "clef-flash"))
    )
    ap.add_argument("--out", default=str(_HERE / "out"))
    ap.add_argument("--port", type=int, default=int(os.environ.get("CLEF_PORT", "8910")))
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--force", action="store_true")
    add_backend_args(ap)
    args = ap.parse_args()
    if not args.force and server_running(args.port):
        print(
            f"REFUSING: a clef server holds the GPU on :{args.port}. Stop it or pass --force.",
            file=sys.stderr,
        )
        return 2

    import torch
    from torch.autograd import DeviceType
    from torch.profiler import ProfilerActivity, profile

    sys.path.insert(0, args.model_path)
    from joint_schema_model import collate_records, encode_record

    backend, _dtype, model, processor = load_model(args.model_path, args.device, args.dtype, args.quant)
    tok, pad, dev = processor.tokenizer, processor.tokenizer.pad_token_id, backend.device
    activities = [ProfilerActivity.CPU]
    if backend.name in ("cuda", "rocm"):
        activities.append(ProfilerActivity.CUDA)
    elif backend.name == "mps":
        print("note: torch.profiler has no device kernel timings on MPS; only CPU-side ops are listed")
    enc = [encode_record(tok, r, processor=processor) for r in make_records(args.batch, False)]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    for label, group in (("single", enc[:1]), (f"batch{args.batch}", enc[: args.batch])):
        batch = collate_records(group, pad, dev)
        with torch.inference_mode():
            for _ in range(3):
                model(batch)
            backend.synchronize()
            t = time.perf_counter()
            with profile(activities=activities) as prof:
                model(batch)
                backend.synchronize()
            wall = (time.perf_counter() - t) * 1000
        events = [e for e in prof.key_averages()]

        def dev_time(e):
            return getattr(e, "self_device_time_total", None) or getattr(e, "self_cuda_time_total", 0)

        # Only real device kernels: on ROCm, CPU-side ops (aten::mm, autograd Functions) also report
        # self device time for the kernels they launch, which would double count.
        def is_kernel(e) -> bool:
            if e.device_type != DeviceType.CUDA:
                return False
            return not e.key.startswith("aten::") and not e.key.endswith("Function")

        kernels = sorted((e for e in events if is_kernel(e) and dev_time(e) > 0), key=dev_time, reverse=True)
        total = sum(dev_time(e) for e in kernels) or 1
        print(
            f"\n=== {label}: {len(group)} record(s), {sum(len(e.input_ids) for e in group)} tokens, "
            f"wall {wall:.1f} ms (profiler overhead included), GPU kernel time {total / 1000:.1f} ms"
        )
        print(f"{'self GPU ms':>11} {'%':>5} {'calls':>6}  kernel")
        for e in kernels[:20]:
            print(
                f"{dev_time(e) / 1000:>11.2f} {100 * dev_time(e) / total:>4.1f}% {e.count:>6}  {e.key[:100]}"
            )
        grouped: dict[str, float] = {}
        for e in kernels:
            grouped[group_of(e.key)] = grouped.get(group_of(e.key), 0.0) + dev_time(e)
        print("-- grouped by name heuristic --")
        for g, v in sorted(grouped.items(), key=lambda kv: -kv[1]):
            print(f"{v / 1000:>11.2f} {100 * v / total:>4.1f}%  {g}")
        trace = out / f"trace_{label}.json"
        prof.export_chrome_trace(str(trace))
        print("trace:", trace)
    return 0


if __name__ == "__main__":
    sys.exit(main())
