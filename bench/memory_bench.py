"""Memory / latency / parity of one memory setting (dtype, CLEF_QUANT, CLEF_OFFLOAD, device cap), in-process.

Runs the real engine (load, warmup, micro-batching) and reports, for one configuration:
peak device memory (torch allocator peak and the device-wide used memory sampled every 100 ms), peak host RSS,
single-record latency (p50 / p95, end to end through Engine.decide), throughput at concurrency 8, and probability
parity against a bf16 reference on the fixed eval set (bench/parity.py, 200 records incl. choice / score / noul
and a few images).

    # 1) reference, once per session (bf16, nothing offloaded):
    python bench/memory_bench.py --label bf16 --save-reference bench/out/ref_bf16.json
    # 2) candidates, each in its own process (memory is only meaningful per process):
    python bench/memory_bench.py --label int8 --quant int8 --reference bench/out/ref_bf16.json
    python bench/memory_bench.py --label off16 --offload cpu --max-device-gb 16 --reference bench/out/ref_bf16.json

Refuses to run while a clef server holds the GPU. Results: bench/out/memory_<label>.json (+ one summary line).
"""

from __future__ import annotations

import argparse
import asyncio
import dataclasses
import json
import os
import resource
import statistics
import sys
import threading
import time
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_SRC = _HERE.parent / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))
sys.path.insert(0, str(_HERE))

from parity import build_eval_set, compare, server_running  # noqa: E402

GB = 1024**3


class _Sampler(threading.Thread):
    """Samples device-wide used memory (total - free) and process RSS until stopped."""

    def __init__(self, torch_mod, device, period: float = 0.1):
        super().__init__(daemon=True)
        self.torch, self.device, self.period = torch_mod, device, period
        self.stop_evt = threading.Event()
        self.peak_used = 0.0
        self.peak_rss = 0.0
        self.baseline = None

    def run(self) -> None:
        import psutil

        proc = psutil.Process()
        while not self.stop_evt.is_set():
            try:
                free, total = self.torch.cuda.mem_get_info(self.device)
                used = (total - free) / GB
                if self.baseline is None:
                    self.baseline = used
                self.peak_used = max(self.peak_used, used)
            except Exception:
                pass
            self.peak_rss = max(self.peak_rss, proc.memory_info().rss / GB)
            self.stop_evt.wait(self.period)


def pct(values: list[float], q: int) -> float:
    if len(values) == 1:
        return values[0]
    return statistics.quantiles(values, n=100, method="inclusive")[q - 1]


def flat_probs(enc, logits) -> dict[str, float]:
    flat = {}
    for q, ql in zip(enc.questions, logits, strict=True):
        for opt, p in zip(q.option_ids, ql.float().softmax(-1).tolist(), strict=True):
            flat[f"{q.question_id}.{opt}"] = p
    return flat


async def run(args) -> dict:
    import torch

    from clef_server.config import Config
    from clef_server.engine import Engine
    from clef_server.stats import Stats

    cfg = dataclasses.replace(
        Config(),
        model_path=args.model_path,
        device=args.device,
        dtype=args.dtype,
        quant=args.quant,
        quant_backend=args.quant_backend,
        offload=args.offload,
        max_device_memory_gb=args.max_device_gb,
        telemetry=False,
    )
    engine = Engine(cfg, Stats())
    sampler = _Sampler(torch, torch.device("cuda", 0))
    sampler.start()
    t0 = time.perf_counter()
    engine.start()
    while engine.status in ("loading",):
        await asyncio.sleep(0.5)
    if engine.status == "error":
        sampler.stop_evt.set()
        return {"label": args.label, "error": engine.error}
    load_s = time.perf_counter() - t0
    while engine.status == "warming":
        await asyncio.sleep(0.5)
    warm_s = time.perf_counter() - t0 - load_s
    info = engine.info()
    load_info = engine._load_info
    print(
        f"[{args.label}] loaded in {load_s:.1f}s, warmup {warm_s:.1f}s; {info['dtype']} quant={cfg.quant}",
        flush=True,
    )
    for note in load_info.notes if load_info else []:
        print(f"[{args.label}] {note}", flush=True)
    import psutil

    after_load_alloc = torch.cuda.memory_allocated() / GB
    steady_rss = psutil.Process().memory_info().rss / GB
    torch.cuda.reset_peak_memory_stats()

    records = build_eval_set(args.n, images=args.images)
    text_records = [r for r in records if "images" not in r]

    # parity: one record per forward through the production path (encode -> collate -> pad -> forward)
    probs = []
    for rec in records:
        enc = engine._encode(rec)
        logits, *_ = engine._forward([enc], media=enc.media is not None)
        probs.append(flat_probs(enc, logits[0]))
    result: dict = {
        "label": args.label,
        "config": {
            "dtype": info["dtype"],
            "quant": cfg.quant,
            "quant_backend": cfg.quant_backend,
            "offload": cfg.offload,
            "max_device_gb": cfg.max_device_memory_gb,
        },
        "load_s": round(load_s, 1),
        "warmup_s": round(warm_s, 1),
        "weights_on_device_gb": round(after_load_alloc, 2),
        "host_rss_after_load_gb": round(steady_rss, 2),
        "records": len(records),
    }
    if args.save_reference:
        Path(args.save_reference).parent.mkdir(parents=True, exist_ok=True)
        Path(args.save_reference).write_text(json.dumps({"probs": probs}))
        print(f"[{args.label}] wrote reference {args.save_reference}", flush=True)
    if args.reference:
        ref = json.loads(Path(args.reference).read_text())["probs"]
        result["parity"] = compare(ref, probs)

    # single-record latency, end to end
    lat = []
    for rec in text_records[: args.singles]:
        t = time.perf_counter()
        await engine.decide([rec])
        lat.append((time.perf_counter() - t) * 1000)
    result["single_p50_ms"] = round(statistics.median(lat), 1)
    result["single_p95_ms"] = round(pct(lat, 95), 1)

    # throughput at concurrency N
    work = [text_records[i % len(text_records)] for i in range(args.requests)]
    sem = asyncio.Semaphore(args.concurrency)

    async def one(rec):
        async with sem:
            await engine.decide([rec])

    t = time.perf_counter()
    await asyncio.gather(*(one(r) for r in work))
    wall = time.perf_counter() - t
    result[f"req_per_s_c{args.concurrency}"] = round(len(work) / wall, 2)

    sampler.stop_evt.set()
    sampler.join(2)
    result["peak_allocated_gb"] = round(torch.cuda.max_memory_allocated() / GB, 2)
    result["peak_reserved_gb"] = round(torch.cuda.max_memory_reserved() / GB, 2)
    result["peak_device_used_gb"] = round(sampler.peak_used, 2)
    result["device_baseline_gb"] = round(sampler.baseline or 0.0, 2)
    result["peak_host_rss_gb"] = round(
        max(sampler.peak_rss, resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6), 2
    )
    engine.shutdown()
    return result


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument(
        "--model-path", default=os.environ.get("CLEF_MODEL_PATH", str(Path.home() / "models" / "clef-flash"))
    )
    ap.add_argument("--label", required=True)
    ap.add_argument("--device", default=os.environ.get("CLEF_DEVICE", "auto"))
    ap.add_argument("--dtype", default=os.environ.get("CLEF_DTYPE", "auto"))
    ap.add_argument("--quant", default=os.environ.get("CLEF_QUANT", "none"))
    ap.add_argument("--quant-backend", default=os.environ.get("CLEF_QUANT_BACKEND", "auto"))
    ap.add_argument("--offload", default=os.environ.get("CLEF_OFFLOAD", "none"))
    ap.add_argument(
        "--max-device-gb", type=float, default=float(os.environ.get("CLEF_MAX_DEVICE_MEMORY_GB", 0))
    )
    ap.add_argument("--n", type=int, default=200, help="eval records (parity)")
    ap.add_argument("--images", type=int, default=10, help="of which image records")
    ap.add_argument("--singles", type=int, default=100)
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--requests", type=int, default=160)
    ap.add_argument("--reference", help="reference probabilities JSON (from --save-reference)")
    ap.add_argument("--save-reference", help="write this run's probabilities as the reference")
    ap.add_argument("--port", type=int, default=int(os.environ.get("CLEF_PORT", "8910")))
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--out", default=str(_HERE / "out"))
    args = ap.parse_args()
    if not args.force and server_running(args.port):
        print(
            f"REFUSING: a clef server on :{args.port} holds the GPU. Stop it or pass --force.",
            file=sys.stderr,
        )
        return 2
    result = asyncio.run(run(args))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / f"memory_{args.label}.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    return 1 if "error" in result else 0


if __name__ == "__main__":
    sys.exit(main())
