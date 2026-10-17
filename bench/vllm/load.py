"""Load test for any /v1/systemone server (clef or vLLM) with identical requests, shape checks and a VRAM peak.

    python bench/vllm/load.py --url http://127.0.0.1:8000 --concurrency 1,8,16,32 --requests 100 \\
        --out bench/out/vllm/bench_vllm.txt --vram-json bench/out/vllm/vram_vllm.json

Reuses clef_server.httpbench (the `clef bench` generator: same bodies, same percentiles) and prints the same table, so
bench/perf_table.py's parser reads it. Differences: every response is validated against the request
(vllm_client.check_response; a mismatch counts as an error, never a silent 0), the first error body is printed, and
the device VRAM is sampled while the load runs: torch.cuda.mem_get_info() of the host venv, which is device-wide
(it includes whatever else uses the card), so the report is "peak used minus the idle value taken at start".
Exit 1 if any request failed. --idle-gb: the value from --print-vram taken before the server started.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import threading
import time
from pathlib import Path

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_HERE.parents[1] / "src"))

import httpx  # noqa: E402
from vllm_client import check_response  # noqa: E402

from clef_server import httpbench  # noqa: E402


def vram_used_gb() -> tuple[float, float] | None:
    """(used, total) GB of device 0, or None when torch / the GPU is not usable here."""
    try:
        import torch

        free, total = torch.cuda.mem_get_info(0)
        return (total - free) / 1e9, total / 1e9
    except Exception:
        return None


class VramSampler(threading.Thread):
    def __init__(self, every: float = 0.5):
        super().__init__(daemon=True)
        self.every, self.peak, self.total, self._stop_evt = every, 0.0, 0.0, threading.Event()

    def run(self) -> None:
        while not self._stop_evt.is_set():
            got = vram_used_gb()
            if got:
                self.peak, self.total = max(self.peak, got[0]), got[1]
            self._stop_evt.wait(self.every)

    def stop(self) -> None:
        self._stop_evt.set()
        self.join(2)


async def amain(args: argparse.Namespace) -> int:
    levels = [int(x) for x in args.concurrency.split(",")]
    now = vram_used_gb()
    idle = (
        (args.idle_gb, now[1]) if (args.idle_gb is not None and now) else now
    )  # idle: before the server started
    sampler = VramSampler()
    sampler.start()
    failed = False
    limits = httpx.Limits(max_connections=max(64, 2 * max(levels)))
    headers = {"X-API-Key": args.api_key} if args.api_key else {}
    async with httpx.AsyncClient(base_url=args.url, headers=headers, timeout=600, limits=limits) as c:
        for _ in range(args.warmup):  # warm shapes through the real path, shape-checked too
            r = await httpbench.run_level(c, max(levels), max(levels), args.mixed, check_response)
            if r["errors"]:
                print(f"warm-up errors: {r['errors']} {r['error_samples']}", file=sys.stderr)
        print(f"server {args.url} label={args.label}")
        print(
            f"\n{'conc':>4} {'ok':>5} {'err':>4} {'req/s':>7} | {'p50':>7} {'p95':>7} {'p99':>7} ms | server batch_size distribution"
        )
        for conc in levels:
            r = await httpbench.run_level(c, conc, args.requests, args.mixed, check_response)
            dist = " ".join(f"{k}:{v}" for k, v in r["batch_sizes"].items())
            total = sum(r["batch_sizes"].values()) or 1
            avg = sum(k * v for k, v in r["batch_sizes"].items()) / total
            err = sum(r["errors"].values())
            print(
                f"{conc:>4} {r['ok']:>5} {err:>4} {r['rps']:>7.1f} | {r['p50']:>7.1f} {r['p95']:>7.1f} {r['p99']:>7.1f}    "
                f"| avg {avg:.2f}  [{dist}]" + (f"  errors={r['errors']}" if err else "")
            )
            if err:
                failed = True
                print(f"  first error bodies: {r['error_samples']}", file=sys.stderr)
    time.sleep(args.every_settle)
    sampler.stop()
    vram = {
        "label": args.label,
        "idle_gb": idle[0] if idle else None,
        "peak_gb": sampler.peak if idle else None,
        "total_gb": idle[1] if idle else None,
        "delta_gb": (sampler.peak - idle[0]) if idle else None,
    }
    if args.vram_json:
        Path(args.vram_json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.vram_json).write_text(json.dumps(vram))
    print(f"VRAM (device-wide, torch mem_get_info): {vram}")
    return 1 if failed else 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", default="http://127.0.0.1:8000")
    ap.add_argument("--api-key")
    ap.add_argument("--concurrency", default="1,8,16,32")
    ap.add_argument("--requests", type=int, default=100)
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--mixed", action="store_true")
    ap.add_argument("--idle-gb", type=float, help="device VRAM used before the server started (--print-vram)")
    ap.add_argument(
        "--print-vram", action="store_true", help="print the device VRAM used in GB (or nan) and exit"
    )
    ap.add_argument("--label", default="server")
    ap.add_argument("--vram-json")
    ap.add_argument(
        "--every-settle", type=float, default=1.0, help="seconds to keep sampling VRAM after the last level"
    )
    args = ap.parse_args(argv)
    if args.print_vram:
        got = vram_used_gb()
        print(f"{got[0]:.3f}" if got else "nan")
        return 0
    return asyncio.run(amain(args))


if __name__ == "__main__":
    sys.exit(main())
