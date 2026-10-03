"""In-process GPU benchmark for clef-flash (no server). Loads the model itself.

Methodology: every measured shape is warmed up first; collate is timed separately from forward; tensors
are synchronised before and after each timed region. Reports forward-only latency (percentiles) and
end-to-end (encode + collate + forward). Refuses to run while a clef server holds the GPU.

    python bench/bench.py --singles 100 --batches 30 --batch-sizes 1,2,4,8,16 [--mixed]

For the realistic production number (HTTP + micro-batching) use bench/http_bench.py.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
import urllib.request
from pathlib import Path

PARAMS = 9e9  # clef-flash parameter count, for the FLOPs estimate


def pct(values: list[float], q: int) -> float:
    """q-th percentile (1..99) with linear interpolation; works for tiny samples."""
    if len(values) == 1:
        return values[0]
    return statistics.quantiles(values, n=100, method="inclusive")[q - 1]


def summarize(ms: list[float]) -> dict[str, float]:
    return {
        "p50": pct(ms, 50),
        "p90": pct(ms, 90),
        "p99": pct(ms, 99),
        "mean": statistics.fmean(ms),
        "min": min(ms),
        "max": max(ms),
    }


def make_records(n: int, mixed: bool) -> list[dict]:
    """Ticket-triage records. With mixed=True the state length varies ~10x to exercise padding."""
    recs = []
    for i in range(n):
        reps = (1, 2, 4, 8, 16, 32)[i % 6] if mixed else 1
        text = "Checkout errors, orders blocked since 03:00 UTC. " * reps
        recs.append(
            {
                "state": {
                    "ticket": {
                        "id": f"t{i}",
                        "text": text,
                        "customers_affected": 1200 + i,
                        "service": "orders-api",
                    }
                },
                "questions": {
                    "department": {
                        "type": "choice",
                        "instructions": "Which team owns this?",
                        "criteria": {"billing": "Payments or invoices", "technical": "Bugs or outages"},
                    },
                    "urgency": {
                        "type": "score",
                        "instructions": "How urgent?",
                        "criteria": ["Can wait", "This week", "Today"],
                    },
                    "outage": {"type": "noul", "instructions": "Is a service down?"},
                },
            }
        )
    return recs


def server_running(port: int) -> bool:
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/livez", timeout=2):
            return True
    except Exception:
        return False


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument(
        "--model-path", default=os.environ.get("CLEF_MODEL_PATH", str(Path.home() / "models" / "clef-flash"))
    )
    ap.add_argument("--singles", type=int, default=100, help="timed single-record runs (>=100 recommended)")
    ap.add_argument("--batches", type=int, default=30, help="timed runs per batch size (>=30 recommended)")
    ap.add_argument("--batch-sizes", default="1,2,4,8,16")
    ap.add_argument("--mixed", action="store_true", help="mixed-length records (default: uniform)")
    ap.add_argument("--warmup", type=int, default=3, help="warm-up forwards per distinct shape")
    ap.add_argument("--port", type=int, default=int(os.environ.get("CLEF_PORT", "8910")))
    ap.add_argument("--force", action="store_true", help="run even if a server answers on --port")
    ap.add_argument("--json", metavar="PATH", help="also write results as JSON (e.g. bench/out/bench.json)")
    args = ap.parse_args()

    if not args.force and server_running(args.port):
        print(
            f"REFUSING: a clef server is answering on :{args.port} and holds the GPU (19+ GB VRAM).\n"
            f"Stop it first (.\\clef.ps1 stop) or pass --force. For load tests of the live server use "
            f"bench/http_bench.py.",
            file=sys.stderr,
        )
        return 2

    import torch

    sys.path.insert(0, args.model_path)
    from joint_schema_model import collate_records, encode_record, load_release_model

    dev = torch.device("cuda")
    t0 = time.perf_counter()
    model, processor = load_release_model(args.model_path, device="cuda")
    print(f"model loaded in {time.perf_counter() - t0:.1f}s")
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()  # measure inference peak, not load transients
    tok, pad = processor.tokenizer, processor.tokenizer.pad_token_id

    sizes = sorted({int(x) for x in args.batch_sizes.split(",")})
    pool = make_records(max(args.singles, 64), args.mixed)
    t0 = time.perf_counter()
    encoded = [encode_record(tok, r, processor=processor) for r in pool]
    encode_ms = (time.perf_counter() - t0) * 1000 / len(pool)
    lens = [len(e.input_ids) for e in encoded]
    print(
        f"records: {len(pool)} ({'mixed' if args.mixed else 'uniform'} length), tokens/record "
        f"min {min(lens)} / mean {statistics.fmean(lens):.0f} / max {max(lens)}; "
        f"encode {encode_ms:.2f} ms/record"
    )

    def run_forward(group):
        batch = collate_records(group, pad, dev)
        return model(batch)

    results = []
    with torch.inference_mode():
        for bs in sizes:
            runs = args.singles if bs == 1 else args.batches
            groups = [[encoded[(r * bs + k) % len(encoded)] for k in range(bs)] for r in range(runs)]
            # warm up EVERY distinct (batch, padded length) shape before timing
            shapes = {}
            for g in groups:
                shapes.setdefault((len(g), max(len(e.input_ids) for e in g)), g)
            for g in shapes.values():
                for _ in range(args.warmup):
                    run_forward(g)
            torch.cuda.synchronize()

            fwd, col, toks, padded = [], [], [], []
            for g in groups:
                torch.cuda.synchronize()
                t = time.perf_counter()
                batch = collate_records(g, pad, dev)
                torch.cuda.synchronize()
                col.append((time.perf_counter() - t) * 1000)
                t = time.perf_counter()
                model(batch)
                torch.cuda.synchronize()
                fwd.append((time.perf_counter() - t) * 1000)
                n = sum(len(e.input_ids) for e in g)
                toks.append(n)
                padded.append(len(g) * max(len(e.input_ids) for e in g))
            e2e = [f + c + encode_ms * bs for f, c in zip(fwd, col, strict=False)]
            f, e = summarize(fwd), summarize(e2e)
            tok_mean = statistics.fmean(toks)
            tflops = 2 * PARAMS * tok_mean / (f["p50"] / 1000) / 1e12
            results.append(
                {
                    "batch_size": bs,
                    "runs": runs,
                    "shapes": len(shapes),
                    "tokens_mean": tok_mean,
                    "padded_mean": statistics.fmean(padded),
                    "forward_ms": f,
                    "collate_ms": statistics.fmean(col),
                    "e2e_ms": e,
                    "records_per_s": bs / (f["p50"] / 1000),
                    "tflops_est": tflops,
                }
            )

    peak = torch.cuda.max_memory_allocated() / 1e9
    print(
        f"\n{'bs':>3} {'runs':>5} {'tok/rec':>8} {'pad%':>5} | {'fwd p50':>8} {'p90':>8} {'p99':>8} | "
        f"{'e2e p50':>8} {'p99':>8} | {'rec/s':>7} {'TFLOPS*':>8}"
    )
    for r in results:
        pad_pct = 100 * (1 - r["tokens_mean"] / r["padded_mean"])
        print(
            f"{r['batch_size']:>3} {r['runs']:>5} {r['tokens_mean'] / r['batch_size']:>8.0f} {pad_pct:>4.0f}% | "
            f"{r['forward_ms']['p50']:>8.1f} {r['forward_ms']['p90']:>8.1f} {r['forward_ms']['p99']:>8.1f} | "
            f"{r['e2e_ms']['p50']:>8.1f} {r['e2e_ms']['p99']:>8.1f} | "
            f"{r['records_per_s']:>7.1f} {r['tflops_est']:>8.1f}"
        )
    print("latencies in ms. fwd = forward only (collate excluded); e2e = encode + collate + forward.")
    print("* TFLOPS = 2 * 9e9 * real_tokens / fwd_p50 (padding not counted; a lower bound of hardware work).")
    print(
        f"peak VRAM (after load): {peak:.2f} GB of {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB"
    )

    if args.json:
        out = Path(args.json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps({"mixed": args.mixed, "peak_vram_gb": peak, "results": results}, indent=2))
        print("wrote", out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
