# ruff: noqa: E501
"""Async HTTP load test against a LIVE clef server (exercises micro-batching). The realistic benchmark.

    clef bench --concurrency 1,2,4,8,16 --requests 100 [--mixed] [--url http://127.0.0.1:8910]
    (bench/http_bench.py is a shim for the same entry point)

Env: CLEF_URL, CLEF_API_KEY.
"""

from __future__ import annotations

import argparse
import asyncio
import collections
import os
import statistics
import sys
import time

import httpx


def pct(values: list[float], q: int) -> float:
    if not values:
        return float("nan")
    if len(values) == 1:
        return values[0]
    return statistics.quantiles(values, n=100, method="inclusive")[q - 1]


def make_body(i: int, mixed: bool) -> dict:
    reps = (1, 2, 4, 8, 16, 32)[i % 6] if mixed else 1
    return {
        "model": "clef-flash",
        "state": {
            "ticket": {
                "id": f"t{i}",
                "text": "Checkout errors, orders blocked since 03:00 UTC. " * reps,
                "customers_affected": 1200 + i,
            }
        },
        "questions": {
            "department": {
                "type": "choice",
                "instructions": "Which team owns this?",
                "criteria": {"billing": "Payments or invoices", "technical": "Bugs or outages"},
            },
            "urgency": {"type": "score", "criteria": ["Can wait", "This week", "Today"]},
            "outage": {"type": "noul", "instructions": "Is a service down?"},
        },
    }


async def run_level(client: httpx.AsyncClient, conc: int, n: int, mixed: bool) -> dict:
    sem = asyncio.Semaphore(conc)
    lat: list[float] = []
    batch_sizes: collections.Counter = collections.Counter()
    errors: collections.Counter = collections.Counter()

    async def one(i: int) -> None:
        async with sem:
            t = time.perf_counter()
            try:
                r = await client.post("/v1/systemone", json=make_body(i, mixed))
            except httpx.HTTPError as exc:
                errors[type(exc).__name__] += 1
                return
            ms = (time.perf_counter() - t) * 1000
            if r.status_code != 200:
                errors[str(r.status_code)] += 1
                return
            lat.append(ms)
            batch_sizes[r.json().get("timing", {}).get("batch_size", 0)] += 1

    t0 = time.perf_counter()
    await asyncio.gather(*(one(i) for i in range(n)))
    wall = time.perf_counter() - t0
    return {
        "conc": conc,
        "ok": len(lat),
        "errors": dict(errors),
        "rps": len(lat) / wall,
        "p50": pct(lat, 50),
        "p95": pct(lat, 95),
        "p99": pct(lat, 99),
        "batch_sizes": dict(sorted(batch_sizes.items())),
    }


async def amain(args: argparse.Namespace) -> int:
    headers = {"X-API-Key": args.api_key} if args.api_key else {}
    limits = httpx.Limits(max_connections=max(64, 2 * max(args.levels)))
    async with httpx.AsyncClient(base_url=args.url, headers=headers, timeout=600, limits=limits) as c:
        try:
            h = (await c.get("/health", timeout=5)).json()
        except Exception as exc:
            print(f"cannot reach {args.url}: {exc!r}", file=sys.stderr)
            return 2
        print(f"server {h.get('version')} status={h.get('status')} {h.get('gpu', {}).get('name')}")
        for _ in range(args.warmup):  # warm shapes through the real path
            await run_level(c, max(args.levels), max(args.levels), args.mixed)
        print(
            f"\n{'conc':>4} {'ok':>5} {'err':>4} {'req/s':>7} | {'p50':>7} {'p95':>7} {'p99':>7} ms | server batch_size distribution"
        )
        for conc in args.levels:
            r = await run_level(c, conc, args.requests, args.mixed)
            dist = " ".join(f"{k}:{v}" for k, v in r["batch_sizes"].items())
            total = sum(r["batch_sizes"].values()) or 1
            avg = sum(k * v for k, v in r["batch_sizes"].items()) / total
            print(
                f"{conc:>4} {r['ok']:>5} {sum(r['errors'].values()):>4} {r['rps']:>7.1f} | "
                f"{r['p50']:>7.1f} {r['p95']:>7.1f} {r['p99']:>7.1f}    | avg {avg:.2f}  [{dist}]"
                + (f"  errors={r['errors']}" if r["errors"] else "")
            )
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="clef bench", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--url", default=os.environ.get("CLEF_URL", "http://127.0.0.1:8910"))
    ap.add_argument("--api-key", default=os.environ.get("CLEF_API_KEY"))
    ap.add_argument("--concurrency", default="1,2,4,8,16", help="comma-separated levels")
    ap.add_argument("--requests", type=int, default=100, help="requests per level")
    ap.add_argument("--warmup", type=int, default=1, help="warm-up rounds before measuring")
    ap.add_argument("--mixed", action="store_true", help="mixed-length states")
    args = ap.parse_args(argv)
    args.levels = [int(x) for x in args.concurrency.split(",")]
    return asyncio.run(amain(args))


if __name__ == "__main__":
    sys.exit(main())
