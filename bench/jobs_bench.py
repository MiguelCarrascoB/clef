"""Job throughput driver for bench/sort_check.sh: submit ONE classify job of mixed-length inputs to a running
server, wait for it, and append a JSON line with items/s (from the job's own duration_s).

    python bench/jobs_bench.py --tag base --items 2000 --repeat 3 --out bench/out/sort/runs.jsonl

Stdlib only. The inputs depend on --seed and the repeat number, not on the variant, so every variant sees the
same records; start the server with CLEF_CACHE_SIZE=0 or repeats would be answered from the result cache.
Lengths are long-tailed (log-normal, median ~--median-words words), the shape of real support or review text.
With --interactive S, one /v1/classify request is sent every S seconds while the job runs and its latency is
recorded (p50 / p95): the sort must not make interactive traffic slower.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import statistics
import sys
import threading
import time
import urllib.error
import urllib.request

LABELS = ["billing", "technical", "account", "shipping", "other"]
_TEXT = (
    "please order account payment refund invoice error login password delivery package late broken item "
    "support agent help issue update address cancel subscription charge card website app crash slow thanks "
    "after before again never always manager reply email phone number problem still working yesterday"
)
WORDS = _TEXT.split()


def call(url: str, path: str, body: dict | None = None, timeout: float = 120.0) -> dict:
    req = urllib.request.Request(url.rstrip("/") + path, method="POST" if body is not None else "GET")
    key = os.environ.get("CLEF_API_KEY")
    if key:
        req.add_header("Authorization", f"Bearer {key}")
    data = None
    if body is not None:
        req.add_header("Content-Type", "application/json")
        data = json.dumps(body).encode()
    with urllib.request.urlopen(req, data=data, timeout=timeout) as resp:
        return json.loads(resp.read() or b"{}")


def make_inputs(n: int, seed: int, median_words: int, max_words: int) -> list[str]:
    rng = random.Random(seed)
    out = []
    for i in range(n):
        k = int(min(max_words, max(4, rng.lognormvariate(math.log(median_words), 0.9))))
        out.append(f"#{seed}-{i} " + " ".join(rng.choice(WORDS) for _ in range(k)))
    return out


def pct(values: list[float], q: float) -> float:
    s = sorted(values)
    return s[min(len(s) - 1, int(q / 100 * len(s)))] if s else 0.0


def ndjson_labels(url: str, job_id: str) -> list[str | None]:
    req = urllib.request.Request(f"{url.rstrip('/')}/v1/jobs/{job_id}/results?format=ndjson")
    key = os.environ.get("CLEF_API_KEY")
    if key:
        req.add_header("Authorization", f"Bearer {key}")
    with urllib.request.urlopen(req, timeout=300) as resp:
        rows = [json.loads(line) for line in resp.read().decode().splitlines() if line]
    return [r.get("label") for r in sorted(rows, key=lambda r: r["index"])]


def run_once(args: argparse.Namespace, repeat: int) -> dict:
    inputs = make_inputs(args.items, args.seed + repeat, args.median_words, args.max_words)
    lat: list[float] = []
    stop = threading.Event()

    def interactive() -> None:
        while not stop.wait(args.interactive):
            t0 = time.perf_counter()
            try:
                call(args.url, "/v1/classify", {"input": "my parcel never arrived", "labels": LABELS})
                lat.append((time.perf_counter() - t0) * 1000)
            except (urllib.error.URLError, TimeoutError, OSError):
                pass

    t_wall = time.perf_counter()
    job = call(args.url, "/v1/jobs", {"kind": "classify", "payload": {"inputs": inputs, "labels": LABELS}})
    if args.interactive:
        threading.Thread(target=interactive, daemon=True).start()
    while job["status"] not in ("succeeded", "failed", "cancelled"):
        time.sleep(args.poll)
        job = call(args.url, f"/v1/jobs/{job['id']}")
    stop.set()
    wall = time.perf_counter() - t_wall
    labels = ndjson_labels(args.url, job["id"]) if job["status"] == "succeeded" else []
    dur = job.get("duration_s") or wall
    return {
        "tag": args.tag,
        "repeat": repeat,
        "status": job["status"],
        "items": args.items,
        "duration_s": round(dur, 3),
        "wall_s": round(wall, 3),
        "items_per_s": round(args.items / dur, 3) if dur else 0.0,
        "failed": job.get("progress", {}).get("failed"),
        "labels_sha": hashlib.sha256(json.dumps(labels).encode()).hexdigest()[:16],
        "labels": labels,
        "interactive_n": len(lat),
        "interactive_p50_ms": round(statistics.median(lat), 1) if lat else None,
        "interactive_p95_ms": round(pct(lat, 95), 1) if lat else None,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--url", default=f"http://127.0.0.1:{os.environ.get('CLEF_PORT', '8910')}")
    ap.add_argument("--tag", required=True, help="variant name (base, sort, ahead, both, ...)")
    ap.add_argument("--items", type=int, default=2000)
    ap.add_argument("--repeat", type=int, default=3)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--median-words", type=int, default=60)
    ap.add_argument("--max-words", type=int, default=700)
    ap.add_argument(
        "--interactive", type=float, default=0.0, help="seconds between interactive probes (0=off)"
    )
    ap.add_argument("--poll", type=float, default=0.5)
    ap.add_argument("--warm", type=int, default=1, help="untimed warm-up jobs (kernels, tuned shapes)")
    ap.add_argument("--out", required=True, help="JSON lines file to append to")
    args = ap.parse_args()
    for w in range(args.warm):
        warm = argparse.Namespace(**{**vars(args), "items": min(args.items, 200), "interactive": 0.0})
        run_once(warm, 100 + w)
    rc = 0
    for r in range(args.repeat):
        rec = run_once(args, r)
        with open(args.out, "a") as fh:
            fh.write(json.dumps(rec) + "\n")
        print(
            f"{args.tag} run {r}: {rec['status']} {rec['items']} items in {rec['duration_s']} s = "
            f"{rec['items_per_s']} items/s, interactive p95 {rec['interactive_p95_ms']} ms"
        )
        rc = rc or (0 if rec["status"] == "succeeded" else 1)
    return rc


if __name__ == "__main__":
    sys.exit(main())
