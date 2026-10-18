"""Probability parity of a dtype / quantization / offload setting against bf16 on a fixed eval set (200 records).

    python bench/parity.py --dtype float16                 # candidate fp16 vs bf16 on the detected backend
    python bench/parity.py --quant int8 --dtype bfloat16   # weight-only int8 vs bf16 (see docs/memory.md)
    python bench/parity.py --offload cpu --max-device-gb 14 --dtype bfloat16
    python bench/parity.py --quant nf4 --device cuda       # bitsandbytes nf4 vs bf16 (NVIDIA)
    python bench/parity.py --dtype float32 --n 20

Loads the model twice (reference first, then the candidate, freeing the GPU in between). Reports the max and mean
absolute probability difference over every answer option, how many top-1 choices flipped, and latency for both.
Results go to bench/out/parity_<backend>_<dtype>_<quant>.json (copy the table into docs/hardware-results.md).
Refuses to run while a clef server holds the GPU.

Engine parity (code changes, not settings): run the eval set through the production `Engine.decide` (one record
per call, then max_microbatch per call) with the current CLEF_* settings and dump the answers, optionally with
another source tree first on sys.path (e.g. pre-refactor main), then compare two dumps:

    python bench/parity.py --engine-dump out/ref.json --src /tmp/ref/src   # the reference tree's engine
    python bench/parity.py --engine-dump out/new.json                     # this tree's engine
    python bench/parity.py --compare out/ref.json out/new.json            # exit 1 on a flip or |dp| > --tol
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import random
import statistics
import sys
import time
import urllib.request
from pathlib import Path

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))
from _model import default_model_path  # noqa: E402

_SRC = _HERE.parent / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:  # run from a checkout without `pip install -e .`
    sys.path.insert(0, str(_SRC))
# --src DIR: import clef_server from another tree (an engine parity reference). Must win before any import.
if "--src" in sys.argv[:-1]:
    sys.path.insert(0, str(Path(sys.argv[sys.argv.index("--src") + 1]).expanduser().resolve()))

_TEXTS = [
    "Checkout errors, orders blocked since 03:00 UTC.",
    "Customer asks to change the billing address on invoice 4411.",
    "The mobile app crashes when opening the settings screen on Android 14.",
    "Please cancel my subscription and refund the last charge.",
    "Great service, the new dashboard is much faster than before.",
    "Servers in eu-west-1 report elevated latency, p99 above 4 seconds.",
    "I was charged twice for the same order and nobody answers my emails.",
    "Feature request: export reports as CSV with custom column order.",
    "Planned maintenance tonight 22:00-23:00 UTC, no customer impact expected.",
    "The invoice total is 1250 USD and it is 45 days past due.",
]


_DEPARTMENTS = {
    "billing": "Payments or invoices",
    "technical": "Bugs or outages",
    "account": "Account or profile changes",
    "sales": "New purchases, upgrades or quotes",
    "abuse": "Spam, fraud or policy violations",
    "other": "Anything else",
}
_SCORE_SCALES = [
    ["Can wait", "This week", "Today"],
    ["Negative", "Neutral", "Positive"],
    ["None", "Low", "Medium", "High", "Critical"],
]


def _image_for(i: int):
    """Small synthetic RGB image (PIL), deterministic per index."""
    from PIL import Image, ImageDraw

    img = Image.new("RGB", (224 + 32 * (i % 3), 192), ((37 * i) % 256, (91 * i) % 256, (153 * i) % 256))
    draw = ImageDraw.Draw(img)
    draw.rectangle((20, 20, 120, 100), outline=(255, 255, 255), width=4)
    draw.text((30, 120), f"ticket {i}", fill=(255, 255, 255))
    return img


def build_eval_set(n: int = 200, images: int = 0) -> list[dict]:
    """Deterministic, diverse eval set: choice (2-6 options) / score (3-5 levels) / noul questions, 1-3 per
    record, states from ~15 to ~1500 tokens. `images` > 0 turns every (n // images)-th record into an image
    record (needs PIL; exercises the vision tower)."""
    rng = random.Random(1234)
    every = max(1, n // images) if images else 0
    records = []
    for i in range(n):
        reps = rng.choice([1, 1, 1, 2, 2, 3, 4, 8, 20, 40]) if i % 7 else rng.choice([60, 90])
        picks = [_TEXTS[(i + k * 3) % len(_TEXTS)] for k in range(reps)]
        text = " ".join(picks)
        state = {"ticket": {"id": f"p{i}", "text": text, "amount": 100 + 37 * i}}
        if i % 5 == 0:
            state["customer"] = {"tier": ["free", "pro", "enterprise"][i % 3], "open_tickets": i % 9}
        names = list(_DEPARTMENTS)
        n_opts = 2 + i % 5
        picked = sorted(rng.sample(names, n_opts))
        pool = {
            "department": {
                "type": "choice",
                "instructions": "Which team owns this?",
                "criteria": {k: _DEPARTMENTS[k] for k in picked},
            },
            "urgency": {
                "type": "score",
                "instructions": "How urgent is this?",
                "criteria": _SCORE_SCALES[i % 3],
            },
            "outage": {"type": "noul", "instructions": "Is a service down?"},
            "refund": {"type": "noul", "instructions": "Does the customer want money back?"},
        }
        keys = ["department", "urgency", "outage", "refund"]
        chosen = [keys[(i + j) % 4] for j in range(1 + i % 3)]
        record: dict = {
            "id": f"p{i}",
            "state": state,
            "questions": {k: pool[k] for k in dict.fromkeys(chosen)},
        }
        if every and i % every == 0:
            record["images"] = [_image_for(i)]
        records.append(record)
    return records


TIE_MARGIN = 1e-3  # a top-1 flip where the reference top-2 gap is below this is a near-tie, not a defect


def compare(ref: list[dict], cand: list[dict]) -> dict:
    """ref / cand: per record {"question.option": prob}. Max / mean abs diff and top-1 flips per question.
    A flip whose reference top-2 margin is below TIE_MARGIN counts in `tie_flips`, not in `top1_flips`."""
    diffs, flips, ties, total_q = [], 0, 0, 0
    for r, c in zip(ref, cand, strict=True):
        if r.keys() != c.keys():
            raise ValueError("reference and candidate answer different options")
        diffs.extend(abs(r[k] - c[k]) for k in r)
        by_q: dict[str, list[str]] = {}
        for k in r:
            by_q.setdefault(k.split(".", 1)[0], []).append(k)
        for keys in by_q.values():
            total_q += 1
            if max(keys, key=r.__getitem__) != max(keys, key=c.__getitem__):
                top = sorted((r[k] for k in keys), reverse=True)
                if len(top) > 1 and top[0] - top[1] < TIE_MARGIN:
                    ties += 1
                else:
                    flips += 1
    return {
        "max_abs_diff": max(diffs) if diffs else 0.0,
        "mean_abs_diff": statistics.fmean(diffs) if diffs else 0.0,
        "top1_flips": flips,
        "tie_flips": ties,
        "questions": total_q,
    }


def noise_dp(a_path: str, b_path: str) -> float:
    """Largest |dp| between two dumps of the same configuration (the run-to-run noise floor)."""
    a, b = json.loads(Path(a_path).read_text()), json.loads(Path(b_path).read_text())
    return max(compare(a[m], b[m])["max_abs_diff"] for m in ("singles", "batched"))


def effective_tol(floor: float, noise: float) -> float:
    """Parity threshold: max(floor, 2 x noise)."""
    return max(floor, 2 * noise)


def server_running(port: int) -> bool:
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/livez", timeout=2):
            return True
    except Exception:
        return False


def run_setting(
    backend_req: str,
    dtype: str,
    quant: str,
    model_path: str,
    records: list[dict],
    warmup: int = 3,
    offload: str = "none",
    max_device_gb: float = 0.0,
):
    """Load the model with one setting, run every record singly. Returns (probs, latencies_ms, info)."""
    import torch

    from bench import load_model

    sys.path.insert(0, model_path)
    from joint_schema_model import collate_records, encode_record

    backend, torch_dtype, model, processor = load_model(
        model_path, backend_req, dtype, quant, offload, max_device_gb
    )
    tok, pad = processor.tokenizer, processor.tokenizer.pad_token_id
    encoded = [encode_record(tok, r, processor=processor) for r in records]
    probs, lat = [], []
    with torch.inference_mode():
        for _ in range(warmup):
            model(collate_records(encoded[:1], pad, backend.device))
        backend.synchronize()
        for enc in encoded:
            batch = collate_records([enc], pad, backend.device)
            t = time.perf_counter()
            logits = model(batch)[0]
            backend.synchronize()
            lat.append((time.perf_counter() - t) * 1000)
            flat = {}
            for q, ql in zip(enc.questions, logits, strict=True):
                for opt, p in zip(q.option_ids, ql.float().softmax(-1).tolist(), strict=True):
                    flat[f"{q.question_id}.{opt}"] = p
            probs.append(flat)
    info = {
        "backend": backend.name,
        "device": backend.device_name(),
        "dtype": str(torch_dtype).replace("torch.", ""),
    }
    del model, processor
    gc.collect()
    backend.empty_cache()
    return probs, lat, info


def flat_answers(res: dict) -> dict:
    """SystemOne answers -> {"question.option": prob} (noul: true / false). Rounded as the API rounds them."""
    flat = {}
    for qid, ans in res["answers"].items():
        if "probabilities" in ans:
            for opt, p in ans["probabilities"].items():
                flat[f"{qid}.{opt}"] = float(p)
        elif "noul" in ans:
            flat[f"{qid}.true"] = float(ans["noul"])
            flat[f"{qid}.false"] = 1.0 - float(ans["noul"])
    return flat


class _NullStats:
    def record_forward(self, *a, **k):
        pass

    def set_queue_depth(self, *a, **k):
        pass


def engine_dump(out_path: str, records: list[dict], timeout: float = 1800.0) -> int:
    """The production path: Engine(Config()) with the CLEF_* environment (warmup off), decide() per record,
    then in max_microbatch chunks. Works with the pre- and post-refactor engines (same public API)."""
    import asyncio
    import dataclasses

    import clef_server
    from clef_server.config import Config
    from clef_server.engine import Engine

    print(f"engine from {Path(clef_server.__file__).parent}", flush=True)
    cfg = dataclasses.replace(Config(), warmup=False)
    # D3: the result cache would answer the batched pass from the singles pass (no forward at all)
    if any(f.name == "cache_size" for f in dataclasses.fields(cfg)):
        cfg = dataclasses.replace(cfg, cache_size=0)
    eng = Engine(cfg, _NullStats())
    eng.start()
    t0 = time.perf_counter()
    while eng.status == "loading" and time.perf_counter() - t0 < timeout:
        time.sleep(0.5)
    if eng.status not in ("ready", "warming"):
        print(f"engine {eng.status}: {eng.error}", file=sys.stderr)
        eng.shutdown()
        return 1
    info = eng.info()
    print(f"loaded on {info['backend']} {info['dtype']} in {eng.load_seconds}s", flush=True)

    async def go():
        singles = [(await eng.decide([dict(r)]))[0] for r in records]
        step = max(1, int(cfg.max_microbatch))
        batched = []
        for i in range(0, len(records), step):
            batched.extend(await eng.decide([dict(r) for r in records[i : i + step]]))
        return singles, batched

    try:
        singles, batched = asyncio.run(go())
    finally:
        eng.shutdown()
    model_ref = eng.model_ref() if hasattr(eng, "model_ref") else None
    result = {
        "src": str(Path(clef_server.__file__).parent),
        "model": model_ref,
        "backend": info["backend"],
        "dtype": info["dtype"],
        "quant": cfg.quant,
        "load_seconds": eng.load_seconds,
        "singles": [flat_answers(r) for r in singles],
        "batched": [flat_answers(r) for r in batched],
        "lat": [r["timing"]["forward_ms"] for r in singles],
        "result_models": sorted({r.get("model") for r in singles + batched}),
    }
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text(json.dumps(result))
    print(f"wrote {out_path} ({len(records)} records, model {model_ref or 'pre-refactor'})")
    return 0


def compare_dumps(a_path: str, b_path: str, tol: float) -> int:
    a, b = json.loads(Path(a_path).read_text()), json.loads(Path(b_path).read_text())
    ok = True
    for mode in ("singles", "batched"):
        st = compare(a[mode], b[mode])
        good = st["top1_flips"] == 0 and st["max_abs_diff"] <= tol
        ok = ok and good
        print(
            f"{mode:<8} max |dp| = {st['max_abs_diff']:.5f}  mean |dp| = {st['mean_abs_diff']:.6f}  "
            f"top-1 flips {st['top1_flips']}/{st['questions']}  {'OK' if good else 'FAIL'}"
            f"  (near-tie flips, ref margin < {TIE_MARGIN:g}: {st['tie_flips']})"
        )
    for name, d in (("A", a), ("B", b)):
        lat = statistics.median(d["lat"]) if d.get("lat") else 0.0
        print(f"{name}: {d['src']} model={d.get('model')} {d['backend']}/{d['dtype']} p50 {lat:.1f} ms")
    print("PARITY OK" if ok else f"PARITY FAIL (tolerance {tol})")
    return 0 if ok else 1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--engine-dump", metavar="OUT.json", help="engine parity: dump Engine.decide answers")
    ap.add_argument(
        "--src", metavar="DIR", help="with --engine-dump: import clef_server from DIR (a src/ dir)"
    )
    ap.add_argument(
        "--compare", nargs=2, metavar=("A.json", "B.json"), help="compare two --engine-dump files"
    )
    ap.add_argument(
        "--tol", type=float, default=2e-3, help="--compare: max |dp| allowed (answers are rounded)"
    )
    ap.add_argument(
        "--noise",
        nargs=2,
        metavar=("A.json", "B.json"),
        help="print the max |dp| between two dumps of one config (noise) and max(--tol, 2 x noise)",
    )
    ap.add_argument("--model-path", default=default_model_path())
    ap.add_argument("--device", default=os.environ.get("CLEF_DEVICE", "auto"))
    ap.add_argument("--dtype", default=os.environ.get("CLEF_DTYPE", "float16"), help="candidate dtype")
    ap.add_argument("--quant", default=os.environ.get("CLEF_QUANT", "none"), help="candidate quantization")
    ap.add_argument("--offload", default=os.environ.get("CLEF_OFFLOAD", "none"), help="candidate offload")
    ap.add_argument("--max-device-gb", type=float, default=0.0, help="candidate device memory cap (GB)")
    ap.add_argument("--n", type=int, default=200, help="eval records")
    ap.add_argument("--images", type=int, default=10, help="of which image records (needs PIL)")
    ap.add_argument(
        "--reference", help="reuse bf16 probabilities from this JSON instead of loading the reference"
    )
    ap.add_argument("--save-reference", help="write the bf16 reference probabilities to this JSON")
    ap.add_argument("--port", type=int, default=int(os.environ.get("CLEF_PORT", "8910")))
    ap.add_argument("--force", action="store_true", help="run even if a server answers on --port")
    ap.add_argument("--out", default=str(_HERE / "out"))
    args = ap.parse_args()

    if args.noise:
        n = noise_dp(*args.noise)
        print(f"noise {n:.6f} tol {effective_tol(args.tol, n):.6f}")
        return 0
    if args.compare:
        return compare_dumps(args.compare[0], args.compare[1], args.tol)
    if not args.force and server_running(args.port):
        print(
            f"REFUSING: a clef server on :{args.port} holds the GPU. Stop it or pass --force.",
            file=sys.stderr,
        )
        return 2

    records = build_eval_set(args.n, images=args.images)
    if args.engine_dump:
        return engine_dump(args.engine_dump, records)
    print(f"eval set: {len(records)} records; reference = bfloat16 / no quant")
    if args.reference:
        saved = json.loads(Path(args.reference).read_text())
        ref, ref_lat, ref_info = (
            saved["probs"],
            saved.get("lat") or [0.0],
            {"dtype": saved.get("dtype", "bfloat16")},
        )
    else:
        ref, ref_lat, ref_info = run_setting(args.device, "bfloat16", "none", args.model_path, records)
        if ref_info["dtype"] != "bfloat16":
            print(f"note: this backend cannot run bfloat16; the reference is {ref_info['dtype']}")
        if args.save_reference:
            Path(args.save_reference).parent.mkdir(parents=True, exist_ok=True)
            Path(args.save_reference).write_text(
                json.dumps({"probs": ref, "lat": ref_lat, "dtype": ref_info["dtype"]})
            )
    cand, cand_lat, cand_info = run_setting(
        args.device,
        args.dtype,
        args.quant,
        args.model_path,
        records,
        offload=args.offload,
        max_device_gb=args.max_device_gb,
    )
    stats = compare(ref, cand)
    result = {
        **stats,
        "backend": cand_info["backend"],
        "device": cand_info["device"],
        "reference": {"dtype": ref_info["dtype"], "latency_p50_ms": statistics.median(ref_lat)},
        "candidate": {
            "dtype": cand_info["dtype"],
            "quant": args.quant,
            "offload": args.offload,
            "latency_p50_ms": statistics.median(cand_lat),
        },
        "records": len(records),
    }
    print(
        f"max |dp| = {stats['max_abs_diff']:.5f}   mean |dp| = {stats['mean_abs_diff']:.6f}   "
        f"top-1 flips {stats['top1_flips']}/{stats['questions']}\n"
        f"latency p50: reference {result['reference']['latency_p50_ms']:.1f} ms, "
        f"candidate {result['candidate']['latency_p50_ms']:.1f} ms"
    )
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"parity_{cand_info['backend']}_{cand_info['dtype']}_{args.quant}.json"
    path.write_text(json.dumps(result, indent=2))
    print("wrote", path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
