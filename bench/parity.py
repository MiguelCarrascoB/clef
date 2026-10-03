"""Probability parity of a dtype / quantization setting against bf16 on a fixed eval set (~50 records).

    python bench/parity.py --dtype float16                 # candidate fp16 vs bf16 on the detected backend
    python bench/parity.py --quant nf4 --device cuda       # bitsandbytes nf4 vs bf16 (NVIDIA only)
    python bench/parity.py --dtype float32 --n 20

Loads the model twice (reference first, then the candidate, freeing the GPU in between). Reports the max and mean
absolute probability difference over every answer option, how many top-1 choices flipped, and latency for both.
Results go to bench/out/parity_<backend>_<dtype>_<quant>.json (copy the table into docs/hardware-results.md).
Refuses to run while a clef server holds the GPU.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import statistics
import sys
import time
import urllib.request
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_SRC = _HERE.parent / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:  # run from a checkout without `pip install -e .`
    sys.path.insert(0, str(_SRC))

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


def build_eval_set(n: int = 50) -> list[dict]:
    """Deterministic mix of choice / score / noul questions over varied states and lengths."""
    records = []
    for i in range(n):
        text = " ".join(_TEXTS[(i + k) % len(_TEXTS)] for k in range(1 + i % 4))
        records.append(
            {
                "state": {"ticket": {"id": f"p{i}", "text": text, "amount": 100 + 37 * i}},
                "questions": {
                    "department": {
                        "type": "choice",
                        "instructions": "Which team owns this?",
                        "criteria": {
                            "billing": "Payments or invoices",
                            "technical": "Bugs or outages",
                            "account": "Account or profile changes",
                        },
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
    return records


def compare(ref: list[dict], cand: list[dict]) -> dict:
    """ref / cand: per record {"question.option": prob}. Max / mean abs diff and top-1 flips per question."""
    diffs, flips, total_q = [], 0, 0
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
                flips += 1
    return {
        "max_abs_diff": max(diffs) if diffs else 0.0,
        "mean_abs_diff": statistics.fmean(diffs) if diffs else 0.0,
        "top1_flips": flips,
        "questions": total_q,
    }


def server_running(port: int) -> bool:
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/livez", timeout=2):
            return True
    except Exception:
        return False


def run_setting(
    backend_req: str, dtype: str, quant: str, model_path: str, records: list[dict], warmup: int = 3
):
    """Load the model with one setting, run every record singly. Returns (probs, latencies_ms, info)."""
    import torch

    from clef_server import backend as backend_mod

    sys.path.insert(0, model_path)
    from joint_schema_model import collate_records, encode_record, load_release_model

    backend = backend_mod.detect(backend_req)
    torch_dtype, warning = backend.resolve_dtype(dtype)
    if warning:
        print("warning:", warning)
    kwargs = {}
    qc = backend_mod.quantization_config(backend, quant, torch_dtype)
    if qc is not None:
        kwargs["quantization_config"] = qc
    model, processor = load_release_model(model_path, device=backend.device, dtype=torch_dtype, **kwargs)
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


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument(
        "--model-path", default=os.environ.get("CLEF_MODEL_PATH", str(Path.home() / "models" / "clef-flash"))
    )
    ap.add_argument("--device", default=os.environ.get("CLEF_DEVICE", "auto"))
    ap.add_argument("--dtype", default=os.environ.get("CLEF_DTYPE", "float16"), help="candidate dtype")
    ap.add_argument("--quant", default=os.environ.get("CLEF_QUANT", "none"), help="candidate quantization")
    ap.add_argument("--n", type=int, default=50, help="eval records")
    ap.add_argument("--port", type=int, default=int(os.environ.get("CLEF_PORT", "8910")))
    ap.add_argument("--force", action="store_true", help="run even if a server answers on --port")
    ap.add_argument("--out", default=str(_HERE / "out"))
    args = ap.parse_args()

    if not args.force and server_running(args.port):
        print(
            f"REFUSING: a clef server on :{args.port} holds the GPU. Stop it or pass --force.",
            file=sys.stderr,
        )
        return 2

    records = build_eval_set(args.n)
    print(f"eval set: {len(records)} records; reference = bfloat16 / no quant")
    ref, ref_lat, ref_info = run_setting(args.device, "bfloat16", "none", args.model_path, records)
    if ref_info["dtype"] != "bfloat16":
        print(f"note: this backend cannot run bfloat16; the reference is {ref_info['dtype']}")
    cand, cand_lat, cand_info = run_setting(args.device, args.dtype, args.quant, args.model_path, records)
    stats = compare(ref, cand)
    result = {
        **stats,
        "backend": cand_info["backend"],
        "device": cand_info["device"],
        "reference": {"dtype": ref_info["dtype"], "latency_p50_ms": statistics.median(ref_lat)},
        "candidate": {
            "dtype": cand_info["dtype"],
            "quant": args.quant,
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
