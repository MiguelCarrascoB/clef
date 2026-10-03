"""Canonical clef-flash example (from the model card) on the local GPU. Loads the model (~45 s)."""

import os
import sys
import time

import torch

path = os.environ.get("CLEF_MODEL_PATH", os.path.expanduser("~/models/clef-flash"))
sys.path.insert(0, path)
from joint_schema_model import collate_records, encode_record, load_release_model

t0 = time.time()
model, processor = load_release_model(path, device="cuda")
print(f"model loaded in {time.time() - t0:.1f}s")

record = {
    "state": {"invoice": {"vendor": "Acme", "total": 1250.0, "currency": "USD", "status": "overdue"}},
    "questions": {
        "status": {
            "type": "choice",
            "instructions": "What is the invoice status?",
            "criteria": {"paid": "Invoice is paid.", "overdue": "Invoice is past due.", "draft": "Not sent."},
        },
        "large": {"type": "noul", "instructions": "Is the total above 1000 USD?"},
    },
}

encoded = encode_record(processor.tokenizer, record, processor=processor)
batch = collate_records([encoded], processor.tokenizer.pad_token_id, torch.device("cuda"))
with torch.inference_mode():
    model(batch)  # warm-up: first call compiles kernels
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    logits = model(batch)[0]
    torch.cuda.synchronize()
print(f"warm forward pass: {(time.perf_counter() - t0) * 1000:.1f} ms")

probs = [
    dict(zip(q.option_ids, ql.float().softmax(-1).tolist(), strict=False))
    for q, ql in zip(encoded.questions, logits, strict=False)
]
for q, p in zip(encoded.questions, probs, strict=False):
    print(q.question_id, p)

assert probs[0]["overdue"] > 0.5, "expected overdue to dominate"
assert probs[1]["true"] > 0.5, "expected true to dominate (1250 > 1000)"
print(f"peak VRAM: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")
print("MODEL RUN PASSED")
