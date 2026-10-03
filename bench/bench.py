"""Warm throughput benchmark for clef-flash on ROCm: median/p95 latency like the Decision Index table."""
import sys
import time

import torch

path = "~/models/clef-flash"
sys.path.insert(0, path)
from joint_schema_model import collate_records, encode_record, load_release_model

model, processor = load_release_model(path, device="cuda")

records = []
for i in range(8):
    records.append({
        "state": {
            "ticket": {"id": f"t{i}", "text": "Checkout errors, orders blocked since 03:00 UTC.",
                       "customers_affected": 1200 + i, "service": "orders-api"}},
        "questions": {
            "department": {"type": "choice", "instructions": "Which team owns this?",
                           "criteria": {"billing": "Payments or invoices", "technical": "Bugs or outages"}},
            "urgency": {"type": "score", "instructions": "How urgent?",
                        "criteria": ["Can wait", "This week", "Today"]},
            "outage": {"type": "noul", "instructions": "Is a service down?"},
        },
    })

encoded_records = [encode_record(processor.tokenizer, r, processor=processor) for r in records]
batch = collate_records(encoded_records, processor.tokenizer.pad_token_id, torch.device("cuda"))

# warmup + capture first pass
with torch.inference_mode():
    _ = model(batch)
torch.cuda.synchronize()
times = []
with torch.inference_mode():
    for run in range(20):
        t0 = time.perf_counter()
        _ = model(collate_records(encoded_records[:1], processor.tokenizer.pad_token_id, torch.device("cuda")))
        torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1000)
times_sorted = sorted(times)
n = len(times_sorted)
median = times_sorted[n // 2]
p95 = times_sorted[int(n * 0.95)]
print(f"SINGLE-RECORD 20 runs: median {median:.1f} ms | p95 {p95:.1f} ms")

btimes = []
with torch.inference_mode():
    for run in range(10):
        t0 = time.perf_counter()
        _ = model(batch)
        torch.cuda.synchronize()
        btimes.append((time.perf_counter() - t0) * 1000)
btimes.sort()
print(f"BATCH-8 (10 runs): median {btimes[5]:.1f} ms -> {8000 / btimes[5]:.0f} decisions/s bucket")
print(f"peak VRAM: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")
