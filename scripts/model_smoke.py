"""Canonical clef-flash example (from the model card) on the local backend. Loads the model (~45 s).

Backend comes from CLEF_DEVICE (auto/cuda/rocm/mps/cpu), dtype from CLEF_DTYPE, weights from CLEF_MODEL_PATH.
"""

import sys
import time
from pathlib import Path

try:
    from clef_server import backend as backend_mod
except ImportError:  # running from a checkout without `pip install -e .`
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    from clef_server import backend as backend_mod

import torch  # noqa: E402

from clef_server.config import Config  # noqa: E402
from clef_server.paths import resolve_model_path  # noqa: E402

cfg = Config()
backend = backend_mod.detect(cfg.device)
dtype, warning = backend.resolve_dtype(cfg.dtype)
if warning:
    print("warning:", warning)
path = str(resolve_model_path(cfg))
print(f"backend: {backend.name} ({backend.device_name()}), dtype {dtype}, model {path}")
sys.path.insert(0, path)
from joint_schema_model import collate_records, encode_record, load_release_model  # noqa: E402

t0 = time.time()
kwargs = {}
qc = backend_mod.quantization_config(backend, cfg.quant, dtype)
if qc is not None:
    kwargs["quantization_config"] = qc
model, processor = load_release_model(path, device=backend.device, dtype=dtype, **kwargs)
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
batch = collate_records([encoded], processor.tokenizer.pad_token_id, backend.device)
with torch.inference_mode():
    model(batch)  # warm-up: first call compiles kernels
    backend.synchronize()
    t0 = time.perf_counter()
    logits = model(batch)[0]
    backend.synchronize()
print(f"warm forward pass: {(time.perf_counter() - t0) * 1000:.1f} ms")

probs = [
    dict(zip(q.option_ids, ql.float().softmax(-1).tolist(), strict=False))
    for q, ql in zip(encoded.questions, logits, strict=False)
]
for q, p in zip(encoded.questions, probs, strict=False):
    print(q.question_id, p)

assert probs[0]["overdue"] > 0.5, "expected overdue to dominate"
assert probs[1]["true"] > 0.5, "expected true to dominate (1250 > 1000)"
mem = backend.memory()
print(f"{mem['kind']} memory allocated: {mem['allocated_gb']} GB of {mem['total_gb']} GB")
print("MODEL RUN PASSED")
