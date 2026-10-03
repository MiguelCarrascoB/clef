"""Canonical clef-flash example (from the model card), running on AMD RX 7900 XTX via ROCm."""
import sys
import time

import torch
from huggingface_hub import snapshot_download  # noqa: F401 (imported to mirror the card)

path = "~/models/clef-flash"
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
t0 = time.time()
with torch.inference_mode():
    logits = model(batch)[0]
elapsed_ms = (time.time() - t0) * 1000
print(f"forward pass: {elapsed_ms:.1f} ms")

for question, question_logits in zip(encoded.questions, logits):
    probabilities = question_logits.float().softmax(-1).tolist()
    print(question.question_id, dict(zip(question.option_ids, probabilities)))

# quick sanity assertions (logits is already this record's per-question tensor list)
probs_status = dict(
    zip(encoded.questions[0].option_ids, logits[0].float().softmax(-1).tolist())
)
assert probs_status["overdue"] > 0.5, "expected overdue to dominate"
probs_large = dict(
    zip(encoded.questions[1].option_ids, logits[1].float().softmax(-1).tolist())
)
assert probs_large["true"] > 0.5, "expected true to dominate (1250 > 1000)"
vr = round(torch.cuda.max_memory_allocated() / 1e9, 2)
print(f"peak VRAM: {vr} GB")
print("MODEL RUN PASSED")
