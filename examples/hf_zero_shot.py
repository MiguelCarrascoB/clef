"""Zero-shot classification through `huggingface_hub.InferenceClient` (pip install huggingface_hub).

clef serves the Inference API shape at /hf/models/<id>. Reads CLEF_URL and CLEF_API_KEY.
"""

import os

from huggingface_hub import InferenceClient

base = os.environ.get("CLEF_URL", "http://127.0.0.1:8910").rstrip("/")
# Point `model` at the full URL: the client posts to it as-is.
client = InferenceClient(model=f"{base}/hf/models/clef-flash", token=os.environ.get("CLEF_API_KEY"))

text = "Checkout is down, orders blocked"
labels = ["billing", "technical", "account"]

for item in client.zero_shot_classification(text, labels):  # one label, scores sum to 1
    print(f"single : {item.label:10s} {item.score:.3f}")

multi = client.zero_shot_classification(  # independent yes/no per label
    "I was charged twice and the app crashes on login",
    labels,
    multi_label=True,
    hypothesis_template="This message is about {}.",  # becomes each label's description
)
for item in multi:
    print(f"multi  : {item.label:10s} {item.score:.3f}")
