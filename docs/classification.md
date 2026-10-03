# Classification API

A short way to ask "which of these labels fits this input?" (or "how high on this scale?"). It sits on the same engine
as `/v1/systemone`: each input becomes exactly one SystemOne record, so the probabilities are identical to the
equivalent SystemOne call (checked live within 1e-3). The normative contract is in
[ARCHITECTURE.md](ARCHITECTURE.md#classification-api-classifypy-classifierspy); the schema is in
[`openapi.json`](../openapi.json) and at `/docs` on a running server.

All routes need an API key when keys are configured (`X-API-Key` or `Authorization: Bearer`), return `usage`, `timing`
and `request_id`, and report errors as `{"detail", "request_id"}`.

Labels are either a list (`["billing", "technical"]`) or an object of label to description
(`{"billing": "Payments or invoices", "technical": "Bugs or outages"}`). Descriptions help the model; use them when
label names are ambiguous. Limits: 2..`CLEF_MAX_LABELS` (64) labels for single-label, 1..64 for multi-label.

## Single-label: `POST /v1/classify`

Picks one label; scores are a distribution over the labels (sum to 1).

```bash
curl -s http://127.0.0.1:8910/v1/classify -H 'Content-Type: application/json' -d '{
  "input": "Checkout is down",
  "labels": {"billing": "Payments or invoices", "technical": "Bugs or outages"},
  "instructions": "Which team should handle this message?"
}'
```

```json
{"model": "clef-flash", "multi_label": false, "label": "technical", "confidence": 0.96,
 "scores": {"billing": 0.04, "technical": 0.96},
 "usage": {"input_tokens": 120, "output_tokens": 0},
 "timing": {"queue_ms": 0.4, "forward_ms": 141.2, "total_ms": 150.3, "batch_size": 1}, "request_id": "..."}
```

`input` can be a string or any JSON value (objects are given to the model as structured data). `images` and `videos`
(data URLs) can be added exactly as on `/v1/systemone`; `instructions` is optional.

```python
from clef_client import ClefClient

c = ClefClient("http://127.0.0.1:8910")
r = c.classify("Checkout is down", ["billing", "technical"])
print(r.label, r.confidence, r.scores)
```

```js
import { ClefClient } from "./clients/js/src/index.js";
const c = new ClefClient({ baseUrl: "http://127.0.0.1:8910" });
const r = await c.classify("Checkout is down", ["billing", "technical"]);
console.log(r.label, r.confidence, r.scores);
```

## Multi-label: `"multi_label": true`

One independent yes/no question per label; scores are P(label applies) and do not sum to 1. `labels` lists every
label with score >= `threshold` (default 0.5, or `CLEF_CLASSIFY_THRESHOLD`), sorted by score; it can be empty.

```bash
curl -s http://127.0.0.1:8910/v1/classify -H 'Content-Type: application/json' -d '{
  "input": "Refund me, and the app also crashes on login",
  "labels": ["billing", "technical", "feature-request"],
  "multi_label": true, "threshold": 0.6
}'
# {"multi_label": true, "labels": ["billing", "technical"], "threshold": 0.6,
#  "scores": {"billing": 0.93, "technical": 0.88, "feature-request": 0.04}, ...}
```

```python
r = c.classify(
    "Refund me, and the app also crashes",
    ["billing", "technical", "feature-request"],
    multi_label=True,
    threshold=0.6,
)
print(r.labels, r.scores)
```

`threshold` is accepted only together with `multi_label: true` and must be within [0, 1].

## Many inputs: `POST /v1/classify/batch`

One shared label set, up to `CLEF_MAX_BATCH` inputs (64), no media. Runs through the micro-batcher and returns results
in input order. A bad input is reported by index (`inputs[3]: ...`).

```bash
curl -s http://127.0.0.1:8910/v1/classify/batch -H 'Content-Type: application/json' \
  -d '{"inputs": ["Checkout is down", "Where is my invoice?"], "labels": ["billing", "technical"]}'
# {"batch_ms": 160.2, "results": [{"label": "technical", ...}, {"label": "billing", ...}], "request_id": "..."}
```

```python
results = c.classify_many(["Checkout is down", "Where is my invoice?"], ["billing", "technical"])
[r.label for r in results]
```

## Ordinal scale: `POST /v1/score`

For an ordered scale. `score` is the expected level index, `level` the most likely level, `distribution` the
probabilities.

```bash
curl -s http://127.0.0.1:8910/v1/score -H 'Content-Type: application/json' \
  -d '{"input": "The server has been down for two hours", "levels": ["low", "medium", "high"]}'
# {"score": 1.78, "level": "high", "level_index": 2, "confidence": 0.86,
#  "distribution": {"low": 0.07, "medium": 0.07, "high": 0.86}, ...}
```

```python
s = c.score("The server has been down for two hours", ["low", "medium", "high"])
print(s.score, s.level, s.distribution)
```

## Saved classifiers

Store the labels and instructions once; callers send only the input. Names: lowercase letters, digits, `-` and `_`,
at most 64 characters, starting with a letter or digit. Definitions are JSON files under the state directory
(`<CLEF_STATE_DIR>/classifiers/`), at most `CLEF_MAX_CLASSIFIERS` (1000).

```bash
# create or overwrite
curl -s -X PUT http://127.0.0.1:8910/v1/classifiers/support-triage -H 'Content-Type: application/json' -d '{
  "kind": "classify",
  "labels": {"billing": "Payments or invoices", "technical": "Bugs or outages"},
  "instructions": "Which team should handle this message?",
  "description": "Inbound support mail"
}'

curl -s http://127.0.0.1:8910/v1/classifiers                    # list
curl -s http://127.0.0.1:8910/v1/classifiers/support-triage     # show
curl -s -X POST http://127.0.0.1:8910/v1/classifiers/support-triage \
  -H 'Content-Type: application/json' -d '{"input": "My invoice is wrong"}'
curl -s -X POST http://127.0.0.1:8910/v1/classifiers/support-triage/batch \
  -H 'Content-Type: application/json' -d '{"inputs": ["...", "..."]}'
curl -s -X DELETE http://127.0.0.1:8910/v1/classifiers/support-triage
```

`kind` is `classify` (with `labels`, optional `multi_label` / `threshold`) or `score` (with `levels`). The response of
a call carries `"classifier": "<name>"`.

```python
c.classifier("support-triage").classify("My invoice is wrong")
```

```js
await c.classifier("support-triage").classify("My invoice is wrong");
```

## Tips

- Write label names the way a person would; add descriptions when two labels could overlap.
- Single-label forces a choice among the labels. If "none of these" is possible, include a label for it, or use
  multi-label and check for an empty `labels` list.
- Scores are calibrated probabilities but not guarantees; test on a labelled sample from your own data before relying
  on a threshold.
- Latency is one forward pass (~140 ms on the verified ROCm setup, [hardware-results.md](hardware-results.md)). Send
  many inputs through `/v1/classify/batch` or concurrent requests to let the server micro-batch.

## Errors

| Status | Meaning |
| --- | --- |
| 400 | invalid body (duplicate or empty labels, bad threshold, bad name, ...) |
| 401 | missing or wrong API key |
| 404 | unknown saved classifier |
| 409 | too many saved classifiers |
| 413 | input too large for `CLEF_MAX_TOKENS` or the body limit |
| 429 | rate limited; honour `Retry-After` |
| 503 | model still loading or device out of memory; the clients retry with backoff (they also honour `Retry-After` on 429) |
