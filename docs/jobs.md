# Async jobs and webhooks

`/v1/classify/batch` holds one HTTP connection open until every record is done. That is fine for 64 rows
and wrong for a 50 000-row CSV (~150 ms per record means about two hours). A **job** is the same work
with the connection closed: submit it, get an id back immediately, then poll (or receive a webhook) and
page or stream the results when it is done.

```mermaid
stateDiagram-v2
    [*] --> queued: POST /v1/jobs (202)
    queued --> running: runner picks it up (FIFO)
    running --> succeeded: all items processed
    running --> failed: systemic error, every item failed, or the circuit breaker tripped
    running --> cancelled: POST .../cancel (between chunks)
    queued --> cancelled: POST .../cancel
    running --> interrupted: server stopped / crashed
    interrupted --> running: server restarted, resumes after the last stored row
    interrupted --> failed: interrupted more than CLEF_JOB_MAX_RESUMES times
    succeeded --> [*]: DELETE or retention TTL
    failed --> [*]
    cancelled --> [*]
```

Interactive requests (`/v1/classify`, `/v1/systemone`, ...) stay responsive while a job runs: the runner
processes one job at a time and keeps **one chunk** (`CLEF_MAX_MICROBATCH` records, 8 by default) in flight
at the engine. A request arriving mid-job waits for the chunk being computed (at most one micro-batch) and is
then served ahead of the job's next chunk. Cancellation also takes effect between chunks.

## Endpoints

All under `/v1`, same API key rules as the rest. Errors are `{detail, request_id}`.

| Method and path | Purpose |
| --- | --- |
| `POST /v1/jobs` | Submit. `202` with the job and a `Location` header. Rate limited like inference. |
| `GET /v1/jobs?status=&kind=&limit=&offset=` | List, newest first (`limit` 1-200, default 50). |
| `GET /v1/jobs/{id}` | Status, progress, ETA, timings, error, result summary, webhook delivery status. |
| `GET /v1/jobs/{id}/results?offset=0&limit=100` | One JSON page (`limit` up to 1000). Follow `next_offset` until it is `null`. |
| `GET /v1/jobs/{id}/results?format=ndjson` or `csv` | Stream every row (`offset` / `limit` optional). Works while the job runs, but then it is **partial**: it holds the rows stored when the request arrived (the `X-Clef-Job-Status` header says which state that was). |
| `POST /v1/jobs/{id}/cancel` | Cancel a queued or running job (`409` if already finished). Rows already produced are kept. |
| `POST /v1/jobs/{id}/webhook/redeliver?event=` | Retry a webhook delivery that ended `failed` (see Webhooks). `event` defaults to the job's own final event. Rate limited like submit. |
| `DELETE /v1/jobs/{id}` | Remove a finished job and its rows (`409` while queued / running: cancel first). |

### Submit

```json
{
  "kind": "classify",
  "payload": {"inputs": ["Checkout is down", "Charged twice"], "labels": ["billing", "technical"]},
  "webhook": {"url": "https://hooks.example.com/clef", "secret": "s3cret", "events": ["job.succeeded", "job.failed"]},
  "metadata": {"batch": "2026-10-03"}
}
```

`webhook` and `metadata` are optional (`metadata` is any JSON up to 16 KB, echoed back on every read).

**Idempotency.** Send an `Idempotency-Key` header (1-128 printable ASCII characters) and a resubmission with the
same key by the same API key returns the job created the first time (`202`, plus `Idempotent-Replay: true`)
instead of queueing a duplicate. Use it when the response to a submit may have been lost. The key is remembered
as long as the job is (until it is deleted or purged); the payload of a replay is not compared. The Python
clients take `idempotency_key=`, the JS client `idempotencyKey`.

The payload is validated completely at submit time: a mistake is a `400` with a path, for example
`payload.inputs: inputs must contain at least one item` or `payload.requests[12].questions: at least one question is required`,
and nothing is queued.

Built-in kinds:

| `kind` | `payload` | Result rows |
| --- | --- | --- |
| `classify` | `inputs` (up to `CLEF_MAX_JOB_ITEMS`), either `labels` or `classifier` (a saved classify classifier), optional `instructions`, `multi_label`, `threshold`, `model` | `{index, label, confidence, scores, input_tokens}` (multi-label: `labels`, `threshold` instead of `label`, `confidence`) |
| `score` | `inputs`, either `levels` or `classifier` (a saved score classifier), optional `instructions`, `model` | `{index, score, level, level_index, confidence, distribution, input_tokens}` |
| `systemone` | `requests`: a list of `/v1/systemone` bodies (media allowed, same per-request limits) | `{index, model, answers, input_tokens}` |
| `evaluate` | the same body as `POST /v1/evaluate` (up to `CLEF_MAX_JOB_EVAL_ROWS` rows), see [evaluation](evaluation.md#as-a-job) | `{index, input, gold, predicted, confidence, correct, scores}`; the job `result` is the metrics object |

Other kinds can be registered by feature modules (`evaluate` is one); `POST` with an unknown
kind answers `400` and lists the available ones. A saved `classifier` is **copied into the job at submit**: the
stored payload carries its labels / levels, instructions, `multi_label` and threshold (and `snapshot_of` with
the name). Editing or deleting the classifier afterwards does not affect the job, queued, running or resumed,
so one result set never mixes two definitions.

A row the engine rejects (for example one record over `CLEF_MAX_TOKENS`) does **not** fail the job: its row is
`{"index": 17, "error": "input is too long ..."}`, `progress.failed` counts it, and the rest of the chunk is
processed normally. Those failures are never silent: the job view carries a `warnings` entry
(`"3 of 100 item(s) failed; ..."`) and clients expose `has_errors` / `hasErrors`. Two guards stop a job
whose failures are systemic (for example a model that rejects every record) instead of grinding through it:
the job is **failed** with the item error as its message when the first 16 items of a run all failed, or when
at least 90% of 50 or more items failed with one and the same message; and a job in which **every** item failed
ends `failed` (`"all 2 item(s) failed (...)"`), never `succeeded`. Rows produced before the abort stay readable.
A job also fails on engine crashes, or when the engine stayed unavailable for `CLEF_JOB_ENGINE_WAIT_S`. While
the model is still loading a job waits and retries; it does not fail. Every failure is logged
(`job <id> (<kind>) failed: ...`); the client sees only a generic message for unexpected errors.

### Job object

```json
{
  "id": "job_3f9c1e7a5b2d4c60e8a1",
  "kind": "classify",
  "status": "running",
  "progress": {"done": 12000, "total": 50000, "failed": 3, "percent": 24.0},
  "eta_s": 4560.2, "items_per_s": 8.33,
  "created_at": "2026-10-03T09:00:00Z", "started_at": "2026-10-03T09:00:01Z", "finished_at": null,
  "queued_s": 1.1, "duration_s": 1440.0,
  "cancel_requested": false, "resumes": 0, "error": null,
  "result": null,
  "warnings": ["3 of 12000 item(s) failed; their result rows carry an 'error' field"],
  "metadata": {"batch": "2026-10-03"},
  "webhook": {"url": "https://hooks.example.com/clef", "events": ["job.succeeded"], "has_secret": true,
              "failed_deliveries": 0,
              "deliveries": {"job.succeeded": {"id": "...", "status": "pending", "attempts": 0}}}
}
```

`result` is set when the job succeeds (classify: `{items, ok, errors, by_label}`, with `by_labels` for
multi-label; score: `by_level`). `status: "succeeded"` means the job ran to the end, **not** that every item
worked: look at `progress.failed` and `warnings` (a non-empty list also reports failed webhook deliveries).
`eta_s` and `items_per_s` are measured on the current run. The webhook secret
is never returned; `deliveries.<event>.status` is `pending`, `delivered` or `failed` (with `attempts`,
`last_status` and `last_error`).

## Examples

### curl

```bash
# submit (50k rows would go in "inputs"; raise CLEF_MAX_BODY_MB if the JSON is larger than 64 MB)
curl -sS -i http://127.0.0.1:8910/v1/jobs -H 'X-API-Key: ...' -H 'Content-Type: application/json' -d '{
  "kind": "classify",
  "payload": {"inputs": ["Checkout is down", "Charged twice"], "labels": ["billing", "technical"]}
}'
# poll
curl -sS http://127.0.0.1:8910/v1/jobs/job_3f9c1e7a5b2d4c60e8a1 -H 'X-API-Key: ...'
# results: first page as JSON, or the whole thing as a CSV file
curl -sS 'http://127.0.0.1:8910/v1/jobs/job_3f9c1e7a5b2d4c60e8a1/results?limit=100' -H 'X-API-Key: ...'
curl -sS -o results.csv 'http://127.0.0.1:8910/v1/jobs/job_3f9c1e7a5b2d4c60e8a1/results?format=csv' -H 'X-API-Key: ...'
```

CSV columns are the flattened row (`index`, `label`, `confidence`, `scores.billing`, ...) and `error` last;
`labels` lists are joined with `|`. Cells that start with `=`, `+`, `-` or `@` get a leading quote so a spreadsheet
does not run them as formulas. Row `index` is the position of the input, so you can join back to your data.

### Python

```python
from clef_client import ClefClient

with ClefClient() as clef:
    job = clef.classify_job(texts, ["billing", "technical", "account"])  # returns at once
    job = clef.wait_job(job.id, poll=2, on_progress=lambda j: print(j.status, j.done, j.total))
    if not job.ok:  # failed / cancelled
        raise SystemExit(job.error)
    if job.has_errors:  # succeeded, but some items failed: their rows carry .error
        print("warning:", *job.warnings, sep="\n  ")
    for item in clef.job_results(job.id):  # pages of 500, in input order
        print(item.index, item.error or item.result.label)
    clef.save_job_results(job.id, "results.csv", format="csv")  # or stream straight to disk
```

`job.ok` means the job ran to its end (`succeeded`); `job.has_errors` / `job.failed` / `job.warnings` tell you
that items failed. Results are **partial while the job runs**: `job_results()` ends at the rows stored right now
unless you call `wait_job` first or pass `wait=True` (it then polls until the job is finished), and
`save_job_results()` raises `JobNotFinished` for an unfinished job unless `require_finished=False`. The file is
written to `<path>.part` and renamed, so a failed download never leaves a truncated file. `jobs()` returns a
`list[Job]` that also has `.total`, `.limit`, `.offset` and `.has_more`. `redeliver_webhook(id)` retries a failed
webhook delivery. Pass `idempotency_key=` to `submit_job` / `classify_job` so a retried submit cannot duplicate
the job.

`AsyncClefClient` has the same methods (`await`, and `async for` over `job_results`). Other methods:
`submit_job(kind, payload, webhook=..., metadata=...)`, `job(id)`, `jobs(status=...)`, `cancel_job(id)`,
`delete_job(id)`. `wait_job` raises `TimeoutError` when `timeout=` passes; the job keeps running. See
`examples/jobs.py` for a CSV in, CSV out script.

### JavaScript

```js
import { ClefClient } from './clients/js/src/index.js';

const clef = new ClefClient();
let job = await clef.classifyJob(texts, ['billing', 'technical', 'account']);
job = await clef.waitJob(job.id, { pollMs: 2000, onProgress: (j) => console.log(j.status, j.done, j.total) });
if (!job.ok) throw new Error(job.error);
if (job.hasErrors) console.warn(job.warnings);
for await (const item of clef.jobResults(job.id)) console.log(item.index, item.error ?? item.result.label);
```

Same semantics as Python: `ok` is "ran to its end", `hasErrors` / `failed` / `warnings` report failed items, `jobResults`
is partial while the job runs (`{ wait: true }` polls until it is finished), and `jobs()` returns an array with
`total`, `limit`, `offset`, `hasMore`. The client does not retry a network error after `POST /v1/jobs` was sent
(the job might exist already); only connect errors, `429` and `503` are retried. With `{ idempotencyKey }` the
server de-duplicates, so those retries are enabled too. `redeliverWebhook(id)` retries a failed delivery.

## Webhooks

Instead of polling, pass `webhook: {url, secret?, events?}` when submitting. clef POSTs a JSON event when the
job reaches a final state:

| Event | When |
| --- | --- |
| `job.succeeded`, `job.failed`, `job.cancelled` | the job ended (these three are the default) |
| `job.progress` | opt-in; at most one every 10 s, one attempt, never retried |

```json
{"event": "job.succeeded", "delivery_id": "9b1f...", "created_at": "2026-10-03T10:00:00Z", "job": { ...the job object... }}
```

The body carries the job object (status, progress, `result` summary), not the result rows; fetch those with
the API. Headers: `X-Clef-Event`, `X-Clef-Delivery` (stable across retries, use it to de-duplicate), and, when
the webhook has a `secret`, `X-Clef-Timestamp` (unix seconds) and `X-Clef-Signature: sha256=<hex>` where the
hex value is the HMAC-SHA256 of `"<timestamp>.<raw body>"` keyed with the secret. The signature is recomputed
per attempt, so the timestamp is fresh each time.

**Delivery.** Any 2xx is success. 408, 425, 429, 5xx and network errors are retried with exponential backoff:
2, 4, 8, 16, 32, 64, then 120 s between attempts (`CLEF_WEBHOOK_BACKOFF_S` and `CLEF_WEBHOOK_BACKOFF_CAP_S`, +-25%
jitter), up to `CLEF_WEBHOOK_ATTEMPTS` attempts in total (default 10, about 8 minutes, so a receiver that restarts
or deploys is covered); other 4xx and every 3xx are final (redirects are never followed). Each attempt has a
`CLEF_WEBHOOK_TIMEOUT_S` (default 10) total timeout. The delivery id (`X-Clef-Delivery`) is minted when the final
event is recorded, together with the job's state change, so it stays the same across retries and server
restarts. A pending delivery survives a restart and is retried on the next start. Status is in
`GET /v1/jobs/{id}` under `webhook.deliveries`; `webhook.failed_deliveries` counts the ones that gave up, and each
shows up in `warnings`. A finished job with a pending final event is never purged, and a failed delivery is not
retried automatically: after fixing the receiver, `POST /v1/jobs/{id}/webhook/redeliver` (optional `?event=`)
sends it again with the same id and a fresh attempt budget (`409` unless that delivery is `failed`).
Webhooks are at-least-once: de-duplicate on `X-Clef-Delivery`.

### Verify on the receiving side

```python
import hashlib, hmac, time


def verify(secret: str, headers, raw_body: bytes, tolerance_s: int = 300) -> bool:
    """headers: any case-insensitive mapping (Flask / FastAPI / requests headers). raw_body: bytes as received."""
    ts = headers["X-Clef-Timestamp"]
    if abs(time.time() - int(ts)) > tolerance_s:  # replay protection
        return False
    expected = (
        "sha256=" + hmac.new(secret.encode(), ts.encode() + b"." + raw_body, hashlib.sha256).hexdigest()
    )
    return hmac.compare_digest(expected, headers["X-Clef-Signature"])
```

Always compute over the raw bytes (not a re-serialised JSON object), reject a missing signature when you
configured a secret, and answer `2xx` quickly (do the work asynchronously).

### Webhooks are off by default (SSRF)

A webhook makes the server send a request to an address chosen by the API caller. So the feature is **disabled
until the operator sets `CLEF_WEBHOOK_ALLOW`** to a comma-separated list of what may be called:

```bash
CLEF_WEBHOOK_ALLOW='hooks.example.com,*.corp.example.net,192.168.1.0/24'
```

* A hostname must be listed exactly (`hooks.example.com`) or by suffix (`*.corp.example.net`). A URL whose
  host is an IP literal must fall inside a listed IP or CIDR.
* Checked at submit (`400` with `webhook.url: ...`) **and again on every delivery attempt**: the hostname is
  resolved right then, **every** returned address must be public or inside a listed IP / CIDR, and the request
  is sent to that checked address (Host header and TLS server name keep the hostname). A DNS answer that
  changes between attempts (DNS rebinding) is therefore refused.
* A private or loopback receiver (for example `192.168.1.20`, or `127.0.0.1` for a local script) must be listed
  by IP or CIDR; listing only its hostname is not enough. Link-local (169.254.0.0/16, which includes cloud
  metadata endpoints), multicast, unspecified and reserved addresses are refused even if a CIDR would cover them.
* IPv6 addresses that embed an IPv4 address (IPv4-mapped, NAT64 `64:ff9b::/96` and `64:ff9b:1::/48`, 6to4, Teredo)
  are judged by the IPv4 inside, so `64:ff9b::a00:5` is refused as the private `10.0.0.5` it translates to
  (listing `10.0.0.0/8` is what allows it). Site-local and reserved IPv6 ranges are refused; `::1` still works
  when listed.
* Only `http` / `https`, no credentials in the URL, no redirects, no proxy settings from the environment, and an
  API key is never sent. Use `https` for anything that leaves your machine.
* Secrets are stored in the job database (`jobs.db` in the state directory) in clear text, because HMAC needs
  them; protect that directory like the API keys file. The secret is write-only through the API.

## Limits, ownership, retention

| Setting | Default | Meaning |
| --- | --- | --- |
| `CLEF_MAX_JOB_ITEMS` | 100000 | Items per job (`classify`, `score` inputs / `systemone` requests). |
| `CLEF_MAX_JOBS` | 100 | Queued + running jobs; more is `409` with `Retry-After`. |
| `CLEF_JOB_TTL_HOURS` | 168 | Finished jobs (and their rows) are purged this long after they finish; `0` keeps them forever. |
| `CLEF_JOB_ENGINE_WAIT_S` | 600 | How long a job waits for an unavailable engine before failing. |
| `CLEF_WEBHOOK_ALLOW` | empty | Allowed webhook hosts / IPs / CIDRs; empty disables webhooks. |
| `CLEF_WEBHOOK_TIMEOUT_S` | 10 | Total timeout of one delivery attempt. |
| `CLEF_WEBHOOK_ATTEMPTS` | 10 | Delivery attempts for final events. |
| `CLEF_WEBHOOK_BACKOFF_S` | 2 | First retry delay; doubles each attempt. |
| `CLEF_WEBHOOK_BACKOFF_CAP_S` | 120 | Longest delay between two attempts. |
| `CLEF_JOB_MAX_RESUMES` | 3 | A job interrupted more often than this fails (`interrupted N times, giving up`); `0` = no limit. |

The request body limit (`CLEF_MAX_BODY_MB`, default 64) applies to `POST /v1/jobs`: roughly 100 000 short
texts fit, 50 000 long ones may not. Media in `systemone` jobs counts too. Per-record limits (labels, images,
tokens) are the same as for the synchronous endpoints.

**Ownership.** With auth off, every caller sees every job. With `CLEF_API_KEY` / `CLEF_API_KEYS` on, a job is
owned by the key name that submitted it (`default` for `CLEF_API_KEY`); other keys get `404` for it (not `403`)
and do not see it in `GET /v1/jobs`. There is no admin view: stop the server and read `jobs.db` if you need
one. Jobs submitted while auth was off stay visible to all keys after you turn it on. Webhook URLs and
metadata are visible to the owner.

**Persistence and restart.** Jobs, their submitted payloads (your inputs) and result rows live in SQLite
(`<state dir>/jobs.db`, in clear text) and survive restarts.
Rows are written after every chunk. If the server stops while a job runs it is marked `interrupted` and, on the
next start, **resumed** from the last stored row (`resumes` counts how often), ahead of queued jobs. A job that
keeps killing the process would otherwise be resumed forever, so after `CLEF_JOB_MAX_RESUMES` interruptions
(default 3) it is failed with `interrupted N times, giving up` and a WARNING is logged. The built-in kinds (including
`evaluate`) resume exactly; a custom kind that does not declare itself resumable restarts from item 0. Resuming re-reads
the stored payload, which already holds the saved classifier as it was at submit. Finished jobs are purged on
their own timer (every ten minutes), also while a job is running. Delete the file to drop all jobs (an older
`jobs.db` is upgraded in place).

**Throughput.** Jobs run one at a time, FIFO, at the engine's batch speed, shared with interactive traffic.
Priority between jobs, and a priority lane inside the engine, do not exist. Measured on the RX 7900 XTX (bf16,
short tickets): a 500-item `classify` job ran at ~15 items/s (32 s); meanwhile interactive `/v1/classify` calls
took p50 422 / p95 458 ms instead of 109 / 118 ms idle. They are never starved, but expect them to be ~4x slower
while a large job runs.

## Writing a job kind

A feature module registers `{"validate": fn, "run": async_fn}` in `ctx.extra["job_kinds"]` (use `setdefault`). An
optional `"snapshot": fn(payload, parsed) -> payload` rewrites what is stored at submit (the built-in kinds use it
to copy a saved classifier into the payload):

```python
def validate(payload: dict) -> Parsed: ...      # raise ValueError -> 400 at submit; return the parsed object

async def run(ctx, parsed, job) -> dict | None:
    job.set_total(len(parsed.items))
    for chunk in chunks(parsed.items, 8):
        if job.cancelled:
            return None
        await job.add_items([{"index": ..., ...}, ...])   # persist rows in order; progress += len(rows)
    return {"summary": "stored as job.result"}
```

Rows containing an `"error"` key count as failed, and the circuit breaker above applies to every kind (it raises
`JobAborted` from `add_items`). Call the model with `ctx.decide(requests, batch=True)`, keep one
call in flight, and check `job.cancelled` between calls. Add `"resumable": True` only if `run` honours
`job.resume_from` (the number of rows already stored) and does not re-emit them.
