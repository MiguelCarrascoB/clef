# OpenAI and Hugging Face compatibility

clef speaks two familiar dialects so existing tools can use it without code changes:

- **OpenAI**: `GET /v1/models`, `POST /v1/chat/completions` (structured outputs and forced function calls),
  and `/v1/files` + `/v1/batches` (the [Batch API](#files-and-batches-the-openai-batch-api)).
- **Hugging Face Inference API**: zero-shot classification at `POST /hf/models/{model_id}`.

Both sit on the same engine and the same `infer` path as `/v1/systemone`: same API keys, same rate limit, same
request log and stats, same limits (`CLEF_MAX_QUESTIONS`, `CLEF_MAX_LABELS`, ...). Each request is one SystemOne
record, so the probabilities are identical to the equivalent `/v1/systemone` call.

!!! note "What clef is, and is not"
    clef-flash is a *decision* model. Given an input and a set of options it returns a calibrated probability for
    every option in one forward pass; it does not generate text. That is a great fit for routing, triage,
    moderation, tagging, ratings and gating, and the wrong tool for chat, summaries or code.

    So a chat completion here is a **structured output**: you describe the decisions you want as a JSON schema
    (enums, booleans, integer scales) and clef answers each one. A request with no usable schema gets a clear `400`
    explaining this instead of a made-up reply.

## OpenAI SDK

Point the SDK at `/v1` and use any API key (or your `CLEF_API_KEY`, sent as `Authorization: Bearer`).

```python
from typing import Literal
from openai import OpenAI
from pydantic import BaseModel

client = OpenAI(base_url="http://127.0.0.1:8910/v1", api_key="not-needed")


class Triage(BaseModel):
    team: Literal["billing", "technical", "account"]
    outage: bool


r = client.beta.chat.completions.parse(
    model="clef-flash",
    messages=[
        {"role": "system", "content": "You triage support tickets."},
        {"role": "user", "content": "Checkout is down, orders blocked"},
    ],
    response_format=Triage,
)
print(r.choices[0].message.parsed)  # Triage(team='technical', outage=True)
```

Plain JSON schema works the same way, and `curl` needs nothing but JSON:

```bash
curl -s http://127.0.0.1:8910/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "model": "clef-flash",
  "messages": [{"role": "user", "content": "Checkout is down"}],
  "response_format": {"type": "json_schema", "json_schema": {"name": "triage", "schema": {
    "type": "object",
    "properties": {
      "team": {"type": "string", "enum": ["billing", "technical"], "description": "Who handles it?"},
      "outage": {"type": "boolean", "description": "Is a service down?"}
    }}}}
}'
```

```json
{"id": "chatcmpl-...", "object": "chat.completion", "model": "clef-flash",
 "choices": [{"index": 0, "finish_reason": "stop", "logprobs": null,
              "message": {"role": "assistant", "content": "{\"team\": \"technical\", \"outage\": true}"}}],
 "usage": {"prompt_tokens": 120, "completion_tokens": 0, "total_tokens": 120},
 "clef": {"questions": {"team": {"type": "choice", "value": "technical", "confidence": 0.96,
                                 "probabilities": {"billing": 0.04, "technical": 0.96}},
                        "outage": {"type": "bool", "value": true, "confidence": 0.97,
                                   "probabilities": {"true": 0.97, "false": 0.03}}},
          "timing": {"forward_ms": 141.2, "total_ms": 150.3}, "request_id": "..."}}
```

LangChain and similar frameworks only need the base URL and a structured-output call. Verified with
`langchain-openai` 1.7.1 and `langchain-core` 1.6.10 (2026-10-12) against a live server: `with_structured_output(Triage)`
(the default `json_schema` method) and `method="function_calling"` both return the right object. `method="json_mode"` and
a plain `invoke()` carry no schema, so they get the `400 missing_schema` described above:

```python
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(base_url="http://127.0.0.1:8910/v1", api_key="not-needed", model="clef-flash")
triage = llm.with_structured_output(Triage)
print(triage.invoke("Checkout is down, orders blocked"))
```

Any tool with an "OpenAI-compatible endpoint" setting works the same way: base URL `http://127.0.0.1:8910/v1`,
model `clef-flash`. The tool just has to ask for structured output.

## How a request maps to clef

| OpenAI | clef |
| --- | --- |
| `messages` | the SystemOne `state` (see below) |
| `image_url` content parts (data URLs) | `images` |
| `video_url` content parts (non-standard, as in vLLM) | `videos` |
| `response_format.json_schema.schema` | one question per top-level property |
| `tools` + `tool_choice` naming one function | the function's `parameters` schema, same rules; answer returned as `tool_calls` |
| `logprobs` / `top_logprobs` | `log(p)` of each option (below) |
| `usage` | clef `usage` (`completion_tokens` is always 0) |

### Messages become the state

A lone user message is passed through verbatim. Anything else becomes a readable transcript, in order, one blank line
between turns:

```text
system: You triage support tickets.

user: Checkout is down

assistant: Noted.

user: Still down
```

Assistant and tool turns are kept because they are context for the decision (`developer` is read as `system`,
`function` as `tool`; an assistant tool call shows as `[called name(args)]`). Text parts of a content array are joined
with newlines. Audio and file parts are rejected with a `400`.

Images follow the same rules as everywhere else: `data:` URLs always, `http(s)` URLs only when
`CLEF_ALLOW_URL_FETCH=1`.

### Schema properties become questions

Each **top-level property** of the schema is one decision. `required`, `additionalProperties`, `strict` and `name`
are ignored (every property is always answered). `$ref`/`$defs`, `allOf` with one entry, `Optional` (`anyOf` with
`null`) and unions of constants are resolved, so schemas generated by Pydantic work as they are.

| Property | clef question | Value returned |
| --- | --- | --- |
| `{"type": "string", "enum": [...]}` (2+ options), a `Literal`, an `Enum` | `choice` | the most likely option |
| `{"type": "boolean"}` | `noul` (yes/no) | `true` when P(yes) >= 0.5 |
| `{"type": "integer", "enum": [1, 2, 3]}` | `score` over the sorted values | the most likely value (an integer) |
| `{"type": "integer", "minimum": a, "maximum": b}` (2..64 values) | `score` over `a..b` | the most likely integer |
| string enum with `"x-clef-ordinal": true` | `score`, options in the order given | the most likely option |
| `{"type": "array", "items": {"enum": [...]}}` | one `noul` per option (multi-label) | the options with P >= `CLEF_CLASSIFY_THRESHOLD` (0.5), best first |

The property `description` becomes the question's instructions, and a description at the top of the schema (or the
function description for tools) is placed in front of every question. Option descriptions help the model when the
option names are ambiguous. Two ways to give them, both plain JSON schema:

```json
{"type": "string", "x-clef-descriptions": {"billing": "Payments, invoices, refunds", "technical": "Bugs, outages"},
 "enum": ["billing", "technical"]}
```

```json
{"oneOf": [{"const": "billing", "description": "Payments, invoices, refunds"},
           {"const": "technical", "description": "Bugs, outages"}]}
```

For integer scales the keys of `x-clef-descriptions` are the numbers as text (`{"1": "Can wait", "3": "Today"}`).
`x-clef-*` keywords are clef extensions and are ignored by everything else; with Pydantic, set them through
`Field(json_schema_extra={"x-clef-ordinal": True})`.

!!! tip "choice vs score"
    Use `score` (integer enum, bounded integer, or `x-clef-ordinal`) when the options are ordered, such as
    low/medium/high or 1-5 stars; clef then reasons about the ordering. Use plain enums for unordered categories.

Properties clef cannot decide (free-form `string`, `number`, nested `object`, ...) are rejected with a `400` whose
`param` points at the property, for example `response_format.json_schema.schema.properties.summary`.

### Forced function calls

`tools` with one function and `tool_choice` set to that function (or `"required"`, or `"auto"` with a single tool)
returns a `tool_calls` message whose `arguments` is the JSON object, exactly like a model that obeyed the call. The
`finish_reason` is `"stop"` when the function was named in `tool_choice` (as OpenAI does) and `"tool_calls"`
otherwise. clef cannot decide *whether* or *which* of several tools to call: with several tools, name one in
`tool_choice` (a `400` says so otherwise). `tool_choice: "none"` ignores the tools. If both a forced tool and a
`response_format` are sent, the tool wins.

### Logprobs and probabilities

With `"logprobs": true`, `choices[0].logprobs.content` has one token-like entry per decision (per option for
arrays): `token` is the chosen value as text, `logprob` is `log(p)` of it. `top_logprobs: N` (0..64) adds the best N
options of that decision with their `log(p)`, so `exp(logprob)` recovers clef's calibrated probability. Without
`top_logprobs` the lists are empty, as in OpenAI. Probabilities that round to zero are reported as `-9999`.

The non-standard top-level **`clef`** field always has the full picture: per property the type, value, confidence and
all probabilities (`{"true": ..., "false": ...}` for booleans, option -> probability otherwise), plus `timing` and
`request_id`. The official SDKs expose unknown fields (`response.model_extra["clef"]` in Python).

### Streaming

`"stream": true` returns valid Server-Sent Events: a `chat.completion.chunk` with the role, one with the whole JSON as
`content` (or the `tool_calls` entry), one with `finish_reason`, then `data: [DONE]`. A decision is a single forward
pass, so there is nothing to stream token by token. `stream_options: {"include_usage": true}` adds the usage chunk.
Validation errors still come back as a normal JSON error response, before any chunk.

## Supported and ignored parameters

| Parameter | Behaviour |
| --- | --- |
| `model` | accepted and ignored: always served by `clef-flash` (so hard-coded names like `gpt-4o-mini` work) |
| `messages` | required; see above |
| `response_format` | `json_schema` required (unless a tool is used); `text` and `json_object` give a `400` explaining why |
| `tools`, `tool_choice` | one function, forced; `type: "function"` only |
| `logprobs`, `top_logprobs` | supported (above); `top_logprobs` needs `logprobs: true` |
| `stream`, `stream_options.include_usage` | supported |
| `n` | `1` only; `n > 1` is a `400` (a decision is deterministic, there is nothing to sample) |
| `temperature`, `top_p`, `max_tokens`, `max_completion_tokens`, `stop`, `seed`, `presence_penalty`, `frequency_penalty`, `logit_bias`, `user`, `metadata`, `store`, `parallel_tool_calls`, `reasoning_effort`, `service_tier`, ... | accepted and ignored |
| `functions`, `function_call` (legacy) | not supported; use `tools` |

Ignored fields are not silent: a chat response (and the final stream chunk) carries the non-standard
`clef.ignored` list naming each ignored field the request sent, e.g. `["model", "temperature", "max_tokens"]`
(`model` appears only when it is not `clef-flash`; the list is `[]` when nothing was ignored). The same list is
logged at DEBUG. Clients that send a fixed model name keep working, and you can see that the name was not used.

`GET /v1/models` lists `clef-flash`; `GET /v1/models/clef-flash` returns it and any other id gives a `404`
(`model_not_found`).

### Errors

Errors on the OpenAI routes use OpenAI's shape so SDKs raise their usual exceptions
(`BadRequestError`, `AuthenticationError`, `RateLimitError`, ...):

```json
{"error": {"message": "...", "type": "invalid_request_error", "param": "n", "code": "unsupported_value"}}
```

Status codes match the rest of the API: `400` bad request or schema, `401` missing/invalid key
(`code: invalid_api_key`), `404` unknown model, `413` too large, `429` rate limited (with `Retry-After`,
`type: rate_limit_error`), `503` model loading or GPU out of memory. The request id is in the `X-Request-ID` header.
The native `/v1/*` routes keep `{detail, request_id}`.

## Files and batches (the OpenAI Batch API)

`/v1/files` and `/v1/batches` follow OpenAI's Batch API, so the `openai` SDK's `client.files` and `client.batches`
work unchanged. A batch is one [async job](jobs.md) of kind `batch` (it shows up in `GET /v1/jobs?kind=batch`), so it
shares the job queue, the daily quota, resume after a restart and the retention timer. All routes need the `jobs`
scope and are private to the key that created them (an admin sees every file and batch). Errors use the OpenAI
shape, `{"error": {"message", "type", "param", "code"}}`.

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="...")
f = client.files.create(file=open("requests.jsonl", "rb"), purpose="batch")
batch = client.batches.create(input_file_id=f.id, endpoint="/v1/chat/completions", completion_window="24h")
# poll client.batches.retrieve(batch.id) until status is completed, then
print(client.files.content(batch.output_file_id).text)
```

### The input file

One JSON object per line (blank lines are ignored), at most `min(CLEF_MAX_BATCH_LINES, CLEF_MAX_JOB_ITEMS)` lines
(50,000 by default), up to `CLEF_MAX_FILE_MB` (200):

```json
{"custom_id": "ticket-1", "method": "POST", "url": "/v1/chat/completions", "body": {"messages": [...], "response_format": {...}}}
```

`custom_id` is unique, `method` is `POST` and `url` equals the batch's `endpoint`. The supported endpoints are
`/v1/chat/completions`, `/v1/classify` and `/v1/score`; each `body` is exactly what that route takes, with the same
validation, the same limits and the same probabilities. (`stream: true` is not allowed in a batch.) The SDK does not
check `endpoint`; clef answers `400` with `param: "endpoint"` for any other value.

### Files

| Route | |
| --- | --- |
| `POST /v1/files` | multipart: `file` (`.jsonl`), `purpose=batch`, optional `expires_after[anchor]=created_at` and `expires_after[seconds]` (3600..2592000). Streamed to disk; the body limit of this route alone is `CLEF_MAX_FILE_MB` |
| `GET /v1/files` | `after`, `limit` (default 20), `order` (`desc`), `purpose`; cursor list |
| `GET /v1/files/{id}`, `GET /v1/files/{id}/content` | the file object; the JSONL, streamed |
| `DELETE /v1/files/{id}` | uploaded files only. A batch's output / error file is deleted with its job (`DELETE /v1/jobs/{job_id}`); an input file an unfinished batch still reads is `409` |

Uploads expire after `expires_after` or `CLEF_FILE_TTL_HOURS` (30 days). The **output and error files are not
copies**: they are streamed from the job's result rows, so they cost no disk and live as long as the job
(`CLEF_JOB_TTL_HOURS`).

### Batches

| Route | |
| --- | --- |
| `POST /v1/batches` | `input_file_id`, `endpoint`, `completion_window` (`"24h"`), optional `metadata` (<= 16 string pairs; keys <= 64, values <= 512 characters) and `extra_body={"clef": {...}}` |
| `GET /v1/batches/{id}`, `GET /v1/batches` | the batch; newest first, `after`, `limit` 1..100 |
| `POST /v1/batches/{id}/cancel` | `cancelling` while a running chunk finishes, then `cancelled`; a queued batch is `cancelled` at once. `409` when it already ended |

`status` is `in_progress`, `cancelling`, `completed`, `failed`, `expired` or `cancelled`, with OpenAI's timestamps and
`request_counts` (`completed` = succeeded, `failed` = non-2xx lines). `usage.input_tokens` is the sum over the lines.
`expires_at` is 24 hours after the batch is eligible to run: its creation time, or the next time its window opens when
`clef.schedule` (or `CLEF_JOB_WINDOW`) holds it back. A batch not finished by then ends `expired`.

- A file that is **structurally** wrong (a line that is not JSON, a duplicate or missing `custom_id`, a wrong `url`
  or `method`, too many lines, an empty file) makes `POST /v1/batches` return a batch with status `failed` and
  `errors.data = [{code, line, message, param}]` (at most 100). No job runs and nothing is charged.
- A **request** that cannot run (a schema with no decisions, `n > 1`, an unknown `model`, a label set over the
  limit) is a line of the error file with its OpenAI error body and status code; the batch still ends `completed`,
  even when every line failed.
- `output_file_id` appears once the batch ended with at least one success, `error_file_id` when a line failed or
  never ran. Output is in input order: `{"id": "batch_req_...", "custom_id", "response": {"status_code",
  "request_id", "body"}, "error": null}`. For a cancelled, expired or failed batch the error file also names, with
  `{"response": null, "error": {"code": "batch_cancelled" | "batch_expired" | "batch_failed", "message"}}`, every
  line that never ran; those custom ids are read back from the input file, which is why it cannot be deleted while
  the batch is unfinished.
- A batch that stops for a systemic reason (the engine went away, the circuit breaker) is `failed` with
  `errors.data = [{"code": "server_error", ...}]`; the rows it did produce are still served.

The `clef` extension, `extra_body={"clef": {"priority": "low", "urgent": false, "schedule": {"window": "20:00-06:00",
"tz": "Europe/Madrid"}}}` (`priority` `low|normal|high`), is the [job scheduler](jobs.md#scheduling)'s: the batch is
queued, ordered and windowed like any job. `urgent` needs the `urgent` scope (`403` otherwise). The daily quota of the key is reserved when the batch is created (`429`
`insufficient_quota` when it does not fit; a failed or cancelled batch gets the unprocessed part back).

| Variable | Default | Meaning |
| --- | --- | --- |
| `CLEF_MAX_FILE_MB` | `200` | body limit of `POST /v1/files` only (every other route keeps `CLEF_MAX_BODY_MB`) |
| `CLEF_MAX_BATCH_LINES` | `50000` | requests per batch (also capped by `CLEF_MAX_JOB_ITEMS`) |
| `CLEF_FILE_TTL_HOURS` | `720` | how long an uploaded file is kept unless it has `expires_after`; `0` = forever |

Not supported: other `purpose` values (`assistants`, `fine-tune`...), other endpoints, `completion_window` other than
`24h`, `output_expires_after` (ignored), `/v1/batches` from a file that another key uploaded (`404`, admins too).

## Hugging Face zero-shot classification

`POST /hf/models/{model_id}` (also `POST /models/{model_id}`) takes the Inference API body. `model_id` is ignored: any value, even an
unknown one, is served by `clef-flash` (no error, no warning), and it may contain slashes (`facebook/bart-large-mnli`).

```bash
curl -s http://127.0.0.1:8910/hf/models/clef-flash -H 'Content-Type: application/json' -d '{
  "inputs": "Checkout is down, orders blocked",
  "parameters": {"candidate_labels": ["billing", "technical", "account"], "multi_label": false}
}'
```

```json
{"sequence": "Checkout is down, orders blocked",
 "labels": ["technical", "billing", "account"], "scores": [0.94, 0.04, 0.02]}
```

| Field | Behaviour |
| --- | --- |
| `inputs` | a string, or a list of strings (answered in one micro-batched pass, a list of results back; max `CLEF_MAX_BATCH`) |
| `parameters.candidate_labels` | list of strings, or a comma-separated string; de-duplicated; 2+ labels (1+ with `multi_label`) |
| `parameters.multi_label` | `false` (default): one label, scores sum to 1. `true`: independent yes/no per label, scores do not sum to 1 |
| `parameters.hypothesis_template` | must contain `{}`. clef has no NLI hypothesis step, so the filled template (`"This is about billing."`) is used as that label's description instead |
| `options`, other fields | ignored |

Labels and scores are sorted by score, best first. Errors are `{"error": "message"}` with the usual status codes.

### Using `huggingface_hub`

`InferenceClient` posts to the model URL as given, so pass the full route as the model:

```python
from huggingface_hub import InferenceClient

client = InferenceClient(
    model="http://127.0.0.1:8910/hf/models/clef-flash", token="your CLEF_API_KEY or None"
)
for item in client.zero_shot_classification("Checkout is down", ["billing", "technical", "account"]):
    print(item.label, item.score)
```

!!! note "Two response shapes"
    The classic Inference API answers `{"sequence", "labels", "scores"}` (the default here). `huggingface_hub` 1.x
    expects the newer `[{"label", "score"}, ...]` list. clef detects `hf_hub/1.x` in the `User-Agent` and answers in
    that shape automatically. For any other client, force a shape with `?format=classic` or `?format=list`.

Verified against `huggingface_hub` 1.33 (single and multi-label, `hypothesis_template`) with the `model=<full URL>`
form above. Pointing `HF_ENDPOINT` at the server is not supported: the library may try Hub lookups for the model id
first, which clef does not serve.

## Limitations

- **No free text.** There is no way to ask clef to write, summarize or explain. If your tool only sends plain chat
  messages, it will get the `400` that explains the schema requirement; configure it for structured output, or call
  `/v1/classify` directly.
- Only top-level properties are decisions; nested objects and free-form strings are not supported.
- One choice per property (argmax). The full distribution is in the `clef` field and in `logprobs`.
- `n`, sampling parameters and token limits have no meaning for a single forward pass and are ignored or rejected as
  listed above.
- Nullability is ignored: an `Optional[...]` property is always decided.
- Streaming delivers the finished decision in one chunk.
