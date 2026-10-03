# clef server v2 - architecture & API contract

This file is the contract between the server modules, the web console, the Python client and the tests.
Change it first, then the code.

## Modules (`server/`, run as `python server/main.py`; siblings import each other as top-level modules)

| Module | Owns |
| --- | --- |
| `config.py` | `Config` dataclass from `CLEF_*` env vars (see file), `VERSION` |
| `schemas.py` | pydantic v2 request/response models + strict validation |
| `media.py` | data-URL / (optional) URL fetching with SSRF guard + size caps, image decode + downscale to `max_pixels`, video decode + frame cap |
| `stats.py` | thread-safe ring buffer of request log entries, latency histogram, counters, forward stats |
| `engine.py` | background model load, single GPU worker thread, micro-batching queue, length bucketing, warmup |
| `main.py` | FastAPI app: lifespan, auth, body-size limit, routes, error mapping, static console mount |
| `static/` | Clef Console (zero-build Preact + htm SPA) served at `/` |

## Engine interface (`engine.py`)

```python
Record = dict  # sanitized: {"model": str, "state": Any, "questions": dict,
               #             "images": list[PIL.Image.Image] | None, "videos": list[np.ndarray] | None}
               # videos: uint8 arrays (T, H, W, 3), already sampled/capped by media.py

class EngineNotReady(RuntimeError): ...          # -> HTTP 503
class InputTooLarge(ValueError): ...             # -> HTTP 413 (e.g. schema alone exceeds max_tokens)
class GpuOutOfMemory(RuntimeError): ...          # -> HTTP 503 (after torch.cuda.empty_cache())

class Engine:
    def __init__(self, cfg: Config, stats: Stats, loader: Callable | None = None): ...
        # loader: optional injection for tests; default imports joint_schema_model from cfg.model_path
        # and returns a module-like object exposing load_release_model, encode_record, collate_records,
        # systemone_answer
    status: str             # "loading" | "warming" | "ready" | "error"
    error: str | None
    load_seconds: float | None
    def start(self) -> None                       # non-blocking: loader thread -> warmup -> worker thread
    async def decide(self, records: list[Record]) -> list[Result]   # results in input order
    def info(self) -> dict                        # {"device","dtype","model_path","torch","transformers",
                                                  #  "gpu": {"available","name","vram_total_gb",
                                                  #  "vram_allocated_gb","vram_reserved_gb"}}
    def shutdown(self) -> None
```

`Result = {"model": str, "answers": {...}, "usage": {"input_tokens": int, "output_tokens": 0},
"timing": {"queue_ms": float, "forward_ms": float, "batch_size": int}}`

Rules:
- Exactly ONE thread ever touches the GPU (the worker). `decide()` enqueues and awaits futures.
- Text-only records from concurrent `decide()` calls are coalesced: the worker drains the queue for up to
  `batch_window_ms` or `max_microbatch` records, groups by length bucket, runs one forward per group.
- Media records run alone (one record per forward) - never merged with others.
- All of a request's media goes into ONE record (no sharding; answers must reflect all media).
- Text batches are grouped by bucket and right-padded to the next `cfg.pad_multiple` (default 64; aligned lengths
  get faster hipBLASLt GEMM tiles). Full padding to the bucket is opt-in (`CLEF_PAD_TO_BUCKET=1`); beyond the
  largest bucket lengths round up to a multiple of 512. Right padding is safe: the model is causal and the head
  slices by attention_mask.
- Warmup (when `cfg.warmup`): one forward per bucket at batch sizes {1, max_microbatch} up to bucket 1024,
  status "warming" meanwhile (requests are accepted and served).
- `torch.cuda.OutOfMemoryError` -> empty_cache, raise `GpuOutOfMemory`, worker keeps running.
- Engine reports forward stats via `stats.record_forward(n_records, n_tokens, padded_tokens, ms)` and queue depth
  via `stats.set_queue_depth(n)`.
- Do NOT modify files under the model directory; wrap/re-implement collation in engine.py.

## Stats interface (`stats.py`)

```python
class Stats:
    def __init__(self, buffer: int = 1000): ...
    def record_request(self, entry: dict) -> None   # entry fields = /v1/log item (minus id/ts, which it assigns)
    def record_forward(self, n_records: int, n_tokens: int, padded_tokens: int, ms: float) -> None
    def set_queue_depth(self, n: int) -> None
    def in_flight(self) -> ContextManager            # increments/decrements in_flight
    def snapshot(self, window_s: int = 300) -> dict   # the /v1/stats body (minus gpu, added by main)
    def log(self, limit: int = 100, since: int | None = None) -> list[dict]
    async def subscribe(self) -> AsyncIterator[dict]  # new log entries, for SSE
```

## HTTP API

Auth: if `CLEF_API_KEY` is set, every `/v1/*` route requires header `X-API-Key: <key>` or
`Authorization: Bearer <key>` (401 otherwise). `/`, `/static/*`, `/health`, `/livez`, `/schema-example`, `/docs`
are open. For SSE (EventSource cannot set headers) `/v1/events?key=<key>` is also accepted.

Body limit: requests with Content-Length > `max_body_mb` -> 413.

### `POST /v1/systemone`
Request:
```json
{"model": "clef-flash", "state": "<string or any JSON>",
 "questions": {"<id>": {"type": "choice", "instructions": "optional", "criteria": {"opt": "desc", "...": "..."}},
               "<id>": {"type": "score", "instructions": "optional", "criteria": ["level0", "level1", "..."]},
               "<id>": {"type": "noul", "instructions": "optional", "criteria": {"true": "desc", "false": "desc"}}},
 "images": ["data:image/...;base64,..."  or "https://..." (only if CLEF_ALLOW_URL_FETCH)],
 "videos": ["data:video/mp4;base64,..."]}
```
- `model` optional, defaults to `"clef-flash"`, must be a string if given (same rule on every route).
- `choice.criteria`: non-empty object of string->string. `score.criteria`: non-empty array of strings.
  `noul.criteria`: optional object with only `true`/`false` keys. `instructions`: optional string.
- Unknown top-level keys are rejected (422-style 400) - in particular `media_kwargs` is NOT client-settable.
- Limits: `max_questions`, `max_images`, `max_videos`, decoded image/video byte caps; images larger than
  `max_pixels` are downscaled (aspect kept), videos sampled at `video_fps` and capped to `max_frames`
  (evenly subsampled).

Response 200 (backwards compatible + `timing`):
```json
{"model": "clef-flash",
 "answers": {"department": {"type": "choice", "choice": "technical", "confidence": 0.96,
                            "probabilities": {"billing": 0.04, "technical": 0.96}},
             "urgency": {"type": "score", "score": 1.78, "confidence": 0.86,
                         "legend": {"0": "Can wait", "1": "This week", "2": "Today"},
                         "probabilities": {"0": 0.07, "1": 0.07, "2": 0.86}},
             "outage": {"type": "noul", "noul": 0.84}},
 "usage": {"input_tokens": 300, "output_tokens": 0},
 "timing": {"queue_ms": 1.2, "forward_ms": 160.3, "total_ms": 171.0, "batch_size": 1}}
```
(answer objects are exactly what the model's `systemone_answer()` returns.)

### `POST /v1/batch`
Request `{"batch": [<systemone request>, ...]}` (1..`max_batch`; media allowed - media records run individually).
Response `{"batch_ms": float, "results": [<systemone response>, ...]}` in input order.
Validation errors name the index: `"batch[3].questions.urgency: criteria must be a non-empty list"`.

### `GET /health`
200 when status is ready or warming, 503 otherwise:
```json
{"ready": true, "status": "ready", "version": "2.0.0", "device": "cuda", "dtype": "bfloat16",
 "model_path": "...", "torch": "2.11.0+rocm7.2", "transformers": "5.10.2",
 "gpu": {"available": true, "name": "AMD Radeon RX 7900 XTX", "vram_total_gb": 25.7,
         "vram_allocated_gb": 19.1, "vram_reserved_gb": 19.8},
 "load_seconds": 45.4, "uptime_s": 3600, "error": null, "limits": {"...": "Config.public_limits()"}}
```
`GET /livez` -> 200 `{"ok": true}` always.

### `GET /v1/stats`
```json
{"uptime_s": 3600, "status": "ready", "total": 412, "errors": 3, "in_flight": 1, "queue_depth": 0,
 "window_s": 300, "rps": 0.8,
 "latency_ms": {"p50": 162, "p95": 222, "p99": 410, "max": 3100,
                "hist": {"edges": [25, 50, 100, 150, 200, 300, 500, 1000, 2000, 5000],
                         "counts": [0, 0, 2, 80, 40, 9, 3, 1, 1, 0, 0]}},
 "forward": {"count": 400, "avg_batch": 1.6, "avg_ms": 150.2, "padding_ratio": 0.12},
 "tokens": {"in_total": 123456, "avg_in": 300},
 "gpu": {"vram_allocated_gb": 19.6, "vram_reserved_gb": 21.0, "vram_total_gb": 25.7}}
```
(`hist.counts` has len(edges)+1 buckets: `<edges[0]`, ..., `>=edges[-1]`. Latency over the window.)

### `GET /v1/log?limit=100&since=<id>`
`{"entries": [{"id": 17, "ts": 1759480000.12, "endpoint": "/v1/systemone", "status": 200, "ms": 171.0,
"input_tokens": 300, "n_records": 1, "n_questions": 3, "media": {"images": 0, "videos": 0},
"error": null, "state_preview": null}]}` newest last. `state_preview` only when `CLEF_LOG_STATE=1`.

### `GET /v1/events` (SSE)
`event: log` + `data: <log entry JSON>` per request; `event: stats` + `data: <stats JSON>` every 2 s.

### Errors
Always `{"detail": "<message>", "request_id": "<hex>"}`; header `X-Request-ID` on every response.
400 invalid input, 401 auth, 413 too large, 503 not ready / OOM, 500 internal (generic message; full traceback
only in the server log).
