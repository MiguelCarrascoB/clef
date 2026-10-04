# clef v3 - architecture & API contract

This file is the contract between the server modules, the backends, the CLI, the web console, the clients and the
tests. Change it first, then the code.

## Package layout

`pip install` / `uv tool install` installs two packages from `src/` and one console script, `clef`.

| Module | Owns |
| --- | --- |
| `clef_server/__init__.py` | `__version__`; calls `backend.prepare_environment()` once (torch-free, before any torch import) |
| `clef_server/config.py` | `Config` dataclass from `CLEF_*` env vars, `VERSION`, `MODEL_REPO`/`MODEL_REVISION`, `parse_api_keys` |
| `clef_server/paths.py` | state dir, log/pid files, classifiers dir, `resolve_model_path(cfg)` (never downloads) |
| `clef_server/backend.py` | **the only place with device-specific code**: detection, dtype, sync, OOM, memory, telemetry, preflight, quantization config (bitsandbytes / torchao), device memory cap, pinned host copies (`HostCopier`) |
| `clef_server/offload.py` | CPU offload: `plan_layers` (which decoder layers stay on the host) and `LayerStreamer` (async copy-in per forward); pure logic plus a `copier` supplied by `backend.py` |
| `clef_server/engine.py` | background model load (`load_model`: standard, quantized or offloaded), single GPU worker thread, micro-batching, length bucketing, warmup |
| `clef_server/schemas.py` | pydantic v2 SystemOne request/response models + strict validation |
| `clef_server/classify.py` | classification API models + translation classify/score <-> SystemOne records/answers |
| `clef_server/classifiers.py` | saved-classifier store (JSON files under `<state_dir>/classifiers/`) |
| `clef_server/security.py` | multi-key auth, per-key rate limiting, CORS setup |
| `clef_server/media.py` | data-URL / (optional) URL fetching with SSRF guard + caps, image/video decode |
| `clef_server/stats.py` | request log ring buffer, latency histogram, counters, forward stats, **time series ring buffer** |
| `clef_server/appctx.py` | `AppContext`: what a feature module gets from `create_app()` (config, stats, engine, auth / rate-limit dependencies, inference helpers, lifecycle hooks) |
| `clef_server/main.py` | FastAPI app factory `create_app(cfg, engine)`, core routes, middleware, error mapping, `FEATURES` (feature modules mounted on the context), `run(cfg)` |
| `clef_server/compat.py`, `compat_openai.py`, `compat_hf.py`, `compat_schema.py`, `compat_errors.py` | OpenAI (`/v1/models`, `/v1/chat/completions`) and Hugging Face zero-shot (`/hf/models/{id}`) routes; JSON-schema -> questions translation; client-shaped errors |
| `clef_server/evaluation.py` | metric math (`compute_metrics`), `POST /v1/evaluate` and `/v1/evaluate/metrics`, the `evaluate` job kind |
| `clef_server/jobs.py`, `clef_server/webhooks.py` | async jobs (SQLite store, FIFO runner, `/v1/jobs*` routes, built-in kinds) and signed, SSRF-guarded webhook delivery |
| `clef_server/cli.py` | the `clef` command (serve/stop/status/logs/doctor/download/bench/open/version); `download` is a pinned, resumable `snapshot_download` with a disk-space check |
| `clef_server/doctor.py` | environment checks per backend, including a recommended memory setting; `main(argv) -> int` |
| `clef_server/httpbench.py` | the async HTTP load test (`clef bench`; `bench/http_bench.py` is a shim) |
| `clef_server/static/` | Clef Console (zero-build Preact + htm SPA, vendored uPlot; screens in `screens/`) served at `/` |
| `clef_client/` | sync + async Python client (`httpx`), typed classify results |
| `clients/js/` | dependency-free ESM JS/TS client (`fetch`), Node 18+ and browsers |

Invariants (unchanged from v2):
- ONE uvicorn worker per process (one model), ONE thread ever touches the GPU (the engine worker).
- Python `>=3.10` (the WSL venv is 3.10): no `typing.Self`, `tomllib`, `ExceptionGroup` or other 3.11+ features.
- Never edit files in the model directory; wrap or re-implement in the engine.

## Configuration (`config.py`)

All settings come from env vars. The CLI flags of `clef serve` set the same env vars.

| Env | Default | Meaning |
| --- | --- | --- |
| `CLEF_MODEL_PATH` | unset | release dir. Unset: `~/models/clef-flash` if it holds a release, else the pinned HF cache snapshot |
| `CLEF_MODEL_REPO` / `CLEF_MODEL_REVISION` | `Cloudflare/clef-flash` / pinned sha | what `clef download` fetches and the cache lookup uses |
| `CLEF_DEVICE` | `auto` | `auto`, `cuda`, `rocm`, `mps`, `cpu` (`cuda:1` / `rocm:1` select an index) |
| `CLEF_DTYPE` | `auto` | `auto`, `bfloat16`, `float16`, `float32` (auto: see backend dtype rules) |
| `CLEF_QUANT` | `none` | `none`, `int8`, `nf4`. int8: bitsandbytes on CUDA, torchao on ROCm / MPS / CPU. nf4: bitsandbytes only (CUDA, and ROCm where it is not recommended); refused elsewhere with a clear error |
| `CLEF_QUANT_BACKEND` | `auto` | `auto`, `bnb`, `torchao`: the library behind `CLEF_QUANT` (auto: CUDA keeps bitsandbytes, everything else torchao; nf4 is bitsandbytes only) |
| `CLEF_OFFLOAD` | `none` | `none` or `cpu`: keep the embeddings and the layers that do not fit under the device budget in host RAM and stream them in per forward (CUDA and ROCm; a warning and no effect on MPS / CPU, which share one memory pool) |
| `CLEF_MAX_DEVICE_MEMORY_GB` | `0` | device memory this process may allocate, applied as a hard allocator limit; `0` = no cap (offload then plans from free memory). 2.5 GB of it is reserved for activations when planning the split |
| `CLEF_PREFLIGHT` | `1` | memory check before loading |
| `CLEF_TELEMETRY` | `1` | GPU telemetry via pynvml / amdsmi / rocm-smi when available |
| `CLEF_HOST` / `CLEF_PORT` | `127.0.0.1` / `8910` | bind address. A non-loopback host without keys logs a loud warning |
| `CLEF_API_KEY` | unset | one key, named `default` |
| `CLEF_API_KEYS` | unset | `name:key,name2:key2` or a path to a file with one `name:key` per line (`#` comments) |
| `CLEF_RATE_LIMIT` | `0` | requests per minute per key (per client IP when auth is off); 0 = off |
| `CLEF_CORS_ORIGINS` | empty | comma-separated allowed origins (`*` allowed); empty = CORS off |
| `CLEF_STATE_DIR` | per OS | Linux `$XDG_STATE_HOME/clef` (`~/.local/state/clef`), macOS `~/Library/Application Support/clef`, Windows `%LOCALAPPDATA%\clef`. Legacy alias `CLEF_LOG_DIR` |
| `CLEF_MAX_LABELS` | `64` | labels per classify request / saved classifier |
| `CLEF_MAX_CLASSIFIERS` | `1000` | saved classifiers |
| `CLEF_CLASSIFY_THRESHOLD` | `0.5` | default multi-label threshold (also used by `/v1/chat/completions` array properties) |
| `CLEF_MAX_EVAL_ROWS` | `500` | rows per `POST /v1/evaluate` request |
| `CLEF_MAX_JOB_EVAL_ROWS` | `100000` | rows per `evaluate` job and per `POST /v1/evaluate/metrics` request |
| `CLEF_MAX_JOB_ITEMS` | `100000` | items per `classify` / `score` / `systemone` job |
| `CLEF_MAX_JOBS` | `100` | queued + running jobs (more: 409 with `Retry-After`) |
| `CLEF_JOB_TTL_HOURS` | `168` | finished jobs and their rows are purged this long after they finish; `0` keeps them forever |
| `CLEF_JOB_ENGINE_WAIT_S` | `600` | how long a job waits for an unavailable engine before failing |
| `CLEF_WEBHOOK_ALLOW` | empty | hosts (`hooks.example.com`, `*.corp.example.net`), IPs and CIDRs webhooks may be delivered to; empty = webhooks off |
| `CLEF_WEBHOOK_TIMEOUT_S` | `10` | total timeout of one delivery attempt |
| `CLEF_WEBHOOK_ATTEMPTS` | `5` | delivery attempts for final events (`job.progress` is never retried) |
| `CLEF_SAMPLE_INTERVAL_S` | `2` | gauge sampling period for the time series (memory, telemetry, queue) |
| v2 knobs | unchanged | `CLEF_MAX_TOKENS`, `CLEF_MAX_BODY_MB`, `CLEF_MAX_IMAGES/VIDEOS/PIXELS/FRAMES`, `CLEF_VIDEO_FPS`, `CLEF_MAX_BATCH`, `CLEF_MAX_QUESTIONS`, `CLEF_ALLOW_URL_FETCH`, `CLEF_URL_FETCH_MAX_MB`, `CLEF_MAX_MICROBATCH`, `CLEF_BATCH_WINDOW_MS`, `CLEF_BUCKETS`, `CLEF_PAD_MULTIPLE` (64, measured on ROCm only), `CLEF_PAD_TO_BUCKET`, `CLEF_WARMUP`, `CLEF_LOG_STATE`, `CLEF_LOG_BUFFER` |

`Config` validates enums (device, dtype, quant, quant backend, offload), `CLEF_MAX_DEVICE_MEMORY_GB >= 0`, the threshold range and `CLEF_RATE_LIMIT >= 0` in `__post_init__` (ValueError on a bad value) and exposes `api_keys() -> dict[name, key]`,
`auth_required`, `is_loopback`, `public_limits()`.

## Backend interface (`backend.py`)

```python
class BackendError(RuntimeError): ...       # unavailable explicit device, unsupported quant/dtype combination
class PreflightError(RuntimeError): ...     # not enough free memory for the chosen dtype/quant (clear message)

def prepare_environment(environ: MutableMapping[str, str] = os.environ) -> list[str]:
    """Torch-free. setdefault() the per-platform env BEFORE torch is imported. Returns the names it set.
    - torch build is ROCm (importlib.metadata version of torch contains "+rocm"):
        TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1, and on WSL (/dev/dxg exists) HSA_ENABLE_DXG_DETECTION=1;
        PYTORCH_TUNABLEOP_* only when CLEF_TUNABLEOP=1. HSA_* / AOTriton are NEVER set on other backends.
    - macOS: PYTORCH_ENABLE_MPS_FALLBACK=1.
    - everywhere: HF_HUB_OFFLINE=1 is NOT set here (the CLI download needs the network; the engine passes
      local paths so it never downloads)."""

def detect(requested: str = "auto") -> Backend:
    """auto: cuda (NVIDIA: torch.cuda.is_available() and torch.version.hip is None) > rocm (torch.version.hip)
    > mps (torch.backends.mps.is_available()) > cpu. An explicit request that is unavailable raises BackendError
    with a hint (wrong torch build, driver missing, HSA_ENABLE_DXG_DETECTION on WSL, ...)."""

@dataclass
class Backend:
    name: str                     # "cuda" | "rocm" | "mps" | "cpu"
    device: torch.device          # cuda:N for cuda AND rocm, mps, cpu
    def synchronize(self) -> None
    def empty_cache(self) -> None
    def is_oom(self, exc: BaseException) -> bool       # torch.OutOfMemoryError / cuda OOM / MPS "out of memory"
    def device_name(self) -> str                       # "NVIDIA GeForce RTX 4090", "Apple M3 Max", CPU brand
    def memory(self) -> dict                           # see below; never raises
    def telemetry(self) -> dict                        # see below; never raises, "unavailable" elsewhere
    def resolve_dtype(self, requested: str) -> tuple[torch.dtype, str | None]   # (dtype, warning or None)
    def versions(self) -> dict                         # {"torch","cuda","hip","driver","macos"} (None if n/a)
    def fast_path(self) -> dict                        # {"causal_conv1d": bool, "fla": bool, "expected": bool}

def preflight(backend: Backend, dtype: torch.dtype, quant: str, weights_gb: float,
              cap_gb: float = 0.0, offload: str = "none") -> list[str]:
    """Raise PreflightError when free memory (or the CLEF_MAX_DEVICE_MEMORY_GB cap) is clearly below what the load
    needs; return warnings when close. With offload=cpu on cuda / rocm the engine plans the split itself and only
    the cap is checked here."""

def quant_method(backend: Backend, quant: str, choice: str = "auto") -> str:
    """"bnb" or "torchao". auto: nf4 -> bnb; int8 -> bnb on cuda, torchao on rocm / mps / cpu.
    An explicit choice wins (nf4 + torchao is a BackendError)."""

def quantization_config(backend: Backend, quant: str, dtype: torch.dtype, choice: str = "auto") -> Any | None:
    """None for "none". Else a transformers config: BitsAndBytesConfig (int8 / nf4; cuda and rocm only) or
    TorchAoConfig(Int8WeightOnlyConfig) with the embeddings and the vision tower left in bf16. BackendError with
    an install hint when the library is missing or the backend cannot run it."""

def recommend_memory_setting(backend: Backend, weights_gb: float = 19.0) -> str | None:
    """One line of advice from the detected memory (printed by `clef doctor`); None when bf16 fits."""
```

Offload support on `Backend`: `has_discrete_memory` (true on cuda / rocm), `pin_host(tensor)`, `host_copier()` (a
`HostCopier` that copies pinned host tensors on a side stream) and `apply_memory_cap(cap_gb)`
(`torch.cuda.set_per_process_memory_fraction`; returns whether a limit was set). `ACTIVATION_RESERVE_GB = 2.5`.

Dtype rules (`resolve_dtype("auto")`): bf16 on cuda (if `torch.cuda.is_bf16_supported()`, else fp16 + warning),
rocm, and mps on macOS >= 14 (fp16 + warning before 14); cpu: bf16 when the CPU supports it, else fp32. An explicit
dtype the device cannot run falls back the same way with a warning. Any fallback must be validated for probability
parity against bf16 (`bench/parity.py`, results in `docs/hardware-results.md`).

`memory()`: `{"kind": "vram"|"unified"|"system", "total_gb", "allocated_gb", "reserved_gb", "free_gb"}` (floats or
None). cuda/rocm: `torch.cuda.mem_get_info/memory_allocated/memory_reserved`; mps:
`torch.mps.current_allocated_memory`, `driver_allocated_memory`, `recommended_max_memory`; cpu: process RSS and
system memory (psutil if installed, else None).

`telemetry()`: `{"available": bool, "source": "nvml"|"amdsmi"|"rocm-smi"|None, "util_pct", "temp_c", "power_w",
"mem_used_gb", "mem_total_gb"}`; values None when unknown. Imports `pynvml` / `amdsmi` lazily; failures are
logged once at debug level and turn the result into `{"available": false, ...}`. rocm-smi (subprocess) results are
cached for >= 2 s. Under WSL (measured, 7900 XTX) amdsmi gives utilization + temperature only; its VRAM usage is impossible (> total) and is dropped.

## Engine interface (`engine.py`)

```python
Record = dict  # sanitized: {"model": str, "state": Any, "questions": dict,
               #             "images": list[PIL.Image.Image] | None, "videos": list[np.ndarray] | None}

class EngineNotReady(RuntimeError): ...          # -> HTTP 503
class InputTooLarge(ValueError): ...             # -> HTTP 413 (e.g. schema alone exceeds max_tokens)
class GpuOutOfMemory(RuntimeError): ...          # -> HTTP 503 (after backend.empty_cache())

class Engine:
    def __init__(self, cfg: Config, stats: Stats, loader: Callable | None = None,
                 backend: Backend | None = None): ...
        # loader: test injection; default resolves the model dir (paths.resolve_model_path) and imports
        #   joint_schema_model from it. backend: test injection; default backend.detect(cfg.device),
        #   done on the worker thread at load time (so a BackendError becomes status "error", not a crash).
    status: str             # "loading" | "warming" | "ready" | "error"
    error: str | None
    load_seconds: float | None
    def start(self) -> None
    async def decide(self, records: list[Record]) -> list[Result]   # results in input order
    def info(self) -> dict
    def sample(self) -> dict    # cheap gauges for the time series, see below; never raises; any thread
    def shutdown(self) -> None
```

`info()` (also the bulk of `/health`):
```json
{"backend": "rocm", "device": "cuda:0", "dtype": "bfloat16", "quant": "none",
 "model_path": "/home/<user>/models/clef-flash", "torch": "2.11.0+rocm7.2", "transformers": "5.10.2",
 "gpu": {"available": true, "name": "AMD Radeon RX 7900 XTX", "memory_kind": "vram", "vram_total_gb": 23.94,
         "vram_allocated_gb": 17.79, "vram_reserved_gb": 18.17, "vram_free_gb": 5.4},
 "telemetry": {"available": false, "source": null, "util_pct": null, "temp_c": null, "power_w": null,
               "mem_used_gb": null, "mem_total_gb": null},
 "fast_path": {"causal_conv1d": false, "fla": true, "expected": true},
 "versions": {"torch": "2.11.0+rocm7.2", "cuda": null, "hip": "7.2", "driver": null, "macos": null},
 "warnings": []}
```
Before the backend is detected (`status == "loading"` early) `backend` is `null` and the `gpu` block has
`available: false`. `gpu` keeps the v2 key names for compatibility; on mps `memory_kind` is `"unified"` and
`vram_total_gb` is `recommended_max_memory`; on cpu `available` is false. `info()` must stay fast (< 5 ms): it
returns the cached telemetry from the last `sample()`.

`sample()` -> `{"mem_used_gb": float|None, "mem_total_gb": float|None, "gpu_util_pct": float|None,
"gpu_temp_c": float|None, "gpu_power_w": float|None}` (memory = allocated on cuda/rocm/mps, RSS on cpu).

Rules (v2 rules kept):
- Exactly ONE thread ever runs forwards (the worker). `decide()` enqueues and awaits futures.
- Text-only records are coalesced for up to `batch_window_ms` / `max_microbatch`, grouped by length bucket, one
  forward per group, right-padded to the next `pad_multiple`. Media records run alone, unpadded.
- Warmup: one forward per bucket (<= 1024) at batch sizes {1, max_microbatch}; status "warming" meanwhile.
- No `torch.cuda.*` outside `backend.py`. OOM: `backend.is_oom(exc)` -> `backend.empty_cache()` ->
  `GpuOutOfMemory`; the worker keeps running. Timing uses `backend.synchronize()`.
- Load order on the worker: detect backend -> resolve dtype -> resolve model path -> preflight (if enabled) ->
  `load_model(...)`: apply the device cap (`CLEF_MAX_DEVICE_MEMORY_GB`), build the quantization config, then either
  the standard path (`load_release_model(path, device=..., dtype=..., quantization_config=...)`, kwarg only when
  quantizing) or, with `CLEF_OFFLOAD=cpu` on cuda / rocm, an offloaded load. Each failure becomes status "error"
  with a one-line actionable message.
- Offloaded load (`offload.py`): token and output embeddings (~4 GB) stay on the host and are gathered on the CPU;
  decoder layers beyond the device budget (cap or free memory, minus `ACTIVATION_RESERVE_GB`) are held in pinned
  host memory, spread evenly through the stack, and copied in on a side stream one or two layers ahead of the
  compute; the vision tower, final norm, rotary tables and the joint head stay on the device. The weights are the
  same tensors, so probabilities are identical to a normal run. After a failed forward the streamer puts the layers
  back on the host. Measured numbers: [memory.md](memory.md).
- The engine reports `stats.record_forward(...)` and `stats.set_queue_depth(n)` as in v2.

## Stats interface (`stats.py`)

```python
class Stats:
    def __init__(self, buffer: int = 1000, retention_s: int = 3600): ...
    def record_request(self, entry: dict) -> None   # entry = /v1/log item minus id/ts; may include "key"
    def record_forward(self, n_records: int, n_tokens: int, padded_tokens: int, ms: float) -> None
    def set_queue_depth(self, n: int) -> None
    def sample(self, gauges: dict) -> None          # engine.sample() + {"queue_depth"}; called every
                                                    # cfg.sample_interval_s by main; unknown keys ignored
    def in_flight(self) -> ContextManager
    def snapshot(self, window_s: int = 300) -> dict   # the /v1/stats body (minus status/gpu, added by main)
    def timeseries(self, window_s: int = 300, step_s: int | None = None) -> dict
    def latest_point(self, step_s: int = 5) -> dict   # the newest (current, partial) step as one point
    def log(self, limit: int = 100, since: int | None = None) -> list[dict]
    async def subscribe(self) -> AsyncIterator[dict]
```

Internally the time series is a ring of 1-second buckets covering `retention_s` (3600). Each bucket holds:
request count, error count, successful latencies (capped at 1000 per second), forward count/records/tokens/padded,
max queue depth seen (from `set_queue_depth` and `sample`), and the last gauge sample. Thread-safe (engine thread
+ event loop + sampler).

## Feature modules (`appctx.py`)

New API surface is added as a **feature module** instead of growing `main.py`. A module exposes
`router(ctx: AppContext) -> APIRouter` and is listed in `main.FEATURES` (currently `["evaluation", "compat",
"jobs"]`, in that order: evaluation registers its job kind before jobs reads the registry). `create_app()` builds the
context, imports each module with `importlib` and includes its router.

`AppContext` fields:

| Field | What it is |
| --- | --- |
| `cfg`, `stats`, `engine`, `store` | the `Config`, `Stats`, `Engine` and `ClassifierStore` of this app |
| `auth`, `limited` | FastAPI dependency lists: `auth` goes on every router that needs a key, `limited` adds the rate limit on inference routes |
| `errors` | the shared OpenAPI `responses` for the standard error statuses |
| `infer(request, reqs, batch, label="batch")` | the request path: limits, media load, one `engine.decide()`, and the request log / stats entry |
| `decide(reqs, batch=True, label="batch")` | the same without a request, for background work (jobs): limits, media and one `engine.decide()`; not logged or counted in stats |
| `map_exception(exc)`, `api_error(status, detail, headers=None)`, `request_id(request)` | the shared error mapping (raise the result `from exc`) |
| `run_classify`, `run_classify_batch`, `run_score` | the classification routes' implementations, reusable by other features |
| `on_startup`, `on_shutdown` | lists of async callables run inside the app lifespan after the engine starts / before it stops (jobs start and stop their runner here) |
| `extra` | a dict features use to publish objects to each other; `extra["job_kinds"]` is the job-kind registry |

Rules for a feature: inference goes through `ctx.infer` or `ctx.decide` (never straight to the engine), routes carry
`ctx.auth` (and `ctx.limited` when they run the model), and all request / response models are pydantic so the route
appears in `openapi.json`.

## HTTP API

| Route | Purpose | Section |
| --- | --- | --- |
| `GET /health`, `GET /livez`, `GET /schema-example` | status, liveness, copy-paste bodies (open) | [below](#get-health) |
| `POST /v1/systemone`, `POST /v1/batch` | the original typed questions API | [below](#post-v1systemone-post-v1batch) |
| `POST /v1/classify`, `/v1/classify/batch`, `/v1/score`; `PUT|GET|DELETE /v1/classifiers/{name}`, `GET /v1/classifiers`, `POST /v1/classifiers/{name}[/batch]` | classification | [below](#classification-api-classifypy-classifierspy) |
| `GET /v1/models[/{id}]`, `POST /v1/chat/completions` | OpenAI-compatible structured outputs | [Compat](#openai-and-hugging-face-compat-compatpy) |
| `POST /hf/models/{id}` (and the hidden `POST /models/{id}`) | Hugging Face zero-shot classification | [Compat](#openai-and-hugging-face-compat-compatpy) |
| `POST /v1/evaluate`, `POST /v1/evaluate/metrics` | accuracy, F1, confusion matrix, calibration | [Evaluation](#evaluation-evaluationpy) |
| `POST /v1/jobs`, `GET /v1/jobs`, `GET /v1/jobs/{id}`, `GET /v1/jobs/{id}/results`, `POST /v1/jobs/{id}/cancel`, `DELETE /v1/jobs/{id}` | async jobs | [Jobs](#async-jobs-and-webhooks-jobspy-webhookspy) |
| `GET /v1/stats`, `/v1/stats/timeseries`, `/v1/log`, `/v1/events` | observability (console) | below |

### Auth, rate limit, CORS (`security.py`)

- Keys: `CLEF_API_KEY` and/or `CLEF_API_KEYS` (see Configuration). When any key exists every `/v1/*` route, and the
  Hugging Face routes (`/hf/models/*`, `/models/*`), need `X-API-Key: <key>` or `Authorization: Bearer <key>` (401 +
  `WWW-Authenticate: Bearer` otherwise; the OpenAI and HF routes render that in their own error shape). For SSE
  (EventSource cannot set headers) `/v1/events?key=<key>` is also accepted. Comparison is constant-time over all
  keys. The matching key's **name** (never the key) is stored on `request.state.key_name` and written to the request
  log (`"key"`) and to `/v1/stats` (`by_key`). Auth off -> `key` is `null`.
- Open routes: `/`, static files, `/health`, `/livez`, `/schema-example`, `/docs`, `/openapi.json`.
- Rate limit (`CLEF_RATE_LIMIT=N`): sliding 60 s window per key name (per client IP when auth is off), counted on
  the POST inference routes only (`/v1/systemone`, `/v1/batch`, `/v1/classify`, `/v1/classify/batch`, `/v1/score`,
  `POST /v1/classifiers/{name}[/batch]`, `/v1/chat/completions`, `/hf/models/*`, `/v1/evaluate`, `POST /v1/jobs`).
  `/v1/evaluate/metrics` and the job read / cancel / delete routes are not counted. Over the limit: `429 {"detail": "rate limit exceeded (N/min)",
  "request_id"}` + `Retry-After: <int seconds>`. Rejected requests are logged (status 429) but not counted.
- CORS: off by default. `CLEF_CORS_ORIGINS=https://a.example,https://b.example` (or `*`) installs Starlette's
  `CORSMiddleware` with methods GET/POST/PUT/DELETE/OPTIONS, headers `*` (X-API-Key, Authorization, Content-Type),
  exposed headers `X-Request-ID`, `Retry-After`; `allow_credentials` false.
- Binding a non-loopback `CLEF_HOST` without any key logs a multi-line WARNING at startup (`main.run`).

Body limit: Content-Length (or streamed bytes) > `max_body_mb` -> 413.

### `POST /v1/systemone`, `POST /v1/batch`

Unchanged from v2 (request, response, validation, error paths). See git history of this file for the full text;
summary:
```json
{"model": "clef-flash", "state": "<string or any JSON>",
 "questions": {"<id>": {"type": "choice", "instructions": "optional", "criteria": {"opt": "desc"}},
               "<id>": {"type": "score", "instructions": "optional", "criteria": ["level0", "level1"]},
               "<id>": {"type": "noul", "instructions": "optional", "criteria": {"true": "desc", "false": "desc"}}},
 "images": ["data:image/...;base64,..."], "videos": ["data:video/mp4;base64,..."]}
```
-> `{"model", "answers": {"<id>": <systemone_answer>}, "usage": {"input_tokens", "output_tokens": 0},
"timing": {"queue_ms", "forward_ms", "total_ms", "batch_size"}}`. Answers: choice
`{"type","choice","confidence","probabilities": {opt: p}}`; score `{"type","score","confidence","legend":
{"0": level0, ...},"probabilities": {"0": p, ...}}`; noul `{"type","noul": p_true}`.
`/v1/batch`: `{"batch": [...]}` -> `{"batch_ms", "results": [...]}`, errors name the index (`batch[3].questions...`).

### Classification API (`classify.py`, `classifiers.py`)

The classification routes are a thin, stable layer over the SAME engine: each input is translated into exactly one
SystemOne record (the same record `/v1/systemone` would build), so probabilities are identical to the equivalent
SystemOne call (verified live within 1e-3). All of them are POST inference routes (logged, rate limited, counted in
stats) and return `usage`, `timing` and `request_id` like the existing routes.

Labels: either a list of unique non-empty strings (`["billing", "technical"]`) or an object label -> description
(`{"billing": "Payments or invoices"}`); 2..`max_labels` labels for single-label, 1..`max_labels` for multi-label.

**Translation (normative - the live parity test rebuilds these exact records):**
- single-label -> `questions = {"label": {"type": "choice", "criteria": C, "instructions": I}}` where
  `C = {l: l for l in labels}` for a list, the object as given for a dict, and `"instructions"` is present only
  when the request has `instructions`.
- multi-label -> one noul question per label, question id = the label:
  `{"type": "noul", "instructions": P + 'Does the label "<label>" apply?'}` where `P = instructions + " "` when
  given, else `""`; plus `"criteria": {"true": desc}` when labels is a dict and `desc != label`.
- score -> `questions = {"score": {"type": "score", "criteria": levels, "instructions": I}}` (instructions only when
  given).
- `state` = `input` verbatim (string or any JSON); `images`/`videos` pass through `load_media` exactly as on
  `/v1/systemone`; `model` defaults to `"clef-flash"`.

#### `POST /v1/classify`
```json
{"input": "Checkout is down", "labels": ["billing", "technical"], "instructions": "optional question",
 "multi_label": false, "threshold": 0.5, "images": [], "videos": [], "model": "clef-flash"}
```
Single-label response:
```json
{"model": "clef-flash", "multi_label": false, "label": "technical", "confidence": 0.96,
 "scores": {"billing": 0.04, "technical": 0.96},
 "usage": {"input_tokens": 120, "output_tokens": 0},
 "timing": {"queue_ms": 0.4, "forward_ms": 141.2, "total_ms": 150.3, "batch_size": 1}, "request_id": "<hex>"}
```
(`label` = the choice answer, `confidence` = its probability, `scores` = all options' probabilities.)
Multi-label response: `"multi_label": true`, `"labels": [...]` = every label with score >= `threshold`, sorted by score
desc (may be empty), `"scores": {label: P(true)}`, `"threshold": 0.5`; no `label`/`confidence`. `threshold` is
allowed only with `multi_label: true` (default `CLEF_CLASSIFY_THRESHOLD`), within [0, 1].

#### `POST /v1/classify/batch`
`{"inputs": [<any>, ...] (1..max_batch), "labels": ..., "instructions", "multi_label", "threshold", "model"}` (no media
in batch). Runs through the engine's micro-batching in one `decide()` call ->
`{"batch_ms": float, "results": [<classify response minus request_id>, ...], "request_id": "<hex>"}` in input order.
Errors name the index: `inputs[3]: ...`.

#### `POST /v1/score`
`{"input": ..., "levels": ["low", "medium", "high"], "instructions": "optional", "images", "videos", "model"}`
(2..max_labels unique non-empty levels) ->
```json
{"model": "clef-flash", "score": 1.78, "level": "high", "level_index": 2, "confidence": 0.86,
 "distribution": {"low": 0.07, "medium": 0.07, "high": 0.86}, "usage": {}, "timing": {}, "request_id": "<hex>"}
```
(`score` = expected level index, `level`/`level_index` = most likely level, `distribution` keyed by level text.)

#### Saved classifiers
- `PUT /v1/classifiers/{name}` body
  `{"kind": "classify" | "score", "labels": ... (classify), "levels": [...] (score), "instructions": str?,
  "multi_label": false, "threshold": float?, "description": str?}` -> 200 the stored definition +
  `{"name", "created_at", "updated_at"}` (ISO-8601 UTC; `created_at` kept on overwrite). Validation as above.
- `GET /v1/classifiers` -> `{"classifiers": [<definition>, ...]}` sorted by name.
- `GET /v1/classifiers/{name}` -> the definition, 404 if missing.
- `DELETE /v1/classifiers/{name}` -> `{"deleted": "<name>"}`, 404 if missing.
- `POST /v1/classifiers/{name}` `{"input": ..., "images"?, "videos"?, "threshold"?}` -> the classify (or score)
  response + `"classifier": "<name>"`.
- `POST /v1/classifiers/{name}/batch` `{"inputs": [...]}` -> the batch response + `"classifier"`.
- Names: `^[a-z0-9][a-z0-9_-]{0,63}$` (400 otherwise). Files: `<state_dir>/classifiers/<name>.json`, path built only
  from a validated name, written atomically (temp file + `os.replace`), at most `max_classifiers` (409 beyond).
  Unreadable/corrupt files are skipped in the list and logged.

### OpenAI and Hugging Face compat (`compat*.py`)

Guide: [openai-compat.md](openai-compat.md). Both dialects translate to ONE SystemOne record per request and go
through `ctx.infer`, so keys, rate limit, log, stats and limits are shared and the probabilities equal the
equivalent `/v1/systemone` call.

- `GET /v1/models` lists `clef-flash`; `GET /v1/models/{id}` returns it, any other id is 404 `model_not_found`.
- `POST /v1/chat/completions`: the schema comes from `response_format` (`json_schema`) or from exactly one forced
  tool. Each top-level property becomes one question (`compat_schema.py`): string enum -> `choice`; boolean ->
  `noul` (true when P >= 0.5); integer enum, bounded integer (2..64 values) or string enum with `x-clef-ordinal` ->
  `score`; array of string enums -> one `noul` per option (options with P >= `CLEF_CLASSIFY_THRESHOLD`). Anything
  else (free text, numbers, nested objects) is a 400 whose `param` names the property. `messages` become the
  `state` (a lone user message verbatim, otherwise a readable transcript); `image_url` / `video_url` parts become
  `images` / `videos`.
- Response: OpenAI `chat.completion` with the argmax as JSON in `message.content` (or a `tool_calls` entry),
  `logprobs` carrying `log p` (`top_logprobs` 0..64, floor `-9999`), `usage` (`completion_tokens` always 0) and a
  non-standard top-level `clef` object with every probability, `timing` and `request_id`. `stream: true` returns
  valid SSE (role chunk, one content chunk, finish chunk, optional usage chunk, `[DONE]`). `n > 1`, `text` /
  `json_object` response formats and requests without a schema are 400s (`missing_schema` when no schema was given).
- `POST /hf/models/{model_id:path}`: Inference API zero-shot body `{inputs, parameters: {candidate_labels,
  multi_label, hypothesis_template}}`. Response `{sequence, labels, scores}` sorted by score, or
  `[{label, score}]` when the User-Agent is `hf_hub/1.x` (force with `?format=classic|list`); `inputs` may be a
  list (one micro-batched pass).
- Errors on these routes use the client's shape (`compat_errors.py`): OpenAI `{"error": {message, type, param,
  code}}`, Hugging Face `{"error": "message"}`; status codes match the rest of the API.

### Evaluation (`evaluation.py`)

Guide: [evaluation.md](evaluation.md). `compute_metrics` is the single implementation of the metric math; the console
and the job kind use it too.

- `POST /v1/evaluate` `{labels | classifier, instructions?, multi_label?, threshold?, bins?, rows: [{input, gold}],
  include_predictions?}`: at most `CLEF_MAX_EVAL_ROWS` rows (400 beyond, pointing at the job kind), classified
  `CLEF_MAX_BATCH` at a time through `ctx.infer`, then scored. A gold label outside the label set is a 400 naming
  the row, before any inference.
- `POST /v1/evaluate/metrics` `{labels, rows: [{gold, scores}], multi_label?, threshold?, bins?,
  include_predictions?}`: no inference, at most `CLEF_MAX_JOB_EVAL_ROWS` rows, not rate limited.
- Response: `n`, `accuracy`, `top2_accuracy`, `macro_f1`, `micro_f1`, `weighted_f1`, `per_label`,
  `confusion_matrix {labels, matrix}`, `calibration {bins, ece, mce, brier, nll, mean_confidence}`,
  `coverage_curve`, `auto_route`, optional `predictions`. Multi-label: `exact_match`, `hamming_loss`, per-label
  `tp fp fn tn`.
- `evaluate` job kind (registered in `ctx.extra["job_kinds"]`): same payload as `POST /v1/evaluate`, up to
  `CLEF_MAX_JOB_EVAL_ROWS` rows, one stored item per row, result = the metrics object. Not resumable (it restarts).

### Async jobs and webhooks (`jobs.py`, `webhooks.py`)

Guide: [jobs.md](jobs.md).

- `POST /v1/jobs` `{kind, payload, webhook?, metadata?}` -> 202 with the job and `Location: /v1/jobs/{id}`. The
  payload is fully validated at submit (400 with a path); an unknown kind lists the available ones. Kinds:
  `classify`, `score`, `systemone` (jobs.py) and `evaluate` (evaluation.py). 409 + `Retry-After` when
  `CLEF_MAX_JOBS` is reached.
- `GET /v1/jobs?status&kind&limit&offset` (newest first), `GET /v1/jobs/{id}` (status, progress, ETA, result summary,
  webhook delivery status), `GET /v1/jobs/{id}/results` (JSON pages with `next_offset`, or a full stream with
  `?format=ndjson|csv`; CSV neutralises cells starting with `=`, `+`, `-`, `@`), `POST /v1/jobs/{id}/cancel`,
  `DELETE /v1/jobs/{id}` (409 while queued or running).
- Storage: SQLite `<state_dir>/jobs.db` (jobs, payloads, result rows, webhook deliveries). Statuses: `queued`,
  `running`, `succeeded`, `failed`, `cancelled`, `interrupted`. One runner, FIFO, one job at a time and ONE
  micro-batch in flight (`CLEF_MAX_MICROBATCH` records through `ctx.decide`), so interactive requests interleave.
  A job interrupted by a stop or crash resumes after its last stored row on the next start. A row the engine rejects
  becomes an `{"index", "error"}` row; the job fails only on systemic errors (engine down for
  `CLEF_JOB_ENGINE_WAIT_S`). Finished jobs are purged `CLEF_JOB_TTL_HOURS` after they finish.
- Ownership: with auth on, a job belongs to the key name that submitted it and other keys get 404 (also in the
  list); jobs created while auth was off stay visible to every key.
- Webhooks: events `job.succeeded`, `job.failed`, `job.cancelled` (default) and opt-in `job.progress` (at most one
  per 10 s, one attempt). Headers `X-Clef-Event`, `X-Clef-Delivery`, and with a secret `X-Clef-Timestamp` and
  `X-Clef-Signature: sha256=` HMAC-SHA256 of `"<timestamp>.<raw body>"`. 2xx = delivered; 408 / 425 / 429 / 5xx and
  network errors retry with exponential backoff up to `CLEF_WEBHOOK_ATTEMPTS`; other statuses and all redirects are
  final. Disabled until `CLEF_WEBHOOK_ALLOW` is set; the allow-list is checked at submit and on every attempt, the
  host is resolved then and the connection is pinned to the checked IP (see [SECURITY.md](../SECURITY.md)).
- Jobs are not counted in `/v1/stats` or the request log (they use `ctx.decide`).
- Custom kinds: `ctx.extra.setdefault("job_kinds", {})[name] = {"validate": fn, "run": async_fn, "resumable": bool}`
  (see [jobs.md](jobs.md#writing-a-job-kind)).

### `GET /health`
200 when status is ready or warming, 503 otherwise. `{"ready", "status", "version": "3.0.0", **engine.info(),
"load_seconds", "uptime_s", "error", "limits": Config.public_limits()}`. `GET /livez` -> 200 `{"ok": true}`.

### `GET /v1/stats?window_s=300`
v2 body plus `"by_key": {"<name>|anonymous": count}` and `"by_endpoint": {"/v1/classify": count, ...}` over the
window, plus `"status"` and `"gpu"` (`vram_allocated_gb`, `vram_reserved_gb`, `vram_total_gb`, `memory_kind`) and
`"telemetry"` (cached engine telemetry) added by main.

### `GET /v1/stats/timeseries?window_s=300&step_s=5`
`window_s` in 10..3600 (console uses 300 / 900 / 3600); `step_s` optional, default `max(1, window_s // 60)`
(5 / 15 / 60 s), must satisfy `1 <= step_s <= window_s` and `window_s / step_s <= 720` (400 otherwise).
Columnar (uPlot-friendly), oldest first, `n = ceil(window_s / step_s)` points aligned to multiples of `step_s`; the
last point is the current, partial step:
```json
{"window_s": 300, "step_s": 5, "now": 1759480000.2,
 "t":             [1759479705, ...],      // step start, unix seconds
 "rps":           [0.8, ...],             // requests completed in the step / step_s (0 when none)
 "p50_ms":        [141.0, null, ...],     // successful requests completed in the step; null when none
 "p95_ms":        [150.2, null, ...],
 "error_rate":    [0.0, null, ...],       // errors / requests; null when no requests
 "avg_batch":     [1.6, null, ...],       // records / forwards; null when no forwards
 "queue_depth":   [0, ...],               // max seen in the step; null before the first sample
 "padding_ratio": [0.12, null, ...],      // 1 - tokens / padded tokens over the step's forwards; null when none
 "mem_used_gb":   [17.8, ...],            // last gauge sample in the step, else null
 "gpu_util_pct":  [null, ...], "gpu_temp_c": [null, ...], "gpu_power_w": [null, ...]}
```

### `GET /v1/log?limit=100&since=<id>`
v2 entries plus `"key": "<name>" | null`. Endpoint values include the new POST routes (actual path, e.g.
`/v1/classifiers/support-triage`).

### `GET /v1/events?step_s=5` (SSE)
`event: log` + `data: <log entry>` per request; `event: stats` + `data: <stats body>` every 2 s. The stats body
carries `"point": stats.latest_point(step_s)` (same fields as one time-series column entry, scalars, plus `t` and
`step_s`), so charts append (new `t`) or replace (same `t`) their last point without polling. `step_s` in 1..60,
default 5.

### Errors
Always `{"detail": "<message>", "request_id": "<hex>"}`; header `X-Request-ID` on every response.
400 invalid input, 401 auth, 404 unknown classifier / model / job, 409 too many classifiers or jobs (or an invalid job
state transition), 413 too large, 429 rate limited (+ `Retry-After`), 503 not ready / OOM, 500 internal (generic
message; traceback only in the server log). The OpenAI and Hugging Face routes use their clients' error shapes (see
Compat).

### OpenAPI
`openapi.json` at the repo root is exported from `create_app()` by `scripts/export_openapi.py` (CI checks it is up to
date). All routes, including the feature modules', have pydantic request/response models so `/docs` shows them.

## CLI (`clef`, `cli.py`)

```
clef serve [--host H] [--port P] [--device D] [--dtype T] [--quant Q] [--offload none|cpu]
           [--max-device-memory-gb GB] [--model-path DIR] [--detach] [--timeout S]
           # flags set the matching CLEF_* env; foreground by default (Ctrl+C stops)
clef stop | clef status [--json] | clef logs [-f] [-n N]
clef doctor [--no-gpu] [--smoke] [--json]    # exit 0 ok/warn, 1 on any FAIL
clef download [--revision SHA] [--dir DIR] [--yes]
clef bench [http_bench args...]              # against the running server
clef open                                    # opens http://<host>:<port>/ in the browser
clef version
```
- `--detach`: spawns `python -m clef_server.main` as a detached process (POSIX `start_new_session`, Windows
  `DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP`), writes `paths.pid_file`, logs to `paths.log_file` (rotating the
  previous log to `.1..3`), then waits for `/livez` and for `/health` status `ready|warming` (health-gated start,
  `--timeout` default 600 s); on exit or `status=error` prints the last 20 log lines and exits 1. Refuses to start
  when something already answers on the port.
- `stop`: SIGTERM (Windows: `taskkill /PID /T`), wait 30 s, then kill; removes the pidfile.
- `status`: process + `/health` summary (backend, device, status); exit 0 when healthy.
- `doctor --no-gpu`: skips everything that needs a GPU or the weights (CI smoke). On a GPU, `doctor` also prints a
  `memory setting` line (from `recommend_memory_setting`) when bf16 would not fit comfortably. `--smoke`: loads the model on
  the detected backend, runs one fixed record, checks probabilities sum to 1 and prints latency.
- `download`: `huggingface_hub.snapshot_download(MODEL_REPO, revision=MODEL_REVISION)` into the HF cache (or
  `--dir` / `CLEF_MODEL_PATH`), resumable; checks free disk >= 21 GB first and warns about the ~19 GB size.
- Wrappers: `scripts/launch_server.sh` and `clef.ps1` are thin wrappers over the CLI; `clef.ps1` (Windows -> WSL)
  keeps the hidden WSL keep-alive session.

## Console (`static/`)

Zero-build Preact + htm; one vendored charting library (`static/vendor/uPlot.*`, MIT) for time series, everything
else hand-written SVG. No runtime CDN calls. Screens: Playground (SystemOne + Classify modes), History, Batch, Evaluate (labelled dataset -> `/v1/classify/batch`
in chunks -> `/v1/evaluate/metrics`; confusion heatmap, reliability diagram, coverage curve, mistakes, export), Ops.
It consumes `/health`, `/v1/stats`, `/v1/stats/timeseries`, `/v1/events?step_s=`, `/v1/log`, `/v1/systemone`,
`/v1/batch`, `/v1/classify`, `/v1/classify/batch`, `/v1/classifiers*`, `/v1/evaluate/metrics`.
