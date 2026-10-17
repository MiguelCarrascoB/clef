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
| `clef_server/logs.py` | stdlib-only logging: text / JSON formatters, the request-id context, secret redaction for the access log, uvicorn's `log_config` |
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
| `clef_server/db.py` (planned, v3.2) | `Database` (SQLAlchemy Core engine from `CLEF_DATABASE_URL`), transactions, Alembic upgrade, `jobs.db`→`clef.db` migration, URL redaction, `audit()` helper |
| `clef_server/dbschema.py` (planned, v3.2) | the SQLAlchemy `MetaData` and every `Table` in this document (one place; migrations mirror it) |
| `clef_server/migrations/` (planned, v3.2) | Alembic `env.py`, `script.py.mako`, `versions/NNNN_*.py` |
| `clef_server/models/` (planned, v3.2) | `base.py` (`DecisionModel`, `LocalTensorAdapter`, `Capabilities`, `ModelSpec`, `UnknownModel`), `registry.py` (`clef.models` entry points, `CLEF_MODEL`), `clef_release.py` (clef-flash, clef 27B) |
| `clef_server/keystore.py` (planned, v3.2) | principals, hashed keys, key cache, quotas (`Quotas`), on `db.py`; shared by the server and `clef keys` |
| `clef_server/admin.py` (planned, v3.2) | feature module: `/v1/whoami`, `/v1/admin/keys*`, `/v1/admin/audit` |
| `clef_server/cache.py` (planned, v3.2) | in-memory LRU result cache used by `Engine.prepare` / `Engine.submit` |
| `clef_server/router.py`, `clef_server/llm.py` (planned, v3.2) | feature module: routers, label-set versions, the Review queue (`/v1/routers*`, `/v1/review*`); the OpenAI-compatible LLM client |
| `clef_server/batches.py` (planned, v3.2) | feature module: OpenAI `/v1/files*` and `/v1/batches*` on the `batch` job kind |

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
| `CLEF_DRAIN_S` | `0` | seconds `main.run` keeps serving after the first SIGTERM / Ctrl-C with `/readyz` 503 `draining` (see [`GET /readyz`](#get-readyz)); `>= 0`, 0 = stop at once. `--drain-s` |
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
| `CLEF_WEBHOOK_ATTEMPTS` | `10` | delivery attempts for final events (`job.progress` is never retried) |
| `CLEF_SAMPLE_INTERVAL_S` | `2` | gauge sampling period for the time series (memory, telemetry, queue) |
| `CLEF_INTERACTIVE_BURST` | `8` | interactive micro-batches in a row before waiting background (job) work gets one; `>= 1` |
| `CLEF_LOG_FORMAT` | `text` | `text` or `json`: server log lines (app, uvicorn, access log); json = one object per line, see [Logging](#logging-logspy) |
| `CLEF_DATABASE_URL` (planned, v3.2) | unset | SQLAlchemy URL. Unset: `sqlite:///<state_dir>/clef.db`. `postgresql://` / `postgres://` are normalised to `postgresql+psycopg://` (needs the `postgres` extra). See Storage |
| `CLEF_DATABASE_POOL_SIZE` (planned, v3.2) | `5` | Postgres pool size (`max_overflow` = the same); ignored on SQLite; `>= 1` |
| `CLEF_MODEL` (planned, v3.2) | `clef-flash` | `<id>[@<revision>]` from the model catalog (`clef-flash`, `clef`). `CLEF_MODEL_PATH` / `_REPO` / `_REVISION` stay as aliases, see Decision model interface |
| `CLEF_MASK_FREE` (planned, v3.2) | `0` | `1`: text batches call the backbone with `attention_mask=None` and the head uses CPU lengths (D2; becomes the default only after `bench/parity.py` passes) |
| `CLEF_CACHE_SIZE` (planned, v3.2) | `10000` | result-cache entries (in-memory LRU); `0` = off |
| `CLEF_JOB_TZ` (planned, v3.2) | `UTC` | IANA zone for job windows that give no `tz` and for `CLEF_JOB_WINDOW` |
| `CLEF_JOB_WINDOW` (planned, v3.2) | empty | default window `HH:MM-HH:MM` for jobs without `schedule`; empty = always open |
| `CLEF_JOB_ACTIVE` (planned, v3.2) | `4` | jobs that may be `running` (taking turns) at once; `>= 1` |
| `CLEF_JOB_AGING_S` (planned, v3.2) | `3600` | a waiting job's class rises one level per this many seconds without a turn (never to `urgent`); `0` = no aging |
| `CLEF_JOB_SORT_WINDOW` (planned, v3.2) | `0` | F2: length-sort jobs in windows of N items (256-1024); `0` = off. Default flips only if the benchmark gain is over 10% |
| `GEMINI_API_KEY` (planned, v3.2) | unset | the router's LLM key. Unset (and no `CLEF_LLM_API_KEY`) = router LLM off. Never logged; `clef doctor` only says whether it is set |
| `CLEF_LLM_API_KEY` (planned, v3.2) | unset | key for another OpenAI-compatible provider; wins over `GEMINI_API_KEY` |
| `CLEF_LLM_BASE_URL` (planned, v3.2) | `https://generativelanguage.googleapis.com/v1beta/openai/` | OpenAI-compatible endpoint (ollama, LM Studio, vLLM, OpenAI work the same) |
| `CLEF_LLM_MODEL` (planned, v3.2) | `gemini-3.8-flash` | model id sent to the endpoint |
| `CLEF_LLM_TIMEOUT_S` (planned, v3.2) | `60` | total timeout of one LLM call (2 retries on 429 / 5xx) |
| `CLEF_ROUTER_BATCH` (planned, v3.2) | `20` | buffered items per proposal call |
| `CLEF_ROUTER_COLD_BATCH` (planned, v3.2) | `20` | buffered items before the cold-start (bootstrap) call |
| `CLEF_ROUTER_FLUSH_S` (planned, v3.2) | `600` | flush a non-empty buffer after this long even if it is below the batch size |
| `CLEF_ROUTER_MODE` (planned, v3.2) | `review` | default mode of new routers: `review` or `auto` |
| `CLEF_ROUTER_MAX_NEW_PER_DAY` (planned, v3.2) | `10` | auto mode: labels a router may add per UTC day |
| `CLEF_ROUTER_MAX_INPUT_CHARS` (planned, v3.2) | `500` | each buffered input is truncated to this before it is sent to the LLM |
| `CLEF_REVIEW_TTL_DAYS` (planned, v3.2) | `30` | review items' input text is deleted this long after creation (and at once when resolved) |
| `CLEF_MAX_FILE_MB` (planned, v3.2) | `200` | body limit for `POST /v1/files` only (everything else keeps `CLEF_MAX_BODY_MB`) |
| `CLEF_MAX_BATCH_LINES` (planned, v3.2) | `50000` | lines per batch input file; the effective cap is `min(this, CLEF_MAX_JOB_ITEMS)` |
| `CLEF_FILE_TTL_HOURS` (planned, v3.2) | `720` | uploaded files without `expires_after` are purged this long after upload |
| v2 knobs | unchanged | `CLEF_MAX_TOKENS`, `CLEF_MAX_BODY_MB`, `CLEF_MAX_IMAGES/VIDEOS/PIXELS/FRAMES`, `CLEF_VIDEO_FPS`, `CLEF_MAX_BATCH`, `CLEF_MAX_QUESTIONS`, `CLEF_ALLOW_URL_FETCH`, `CLEF_URL_FETCH_MAX_MB`, `CLEF_MAX_MICROBATCH`, `CLEF_BATCH_WINDOW_MS`, `CLEF_BUCKETS`, `CLEF_PAD_MULTIPLE` (64, measured on ROCm only), `CLEF_PAD_TO_BUCKET`, `CLEF_WARMUP`, `CLEF_LOG_STATE`, `CLEF_LOG_BUFFER` |

`Config` validates enums (device, dtype, quant, quant backend, offload, log format), `CLEF_MAX_DEVICE_MEMORY_GB >= 0`, the threshold range, `CLEF_RATE_LIMIT >= 0`, `CLEF_INTERACTIVE_BURST >= 1` and `CLEF_DRAIN_S >= 0` in `__post_init__` (ValueError on a bad value) and exposes `api_keys() -> dict[name, key]`,
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
    async def decide(self, records: list[Record], priority: str = "interactive") -> list[Result]
        # results in input order; priority is "interactive" (default) or "background" (ValueError otherwise)
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
- Two priority lanes: `interactive` (everything through `ctx.infer`, and direct `engine.decide()` calls) and
  `background` (`ctx.decide`: jobs, the `evaluate` job kind). Each lane is FIFO. The worker waits for the first item
  of either lane, lets the batch window run, and only then picks the lane for the micro-batch, so an interactive
  item that arrives during a window opened by background work still goes first. A micro-batch holds ONE class:
  interactive whenever any is queued (up to the starvation bound below), else background. Micro-batching and
  bucketing work as before inside it.
- Classes are not mixed even when the micro-batch has room. The model is compute-bound (batching adds ~30%
  throughput), so an interactive record riding in a batch of 8 background records pays nearly the whole batch's
  forward time; running it alone first costs the job one extra host overhead (~50% of a single forward).
- Starvation bound: after `CLEF_INTERACTIVE_BURST` (default 8) interactive micro-batches in a row while background
  work was waiting, the next micro-batch is background. A steady interactive load therefore leaves jobs at least
  1 of every N+1 micro-batches. A micro-batch already taken is never preempted: all of its groups (one forward
  per length bucket, one per media record) run before the worker looks at the lanes again.
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
- The engine reports `stats.record_forward(...)` and `stats.set_queue_depth(n, by_class={"interactive": i,
  "background": b})` (`n` = `i + b`, the v2 total).

## Stats interface (`stats.py`)

```python
class Stats:
    def __init__(self, buffer: int = 1000, retention_s: int = 3600): ...
    def record_request(self, entry: dict) -> None   # entry = /v1/log item minus id/ts; may include "key"
    def record_forward(self, n_records: int, n_tokens: int, padded_tokens: int, ms: float) -> None
    def set_queue_depth(self, n: int, by_class: dict[str, int] | None = None) -> None
                                                    # n = total; by_class -> /v1/stats queue_depth_by_class
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
| `infer(request, reqs, batch, label="batch")` | the request path: limits, media load, one `engine.decide(priority="interactive")`, and the request log / stats entry |
| `decide(reqs, batch=True, label="batch", priority="background")` | the same without a request, for background work (jobs, the `evaluate` kind): limits, media and one `engine.decide()` in the background lane; not logged or counted in stats |
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
| `GET /health`, `GET /livez`, `GET /readyz`, `GET /schema-example` | status, liveness, readiness, copy-paste bodies (open) | [below](#get-health) |
| `POST /v1/systemone`, `POST /v1/batch` | the original typed questions API | [below](#post-v1systemone-post-v1batch) |
| `POST /v1/classify`, `/v1/classify/batch`, `/v1/score`; `PUT|GET|DELETE /v1/classifiers/{name}`, `GET /v1/classifiers`, `POST /v1/classifiers/{name}[/batch]` | classification | [below](#classification-api-classifypy-classifierspy) |
| `GET /v1/models[/{id}]`, `POST /v1/chat/completions` | OpenAI-compatible structured outputs | [Compat](#openai-and-hugging-face-compat-compatpy) |
| `POST /hf/models/{id}` (and the hidden `POST /models/{id}`) | Hugging Face zero-shot classification | [Compat](#openai-and-hugging-face-compat-compatpy) |
| `POST /v1/evaluate`, `POST /v1/evaluate/metrics` | accuracy, F1, confusion matrix, calibration | [Evaluation](#evaluation-evaluationpy) |
| `POST /v1/jobs`, `GET /v1/jobs`, `GET /v1/jobs/{id}`, `GET /v1/jobs/{id}/results`, `POST /v1/jobs/{id}/cancel`, `DELETE /v1/jobs/{id}` | async jobs | [Jobs](#async-jobs-and-webhooks-jobspy-webhookspy) |
| `GET /v1/stats`, `/v1/stats/timeseries`, `/v1/log`, `/v1/events` | observability (console) | below |
| `GET /v1/classifiers/{name}/versions[/{v}]` (planned, v3.2) | classifier history | Storage |
| `GET /v1/whoami`, `/v1/admin/keys*`, `GET /v1/admin/audit` (planned, v3.2) | principals, keys, quotas, audit | Keys and principals |
| `/v1/routers*`, `/v1/review*` (planned, v3.2) | self-evolving router, human review queue | Review queue and router |
| `/v1/files*`, `/v1/batches*` (planned, v3.2) | OpenAI Files + Batch API | OpenAI Files and Batches |

### Auth, rate limit, CORS (`security.py`)

- Keys: `CLEF_API_KEY` and/or `CLEF_API_KEYS` (see Configuration). When any key exists every `/v1/*` route, and the
  Hugging Face routes (`/hf/models/*`, `/models/*`), need `X-API-Key: <key>` or `Authorization: Bearer <key>` (401 +
  `WWW-Authenticate: Bearer` otherwise; the OpenAI and HF routes render that in their own error shape). For SSE
  `/v1/events?key=<key>` is also accepted for plain `EventSource` clients, which cannot set headers (the console
  sends the header; see [Logging](#logging-logspy) for how the query value is masked). Comparison is constant-time over all
  keys. The matching key's **name** (never the key) is stored on `request.state.key_name` and written to the request
  log (`"key"`) and to `/v1/stats` (`by_key`). Auth off -> `key` is `null`.
- Open routes: `/`, static files, `/health`, `/livez`, `/readyz`, `/schema-example`, `/docs`, `/openapi.json`.
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
  A job interrupted by a stop or crash resumes after its last stored row on the next start. Rows are appended in
  one transaction that reads `done` inside it, so an in-flight chunk is either stored whole or recomputed (no gap,
  no duplicate). A clean stop (lifespan shutdown) parks the running job as `interrupted` without counting it;
  only a job found still `running` at startup (a crash) increments `resumes`, and more than
  `CLEF_JOB_MAX_RESUMES` crashes fail it. A job whose cancel was requested before the stop or crash ends
  `cancelled` at that point (or at the next start), never resumed. Cancel is atomic per state:
  `JobStore.cancel_queued` (queued / interrupted -> cancelled), else `JobStore.flag_cancel` in one transaction
  (running -> `cancel_requested=1`; interrupted, i.e. parked meanwhile by a stop or runner error -> cancelled at
  once), so a park racing the cancel cannot drop it. Every terminal transition (finish, cancel while queued or
  interrupted, cancel / give-up at recovery) writes the status and the pending terminal webhook record in ONE
  SQLite transaction, so a crash cannot lose the event. A row the engine rejects
  becomes an `{"index", "error"}` row; the job fails only on systemic errors (engine down for
  `CLEF_JOB_ENGINE_WAIT_S`). Finished jobs are purged `CLEF_JOB_TTL_HOURS` after they finish.
- Ownership: with auth on, a job belongs to the key name that submitted it and other keys get 404 (also in the
  list); jobs created while auth was off stay visible to every key.
- Webhooks: events `job.succeeded`, `job.failed`, `job.cancelled` (default) and opt-in `job.progress` (at most one
  per 10 s, one attempt). Headers `X-Clef-Event`, `X-Clef-Delivery`, and with a secret `X-Clef-Timestamp` and
  `X-Clef-Signature: sha256=` HMAC-SHA256 of `"<timestamp>.<raw body>"`. 2xx = delivered; 408 / 425 / 429 / 5xx and
  network errors retry with exponential backoff up to `CLEF_WEBHOOK_ATTEMPTS`; other statuses and all redirects are
  final. A delivery still pending at restart is retried with the same delivery id, but its attempt count and backoff
  start over (known behaviour). Disabled until `CLEF_WEBHOOK_ALLOW` is set; the allow-list is checked at submit and on every attempt, the
  host is resolved then and the connection is pinned to the checked IP (see [SECURITY.md](../SECURITY.md)).
- Jobs are not counted in `/v1/stats` or the request log (they use `ctx.decide`).
- Custom kinds: `ctx.extra.setdefault("job_kinds", {})[name] = {"validate": fn, "run": async_fn, "resumable": bool}`
  (see [jobs.md](jobs.md#writing-a-job-kind)).

### `GET /health`
200 when status is ready or warming, 503 otherwise. `{"ready", "status", "version": "3.1.0", **engine.info(),
"load_seconds", "uptime_s", "error", "limits": Config.public_limits()}`. `GET /livez` -> 200 `{"ok": true}`.

### `GET /readyz`
Readiness for load balancers and orchestrators (open, like `/livez`). 200 `{"ready": true}` only when the engine
status is `ready` and the app is not draining; otherwise 503 `{"ready": false, "status":
"loading|warming|error|draining", "error": <engine error or null>}`. Unlike `/health`, `warming` is not ready.
Draining: uvicorn closes its listening sockets and drains connections BEFORE the lifespan shutdown runs, so a flag
set there is never seen by a client. `main.run` therefore serves through `DrainingServer` (a `uvicorn.Server`
subclass): the FIRST SIGTERM / Ctrl-C sets `app.state.draining = True` and keeps serving for `CLEF_DRAIN_S`
seconds (`/readyz` answers 503 `draining`, every other route works as usual), then calls uvicorn's own
`handle_exit` and the normal shutdown starts. A second signal during the drain skips the rest of it; after that
uvicorn's semantics apply (another Ctrl-C forces the exit). `CLEF_DRAIN_S=0` (default) stops at once, as before.
The lifespan still sets `draining = True` first thing at shutdown, for launchers other than `main.run` (plain
`uvicorn`). A drain only helps a load balancer that probes `/readyz` faster than it lasts: probe interval x
failure threshold must be below `CLEF_DRAIN_S`, and `CLEF_DRAIN_S` below the supervisor's stop timeout (Docker
sends SIGKILL after 10 s by default; `clef stop` waits 30 s). Liveness stays on `/livez` (process up); the Docker
HEALTHCHECK uses `/readyz` with a long start period so a container whose model failed to load turns unhealthy.

### `GET /v1/stats?window_s=300`
v2 body plus `"by_key": {"<name>|anonymous": count}` and `"by_endpoint": {"/v1/classify": count, ...}` over the
window, `"queue_depth_by_class": {"interactive": n, "background": n}` next to the v2 total `queue_depth`, plus `"status"` and `"gpu"` (`vram_allocated_gb`, `vram_reserved_gb`, `vram_total_gb`, `memory_kind`) and
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
default 5. Auth: the `X-API-Key` / Bearer header like every route, or `?key=` (kept for `EventSource` clients; masked
in the access log).

### Logging (`logs.py`)
- `setup_logging(fmt)` installs one root handler (idempotent; a later call only swaps its formatter). `run(cfg)`
  passes `uvicorn_log_config(fmt)` to `uvicorn.run(log_config=...)`, so the foreground `clef serve` and the
  `--detach` child (`python -m clef_server.main`) log the same way.
- `CLEF_LOG_FORMAT=text` (default): the app keeps `"%(asctime)s %(levelname)-7s %(name)s %(message)s"`, uvicorn
  keeps its own formatters. `json`: the app, `uvicorn` / `uvicorn.error` and `uvicorn.access` all write one JSON
  object per line: `{"ts": "<ISO 8601 UTC, ms>", "level", "logger", "msg"}` plus `"request_id"` when the record is
  emitted inside a request (a `ContextVar` set by `GuardMiddleware`; the access line is logged inside it) and
  `"exc"` (the formatted traceback) when there is one. Stdlib only.
- Redaction: `RedactSecretsFilter` rewrites the values of the `key`, `api_key` and `token` query parameters to
  `REDACTED` in every string of `record.args` (keeping the args tuple's shape, which uvicorn's `AccessFormatter`
  unpacks) and in `record.msg`. It parses each `?query` (up to `#`, whitespace or `"`): the query is split on `&`
  and a parameter is redacted when its name, percent-decoded (as auth reads it) and lower-cased, is one of the
  three, so `k%65y=` and `KEY=` are masked and `step_key=` is not; every other parameter stays byte-identical. It sits on uvicorn's handlers (via `log_config`), on clef's root handler and, from
  the lifespan, on the `uvicorn.access` logger, which covers servers started with plain `uvicorn`. Keys therefore never reach
  clef's logs; reverse-proxy logs are the operator's (SECURITY.md has Caddy / nginx masking snippets).

### Errors
Always `{"detail": "<message>", "request_id": "<hex>"}`; header `X-Request-ID` on every response.
400 invalid input, 401 auth, 404 unknown classifier / model / job, 409 too many classifiers or jobs (or an invalid job
state transition), 413 too large, 429 rate limited (+ `Retry-After`), 503 not ready / OOM, 500 internal (generic
message; traceback only in the server log). The OpenAI and Hugging Face routes use their clients' error shapes (see
Compat).

Planned (v3.2): **403** a valid key without the route's scope (`{"detail": "key 'ci-bot' lacks scope 'jobs'"}`;
OpenAI shape `permission_denied`); **404** also for a `model` the server does not serve (`model_not_found` on the
OpenAI routes); **429** also for an exhausted daily quota (`daily quota exceeded (N records/day)`, `Retry-After` =
seconds to the next UTC midnight; OpenAI code `insufficient_quota`). `ERRORS` in `main.py` and
`compat_errors._OPENAI_TYPES/_CODES` gain 403.

### OpenAPI
`openapi.json` at the repo root is exported from `create_app()` by `scripts/export_openapi.py` (`--check` verifies it is up to
date). All routes, including the feature modules', have pydantic request/response models so `/docs` shows them.

## CLI (`clef`, `cli.py`)

```
clef serve [--host H] [--port P] [--device D] [--dtype T] [--quant Q] [--offload none|cpu]
           [--max-device-memory-gb GB] [--model-path DIR] [--log-format text|json] [--detach] [--timeout S]
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
- `doctor --no-gpu`: skips everything that needs a GPU or the weights (smoke check). On a GPU, `doctor` also prints a
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
It consumes `/health`, `/v1/stats`, `/v1/stats/timeseries`, `/v1/events?step_s=` (read with `fetch()` and the
`X-API-Key` header, not `EventSource`, so the key never goes in the URL), `/v1/log`, `/v1/systemone`,
`/v1/batch`, `/v1/classify`, `/v1/classify/batch`, `/v1/classifiers*`, `/v1/evaluate/metrics`.

## Planned work (v3.2)

The sections below are the contract for the v3.2 build waves (C to F). Each heading carries `(planned, v3.2)`; the
coordinator removes the marker when the work lands, and the "Supersedes" line says which text above it replaces
then. Do not link to these headings by anchor (the slug changes when the marker goes).

Rules shared by every planned section:
- One clef server per database (D3). No leases, no outbox, no cross-process cancel. `recover()` keeps turning
  every `running` job into `interrupted` at startup.
- SQL below is the logical DDL. It is valid on SQLite and Postgres except for identity columns, written
  `BIGINT PRIMARY KEY /* identity */`: SQLAlchemy emits `INTEGER PRIMARY KEY AUTOINCREMENT` on SQLite
  (`sqlite_autoincrement=True`) and `BIGINT GENERATED BY DEFAULT AS IDENTITY` on Postgres. Timestamps are
  `DOUBLE PRECISION` unix seconds (as `jobs` today), days are `TEXT 'YYYY-MM-DD'` (UTC), booleans are `INTEGER`
  0/1, JSON is `TEXT`, upserts use `INSERT ... ON CONFLICT (...) DO UPDATE` (SQLite >= 3.24). No partial indexes in
  new tables, no `WITHOUT ROWID`, no `PRAGMA` outside `db.py`.
- Every DB call runs in a thread (`asyncio.to_thread` / `run_in_threadpool`), never on the event loop. The auth
  path never touches the DB per request (keys come from an in-memory cache).
- Python >= 3.10: `datetime.timezone.utc` (not `datetime.UTC`), `zoneinfo` (plus `tzdata` on Windows), no
  `StrEnum`, no `typing.Self`.
- `cli.py` stays stdlib-only **at import time**. `clef db`, `clef backup`, `clef keys` and `clef classifiers`
  import `clef_server.db` / `keystore` inside the command function and exit 2 with
  `needs the server extra: pip install 'clef-local[server]'` when SQLAlchemy is missing.
- New tests go in new files named after the workstream (`tests/unit/test_db.py`, `test_models.py`,
  `test_keys.py`, `test_cache.py`, `test_scheduler.py`, `test_router.py`, `test_batches.py`); edits to existing
  test files are limited to what a changed contract forces.
- Shared files are edited **append-only** where possible: new `Config` fields, `AppContext` fields, Configuration
  rows and `pyproject.toml` entries go at the end of their block under a `# <workstream id>` comment, so the
  coordinator can keep both sides of a conflict. `openapi.json`, `uv.lock` and `CHANGELOG.md` are regenerated or
  merged by the coordinator at the wave merge, never hand-merged.

## Storage

Module boundary: `db.py` owns the engine, connections, transactions, migrations and URL handling. `dbschema.py` owns
the table definitions. Stores (`JobStore` in `jobs.py`, `ClassifierStore` in `classifiers.py`, `KeyStore` in
`keystore.py`, the router / review / files stores) take a `Database` and use SQLAlchemy Core (`select`, `insert`,
`update`; `text()` only where Core cannot express it). No ORM. No SQL outside the stores and `db.py`.

Dependencies: the `server` extra gains `sqlalchemy>=2.0,<2.1` and `alembic>=1.13`; new extra
`postgres = ["psycopg[binary]>=3.2", "psycopg-pool>=3.2"]`. A Postgres URL without psycopg fails at startup with
`CLEF_DATABASE_URL is Postgres but psycopg is not installed: pip install 'clef-local[postgres]'`.

```python
class DatabaseError(RuntimeError): ...        # bad URL, missing driver, second server on the same DB, failed upgrade

class Database:
    url: str                                   # normalised SQLAlchemy URL (contains the password: never log it)
    display_url: str                           # redact_url(url)
    dialect: str                               # "sqlite" | "postgresql"
    path: Path | None                          # the SQLite file, else None
    def __init__(self, url: str, pool_size: int = 5) -> None
    def tx(self) -> ContextManager[Connection]       # write transaction: BEGIN IMMEDIATE on SQLite (under a process
                                                     # RLock, like JobStore today), BEGIN on Postgres; commit/rollback
    def read(self) -> ContextManager[Connection]     # read-only, no write lock
    def upgrade(self) -> None                        # alembic `upgrade heads`; idempotent
    def schema_version(self) -> str                  # alembic head revision(s), e.g. "0002" ("+"-joined if several)
    def bump(self, conn: Connection, key: str) -> int  # clef_meta counter += 1 inside the caller's tx; returns it
    def meta(self, key: str) -> str | None
    def close(self) -> None

def database_url(cfg: Config) -> str           # CLEF_DATABASE_URL or sqlite:///<state_dir>/clef.db; normalises postgres
def redact_url(url: str) -> str                # postgresql+psycopg://user:***@host:5432/clef
def open_database(cfg: Config) -> Database     # legacy rename -> Database -> PG lock -> upgrade -> classifier import
def audit(conn: Connection, actor: str, action: str, target: str, detail: dict | None = None,
          ip: str | None = None, request_id: str | None = None) -> None   # one audit_log row in the caller's tx
```

Engine setup:
- SQLite: a `QueuePool` of file connections; a `connect` event sets `journal_mode=WAL`, `synchronous=NORMAL`,
  `foreign_keys=ON`, `busy_timeout=30000`; writes are serialised by the `tx()` RLock + `BEGIN IMMEDIATE`. A new file
  is created `0600` on POSIX.
- Postgres: `pool_size=CLEF_DATABASE_POOL_SIZE`, `max_overflow=pool_size`, `pool_pre_ping=True`. At open, a dedicated
  connection takes `pg_try_advisory_lock(1667032422)` and holds it for the process lifetime; a second server on the
  same DB fails with `DatabaseError("another clef server holds this database (one server per database)")`. The CLI
  commands do not take the lock.
- `open_database` is called once in `create_app` and published as `ctx.db`. `JobStore`, `ClassifierStore` and the
  later stores receive it; nothing else opens connections. The lifespan closes `ctx.db` after the shutdown hooks
  (`runner.stop` no longer closes the store).

`jobs.db` → `clef.db` (only when `CLEF_DATABASE_URL` is unset): if `<state_dir>/clef.db` does not exist and
`<state_dir>/jobs.db` does, open `jobs.db` with `sqlite3`, run `PRAGMA wal_checkpoint(TRUNCATE)`, close it,
`os.replace(jobs.db, clef.db)` and delete any leftover `jobs.db-wal` / `jobs.db-shm`. Logged once at INFO. When both
files exist: a WARNING, and `clef.db` is used.

Alembic layout (`src/clef_server/migrations/`, shipped in the wheel; no `alembic.ini`, `db.py` builds the Alembic
`Config` in code):

| Revision | Creates | Owner |
| --- | --- | --- |
| `0001_baseline` | `jobs`, `items`, `jobs_status`, `jobs_idem` exactly as today | C1 |
| `0002_v32` | `clef_meta`, `classifiers`, `classifier_versions`, `audit_log`, the new `jobs` columns, `api_principals`, `api_keys`, `api_usage`, `routers`, `router_label_sets`, `review`, `llm_calls`, `files`, `batches` | C1 |

- Legacy DB (a `jobs` table, no `alembic_version`): run today's `idem_key` ALTER if the column is missing,
  `stamp 0001`, then `upgrade heads`.
- C1 creates **every table in this document** in `0002`, so waves D to F normally need no migration. A later
  workstream that must change the schema adds `NNNN_<ws>_<what>.py` with `down_revision` = the head at the start of
  its wave. Two in one wave give two heads: startup always runs `upgrade heads` (never `head`), and the coordinator
  adds an `alembic merge` revision when merging the wave.
- `dbschema.py` mirrors the migrations; a unit test compares `metadata` with a freshly migrated SQLite DB.

`schema_version` (= `Database.schema_version()`) is shown in `/health` (`"db": {"dialect", "schema_version", "url":
display_url}`), in `clef doctor` (a `database` check: connect and report the version; FAIL when the DB is behind or
ahead of the code) and by `clef db current`. No separate table.

Tables (new or changed by `0002`):

```sql
CREATE TABLE clef_meta (k TEXT PRIMARY KEY, v TEXT NOT NULL);
-- rows: keys_gen (int, bumped by every key write), classifiers_imported ('1' once the JSON import ran)

ALTER TABLE jobs ADD COLUMN model TEXT;                               -- C1: "id@revision" served at submit
ALTER TABLE jobs ADD COLUMN priority INTEGER NOT NULL DEFAULT 1;      -- E1: 0 low, 1 normal, 2 high, 3 urgent
ALTER TABLE jobs ADD COLUMN schedule TEXT;                            -- E1: JSON {"window": "20:00-06:00", "tz": ...}
ALTER TABLE jobs ADD COLUMN expire_after_s DOUBLE PRECISION;          -- E1: NULL = never expires
ALTER TABLE jobs ADD COLUMN eligible_at DOUBLE PRECISION;             -- E1: first time the job could take a turn
ALTER TABLE jobs ADD COLUMN quota_day TEXT;                           -- E1: UTC day the reservation was charged to
ALTER TABLE jobs ADD COLUMN quota_reserved BIGINT NOT NULL DEFAULT 0; -- E1: records reserved at submit
ALTER TABLE jobs ADD COLUMN tokens BIGINT NOT NULL DEFAULT 0;         -- E1: sum of input_tokens of the stored rows

CREATE TABLE classifiers (
    name           TEXT PRIMARY KEY,             -- NAME_RE
    owner          TEXT,                         -- principal that created it; NULL = auth off / imported
    latest_version INTEGER NOT NULL,
    created_at     DOUBLE PRECISION NOT NULL,
    updated_at     DOUBLE PRECISION NOT NULL
);
CREATE TABLE classifier_versions (
    name       TEXT NOT NULL REFERENCES classifiers(name) ON DELETE CASCADE,
    version    INTEGER NOT NULL,                 -- 1, 2, ...; never reused while the name exists
    definition TEXT NOT NULL,                    -- JSON ClassifierDef.model_dump() (exactly today's body)
    model      TEXT,                             -- ctx.model_ref() when written
    created_at DOUBLE PRECISION NOT NULL,
    created_by TEXT,                             -- principal name | "router:<name>" | "cli" | "import"
    PRIMARY KEY (name, version)
);
CREATE TABLE audit_log (
    id         BIGINT PRIMARY KEY /* identity */,
    ts         DOUBLE PRECISION NOT NULL,
    actor      TEXT NOT NULL,                    -- principal name, "cli:<os user>", "router:<name>", "anonymous"
    action     TEXT NOT NULL,                    -- see the action list below
    target     TEXT NOT NULL,
    detail     TEXT,                             -- JSON; never a secret, never request content
    ip         TEXT,
    request_id TEXT
);
CREATE INDEX audit_ts ON audit_log(ts);
```

Audit actions: `classifier.put`, `classifier.delete` (C1); `key.create`, `key.rotate`, `key.revoke`,
`principal.update`, `principal.disable`, `job.cancel`, `job.delete` (D1); `router.put`, `router.delete`,
`router.activate`, `review.resolve` (E2); `batch.cancel`, `file.delete` (F1). The log is append-only; there is no
purge in v3.2.

`JobStore(db)` keeps today's public method set and return shapes (`create`, `find_idem`, `get`, `list`, `payload`,
`set_total`, `add_items`, `finish`, `cancel_queued`, `flag_cancel`, `park`, `park_running`, `recover`,
`reset_items`, `delete`, `purge`, `items`, `set_delivery`, `redeliver`, `pending_deliveries`) and the A2 guarantees
(terminal status + pending webhook record in one transaction; `add_items` reads `done` inside its transaction).
Changes:
- `next_runnable()` + `mark_running()` become `claim_next() -> dict | None`: one transaction that selects
  `status IN ('queued','interrupted') ORDER BY CASE WHEN status='interrupted' THEN 0 ELSE 1 END, seq LIMIT 1`
  with `.with_for_update(skip_locked=True)` (ignored on SQLite, where `BEGIN IMMEDIATE` serialises) and sets it
  `running`. `claim(job_id) -> bool` is the same conditional update for one id (E1 picks the id itself).
- `create(..., model: str | None)` stores `jobs.model`. The submit route fills it with `ctx.model_ref()`; at run
  time a job whose `model` differs from `ctx.model_ref()` fails with `job was submitted for <model>, this server
  serves <ref>` (one model per process, D6). `model` NULL (rows from before v3.2) runs.
- Tests must not touch `store._db`; use `store.db.tx()` with `text()`.

Classifiers (`ClassifierStore(db, max_classifiers)`): `get(name)`, `list()`, `put(name, definition)`,
`delete(name)` keep their return shapes (`jobs.py` and `evaluation.py` call them). Extended signatures:
```python
def get(self, name: str, version: int | None = None) -> dict | None   # latest when version is None
def put(self, name: str, definition: dict, owner: str | None = None, created_by: str | None = None,
        model: str | None = None) -> dict           # appends version n+1; a definition equal to the latest is a
                                                    # no-op returning the latest (idempotent PUT); writes audit
def versions(self, name: str) -> list[dict] | None  # [{version, created_at, created_by, model}], newest first
def owner(self, name: str) -> str | None
```
The returned definition = today's fields + `{"version", "owner", "model"}`; `created_at` = version 1,
`updated_at` = latest. `CLEF_MAX_CLASSIFIERS` counts names. Delete removes the name and all its versions (jobs hold
a snapshot). JSON import: once (`clef_meta.classifiers_imported`), every readable `<state_dir>/classifiers/*.json`
becomes version 1 with `owner` NULL and `created_by` `import`; the directory is then renamed
`classifiers.imported/`. Unreadable files are logged and skipped.

Routes (changes; every existing body stays valid):
- `PUT /v1/classifiers/{name}` -> 200 the definition with `version`, `owner`, `model`. Owner = the caller's
  principal on the first PUT (NULL with auth off). A later PUT or DELETE by another principal is 403 unless the
  caller has `admin`; anyone may change a classifier whose owner is NULL. The admin test is one helper: the
  request's `principal.has("admin")` when `request.state.principal` is set (D1), else any caller with a key is
  admin (until D1 lands every key is an env key, and env principals have every scope), so before D1 the owner is
  recorded but no PUT / DELETE is refused.
- `GET /v1/classifiers/{name}?version=N`; `GET /v1/classifiers/{name}/versions` -> `{"name", "versions": [...]}`;
  `GET /v1/classifiers/{name}/versions/{v}` -> that definition. 404 for an unknown name or version.
- `POST /v1/classifiers/{name}[/batch]`, and the `classifier` field of jobs and `POST /v1/evaluate`, accept an
  optional `"version": int` (default latest). Job snapshots record the version they copied
  (`payload.classifier_version`).

CLI (direct DB access through `CLEF_DATABASE_URL` or the default file; works with the server stopped):
```
clef db current                          # schema version and display URL
clef db upgrade                          # alembic upgrade heads (the server also does it at startup)
clef db copy --from URL --to URL [--yes] # every table in FK order, 1000 rows per batch; the target is upgraded
                                         # first and must be empty (exit 1 otherwise); Postgres identity sequences
                                         # are reset (setval to max); refuses while a server answers on the port
clef backup [--out PATH]                 # SQLite: the sqlite3 online backup API into
                                         # <state_dir>/backups/clef-<UTC yyyymmddThhmmssZ>.db (file 0600; safe
                                         # while the server runs); Postgres: exit 2 with a pg_dump hint
clef classifiers export [--out DIR] [--name N]   # latest version of each as <DIR>/<name>.json (today's file format
                                                 # + "version"); default DIR ./classifiers
```

Error codes: 403 (not the owner), 404 (unknown version), 409 (`CLEF_MAX_CLASSIFIERS`), 503 `database unavailable`
(`DatabaseError` and SQLAlchemy `OperationalError`: an exception handler registered in `create_app`, plus the
classifier routes, which map them before `map_exception`; `map_exception` itself stays C2's hunk).

Invariants: payload columns stay out of list selects (the `_COLS` discipline); store APIs are sync; one writer
transaction at a time on SQLite; the request path reaches the DB only through stores called in the threadpool.

As built (C1; additions to the interface above, nothing in it changed):
- `Database` also has `serialize(conn, name)` (inside `tx()`: a transaction advisory lock on Postgres for
  check-then-write sections such as `CLEF_MAX_JOBS` / `CLEF_MAX_CLASSIFIERS`; a no-op on SQLite, where `BEGIN
  IMMEDIATE` already serialises), `set_meta(conn, key, value)`, `acquire_server_lock()` (called by
  `open_database`) and `info()` (the cached `/health` block; no I/O). Nested `tx()` / `read()` on the same thread
  join the open transaction. `db.py` also holds the CLI / doctor helpers (`peek_version` opens nothing for
  writing, `version_state`, `copy_database`, `backup_sqlite`, `migrate_legacy_file`) and `is_unavailable(exc)`.
- `claim_next()` returns the running row plus `claimed_from` (`"queued"` | `"interrupted"`, the status before
  the claim); `run_base_done` is set to `done` by the claim and zeroed by `reset_items`. The runner seeds
  `cancel_ids` from the claimed row and never discards an id after a claim. `JobStore.close()` closes its
  `Database` (tests); the server never calls it.
- `ClassifierStore.put` / `delete` take keyword-only `actor`, `ip`, `request_id` for the audit row (`actor`
  defaults to `created_by`, then `"anonymous"`). The JSON import writes `classifier.put` rows with actor `import`.
- `AppContext.model_ref()` / `check_model()` are methods (they read the engine's hooks at call time).
- Job payloads of `classify` / `score` accept `version` next to `classifier`; the snapshot drops it and records
  `classifier_version`. `EvaluateRequest` has both fields too.
- The alembic logger is set to WARNING; `upgrade()` logs `database: schema <from> -> <to>` at INFO when it
  changed something.

Supersedes when landed: "Storage: SQLite `<state_dir>/jobs.db`" in Async jobs, the `classifiers.py` row of the
package layout, and the "Files: `<state_dir>/classifiers/<name>.json`" bullet in Saved classifiers.

## Decision model interface (planned, v3.2)

Scope (D5, D6, D13, D14): one model per process, chosen by `CLEF_MODEL` and a restart. v3.2 ships one adapter,
`clef-release`, which serves every Cloudflare-style release (`joint_schema_model.py` + `joint_head.*` next to the
weights): `clef-flash` and `clef` (27B). Not in v3.2: hf-label-lm, encoders, remote adapters, several models at
once, hot-swap, falling back to the default model for an unknown id. The interface stays generic enough to add them.

Module boundary: `models/base.py` and `models/registry.py` are torch-free at import. `models/clef_release.py`
imports torch lazily (inside `load` and the tensor methods). The engine (`engine.py`) owns the worker thread, the two
priority lanes (A3), micro-batching, bucketing, padding, warmup, OOM handling and stats; it calls the adapter for
everything model-specific. `offload.py` keeps `plan_layers` / `LayerStreamer` and takes the layer names from the
adapter's `OffloadLayout`. `backend.py` stays the only device code.

```python
# models/base.py
class UnknownModel(LookupError): ...          # -> 404 (`model_not_found` on the OpenAI routes)

@dataclass(frozen=True)
class Capabilities:
    question_types: frozenset[str] = frozenset({"choice", "score", "noul"})
    joint_questions: bool = True              # all questions of a record in ONE forward
    images: bool = False
    videos: bool = False
    max_tokens: int = 16384                   # hard input limit
    max_options: int | None = None            # None = CLEF_MAX_LABELS
    calibrated: str = "trained"               # "trained" | "temperature-fitted" | "raw" | "unknown"
    supports_offload: bool = False
    quant: frozenset[str] = frozenset({"none"})
    optional_kernels: tuple[str, ...] = ()    # ("fla", "causal_conv1d") for Qwen3.5 hybrids

@dataclass(frozen=True)
class ModelSpec:
    id: str                                   # public id: "clef-flash", "clef"
    adapter: str                              # entry-point name: "clef-release"
    repo: str | None                          # HF repo, e.g. "Cloudflare/clef-flash"
    revision: str | None                      # full 40-char sha (pinned), or None for a local dir
    weights_gb: float                         # catalog estimate; the adapter reads the real size at load
    options: dict = field(default_factory=dict)
    @property
    def ref(self) -> str                      # f"{id}@{revision or 'local'}"

@dataclass(frozen=True)
class OffloadLayout:
    layers_prefix: str                        # "model.language_model.layers."
    embed_key: str                            # "model.language_model.embed_tokens"
    head_key: str                             # "lm_head"
    device_prefixes: tuple[str, ...]          # kept on the device: vision tower, final norm, rotary, joint head

class Encoded(Protocol):                      # opaque to the engine except these
    length: int                               # real tokens (bucketing, usage.input_tokens)
    is_media: bool                            # media records run alone and unpadded
    questions: Sequence[Any]                  # each has .question_id and .option_ids

class DecisionModel(Protocol):
    name: ClassVar[str]                       # adapter (entry-point) name
    spec: ModelSpec
    capabilities: Capabilities
    @classmethod
    def catalog(cls) -> list[ModelSpec]       # the ids this adapter serves out of the box
    def __init__(self, spec: ModelSpec, cfg: Config) -> None
    def resolve(self) -> Path                 # local weights dir; never downloads; paths.ModelNotFound with a hint
    def weights_gb(self, path: Path) -> float # from the safetensors headers
    def load(self, backend: Backend, dtype: Any) -> LoadInfo   # worker thread: standard / quantized / offloaded
    def validate(self, record: Record) -> None  # capability check, ValueError (400) naming the field
    def info(self) -> dict                    # {"id", "adapter", "revision", "ref", "capabilities"}

class LocalTensorAdapter(DecisionModel):     # what the engine drives
    def encode(self, record: Record) -> Encoded         # any thread (holds its own tokenizer lock);
                                                        # raises engine.InputTooLarge (the string match lives here)
    def collate(self, encs: list[Encoded], pad_to: int | None) -> dict
        # CPU batch: {"input_ids", "attention_mask", "media", "lengths": list[int]}; right-padded to pad_to
        # (None for media records); "lengths" are the real lengths, known on the CPU (no .item())
    def to_device(self, batch: dict, device: Any) -> dict
    def forward(self, batch: dict) -> list[list[Any]]   # worker thread, inside torch.inference_mode():
                                                        # per record, per question, the option logits
    def answer(self, enc: Encoded, question: dict, logits: Any) -> dict  # SystemOne answer (softmax inside)
    def pad_id(self) -> int
    def warm_record(self, repeats: int) -> Record
    def offload_layout(self) -> OffloadLayout | None
```

Registry (`models/registry.py`):
```python
def adapters() -> dict[str, type[DecisionModel]]   # importlib.metadata.entry_points(group="clef.models")
def catalog() -> dict[str, ModelSpec]              # union of adapter catalogs; duplicate id -> the built-in wins + WARNING
def resolve_spec(cfg: Config) -> ModelSpec         # CLEF_MODEL + the legacy aliases (rules below); UnknownModel
def create(cfg: Config) -> DecisionModel
```
- `pyproject.toml`: `[project.entry-points."clef.models"] clef-release = "clef_server.models.clef_release:ClefReleaseAdapter"`.
  Built-ins register exactly like third-party adapters (D14). Entry points are only read from installed
  distributions; the adapter that loaded and its distribution version go in `/health` and the startup log.
- Catalog (`ClefReleaseAdapter.catalog()`): `clef-flash` → `Cloudflare/clef-flash@<MODEL_REVISION>` (today's
  pin, `weights_gb` 19.0); `clef` → `Cloudflare/clef@<sha>` with the full sha taken by C2 from the HF API (never
  typed from memory), `weights_gb` from its index.
- Loading `joint_schema_model.py` uses `importlib.util.spec_from_file_location("clef_release_<sha8>", path)`, so the
  module name no longer collides.

`CLEF_MODEL` and the old variables (precedence, checked in `Config.__post_init__`, `ValueError` on a conflict):
1. `CLEF_MODEL=<id>[@<rev>]` selects the catalog entry; unset = `clef-flash`, except that an unset `CLEF_MODEL`
   with a `CLEF_MODEL_REPO` equal to a catalog entry's repo selects that entry.
2. `@<rev>` overrides the pinned revision. `CLEF_MODEL_REVISION` is an alias for it; both set and different is an
   error.
3. `CLEF_MODEL_REPO` overrides the entry's repo (download source + cache lookup).
4. `CLEF_MODEL_PATH` is the local release dir for the selected entry (no download, no cache lookup); its `ref` is
   `<id>@<rev>` when a revision was given, else `<id>@local`.
An id that is not in the catalog is an engine load error (status `error`, message listing the known ids), not a
crash; `clef doctor` reports it as FAIL.

Engine (changes to the interface above; `Engine(cfg, stats, loader=None, backend=None, model=None)`):
```python
class Engine:
    model: DecisionModel | None        # set on the worker thread at load; test injection via `model=`
    spec: ModelSpec                    # resolved at construction (torch-free)
    def model_ref(self) -> str         # spec.ref
    def check_model(self, requested: str | None) -> str
        # None / "" -> spec.id. Accepted: spec.id, f"{spec.id}@{rev}" (full sha or a >= 7-char prefix), spec.repo.
        # Anything else -> UnknownModel(f"model {requested!r} is not served here (this server serves {spec.id!r})")
    async def prepare(self, records: list[Record], cache: bool = True) -> Prepared
        # validate + check_model per record, cache lookup (D3; no-op until then), encode the misses in threads
    async def submit(self, prepared: Prepared, priority: str = "interactive") -> list[Result]
        # enqueue the misses in the lane, await, fill the cache, merge hits back in input order
    async def decide(self, records: list[Record], priority: str = "interactive", cache: bool = True) -> list[Result]
        # = submit(await prepare(records, cache), priority); unchanged for every existing caller

@dataclass
class Prepared:
    records: list[Record]
    keys: list[str | None]             # cache keys; None = not cacheable (filled by D3)
    hits: dict[int, Result]            # index -> cached result (filled by D3)
    encoded: list[Encoded | None]      # None at hit positions
    def slice(self, start: int, stop: int) -> Prepared   # for bisection in decide_resilient
```
- Compatibility: `Engine(cfg, stats, loader=fn)` still accepts the test namespace (`load_release_model`,
  `encode_record`, `collate_records`, `systemone_answer`); it is wrapped with `ClefReleaseAdapter.from_module(...)`.
  `clef_server.engine` keeps exporting the names tests patch: `load_model`, `_load_offloaded`,
  `_embeddings_to_host`, `backend_mod`, `bucket_for`, `pad_batch_right`, `LoadInfo`, `device_budget_gb`,
  `_import_model_module`, and the attributes `_load_info`, `_thread`.
- Results echo `model` = `spec.id` (no longer the requested string). `usage` and `timing` are unchanged.
- The priority lane stays in the engine, above the adapter: the lanes hold `_Item`s carrying `Encoded`, and the
  worker calls `collate` → `to_device` → `forward` → `answer` for each length group. The offload layer is set up
  inside `adapter.load()` (it calls `engine.load_model` / `offload.plan_layers` with its `OffloadLayout`); the
  engine only resets the streamer after a failed forward (`LoadInfo.streamer.reset()`).
- `paths.WEIGHTS_GB`, `MIN_FREE_GB` (cli) and doctor's file list come from the selected spec / adapter.

Perf hooks (D2, F2, wave G; nothing here changes probabilities by itself):
- D2 works inside `ClefReleaseAdapter.forward` and its head: with `CLEF_MASK_FREE=1` a text-only batch calls the
  backbone with `attention_mask=None` and the head slices by `batch["lengths"]` (sync-free, vectorised). Media
  batches keep the mask. Gate: `bench/parity.py` (max |dp| within the fp16 noise floor, 0 flips).
- TunableOp results are kept per model: `PYTORCH_TUNABLEOP_FILENAME` under
  `<state_dir>/tunableop/<spec.id>@<rev8>/` (set by `backend.prepare_environment` from `CLEF_MODEL`).
- Graph capture (wave G) belongs in `forward`, keyed by `(batch["input_ids"].shape)`; warmup already visits every
  bucket <= 1024 at batch sizes {1, max_microbatch}.
- F2 tokenizes ahead with `ctx.prepare` (below): encoding is not "in flight"; only `submit` enqueues.

`AppContext` additions (C2): `prepare(reqs, batch=True, label="batch", cache=True) -> Prepared` (limits, media,
`engine.prepare`) and `decide_prepared(prepared, priority="background") -> list[Result]`. `ctx.decide(...)`
gains `cache: bool = True` and equals `decide_prepared(await prepare(...))`. C1 adds `model_ref() -> str | None`
and `check_model(requested) -> str`, wired to the engine's methods when it has them (test fakes fall back to
`None` / returning `requested or "clef-flash"`).

API changes:
- `model` on every request body (SystemOne, classify, score, classifiers, jobs payloads, chat, evaluate) defaults
  to `None` = the served model. A model the server does not serve is **404** before any inference:
  `{"detail": "model 'x' is not served here (this server serves 'clef-flash')"}`; OpenAI routes
  `{"error": {"type": "invalid_request_error", "code": "model_not_found", "param": "model"}}`. `/v1/chat/completions`
  no longer lists `model` in `clef.ignored`.
- Jobs validate every `model` in the payload at submit (404), and store `jobs.model = ctx.model_ref()` (C1).
- `POST /v1/evaluate` and the `evaluate` job kind accept `model` (validated the same way).
- `GET /v1/models` lists the one served model: `{"id": spec.id, "object": "model", "created", "owned_by":
  "cloudflare", "clef": {"adapter", "revision", "ref", "status", "capabilities"}}`; `GET /v1/models/{id}` accepts
  the same strings as `check_model`, else 404 `model_not_found`.
- `POST /hf/models/{id}`: the path id is still ignored (Hugging Face clients send their own model names);
  responses report `spec.id`.
- `/health` gains `"model": engine.model.info()` (or the spec before load). The `fast_path` block stays (filled from
  `capabilities.optional_kernels`).
- `clef download [--model ID[@REV]]` (default: the selected spec), `clef models` (catalog + installed adapters,
  marking the selected one; `--json`).
- Clients and console stop hard-coding `"clef-flash"`: they omit `model` unless the user sets one, and the console
  shows the id from `/v1/models`.

Tables: none (the model ref is stored on `jobs.model` and `classifier_versions.model`, see Storage).

Error codes: 404 unknown model; 400 capability mismatch (images to a text-only model, a question type the adapter
lacks, more options than `max_options`); 413 `InputTooLarge`; 503 not ready / OOM (unchanged).

Invariants: ONE GPU worker thread, ONE adapter per process; the adapter never edits files in the model dir; every
result for a record is computed by exactly one `forward` call (cache hits excepted).

Supersedes when landed: "ONE uvicorn worker per process (one model)" (becomes "one adapter"), the engine's
`loader` / `load_release_model` wording in Engine interface, "`GET /v1/models` lists `clef-flash`" and "`model`
defaults to `"clef-flash"`" in the Classification and Compat sections.

## Keys and principals (planned, v3.2)

Model: a **principal** is a name with scopes, a daily quota and a status; it owns jobs, classifiers, routers, files
and batches (all keyed by name, as `jobs.owner` is today). A principal has 1 to 5 active **keys** (overlapping
rotation). Env keys (`CLEF_API_KEY`, `CLEF_API_KEYS`, `name:key` kept) are principals with `source="env"`, every
scope (full admin), no quota and the global rate limit (break-glass + bootstrap, D10).

Module boundary: `keystore.py` (schema access through `db.py`, key generation / parsing / hashing, CRUD, the
in-memory `KeyCache`, `Quotas`) is shared by the server and `clef keys`. `security.py` keeps CORS, the rate limiter
and `RedactSecretsFilter`, and gains `Authenticator` (env + DB resolution) and `require_scope`. `admin.py` is the
feature module with the routes.

Key format: `clef_live_<id>_<secret>`: `id` = 12 chars `[a-z2-7]` (public, shown in lists); `secret` =
`secrets.token_urlsafe(32)` (43 chars, may contain `_` and `-`); 66 chars total. Parse by fixed offsets (prefix
`clef_live_`, `id = s[10:22]`, `s[22] == "_"`, `secret = s[23:]`), never by splitting on `_`. Only
`sha256(whole string)` (hex) is stored. Lists show `clef_live_<id>_…`. The secret is returned once, at create /
rotate.

Tables (created by C1 in `0002`):
```sql
CREATE TABLE api_principals (
    name        TEXT PRIMARY KEY,                -- KEY_NAME_RE; also jobs.owner
    scopes      TEXT NOT NULL,                   -- sorted, comma-separated: "classify,jobs"
    daily_quota BIGINT,                          -- records per UTC day; NULL = unlimited
    rate_limit  INTEGER,                         -- requests/min; NULL = CLEF_RATE_LIMIT
    note        TEXT,
    created_at  DOUBLE PRECISION NOT NULL,
    created_by  TEXT,                            -- principal name or "cli:<os user>"
    disabled_at DOUBLE PRECISION                 -- tombstone: rows are never deleted, names never reused
);
CREATE TABLE api_keys (
    id            TEXT PRIMARY KEY,              -- the 12-char public id
    principal     TEXT NOT NULL REFERENCES api_principals(name),
    secret_sha256 TEXT NOT NULL,
    created_at    DOUBLE PRECISION NOT NULL,
    created_by    TEXT,
    expires_at    DOUBLE PRECISION,              -- NULL = never; rotate sets now + grace on the old keys
    revoked_at    DOUBLE PRECISION,
    last_used_at  DOUBLE PRECISION               -- approximate, flushed every 10 s
);
CREATE INDEX api_keys_principal ON api_keys(principal);
CREATE TABLE api_usage (
    principal TEXT NOT NULL,
    day       TEXT NOT NULL,                     -- 'YYYY-MM-DD' UTC
    records   BIGINT NOT NULL DEFAULT 0,         -- interactive records + job reservations (net of refunds)
    requests  BIGINT NOT NULL DEFAULT 0,
    rejected  BIGINT NOT NULL DEFAULT 0,         -- requests refused for quota
    PRIMARY KEY (principal, day)
);
```
Every write to `api_principals` / `api_keys` bumps `clef_meta.keys_gen` in the same transaction. Key actions are
written to `audit_log` (Storage) in the same transaction.

Scopes: `classify`, `jobs`, `classifiers`, `ops`, `admin`, `urgent`. `admin` implies all the others.

| Routes | Scope |
| --- | --- |
| `POST /v1/systemone`, `/v1/batch`, `/v1/classify[/batch]`, `/v1/score`, `POST /v1/classifiers/{n}[/batch]`, `GET /v1/classifiers*`, `GET /v1/models*`, `POST /v1/chat/completions`, `/hf/models/*`, `/models/*`, `POST /v1/evaluate[/metrics]`, `POST /v1/routers/{n}/route[/batch]` | `classify` |
| `/v1/jobs*`, `/v1/files*`, `/v1/batches*` | `jobs` (submitting needs only `jobs`) |
| `PUT` / `DELETE /v1/classifiers/{n}`, `/v1/routers*` except route, `/v1/review*` | `classifiers` |
| `GET /v1/stats*`, `/v1/log`, `/v1/events` | `ops` |
| `/v1/admin/*` | `admin` |
| `urgent: true` on a job or batch | `urgent` (403 otherwise) |
| `GET /v1/whoami` | any valid key |

Admins see every principal's jobs, files, batches, routers and review items (D19). A default-deny unit test walks
`app.routes` and fails when a `/v1/*`, `/hf/*` or `/models/*` route has no scope dependency.

```python
ALL_SCOPES = frozenset({"classify", "jobs", "classifiers", "ops", "admin", "urgent"})

@dataclass(frozen=True)
class Principal:
    name: str
    source: str                          # "env" | "db"
    scopes: frozenset[str]               # env: ALL_SCOPES
    daily_quota: int | None              # env: None
    rate_limit: int | None               # None = CLEF_RATE_LIMIT
    key_id: str | None                   # env: None
    expires_at: float | None
    def has(self, scope: str) -> bool    # "admin" in scopes or scope in scopes

class Authenticator:                     # security.py
    def __init__(self, env_keys: dict[str, str], cache: KeyCache | None): ...
    @property
    def enabled(self) -> bool            # env keys exist OR any api_principals row exists (even disabled), so
                                         # revoking the last DB key never switches auth off
    def resolve(self, token: str | None) -> Principal | None
        # 1. token has the clef_live_ format and its id is cached: compare sha256 (hmac.compare_digest), then
        #    check revoked_at, expires_at (against now, at request time) and the principal's disabled_at.
        # 2. otherwise the constant-time walk over env keys (as KeyAuth does today).
        # 3. else None (-> 401).

class KeyCache:                          # keystore.py; no DB I/O on the request path
    def __init__(self, db: Database, refresh_s: float = 10.0): ...
    def get(self, key_id: str) -> tuple[KeyRow, PrincipalRow] | None
    def any_principal(self) -> bool
    def reload(self) -> None             # full reload (a few hundred rows)
    async def run(self) -> None          # ctx.on_startup task: every 10 s read clef_meta.keys_gen in a thread and
                                         # reload when it changed; full reload every 60 s regardless
    # admin API writes call reload() synchronously, so they apply at once; CLI writes apply within 10 s

class QuotaExceeded(Exception):
    limit: int; remaining: int; retry_after: int   # seconds to the next UTC midnight

class Quotas:                            # keystore.py; in-memory and authoritative (one server per DB)
    def reserve(self, principal: Principal | None, n: int) -> Reservation | None
        # None for env principals / auth off / no quota. Raises QuotaExceeded when used_today + n > daily_quota.
        # Records the reservation on today's UTC row at once.
    def refund(self, reservation: Reservation, n: int) -> None   # n <= reservation.n; same day row
    def used(self, name: str, day: str | None = None) -> int
    def reject(self, name: str) -> None                          # api_usage.rejected += 1
    async def run(self) -> None          # ctx.on_startup: load today's rows, flush deltas every 5 s (UPSERT in a
                                         # thread), and once more on shutdown; a crash loses at most 5 s of counts

@dataclass(frozen=True)
class Reservation:
    principal: str; day: str; n: int
```

Request path: `require_key` sets `request.state.principal` (or `None` with auth off) and keeps
`request.state.key_name = principal.name` (log, `by_key`, job owner, rate-limit identity unchanged). The rate
limiter uses `principal.rate_limit or CLEF_RATE_LIMIT`. `ctx.infer` calls `ctx.quotas.reserve(principal, len(reqs))`
before inference and refunds the whole reservation when the request ends in a 5xx. Cache hits do NOT count: the
reservation is settled after the cache lookup and only computed records are charged (hits cost no GPU time). Every quota'd response carries `X-Quota-Limit` and `X-Quota-Remaining`. Over quota:
429 `daily quota exceeded (N records/day)` + `Retry-After` (OpenAI shape `insufficient_quota`). Jobs reserve at
submit (Job scheduler).

`AppContext` additions (D1): `scope(name: str) -> list[Depends]`, `principal(request) -> Principal | None`,
`quotas: Quotas`, `auth_enabled() -> bool` (replaces `cfg.auth_required` in `jobs.scoped()`, `/health` limits and
the bind warning, which moves into the lifespan because it needs the DB).

Routes (`admin.py`; all pydantic; all `admin` except whoami; rate limited, never quota'd; admin routes need a
principal even when auth is off: 401 `admin routes need a key: set CLEF_API_KEY or run clef keys create`):

| Route | Body | Response |
| --- | --- | --- |
| `GET /v1/whoami` | | `{"auth": bool, "name", "source", "scopes": [...], "daily_quota", "used_today", "key_id", "expires_at"}` (all null with auth off) |
| `GET /v1/admin/keys?include_disabled=false` | | `{"principals": [{"name", "source", "scopes", "daily_quota", "rate_limit", "note", "created_at", "created_by", "disabled_at", "used_today", "keys": [{"id", "display", "created_at", "expires_at", "revoked_at", "last_used_at", "status": "active|expiring|expired|revoked"}]}]}`; env principals listed read-only with `keys: []` |
| `POST /v1/admin/keys` | `{"name", "scopes": [...], "daily_quota"?, "rate_limit"?, "expires_in_days"?, "note"?}` | 201 principal + `{"key": "clef_live_..."}` |
| `PATCH /v1/admin/keys/{name}` | `{"scopes"?, "daily_quota"?, "rate_limit"?, "note"?}` (explicit `null` clears quota / rate limit) | principal |
| `POST /v1/admin/keys/{name}/rotate` | `{"grace_s": 86400, "expires_in_days"?}` | 201 principal + `{"key"}`; the other active keys get `expires_at = min(current, now + grace_s)` in one tx |
| `POST /v1/admin/keys/{name}/revoke` | `{"key_id"?}` | principal; without `key_id` the principal is disabled and all its keys revoked |
| `GET /v1/admin/keys/{name}/usage?days=30` | | `{"name", "days": [{"day", "records", "requests", "rejected"}]}` oldest first, days 1..366 |
| `GET /v1/admin/audit?limit=100&before=<id>&action=<prefix>` | | `{"entries": [{"id", "ts", "actor", "action", "target", "detail", "ip", "request_id"}], "next_before": id | null}` newest first, limit 1..1000 |

Errors: 400 bad name (`KEY_NAME_RE`) or unknown scope; 404 unknown principal or key id; 409 name exists, is a
tombstone (disabled names are never reused, D19), or equals an env key name; 409 a 6th active key; 403 changing an
env principal, or a DB admin removing its own `admin` scope / revoking its own last key without `?force=true`.
Startup logs a WARNING for an env name that is also a DB principal (the env key wins).

CLI (`clef keys`, direct DB, works with the server stopped; the first DB principal turns auth on and the command
prints `auth is now required for every /v1 route`; secrets are printed once):
```
clef keys create NAME --scope classify,jobs [--quota N] [--rate-limit N] [--expires 90d|2027-01-31] [--note T] [--json]
clef keys list [--all] [--json]
clef keys rotate NAME [--grace 24h] [--json]
clef keys revoke NAME [--key-id ID]
clef keys update NAME [--scope S] [--quota N|none] [--rate-limit N|none] [--note T]
clef keys usage NAME [--days 30] [--json]
```
Audit actor for CLI writes: `cli:<os user>`.

Console: Settings calls `GET /v1/whoami` after a key is saved and shows `name (env|db) · scopes · used/quota`.
A **Keys** tab (`static/screens/keys.js`) is added to `TABS` only when the scopes include `admin`: principals table
(scope chips, quota bar, 30-day usage sparkline in hand-written SVG, keys with status badges), New key dialog,
secret-once panel (copy button; the secret is never stored in the store, history or localStorage), rotate (grace
picker), revoke (type the name to confirm), edit, audit list, env principals read-only.

Invariants: no plaintext secret is ever stored or logged; the request path does no DB I/O for auth; a principal
row is never deleted; env keys always work and are always admin; auth, once on because of a DB principal, never
turns off by itself.

Supersedes when landed: the Keys bullet of "Auth, rate limit, CORS" (`KeyAuth`, "auth off -> `key` is `null`"), the
`security.py` row's "multi-key auth", and "jobs created while auth was off stay visible to every key" (still true)
plus "other keys get 404" (admins now see all).

## Result cache (planned, v3.2)

Module boundary: `cache.py` holds `ResultCache`; the engine owns the only instance (`Engine.cache`, built from
`cfg.cache_size` in `__init__`) and uses it in `Engine.prepare` (lookup) and `Engine.submit` (store). Nothing else
reads or writes it, so interactive routes, jobs, evaluate, batches and the router all share it (D16).

```python
class ResultCache:
    def __init__(self, capacity: int = 10000) -> None     # capacity 0 = disabled (every call is a no-op)
    def key(self, model_ref: str, record: Record) -> str | None
        # None (not cacheable) when the record has images or videos.
        # Else sha256 hex of canonical JSON (sort_keys=True, separators=(",", ":"), ensure_ascii=False) of
        # {"m": model_ref, "s": record["state"], "q": record["questions"]}: the state verbatim, every question with
        # its type, instructions and criteria. Thresholds, the request id and the label order of a multi-label
        # response are applied after the model and are not in the key.
    def get(self, key: str) -> Result | None             # moves the entry to the MRU end
    def put(self, key: str, result: Result) -> None       # stores the result without "timing"; evicts the LRU entry
    def stats(self) -> dict                               # {"size", "capacity", "hits", "misses", "bypassed"}
```

Semantics:
- Thread-safe (`threading.Lock`); in memory only, lost on restart. No invalidation: the questions are in the key,
  and a model change needs a restart (D6).
- A hit returns the stored result with `usage` as stored (the original `input_tokens`) and `timing` =
  `{"queue_ms": 0, "forward_ms": 0, "batch_size": 0, "cached": true}` (main adds `total_ms`). Misses carry
  `"cached": false`. Probabilities are bit-identical to the stored forward.
- Bypass: a request with `Cache-Control: no-cache` (case-insensitive token anywhere in the header) skips the
  lookup and still stores the fresh result. `ctx.infer` reads the header and passes `cache=False` down
  (`ctx.decide(..., cache=False)` → `engine.prepare(records, cache=False)`); background callers pass it explicitly.
- Identical records inside one call are computed once (the later ones are filled from the first after the forward).
- Quotas count cached records like computed ones; the request log and stats count the request as usual.
- `/v1/stats` gains `"cache": ResultCache.stats() + {"hit_rate"}` (from `Stats.snapshot`, which reads
  `engine.cache_stats()` when the engine has it).

Env: `CLEF_CACHE_SIZE` (Configuration). Tables: none. Routes: none (header only). Error codes: none.

Invariants: a cache entry is only ever written from a successful forward of the same process; media records never
hit the cache; the key never contains secrets or the API key.

## Review queue and router (planned, v3.2)

Scope (D7, D8, D15, D17, D19): a **router** is its own resource that grows the label set of one single-label saved
classifier. It runs in review mode (default) or auto mode, only **adds** labels (grow-only; no merge, split, rename
or retire in v3.2), and publishes each new label set as a new classifier version. One **Review** table and one
console screen hold both router proposals and low-confidence items; human labels given there feed the router's
re-checks.

Module boundary: `router.py` is the feature module (routes, buffers, flush loop, publishing, the Review routes);
its pure logic (gate, "other" description, proposal validation, lexical dedup, acceptance test over a supplied
`decide`) sits in functions with no FastAPI imports, unit-tested with a fake decide. `llm.py` is the LLM client.
All inference goes through `ctx.decide` (background lane, one micro-batch at a time); no job kind is added, so
the router does not depend on `jobs.py`. Routers publish through `ClassifierStore.put(..., created_by="router:<name>")`.

LLM client (`llm.py`, httpx only, no SDK):
```python
class LLMError(RuntimeError): ...           # transport, HTTP status, timeout, schema-invalid response

@dataclass(frozen=True)
class LLMResult:
    data: dict                              # the parsed JSON, already validated against the schema
    model: str
    input_tokens: int | None
    output_tokens: int | None
    latency_ms: float

class LLMClient(Protocol):
    async def complete_json(self, *, system: str, user: str, schema: dict, name: str,
                            max_tokens: int = 2048) -> LLMResult

class OpenAICompatClient:                   # POST {base_url}chat/completions, Authorization: Bearer <key>,
    def __init__(self, base_url: str, api_key: str, model: str, timeout_s: float = 60.0,
                 http: httpx.AsyncClient | None = None) -> None
        # body: {"model", "messages": [system, user], "response_format": {"type": "json_schema",
        #        "json_schema": {"name", "schema", "strict": true}}, "max_tokens"}; 2 retries with backoff on
        #        429 / 5xx honouring Retry-After; if the content is not valid JSON, one tolerant parse (first
        #        JSON object in the text) before LLMError

def client_from_config(cfg: Config) -> LLMClient | None
    # None (router LLM off) unless CLEF_LLM_API_KEY or GEMINI_API_KEY is set in the process env (clef does not
    # read .env files: the systemd EnvironmentFile / compose .env put it there). Every call is logged to llm_calls.
```

Tables (created by C1 in `0002`):
```sql
CREATE TABLE routers (
    name           TEXT PRIMARY KEY,               -- NAME_RE
    owner          TEXT,
    classifier     TEXT NOT NULL UNIQUE,           -- the saved classifier it publishes to
    config         TEXT NOT NULL,                  -- JSON: instructions, domain_hint, gate, max_labels
    mode           TEXT NOT NULL,                  -- "review" | "auto"
    active_version INTEGER,                        -- NULL while cold (fewer than 1 real label)
    created_at     DOUBLE PRECISION NOT NULL,
    updated_at     DOUBLE PRECISION NOT NULL
);
CREATE TABLE router_label_sets (
    router             TEXT NOT NULL REFERENCES routers(name) ON DELETE CASCADE,
    version            INTEGER NOT NULL,
    labels             TEXT NOT NULL,              -- JSON {name: description}, including "other"
    source             TEXT NOT NULL,              -- "seed" | "bootstrap" | "proposal" | "manual" | "rollback"
    classifier_version INTEGER,                    -- the classifier version it was published as
    created_at         DOUBLE PRECISION NOT NULL,
    created_by         TEXT,
    PRIMARY KEY (router, version)
);
CREATE TABLE review (
    id                 TEXT PRIMARY KEY,           -- "rev_" + 20 hex
    kind               TEXT NOT NULL,              -- "item" | "proposal"
    state              TEXT NOT NULL,              -- item: "open" | "buffered" | "resolved" | "discarded" |
                                                   --       "permanent_other" | "expired"
                                                   -- proposal: "open" | "accepted" | "rejected" | "failed"
    owner              TEXT,                       -- owner of the router / classifier
    router             TEXT,                       -- NULL for classifier low-confidence items
    classifier         TEXT,
    classifier_version INTEGER,
    reason             TEXT NOT NULL,              -- item: "other" | "low_confidence" | "ambiguous" | "cold";
                                                   -- proposal: "proposal" | "bootstrap"
    input              TEXT,                       -- JSON input; NULL once resolved or after CLEF_REVIEW_TTL_DAYS
    input_sha256       TEXT,                       -- of the canonical input (dedup, kept after the purge)
    prediction         TEXT,                       -- JSON {"label", "confidence", "scores"} at capture time
    proposal           TEXT,                       -- JSON (kind=proposal): {"labels": [{"name", "description",
                                                   --   "member_ids", "rationale", "nearest_existing"}], "stats"}
    label              TEXT,                       -- the human label (item) / accepted names (proposal, JSON)
    reroutes           INTEGER NOT NULL DEFAULT 0,
    created_at         DOUBLE PRECISION NOT NULL,
    resolved_at        DOUBLE PRECISION,
    resolved_by        TEXT
);
CREATE INDEX review_state ON review(state, created_at);
CREATE INDEX review_router ON review(router, state);
CREATE TABLE llm_calls (
    id            BIGINT PRIMARY KEY /* identity */,
    ts            DOUBLE PRECISION NOT NULL,
    router        TEXT,
    purpose       TEXT NOT NULL,                   -- "propose" | "bootstrap"
    provider_url  TEXT NOT NULL,                   -- base URL (no key)
    model         TEXT NOT NULL,
    status        TEXT NOT NULL,                   -- "ok" | "error"
    latency_ms    DOUBLE PRECISION,
    input_tokens  BIGINT,
    output_tokens BIGINT,
    items         INTEGER,
    error         TEXT
);
```

Trigger (per routed item, under the active label set with `n` options including `other`):
- `other` wins → reason `other`, buffered.
- top probability `< τ(n)` → reason `low_confidence`, buffered. `τ(n)` rises with the label count: 0.5 for
  n <= 10, 0.7 for 11-30, 0.85 for 31-64 (defaults; tuned on `tickets_labelled.csv`, per-router override
  `config.gate.thresholds = [[10, 0.5], [30, 0.7], [64, 0.85]]`).
- top two within `config.gate.margin` (default 0.15) → reason `ambiguous`, review item, **not** buffered.
- "Buffered" = a `review` row `kind=item, state=buffered` with the input stored (D17).
The `other` label's description names the other labels in negative form up to 15 labels, then a generic sentence.

Buffering and calls: a flush happens when a router has `CLEF_ROUTER_BATCH` (20) buffered items, when the oldest
buffered item is `CLEF_ROUTER_FLUSH_S` old, or on `POST .../propose`. One call in flight per router and one
globally. The prompt carries the current labels and the buffered inputs truncated to
`CLEF_ROUTER_MAX_INPUT_CHARS`, numbered (members are returned as indices, never as copied text), and asks for at
most `min(3, capacity)` new labels. Capacity = `min(config.max_labels, CLEF_MAX_LABELS - 1)` real labels; at the
cap no call is made. Response schema: `{"proposals": [{"name", "description", "member_indices", "nearest_existing",
"duplicate_of", "rationale"}], "covered_by": [{"index", "label"}], "unclear_indices": [int]}` (strict; `name`
matches `NAME_RE`, description <= 15 words).

Acceptance test (before a proposal is shown or activated): build the candidate set (current + proposal + other),
re-classify the proposal's members with `ctx.decide`; it passes when at least `max(2, 50%)` land on the new label
with p >= τ(n'). It fails as a duplicate when most land on one existing label. Human labels feed it: a member a
human already labelled with an existing label counts against the proposal, and resolving an item with a label
re-classifies its text under the active set before the text is purged, recording agreement in the item's
`prediction.recheck`.

Review mode: passing proposals become `review` rows `kind=proposal, state=open` with the acceptance numbers.
Auto mode: a passing proposal is activated at once, at most `CLEF_ROUTER_MAX_NEW_PER_DAY` labels per UTC day
(beyond that it waits as a review row). Activation = new `router_label_sets` version → `ClassifierStore.put` →
re-route the buffered items in the background; an item that still fails the gate gets `reroutes += 1`, and after 2
it becomes `permanent_other` (out of future batches).

Cold start: a router with no seeds has no label set (clef needs >= 2 options). Every routed item is answered
`other` with reason `cold` without a forward, and buffered. At `CLEF_ROUTER_COLD_BATCH` (20) items one bootstrap
call proposes 3-8 labels (`maxItems` 8); the result is one proposal row holding the whole starter set (approve all,
or edit, at once). One seed is enough to start the normal loop (`[seed, other]`).

Low-confidence items from saved classifiers: `ClassifierDef` gains `review_threshold: float | None` (0..1, default
None = off). On interactive single-label runs (`POST /v1/classifiers/{name}[/batch]`, not jobs), a result with
`confidence < review_threshold` creates a `review` row `kind=item, reason=low_confidence, router=NULL` with the
input. Media inputs are never captured.

Routes (scopes from Keys; owner-scoped like jobs, admins see all; OpenAPI models for every body):

| Route | Body / query | Response |
| --- | --- | --- |
| `PUT /v1/routers/{name}` | `{"classifier"?: name (default = router name), "seeds"?: labels dict, "instructions"?, "domain_hint"?, "mode"?: "review"|"auto", "max_labels"?: 2..63, "gate"?: {"thresholds", "margin"}}` | the router: `{"name", "owner", "classifier", "mode", "config", "active_version", "labels", "buffered", "open_reviews", "created_at", "updated_at"}`. An existing single-label classifier's latest labels become the seeds when `seeds` is omitted; multi-label or score: 400; a classifier already owned by another router: 409 |
| `GET /v1/routers`, `GET /v1/routers/{name}`, `DELETE /v1/routers/{name}` | | list / router / `{"deleted"}` (the classifier stays) |
| `POST /v1/routers/{name}/route` | `{"input"}` | the classify response + `{"router", "reason": "classified|other|low_confidence|ambiguous|cold", "label_set_version", "review_id"}` |
| `POST /v1/routers/{name}/route/batch` | `{"inputs": [...]}` | `{"results": [...], "batch_ms", "request_id"}` |
| `POST /v1/routers/{name}/propose` | | 202 `{"router", "items", "llm_call_id"}`; 409 when a call is in flight; 503 `router LLM is off (set GEMINI_API_KEY)` |
| `GET /v1/routers/{name}/versions`, `POST /v1/routers/{name}/versions/{v}/activate` | | versions newest first / the router (rollback publishes that version again as a new classifier version) |
| `GET /v1/review?kind&state&router&classifier&limit=50&offset=0` | | `{"items": [<review row; input included while stored>], "total"}` newest first |
| `GET /v1/review/{id}` | | the review row |
| `POST /v1/review/{id}/resolve` | item: `{"action": "label", "label"}` or `{"action": "discard"}`; proposal: `{"action": "approve", "labels"?: [{"name", "description"}] (edited subset)}` or `{"action": "reject"}` | the row; input set NULL for items |

Errors: 400 (name, labels, capacity), 403 (scope / owner), 404, 409 (in-flight call, classifier taken, resolving a
resolved row), 503 (LLM off for `/propose`; LLM failures are recorded in `llm_calls` and on the proposal, never
surfaced on `route`).

Purge: every hour, `review.input` is set NULL for rows older than `CLEF_REVIEW_TTL_DAYS` (open / buffered items
become `expired`). Resolved rows keep `label`, `input_sha256` and `prediction` without text.

Console: one **Review** screen (`static/screens/review.js`): queue filters (kind, state, router, classifier),
item cards (input, prediction, label picker, discard), proposal cards (labels, members, acceptance bars,
approve / edit / reject), per-router panel (label set + versions with rollback, buffer size, mode, LLM status with
an "external LLM: <host>" badge, calls and tokens today).

Invariants: input text is stored only in `review.input` and only until resolved / TTL (D17); nothing leaves the
machine unless an LLM key is set; the router never edits a classifier except by appending a version; single-label
only (D8).

## Job scheduler (planned, v3.2)

Module boundary: all in `jobs.py` (`JobRunner` + a `Scheduler` class). Kinds keep their `validate` / `run`
signatures; scheduling happens per chunk inside `decide_resilient`, which every built-in kind (and `evaluate`, the
`batch` kind) already uses. A custom kind that calls `ctx.decide` directly bypasses the scheduler; the contract is
that kinds call `job.decide(reqs)` (= `decide_resilient(ctx, job, reqs)`).

Policy:
- Classes, strict between classes: `urgent` (3) > `high` (2) > `normal` (1, default) > `low` (0). Aging: a waiting
  job's effective class rises one level per `CLEF_JOB_AGING_S` without a turn, never to `urgent`.
- Fair share inside a class: stride scheduling per **owner** (principal name; `NULL` = one owner). `pass +=
  cost / weight`, weight 1 for everyone in v3.2, cost = the chunk's records. The next turn goes to the lowest pass;
  an owner that comes back sets `pass = max(own, current minimum)`. Within one owner: higher class first, then
  interrupted, then `seq`. State is in memory and rebuilt after a restart.
- Turn gate: `await scheduler.turn(job)` before every engine call of a job; `scheduler.done(job, cost)` after.
  **One job chunk in flight across all jobs** (as today): interactive work still goes first through the engine's
  lanes. A unit test asserts the background lane never holds more than one job chunk.
- Active set: up to `CLEF_JOB_ACTIVE` jobs are claimed (`JobStore.claim(job_id)`, chosen in stride order) and run
  as their own asyncio tasks; all of them show `running` (D9). `runner.current` becomes a set; progress webhooks
  are per job.
- Windows: `schedule = {"window": "HH:MM-HH:MM", "tz": "<IANA>"}` per job (default `CLEF_JOB_WINDOW` in
  `CLEF_JOB_TZ`). `start > end` crosses midnight; `start == end` = 24 h. Aware datetimes via `zoneinfo`; on DST
  days an edge inside the skipped hour starts at the next valid minute, the repeated hour uses `fold=0`. A job gets
  turns only inside its window. A queued job outside it stays `queued`; a running job that leaves it stays
  `running` but gets no turns and frees its active slot. Both show `waiting_reason: "off_hours"` and
  `next_window_at`. Unknown tz or a bad window: 400 at submit.
- `urgent: true` needs the `urgent` scope (403 otherwise), sets class `urgent` and ignores the window.
- Expiry: `expire_after_s` counts from `eligible_at` (the first moment the job could take a turn, i.e. when its
  window first opens; D19). An expired job stops between chunks and ends `cancelled` with `error = "expired"`
  (no new status; `JobInfo.expired: true`).
- Quota at submit, in this order: idempotency lookup (a replay returns the existing job and is **not** charged) →
  validate / snapshot → `ctx.quotas.reserve(principal, total)` (in memory; 429 when it does not fit) →
  `JobStore.create(..., quota_day, quota_reserved)`. If `create` raises (`JobsFull`, an idempotency race that
  returns another job, a DB error), the reservation is refunded in full before the error is returned. The whole
  job is charged to the submit day; on `cancelled` / `failed` / expiry, `total - done` is refunded to that day.
  Env principals and auth off: no reservation.

Python API (for E2-free kinds and F1, so they never edit `jobs.py`):
```python
class JobRunner:
    async def submit(self, kind: str, payload: dict, *, principal: Principal | None, metadata: dict | None = None,
                     webhook: dict | None = None, priority: str = "normal", urgent: bool = False,
                     schedule: dict | None = None, expire_after_s: float | None = None,
                     idem_key: str | None = None) -> dict
        # validation, snapshot, model check, quota reservation, create, poke; raises ValueError (400),
        # PermissionError (403, urgent), QuotaExceeded (429), JobsFull (409). POST /v1/jobs uses it.
    async def cancel(self, job_id: str) -> dict
    def get(self, job_id: str) -> dict | None
class JobHandle:
    async def decide(self, reqs: list[SystemOneRequest]) -> list[Any]   # turn gate + decide_resilient
    async def add_items(self, items: list[dict], tokens: int = 0) -> None  # tokens -> jobs.tokens
```
Kind spec keys (added): `"completed_if_all_failed": bool` (default False; True = a run where every item failed
still ends `succeeded`, as OpenAI batches do). Existing keys unchanged.

HTTP: `POST /v1/jobs` gains `priority` (`low|normal|high`), `urgent` (bool), `schedule`, `expire_after_s`.
`JobInfo` gains `priority`, `urgent`, `schedule`, `waiting_reason` (`null | "off_hours" | "active_limit" |
"turn"`), `next_window_at` (ISO or null), `expired`, `model`, `tokens`. `GET /v1/jobs` gains `?priority=`.
Statuses are unchanged.

Env: `CLEF_JOB_TZ`, `CLEF_JOB_WINDOW`, `CLEF_JOB_ACTIVE`, `CLEF_JOB_AGING_S`. New dep `tzdata` in the `server`
extra. Tables: the E1 `jobs` columns in Storage.

Errors: 400 bad window / tz / priority; 403 `urgent` without the scope; 409 `CLEF_MAX_JOBS`; 429 quota.

Invariants: one job chunk in flight; rows stay contiguous `0..done-1` (F2's length sort persists each window in
input order); `recover()` unchanged; webhooks unchanged.

Supersedes when landed: "One runner, FIFO, one job at a time and ONE micro-batch in flight" in Async jobs (the
"one micro-batch in flight" part stays).

## OpenAI Files and Batches (planned, v3.2)

Module boundary: `batches.py` is a feature module listed in `FEATURES` before `jobs` (it registers the `batch` job
kind); it reaches the runner lazily through `ctx.extra["jobs"]` and only through the `JobRunner` / `JobHandle`
API (Job scheduler). `compat_openai.py` is split into pure `prepare_chat(body: dict, cfg) -> ChatPlan` (raises
`SchemaError`) and `render_chat(plan, result, request_id) -> dict`; the live route and the batch kind both use
them. Errors on these routes use the OpenAI shape (`route_class(ctx, openai_error)`). New dep `python-multipart`
in the `server` extra.

Tables (created by C1 in `0002`):
```sql
CREATE TABLE files (
    id         TEXT PRIMARY KEY,                  -- "file-" + 24 hex
    owner      TEXT,
    purpose    TEXT NOT NULL,                     -- "batch" (uploads) | "batch_output" (generated)
    role       TEXT NOT NULL,                     -- "input" | "output" | "error"
    filename   TEXT NOT NULL,
    bytes      BIGINT NOT NULL,                   -- input: on disk; output/error: computed when the job ends
    lines      INTEGER,
    job_id     TEXT REFERENCES jobs(id) ON DELETE CASCADE,   -- output/error files only
    created_at DOUBLE PRECISION NOT NULL,
    expires_at DOUBLE PRECISION
);
CREATE INDEX files_owner ON files(owner, created_at);
CREATE TABLE batches (
    id                TEXT PRIMARY KEY,           -- "batch_" + the job id's 20 hex
    job_id            TEXT NOT NULL UNIQUE REFERENCES jobs(id) ON DELETE CASCADE,
    owner             TEXT,
    endpoint          TEXT NOT NULL,              -- "/v1/chat/completions" | "/v1/classify" | "/v1/score"
    input_file_id     TEXT NOT NULL,
    output_file_id    TEXT NOT NULL,
    error_file_id     TEXT NOT NULL,
    completion_window TEXT NOT NULL,              -- "24h"
    metadata          TEXT,
    errors            TEXT,                       -- JSON list of {code, line, message, param} (status failed)
    created_at        DOUBLE PRECISION NOT NULL,
    cancelling_at     DOUBLE PRECISION
);
```
Input content lives on disk at `<state_dir>/files/<id>.jsonl` (path built only from a validated id, streamed to a
temp file with the size cap, then renamed). Output and error files are **virtual**: their content is streamed from
the job's rows. They live as long as the job (`CLEF_JOB_TTL_HOURS`); input files until `expires_at`
(`expires_after` or `CLEF_FILE_TTL_HOURS`).

Routes (scope `jobs`; owner-scoped, admins see all; integer unix-second timestamps):

| Route | Request | Response |
| --- | --- | --- |
| `POST /v1/files` | multipart: `file` (`.jsonl`), `purpose=batch`, optional `expires_after[anchor]=created_at`, `expires_after[seconds]` (3600..2592000) | FileObject |
| `GET /v1/files?after&limit=20&order=desc&purpose` | | `{"object": "list", "data": [FileObject], "first_id", "last_id", "has_more"}` |
| `GET /v1/files/{id}` | | FileObject |
| `DELETE /v1/files/{id}` | | `{"id", "object": "file", "deleted": true}`; output / error files: 400 (delete the batch's job instead) |
| `GET /v1/files/{id}/content` | | the JSONL bytes, `application/jsonl`, streamed |
| `POST /v1/batches` | `{"input_file_id", "endpoint", "completion_window": "24h", "metadata"?, "clef"?: {"priority", "urgent", "schedule"}}` | Batch |
| `GET /v1/batches/{id}` | | Batch |
| `GET /v1/batches?after&limit=20` | limit 1..100, newest first | cursor list as for files |
| `POST /v1/batches/{id}/cancel` | | Batch (`cancelling` or `cancelled`) |

FileObject: `{"id", "object": "file", "bytes", "created_at", "expires_at", "filename", "purpose", "status":
"processed", "status_details": null}`.

Batch: `{"id", "object": "batch", "endpoint", "errors", "input_file_id", "completion_window", "status",
"output_file_id", "error_file_id", "created_at", "in_progress_at", "expires_at", "finalizing_at", "completed_at",
"failed_at", "expired_at", "cancelling_at", "cancelled_at", "request_counts": {"total", "completed", "failed"},
"metadata", "model", "usage": {"input_tokens", "input_tokens_details": {"cached_tokens": 0}, "output_tokens": 0,
"output_tokens_details": {"reasoning_tokens": 0}, "total_tokens"}}`. `request_counts.completed = done - failed`;
`usage.input_tokens = jobs.tokens`; `model` = `spec.id`; `expires_at = eligible_at + 24h` (null before the window
opens). The SDK's `extra_body={"clef": {...}}` lands as the top-level `clef` field.

Mapping to jobs (one batch = one job of kind `batch`, visible in `GET /v1/jobs?kind=batch`; submitted through
`JobRunner.submit(..., expire_after_s=86400)`; the `batch` kind spec itself sets `"completed_if_all_failed": True`):
- Create validates at submit, in the threadpool: the file exists, belongs to the caller and has `purpose=batch`;
  every line is JSON with a unique `custom_id`, `method == "POST"` and `url == endpoint`; at most
  `min(CLEF_MAX_BATCH_LINES, CLEF_MAX_JOB_ITEMS)` lines. A structural failure creates the batch with status
  `failed` and `errors.data = [{code, line, message, param}]` (no job runs), as OpenAI does. A body that fails
  translation (`prepare_chat` / classify / score validation) becomes an error-file line, not a batch failure.
  `endpoint` outside the three: 400 `param: "endpoint"`. `stream: true` in a body: an error line.

| Job state | Batch `status` |
| --- | --- |
| validation failed (no job) | `failed` |
| `queued`, `running`, `interrupted` | `in_progress` (`cancelling` once a cancel was requested) |
| `succeeded` | `completed` |
| `failed` (systemic) | `failed`, `errors.data = [{"code": "server_error", "message"}]`; rows so far are still served |
| `cancelled`, `error = "expired"` | `expired` |
| `cancelled` | `cancelled` |

Rows: `{"index", "custom_id", "status_code", "body" | "error", "input_tokens"}`. Output file line =
`{"id": "batch_req_<20hex>_<index>", "custom_id", "response": {"status_code": 200, "request_id", "body"}, "error":
null}` for 2xx rows; the error file streams the non-2xx rows (OpenAI error body) plus, for a finished job,
`{"response": null, "error": {"code": "batch_expired" | "batch_cancelled", "message"}}` for every input line with
no row (custom ids re-read from the input file). Output order is input order.

Limits: `CLEF_MAX_FILE_MB` (body limit of `POST /v1/files` only, applied in `GuardMiddleware`),
`CLEF_MAX_BATCH_LINES`, `CLEF_FILE_TTL_HOURS`; `metadata` <= 16 pairs, keys <= 64 chars, values <= 512 chars;
`completion_window` must be `"24h"`. Rate limit (`ctx.limited`) on `POST /v1/files` and `POST /v1/batches`.
Quota: the batch reserves its line count at create (Job scheduler).

Errors (OpenAI shape): 400 `invalid_request_error` (bad purpose, endpoint, window, metadata, file not `.jsonl`),
403, 404 (unknown or foreign file / batch), 409 (`CLEF_MAX_JOBS`), 413 (file too large), 429
(`rate_limit_exceeded` / `insufficient_quota`).

Invariants: files are never in the DB; virtual files never exist on disk; deleting a batch's job deletes its batch
and output / error file rows; the batch path uses the same translation as the live routes, so probabilities match
`/v1/chat/completions`.

## Build ownership (planned, v3.2)

Files a workstream may create (C) or modify (M). Tests: new files per workstream (see the shared rules), plus the
existing tests a contract change forces. Every workstream may append to `CHANGELOG.md` and its own docs page;
`openapi.json` and `uv.lock` are regenerated by the coordinator after the wave merge.

| WS | Creates | Modifies |
| --- | --- | --- |
| C1 DB | `db.py`, `dbschema.py`, `migrations/**` | `jobs.py` (store port, `claim_next`, `jobs.model`, payload `model` default + submit check, `classifier_version`), `classifiers.py`, `classify.py` (`version` on `ClassifierRunRequest` / `ClassifierBatchRequest` only), `evaluation.py` (`version` and the server-set `classifier_version` next to `classifier` on `EvaluateRequest`, its `ctx.store.get` lookup, and the `classifier_version` line in `snapshot_job`), `main.py` (create_app: `open_database`, store wiring, `AppContext(...)` db/model fields; classifier routes; `/health` `db`; lifespan close), `appctx.py` (`db`, `model_ref`, `check_model`), `paths.py` (`database_path`, `jobs_db` kept for the legacy rename; the rename and its `PRAGMA` live in `db.py`), `config.py` (`CLEF_DATABASE_*`), `cli.py` (`db`, `backup`, `classifiers export`), `doctor.py` (database check), `pyproject.toml` (sqlalchemy, alembic, `postgres` extra), `docker-compose.yml` (comment), `docs/jobs.md`, `SECURITY.md` |
| C2 models | `models/__init__.py`, `models/base.py`, `models/registry.py`, `models/clef_release.py` | `engine.py`, `offload.py`, `backend.py` (TunableOp path per model, `fast_path` from capabilities), `paths.py` (`resolve_model_path(spec)`, `WEIGHTS_GB`), `config.py` (`CLEF_MODEL`, aliases), `schemas.py`, `classify.py`, `evaluation.py` (`model`), `compat_openai.py`, `compat_hf.py`, `main.py` (`infer` / `decide` / `prepare` region, `Cache-Control` read, `map_exception` UnknownModel, `/health` `model`, OpenAPI title, `/schema-example`), `appctx.py` (`prepare`, `decide_prepared`), `doctor.py` (model files from the adapter), `cli.py` (`download --model`, `models`), `pyproject.toml` (entry points), `clef_client/*`, `clients/js/*`, `static/screens/{playground,batch}.js`, `static/util.js`, `bench/parity.py` |
| D1 keys | `keystore.py`, `admin.py`, `static/screens/keys.js` | `security.py`, `main.py` (auth / rate-limit region, `ctx.scope` on core routes, quota in `infer`, `FEATURES` += `admin`, `AppContext(...)` D1 fields, 403 in `ERRORS`), `appctx.py`, `jobs.py` (scopes, admin view, `auth_enabled`, audit calls), `compat_openai.py`, `compat_hf.py`, `compat_errors.py` (403), `evaluation.py` (scopes), `cli.py` (`keys`), `config.py` (none expected), `static/app.js`, `static/api.js`, `static/components/topbar.js`, `docs/remote-access.md`, `SECURITY.md` |
| D2 perf | `bench/` scripts as needed | `models/clef_release.py` (forward, head), `backend.py` (`prepare_environment`), `config.py` (`CLEF_MASK_FREE`), `docs/performance.md`, `docs/hardware-results.md` |
| D3 cache | `cache.py` | `engine.py` (`prepare` / `submit` cache calls, `cache_stats`), `stats.py` (`cache` block), `config.py` (`CLEF_CACHE_SIZE`) |
| E1 scheduler | | `jobs.py`, `evaluation.py` (`job.decide`), `config.py` (`CLEF_JOB_*`), `pyproject.toml` (`tzdata`), `clef_client/*`, `clients/js/*` (job fields), `static/screens/batch.js` (job fields, optional), `docs/jobs.md` |
| E2 router | `router.py`, `llm.py`, `static/screens/review.js` | `main.py` (`FEATURES` += `router`, low-confidence capture in the classifier-run routes), `classify.py` (`ClassifierDef.review_threshold`), `config.py` (`CLEF_LLM_*`, `CLEF_ROUTER_*`, `CLEF_REVIEW_TTL_DAYS`), `doctor.py` (LLM key check), `static/app.js` (tab), `clef_client/*`, `clients/js/*` (router helpers), new `docs/router.md` |
| F1 batches | `batches.py` | `compat_openai.py` (`prepare_chat` / `render_chat`), `main.py` (`FEATURES` += `batches`, `GuardMiddleware` file limit), `config.py` (`CLEF_MAX_FILE_MB`, `CLEF_MAX_BATCH_LINES`, `CLEF_FILE_TTL_HOURS`), `pyproject.toml` (`python-multipart`), `docs/openai-compat.md` |
| F2 throughput | | `jobs.py` (window sort in `run_chunked`, prefetch via `ctx.prepare`), `evaluation.py` (same for the evaluate kind), `engine.py` only if `Engine.prepare` / `submit` need a fix, `config.py` (`CLEF_JOB_SORT_WINDOW`) |

Same-wave shared files and how they split:

| Wave | File | Split |
| --- | --- | --- |
| C | `main.py` | C1: `create_app` store / DB construction, the `AppContext(...)` call, classifier routes, `/health` `db` key, lifespan close. C2: `map_exception`, `infer` / `decide` / `prepare`, `/health` `model` key, title, `/schema-example`. Distinct hunks; the two `/health` keys are adjacent lines (keep both). C2 passes its new ctx fields by appending keyword arguments at the end of the `AppContext(...)` call |
| C | `appctx.py` | append-only: C1 adds `db`, `model_ref`, `check_model`; C2 adds `prepare`, `decide_prepared`. The `model_ref` / `check_model` hook lets C1 store and validate models in `jobs.py` without C2 touching it (C2 implements `Engine.model_ref` / `check_model`; fakes fall back) |
| C | `jobs.py` | C1 only. C2 must not edit it; the model default and the submit 404 are C1's, through `ctx.check_model` |
| C | `classify.py`, `evaluation.py` | C1: only the classifier `version` field (`ClassifierRunRequest`, `ClassifierBatchRequest`, `EvaluateRequest`, plus `classifier_version` on `EvaluateRequest`), the versioned `store.get(name, version)` lookup and the `classifier_version` line of the evaluate `snapshot_job`. C2: the `model` fields / defaults and the evaluate `model` check. Different fields; each adds its field on its own line |
| C | `config.py`, `pyproject.toml` | append-only blocks (`# C1`, `# C2`); different sections of `pyproject.toml` (extras vs `[project.entry-points]`) |
| C | `paths.py` | C1: state-dir / DB helpers (new functions). C2: `resolve_model_path`, `WEIGHTS_GB`, `MODEL_MARKER`. Different functions |
| C | `cli.py` | C1: new `db`, `backup`, `classifiers` subparsers and `cmd_*` functions. C2: the `download` parser + `cmd_download`, new `models` subparser. New subparsers are appended in `build_parser` in that order (C1 block, then C2 block) |
| C | `doctor.py` | C1 appends a `database` check; C2 changes the model-file / weights checks. Different functions |
| D | `main.py` | D1 only. D3 does not touch it (the cache lives in the engine; the bypass header is plumbed by C2) |
| D | `engine.py` vs `models/clef_release.py` | D3 owns `engine.py`, D2 owns the adapter. D2 needs no engine change: `batch["lengths"]` and `CLEF_MASK_FREE` are read inside the adapter |
| D | `backend.py` | D2 only |
| D | `config.py` | append-only (`# D1`, `# D2`, `# D3`) |
| D | `stats.py` | D3 only |
| E | `jobs.py` | E1 only. E2 needs no job kind (re-routing runs in-process through `ctx.decide`) |
| E | `main.py` | E2 only |
| E | `config.py`, `clef_client/*`, `clients/js/*` | append-only; E1 adds job fields to the existing job methods, E2 adds new router / review methods at the end of each client class |
| E | `evaluation.py` | E1 only (E2 does not touch it) |
| F | `jobs.py`, `evaluation.py`, `engine.py` | F2 only. F1 uses only `JobRunner.submit/cancel/get`, `JobHandle.decide/add_items` and the `completed_if_all_failed` kind key, all delivered by E1 |
| F | `main.py`, `compat_openai.py`, `pyproject.toml` | F1 only |
| F | `config.py` | append-only (`# F1`, `# F2`) |
| all | Alembic heads | no new revision expected after C1; if two appear in one wave, the coordinator adds `alembic merge` (startup runs `upgrade heads`) |
| all | `docs/ARCHITECTURE.md` | each workstream edits only its own planned section; the coordinator removes markers and applies the "Supersedes" lines |

## Contract decisions (planned, v3.2)

Confirmed by the user (2026-10-17):
1. **Unknown model (C2):** 404 `model_not_found` on every route except the HF path id, which stays ignored. This
   replaces the `clef.ignored` behaviour of `/v1/chat/completions`; update `test_compat.py`.
2. **Classifier owner (C1):** enforced. Only the owner or an admin may PUT a new version or DELETE (403 otherwise);
   anyone with `classify` may run it.
3. **Cache hits and quotas (D1/D3):** hits do **not** count. Only computed records are charged.
4. **Multi-day jobs (E1):** the whole job is reserved on the submit day; the unprocessed part is refunded to that
   day on cancel / fail / expiry.
5. **Paused jobs (E1):** a running job outside its window stays `running` with `waiting_reason: "off_hours"`.
6. **Human labels (E2):** the re-check runs at resolve time, before the text is purged.
7. **Low-confidence sources (E2):** routers, plus saved classifiers with `review_threshold`, on interactive
   single-label runs only.
8. **Privacy (E2):** setting `GEMINI_API_KEY` is the consent to send up to 500 characters per buffered input. No
   second switch.
9. **`clef backup` on Postgres (C1):** exit 2 with a `pg_dump` hint.
10. **27B pin (C2):** C2 fetches the `Cloudflare/clef` revision sha from the HF API; doctor warns it fits 24 GB only
    with nf4.
11. **Fair-share weights (E1):** 1 for every principal; no weight column.
