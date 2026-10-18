# Configuration

All settings are `CLEF_*` environment variables; the flags of `clef serve` set the same variables. Values are read
once at startup, so restart the server after changing them. They can also live in a [`.env` file](#env-files). The tables below are checked against `config.py`; the
same list with the engine and padding knobs is in [Architecture](ARCHITECTURE.md#configuration-configpy).

## Model and device

| Variable | Default | Meaning |
| --- | --- | --- |
| `CLEF_DEVICE` | `auto` | `auto`, `cuda`, `rocm`, `mps`, `cpu` (`cuda:1` / `rocm:1` select an index). `--device` |
| `CLEF_DTYPE` | `auto` | `auto`, `bfloat16`, `float16`, `float32`. `--dtype` |
| `CLEF_MODEL_PATH` | unset | release dir. Unset: `~/models/clef-flash` if it holds a release, else the pinned HF cache snapshot. `--model-path` |
| `CLEF_QUANT` | `none` | `none`, `int8`, `nf4`. int8 uses torchao on ROCm / MPS / CPU and bitsandbytes on CUDA; nf4 is bitsandbytes only and not recommended. `--quant` |
| `CLEF_QUANT_BACKEND` | `auto` | `auto`, `bnb`, `torchao`: the library behind `CLEF_QUANT` |
| `CLEF_OFFLOAD` | `none` | `cpu`: keep the embeddings and the layers that do not fit in host RAM and stream them in per forward (CUDA and ROCm; no effect on MPS / CPU). `--offload` |
| `CLEF_MAX_DEVICE_MEMORY_GB` | `0` | device memory the process may allocate, as a hard limit; `0` = no cap. Use `N - 1` on an `N` GB card. `--max-device-memory-gb` |
| `CLEF_PREFLIGHT` | `1` | check free memory before loading |
| `CLEF_PAD_MULTIPLE` | `64` | pad text batches to a multiple of this (measured on ROCm only) |

See [Smaller GPUs](memory.md) for which combination fits which card.

## Network, keys and state

| Variable | Default | Meaning |
| --- | --- | --- |
| `CLEF_HOST` / `CLEF_PORT` | `127.0.0.1` / `8910` | bind address. A non-loopback host without keys logs a loud warning |
| `CLEF_API_KEY` / `CLEF_API_KEYS` | unset | one key, or `name:key,name2:key2` / a path to a file with one `name:key` per line |
| `CLEF_RATE_LIMIT` | `0` | requests per minute per key (per client IP when auth is off); 0 = off |
| `CLEF_CORS_ORIGINS` | empty | comma-separated allowed origins (`*` allowed); empty = CORS off |
| `CLEF_DRAIN_S` | `0` | seconds to keep serving after the first SIGTERM / Ctrl-C while `/readyz` answers 503 `draining`, so a load balancer stops routing first; a second signal stops at once. `0` = stop at once. Behind a load balancer use e.g. `5` (below Docker's 10 s stop timeout). `--drain-s` |
| `CLEF_STATE_DIR` | per OS | logs, pidfile and the database `clef.db` (jobs, saved classifiers, audit log; unless `CLEF_DATABASE_URL` is set) |

## Limits

| Variable | Default | Meaning |
| --- | --- | --- |
| `CLEF_MAX_LABELS` | `64` | labels per classify request / saved classifier |
| `CLEF_CLASSIFY_THRESHOLD` | `0.5` | default multi-label threshold (also for array properties in `/v1/chat/completions`) |
| `CLEF_MAX_BATCH` / `CLEF_MAX_QUESTIONS` | `64` / `64` | records per batch / questions per record |
| `CLEF_MAX_BODY_MB` | `64` | request body limit |
| `CLEF_MASK_FREE` | `0` | `1`: text-only batches call the backbone with `attention_mask=None` and the head uses CPU lengths (no device sync). Opt-in until `bench/perf_check.sh` passes on your GPU; see [Performance](performance.md) |
| `CLEF_FAST_HEAD` | `0` | `1`: use the batch-vectorised, sync-free joint head (`fast_joint_head`) instead of the release's per-record loops. Same maths, opt-in; see [Performance](performance.md) |
| `CLEF_JOB_SORT_WINDOW` | `0` | `N > 0`: a job sorts N items at a time by token length (256-1024 recommended) before sending them to the engine; results are persisted per window in input order. Opt-in until `bench/sort_check.sh` shows a gain over 10% on your GPU; see [Performance](performance.md) |
| `CLEF_TOKENIZE_AHEAD` | `0` | `1`: the next window of a job is tokenized on CPU threads while the current one is on the GPU. Same opt-in rule as the sort |
| `CLEF_MAX_TOKENS` | `16384` | input tokens per record |
| `CLEF_MAX_EVAL_ROWS` | `500` | rows per `POST /v1/evaluate` request |
| `CLEF_CACHE_SIZE` | `10000` | result-cache entries (in-memory LRU, lost on restart); `0` = off. Also bounded by `CLEF_CACHE_MB`. Interactive calls, jobs and evaluate share it; media records are never cached; send `Cache-Control: no-cache` to skip the lookup. Cache hits do not count toward quotas. Stats: `cache` block of `GET /v1/stats` |
| `CLEF_CACHE_MB` | `256` | result-cache byte budget (estimated as the stored JSON size, typically about 1-2 KB per entry, so 10000 entries are about 10-20 MB); LRU entries are evicted until it fits, and an entry larger than a tenth of the budget is not stored; `0` = off |
| `CLEF_MAX_JOB_EVAL_ROWS` | `100000` | rows per `evaluate` job and per `POST /v1/evaluate/metrics` request |

The media limits (`CLEF_MAX_IMAGES`, `CLEF_MAX_VIDEOS`, `CLEF_MAX_PIXELS`, `CLEF_MAX_FRAMES`, `CLEF_VIDEO_FPS`,
`CLEF_ALLOW_URL_FETCH`, `CLEF_URL_FETCH_MAX_MB`) and the batching knobs (`CLEF_MAX_MICROBATCH`,
`CLEF_BATCH_WINDOW_MS`, `CLEF_BUCKETS`, `CLEF_WARMUP`) are in [Architecture](ARCHITECTURE.md#configuration-configpy).

## Jobs and webhooks

See [Jobs & webhooks](jobs.md).

| Variable | Default | Meaning |
| --- | --- | --- |
| `CLEF_MAX_JOB_ITEMS` | `100000` | items per `classify` / `score` / `systemone` job |
| `CLEF_MAX_FILE_MB` | `200` | body limit of `POST /v1/files` (OpenAI batch uploads); other routes use `CLEF_MAX_BODY_MB` |
| `CLEF_MAX_BATCH_LINES` | `50000` | requests per OpenAI batch (also capped by `CLEF_MAX_JOB_ITEMS`) |
| `CLEF_FILE_TTL_HOURS` | `720` | how long an uploaded batch file is kept (`0` = forever) |
| `CLEF_MAX_JOBS` | `100` | queued + running jobs; more is `409` with `Retry-After` |
| `CLEF_JOB_TTL_HOURS` | `168` | finished jobs and their rows are purged this long after they finish; `0` keeps them forever |
| `CLEF_JOB_ENGINE_WAIT_S` | `600` | how long a job waits for an unavailable engine before failing |
| `CLEF_JOB_ACTIVE` | `4` | jobs that take turns at the engine at once (`>= 1`; `1` = strictly one after another). Shared fairly: when another key waits, a key holding more than its share lets its newest job wait ([jobs](jobs.md#scheduling)) |
| `CLEF_JOB_AGING_S` | `3600` | a waiting job rises one priority class per this many seconds without a turn (never to `urgent`); `0` = no aging |
| `CLEF_JOB_TZ` | `UTC` | IANA zone of job windows that give no `tz`, and of `CLEF_JOB_WINDOW` |
| `CLEF_JOB_WINDOW` | empty | default off-hours window `HH:MM-HH:MM` for jobs without `schedule`; empty = always open |
| `CLEF_INTERACTIVE_BURST` | `8` | interactive micro-batches in a row before waiting job work gets one (the engine's priority lane; `>= 1`) |
| `CLEF_WEBHOOK_ALLOW` | empty | hosts (`hooks.example.com`, `*.corp.example.net`), IPs and CIDRs webhooks may be sent to. **Empty = webhooks off** |
| `CLEF_WEBHOOK_TIMEOUT_S` | `10` | total timeout of one delivery attempt |
| `CLEF_WEBHOOK_ATTEMPTS` | `10` | delivery attempts for final events |

## Observability

| Variable | Default | Meaning |
| --- | --- | --- |
| `CLEF_TELEMETRY` | `1` | GPU telemetry via pynvml / amdsmi / rocm-smi when available |
| `CLEF_LOG_STATE` | `0` | store a 200-character preview of each request's input in the request log |
| `CLEF_LOG_BUFFER` | `1000` | request log ring buffer size |
| `CLEF_SAMPLE_INTERVAL_S` | `2` | gauge sampling period for the console charts |
| `CLEF_LOG_FORMAT` | `text` | `text` or `json`. json writes the app, uvicorn and access log lines as one JSON object per line (`ts`, `level`, `logger`, `msg`, plus `request_id` inside a request and `exc` for a traceback). `--log-format` |

## Router and review

The router is off unless a key is set in the environment or an [env file](#env-files) ([Router and Review](router.md)).

| Variable | Default | Meaning |
| --- | --- | --- |
| `GEMINI_API_KEY` | unset | the router's LLM key. Unset (and no `CLEF_LLM_API_KEY`) = off. Never logged |
| `CLEF_LLM_API_KEY` | unset | key for another OpenAI-compatible provider; wins over `GEMINI_API_KEY` |
| `CLEF_LLM_BASE_URL` | `https://generativelanguage.googleapis.com/v1beta/openai/` | OpenAI-compatible endpoint (ollama, LM Studio, OpenAI) |
| `CLEF_LLM_MODEL` | `gemini-3.8-flash` | model id sent to the endpoint |
| `CLEF_LLM_TIMEOUT_S` | `60` | timeout of one LLM request (2 retries on 429 / 5xx) |
| `CLEF_ROUTER_BATCH` | `20` | buffered items per proposal call |
| `CLEF_ROUTER_COLD_BATCH` | `20` | buffered items before the cold-start call |
| `CLEF_ROUTER_FLUSH_S` | `600` | call after this long even below the batch size (0 = never) |
| `CLEF_ROUTER_MODE` | `review` | default mode of new routers: `review` or `auto` |
| `CLEF_ROUTER_MAX_NEW_PER_DAY` | `10` | auto mode: labels a router may add per UTC day |
| `CLEF_ROUTER_MAX_INPUT_CHARS` | `500` | characters of each buffered input sent to the LLM |
| `CLEF_REVIEW_TTL_DAYS` | `30` | review rows lose their input text this long after creation (and at once when resolved); 0 = keep |

## .env files

Instead of exporting variables, put `KEY=VALUE` lines in a file. At startup, before the settings are read, clef loads
`./.env` (the directory you run `clef serve` in) and then `~/.config/clef/env` (the file `deploy/clef.service` also
reads):

```bash
# .env  (keep it out of git; clef's own .gitignore lists it)
GEMINI_API_KEY=your-key
CLEF_API_KEYS=~/.config/clef/keys
```

- A variable that is already set (in the shell, by a flag, by systemd) always wins over both files; `./.env` wins
  over `~/.config/clef/env`. A variable set to an empty value counts as unset, so the file value is used.
- Only `CLEF_*` names and `GEMINI_API_KEY` are read from these files. Anything else (`PATH`, `HF_HOME`,
  `LD_PRELOAD`...) is ignored, with one warning that names the ignored variables (never their values); set those
  in the real environment.
- Blank lines and `#` comments are ignored. `export KEY=value` works. Values may be `'single'` (literal) or
  `"double"` quoted (`\n`, `\"`, `\\` escapes); an unquoted value ends at ` #`.
- `CLEF_ENV_FILE=/path/to/file` loads only that file. `CLEF_ENV_FILE=` (empty) loads none. It is read from the real
  environment only, once per process: a `CLEF_ENV_FILE` line inside an env file is ignored.
- clef never logs values. `clef serve` logs one line at startup naming the env files it loaded (`env files loaded:
  ~/work/.env`, or `env files: none loaded`), so a `./.env` picked up from an unexpected directory is visible;
  `clef doctor` lists them too.

## Example

Example for LAN access:

```bash
CLEF_HOST=0.0.0.0 CLEF_API_KEYS=~/.config/clef/keys CLEF_RATE_LIMIT=120 clef serve
```

See [remote access](remote-access.md) before exposing the server beyond the machine.
