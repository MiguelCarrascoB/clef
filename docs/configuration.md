# Configuration

All settings are `CLEF_*` environment variables; the flags of `clef serve` set the same variables. Values are read
once at startup, so restart the server after changing them. The tables below are checked against `config.py`; the
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
| `CLEF_STATE_DIR` | per OS | logs, pidfile, saved classifiers and the jobs database (`jobs.db`) |

## Limits

| Variable | Default | Meaning |
| --- | --- | --- |
| `CLEF_MAX_LABELS` | `64` | labels per classify request / saved classifier |
| `CLEF_CLASSIFY_THRESHOLD` | `0.5` | default multi-label threshold (also for array properties in `/v1/chat/completions`) |
| `CLEF_MAX_BATCH` / `CLEF_MAX_QUESTIONS` | `64` / `64` | records per batch / questions per record |
| `CLEF_MAX_BODY_MB` | `64` | request body limit |
| `CLEF_MAX_TOKENS` | `16384` | input tokens per record |
| `CLEF_MAX_EVAL_ROWS` | `500` | rows per `POST /v1/evaluate` request |
| `CLEF_MAX_JOB_EVAL_ROWS` | `100000` | rows per `evaluate` job and per `POST /v1/evaluate/metrics` request |

The media limits (`CLEF_MAX_IMAGES`, `CLEF_MAX_VIDEOS`, `CLEF_MAX_PIXELS`, `CLEF_MAX_FRAMES`, `CLEF_VIDEO_FPS`,
`CLEF_ALLOW_URL_FETCH`, `CLEF_URL_FETCH_MAX_MB`) and the batching knobs (`CLEF_MAX_MICROBATCH`,
`CLEF_BATCH_WINDOW_MS`, `CLEF_BUCKETS`, `CLEF_WARMUP`) are in [Architecture](ARCHITECTURE.md#configuration-configpy).

## Jobs and webhooks

See [Jobs & webhooks](jobs.md).

| Variable | Default | Meaning |
| --- | --- | --- |
| `CLEF_MAX_JOB_ITEMS` | `100000` | items per `classify` / `score` / `systemone` job |
| `CLEF_MAX_JOBS` | `100` | queued + running jobs; more is `409` with `Retry-After` |
| `CLEF_JOB_TTL_HOURS` | `168` | finished jobs and their rows are purged this long after they finish; `0` keeps them forever |
| `CLEF_JOB_ENGINE_WAIT_S` | `600` | how long a job waits for an unavailable engine before failing |
| `CLEF_WEBHOOK_ALLOW` | empty | hosts (`hooks.example.com`, `*.corp.example.net`), IPs and CIDRs webhooks may be sent to. **Empty = webhooks off** |
| `CLEF_WEBHOOK_TIMEOUT_S` | `10` | total timeout of one delivery attempt |
| `CLEF_WEBHOOK_ATTEMPTS` | `5` | delivery attempts for final events |

## Observability

| Variable | Default | Meaning |
| --- | --- | --- |
| `CLEF_TELEMETRY` | `1` | GPU telemetry via pynvml / amdsmi / rocm-smi when available |
| `CLEF_LOG_STATE` | `0` | store a 200-character preview of each request's input in the request log |
| `CLEF_LOG_BUFFER` | `1000` | request log ring buffer size |
| `CLEF_SAMPLE_INTERVAL_S` | `2` | gauge sampling period for the console charts |

Example for LAN access:

```bash
CLEF_HOST=0.0.0.0 CLEF_API_KEYS=~/.config/clef/keys CLEF_RATE_LIMIT=120 clef serve
```

See [remote access](remote-access.md) before exposing the server beyond the machine.
