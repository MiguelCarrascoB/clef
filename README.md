# clef-flash local (AMD RX 7900 XTX / WSL2 ROCm)

[Cloudflare/clef-flash](https://huggingface.co/Cloudflare/clef-flash) running 100% locally: a 9B multimodal
**decision model** served by FastAPI on an RX 7900 XTX through ROCm 7.2 in WSL2 (ROCDXG).

It reads a `state` (text, JSON, images, videos) plus typed questions (`choice`, `score`, `noul`) and returns a
calibrated probability for every option of every question in **one forward pass** - no text generation.
Concurrent requests are micro-batched on a single GPU worker. API is Jev / SystemOne compatible.

## Quick start (Windows PowerShell)

```powershell
.\clef.ps1 doctor     # preflight: GPU, pins, model files, env
.\clef.ps1 start      # detached in WSL, waits until the model is loaded (~45 s + warmup)
.\clef.ps1 status     # table from /health
.\clef.ps1 open       # the Console: http://localhost:8910/
.\clef.ps1 test       # unit tests + integration tests against the live server
.\clef.ps1 logs       # tail the server log (Ctrl+C to leave)
.\clef.ps1 stop
```

`bench [http]` runs the benchmarks (see below). From inside WSL the same commands are
`bash scripts/launch_server.sh start|stop|restart|status|logs`.

## Console

`http://localhost:8910/` serves the Clef Console (request playground, live stats, request log).

## API

`POST /v1/systemone`, `POST /v1/batch`, `GET /health`, `GET /livez`, `GET /v1/stats`, `GET /v1/log`,
`GET /v1/events` (SSE). Full contract (schemas, limits, errors, timing fields):
[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md). Interactive docs at `/docs`.

```powershell
$body = @{
  model = 'clef-flash'; state = 'Our checkout started returning errors and orders are blocked.'
  questions = @{
    department = @{ type = 'choice'; instructions = 'Which team should handle the message?'
                    criteria = @{ billing = 'Payments or invoices'; technical = 'Bugs or outages' } }
    outage     = @{ type = 'noul'; instructions = 'Is a service down?' }
  }
} | ConvertTo-Json -Depth 6
Invoke-RestMethod http://localhost:8910/v1/systemone -Method Post -Body $body -ContentType 'application/json'
```

Answers: `choice` -> `choice`, `confidence`, `probabilities`; `score` -> expected `score`, `confidence`, `legend`,
`probabilities`; `noul` -> `noul` (P(true)). Images/videos go in as `data:` URLs (`http(s)` only with
`CLEF_ALLOW_URL_FETCH=1`); all media of a request is evaluated together in one record.

## Python client

```python
from clef_client import ClefClient, image_to_data_url

c = ClefClient("http://127.0.0.1:8910", api_key=None)  # AsyncClefClient has the same methods
out = c.decide(
    "A photo is attached.",
    {"red": {"type": "noul", "instructions": "Does any image show a red square?"}},
    images=[image_to_data_url("photo.png")],
)
print(out["answers"]["red"]["noul"], out["timing"])
c.batch([{"model": "clef-flash", "state": "...", "questions": {...}}])
c.health()
c.stats()  # errors raise ClefError(status, detail, request_id)
```

## Configuration

Server settings are `CLEF_*` environment variables read by `server/config.py`; shell defaults live in
`scripts/env.sh` (single source of truth, sourced by every script). Everything is overridable.

| Variable | Default | Meaning |
| --- | --- | --- |
| `CLEF_HOME` | repo root | repo location (env.sh) |
| `CLEF_VENV` | `~/venvs/clef` | Python venv (env.sh) |
| `CLEF_MODEL_PATH` | `~/models/clef-flash` | model directory |
| `CLEF_LOG_DIR` | `~/.local/state/clef` | `server.log` (+ `.1`-`.3`) and `server.pid` (env.sh) |
| `CLEF_HOST` / `CLEF_PORT` | `127.0.0.1` / `8910` | bind address |
| `CLEF_API_KEY` | unset | require `X-API-Key` / Bearer on `/v1/*` |
| `CLEF_DEVICE` / `CLEF_DTYPE` | `cuda` / `bfloat16` | |
| `CLEF_MAX_TOKENS` | 16384 | max tokens per record |
| `CLEF_MAX_BODY_MB` | 64 | request body limit |
| `CLEF_MAX_IMAGES` / `CLEF_MAX_VIDEOS` | 8 / 2 | media per request |
| `CLEF_MAX_PIXELS` | 1048576 | images above this are downscaled |
| `CLEF_MAX_FRAMES` / `CLEF_VIDEO_FPS` | 32 / 2.0 | video sampling |
| `CLEF_MAX_BATCH` / `CLEF_MAX_QUESTIONS` | 64 / 64 | request caps |
| `CLEF_ALLOW_URL_FETCH` / `CLEF_URL_FETCH_MAX_MB` | 0 / 32 | remote media (SSRF-guarded) |
| `CLEF_MAX_MICROBATCH` / `CLEF_BATCH_WINDOW_MS` | 8 / 4.0 | micro-batching |
| `CLEF_BUCKETS` | `128,...,4096` | length groups for micro-batching (and warmup shapes) |
| `CLEF_PAD_MULTIPLE` | 64 | pad text batches to a multiple of this (aligned GEMMs are faster) |
| `CLEF_PAD_TO_BUCKET` | 0 | pad to the full bucket instead (wastes compute; for experiments) |
| `CLEF_WARMUP` | 1 | warm every bucket at startup |
| `CLEF_LOG_STATE` / `CLEF_LOG_BUFFER` | 0 / 1000 | request log (state preview off by default) |
| `CLEF_START_TIMEOUT` | 300 | seconds `launch_server.sh start` waits (env.sh) |
| `CLEF_TUNABLEOP` | 0 | `1` enables PyTorch TunableOp (results cached in `~/.cache/clef`) |
| `HSA_ENABLE_DXG_DETECTION` / `HSA_OVERRIDE_GFX_VERSION` | `1` / `11.0.0` | ROCm on WSL |
| `HF_HUB_OFFLINE` | 1 | never touch the network |
| `TRITON_CACHE_DIR`, `TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL` | see env.sh | perf defaults |

## Benchmarks

Measured on the RX 7900 XTX (ROCm 7.2, WSL2, bf16), v2 server, ~330-token records, 100 requests per level:

| HTTP concurrency | req/s | p50 ms | p95 ms | avg server batch |
| --- | --- | --- | --- | --- |
| 1 | 7.0 | 142 | 150 | 1.0 |
| 8 | 8.2 | 970 | 985 | 6.1 |
| 16 | 8.9 | 1756 | 1931 | 7.8 |

v1 (same card): single-record p50 162 ms / p95 222 ms. Warm restart: model load ~45-100 s (page cache), warmup
~12 s with a warm Triton cache (~110 s the very first time).

What the profiler says (`bench/profile_forward.py`): GEMMs are ~83% of GPU kernel time and already run at
~95 TFLOPS, so the card is compute-bound and batching adds little throughput (6.9 -> 8.9 req/s). Single-record
forwards are ~50% host overhead (75 ms of kernels in ~150 ms); HIP-graph capture would remove most of it but
is blocked by a device sync in transformers' SDPA masking (`masking_utils._ignore_causal_mask_sdpa`). The joint
head is ~5% of the forward. Padding to multiples of 64 beats both raw lengths and full buckets (GEMM tile
alignment), see `CLEF_PAD_MULTIPLE`.

- `bench/http_bench.py` - realistic: async load against the live server at concurrency 1/2/4/8/16, throughput,
  p50/p95/p99 and the server-reported batch-size distribution (exercises micro-batching).
  `.\clef.ps1 bench http --requests 100`
- `bench/bench.py` - in-process: forward-only vs end-to-end, p50/p90/p99, batch sizes, mixed lengths, token
  counts and TFLOPS estimate. Needs the GPU, so stop the server first. `.\clef.ps1 bench --mixed`
- `bench/profile_forward.py` - torch.profiler: top kernels and GEMM/attention/linear-attn/conv grouping, chrome trace
  in `bench/out/`.

## Testing

```powershell
.\clef.ps1 test        # = unit (tests/unit, no GPU/weights) + integration (tests/integration, live server)
```

Integration tests (`pytest -m gpu`) use `CLEF_URL` / `CLEF_API_KEY` and skip when no server answers. CI runs
ruff, shellcheck and the unit tests on CPU.

## Service / autostart

Optional systemd user service (WSL needs `systemd=true`, see `deploy/wsl.conf.sample`):

```bash
bash scripts/install_service.sh      # copies deploy/clef.service, daemon-reload, enable (does not start)
systemctl --user start clef          # journalctl --user -u clef -f
```

Overrides (API key, port) go in `~/.config/clef/env` (`KEY=value` per line). `deploy/wslconfig.sample` is the
Windows-side `.wslconfig` (memory=26GB for a 32 GB host).

## Setup from scratch

As root in WSL: `bash scripts/wsl_setup_root.sh` (ROCm 7.2.4 + librocdxg 1.2.2; versions overridable). As
user: `bash scripts/wsl_setup_user.sh` (idempotent: clones weights only if missing at the pinned
`CLEF_MODEL_REV`, creates the venv if missing, `pip install -r requirements/server.txt`). The venv is
Python 3.10; `requirements/server.txt` holds exact pins (torch 2.11.0+rocm7.2, transformers 5.10.2,
flash-linear-attention 0.5.2). Dev tools: `CLEF_INSTALL_DEV=1` or `pip install -r requirements/dev.txt`.

## Troubleshooting

Run `.\clef.ps1 doctor` first.

- **`No CUDA GPUs are available`**: `HSA_ENABLE_DXG_DETECTION=1` must be set (env.sh does; required until
  ROCm 7.13+ in WSL). Update the Windows Adrenalin driver if `/dev/dxg` is missing.
- **First forwards are slow (seconds to minutes)**: one-time Triton/SDPA kernel compilation per input shape.
  Startup warmup covers the length buckets; caches live in `~/.triton/cache`.
- **OOM in WSL**: `%USERPROFILE%\.wslconfig` `memory=26GB`; applies after `wsl --shutdown`.
- **`start` says something already answers on the port**: a server not started by `launch_server.sh` (no pidfile)
  holds it; stop that process or use another port (`.\clef.ps1` honours `$env:CLEF_PORT`).
- **bench refuses to run**: the server holds ~19.6 GB of VRAM; `.\clef.ps1 stop` first (or `--force`).
- **causal-conv1d**: its HIP build does not work on WSL-ROCm yet; `flash-linear-attention` is active and the
  torch fallback is already fast.
- Weights live only in the WSL filesystem (fast safetensors I/O, no double storage).

## Layout

```
clef.ps1            Windows wrapper (start/stop/status/logs/test/bench/doctor/open)
server/             FastAPI app: main, schemas, media, stats, engine, config, static/ (Console)
clef_client/        sync + async Python client
scripts/            env.sh, launch_server.sh, run_server.sh, doctor.py, setup + service scripts, model_smoke.py
deploy/             clef.service, wsl.conf.sample, wslconfig.sample
requirements/       server.txt (exact pins), dev.txt
bench/              bench.py, http_bench.py, profile_forward.py
tests/              unit/ (CPU) and integration/ (live server, marker gpu)
docs/ARCHITECTURE.md  API + module contract
```

## License

Model: Apache-2.0 (Cloudflare clef-flash, base model Qwen/Qwen3.5-9B).
