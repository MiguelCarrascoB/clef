# Changelog

## 3.0.0

Cross-platform release: the model now runs on NVIDIA (CUDA), AMD (ROCm), Apple Silicon (MPS) and CPU. Only the
Windows + WSL2 + ROCm path has been run on real hardware; CUDA and MPS are unit-tested with mocked backends and marked
"untested on hardware" until `docs/hardware-validation.md` has been run. Quantization accuracy and latency are pending
hardware.

### Added
- **Backends** (`clef_server/backend.py`): `CLEF_DEVICE=auto` detects cuda, rocm, mps or cpu; per-backend dtype rules
  with warnings and fallbacks, memory stats, optional GPU telemetry (pynvml, amdsmi, rocm-smi), a preflight memory
  check before loading, and `backend`, `fast_path` and `versions` in `/health`.
- **`CLEF_QUANT=int8|nf4`** (bitsandbytes, CUDA only, refused elsewhere) for 16 GB NVIDIA GPUs.
- **`clef` CLI**: `serve` (foreground or `--detach`, health-gated), `stop`, `status`, `logs`, `doctor` (`--no-gpu`,
  `--smoke`, `--json`), `download` (pinned, resumable, disk-space check), `bench`, `open`, `version`.
- **Classification API**: `POST /v1/classify` (single and multi-label), `/v1/classify/batch`, `/v1/score`, and saved
  classifiers (`PUT|GET|DELETE /v1/classifiers/{name}`, `POST /v1/classifiers/{name}[/batch]`). Same engine and
  probabilities as `/v1/systemone`.
- **Clients**: typed Python client (sync + async, retries on 503) and a dependency-free JavaScript/TypeScript client
  (`clients/js`); `examples/`; `openapi.json` exported from the app and checked in CI.
- **Remote access**: named API keys (`CLEF_API_KEYS`), per-key rate limit (`CLEF_RATE_LIMIT`, 429 + `Retry-After`),
  opt-in CORS (`CLEF_CORS_ORIGINS`), key names in `/v1/log` and `/v1/stats`, warning when binding a non-loopback host
  without keys. Docs: `docs/remote-access.md` (LAN setup, firewall, Caddy TLS).
- **Console**: time-series charts, KPI tiles, histograms, Classify mode, redesigned light/dark theme.
  `GET /v1/stats/timeseries` and a live `point` in the SSE `stats` event.
- **Packaging**: installable package (`src/clef_server`, `src/clef_client`), per-backend lock files
  (`requirements/{rocm,cuda,cpu,macos}.txt`), Dockerfiles for CUDA and ROCm, compose file, systemd/launchd/Task
  Scheduler service files.
- **CI/release**: matrix on Ubuntu, macOS and Windows (ruff, unit tests, wheel build, CLI smoke from the wheel),
  JS tests, shellcheck, build-only Docker jobs; a `v*` tag attaches the wheel, sdist and `openapi.json` to a GitHub
  Release. Nothing is published to PyPI, GHCR or npm.
- **Repo**: Apache-2.0 `LICENSE`, `CONTRIBUTING.md`, `SECURITY.md`, issue and PR templates, `docs/troubleshooting.md`,
  `docs/hardware-validation.md`, `docs/hardware-results.md`, `docs/classification.md`.

### Changed
- `CLEF_DEVICE` defaults to `auto` (was `cuda`); `CLEF_DTYPE` defaults to `auto` (was `bfloat16`).
- ROCm/WSL-only environment (`HSA_*`, AOTriton) is applied only when the torch build is ROCm; macOS sets
  `PYTORCH_ENABLE_MPS_FALLBACK=1`.
- Model path resolution: `CLEF_MODEL_PATH`, else `~/models/clef-flash` if it holds a release, else the pinned Hugging
  Face cache snapshot. The server never downloads; use `clef download`.
- State (log, pidfile, saved classifiers) lives in a per-OS state directory (`CLEF_STATE_DIR`).
- `scripts/launch_server.sh` and `clef.ps1` are thin wrappers over the CLI (`clef.ps1` keeps the WSL keep-alive).
- README rewritten with per-platform quick starts, a support matrix and a hardware table.

### Removed
- `server/` directory (moved to `src/clef_server/`), `requirements/server.txt` (replaced by per-backend locks).

### Migration notes from 2.x
- Code moved from `server/` to `src/clef_server/`; `python server/main.py` becomes `clef serve`
  (`clef serve --detach` replaces `launch_server.sh start`). Imports: `clef_server.*` instead of `server.*`.
- `pip install -r requirements/server.txt` becomes `pip install -r requirements/<rocm|cuda|cpu|macos>.txt` followed by
  `pip install -e ".[server]"`.
- `CLEF_LOG_DIR` still works as an alias for `CLEF_STATE_DIR`. The new default location per OS is
  `~/.local/state/clef` (Linux, same as before), `~/Library/Application Support/clef` (macOS) and
  `%LOCALAPPDATA%\clef` (Windows).
- Set `CLEF_DEVICE=cuda` explicitly if you relied on the old default on an NVIDIA/ROCm machine and want an error
  instead of a fallback; with `auto` a ROCm build is reported as `rocm`.
- `CLEF_API_KEY` keeps working (one key named `default`). Use `CLEF_API_KEYS` for several named keys.
- Weights: if they are not at `~/models/clef-flash` or `CLEF_MODEL_PATH`, run `clef download`.

## 2.0.0

### Fixed
- **Media sharding bug**: v1 split a request's images/videos across several records and answered from the last
  shard only. All media of a request now goes into ONE record, so answers reflect every image and video.

### Changed
- **GPU serialization + micro-batching**: exactly one worker thread touches the GPU; concurrent requests are
  coalesced (`CLEF_MAX_MICROBATCH`, `CLEF_BATCH_WINDOW_MS`). Media records run alone. Responses carry `timing`
  (queue, forward, total, batch_size).
- **Length grouping + alignment + warmup**: concurrent text records are grouped by length bucket, padded to a
  multiple of 64 (`CLEF_PAD_MULTIPLE`; aligned GEMMs are faster on RDNA3), and common shapes are warmed at
  startup in the background (`CLEF_BUCKETS`, `CLEF_WARMUP`). Single-record p50/p95 142/150 ms (v1: 162/222).
- **Security hardening**: optional API key (`CLEF_API_KEY`), loopback bind by default, body-size limit, SSRF-guarded
  opt-in URL fetching, `media_kwargs` no longer client-settable, generic 500s with request ids, no request
  state logged unless `CLEF_LOG_STATE=1`.
- **Strict validation**: pydantic schemas for every route, per-index errors in batches, caps on questions,
  images, videos, pixels and frames. Uniform `{detail, request_id}` errors and `X-Request-ID`.
- **Clef Console** at `/`: playground, live stats, request log (`/v1/stats`, `/v1/log`, `/v1/events`).
- `/health` reports `ready|warming|loading|error` plus GPU and limits; new `/livez`.

### Ops / packaging
- `scripts/env.sh` single source of environment; `launch_server.sh` with pidfile, log rotation and
  start|stop|restart|status|logs; `clef.ps1` Windows wrapper; systemd user unit (`deploy/`).
- Exact dependency pins (`requirements/`), idempotent setup scripts with pinned model revision, `doctor.py`.
- Benchmarks rewritten (warm every shape, forward vs end-to-end, proper percentiles); `http_bench.py` and
  `profile_forward.py` added.
- `clef_client` Python package, unit tests, live integration tests, CI (ruff, shellcheck, pytest), pyproject.
- Removed `fix_cc.sh`, `fastpath.py`, `fla_check.py`, `gpu_check.py` and the `server/test_*` scripts.

## 1.0.0

- Initial local deployment of clef-flash on RX 7900 XTX via WSL2 ROCm 7.2 (torch 2.11.0+rocm7.2,
  transformers 5.10.2) behind a FastAPI SystemOne-compatible server (`/v1/systemone`, `/v1/batch`, `/health`).
- Verified text/JSON, image, video and batch requests; single-record median 162 ms, peak VRAM 19.6 GB.
