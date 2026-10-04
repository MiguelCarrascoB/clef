# Changelog

## 3.1.0 (2026-10-04)

Four features on top of 3.0.0, plus the public-release sweep. Everything was run on the verified Windows + WSL2 +
ROCm setup (RX 7900 XTX): the compat routes with the official `openai` and `huggingface_hub` SDKs, evaluation on
`examples/tickets_labelled.csv` (accuracy 0.87, ECE 0.04), a 500-item job alongside interactive traffic, and every
memory mode. CUDA and Apple Silicon remain untested on hardware.

### Added
- **OpenAI and Hugging Face compatibility** ([guide](docs/openai-compat.md)): `GET /v1/models[/{id}]` and
  `POST /v1/chat/completions` turn a JSON schema (`response_format` or one forced tool) into decisions: string
  enum -> choice, boolean -> yes/no, integer enum / bounded integer / `x-clef-ordinal` string enum -> score, array
  of string enums -> multi-label. Answers are argmax JSON in `message.content`, with `logprobs` (log of the
  calibrated probability), a non-standard `clef` field with every probability, streaming (SSE) and OpenAI-shaped
  errors. A request without a usable schema is a `400 missing_schema`. `POST /hf/models/{id}` is the Hugging Face
  zero-shot endpoint (classic `{sequence, labels, scores}` or the `[{label, score}]` list that `huggingface_hub`
  1.x expects). LangChain is not tested.
- **Evaluation** ([guide](docs/evaluation.md)): `POST /v1/evaluate` (labelled rows, optionally a saved classifier;
  up to `CLEF_MAX_EVAL_ROWS`, default 500) and `POST /v1/evaluate/metrics` (scores you already have, no
  inference). Accuracy, top-2, macro / micro / weighted F1, per-label precision / recall / F1, confusion matrix,
  reliability bins, ECE, MCE, Brier, log-loss, a coverage-vs-accuracy curve and `auto_route` thresholds;
  multi-label reports exact match and Hamming loss. An `evaluate` job kind handles up to
  `CLEF_MAX_JOB_EVAL_ROWS` (100000) rows. `examples/tickets_labelled.csv` is a 47-ticket sample.
- **Console: Evaluate tab**: upload CSV / JSON / JSONL, pick the input and gold columns, run with progress and
  cancel, then read the confusion heatmap, reliability diagram, coverage curve with a threshold readout, per-label
  table and mistakes list; export predictions or metrics.
- **Async jobs and webhooks** ([guide](docs/jobs.md)): `POST /v1/jobs` (kinds `classify`, `score`, `systemone`,
  `evaluate`; `202` with `Location`), `GET /v1/jobs`, `GET /v1/jobs/{id}`, `GET /v1/jobs/{id}/results` (JSON pages,
  or `?format=ndjson|csv`), `POST /v1/jobs/{id}/cancel`, `DELETE /v1/jobs/{id}`. Jobs live in SQLite
  (`<state dir>/jobs.db`), run FIFO with one micro-batch in flight so interactive calls interleave, resume after a
  restart, and belong to the API key that submitted them. Webhooks (`job.succeeded`, `job.failed`, `job.cancelled`,
  opt-in `job.progress`) are signed with `X-Clef-Signature: sha256=HMAC(secret, "<timestamp>.<body>")` and are off
  until `CLEF_WEBHOOK_ALLOW` lists the receivers. New settings: `CLEF_MAX_JOB_ITEMS`, `CLEF_MAX_JOBS`,
  `CLEF_JOB_TTL_HOURS`, `CLEF_JOB_ENGINE_WAIT_S`, `CLEF_WEBHOOK_ALLOW`, `CLEF_WEBHOOK_TIMEOUT_S`,
  `CLEF_WEBHOOK_ATTEMPTS`.
- **Clients**: Python (`submit_job`, `classify_job`, `job`, `jobs`, `cancel_job`, `delete_job`, `wait_job`,
  `job_results`, `save_job_results`; sync and async) and JavaScript (`submitJob`, `classifyJob`, `job`, `jobs`,
  `cancelJob`, `deleteJob`, `waitJob`, `jobResults`). Examples: `openai_sdk.py`, `hf_zero_shot.py`, `jobs.py`.
- **Smaller GPUs** ([guide](docs/memory.md)): `CLEF_OFFLOAD=cpu` (`--offload`) keeps the embeddings and any layers
  that do not fit in host RAM and streams them in per forward; `CLEF_MAX_DEVICE_MEMORY_GB`
  (`--max-device-memory-gb`) caps device memory as a hard allocator limit. `CLEF_QUANT=int8` now works on ROCm,
  Apple Silicon and CPU through torchao (CUDA keeps bitsandbytes); `CLEF_QUANT_BACKEND=auto|bnb|torchao` picks the
  library. `clef doctor` prints a recommended memory setting. torchao is part of the `rocm`, `mps`, `cpu` and new
  `quant` extras. `bench/memory_bench.py` reproduces the measurements.
- **Extension point**: `clef_server/appctx.py` defines `AppContext`. A feature module exposes
  `router(ctx) -> APIRouter` and is listed in `main.FEATURES` (`evaluation`, `compat`, `jobs`); the context carries
  config, stats, engine, auth and rate-limit dependencies, the inference helpers and startup / shutdown hooks.
  Evaluation, compat and jobs are built on it.
- **Docs site** (MkDocs Material, built with `--strict` and published to the `gh-pages` branch by
  `scripts/publish_docs.sh`): <https://miguelcarrascob.github.io/clef/>, with the interactive API reference
  generated from `openapi.json`.

### Changed
- Measured on the RX 7900 XTX (see [docs/memory.md](docs/memory.md)): offload is bit-identical to bf16. A 16 GB
  card runs with `CLEF_OFFLOAD=cpu CLEF_MAX_DEVICE_MEMORY_GB=15` (p50 152 -> 175 ms), a 12 GB card with
  `CLEF_QUANT=int8 CLEF_OFFLOAD=cpu` (about 206 ms, mean probability error 0.006, 2 of 399 top choices flipped) and
  an 8 GB card with a cap of 7 (about 540 ms, ~15 GB of host RAM). The old advice for 16 GB cards
  (`CLEF_QUANT=nf4` with bitsandbytes) is replaced: nf4 flipped about 9% of top choices on ROCm and is not
  recommended. CUDA and Apple Silicon are untested on hardware.
- `CLEF_QUANT=int8` is no longer CUDA only (see above). `pip install "clef-local[quant]"` adds torchao anywhere.
- API keys also protect the new `/hf/models/*` routes (they live outside `/v1`). `/health` `limits` gains
  `max_job_items` and `webhooks_enabled`.
- README restructured for the public release (hero GIF, usage sections for the OpenAI SDK, evaluation and jobs,
  measured memory options); the docs site gains pages for the four features.
- `/health` gains a `memory` block (offload mode, cap, quant and quant backend, layers / embeddings on the host,
  host and device weight GB), shown on the Ops screen when not the default. Load-time fallbacks and caveats (cap not
  enforced, offload fallback, unpinned host memory, nf4 on ROCm, int8 on MPS, cap above free memory) are listed in
  `/health` `warnings`, and `clef doctor` warns for those combinations and below a 7 GB offload cap on WSL.
- Console: at 560 px and below the top bar wraps into two rows with a scrolling tab strip (no horizontal page
  scroll at 375 px).
- Jobs: a job whose items all fail (or that trips the circuit breaker on a systemic error) ends `failed` instead
  of `succeeded`; the job view gains `warnings` and `webhook.failed_deliveries`, clients gain `has_errors`. Saved
  classifiers are copied into the job at submit (`snapshot_of`) for every kind, including `evaluate`. Jobs
  interrupted more than `CLEF_JOB_MAX_RESUMES` (3) times are failed. `POST /v1/jobs` honours `Idempotency-Key`.
  Webhooks retry for ~8 minutes by default (`CLEF_WEBHOOK_ATTEMPTS` 10, `CLEF_WEBHOOK_BACKOFF_S`,
  `CLEF_WEBHOOK_BACKOFF_CAP_S`) and a failed delivery can be resent with `POST /v1/jobs/{id}/webhook/redeliver`.
- Evaluation: the `evaluate` job kind waits for the model, isolates bad rows (`n_errors`) and resumes; rows with
  NaN / non-numeric scores or none of the labels are rejected, defaulted or clipped scores are counted;
  `POST /v1/evaluate` counts in stats and the request log.
- Compat chat responses list ignored request fields in `clef.ignored`.
- Python and JS clients: `save_job_results(require_finished=True)` writes atomically; `jobs()` exposes `total`;
  the JS client no longer retries a job submission after it was sent (unless an idempotency key is given).
- GitHub Actions workflows (CI, release, docs) and Dependabot are removed; GitHub Actions is disabled for the
  repository. The checks in CONTRIBUTING.md are run locally before a PR, and the docs are published by hand.

### Fixed
- Server start with `CLEF_MAX_DEVICE_MEMORY_GB` segfaulted on ROCm / WSL2: GPU telemetry sampling raced torch's
  own device initialisation. The sampler now skips GPU telemetry while the model loads.
- Blank `CLEF_*` variables (WSLENV forwards unset Windows variables as empty strings) now mean "unset" instead of
  failing config validation.
- CSV export no longer prefixes negative numbers with a quote, and an export of a running job keeps every column.

### Security
- Webhook deliveries are an SSRF surface, so they are off by default and gated by `CLEF_WEBHOOK_ALLOW`. The
  allow-list is checked at submit and again on every attempt, the host is resolved right then and the request is
  pinned to the checked address (no DNS rebinding), link-local (cloud metadata), multicast, unspecified and
  reserved addresses are always refused, redirects and environment proxies are ignored and API keys are never
  sent. Webhook secrets are stored in clear text in `jobs.db` (HMAC needs them).
- Webhook addresses that embed an IPv4 address (NAT64 `64:ff9b::/96` and `64:ff9b:1::/48`, 6to4, Teredo) are
  judged by that IPv4 address, and site-local / reserved IPv6 ranges are refused, so a translated address can no
  longer reach a private host. Redelivery reuses the stored URL and re-runs every check.
- Job results exported as CSV neutralise spreadsheet formulas (text cells starting with `=`, `+`, `-` or `@`).
- Jobs are owned by the API key name that submitted them; other keys get `404`. Job payloads and results are kept on
  disk until deleted or until `CLEF_JOB_TTL_HOURS` expires.
- `THIRD_PARTY_NOTICES` for the vendored Preact, htm and uPlot console libraries
  (`src/clef_server/static/vendor/THIRD_PARTY_NOTICES.md`).

### Known gaps
- Jobs are not counted in `/v1/stats` or the request log, and there is no priority lane: interactive calls are
  ~4x slower while a large job runs (p50 109 -> 422 ms measured). The webhook TLS path was not tested against a real
  TLS server; LangChain was not tested against the compat routes; offload, torchao int8 and the device cap are
  untested on NVIDIA and Apple Silicon.

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
