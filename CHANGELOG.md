# Changelog

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
