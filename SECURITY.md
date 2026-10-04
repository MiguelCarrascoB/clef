# Security policy

## Reporting a vulnerability

Please report privately through GitHub: Security tab > "Report a vulnerability"
(<https://github.com/MiguelCarrascoB/clef/security/advisories/new>). Do not open a public issue for security problems.
Include the version (`clef version`), the backend, and steps to reproduce. You can expect an acknowledgement within a
few days; this is a one-maintainer project, so there is no formal SLA.

Supported version: the latest release (3.x).

## Threat model

clef runs a model on your own hardware and exposes an HTTP API. It is designed for a single trusted operator and, at
most, trusted callers on a local network.

- **Local by default.** `CLEF_HOST` defaults to `127.0.0.1`. Binding any other address without API keys logs a loud
  warning at startup.
- **API keys.** `CLEF_API_KEY` or `CLEF_API_KEYS` (named keys, from a list or a file) protect every `/v1/*` route and the Hugging Face compatibility routes (`/hf/models/*`).
  The OpenAI-compatible routes read the same keys as `Authorization: Bearer`, so the official SDKs work unchanged. Keys are compared in constant time, never logged, and only the key name is recorded in the request log and stats.
  Keep the keys file out of version control and readable only by you (`chmod 600`). The SSE endpoint also accepts
  `?key=`, which puts the key in the URL; do not use it across untrusted networks without TLS.
- **Rate limit.** `CLEF_RATE_LIMIT` limits inference requests per minute per key (per IP when auth is off) and returns
  429 with `Retry-After`. It is a safety net against accidents and noisy clients, not a DDoS defence.
- **Rate limit scope.** Besides the classification routes, `POST /v1/chat/completions`, `POST /hf/models/*`,
  `POST /v1/evaluate` and `POST /v1/jobs` count against the limit. `POST /v1/evaluate/metrics` runs no inference and is
  not counted.
- **CORS** is off unless `CLEF_CORS_ORIGINS` is set. Do not embed API keys in browser code.
- **SSRF guard.** Fetching media from `http(s)` URLs is off by default (`CLEF_ALLOW_URL_FETCH=0`). When enabled,
  hosts are resolved and any non-global address (private, loopback, link-local, multicast) is refused, redirects are
  not followed, and downloads are size-capped (`CLEF_URL_FETCH_MAX_MB`) with a total timeout. Data URLs are the
  default way to send media.
- **Input limits.** Body size, token count, images, videos, pixels, frames, labels, questions and batch size are all
  capped and validated (`CLEF_MAX_*`). Evaluation and jobs add their own caps: `CLEF_MAX_EVAL_ROWS` (500 rows per
  `/v1/evaluate` call), `CLEF_MAX_JOB_EVAL_ROWS` (100000), `CLEF_MAX_JOB_ITEMS` (100000 items per job),
  `CLEF_MAX_JOBS` (100 queued or running jobs, then `409`) and a 16 KB limit on job metadata. Errors are generic 500s with a request id; tracebacks stay in the server log.
- **Webhooks and SSRF.** A webhook makes the server send a request to an address chosen by the API caller, so the
  feature is **off** until the operator sets `CLEF_WEBHOOK_ALLOW` (hosts, `*.suffix` patterns, IPs or CIDRs).
  The list is checked when the job is submitted and again on **every delivery attempt**: the hostname is resolved
  at that moment, every returned address must be public or inside a listed IP / CIDR, and the connection is pinned to
  the checked address, so a DNS answer that changes later (rebinding) is refused. Link-local (cloud metadata),
  multicast, unspecified and reserved addresses are refused even when a CIDR covers them. Redirects are never
  followed, environment proxies are ignored, credentials in the URL are rejected and API keys are never sent.
  Deliveries are signed (`X-Clef-Signature`, HMAC-SHA256 over `"<timestamp>.<body>"`); receivers should verify it and
  check the timestamp. Use `https` for anything that leaves your machine. The TLS path has not been tested against a
  real TLS server yet. Details: [docs/jobs.md](docs/jobs.md#webhooks).
- **Jobs: ownership, retention, disk.** With keys configured, a job belongs to the key name that submitted it; other
  keys get `404` and do not see it in the list. This is a convenience between cooperating callers, not tenant
  isolation: there is no admin view, and jobs created while auth was off stay visible to every key. Jobs are stored in
  `jobs.db` in the state directory: the **submitted payload of every job (your input texts), its result rows and
  webhook secrets are on disk in clear text** (HMAC needs the secret; `evaluate` rows also keep the input and gold
  label). Finished jobs are deleted `CLEF_JOB_TTL_HOURS` after they finish (default 168; `0` keeps them forever) or on `DELETE /v1/jobs/{id}`.
  Protect the state directory like the API keys file, and delete `jobs.db` to drop everything. Job results exported as
  CSV neutralise cells that start with `=`, `+`, `-` or `@`, so opening them in a spreadsheet does not run formulas.
- **No secrets in logs.** Request state (the text you classify) is not logged unless `CLEF_LOG_STATE=1`, but jobs do
  store it on disk (see above). Keys are never logged. Saved classifier names are validated and never used to build paths from raw input.
- **Console and vendored code.** The console is a zero-build app served by the server with no runtime CDN calls. Its
  three vendored libraries (Preact, htm, uPlot) and their licences are listed in
  [`THIRD_PARTY_NOTICES.md`](src/clef_server/static/vendor/THIRD_PARTY_NOTICES.md).
- **Model and dependencies.** Weights come from the pinned Hugging Face revision; `CLEF_MODEL_PATH` should point only
  at files you trust (the model code in the release directory is imported and executed). Dependencies are pinned in
  `requirements/`; Dependabot updates the GitHub Actions.

Out of scope / not provided: TLS (use a reverse proxy, see [docs/remote-access.md](docs/remote-access.md)), per-user
authorization (all keys have the same access to the model; job ownership only scopes job visibility), multi-tenant isolation, protection against a malicious model directory.
