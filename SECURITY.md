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
- **API keys.** `CLEF_API_KEY` or `CLEF_API_KEYS` (named keys, from a list or a file) protect every `/v1/*` route.
  Keys are compared in constant time, never logged, and only the key name is recorded in the request log and stats.
  Keep the keys file out of version control and readable only by you (`chmod 600`). The SSE endpoint also accepts
  `?key=`, which puts the key in the URL; do not use it across untrusted networks without TLS.
- **Rate limit.** `CLEF_RATE_LIMIT` limits inference requests per minute per key (per IP when auth is off) and returns
  429 with `Retry-After`. It is a safety net against accidents and noisy clients, not a DDoS defence.
- **CORS** is off unless `CLEF_CORS_ORIGINS` is set. Do not embed API keys in browser code.
- **SSRF guard.** Fetching media from `http(s)` URLs is off by default (`CLEF_ALLOW_URL_FETCH=0`). When enabled,
  hosts are resolved and any non-global address (private, loopback, link-local, multicast) is refused, redirects are
  not followed, and downloads are size-capped (`CLEF_URL_FETCH_MAX_MB`) with a total timeout. Data URLs are the
  default way to send media.
- **Input limits.** Body size, token count, images, videos, pixels, frames, labels, questions and batch size are all
  capped and validated (`CLEF_MAX_*`). Errors are generic 500s with a request id; tracebacks stay in the server log.
- **No secrets in logs.** Request state (the text you classify) is not logged unless `CLEF_LOG_STATE=1`. Keys are never
  logged. Saved classifier names are validated and never used to build paths from raw input.
- **Model and dependencies.** Weights come from the pinned Hugging Face revision; `CLEF_MODEL_PATH` should point only
  at files you trust (the model code in the release directory is imported and executed). Dependencies are pinned in
  `requirements/`; Dependabot updates the GitHub Actions.

Out of scope / not provided: TLS (use a reverse proxy, see `docs/remote-access.md`), per-user authorization (all keys
have the same access), multi-tenant isolation, protection against a malicious model directory.
