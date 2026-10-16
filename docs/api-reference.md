---
hide:
  - toc
---

# HTTP API

Interactive reference generated from [`openapi.json`](openapi.json), which is copied from the repository root at build
time, so it always matches the code. A running server serves the same schema at `/openapi.json`, with a live Swagger
UI at `/docs`. Use the schema to generate a client in any language.

"Try it out" is disabled here because no server sits behind this site. Start one with `clef serve` and open
<http://127.0.0.1:8910/docs> to call the API for real, or see the [classification guide](classification.md) and
[first request](first-request.md) for curl examples.

Open routes (no API key): `/health` (model and GPU status), `/livez` (process up), `/readyz` (200 only when the
model is ready and the server is not shutting down; for load balancers and container health), `/schema-example`,
`/docs`, `/openapi.json`.

<swagger-ui src="openapi.json"/>
