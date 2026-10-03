# Console

`http://127.0.0.1:8910/` serves the Clef Console (or run `clef open`). It is a zero-build single-page app bundled with
the server and makes no CDN calls at runtime.

| Screen | What it does |
| --- | --- |
| **Ops** | live KPIs and time-series charts (latency, throughput, memory, queue) |
| **Playground** | try SystemOne and Classify requests interactively |
| **Batch** | per-question charts, CSV / JSON in |
| **History** | recent requests from the server's log buffer |

=== "Ops"

    ![Ops light](screenshots/ops-light.png#only-light)
    ![Ops dark](screenshots/ops-dark.png#only-dark)

=== "Playground"

    ![Playground light](screenshots/playground-light.png#only-light)
    ![Playground dark](screenshots/playground-dark.png#only-dark)

=== "Classify"

    ![Classify light](screenshots/classify-light.png#only-light)
    ![Classify dark](screenshots/classify-dark.png#only-dark)

=== "Batch"

    ![Batch light](screenshots/batch-light.png#only-light)
    ![Batch dark](screenshots/batch-dark.png#only-dark)

=== "History"

    ![History light](screenshots/history-light.png#only-light)
    ![History dark](screenshots/history-dark.png#only-dark)

Screenshots were taken against the live server on the RX 7900 XTX with `bench/http_bench.py` traffic
(`python scripts/screenshots.py`).

The console consumes `/health`, `/v1/stats`, `/v1/stats/timeseries`, `/v1/events` (SSE), `/v1/log`, `/v1/systemone`,
`/v1/batch`, `/v1/classify` and `/v1/classifiers*`. When API keys are configured, enter a key in the console first;
see [remote access](remote-access.md).
