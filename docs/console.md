# Console

`http://127.0.0.1:8910/` serves the Clef Console (or run `clef open`). It is a zero-build single-page app bundled with
the server and makes no CDN calls at runtime.

| Screen | What it does |
| --- | --- |
| **Playground** | try SystemOne and Classify requests interactively |
| **History** | recent requests from the server's log buffer |
| **Batch** | per-question charts, CSV / JSON in |
| **Evaluate** | a labelled dataset in, accuracy, confusion matrix and calibration out |
| **Ops** | live KPIs and time-series charts (latency, throughput, memory, queue) |

![Clef Console tour](screenshots/console.gif)

## Evaluate

Drop a CSV, JSON or JSONL file (or press *Load sample*), pick the **input** and **gold label** columns, optionally a
saved classifier or multi-label, and press **Run evaluation**. Rows go through `/v1/classify/batch` in chunks with a
progress bar, ETA and Cancel; the scores are then posted to `/v1/evaluate/metrics`. You get accuracy, per-label
precision / recall / F1, a clickable confusion heatmap, a reliability diagram with ECE and Brier, a
coverage-vs-accuracy curve with a threshold readout for choosing an auto-route threshold, and a mistakes table you
can export. [`examples/tickets_labelled.csv`](https://github.com/MiguelCarrascoB/clef/blob/main/examples/tickets_labelled.csv)
is a ready 47-ticket sample. How to read the numbers: [Evaluation](evaluation.md).

## Screens

=== "Evaluate"

    ![Evaluate light](screenshots/evaluate-light.png#only-light)
    ![Evaluate dark](screenshots/evaluate-dark.png#only-dark)

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
`/v1/batch`, `/v1/classify`, `/v1/classify/batch`, `/v1/classifiers*` and `/v1/evaluate/metrics`. When API keys are configured, enter a key in the console first;
see [remote access](remote-access.md).
