# Evaluation

Clef returns a probability for every option, so you can check not only *how often* it is right but whether its
confidence can be trusted. Upload a labelled dataset and you get accuracy, per-label precision / recall / F1, a
confusion matrix and calibration (reliability diagram, ECE, Brier score), plus a coverage-vs-accuracy curve for
choosing an auto-route threshold.

Three ways to run it, all using the same metric code (`src/clef_server/evaluation.py`):

| | Use it for | Limit |
| --- | --- | --- |
| Console: **Evaluate** tab | exploring a dataset interactively | what your browser can hold |
| `POST /v1/evaluate` | scripts and CI, up to a few hundred rows | `CLEF_MAX_EVAL_ROWS` (500) |
| `evaluate` job kind (`POST /v1/jobs`) | big datasets, runs in the background | `CLEF_MAX_JOB_EVAL_ROWS` (100000) |
| `POST /v1/evaluate/metrics` | you already have the scores | no inference; `CLEF_MAX_JOB_EVAL_ROWS` |

## In the console

1. Open **Evaluate**, drop a CSV / JSON / JSONL file (or press *Load sample*). The first rows are previewed.
2. Pick the **input column** and the **gold-label column**. Labels are derived from the gold column (most frequent
   first); edit the list if you want to add a label that never occurs. Rows whose gold label is not in the list are
   skipped and counted.
3. Optional: instructions, a saved classifier (it then supplies labels, instructions and multi-label), or
   **multi-label** with a separator for the gold column (`billing;technical`) and a threshold.
4. **Run evaluation**. Rows go through `/v1/classify/batch` in chunks with a progress bar, ETA and Cancel; failed
   chunks are retried and can be re-run. When done (or cancelled with partial results) the scores are posted to
   `/v1/evaluate/metrics`.
5. Read the results, click a confusion-matrix cell to list those rows, then export predictions (CSV) or metrics (JSON).

A ready-made file with 47 tickets, including a few genuinely ambiguous ones, is in `examples/tickets_labelled.csv`.

## With the API

```bash
curl -s http://127.0.0.1:8910/v1/evaluate -H 'Content-Type: application/json' -d '{
  "labels": ["billing", "technical", "account"],
  "instructions": "Which team should handle the message?",
  "rows": [
    {"input": "I was charged twice", "gold": "billing"},
    {"input": "API returns 502",     "gold": "technical"}
  ],
  "include_predictions": true
}'
```

Send `classifier: "<name>"` instead of `labels` / `instructions` / `multi_label` to use a saved classifier. For
multi-label, set `multi_label: true` and make `gold` a list (`[]` is allowed: no label applies). A gold label that is
not in the label set is a 400 that names the row, before any inference runs. Rows are classified `CLEF_MAX_BATCH` at a
time through the normal inference path, so Ops stats and the request log count them.

`POST /v1/evaluate/metrics` takes `{labels, rows: [{gold, scores}], multi_label?, threshold?, bins?, include_predictions?}`
where `scores` is the `scores` object `/v1/classify` returned. Use it to score predictions you collected elsewhere.

The response contains `n`, `accuracy`, `top2_accuracy`, `macro_f1`, `micro_f1`, `weighted_f1`, `per_label`,
`confusion_matrix {labels, matrix}` (rows gold, columns predicted), `calibration {bins, ece, mce, brier, nll,
mean_confidence}`, `coverage_curve` (thresholds 0.00 to 1.00), `auto_route` and, with `include_predictions`,
`predictions` (`input` truncated to 200 characters, `gold`, `predicted`, `confidence`, `correct`). Multi-label replaces
`accuracy` / confusion matrix / coverage with `exact_match`, `hamming_loss` and per-label `tp fp fn tn`.

## As a job

When the jobs API is present, `evaluate` is a job kind with the same payload as `POST /v1/evaluate`:

```bash
curl -s http://127.0.0.1:8910/v1/jobs -H 'Content-Type: application/json' \
  -d '{"kind": "evaluate", "payload": {"labels": ["billing","technical"], "rows": [...]}}'
```

Invalid payloads are rejected at submit (400). Every classified row is stored as a job item (`index`, `input`, `gold`,
`predicted`, `confidence`, `correct`, `scores`), progress counts rows, and the job result is the metrics object above.
Cancelling a job returns the metrics of the rows finished so far, marked `cancelled` and `partial`.

## What the numbers mean

- **Accuracy**: share of rows where the top label equals the gold label. **Top-2 accuracy**: the gold label is among
  the two most probable.
- **Precision** (of what it called X, how much was X), **Recall** (of the real X, how much it found), **F1** (their
  harmonic mean). **Macro F1** averages labels equally, so rare labels count; **weighted F1** weighs by support;
  **micro F1** pools all decisions (equal to accuracy for single-label). Labels with no gold rows and no predictions
  are left out of the macro average.
- **Confusion matrix**: rows are the truth, columns the prediction. The diagonal is correct; any other cell is a
  specific confusion worth reading examples for.
- **Calibration**: does a stated 80% mean 80% right? **ECE** (expected calibration error) is the average gap between
  confidence and actual accuracy across confidence bins, weighted by bin size: under 0.05 is good, over 0.10 means
  you should not trust the raw numbers. **MCE** is the worst bin. **Brier** is the mean squared error of the whole
  probability vector (0 is perfect; for 2 labels, always answering 50/50 scores 0.5). **Log-loss (NLL)** punishes
  confident mistakes hard.
- **Multi-label**: **exact match** needs every label right; **Hamming loss** is the share of wrong (row, label)
  decisions; calibration is pooled over all (row, label) pairs.

### Reading a reliability diagram

Rows are grouped by the confidence of their top answer (0 to 10%, 10 to 20%, ...). Each bar's height is how often
those rows were actually right; the dashed diagonal is perfect calibration. Bars on the diagonal: trustworthy. Bars
below it: over-confident (says 90%, right 70%). Bars above it: under-confident. The strip underneath shows how many
rows are in each bin; ignore bars with only a handful of rows. With single-label classification the top answer
always has at least `1/labels` probability, so bins below that stay empty.

## Choosing an auto-route threshold

The **coverage vs accuracy** curve answers "if I only act automatically when confidence is at least *t*, what do I
get?". Coverage is the share of rows that clear the bar; accuracy is measured on those rows only; the rest go to a
human. Drag the threshold (or use the slider) and read the line, for example: "at >= 0.85 confidence: 72% coverage,
98.6% accuracy". The buttons below the chart jump to the lowest threshold that reaches 80 / 90 / 95 / 98 / 99%
accuracy, which is the one that automates the most rows.

Pick the accuracy your workflow can live with, take that threshold, and check it on data the threshold was not
chosen from (a threshold tuned on 30 rows is noisy). Re-check after changing labels, instructions or the model. In
production, send anything below the threshold to review and keep sampling the automated part.
