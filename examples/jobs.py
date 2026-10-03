"""Classify a CSV column as an async job: submit, watch progress, write the results.

    python examples/jobs.py examples/tickets.csv -o out.csv --column text --labels billing,technical,account

Unlike classify_csv.py no HTTP connection stays open while the model works, so this scales to tens of
thousands of rows (raise CLEF_MAX_JOB_ITEMS / CLEF_MAX_BODY_MB on the server for very large files) and
survives a flaky network: the job keeps running on the server, and the printed job id lets you re-attach
with `--job <id>`. Reads CLEF_URL / CLEF_API_KEY from the environment.
"""

from __future__ import annotations

import argparse
import csv
import sys

from clef_client import ClefClient, ClefError


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("input", help="input CSV (with header row)")
    ap.add_argument("-o", "--output", required=True, help="output CSV path")
    ap.add_argument("--column", default="text", help="column holding the text to classify (default: text)")
    ap.add_argument("--labels", required=True, help="comma-separated labels, e.g. billing,technical,account")
    ap.add_argument("--multi-label", action="store_true", help="allow several labels per row")
    ap.add_argument("--instructions", default=None, help="optional question / context for the model")
    ap.add_argument("--job", default=None, help="re-attach to an existing job id instead of submitting")
    ap.add_argument("--webhook", default=None, help="URL to notify when done (needs CLEF_WEBHOOK_ALLOW)")
    ap.add_argument("--poll", type=float, default=1.0, help="seconds between status checks (default: 1)")
    args = ap.parse_args(argv)

    labels = [s.strip() for s in args.labels.split(",") if s.strip()]
    with open(args.input, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None or args.column not in reader.fieldnames:
            ap.error(f"column {args.column!r} not found; columns are {reader.fieldnames}")
        fieldnames = list(reader.fieldnames)
        rows = list(reader)

    try:
        with ClefClient() as clef:
            if args.job:
                job = clef.job(args.job)
            else:
                job = clef.classify_job(
                    [r[args.column] for r in rows],
                    labels,
                    instructions=args.instructions,
                    multi_label=args.multi_label,
                    webhook=args.webhook,
                    metadata={"source": args.input},
                )
                print(f"submitted {job.id} ({len(rows)} rows)")
            job = clef.wait_job(
                job.id,
                poll=args.poll,
                on_progress=lambda j: print(f"\r{j.status}: {j.done}/{j.total or '?'}", end="", flush=True),
            )
            print()
            if not job.ok:
                print(f"error: job {job.id} {job.status}: {job.error or 'no details'}", file=sys.stderr)
                return 1
            by_index = {item.index: item for item in clef.job_results(job.id)}
    except ClefError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    extra = ["label", "confidence", "error"] + [f"score_{lab}" for lab in labels]
    out_fields = fieldnames + [c for c in extra if c not in fieldnames]
    with open(args.output, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=out_fields)
        writer.writeheader()
        for i, row in enumerate(rows):
            item = by_index.get(i)
            if item is None or item.result is None:  # a row the model could not handle (e.g. too long)
                row["error"] = item.error if item else "missing result"
            else:
                res = item.result
                row["label"] = "|".join(res.labels)
                row["confidence"] = "" if res.confidence is None else f"{res.confidence:.4f}"
                for lab in labels:
                    row[f"score_{lab}"] = f"{res.scores.get(lab, 0.0):.4f}"
            writer.writerow(row)
    summary = job.result or {}
    print(f"wrote {len(rows)} rows to {args.output} ({summary.get('errors', 0)} failed)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
