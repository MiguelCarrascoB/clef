"""Classify one CSV column and write label + confidence + per-label scores to an output CSV.

    python examples/classify_csv.py examples/tickets.csv -o out.csv --column text \
        --labels billing,technical,account

Reads CLEF_URL / CLEF_API_KEY from the environment. Rows are sent in chunks through /v1/classify/batch.
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
    ap.add_argument("--threshold", type=float, default=None, help="multi-label threshold (0..1)")
    ap.add_argument("--instructions", default=None, help="optional question / context for the model")
    ap.add_argument("--chunk-size", type=int, default=32, help="rows per request (default: 32)")
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
            results = clef.classify_many(
                [r[args.column] for r in rows],
                labels,
                instructions=args.instructions,
                multi_label=args.multi_label,
                threshold=args.threshold,
                chunk_size=args.chunk_size,
            )
    except ClefError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    extra = ["label", "confidence"] + [f"score_{lab}" for lab in labels]
    out_fields = fieldnames + [c for c in extra if c not in fieldnames]
    with open(args.output, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=out_fields)
        writer.writeheader()
        for row, res in zip(rows, results, strict=True):
            # multi-label: all hits joined with "|" (empty when none reached the threshold)
            row["label"] = "|".join(res.labels)
            row["confidence"] = "" if res.confidence is None else f"{res.confidence:.4f}"
            for lab in labels:
                row[f"score_{lab}"] = f"{res.scores.get(lab, 0.0):.4f}"
            writer.writerow(row)
    print(f"wrote {len(rows)} rows to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
