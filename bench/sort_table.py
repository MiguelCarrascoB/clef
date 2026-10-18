"""Summarise a bench/sort_check.sh run: items/s per variant, gain over base, label agreement, verdict.

    python bench/sort_table.py bench/out/sort            # reads runs.jsonl written by bench/jobs_bench.py

The flags flip on only if the median gain over base is above GATE (10%) AND above the spread of the base
repeats (so run-to-run noise cannot pass). Label agreement with base is shown for the first repeat: sorting
changes padding, so a few last-bit differences are normal, a large disagreement is not. Writes summary.md and
summary.json. Exit 1 when a variant's job failed or no base run exists.
"""

from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path

GATE = 0.10
MIN_AGREEMENT = 0.99
VARIANTS = {
    "base": "all flags off",
    "sort": "CLEF_JOB_SORT_WINDOW=512",
    "ahead": "CLEF_TOKENIZE_AHEAD=1",
    "both": "window 512 + tokenize ahead",
    "sort256": "CLEF_JOB_SORT_WINDOW=256",
    "sort1024": "CLEF_JOB_SORT_WINDOW=1024",
}


def load(path: Path) -> dict[str, list[dict]]:
    runs: dict[str, list[dict]] = {}
    for line in path.read_text().splitlines() if path.exists() else []:
        if line.strip():
            rec = json.loads(line)
            runs.setdefault(rec["tag"], []).append(rec)
    return runs


def agreement(a: list, b: list) -> float | None:
    if not a or len(a) != len(b):
        return None
    return sum(x == y for x, y in zip(a, b, strict=True)) / len(a)


def summarise(runs: dict[str, list[dict]]) -> dict:
    base = [r for r in runs.get("base", []) if r["status"] == "succeeded"]
    base_med = statistics.median(r["items_per_s"] for r in base) if base else None
    spread = (
        (max(r["items_per_s"] for r in base) - min(r["items_per_s"] for r in base)) / base_med
        if base and base_med
        else 0.0
    )
    out: dict = {"gate": GATE, "base_items_per_s": base_med, "base_spread": spread, "variants": {}}
    base_p95 = [r["interactive_p95_ms"] for r in base if r.get("interactive_p95_ms")]
    for tag, rs in runs.items():
        ok = [r for r in rs if r["status"] == "succeeded"]
        v: dict = {"desc": VARIANTS.get(tag, tag), "runs": len(rs), "failed_runs": len(rs) - len(ok)}
        if ok:
            med = statistics.median(r["items_per_s"] for r in ok)
            v["items_per_s"] = med
            v["min"], v["max"] = min(r["items_per_s"] for r in ok), max(r["items_per_s"] for r in ok)
            p95 = [r["interactive_p95_ms"] for r in ok if r.get("interactive_p95_ms")]
            v["interactive_p95_ms"] = statistics.median(p95) if p95 else None
            if tag != "base" and base_med:
                v["gain"] = med / base_med - 1.0
                v["verdict"] = "FLIP" if v["gain"] > max(GATE, spread) else "keep off"
                v["agreement"] = agreement(base[0].get("labels", []), ok[0].get("labels", []))
                if v["agreement"] is not None and v["agreement"] < MIN_AGREEMENT:
                    v["verdict"] = "CHECK labels"
                if p95 and base_p95 and statistics.median(p95) > 1.25 * statistics.median(base_p95):
                    v["verdict"] += " (interactive p95 worse)"
        else:
            v["verdict"] = "FAILED"
        out["variants"][tag] = v
    return out


def render(data: dict) -> str:
    lines = ["# sort_check summary", ""]
    if data["base_items_per_s"] is None:
        return "\n".join([*lines, "No successful base run: nothing to compare against.", ""])
    lines += [
        f"Gate: flip a flag only if the median gain over base is above {data['gate']:.0%} and above the base "
        f"spread ({data['base_spread']:.1%}).",
        "",
        "| variant | settings | items/s (median, min-max) | gain | label agreement | interactive p95 ms | verdict |",
        "|---|---|---|---|---|---|---|",
    ]
    for tag, v in data["variants"].items():
        if "items_per_s" not in v:
            lines.append(f"| {tag} | {v['desc']} | failed | - | - | - | {v['verdict']} |")
            continue
        gain = f"{v['gain']:+.1%}" if "gain" in v else "-"
        agree = f"{v['agreement']:.2%}" if v.get("agreement") is not None else "-"
        p95 = f"{v['interactive_p95_ms']:.0f}" if v.get("interactive_p95_ms") else "-"
        rng = f"{v['items_per_s']:.1f} ({v['min']:.1f}-{v['max']:.1f})"
        verdict = v.get("verdict", "baseline")
        lines.append(f"| {tag} | {v['desc']} | {rng} | {gain} | {agree} | {p95} | {verdict} |")
    lines.append("")
    return "\n".join(lines)


def main(argv: list[str]) -> int:
    out = Path(argv[1] if len(argv) > 1 else "bench/out/sort")
    data = summarise(load(out / "runs.jsonl"))
    text = render(data)
    (out / "summary.md").write_text(text)
    (out / "summary.json").write_text(json.dumps(data, indent=2))
    print(text)
    failed = data["base_items_per_s"] is None or any(v.get("failed_runs") for v in data["variants"].values())
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
