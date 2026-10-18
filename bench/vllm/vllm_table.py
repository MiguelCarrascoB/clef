"""Summarise a bench/vllm/vllm_check.sh run: speed table, parity, VRAM and the D12 decision.

    python bench/vllm/vllm_table.py bench/out/vllm [--min-speedup 1.3 --level 16 ...]

Inputs in the directory (missing ones show as "-" and the matching rule can only FAIL/UNKNOWN):
    bench_clef_{base,both,tun}.txt, bench_vllm.txt   bench/vllm/load.py output (`clef bench` table)
    vram_<line>.json                                           load.py --vram-json
    parity_vllm.txt/.rc                                        bench/parity.py --compare dump_clef.json dump_vllm.json
Writes summary.md and summary.json next to them and prints the same. Exit 0 ADOPT, 1 DO NOT ADOPT / INCONCLUSIVE.

D12 (UPGRADES.md): vLLM is adopted only if a benchmark shows a real gain. The thresholds are PROPOSED (see
bench/vllm/README.md) and are arguments here, so changing them needs no code change.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

_PT = Path(__file__).resolve().parents[1] / "perf_table.py"
_spec = importlib.util.spec_from_file_location("perf_table", _PT)
perf_table = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(perf_table)

LINES = [
    ("clef_base", "clef, all flags off"),
    ("clef_both", "clef, CLEF_MASK_FREE=1 CLEF_FAST_HEAD=1"),
    ("clef_tun", "clef, CLEF_TUNABLEOP=1 (tuned at its first start)"),
    ("vllm", "vLLM nightly (Docker)"),
]


@dataclass
class Thresholds:
    min_speedup: float = 1.3  # vLLM req/s >= this x the best clef line, at `level`
    level: int = 16  # the concurrency the rule is judged at
    max_p95_ratio: float = 1.0  # vLLM p95 <= this x the best clef p95, at `level`
    max_p95_ratio_c1: float = 1.1  # and at concurrency 1 (single-request latency must not regress by > 10%)
    max_dp: float = 2e-3  # parity gate lives in bench/parity.py --tol; recorded here for the report
    vram_headroom_gb: float = 1.0  # vLLM peak VRAM must leave this much of the card free


def read(path: Path) -> str | None:
    return path.read_text(errors="replace") if path.exists() else None


def load(out: Path) -> dict:
    data: dict = {"lines": {}}
    for name, desc in LINES:
        bench = read(out / f"bench_{name}.txt")
        vram = read(out / f"vram_{name}.json")
        data["lines"][name] = {
            "desc": desc,
            "bench": perf_table.parse_bench(bench) if bench else {},
            "vram": json.loads(vram) if vram else None,
        }
    rc = read(out / "parity_vllm.rc")
    data["parity"] = None
    if rc is not None:
        txt = read(out / "parity_vllm.txt") or ""
        data["parity"] = {"pass": rc.strip() == "0", "modes": perf_table.parse_parity(txt)}
    return data


def best_clef(data: dict, level: int, key: str, high: bool) -> tuple[str, float] | None:
    """The better of the clef lines at `level` for `key` (rps: highest; p95: lowest)."""
    cands = [
        (n, v["bench"][level][key])
        for n, v in data["lines"].items()
        if n.startswith("clef") and level in v["bench"] and v["bench"][level]["ok"] > 0
    ]
    if not cands:
        return None
    return max(cands, key=lambda c: c[1]) if high else min(cands, key=lambda c: c[1])


def decide(data: dict, t: Thresholds) -> tuple[str, list[dict]]:
    """Returns (verdict, rules). Verdict: ADOPT (all rules pass), DO NOT ADOPT (parity or speed rule fails),
    INCONCLUSIVE (a needed number is missing, or only the p95 / VRAM / errors rule failed: look at the table)."""
    v = data["lines"]["vllm"]["bench"]
    rules: list[dict] = []

    def rule(name: str, ok: bool | None, detail: str, hard: bool) -> None:
        rules.append({"rule": name, "ok": ok, "detail": detail, "hard": hard})

    p = data["parity"]
    if p is None:
        rule("parity", None, "not run", True)
    else:
        det = "; ".join(f"{m} max |dp| {s['max_dp']:.5f}, flips {s['flips']}" for m, s in p["modes"].items())
        rule(
            "parity: 0 top-1 flips (near-ties apart) and max |dp| <= gate",
            p["pass"],
            det or "see parity_vllm.txt",
            True,
        )

    base_rps = best_clef(data, t.level, "rps", True)
    if t.level in v and v[t.level]["ok"] > 0 and base_rps:
        ratio = v[t.level]["rps"] / base_rps[1]
        rule(
            f"speed: req/s at c={t.level} >= {t.min_speedup}x best clef line",
            ratio >= t.min_speedup,
            f"vLLM {v[t.level]['rps']:.2f} vs {base_rps[0]} {base_rps[1]:.2f} = {ratio:.2f}x",
            True,
        )
    else:
        rule(f"speed: req/s at c={t.level}", None, "missing numbers", True)

    for lvl, ratio_max in ((t.level, t.max_p95_ratio), (1, t.max_p95_ratio_c1)):
        base_p95 = best_clef(data, lvl, "p95", False)
        if lvl in v and v[lvl]["ok"] > 0 and base_p95:
            r = v[lvl]["p95"] / base_p95[1]
            rule(
                f"latency: p95 at c={lvl} <= {ratio_max}x best clef p95",
                r <= ratio_max,
                f"vLLM {v[lvl]['p95']:.1f} ms vs {base_p95[0]} {base_p95[1]:.1f} ms = {r:.2f}x",
                False,
            )
        else:
            rule(f"latency: p95 at c={lvl}", None, "missing numbers", False)

    errs = sum(r["err"] for r in v.values())
    rule(
        "stability: 0 failed or mis-shaped requests (vLLM)", errs == 0 if v else None, f"{errs} errors", False
    )

    vr = data["lines"]["vllm"]["vram"]
    if vr and vr.get("peak_gb") is not None and vr.get("total_gb"):
        free = vr["total_gb"] - vr["peak_gb"]
        rule(
            f"memory: >= {t.vram_headroom_gb} GB of the card left free at vLLM's peak",
            free >= t.vram_headroom_gb,
            f"peak {vr['peak_gb']:.1f} of {vr['total_gb']:.1f} GB (delta over idle {vr['delta_gb']:.1f})",
            False,
        )
    else:
        rule("memory headroom", None, "VRAM not measured (torch could not see the GPU)", False)

    hard_fail = any(r["ok"] is False and r["hard"] for r in rules)
    unknown = any(r["ok"] is None for r in rules)
    soft_fail = any(r["ok"] is False and not r["hard"] for r in rules)
    if hard_fail:
        return "DO NOT ADOPT", rules
    if unknown or soft_fail:
        return "INCONCLUSIVE", rules
    return "ADOPT", rules


def build(out: Path, t: Thresholds) -> tuple[dict, str, str]:
    data = load(out)
    verdict, rules = decide(data, t)
    data.update({"verdict": verdict, "rules": rules, "thresholds": asdict(t)})
    levels = sorted({c for v in data["lines"].values() for c in v["bench"]})
    md = ["# vLLM vs clef summary", ""]
    head = ["line"] + [f"c={c} p50 / p95 ms, req/s" for c in levels] + ["VRAM peak GB (delta)"]
    md += ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for name, v in data["lines"].items():
        if not v["bench"] and not v["vram"]:
            md.append(f"| {name} ({v['desc']}) | " + " | ".join("-" for _ in head[1:]) + " |")
            continue
        cells = []
        for c in levels:
            r = v["bench"].get(c)
            cells.append(
                f"{r['p50']:.0f} / {r['p95']:.0f}, {r['rps']:.2f}"
                + (f" ({r['err']} err)" if r and r["err"] else "")
                if r
                else "-"
            )
        vr = v["vram"]
        cells.append(
            f"{vr['peak_gb']:.1f} ({vr['delta_gb']:+.1f})" if vr and vr.get("peak_gb") is not None else "-"
        )
        md.append(f"| {name} ({v['desc']}) | " + " | ".join(cells) + " |")
    md += [
        "",
        "VRAM is device-wide (torch mem_get_info) and includes the sampler's own context; vLLM preallocates "
        "`--gpu-memory-utilization` of the card for its KV cache, so its peak is a reservation, not a need.",
        "",
    ]
    md += [f"## Decision (D12): {verdict}", "", "| rule | result | detail |", "|---|---|---|"]
    for r in rules:
        md.append(
            f"| {r['rule']} | {'PASS' if r['ok'] else 'FAIL' if r['ok'] is False else 'n/a'} | {r['detail']} |"
        )
    md += [
        "",
        f"Thresholds (proposed, not yet confirmed): `{json.dumps(asdict(t))}`.",
        "Hard rules (parity, speed) decide DO NOT ADOPT; a failing or missing soft rule (p95, errors, memory) gives "
        "INCONCLUSIVE, to be read by a person.",
        "",
    ]
    return data, "\n".join(md) + "\n", verdict


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("out", nargs="?", default="bench/out/vllm")
    d = Thresholds()
    ap.add_argument("--min-speedup", type=float, default=d.min_speedup)
    ap.add_argument("--level", type=int, default=d.level)
    ap.add_argument("--max-p95-ratio", type=float, default=d.max_p95_ratio)
    ap.add_argument("--max-p95-ratio-c1", type=float, default=d.max_p95_ratio_c1)
    ap.add_argument("--vram-headroom-gb", type=float, default=d.vram_headroom_gb)
    a = ap.parse_args(argv)
    t = Thresholds(a.min_speedup, a.level, a.max_p95_ratio, a.max_p95_ratio_c1, d.max_dp, a.vram_headroom_gb)
    out = Path(a.out)
    data, md, verdict = build(out, t)
    (out / "summary.json").write_text(json.dumps(data, indent=2))
    (out / "summary.md").write_text(md)
    print(md)
    return 0 if verdict == "ADOPT" else 1


if __name__ == "__main__":
    sys.exit(main())
