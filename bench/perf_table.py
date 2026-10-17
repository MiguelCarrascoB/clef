"""Summarise a bench/perf_check.sh run: parity PASS/FAIL per variant and a speed table.

    python bench/perf_table.py bench/out/perf            # reads what perf_check.sh wrote there

Inputs in the directory (all optional; missing ones show as "-"):
    profile.txt                 bench/profile_forward.py output (step 0)
    parity_<variant>.txt/.rc    bench/parity.py --compare output and its exit code (0 = PASS)
    bench_<variant>.txt         `clef bench` output
Writes summary.md and summary.json next to them and prints the same. Exit 1 if any parity check failed.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

VARIANTS = [
    ("base", "all flags off"),
    ("mask", "CLEF_MASK_FREE=1"),
    ("head", "CLEF_FAST_HEAD=1"),
    ("both", "MASK_FREE + FAST_HEAD"),
    ("tun", "base + CLEF_TUNABLEOP=1"),
    ("tun_both", "both + CLEF_TUNABLEOP=1"),
]
ROW = re.compile(
    r"^\s*(\d+)\s+(\d+)\s+(\d+)\s+([\d.]+)\s*\|\s*([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+\|\s*avg\s+([\d.]+)"
)
STAT = re.compile(r"(single|batch\d+)\b.*?wall ([\d.]+) ms.*?GPU kernel time ([\d.]+) ms")
GROUP = re.compile(r"^\s*([\d.]+)\s+([\d.]+)%\s+(.+)$")
PARITY_LINE = re.compile(r"^(singles|batched)\s+max \|dp\| = ([\d.]+).*?flips (\d+)/(\d+)", re.M)


def parse_bench(text: str) -> dict[int, dict]:
    out = {}
    for line in text.splitlines():
        m = ROW.match(line)
        if m:
            conc, ok, err, rps, p50, p95, p99, avg = m.groups()
            out[int(conc)] = {
                "ok": int(ok),
                "err": int(err),
                "rps": float(rps),
                "p50": float(p50),
                "p95": float(p95),
                "p99": float(p99),
                "avg_batch": float(avg),
            }
    return out


def parse_parity(text: str) -> dict:
    """The per-mode lines of bench/parity.py --compare: 'singles  max |dp| = ...  top-1 flips a/b  OK'."""
    return {
        m.group(1): {"max_dp": float(m.group(2)), "flips": int(m.group(3)), "n": int(m.group(4))}
        for m in PARITY_LINE.finditer(text)
    }


def parse_profile(text: str) -> dict:
    """Per profiled forward: wall ms (profiler overhead included), GPU kernel ms, GEMM share of kernel time."""
    out: dict[str, dict] = {}
    label = None
    for line in text.splitlines():
        m = STAT.search(line)
        if line.startswith("===") and m:
            label = m.group(1)
            wall, kern = float(m.group(2)), float(m.group(3))
            out[label] = {
                "wall_ms": wall,
                "kernel_ms": kern,
                "kernel_over_wall": kern / wall if wall else 0.0,
            }
        elif label and line.rstrip().endswith("GEMM"):
            g = GROUP.match(line)
            if g:
                out[label]["gemm_pct"] = float(g.group(2))
    return out


def read(path: Path) -> str | None:
    return path.read_text(errors="replace") if path.exists() else None


def _fmt_modes(modes: dict) -> str:
    return "; ".join(f"{k} max |dp| {v['max_dp']:.5f}, flips {v['flips']}" for k, v in modes.items())


def build(out: Path) -> tuple[dict, str, bool]:
    data: dict = {"variants": {}, "profile": {}}
    prof = read(out / "profile.txt")
    if prof:
        data["profile"] = parse_profile(prof)
    failed = False
    for name, desc in VARIANTS:
        v: dict = {"desc": desc}
        bench = read(out / f"bench_{name}.txt")
        v["bench"] = parse_bench(bench) if bench else {}
        rc = read(out / f"parity_{name}.rc")
        if rc is not None:
            v["parity_pass"] = rc.strip() == "0"
            v["parity"] = parse_parity(read(out / f"parity_{name}.txt") or "")
            failed = failed or not v["parity_pass"]
        data["variants"][name] = v
    if read(out / "parity_noise.rc") is not None:
        data["noise"] = parse_parity(read(out / "parity_noise.txt") or "")
    lines = ["# perf_check summary", ""]
    if data["profile"]:
        lines += ["## Step 0: is batched serving GEMM-bound?", ""]
        for label, p in data["profile"].items():
            gemm = p.get("gemm_pct")
            lines.append(
                f"- {label}: wall {p['wall_ms']:.1f} ms, GPU kernels {p['kernel_ms']:.1f} ms "
                f"(kernel/wall {p['kernel_over_wall']:.2f})"
                + (f", GEMM {gemm:.0f}% of kernel time" if gemm else "")
            )
        b8 = next((p for k, p in data["profile"].items() if k.startswith("batch")), None)
        if b8:
            verdict = (
                "GEMM-bound: only GEMM tuning (TunableOp) can help batched throughput"
                if b8["kernel_over_wall"] >= 0.9
                else "NOT GEMM-bound: host/sync overhead is left, the flags can help batched throughput"
            )
            lines += ["", f"Verdict (kernel/wall >= 0.9 means GEMM-bound): {verdict}"]
        lines.append("")
    lines += ["## Parity vs flags off (gate: 0 top-1 flips and max |dp| <= 2e-3)", ""]
    if data.get("noise"):
        lines += ["Run-to-run noise of the flags-off path: " + _fmt_modes(data["noise"]), ""]
    lines += ["| variant | verdict | singles max dp / flips | batched max dp / flips |", "|---|---|---|---|"]
    for name, v in data["variants"].items():
        if "parity_pass" not in v:
            if name != "base":
                lines.append(f"| {name} | not run | - | - |")
            continue
        verdict = "PASS" if v["parity_pass"] else "FAIL"
        cells = [
            f"{v['parity'][m]['max_dp']:.5f} / {v['parity'][m]['flips']}" if m in v["parity"] else "-"
            for m in ("singles", "batched")
        ]
        lines.append(f"| {name} | {verdict} | {cells[0]} | {cells[1]} |")
    lines += ["", "## Speed (`clef bench`; same session, ~5% same-day drift)", ""]
    levels = sorted({c for v in data["variants"].values() for c in v["bench"]})
    head = (
        ["variant"]
        + [f"c={c} p50 ms / p95 / req/s" for c in levels]
        + [f"c={c} req/s vs base" for c in levels]
    )
    lines += ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    base = data["variants"]["base"]["bench"]
    for name, v in data["variants"].items():
        if not v["bench"]:
            continue
        cells = []
        for c in levels:
            r = v["bench"].get(c)
            cells.append(f"{r['p50']:.1f} / {r['p95']:.1f} / {r['rps']:.2f}" if r else "-")
        for c in levels:
            r, b = v["bench"].get(c), base.get(c)
            cells.append(f"{100 * (r['rps'] / b['rps'] - 1):+.1f}%" if r and b and b["rps"] else "-")
        lines.append(f"| {name} ({v['desc']}) | " + " | ".join(cells) + " |")
    lines += ["", "Overall: " + ("PARITY FAIL" if failed else "parity PASS (for the variants that ran)")]
    return data, "\n".join(lines) + "\n", failed


def main(argv: list[str]) -> int:
    out = Path(argv[1] if len(argv) > 1 else "bench/out/perf")
    data, md, failed = build(out)
    (out / "summary.json").write_text(json.dumps(data, indent=2))
    (out / "summary.md").write_text(md)
    print(md)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
