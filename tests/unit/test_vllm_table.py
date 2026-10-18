# ruff: noqa: E501
"""bench/vllm/vllm_table.py: the speed table and the D12 decision (no GPU)."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

_BENCH = Path(__file__).resolve().parents[2] / "bench" / "vllm"
spec = importlib.util.spec_from_file_location("vllm_table", _BENCH / "vllm_table.py")
vt = importlib.util.module_from_spec(spec)
sys.modules["vllm_table"] = vt
spec.loader.exec_module(vt)


def table(rows):  # rows: {conc: (ok, err, rps, p50, p95)}
    out = ["server x", "conc ok err req/s | p50 p95 p99 ms | server batch_size distribution"]
    for c, (ok, err, rps, p50, p95) in rows.items():
        out.append(
            f"{c:>4} {ok:>5} {err:>4} {rps:>7.1f} | {p50:>7.1f} {p95:>7.1f} {p95 + 5:>7.1f}    | avg 1.00  [1:{ok}]"
        )
    return "\n".join(out) + "\n"


CLEF = {1: (100, 0, 6.6, 149.0, 159.0), 16: (100, 0, 8.9, 1700.0, 1800.0)}
GOOD = {1: (100, 0, 7.0, 120.0, 130.0), 16: (100, 0, 13.0, 900.0, 1000.0)}
PARITY_OK = "singles  max |dp| = 0.00100  mean |dp| = 0.00002  top-1 flips 0/600  OK\nPARITY OK\n"
PARITY_BAD = PARITY_OK.replace("flips 0/600  OK", "flips 3/600  FAIL")


def write(tmp, vllm, parity=(PARITY_OK, 0), both=None):
    (tmp / "bench_clef_base.txt").write_text(table(CLEF))
    if both:
        (tmp / "bench_clef_both.txt").write_text(table(both))
    (tmp / "bench_vllm.txt").write_text(table(vllm))
    if parity:
        (tmp / "parity_vllm.txt").write_text(parity[0])
        (tmp / "parity_vllm.rc").write_text(str(parity[1]))
    (tmp / "vram_vllm.json").write_text(
        json.dumps({"peak_gb": 20.0, "total_gb": 24.0, "idle_gb": 1.0, "delta_gb": 19.0})
    )


def verdict(tmp, **kw):
    return vt.build(tmp, vt.Thresholds(**kw))[2]


def test_adopt_when_every_rule_passes(tmp_path):
    write(tmp_path, GOOD)
    assert verdict(tmp_path) == "ADOPT"


def test_speedup_below_threshold_is_do_not_adopt(tmp_path):
    write(tmp_path, {1: (100, 0, 7.0, 120.0, 130.0), 16: (100, 0, 10.0, 900.0, 1000.0)})  # 10 / 8.9 = 1.12x
    assert verdict(tmp_path) == "DO NOT ADOPT"
    assert verdict(tmp_path, min_speedup=1.1) == "ADOPT"  # thresholds are arguments


def test_best_clef_line_is_the_baseline(tmp_path):
    fast = {1: (100, 0, 6.6, 140.0, 150.0), 16: (100, 0, 11.0, 1500.0, 1600.0)}
    write(tmp_path, GOOD, both=fast)
    assert verdict(tmp_path) == "DO NOT ADOPT"  # 13 / 11 = 1.18x against the faster clef line


def test_parity_failure_beats_a_big_speedup(tmp_path):
    write(
        tmp_path, {1: (100, 0, 7.0, 120.0, 130.0), 16: (100, 0, 30.0, 400.0, 500.0)}, parity=(PARITY_BAD, 1)
    )
    assert verdict(tmp_path) == "DO NOT ADOPT"


def test_worse_p95_or_errors_or_missing_parity_is_inconclusive(tmp_path):
    write(tmp_path, {1: (100, 0, 7.0, 120.0, 130.0), 16: (100, 0, 13.0, 900.0, 2500.0)})
    assert verdict(tmp_path) == "INCONCLUSIVE"  # p95 worse at c=16
    write(tmp_path, {1: (99, 1, 7.0, 120.0, 130.0), 16: (100, 0, 13.0, 900.0, 1000.0)})
    assert verdict(tmp_path) == "INCONCLUSIVE"
    (tmp_path / "parity_vllm.rc").unlink()
    write(tmp_path, GOOD, parity=None)
    assert verdict(tmp_path) == "INCONCLUSIVE"


def test_nothing_ran_is_inconclusive_and_the_table_still_builds(tmp_path):
    data, md, v = vt.build(tmp_path, vt.Thresholds())
    assert v == "INCONCLUSIVE" and "vllm" in md and data["verdict"] == v


def test_table_and_cli(tmp_path):
    write(tmp_path, GOOD)
    assert vt.main([str(tmp_path)]) == 0
    md = (tmp_path / "summary.md").read_text()
    assert "Decision (D12): ADOPT" in md and "1.46x" in md and "20.0 (+19.0)" in md
    assert json.loads((tmp_path / "summary.json").read_text())["verdict"] == "ADOPT"
