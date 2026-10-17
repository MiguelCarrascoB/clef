"""bench/perf_table.py parses the outputs of the other bench scripts (no GPU needed)."""

from __future__ import annotations

import importlib.util
from pathlib import Path

spec = importlib.util.spec_from_file_location(
    "perf_table", Path(__file__).resolve().parents[2] / "bench" / "perf_table.py"
)
perf_table = importlib.util.module_from_spec(spec)
spec.loader.exec_module(perf_table)

BENCH = """server 3.2.0 status=ready AMD
conc    ok  err   req/s |     p50     p95     p99 ms | server batch_size distribution
   1    30    0     6.6 |   149.0   159.2   160.1    | avg 1.00  [1:30]
  16    30    0     8.9 |  1700.5  1800.0  1850.2    | avg 7.80  [7:10 8:20]
"""
PARITY_OK = """singles  max |dp| = 0.00100  mean |dp| = 0.000020  top-1 flips 0/600  OK
batched  max |dp| = 0.00120  mean |dp| = 0.000030  top-1 flips 0/600  OK
PARITY OK
"""
PARITY_BAD = PARITY_OK.replace("flips 0/600  OK", "flips 2/600  FAIL", 1)
PROFILE = """=== single: 1 record(s), 330 tokens, wall 160.0 ms (profiler overhead), GPU kernel time 70.0 ms
       58.1 83.0%  GEMM
=== batch8: 8 record(s), 2640 tokens, wall 900.0 ms (profiler overhead included), GPU kernel time 450.0 ms
      370.0 82.2%  GEMM
"""


def test_parsers():
    rows = perf_table.parse_bench(BENCH)
    assert rows[1]["p50"] == 149.0 and rows[16]["rps"] == 8.9 and rows[16]["avg_batch"] == 7.8
    assert perf_table.parse_parity(PARITY_BAD)["singles"]["flips"] == 2
    prof = perf_table.parse_profile(PROFILE)
    assert round(prof["batch8"]["kernel_over_wall"], 2) == 0.5 and prof["batch8"]["gemm_pct"] == 82.2


def test_build_pass_and_fail(tmp_path):
    (tmp_path / "bench_base.txt").write_text(BENCH)
    (tmp_path / "bench_mask.txt").write_text(BENCH.replace("6.6", "7.4"))
    (tmp_path / "profile.txt").write_text(PROFILE)
    (tmp_path / "parity_mask.txt").write_text(PARITY_OK)
    (tmp_path / "parity_mask.rc").write_text("0\n")
    data, md, failed = perf_table.build(tmp_path)
    assert not failed and "| mask | PASS |" in md and "+12.1%" in md and "NOT GEMM-bound" in md
    (tmp_path / "parity_head.txt").write_text(PARITY_BAD)
    (tmp_path / "parity_head.rc").write_text("1\n")
    _, md, failed = perf_table.build(tmp_path)
    assert failed and "| head | FAIL |" in md and "PARITY FAIL" in md
