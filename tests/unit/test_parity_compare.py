"""bench/parity.py compare(): near-tie flips are reported apart from real top-1 flips."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

BENCH = Path(__file__).resolve().parents[2] / "bench"
spec = importlib.util.spec_from_file_location("parity_bench", BENCH / "parity.py")
parity = importlib.util.module_from_spec(spec)
spec.loader.exec_module(parity)


def test_near_tie_flip_is_reported_separately():
    ref = [{"q.a": 0.5003, "q.b": 0.4997, "r.a": 0.9, "r.b": 0.1}]
    cand = [{"q.a": 0.4998, "q.b": 0.5002, "r.a": 0.1, "r.b": 0.9}]
    st = parity.compare(ref, cand)
    assert st["tie_flips"] == 1  # q: ref margin 6e-4 < 1e-3
    assert st["top1_flips"] == 1  # r: a real flip
    assert st["questions"] == 2


def test_no_flip_no_counts():
    ref = [{"q.a": 0.6, "q.b": 0.4}]
    st = parity.compare(ref, ref)
    assert st["top1_flips"] == 0 and st["tie_flips"] == 0


def test_noise_tolerance_is_max_of_floor_and_twice_noise(tmp_path, capsys):
    dump = {"singles": [{"q.a": 0.5, "q.b": 0.5}], "batched": [{"q.a": 0.5, "q.b": 0.5}]}
    other = {"singles": [{"q.a": 0.504, "q.b": 0.496}], "batched": [{"q.a": 0.5, "q.b": 0.5}]}
    a, b = tmp_path / "a.json", tmp_path / "b.json"
    a.write_text(json.dumps(dump))
    b.write_text(json.dumps(other))
    assert parity.noise_dp(str(a), str(b)) == pytest.approx(0.004)
    assert parity.effective_tol(0.002, 0.004) == pytest.approx(0.008)
    assert parity.effective_tol(0.002, 0.0001) == 0.002
