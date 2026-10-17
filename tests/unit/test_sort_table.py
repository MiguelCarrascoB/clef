"""bench/sort_table.py and the input generator of bench/jobs_bench.py (no GPU needed)."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path


def load(name: str):
    spec = importlib.util.spec_from_file_location(
        name, Path(__file__).resolve().parents[2] / "bench" / f"{name}.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


sort_table = load("sort_table")
jobs_bench = load("jobs_bench")


def run(tag: str, rate: float, repeat: int = 0, labels=None, p95=100.0, status="succeeded") -> dict:
    return {
        "tag": tag,
        "repeat": repeat,
        "status": status,
        "items_per_s": rate,
        "labels": labels if labels is not None else ["a", "b", "a", "b"],
        "interactive_p95_ms": p95,
    }


def summarise(*runs: dict) -> dict:
    grouped: dict[str, list[dict]] = {}
    for r in runs:
        grouped.setdefault(r["tag"], []).append(r)
    return sort_table.summarise(grouped)


def test_gain_above_ten_percent_and_the_noise_flips():
    data = summarise(run("base", 100, 0), run("base", 102, 1), run("sort", 125, 0), run("sort", 128, 1))
    v = data["variants"]["sort"]
    assert round(v["gain"], 3) == round(126.5 / 101 - 1, 3) and v["verdict"] == "FLIP"
    assert v["agreement"] == 1.0
    assert "FLIP" in sort_table.render(data)


def test_gain_under_the_gate_or_inside_the_noise_keeps_the_flag_off():
    small = summarise(run("base", 100), run("sort", 108))
    assert small["variants"]["sort"]["verdict"] == "keep off"
    noisy = summarise(run("base", 100, 0), run("base", 140, 1), run("sort", 125, 0))  # base spread 33%
    assert noisy["variants"]["sort"]["verdict"] == "keep off"


def test_label_disagreement_and_interactive_regression_are_called_out():
    bad_labels = summarise(run("base", 100), run("both", 150, labels=["b", "a", "b", "a"]))
    assert bad_labels["variants"]["both"]["verdict"] == "CHECK labels"
    slow = summarise(run("base", 100, p95=100), run("ahead", 150, p95=300))
    assert slow["variants"]["ahead"]["verdict"] == "FLIP (interactive p95 worse)"


def test_failed_variant_and_missing_base(tmp_path):
    data = summarise(run("base", 100), run("sort", 0, status="failed"))
    assert data["variants"]["sort"]["verdict"] == "FAILED"
    assert "No successful base run" in sort_table.render(summarise(run("sort", 100)))
    (tmp_path / "runs.jsonl").write_text(
        "\n".join(json.dumps(r) for r in (run("base", 100), run("sort", 120)))
    )
    assert sort_table.main(["x", str(tmp_path)]) == 0
    assert (tmp_path / "summary.md").exists() and json.loads((tmp_path / "summary.json").read_text())


def test_bench_inputs_are_long_tailed_unique_and_reproducible():
    a = jobs_bench.make_inputs(300, seed=7, median_words=60, max_words=700)
    assert a == jobs_bench.make_inputs(300, seed=7, median_words=60, max_words=700)
    assert len(set(a)) == 300 and a != jobs_bench.make_inputs(300, seed=8, median_words=60, max_words=700)
    sizes = sorted(len(x.split()) for x in a)
    assert sizes[-10] > 3 * sizes[len(sizes) // 2] and max(sizes) <= 701  # a real tail, capped
