"""clef_server.tuner: the TunableOp background tuner process (CPU only: torch.cuda.tunable is replaced)."""

from __future__ import annotations

import os
from pathlib import Path

import torch

from clef_server import tuner

OP = "GemmTunableOp_BFloat16_TN"


def test_pending_skips_tuned_given_up_and_repeated_gemms(tmp_path):
    (tmp_path / "tunableop_untuned0.csv").write_text(
        f"Validator,PT_VERSION,2.11.0\n{OP},tn_a\n{OP},tn_a\n{OP},tn_done\n{OP},tn_skip\n{OP},tn_b\n"
    )
    (tmp_path / "tunableop_skip.csv").write_text(f"{OP},tn_skip\n")
    todo = tuner.pending(
        tmp_path / "tunableop_untuned.csv", tmp_path / "tunableop_skip.csv", {(OP, "tn_done")}
    )
    assert todo == [f"{OP},tn_a", f"{OP},tn_b"]


def test_main_tunes_each_pending_gemm_and_gives_up_on_silent_ones(tmp_path, monkeypatch):
    (tmp_path / "tunableop_untuned0.csv").write_text(f"{OP},tn_a\n{OP},tn_silent\n{OP},tn_b\n")
    results, tuned = [], []

    def tune_gemm_in_file(path):
        params = Path(path).read_text().strip().split(",")[1]
        tuned.append(params)
        if params != "tn_silent":  # torch tunes nothing for some shapes, without an error
            results.append((OP, params, "Gemm_Rocblas_1", 0.1))

    monkeypatch.setattr(torch.cuda.tunable, "get_results", lambda: list(results))
    monkeypatch.setattr(torch.cuda.tunable, "tune_gemm_in_file", tune_gemm_in_file)
    for name in ("PYTORCH_TUNABLEOP_ENABLED", "PYTORCH_TUNABLEOP_TUNING", "PYTORCH_TUNABLEOP_FILENAME"):
        monkeypatch.delenv(name, raising=False)
    base, skip = tmp_path / "tunableop_untuned.csv", tmp_path / "tunableop_skip.csv"
    assert tuner.main(["tuner", str(base), str(tmp_path / "tunableop_0.csv"), str(skip)]) == 0
    assert tuned == ["tn_a", "tn_silent", "tn_b"]  # each once, in recording order
    assert skip.read_text() == f"{OP},tn_silent\n"  # never retried
    assert (
        os.environ["PYTORCH_TUNABLEOP_TUNING"] == "1" and not (tmp_path / "tunableop_tuner_step.csv").exists()
    )
