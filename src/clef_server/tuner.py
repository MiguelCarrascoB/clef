"""TunableOp background tuner: a separate process, so the server's one GPU thread never blocks on tuning.

    python -m clef_server.tuner <untuned base> <results file> <skip file>

The server (engine.py) records the GEMMs it ran untuned to ``<untuned base>`` (torch writes
``<stem><ordinal><suffix>``), starts this process once it has been quiet for a while, pauses it with SIGSTOP
while requests arrive, resumes it with SIGCONT, and reloads ``<results file>`` whenever it changes.

This process owns the results file. It loads it, tunes each recorded GEMM that has no result yet, one at a
time (torch writes each result at once), and appends the GEMMs the offline tuner cannot tune to
``<skip file>`` so nobody retries them. It exits when nothing is left, or when the server is gone. A tuning
budget set in the environment (PYTORCH_TUNABLEOP_MAX_TUNING_*) applies here too.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path


def gemm_key(line: str) -> tuple[str, ...]:
    """(operator, parameters) of a TunableOp CSV line: what a tuned result is keyed on."""
    return tuple(line.split(",")[:2])


def recorded(base: Path) -> list[str]:
    """Recorded GEMM lines, unique by key, in recording order."""
    seen: set[tuple[str, ...]] = set()
    out = []
    for f in sorted(base.parent.glob(f"{base.stem}*{base.suffix}")):
        try:
            text = f.read_text()
        except OSError:
            continue
        for line in text.splitlines():
            key = gemm_key(line)
            if line.startswith(("Gemm", "ScaledGemm")) and key not in seen:
                seen.add(key)
                out.append(line)
    return out


def skipped(skip: Path) -> set[tuple[str, ...]]:
    try:
        return {gemm_key(line) for line in skip.read_text().splitlines() if line}
    except OSError:
        return set()


def pending(base: Path, skip: Path, tuned: set[tuple[str, ...]]) -> list[str]:
    """Recorded GEMMs with no tuned result that were not given up on."""
    gave_up = skipped(skip)
    return [line for line in recorded(base) if gemm_key(line) not in tuned and gemm_key(line) not in gave_up]


def main(argv: list[str]) -> int:
    base, results, skip = Path(argv[1]), Path(argv[2]), Path(argv[3])
    parent = os.getppid()
    # the server's own PYTORCH_TUNABLEOP_FILENAME pattern (with %d) is inherited, so torch writes ``results``
    os.environ.setdefault("PYTORCH_TUNABLEOP_FILENAME", str(results))
    os.environ.update(
        PYTORCH_TUNABLEOP_ENABLED="1", PYTORCH_TUNABLEOP_TUNING="1", PYTORCH_TUNABLEOP_RECORD_UNTUNED="0"
    )
    os.environ.pop("PYTORCH_TUNABLEOP_UNTUNED_FILENAME", None)
    from torch.cuda import tunable

    tunable.get_results()  # loads the results file while tuning is on (torch reads it on first use only)
    step = results.with_name("tunableop_tuner_step.csv")  # outside the untuned <stem>*<suffix> glob
    done = 0
    while os.getppid() == parent:  # the server is gone: stop
        tuned = {(r[0], r[1]) for r in tunable.get_results()}
        todo = pending(base, skip, tuned)
        if not todo:
            break
        line = todo[0]
        step.write_text(line + "\n")
        try:
            tunable.tune_gemm_in_file(str(step))
            ok = any((r[0], r[1]) == gemm_key(line) for r in tunable.get_results())
        except Exception as exc:  # e.g. a shape the offline tuner does not support
            print(f"tuner: cannot tune {line[:80]}: {exc}", flush=True)
            ok = False
        if ok:
            done += 1
            print(f"tuner: tuned {line[:80]} ({done}, {len(todo) - 1} left)", flush=True)
        else:
            with skip.open("a") as f:
                f.write(line + "\n")
    step.unlink(missing_ok=True)
    print(f"tuner: done, {done} GEMMs tuned", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
