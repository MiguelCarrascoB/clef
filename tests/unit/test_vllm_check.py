"""bench/vllm/vllm_check.sh with stub docker / curl / python (no GPU, no Docker): review E/F #5.

A run that preflight refuses must not touch another run's container or clef server; a run cleans up only the
container id it wrote and the clef state dirs of its own run; an unreadable GPU (VRAM ``nan``) is refused
unless FORCE=1.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "bench" / "vllm" / "vllm_check.sh"

pytestmark = pytest.mark.skipif(sys.platform == "win32" or shutil.which("bash") is None, reason="needs bash")

DOCKER = """#!/usr/bin/env bash
echo "docker $*" >> "$STUBLOG"
case "$1" in
  info) [ "${2:-}" = --format ] && echo /; exit 0;;
  ps) [ -n "${STUB_PS:-}" ] && echo "$STUB_PS"; exit 0;;
  run) [ "${2:-}" = -d ] && echo cid-of-this-run; exit 0;;
  inspect) echo false; exit 0;;
  image) echo "image id sha256:x"; exit 0;;
esac
exit 0
"""

PY = """#!/usr/bin/env bash
case "$1" in
  -) cat >/dev/null; echo "Cloudflare/clef-flash rev0123";;
  -m) echo "clef ${*:3} state=$CLEF_STATE_DIR" >> "$STUBLOG";;
  bench/vllm/load.py) [ "${2:-}" = --print-vram ] && echo "${STUB_VRAM:-nan}";;
  bench/parity.py) exit "${STUB_PARITY_RC:-0}";;
esac
exit 0
"""


def run(tmp_path: Path, **env: str) -> tuple[subprocess.CompletedProcess[str], list[str]]:
    stubs = tmp_path / "bin"
    stubs.mkdir(exist_ok=True)
    for name, body in (("docker", DOCKER), ("curl", "#!/usr/bin/env bash\nexit 7\n"), ("py", PY)):
        (stubs / name).write_text(body)
        (stubs / name).chmod(0o755)
    wsl = tmp_path / "wsl"
    wsl.mkdir(exist_ok=True)
    for name in ("dxg", "libdxcore.so", "librocdxg.so", "dids.conf"):
        (wsl / name).write_text("")
    log = tmp_path / "stub.log"
    log.write_text("")
    full = {
        "PATH": f"{stubs}:{os.environ['PATH']}",
        "HOME": str(tmp_path),
        "STUBLOG": str(log),
        "PY": str(stubs / "py"),
        "OUT": str(tmp_path / "out"),
        "HF_HOME": str(tmp_path / "hf"),
        "DXG": str(wsl / "dxg"),
        "DXCORE": str(wsl / "libdxcore.so"),
        "ROCDXG": str(wsl / "librocdxg.so"),
        "DIDS": str(wsl / "dids.conf"),
        "SKIP_PULL": "1",
        "VLLM_TIMEOUT": "5",
        **env,
    }
    proc = subprocess.run(
        ["bash", str(SCRIPT)], env=full, capture_output=True, text=True, timeout=120, check=False
    )
    return proc, log.read_text().splitlines()


def test_a_refused_run_does_not_remove_another_runs_container_or_stop_its_clef(tmp_path):
    proc, log = run(tmp_path, STUB_PS="clef-vllm-bench")  # another run's container is up
    assert proc.returncode == 2 and "exists" in proc.stderr
    assert not [x for x in log if x.startswith("docker rm")], log
    assert not [x for x in log if x.startswith("clef stop")], log


def test_an_unreadable_gpu_is_refused_unless_forced(tmp_path):
    proc, log = run(tmp_path, STUB_VRAM="nan")
    assert proc.returncode == 2 and "FORCE=1" in proc.stderr and "VRAM" in proc.stderr
    assert not [x for x in log if x.startswith("docker run") or x.startswith("clef ")], log
    forced, log = run(tmp_path, STUB_VRAM="nan", FORCE="1", STUB_PARITY_RC="1")
    assert forced.returncode == 2 and "engine dump failed" in forced.stderr  # got past preflight


def test_a_run_removes_only_the_container_it_started_and_stops_only_its_own_clef(tmp_path):
    proc, log = run(tmp_path, STUB_VRAM="1.0")
    assert proc.returncode == 1, proc.stdout[-2000:] + proc.stderr  # vLLM "never became healthy"
    removed = [x for x in log if x.startswith("docker rm")]
    assert removed and all(x.split()[-1] == "cid-of-this-run" for x in removed), removed
    stops = [x for x in log if x.startswith("clef stop")]
    runs = sorted((tmp_path / "out").glob("run-*"))
    assert len(runs) == 1 and stops and all(f"state={runs[0]}/state-" in x for x in stops), stops


def test_a_failure_after_preflight_cleans_up_this_run_only(tmp_path):
    proc, log = run(tmp_path, STUB_VRAM="1.0", STUB_PARITY_RC="1")  # step 2 fails: no container yet
    assert proc.returncode == 2
    assert not [x for x in log if x.startswith("docker rm")], log
    runs = sorted((tmp_path / "out").glob("run-*"))
    assert all(f"state={runs[0]}/state-" in x for x in log if x.startswith("clef stop"))


def test_script_is_clean_for_bash_n_and_shellcheck():
    assert subprocess.run(["bash", "-n", str(SCRIPT)], check=False).returncode == 0
    if shutil.which("shellcheck") is None:
        pytest.skip("shellcheck not installed")
    res = subprocess.run(["shellcheck", str(SCRIPT)], capture_output=True, text=True, check=False)
    assert res.returncode == 0, res.stdout
