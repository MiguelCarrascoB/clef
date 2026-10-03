"""`clef` command line: serve / stop / status / logs / doctor / download / bench / open / version.

Stdlib only at import time (no torch, no httpx, no fastapi): `clef --help`, `clef version` and
`clef doctor --no-gpu` must start fast on a machine without a GPU. Heavy modules are imported lazily
inside the command that needs them.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import shutil
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request
from collections import deque
from collections.abc import Sequence
from pathlib import Path
from typing import Any

# Keep in sync with config.DEVICES / DTYPES / QUANTS (tests/unit/test_cli.py asserts it).
DEVICES = ("auto", "cuda", "rocm", "mps", "cpu")
DTYPES = ("auto", "bfloat16", "float16", "float32")
QUANTS = ("none", "int8", "nf4")
OFFLOADS = ("none", "cpu")

# flag attribute -> environment variable set by `clef serve`
FLAG_ENV = {
    "host": "CLEF_HOST",
    "port": "CLEF_PORT",
    "device": "CLEF_DEVICE",
    "dtype": "CLEF_DTYPE",
    "quant": "CLEF_QUANT",
    "offload": "CLEF_OFFLOAD",
    "max_device_memory_gb": "CLEF_MAX_DEVICE_MEMORY_GB",
    "model_path": "CLEF_MODEL_PATH",
}

MIN_FREE_GB = 21.0  # the release is ~19 GB; leave headroom for the partial download
STOP_GRACE_S = 30.0
LOG_KEEP = 3
READY_STATES = ("ready", "warming")

# creationflags for a detached process on Windows (literals so POSIX tests can check them)
DETACHED_PROCESS = 0x00000008
CREATE_NEW_PROCESS_GROUP = 0x00000200


# --------------------------------------------------------------------------- helpers


def serve_env(args: argparse.Namespace, base: dict[str, str] | None = None) -> dict[str, str]:
    """Environment for the server: `base` (default os.environ) plus the CLEF_* values set by flags."""
    env = dict(os.environ if base is None else base)
    for attr, name in FLAG_ENV.items():
        value = getattr(args, attr, None)
        if value is not None:
            env[name] = str(value)
    return env


def load_config(env: dict[str, str] | None = None) -> Any:
    """clef_server.config.load() with `env` applied to os.environ (torch-free import)."""
    if env is not None:
        os.environ.update(env)
    from clef_server import config

    return config.load()


def client_host(host: str) -> str:
    return "127.0.0.1" if host in ("0.0.0.0", "::", "") else host  # noqa: S104


def base_url(cfg: Any) -> str:
    host = client_host(cfg.host)
    if ":" in host:
        host = f"[{host}]"
    return f"http://{host}:{cfg.port}"


def http_json(url: str, timeout: float = 3.0) -> tuple[int, dict[str, Any]] | None:
    """GET `url`; returns (status, json body) or None when unreachable. /health answers 503 with a body."""
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:  # noqa: S310 (local URL)
            code, raw = resp.status, resp.read()
    except urllib.error.HTTPError as exc:
        code, raw = exc.code, exc.read()
    except (urllib.error.URLError, OSError, ValueError):
        return None
    try:
        body = json.loads(raw.decode("utf-8", "replace") or "{}")
    except ValueError:
        body = {}
    return code, body if isinstance(body, dict) else {}


def livez(url: str) -> bool:
    got = http_json(url + "/livez", timeout=2.0)
    return got is not None and got[0] == 200


def health_status(url: str) -> str:
    got = http_json(url + "/health", timeout=3.0)
    return str(got[1].get("status", "")) if got else ""


# ---- pidfile / process control


def read_pid(path: Path) -> int | None:
    try:
        return int(path.read_text().strip().split()[0])
    except (OSError, ValueError, IndexError):
        return None


def pid_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    if os.name == "nt":
        import ctypes

        kernel32 = ctypes.windll.kernel32  # type: ignore[attr-defined]
        handle = kernel32.OpenProcess(0x1000, False, pid)  # PROCESS_QUERY_LIMITED_INFORMATION
        if not handle:
            return False
        try:
            code = ctypes.c_ulong()
            if not kernel32.GetExitCodeProcess(handle, ctypes.byref(code)):
                return False
            return code.value == 259  # STILL_ACTIVE
        finally:
            kernel32.CloseHandle(handle)
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def running_pid(pidfile: Path) -> int | None:
    """Pid from the pidfile when that process is alive; a stale pidfile is removed."""
    pid = read_pid(pidfile)
    if pid is not None and pid_alive(pid):
        return pid
    if pidfile.exists():
        pidfile.unlink(missing_ok=True)
    return None


def terminate(pid: int, force: bool = False) -> None:
    if os.name == "nt":
        cmd = ["taskkill", "/PID", str(pid), "/T"] + (["/F"] if force else [])
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)
        return
    with contextlib.suppress(ProcessLookupError):
        os.kill(pid, signal.SIGKILL if force else signal.SIGTERM)


def rotate_logs(log: Path, keep: int = LOG_KEEP) -> None:
    """server.log -> server.log.1 -> .2 -> .3 (the oldest is dropped)."""
    try:
        if not log.exists() or log.stat().st_size == 0:
            return
        for i in range(keep - 1, 0, -1):
            src = log.with_name(f"{log.name}.{i}")
            if src.exists():
                os.replace(src, log.with_name(f"{log.name}.{i + 1}"))
        os.replace(log, log.with_name(f"{log.name}.1"))
    except OSError:
        pass


def tail_lines(path: Path, n: int) -> list[str]:
    try:
        with open(path, encoding="utf-8", errors="replace") as fh:
            return [line.rstrip("\n") for line in deque(fh, maxlen=max(n, 0))]
    except OSError:
        return []


def detach_command() -> list[str]:
    return [sys.executable, "-m", "clef_server.main"]


def popen_kwargs() -> dict[str, Any]:
    if os.name == "nt":
        return {"creationflags": DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP, "close_fds": True}
    return {"start_new_session": True, "close_fds": True}


# --------------------------------------------------------------------------- commands


def cmd_serve(args: argparse.Namespace) -> int:
    env = serve_env(args)
    try:
        cfg = load_config(env)
    except ValueError as exc:
        print(f"clef: invalid configuration: {exc}", file=sys.stderr)
        return 2
    if not args.detach:
        from clef_server.main import run

        try:
            run(cfg)
        except KeyboardInterrupt:
            print("\nclef: stopped")
        return 0
    return serve_detached(cfg, env, args.timeout)


def serve_detached(cfg: Any, env: dict[str, str], timeout: float) -> int:
    from clef_server import paths

    url = base_url(cfg)
    pidfile, log = paths.pid_file(cfg), paths.log_file(cfg)
    existing = running_pid(pidfile)
    if existing is not None:
        print(f"clef: already running (pid {existing}); use `clef stop` first", file=sys.stderr)
        return 1
    if livez(url):
        print(
            f"clef: something already answers on {url} (not managed by {pidfile}). Stop it first.",
            file=sys.stderr,
        )
        return 1
    rotate_logs(log)
    print(f"starting clef on {url} (log: {log})")
    with open(log, "ab") as logfh:
        proc = subprocess.Popen(  # noqa: S603
            detach_command(),
            stdin=subprocess.DEVNULL,
            stdout=logfh,
            stderr=subprocess.STDOUT,
            env=env,
            **popen_kwargs(),
        )
    pidfile.write_text(f"{proc.pid}\n")

    def fail(msg: str) -> int:
        print(f"clef: {msg}; last log lines:", file=sys.stderr)
        for line in tail_lines(log, 20):
            print("  " + line, file=sys.stderr)
        return 1

    start = time.monotonic()
    while not livez(url):  # phase 1: process is up
        if proc.poll() is not None:
            pidfile.unlink(missing_ok=True)
            return fail(f"server exited during startup (code {proc.returncode})")
        if time.monotonic() - start > timeout:
            return fail(f"timeout ({timeout:.0f}s) waiting for /livez")
        time.sleep(0.5)
    status = ""
    while status not in READY_STATES:  # phase 2: model loaded
        status = health_status(url)
        if status in READY_STATES:
            break
        if proc.poll() is not None:
            pidfile.unlink(missing_ok=True)
            return fail(f"server exited while loading (code {proc.returncode})")
        if status == "error":
            terminate(proc.pid, force=True)  # do not leave a broken server behind
            pidfile.unlink(missing_ok=True)
            return fail("server reports status=error (stopped it)")
        if time.monotonic() - start > timeout:
            return fail(f"timeout ({timeout:.0f}s) waiting for the model; status={status or 'unknown'}")
        time.sleep(1.0)
    print(f"SERVER UP (pid {proc.pid}, status={status}, {time.monotonic() - start:.0f}s) -> {url}")
    return 0


def cmd_stop(args: argparse.Namespace) -> int:
    cfg = load_config(serve_env(args))
    from clef_server import paths

    pidfile = paths.pid_file(cfg)
    pid = running_pid(pidfile)
    if pid is None:
        if livez(base_url(cfg)):
            print(
                f"clef: a server answers on {base_url(cfg)} but was not started with `clef serve --detach`"
                " (no pidfile); stop it where it runs",
                file=sys.stderr,
            )
            return 1
        print("not running")
        return 0
    print(f"stopping pid {pid}")
    terminate(pid)
    # Windows has no graceful signal for a console-less process: taskkill /T only works after a short grace.
    grace = 5.0 if os.name == "nt" else STOP_GRACE_S
    deadline = time.monotonic() + grace
    while pid_alive(pid) and time.monotonic() < deadline:
        time.sleep(0.5)
    if pid_alive(pid):
        print(f"still alive after {grace:.0f}s, killing")
        terminate(pid, force=True)
        time.sleep(1.0)
    pidfile.unlink(missing_ok=True)
    print("stopped")
    return 0


def cmd_status(args: argparse.Namespace) -> int:
    cfg = load_config(serve_env(args))
    from clef_server import paths

    url = base_url(cfg)
    pid = running_pid(paths.pid_file(cfg))
    got = http_json(url + "/health", timeout=5.0)
    body = got[1] if got else {}
    status = str(body.get("status", "")) if got else "unreachable"
    healthy = status in READY_STATES
    if args.json:
        print(
            json.dumps(
                {
                    "url": url,
                    "pid": pid,
                    "running": pid is not None,
                    "health": body or None,
                    "status": status,
                },
                indent=2,
            )
        )
        return 0 if healthy else 1
    proc = f"running (pid {pid})" if pid is not None else "not managed / not running"
    print(f"process: {proc}")
    print(f"url:     {url}")
    print(f"status:  {status or 'unknown'}")
    if got:
        gpu = body.get("gpu") if isinstance(body.get("gpu"), dict) else {}
        backend = body.get("backend")
        backend = backend.get("name") if isinstance(backend, dict) else backend
        for label, value in (
            ("version", body.get("version")),
            ("backend", backend),
            ("device", body.get("device") or gpu.get("name")),
            ("dtype", body.get("dtype")),
            ("error", body.get("error")),
        ):
            if value not in (None, ""):
                print(f"{label + ':':<9}{value}")
    return 0 if healthy else 1


def cmd_logs(args: argparse.Namespace) -> int:
    cfg = load_config(serve_env(args))
    from clef_server import paths

    log = paths.log_file(cfg)
    if not log.exists() and not args.follow:
        print(f"no log file at {log}", file=sys.stderr)
        return 1
    for line in tail_lines(log, args.lines):
        print(line)
    if not args.follow:
        return 0
    with contextlib.suppress(KeyboardInterrupt):
        follow(log)
    return 0


def follow(path: Path, poll_s: float = 0.5, stop: Any = None) -> None:
    """Pure-Python `tail -F`: prints new lines, reopens after rotation/truncation."""
    fh = None
    pos = 0
    while stop is None or not stop():
        try:
            size = path.stat().st_size
        except OSError:
            size = -1
        if fh is None and size >= 0:
            fh = open(path, encoding="utf-8", errors="replace")  # noqa: SIM115
            fh.seek(0, os.SEEK_END)
            pos = fh.tell()
        elif fh is not None and (size < 0 or size < pos):
            fh.close()
            fh, pos = None, 0
            if size >= 0:
                fh = open(path, encoding="utf-8", errors="replace")  # noqa: SIM115
        if fh is not None:
            chunk = fh.read()
            if chunk:
                sys.stdout.write(chunk)
                sys.stdout.flush()
                pos = fh.tell()
                continue
        time.sleep(poll_s)


def cmd_doctor(args: argparse.Namespace) -> int:
    argv = []
    if args.no_gpu:
        argv.append("--no-gpu")
    if args.smoke:
        argv.append("--smoke")
    if args.json:
        argv.append("--json")
    try:
        from clef_server import doctor
    except ImportError as exc:
        print(f"clef: doctor is unavailable ({exc})", file=sys.stderr)
        return 1
    return int(doctor.main(argv) or 0)


def hf_cache_dir() -> Path:
    if os.environ.get("HF_HUB_CACHE"):
        return Path(os.environ["HF_HUB_CACHE"]).expanduser()
    if os.environ.get("HF_HOME"):
        return Path(os.environ["HF_HOME"]).expanduser() / "hub"
    return Path.home() / ".cache" / "huggingface" / "hub"


def free_gb(path: Path) -> float:
    """Free space (GB) on the filesystem that holds `path` (walks up to an existing parent)."""
    probe = path.expanduser()
    while not probe.exists() and probe != probe.parent:
        probe = probe.parent
    return shutil.disk_usage(probe).free / 1e9


def cmd_download(args: argparse.Namespace) -> int:
    from clef_server import config, paths

    repo = os.environ.get("CLEF_MODEL_REPO", config.MODEL_REPO)
    revision = args.revision or os.environ.get("CLEF_MODEL_REVISION", config.MODEL_REVISION)
    target = args.dir or os.environ.get("CLEF_MODEL_PATH") or None
    dest = Path(target).expanduser() if target else hf_cache_dir()
    free = free_gb(dest)
    print(f"model:       {repo}@{revision[:12]}")
    print(f"destination: {dest}{'' if target else ' (Hugging Face cache)'}")
    print(f"size:        about {paths.WEIGHTS_GB:.0f} GB; free space: {free:.1f} GB")
    if free < MIN_FREE_GB:
        print(f"clef: need at least {MIN_FREE_GB:.0f} GB free on that disk; aborting", file=sys.stderr)
        return 1
    if not args.yes:
        if not sys.stdin.isatty():
            print("clef: pass --yes to download non-interactively", file=sys.stderr)
            return 1
        if input("Download now? [y/N] ").strip().lower() not in ("y", "yes"):
            print("aborted")
            return 1
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print(
            "clef: huggingface_hub is missing; install the server extra: pip install 'clef-local[server]'",
            file=sys.stderr,
        )
        return 1
    kwargs: dict[str, Any] = {"revision": revision}
    if target:
        kwargs["local_dir"] = str(dest)
    path = snapshot_download(repo, **kwargs)  # resumable; re-running continues where it stopped
    print(f"downloaded to {path}")
    if target:
        print(f"use it with: CLEF_MODEL_PATH={path} clef serve")
    return 0


def cmd_bench(args: argparse.Namespace) -> int:
    from clef_server import httpbench

    return int(httpbench.main(args.rest))


def cmd_open(args: argparse.Namespace) -> int:
    import webbrowser

    url = base_url(load_config(serve_env(args))) + "/"
    print(url)
    if not webbrowser.open(url):
        print("clef: could not open a browser; open the URL manually", file=sys.stderr)
        return 1
    return 0


def cmd_version(_args: argparse.Namespace) -> int:
    from clef_server.config import VERSION

    print(f"clef-local {VERSION}")
    return 0


# --------------------------------------------------------------------------- parser


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="clef",
        description="Local server for the clef-flash decision model (CUDA, ROCm, Apple Silicon, CPU).",
    )
    sub = p.add_subparsers(dest="command", metavar="<command>")

    def netflags(sp: argparse.ArgumentParser) -> None:
        sp.add_argument("--host", help="server host (default CLEF_HOST or 127.0.0.1)")
        sp.add_argument("--port", type=int, help="server port (default CLEF_PORT or 8910)")

    sp = sub.add_parser("serve", help="run the server (foreground; --detach to background it)")
    netflags(sp)
    sp.add_argument("--device", choices=DEVICES, help="CLEF_DEVICE")
    sp.add_argument("--dtype", choices=DTYPES, help="CLEF_DTYPE")
    sp.add_argument("--quant", choices=QUANTS, help="CLEF_QUANT: smaller weights (see docs/memory.md)")
    sp.add_argument(
        "--offload",
        choices=OFFLOADS,
        help="CLEF_OFFLOAD: cpu keeps embeddings and what exceeds the device cap in host RAM",
    )
    sp.add_argument(
        "--max-device-memory-gb",
        dest="max_device_memory_gb",
        type=float,
        metavar="GB",
        help="CLEF_MAX_DEVICE_MEMORY_GB: device memory this server may use (0 = no cap)",
    )
    sp.add_argument("--model-path", dest="model_path", metavar="DIR", help="CLEF_MODEL_PATH")
    sp.add_argument("--detach", action="store_true", help="start in the background and wait until healthy")
    sp.add_argument(
        "--timeout",
        type=float,
        default=600.0,
        metavar="S",
        help="--detach: seconds to wait for the model (default 600)",
    )
    sp.set_defaults(func=cmd_serve)

    sp = sub.add_parser("stop", help="stop the detached server")
    netflags(sp)
    sp.set_defaults(func=cmd_stop)

    sp = sub.add_parser("status", help="process and /health summary (exit 0 when healthy)")
    netflags(sp)
    sp.add_argument("--json", action="store_true")
    sp.set_defaults(func=cmd_status)

    sp = sub.add_parser("logs", help="print the server log")
    sp.add_argument("-f", "--follow", action="store_true")
    sp.add_argument("-n", "--lines", type=int, default=100)
    sp.set_defaults(func=cmd_logs)

    sp = sub.add_parser("doctor", help="check the environment (exit 1 on any FAIL)")
    sp.add_argument("--no-gpu", action="store_true", help="skip checks that need a GPU or the weights")
    sp.add_argument("--smoke", action="store_true", help="load the model and run one record")
    sp.add_argument("--json", action="store_true")
    sp.set_defaults(func=cmd_doctor)

    sp = sub.add_parser("download", help="download the pinned model weights (about 19 GB, resumable)")
    sp.add_argument("--revision", metavar="SHA")
    sp.add_argument("--dir", metavar="DIR", help="download here instead of the Hugging Face cache")
    sp.add_argument("--yes", "-y", action="store_true", help="do not ask for confirmation")
    sp.set_defaults(func=cmd_download)

    sp = sub.add_parser(
        "bench", help="HTTP load test against the running server (see --help after bench)", add_help=False
    )
    sp.add_argument("rest", nargs=argparse.REMAINDER)
    sp.set_defaults(func=cmd_bench)

    sp = sub.add_parser("open", help="open the console in the browser")
    netflags(sp)
    sp.set_defaults(func=cmd_open)

    sp = sub.add_parser("version", help="print the version")
    sp.set_defaults(func=cmd_version)
    return p


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    raw = list(sys.argv[1:] if argv is None else argv)
    if raw and raw[0] == "bench":  # argparse REMAINDER mishandles a leading --option: pass it straight on
        return cmd_bench(argparse.Namespace(rest=raw[1:]))
    args = parser.parse_args(raw)
    if not getattr(args, "func", None):
        parser.print_help()
        return 0
    try:
        return int(args.func(args) or 0)
    except KeyboardInterrupt:
        return 130
    except BrokenPipeError:
        return 0


if __name__ == "__main__":
    sys.exit(main())
