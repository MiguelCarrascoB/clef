"""`clef` command line: serve / stop / status / logs / doctor / download / bench / open / version / models.

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

# Keep in sync with config.DEVICES / DTYPES / QUANTS / OFFLOADS / LOG_FORMATS (test_cli.py asserts it).
DEVICES = ("auto", "cuda", "rocm", "mps", "cpu")
DTYPES = ("auto", "bfloat16", "float16", "float32")
QUANTS = ("none", "int8", "nf4")
OFFLOADS = ("none", "cpu")
LOG_FORMATS = ("text", "json")

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
    "log_format": "CLEF_LOG_FORMAT",
    "drain_s": "CLEF_DRAIN_S",
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


def selected_spec(model: str | None = None) -> Any:
    """The ModelSpec `clef serve` would load (CLEF_MODEL + aliases), or `model` (`id[@rev]`) instead of
    CLEF_MODEL. Its revision is the pin even when CLEF_MODEL_PATH is set. Raises ValueError / UnknownModel."""
    import dataclasses

    from clef_server.models.registry import resolve_spec

    cfg = load_config()
    changes: dict[str, Any] = {"model_path": None}
    if model:
        changes["model"] = model
    return resolve_spec(dataclasses.replace(cfg, **changes))


def cmd_download(args: argparse.Namespace) -> int:
    from clef_server.models import UnknownModel

    try:
        spec = selected_spec(args.model)
    except (ValueError, UnknownModel) as exc:
        print(f"clef: {exc}", file=sys.stderr)
        return 2
    repo = spec.repo
    revision = args.revision or spec.revision or "main"
    target = args.dir or os.environ.get("CLEF_MODEL_PATH") or None
    dest = Path(target).expanduser() if target else hf_cache_dir()
    free = free_gb(dest)
    need = max(MIN_FREE_GB, spec.weights_gb + 2.0)  # headroom for the partial download
    print(f"model:       {spec.id} = {repo}@{revision[:12]}")
    print(f"destination: {dest}{'' if target else ' (Hugging Face cache)'}")
    print(f"size:        about {spec.weights_gb:.0f} GB; free space: {free:.1f} GB")
    if free < need:
        print(f"clef: need at least {need:.0f} GB free on that disk; aborting", file=sys.stderr)
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
    serve = "clef serve" if spec.id == "clef-flash" else f"CLEF_MODEL={spec.id} clef serve"
    if target:
        print(f"use it with: CLEF_MODEL_PATH={path} {serve}")
    elif spec.id != "clef-flash":
        print(f"use it with: {serve}")
    return 0


def cmd_models(args: argparse.Namespace) -> int:
    """The model catalog (every installed adapter's ids), marking the one `clef serve` would load."""
    from clef_server.models import UnknownModel, registry

    try:
        selected = selected_spec()
        error = None
    except (ValueError, UnknownModel) as exc:
        selected, error = None, str(exc)
    rows = []
    for spec in registry.catalog().values():
        rows.append(
            {
                "id": spec.id,
                "adapter": spec.adapter,
                "repo": spec.repo,
                "revision": spec.revision,
                "weights_gb": spec.weights_gb,
                "selected": selected is not None and spec.id == selected.id,
            }
        )
    adapters = [
        {"name": name, "class": f"{cls.__module__}.{cls.__qualname__}", "from": registry.adapter_origin(name)}
        for name, cls in sorted(registry.adapters().items())
    ]
    if args.json:
        print(json.dumps({"models": rows, "adapters": adapters, "error": error}, indent=2))
        return 0 if error is None else 1
    for row in rows:
        mark = "*" if row["selected"] else " "
        rev = (row["revision"] or "local")[:12]
        where = f"{row['repo'] or '-'}@{rev}"
        print(f"{mark} {row['id']:<16} {where:<36} ~{row['weights_gb']:.0f} GB  ({row['adapter']})")
    print("adapters: " + ", ".join(f"{a['name']} ({a['from']})" for a in adapters))
    if error:
        print(f"clef: {error}", file=sys.stderr)
        return 1
    print("select one with CLEF_MODEL=<id>[@<revision>]; fetch it with `clef download --model <id>`")
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


# ---- C1: database commands (stdlib at import time; clef_server.db is imported inside each command)

NEEDS_SERVER_EXTRA = "needs the server extra: pip install 'clef-local[server]'"


def _db_module() -> Any:
    try:
        from clef_server import db
    except ImportError:
        return None
    return db


def cmd_db(args: argparse.Namespace) -> int:
    db = _db_module()
    if db is None:
        print(f"clef db: {NEEDS_SERVER_EXTRA}", file=sys.stderr)
        return 2
    cfg = load_config()
    try:
        if args.db_command == "current":
            url = db.database_url(cfg)
            version = db.peek_version(url)
            state = db.version_state(version)
            shown = {None: "(no database yet)", "": "(empty)", "legacy": "legacy v3.1 jobs schema"}
            print(f"database: {db.redact_url(url)}")
            print(f"schema:   {shown.get(version, version)} (this clef expects {db.code_version()}: {state})")
            return 0 if state == "current" else 1
        if args.db_command == "upgrade":
            if not cfg.database_url:
                db.migrate_legacy_file(cfg)
            database = db.Database(db.database_url(cfg))
            try:
                before = database.schema_version()
                database.upgrade()
                print(f"{database.display_url}: schema {before or 'none'} -> {database.schema_version()}")
            finally:
                database.close()
            return 0
        if args.db_command == "copy":
            return _db_copy(db, cfg, args)
    except db.DatabaseError as exc:
        print(f"clef db: {exc}", file=sys.stderr)
        return 1
    return 2


def _db_copy(db: Any, cfg: Any, args: argparse.Namespace) -> int:
    if livez(base_url(cfg)):
        print(
            f"clef db copy: a clef server answers on {base_url(cfg)}; stop it first (clef stop)",
            file=sys.stderr,
        )
        return 1
    src_url, dst_url = db.normalise_url(args.source), db.normalise_url(args.target)
    if src_url == dst_url:
        print("clef db copy: --from and --to are the same database", file=sys.stderr)
        return 1
    print(f"from: {db.redact_url(src_url)}\nto:   {db.redact_url(dst_url)}")
    if not args.yes:
        if not sys.stdin.isatty():
            print("clef db copy: pass --yes to copy non-interactively", file=sys.stderr)
            return 1
        if input("Copy every table? [y/N] ").strip().lower() not in ("y", "yes"):
            print("aborted")
            return 1
    if db.peek_version(src_url) is None:
        print(f"clef db copy: {db.redact_url(src_url)} does not exist", file=sys.stderr)
        return 1
    source, target = db.Database(src_url), db.Database(dst_url)
    try:
        copied = db.copy_database(source, target)
    finally:
        source.close()
        target.close()
    for name, n in copied.items():
        if n:
            print(f"  {name}: {n} row(s)")
    print(f"copied {sum(copied.values())} row(s); point CLEF_DATABASE_URL at the target to use it")
    return 0


def cmd_backup(args: argparse.Namespace) -> int:
    db = _db_module()
    if db is None:
        print(f"clef backup: {NEEDS_SERVER_EXTRA}", file=sys.stderr)
        return 2
    cfg = load_config()
    url = db.database_url(cfg)
    if not url.startswith("sqlite"):
        print(
            "clef backup: the database is Postgres; back it up with pg_dump, e.g.\n"
            f"  pg_dump --format=custom --file clef.dump '{db.redact_url(url)}'  (fill in the password)",
            file=sys.stderr,
        )
        return 2
    from sqlalchemy.engine import make_url

    from clef_server.paths import database_path, state_dir

    path = database_path(cfg) if not cfg.database_url else Path(make_url(url).database or "")
    if not path.is_file():
        print(f"clef backup: no database at {path}", file=sys.stderr)
        return 1
    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    out = Path(args.out).expanduser() if args.out else state_dir(cfg) / "backups" / f"clef-{stamp}.db"
    if out.exists():
        print(f"clef backup: {out} already exists", file=sys.stderr)
        return 1
    db.backup_sqlite(path, out)
    print(f"backup written to {out} ({out.stat().st_size / 1e6:.1f} MB)")
    return 0


def cmd_classifiers(args: argparse.Namespace) -> int:
    db = _db_module()
    if db is None:
        print(f"clef classifiers: {NEEDS_SERVER_EXTRA}", file=sys.stderr)
        return 2
    from clef_server.classifiers import ClassifierStore, InvalidName, export_file

    cfg = load_config()
    url = db.database_url(cfg)
    try:
        if db.peek_version(url) is None:
            print(
                f"clef classifiers: no database at {db.redact_url(url)} (start the server once)",
                file=sys.stderr,
            )
            return 1
        database = db.Database(url)
        try:
            store = ClassifierStore(database)
            docs = [store.get(args.name)] if args.name else store.list()
        finally:
            database.close()
    except InvalidName as exc:
        print(f"clef classifiers: {exc}", file=sys.stderr)
        return 1
    except db.DatabaseError as exc:
        print(f"clef classifiers: {exc}", file=sys.stderr)
        return 1
    if args.name and docs[0] is None:
        print(f"clef classifiers: {args.name!r} not found", file=sys.stderr)
        return 1
    out = Path(args.out).expanduser()
    out.mkdir(parents=True, exist_ok=True)
    for doc in docs:
        assert doc is not None
        target = out / f"{doc['name']}.json"
        target.write_text(json.dumps(export_file(doc), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"{target} (version {doc['version']})")
    print(f"exported {len(docs)} classifier(s) to {out}")
    return 0


# ---- D1: keys (direct database access; works with the server stopped: this is the bootstrap)


def _parse_duration(text: str) -> float:
    """``90d``, ``24h``, ``30m``, ``45s`` or plain seconds -> seconds."""
    units = {"d": 86400.0, "h": 3600.0, "m": 60.0, "s": 1.0}
    raw = text.strip().lower()
    try:
        return float(raw[:-1]) * units[raw[-1]] if raw and raw[-1] in units else float(raw)
    except ValueError:
        raise ValueError(f"{text!r} is not a duration (use 90d, 24h, 30m or seconds)") from None


def _parse_expiry(text: str) -> float:
    """``--expires``: a duration from now (``90d``) or a date (``2027-01-31``, valid through that UTC day)."""
    from datetime import datetime, timedelta, timezone

    if len(text) == 10 and text[4:5] == "-":
        try:
            day = datetime.strptime(text, "%Y-%m-%d").replace(tzinfo=timezone.utc)
        except ValueError:
            raise ValueError(f"{text!r} is not a date (YYYY-MM-DD)") from None
        return (day + timedelta(days=1)).timestamp()
    return time.time() + _parse_duration(text)


def _opt_int(text: str) -> int | None:
    """``--quota N`` / ``none``."""
    if text.strip().lower() in ("none", "null", "off", "unlimited"):
        return None
    try:
        return int(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"{text!r} is not a number (or 'none')") from None


def cmd_keys(args: argparse.Namespace) -> int:
    db = _db_module()
    if db is None:
        print(f"clef keys: {NEEDS_SERVER_EXTRA}", file=sys.stderr)
        return 2
    from clef_server.keystore import KeyStore, KeystoreError, Quotas, cli_actor

    cfg = load_config()
    try:
        if not cfg.database_url:
            db.migrate_legacy_file(cfg)
        database = db.Database(db.database_url(cfg))
        try:
            database.upgrade()  # a fresh state dir works: `clef keys create` is the bootstrap
            return _keys_command(args, KeyStore(database), Quotas(database), cfg, cli_actor())
        finally:
            database.close()
    except (KeystoreError, ValueError, db.DatabaseError) as exc:
        print(f"clef keys: {exc}", file=sys.stderr)
        return 1


def _keys_command(args: argparse.Namespace, store: Any, quotas: Any, cfg: Any, actor: Any) -> int:
    cmd = args.keys_command
    env_names = frozenset(cfg.api_keys())
    as_json = getattr(args, "json", False)
    if cmd in ("revoke", "update", "rotate") and args.name in env_names:
        raise ValueError(f"{args.name!r} is an env key (CLEF_API_KEY / CLEF_API_KEYS): change it there")
    if cmd in ("create", "rotate"):
        expires = _parse_expiry(args.expires) if args.expires else None
        first = False
        if cmd == "create":
            first = not cfg.auth_required and not store.list_principals(include_disabled=True)
            row, secret = store.create_principal(
                args.name,
                args.scope,
                actor=actor,
                daily_quota=args.quota,
                rate_limit=args.rate_limit,
                note=args.note,
                expires_at=expires,
                env_names=env_names,
            )
        else:
            grace = _parse_duration(args.grace)
            row, secret = store.rotate(args.name, actor=actor, grace_s=grace, expires_at=expires)
        if as_json:
            print(json.dumps({**row, "key": secret}, indent=2))
        else:
            verb = "created" if cmd == "create" else f"rotated (the old keys work for {args.grace})"
            print(f"principal {row['name']!r} {verb}; scopes: {','.join(row['scopes'])}")
            print(f"key: {secret}")
            print("Store it now: only its hash is kept, so it cannot be shown again.")
        if first:
            print("auth is now required for every /v1 route", file=sys.stderr if as_json else sys.stdout)
        return 0
    quotas.load()
    if cmd == "list":
        envs = [
            {"name": n, "source": "env", "scopes": ["admin"], "daily_quota": None, "used_today": 0}
            | {"keys": [], "disabled_at": None}
            for n in sorted(env_names)
        ]
        rows = envs + [{**r, "used_today": quotas.used(r["name"])} for r in store.list_principals(args.all)]
        if as_json:
            print(json.dumps({"principals": rows}, indent=2))
        elif not rows:
            print("no principals (auth is off); create one: clef keys create NAME --scope admin")
        else:
            print(f"{'NAME':<20} {'SRC':<4} {'SCOPES':<34} {'QUOTA':>9} {'USED':>8}  KEYS")
            for r in rows:
                keys = ", ".join(f"{k['id']} {k['status']}" for k in r["keys"]) or "-"
                name = r["name"] + (" (disabled)" if r["disabled_at"] else "")
                scopes = "all (env key)" if r["source"] == "env" else ",".join(r["scopes"])
                quota = "-" if r["daily_quota"] is None else r["daily_quota"]
                print(f"{name:<20} {r['source']:<4} {scopes:<34} {quota!s:>9} {r['used_today']:>8}  {keys}")
        return 0
    if cmd == "revoke":
        store.revoke(args.name, actor=actor, key_id=args.key_id)
        what = f"key {args.key_id} revoked" if args.key_id else f"principal {args.name!r} disabled"
        print(what)
        return 0
    if cmd == "update":
        given = vars(args)  # the update flags default to SUPPRESS: "--quota none" (None) differs from absent
        pairs = (
            ("scope", "scopes"),
            ("quota", "daily_quota"),
            ("rate_limit", "rate_limit"),
            ("note", "note"),
        )
        changes = {field: given[flag] for flag, field in pairs if flag in given}
        if not changes:
            raise ValueError("nothing to change (give --scope, --quota, --rate-limit or --note)")
        row = store.update_principal(args.name, changes, actor=actor)
        print(f"updated {row['name']!r}: scopes {','.join(row['scopes'])}, quota {row['daily_quota']}")
        return 0
    if cmd == "usage":
        if store.get(args.name) is None:
            raise ValueError(f"principal {args.name!r} not found")
        days = quotas.history(args.name, args.days)
        if as_json:
            print(json.dumps({"name": args.name, "days": days}, indent=2))
        else:
            print(f"{'DAY':<12} {'RECORDS':>10} {'REQUESTS':>10} {'REJECTED':>10}")
            for d in days:
                print(f"{d['day']:<12} {d['records']:>10} {d['requests']:>10} {d['rejected']:>10}")
        return 0
    return 2


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
    sp.add_argument(
        "--log-format",
        dest="log_format",
        choices=LOG_FORMATS,
        help="CLEF_LOG_FORMAT: json = one object per line",
    )
    sp.add_argument(
        "--drain-s",
        dest="drain_s",
        type=float,
        metavar="S",
        help="CLEF_DRAIN_S: keep serving S seconds after SIGTERM with /readyz 503 (default 0)",
    )
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

    sp = sub.add_parser("download", help="download the pinned model weights (clef-flash ~19 GB, resumable)")
    sp.add_argument(
        "--model",
        metavar="ID[@REV]",
        help="catalog id (`clef models`); default: the selected model (CLEF_MODEL)",
    )
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

    # C1: database (direct access via CLEF_DATABASE_URL or the default file; works with the server stopped)
    sp = sub.add_parser("db", help="database: current, upgrade, copy (see docs/jobs.md)")
    dbsub = sp.add_subparsers(dest="db_command", metavar="<db command>", required=True)
    dbsub.add_parser("current", help="schema version and (redacted) database URL")
    dbsub.add_parser("upgrade", help="migrate to this version's schema (the server also does it at startup)")
    cp = dbsub.add_parser("copy", help="copy every table into an empty database (e.g. SQLite -> Postgres)")
    cp.add_argument("--from", dest="source", required=True, metavar="URL", help="source database URL")
    cp.add_argument("--to", dest="target", required=True, metavar="URL", help="target database URL (empty)")
    cp.add_argument("--yes", "-y", action="store_true", help="do not ask for confirmation")
    sp.set_defaults(func=cmd_db)

    sp = sub.add_parser("backup", help="consistent copy of the SQLite database (safe while the server runs)")
    sp.add_argument("--out", metavar="PATH", help="default <state_dir>/backups/clef-<UTC time>.db")
    sp.set_defaults(func=cmd_backup)

    sp = sub.add_parser("classifiers", help="saved classifiers: export")
    csub = sp.add_subparsers(dest="classifiers_command", metavar="<classifiers command>", required=True)
    ex = csub.add_parser("export", help="latest version of each as <DIR>/<name>.json")
    ex.add_argument("--out", default="classifiers", metavar="DIR", help="default ./classifiers")
    ex.add_argument("--name", metavar="NAME", help="only this classifier")
    sp.set_defaults(func=cmd_classifiers)
    # C2
    sp = sub.add_parser("models", help="list the model catalog and installed adapters (* = selected)")
    sp.add_argument("--json", action="store_true")
    sp.set_defaults(func=cmd_models)

    # D1: keys (direct database access; the first `create` is the bootstrap)
    sp = sub.add_parser("keys", help="API keys and principals: create, list, rotate, revoke, update, usage")
    ksub = sp.add_subparsers(dest="keys_command", metavar="<keys command>", required=True)
    kp = ksub.add_parser("create", help="a principal and its first key (printed once)")
    kp.add_argument("name")
    scopes_help = "comma-separated: classify,jobs,classifiers,ops,admin,urgent"
    kp.add_argument("--scope", required=True, metavar="S", help=scopes_help)
    kp.add_argument("--quota", type=int, metavar="N", help="records per UTC day (default unlimited)")
    kp.add_argument("--rate-limit", type=int, metavar="N", help="requests/min (default CLEF_RATE_LIMIT)")
    kp.add_argument("--expires", metavar="90d|2027-01-31", help="expiry of the key (default never)")
    kp.add_argument("--note", metavar="TEXT")
    kp.add_argument("--json", action="store_true")
    kp = ksub.add_parser("list", help="principals and key statuses")
    kp.add_argument("--all", action="store_true", help="include disabled principals")
    kp.add_argument("--json", action="store_true")
    kp = ksub.add_parser("rotate", help="a new key; the old ones keep working for --grace")
    kp.add_argument("name")
    kp.add_argument("--grace", default="24h", metavar="24h", help="how long the old keys keep working")
    kp.add_argument("--expires", metavar="90d|2027-01-31", help="expiry of the new key")
    kp.add_argument("--json", action="store_true")
    kp = ksub.add_parser("revoke", help="one key (--key-id) or the whole principal")
    kp.add_argument("name")
    kp.add_argument("--key-id", metavar="ID")
    kp = ksub.add_parser("update", help="scopes, quota, rate limit or note")
    kp.add_argument("name")
    kp.add_argument("--scope", metavar="S", default=argparse.SUPPRESS)
    kp.add_argument("--quota", type=_opt_int, metavar="N|none", default=argparse.SUPPRESS)
    kp.add_argument("--rate-limit", type=_opt_int, metavar="N|none", default=argparse.SUPPRESS)
    kp.add_argument("--note", metavar="TEXT", default=argparse.SUPPRESS)
    kp = ksub.add_parser("usage", help="daily records / requests / quota rejections")
    kp.add_argument("name")
    kp.add_argument("--days", type=int, default=30, metavar="N")
    kp.add_argument("--json", action="store_true")
    sp.set_defaults(func=cmd_keys)
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
