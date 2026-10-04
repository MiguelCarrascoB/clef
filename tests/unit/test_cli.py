"""CLI: argument parsing, flag -> env mapping, pidfile handling, detach command, download disk check."""

from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from clef_server import cli, config


def parse(*argv: str):
    return cli.build_parser().parse_args(list(argv))


def test_choices_match_config():
    assert cli.DEVICES == config.DEVICES
    assert cli.DTYPES == config.DTYPES
    assert cli.QUANTS == config.QUANTS
    assert cli.OFFLOADS == config.OFFLOADS


def test_serve_memory_flags_map_to_env():
    a = parse("serve", "--offload", "cpu", "--max-device-memory-gb", "14.5", "--quant", "int8")
    env = cli.serve_env(a, base={})
    assert env["CLEF_OFFLOAD"] == "cpu"
    assert env["CLEF_MAX_DEVICE_MEMORY_GB"] == "14.5"
    assert env["CLEF_QUANT"] == "int8"
    assert "CLEF_OFFLOAD" not in cli.serve_env(parse("serve"), base={})
    with pytest.raises(SystemExit):
        parse("serve", "--offload", "disk")


def test_serve_defaults_and_flags():
    a = parse("serve")
    assert a.detach is False and a.timeout == 600 and a.host is None and a.port is None
    a = parse(
        "serve", "--host", "0.0.0.0", "--port", "9000", "--device", "cpu", "--dtype", "float32",
        "--quant", "none", "--model-path", "/m", "--detach", "--timeout", "30",
    )  # fmt: skip
    assert (a.host, a.port, a.device, a.dtype, a.quant, a.model_path) == (
        "0.0.0.0", 9000, "cpu", "float32", "none", "/m",
    )  # fmt: skip
    assert a.detach and a.timeout == 30


def test_serve_rejects_bad_choice():
    with pytest.raises(SystemExit):
        parse("serve", "--device", "tpu")


def test_serve_env_maps_flags_and_keeps_rest():
    a = parse("serve", "--port", "9001", "--device", "cuda", "--model-path", "/w")
    env = cli.serve_env(a, base={"CLEF_HOST": "10.0.0.5", "CLEF_DTYPE": "bfloat16"})
    assert env["CLEF_PORT"] == "9001"
    assert env["CLEF_DEVICE"] == "cuda"
    assert env["CLEF_MODEL_PATH"] == "/w"
    assert env["CLEF_HOST"] == "10.0.0.5"  # untouched when the flag is absent
    assert env["CLEF_DTYPE"] == "bfloat16"
    assert "CLEF_QUANT" not in env


def test_other_commands_parse():
    assert parse("logs", "-f", "-n", "5").follow and parse("logs", "-n", "5").lines == 5
    assert parse("status", "--json").json
    d = parse("doctor", "--no-gpu", "--smoke", "--json")
    assert d.no_gpu and d.smoke and d.json
    dl = parse("download", "--revision", "abc", "--dir", "/x", "--yes")
    assert (dl.revision, dl.dir, dl.yes) == ("abc", "/x", True)
    assert parse("bench").rest == []


def test_main_without_command_prints_help(capsys):
    assert cli.main([]) == 0
    assert "serve" in capsys.readouterr().out


def test_version_command(capsys):
    assert cli.main(["version"]) == 0
    assert capsys.readouterr().out.strip() == f"clef-local {config.VERSION}"


def test_version_in_sync_with_pyproject():
    text = (Path(__file__).resolve().parents[2] / "pyproject.toml").read_text(encoding="utf-8")
    static = re.search(r'^version\s*=\s*"([^"]+)"', text, re.M)
    if static:
        assert static.group(1) == config.VERSION
    else:  # dynamic version read from config.py by the build backend
        assert 'dynamic = ["version"]' in text
        assert re.search(r'^VERSION = "[\d.]+"', Path(config.__file__).read_text(encoding="utf-8"), re.M)


def test_cli_import_is_light():
    code = (
        "import sys; import clef_server.cli; "
        "bad=[m for m in ('torch','fastapi','httpx','transformers') if m in sys.modules]; print(bad)"
    )
    out = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        check=True,
        env={**os.environ, "PYTHONPATH": str(Path(config.__file__).parents[1])},
    )
    assert out.stdout.strip() == "[]"


# ---- pidfile


def test_read_pid_and_stale_removal(tmp_path):
    pf = tmp_path / "server.pid"
    assert cli.read_pid(pf) is None
    pf.write_text("garbage")
    assert cli.read_pid(pf) is None
    pf.write_text(f"{os.getpid()}\n")
    assert cli.read_pid(pf) == os.getpid()
    assert cli.running_pid(pf) == os.getpid()
    pf.write_text("999999999\n")
    assert cli.running_pid(pf) is None
    assert not pf.exists()  # stale pidfile cleaned


def test_pid_alive_self_and_bogus():
    assert cli.pid_alive(os.getpid())
    assert not cli.pid_alive(0)
    assert not cli.pid_alive(999999999)


def test_stop_without_pidfile(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("CLEF_STATE_DIR", str(tmp_path))
    monkeypatch.setattr(cli, "livez", lambda url: False)
    assert cli.main(["stop"]) == 0
    assert "not running" in capsys.readouterr().out


def test_stop_signals_pid_and_removes_pidfile(tmp_path, monkeypatch):
    monkeypatch.setenv("CLEF_STATE_DIR", str(tmp_path))
    (tmp_path / "server.pid").write_text("4242\n")
    state = {"alive": True}
    monkeypatch.setattr(cli, "pid_alive", lambda pid: state["alive"])
    calls = []

    def fake_terminate(pid, force=False):
        calls.append((pid, force))
        state["alive"] = False

    monkeypatch.setattr(cli, "terminate", fake_terminate)
    monkeypatch.setattr(cli.time, "sleep", lambda s: None)
    assert cli.main(["stop"]) == 0
    assert calls == [(4242, False)]
    assert not (tmp_path / "server.pid").exists()


def test_rotate_logs(tmp_path):
    log = tmp_path / "server.log"
    for i, name in enumerate(["server.log", "server.log.1", "server.log.2", "server.log.3"]):
        (tmp_path / name).write_text(f"v{i}")
    cli.rotate_logs(log)
    assert not log.exists()
    assert (tmp_path / "server.log.1").read_text() == "v0"
    assert (tmp_path / "server.log.2").read_text() == "v1"
    assert (tmp_path / "server.log.3").read_text() == "v2"  # oldest dropped
    cli.rotate_logs(log)  # nothing to rotate: no error


def test_tail_lines(tmp_path):
    f = tmp_path / "x.log"
    f.write_text("\n".join(f"l{i}" for i in range(10)) + "\n")
    assert cli.tail_lines(f, 3) == ["l7", "l8", "l9"]
    assert cli.tail_lines(tmp_path / "missing", 3) == []


# ---- detach


def test_detach_command_and_popen_kwargs(monkeypatch):
    assert cli.detach_command() == [sys.executable, "-m", "clef_server.main"]
    monkeypatch.setattr(cli.os, "name", "nt")
    kw = cli.popen_kwargs()
    assert kw["creationflags"] == cli.DETACHED_PROCESS | cli.CREATE_NEW_PROCESS_GROUP
    assert "start_new_session" not in kw
    monkeypatch.setattr(cli.os, "name", "posix")
    assert cli.popen_kwargs()["start_new_session"] is True


class FakeProc:
    pid = 31337
    returncode = None

    def poll(self):
        return None


def test_serve_detach_spawns_and_waits_for_health(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("CLEF_STATE_DIR", str(tmp_path))
    captured = {}

    def fake_popen(cmd, **kw):
        captured.update(cmd=cmd, kw=kw)
        return FakeProc()

    monkeypatch.setattr(cli.subprocess, "Popen", fake_popen)
    states = iter(["", "warming"])
    lives = iter([False, False, True])  # pre-check, first poll, then up
    monkeypatch.setattr(cli, "livez", lambda url: next(lives))
    monkeypatch.setattr(cli, "health_status", lambda url: next(states))
    monkeypatch.setattr(cli.time, "sleep", lambda s: None)
    rc = cli.main(["serve", "--detach", "--port", "9123", "--device", "cpu", "--timeout", "5"])
    assert rc == 0
    assert captured["cmd"] == cli.detach_command()
    assert captured["kw"]["env"]["CLEF_PORT"] == "9123"
    assert captured["kw"]["env"]["CLEF_DEVICE"] == "cpu"
    assert (tmp_path / "server.pid").read_text().strip() == "31337"
    assert "SERVER UP" in capsys.readouterr().out


def test_serve_detach_refuses_when_port_answers(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("CLEF_STATE_DIR", str(tmp_path))
    monkeypatch.setattr(cli, "livez", lambda url: True)
    monkeypatch.setattr(cli.subprocess, "Popen", lambda *a, **k: pytest.fail("must not spawn"))
    assert cli.main(["serve", "--detach"]) == 1
    assert "already answers" in capsys.readouterr().err


def test_serve_detach_reports_child_exit(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("CLEF_STATE_DIR", str(tmp_path))

    class Dead(FakeProc):
        returncode = 3

        def poll(self):
            return 3

    monkeypatch.setattr(cli.subprocess, "Popen", lambda *a, **k: Dead())
    monkeypatch.setattr(cli, "livez", lambda url: False)
    monkeypatch.setattr(cli.time, "sleep", lambda s: None)
    assert cli.main(["serve", "--detach", "--timeout", "2"]) == 1
    assert "exited during startup" in capsys.readouterr().err
    assert not (tmp_path / "server.pid").exists()


# ---- download


def test_download_aborts_on_low_disk(monkeypatch, capsys, tmp_path):
    def boom(*a, **k):
        pytest.fail("must not download")

    monkeypatch.setattr(cli, "free_gb", lambda p: 5.0)
    monkeypatch.setitem(sys.modules, "huggingface_hub", SimpleNamespace(snapshot_download=boom))
    assert cli.main(["download", "--yes", "--dir", str(tmp_path / "w")]) == 1
    assert "at least" in capsys.readouterr().err


def test_download_calls_snapshot_with_pinned_revision(monkeypatch, tmp_path):
    seen = {}

    def fake(repo, **kw):
        seen.clear()
        seen.update(repo=repo, **kw)
        return str(tmp_path)

    monkeypatch.setattr(cli, "free_gb", lambda p: 500.0)
    monkeypatch.delenv("CLEF_MODEL_REVISION", raising=False)
    monkeypatch.delenv("CLEF_MODEL_PATH", raising=False)
    monkeypatch.setitem(sys.modules, "huggingface_hub", SimpleNamespace(snapshot_download=fake))
    assert cli.main(["download", "--yes"]) == 0
    assert seen["repo"] == config.MODEL_REPO and seen["revision"] == config.MODEL_REVISION
    assert "local_dir" not in seen
    assert cli.main(["download", "--yes", "--revision", "abc", "--dir", str(tmp_path)]) == 0
    assert seen["revision"] == "abc" and seen["local_dir"] == str(tmp_path)


def test_download_needs_yes_without_tty(monkeypatch, capsys):
    monkeypatch.setattr(cli, "free_gb", lambda p: 500.0)
    monkeypatch.delenv("CLEF_MODEL_PATH", raising=False)
    monkeypatch.setattr(sys.stdin, "isatty", lambda: False)
    assert cli.main(["download"]) == 1
    assert "--yes" in capsys.readouterr().err


def test_free_gb_walks_to_existing_parent(tmp_path):
    assert cli.free_gb(tmp_path / "a" / "b" / "c") > 0


# ---- status / doctor / bench delegation


def test_status_unreachable_exit_1(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("CLEF_STATE_DIR", str(tmp_path))
    monkeypatch.setattr(cli, "http_json", lambda url, timeout=3.0: None)
    assert cli.main(["status"]) == 1
    assert "unreachable" in capsys.readouterr().out


def test_status_ready_exit_0(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("CLEF_STATE_DIR", str(tmp_path))
    ok = (200, {"status": "ready", "version": "3.1.0"})
    monkeypatch.setattr(cli, "http_json", lambda url, timeout=3.0: ok)
    assert cli.main(["status", "--json"]) == 0
    assert '"ready"' in capsys.readouterr().out


def _fake_module(monkeypatch, name, got):
    def main(argv):
        got["argv"] = argv
        return 0

    import importlib

    pkg = importlib.import_module("clef_server")
    fake = SimpleNamespace(main=main)
    monkeypatch.setitem(sys.modules, f"clef_server.{name}", fake)
    monkeypatch.setattr(pkg, name, fake, raising=False)


def test_doctor_delegates(monkeypatch):
    got = {}
    _fake_module(monkeypatch, "doctor", got)
    assert cli.main(["doctor", "--no-gpu", "--json"]) == 0
    assert got["argv"] == ["--no-gpu", "--json"]


def test_bench_delegates(monkeypatch):
    got = {}
    _fake_module(monkeypatch, "httpbench", got)
    assert cli.main(["bench", "--requests", "5"]) == 0
    assert got["argv"] == ["--requests", "5"]


def test_client_host_and_base_url():
    assert cli.client_host("0.0.0.0") == "127.0.0.1"
    assert cli.base_url(SimpleNamespace(host="::1", port=1)) == "http://[::1]:1"


def test_serve_detach_stops_server_on_error_status(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("CLEF_STATE_DIR", str(tmp_path))
    killed = []
    monkeypatch.setattr(cli.subprocess, "Popen", lambda *a, **k: FakeProc())
    calls = iter([False, True])
    monkeypatch.setattr(cli, "livez", lambda url: next(calls))
    monkeypatch.setattr(cli, "health_status", lambda url: "error")
    monkeypatch.setattr(cli, "terminate", lambda pid, force=False: killed.append((pid, force)))
    monkeypatch.setattr(cli.time, "sleep", lambda s: None)
    assert cli.main(["serve", "--detach", "--timeout", "5"]) == 1
    assert killed == [(31337, True)]
    assert not (tmp_path / "server.pid").exists()
    assert "status=error" in capsys.readouterr().err
