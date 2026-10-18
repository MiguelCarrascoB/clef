"""`.env` loading: syntax, precedence (set variables win), CLEF_ENV_FILE, doctor output, no value in logs."""

from __future__ import annotations

import logging
import os
from pathlib import Path

import pytest

from clef_server import config, doctor, dotenv


@pytest.fixture(autouse=True)
def _fresh():
    dotenv.reset()  # conftest resets too; explicit here because these tests are about that state
    yield
    dotenv.reset()


def test_parse_syntax():
    text = (
        "# a comment\n\n"
        "A=1\n"
        "export B = two words  \n"
        "C='single # not a comment \\n'\n"
        'D="double # kept \\"q\\" \\n end"\n'
        "E=plain # trailing comment\n"
        "F=\n"
        "G='unterminated\n"
        "not a pair\n"
        "1BAD=x\n"
        "A=last wins\n"
        "H=a=b\n"
    )
    assert dotenv.parse(text) == {
        "A": "last wins",
        "B": "two words",
        "C": "single # not a comment \\n",
        "D": 'double # kept "q" \n end',
        "E": "plain",
        "F": "",
        "G": "unterminated",
        "H": "a=b",
    }


def test_set_variables_win_and_the_cwd_file_beats_the_config_file(tmp_path, monkeypatch):
    cwd, cfgdir = tmp_path / "work", tmp_path / "home" / ".config" / "clef"
    cwd.mkdir()
    cfgdir.mkdir(parents=True)
    (cwd / ".env").write_text("GEMINI_API_KEY=from-dotenv\nCLEF_PORT=1111\nCLEF_ONLY_CWD=1\n")
    (cfgdir / "env").write_text("GEMINI_API_KEY=from-config\nCLEF_ONLY_CONFIG=2\nCLEF_PORT=2222\n")
    monkeypatch.chdir(cwd)
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.delenv("CLEF_ENV_FILE")
    for name in ("GEMINI_API_KEY", "CLEF_ONLY_CWD", "CLEF_ONLY_CONFIG"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("CLEF_PORT", "9999")  # already set: never overwritten
    read = dotenv.load_env()
    assert len(read) == 2 and read[0].endswith(".env") and read[1].endswith("clef/env")
    assert os.environ["GEMINI_API_KEY"] == "from-dotenv"
    assert os.environ["CLEF_ONLY_CWD"] == "1" and os.environ["CLEF_ONLY_CONFIG"] == "2"
    assert os.environ["CLEF_PORT"] == "9999"
    for name in ("GEMINI_API_KEY", "CLEF_ONLY_CWD", "CLEF_ONLY_CONFIG"):
        monkeypatch.delenv(name)  # setdefault wrote them: undo
    assert dotenv.load_env() == read  # idempotent


def test_config_load_reads_the_key_from_dotenv(tmp_path, monkeypatch):
    env = tmp_path / "my.env"
    env.write_text("export GEMINI_API_KEY='k-123'\n")
    monkeypatch.setenv("CLEF_ENV_FILE", str(env))
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    try:
        assert config.load().llm_api_key == "k-123"
    finally:
        monkeypatch.delenv("GEMINI_API_KEY", raising=False)


def test_clef_env_file_empty_disables_and_a_path_replaces_the_defaults(tmp_path, monkeypatch):
    (tmp_path / ".env").write_text("X_DISABLED=1\n")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("CLEF_ENV_FILE", "")
    assert dotenv.env_files() == [] and dotenv.load_env() == []
    only = tmp_path / "only.env"
    monkeypatch.setenv("CLEF_ENV_FILE", str(only))
    dotenv.reset()  # CLEF_ENV_FILE is read once per process (first load): a new process sees the change
    assert dotenv.env_files() == [only]
    assert dotenv.load_env() == []  # missing file: nothing, no error


def test_env_files_never_log_values(tmp_path, monkeypatch, caplog):
    env = tmp_path / "secret.env"
    env.write_text("GEMINI_API_KEY=super-secret-value\n")
    target: dict[str, str] = {"CLEF_ENV_FILE": str(env)}
    with caplog.at_level(logging.DEBUG):
        dotenv.load_env(target)
        (tmp_path / "dir.env").mkdir()
        dotenv.load_env({"CLEF_ENV_FILE": str(tmp_path / "dir.env")})
    assert target["GEMINI_API_KEY"] == "super-secret-value"
    assert "super-secret-value" not in caplog.text


def test_doctor_names_the_files_it_loaded(tmp_path, monkeypatch):
    env = tmp_path / "doctor.env"
    env.write_text("CLEF_DOCTOR_TEST=secret-value\n")
    monkeypatch.setenv("CLEF_ENV_FILE", str(env))
    monkeypatch.delenv("CLEF_DOCTOR_TEST", raising=False)
    try:
        r = doctor.Report(echo=False)
        doctor.check_config(r)
    finally:
        monkeypatch.delenv("CLEF_DOCTOR_TEST", raising=False)
    line = next(x for x in r.items if x["name"] == "env files")
    assert "doctor.env" in line["detail"] and "secret-value" not in line["detail"]


def test_gitignore_lists_dotenv():
    root = Path(__file__).resolve().parents[2]
    assert ".env" in (root / ".gitignore").read_text().split()


# ------------------------------------------------------------------ review E/F #4


def _home(tmp_path, monkeypatch):
    home = tmp_path / "home"
    (home / ".config" / "clef").mkdir(parents=True)
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.delenv("XDG_CONFIG_HOME", raising=False)
    return home


def test_a_dotenv_cannot_redirect_loading_and_a_second_load_is_idempotent(tmp_path, monkeypatch):
    """(a) CLEF_ENV_FILE comes from the real environment only, read once: a .env cannot redirect loading."""
    _home(tmp_path, monkeypatch)
    other = tmp_path / "other.env"
    other.write_text("CLEF_FROM_OTHER=1\nGEMINI_API_KEY=sneaky\n")
    (tmp_path / ".env").write_text(f"CLEF_ENV_FILE={other}\nCLEF_FROM_DOTENV=1\n")
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("CLEF_ENV_FILE")
    for name in ("CLEF_FROM_OTHER", "CLEF_FROM_DOTENV", "GEMINI_API_KEY"):
        monkeypatch.delenv(name, raising=False)
    try:
        first = dotenv.load_env()
        assert os.environ.get("CLEF_ENV_FILE") is None and os.environ["CLEF_FROM_DOTENV"] == "1"
        monkeypatch.setenv("CLEF_ENV_FILE", str(other))  # changed later in the process: the snapshot holds
        assert dotenv.load_env() == first
        assert "CLEF_FROM_OTHER" not in os.environ and "GEMINI_API_KEY" not in os.environ
    finally:
        for name in ("CLEF_FROM_OTHER", "CLEF_FROM_DOTENV", "GEMINI_API_KEY"):
            monkeypatch.delenv(name, raising=False)


def test_only_clef_and_llm_names_are_loaded_and_the_rest_is_named_in_a_warning(tmp_path, caplog):
    """(b) an env file can set CLEF_* and GEMINI_API_KEY only; ignored names (never values) are logged."""
    env = tmp_path / "x.env"
    env.write_text("CLEF_PORT=1\nGEMINI_API_KEY=k\nPATH=/evil\nLD_PRELOAD=/evil.so\nCLEF_ENV_FILE=/x\n")
    target: dict[str, str] = {"CLEF_ENV_FILE": str(env), "PATH": "/usr/bin"}
    with caplog.at_level(logging.WARNING, logger="clef.dotenv"):
        dotenv.load_env(target)
    assert target == {"CLEF_ENV_FILE": str(env), "PATH": "/usr/bin", "CLEF_PORT": "1", "GEMINI_API_KEY": "k"}
    [rec] = [r for r in caplog.records if "ignored" in r.getMessage()]
    msg = rec.getMessage()
    assert "LD_PRELOAD" in msg and "PATH" in msg and "CLEF_ENV_FILE" in msg and "/evil" not in msg
    assert "CLEF_PORT" not in msg


def test_an_empty_variable_does_not_block_the_file_value_but_an_empty_clef_env_file_disables(tmp_path):
    """(c) set-but-empty counts as unset (like config._env); CLEF_ENV_FILE= (empty) still loads nothing."""
    env = tmp_path / "x.env"
    env.write_text("GEMINI_API_KEY=from-file\nCLEF_PORT=7\n")
    target = {"CLEF_ENV_FILE": str(env), "GEMINI_API_KEY": "", "CLEF_PORT": "  "}
    dotenv.load_env(target)
    assert target["GEMINI_API_KEY"] == "from-file" and target["CLEF_PORT"] == "7"
    off = {"CLEF_ENV_FILE": "", "GEMINI_API_KEY": ""}
    assert dotenv.load_env(off) == [] and off["GEMINI_API_KEY"] == ""


def test_server_start_logs_which_env_files_were_loaded(tmp_path, monkeypatch, caplog):
    """(d) one INFO line at startup with the file names (never the values), also when none was loaded."""
    from fastapi.testclient import TestClient
    from test_api import FakeEngine

    from clef_server.main import create_app

    env = tmp_path / "start.env"
    env.write_text("CLEF_START_TEST=secret-start-value\n")
    monkeypatch.setenv("CLEF_ENV_FILE", str(env))
    monkeypatch.delenv("CLEF_START_TEST", raising=False)
    try:
        dotenv.load_env()
        app = create_app(config.Config(state_dir=str(tmp_path / "s"), api_key=None), FakeEngine())
        with caplog.at_level(logging.INFO, logger="clef"), TestClient(app):
            pass
    finally:
        monkeypatch.delenv("CLEF_START_TEST", raising=False)
    lines = [r.getMessage() for r in caplog.records if r.getMessage().startswith("env files")]
    assert len(lines) == 1 and "start.env" in lines[0] and "secret-start-value" not in caplog.text
