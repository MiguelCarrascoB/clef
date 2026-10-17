"""`.env` loading: syntax, precedence (set variables win), CLEF_ENV_FILE, doctor output, no value in logs."""

from __future__ import annotations

import logging
import os
from pathlib import Path

import pytest

from clef_server import config, doctor, dotenv


@pytest.fixture(autouse=True)
def _fresh(monkeypatch):
    monkeypatch.setattr(dotenv, "loaded", [])


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
    (cwd / ".env").write_text("GEMINI_API_KEY=from-dotenv\nCLEF_PORT=1111\nONLY_CWD=1\n")
    (cfgdir / "env").write_text("GEMINI_API_KEY=from-config\nONLY_CONFIG=2\nCLEF_PORT=2222\n")
    monkeypatch.chdir(cwd)
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.delenv("CLEF_ENV_FILE")
    for name in ("GEMINI_API_KEY", "ONLY_CWD", "ONLY_CONFIG"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("CLEF_PORT", "9999")  # already set: never overwritten
    read = dotenv.load_env()
    assert len(read) == 2 and read[0].endswith(".env") and read[1].endswith("clef/env")
    assert os.environ["GEMINI_API_KEY"] == "from-dotenv"
    assert os.environ["ONLY_CWD"] == "1" and os.environ["ONLY_CONFIG"] == "2"
    assert os.environ["CLEF_PORT"] == "9999"
    for name in ("GEMINI_API_KEY", "ONLY_CWD", "ONLY_CONFIG"):
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
