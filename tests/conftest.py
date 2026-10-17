"""Shared pytest setup. The package is imported from src/ (pyproject sets pythonpath).

Postgres: tests marked ``postgres`` run once on SQLite and, when ``CLEF_TEST_DATABASE_URL`` points at a
Postgres server (an admin URL, e.g. ``postgresql://postgres:test@localhost:55432/postgres``), once more on a
fresh database created for that test (``CLEF_DATABASE_URL`` points at it; dropped afterwards). The
``db_backend`` fixture says which ("sqlite" | "postgres").
"""

from __future__ import annotations

import os
import secrets
from collections.abc import Iterator

import pytest

PG_ADMIN_URL = os.environ.get("CLEF_TEST_DATABASE_URL", "").strip()


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers", "postgres: also runs against a fresh Postgres database when CLEF_TEST_DATABASE_URL is set"
    )


def pytest_generate_tests(metafunc: pytest.Metafunc) -> None:
    if "db_backend" in metafunc.fixturenames and metafunc.definition.get_closest_marker("postgres"):
        backends = ["sqlite", "postgres"] if PG_ADMIN_URL else ["sqlite"]
        metafunc.parametrize("db_backend", backends, indirect=True)


def _pg_admin():  # psycopg is only needed when the Postgres runs are enabled
    import psycopg
    from sqlalchemy.engine import make_url

    url = make_url(PG_ADMIN_URL.replace("postgresql+psycopg://", "postgresql://"))
    conninfo = url.set(drivername="postgresql").render_as_string(hide_password=False)
    return psycopg.connect(conninfo, autocommit=True), url


@pytest.fixture(autouse=True)
def _private_state_dir(tmp_path_factory: pytest.TempPathFactory, monkeypatch: pytest.MonkeyPatch) -> None:
    """A test that builds ``Config()`` without ``state_dir`` must never open (or migrate) the developer's real
    state dir: its clef.db, its v3.1 jobs.db or its classifier files."""
    monkeypatch.setenv("CLEF_STATE_DIR", str(tmp_path_factory.mktemp("state")))
    monkeypatch.delenv("CLEF_LOG_DIR", raising=False)


@pytest.fixture(autouse=True)
def _no_env_files(monkeypatch: pytest.MonkeyPatch) -> None:
    """Never read a real ``./.env`` or ``~/.config/clef/env`` (CLEF_ENV_FILE= disables loading)."""
    monkeypatch.setenv("CLEF_ENV_FILE", "")


@pytest.fixture(autouse=True)
def _no_llm_key(monkeypatch: pytest.MonkeyPatch) -> None:
    """The router's LLM is off unless a test turns it on with a fake: never a live client in tests."""
    for name in ("GEMINI_API_KEY", "CLEF_LLM_API_KEY", "CLEF_LLM_BASE_URL", "CLEF_LLM_MODEL"):
        monkeypatch.delenv(name, raising=False)


@pytest.fixture(autouse=True)
def db_backend(request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch) -> Iterator[str]:
    backend = getattr(request, "param", "sqlite")
    if backend == "sqlite":
        monkeypatch.delenv("CLEF_DATABASE_URL", raising=False)
        yield backend
        return
    name = "clef_t_" + secrets.token_hex(6)
    admin, url = _pg_admin()
    try:
        admin.execute(f'CREATE DATABASE "{name}"')
        monkeypatch.setenv("CLEF_DATABASE_URL", url.set(database=name).render_as_string(hide_password=False))
        yield backend
    finally:
        admin.execute(f'DROP DATABASE IF EXISTS "{name}" WITH (FORCE)')
        admin.close()


@pytest.fixture
def sqlite_only(db_backend: str) -> None:
    """For tests that exercise SQLite specifics (triggers, legacy jobs.db files)."""
    if db_backend != "sqlite":
        pytest.skip("SQLite-specific test")
