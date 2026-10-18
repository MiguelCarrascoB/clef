"""Optional ``.env`` loading: ``KEY=VALUE`` files read into ``os.environ`` before ``Config`` is built.

Files, in order: ``./.env`` (the directory ``clef serve`` runs in), then ``<clef config dir>/env``
(``~/.config/clef/env``, the file ``deploy/clef.service`` also reads). A variable that is already set to a
non-empty value (the real environment, a CLI flag, or an earlier file) always wins; set but empty counts as
unset, as in ``config._env``. ``CLEF_ENV_FILE=<path>`` loads only that file; ``CLEF_ENV_FILE=`` (empty) loads
nothing. ``CLEF_ENV_FILE`` is read from the real process environment once, at the first load of the process,
so a file can never redirect loading and a second load reads the same files. Only ``CLEF_*`` and
``GEMINI_API_KEY`` (the router's LLM key) are taken from files; other names are ignored with a warning that
lists the names. No new dependency, and values are never logged: only file names and variable names.

Syntax: one ``KEY=VALUE`` per line; blank lines and ``#`` comments are ignored; an optional ``export ``
prefix; a value may be single-quoted (literal) or double-quoted (``\\n``, ``\\"`` and ``\\\\`` escapes); an
unquoted value ends at `` #``.
"""

from __future__ import annotations

import logging
import os
import re
from collections.abc import MutableMapping
from pathlib import Path

from .paths import config_dir

log = logging.getLogger("clef.dotenv")

ENV_FILE_VAR = "CLEF_ENV_FILE"
_KEY = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_ESCAPES = {"n": "\n", "t": "\t", "r": "\r", '"': '"', "\\": "\\"}

ALLOWED_NAMES = frozenset({"GEMINI_API_KEY"})  # besides CLEF_*: the names config.py reads

loaded: list[str] = []  # display names of the env files read so far in this process (`clef doctor`)
_snapshot: list[str | None] = []  # [CLEF_ENV_FILE as the real os.environ had it at the first load], or empty
_warned: set[str] = set()  # files whose ignored names were already logged


def allowed(name: str) -> bool:
    """May an env file set ``name``? ``CLEF_*`` (but never ``CLEF_ENV_FILE``) and ``GEMINI_API_KEY``."""
    return name != ENV_FILE_VAR and (name.startswith("CLEF_") or name in ALLOWED_NAMES)


def reset() -> None:
    """Forget the per-process state (the CLEF_ENV_FILE snapshot, the files loaded): tests only."""
    loaded.clear()
    _snapshot.clear()
    _warned.clear()


def _unescape(raw: str) -> str:
    out, i = [], 0
    while i < len(raw):
        c = raw[i]
        if c == "\\" and i + 1 < len(raw) and raw[i + 1] in _ESCAPES:
            out.append(_ESCAPES[raw[i + 1]])
            i += 2
        else:
            out.append(c)
            i += 1
    return "".join(out)


def _value(raw: str) -> str:
    raw = raw.strip()
    if raw[:1] in ("'", '"'):
        quote = raw[0]
        i = 1
        while i < len(raw):
            if raw[i] == "\\" and quote == '"':
                i += 2
                continue
            if raw[i] == quote:
                inner = raw[1:i]
                return _unescape(inner) if quote == '"' else inner
            i += 1
        return raw[1:]  # unterminated: take the rest as it is
    m = re.search(r"\s#", raw)
    return (raw[: m.start()] if m else raw).strip()


def parse(text: str) -> dict[str, str]:
    """The ``KEY=VALUE`` pairs of an env file's text (later lines win over earlier ones)."""
    out: dict[str, str] = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export ") or line.startswith("export\t"):
            line = line[7:].lstrip()
        key, eq, raw = line.partition("=")
        key = key.strip()
        if eq and _KEY.match(key):
            out[key] = _value(raw)
    return out


def _setting(environ: MutableMapping[str, str] | None) -> str | None:
    """``CLEF_ENV_FILE``: for the process environment, the value it had at the first load (snapshot)."""
    if environ is not None and environ is not os.environ:
        return environ.get(ENV_FILE_VAR)
    return _snapshot[0] if _snapshot else os.environ.get(ENV_FILE_VAR)


def env_files(environ: MutableMapping[str, str] | None = None) -> list[Path]:
    """``CLEF_ENV_FILE`` alone when set (empty = none), else ``./.env`` then ``~/.config/clef/env``."""
    explicit = _setting(environ)
    if explicit is not None:
        explicit = explicit.strip()
        return [Path(explicit).expanduser()] if explicit else []
    return [Path.cwd() / ".env", config_dir() / "env"]


def _name(path: Path) -> str:
    home = Path.home()
    try:
        return "~/" + path.relative_to(home).as_posix() if path.is_absolute() else path.as_posix()
    except ValueError:
        return path.as_posix()


def load_env(environ: MutableMapping[str, str] | None = None) -> list[str]:
    """Read the env files into ``environ`` (default ``os.environ``); variables already set (non-empty) win.

    Returns the display names of the files that were read (also kept in :data:`loaded`). Safe to call twice:
    the second call reads the same files and changes nothing.
    """
    env = os.environ if environ is None else environ
    if env is os.environ and not _snapshot:
        _snapshot.append(os.environ.get(ENV_FILE_VAR))
    read: list[str] = []
    for path in env_files(env):
        name = _name(path)
        try:
            if not path.is_file():
                continue
            pairs = parse(path.read_text(encoding="utf-8-sig"))
        except (OSError, UnicodeDecodeError) as exc:
            log.warning("env file %s could not be read (%s)", name, type(exc).__name__)
            continue
        ignored = sorted(k for k in pairs if not allowed(k))
        for key, value in pairs.items():
            if allowed(key) and not (env.get(key) or "").strip():  # empty = unset, as in config._env
                env[key] = value
        if ignored and name not in _warned:
            _warned.add(name)
            log.warning(
                "env file %s: ignored %s (only CLEF_* and %s are read from env files)",
                name, ", ".join(ignored), ", ".join(sorted(ALLOWED_NAMES)),
            )  # fmt: skip
        read.append(name)
        if name not in loaded:
            loaded.append(name)
    return read


def startup_line() -> str:
    """The INFO line logged at server start: which env files were loaded (names only, never values)."""
    if loaded:
        return "env files loaded: " + ", ".join(loaded)
    return "env files: none loaded" + (" (CLEF_ENV_FILE is empty)" if _setting(None) == "" else "")
