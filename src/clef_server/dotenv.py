"""Optional ``.env`` loading: ``KEY=VALUE`` files read into ``os.environ`` before ``Config`` is built.

Files, in order: ``./.env`` (the directory ``clef serve`` runs in), then ``<clef config dir>/env``
(``~/.config/clef/env``, the file ``deploy/clef.service`` also reads). A variable that is already set
(the real environment, a CLI flag, or an earlier file) always wins. ``CLEF_ENV_FILE=<path>`` loads only that
file; ``CLEF_ENV_FILE=`` (empty) loads nothing. No new dependency, and values are never logged: only the
names of the files read.

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

loaded: list[str] = []  # display names of the env files read so far in this process (`clef doctor`)


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


def env_files(environ: MutableMapping[str, str] | None = None) -> list[Path]:
    """``CLEF_ENV_FILE`` alone when set (empty = none), else ``./.env`` then ``~/.config/clef/env``."""
    env = os.environ if environ is None else environ
    if ENV_FILE_VAR in env:
        explicit = env[ENV_FILE_VAR].strip()
        return [Path(explicit).expanduser()] if explicit else []
    return [Path.cwd() / ".env", config_dir() / "env"]


def _name(path: Path) -> str:
    home = Path.home()
    try:
        return "~/" + path.relative_to(home).as_posix() if path.is_absolute() else path.as_posix()
    except ValueError:
        return path.as_posix()


def load_env(environ: MutableMapping[str, str] | None = None) -> list[str]:
    """Read the env files into ``environ`` (default ``os.environ``); variables already set win.

    Returns the display names of the files that were read (also kept in :data:`loaded`). Safe to call twice.
    """
    env = os.environ if environ is None else environ
    read: list[str] = []
    for path in env_files(env):
        try:
            if not path.is_file():
                continue
            pairs = parse(path.read_text(encoding="utf-8-sig"))
        except (OSError, UnicodeDecodeError) as exc:
            log.warning("env file %s could not be read (%s)", _name(path), type(exc).__name__)
            continue
        for key, value in pairs.items():
            env.setdefault(key, value)
        name = _name(path)
        read.append(name)
        if name not in loaded:
            loaded.append(name)
    return read
