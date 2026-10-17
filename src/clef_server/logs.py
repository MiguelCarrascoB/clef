"""Server logging: text / JSON formatters, request-id context, secret redaction (ARCHITECTURE.md#logging).

Stdlib only. uvicorn is imported lazily in `uvicorn_log_config` so the CLI can import this module cheaply.
"""

from __future__ import annotations

import copy
import json
import logging
import re
from contextvars import ContextVar
from datetime import datetime, timezone
from typing import Any
from urllib.parse import unquote_plus

TEXT_FORMAT = "%(asctime)s %(levelname)-7s %(name)s %(message)s"
REDACTED = "REDACTED"

# Query parameters whose value is a secret, matched on the whole percent-decoded, lower-cased name (what auth
# reads): `k%65y=` and `KEY=` are redacted, `monkey=` / `step_key=` are left alone.
_SECRET_NAMES = frozenset({"key", "api_key", "token"})
# A `?query` part of a logged line: up to a fragment, whitespace or the quote closing uvicorn's request line.
_QUERY = re.compile(r"\?([^#\s\"]*)")

# Set by GuardMiddleware for the duration of a request; uvicorn logs the access line inside it.
request_id_var: ContextVar[str | None] = ContextVar("clef_request_id", default=None)


def _redact_query(match: re.Match[str]) -> str:
    parts = match.group(1).split("&")
    for i, part in enumerate(parts):
        name, eq, _ = part.partition("=")
        if eq and unquote_plus(name).lower() in _SECRET_NAMES:
            parts[i] = f"{name}={REDACTED}"
    return "?" + "&".join(parts)


def redact(text: str) -> str:
    """`/x?step_s=5&k%65y=abc` -> `/x?step_s=5&k%65y=REDACTED`; other params stay byte-identical."""
    return _QUERY.sub(_redact_query, text) if "=" in text else text


class RedactSecretsFilter(logging.Filter):
    """Masks secret query values in `record.msg` and every string of `record.args`; never drops a record.

    uvicorn's access record carries `(client, method, path_with_query, http_version, status)` as args, and its
    AccessFormatter unpacks that tuple, so the shape is kept and only the strings are rewritten.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        if isinstance(record.msg, str):
            record.msg = redact(record.msg)
        args = record.args
        if isinstance(args, tuple):
            record.args = tuple(redact(a) if isinstance(a, str) else a for a in args)
        elif isinstance(args, dict):
            record.args = {k: redact(v) if isinstance(v, str) else v for k, v in args.items()}
        return True


class JsonFormatter(logging.Formatter):
    """One JSON object per line: ts, level, logger, msg, plus request_id (when known) and exc (traceback)."""

    def format(self, record: logging.LogRecord) -> str:
        out: dict[str, Any] = {
            "ts": datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(timespec="milliseconds"),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        rid = getattr(record, "request_id", None) or request_id_var.get()
        if rid:
            out["request_id"] = rid
        if record.exc_info:
            out["exc"] = self.formatException(record.exc_info)
        elif record.exc_text:
            out["exc"] = record.exc_text
        if record.stack_info:
            out["stack"] = self.formatStack(record.stack_info)
        return json.dumps(out, default=str, ensure_ascii=False)


def make_formatter(fmt: str) -> logging.Formatter:
    return JsonFormatter() if fmt == "json" else logging.Formatter(TEXT_FORMAT)


def setup_logging(fmt: str = "text") -> None:
    """One root handler (stderr) for the app loggers; a repeat call only swaps the formatter."""
    root = logging.getLogger()
    handler = next((h for h in root.handlers if getattr(h, "_clef", False)), None)
    if handler is None:
        handler = logging.StreamHandler()
        handler._clef = True  # type: ignore[attr-defined]
        handler.addFilter(RedactSecretsFilter())
        root.addHandler(handler)
    handler.setFormatter(make_formatter(fmt))
    if root.level in (logging.NOTSET, logging.WARNING):
        root.setLevel(logging.INFO)


def install_access_filter() -> None:
    """Redact on the `uvicorn.access` logger too (plain `uvicorn` launches, no clef log_config)."""
    access = logging.getLogger("uvicorn.access")
    if not any(isinstance(f, RedactSecretsFilter) for f in access.filters):
        access.addFilter(RedactSecretsFilter())


def uvicorn_log_config(fmt: str = "text") -> dict[str, Any]:
    """uvicorn's default LOGGING_CONFIG plus the redaction filter on its handlers (and JSON formatters)."""
    from uvicorn.config import LOGGING_CONFIG

    cfg = copy.deepcopy(LOGGING_CONFIG)
    cfg.setdefault("filters", {})["redact_secrets"] = {"()": RedactSecretsFilter}
    for handler in cfg["handlers"].values():
        handler.setdefault("filters", []).append("redact_secrets")
    if fmt == "json":
        for name in cfg["formatters"]:
            cfg["formatters"][name] = {"()": JsonFormatter}
    return cfg
