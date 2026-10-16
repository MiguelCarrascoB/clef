"""Unit tests for logs.py: access-log redaction, JSON formatter, uvicorn log_config, CLEF_LOG_FORMAT."""

from __future__ import annotations

import json
import logging
import logging.config
import sys

import pytest
from uvicorn.logging import AccessFormatter

from clef_server import logs, main
from clef_server.config import Config
from clef_server.logs import JsonFormatter, RedactSecretsFilter, redact, request_id_var, uvicorn_log_config
from tests.unit.test_api import make


def access_record(path: str) -> logging.LogRecord:
    """What uvicorn's h11 / httptools protocol logs on `uvicorn.access` (uvicorn 0.30 .. 0.54)."""
    return logging.LogRecord(
        "uvicorn.access", logging.INFO, __file__, 1, '%s - "%s %s HTTP/%s" %d',
        ("127.0.0.1:5000", "GET", path, "1.1", 200), None,
    )  # fmt: skip


# ------------------------------------------------------------------ redaction


def test_redact_masks_key_and_keeps_other_params():
    assert redact("/v1/events?key=abc&step_s=5") == "/v1/events?key=REDACTED&step_s=5"
    assert redact("/v1/events?step_s=5&key=abc") == "/v1/events?step_s=5&key=REDACTED"
    assert redact("/x?api_key=s3cr%2Bt&token=t0k&Key=K") == "/x?api_key=REDACTED&token=REDACTED&Key=REDACTED"
    # only whole parameter names: these are not secrets
    assert redact("/x?monkey=1&step_key=2&keys=3") == "/x?monkey=1&step_key=2&keys=3"
    assert redact("/v1/events") == "/v1/events"


def test_filter_masks_access_record_and_keeps_tuple_shape():
    rec = access_record("/v1/events?key=abc&step_s=5")
    assert RedactSecretsFilter().filter(rec) is True
    assert isinstance(rec.args, tuple) and len(rec.args) == 5
    assert rec.args[2] == "/v1/events?key=REDACTED&step_s=5"
    assert rec.args[4] == 200  # non-strings untouched
    # the real uvicorn formatter still unpacks it
    line = AccessFormatter('%(client_addr)s - "%(request_line)s" %(status_code)s', use_colors=False).format(
        rec
    )
    assert "abc" not in line
    assert "/v1/events?key=REDACTED&step_s=5" in line and "step_s=5" in line


def test_filter_masks_msg_and_dict_args():
    rec = logging.LogRecord("x", logging.INFO, __file__, 1, "GET /v1/events?key=abc", None, None)
    RedactSecretsFilter().filter(rec)
    assert rec.getMessage() == "GET /v1/events?key=REDACTED"
    rec = logging.LogRecord("x", logging.INFO, __file__, 1, "%(p)s", ({"p": "/a?token=t"},), None)
    RedactSecretsFilter().filter(rec)
    assert rec.getMessage() == "/a?token=REDACTED"


# ------------------------------------------------------------------ JSON formatter


def test_json_formatter_fields_one_line():
    rec = logging.LogRecord("clef.engine", logging.WARNING, __file__, 1, "loaded %s\nin %d s", ("m", 3), None)
    rec.created = 0.0
    line = JsonFormatter().format(rec)
    assert "\n" not in line
    out = json.loads(line)
    assert out == {"ts": "1970-01-01T00:00:00.000+00:00", "level": "WARNING", "logger": "clef.engine",
                   "msg": "loaded m\nin 3 s"}  # fmt: skip


def test_json_formatter_request_id_from_context():
    rec = access_record("/v1/classify")
    assert "request_id" not in json.loads(JsonFormatter().format(rec))
    token = request_id_var.set("rid-1")
    try:
        out = json.loads(JsonFormatter().format(rec))
    finally:
        request_id_var.reset(token)
    assert out["request_id"] == "rid-1"
    assert out["msg"] == '127.0.0.1:5000 - "GET /v1/classify HTTP/1.1" 200'


def test_json_formatter_exception_stays_on_one_line():
    try:
        raise RuntimeError("boom")
    except RuntimeError:
        rec = logging.LogRecord("clef", logging.ERROR, __file__, 1, "unhandled error", None, sys.exc_info())
    line = JsonFormatter().format(rec)
    assert "\n" not in line
    out = json.loads(line)
    assert out["msg"] == "unhandled error"
    assert "RuntimeError: boom" in out["exc"] and "Traceback" in out["exc"]


# ------------------------------------------------------------------ uvicorn log_config + wiring


def test_uvicorn_log_config_text_keeps_uvicorn_formatters():
    cfg = uvicorn_log_config("text")
    assert cfg["formatters"]["access"]["()"] == "uvicorn.logging.AccessFormatter"
    assert "redact_secrets" in cfg["handlers"]["access"]["filters"]
    assert cfg["filters"]["redact_secrets"]["()"] is RedactSecretsFilter
    from uvicorn.config import LOGGING_CONFIG

    assert "filters" not in LOGGING_CONFIG["handlers"]["access"]  # deep copy, uvicorn's default untouched


def test_uvicorn_log_config_json_end_to_end(capsys):
    logging.config.dictConfig(uvicorn_log_config("json"))
    try:
        token = request_id_var.set("abc123")
        try:
            logging.getLogger("uvicorn.access").info(
                '%s - "%s %s HTTP/%s" %d', "1.2.3.4:1", "GET", "/v1/events?step_s=5&key=SECRET", "1.1", 200
            )
        finally:
            request_id_var.reset(token)
        line = capsys.readouterr().out.strip()
    finally:
        for name in ("uvicorn", "uvicorn.error", "uvicorn.access"):  # do not leak handlers on capsys streams
            logging.getLogger(name).handlers.clear()
    out = json.loads(line)
    assert out["logger"] == "uvicorn.access" and out["level"] == "INFO" and out["request_id"] == "abc123"
    assert "SECRET" not in line and "key=REDACTED" in out["msg"]


@pytest.mark.parametrize("fmt", ["text", "json"])
def test_run_passes_log_config(monkeypatch, fmt):
    seen = {}
    monkeypatch.setattr(main, "create_app", lambda cfg: "app")
    monkeypatch.setattr(main.uvicorn, "run", lambda app, **kw: seen.update(kw))
    main.run(Config(log_format=fmt))
    assert seen["workers"] == 1
    lc = seen["log_config"]
    assert "redact_secrets" in lc["handlers"]["access"]["filters"]
    assert (lc["formatters"]["access"].get("()") is JsonFormatter) == (fmt == "json")


def test_setup_logging_swaps_formatter_on_one_handler():
    root = logging.getLogger()
    try:
        logs.setup_logging("json")
        logs.setup_logging("json")
        ours = [h for h in root.handlers if getattr(h, "_clef", False)]
        assert len(ours) == 1 and isinstance(ours[0].formatter, JsonFormatter)
        logs.setup_logging("text")
        assert not isinstance(ours[0].formatter, JsonFormatter)
    finally:
        logs.setup_logging("text")


def test_lifespan_installs_access_filter():
    access = logging.getLogger("uvicorn.access")
    client, _, _ = make()
    with client:
        filters = [f for f in access.filters if isinstance(f, RedactSecretsFilter)]
    assert len(filters) == 1


# ------------------------------------------------------------------ config


def test_config_log_format(monkeypatch):
    monkeypatch.delenv("CLEF_LOG_FORMAT", raising=False)
    assert Config().log_format == "text"
    monkeypatch.setenv("CLEF_LOG_FORMAT", "JSON")
    assert Config().log_format == "json"
    with pytest.raises(ValueError, match="CLEF_LOG_FORMAT"):
        Config(log_format="xml")
