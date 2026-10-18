"""OpenAI-compatible Files and Batches: ``/v1/files*`` and ``/v1/batches*``.

Feature module (``main.FEATURES``, listed before ``jobs``): it registers the ``batch`` job kind and the
routes. A batch is ONE job of kind ``batch`` (visible in ``GET /v1/jobs?kind=batch``); the files and batches
tables hold what OpenAI clients see on top of it. Input files live on disk (``<state_dir>/files/<id>.jsonl``,
streamed in and renamed); output and error files are virtual, streamed from the job's rows. See
docs/openai-compat.md and docs/ARCHITECTURE.md ("OpenAI Files and Batches").

The batch logic stays here. ``jobs.py`` only knows the ``completed_if_all_failed`` kind key, the 4xx-rows rule
of the circuit breaker and the ``jobs.tokens`` counter. The job scheduler is reached through
``submit_batch_job`` (the one seam, ``JobRunner.submit``) and ``run_chunked``.
"""

from __future__ import annotations

import asyncio
import contextlib
import contextvars
import json
import logging
import os
import re
import secrets
import time
import uuid
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from fastapi import APIRouter, File, Form, Query, Request, UploadFile
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, ConfigDict, Field, ValidationError
from sqlalchemy import delete, insert, select, tuple_, update
from starlette.concurrency import run_in_threadpool

from . import dbschema
from .appctx import AppContext
from .classify import (
    ClassifyRequest,
    ClassifyResponse,
    ScoreRequest,
    ScoreResponse,
    check_label_count,
    check_levels_count,
    classify_result,
    classify_to_systemone,
    resolve_threshold,
    score_result,
    score_to_systemone,
)
from .compat_errors import openai_error, route_class
from .compat_openai import prepare_chat, render_chat
from .compat_schema import SchemaError
from .db import Database, audit
from .jobs import (
    ACTIVE,
    FINISHED,
    STREAM_BATCH,
    SUBMIT_OWNER,
    JobAborted,
    JobHandle,
    JobsFull,
    JobStore,
    _Parsed,
    run_chunked,
)
from .keystore import QuotaExceeded
from .models.registry import served_model
from .schemas import format_errors

log = logging.getLogger("clef.batches")

KIND = "batch"
ENDPOINTS = ("/v1/chat/completions", "/v1/classify", "/v1/score")
COMPLETION_WINDOW = "24h"
WINDOW_S = 24 * 3600
FILE_ID_RE = re.compile(r"^file-[0-9a-f]{24}$")
BATCH_ID_RE = re.compile(r"^batch_[0-9a-f]{20}$")
JOB_ID_RE = re.compile(r"^job_[0-9a-f]{20}$")
MAX_STRUCTURE_ERRORS = 100  # a structurally bad file reports at most this many problems
MAX_METADATA_PAIRS, MAX_METADATA_KEY, MAX_METADATA_VALUE = 16, 64, 512
EXPIRES_MIN_S, EXPIRES_MAX_S = 3600, 2592000
COPY_CHUNK = 1024 * 1024
PURGE_INTERVAL_S = 600.0
PRIORITIES = ("low", "normal", "high")


def _now() -> float:
    return time.time()


def _ts(value: float | None) -> int | None:
    return None if value is None else int(value)


# ------------------------------------------------------------------ storage


class BatchStore:
    """The ``files`` and ``batches`` tables plus the input files on disk. Sync, short calls (threadpool)."""

    def __init__(self, db: Database, files_dir: Path):
        self.db = db
        self.dir = files_dir

    # ---- disk
    def path(self, file_id: str) -> Path:
        """The path of an input file; built only from a validated id."""
        if not FILE_ID_RE.match(file_id):
            raise ValueError("bad file id")
        return self.dir / f"{file_id}.jsonl"

    def ensure_dir(self) -> None:
        self.dir.mkdir(parents=True, exist_ok=True, mode=0o700)

    def sweep_parts(self, older_than_s: float = 3600.0) -> None:
        """Leftovers of an upload that died half way."""
        with contextlib.suppress(OSError):
            for p in self.dir.glob("*.part"):
                if _now() - p.stat().st_mtime > older_than_s:
                    p.unlink(missing_ok=True)

    # ---- files
    @staticmethod
    def _row(row: Any) -> dict[str, Any] | None:
        return None if row is None else dict(row._mapping)

    def file(self, file_id: str) -> dict[str, Any] | None:
        f = dbschema.files
        with self.db.read() as conn:
            return self._row(conn.execute(select(f).where(f.c.id == file_id)).first())

    def add_input(self, values: dict[str, Any]) -> None:
        with self.db.tx() as conn:
            conn.execute(insert(dbschema.files).values(**values))

    def set_bytes(self, file_id: str, nbytes: int) -> None:
        with self.db.tx() as conn:
            conn.execute(update(dbschema.files).where(dbschema.files.c.id == file_id).values(bytes=nbytes))

    def list_files(
        self, owner: str | None, scoped: bool, after: str | None, limit: int, order: str, purpose: str | None
    ) -> tuple[list[dict[str, Any]], bool] | None:
        """One page (``limit`` rows + has_more), or None when ``after`` is not a file of the caller."""
        f = dbschema.files
        where: list[Any] = []
        if scoped:
            where.append(f.c.owner == owner)
        if purpose:
            where.append(f.c.purpose == purpose)
        return self._page(f, where, after, limit, order, owner if scoped else None, scoped)

    def _page(
        self,
        t: Any,
        where: list[Any],
        after: str | None,
        limit: int,
        order: str,
        owner: str | None,
        scoped: bool,
    ) -> tuple[list[dict[str, Any]], bool] | None:
        desc = order == "desc"
        with self.db.read() as conn:
            if after:
                ref = conn.execute(select(t.c.created_at, t.c.owner).where(t.c.id == after)).first()
                if ref is None or (scoped and ref.owner != owner):
                    return None
                key = tuple_(t.c.created_at, t.c.id)
                where = [
                    *where,
                    key < tuple_(ref.created_at, after) if desc else key > tuple_(ref.created_at, after),
                ]
            sort = (t.c.created_at.desc(), t.c.id.desc()) if desc else (t.c.created_at.asc(), t.c.id.asc())
            rows = conn.execute(select(t).where(*where).order_by(*sort).limit(limit + 1)).all()
        return [dict(r._mapping) for r in rows[:limit]], len(rows) > limit

    def delete_file(self, file_id: str) -> bool:
        with self.db.tx() as conn:
            cur = conn.execute(delete(dbschema.files).where(dbschema.files.c.id == file_id))
            return cur.rowcount == 1

    def input_in_use(self, file_id: str) -> bool:
        """True while an unfinished batch reads this input file (its error file re-reads the custom ids)."""
        b, j = dbschema.batches, dbschema.jobs
        with self.db.read() as conn:
            return (
                conn.execute(
                    select(b.c.id)
                    .join(j, j.c.id == b.c.job_id)
                    .where(b.c.input_file_id == file_id, j.c.status.in_(ACTIVE))
                    .limit(1)
                ).first()
                is not None
            )

    def purge_expired(self) -> int:
        """Delete input files past ``expires_at`` unless an unfinished batch still reads them."""
        f = dbschema.files
        with self.db.read() as conn:
            ids = list(
                conn.execute(
                    select(f.c.id).where(
                        f.c.role == "input", f.c.expires_at.is_not(None), f.c.expires_at < _now()
                    )
                ).scalars()
            )
        n = 0
        for file_id in ids:
            if self.input_in_use(file_id):
                continue
            if self.delete_file(file_id):
                with contextlib.suppress(OSError):
                    self.path(file_id).unlink(missing_ok=True)
                n += 1
        return n

    # ---- batches
    def batch(self, batch_id: str) -> dict[str, Any] | None:
        b = dbschema.batches
        with self.db.read() as conn:
            return self._row(conn.execute(select(b).where(b.c.id == batch_id)).first())

    def list_batches(
        self, owner: str | None, scoped: bool, after: str | None, limit: int
    ) -> tuple[list[dict[str, Any]], bool] | None:
        b = dbschema.batches
        where = [b.c.owner == owner] if scoped else []
        return self._page(b, where, after, limit, "desc", owner if scoped else None, scoped)

    def create_batch(self, batch: dict[str, Any], output: dict[str, Any], error: dict[str, Any]) -> None:
        """The batch row and its two virtual file rows, in one transaction (the job exists already)."""
        with self.db.tx() as conn:
            conn.execute(insert(dbschema.files), [output, error])
            conn.execute(insert(dbschema.batches).values(**batch))

    def create_failed_batch(
        self, job: dict[str, Any], batch: dict[str, Any], output: dict[str, Any], error: dict[str, Any]
    ) -> None:
        """A batch whose file failed validation: a terminal ``failed`` job row (never runnable, nothing
        reserved) plus the batch, in one transaction."""
        with self.db.tx() as conn:
            conn.execute(insert(dbschema.jobs).values(**job))
            conn.execute(insert(dbschema.files), [output, error])
            conn.execute(insert(dbschema.batches).values(**batch))

    def batch_for_job(self, job_id: str) -> dict[str, Any] | None:
        b = dbschema.batches
        with self.db.read() as conn:
            return self._row(conn.execute(select(b).where(b.c.job_id == job_id)).first())

    def mark_cancelling(self, batch_id: str) -> None:
        b = dbschema.batches
        with self.db.tx() as conn:
            conn.execute(
                update(b).where(b.c.id == batch_id, b.c.cancelling_at.is_(None)).values(cancelling_at=_now())
            )


# ------------------------------------------------------------------ input file validation


def _line_error(line: int, code: str, message: str, param: str | None = None) -> dict[str, Any]:
    return {"code": code, "line": line, "message": message, "param": param}


def scan_input(path: Path, endpoint: str, max_lines: int) -> tuple[int, list[dict[str, Any]]]:
    """Structural check of a batch input file: ``(non-blank lines, errors)``.

    Every non-blank line must be a JSON object with a unique string ``custom_id``, ``method == "POST"``,
    ``url == endpoint`` and an object ``body``. Whether the body *translates* (a schema with decisions, a
    classify label set...) is decided when the line runs: that is an error-file line, not a batch error.
    Stops after MAX_STRUCTURE_ERRORS problems.
    """
    errors: list[dict[str, Any]] = []
    seen: set[str] = set()
    lines = 0
    with path.open("rb") as fh:
        for no, raw in enumerate(fh, 1):
            if not raw.strip():
                continue
            lines += 1
            if lines > max_lines:
                errors.append(
                    _line_error(no, "too_many_lines", f"the file has more than {max_lines} requests")
                )
                break
            problem = _check_line(raw, endpoint, seen)
            if problem is not None:
                errors.append(_line_error(no, *problem))
                if len(errors) >= MAX_STRUCTURE_ERRORS:
                    break
    if lines == 0:
        errors.append(_line_error(1, "empty_file", "the file has no requests"))
    return lines, errors


def _check_line(raw: bytes, endpoint: str, seen: set[str]) -> tuple[str, str, str | None] | None:
    try:
        obj = json.loads(raw)
    except (ValueError, RecursionError):  # RecursionError: nested too deeply for the parser
        return "invalid_json_line", "the line is not valid JSON", None
    if not isinstance(obj, dict):
        return "invalid_request", "the line must be a JSON object", None
    cid = obj.get("custom_id")
    if not isinstance(cid, str) or not cid:
        return "missing_custom_id", "custom_id must be a non-empty string", "custom_id"
    if cid in seen:
        return "duplicate_custom_id", f"custom_id {cid[:64]!r} appears more than once", "custom_id"
    seen.add(cid)
    if obj.get("method") != "POST":
        return "invalid_method", 'method must be "POST"', "method"
    if obj.get("url") != endpoint:
        return "invalid_url", f"url must be {endpoint!r} (the batch's endpoint)", "url"
    if not isinstance(obj.get("body"), dict):
        return "invalid_request", "body must be a JSON object", "body"
    return None


def line_offsets(path: Path, limit: int) -> list[int]:
    """Byte offset of every non-blank line (at most ``limit``): line ``k`` of the batch is ``offsets[k]``."""
    offsets: list[int] = []
    pos = 0
    with path.open("rb") as fh:
        for raw in fh:
            if raw.strip():
                offsets.append(pos)
                if len(offsets) >= limit:
                    break
            pos += len(raw)
    return offsets


def input_custom_ids(path: Path, start: int) -> Iterator[tuple[int, str | None]]:
    """``(ordinal, custom_id)`` of the non-blank lines from ordinal ``start`` on (None if unreadable)."""
    k = -1
    with path.open("rb") as fh:
        for raw in fh:
            if not raw.strip():
                continue
            k += 1
            if k < start:
                continue
            try:
                cid = json.loads(raw).get("custom_id")
            except (ValueError, AttributeError, RecursionError):
                cid = None
            yield k, cid if isinstance(cid, str) else None


# ------------------------------------------------------------------ the batch job kind


@dataclass
class _LineError(Exception):
    """A request that cannot run: becomes an error-file line with this status and OpenAI error body."""

    status: int
    message: str
    code: str | None = None
    param: str | None = None
    type: str = "invalid_request_error"

    def body(self) -> dict[str, Any]:
        return {"message": self.message, "type": self.type, "param": self.param, "code": self.code}


def _first_param(exc: ValidationError) -> str | None:
    errs = exc.errors()
    return ".".join(str(p) for p in errs[0]["loc"]) if errs and errs[0].get("loc") else None


class _Request:
    """One input line, translated: what to ask the engine and how to shape the answer."""

    def __init__(self, custom_id: str | None, request_id: str, sreq: Any, shape: Any):
        self.custom_id, self.request_id, self.sreq, self.shape = custom_id, request_id, sreq, shape


@dataclass
class _Failed:
    """One input line that did not translate."""

    error: _LineError
    custom_id: str | None
    request_id: str


def translate_line(ctx: AppContext, endpoint: str, obj: Any, request_id: str) -> _Request:
    """Line -> engine request, with the live route's validation and model check. Raises _LineError."""
    if not isinstance(obj, dict) or not isinstance(obj.get("body"), dict):
        raise _LineError(400, "the line must be a JSON object with a body", "invalid_request")
    cid = obj.get("custom_id") if isinstance(obj.get("custom_id"), str) else None
    body = obj["body"]
    try:
        if endpoint == "/v1/chat/completions":
            return _chat_line(ctx, body, cid, request_id)
        if endpoint == "/v1/classify":
            return _classify_line(ctx, body, cid, request_id)
        return _score_line(ctx, body, cid, request_id)
    except SchemaError as exc:
        raise _LineError(400, str(exc), exc.code, exc.param) from exc
    except ValidationError as exc:
        raise _LineError(
            400, format_errors(list(exc.errors())), "invalid_request", _first_param(exc)
        ) from exc
    except LookupError as exc:  # UnknownModel: one model per server
        raise _LineError(404, str(exc), "model_not_found", "model") from exc
    except ValueError as exc:
        raise _LineError(400, str(exc), "invalid_request") from exc


def _chat_line(ctx: AppContext, body: dict[str, Any], cid: str | None, request_id: str) -> _Request:
    requested = body.get("model")
    if requested is not None and not isinstance(requested, str):
        raise SchemaError("model must be a string", "model")
    if body.get("stream") is True:
        raise SchemaError("stream is not supported in a batch", "stream", "unsupported_value")
    model = served_model(ctx.engine, requested)
    chat = prepare_chat(body, ctx.cfg, model)

    def shape(res: dict[str, Any]) -> dict[str, Any]:
        return render_chat(chat, res, request_id)

    return _Request(cid, request_id, chat.sreq, shape)


def _classify_line(ctx: AppContext, body: dict[str, Any], cid: str | None, request_id: str) -> _Request:
    req = ClassifyRequest.model_validate(body)
    model = ctx.check_model(req.model)
    check_label_count(req.labels, req.multi_label, ctx.cfg.max_labels)
    thr = resolve_threshold(req.threshold, ctx.cfg) if req.multi_label else None
    sreq = classify_to_systemone(
        req.input, req.labels, req.instructions, req.multi_label, model, req.images, req.videos
    )

    def shape(res: dict[str, Any]) -> dict[str, Any]:
        served = res.get("model") or model
        raw = {
            **classify_result(res["answers"], req.labels, req.multi_label, thr, served),
            "usage": res.get("usage", {}),
            "timing": res.get("timing") or {},
            "request_id": request_id,
        }
        return ClassifyResponse.model_validate(raw).model_dump(mode="json", exclude_none=True)

    return _Request(cid, request_id, sreq, shape)


def _score_line(ctx: AppContext, body: dict[str, Any], cid: str | None, request_id: str) -> _Request:
    req = ScoreRequest.model_validate(body)
    model = ctx.check_model(req.model)
    check_levels_count(req.levels, ctx.cfg.max_labels)
    sreq = score_to_systemone(req.input, req.levels, req.instructions, model, req.images, req.videos)

    def shape(res: dict[str, Any]) -> dict[str, Any]:
        raw = {
            **score_result(res["answers"], req.levels, res.get("model") or model),
            "usage": res.get("usage", {}),
            "timing": res.get("timing") or {},
            "request_id": request_id,
        }
        return ScoreResponse.model_validate(raw).model_dump(mode="json", exclude_none=True)

    return _Request(cid, request_id, sreq, shape)


def _engine_error(message: str) -> _LineError:
    """An item the engine could not run (decide_resilient isolated it): too long / GPU out of memory."""
    if message.startswith("GPU out of memory"):
        return _LineError(500, message, "server_error", None, "server_error")
    return _LineError(400, message, "invalid_request")


# Set by POST /v1/batches around its submit: that route already scanned (input_file_id, endpoint) itself.
_SCANNED: contextvars.ContextVar[tuple[str, str] | None] = contextvars.ContextVar(
    "clef_batch_scanned", default=None
)


def batch_kind(ctx: AppContext, store: BatchStore) -> dict[str, Any]:
    def scan_at_submit(payload: dict[str, Any], p: _Parsed) -> dict[str, Any]:
        """``snapshot`` hook (submit only, not on every run): POST /v1/jobs kind=batch gets the
        structural scan of POST /v1/batches, as a 400 naming the first bad line. Only for the submitter's
        own file (another key's file is refused when the job runs, and the scan must not describe it)."""
        if _SCANNED.get() == (p.input_file_id, p.endpoint):
            return payload  # POST /v1/batches scanned it already
        submitter, frow = SUBMIT_OWNER.get(), store.file(p.input_file_id)
        if submitter is None or frow is None or (frow["owner"] or None) != (submitter[0] or None):
            return payload
        limit = min(ctx.cfg.max_batch_lines, ctx.cfg.max_job_items)
        try:
            _, errors = scan_input(store.path(p.input_file_id), p.endpoint, limit)
        except OSError as exc:
            raise ValueError(f"input_file_id: file {p.input_file_id} not found") from exc
        if errors:
            e = errors[0]
            more = f" (and {len(errors) - 1} more)" if len(errors) > 1 else ""
            raise ValueError(f"input file failed validation: line {e['line']}: {e['message']}{more}")
        return payload

    def validate(payload: dict[str, Any]) -> _Parsed:
        if not isinstance(payload, dict):
            raise ValueError("payload must be an object")
        fid, endpoint = payload.get("input_file_id"), payload.get("endpoint")
        if not isinstance(fid, str) or not FILE_ID_RE.match(fid):
            raise ValueError("input_file_id: not a file id")
        if endpoint not in ENDPOINTS:
            raise ValueError(f"endpoint: must be one of {', '.join(ENDPOINTS)}")
        row = store.file(fid)
        if row is None or row["role"] != "input":
            raise ValueError(f"input_file_id: file {fid} not found")
        # The line count comes from the file row, never from the payload: it sizes the quota reservation.
        return _Parsed(input_file_id=fid, endpoint=endpoint, lines=max(1, int(row["lines"] or 1)))

    async def run(ctx_: AppContext, p: _Parsed, job: JobHandle) -> dict[str, Any]:
        runner = ctx_.extra["jobs"]
        jrow = await asyncio.to_thread(runner.store.get, job.id) or {}
        frow = await asyncio.to_thread(store.file, p.input_file_id)
        # POST /v1/jobs can submit kind=batch with any file id: the file must belong to the job's owner.
        if frow is None or (frow["owner"] or None) != (jrow.get("owner") or None):
            raise JobAborted(f"input file {p.input_file_id} not found")
        path = store.path(p.input_file_id)
        limit = min(ctx_.cfg.max_batch_lines, ctx_.cfg.max_job_items)
        try:
            offsets = await asyncio.to_thread(line_offsets, path, limit)
        except OSError as exc:
            raise JobAborted(f"input file {p.input_file_id} is gone") from exc
        pending: dict[int, _Request | _Failed] = {}
        errors = int(jrow.get("failed") or 0)  # on resume: the failed rows already stored
        fh = path.open("rb")

        def build(k: int) -> Any:
            request_id = "req_" + uuid.uuid4().hex[:24]
            fh.seek(offsets[k])
            raw = fh.readline()
            try:
                obj = json.loads(raw)
            except (ValueError, RecursionError):
                obj = None
            try:
                req = translate_line(ctx_, p.endpoint, obj, request_id)
            except _LineError as exc:
                cid = obj.get("custom_id") if isinstance(obj, dict) else None
                pending[k] = _Failed(exc, cid if isinstance(cid, str) else None, request_id)
                raise ValueError(exc.message) from exc  # run_chunked turns it into an error row for tally
            pending[k] = req
            return req.sreq

        def shape(k: int, res: dict[str, Any]) -> dict[str, Any]:
            req = pending[k]
            assert isinstance(req, _Request)
            return {
                "index": k,
                "custom_id": req.custom_id,
                "status_code": 200,
                "request_id": req.request_id,
                "body": req.shape(res),
                "input_tokens": int((res.get("usage") or {}).get("input_tokens") or 0),
            }

        def tally(row: dict[str, Any]) -> None:
            """Runs on every row before it is stored: give an item error its custom_id, status and body."""
            nonlocal errors
            k = row["index"]
            entry = pending.pop(k)
            if "error" in row:  # a translation error (build) or an item the engine rejected
                # else: the line translated and the engine refused it (the ItemError text is row["error"])
                exc = entry.error if isinstance(entry, _Failed) else _engine_error(str(row["error"]))
                row.update(
                    custom_id=entry.custom_id,
                    status_code=exc.status,
                    request_id=entry.request_id,
                    error=exc.body(),
                    input_tokens=0,
                )
                errors += 1

        try:
            await run_chunked(ctx_, job, len(offsets), build, shape, tally)
        finally:
            fh.close()
        return {"items": job.done, "ok": job.done - errors, "errors": errors}

    return {
        "validate": validate,
        "snapshot": scan_at_submit,
        "run": run,
        "resumable": True,
        "completed_if_all_failed": True,  # OpenAI: a batch where every request failed is still `completed`
        "records": lambda p: p.lines,  # quota: one record per request line
    }


# ------------------------------------------------------------------ output and error files (virtual)


def _finished(job: dict[str, Any]) -> bool:
    return job["status"] in FINISHED


def virtual_file(
    jobs: JobStore, store: BatchStore, job: dict[str, Any], which: str, input_id: str | None
) -> Iterator[bytes]:
    """The JSONL content of a batch's output (2xx rows) or error file (the rest), in input order.

    Streamed from the job's rows; for a finished job the error file ends with one line per request that never
    got a row (the batch expired, was cancelled or failed), with the custom ids re-read from the input file.
    """
    hexid = job["id"].removeprefix("job_")
    done = job["done"]  # frozen: rows keep arriving while a job runs
    offset = 0
    while offset < done:
        page = jobs.items(job["id"], offset, STREAM_BATCH)
        if not page:
            break
        out: list[str] = []
        for idx, raw in page:
            if idx >= done:
                break
            row = json.loads(raw)
            bad = "error" in row
            if bad != (which == "error"):
                continue
            body = {"error": row["error"]} if bad else row["body"]
            out.append(
                json.dumps(
                    {
                        "id": f"batch_req_{hexid}_{idx}",
                        "custom_id": row.get("custom_id"),
                        "response": {
                            "status_code": row.get("status_code", 500 if bad else 200),
                            "request_id": row.get("request_id"),
                            "body": body,
                        },
                        "error": None,
                    },
                    ensure_ascii=False,
                    separators=(",", ":"),
                )
            )
        offset = page[-1][0] + 1
        if out:
            yield ("\n".join(out) + "\n").encode()
    if which != "error" or not _finished(job) or job["status"] == "succeeded" or input_id is None:
        return
    code = (
        "batch_expired"
        if job["error"] == "expired"
        else "batch_cancelled"
        if job["status"] == "cancelled"
        else "batch_failed"
    )
    why = {"batch_expired": "expired", "batch_cancelled": "was cancelled", "batch_failed": "failed"}[code]
    try:
        for k, cid in input_custom_ids(store.path(input_id), done):
            yield (
                json.dumps(
                    {
                        "id": f"batch_req_{hexid}_{k}",
                        "custom_id": cid,
                        "response": None,
                        "error": {"code": code, "message": f"this request did not run: the batch {why}"},
                    },
                    separators=(",", ":"),
                )
                + "\n"
            ).encode()
    except OSError:  # the input file was deleted after the batch finished: nothing to name the lines with
        return


def file_chunks(path: Path) -> Iterator[bytes]:
    with path.open("rb") as fh:
        while chunk := fh.read(COPY_CHUNK):
            yield chunk


# ------------------------------------------------------------------ OpenAI objects


class FileObject(BaseModel):
    id: str
    object: Literal["file"] = "file"
    bytes: int
    created_at: int
    expires_at: int | None = None
    filename: str
    purpose: str
    status: Literal["processed"] = "processed"
    status_details: str | None = None


class FileList(BaseModel):
    object: Literal["list"] = "list"
    data: list[FileObject]
    first_id: str | None = None
    last_id: str | None = None
    has_more: bool = False


class FileDeleted(BaseModel):
    id: str
    object: Literal["file"] = "file"
    deleted: bool = True


class RequestCounts(BaseModel):
    total: int
    completed: int
    failed: int


class BatchError(BaseModel):
    code: str | None = None
    line: int | None = None
    message: str | None = None
    param: str | None = None


class BatchErrors(BaseModel):
    object: Literal["list"] = "list"
    data: list[BatchError]


class InputTokensDetails(BaseModel):
    cached_tokens: int = 0


class OutputTokensDetails(BaseModel):
    reasoning_tokens: int = 0


class BatchUsage(BaseModel):
    input_tokens: int = 0
    input_tokens_details: InputTokensDetails = Field(default_factory=InputTokensDetails)
    output_tokens: int = 0
    output_tokens_details: OutputTokensDetails = Field(default_factory=OutputTokensDetails)
    total_tokens: int = 0


class BatchObject(BaseModel):
    id: str
    object: Literal["batch"] = "batch"
    endpoint: str
    errors: BatchErrors | None = None
    input_file_id: str
    completion_window: str
    status: Literal[
        "validating", "failed", "in_progress", "finalizing", "completed", "expired", "cancelling", "cancelled"
    ]
    output_file_id: str | None = None
    error_file_id: str | None = None
    created_at: int
    in_progress_at: int | None = None
    expires_at: int | None = None
    finalizing_at: int | None = None
    completed_at: int | None = None
    failed_at: int | None = None
    expired_at: int | None = None
    cancelling_at: int | None = None
    cancelled_at: int | None = None
    request_counts: RequestCounts
    metadata: dict[str, str] | None = None
    model: str | None = None
    usage: BatchUsage = Field(default_factory=BatchUsage)


class BatchList(BaseModel):
    object: Literal["list"] = "list"
    data: list[BatchObject]
    first_id: str | None = None
    last_id: str | None = None
    has_more: bool = False


class ClefOptions(BaseModel):
    """clef's extension, sent as ``extra_body={"clef": {...}}`` (the job scheduler's knobs)."""

    model_config = ConfigDict(extra="forbid")
    priority: Literal["low", "normal", "high"] = "normal"
    urgent: bool = Field(False, description="needs the `urgent` scope; ignores the off-hours window")
    schedule: dict[str, str] | None = Field(None, description='{"window": "HH:MM-HH:MM", "tz": "<IANA>"}')


class BatchCreate(BaseModel):
    model_config = ConfigDict(extra="ignore")
    input_file_id: str
    endpoint: str
    completion_window: str
    metadata: dict[str, Any] | None = None
    clef: ClefOptions | None = None


def file_view(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": row["id"],
        "object": "file",
        "bytes": int(row["bytes"]),
        "created_at": int(row["created_at"]),
        "expires_at": _ts(row["expires_at"]),
        "filename": row["filename"],
        "purpose": row["purpose"],
        "status": "processed",
        "status_details": None,
    }


def batch_view(b: dict[str, Any], j: dict[str, Any], lines: int | None, model: str) -> dict[str, Any]:
    """The Batch object from the batch row, its job row and the input line count."""
    status = j["status"]
    ran = _finished(j) and not b["errors"]  # a batch that failed validation never ran: no files
    done, failed = j["done"], j["failed"]
    total = lines if lines else (j["total"] or done)
    if status in ACTIVE:
        state = "cancelling" if (j["cancel_requested"] or b["cancelling_at"]) else "in_progress"
    elif status == "succeeded":
        state = "completed"
    elif status == "failed":
        state = "failed"
    else:
        state = "expired" if j["error"] == "expired" else "cancelled"
    errors = None
    if b["errors"]:
        errors = {"object": "list", "data": json.loads(b["errors"])}
    elif state == "failed":
        errors = {
            "object": "list",
            "data": [{"code": "server_error", "message": j["error"] or "the batch failed"}],
        }
    end = _ts(j["finished_at"])
    eligible = j["eligible_at"]
    window = j["expire_after_s"] or WINDOW_S
    ok = done - failed
    started = _ts(j["started_at"])
    return {
        "id": b["id"],
        "object": "batch",
        "endpoint": b["endpoint"],
        "errors": errors,
        "input_file_id": b["input_file_id"],
        "completion_window": b["completion_window"],
        "status": state,
        # The output file exists once it has a line; the error file when a request failed or never ran.
        "output_file_id": b["output_file_id"] if ran and ok > 0 else None,
        "error_file_id": b["error_file_id"]
        if ran and (failed > 0 or (status != "succeeded" and done < total))
        else None,
        "created_at": int(b["created_at"]),
        "in_progress_at": started,
        # D19: the 24 h start when the window opens (jobs.eligible_at); null while the batch is waiting for it
        "expires_at": None if eligible is None else int(eligible + window),
        "finalizing_at": end if state == "completed" else None,
        "completed_at": end if state == "completed" else None,
        "failed_at": end if state == "failed" else None,
        "expired_at": end if state == "expired" else None,
        "cancelling_at": _ts(b["cancelling_at"]),
        "cancelled_at": end if state == "cancelled" else None,
        "request_counts": {"total": total, "completed": max(0, ok), "failed": failed},
        "metadata": json.loads(b["metadata"]) if b["metadata"] else None,
        "model": model,
        "usage": {
            "input_tokens": int(j["tokens"]),
            "input_tokens_details": {"cached_tokens": 0},
            "output_tokens": 0,
            "output_tokens_details": {"reasoning_tokens": 0},
            "total_tokens": int(j["tokens"]),
        },
    }


# ------------------------------------------------------------------ the job-scheduler seam


async def submit_batch_job(
    ctx: AppContext, request: Request, payload: dict[str, Any], options: ClefOptions
) -> dict[str, Any]:
    """Create the job of a new batch; returns its public job view (``id`` at least). The one seam to E1.

    ``JobRunner.submit`` validates the payload, reserves the daily quota (the file's stored line count, once),
    queues the job with its priority / window / 24 h expiry and wakes the scheduler.

    Raises ApiError: 400 (a bad option), 403 (`urgent` without the scope), 404 (a model not served),
    409 (CLEF_MAX_JOBS), 429 (daily quota; nothing stays reserved).
    """
    runner = ctx.extra["jobs"]
    principal = ctx.principal(request)
    try:
        row = await runner.submit(
            KIND,
            payload,
            principal=principal,
            priority=options.priority,
            urgent=options.urgent,
            schedule=options.schedule,
            expire_after_s=WINDOW_S,
        )
    except QuotaExceeded as exc:
        request.state.quota = (exc.limit, exc.remaining)
        raise _scheduler_error(ctx, exc) from exc
    except (KeyError, IndexError):
        raise
    except LookupError as exc:
        raise ctx.api_error(404, str(exc)) from exc
    except (JobsFull, PermissionError, ValueError) as exc:
        raise _scheduler_error(ctx, exc) from exc
    request.state.quota = ctx.quotas.remaining(principal)
    return row


def _scheduler_error(ctx: AppContext, exc: Exception) -> Exception:
    if isinstance(exc, QuotaExceeded):
        return ctx.api_error(429, str(exc), {"Retry-After": str(exc.retry_after)}, code="insufficient_quota")  # type: ignore[call-arg]
    if isinstance(exc, JobsFull):
        return ctx.api_error(409, str(exc), {"Retry-After": "30"})
    if isinstance(exc, PermissionError):
        return ctx.api_error(403, str(exc))
    return ctx.api_error(400, str(exc))


# ------------------------------------------------------------------ routes


def router(ctx: AppContext) -> APIRouter:
    cfg, ApiError = ctx.cfg, ctx.api_error  # noqa: N806
    store = BatchStore(ctx.db, Path(cfg.state_dir) / "files")
    jobs = JobStore(ctx.db)  # a second handle on the same Database: reads, and the batch row lookups
    ctx.extra.setdefault("job_kinds", {})[KIND] = batch_kind(ctx, store)
    ctx.extra["batches"] = store
    purge_task: list[asyncio.Task[None]] = []

    async def purge_loop() -> None:
        while True:
            try:
                n = await asyncio.to_thread(store.purge_expired)
                if n:
                    log.info("files: purged %d expired input file(s)", n)
            except asyncio.CancelledError:
                raise
            except Exception:
                log.exception("file purge failed")
            await asyncio.sleep(PURGE_INTERVAL_S)

    async def start() -> None:
        await asyncio.to_thread(store.ensure_dir)
        await asyncio.to_thread(store.sweep_parts)
        purge_task.append(asyncio.create_task(purge_loop(), name="clef-files-purge"))

    async def stop() -> None:
        for t in purge_task:
            t.cancel()
        await asyncio.gather(*purge_task, return_exceptions=True)

    ctx.on_startup.append(start)
    ctx.on_shutdown.append(stop)

    api = APIRouter(
        prefix="/v1",
        tags=["files and batches (OpenAI)"],
        dependencies=ctx.scope("jobs"),
        responses=ctx.errors,
        route_class=route_class(ctx, openai_error),
    )

    def caller(request: Request) -> str | None:
        return getattr(request.state, "key_name", None)

    def scoped(request: Request) -> bool:
        """Files and batches are private to their owner when auth is on; admins see every one."""
        principal = ctx.principal(request)
        return ctx.auth_enabled() and not (principal is not None and principal.has("admin"))

    def not_found(what: str, ident: str) -> Exception:
        return ApiError(404, f"No such {what}: {ident!r}")

    def mine(request: Request, row: dict[str, Any] | None) -> bool:
        return row is not None and (not scoped(request) or row["owner"] == caller(request))

    async def audit_action(request: Request, action: str, target: str, detail: dict[str, Any]) -> None:
        def write() -> None:
            with ctx.db.tx() as conn:
                audit(
                    conn,
                    caller(request) or "anonymous",
                    action,
                    target,
                    detail,
                    request.client.host if request.client else None,
                    ctx.request_id(request),
                )

        await run_in_threadpool(write)

    def new_file_row(
        file_id: str,
        owner: str | None,
        purpose: str,
        role: str,
        filename: str,
        job_id: str | None = None,
        **more: Any,
    ) -> dict[str, Any]:
        return {
            "id": file_id,
            "owner": owner,
            "purpose": purpose,
            "role": role,
            "filename": filename,
            "bytes": 0,
            "lines": None,
            "job_id": job_id,
            "created_at": _now(),
            "expires_at": None,
            **more,
        }

    # ---------------------------------------------------------------- files
    @api.post(
        "/files",
        response_model=FileObject,
        summary="Upload a batch input file (multipart, purpose=batch)",
        dependencies=ctx.limited,
    )
    async def upload_file(
        request: Request,
        file: UploadFile = File(description="a .jsonl file, one request per line"),  # noqa: B008
        purpose: str = Form(description='only "batch"'),
        anchor: str | None = Form(None, alias="expires_after[anchor]"),
        seconds: int | None = Form(None, alias="expires_after[seconds]"),
    ) -> dict[str, Any]:
        if purpose != "batch":
            raise SchemaError("purpose must be 'batch': clef stores batch input files only", "purpose")
        name = os.path.basename((file.filename or "").replace("\\", "/")) or "upload.jsonl"
        if not name.lower().endswith(".jsonl"):
            raise SchemaError("file must be a .jsonl file", "file")
        if (anchor is None) != (seconds is None):
            raise SchemaError("expires_after needs both anchor and seconds", "expires_after")
        if anchor is not None and anchor != "created_at":
            raise SchemaError("expires_after.anchor must be 'created_at'", "expires_after.anchor")
        if seconds is not None and not EXPIRES_MIN_S <= seconds <= EXPIRES_MAX_S:
            raise SchemaError(
                f"expires_after.seconds must be between {EXPIRES_MIN_S} and {EXPIRES_MAX_S}",
                "expires_after.seconds",
            )
        file_id = "file-" + secrets.token_hex(12)
        now = _now()
        ttl = float(seconds) if seconds is not None else cfg.file_ttl_hours * 3600
        row = new_file_row(file_id, caller(request), "batch", "input", name, created_at=now)
        row["expires_at"] = now + ttl if ttl > 0 else None

        def save() -> None:
            store.ensure_dir()
            final = store.path(file_id)
            part = final.with_suffix(".part")
            nbytes = 0
            lines = 0
            try:
                with part.open("wb") as out:
                    while chunk := file.file.read(COPY_CHUNK):
                        out.write(chunk)
                        nbytes += len(chunk)
                    out.flush()
                    os.fsync(out.fileno())
                with part.open("rb") as fh:
                    lines = sum(1 for raw in fh if raw.strip())
                if nbytes == 0 or lines == 0:
                    raise SchemaError("the file is empty", "file", "empty_file")
                os.replace(part, final)
                row.update(bytes=nbytes, lines=lines)
                store.add_input(row)
            except BaseException:
                part.unlink(missing_ok=True)
                final.unlink(missing_ok=True)
                raise

        await run_in_threadpool(save)
        return file_view(row)

    async def visible_file(request: Request, file_id: str) -> dict[str, Any]:
        row = await run_in_threadpool(store.file, file_id) if FILE_ID_RE.match(file_id) else None
        if not mine(request, row):
            raise not_found("file", file_id)
        return row  # type: ignore[return-value]

    async def with_bytes(row: dict[str, Any]) -> dict[str, Any]:
        """A virtual file's size is only known once its batch ended: count it once, then keep it."""
        if row["role"] == "input" or row["bytes"] or not row["job_id"]:
            return row
        job = await run_in_threadpool(jobs.get, row["job_id"])
        if job is None or not _finished(job):
            return row

        def count() -> int:
            batch = store.batch_for_job(job["id"])
            return sum(
                len(c) for c in virtual_file(jobs, store, job, row["role"], batch and batch["input_file_id"])
            )

        n = await run_in_threadpool(count)
        await run_in_threadpool(store.set_bytes, row["id"], n)
        return {**row, "bytes": n}

    @api.get("/files", response_model=FileList, summary="List your files")
    async def list_files(
        request: Request,
        after: str | None = None,
        limit: int = Query(20, ge=1, le=10000),
        order: Literal["asc", "desc"] = "desc",
        purpose: str | None = None,
    ) -> dict[str, Any]:
        page = await run_in_threadpool(
            store.list_files, caller(request), scoped(request), after, limit, order, purpose
        )
        if page is None:
            raise SchemaError(f"after: no such file {after!r}", "after")
        rows, more = page
        rows = [await with_bytes(r) for r in rows]
        return _list(rows, more, file_view)

    @api.get("/files/{file_id}", response_model=FileObject, summary="Retrieve a file")
    async def get_file(file_id: str, request: Request) -> dict[str, Any]:
        return file_view(await with_bytes(await visible_file(request, file_id)))

    @api.delete("/files/{file_id}", response_model=FileDeleted, summary="Delete an input file")
    async def delete_file(file_id: str, request: Request) -> dict[str, Any]:
        row = await visible_file(request, file_id)
        if row["role"] != "input":
            raise ApiError(
                400,
                f"{file_id} is a batch output file: delete the batch's job instead "
                f"(DELETE /v1/jobs/{row['job_id']})",
            )
        if await run_in_threadpool(store.input_in_use, file_id):
            raise ApiError(409, "a batch that has not finished still reads this file: cancel it first")
        await run_in_threadpool(store.delete_file, file_id)
        with contextlib.suppress(OSError):
            await run_in_threadpool(store.path(file_id).unlink, True)
        await audit_action(
            request, "file.delete", file_id, {"owner": row["owner"], "filename": row["filename"]}
        )
        return {"id": file_id, "object": "file", "deleted": True}

    @api.get(
        "/files/{file_id}/content",
        summary="Download a file (JSONL, streamed)",
        responses={200: {"content": {"application/jsonl": {}}}},
    )
    async def file_content(file_id: str, request: Request) -> StreamingResponse:
        row = await visible_file(request, file_id)
        headers = {"Cache-Control": "no-store"}
        if row["role"] == "input":
            try:
                path = store.path(file_id)
                size = path.stat().st_size
            except OSError:
                raise not_found("file", file_id) from None
            return StreamingResponse(
                file_chunks(path),
                media_type="application/jsonl",
                headers={**headers, "Content-Length": str(size)},
            )
        job = await run_in_threadpool(jobs.get, row["job_id"])
        if job is None:
            raise not_found("file", file_id)
        batch = await run_in_threadpool(store.batch_for_job, job["id"])
        return StreamingResponse(
            virtual_file(jobs, store, job, row["role"], batch and batch["input_file_id"]),
            media_type="application/jsonl",
            headers=headers,
        )

    # ---------------------------------------------------------------- batches
    async def view(b: dict[str, Any], j: dict[str, Any] | None = None) -> dict[str, Any]:
        j = j or await run_in_threadpool(jobs.get, b["job_id"])
        if j is None:
            raise not_found("batch", b["id"])
        frow = await run_in_threadpool(store.file, b["input_file_id"])
        lines = frow["lines"] if frow else None
        return batch_view(b, j, lines, (j.get("model") or served_model(ctx.engine, None)).split("@")[0])

    async def visible_batch(request: Request, batch_id: str) -> dict[str, Any]:
        row = await run_in_threadpool(store.batch, batch_id) if BATCH_ID_RE.match(batch_id) else None
        if not mine(request, row):
            raise not_found("batch", batch_id)
        return row  # type: ignore[return-value]

    def check_metadata(meta: dict[str, Any] | None) -> None:
        if meta is None:
            return
        if len(meta) > MAX_METADATA_PAIRS:
            raise SchemaError(f"metadata: at most {MAX_METADATA_PAIRS} key-value pairs", "metadata")
        for k, v in meta.items():
            if len(k) > MAX_METADATA_KEY:
                raise SchemaError(f"metadata: keys are at most {MAX_METADATA_KEY} characters", "metadata")
            if not isinstance(v, str) or len(v) > MAX_METADATA_VALUE:
                raise SchemaError(
                    f"metadata: values are strings of at most {MAX_METADATA_VALUE} characters", "metadata"
                )

    @api.post(
        "/batches",
        response_model=BatchObject,
        summary="Create a batch from an uploaded file",
        dependencies=ctx.limited,
    )
    async def create_batch(body: BatchCreate, request: Request) -> dict[str, Any]:
        if body.endpoint not in ENDPOINTS:
            raise SchemaError(f"endpoint must be one of {', '.join(ENDPOINTS)}", "endpoint")
        if body.completion_window != COMPLETION_WINDOW:
            raise SchemaError(f"completion_window must be {COMPLETION_WINDOW!r}", "completion_window")
        check_metadata(body.metadata)
        frow = (
            await run_in_threadpool(store.file, body.input_file_id)
            if FILE_ID_RE.match(body.input_file_id)
            else None
        )
        # Own files only, admins included: the job re-checks it when it runs (POST /v1/jobs can submit one).
        if (
            frow is None
            or frow["role"] != "input"
            or frow["purpose"] != "batch"
            or frow["owner"] != caller(request)
        ):
            raise not_found("file", body.input_file_id)
        limit = min(cfg.max_batch_lines, cfg.max_job_items)
        try:
            lines, errors = await run_in_threadpool(scan_input, store.path(frow["id"]), body.endpoint, limit)
        except OSError:
            raise not_found("file", body.input_file_id) from None
        options = body.clef or ClefOptions()
        owner, now = caller(request), _now()

        def rows_for(job_id: str) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
            """The batch row and its two virtual file rows; the batch id is the job id's hex."""
            batch_id = "batch_" + job_id.removeprefix("job_")
            files = {
                role: new_file_row(
                    "file-" + secrets.token_hex(12),
                    owner,
                    "batch_output",
                    role,
                    f"{batch_id}_{role}.jsonl",
                    job_id,
                    created_at=now,
                )
                for role in ("output", "error")
            }
            batch = {
                "id": batch_id,
                "job_id": job_id,
                "owner": owner,
                "endpoint": body.endpoint,
                "input_file_id": frow["id"],
                "output_file_id": files["output"]["id"],
                "error_file_id": files["error"]["id"],
                "completion_window": COMPLETION_WINDOW,
                "metadata": json.dumps(body.metadata) if body.metadata else None,
                "errors": json.dumps(errors) if errors else None,
                "created_at": now,
                "cancelling_at": None,
            }
            return batch, files["output"], files["error"]

        if errors:  # a structural failure is a `failed` batch (200), as OpenAI does: no job runs, no charge
            job_id = "job_" + secrets.token_hex(10)
            job_row = {
                "id": job_id,
                "kind": KIND,
                "status": "failed",
                "owner": owner,
                "payload": "{}",
                "created_at": now,
                "finished_at": now,
                "error": f"input file failed validation: {errors[0]['message']}",
                "total": 0,
                "model": ctx.model_ref(),
            }
            batch_row, out_row, err_row = rows_for(job_id)
            await run_in_threadpool(store.create_failed_batch, job_row, batch_row, out_row, err_row)
            return await view(batch_row)
        payload = {"input_file_id": frow["id"], "endpoint": body.endpoint, "lines": lines}
        scanned = _SCANNED.set((frow["id"], body.endpoint))  # scanned above: the kind need not scan it again
        try:
            job = await submit_batch_job(ctx, request, payload, options)
        finally:
            _SCANNED.reset(scanned)
        # The job is queued already and the runner may claim it, even finish it, before the batch row below
        # exists (submit pokes the runner, and the runner also polls). That is safe: the kind needs only its
        # payload and the input file, never the batch row, and every batch view starts from the row, so nobody
        # sees a batch without it. If the insert fails, request_cancel stops a queued / running job and
        # refunds what it did not process; a job that already finished stays a plain `batch` job (GET
        # /v1/jobs, deletable) with no batch view.
        batch_row, out_row, err_row = rows_for(job["id"])
        try:
            await run_in_threadpool(store.create_batch, batch_row, out_row, err_row)
        except BaseException:
            with contextlib.suppress(Exception):
                await ctx.extra["jobs"].request_cancel(job)
            raise
        return await view(batch_row)

    @api.get("/batches", response_model=BatchList, summary="List your batches (newest first)")
    async def list_batches(
        request: Request, after: str | None = None, limit: int = Query(20, ge=1, le=100)
    ) -> dict[str, Any]:
        page = await run_in_threadpool(store.list_batches, caller(request), scoped(request), after, limit)
        if page is None:
            raise SchemaError(f"after: no such batch {after!r}", "after")
        rows, more = page
        views = [await view(b) for b in rows]
        return _list(views, more, lambda v: v)

    @api.get("/batches/{batch_id}", response_model=BatchObject, summary="Retrieve a batch")
    async def get_batch(batch_id: str, request: Request) -> dict[str, Any]:
        return await view(await visible_batch(request, batch_id))

    @api.post("/batches/{batch_id}/cancel", response_model=BatchObject, summary="Cancel a batch")
    async def cancel_batch(batch_id: str, request: Request) -> dict[str, Any]:
        b = await visible_batch(request, batch_id)
        job = await run_in_threadpool(jobs.get, b["job_id"])
        if job is None:
            raise not_found("batch", batch_id)
        if _finished(job):
            state = (await view(b, job))["status"]
            raise ApiError(409, f"batch is already {state}: only an unfinished batch can be cancelled")
        if not job["cancel_requested"]:
            await run_in_threadpool(store.mark_cancelling, batch_id)
            job = await ctx.extra["jobs"].request_cancel(job)
            await audit_action(request, "batch.cancel", batch_id, {"job": job["id"], "owner": b["owner"]})
        return await view(await run_in_threadpool(store.batch, batch_id) or b, job)

    return api


def _list(rows: list[dict[str, Any]], more: bool, fn: Any) -> dict[str, Any]:
    data = [fn(r) for r in rows]
    return {
        "object": "list",
        "data": data,
        "first_id": data[0]["id"] if data else None,
        "last_id": data[-1]["id"] if data else None,
        "has_more": more,
    }


__all__ = ["BatchStore", "batch_kind", "router", "scan_input", "submit_batch_job"]
