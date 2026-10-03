"""Async job API: submit a large batch, poll (or get a webhook), page / stream the results.

Feature module (``main.FEATURES``): ``router(ctx)`` registers ``/v1/jobs*``, the background runner and the
webhook dispatcher. See docs/jobs.md for the contract.

* One runner, jobs FIFO, one at a time. A job feeds the engine in chunks of ``max_microbatch`` records and
  keeps ONE chunk in flight, so interactive requests queue behind at most one chunk.
* SQLite (``<state_dir>/jobs.db``) holds jobs and result rows. Rows are persisted per chunk, so a job
  interrupted by a restart is resumed from the last persisted row (kinds that do not declare themselves
  ``resumable`` restart from scratch).
* Kinds are pluggable: ``ctx.extra["job_kinds"][name] = {"validate": fn, "run": async_fn, "resumable": bool}``
  (``resumable`` optional).
"""

from __future__ import annotations

import asyncio
import contextlib
import csv
import io
import json
import logging
import re
import secrets
import sqlite3
import threading
import time
from collections import Counter
from collections.abc import AsyncIterator, Callable
from datetime import datetime, timezone
from typing import Any, Literal

from fastapi import APIRouter, Query, Request, Response
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator
from starlette.concurrency import run_in_threadpool

from .appctx import AppContext
from .classify import (
    Labels,
    check_label_count,
    check_labels_shape,
    check_levels_count,
    classify_result,
    classify_to_systemone,
    resolve_threshold,
    score_result,
    score_to_systemone,
)
from .engine import EngineNotReady, GpuOutOfMemory
from .paths import jobs_db
from .schemas import DEFAULT_MODEL, SystemOneRequest, check_limits, format_errors
from .webhooks import DEFAULT_EVENTS, WebhookDispatcher, WebhookPolicy, WebhookRefused

log = logging.getLogger("clef")

STATUSES = ("queued", "running", "succeeded", "failed", "cancelled", "interrupted")
ACTIVE = ("queued", "running", "interrupted")
FINISHED = ("succeeded", "failed", "cancelled")
KIND_RE = re.compile(r"^[a-z][a-z0-9_.-]{0,63}$")
MAX_METADATA_BYTES = 16 * 1024
PAGE_MAX = 1000
STREAM_BATCH = 500
PURGE_INTERVAL_S = 600.0
OOM_PAUSE_S = 0.5


class JobsFull(RuntimeError):
    """max_jobs queued/running jobs reached (-> HTTP 409)."""


class JobAborted(Exception):
    """Systemic failure inside a kind: the job fails with this message (shown to the client as is)."""


class JobCancelled(Exception):
    """Raised inside a kind's runner to stop early (equivalent to returning after ``job.cancelled``)."""


def _iso(ts: float | None) -> str | None:
    if ts is None:
        return None
    return datetime.fromtimestamp(ts, timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


# ------------------------------------------------------------------ storage

_SCHEMA = """
CREATE TABLE IF NOT EXISTS jobs (
    seq INTEGER PRIMARY KEY AUTOINCREMENT,
    id TEXT NOT NULL UNIQUE,
    kind TEXT NOT NULL,
    status TEXT NOT NULL,
    owner TEXT,
    payload TEXT NOT NULL,
    metadata TEXT,
    webhook TEXT,
    deliveries TEXT NOT NULL DEFAULT '{}',
    total INTEGER,
    done INTEGER NOT NULL DEFAULT 0,
    failed INTEGER NOT NULL DEFAULT 0,
    created_at REAL NOT NULL,
    started_at REAL,
    finished_at REAL,
    run_started_at REAL,
    run_base_done INTEGER NOT NULL DEFAULT 0,
    resumes INTEGER NOT NULL DEFAULT 0,
    cancel_requested INTEGER NOT NULL DEFAULT 0,
    error TEXT,
    result TEXT
);
CREATE INDEX IF NOT EXISTS jobs_status ON jobs(status, seq);
CREATE TABLE IF NOT EXISTS items (
    job_id TEXT NOT NULL REFERENCES jobs(id) ON DELETE CASCADE,
    idx INTEGER NOT NULL,
    data TEXT NOT NULL,
    PRIMARY KEY (job_id, idx)
) WITHOUT ROWID;
"""


_COLS = (
    "seq, id, kind, status, owner, metadata, webhook, deliveries, total, done, failed, created_at,"
    " started_at, finished_at, run_started_at, run_base_done, resumes, cancel_requested, error, result"
)  # everything but `payload` (can be tens of MB)


def _loads(raw: str | None) -> Any:
    return None if raw is None else json.loads(raw)


class JobStore:
    """Thread-safe sqlite wrapper (one connection, one lock; every call is short)."""

    def __init__(self, path: Any):
        self._lock = threading.RLock()
        self._db = sqlite3.connect(str(path), check_same_thread=False, isolation_level=None, timeout=30)
        self._db.row_factory = sqlite3.Row
        with contextlib.suppress(sqlite3.Error):
            self._db.execute("PRAGMA journal_mode=WAL")
        self._db.execute("PRAGMA synchronous=NORMAL")
        self._db.execute("PRAGMA foreign_keys=ON")
        self._db.executescript(_SCHEMA)

    def close(self) -> None:
        with self._lock:
            self._db.close()

    def _row(self, row: sqlite3.Row | None) -> dict[str, Any] | None:
        if row is None:
            return None
        d = dict(row)
        for k in ("metadata", "webhook", "deliveries", "result"):
            d[k] = _loads(d[k])
        d["cancel_requested"] = bool(d["cancel_requested"])
        return d

    # ---- jobs
    def create(
        self,
        kind: str,
        owner: str | None,
        payload: dict[str, Any],
        metadata: dict[str, Any] | None,
        webhook: dict[str, Any] | None,
        max_active: int,
    ) -> dict[str, Any]:
        job_id = "job_" + secrets.token_hex(10)
        with self._lock:
            (n,) = self._db.execute(
                "SELECT COUNT(*) FROM jobs WHERE status IN ('queued','running','interrupted')"
            ).fetchone()
            if max_active > 0 and n >= max_active:
                raise JobsFull(f"too many queued jobs (max {max_active}); retry later or cancel some")
            self._db.execute(
                "INSERT INTO jobs (id, kind, status, owner, payload, metadata, webhook, created_at)"
                " VALUES (?, ?, 'queued', ?, ?, ?, ?, ?)",
                (
                    job_id,
                    kind,
                    owner,
                    json.dumps(payload, separators=(",", ":")),
                    None if metadata is None else json.dumps(metadata),
                    None if webhook is None else json.dumps(webhook),
                    time.time(),
                ),
            )
            return self.get(job_id) or {}

    def get(self, job_id: str) -> dict[str, Any] | None:
        with self._lock:
            return self._row(self._db.execute(f"SELECT {_COLS} FROM jobs WHERE id=?", (job_id,)).fetchone())

    def list(
        self, owner: str | None, scoped: bool, status: str | None, kind: str | None, limit: int, offset: int
    ) -> tuple[list[dict[str, Any]], int]:
        where, args = ["1=1"], []
        if scoped:
            where.append("(owner IS NULL OR owner=?)")
            args.append(owner)
        if status:
            where.append("status=?")
            args.append(status)
        if kind:
            where.append("kind=?")
            args.append(kind)
        clause = " AND ".join(where)
        with self._lock:
            (total,) = self._db.execute(f"SELECT COUNT(*) FROM jobs WHERE {clause}", args).fetchone()
            rows = self._db.execute(
                f"SELECT {_COLS} FROM jobs WHERE {clause} ORDER BY seq DESC LIMIT ? OFFSET ?",
                [*args, limit, offset],
            ).fetchall()
        return [self._row(r) for r in rows], total  # type: ignore[misc]

    def next_runnable(self) -> dict[str, Any] | None:
        with self._lock:
            return self._row(
                self._db.execute(
                    f"SELECT {_COLS} FROM jobs WHERE status IN ('queued','interrupted')"
                    " ORDER BY (status='interrupted') DESC, seq LIMIT 1"
                ).fetchone()
            )

    def payload(self, job_id: str) -> str:
        with self._lock:
            row = self._db.execute("SELECT payload FROM jobs WHERE id=?", (job_id,)).fetchone()
        return row["payload"] if row else "{}"

    def mark_running(self, job_id: str, resume_done: int) -> bool:
        """queued/interrupted -> running. False when a cancel got there first."""
        now = time.time()
        with self._lock:
            cur = self._db.execute(
                "UPDATE jobs SET status='running', started_at=COALESCE(started_at, ?), run_started_at=?,"
                " run_base_done=?, finished_at=NULL WHERE id=? AND status IN ('queued','interrupted')",
                (now, now, resume_done, job_id),
            )
            return cur.rowcount == 1

    def set_total(self, job_id: str, total: int) -> None:
        with self._lock:
            self._db.execute("UPDATE jobs SET total=? WHERE id=?", (int(total), job_id))

    def add_items(self, job_id: str, items: list[dict[str, Any]]) -> int:
        """Append rows after the last persisted one (one transaction); returns the new ``done``."""
        with self._lock:
            self._db.execute("BEGIN IMMEDIATE")
            try:
                row = self._db.execute("SELECT done FROM jobs WHERE id=?", (job_id,)).fetchone()
                if row is None:
                    raise KeyError(job_id)
                start = row["done"]
                self._db.executemany(
                    "INSERT INTO items (job_id, idx, data) VALUES (?, ?, ?)",
                    [
                        (job_id, start + i, json.dumps(it, separators=(",", ":"), default=str))
                        for i, it in enumerate(items)
                    ],
                )
                failed = sum(1 for it in items if isinstance(it, dict) and "error" in it)
                self._db.execute(
                    "UPDATE jobs SET done=done+?, failed=failed+? WHERE id=?", (len(items), failed, job_id)
                )
                self._db.execute("COMMIT")
            except BaseException:
                self._db.execute("ROLLBACK")
                raise
            return start + len(items)

    def finish(self, job_id: str, status: str, error: str | None = None, result: Any = None) -> bool:
        with self._lock:
            cur = self._db.execute(
                "UPDATE jobs SET status=?, finished_at=?, error=?, result=?, total=COALESCE(total, done)"
                " WHERE id=? AND status IN ('queued','running','interrupted')",
                (
                    status,
                    time.time(),
                    error,
                    None if result is None else json.dumps(result, default=str),
                    job_id,
                ),
            )
            return cur.rowcount == 1

    def cancel_queued(self, job_id: str) -> bool:
        with self._lock:
            cur = self._db.execute(
                "UPDATE jobs SET status='cancelled', finished_at=?, total=COALESCE(total, done)"
                " WHERE id=? AND status IN ('queued','interrupted')",
                (time.time(), job_id),
            )
            return cur.rowcount == 1

    def flag_cancel(self, job_id: str) -> None:
        with self._lock:
            self._db.execute("UPDATE jobs SET cancel_requested=1 WHERE id=? AND status='running'", (job_id,))

    def interrupt(self, job_id: str) -> None:
        with self._lock:
            self._db.execute(
                "UPDATE jobs SET status='interrupted', resumes=resumes+1 WHERE id=? AND status='running'",
                (job_id,),
            )

    def recover(self) -> int:
        """Startup: whatever was running when the process died is interrupted (and will be resumed)."""
        with self._lock:
            cur = self._db.execute(
                "UPDATE jobs SET status='interrupted', resumes=resumes+1, cancel_requested=0"
                " WHERE status='running'"
            )
            return cur.rowcount

    def reset_items(self, job_id: str) -> None:
        with self._lock:
            self._db.execute("DELETE FROM items WHERE job_id=?", (job_id,))
            self._db.execute("UPDATE jobs SET done=0, failed=0, total=NULL WHERE id=?", (job_id,))

    def delete(self, job_id: str) -> bool:
        with self._lock:
            cur = self._db.execute(
                "DELETE FROM jobs WHERE id=? AND status IN ('succeeded','failed','cancelled')", (job_id,)
            )
            return cur.rowcount == 1

    def purge(self, older_than: float) -> int:
        with self._lock:
            cur = self._db.execute(
                "DELETE FROM jobs WHERE status IN ('succeeded','failed','cancelled') AND finished_at < ?",
                (older_than,),
            )
            return cur.rowcount

    # ---- results
    def items(self, job_id: str, offset: int, limit: int) -> list[tuple[int, str]]:
        with self._lock:
            rows = self._db.execute(
                "SELECT idx, data FROM items WHERE job_id=? AND idx>=? ORDER BY idx LIMIT ?",
                (job_id, offset, limit),
            ).fetchall()
        return [(r["idx"], r["data"]) for r in rows]

    # ---- webhook deliveries
    def set_delivery(self, job_id: str, event: str, record: dict[str, Any]) -> None:
        with self._lock:
            row = self._db.execute("SELECT deliveries FROM jobs WHERE id=?", (job_id,)).fetchone()
            if row is None:
                return
            deliveries = json.loads(row["deliveries"] or "{}")
            deliveries[event] = record
            self._db.execute("UPDATE jobs SET deliveries=? WHERE id=?", (json.dumps(deliveries), job_id))

    def pending_deliveries(self) -> list[tuple[str, str]]:
        with self._lock:
            rows = self._db.execute(
                "SELECT id, deliveries FROM jobs WHERE webhook IS NOT NULL AND deliveries != '{}'"
            ).fetchall()
        out = []
        for r in rows:
            for event, rec in json.loads(r["deliveries"]).items():
                if rec.get("status") == "pending" and event != "job.progress":
                    out.append((r["id"], event))
        return out


# ------------------------------------------------------------------ public views


def public_job(row: dict[str, Any], now: float | None = None) -> dict[str, Any]:
    """The JSON a client sees (never the payload, owner or webhook secret)."""
    now = time.time() if now is None else now
    done, total, status = row["done"], row["total"], row["status"]
    eta = rate = None
    if status == "running" and row["run_started_at"]:
        elapsed = now - row["run_started_at"]
        fresh = done - row["run_base_done"]
        if fresh > 0 and elapsed > 0:
            rate = fresh / elapsed
            if total is not None:
                eta = max(0.0, (total - done) / rate)
    end = row["finished_at"] or now
    hook = row["webhook"]
    out: dict[str, Any] = {
        "id": row["id"],
        "kind": row["kind"],
        "status": status,
        "progress": {
            "done": done,
            "total": total,
            "failed": row["failed"],
            "percent": None if not total else round(100.0 * done / total, 1),
        },
        "eta_s": None if eta is None else round(eta, 1),
        "items_per_s": None if rate is None else round(rate, 3),
        "created_at": _iso(row["created_at"]),
        "started_at": _iso(row["started_at"]),
        "finished_at": _iso(row["finished_at"]),
        "queued_s": round((row["started_at"] or now) - row["created_at"], 3),
        "duration_s": None if not row["started_at"] else round(end - row["started_at"], 3),
        "cancel_requested": row["cancel_requested"],
        "resumes": row["resumes"],
        "error": row["error"],
        "result": row["result"],
        "metadata": row["metadata"],
        "webhook": None
        if not hook
        else {
            "url": hook["url"],
            "events": hook.get("events") or list(DEFAULT_EVENTS),
            "has_secret": bool(hook.get("secret")),
            "deliveries": row["deliveries"] or {},
        },
    }
    return out


# ------------------------------------------------------------------ request / response models


class _Base(BaseModel):
    model_config = ConfigDict(extra="forbid")


class WebhookSpec(_Base):
    url: str = Field(max_length=2048, description="http(s) URL; host must be in CLEF_WEBHOOK_ALLOW")
    secret: str | None = Field(None, max_length=256, description="signs deliveries (HMAC-SHA256)")
    events: list[Literal["job.succeeded", "job.failed", "job.cancelled", "job.progress"]] | None = Field(
        None, description="default: succeeded, failed, cancelled"
    )

    @field_validator("events")
    @classmethod
    def _events(cls, v: list[str] | None) -> list[str] | None:
        if v is not None and not v:
            raise ValueError("events must not be empty (omit it for the defaults)")
        return None if v is None else list(dict.fromkeys(v))


class JobCreate(_Base):
    kind: str = Field(description="classify | score | systemone | any registered kind")
    payload: dict[str, Any] = Field(description="kind-specific; see docs/jobs.md")
    webhook: WebhookSpec | None = None
    metadata: dict[str, Any] | None = Field(None, description="free-form JSON echoed back (<= 16 KB)")


class JobProgress(BaseModel):
    done: int
    total: int | None
    failed: int
    percent: float | None


class JobInfo(BaseModel):
    model_config = ConfigDict(extra="allow")
    id: str
    kind: str
    status: Literal["queued", "running", "succeeded", "failed", "cancelled", "interrupted"]
    progress: JobProgress
    eta_s: float | None = None
    items_per_s: float | None = None
    created_at: str
    started_at: str | None = None
    finished_at: str | None = None
    queued_s: float | None = None
    duration_s: float | None = None
    cancel_requested: bool = False
    resumes: int = 0
    error: str | None = None
    result: Any = None
    metadata: dict[str, Any] | None = None
    webhook: dict[str, Any] | None = None


class JobList(BaseModel):
    jobs: list[JobInfo]
    total: int
    limit: int
    offset: int


class JobResultsPage(BaseModel):
    id: str
    kind: str
    status: str
    offset: int
    count: int
    total: int | None
    next_offset: int | None = Field(None, description="null once the job is finished and nothing is left")
    items: list[dict[str, Any]]


class JobDeleted(BaseModel):
    deleted: str


# ------------------------------------------------------------------ job handle given to kinds


class JobHandle:
    """Handle given to ``run_fn``; ``set_total``, ``add_items`` and ``cancelled`` are the contract."""

    def __init__(self, runner: JobRunner, row: dict[str, Any]):
        self._runner = runner
        self.id: str = row["id"]
        self.kind: str = row["kind"]
        self.metadata: dict[str, Any] | None = row["metadata"]
        self.done: int = row["done"]  # rows already persisted (non-zero when resuming a resumable kind)
        self.resumed: bool = row["done"] > 0

    @property
    def resume_from(self) -> int:
        return self.done

    @property
    def cancelled(self) -> bool:
        return self.id in self._runner.cancel_ids

    def set_total(self, n: int) -> None:
        self._runner.store.set_total(self.id, int(n))

    async def add_items(self, items: list[dict[str, Any]]) -> None:
        """Persist rows in order; progress advances by ``len(items)``; a row with "error" counts as failed."""
        if not items:
            return
        self.done = await asyncio.to_thread(self._runner.store.add_items, self.id, list(items))
        self._runner.notify_progress(self.id)

    async def stored_items(self) -> list[dict[str, Any]]:
        """Every row persisted so far (used to rebuild summaries when resuming)."""
        out: list[dict[str, Any]] = []
        offset = 0
        while True:
            page = await asyncio.to_thread(self._runner.store.items, self.id, offset, STREAM_BATCH)
            if not page:
                return out
            out.extend(json.loads(d) for _, d in page)
            offset = page[-1][0] + 1


# ------------------------------------------------------------------ runner


class JobRunner:
    def __init__(self, ctx: AppContext, store: JobStore, dispatcher: WebhookDispatcher):
        self.ctx, self.store, self.dispatcher = ctx, store, dispatcher
        self.cancel_ids: set[str] = set()
        self.wake: asyncio.Event | None = None  # created on start (needs the running loop)
        self._wants_progress = False
        self._task: asyncio.Task[None] | None = None
        self.current: str | None = None

    def kinds(self) -> dict[str, Any]:
        return self.ctx.extra.setdefault("job_kinds", {})

    # ---- lifecycle
    async def start(self) -> None:
        self.wake = asyncio.Event()
        n = await asyncio.to_thread(self.store.recover)
        if n:
            log.info("jobs: %d job(s) interrupted by the last shutdown will be resumed", n)
        await asyncio.to_thread(self._purge)
        for job_id, event in await asyncio.to_thread(self.store.pending_deliveries):
            self.dispatcher.enqueue(job_id, event)
        self._task = asyncio.create_task(self._loop(), name="clef-jobs")

    async def stop(self) -> None:
        if self._task:
            self._task.cancel()
            await asyncio.gather(self._task, return_exceptions=True)
        await self.dispatcher.close()
        self.store.close()

    def poke(self) -> None:
        if self.wake is not None:
            self.wake.set()

    def _purge(self) -> None:
        ttl = self.ctx.cfg.job_ttl_hours
        if ttl > 0:
            n = self.store.purge(time.time() - ttl * 3600)
            if n:
                log.info("jobs: purged %d finished job(s) older than %g h", n, ttl)

    async def _loop(self) -> None:
        assert self.wake is not None
        next_purge = time.monotonic() + PURGE_INTERVAL_S
        while True:
            try:
                row = await asyncio.to_thread(self.store.next_runnable)
                if row is None:
                    with contextlib.suppress(asyncio.TimeoutError):
                        await asyncio.wait_for(self.wake.wait(), timeout=PURGE_INTERVAL_S)
                    self.wake.clear()
                    if time.monotonic() >= next_purge:
                        next_purge = time.monotonic() + PURGE_INTERVAL_S
                        await asyncio.to_thread(self._purge)
                    continue
                await self._execute(row)
            except asyncio.CancelledError:
                raise
            except Exception:
                log.exception("job runner error")
                await asyncio.sleep(1.0)

    # ---- cancel / webhooks
    async def request_cancel(self, row: dict[str, Any]) -> dict[str, Any]:
        job_id = row["id"]
        if await asyncio.to_thread(self.store.cancel_queued, job_id):
            await self._emit(job_id, "cancelled")
        else:
            current = await asyncio.to_thread(self.store.get, job_id)
            if current and current["status"] == "running":
                self.cancel_ids.add(job_id)
                await asyncio.to_thread(self.store.flag_cancel, job_id)
        return await asyncio.to_thread(self.store.get, job_id) or row

    def _record_event(self, job_id: str, status: str) -> str | None:
        """Mark the terminal webhook pending (if subscribed) so a restart retries it; returns the event."""
        row = self.store.get(job_id)
        hook = row and row["webhook"]
        event = f"job.{status}"
        if not hook or event not in (hook.get("events") or DEFAULT_EVENTS):
            return None
        self.store.set_delivery(job_id, event, {"event": event, "status": "pending", "attempts": 0})
        return event

    async def _emit(self, job_id: str, status: str) -> None:
        event = await asyncio.to_thread(self._record_event, job_id, status)
        if event:
            self.dispatcher.enqueue(job_id, event)

    def notify_progress(self, job_id: str) -> None:
        if self._wants_progress:
            self.dispatcher.progress(job_id, True)

    # ---- execution
    async def _execute(self, row: dict[str, Any]) -> None:
        job_id = row["id"]
        spec = self.kinds().get(row["kind"])
        if spec is None:
            await self._finish(job_id, "failed", error=f"unknown job kind {row['kind']!r}")
            return
        resuming = row["status"] == "interrupted"
        done = row["done"] if resuming and spec.get("resumable") else 0
        if resuming and not spec.get("resumable"):
            await asyncio.to_thread(self.store.reset_items, job_id)
        try:
            payload = json.loads(await asyncio.to_thread(self.store.payload, job_id))
            parsed = await asyncio.to_thread(spec["validate"], payload)
        except (ValueError, ValidationError) as exc:
            await self._finish(job_id, "failed", error=f"payload is no longer valid: {_msg(exc)}")
            return
        if not await asyncio.to_thread(self.store.mark_running, job_id, done):
            return  # cancelled while queued
        self.cancel_ids.discard(job_id)
        self.current = job_id
        self._wants_progress = "job.progress" in ((row["webhook"] or {}).get("events") or ())
        handle = JobHandle(self, {**row, "done": done})
        t0 = time.monotonic()
        try:
            result = await spec["run"](self.ctx, parsed, handle)
        except asyncio.CancelledError:
            self.store.interrupt(job_id)  # shutdown: resumed on the next start
            raise
        except JobCancelled:
            result = None
        except Exception as exc:
            detail = str(exc) if isinstance(exc, JobAborted) else self.ctx.map_exception(exc).detail
            await self._finish(job_id, "failed", error=detail)
            return
        finally:
            self.current = None
        cancelled = job_id in self.cancel_ids
        self.cancel_ids.discard(job_id)
        await self._finish(job_id, "cancelled" if cancelled else "succeeded", result=result)
        log.info(
            "job %s (%s) %s: %d item(s) in %.1fs",
            job_id,
            row["kind"],
            "cancelled" if cancelled else "done",
            handle.done,
            time.monotonic() - t0,
        )

    async def _finish(self, job_id: str, status: str, error: str | None = None, result: Any = None) -> None:
        if await asyncio.to_thread(self.store.finish, job_id, status, error, result):
            await self._emit(job_id, status)


def _msg(exc: Exception) -> str:
    if isinstance(exc, ValidationError):
        return format_errors(list(exc.errors()))
    return str(exc)


# ------------------------------------------------------------------ chunked execution (built-in kinds)


class ItemError:
    def __init__(self, message: str):
        self.message = message[:500]


async def decide_resilient(ctx: AppContext, job: JobHandle, reqs: list[SystemOneRequest]) -> list[Any]:
    """Results for ``reqs`` (one engine call when everything is fine).

    A record the engine / limits reject (ValueError, e.g. too long) or that hits a GPU OOM is isolated by
    bisecting the chunk and becomes an ``ItemError``; the rest of the chunk is unaffected. Engine not ready
    waits with backoff (up to CLEF_JOB_ENGINE_WAIT_S). Any other exception is systemic and propagates.
    """
    wait_s = float(ctx.cfg.job_engine_wait_s)

    async def call(part: list[SystemOneRequest]) -> list[Any]:
        delay, waited = 0.5, 0.0
        while True:
            if job.cancelled:
                raise JobCancelled
            try:
                return await ctx.decide(part, batch=len(part) > 1, label="items")
            except EngineNotReady as exc:
                if waited >= wait_s:
                    raise JobAborted(f"engine not available for {wait_s:g}s: {exc}") from exc
                await asyncio.sleep(delay)
                waited += delay
                delay = min(delay * 2, 15.0)

    async def go(part: list[SystemOneRequest]) -> list[Any]:
        try:
            return await call(part)
        except (ValueError, GpuOutOfMemory) as exc:
            if len(part) == 1:
                msg = "GPU out of memory for this item" if isinstance(exc, GpuOutOfMemory) else str(exc)
                return [ItemError(msg or type(exc).__name__)]
            if isinstance(exc, GpuOutOfMemory):
                await asyncio.sleep(OOM_PAUSE_S)
            mid = len(part) // 2
            return [*await go(part[:mid]), *await go(part[mid:])]

    return await go(reqs)


async def run_chunked(
    ctx: AppContext,
    job: JobHandle,
    n: int,
    build: Callable[[int], SystemOneRequest],
    shape: Callable[[int, dict[str, Any]], dict[str, Any]],
    tally: Callable[[dict[str, Any]], None],
) -> None:
    """Feed items ``job.resume_from .. n`` through the engine, ONE chunk in flight, persisting after each."""
    job.set_total(n)
    size = max(1, int(ctx.cfg.max_microbatch))
    i = job.resume_from
    while i < n:
        if job.cancelled:
            return
        j = min(n, i + size)
        built: list[tuple[int, SystemOneRequest | ItemError]] = []
        for k in range(i, j):
            try:
                built.append((k, build(k)))
            except (ValueError, ValidationError) as exc:
                built.append((k, ItemError(_msg(exc))))
        good = [r for _, r in built if not isinstance(r, ItemError)]
        results = iter(await decide_resilient(ctx, job, good) if good else [])
        rows = []
        for k, r in built:
            out = r if isinstance(r, ItemError) else next(results)
            row = {"index": k, "error": out.message} if isinstance(out, ItemError) else shape(k, out)
            tally(row)
            rows.append(row)
        await job.add_items(rows)
        i = j
        await asyncio.sleep(0)  # let interactive requests reach the engine queue before the next chunk


# ------------------------------------------------------------------ built-in kinds


class _Items(_Base):
    inputs: list[Any] = Field(description="text or any JSON per item")
    instructions: str | None = None
    model: str = DEFAULT_MODEL

    @field_validator("inputs")
    @classmethod
    def _inputs(cls, v: list[Any]) -> list[Any]:
        if not v:
            raise ValueError("inputs must contain at least one item")
        return v


class ClassifyJob(_Items):
    labels: Labels | None = None
    classifier: str | None = Field(
        None, description="name of a saved classify classifier (instead of labels)"
    )
    multi_label: bool = False
    threshold: float | None = Field(None, ge=0.0, le=1.0)

    @field_validator("labels")
    @classmethod
    def _labels(cls, v: Any) -> Any:
        return v if v is None else check_labels_shape(v)


class ScoreJob(_Items):
    levels: list[str] | None = None
    classifier: str | None = Field(None, description="name of a saved score classifier (instead of levels)")


class SystemOneJob(_Base):
    requests: list[SystemOneRequest] = Field(description="SystemOne requests (media allowed)")

    @field_validator("requests")
    @classmethod
    def _requests(cls, v: list[SystemOneRequest]) -> list[SystemOneRequest]:
        if not v:
            raise ValueError("requests must contain at least one item")
        return v


class _Parsed:
    """Output of a built-in validator: everything the runner needs, classifier already resolved."""

    def __init__(self, **kw: Any):
        self.__dict__.update(kw)


def _max_items(ctx: AppContext, n: int, field_name: str) -> None:
    if n > ctx.cfg.max_job_items:
        raise ValueError(f"{field_name}: too many items (max {ctx.cfg.max_job_items}, CLEF_MAX_JOB_ITEMS)")


def _resolve_classifier(ctx: AppContext, name: str, kind: str) -> dict[str, Any]:
    from .classifiers import InvalidName, validate_name

    try:
        validate_name(name)
    except InvalidName as exc:
        raise ValueError(f"classifier: {exc}") from exc
    doc = ctx.store.get(name)
    if doc is None:
        raise ValueError(f"classifier: {name!r} not found")
    if doc["kind"] != kind:
        other = "score" if kind == "classify" else "classify"
        raise ValueError(f"classifier: {name!r} is a {doc['kind']} classifier; use kind={other!r}")
    return doc


def _tally_labels(counter: Counter[str], key: str) -> Callable[[dict[str, Any]], None]:
    def tally(row: dict[str, Any]) -> None:
        if "error" in row:
            counter["__errors__"] += 1
        elif key == "labels":
            counter.update(row.get("labels") or [])
        elif row.get(key) is not None:
            counter[row[key]] += 1

    return tally


def _summary(job: JobHandle, counter: Counter[str], key: str) -> dict[str, Any]:
    errors = counter.pop("__errors__", 0)
    return {"items": job.done, "ok": job.done - errors, "errors": errors, key: dict(counter.most_common())}


def builtin_kinds(ctx: AppContext) -> dict[str, dict[str, Any]]:
    cfg = ctx.cfg

    def validate_classify(payload: dict[str, Any]) -> _Parsed:
        try:
            body = ClassifyJob.model_validate(payload)
        except ValidationError as exc:
            raise ValueError(format_errors(list(exc.errors()))) from exc
        _max_items(ctx, len(body.inputs), "inputs")
        if (body.labels is None) == (body.classifier is None):
            raise ValueError("labels: give exactly one of labels or classifier")
        instructions, multi, thr = body.instructions, body.multi_label, body.threshold
        labels = body.labels
        if body.classifier is not None:
            doc = _resolve_classifier(ctx, body.classifier, "classify")
            labels = doc["labels"]
            instructions = instructions if instructions is not None else doc.get("instructions")
            multi = bool(doc.get("multi_label")) if "multi_label" not in payload else multi
            thr = thr if thr is not None else doc.get("threshold")
        assert labels is not None
        check_label_count(labels, multi, cfg.max_labels)
        if thr is not None and not multi:
            raise ValueError("threshold: only allowed with multi_label=true")
        threshold = resolve_threshold(thr, cfg) if multi else None
        try:
            classify_to_systemone(body.inputs[0], labels, instructions, multi, body.model)
        except ValidationError as exc:
            raise ValueError(format_errors(list(exc.errors()))) from exc
        return _Parsed(
            inputs=body.inputs, labels=labels, instructions=instructions, multi=multi, threshold=threshold,
            model=body.model,
        )  # fmt: skip

    async def run_classify(ctx_: AppContext, p: _Parsed, job: JobHandle) -> dict[str, Any]:
        counts: Counter[str] = Counter()
        key = "labels" if p.multi else "label"
        tally = _tally_labels(counts, key)
        if job.resumed:
            for row in await job.stored_items():
                tally(row)

        def shape(k: int, res: dict[str, Any]) -> dict[str, Any]:
            body = classify_result(res["answers"], p.labels, p.multi, p.threshold, p.model)
            body.pop("model"), body.pop("multi_label")
            return {"index": k, **body, "input_tokens": res.get("usage", {}).get("input_tokens", 0)}

        await run_chunked(
            ctx_, job, len(p.inputs),
            lambda k: classify_to_systemone(p.inputs[k], p.labels, p.instructions, p.multi, p.model),
            shape, tally,
        )  # fmt: skip
        return _summary(job, counts, "by_labels" if p.multi else "by_label")

    def validate_score(payload: dict[str, Any]) -> _Parsed:
        try:
            body = ScoreJob.model_validate(payload)
        except ValidationError as exc:
            raise ValueError(format_errors(list(exc.errors()))) from exc
        _max_items(ctx, len(body.inputs), "inputs")
        if (body.levels is None) == (body.classifier is None):
            raise ValueError("levels: give exactly one of levels or classifier")
        levels, instructions = body.levels, body.instructions
        if body.classifier is not None:
            doc = _resolve_classifier(ctx, body.classifier, "score")
            levels = doc["levels"]
            instructions = instructions if instructions is not None else doc.get("instructions")
        assert levels is not None
        if len(set(levels)) != len(levels) or any(not isinstance(x, str) or not x.strip() for x in levels):
            raise ValueError("levels: must be unique, non-empty strings")
        check_levels_count(levels, cfg.max_labels)
        try:
            score_to_systemone(body.inputs[0], levels, instructions, body.model)
        except ValidationError as exc:
            raise ValueError(format_errors(list(exc.errors()))) from exc
        return _Parsed(inputs=body.inputs, levels=levels, instructions=instructions, model=body.model)

    async def run_score(ctx_: AppContext, p: _Parsed, job: JobHandle) -> dict[str, Any]:
        counts: Counter[str] = Counter()
        tally = _tally_labels(counts, "level")
        if job.resumed:
            for row in await job.stored_items():
                tally(row)

        def shape(k: int, res: dict[str, Any]) -> dict[str, Any]:
            body = score_result(res["answers"], p.levels, p.model)
            body.pop("model")
            return {"index": k, **body, "input_tokens": res.get("usage", {}).get("input_tokens", 0)}

        await run_chunked(
            ctx_, job, len(p.inputs),
            lambda k: score_to_systemone(p.inputs[k], p.levels, p.instructions, p.model),
            shape, tally,
        )  # fmt: skip
        return _summary(job, counts, "by_level")

    def validate_systemone(payload: dict[str, Any]) -> _Parsed:
        try:
            body = SystemOneJob.model_validate(payload)
        except ValidationError as exc:
            raise ValueError(format_errors(list(exc.errors()))) from exc
        _max_items(ctx, len(body.requests), "requests")
        for i, r in enumerate(body.requests):
            check_limits(r, cfg, f"requests[{i}].")
        return _Parsed(requests=body.requests)

    async def run_systemone(ctx_: AppContext, p: _Parsed, job: JobHandle) -> dict[str, Any]:
        errors = 0
        if job.resumed:
            errors = sum(1 for row in await job.stored_items() if "error" in row)

        def shape(k: int, res: dict[str, Any]) -> dict[str, Any]:
            return {
                "index": k,
                "model": res.get("model"),
                "answers": res["answers"],
                "input_tokens": res.get("usage", {}).get("input_tokens", 0),
            }

        def tally(row: dict[str, Any]) -> None:
            nonlocal errors
            errors += "error" in row

        await run_chunked(ctx_, job, len(p.requests), lambda k: p.requests[k], shape, tally)
        return {"items": job.done, "ok": job.done - errors, "errors": errors}

    return {
        "classify": {"validate": validate_classify, "run": run_classify, "resumable": True},
        "score": {"validate": validate_score, "run": run_score, "resumable": True},
        "systemone": {"validate": validate_systemone, "run": run_systemone, "resumable": True},
    }


# ------------------------------------------------------------------ CSV flattening


def flatten(value: Any, prefix: str = "") -> dict[str, str]:
    """Nested row -> flat ``a.b`` columns (lists of scalars joined with ``|``)."""
    out: dict[str, str] = {}
    if isinstance(value, dict):
        for k, v in value.items():
            out.update(flatten(v, f"{prefix}.{k}" if prefix else str(k)))
    elif isinstance(value, list) and all(not isinstance(x, (dict, list)) for x in value):
        out[prefix] = "|".join("" if x is None else str(x) for x in value)
    elif isinstance(value, (list,)):
        out[prefix] = json.dumps(value, ensure_ascii=False)
    else:
        out[prefix] = "" if value is None else str(value)
    return out


# ------------------------------------------------------------------ router


def router(ctx: AppContext) -> APIRouter:
    cfg, ApiError = ctx.cfg, ctx.api_error  # noqa: N806
    store = JobStore(jobs_db(cfg))
    policy = WebhookPolicy(cfg.webhook_allow)

    def hook_view(job_id: str) -> dict[str, Any] | None:
        row = store.get(job_id)
        return None if row is None else {"webhook": row["webhook"], "job": public_job(row)}

    dispatcher = WebhookDispatcher(
        policy, hook_view, store.set_delivery, timeout_s=cfg.webhook_timeout_s, attempts=cfg.webhook_attempts
    )
    runner = JobRunner(ctx, store, dispatcher)
    ctx.extra["jobs"] = runner
    kinds = ctx.extra.setdefault("job_kinds", {})
    for name, spec in builtin_kinds(ctx).items():
        kinds.setdefault(name, spec)

    ctx.on_startup.append(runner.start)
    ctx.on_shutdown.append(runner.stop)

    api = APIRouter(prefix="/v1/jobs", tags=["jobs"], dependencies=ctx.auth, responses=ctx.errors)

    def caller(request: Request) -> str | None:
        return getattr(request.state, "key_name", None)

    def scoped(request: Request) -> bool:
        return cfg.auth_required

    async def load(request: Request, job_id: str) -> dict[str, Any]:
        row = await run_in_threadpool(store.get, job_id)
        # Someone else's job is indistinguishable from a missing one.
        if row is None or (scoped(request) and row["owner"] not in (None, caller(request))):
            raise ApiError(404, f"job {job_id!r} not found")
        return row

    @api.post(
        "",
        status_code=202,
        response_model=JobInfo,
        summary="Submit an async job",
        dependencies=ctx.limited,
    )
    async def submit(body: JobCreate, request: Request, response: Response) -> dict[str, Any]:
        if not KIND_RE.match(body.kind) or body.kind not in kinds:
            raise ApiError(
                400, f"kind: unknown job kind {body.kind[:64]!r} (available: {', '.join(sorted(kinds))})"
            )
        if body.metadata is not None and len(json.dumps(body.metadata)) > MAX_METADATA_BYTES:
            raise ApiError(400, f"metadata: too large (max {MAX_METADATA_BYTES // 1024} KB)")
        hook = None
        if body.webhook is not None:
            try:
                policy.check_url(body.webhook.url)
            except WebhookRefused as exc:
                raise ApiError(400, f"webhook.url: {exc}") from exc
            hook = body.webhook.model_dump(exclude_none=True)
        try:
            await run_in_threadpool(kinds[body.kind]["validate"], body.payload)
        except (ValueError, ValidationError) as exc:
            raise ApiError(400, f"payload.{_msg(exc)}") from exc
        try:
            row = await run_in_threadpool(
                store.create, body.kind, caller(request), body.payload, body.metadata, hook, cfg.max_jobs
            )
        except JobsFull as exc:
            raise ApiError(409, str(exc), {"Retry-After": "30"}) from exc
        runner.poke()
        response.headers["Location"] = f"/v1/jobs/{row['id']}"
        return public_job(row)

    @api.get("", response_model=JobList, summary="List jobs (newest first)")
    async def list_jobs(
        request: Request,
        status: Literal["queued", "running", "succeeded", "failed", "cancelled", "interrupted"] | None = None,
        kind: str | None = None,
        limit: int = Query(50, ge=1, le=200),
        offset: int = Query(0, ge=0),
    ) -> dict[str, Any]:
        rows, total = await run_in_threadpool(
            store.list, caller(request), scoped(request), status, kind, limit, offset
        )
        now = time.time()
        return {"jobs": [public_job(r, now) for r in rows], "total": total, "limit": limit, "offset": offset}

    @api.get("/{job_id}", response_model=JobInfo, summary="Job status, progress, ETA and result summary")
    async def get_job(job_id: str, request: Request) -> dict[str, Any]:
        return public_job(await load(request, job_id))

    @api.post("/{job_id}/cancel", response_model=JobInfo, summary="Cancel a queued or running job")
    async def cancel_job(job_id: str, request: Request) -> dict[str, Any]:
        row = await load(request, job_id)
        if row["status"] in FINISHED:
            raise ApiError(409, f"job is already {row['status']}")
        return public_job(await runner.request_cancel(row))

    @api.delete("/{job_id}", response_model=JobDeleted, summary="Delete a finished job and its results")
    async def delete_job(job_id: str, request: Request) -> dict[str, str]:
        row = await load(request, job_id)
        if row["status"] in ACTIVE:
            raise ApiError(409, f"job is {row['status']}; cancel it first (POST /v1/jobs/{job_id}/cancel)")
        await run_in_threadpool(store.delete, job_id)
        return {"deleted": job_id}

    @api.get(
        "/{job_id}/results",
        response_model=JobResultsPage,
        responses={200: {"content": {"application/x-ndjson": {}, "text/csv": {}}}},
        summary="Result rows: JSON page, or a full ndjson / csv stream",
    )
    async def results(
        job_id: str,
        request: Request,
        offset: int = Query(0, ge=0),
        limit: int | None = Query(None, ge=1),
        fmt: Literal["json", "ndjson", "csv"] = Query("json", alias="format"),
    ) -> Any:
        row = await load(request, job_id)
        if fmt == "json":
            n = min(limit or 100, PAGE_MAX)
            page = await run_in_threadpool(store.items, job_id, offset, n)
            fresh = await run_in_threadpool(store.get, job_id) or row
            nxt = offset + len(page)
            more = nxt < fresh["done"] or fresh["status"] in ACTIVE
            return {
                "id": job_id,
                "kind": fresh["kind"],
                "status": fresh["status"],
                "offset": offset,
                "count": len(page),
                "total": fresh["total"],
                "next_offset": nxt if more else None,
                "items": [json.loads(d) for _, d in page],
            }
        headers = {"X-Clef-Job-Status": row["status"], "Cache-Control": "no-store"}
        if fmt == "ndjson":
            return StreamingResponse(
                _ndjson(store, job_id, offset, limit), media_type="application/x-ndjson", headers=headers
            )
        headers["Content-Disposition"] = f'attachment; filename="{job_id}.csv"'
        return StreamingResponse(
            _csv(store, job_id, offset, limit), media_type="text/csv; charset=utf-8", headers=headers
        )

    return api


async def _batches(
    store: JobStore, job_id: str, offset: int, limit: int | None
) -> AsyncIterator[list[tuple[int, str]]]:
    left = limit
    while left is None or left > 0:
        page = await asyncio.to_thread(
            store.items, job_id, offset, STREAM_BATCH if left is None else min(STREAM_BATCH, left)
        )
        if not page:
            return
        yield page
        offset = page[-1][0] + 1
        if left is not None:
            left -= len(page)


async def _ndjson(store: JobStore, job_id: str, offset: int, limit: int | None) -> AsyncIterator[str]:
    async for page in _batches(store, job_id, offset, limit):
        yield "".join(d + "\n" for _, d in page)


_FORMULA_START = ("=", "+", "-", "@", "\t", "\r")


def _csv_safe(row: dict[str, Any]) -> dict[str, Any]:
    """Neutralise spreadsheet formulas (CSV injection): text starting with = + - @ gets a leading quote."""
    return {k: f"'{v}" if isinstance(v, str) and v.startswith(_FORMULA_START) else v for k, v in row.items()}


async def _csv(store: JobStore, job_id: str, offset: int, limit: int | None) -> AsyncIterator[str]:
    header: dict[str, None] = {}
    async for page in _batches(store, job_id, offset, limit):  # pass 1: the union of columns
        for _, d in page:
            header.update(dict.fromkeys(flatten(json.loads(d))))
    columns = [c for c in header if c != "error"] + (["error"] if "error" in header else [])
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=columns, extrasaction="ignore", lineterminator="\r\n")
    writer.writeheader()
    yield buf.getvalue()
    async for page in _batches(store, job_id, offset, limit):  # pass 2: the rows
        buf.seek(0), buf.truncate()
        for _, d in page:
            writer.writerow(_csv_safe(flatten(json.loads(d))))
        yield buf.getvalue()


__all__ = ["JobCancelled", "JobHandle", "JobStore", "public_job", "router"]
