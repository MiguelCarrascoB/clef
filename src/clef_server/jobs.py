"""Async job API: submit a large batch, poll (or get a webhook), page / stream the results.

Feature module (``main.FEATURES``): ``router(ctx)`` registers ``/v1/jobs*``, the background runner and the
webhook dispatcher. See docs/jobs.md for the contract.

* One runner, jobs FIFO, one at a time. A job feeds the engine in chunks of ``max_microbatch`` records and
  keeps ONE chunk in flight, so interactive requests queue behind at most one chunk.
* The database (``ctx.db``: SQLite ``<state_dir>/clef.db`` or ``CLEF_DATABASE_URL``) holds jobs and result
  rows. Rows are persisted per chunk, so a job interrupted by a restart is resumed from the last persisted row
  (kinds that do not declare themselves ``resumable`` restart from scratch).
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
import time
import uuid
from collections import Counter
from collections.abc import AsyncIterator, Callable
from datetime import datetime, timezone
from typing import Any, Literal

from fastapi import APIRouter, Header, Query, Request, Response
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator
from sqlalchemy import and_, case, delete, func, insert, or_, select, update
from sqlalchemy.engine import Connection
from sqlalchemy.exc import IntegrityError
from starlette.concurrency import run_in_threadpool

from . import dbschema
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
from .db import Database
from .engine import EngineNotReady, GpuOutOfMemory
from .schemas import SystemOneRequest, check_limits, format_errors
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
RUNNER_RETRY_PAUSE_S = 1.0
RUNNER_MAX_FAILURES = 3  # consecutive runner-level failures on one job before it is failed (poison pill)
# Circuit breaker (JobHandle.add_items): a job whose rows keep failing is systemic, not bad data.
BREAKER_MIN_ITEMS = 16  # ... every one of the first N items of a run failed
BREAKER_SAME_MIN = 50  # ... or, from N items on, this share failed with one and the same message
BREAKER_SAME_SHARE = 0.9
IDEMPOTENCY_KEY_RE = re.compile(r"^[!-~]{1,128}$")


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

_JOBS = dbschema.jobs
_COLS = [c for c in _JOBS.c if c.name != "payload"]  # everything but `payload` (can be tens of MB)
_ACTIVE_SQL = ("queued", "running", "interrupted")
_RUNNABLE = ("queued", "interrupted")
_DONE_SQL = ("succeeded", "failed", "cancelled")


def _loads(raw: str | None) -> Any:
    return None if raw is None else json.loads(raw)


class JobStore:
    """Jobs and their result rows on the shared :class:`~clef_server.db.Database` (every call is short).

    Writes go through ``db.tx()``: one transaction each (BEGIN IMMEDIATE under the process lock on SQLite).
    Helpers that take ``conn`` run inside the caller's transaction.
    """

    def __init__(self, db: Database):
        self.db = db

    def close(self) -> None:
        self.db.close()

    def _row(self, row: Any) -> dict[str, Any] | None:
        if row is None:
            return None
        d = dict(row._mapping)
        for k in ("metadata", "webhook", "deliveries", "result"):
            d[k] = _loads(d[k])
        d["cancel_requested"] = bool(d["cancel_requested"])
        return d

    def _get(self, conn: Connection, job_id: str) -> dict[str, Any] | None:
        return self._row(conn.execute(select(*_COLS).where(_JOBS.c.id == job_id)).first())

    # ---- jobs
    def create(
        self,
        kind: str,
        owner: str | None,
        payload: dict[str, Any],
        metadata: dict[str, Any] | None,
        webhook: dict[str, Any] | None,
        max_active: int,
        idem_key: str | None = None,
        model: str | None = None,
    ) -> dict[str, Any]:
        """Insert a queued job. With ``idem_key``, a job this owner already submitted under it is returned
        instead (``row["replayed"]`` is True). ``model`` = the model ref served at submit (``jobs.model``)."""
        job_id = "job_" + secrets.token_hex(10)
        try:
            with self.db.tx() as conn:
                if idem_key:
                    found = self._find_idem(conn, owner, idem_key)
                    if found:
                        return {**found, "replayed": True}
                self.db.serialize(conn, "jobs.create")  # count-then-insert (max_active) on Postgres
                n = conn.execute(
                    select(func.count()).select_from(_JOBS).where(_JOBS.c.status.in_(_ACTIVE_SQL))
                ).scalar_one()
                if max_active > 0 and n >= max_active:
                    raise JobsFull(f"too many queued jobs (max {max_active}); retry later or cancel some")
                conn.execute(
                    insert(_JOBS).values(
                        id=job_id,
                        kind=kind,
                        status="queued",
                        owner=owner,
                        payload=json.dumps(payload, separators=(",", ":")),
                        metadata=None if metadata is None else json.dumps(metadata),
                        webhook=None if webhook is None else json.dumps(webhook),
                        created_at=time.time(),
                        idem_key=idem_key or None,
                        model=model,
                    )
                )
                return self._get(conn, job_id) or {}
        except IntegrityError:
            # Postgres: a concurrent submit with the same Idempotency-Key won the unique index.
            found = self.find_idem(owner, idem_key) if idem_key else None
            if found:
                return {**found, "replayed": True}
            raise

    def _find_idem(self, conn: Connection, owner: str | None, idem_key: str) -> dict[str, Any] | None:
        return self._row(
            conn.execute(
                select(*_COLS).where(
                    func.coalesce(_JOBS.c.owner, "") == (owner or ""), _JOBS.c.idem_key == idem_key
                )
            ).first()
        )

    def find_idem(self, owner: str | None, idem_key: str) -> dict[str, Any] | None:
        with self.db.read() as conn:
            return self._find_idem(conn, owner, idem_key)

    def get(self, job_id: str) -> dict[str, Any] | None:
        with self.db.read() as conn:
            return self._get(conn, job_id)

    def list(
        self, owner: str | None, scoped: bool, status: str | None, kind: str | None, limit: int, offset: int
    ) -> tuple[list[dict[str, Any]], int]:
        where = []
        if scoped:
            where.append(or_(_JOBS.c.owner.is_(None), _JOBS.c.owner == owner))
        if status:
            where.append(_JOBS.c.status == status)
        if kind:
            where.append(_JOBS.c.kind == kind)
        with self.db.read() as conn:
            total = conn.execute(select(func.count()).select_from(_JOBS).where(*where)).scalar_one()
            rows = conn.execute(
                select(*_COLS).where(*where).order_by(_JOBS.c.seq.desc()).limit(limit).offset(offset)
            ).all()
        return [self._row(r) for r in rows], total  # type: ignore[misc]

    def _claim(self, conn: Connection, job_id: str, now: float) -> bool:
        cur = conn.execute(
            update(_JOBS)
            .where(_JOBS.c.id == job_id, _JOBS.c.status.in_(_RUNNABLE))
            .values(
                status="running",
                started_at=func.coalesce(_JOBS.c.started_at, now),
                run_started_at=now,
                run_base_done=_JOBS.c.done,
                finished_at=None,
            )
        )
        return cur.rowcount == 1

    def claim_next(self) -> dict[str, Any] | None:
        """Pick the next runnable job and set it ``running``, in ONE transaction.

        Interrupted jobs first (they were running), then FIFO by ``seq``. ``FOR UPDATE SKIP LOCKED`` on
        Postgres; on SQLite ``BEGIN IMMEDIATE`` serialises. The returned row is the running job plus
        ``claimed_from`` (its status before the claim: "queued" or "interrupted").
        """
        with self.db.tx() as conn:
            row = conn.execute(
                select(_JOBS.c.id, _JOBS.c.status)
                .where(_JOBS.c.status.in_(_RUNNABLE))
                .order_by(case((_JOBS.c.status == "interrupted", 0), else_=1), _JOBS.c.seq)
                .limit(1)
                .with_for_update(skip_locked=True)
            ).first()
            if row is None or not self._claim(conn, row.id, time.time()):
                return None
            claimed = self._get(conn, row.id)
            return None if claimed is None else {**claimed, "claimed_from": row.status}

    def claim(self, job_id: str) -> bool:
        """queued/interrupted -> running for one id (the caller picked it). False when it was not runnable."""
        with self.db.tx() as conn:
            return self._claim(conn, job_id, time.time())

    def payload(self, job_id: str) -> str:
        with self.db.read() as conn:
            raw = conn.execute(select(_JOBS.c.payload).where(_JOBS.c.id == job_id)).scalar()
        return raw if raw is not None else "{}"

    def set_total(self, job_id: str, total: int) -> None:
        with self.db.tx() as conn:
            conn.execute(update(_JOBS).where(_JOBS.c.id == job_id).values(total=int(total)))

    def add_items(self, job_id: str, items: list[dict[str, Any]]) -> int:
        """Append rows after the last persisted one (one transaction); returns the new ``done``.

        ``done`` is read inside the transaction, so a chunk is stored whole at the right offset or not at
        all: whatever interrupts the job, the rows stay gap-free and a resume recomputes only what is missing.
        """
        with self.db.tx() as conn:
            start = conn.execute(select(_JOBS.c.done).where(_JOBS.c.id == job_id).with_for_update()).scalar()
            if start is None:
                raise KeyError(job_id)
            conn.execute(
                insert(dbschema.items),
                [
                    {
                        "job_id": job_id,
                        "idx": start + i,
                        "data": json.dumps(it, separators=(",", ":"), default=str),
                    }
                    for i, it in enumerate(items)
                ],
            )
            failed = sum(1 for it in items if isinstance(it, dict) and "error" in it)
            conn.execute(
                update(_JOBS)
                .where(_JOBS.c.id == job_id)
                .values(done=_JOBS.c.done + len(items), failed=_JOBS.c.failed + failed)
            )
        return start + len(items)

    def _terminal_event(self, conn: Connection, job_id: str, status: str) -> str | None:
        """Inside a transaction: mark ``job.<status>`` pending (if subscribed) so a restart retries it.

        Written in the same transaction as the status change, so a crash cannot leave a finished job without
        its event. The delivery id is minted here, so every retry (also after a crash) carries the same
        X-Clef-Delivery. Returns the event, or None when the job has no webhook for it.
        """
        row = conn.execute(select(_JOBS.c.webhook, _JOBS.c.deliveries).where(_JOBS.c.id == job_id)).first()
        hook = row and _loads(row.webhook)
        event = f"job.{status}"
        if not hook or event not in (hook.get("events") or DEFAULT_EVENTS):
            return None
        deliveries = json.loads(row.deliveries or "{}")
        deliveries[event] = {"id": uuid.uuid4().hex, "event": event, "status": "pending", "attempts": 0}
        conn.execute(update(_JOBS).where(_JOBS.c.id == job_id).values(deliveries=json.dumps(deliveries)))
        return event

    def finish(
        self, job_id: str, status: str, error: str | None = None, result: Any = None
    ) -> tuple[bool, str | None]:
        """Active -> ``status`` plus its pending webhook event, in ONE transaction; ``(changed, event)``."""
        with self.db.tx() as conn:
            cur = conn.execute(
                update(_JOBS)
                .where(_JOBS.c.id == job_id, _JOBS.c.status.in_(_ACTIVE_SQL))
                .values(
                    status=status,
                    finished_at=time.time(),
                    error=error,
                    result=None if result is None else json.dumps(result, default=str),
                    total=func.coalesce(_JOBS.c.total, _JOBS.c.done),
                )
            )
            if cur.rowcount != 1:
                return False, None
            return True, self._terminal_event(conn, job_id, status)

    def cancel_queued(self, job_id: str) -> tuple[bool, str | None]:
        """queued/interrupted -> cancelled plus its pending event (one transaction); ``(changed, event)``."""
        with self.db.tx() as conn:
            cur = conn.execute(
                update(_JOBS)
                .where(_JOBS.c.id == job_id, _JOBS.c.status.in_(_RUNNABLE))
                .values(
                    status="cancelled",
                    finished_at=time.time(),
                    total=func.coalesce(_JOBS.c.total, _JOBS.c.done),
                )
            )
            if cur.rowcount != 1:
                return False, None
            return True, self._terminal_event(conn, job_id, "cancelled")

    def flag_cancel(self, job_id: str) -> tuple[str | None, str | None]:
        """Cancel a job that is not queued, in ONE transaction; ``(status, event)``.

        running -> ``cancel_requested=1`` (the runner stops it between chunks): ``("running", None)``.
        interrupted (parked by a clean stop or a runner error after the caller saw it running) -> cancelled
        at once, never resumed: ``("cancelled", event)``. Anything else is left alone: ``(None, None)``.
        """
        with self.db.tx() as conn:
            status = conn.execute(
                select(_JOBS.c.status).where(_JOBS.c.id == job_id).with_for_update()
            ).scalar()
            if status == "running":
                conn.execute(update(_JOBS).where(_JOBS.c.id == job_id).values(cancel_requested=1))
                return "running", None
            if status == "interrupted":
                return "cancelled", self._cancel_now(conn, job_id)
            return None, None

    def _cancel_now(self, conn: Connection, job_id: str) -> str | None:
        """Inside a transaction: a job whose cancel was requested ends cancelled (never resumed)."""
        conn.execute(
            update(_JOBS)
            .where(_JOBS.c.id == job_id)
            .values(
                status="cancelled", finished_at=time.time(), total=func.coalesce(_JOBS.c.total, _JOBS.c.done)
            )
        )
        return self._terminal_event(conn, job_id, "cancelled")

    def _park(self, conn: Connection, job_id: str, cancel: bool) -> str | None:
        flag = conn.execute(
            select(_JOBS.c.cancel_requested)
            .where(_JOBS.c.id == job_id, _JOBS.c.status == "running")
            .with_for_update()
        ).first()
        if flag is None:
            return None
        if cancel or flag.cancel_requested:
            return self._cancel_now(conn, job_id)
        conn.execute(update(_JOBS).where(_JOBS.c.id == job_id).values(status="interrupted"))
        return None

    def park(self, job_id: str, cancel: bool = False) -> str | None:
        """Clean stop or runner error: running -> interrupted WITHOUT counting a resume (only a crash counts).

        A job whose cancel was requested (in the row, or ``cancel`` from the runner's memory) ends
        ``cancelled`` instead, with its pending event in the same transaction; that event is returned.
        """
        with self.db.tx() as conn:
            return self._park(conn, job_id, cancel)

    def park_running(self) -> list[str]:
        """Clean stop safety net: park every job still 'running' (one server per DB). Returns their ids."""
        with self.db.tx() as conn:
            ids = list(conn.execute(select(_JOBS.c.id).where(_JOBS.c.status == "running")).scalars())
            for job_id in ids:
                self._park(conn, job_id, False)
            return ids

    def recover(self, max_resumes: int = 0) -> tuple[int, list[str], list[str]]:
        """Startup: whatever was running when the process died crashed; it is interrupted (and resumed).

        That counts as a resume. A job whose cancel was requested before the crash is cancelled instead. A job
        already resumed ``max_resumes`` times (> 0) is failed: a poison item that kills the process would
        otherwise be resumed forever ahead of every other job. The terminal events are recorded pending in
        the same transaction (``pending_deliveries`` sends them). Returns ``(resumed, given_up, cancelled)``.
        """
        with self.db.tx() as conn:
            rows = conn.execute(
                select(_JOBS.c.id, _JOBS.c.status, _JOBS.c.resumes, _JOBS.c.cancel_requested)
                .where(
                    or_(
                        _JOBS.c.status == "running",
                        and_(_JOBS.c.status == "interrupted", _JOBS.c.cancel_requested == 1),
                    )
                )
                .order_by(_JOBS.c.seq)
            ).all()
            resumed, given_up, cancelled = 0, [], []
            for r in rows:
                if r.cancel_requested:  # the user's cancel wins over resuming or giving up
                    self._cancel_now(conn, r.id)
                    cancelled.append(r.id)
                elif max_resumes > 0 and r.resumes + 1 > max_resumes:
                    conn.execute(
                        update(_JOBS)
                        .where(_JOBS.c.id == r.id)
                        .values(
                            status="failed",
                            finished_at=time.time(),
                            error=f"interrupted {r.resumes + 1} times, giving up",
                            resumes=_JOBS.c.resumes + 1,
                            total=func.coalesce(_JOBS.c.total, _JOBS.c.done),
                        )
                    )
                    self._terminal_event(conn, r.id, "failed")
                    given_up.append(r.id)
                else:
                    conn.execute(
                        update(_JOBS)
                        .where(_JOBS.c.id == r.id)
                        .values(status="interrupted", resumes=_JOBS.c.resumes + 1)
                    )
                    resumed += 1
            return resumed, given_up, cancelled

    def reset_items(self, job_id: str) -> None:
        with self.db.tx() as conn:
            conn.execute(delete(dbschema.items).where(dbschema.items.c.job_id == job_id))
            conn.execute(
                update(_JOBS)
                .where(_JOBS.c.id == job_id)
                .values(done=0, failed=0, total=None, run_base_done=0)
            )

    def delete(self, job_id: str) -> bool:
        with self.db.tx() as conn:
            cur = conn.execute(delete(_JOBS).where(_JOBS.c.id == job_id, _JOBS.c.status.in_(_DONE_SQL)))
            return cur.rowcount == 1

    def purge(self, older_than: float) -> int:
        """Delete finished jobs older than ``older_than`` unless a webhook delivery is still pending."""
        with self.db.tx() as conn:
            rows = conn.execute(
                select(_JOBS.c.id, _JOBS.c.deliveries).where(
                    _JOBS.c.status.in_(_DONE_SQL), _JOBS.c.finished_at < older_than
                )
            ).all()
            doomed = [
                r.id
                for r in rows
                if not any(
                    rec.get("status") == "pending" for rec in json.loads(r.deliveries or "{}").values()
                )
            ]
            for i in range(0, len(doomed), 500):
                conn.execute(delete(_JOBS).where(_JOBS.c.id.in_(doomed[i : i + 500])))
            return len(doomed)

    # ---- results
    def items(self, job_id: str, offset: int, limit: int) -> list[tuple[int, str]]:
        it = dbschema.items
        with self.db.read() as conn:
            rows = conn.execute(
                select(it.c.idx, it.c.data)
                .where(it.c.job_id == job_id, it.c.idx >= offset)
                .order_by(it.c.idx)
                .limit(limit)
            ).all()
        return [(r.idx, r.data) for r in rows]

    # ---- webhook deliveries
    def set_delivery(self, job_id: str, event: str, record: dict[str, Any]) -> None:
        with self.db.tx() as conn:
            raw = conn.execute(
                select(_JOBS.c.deliveries).where(_JOBS.c.id == job_id).with_for_update()
            ).first()
            if raw is None:
                return
            deliveries = json.loads(raw.deliveries or "{}")
            deliveries[event] = record
            conn.execute(update(_JOBS).where(_JOBS.c.id == job_id).values(deliveries=json.dumps(deliveries)))

    def redeliver(self, job_id: str, event: str) -> dict[str, Any] | None:
        """failed -> pending (same delivery id, attempts reset). None unless the record exists and failed."""
        with self.db.tx() as conn:
            raw = conn.execute(
                select(_JOBS.c.deliveries).where(_JOBS.c.id == job_id).with_for_update()
            ).first()
            if raw is None:
                return None
            deliveries = json.loads(raw.deliveries or "{}")
            rec = deliveries.get(event)
            if not rec or rec.get("status") != "failed":
                return None
            rec = {**rec, "status": "pending", "attempts": 0}
            deliveries[event] = rec
            conn.execute(update(_JOBS).where(_JOBS.c.id == job_id).values(deliveries=json.dumps(deliveries)))
            return rec

    def pending_deliveries(self) -> list[tuple[str, str]]:
        with self.db.read() as conn:
            rows = conn.execute(
                select(_JOBS.c.id, _JOBS.c.deliveries)
                .where(_JOBS.c.webhook.isnot(None), _JOBS.c.deliveries != "{}")
                .order_by(_JOBS.c.seq)
            ).all()
        out = []
        for r in rows:
            for event, rec in json.loads(r.deliveries).items():
                if rec.get("status") == "pending" and event != "job.progress":
                    out.append((r.id, event))
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
    deliveries = row["deliveries"] or {}
    bad_hooks = sorted(ev for ev, rec in deliveries.items() if rec.get("status") == "failed")
    warnings: list[str] = []
    if row["failed"]:
        warnings.append(f"{row['failed']} of {done} item(s) failed; their result rows carry an 'error' field")
    if bad_hooks:
        warnings.append(
            f"webhook delivery failed for {', '.join(bad_hooks)};"
            f" retry with POST /v1/jobs/{row['id']}/webhook/redeliver"
        )
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
        "warnings": warnings,
        "metadata": row["metadata"],
        "webhook": None
        if not hook
        else {
            "url": hook["url"],
            "events": hook.get("events") or list(DEFAULT_EVENTS),
            "has_secret": bool(hook.get("secret")),
            "failed_deliveries": len(bad_hooks),
            "deliveries": deliveries,
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
    warnings: list[str] = Field(
        default_factory=list, description="partial failure: failed items, failed webhook deliveries"
    )
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
        self.first_error: str | None = None  # first failed row's message in this run
        self._seen = 0  # rows added in this run (circuit breaker)
        self._bad = 0
        self._errors: Counter[str] = Counter()

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
        self._trip_breaker(items)

    def _trip_breaker(self, items: list[dict[str, Any]]) -> None:
        """Abort the job when failures look systemic (rows stay stored): every one of the first
        BREAKER_MIN_ITEMS items failed, or >= BREAKER_SAME_SHARE of >= BREAKER_SAME_MIN failed alike."""
        for it in items:
            self._seen += 1
            if isinstance(it, dict) and "error" in it:
                msg = str(it["error"])[:200]
                self._bad += 1
                self._errors[msg] += 1
                if self.first_error is None:
                    self.first_error = msg
        if not self._bad:
            return
        if self._seen >= BREAKER_MIN_ITEMS and self._bad == self._seen:
            raise JobAborted(f"aborted: the first {self._seen} items all failed ({self.first_error})")
        top, n = self._errors.most_common(1)[0]
        if self._seen >= BREAKER_SAME_MIN and n >= BREAKER_SAME_SHARE * self._seen:
            raise JobAborted(f"aborted: {n} of {self._seen} items failed with the same error ({top})")

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
        self._purge_task: asyncio.Task[None] | None = None
        self._failures: dict[str, int] = {}  # job id -> consecutive runner-level failures
        self.current: str | None = None

    def kinds(self) -> dict[str, Any]:
        return self.ctx.extra.setdefault("job_kinds", {})

    # ---- lifecycle
    async def start(self) -> None:
        self.wake = asyncio.Event()
        # recover() records the terminal events itself (same transaction); pending_deliveries() sends them.
        resumed, given_up, cancelled = await asyncio.to_thread(
            self.store.recover, int(self.ctx.cfg.job_max_resumes)
        )
        if resumed:
            log.info("jobs: %d job(s) interrupted by a crash will be resumed", resumed)
        for job_id in cancelled:
            log.info("jobs: job %s was cancelled before the last shutdown; not resuming it", job_id)
        for job_id in given_up:
            log.warning(
                "jobs: job %s was interrupted too many times (CLEF_JOB_MAX_RESUMES=%d); giving up",
                job_id,
                self.ctx.cfg.job_max_resumes,
            )
        await asyncio.to_thread(self._purge)
        for job_id, event in await asyncio.to_thread(self.store.pending_deliveries):
            self.dispatcher.enqueue(job_id, event)
        self._task = asyncio.create_task(self._loop(), name="clef-jobs")
        self._purge_task = asyncio.create_task(self._purge_loop(), name="clef-jobs-purge")

    async def stop(self) -> None:
        tasks = [t for t in (self._task, self._purge_task) if t]
        for t in tasks:
            t.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        # Safety net for a clean stop that landed outside the kind (e.g. right after claim_next): park the
        # job without counting a resume. A store call still in a worker thread holds the lock: this waits.
        with contextlib.suppress(Exception):
            await asyncio.to_thread(self.store.park_running)
        await self.dispatcher.close()  # the database itself is closed by the app lifespan (ctx.db)

    def poke(self) -> None:
        if self.wake is not None:
            self.wake.set()

    def _purge(self) -> None:
        ttl = self.ctx.cfg.job_ttl_hours
        if ttl > 0:
            n = self.store.purge(time.time() - ttl * 3600)
            if n:
                log.info("jobs: purged %d finished job(s) older than %g h", n, ttl)

    async def _purge_loop(self) -> None:
        """Retention runs on its own timer, whether the queue is idle or a job has been running for days."""
        while True:
            await asyncio.sleep(PURGE_INTERVAL_S)
            try:
                await asyncio.to_thread(self._purge)
            except asyncio.CancelledError:
                raise
            except Exception:
                log.exception("job purge failed")

    async def _loop(self) -> None:
        assert self.wake is not None
        while True:
            row = None
            try:
                row = await asyncio.to_thread(self.store.claim_next)  # the job is 'running' from here
                if row is None:
                    with contextlib.suppress(asyncio.TimeoutError):
                        await asyncio.wait_for(self.wake.wait(), timeout=PURGE_INTERVAL_S)
                    self.wake.clear()
                    continue
                await self._execute(row)
                self._failures.pop(row["id"], None)
            except asyncio.CancelledError:
                raise
            except Exception:
                await self._runner_failed(row)

    async def _runner_failed(self, row: dict[str, Any] | None) -> None:
        """The runner itself (not the kind) raised: retry a few times, then fail the job (poison pill)."""
        log.exception("job runner error")
        if row is None:
            # claim_next itself failed, possibly after its commit: park whatever is left 'running' (there is
            # one runner and nothing is executing now), so the job is picked up again without a resume.
            with contextlib.suppress(Exception):
                await asyncio.to_thread(self.store.park_running)
        else:
            job_id = row["id"]
            n = self._failures[job_id] = self._failures.get(job_id, 0) + 1
            try:
                # Left 'running'? Park it again (not a crash: RUNNER_MAX_FAILURES is the budget here).
                event = await asyncio.to_thread(self.store.park, job_id, job_id in self.cancel_ids)
                if event:
                    self.dispatcher.enqueue(job_id, event)
                if n >= RUNNER_MAX_FAILURES:
                    self._failures.pop(job_id, None)
                    log.error("jobs: job %s failed %d times in the runner; failing it", job_id, n)
                    await self._finish(job_id, "failed", error="internal error: the job could not be run")
                    return
            except asyncio.CancelledError:
                raise
            except Exception:
                log.exception("job runner error while failing job %s", job_id)
        await asyncio.sleep(RUNNER_RETRY_PAUSE_S)

    # ---- cancel / webhooks
    async def request_cancel(self, row: dict[str, Any]) -> dict[str, Any]:
        job_id = row["id"]
        cancelled, event = await asyncio.to_thread(self._cancel_queued_sync, job_id)
        if cancelled:
            if event:
                self.dispatcher.enqueue(job_id, event)
        else:
            # Not queued: flag it if running, or cancel it now if a park (clean stop, runner error) got
            # there first. One atomic store call, not gated on a stale read: a racing park cannot drop it.
            self.cancel_ids.add(job_id)  # before the flag: a running job may check either one
            status, event = await asyncio.to_thread(self.store.flag_cancel, job_id)
            if status != "running":
                self.cancel_ids.discard(job_id)
            if event:
                self.dispatcher.enqueue(job_id, event)
        return await asyncio.to_thread(self.store.get, job_id) or row

    def _finish_sync(
        self, job_id: str, status: str, error: str | None, result: Any
    ) -> tuple[bool, str | None]:
        """State change + pending event in ONE transaction (``JobStore.finish``): neither a cancelled
        awaiting task (shutdown) nor a hard crash can land between them, so a terminal event is never lost."""
        return self.store.finish(job_id, status, error, result)

    def _cancel_queued_sync(self, job_id: str) -> tuple[bool, str | None]:
        return self.store.cancel_queued(job_id)

    def notify_progress(self, job_id: str) -> None:
        if self._wants_progress:
            self.dispatcher.progress(job_id, True)

    async def redeliver(self, job_id: str, event: str) -> bool:
        """failed -> pending -> delivered again (same delivery id). False when that event did not fail."""
        if await asyncio.to_thread(self.store.redeliver, job_id, event) is None:
            return False
        self.dispatcher.enqueue(job_id, event)
        return True

    # ---- execution
    async def _execute(self, row: dict[str, Any]) -> None:
        """Run a job ``claim_next`` already set 'running' (a cancel may target it from that moment on)."""
        job_id = row["id"]
        # Never discard cancel_ids here: a cancel that arrived right after the claim is already in it (and in
        # the row). Stale ids cannot exist: park / finish end a job whose cancel was requested.
        if row["cancel_requested"]:
            self.cancel_ids.add(job_id)
        spec = self.kinds().get(row["kind"])
        if spec is None:
            await self._finish(job_id, "failed", error=f"unknown job kind {row['kind']!r}")
            return
        served = self.ctx.model_ref()
        if row.get("model") and served and row["model"] != served:  # one model per process (D6)
            await self._finish(
                job_id, "failed", error=f"job was submitted for {row['model']}, this server serves {served}"
            )
            return
        resuming = row.get("claimed_from", row["status"]) == "interrupted"
        done = row["done"] if resuming and spec.get("resumable") else 0
        if resuming and not spec.get("resumable"):
            await asyncio.to_thread(self.store.reset_items, job_id)
        try:
            payload = json.loads(await asyncio.to_thread(self.store.payload, job_id))
            parsed = await asyncio.to_thread(spec["validate"], payload)
        except (ValueError, ValidationError, LookupError) as exc:
            await self._finish(job_id, "failed", error=f"payload is no longer valid: {_msg(exc)}")
            return
        self.current = job_id
        self._wants_progress = "job.progress" in ((row["webhook"] or {}).get("events") or ())
        handle = JobHandle(self, {**row, "done": done})
        t0 = time.monotonic()
        try:
            result = await spec["run"](self.ctx, parsed, handle)
        except asyncio.CancelledError:
            # Clean shutdown: park without counting a resume (a requested cancel ends it cancelled; its
            # event is sent on the next start). An add_items still in its worker thread holds the write
            # lock, so this waits for that chunk to commit; a chunk not yet written is recomputed on resume.
            self.store.park(job_id, cancel=job_id in self.cancel_ids)
            raise
        except JobCancelled:
            result = None
        except Exception as exc:
            # map_exception logs the traceback of unexpected types; the client only sees the mapped text
            detail = str(exc) if isinstance(exc, JobAborted) else self.ctx.map_exception(exc).detail
            log.warning("job %s (%s) failed: %s", job_id, row["kind"], detail)
            await self._finish(job_id, "failed", error=detail)
            return
        finally:
            self.current = None
        cancelled = job_id in self.cancel_ids
        self.cancel_ids.discard(job_id)
        status, error = ("cancelled" if cancelled else "succeeded"), None
        if not cancelled:
            now = await asyncio.to_thread(self.store.get, job_id)
            if now and now["done"] > 0 and now["failed"] >= now["done"]:  # nothing worked: not a success
                status = "failed"
                error = f"all {now['done']} item(s) failed" + (
                    f" ({handle.first_error})" if handle.first_error else ""
                )
                log.warning("job %s (%s) failed: %s", job_id, row["kind"], error)
        await self._finish(job_id, status, error=error, result=result)
        log.info(
            "job %s (%s) %s: %d item(s) in %.1fs",
            job_id,
            row["kind"],
            "done" if status == "succeeded" else status,
            handle.done,
            time.monotonic() - t0,
        )

    async def _finish(self, job_id: str, status: str, error: str | None = None, result: Any = None) -> None:
        _, event = await asyncio.to_thread(self._finish_sync, job_id, status, error, result)
        if event:
            self.dispatcher.enqueue(job_id, event)


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
    model: str | None = Field(None, description="default: the served model")

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
    version: int | None = Field(None, ge=1, description="classifier version (default: the latest)")
    classifier_version: int | None = Field(
        None, description="set by the server: the version of `snapshot_of` that was copied"
    )
    multi_label: bool = False
    threshold: float | None = Field(None, ge=0.0, le=1.0)
    snapshot_of: str | None = Field(
        None, description="set by the server: the saved classifier whose definition was copied at submit"
    )

    @field_validator("labels")
    @classmethod
    def _labels(cls, v: Any) -> Any:
        return v if v is None else check_labels_shape(v)


class ScoreJob(_Items):
    levels: list[str] | None = None
    classifier: str | None = Field(None, description="name of a saved score classifier (instead of levels)")
    version: int | None = Field(None, ge=1, description="classifier version (default: the latest)")
    classifier_version: int | None = Field(
        None, description="set by the server: the version of `snapshot_of` that was copied"
    )
    snapshot_of: str | None = Field(
        None, description="set by the server: the saved classifier whose definition was copied at submit"
    )


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


def _resolve_classifier(ctx: AppContext, name: str, kind: str, version: int | None = None) -> dict[str, Any]:
    from .classifiers import InvalidName, validate_name

    try:
        validate_name(name)
    except InvalidName as exc:
        raise ValueError(f"classifier: {exc}") from exc
    doc = ctx.store.get(name) if version is None else ctx.store.get(name, version)
    if doc is None:
        if version is not None and ctx.store.get(name) is not None:
            raise ValueError(f"version: classifier {name!r} has no version {version}")
        raise ValueError(f"classifier: {name!r} not found")
    if doc["kind"] != kind:
        other = "score" if kind == "classify" else "classify"
        raise ValueError(f"classifier: {name!r} is a {doc['kind']} classifier; use kind={other!r}")
    return doc


def _check_version(classifier: str | None, version: int | None) -> None:
    if version is not None and classifier is None:
        raise ValueError("version: only allowed with classifier")


def payload_models(payload: dict[str, Any]) -> list[str | None]:
    """Every ``model`` a job payload asks for (top level, and each SystemOne request)."""
    models: list[str | None] = [payload.get("model")] if "model" in payload else []
    reqs = payload.get("requests")
    if isinstance(reqs, list):
        models += [r.get("model") for r in reqs if isinstance(r, dict) and "model" in r]
    return models


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
        _check_version(body.classifier, body.version)
        instructions, multi, thr = body.instructions, body.multi_label, body.threshold
        labels, version = body.labels, None
        if body.classifier is not None:
            doc = _resolve_classifier(ctx, body.classifier, "classify", body.version)
            labels, version = doc["labels"], doc.get("version")
            instructions = instructions if instructions is not None else doc.get("instructions")
            multi = bool(doc.get("multi_label")) if "multi_label" not in payload else multi
            thr = thr if thr is not None else doc.get("threshold")
        assert labels is not None
        check_label_count(labels, multi, cfg.max_labels)
        if thr is not None and not multi:
            raise ValueError("threshold: only allowed with multi_label=true")
        threshold = resolve_threshold(thr, cfg) if multi else None
        model = ctx.check_model(body.model)
        try:
            classify_to_systemone(body.inputs[0], labels, instructions, multi, model)
        except ValidationError as exc:
            raise ValueError(format_errors(list(exc.errors()))) from exc
        return _Parsed(
            inputs=body.inputs, labels=labels, instructions=instructions, multi=multi, threshold=threshold,
            model=model, version=version,
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

    def snapshot_classify(payload: dict[str, Any], p: _Parsed) -> dict[str, Any]:
        """Replace ``classifier`` by the definition it resolved to, so a resumed job never mixes versions."""
        if payload.get("classifier") is None:
            return payload
        drop = ("classifier", "version", "labels", "snapshot_of", "classifier_version")
        snap = {k: v for k, v in payload.items() if k not in drop}
        snap.update(labels=p.labels, multi_label=p.multi, snapshot_of=payload["classifier"])
        if p.version is not None:
            snap["classifier_version"] = p.version
        if p.instructions is not None:
            snap["instructions"] = p.instructions
        if p.multi and p.threshold is not None:
            snap["threshold"] = p.threshold
        return snap

    def snapshot_score(payload: dict[str, Any], p: _Parsed) -> dict[str, Any]:
        if payload.get("classifier") is None:
            return payload
        drop = ("classifier", "version", "levels", "snapshot_of", "classifier_version")
        snap = {k: v for k, v in payload.items() if k not in drop}
        snap.update(levels=p.levels, snapshot_of=payload["classifier"])
        if p.version is not None:
            snap["classifier_version"] = p.version
        if p.instructions is not None:
            snap["instructions"] = p.instructions
        return snap

    def validate_score(payload: dict[str, Any]) -> _Parsed:
        try:
            body = ScoreJob.model_validate(payload)
        except ValidationError as exc:
            raise ValueError(format_errors(list(exc.errors()))) from exc
        _max_items(ctx, len(body.inputs), "inputs")
        if (body.levels is None) == (body.classifier is None):
            raise ValueError("levels: give exactly one of levels or classifier")
        _check_version(body.classifier, body.version)
        levels, instructions, version = body.levels, body.instructions, None
        if body.classifier is not None:
            doc = _resolve_classifier(ctx, body.classifier, "score", body.version)
            levels, version = doc["levels"], doc.get("version")
            instructions = instructions if instructions is not None else doc.get("instructions")
        assert levels is not None
        if len(set(levels)) != len(levels) or any(not isinstance(x, str) or not x.strip() for x in levels):
            raise ValueError("levels: must be unique, non-empty strings")
        check_levels_count(levels, cfg.max_labels)
        model = ctx.check_model(body.model)
        try:
            score_to_systemone(body.inputs[0], levels, instructions, model)
        except ValidationError as exc:
            raise ValueError(format_errors(list(exc.errors()))) from exc
        return _Parsed(
            inputs=body.inputs, levels=levels, instructions=instructions, model=model, version=version
        )

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
        "classify": {
            "validate": validate_classify,
            "run": run_classify,
            "resumable": True,
            "snapshot": snapshot_classify,
        },
        "score": {
            "validate": validate_score,
            "run": run_score,
            "resumable": True,
            "snapshot": snapshot_score,
        },
        "systemone": {"validate": validate_systemone, "run": run_systemone, "resumable": True},
    }


# ------------------------------------------------------------------ CSV flattening


_FORMULA_START = ("=", "+", "-", "@", "\t", "\r")


def _neutralise(text: str) -> str:
    """Spreadsheet formulas (CSV injection): text starting with = + - @ gets a leading quote."""
    return f"'{text}" if text.startswith(_FORMULA_START) else text


def flatten(value: Any, prefix: str = "", csv_safe: bool = False) -> dict[str, str]:
    """Nested row -> flat ``a.b`` columns (lists of scalars joined with ``|``).

    ``csv_safe`` neutralises formula-looking *strings* (decided on the original type, so the number -0.5 is
    left alone).
    """
    out: dict[str, str] = {}
    if isinstance(value, dict):
        for k, v in value.items():
            out.update(flatten(v, f"{prefix}.{k}" if prefix else str(k), csv_safe))
    elif isinstance(value, list) and all(not isinstance(x, (dict, list)) for x in value):
        cell = "|".join("" if x is None else str(x) for x in value)
        out[prefix] = _neutralise(cell) if csv_safe and value and isinstance(value[0], str) else cell
    elif isinstance(value, list):
        out[prefix] = json.dumps(value, ensure_ascii=False)
    elif value is None:
        out[prefix] = ""
    elif isinstance(value, str):
        out[prefix] = _neutralise(value) if csv_safe else value
    else:
        out[prefix] = str(value)
    return out


# ------------------------------------------------------------------ router


def router(ctx: AppContext) -> APIRouter:
    cfg, ApiError = ctx.cfg, ctx.api_error  # noqa: N806
    store = JobStore(ctx.db)
    policy = WebhookPolicy(cfg.webhook_allow)

    def hook_view(job_id: str) -> dict[str, Any] | None:
        row = store.get(job_id)
        return None if row is None else {"webhook": row["webhook"], "job": public_job(row)}

    dispatcher = WebhookDispatcher(
        policy,
        hook_view,
        store.set_delivery,
        timeout_s=cfg.webhook_timeout_s,
        attempts=cfg.webhook_attempts,
        base_delay=cfg.webhook_backoff_s,
        max_delay=cfg.webhook_backoff_cap_s,
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
    async def submit(
        body: JobCreate,
        request: Request,
        response: Response,
        idempotency_key: str | None = Header(
            None,
            alias="Idempotency-Key",
            description="same key again (same API key) returns the job created the first time",
        ),
    ) -> dict[str, Any]:
        if idempotency_key is not None:
            if not IDEMPOTENCY_KEY_RE.match(idempotency_key):
                raise ApiError(400, "Idempotency-Key: 1-128 printable ASCII characters")
            known = await run_in_threadpool(store.find_idem, caller(request), idempotency_key)
            if known:
                response.headers["Location"] = f"/v1/jobs/{known['id']}"
                response.headers["Idempotent-Replay"] = "true"
                return public_job(known)
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
        spec = kinds[body.kind]
        payload = body.payload
        for requested in payload_models(payload):  # a model this server does not serve: 404 before anything
            try:
                ctx.check_model(requested)
            except LookupError as exc:
                if isinstance(exc, (KeyError, IndexError)):
                    raise
                raise ApiError(404, str(exc)) from exc
        try:
            parsed = await run_in_threadpool(spec["validate"], payload)
            if spec.get("snapshot"):  # saved classifiers are copied into the payload now (see docs/jobs.md)
                payload = await run_in_threadpool(spec["snapshot"], payload, parsed)
        except (ValueError, ValidationError) as exc:
            raise ApiError(400, f"payload.{_msg(exc)}") from exc
        try:
            row = await run_in_threadpool(
                store.create,
                body.kind,
                caller(request),
                payload,
                body.metadata,
                hook,
                cfg.max_jobs,
                idempotency_key,
                ctx.model_ref(),
            )
        except JobsFull as exc:
            raise ApiError(409, str(exc), {"Retry-After": "30"}) from exc
        if row.get("replayed"):
            response.headers["Idempotent-Replay"] = "true"
        else:
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

    @api.post(
        "/{job_id}/webhook/redeliver",
        response_model=JobInfo,
        summary="Retry a webhook delivery that failed",
        dependencies=ctx.limited,
    )
    async def redeliver_webhook(
        job_id: str,
        request: Request,
        event: Literal["job.succeeded", "job.failed", "job.cancelled", "job.progress"] | None = Query(
            None, description="default: the job's own terminal event"
        ),
    ) -> dict[str, Any]:
        row = await load(request, job_id)
        if not row["webhook"]:
            raise ApiError(409, "job has no webhook")
        if row["status"] not in FINISHED:
            raise ApiError(409, f"job is {row['status']}; wait until it has finished")
        event = event or f"job.{row['status']}"
        try:
            policy.check_url(row["webhook"]["url"])
        except WebhookRefused as exc:
            raise ApiError(400, f"webhook.url: {exc}") from exc
        if not await runner.redeliver(job_id, event):
            state = ((row["deliveries"] or {}).get(event) or {}).get("status")
            raise ApiError(
                409,
                f"nothing to redeliver for {event}: "
                + (f"its delivery is {state}" if state else "no such delivery was recorded"),
            )
        return public_job(await run_in_threadpool(store.get, job_id) or row)

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
        # Rows keep arriving while a job runs: freeze the row count now so the two CSV passes (columns,
        # then rows) see exactly the same rows.
        upto = max(0, row["done"] - offset)
        limit = upto if limit is None else min(limit, upto)
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


def _csv_safe(row: dict[str, Any]) -> dict[str, Any]:
    """Neutralise formulas in the string values of a flat row; numbers and other types are left alone."""
    return {k: _neutralise(v) if isinstance(v, str) else v for k, v in row.items()}


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
            writer.writerow(flatten(json.loads(d), csv_safe=True))
        yield buf.getvalue()


__all__ = ["JobCancelled", "JobHandle", "JobStore", "public_job", "router"]
