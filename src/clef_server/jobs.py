"""Async job API: submit a large batch, poll (or get a webhook), page / stream the results.

Feature module (``main.FEATURES``): ``router(ctx)`` registers ``/v1/jobs*``, the background runner and the
webhook dispatcher. See docs/jobs.md for the contract.

* Up to ``CLEF_JOB_ACTIVE`` jobs run as asyncio tasks, but a :class:`Scheduler` hands out turns: strict
  classes (urgent > high > normal > low, with aging), stride fair share per owner, optional off-hours
  windows. A job feeds the engine in chunks of ``max_microbatch`` records and ONE chunk is in flight across
  all jobs, so interactive requests queue behind at most one chunk.
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
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Literal

from fastapi import APIRouter, Header, Query, Request, Response
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator
from sqlalchemy import and_, case, delete, func, insert, or_, select, update
from sqlalchemy.engine import Connection
from sqlalchemy.exc import IntegrityError
from starlette.concurrency import run_in_threadpool

from . import dbschema, jobsort
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
from .db import Database, audit
from .engine import EngineNotReady, GpuOutOfMemory
from .keystore import QuotaExceeded, Reservation
from .schedule import WINDOW_RE, Window, parse_window
from .schemas import SystemOneRequest, check_limits, format_errors
from .webhooks import DEFAULT_EVENTS, WebhookDispatcher, WebhookPolicy, WebhookRefused

log = logging.getLogger("clef")

STATUSES = ("queued", "running", "succeeded", "failed", "cancelled", "interrupted")
ACTIVE = ("queued", "running", "interrupted")
FINISHED = ("succeeded", "failed", "cancelled")
EXPIRED = "expired"  # jobs.error of a job that ran out of expire_after_s
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
PRIORITY_LOW, PRIORITY_NORMAL, PRIORITY_HIGH, PRIORITY_URGENT = range(4)
PRIORITY_NAMES = ("low", "normal", "high", "urgent")  # jobs.priority is the index
SCHEDULER_MAX_SLEEP_S = 600.0


class JobsFull(RuntimeError):
    """max_jobs queued/running jobs reached (-> HTTP 409)."""


class JobAborted(Exception):
    """Systemic failure inside a kind: the job fails with this message (shown to the client as is)."""


class JobCancelled(Exception):
    """Raised inside a kind's runner to stop early (equivalent to returning after ``job.cancelled``)."""


class JobExpired(JobCancelled):
    """``expire_after_s`` ran out: the job stops between chunks, ends ``cancelled`` with error "expired"."""


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
        for k in ("metadata", "webhook", "deliveries", "result", "schedule"):
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
        quota_day: str | None = None,
        quota_reserved: int = 0,
        priority: int = PRIORITY_NORMAL,
        schedule: dict[str, Any] | None = None,
        expire_after_s: float | None = None,
        eligible_at: float | None = None,
    ) -> dict[str, Any]:
        """Insert a queued job. With ``idem_key``, a job this owner already submitted under it is returned
        instead (``row["replayed"]`` is True). ``model`` = the model ref served at submit (``jobs.model``).
        ``quota_day`` / ``quota_reserved``: the daily quota reservation made for it (D1). ``priority`` (0..3),
        ``schedule``, ``expire_after_s`` and ``eligible_at`` are the scheduler's (E1)."""
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
                        quota_day=quota_day,
                        quota_reserved=quota_reserved,
                        priority=priority,
                        schedule=None if schedule is None else json.dumps(schedule),
                        expire_after_s=expire_after_s,
                        eligible_at=eligible_at,
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
        self,
        owner: str | None,
        scoped: bool,
        status: str | None,
        kind: str | None,
        limit: int,
        offset: int,
        priority: int | None = None,
    ) -> tuple[list[dict[str, Any]], int]:
        where = []
        if scoped:
            where.append(_JOBS.c.owner == owner)  # owner-less jobs (auth-off era) are admin-only
        if status:
            where.append(_JOBS.c.status == status)
        if kind:
            where.append(_JOBS.c.kind == kind)
        if priority is not None:
            where.append(_JOBS.c.priority == priority)
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
        ``claimed_from`` (its status before the claim: "queued" or "interrupted"). None once the server lock
        on Postgres is lost (another server may own the database; this one is exiting).
        """
        if self.db.lock_lost:
            return None
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

    def claim(self, job_id: str) -> dict[str, Any] | None:
        """queued/interrupted -> running for one id (the scheduler picked it): the running row plus
        ``claimed_from``, or None when it was not runnable. Raises once the Postgres server lock is lost."""
        with self.db.tx() as conn:
            before = conn.execute(
                select(_JOBS.c.status).where(_JOBS.c.id == job_id).with_for_update()
            ).scalar()
            if before is None or not self._claim(conn, job_id, time.time()):
                return None
            claimed = self._get(conn, job_id)
            return None if claimed is None else {**claimed, "claimed_from": before}

    def runnable(self) -> list[dict[str, Any]]:
        """Light rows (no payload) of the jobs waiting for a slot, for the scheduler to order."""
        cols = (
            _JOBS.c.id, _JOBS.c.owner, _JOBS.c.status, _JOBS.c.priority, _JOBS.c.schedule, _JOBS.c.seq,
            _JOBS.c.created_at, _JOBS.c.eligible_at, _JOBS.c.expire_after_s,
        )  # fmt: skip
        with self.db.read() as conn:
            rows = conn.execute(
                select(*cols).where(_JOBS.c.status.in_(_RUNNABLE)).order_by(_JOBS.c.seq)
            ).all()
        return [{**r._mapping, "schedule": _loads(r.schedule)} for r in rows]

    def payload(self, job_id: str) -> str:
        with self.db.read() as conn:
            raw = conn.execute(select(_JOBS.c.payload).where(_JOBS.c.id == job_id)).scalar()
        return raw if raw is not None else "{}"

    def set_total(self, job_id: str, total: int) -> None:
        with self.db.tx() as conn:
            conn.execute(update(_JOBS).where(_JOBS.c.id == job_id).values(total=int(total)))

    def add_items(self, job_id: str, items: list[dict[str, Any]], tokens: int = 0, hits: int = 0) -> int:
        """Append rows after the last persisted one (one transaction); returns the new ``done``.

        ``done`` is read inside the transaction, so a chunk is stored whole at the right offset or not at
        all: whatever interrupts the job, the rows stay gap-free and a resume recomputes only what is missing.
        ``tokens`` adds to ``jobs.tokens``; ``hits`` (cache hits among these rows) lowers ``quota_reserved``
        in the same transaction (the caller hands them back to the quota once this returns).
        """
        with self.db.tx() as conn:
            cur = conn.execute(
                select(_JOBS.c.done, _JOBS.c.quota_reserved, _JOBS.c.quota_day)
                .where(_JOBS.c.id == job_id)
                .with_for_update()
            ).first()
            if cur is None:
                raise KeyError(job_id)
            start = cur.done
            hits = min(hits, cur.quota_reserved) if cur.quota_day is not None else 0
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
            # an explicit count wins; otherwise jobs.tokens sums the rows' input_tokens (batch usage)
            if not tokens:
                tokens = sum(int(it.get("input_tokens") or 0) for it in items if isinstance(it, dict))
            conn.execute(
                update(_JOBS)
                .where(_JOBS.c.id == job_id)
                .values(
                    done=_JOBS.c.done + len(items),
                    failed=_JOBS.c.failed + failed,
                    tokens=_JOBS.c.tokens + max(0, int(tokens)),
                    quota_reserved=_JOBS.c.quota_reserved - hits,
                )
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

    def settle_quota(
        self, job_id: str | None = None, hits: dict[str, int] | None = None
    ) -> list[tuple[str, str, int]]:
        """Hand back the unprocessed part of the reservation of failed / cancelled jobs: ``(owner, day, n)``.

        ``n = quota_reserved - done + hits``; ``quota_reserved`` is then set to ``done - hits`` (what was
        really charged), so a second call refunds nothing. ``hits`` = per job, the cache hits already
        handed back (they lowered ``quota_reserved`` but are part of ``done``); empty after a restart, which
        can only make the refund short. One job, or (``job_id`` None) every finished one: startup uses that
        for jobs that ended while the process was stopping.
        """
        out: list[tuple[str, str, int]] = []
        with self.db.tx() as conn:
            q = select(
                _JOBS.c.id, _JOBS.c.owner, _JOBS.c.quota_day, _JOBS.c.quota_reserved, _JOBS.c.done
            ).where(
                _JOBS.c.status.in_(("failed", "cancelled")),
                _JOBS.c.quota_day.is_not(None),
                or_(_JOBS.c.quota_reserved > _JOBS.c.done, _JOBS.c.id.in_(list(hits or ()))),
            )
            if job_id is not None:
                q = q.where(_JOBS.c.id == job_id)
            for r in conn.execute(q).all():
                h = (hits or {}).get(r.id, 0)
                conn.execute(
                    update(_JOBS).where(_JOBS.c.id == r.id).values(quota_reserved=max(0, r.done - h))
                )
                if r.owner and r.quota_reserved - r.done + h > 0:
                    out.append((r.owner, r.quota_day, r.quota_reserved - r.done + h))
        return out

    def cancel_queued(self, job_id: str, error: str | None = None) -> tuple[bool, str | None]:
        """queued/interrupted -> cancelled plus its pending event (one transaction); ``(changed, event)``.
        ``error="expired"`` for a job that ran out of ``expire_after_s`` before it could start."""
        with self.db.tx() as conn:
            cur = conn.execute(
                update(_JOBS)
                .where(_JOBS.c.id == job_id, _JOBS.c.status.in_(_RUNNABLE))
                .values(
                    status="cancelled",
                    error=error,
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


def public_job(
    row: dict[str, Any], now: float | None = None, waiting: tuple[str | None, float | None] = (None, None)
) -> dict[str, Any]:
    """The JSON a client sees (never the payload, owner or webhook secret).

    ``waiting`` = ``(waiting_reason, next_window_at)`` from :meth:`JobRunner.waiting` (runtime state).
    """
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
        "priority": PRIORITY_NAMES[min(max(row["priority"], 0), PRIORITY_URGENT)],
        "urgent": row["priority"] >= PRIORITY_URGENT,
        "schedule": row["schedule"],
        "waiting_reason": waiting[0],
        "next_window_at": _iso(waiting[1]),
        "expired": status == "cancelled" and row["error"] == EXPIRED,
        "model": row["model"],
        "tokens": row["tokens"],
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


class JobSchedule(_Base):
    window: str = Field(
        pattern=WINDOW_RE.pattern, description="'HH:MM-HH:MM'; start > end crosses midnight, equal = 24 h"
    )
    tz: str | None = Field(None, max_length=64, description="IANA zone (default: CLEF_JOB_TZ)")


class JobCreate(_Base):
    kind: str = Field(description="classify | score | systemone | any registered kind")
    payload: dict[str, Any] = Field(description="kind-specific; see docs/jobs.md")
    webhook: WebhookSpec | None = None
    metadata: dict[str, Any] | None = Field(None, description="free-form JSON echoed back (<= 16 KB)")
    priority: Literal["low", "normal", "high"] = Field(
        "normal", description="strict between classes; waiting jobs age upwards (CLEF_JOB_AGING_S)"
    )
    urgent: bool = Field(False, description="needs the `urgent` scope: top class, ignores the window")
    schedule: JobSchedule | None = Field(
        None, description="run only inside this window (default: CLEF_JOB_WINDOW in CLEF_JOB_TZ)"
    )
    expire_after_s: float | None = Field(
        None, gt=0, le=31_536_000, description="cancel it ('expired') this long after it first could run"
    )


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
    priority: Literal["low", "normal", "high", "urgent"] = "normal"
    urgent: bool = False
    schedule: dict[str, Any] | None = Field(None, description="{window, tz} the job is restricted to")
    waiting_reason: Literal["off_hours", "active_limit", "turn"] | None = Field(
        None, description="why a queued / running job is not advancing right now"
    )
    next_window_at: str | None = Field(None, description="when the window opens next (off_hours only)")
    expired: bool = Field(False, description="cancelled because expire_after_s ran out")
    model: str | None = Field(None, description="model the job was submitted for")
    tokens: int = Field(0, description="sum of input_tokens of the stored rows")


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


# ------------------------------------------------------------------ scheduler


@dataclass
class Entry:
    """One job as the scheduler sees it (in memory only; rebuilt from the database after a restart)."""

    id: str
    owner: str  # "" = owner-less (auth off)
    cls: int  # PRIORITY_*
    seq: int
    interrupted: bool
    window: Window | None  # None = always open (also for urgent)
    deadline: float | None  # eligible_at + expire_after_s
    ref: float  # aging reference: when the job last had a turn (or became eligible)
    pause: str | None = None  # "off_hours" | "active_limit": holds no slot and gets no turns
    waiter: asyncio.Future[None] | None = None  # set while the job waits for its turn


class Scheduler:
    """Who may use the engine next. Pure policy plus the turn gate; all state is in memory.

    * Classes are strict (urgent > high > normal > low). A waiting job's class rises one level per
      ``aging_s`` without a turn, never to urgent.
    * Inside a class, stride scheduling per owner: ``pass += cost`` per turn (weight 1 for everyone), the
      lowest pass goes next, an owner that comes back starts at the current minimum. Then the owner with fewer
      jobs in the scheduler, higher base class, interrupted, ``seq``.
    * ``turn`` / ``done`` bracket every engine call of a job; only one call is in flight across all jobs.
    * Slots: ``active`` jobs may take turns at once. A job outside its window frees its slot and gets no turns
      (it stays ``running``); urgent jobs ignore the window and the slot limit.

    All methods run on the event loop; ``clock`` returns unix seconds (a fake one in tests).
    """

    def __init__(self, active: int, aging_s: float, clock: Callable[[], float] = time.time):
        self.active, self.aging_s, self.clock = max(1, active), aging_s, clock
        self.entries: dict[str, Entry] = {}
        self.passes: dict[str, float] = {}
        self.vtime = 0.0  # the pass of the owner served last: where a returning owner starts
        self.inflight: str | None = None

    # ---- ordering
    def eff_class(self, e: Entry, now: float) -> int:
        if e.cls >= PRIORITY_URGENT or self.aging_s <= 0:
            return e.cls
        return min(e.cls + int(max(0.0, now - e.ref) // self.aging_s), max(e.cls, PRIORITY_HIGH))

    def running_of(self, owner: str) -> int:
        return sum(1 for x in self.entries.values() if x.owner == owner)

    def key(self, e: Entry, now: float) -> tuple[Any, ...]:
        return (
            -self.eff_class(e, now), self.passes.get(e.owner, 0.0), self.running_of(e.owner),
            -e.cls, not e.interrupted, e.seq,
        )  # fmt: skip

    # ---- membership and slots
    def slots_used(self) -> int:
        return sum(1 for e in self.entries.values() if e.pause is None)

    def has_slot(self) -> bool:
        return self.slots_used() < self.active

    def runnable_owners(self) -> set[str]:
        """Owners with at least one entry that may take turns (not paused)."""
        return {x.owner for x in self.entries.values() if x.pause is None}

    def _lift(self, owner: str, runnable: set[str]) -> None:
        """An owner that becomes runnable again starts level with the owners in ``runnable`` (the lowest of
        their passes, or ``vtime`` when nobody else runs): it does not collect the turns it missed."""
        others = [self.passes.get(o, 0.0) for o in runnable if o != owner]
        self.passes[owner] = max(self.passes.get(owner, 0.0), min(others) if others else self.vtime)

    def register(self, e: Entry) -> None:
        runnable = self.runnable_owners()
        if e.pause is None and e.owner not in runnable:  # new, or all its other jobs are paused
            self._lift(e.owner, runnable)
        self.entries[e.id] = e

    def unregister(self, job_id: str) -> None:
        e = self.entries.pop(job_id, None)
        if e is not None and e.waiter is not None and not e.waiter.done():
            e.waiter.cancel()
        if self.inflight == job_id:
            self.inflight = None
        self.dispatch()

    def interrupt(self, job_id: str, exc: BaseException) -> None:
        """Wake a job that waits for a turn with ``exc`` (cancel, expiry): it must not wait for its window."""
        e = self.entries.get(job_id)
        if e is not None and e.waiter is not None and not e.waiter.done():
            e.waiter.set_exception(exc)

    # ---- windows, slots, expiry
    def refresh(self, now: float) -> None:
        """Pause jobs that left their window, give free slots to those back in, expire the late ones.

        An owner that goes from no runnable job back to runnable (its window reopened, a slot came free) is
        lifted like a new owner in ``register``, so a pause never banks turns; a job that leaves a pause also
        restarts its aging clock (it was not waiting for a turn while paused).
        """
        before = self.runnable_owners()
        for e in self.entries.values():
            if e.deadline is not None and now >= e.deadline:
                self.interrupt(e.id, JobExpired(EXPIRED))
            if e.window is not None and not e.window.contains(now):
                e.pause = "off_hours"
            elif e.pause == "off_hours":
                e.pause = "active_limit"
        for e in sorted(
            (x for x in self.entries.values() if x.pause == "active_limit"), key=lambda x: self.key(x, now)
        ):
            if self.has_slot():
                e.pause = None
                e.ref = now
        self._lift_returning(before)

    def _lift_returning(self, before: set[str]) -> None:
        after = self.runnable_owners()
        still = before & after
        for owner in sorted(after - before):
            self._lift(owner, still)

    def waiting(self, job_id: str) -> str | None:
        """``waiting_reason`` of a job that is in the scheduler."""
        e = self.entries.get(job_id)
        if e is None:
            return None
        return e.pause or ("turn" if e.waiter is not None and not e.waiter.done() else None)

    def next_edge(self, now: float) -> float | None:
        """The next moment (after ``now``) a window opens / closes or a job expires, for the idle timer."""
        edges = []
        for e in self.entries.values():
            if e.deadline is not None and e.deadline > now:
                edges.append(e.deadline)
            if e.window is not None:
                edges.append((e.window.next_start(now) if e.pause else e.window.next_end(now)) or now)
        return min((x for x in edges if x > now), default=None)

    # ---- the turn gate
    def dispatch(self) -> None:
        if self.inflight is not None:
            return
        now = self.clock()
        ready = [
            e
            for e in self.entries.values()
            if e.pause is None and e.waiter is not None and not e.waiter.done()
        ]
        if not ready:
            return
        e = min(ready, key=lambda x: self.key(x, now))
        self.inflight = e.id
        self.vtime = max(self.vtime, self.passes.get(e.owner, 0.0))
        assert e.waiter is not None
        e.waiter.set_result(None)

    async def turn(self, job_id: str) -> None:
        """Wait until this job may make one engine call (a job the scheduler does not know passes at once)."""
        e = self.entries.get(job_id)
        if e is None:
            return
        now = self.clock()
        if e.deadline is not None and now >= e.deadline:
            raise JobExpired(EXPIRED)
        self.refresh(now)
        e.waiter = asyncio.get_running_loop().create_future()
        self.dispatch()
        try:
            await e.waiter
        except BaseException:
            if self.inflight == job_id:  # granted, but the job was cancelled before it used the turn
                self.inflight = None
                self.dispatch()
            raise
        finally:
            e.waiter = None

    def done(self, job_id: str, cost: int) -> None:
        """The engine call is over: charge ``cost`` records to the owner and give the next turn."""
        e = self.entries.get(job_id)
        if e is None:
            return
        if self.inflight == job_id:
            self.inflight = None
        self.passes[e.owner] = self.passes.get(e.owner, 0.0) + max(1, cost)
        e.ref = self.clock()
        self.refresh(e.ref)
        self.dispatch()


# ------------------------------------------------------------------ job handle given to kinds


class JobHandle:
    """Handle given to ``run_fn``; ``set_total``, ``add_items``, ``decide``, ``cancelled`` are its API."""

    def __init__(self, runner: JobRunner, row: dict[str, Any]):
        self._runner = runner
        self.id: str = row["id"]
        self.kind: str = row["kind"]
        self.metadata: dict[str, Any] | None = row["metadata"]
        self.done: int = row["done"]  # rows already persisted (non-zero when resuming a resumable kind)
        self.resumed: bool = row["done"] > 0
        self.first_error: str | None = None  # first failed row's message in this run
        self.expired = False  # expire_after_s ran out (set when `cancelled` first notices)
        eligible, ttl = row.get("eligible_at"), row.get("expire_after_s")
        self._deadline: float | None = None if eligible is None or ttl is None else eligible + ttl
        self._hits = 0  # cache hits among the chunk being decided (not charged: see add_items)
        self._seen = 0  # rows added in this run (circuit breaker)
        self._bad = 0
        self._errors: Counter[str] = Counter()

    @property
    def resume_from(self) -> int:
        return self.done

    @property
    def cancelled(self) -> bool:
        if self.id in self._runner.cancel_ids:
            return True
        if self._deadline is not None and self._runner.clock() >= self._deadline:
            self.expired = True
        return self.expired

    def set_total(self, n: int) -> None:
        self._runner.store.set_total(self.id, int(n))

    async def decide(self, reqs: list[SystemOneRequest]) -> list[Any]:
        """``decide_resilient`` with the scheduler's turn gate: kinds call this, never ``ctx.decide``."""
        return await decide_resilient(self._runner.ctx, self, reqs)

    async def engine(self, part: list[SystemOneRequest]) -> list[Any]:
        """ONE engine call, inside this job's turn. Cache hits are counted (they are not charged)."""
        ctx, sched = self._runner.ctx, self._runner.sched
        try:
            await sched.turn(self.id)
        except JobExpired:
            self.expired = True
            raise
        try:
            if ctx.prepare is not None and ctx.decide_prepared is not None:
                prepared = await ctx.prepare(part, batch=len(part) > 1, label="items")
                out = await ctx.decide_prepared(prepared)
                self._hits += len(prepared.hits)
                return out
            return await ctx.decide(part, batch=len(part) > 1, label="items")
        finally:
            sched.done(self.id, len(part))

    async def engine_prepared(self, prepared: Any) -> list[Any]:
        """ONE engine call for records already encoded (F2's window sorter), inside this job's turn."""
        ctx, sched = self._runner.ctx, self._runner.sched
        try:
            await sched.turn(self.id)
        except JobExpired:
            self.expired = True
            raise
        try:
            return await ctx.decide_prepared(prepared, priority="background")
        finally:
            sched.done(self.id, len(prepared.records))

    def note_hits(self, n: int) -> None:
        """Cache hits served outside ``engine`` (a sorted window's): not charged, like ``engine``'s own."""
        self._hits += n

    async def add_items(self, items: list[dict[str, Any]], tokens: int = 0) -> None:
        """Persist rows in order; progress advances by ``len(items)``; a row with "error" counts as failed.

        ``tokens`` adds to ``jobs.tokens``. Cache hits found while deciding these rows are handed back to
        the daily quota once the rows are stored.
        """
        if not items:
            return
        hits, self._hits = self._hits, 0
        self.done = await asyncio.to_thread(self._runner.store.add_items, self.id, list(items), tokens, hits)
        if hits:
            self._runner.credit_hits(self.id, hits)
        self._runner.notify_progress(self.id)
        self._trip_breaker(items)

    def _trip_breaker(self, items: list[dict[str, Any]]) -> None:
        """Abort the job when failures look systemic (rows stay stored): every one of the first
        BREAKER_MIN_ITEMS items failed, or >= BREAKER_SAME_SHARE of >= BREAKER_SAME_MIN failed alike."""
        for it in items:
            self._seen += 1
            if isinstance(it, dict) and "error" in it and not 400 <= (it.get("status_code") or 0) < 500:
                # F1: a row that carries a 4xx status_code is the client's mistake (a bad batch line), not a
                # systemic failure; it still counts as failed in the job, but never trips the breaker.
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


def _entry(row: dict[str, Any], scheduler: Scheduler) -> Entry:
    """The scheduler's view of a job row (``JobStore.runnable`` or a claimed row)."""
    sched, cls = row.get("schedule"), int(row["priority"])
    window = None
    if sched and cls < PRIORITY_URGENT:  # urgent ignores the window
        try:
            window = parse_window(sched["window"], sched["tz"])
        except (ValueError, KeyError):  # a zone that vanished from tzdata: run rather than never run
            log.warning("jobs: job %s has an unusable schedule %r; ignoring the window", row["id"], sched)
    ttl, eligible = row.get("expire_after_s"), row.get("eligible_at")
    return Entry(
        id=row["id"],
        owner=row["owner"] or "",
        cls=cls,
        seq=row["seq"],
        interrupted=row["status"] == "interrupted",
        window=window,
        deadline=None if ttl is None or eligible is None else eligible + ttl,
        ref=eligible if eligible is not None else row["created_at"],
    )


class JobRunner:
    def __init__(
        self,
        ctx: AppContext,
        store: JobStore,
        dispatcher: WebhookDispatcher,
        clock: Callable[[], float] = time.time,
    ):
        self.ctx, self.store, self.dispatcher = ctx, store, dispatcher
        self.clock = clock  # unix seconds; tests pass a fake one (windows, aging, expiry)
        self.policy: WebhookPolicy = dispatcher.policy
        cfg = ctx.cfg
        self.sched = Scheduler(cfg.job_active, cfg.job_aging_s, lambda: self.clock())
        self.cancel_ids: set[str] = set()
        self.wake: asyncio.Event | None = None  # created on start (needs the running loop)
        self._progress_ids: set[str] = set()  # running jobs whose webhook wants job.progress
        self._task: asyncio.Task[None] | None = None
        self._purge_task: asyncio.Task[None] | None = None
        self.tasks: dict[str, asyncio.Task[None]] = {}  # job id -> its task (every active job)
        self._failures: dict[str, int] = {}  # job id -> consecutive runner-level failures
        self._hits: dict[str, int] = {}  # job id -> cache hits already handed back to the quota
        self._rows: dict[str, dict[str, Any]] = {}  # job id -> its claimed row (owner, quota day)
        self.current: set[str] = set()  # jobs running a kind right now

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
        await self.settle()  # jobs that ended failed / cancelled while the server was down
        await asyncio.to_thread(self._purge)
        for job_id, event in await asyncio.to_thread(self.store.pending_deliveries):
            self.dispatcher.enqueue(job_id, event)
        self._task = asyncio.create_task(self._loop(), name="clef-jobs")
        self._purge_task = asyncio.create_task(self._purge_loop(), name="clef-jobs-purge")

    async def stop(self) -> None:
        tasks = [t for t in (self._task, self._purge_task, *self.tasks.values()) if t]
        for t in tasks:
            t.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)  # each job task parks its own job
        # Safety net for a clean stop that landed outside the kind (e.g. right after claim): park the
        # job without counting a resume. A store call still in a worker thread holds the lock: this waits.
        with contextlib.suppress(Exception):
            await asyncio.to_thread(self.store.park_running)
        await self.dispatcher.close()  # the database itself is closed by the app lifespan (ctx.db)

    def poke(self) -> None:
        if self.wake is not None:
            self.wake.set()

    async def settle(self, job_id: str | None = None) -> None:
        """Refund the unprocessed records of a failed / cancelled job (all if ``job_id`` is None)."""
        quotas = self.ctx.quotas
        if quotas is None:
            return
        hits = {job_id: self._hits.pop(job_id, 0)} if job_id is not None else None
        for owner, day, n in await asyncio.to_thread(self.store.settle_quota, job_id, hits):
            quotas.refund(Reservation(owner, day, n), n)

    def credit_hits(self, job_id: str, n: int) -> None:
        """``n`` cache hits of a stored chunk are not charged: hand them back to the job's reservation."""
        row = self._rows.get(job_id)
        if row is None or self.ctx.quotas is None or not row.get("quota_day") or not row.get("owner"):
            return
        self._hits[job_id] = self._hits.get(job_id, 0) + n
        self.ctx.quotas.refund(Reservation(row["owner"], row["quota_day"], n), n)

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

    # ---- scheduling
    async def _loop(self) -> None:
        assert self.wake is not None
        while True:
            self.wake.clear()  # a poke that lands during the tick makes the next wait return at once
            wait_s = RUNNER_RETRY_PAUSE_S
            try:
                wait_s = await self._tick()
            except asyncio.CancelledError:
                raise
            except Exception:
                log.exception("job scheduler error")
            with contextlib.suppress(asyncio.TimeoutError):
                await asyncio.wait_for(self.wake.wait(), timeout=wait_s)

    async def _tick(self) -> float:
        """Apply windows / expiry to the active jobs, admit waiting ones into free slots; seconds to sleep."""
        now = self.clock()
        self.sched.refresh(now)
        self.sched.dispatch()
        rows = [r for r in await asyncio.to_thread(self.store.runnable) if r["id"] not in self.tasks]
        edges: list[float] = []
        ready: list[Entry] = []
        for row in rows:
            e = _entry(row, self.sched)
            if e.deadline is not None:
                if now >= e.deadline:
                    await self._expire_queued(e.id)
                    continue
                edges.append(e.deadline)
            if e.window is None or e.window.contains(now):
                ready.append(e)
            else:
                edges.append(e.window.next_start(now) or now)
        while ready:
            e = min(ready, key=lambda x: self.sched.key(x, now))
            if not self.sched.has_slot() and e.cls < PRIORITY_URGENT:
                break  # ordered by class first: nothing behind it may jump the queue either
            ready.remove(e)
            await self._admit(e)
        nxt = self.sched.next_edge(now)
        if nxt is not None:
            edges.append(nxt)
        return max(0.05, min([SCHEDULER_MAX_SLEEP_S, *(x - now for x in edges)]))

    async def _expire_queued(self, job_id: str) -> None:
        changed, event = await asyncio.to_thread(self.store.cancel_queued, job_id, EXPIRED)
        if changed:
            log.info("jobs: job %s expired before it could finish", job_id)
            if event:
                self.dispatcher.enqueue(job_id, event)
            await self.settle(job_id)

    async def _admit(self, e: Entry) -> None:
        try:
            row = await asyncio.to_thread(self.store.claim, e.id)  # the job is 'running' from here
        except BaseException:
            # The claim may have committed before it raised: put the job back (only this id; the other
            # active jobs are running and must not be parked).
            if e.id not in self.tasks:
                with contextlib.suppress(Exception):
                    await asyncio.to_thread(self.store.park, e.id)
            raise
        if row is None:
            return  # cancelled or taken meanwhile
        self._rows[e.id] = row
        e.interrupted = row["claimed_from"] == "interrupted"
        self.sched.register(e)
        task = asyncio.create_task(self._run(row), name=f"clef-job-{e.id}")
        self.tasks[e.id] = task
        task.add_done_callback(lambda _t, job_id=e.id: self._ended(job_id))

    def _ended(self, job_id: str) -> None:
        self.tasks.pop(job_id, None)
        self._rows.pop(job_id, None)
        self.poke()

    async def _run(self, row: dict[str, Any]) -> None:
        job_id = row["id"]
        try:
            await self._execute(row)
            self._failures.pop(job_id, None)
        except asyncio.CancelledError:
            raise
        except Exception:
            await self._runner_failed(row)
        finally:
            self.sched.unregister(job_id)
            self.current.discard(job_id)
            self._progress_ids.discard(job_id)

    async def _runner_failed(self, row: dict[str, Any]) -> None:
        """The runner itself (not the kind) raised: retry a few times, then fail the job (poison pill)."""
        log.exception("job runner error")
        job_id = row["id"]
        n = self._failures[job_id] = self._failures.get(job_id, 0) + 1
        try:
            # Left 'running'? Park it again (not a crash: RUNNER_MAX_FAILURES is the budget here).
            event = await asyncio.to_thread(self.store.park, job_id, job_id in self.cancel_ids)
            if event:
                self.dispatcher.enqueue(job_id, event)
                await self.settle(job_id)
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

    # ---- submit / cancel / get (the Python API; POST /v1/jobs and F1 use it)
    async def submit_ex(
        self,
        kind: str,
        payload: dict[str, Any],
        *,
        principal: Any = None,
        metadata: dict[str, Any] | None = None,
        webhook: dict[str, Any] | None = None,
        priority: str = "normal",
        urgent: bool = False,
        schedule: dict[str, Any] | None = None,
        expire_after_s: float | None = None,
        idem_key: str | None = None,
    ) -> tuple[dict[str, Any], bool]:
        """Validate, reserve the daily quota, queue: ``(public job, replayed)``. See :meth:`submit`."""
        ctx, cfg = self.ctx, self.ctx.cfg
        owner = principal.name if principal is not None else None
        if idem_key is not None:
            if not IDEMPOTENCY_KEY_RE.match(idem_key):
                raise ValueError("Idempotency-Key: 1-128 printable ASCII characters")
            known = await asyncio.to_thread(self.store.find_idem, owner, idem_key)
            if known:  # a replay returns the first job and is not charged again
                return public_job(known, waiting=self.waiting(known)), True
        kinds = self.kinds()
        if not KIND_RE.match(kind) or kind not in kinds:
            raise ValueError(f"kind: unknown job kind {kind[:64]!r} (available: {', '.join(sorted(kinds))})")
        if metadata is not None and len(json.dumps(metadata)) > MAX_METADATA_BYTES:
            raise ValueError(f"metadata: too large (max {MAX_METADATA_BYTES // 1024} KB)")
        if priority not in PRIORITY_NAMES[:3]:
            raise ValueError(f"priority: {priority!r} is not one of low, normal, high")
        if urgent and principal is not None and not principal.has("urgent"):
            raise PermissionError(f"key {principal.name!r} lacks scope 'urgent'")
        cls = PRIORITY_URGENT if urgent else PRIORITY_NAMES.index(priority)
        stored, window = self._schedule(schedule)
        if webhook is not None:
            try:
                self.policy.check_url(webhook["url"])
            except WebhookRefused as exc:
                raise ValueError(f"webhook.url: {exc}") from exc
        spec = kinds[kind]
        for requested in payload_models(payload):  # a model this server does not serve: 404 before anything
            ctx.check_model(requested)  # LookupError
        try:
            parsed = await asyncio.to_thread(spec["validate"], payload)
            if spec.get("snapshot"):  # saved classifiers are copied into the payload now (see docs/jobs.md)
                payload = await asyncio.to_thread(spec["snapshot"], payload, parsed)
        except (ValueError, ValidationError) as exc:
            raise ValueError(f"payload.{_msg(exc)}") from exc
        now = self.clock()
        eligible = now
        if window is not None and cls < PRIORITY_URGENT and not window.contains(now):
            eligible = window.next_start(now) or now  # expiry counts from when it could first run
        reservation = ctx.quotas.reserve(principal, job_records(spec, parsed, payload))  # QuotaExceeded
        day, reserved = (reservation.day, reservation.n) if reservation else (None, 0)
        try:  # the whole job is charged to the submit day; the unprocessed part comes back when it ends
            row = await asyncio.to_thread(
                self.store.create, kind, owner, payload, metadata, webhook, cfg.max_jobs, idem_key,
                ctx.model_ref(), day, reserved, cls, stored, expire_after_s, eligible,
            )  # fmt: skip
        except BaseException:  # JobsFull, a DB error: nothing was queued, charge nothing
            if reservation is not None:
                ctx.quotas.refund(reservation, reservation.n)
            raise
        replayed = bool(row.get("replayed"))
        if replayed:  # lost an idempotency race: the other job already holds its reservation
            if reservation is not None:
                ctx.quotas.refund(reservation, reservation.n)
        else:
            self.poke()
        return public_job(row, waiting=self.waiting(row)), replayed

    async def submit(self, kind: str, payload: dict[str, Any], **kw: Any) -> dict[str, Any]:
        """Queue a job and return its public view.

        Keyword arguments: ``principal`` (owner and scopes; None = auth off), ``metadata``, ``webhook``,
        ``priority`` (low | normal | high), ``urgent``, ``schedule`` (``{"window", "tz"?}``),
        ``expire_after_s``, ``idem_key``. Raises ValueError (400), PermissionError (403, urgent without the
        scope), QuotaExceeded (429), JobsFull (409), LookupError (404, a model this server does not serve).
        """
        return (await self.submit_ex(kind, payload, **kw))[0]

    def _schedule(self, schedule: dict[str, Any] | None) -> tuple[dict[str, str] | None, Window | None]:
        """The window a new job gets (its own, else ``CLEF_JOB_WINDOW``), as stored and parsed."""
        cfg = self.ctx.cfg
        spec = (schedule or {}).get("window") or cfg.job_window or None
        tz = (schedule or {}).get("tz") or cfg.job_tz
        try:
            window = parse_window(spec, tz)
        except ValueError as exc:
            raise ValueError(f"schedule: {exc}") from exc
        return (None if window is None else {"window": spec, "tz": tz}), window  # type: ignore[dict-item]

    def get(self, job_id: str) -> dict[str, Any] | None:
        row = self.store.get(job_id)
        return None if row is None else public_job(row, waiting=self.waiting(row))

    def waiting(self, row: dict[str, Any]) -> tuple[str | None, float | None]:
        """``(waiting_reason, next_window_at)`` of a job row: why it is not advancing right now."""
        status = row["status"]
        if status not in ACTIVE:
            return None, None
        now = self.clock()
        e = self.sched.entries.get(row["id"])
        if e is not None:
            reason = self.sched.waiting(row["id"])
            nxt = e.window.next_start(now) if e.window is not None and reason == "off_hours" else None
            return reason, nxt
        if status != "running" and row["priority"] < PRIORITY_URGENT and row["schedule"]:
            try:
                window = parse_window(row["schedule"]["window"], row["schedule"]["tz"])
            except (ValueError, KeyError):
                window = None
            if window is not None and not window.contains(now):
                return "off_hours", window.next_start(now)
        if status != "running" and not self.sched.has_slot() and row["priority"] < PRIORITY_URGENT:
            return "active_limit", None
        return None, None

    async def cancel(self, job_id: str) -> dict[str, Any]:
        """Cancel a queued or running job (KeyError: no such job); returns its public view."""
        row = await asyncio.to_thread(self.store.get, job_id)
        if row is None:
            raise KeyError(job_id)
        return public_job(await self.request_cancel(row), waiting=self.waiting(row))

    # ---- cancel / webhooks
    async def request_cancel(self, row: dict[str, Any]) -> dict[str, Any]:
        job_id = row["id"]
        cancelled, event = await asyncio.to_thread(self._cancel_queued_sync, job_id)
        if cancelled:
            if event:
                self.dispatcher.enqueue(job_id, event)
            await self.settle(job_id)
        else:
            # Not queued: flag it if running, or cancel it now if a park (clean stop, runner error) got
            # there first. One atomic store call, not gated on a stale read: a racing park cannot drop it.
            self.cancel_ids.add(job_id)  # before the flag: a running job may check either one
            status, event = await asyncio.to_thread(self.store.flag_cancel, job_id)
            if status != "running":
                self.cancel_ids.discard(job_id)
            else:
                self.sched.interrupt(
                    job_id, JobCancelled()
                )  # a job paused off-hours must not wait for its window
            if event:
                self.dispatcher.enqueue(job_id, event)
            if status == "cancelled":
                await self.settle(job_id)
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
        if job_id in self._progress_ids:
            self.dispatcher.progress(job_id, True)

    async def redeliver(self, job_id: str, event: str) -> bool:
        """failed -> pending -> delivered again (same delivery id). False when that event did not fail."""
        if await asyncio.to_thread(self.store.redeliver, job_id, event) is None:
            return False
        self.dispatcher.enqueue(job_id, event)
        return True

    # ---- execution
    async def _execute(self, row: dict[str, Any]) -> None:
        """Run a job ``claim`` already set 'running' (a cancel may target it from that moment on)."""
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
        self.current.add(job_id)
        if "job.progress" in ((row["webhook"] or {}).get("events") or ()):
            self._progress_ids.add(job_id)
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
            self.current.discard(job_id)
        cancelled = job_id in self.cancel_ids
        self.cancel_ids.discard(job_id)
        expired = handle.expired and not cancelled  # the user's cancel wins over the clock
        status, error = ("cancelled" if cancelled or expired else "succeeded"), (EXPIRED if expired else None)
        if status == "succeeded" and not spec.get("completed_if_all_failed"):
            # A kind with "completed_if_all_failed" (batch, OpenAI semantics) skips this check.
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
        if status in ("failed", "cancelled"):
            await self.settle(job_id)


def _msg(exc: Exception) -> str:
    if isinstance(exc, ValidationError):
        return format_errors(list(exc.errors()))
    return str(exc)


# ------------------------------------------------------------------ chunked execution (built-in kinds)


class ItemError:
    def __init__(self, message: str):
        self.message = message[:500]


async def decide_resilient(ctx: AppContext, job: JobHandle, reqs: list[SystemOneRequest]) -> list[Any]:
    """Results for ``reqs`` (one engine call when everything is fine). Kinds call ``job.decide(reqs)``.

    Each engine call takes a turn from the scheduler (``JobHandle.engine``); none is held while a call
    waits for an unavailable engine or pauses after an OOM.

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
                return await job.engine(part)  # waits for the job's turn; one engine call
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
    sorter = jobsort.windowed(ctx, n, build, decide_resilient)  # F2: None unless a flag is on
    size = sorter.step if sorter else max(1, int(ctx.cfg.max_microbatch))
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
        results = iter([])
        if good:
            results = iter(await (sorter.decide(job, i, good) if sorter else job.decide(good)))
        rows = []
        for k, r in built:
            out = r if isinstance(r, ItemError) else next(results)
            row = {"index": k, "error": out.message} if isinstance(out, ItemError) else shape(k, out)
            tally(row)
            rows.append(row)
        await job.add_items(rows)
        i = j
        await asyncio.sleep(0)  # let interactive requests reach the engine queue before the next chunk


def job_records(spec: dict[str, Any], parsed: Any, payload: dict[str, Any]) -> int:
    """How many records a job will run, for the daily quota (>= 1).

    A kind may give ``"records": callable(parsed) -> int``; else the length of the parsed ``inputs`` /
    ``requests`` / ``rows`` (the built-in kinds and ``evaluate``), else of the same payload keys.
    """
    if callable(spec.get("records")):
        return max(1, int(spec["records"](parsed)))
    for name in ("inputs", "requests", "rows"):
        for src in (getattr(parsed, name, None), payload.get(name)):
            if isinstance(src, list):
                return max(1, len(src))
    return 1


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

    api = APIRouter(prefix="/v1/jobs", tags=["jobs"], dependencies=ctx.scope("jobs"), responses=ctx.errors)

    def caller(request: Request) -> str | None:
        return getattr(request.state, "key_name", None)

    def scoped(request: Request) -> bool:
        """Jobs are private to their owner when auth is on; admins see every job, owner-less ones included."""
        principal = ctx.principal(request)
        return ctx.auth_enabled() and not (principal is not None and principal.has("admin"))

    async def audit_job(request: Request, action: str, row: dict[str, Any]) -> None:
        def write() -> None:
            with ctx.db.tx() as conn:
                audit(
                    conn,
                    caller(request) or "anonymous",
                    action,
                    row["id"],
                    {"kind": row["kind"], "owner": row["owner"]},
                    request.client.host if request.client else None,
                    ctx.request_id(request),
                )

        await run_in_threadpool(write)

    async def load(request: Request, job_id: str) -> dict[str, Any]:
        row = await run_in_threadpool(store.get, job_id)
        # Someone else's job is indistinguishable from a missing one.
        if row is None or (scoped(request) and row["owner"] != caller(request)):
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
        principal = ctx.principal(request)
        try:
            info, replayed = await runner.submit_ex(
                body.kind,
                body.payload,
                principal=principal,
                metadata=body.metadata,
                webhook=None if body.webhook is None else body.webhook.model_dump(exclude_none=True),
                priority=body.priority,
                urgent=body.urgent,
                schedule=None if body.schedule is None else body.schedule.model_dump(exclude_none=True),
                expire_after_s=body.expire_after_s,
                idem_key=idempotency_key,
            )
        except QuotaExceeded as exc:
            request.state.quota = (exc.limit, exc.remaining)
            raise ApiError(
                429, str(exc), {"Retry-After": str(exc.retry_after)}, code="insufficient_quota"
            ) from exc
        except PermissionError as exc:
            raise ApiError(403, str(exc)) from exc
        except JobsFull as exc:
            raise ApiError(409, str(exc), {"Retry-After": "30"}) from exc
        except (KeyError, IndexError):
            raise
        except LookupError as exc:  # a model this server does not serve
            raise ApiError(404, str(exc)) from exc
        except ValueError as exc:
            raise ApiError(400, str(exc)) from exc
        request.state.quota = ctx.quotas.remaining(principal)
        if replayed:
            response.headers["Idempotent-Replay"] = "true"
        response.headers["Location"] = f"/v1/jobs/{info['id']}"
        return info

    @api.get("", response_model=JobList, summary="List jobs (newest first)")
    async def list_jobs(
        request: Request,
        status: Literal["queued", "running", "succeeded", "failed", "cancelled", "interrupted"] | None = None,
        kind: str | None = None,
        priority: Literal["low", "normal", "high", "urgent"] | None = None,
        limit: int = Query(50, ge=1, le=200),
        offset: int = Query(0, ge=0),
    ) -> dict[str, Any]:
        rows, total = await run_in_threadpool(
            store.list,
            caller(request),
            scoped(request),
            status,
            kind,
            limit,
            offset,
            None if priority is None else PRIORITY_NAMES.index(priority),
        )
        now = time.time()
        jobs = [public_job(r, now, runner.waiting(r)) for r in rows]
        return {"jobs": jobs, "total": total, "limit": limit, "offset": offset}

    @api.get("/{job_id}", response_model=JobInfo, summary="Job status, progress, ETA and result summary")
    async def get_job(job_id: str, request: Request) -> dict[str, Any]:
        row = await load(request, job_id)
        return public_job(row, waiting=runner.waiting(row))

    @api.post("/{job_id}/cancel", response_model=JobInfo, summary="Cancel a queued or running job")
    async def cancel_job(job_id: str, request: Request) -> dict[str, Any]:
        row = await load(request, job_id)
        if row["status"] in FINISHED:
            raise ApiError(409, f"job is already {row['status']}")
        fresh = await runner.request_cancel(row)
        out = public_job(fresh, waiting=runner.waiting(fresh))
        await audit_job(request, "job.cancel", row)
        return out

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
        fresh = await run_in_threadpool(store.get, job_id) or row
        return public_job(fresh, waiting=runner.waiting(fresh))

    @api.delete("/{job_id}", response_model=JobDeleted, summary="Delete a finished job and its results")
    async def delete_job(job_id: str, request: Request) -> dict[str, str]:
        row = await load(request, job_id)
        if row["status"] in ACTIVE:
            raise ApiError(409, f"job is {row['status']}; cancel it first (POST /v1/jobs/{job_id}/cancel)")

        def remove() -> None:
            with ctx.db.tx() as conn:  # the delete and its audit row commit together
                store.delete(job_id)
                audit(
                    conn,
                    caller(request) or "anonymous",
                    "job.delete",
                    job_id,
                    {"kind": row["kind"], "owner": row["owner"]},
                    request.client.host if request.client else None,
                    ctx.request_id(request),
                )

        await run_in_threadpool(remove)
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
