"""Job throughput (F2): length sort inside a window, and tokenizing ahead of the GPU. Both are opt-in.

``CLEF_JOB_SORT_WINDOW=N`` (N > 0): a job takes N items at a time, encodes them all (``ctx.prepare``, CPU
threads), sorts the computed ones by token length and sends them to the engine in micro-batches of
``CLEF_MAX_MICROBATCH``, so every forward carries similar lengths (less padding, fewer bucket groups per
micro-batch). The results go back to input order and the caller persists the whole window in ONE
``add_items``: rows stay contiguous ``0..done-1``, a crash or clean stop mid-window leaves ``done`` at the
window start, and a resume recomputes at most one window. Nothing is persisted out of order.

``CLEF_TOKENIZE_AHEAD=1``: while a window is on the GPU, the next one is already being encoded on CPU threads
(a task on ``ctx.prepare``). It only feeds the existing queue: there is still ONE GPU worker and ONE job
chunk in flight, and interactive requests keep going first through the engine lanes.

With both flags off ``windowed`` returns ``None`` and the callers run their original loop untouched. Anything
unusual (a record the engine rejects, a GPU OOM, engine not ready, a failed prefetch) falls back to the
caller's ``decide_resilient`` for just those records, so error isolation is the same as without these flags.

The hand-off to the engine (``_submit``) is the single place a job scheduler's turn gate must wrap.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable
from typing import Any

from .engine import EngineNotReady, GpuOutOfMemory, Prepared, _as_hit

log = logging.getLogger("clef.jobsort")

# Rejections that the resilient path knows how to isolate or wait out.
_RECOVERABLE = (ValueError, GpuOutOfMemory, EngineNotReady)

Decide = Callable[[Any, Any, list[Any]], Awaitable[list[Any]]]  # decide_resilient(ctx, job, reqs)


def length_key(prepared: Prepared, i: int) -> tuple[int, int]:
    """Sort key of record ``i``: (media last, token length). Cache hits and duplicates cost nothing (0)."""
    enc = prepared.encoded[i]
    if enc is not None and hasattr(enc, "length"):
        return (1 if getattr(enc, "is_media", False) else 0, int(enc.length))
    if i in prepared.hits or i in prepared.dups:
        return (0, 0)
    rec = prepared.records[i]  # engines without prepare (test fakes): characters as a proxy
    return (0, len(str(rec.get("state"))) + len(str(rec.get("questions"))))


def sorted_order(prepared: Prepared) -> list[int]:
    """Indices that need a forward (no cache hit, no duplicate), shortest first; ties keep input order."""
    todo = [i for i in range(len(prepared)) if i not in prepared.hits and i not in prepared.dups]
    todo.sort(key=lambda i: length_key(prepared, i))  # list.sort is stable
    return todo


def pick(prepared: Prepared, idx: list[int]) -> Prepared:
    """The records at ``idx`` as their own Prepared (no hits, no dups: only what needs a forward)."""
    return Prepared(
        [prepared.records[i] for i in idx],
        [prepared.keys[i] for i in idx],
        {},
        [prepared.encoded[i] for i in idx],
        {},
    )


def merge(prepared: Prepared, computed: dict[int, Any]) -> list[Any]:
    """Results in input order: computed ones, cache hits, duplicates copied from their first occurrence."""
    out: list[Any] = [None] * len(prepared)
    out_hits = prepared.hits
    for i in range(len(prepared)):
        if i in computed:
            out[i] = computed[i]
        elif i in out_hits:
            out[i] = out_hits[i]
    for i, src in prepared.dups.items():
        first = computed[src]
        out[i] = _as_hit(first) if isinstance(first, dict) else first  # an ItemError is shared as it is
    return out


class WindowSorter:
    """Windowed engine feed for one job run. Create it with :func:`windowed`."""

    def __init__(
        self, ctx: Any, n: int, build: Callable[[int], Any], fallback: Decide, window: int, ahead: bool
    ):
        self.ctx, self.n, self.build, self.fallback = ctx, n, build, fallback
        self.micro = max(1, int(ctx.cfg.max_microbatch))
        self.sort = window > 0
        # step = items per window: the whole sort window, or one micro-batch when only tokenizing ahead
        self.step = max(self.micro, window) if self.sort else self.micro
        self.ahead = ahead
        self._next: tuple[int, int, asyncio.Task[Prepared]] | None = None  # (start, n_reqs, prepare task)

    # ---- encoding
    def _encode(self, reqs: list[Any]) -> asyncio.Task[Prepared]:
        task = asyncio.ensure_future(self.ctx.prepare(reqs, batch=len(reqs) > 1, label="items"))
        # a prefetch nobody awaits (job cancelled or stopped) must not log "exception was never retrieved"
        task.add_done_callback(lambda t: t.cancelled() or t.exception())
        return task

    def _good(self, start: int) -> list[Any]:
        """The requests of the window at ``start`` that build (what the caller will pass to ``decide``)."""
        out = []
        for k in range(start, min(self.n, start + self.step)):
            try:
                out.append(self.build(k))
            except ValueError:  # includes pydantic's ValidationError; the caller records the ItemError
                continue
        return out

    def _prefetch(self, start: int) -> None:
        if not self.ahead or start >= self.n or self._next is not None:
            return
        reqs = self._good(start)
        if reqs:
            self._next = (start, len(reqs), self._encode(reqs))

    # ---- the one call per window
    async def decide(self, job: Any, start: int, reqs: list[Any]) -> list[Any]:
        """Results for ``reqs`` (the window at ``start``, already built), in the same order."""
        task = None
        if self._next is not None:
            nstart, count, ntask = self._next
            self._next = None
            if nstart == start and count == len(reqs):
                task = ntask
            else:
                ntask.cancel()
        if task is None:
            task = self._encode(reqs)
        try:
            prepared = await task
        except _RECOVERABLE as exc:
            log.debug("window encode failed (%s); falling back to the per-record path", exc)
            prepared = None
        self._prefetch(start + self.step)  # encode the next window while this one is on the GPU
        if prepared is None:
            return await self._plain(job, reqs)
        computed: dict[int, Any] = {}
        order = (
            sorted_order(prepared)
            if self.sort
            else [i for i in range(len(prepared)) if i not in prepared.hits and i not in prepared.dups]
        )
        for s in range(0, len(order), self.micro):
            idx = order[s : s + self.micro]
            if job.cancelled:
                from .jobs import JobCancelled  # late: jobs imports this module

                raise JobCancelled
            try:
                res = await self._submit(job, pick(prepared, idx))
            except _RECOVERABLE:
                res = await self.fallback(self.ctx, job, [reqs[i] for i in idx])  # isolate / wait / bisect
            computed.update(zip(idx, res, strict=True))
            await asyncio.sleep(0)  # let interactive requests reach the engine queue
        return merge(prepared, computed)

    async def _plain(self, job: Any, reqs: list[Any]) -> list[Any]:
        out: list[Any] = []
        for s in range(0, len(reqs), self.micro):
            out.extend(await self.fallback(self.ctx, job, reqs[s : s + self.micro]))
        return out

    async def _submit(self, job: Any, sub: Prepared) -> list[Any]:
        """Queue one micro-batch in the background lane and await it. A scheduler's turn gate wraps this."""
        return await self.ctx.decide_prepared(sub, priority="background")


def windowed(ctx: Any, n: int, build: Callable[[int], Any], fallback: Decide) -> WindowSorter | None:
    """A sorter for a job run, or ``None`` when both flags are off (or the context cannot prepare)."""
    cfg = ctx.cfg
    window = int(getattr(cfg, "job_sort_window", 0))
    ahead = bool(getattr(cfg, "tokenize_ahead", False))
    if window <= 0 and not ahead:
        return None
    if getattr(ctx, "prepare", None) is None or getattr(ctx, "decide_prepared", None) is None:
        return None
    return WindowSorter(ctx, n, build, fallback, window, ahead)
