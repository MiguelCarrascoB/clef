"""Thread-safe request log, latency histogram, counters, forward stats and a 1-second-bucket time series
(engine thread + event loop + sampler)."""

from __future__ import annotations

import asyncio
import contextlib
import threading
import time
from bisect import bisect_right
from collections import deque
from collections.abc import AsyncIterator, Callable, Iterator
from contextlib import contextmanager
from typing import Any

HIST_EDGES = [25, 50, 100, 150, 200, 300, 500, 1000, 2000, 5000]
SUBSCRIBER_QUEUE = 256
LAT_CAP_PER_SEC = 1000
GAUGE_KEYS = ("mem_used_gb", "gpu_util_pct", "gpu_temp_c", "gpu_power_w")
TS_WINDOW_MIN, TS_WINDOW_MAX, TS_MAX_POINTS = 10, 3600, 720


def _percentile(sorted_vals: list[float], q: float) -> float:
    if not sorted_vals:
        return 0.0
    rank = max(1, -(-len(sorted_vals) * q // 100))  # nearest-rank, ceil
    return sorted_vals[int(rank) - 1]


class _Bucket:
    """One second of activity. Replaced when its ring slot is reused by a later second."""

    __slots__ = ("sec", "reqs", "errs", "lats", "fwd", "recs", "toks", "pad", "qmax", "gauges")

    def __init__(self, sec: int):
        self.sec = sec
        self.reqs = 0
        self.errs = 0
        self.lats: list[float] = []
        self.fwd = 0
        self.recs = 0
        self.toks = 0
        self.pad = 0
        self.qmax: int | None = None
        self.gauges: dict[str, float] = {}


def _r(v: float | None, nd: int) -> float | None:
    return None if v is None else round(v, nd)


class Stats:
    """Request log + counters + a time series of 1-second buckets covering ``retention_s`` seconds.

    ``clock`` is injectable (defaults to ``time.time``) for deterministic tests.
    """

    def __init__(
        self,
        buffer: int = 1000,
        retention_s: int = 3600,
        clock: Callable[[], float] = time.time,
    ):
        self._clock = clock
        self._retention = max(1, int(retention_s))
        self._ring: list[_Bucket | None] = [None] * self._retention
        self._lock = threading.Lock()
        self._entries: deque[dict[str, Any]] = deque(maxlen=max(1, buffer))
        self._next_id = 1
        self._start = clock()
        self._total = 0
        self._errors = 0
        self._tokens = 0
        self._token_reqs = 0
        self._in_flight = 0
        self._queue_depth = 0
        self._fwd_count = 0
        self._fwd_records = 0
        self._fwd_ms = 0.0
        self._fwd_tokens = 0
        self._fwd_padded = 0
        self._subs: list[tuple[asyncio.AbstractEventLoop, asyncio.Queue[dict[str, Any]]]] = []

    # ---- time-series internals (call with the lock held)
    def _bucket(self, now: float) -> _Bucket:
        sec = int(now)
        i = sec % self._retention
        b = self._ring[i]
        if b is None or b.sec != sec:
            b = self._ring[i] = _Bucket(sec)
        return b

    @staticmethod
    def _note_queue(b: _Bucket, n: int) -> None:
        if b.qmax is None or n > b.qmax:
            b.qmax = n

    # ---- writers
    def record_request(self, entry: dict[str, Any]) -> None:
        with self._lock:
            now = self._clock()
            item = {"id": self._next_id, "ts": round(now, 3)}
            item.update(entry)
            item.setdefault("key", None)
            item.setdefault("media", {"images": 0, "videos": 0})
            item.setdefault("error", None)
            item.setdefault("state_preview", None)
            self._next_id += 1
            self._entries.append(item)
            self._total += 1
            b = self._bucket(now)
            b.reqs += 1
            if int(item.get("status", 200)) >= 400:
                self._errors += 1
                b.errs += 1
            elif item.get("ms") is not None and len(b.lats) < LAT_CAP_PER_SEC:
                b.lats.append(float(item["ms"]))
            tokens = int(item.get("input_tokens") or 0)
            if tokens:
                self._tokens += tokens
                self._token_reqs += 1
            subs = list(self._subs)
        for loop, queue in subs:
            with contextlib.suppress(RuntimeError):  # loop closed
                loop.call_soon_threadsafe(self._offer, queue, item)

    @staticmethod
    def _offer(queue: asyncio.Queue[dict[str, Any]], item: dict[str, Any]) -> None:
        with contextlib.suppress(asyncio.QueueFull):  # slow consumer: drop
            queue.put_nowait(item)

    def record_forward(self, n_records: int, n_tokens: int, padded_tokens: int, ms: float) -> None:
        with self._lock:
            self._fwd_count += 1
            self._fwd_records += n_records
            self._fwd_ms += ms
            self._fwd_tokens += n_tokens
            self._fwd_padded += padded_tokens
            b = self._bucket(self._clock())
            b.fwd += 1
            b.recs += n_records
            b.toks += n_tokens
            b.pad += padded_tokens

    @property
    def queue_depth(self) -> int:
        """Current engine queue depth (last set_queue_depth / sample)."""
        with self._lock:
            return self._queue_depth

    def set_queue_depth(self, n: int) -> None:
        with self._lock:
            self._queue_depth = n
            self._note_queue(self._bucket(self._clock()), n)

    def sample(self, gauges: dict[str, Any]) -> None:
        """Record a periodic gauge sample into the current second.

        Known keys: mem_used_gb, gpu_util_pct, gpu_temp_c, gpu_power_w, queue_depth. Unknown keys are ignored;
        None values are skipped (an earlier non-None value in the same second is kept).
        """
        with self._lock:
            b = self._bucket(self._clock())
            for k in GAUGE_KEYS:
                v = gauges.get(k)
                if v is not None:
                    b.gauges[k] = float(v)
            q = gauges.get("queue_depth")
            if q is not None:
                self._queue_depth = int(q)
                self._note_queue(b, int(q))

    @contextmanager
    def in_flight(self) -> Iterator[None]:
        with self._lock:
            self._in_flight += 1
        try:
            yield
        finally:
            with self._lock:
                self._in_flight -= 1

    # ---- readers
    def snapshot(self, window_s: int = 300) -> dict[str, Any]:
        with self._lock:
            now = self._clock()
            recent = [e for e in self._entries if e["ts"] >= now - window_s]
            by_key: dict[str, int] = {}
            by_endpoint: dict[str, int] = {}
            for e in recent:
                k = e.get("key") or "anonymous"
                by_key[k] = by_key.get(k, 0) + 1
                ep = e.get("endpoint")
                if ep is not None:
                    by_endpoint[ep] = by_endpoint.get(ep, 0) + 1
            lat = sorted(
                float(e["ms"]) for e in recent if e.get("status", 200) < 400 and e.get("ms") is not None
            )
            counts = [0] * (len(HIST_EDGES) + 1)
            for v in lat:
                counts[bisect_right(HIST_EDGES, v)] += 1
            uptime = now - self._start
            span = max(1.0, min(float(window_s), uptime))
            fwd = self._fwd_count
            return {
                "uptime_s": int(uptime),
                "total": self._total,
                "errors": self._errors,
                "in_flight": self._in_flight,
                "queue_depth": self._queue_depth,
                "window_s": window_s,
                "rps": round(len(recent) / span, 3),
                "by_key": by_key,
                "by_endpoint": by_endpoint,
                "latency_ms": {
                    "p50": round(_percentile(lat, 50), 1),
                    "p95": round(_percentile(lat, 95), 1),
                    "p99": round(_percentile(lat, 99), 1),
                    "max": round(lat[-1], 1) if lat else 0.0,
                    "hist": {"edges": list(HIST_EDGES), "counts": counts},
                },
                "forward": {
                    "count": fwd,
                    "avg_batch": round(self._fwd_records / fwd, 2) if fwd else 0.0,
                    "avg_ms": round(self._fwd_ms / fwd, 1) if fwd else 0.0,
                    "padding_ratio": round(1 - self._fwd_tokens / self._fwd_padded, 4)
                    if self._fwd_padded
                    else 0.0,
                },
                "tokens": {
                    "in_total": self._tokens,
                    "avg_in": round(self._tokens / self._token_reqs) if self._token_reqs else 0,
                },
            }

    def _aggregate(self, lo: int, hi: int, now_sec: int) -> dict[str, Any]:
        """Aggregate seconds [lo, hi) from the ring (lock held). Stale slots (wrapped ring) are skipped."""
        ret = self._retention
        reqs = errs = fwd = recs = toks = padded = 0
        lats: list[float] = []
        qmax: int | None = None
        gauges: dict[str, float] = {}
        for sec in range(max(lo, now_sec - ret + 1), hi):
            b = self._ring[sec % ret]
            if b is None or b.sec != sec:
                continue
            reqs += b.reqs
            errs += b.errs
            fwd += b.fwd
            recs += b.recs
            toks += b.toks
            padded += b.pad
            lats.extend(b.lats)
            if b.qmax is not None and (qmax is None or b.qmax > qmax):
                qmax = b.qmax
            gauges.update(b.gauges)
        lats.sort()
        return {
            "reqs": reqs,
            "p50": _percentile(lats, 50) if lats else None,
            "p95": _percentile(lats, 95) if lats else None,
            "err": errs / reqs if reqs else None,
            "batch": recs / fwd if fwd else None,
            "q": qmax,
            "pad": 1 - toks / padded if padded else None,
            "gauges": gauges,
        }

    def timeseries(self, window_s: int = 300, step_s: int | None = None) -> dict[str, Any]:
        """Columnar time series, oldest first: ``ceil(window_s / step_s)`` points aligned to multiples of
        ``step_s``; the last point is the current, partial step.

        ``rps`` of the partial step is requests so far divided by the full ``step_s`` (not elapsed seconds),
        so it ramps up through the step. ``queue_depth`` is the max seen in the step; a step with no
        observation carries the previous step's value forward (null until the first observation in the
        window). Gauges are the last non-null sample in the step. Raises ValueError on invalid arguments.
        """
        if not isinstance(window_s, int) or not TS_WINDOW_MIN <= window_s <= TS_WINDOW_MAX:
            raise ValueError(f"window_s must be an integer in {TS_WINDOW_MIN}..{TS_WINDOW_MAX}")
        if step_s is None:
            step_s = max(1, window_s // 60)
        if not isinstance(step_s, int) or not 1 <= step_s <= window_s:
            raise ValueError(f"step_s must be an integer in 1..window_s ({window_s})")
        n = -(-window_s // step_s)
        if n > TS_MAX_POINTS:
            raise ValueError(f"window_s / step_s must be <= {TS_MAX_POINTS} (got {n} points)")
        names = ("t", "rps", "p50_ms", "p95_ms", "error_rate", "avg_batch", "queue_depth", "padding_ratio")
        cols: dict[str, list[Any]] = {k: [] for k in (*names, *GAUGE_KEYS)}
        with self._lock:
            now = self._clock()
            last = int(now) // step_s * step_s
            carry_q: int | None = None
            for i in range(n):
                start = last - (n - 1 - i) * step_s
                a = self._aggregate(start, min(start + step_s, int(now) + 1), int(now))
                if a["q"] is not None:
                    carry_q = a["q"]
                cols["t"].append(start)
                cols["rps"].append(round(a["reqs"] / step_s, 3))
                cols["p50_ms"].append(_r(a["p50"], 1))
                cols["p95_ms"].append(_r(a["p95"], 1))
                cols["error_rate"].append(_r(a["err"], 4))
                cols["avg_batch"].append(_r(a["batch"], 2))
                cols["queue_depth"].append(carry_q)
                cols["padding_ratio"].append(_r(a["pad"], 4))
                for k in GAUGE_KEYS:
                    cols[k].append(a["gauges"].get(k))
        return {"window_s": window_s, "step_s": step_s, "now": round(now, 3), **cols}

    def latest_point(self, step_s: int = 5) -> dict[str, Any]:
        """The current (partial) step as scalars: one timeseries entry's fields plus ``step_s``.

        rps uses the same convention as timeseries(). ``queue_depth`` falls back to the live queue depth when
        the step saw no observation.
        """
        if not isinstance(step_s, int) or step_s < 1:
            raise ValueError("step_s must be a positive integer")
        with self._lock:
            now = self._clock()
            start = int(now) // step_s * step_s
            a = self._aggregate(start, int(now) + 1, int(now))
            return {
                "t": start,
                "step_s": step_s,
                "rps": round(a["reqs"] / step_s, 3),
                "p50_ms": _r(a["p50"], 1),
                "p95_ms": _r(a["p95"], 1),
                "error_rate": _r(a["err"], 4),
                "avg_batch": _r(a["batch"], 2),
                "queue_depth": a["q"] if a["q"] is not None else self._queue_depth,
                "padding_ratio": _r(a["pad"], 4),
                **{k: a["gauges"].get(k) for k in GAUGE_KEYS},
            }

    def log(self, limit: int = 100, since: int | None = None) -> list[dict[str, Any]]:
        with self._lock:
            items = [e for e in self._entries if since is None or e["id"] > since]
        return items[-limit:] if limit > 0 else []

    async def subscribe(self) -> AsyncIterator[dict[str, Any]]:
        """Yield new log entries as they are recorded (from any thread). Unsubscribes on close."""
        queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue(SUBSCRIBER_QUEUE)
        sub = (asyncio.get_running_loop(), queue)
        with self._lock:
            self._subs.append(sub)
        try:
            while True:
                yield await queue.get()
        finally:
            with self._lock:
                self._subs.remove(sub)
