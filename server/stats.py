"""Thread-safe request log, latency histogram, counters and forward stats (engine thread + event loop)."""

from __future__ import annotations

import asyncio
import contextlib
import threading
import time
from bisect import bisect_right
from collections import deque
from collections.abc import AsyncIterator, Iterator
from contextlib import contextmanager
from typing import Any

HIST_EDGES = [25, 50, 100, 150, 200, 300, 500, 1000, 2000, 5000]
SUBSCRIBER_QUEUE = 256


def _percentile(sorted_vals: list[float], q: float) -> float:
    if not sorted_vals:
        return 0.0
    rank = max(1, -(-len(sorted_vals) * q // 100))  # nearest-rank, ceil
    return sorted_vals[int(rank) - 1]


class Stats:
    def __init__(self, buffer: int = 1000):
        self._lock = threading.Lock()
        self._entries: deque[dict[str, Any]] = deque(maxlen=max(1, buffer))
        self._next_id = 1
        self._start = time.time()
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

    # ---- writers
    def record_request(self, entry: dict[str, Any]) -> None:
        with self._lock:
            item = {"id": self._next_id, "ts": round(time.time(), 3)}
            item.update(entry)
            item.setdefault("media", {"images": 0, "videos": 0})
            item.setdefault("error", None)
            item.setdefault("state_preview", None)
            self._next_id += 1
            self._entries.append(item)
            self._total += 1
            if int(item.get("status", 200)) >= 400:
                self._errors += 1
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

    def set_queue_depth(self, n: int) -> None:
        with self._lock:
            self._queue_depth = n

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
        now = time.time()
        with self._lock:
            recent = [e for e in self._entries if e["ts"] >= now - window_s]
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
