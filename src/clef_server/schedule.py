"""Off-hours windows for batch jobs (``schedule = {"window": "20:00-06:00", "tz": "Europe/Madrid"}``).

Pure and clock-free: every function takes the time to look at, so tests use a fake clock. See
docs/ARCHITECTURE.md (Job scheduler) for the contract.

* ``start > end`` crosses midnight, ``start == end`` is 24 h (always open).
* Edges are wall-clock times in the zone. In the skipped hour of a spring-forward day an edge starts at the
  next valid minute; in the repeated hour of a fall-back day it uses the first occurrence (``fold=0``).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta
from functools import lru_cache
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

WINDOW_RE = re.compile(r"^([01]\d|2[0-3]):([0-5]\d)-([01]\d|2[0-3]):([0-5]\d)$")


def load_zone(name: str) -> ZoneInfo:
    try:
        return ZoneInfo(name)
    except (ZoneInfoNotFoundError, ValueError, OSError) as exc:
        raise ValueError(
            f"unknown time zone {name!r} (an IANA name such as 'Europe/Madrid'; on Windows install tzdata)"
        ) from exc


def _at(tz: ZoneInfo, day: date, t: time) -> float:
    """Unix time of wall time ``t`` on ``day``; inside a skipped hour: the next valid minute."""
    naive = datetime.combine(day, t)
    aware = naive.replace(tzinfo=tz, fold=0)
    for _ in range(
        180
    ):  # a gap is at most an hour (a few zones had 24 h once; not a window edge worth supporting)
        aware = naive.replace(tzinfo=tz, fold=0)
        if datetime.fromtimestamp(aware.timestamp(), tz).replace(tzinfo=None) == naive:
            break
        naive += timedelta(minutes=1)
    return aware.timestamp()


@dataclass(frozen=True)
class Window:
    start: time
    end: time
    tz: ZoneInfo

    @property
    def always(self) -> bool:
        return self.start == self.end

    def _spans(self, ts: float) -> list[tuple[float, float]]:
        """The window instances that could contain ``ts`` (those starting yesterday or today, local)."""
        today = datetime.fromtimestamp(ts, self.tz).date()
        out = []
        for d in (today - timedelta(days=1), today):
            end_day = d + timedelta(days=1) if self.start > self.end else d
            out.append((_at(self.tz, d, self.start), _at(self.tz, end_day, self.end)))
        return out

    def contains(self, ts: float) -> bool:
        return self.always or any(s <= ts < e for s, e in self._spans(ts))

    def next_start(self, ts: float) -> float | None:
        """The first opening strictly after ``ts`` (None for a 24 h window)."""
        if self.always:
            return None
        today = datetime.fromtimestamp(ts, self.tz).date()
        for k in range(0, 3):
            s = _at(self.tz, today + timedelta(days=k), self.start)
            if s > ts:
                return s
        raise AssertionError("unreachable: some day within 3 has an opening after ts")  # pragma: no cover

    def next_end(self, ts: float) -> float | None:
        """When the instance that contains ``ts`` closes (None when outside or always open)."""
        if self.always:
            return None
        return next((e for s, e in self._spans(ts) if s <= ts < e), None)


@lru_cache(maxsize=256)
def _parse(spec: str, tz: str) -> Window:
    m = WINDOW_RE.match(spec)
    if not m:
        raise ValueError(f"window {spec!r} must look like 'HH:MM-HH:MM' (24 h clock), e.g. '20:00-06:00'")
    h1, m1, h2, m2 = (int(x) for x in m.groups())
    return Window(time(h1, m1), time(h2, m2), load_zone(tz))


def parse_window(spec: str | None, tz: str) -> Window | None:
    """``Window`` for ``spec`` in zone ``tz`` (None = no window); ValueError for a bad zone or window."""
    load_zone(tz)  # also validates the zone when there is no window
    return None if not spec else _parse(spec.strip(), tz)
