"""In-memory LRU result cache (docs/ARCHITECTURE.md, Result cache).

The engine owns the only instance: `Engine.prepare` looks up, `Engine.submit` stores. Entries are
whole per-record results (answers + usage) without timing. Records with images or videos are never cached.
"""

from __future__ import annotations

import copy
import hashlib
import json
import threading
from collections import OrderedDict
from typing import Any

Record = dict
Result = dict


def _ordered(q: Any) -> Any:
    """A question with its `criteria` as an ordered [[key, text], ...] list (dict order is meaningful)."""
    if isinstance(q, dict) and isinstance(q.get("criteria"), dict):
        return {**q, "criteria": [[k, v] for k, v in q["criteria"].items()]}
    return q


class ResultCache:
    def __init__(self, capacity: int = 10000, max_bytes: int = 256 * 1024 * 1024) -> None:
        self.capacity = max(0, int(capacity))
        self.max_bytes = max(0, int(max_bytes))
        self._sizes: dict[str, int] = {}
        self._bytes = 0
        self._lock = threading.Lock()
        self._data: OrderedDict[str, Result] = OrderedDict()
        self._hits = 0
        self._misses = 0
        self._bypassed = 0
        self._evictions = 0

    @property
    def enabled(self) -> bool:
        return self.capacity > 0 and self.max_bytes > 0

    @staticmethod
    def key(model_ref: str, record: Record) -> str | None:
        """sha256 of the canonical JSON of the model ref, state and questions; None for media records."""
        if record.get("images") or record.get("videos"):
            return None
        # Questions (and choice criteria) keep their order: the release lays them out in dict order, so a
        # different order is a different forward. Only the state (sorted by the release) is order-free.
        qs = record.get("questions")
        if isinstance(qs, dict):
            qs = [[qid, _ordered(q)] for qid, q in qs.items()]
        payload = {"m": model_ref, "s": record.get("state"), "q": qs}
        blob = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str)
        return hashlib.sha256(blob.encode("utf-8", "surrogatepass")).hexdigest()

    def get(self, key: str) -> Result | None:
        """A private copy of the stored result (moved to the MRU end), or None."""
        if not self.enabled:
            return None
        with self._lock:
            value = self._data.get(key)
            if value is None:
                self._misses += 1
                return None
            self._data.move_to_end(key)
            self._hits += 1
            return copy.deepcopy(value)

    def put(self, key: str, result: Result) -> None:
        """Store a copy of `result` without its timing; evicts the LRU entry past capacity."""
        if not self.enabled:
            return
        stored = copy.deepcopy({k: v for k, v in result.items() if k != "timing"})
        size = len(json.dumps(stored, default=str))
        if size > self.max_bytes // 10:
            return  # one entry never takes more than a tenth of the budget
        with self._lock:
            self._bytes -= self._sizes.get(key, 0)
            self._data[key] = stored
            self._sizes[key] = size
            self._bytes += size
            self._data.move_to_end(key)
            while len(self._data) > self.capacity or self._bytes > self.max_bytes:
                old, _ = self._data.popitem(last=False)
                self._bytes -= self._sizes.pop(old, 0)
                self._evictions += 1

    def note_dup(self) -> None:
        """A duplicate inside one call: `get` counted a miss, but the record is served as a hit."""
        with self._lock:
            self._misses -= 1
            self._hits += 1

    def note_bypass(self, n: int = 1) -> None:
        """Count lookups skipped by `Cache-Control: no-cache` or because the record is not cacheable."""
        if self.enabled:
            with self._lock:
                self._bypassed += n

    def clear(self) -> None:
        with self._lock:
            self._data.clear()
            self._sizes.clear()
            self._bytes = 0

    def __len__(self) -> int:
        with self._lock:
            return len(self._data)

    def stats(self) -> dict[str, Any]:
        with self._lock:
            return {
                "enabled": self.enabled,
                "size": len(self._data),
                "capacity": self.capacity,
                "bytes": self._bytes,
                "max_bytes": self.max_bytes,
                "hits": self._hits,
                "misses": self._misses,
                "bypassed": self._bypassed,
                "evictions": self._evictions,
            }
