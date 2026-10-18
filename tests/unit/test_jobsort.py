"""F2: window length sort and tokenize-ahead for jobs (CLEF_JOB_SORT_WINDOW, CLEF_TOKENIZE_AHEAD).

The engine is a fake with the two-step API (``prepare`` / ``submit``) whose "token length" is the length of
the record's state text, so the order the engine sees is observable. Flags off must behave like today.
"""

from __future__ import annotations

import json
import tempfile
import time
from types import SimpleNamespace
from typing import Any

import pytest
from test_jobs import (
    LABELS,
    SlowEngine,
    all_rows,
    hold_runner,
    make,
    open_store,
    submit,
    wait,
)
from test_keys import add_key, hdr

from clef_server.config import Config
from clef_server.engine import Prepared
from clef_server.jobsort import length_key, merge, pick, sorted_order

pytestmark = pytest.mark.postgres


def n_chars(record: dict[str, Any]) -> int:
    return len(json.dumps(record["state"]))


class TwoStepEngine(SlowEngine):
    """SlowEngine plus ``prepare`` / ``submit``; logs what the worker would see, in order."""

    def __init__(self, delay: float = 0.0) -> None:
        super().__init__(delay)
        self.events: list[tuple[str, Any]] = []  # ("prepare", n) / ("submit", [token lengths])
        self.lanes: list[str] = []
        self.batches: list[list[int]] = []  # token lengths per submitted micro-batch, background lane only

    async def prepare(self, records: list[dict[str, Any]], cache: bool = True) -> Prepared:
        self.events.append(("prepare", len(records)))
        enc = [SimpleNamespace(length=n_chars(r), is_media=False) for r in records]
        return Prepared(records, [None] * len(records), {}, enc)

    async def submit(self, prepared: Prepared, priority: str = "interactive") -> list[dict[str, Any]]:
        lengths = [e.length for e in prepared.encoded]
        self.lanes.append(priority)
        if priority == "background":
            self.batches.append(lengths)
        self.events.append(("submit", lengths))
        return await self.decide(prepared.records, priority)


def ahead_of_gpu(eng: TwoStepEngine) -> bool:
    """True when the second encode started before the first window's last micro-batch (two per window)."""
    kinds = [e[0] for e in eng.events]
    second_prepare = [i for i, k in enumerate(kinds) if k == "prepare"][1]
    submits = [i for i, k in enumerate(kinds) if k == "submit"]
    return second_prepare < submits[1]


def inputs(n: int) -> list[str]:
    """Lengths scrambled but deterministic, all different enough to sort."""
    return ["x" * (1 + (i * 7) % 23) + str(i) for i in range(n)]


def run_job(n: int = 40, eng: TwoStepEngine | None = None, **cfg: Any) -> tuple[TwoStepEngine, list[dict]]:
    eng = eng or TwoStepEngine()
    client, _, _ = make(eng, **{"max_microbatch": 4, **cfg})
    with client:
        jid = submit(client, payload={"inputs": inputs(n), "labels": LABELS}).json()["id"]
        job = wait(client, jid)
        assert job["status"] == "succeeded", job
        return eng, all_rows(client, jid)


# ------------------------------------------------------------------ flags off = today


def test_flags_off_sends_micro_batches_in_input_order():
    eng, rows = run_job(n=19)
    assert [r["index"] for r in rows] == list(range(19))
    assert [n for b in eng.batches for n in b] == [n_chars({"state": s}) for s in inputs(19)]
    assert [len(b) for b in eng.batches] == [4, 4, 4, 4, 3]
    assert [e[0] for e in eng.events] == ["prepare", "submit"] * 5  # one chunk at a time, no look-ahead


def test_flags_off_do_not_touch_jobsort(monkeypatch):
    from clef_server import jobsort

    def boom(*a: Any, **k: Any) -> None:
        raise AssertionError("WindowSorter built with both flags off")

    monkeypatch.setattr(jobsort, "WindowSorter", boom)
    run_job(n=9)


def test_flags_off_rows_equal_flags_on_rows():
    _, base = run_job(n=37)
    _, sort = run_job(n=37, job_sort_window=16)
    _, ahead = run_job(n=37, tokenize_ahead=True)
    _, both = run_job(n=37, job_sort_window=16, tokenize_ahead=True)
    assert base == sort == ahead == both


# ------------------------------------------------------------------ sorting


def test_window_is_sorted_by_length_and_results_come_back_in_order():
    eng, rows = run_job(n=48, job_sort_window=16)
    assert [r["index"] for r in rows] == list(range(48))
    # window 16 = four micro-batches of 4; inside a window the engine sees non-decreasing lengths
    for w in range(0, len(eng.batches), 4):
        flat = [n for b in eng.batches[w : w + 4] for n in b]
        assert flat == sorted(flat) and len(flat) == 16
    unsorted = [n for b in eng.batches for n in b]
    assert unsorted != sorted(unsorted)  # but not one global sort: windows are independent
    assert all(len(b) <= 4 for b in eng.batches)  # still micro-batch sized, never a window at once


def test_window_smaller_than_a_micro_batch_is_raised_to_one():
    eng, rows = run_job(n=10, job_sort_window=2)
    assert [r["index"] for r in rows] == list(range(10))
    assert all(len(b) <= 4 for b in eng.batches)


def test_tokenize_ahead_encodes_the_next_window_before_the_current_one_is_submitted():
    eng, rows = run_job(n=24, eng=TwoStepEngine(delay=0.05), job_sort_window=8, tokenize_ahead=True)
    assert [r["index"] for r in rows] == list(range(24))
    assert ahead_of_gpu(eng)  # window 2 is encoded while window 1 (two micro-batches) is still on the GPU
    base, _ = run_job(n=24, eng=TwoStepEngine(delay=0.05), job_sort_window=8)
    assert not ahead_of_gpu(base)  # sanity: without the flag the order is encode, run, run, encode, ...


def test_ahead_without_sort_keeps_input_order():
    eng, rows = run_job(n=17, eng=TwoStepEngine(delay=0.05), tokenize_ahead=True)
    assert [r["index"] for r in rows] == list(range(17))
    assert [len(b) for b in eng.batches] == [4, 4, 4, 4, 1]
    assert ahead_of_gpu(eng)


def test_rejected_record_is_isolated_in_a_sorted_window():
    eng = TwoStepEngine()
    client, _, _ = make(eng, max_microbatch=4, job_sort_window=16)
    ins = inputs(20)
    ins[6] = "TOOLONG"
    with client:
        jid = submit(client, payload={"inputs": ins, "labels": LABELS}).json()["id"]
        job = wait(client, jid)
        rows = all_rows(client, jid)
    assert job["status"] == "succeeded" and job["progress"]["failed"] == 1, job
    assert [r["index"] for r in rows] == list(range(20))
    assert "error" in rows[6] and all("error" not in r for i, r in enumerate(rows) if i != 6)


# ------------------------------------------------------------------ crash / resume


def test_clean_stop_mid_window_resumes_at_the_window_start():
    state = tempfile.mkdtemp(prefix="clef-sort-")
    n, window = 64, 16
    client, _, _ = make(TwoStepEngine(delay=0.05), state_dir=state, max_microbatch=4, job_sort_window=window)
    with client:
        jid = submit(client, payload={"inputs": inputs(n), "labels": LABELS}).json()["id"]
        while client.get(f"/v1/jobs/{jid}").json()["progress"]["done"] < window:
            time.sleep(0.01)
        time.sleep(0.12)  # inside the next window: some micro-batches done, nothing of it persisted
    store = open_store(Config(state_dir=state))
    parked = store.get(jid)
    store.close()
    assert parked["status"] == "interrupted" and parked["resumes"] == 0
    assert parked["done"] % window == 0 and 16 <= parked["done"] < n  # whole windows only
    persisted = parked["done"]

    fast = TwoStepEngine()
    client2, _, _ = make(fast, state_dir=state, max_microbatch=4, job_sort_window=window)
    with client2:
        job = wait(client2, jid)
        assert job["status"] == "succeeded" and job["progress"]["done"] == n
        rows = all_rows(client2, jid)
    assert [r["index"] for r in rows] == list(range(n))  # no gap, no duplicate
    assert sum(fast.sizes) == n - persisted  # only the missing items were recomputed


def test_hard_crash_resumes_even_when_done_is_not_window_aligned():
    state = tempfile.mkdtemp(prefix="clef-sort-")
    store = open_store(Config(state_dir=state, api_key=None))
    row = store.create("classify", None, {"inputs": inputs(20), "labels": LABELS}, None, None, 10)
    assert store.claim(row["id"])
    store.add_items(row["id"], [{"index": i, "label": "billing"} for i in range(5)])  # a run without the flag
    store.close()
    client, eng, _ = make(TwoStepEngine(), state_dir=state, max_microbatch=4, job_sort_window=8)
    with client:
        job = wait(client, row["id"])
        rows = all_rows(client, row["id"])
    assert job["status"] == "succeeded" and job["resumes"] == 1 and job["progress"]["done"] == 20
    assert [r["index"] for r in rows] == list(range(20)) and rows[0]["label"] == "billing"
    assert sum(eng.sizes) == 15


def test_failure_mid_window_persists_nothing_of_that_window():
    class Dies(TwoStepEngine):
        async def submit(self, prepared, priority="interactive"):
            if priority == "background" and len(self.batches) >= 2:
                raise RuntimeError("boom")  # systemic, not a rejected record
            return await super().submit(prepared, priority)

    eng = Dies()
    client, _, _ = make(eng, max_microbatch=4, job_sort_window=16)
    with client:
        jid = submit(client, payload={"inputs": inputs(16), "labels": LABELS}).json()["id"]
        job = wait(client, jid)
        assert job["status"] == "failed" and job["progress"]["done"] == 0
        assert all_rows(client, jid) == []


def test_cancel_mid_window_keeps_earlier_windows_only():
    eng = TwoStepEngine(delay=0.05)
    client, _, _ = make(eng, max_microbatch=4, job_sort_window=16)
    with client:
        jid = submit(client, payload={"inputs": inputs(80), "labels": LABELS}).json()["id"]
        while client.get(f"/v1/jobs/{jid}").json()["progress"]["done"] < 16:
            time.sleep(0.01)
        assert client.post(f"/v1/jobs/{jid}/cancel").status_code == 200
        job = wait(client, jid)
        assert job["status"] == "cancelled" and job["progress"]["done"] % 16 == 0
        rows = all_rows(client, jid)
    assert [r["index"] for r in rows] == list(range(len(rows)))


# ------------------------------------------------------------------ with the job scheduler (E1 seam)


class OwnerProbe(TwoStepEngine):
    """TwoStepEngine that records which owner each background micro-batch serves and how many overlap."""

    def __init__(self, delay: float = 0.01) -> None:
        super().__init__(delay)
        self.inflight = self.max_inflight = 0
        self.seq: list[str] = []

    async def submit(self, prepared, priority="interactive"):
        if priority != "background":
            return await super().submit(prepared, priority)
        self.inflight += 1
        self.max_inflight = max(self.max_inflight, self.inflight)
        text = json.dumps(prepared.records[0]["state"])
        self.seq.append("a" if "ownerA" in text else "b" if "ownerB" in text else "?")
        try:
            return await super().submit(prepared, priority)
        finally:
            self.inflight -= 1


def test_two_principals_still_interleave_with_a_sort_window_and_one_chunk_is_in_flight(monkeypatch):
    eng = OwnerProbe()
    client, _, app = make(eng, api_key="root", max_microbatch=4, job_sort_window=8)
    ha, hb = hdr(add_key(app, "alice", "jobs")), hdr(add_key(app, "bob", "jobs"))
    hold = hold_runner(monkeypatch, app.state.ctx.extra["jobs"].store)
    with client:
        ids = {}
        for name, h in (("A", ha), ("B", hb)):
            body = {"inputs": [f"owner{name}-{i}" for i in range(24)], "labels": LABELS}
            ids[name] = submit(client, payload=body, headers=h).json()["id"]
        hold.clear()
        client.portal.call(app.state.ctx.extra["jobs"].poke)
        for name, h in (("A", ha), ("B", hb)):
            assert wait(client, ids[name], headers=h)["status"] == "succeeded"
    assert eng.max_inflight == 1  # the turn gate wraps the sorter's hand-off: one chunk across all jobs
    assert sorted(eng.seq) == ["a"] * 6 + ["b"] * 6
    first = "".join(eng.seq[:10])
    assert "aaa" not in first and "bbb" not in first, eng.seq


class HitsEngine(TwoStepEngine):
    """The first record of every window is a result-cache hit (no forward)."""

    async def prepare(self, records, cache=True):
        prepared = await super().prepare(records, cache)
        first = records[0]
        hit = {
            "model": first["model"],
            "answers": {q: self.answer(q, spec) for q, spec in first["questions"].items()},
            "usage": {"input_tokens": 1},
            "timing": {},
        }
        prepared.hits[0] = hit
        return prepared


def test_cache_hits_in_a_sorted_window_are_not_charged():
    eng = HitsEngine()
    client, _, app = make(eng, api_key="root", max_microbatch=4, job_sort_window=8)
    h = hdr(add_key(app, "ci", "jobs", daily_quota=100))
    with client:
        body = {"inputs": inputs(16), "labels": LABELS}  # two windows of 8, one hit each
        jid = submit(client, payload=body, headers=h).json()["id"]
        assert app.state.ctx.quotas.used("ci") == 16
        assert wait(client, jid, headers=h)["status"] == "succeeded"
        for _ in range(100):
            if app.state.ctx.quotas.used("ci") == 14:
                break
            time.sleep(0.02)
        assert app.state.ctx.quotas.used("ci") == 14
        assert app.state.ctx.extra["jobs"].store.get(jid)["quota_reserved"] == 14
    assert sum(len(b) for b in eng.batches) == 14  # the hits never reached the engine


def test_an_expired_job_stops_cleanly_in_the_sorted_path():
    eng = TwoStepEngine(delay=0.05)
    client, _, _ = make(eng, max_microbatch=4, job_sort_window=8)
    with client:
        body = {"inputs": inputs(160), "labels": LABELS}
        jid = submit(client, payload=body, expire_after_s=0.6).json()["id"]
        job = wait(client, jid)
        assert job["status"] == "cancelled" and job["expired"] is True and job["error"] == "expired"
        done = job["progress"]["done"]
        assert 0 < done < 160 and done % 8 == 0  # whole windows only: nothing out of order
        rows = all_rows(client, jid)
    assert [r["index"] for r in rows] == list(range(done))


# ------------------------------------------------------------------ interactive priority


@pytest.mark.parametrize(
    "cfg",
    [{"job_sort_window": 32}, {"tokenize_ahead": True}, {"job_sort_window": 32, "tokenize_ahead": True}],
)
def test_interactive_requests_still_interleave_with_a_sorted_job(cfg):
    chunk = 0.3
    eng = TwoStepEngine(delay=chunk)
    client, _, _ = make(eng, max_microbatch=8, **cfg)
    with client:
        jid = submit(client, payload={"inputs": [f"t{i}" for i in range(80)], "labels": LABELS}).json()["id"]
        wait(client, jid, until=("running",))
        t0 = time.monotonic()
        while not eng.batches:
            assert time.monotonic() - t0 < 10
            time.sleep(0.01)
        latencies = []
        for _ in range(3):
            t1 = time.perf_counter()
            assert client.post("/v1/classify", json={"input": "hi", "labels": LABELS}).status_code == 200
            latencies.append(time.perf_counter() - t1)
            time.sleep(0.11)
        assert max(latencies) < 3 * chunk, latencies  # one micro-batch in flight, as without the flags
        assert wait(client, jid)["status"] == "succeeded"
    assert max(len(b) for b in eng.batches) <= 8
    assert "interactive" in eng.lanes and "background" in eng.lanes


# ------------------------------------------------------------------ helpers


def test_sorted_order_is_stable_and_skips_hits_and_dups():
    enc = [SimpleNamespace(length=n, is_media=False) for n in (5, 3, 5, 3, 1)]
    enc[1] = None
    prepared = Prepared([{"state": ""}] * 5, [None] * 5, {1: {"hit": True}}, enc, {3: 0})
    assert sorted_order(prepared) == [4, 0, 2]  # 1 is a hit, 3 a duplicate of 0; ties keep input order


def test_media_sorts_last():
    enc = [SimpleNamespace(length=100, is_media=True), SimpleNamespace(length=900, is_media=False)]
    prepared = Prepared([{}, {}], [None, None], {}, enc)
    assert sorted_order(prepared) == [1, 0] and length_key(prepared, 0) > length_key(prepared, 1)


def test_pick_and_merge_restore_input_order_with_hits_and_dups():
    enc = [SimpleNamespace(length=n, is_media=False) for n in (9, 2, 4)]
    hit = {"answers": 1, "timing": {"cached": True}}
    prepared = Prepared(
        [{"i": 0}, {"i": 1}, {"i": 2}, {"i": 3}], ["a", None, "b", "c"], {3: hit}, [*enc, None], {1: 0}
    )
    order = sorted_order(prepared)  # 0 is 9, 2 is 4; index 1 is a dup of 0, 3 a hit
    assert order == [2, 0]
    sub = pick(prepared, order)
    assert (
        [r["i"] for r in sub.records] == [2, 0] and sub.keys == ["b", "a"] and not sub.hits and not sub.dups
    )
    computed = {2: {"answers": "two", "timing": {}}, 0: {"answers": "zero", "timing": {}}}
    out = merge(prepared, computed)
    assert [o["answers"] for o in out] == ["zero", "zero", "two", 1]
    assert out[1]["timing"]["cached"] is True and out[1] is not computed[0]  # a private copy, marked cached


def test_config_validates_the_window():
    assert Config(job_sort_window=512).job_sort_window == 512
    assert Config().job_sort_window == 0 and Config().tokenize_ahead is False
    with pytest.raises(ValueError, match="CLEF_JOB_SORT_WINDOW"):
        Config(job_sort_window=-1)
