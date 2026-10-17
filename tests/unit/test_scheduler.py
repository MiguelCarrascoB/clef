"""Job scheduler (E1): classes, fair share, aging, off-hours windows, urgent, quota at submit, restart.

The policy tests drive `Scheduler` with a fake clock (no sleeping); the integration tests run the real runner
on FakeEngine with `JobRunner.clock` replaced.
"""

from __future__ import annotations

import asyncio
import json
import tempfile
import time
from datetime import date, datetime, timezone
from typing import Any

import pytest
from test_api import FakeEngine
from test_jobs import LABELS, SlowEngine, hold_runner, make, submit, wait
from test_keys import HitEngine, add_key, hdr

from clef_server.jobs import (
    EXPIRED,
    PRIORITY_URGENT,
    Entry,
    Scheduler,
    _entry,
)
from clef_server.schedule import _at, parse_window

pytestmark = (
    pytest.mark.postgres
)  # the integration tests also run on Postgres when CLEF_TEST_DATABASE_URL is set

NOON = datetime(2026, 10, 19, 12, 0, tzinfo=timezone.utc).timestamp()
HOUR = 3600.0
WINDOW = {"window": "20:00-06:00", "tz": "UTC"}  # closed at NOON, open from 20:00


def ent(job_id: str, owner: str = "", cls: int = 1, seq: int = 0, **kw: Any) -> Entry:
    kw.setdefault("window", None)
    kw.setdefault("deadline", None)
    kw.setdefault("ref", 0.0)
    return Entry(id=job_id, owner=owner, cls=cls, seq=seq, interrupted=False, **kw)


def sched(active: int = 4, aging_s: float = 0.0, now: float = 0.0) -> tuple[Scheduler, list[float]]:
    t = [now]
    return Scheduler(active, aging_s, lambda: t[0]), t


async def settle_loop() -> None:
    for _ in range(5):
        await asyncio.sleep(0)


# ------------------------------------------------------------------ policy: fair share


def test_two_owners_interleave_whatever_their_job_counts():
    async def main() -> list[str]:
        s, _ = sched()
        for e in (ent("a1", "a", seq=1), ent("a2", "a", seq=2), ent("b1", "b", seq=3)):
            s.register(e)
        order: list[str] = []

        async def job(job_id: str, n: int) -> None:
            for _ in range(n):
                await s.turn(job_id)
                order.append(job_id[0])
                await asyncio.sleep(0)  # the engine call
                s.done(job_id, 10)

        await asyncio.gather(job("a1", 6), job("a2", 6), job("b1", 6))
        return order

    order = "".join(asyncio.run(main()))
    assert len(order) == 18
    head = order[:12]  # while both owners have work: one owner never gets two turns in a row
    assert head.count("a") == head.count("b") == 6 and "aa" not in head and "bb" not in head


def test_a_returning_owner_does_not_collect_the_turns_it_missed():
    async def main() -> str:
        s, _ = sched()
        s.register(ent("a1", "a", seq=1))
        for _ in range(5):  # a runs alone for 5 turns: pass 50
            await s.turn("a1")
            s.done("a1", 10)
        s.register(ent("b1", "b", seq=2))
        assert s.passes["b"] == s.passes["a"]  # starts at the current minimum, not at 0
        order = ""

        async def job(job_id: str) -> None:
            nonlocal order
            for _ in range(4):
                await s.turn(job_id)
                order += job_id[0]
                await asyncio.sleep(0)
                s.done(job_id, 10)

        await asyncio.gather(job("a1"), job("b1"))
        return order

    order = asyncio.run(main())
    assert "aa" not in order and "bb" not in order  # alternating, b did not get a burst of 5


# ------------------------------------------------------------------ policy: priority and aging


def test_classes_are_strict_between_classes():
    async def main() -> list[str]:
        s, _ = sched()
        for e in (ent("x", "x"), ent("lo", "l", cls=0), ent("hi", "h", cls=2), ent("no", "n", cls=1)):
            s.register(e)
        await s.turn("x")  # holds the engine while the others queue up
        order: list[str] = []

        async def job(job_id: str) -> None:
            await s.turn(job_id)
            order.append(job_id)
            await asyncio.sleep(0)
            s.done(job_id, 1)

        tasks = [asyncio.create_task(job(j)) for j in ("lo", "no", "hi")]
        await settle_loop()
        s.done("x", 1)
        await asyncio.gather(*tasks)
        return order

    assert asyncio.run(main()) == ["hi", "no", "lo"]


def test_aging_raises_a_waiting_job_one_class_per_period_but_never_to_urgent():
    s, _ = sched(aging_s=100.0)
    low, urgent, high = ent("l", cls=0), ent("u", cls=PRIORITY_URGENT), ent("h", cls=2)
    assert [s.eff_class(low, t) for t in (0, 99, 100, 250, 10_000)] == [0, 0, 1, 2, 2]
    assert s.eff_class(high, 10_000) == 2 and s.eff_class(urgent, 10_000) == PRIORITY_URGENT
    off, _ = sched(aging_s=0.0)
    assert off.eff_class(low, 10_000) == 0


def test_an_aged_low_job_is_not_starved_by_a_busy_high_owner():
    async def first_served(aging_s: float) -> str:
        s, t = sched(aging_s=aging_s)
        for e in (ent("x", "x"), ent("hi", "h", cls=2, ref=200.0), ent("lo", "l", cls=0, ref=0.0)):
            s.register(e)
        s.passes["h"] = 50.0  # h has been served a lot
        t[0] = 200.0
        await s.turn("x")
        order: list[str] = []

        async def job(job_id: str) -> None:
            await s.turn(job_id)
            order.append(job_id)

        tasks = [asyncio.create_task(job(j)) for j in ("hi", "lo")]
        await settle_loop()
        s.done("x", 1)
        await settle_loop()
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        return order[0]

    assert asyncio.run(first_served(0.0)) == "hi"  # no aging: strict classes
    assert asyncio.run(first_served(100.0)) == "lo"  # waited two periods: high class, lower pass


# ------------------------------------------------------------------ policy: windows, expiry, urgent


def test_window_pause_frees_the_slot_and_resumes_when_it_opens():
    async def main() -> None:
        s, t = sched(active=1, now=NOON)
        win = parse_window("20:00-06:00", "UTC")
        s.register(ent("w", "a", window=win))
        task = asyncio.create_task(s.turn("w"))
        await settle_loop()
        assert not task.done() and s.waiting("w") == "off_hours" and s.has_slot()
        s.register(ent("n", "b", seq=1))  # another job takes the freed slot
        t[0] = NOON + 9 * HOUR  # 21:00: the window is open, but the slot is taken
        s.refresh(t[0])
        s.dispatch()
        await settle_loop()
        assert not task.done() and s.waiting("w") == "active_limit"
        s.unregister("n")
        s.refresh(t[0])
        s.dispatch()
        await settle_loop()
        assert task.done() and s.inflight == "w"
        s.done("w", 1)
        t[0] = NOON + 19 * HOUR  # 07:00 next day: out again
        s.done("w", 0)
        assert s.waiting("w") == "off_hours"

    asyncio.run(main())


def test_a_window_does_not_stop_the_chunk_in_flight_and_a_late_job_expires():
    async def main() -> None:
        s, t = sched(now=NOON)
        s.register(ent("w", window=parse_window("20:00-06:00", "UTC"), deadline=NOON + 10 * HOUR))
        waiter = asyncio.create_task(s.turn("w"))
        await settle_loop()
        t[0] = NOON + 11 * HOUR
        s.refresh(t[0])  # the deadline passed while it was paused: the waiting turn is interrupted
        with pytest.raises(Exception, match=EXPIRED):
            await waiter

    asyncio.run(main())


def test_urgent_ignores_the_window_and_the_slot_limit():
    row = {
        "id": "j", "owner": None, "status": "queued", "priority": PRIORITY_URGENT, "seq": 1,
        "created_at": 0.0, "schedule": WINDOW, "eligible_at": None, "expire_after_s": None,
    }  # fmt: skip
    s, _ = sched(active=1, now=NOON)
    assert _entry(row, s).window is None  # urgent: always open
    assert _entry({**row, "priority": 1}, s).window is not None
    s.register(ent("n"))
    assert not s.has_slot()  # full; the runner still admits urgent jobs (see test_urgent_job_runs_outside...)


def test_window_edges_around_dst():
    ny = "America/New_York"
    gap = parse_window("02:30-04:00", ny)  # 2026-03-08 02:00 -> 03:00: 02:30 does not exist
    assert (
        datetime.fromtimestamp(_at(gap.tz, date(2026, 3, 8), gap.start), gap.tz).strftime("%H:%M") == "03:00"
    )
    repeat = parse_window("01:30-02:30", ny)  # 2026-11-01: 01:30 happens twice; fold=0 is the first
    start, end = (
        _at(repeat.tz, date(2026, 11, 1), repeat.start),
        _at(repeat.tz, date(2026, 11, 1), repeat.end),
    )
    assert end - start == 2 * HOUR
    assert repeat.contains(start + 90 * 60) and not repeat.contains(end)
    over = parse_window("20:00-06:00", "UTC")  # crosses midnight
    assert over.contains(NOON + 12 * HOUR) and over.contains(NOON + 17 * HOUR) and not over.contains(NOON)
    assert over.next_start(NOON) == NOON + 8 * HOUR and over.next_end(NOON + 17 * HOUR) == NOON + 18 * HOUR
    assert parse_window("08:00-08:00", "UTC").contains(NOON) and parse_window("08:00-08:00", "UTC").always
    for bad in (("25:00-06:00", "UTC"), ("8-6", "UTC"), ("20:00-06:00", "Nowhere/Land")):
        with pytest.raises(ValueError):
            parse_window(*bad)


# ------------------------------------------------------------------ integration helpers


class ProbeEngine(FakeEngine):
    """Records which job each engine call serves and how many overlap (no lock, so it shows concurrency)."""

    def __init__(self) -> None:
        super().__init__()
        self.inflight = self.max_inflight = 0
        self.seq: list[str] = []

    async def decide(self, records, priority="interactive"):
        self.inflight += 1
        self.max_inflight = max(self.max_inflight, self.inflight)
        text = json.dumps(records[0]["state"])
        self.seq.append("a" if "ownerA" in text else "b" if "ownerB" in text else "?")
        await asyncio.sleep(0.01)
        try:
            return await super().decide(records)
        finally:
            self.inflight -= 1


def fake_clock(app: Any, start: float = NOON) -> list[float]:
    t = [start]
    app.state.ctx.extra["jobs"].clock = lambda: t[0]
    return t


def poke(client: Any, app: Any) -> None:
    client.portal.call(app.state.ctx.extra["jobs"].poke)  # the event is not thread safe


def progress(client: Any, jid: str, **kw: Any) -> int:
    return client.get(f"/v1/jobs/{jid}", **kw).json()["progress"]["done"]


# ------------------------------------------------------------------ integration: fairness and one chunk


def test_two_principals_advance_in_turns_and_only_one_chunk_is_in_flight(monkeypatch):
    eng = ProbeEngine()
    client, _, app = make(eng, api_key="root", max_microbatch=4)
    ha, hb = hdr(add_key(app, "alice", "jobs")), hdr(add_key(app, "bob", "jobs"))
    hold = hold_runner(monkeypatch, app.state.ctx.extra["jobs"].store)
    with client:
        ids = {}
        for name, h in (("A", ha), ("B", hb)):
            body = {"inputs": [f"owner{name}-{i}" for i in range(24)], "labels": LABELS}
            ids[name] = submit(client, payload=body, headers=h).json()["id"]
        hold.clear()
        poke(client, app)
        seen = set()
        deadline = time.monotonic() + 15
        while len(seen) < 2 and time.monotonic() < deadline:  # both show `running` (D9) while advancing
            for name, h in (("A", ha), ("B", hb)):
                job = client.get(f"/v1/jobs/{ids[name]}", headers=h).json()
                if job["status"] == "running" and 0 < job["progress"]["done"] < 24:
                    seen.add(name)
            time.sleep(0.005)
        assert seen == {"A", "B"}
        for name, h in (("A", ha), ("B", hb)):
            assert wait(client, ids[name], headers=h)["status"] == "succeeded"
    assert eng.max_inflight == 1  # one job chunk in flight across all jobs
    assert sorted(eng.seq) == ["a"] * 6 + ["b"] * 6
    first = "".join(eng.seq[:10])
    assert "aaa" not in first and "bbb" not in first, eng.seq


# ------------------------------------------------------------------ integration: priority


def test_priority_field_validation_listing_and_job_info(monkeypatch):
    client, _, app = make()
    hold = hold_runner(monkeypatch, app.state.ctx.extra["jobs"].store)
    with client:
        ok = submit(client, priority="high").json()
        assert ok["priority"] == "high" and ok["urgent"] is False and ok["schedule"] is None
        assert ok["waiting_reason"] is None and ok["expired"] is False and ok["tokens"] == 0
        assert submit(client, priority="bogus").status_code == 400
        assert submit(client, priority="urgent").status_code == 400  # urgent is a flag with its own scope
        assert submit(client, expire_after_s=0).status_code == 400
        norm = submit(client).json()
        listed = client.get("/v1/jobs", params={"priority": "high"}).json()
        assert [j["id"] for j in listed["jobs"]] == [ok["id"]] and norm["priority"] == "normal"
        assert client.get("/v1/jobs", params={"priority": "nope"}).status_code == 400
        hold.clear()


def test_a_high_job_goes_before_older_normal_jobs(monkeypatch):
    eng = SlowEngine(delay=0.02)
    client, _, app = make(eng, job_active=1)
    hold = hold_runner(monkeypatch, app.state.ctx.extra["jobs"].store)
    with client:
        a = submit(client).json()["id"]
        b = submit(client, priority="low").json()["id"]
        c = submit(client, priority="high").json()["id"]
        hold.clear()
        poke(client, app)
        jobs = [wait(client, j) for j in (a, b, c)]
        started = {j["id"]: j["started_at"] for j in jobs}
        assert started[c] < started[a] < started[b]  # high, then FIFO normal, then low


# ------------------------------------------------------------------ integration: windows, urgent, expiry


def test_off_hours_job_waits_in_the_queue_until_its_window_opens():
    client, _, app = make()
    t = fake_clock(app)
    with client:
        job = submit(client, schedule=WINDOW).json()
        assert job["status"] == "queued" and job["schedule"] == WINDOW
        assert job["waiting_reason"] == "off_hours" and job["next_window_at"] == "2026-10-19T20:00:00.000000Z"
        time.sleep(0.3)
        assert client.get(f"/v1/jobs/{job['id']}").json()["status"] == "queued"
        t[0] = NOON + 9 * HOUR
        poke(client, app)
        done = wait(client, job["id"])
        assert done["status"] == "succeeded" and done["waiting_reason"] is None


def test_default_window_comes_from_the_environment():
    client, _, app = make(job_window="20:00-06:00", job_tz="UTC")
    fake_clock(app)
    with client:
        job = submit(client).json()
        assert job["schedule"] == WINDOW and job["waiting_reason"] == "off_hours"
        assert client.post(f"/v1/jobs/{job['id']}/cancel").json()["status"] == "cancelled"
        bad = submit(client, schedule={"window": "nonsense"})
        assert bad.status_code == 400
        assert submit(client, schedule={"window": "01:00-02:00", "tz": "Mars/Base"}).status_code == 400


def test_a_running_job_that_leaves_its_window_stays_running_and_gets_no_turns():
    eng = SlowEngine(delay=0.03)
    client, _, app = make(eng, max_microbatch=4)
    t = fake_clock(app, NOON + 9 * HOUR)  # 21:00: inside
    with client:
        body = {"inputs": [f"t{i}" for i in range(60)], "labels": LABELS}
        jid = submit(client, payload=body, schedule=WINDOW).json()["id"]
        while progress(client, jid) < 8:
            time.sleep(0.01)
        t[0] = NOON + 19 * HOUR  # 07:00: closed
        poke(client, app)
        time.sleep(0.4)  # the chunk in flight may finish; then nothing more
        job = client.get(f"/v1/jobs/{jid}").json()
        frozen = job["progress"]["done"]
        assert job["status"] == "running" and job["waiting_reason"] == "off_hours"  # D23 #5
        assert job["next_window_at"] == "2026-10-20T20:00:00.000000Z"
        time.sleep(0.3)
        assert progress(client, jid) == frozen < 60
        t[0] = NOON + 33 * HOUR  # 21:00 next day
        poke(client, app)
        assert wait(client, jid)["status"] == "succeeded"


def test_a_paused_job_can_be_cancelled_without_waiting_for_its_window():
    eng = SlowEngine(delay=0.03)
    client, _, app = make(eng, max_microbatch=4)
    t = fake_clock(app, NOON + 9 * HOUR)
    with client:
        body = {"inputs": [f"t{i}" for i in range(60)], "labels": LABELS}
        jid = submit(client, payload=body, schedule=WINDOW).json()["id"]
        while progress(client, jid) < 4:
            time.sleep(0.01)
        t[0] = NOON + 19 * HOUR
        poke(client, app)
        time.sleep(0.3)
        assert client.post(f"/v1/jobs/{jid}/cancel").status_code == 200
        assert wait(client, jid, timeout=5)["status"] == "cancelled"


def test_urgent_needs_the_scope_and_runs_outside_the_window_and_slot_limit():
    client, _, app = make(SlowEngine(delay=0.02), api_key="root", job_active=1, max_microbatch=4)
    fake_clock(app)  # NOON: the window is closed
    plain, boss = hdr(add_key(app, "plain", "jobs")), hdr(add_key(app, "boss", "jobs,urgent"))
    with client:
        denied = submit(client, urgent=True, schedule=WINDOW, headers=plain)
        assert denied.status_code == 403 and "urgent" in denied.json()["detail"]
        # fill the only slot with a long job that has no window
        long = submit(
            client, payload={"inputs": [f"t{i}" for i in range(200)], "labels": LABELS}, headers=boss
        )
        blocker = long.json()["id"]
        queued = submit(client, schedule=WINDOW, headers=boss).json()
        assert queued["status"] == "queued" and queued["waiting_reason"] == "off_hours"
        urgent = submit(client, urgent=True, schedule=WINDOW, headers=boss)
        assert urgent.status_code == 202 and urgent.json()["priority"] == "urgent" and urgent.json()["urgent"]
        done = wait(client, urgent.json()["id"], headers=boss)
        assert done["status"] == "succeeded"  # window ignored, no free slot needed
        assert client.get(f"/v1/jobs/{blocker}", headers=boss).json()["status"] in ("running", "succeeded")
        client.post(f"/v1/jobs/{blocker}/cancel", headers=boss)


def test_expiry_counts_from_the_window_opening_and_cancels_the_job(monkeypatch):
    client, _, app = make(SlowEngine())
    t = fake_clock(app)
    store = app.state.ctx.extra["jobs"].store
    hold = hold_runner(monkeypatch, store)
    with client:
        job = submit(client, schedule=WINDOW, expire_after_s=HOUR).json()
        t[0] = NOON + 6 * HOUR  # 18:00: six hours after submit, but the window has not opened yet
        poke(client, app)
        time.sleep(0.2)
        assert client.get(f"/v1/jobs/{job['id']}").json()["status"] == "queued"
        t[0] = NOON + 11 * HOUR  # 23:00: window opened at 20:00, the hour is over (and it never got a slot)
        hold.clear()
        poke(client, app)
        out = wait(client, job["id"])
        assert out["status"] == "cancelled" and out["error"] == EXPIRED and out["expired"] is True


def test_a_running_job_expires_between_chunks_and_keeps_its_rows():
    client, _, app = make(SlowEngine(delay=0.1), max_microbatch=4)
    with client:
        body = {"inputs": [f"t{i}" for i in range(120)], "labels": LABELS}
        jid = submit(client, payload=body, expire_after_s=0.5).json()["id"]
        out = wait(client, jid)
        assert out["status"] == "cancelled" and out["expired"] is True and out["error"] == EXPIRED
        assert 0 < out["progress"]["done"] < 120
        assert (
            len(client.get(f"/v1/jobs/{jid}/results", params={"format": "ndjson"}).text.splitlines())
            == (out["progress"]["done"])
        )


# ------------------------------------------------------------------ integration: quota at submit


def used(app: Any, name: str) -> int:
    return app.state.ctx.quotas.used(name)


def test_quota_is_reserved_at_submit_rejected_when_it_does_not_fit_and_refunded_on_cancel(monkeypatch):
    client, _, app = make(SlowEngine(delay=0.02), api_key="root")
    h = hdr(add_key(app, "ci", "classify,jobs", daily_quota=5))
    hold = hold_runner(monkeypatch, app.state.ctx.extra["jobs"].store)
    with client:
        first = submit(client, headers=h)  # 3 records
        assert first.status_code == 202 and first.headers["X-Quota-Remaining"] == "2"
        assert used(app, "ci") == 3
        over = submit(client, headers=h)
        assert over.status_code == 429 and over.json()["detail"] == "daily quota exceeded (5 records/day)"
        assert over.headers["X-Quota-Remaining"] == "2" and "Retry-After" in over.headers
        assert used(app, "ci") == 3  # a rejection charges nothing
        client.post(f"/v1/jobs/{first.json()['id']}/cancel", headers=h)
        for _ in range(100):
            if used(app, "ci") == 0:
                break
            time.sleep(0.02)
        assert used(app, "ci") == 0  # nothing was processed: the whole reservation came back
        hold.clear()
        assert submit(client, headers=h).status_code == 202


def test_a_multi_day_job_is_charged_to_the_submit_day(monkeypatch):
    client, _, app = make(SlowEngine(), api_key="root")
    h = hdr(add_key(app, "ci", "jobs", daily_quota=10))
    with client:
        job = submit(client, schedule=WINDOW, headers=h).json()  # waits for tonight
        assert job["status"] == "queued"
        row = app.state.ctx.extra["jobs"].store.get(job["id"])
        assert row["quota_day"] == app.state.ctx.quotas.today() and row["quota_reserved"] == 3
        client.post(f"/v1/jobs/{job['id']}/cancel", headers=h)


def test_cache_hits_of_a_job_are_not_charged():
    client, _, app = make(HitEngine(), api_key="root", max_microbatch=4)
    h = hdr(add_key(app, "ci", "jobs", daily_quota=100))
    with client:
        body = {"inputs": [f"t{i}" for i in range(8)], "labels": LABELS}  # two chunks, one hit each
        jid = submit(client, payload=body, headers=h).json()["id"]
        assert used(app, "ci") == 8
        assert wait(client, jid, headers=h)["status"] == "succeeded"
        for _ in range(100):
            if used(app, "ci") == 6:
                break
            time.sleep(0.02)
        assert used(app, "ci") == 6
        row = app.state.ctx.extra["jobs"].store.get(jid)
        assert row["quota_reserved"] == 6


# ------------------------------------------------------------------ integration: restart and kinds


def test_scheduler_state_survives_a_restart():
    state = tempfile.mkdtemp(prefix="clef-sched-")
    client, _, app = make(SlowEngine(), state_dir=state, api_key="root")
    fake_clock(app)
    secret = add_key(app, "ci", "jobs", daily_quota=20)
    h = hdr(secret)
    with client:
        job = submit(client, schedule=WINDOW, priority="high", expire_after_s=2 * HOUR, headers=h).json()
        assert used(app, "ci") == 3
    client2, _, app2 = make(SlowEngine(), state_dir=state, api_key="root")
    t = fake_clock(app2)
    with client2:
        again = client2.get(f"/v1/jobs/{job['id']}", headers=h).json()
        assert again["status"] == "queued" and again["priority"] == "high" and again["schedule"] == WINDOW
        assert again["waiting_reason"] == "off_hours" and again["next_window_at"] == job["next_window_at"]
        assert used(app2, "ci") == 3  # the reservation was flushed at shutdown
        t[0] = NOON + 9 * HOUR
        poke(client2, app2)
        assert wait(client2, job["id"], headers=h)["status"] == "succeeded"
        assert used(app2, "ci") == 3


def test_a_restarted_job_in_progress_resumes_and_keeps_its_class():
    state = tempfile.mkdtemp(prefix="clef-sched-")
    body = {"inputs": [f"t{i}" for i in range(40)], "labels": LABELS}
    client, _, _ = make(SlowEngine(delay=0.05), state_dir=state, max_microbatch=4)
    with client:
        jid = submit(client, payload=body, priority="low").json()["id"]
        while progress(client, jid) < 8:
            time.sleep(0.01)
    client2, _, _ = make(SlowEngine(), state_dir=state, max_microbatch=4)
    with client2:
        done = wait(client2, jid)
        assert done["status"] == "succeeded" and done["priority"] == "low" and done["progress"]["done"] == 40
        rows = client2.get(f"/v1/jobs/{jid}/results", params={"format": "ndjson"}).text.splitlines()
        assert [json.loads(r)["index"] for r in rows] == list(range(40))


def test_completed_if_all_failed_and_tokens_for_custom_kinds():
    client, _, app = make()

    async def run(ctx: Any, parsed: Any, job: Any) -> dict:
        job.set_total(3)
        await job.add_items([{"index": i, "error": "bad"} for i in range(3)], tokens=7)
        return {}

    kinds = app.state.ctx.extra["job_kinds"]
    kinds["flaky"] = {"validate": lambda p: p, "run": run, "completed_if_all_failed": True}
    kinds["strict"] = {"validate": lambda p: p, "run": run}
    with client:
        ok = wait(client, submit(client, "flaky", {"inputs": [1, 2, 3]}).json()["id"])
        bad = wait(client, submit(client, "strict", {"inputs": [1, 2, 3]}).json()["id"])
    assert ok["status"] == "succeeded" and ok["tokens"] == 7 and ok["progress"]["failed"] == 3
    assert bad["status"] == "failed"
