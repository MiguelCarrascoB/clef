"""Stats: ring buffer, windowed percentiles, histogram, forward stats, SSE fan-out."""

from __future__ import annotations

import asyncio
import threading
import time

from clef_server.stats import HIST_EDGES, Stats


def req(ms: float, status: int = 200, **kw):
    return {"endpoint": "/v1/systemone", "status": status, "ms": ms, **kw}


def test_ring_buffer_ids_and_log():
    s = Stats(buffer=5)
    for i in range(8):
        s.record_request(req(float(i)))
    log = s.log(limit=100)
    assert [e["id"] for e in log] == [4, 5, 6, 7, 8]
    assert [e["id"] for e in s.log(limit=2)] == [7, 8]
    assert [e["id"] for e in s.log(since=6)] == [7, 8]
    assert log[0]["media"] == {"images": 0, "videos": 0} and log[0]["state_preview"] is None
    assert s.snapshot()["total"] == 8


def test_percentiles_and_hist():
    s = Stats()
    for ms in range(1, 101):  # 1..100 ms
        s.record_request(req(float(ms)))
    s.record_request(req(9000.0))
    s.record_request(req(5.0, status=400))  # errors excluded from latency, counted in errors
    snap = s.snapshot()
    lat = snap["latency_ms"]
    assert lat["p50"] == 51.0 and lat["p95"] == 96.0 and lat["max"] == 9000.0
    assert snap["errors"] == 1 and snap["total"] == 102
    hist = lat["hist"]
    assert hist["edges"] == HIST_EDGES and len(hist["counts"]) == len(HIST_EDGES) + 1
    assert hist["counts"][0] == 24  # 1..24 < 25
    assert hist["counts"][1] == 25  # 25..49
    assert hist["counts"][2] == 50  # 50..99
    assert hist["counts"][3] == 1  # 100
    assert hist["counts"][-1] == 1  # 9000 >= 5000
    assert sum(hist["counts"]) == 101


def test_window_excludes_old_entries():
    s = Stats()
    s.record_request(req(10.0))
    s._entries[0]["ts"] -= 1000
    s.record_request(req(20.0))
    snap = s.snapshot(window_s=60)
    assert snap["latency_ms"]["p50"] == 20.0 and snap["rps"] > 0
    assert s.snapshot(window_s=3600)["latency_ms"]["p50"] == 10.0


def test_empty_snapshot():
    snap = Stats().snapshot()
    assert snap["latency_ms"]["p50"] == 0 and snap["rps"] == 0 and snap["forward"]["padding_ratio"] == 0


def test_forward_and_tokens():
    s = Stats()
    s.record_forward(2, 300, 400, 100.0)
    s.record_forward(4, 700, 800, 200.0)
    s.set_queue_depth(3)
    s.record_request(req(5.0, input_tokens=300))
    s.record_request(req(5.0, input_tokens=100))
    snap = s.snapshot()
    f = snap["forward"]
    assert f["count"] == 2 and f["avg_batch"] == 3.0 and f["avg_ms"] == 150.0
    assert f["padding_ratio"] == round(1 - 1000 / 1200, 4)
    assert snap["queue_depth"] == 3 and snap["tokens"] == {"in_total": 400, "avg_in": 200}


def test_in_flight():
    s = Stats()
    with s.in_flight():
        assert s.snapshot()["in_flight"] == 1
    assert s.snapshot()["in_flight"] == 0
    try:
        with s.in_flight():
            raise RuntimeError
    except RuntimeError:
        pass
    assert s.snapshot()["in_flight"] == 0


def test_subscribe_receives_from_other_thread_and_unsubscribes():
    s = Stats()

    async def main():
        agen = s.subscribe()
        first = asyncio.ensure_future(agen.__anext__())
        await asyncio.sleep(0.05)  # let it register
        threading.Thread(target=lambda: s.record_request(req(1.0))).start()
        entry = await asyncio.wait_for(first, 2)
        assert entry["id"] == 1 and entry["endpoint"] == "/v1/systemone"
        s.record_request(req(2.0))
        assert (await asyncio.wait_for(agen.__anext__(), 2))["id"] == 2
        await agen.aclose()
        assert s._subs == []

    asyncio.run(main())


def test_concurrent_writers_and_readers():
    s = Stats(buffer=50)
    stop = time.time() + 0.3

    def writer():
        while time.time() < stop:
            s.record_request(req(1.0))
            s.record_forward(1, 1, 2, 1.0)

    ts = [threading.Thread(target=writer) for _ in range(3)]
    for t in ts:
        t.start()
    while time.time() < stop:
        s.snapshot()
        s.log()
    for t in ts:
        t.join()
    assert s.snapshot()["total"] > 0
