"""Stats: ring buffer, windowed percentiles, histogram, forward stats, SSE fan-out."""

from __future__ import annotations

import asyncio
import threading
import time

import pytest

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


# ---- time series (deterministic clock)
class Clock:
    def __init__(self, t: float = 1_000_000.0):
        self.t = t

    def __call__(self) -> float:
        return self.t


def mk(**kw):
    c = Clock()
    return Stats(clock=c, **kw), c


def test_ts_alignment_and_shape():
    s, c = mk()
    c.t = 1_000_003.4
    ts = s.timeseries(window_s=300, step_s=5)
    assert ts["window_s"] == 300 and ts["step_s"] == 5 and ts["now"] == 1_000_003.4
    assert len(ts["t"]) == 60
    assert all(t % 5 == 0 for t in ts["t"])
    assert ts["t"][-1] == 1_000_000 and ts["t"][0] == 1_000_000 - 59 * 5
    assert all(b - a == 5 for a, b in zip(ts["t"], ts["t"][1:], strict=False))
    for k in (
        "rps",
        "p50_ms",
        "p95_ms",
        "error_rate",
        "avg_batch",
        "queue_depth",
        "padding_ratio",
        "mem_used_gb",
        "gpu_util_pct",
        "gpu_temp_c",
        "gpu_power_w",
    ):
        assert len(ts[k]) == 60
    # empty: rps 0, everything else null
    assert set(ts["rps"]) == {0}
    for k in ("p50_ms", "p95_ms", "error_rate", "avg_batch", "queue_depth", "padding_ratio", "mem_used_gb"):
        assert set(ts[k]) == {None}


def test_ts_default_step_and_non_divisible_window():
    s, _ = mk()
    assert s.timeseries(300)["step_s"] == 5
    assert s.timeseries(900)["step_s"] == 15
    assert s.timeseries(3600)["step_s"] == 60
    assert s.timeseries(10)["step_s"] == 1
    ts = s.timeseries(window_s=100, step_s=30)
    assert len(ts["t"]) == 4  # ceil(100/30)


def test_ts_partial_step_rps_divides_by_full_step():
    s, c = mk()
    c.t = 1_000_000.0  # aligned to 5
    for _ in range(10):
        s.record_request(req(10.0))
    c.t = 1_000_001.5  # 1.5 s into the step; rps still /5
    ts = s.timeseries(300, 5)
    assert ts["t"][-1] == 1_000_000 and ts["rps"][-1] == 2.0
    lp = s.latest_point(5)
    assert lp["t"] == 1_000_000 and lp["rps"] == 2.0 and lp["step_s"] == 5


def test_ts_requests_land_in_correct_steps_and_percentiles():
    s, c = mk()
    c.t = 1_000_000.2
    for ms in range(1, 101):
        s.record_request(req(float(ms)))
    c.t = 1_000_005.1
    s.record_request(req(7.0))
    c.t = 1_000_007.0
    ts = s.timeseries(300, 5)
    assert ts["t"][-2:] == [1_000_000, 1_000_005]
    assert ts["rps"][-2:] == [20.0, 0.2]
    assert ts["p50_ms"][-2:] == [50.0, 7.0]
    assert ts["p95_ms"][-2:] == [95.0, 7.0]
    assert ts["p50_ms"][-3] is None


def test_ts_error_rate_and_errors_excluded_from_latency():
    s, c = mk()
    s.record_request(req(10.0))
    s.record_request(req(1000.0, status=500))
    s.record_request(req(20.0, status=429))
    s.record_request(req(30.0))
    ts = s.timeseries(10, 1)
    assert ts["error_rate"][-1] == round(2 / 4, 4)
    assert ts["p95_ms"][-1] == 30.0 and ts["p50_ms"][-1] == 10.0
    assert ts["rps"][-1] == 4.0
    # only-errors step: error_rate 1.0, latency null
    c.t += 1
    s.record_request(req(5.0, status=400))
    ts = s.timeseries(10, 1)
    assert ts["error_rate"][-1] == 1.0 and ts["p50_ms"][-1] is None


def test_ts_forward_avg_batch_and_padding():
    s, c = mk()
    s.record_forward(2, 300, 400, 10.0)
    s.record_forward(4, 700, 800, 10.0)
    c.t += 1
    s.record_forward(1, 64, 64, 1.0)
    ts = s.timeseries(10, 1)
    assert ts["avg_batch"][-2:] == [3.0, 1.0]
    assert ts["padding_ratio"][-2:] == [round(1 - 1000 / 1200, 4), 0.0]
    assert ts["avg_batch"][-3] is None and ts["padding_ratio"][-3] is None


def test_ts_gauges_last_sample_and_none_and_unknown_keys():
    s, c = mk()
    s.sample({"mem_used_gb": 10.0, "gpu_util_pct": 50, "gpu_temp_c": 60, "gpu_power_w": 100, "bogus": 1})
    c.t += 1
    s.sample({"mem_used_gb": 11.0, "gpu_util_pct": 70, "gpu_temp_c": None, "gpu_power_w": None})
    c.t += 1
    s.sample({"mem_used_gb": 12.0})
    ts = s.timeseries(10, 5)
    # steps are 5s wide: the last sample in the step wins, None does not erase
    assert ts["mem_used_gb"][-1] == 12.0 and ts["gpu_util_pct"][-1] == 70.0
    assert ts["gpu_temp_c"][-1] == 60.0 and ts["gpu_power_w"][-1] == 100.0
    assert "bogus" not in ts
    ts1 = s.timeseries(10, 1)
    assert ts1["mem_used_gb"][-1] == 12.0 and ts1["mem_used_gb"][-2] == 11.0
    assert ts1["gpu_temp_c"][-2] is None
    assert ts1["mem_used_gb"][-3] == 10.0 and ts1["mem_used_gb"][-4] is None


def test_ts_queue_depth_max_from_set_and_sample():
    s, c = mk()
    c.t = 1_000_000.0
    assert s.timeseries(10, 5)["queue_depth"][-1] is None  # before first observation
    s.set_queue_depth(2)
    s.set_queue_depth(7)
    s.set_queue_depth(1)
    c.t = 1_000_001.0
    s.sample({"queue_depth": 4})
    ts = s.timeseries(10, 5)
    assert ts["queue_depth"][-1] == 7
    assert s.snapshot()["queue_depth"] == 4  # sample updates the live gauge
    # later empty step carries forward; latest_point falls back to the live depth
    c.t = 1_000_005.0
    assert s.timeseries(10, 5)["queue_depth"][-1] == 7
    assert s.latest_point(5)["queue_depth"] == 4
    s.set_queue_depth(0)
    assert s.latest_point(5)["queue_depth"] == 0


def test_ts_latency_cap_per_second():
    s, _ = mk()
    for i in range(1500):
        s.record_request(req(float(i)))
    assert len(s._ring[1_000_000 % 3600].lats) == 1000
    assert s.timeseries(10, 1)["rps"][-1] == 1500.0  # counts are not capped


def test_ts_retention_wraparound_does_not_leak():
    s, c = mk(retention_s=60)
    c.t = 1_000_000.0
    for _ in range(5):
        s.record_request(req(111.0))
    s.sample({"mem_used_gb": 9.0})
    s.record_forward(3, 10, 20, 1.0)
    c.t = 1_000_060.0  # same ring slot, one full lap later
    ts = s.timeseries(60, 1)
    assert sum(ts["rps"]) == 0 and set(ts["p50_ms"]) == {None}
    assert set(ts["mem_used_gb"]) == {None} and set(ts["avg_batch"]) == {None}
    s.record_request(req(5.0))
    ts = s.timeseries(60, 1)
    assert sum(ts["rps"]) == 1.0 and ts["p50_ms"][-1] == 5.0
    # window larger than retention: data older than retention is simply gone
    s2, c2 = mk(retention_s=20)
    c2.t = 1_000_000.0
    s2.record_request(req(1.0))
    c2.t = 1_000_019.0
    assert sum(s2.timeseries(60, 1)["rps"]) == 1.0
    c2.t = 1_000_020.0
    assert sum(s2.timeseries(60, 1)["rps"]) == 0.0


def test_ts_old_data_ages_out_of_window():
    s, c = mk()
    s.record_request(req(10.0))
    c.t += 400
    assert sum(s.timeseries(300, 5)["rps"]) == 0
    assert sum(s.timeseries(900, 15)["rps"]) > 0


def test_ts_validation_errors():
    s, _ = mk()
    bad = [(9, None), (3601, None), (300, 0), (300, 301), (3600, 1), (300, -5), (0, 1)]
    for w, st in bad:
        with pytest.raises(ValueError):
            s.timeseries(w, st)
    s.timeseries(3600, 5)  # 720 points: ok
    with pytest.raises(ValueError, match="step_s"):
        s.latest_point(0)


def test_latest_point_fields():
    s, c = mk()
    c.t = 1_000_007.0
    s.record_request(req(40.0))
    s.record_request(req(50.0, status=500))
    s.record_forward(2, 100, 128, 5.0)
    s.sample({"mem_used_gb": 3.5, "gpu_util_pct": 12})
    lp = s.latest_point(5)
    assert lp == {
        "t": 1_000_005,
        "step_s": 5,
        "rps": 0.4,
        "p50_ms": 40.0,
        "p95_ms": 40.0,
        "error_rate": 0.5,
        "avg_batch": 2.0,
        "queue_depth": 0,
        "padding_ratio": round(1 - 100 / 128, 4),
        "mem_used_gb": 3.5,
        "gpu_util_pct": 12.0,
        "gpu_temp_c": None,
        "gpu_power_w": None,
    }
    ts = s.timeseries(300, 5)
    for k, v in lp.items():
        if k not in ("step_s", "queue_depth"):
            assert ts[k][-1] == v


def test_by_key_and_by_endpoint():
    s, c = mk()
    s.record_request(req(1.0, key="alice"))
    s.record_request(req(1.0, key="alice", endpoint="/v1/classify"))
    s.record_request(req(1.0))
    s.record_request(req(1.0, key=None, endpoint="/v1/classify"))
    assert s.log()[0]["key"] == "alice" and s.log()[2]["key"] is None
    c.t += 1000
    s.record_request(req(1.0, key="bob"))
    assert s.snapshot(window_s=300)["by_key"] == {"bob": 1}
    snap = s.snapshot(window_s=3600)
    assert snap["by_key"] == {"alice": 2, "anonymous": 2, "bob": 1}
    assert snap["by_endpoint"] == {"/v1/systemone": 3, "/v1/classify": 2}
    assert Stats().snapshot()["by_key"] == {} and Stats().snapshot()["by_endpoint"] == {}


def test_ts_thread_safety_smoke():
    s = Stats(buffer=50, retention_s=30)
    stop = time.time() + 0.4
    errors: list[Exception] = []

    def writer():
        while time.time() < stop:
            s.record_request(req(1.0, key="k"))
            s.record_forward(1, 1, 2, 1.0)
            s.set_queue_depth(1)
            s.sample({"mem_used_gb": 1.0, "queue_depth": 2})

    def reader():
        try:
            while time.time() < stop:
                s.timeseries(10, 1)
                s.latest_point(1)
                s.snapshot()
        except Exception as e:  # pragma: no cover
            errors.append(e)

    ts = [threading.Thread(target=writer) for _ in range(3)] + [threading.Thread(target=reader)]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    assert not errors
    assert sum(s.timeseries(10, 10)["rps"]) > 0
