"""Result cache (D3): the LRU itself and its engine integration (CPU only, fake model)."""

from __future__ import annotations

import asyncio
import threading

import pytest
from test_engine import FakeModel, expected, make_cfg, rec, started  # noqa: F401  (fixture)

from clef_server.cache import ResultCache
from clef_server.config import Config
from clef_server.engine import GpuOutOfMemory
from clef_server.stats import Stats


def run(coro):
    return asyncio.run(coro)


def res(tag="a"):
    return {
        "model": "m",
        "answers": {"d": {"probs": {tag: 1.0}}},
        "usage": {"input_tokens": 3, "output_tokens": 0},
    }


# ---------------------------------------------------------------- the LRU
def test_key_depends_on_question_order():
    qa = {"type": "noul", "instructions": "Is it positive?"}
    qb = {"type": "choice", "instructions": "Topic", "criteria": {"x": "sports", "y": "politics"}}
    r1 = {"state": "hello", "questions": {"a": qa, "b": qb}}
    r2 = {"state": "hello", "questions": {"b": qb, "a": qa}}
    assert ResultCache.key("m@1", r1) != ResultCache.key("m@1", r2)
    assert ResultCache.key("m@1", r1) == ResultCache.key(
        "m@1", {"state": "hello", "questions": {"a": qa, "b": qb}}
    )


def test_key_depends_on_criteria_order():
    def r(c):
        return {"state": "s", "questions": {"q": {"type": "choice", "criteria": c}}}

    assert ResultCache.key("m", r({"x": "1", "y": "2"})) != ResultCache.key("m", r({"y": "2", "x": "1"}))
    assert ResultCache.key("m", r({"x": "1", "y": "2"})) == ResultCache.key("m", r({"x": "1", "y": "2"}))


def test_byte_budget_evicts_lru_and_skips_oversized():
    def big(t, n):
        return {**res(t), "pad": "x" * n}

    c = ResultCache(1000, max_bytes=10000)
    for k in "abcdefghijklmnop":  # ~700 bytes each, 16 of them exceed the budget
        c.put(k, big(k, 600))
    st = c.stats()
    assert st["bytes"] <= 10000 and st["evictions"] >= 1 and c.get("a") is None and c.get("p") is not None
    c.put("huge", big("h", 1500))  # > 1/10 of the budget: skipped
    assert c.get("huge") is None
    assert ResultCache(100, max_bytes=0).enabled is False
    c.clear()
    assert c.stats()["bytes"] == 0


def test_key_is_canonical_and_ignores_field_order():
    a = {"state": "hi", "questions": {"d": {"type": "choice", "criteria": {"a": "x", "b": "y"}}}}
    b = {
        "questions": {"d": {"criteria": {"a": "x", "b": "y"}, "type": "choice"}},
        "state": "hi",
        "model": "z",
    }
    assert ResultCache.key("m@1", a) == ResultCache.key("m@1", b)
    assert ResultCache.key("m@1", a) != ResultCache.key("m@2", a)
    assert ResultCache.key("m@1", a) != ResultCache.key("m@1", {**a, "state": "ho"})
    assert ResultCache.key("m@1", {**a, "images": ["x"]}) is None
    assert ResultCache.key("m@1", {**a, "videos": ["x"]}) is None


def test_lru_eviction_and_recency():
    c = ResultCache(2)
    c.put("a", res())
    c.put("b", res())
    assert c.get("a") is not None  # a is now the MRU
    c.put("c", res())  # evicts b
    assert c.get("b") is None
    assert c.get("a") is not None and c.get("c") is not None
    s = c.stats()
    assert (s["size"], s["capacity"], s["evictions"]) == (2, 2, 1)
    assert (s["hits"], s["misses"]) == (3, 1)


def test_disabled_cache_is_a_noop():
    c = ResultCache(0)
    c.put("a", res())
    assert c.get("a") is None and len(c) == 0 and not c.enabled
    assert c.stats()["misses"] == 0


def test_get_returns_private_copy_and_put_drops_timing():
    c = ResultCache(4)
    r = {**res(), "timing": {"forward_ms": 5}}
    c.put("a", r)
    got = c.get("a")
    assert "timing" not in got
    got["answers"]["d"]["probs"]["a"] = 0.0
    assert c.get("a")["answers"]["d"]["probs"]["a"] == 1.0
    assert r["timing"] == {"forward_ms": 5}  # the caller's object is untouched


def test_thread_safety():
    c = ResultCache(50)
    errors = []

    def work(seed):
        try:
            for i in range(2000):
                k = str((i * 7 + seed) % 120)
                if c.get(k) is None:
                    c.put(k, res())
        except Exception as exc:  # pragma: no cover
            errors.append(exc)

    ts = [threading.Thread(target=work, args=(s,)) for s in range(8)]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    assert not errors
    s = c.stats()
    assert s["size"] <= 50 and s["hits"] + s["misses"] == 8 * 2000
    assert s["evictions"] >= s["size"] and len(c) == s["size"]


def test_config_cache_size(monkeypatch):
    assert Config().cache_size == 10000
    monkeypatch.setenv("CLEF_CACHE_SIZE", "0")
    assert Config().cache_size == 0
    monkeypatch.setenv("CLEF_CACHE_SIZE", "-1")
    with pytest.raises(ValueError):
        Config()


# ---------------------------------------------------------------- engine integration
def probs(r):
    return r["answers"]["d"]["probs"]


def test_hit_returns_identical_result(started):  # noqa: F811
    eng, model = started()
    first = run(eng.decide([rec("alpha beta")]))[0]
    second = run(eng.decide([rec("alpha beta")]))[0]
    assert len(model.calls) == 1  # the second call never reached the model
    assert first["timing"]["cached"] is False
    assert second["timing"] == {"queue_ms": 0, "forward_ms": 0, "batch_size": 0, "cached": True}
    assert probs(second) == probs(first) == pytest.approx(expected("alpha beta"), abs=1e-4)
    assert second["usage"] == first["usage"] and second["model"] == first["model"]
    second["answers"]["d"]["probs"]["a"] = -1  # callers may mutate a result
    assert probs(run(eng.decide([rec("alpha beta")]))[0])["a"] != -1


def test_mixed_batch_keeps_input_order(started):  # noqa: F811
    eng, model = started()
    run(eng.decide([rec("one"), rec("three words here")]))
    model.calls.clear()
    states = ["new a", "one", "new bb", "three words here", "new ccc"]
    out = run(eng.decide([rec(s) for s in states]))
    assert [r["timing"]["cached"] for r in out] == [False, True, False, True, False]
    for s, r in zip(states, out, strict=True):
        assert probs(r) == pytest.approx(expected(s), abs=1e-4)
    assert sum(c[0] for c in model.calls) == 3  # only the misses were forwarded


def test_duplicates_in_one_call_run_once(started):  # noqa: F811
    eng, model = started()
    out = run(eng.decide([rec("same"), rec("other"), rec("same")]))
    assert sum(c[0] for c in model.calls) == 2
    assert [r["timing"]["cached"] for r in out] == [False, False, True]
    assert probs(out[0]) == probs(out[2])
    st = eng.cache_stats()  # the duplicate is a hit, matching its timing.cached marker
    assert st["hits"] == 1 and st["misses"] == 2


def test_no_cache_bypasses_lookup_but_stores(started):  # noqa: F811
    eng, model = started()
    run(eng.decide([rec("fresh")], cache=False))
    run(eng.decide([rec("fresh")], cache=False))
    assert sum(c[0] for c in model.calls) == 2
    assert eng.cache_stats()["bypassed"] == 2 and eng.cache_stats()["hits"] == 0
    assert run(eng.decide([rec("fresh")]))[0]["timing"]["cached"] is True  # stored by the bypassing calls


def test_media_is_never_cached(started):  # noqa: F811
    eng, model = started()
    run(eng.decide([rec("pic", media=True)]))
    run(eng.decide([rec("pic", media=True)]))
    assert sum(c[0] for c in model.calls) == 2
    assert eng.cache_stats()["size"] == 0


def test_errors_are_not_cached(started):  # noqa: F811
    eng, model = started()
    for _ in range(2):
        with pytest.raises(GpuOutOfMemory):
            run(eng.decide([rec("OOM")]))
    assert eng.cache_stats()["size"] == 0 and len(model.calls) == 2
    # a sibling that forwarded fine (another length bucket) is still stored when the call fails
    long = " ".join(["w"] * 20)
    with pytest.raises(GpuOutOfMemory):
        run(eng.decide([rec(long), rec("OOM")]))
    assert run(eng.decide([rec(long)]))[0]["timing"]["cached"] is True


def test_cache_off(started):  # noqa: F811
    eng, model = started(make_cfg(cache_size=0))
    run(eng.decide([rec("x")]))
    out = run(eng.decide([rec("x")]))
    assert len(model.calls) == 2 and out[0]["timing"]["cached"] is False
    assert eng.cache_stats()["enabled"] is False


def test_background_lane_shares_the_cache(started):  # noqa: F811
    """Jobs call decide(priority='background'); interactive hits serve them and vice versa."""
    eng, model = started()
    run(eng.decide([rec("shared text")]))
    out = run(eng.decide([rec("shared text"), rec("job only")], priority="background"))
    assert [r["timing"]["cached"] for r in out] == [True, False]
    assert run(eng.decide([rec("job only")]))[0]["timing"]["cached"] is True
    assert sum(c[0] for c in model.calls) == 2


def test_prepared_slice_keeps_hits_and_dups(started):  # noqa: F811
    eng, _ = started()
    run(eng.decide([rec("h")]))
    prep = run(eng.prepare([rec("h"), rec("d"), rec("d"), rec("z")]))
    assert set(prep.hits) == {0} and prep.dups == {2: 1}
    part = prep.slice(2, 4)  # the dup lost its source: it becomes a plain miss (shared Encoded)
    assert part.dups == {} and part.encoded[0] is not None
    out = run(eng.submit(part))
    assert probs(out[0]) == pytest.approx(expected("d"), abs=1e-4)


def test_model_load_clears_cache(started):  # noqa: F811
    eng, _ = started()
    run(eng.decide([rec("keep")]))
    assert eng.cache_stats()["size"] == 1
    eng.cache.clear()
    assert run(eng.decide([rec("keep")]))[0]["timing"]["cached"] is False


def test_stats_cache_block(started):  # noqa: F811
    stats = Stats()
    eng, _ = started()
    stats.set_cache_source(eng.cache_stats)
    run(eng.decide([rec("s1"), rec("s2")]))
    run(eng.decide([rec("s1"), rec("s3")]))
    block = stats.snapshot()["cache"]
    assert block["hits"] == 1 and block["misses"] == 3 and block["size"] == 3
    assert block["hit_rate"] == 0.25 and block["evictions"] == 0 and block["capacity"] == 10000
    assert Stats().snapshot()["cache"]["enabled"] is False  # no engine registered
