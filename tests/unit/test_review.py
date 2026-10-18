"""The review store: dedup, states, purge / TTL (D17), per-owner listing, LLM-call log."""

from __future__ import annotations

import time

import pytest

from clef_server.config import Config
from clef_server.db import Database, database_url
from clef_server.review import ReviewStore, input_sha

pytestmark = pytest.mark.postgres  # also run on Postgres when CLEF_TEST_DATABASE_URL is set (conftest.py)

DAY = 86400.0


@pytest.fixture
def store(tmp_path):
    db = Database(database_url(Config(state_dir=str(tmp_path))))
    db.upgrade()
    return ReviewStore(db)


def item(text="hello", **kw):
    return {"input": text, "reason": "other", "state": "buffered", "router": "r1", "owner": "alice", **kw}


def age(store, rid, days):
    from sqlalchemy import text

    with store.db.tx() as conn:
        conn.execute(
            text("UPDATE review SET created_at = :t WHERE id = :i"), {"t": time.time() - days * DAY, "i": rid}
        )


def test_add_items_stores_text_and_dedups_open_inputs_per_router(store):
    [(a, created_a)] = store.add_items([item("same text")])
    [(b, created_b)] = store.add_items([item("same text")])
    [(c, created_c)] = store.add_items([item("same text", router="r2")])
    assert created_a and not created_b and created_c and a == b != c
    row = store.get(a)
    assert row["input"] == "same text" and row["input_sha256"] == input_sha("same text")
    assert row["state"] == "buffered" and row["kind"] == "item" and row["id"].startswith("rev_")


def test_a_resolved_input_can_be_captured_again(store):
    [(a, _)] = store.add_items([item("again")])
    store.update_row(a, state="resolved", input=None, label="billing")
    [(b, created)] = store.add_items([item("again")])
    assert created and b != a


def test_classifier_items_dedup_per_classifier(store):
    base = {"reason": "low_confidence", "state": "open", "router": None, "owner": "alice"}
    [(a, _)] = store.add_items([{**base, "input": "x", "classifier": "c1"}])
    [(b, created_b)] = store.add_items([{**base, "input": "x", "classifier": "c1"}])
    [(c, created_c)] = store.add_items([{**base, "input": "x", "classifier": "c2"}])
    assert a == b and not created_b and created_c and c != a


def test_purge_drops_text_after_the_ttl_and_expires_open_items(store):
    old_buf, old_open, fresh, resolved = (
        store.add_items([item("1")])[0][0],
        store.add_items([item("2", state="open")])[0][0],
        store.add_items([item("3")])[0][0],
        store.add_items([item("4")])[0][0],
    )
    store.set_state([resolved], "resolved", by="alice")  # resolved but (as if) the text lingered
    for rid in (old_buf, old_open, resolved):
        age(store, rid, 31)
    assert store.purge(30) == 3
    assert store.get(old_buf)["state"] == "expired" and store.get(old_buf)["input"] is None
    assert store.get(old_open)["state"] == "expired" and store.get(old_open)["input"] is None
    kept = store.get(resolved)
    assert kept["state"] == "resolved" and kept["input"] is None and kept["input_sha256"]
    assert store.get(fresh)["state"] == "buffered" and store.get(fresh)["input"] == "3"
    assert store.purge(30) == 0  # idempotent


def test_purge_is_off_with_a_zero_ttl_and_uses_the_given_clock(store):
    rid = store.add_items([item("x")])[0][0]
    age(store, rid, 400)
    assert store.purge(0) == 0 and store.get(rid)["input"] == "x"
    assert store.purge(30, now=time.time() - 500 * DAY) == 0  # the clock before the row was created
    assert store.purge(30) == 1


def test_bump_attempts_makes_permanent_other_after_two(store):
    rid = store.add_items([item("x")])[0][0]
    store.bump_attempts([rid])
    assert store.get(rid)["state"] == "buffered" and store.get(rid)["reroutes"] == 1
    store.bump_attempts([rid])
    row = store.get(rid)
    assert row["state"] == "permanent_other" and row["reroutes"] == 2
    assert row["input"] is None and row["input_sha256"] == input_sha("x")  # D17: the text goes with it
    assert store.buffered("r1", 10) == []  # out of future batches


def test_buffered_returns_oldest_first_and_only_items_with_text(store):
    ids = [store.add_items([item(str(i))])[0][0] for i in range(3)]
    for n, rid in enumerate(ids):
        age(store, rid, 3 - n)
    store.update_row(ids[0], input=None)
    assert [r["id"] for r in store.buffered("r1", 10)] == ids[1:]
    assert store.counts("r1") == {"buffered": 3, "open_reviews": 0}


def test_human_labels_only_come_from_people_not_from_the_router(store):
    a = store.add_items([item("a")])[0][0]
    b = store.add_items([item("b")])[0][0]
    store.update_row(a, state="resolved", label="billing", resolved_by="alice", input=None)
    store.update_row(b, state="resolved", label="billing", resolved_by="router:r1", input=None)
    got = store.human_labels("r1", [input_sha("a"), input_sha("b"), input_sha("c")])
    assert got == {input_sha("a"): "billing"}


def test_claim_resolves_once(store):
    rid = store.add_items([item("x", state="open")])[0][0]
    assert store.claim(rid, ("open", "buffered"), "resolved")
    assert not store.claim(rid, ("open", "buffered"), "resolved")


def test_list_filters_and_scopes_by_owner(store):
    store.add_items([item("1", owner="alice"), item("2", owner="bob", router="r2", state="open")])
    store.add_proposal(
        state="open",
        owner="alice",
        router="r1",
        classifier="c1",
        classifier_version=1,
        reason="proposal",
        proposal={"labels": []},
    )
    rows, total = store.list()
    assert total == 3 and {r["kind"] for r in rows} == {"item", "proposal"}
    assert store.list(kind="proposal")[1] == 1 and store.list(state="open")[1] == 2
    assert store.list(router="r2")[1] == 1
    mine, n = store.list(owner="alice", scoped=True)
    assert n == 2 and all(r["owner"] == "alice" for r in mine)
    assert store.list(owner=None, scoped=True)[1] == 0  # owner-less rows are admin-only
    assert len(store.list(limit=1, offset=1)[0]) == 1


def test_drop_router_discards_items_text_and_rejects_open_proposals(store):
    a = store.add_items([item("1")])[0][0]
    p = store.add_proposal(
        state="open",
        owner="alice",
        router="r1",
        classifier="c1",
        classifier_version=None,
        reason="proposal",
        proposal={},
    )
    store.drop_router("r1", by="alice")
    assert store.get(a)["state"] == "discarded" and store.get(a)["input"] is None
    assert store.get(p)["state"] == "rejected"


def test_llm_call_log_and_usage(store):
    kw = dict(router="r1", purpose="propose", provider_url="https://x/v1/", model="m", items=20)
    store.log_call(status="ok", latency_ms=12.0, input_tokens=100, output_tokens=40, **kw)
    store.log_call(status="error", latency_ms=None, input_tokens=None, output_tokens=None, error="boom", **kw)
    use = store.llm_usage(time.time() - 60, "r1")
    assert use == {"calls": 2, "input_tokens": 100, "output_tokens": 40, "errors": 1}
    assert store.llm_usage(time.time() + 60)["calls"] == 0
