"""The router and the Review queue over HTTP: gate + buffering, cold start, LLM proposals, the re-check,
review vs auto mode, grow-only and the cap, human labels, purge, scopes.

No network: a fake LLM and a scripted engine."""

from __future__ import annotations

import asyncio
import tempfile
import time
from typing import Any

import pytest
from fastapi.testclient import TestClient
from test_api import FakeEngine
from test_keys import add_key, hdr

from clef_server.config import Config
from clef_server.llm import LLMError, LLMResult, validate_schema
from clef_server.main import create_app

DAY = 86400.0
SEEDS = {"billing": "money, invoices, refunds", "technical": "bugs and outages"}


class ScriptEngine(FakeEngine):
    """Picks the winner of a choice question by keyword: ``rules`` = (keyword, label, probability). The label
    only counts when it is one of the options; anything else is "other" (when offered) at 0.9."""

    def __init__(self) -> None:
        super().__init__()
        self.rules: list[tuple[str, str, float]] = [
            ("vague", "billing", 0.4),  # below tau: a low-confidence winner
            ("tie", "billing", 0.5),  # a close second place is added in answer()
            ("invoice", "billing", 0.9),
            ("crash", "technical", 0.9),
            ("password", "security", 0.9),
            ("shipping", "logistics", 0.9),
        ]

    def answer(self, qid: str, q: dict[str, Any], state: Any = "") -> dict[str, Any]:
        if q["type"] != "choice":
            return super().answer(qid, q)
        opts = list(q["criteria"])
        text_ = str(state).lower()
        winner, p = ("other" if "other" in opts else opts[-1]), 0.9
        for kw, label, prob in self.rules:
            if kw in text_ and label in opts:
                winner, p = label, prob
                break
        rest = [o for o in opts if o != winner]
        probs = {winner: p}
        if "tie" in text_ and rest:  # a close second place
            probs[rest[0]] = 0.45
            rest = rest[1:]
            p_left = 1.0 - p - 0.45
        else:
            p_left = 1.0 - p
        for o in rest:
            probs[o] = max(p_left, 0.0) / max(len(rest), 1)
        return {"type": "choice", "choice": winner, "confidence": p, "probabilities": probs}

    async def decide(self, records, priority="interactive"):
        self.calls.append(records)
        self.priorities.append(priority)
        return [
            {
                "model": r["model"],
                "answers": {q: self.answer(q, spec, r["state"]) for q, spec in r["questions"].items()},
                "usage": {"input_tokens": 10, "output_tokens": 0},
                "timing": {"queue_ms": 1.0, "forward_ms": 2.0, "batch_size": len(records)},
            }
            for r in records
        ]


class FakeLLM:
    base_url = "https://llm.test/v1/"
    model = "fake-1"

    def __init__(self) -> None:
        self.responses: list[dict[str, Any]] = []
        self.calls: list[dict[str, Any]] = []
        self.fail: str | None = None

    async def complete_json(self, *, system, user, schema, name, max_tokens=2048):
        self.calls.append({"system": system, "user": user, "schema": schema, "name": name})
        if self.fail:
            raise LLMError(self.fail)
        data = self.responses.pop(0)
        validate_schema(data, schema)  # a typo in a test's canned answer fails loudly
        return LLMResult(data, "fake-1", 120, 30, 5.0)


def prop(name, members, desc="things about it", dup="", nearest=""):
    return {
        "name": name,
        "description": desc,
        "member_indices": list(members),
        "nearest_existing": nearest,
        "duplicate_of": dup,
        "rationale": "because",
    }


def answer(*proposals, covered=(), unclear=()):
    return {"proposals": list(proposals), "covered_by": list(covered), "unclear_indices": list(unclear)}


def make(llm: bool = True, **cfg_kw: Any):
    eng = ScriptEngine()
    cfg_kw.setdefault("state_dir", tempfile.mkdtemp(prefix="clef-router-"))
    cfg_kw.setdefault("router_batch", 50)  # big: a test calls /propose itself; the trigger test shrinks it
    cfg_kw.setdefault("router_cold_batch", 50)
    cfg_kw.setdefault("api_key", None)
    app = create_app(Config(**cfg_kw), eng)
    fake = FakeLLM()
    if llm:
        app.state.ctx.extra["llm_client"] = fake
    return TestClient(app, raise_server_exceptions=False), eng, app, fake


@pytest.fixture
def env():
    client, eng, app, fake = make()
    with client:
        yield client, eng, app, fake


def put_router(client, name="tickets", seeds=SEEDS, **body):
    r = client.put(f"/v1/routers/{name}", json={"seeds": seeds, **body} if seeds is not None else body)
    assert r.status_code == 200, r.text
    return r.json()


def route(client, text_, name="tickets"):
    r = client.post(f"/v1/routers/{name}/route", json={"input": text_})
    assert r.status_code == 200, r.text
    return r.json()


def buffer(client, texts, name="tickets"):
    """Route texts the router cannot place; returns the review ids."""
    out = [route(client, x, name) for x in texts]
    assert all(o["review_id"] for o in out), out
    return [o["review_id"] for o in out]


def review(client, rid):
    r = client.get(f"/v1/review/{rid}")
    assert r.status_code == 200, r.text
    return r.json()


def listing(client, **q):
    r = client.get("/v1/review", params=q)
    assert r.status_code == 200, r.text
    return r.json()


def svc_of(app):
    return app.state.ctx.extra["router_service"]


# ------------------------------------------------------------------ router resource


def test_put_router_publishes_the_seeds_as_a_classifier_version(env):
    client, _, _, _ = env
    rt = put_router(client)
    assert rt["active_version"] == 1 and rt["mode"] == "review" and rt["classifier"] == "tickets"
    assert list(rt["labels"]) == ["billing", "technical", "other"]
    assert "not billing" in rt["labels"]["other"]
    assert rt["llm"]["enabled"] and rt["llm"]["host"] == "llm.test"
    cl = client.get("/v1/classifiers/tickets").json()
    assert cl["labels"] == rt["labels"] and cl["version"] == 1
    assert (
        client.get("/v1/classifiers/tickets/versions").json()["versions"][0]["created_by"] == "router:tickets"
    )
    assert client.get("/v1/routers").json()["routers"][0]["name"] == "tickets"
    versions = client.get("/v1/routers/tickets/versions").json()["versions"]
    assert [v["version"] for v in versions] == [1] and versions[0]["source"] == "seed"


def test_put_router_adopts_an_existing_single_label_classifier(env):
    client, _, _, _ = env
    client.put(
        "/v1/classifiers/support", json={"labels": {"a": "first", "b": "second"}, "instructions": "pick"}
    )
    rt = put_router(client, "r", seeds=None, classifier="support")
    assert list(rt["labels"]) == ["a", "b", "other"] and rt["config"]["instructions"] == "pick"
    assert client.get("/v1/classifiers/support").json()["version"] == 2  # + other
    assert client.put("/v1/routers/again", json={"classifier": "support"}).status_code == 409


def test_put_router_rejections(env):
    client, _, _, _ = env
    client.put("/v1/classifiers/multi", json={"labels": ["a", "b"], "multi_label": True})
    assert client.put("/v1/routers/m", json={"classifier": "multi"}).status_code == 400
    assert client.put("/v1/routers/Bad Name", json={}).status_code == 400
    assert client.put("/v1/routers/x", json={"max_labels": 64, "seeds": SEEDS}).status_code == 400
    assert (
        client.put("/v1/routers/x", json={"gate": {"thresholds": [[10, 0.9], [20, 0.5]]}}).status_code == 400
    )
    put_router(client)
    assert client.put("/v1/routers/tickets", json={"classifier": "other-name"}).status_code == 400
    assert client.put("/v1/routers/tickets", json={"seeds": {"x": "y"}}).status_code == 400  # grow-only
    assert client.get("/v1/routers/none").status_code == 404


def test_seed_cap_is_63_labels_plus_other(env):
    client, _, _, _ = env
    r = client.put("/v1/routers/big", json={"seeds": {f"l{i}": f"label {i}" for i in range(64)}})
    assert r.status_code == 400 and "too many seeds" in r.json()["detail"]
    ok = client.put("/v1/routers/big", json={"seeds": {f"l{i}": f"label {i}" for i in range(63)}})
    assert ok.status_code == 200 and len(ok.json()["labels"]) == 64
    small = client.put("/v1/routers/small", json={"max_labels": 3, "seeds": {f"l{i}": "x" for i in range(4)}})
    assert small.status_code == 400


def test_seeds_named_other_are_dropped_and_other_is_regenerated(env):
    client, _, _, _ = env
    rt = put_router(client, seeds={"billing": "b", "Other": "mine", "technical": "t"})
    assert list(rt["labels"]) == ["billing", "technical", "other"]


def test_settings_change_without_new_label_sets(env):
    client, _, _, _ = env
    put_router(client)
    r = client.put(
        "/v1/routers/tickets",
        json={"mode": "auto", "domain_hint": "support", "gate": {"margin": 0.3, "thresholds": [[5, 0.6]]}},
    )
    body = r.json()
    assert body["mode"] == "auto" and body["config"]["domain_hint"] == "support"
    assert body["config"]["gate"] == {"thresholds": [[5, 0.6]], "margin": 0.3} and body["active_version"] == 1


# ------------------------------------------------------------------ gate and buffering


def test_classified_items_are_not_kept(env):
    client, eng, _, _ = env
    put_router(client)
    out = route(client, "my invoice is wrong")
    assert out["label"] == "billing" and out["reason"] == "classified" and out["review_id"] is None
    assert out["router"] == "tickets" and out["label_set_version"] == 1 and out["request_id"]
    assert eng.priorities[-1] == "interactive"  # routing is the request path
    assert listing(client)["total"] == 0


def test_other_low_confidence_and_ambiguous_are_kept_differently(env):
    client, _, _, _ = env
    put_router(client)
    other = route(client, "something new about warranty")
    assert other["label"] == "other" and other["reason"] == "other"
    low = route(client, "vague question")  # billing at 0.4 < tau(3) = 0.5
    assert low["reason"] == "low_confidence"
    tie = route(client, "tie breaker")  # 0.5 vs 0.45: confident enough but too close
    assert tie["reason"] == "ambiguous"
    rows = {r["id"]: r for r in listing(client)["items"]}
    assert rows[other["review_id"]]["state"] == "buffered" and rows[other["review_id"]]["input"].startswith(
        "something"
    )
    assert rows[low["review_id"]]["state"] == "buffered"
    assert rows[tie["review_id"]]["state"] == "open"  # a human looks, the LLM never sees it
    assert rows[other["review_id"]]["prediction"]["label"] == "other"
    assert client.get("/v1/routers/tickets").json()["buffered"] == 2


def test_the_threshold_rises_with_the_label_count(env):
    client, _, _, _ = env
    put_router(client, seeds={f"l{i}": f"label {i}" for i in range(11)} | {"billing": "b"}, name="wide")
    # 13 options (12 + other): tau = 0.7, so a 0.4 winner and the 0.9 winner differ; custom tau also works
    assert route(client, "vague question", "wide")["reason"] == "low_confidence"
    assert route(client, "invoice", "wide")["reason"] == "classified"
    put_router(client, seeds=SEEDS, name="strict", gate={"thresholds": [[64, 0.95]]})
    assert route(client, "my invoice", "strict")["reason"] == "low_confidence"  # 0.9 < 0.95


def test_a_repeated_input_is_buffered_once(env):
    client, _, _, _ = env
    put_router(client)
    a = route(client, "warranty claim")
    b = route(client, "warranty claim")
    assert a["review_id"] == b["review_id"] and listing(client)["total"] == 1


def test_route_batch_gates_each_item(env):
    client, _, _, _ = env
    put_router(client)
    r = client.post("/v1/routers/tickets/route/batch", json={"inputs": ["invoice", "warranty", "crash"]})
    body = r.json()
    assert [x["reason"] for x in body["results"]] == ["classified", "other", "classified"] and body[
        "request_id"
    ]
    assert body["results"][1]["review_id"] and body["results"][0]["review_id"] is None


# ------------------------------------------------------------------ propose, re-check, review mode


TEXTS = ["password reset fails", "forgot my password", "password leaked?", "warranty claim"]


def buffered_security(client):
    put_router(client)
    return buffer(client, TEXTS)


def test_propose_review_mode_queues_a_confirmed_proposal(env):
    client, _, app, fake = env
    ids = buffered_security(client)
    fake.responses.append(answer(prop("security", [0, 1, 2], "account access and passwords")))
    r = client.post("/v1/routers/tickets/propose")
    assert r.status_code == 202, r.text
    assert r.json() | {"llm_call_id": 0} == {
        "router": "tickets",
        "items": 4,
        "llm_call_id": 0,
        "proposals": 1,
        "queued": 1,
        "activated": 0,
    }
    call = fake.calls[0]
    assert (
        call["name"] == "propose"
        and "- billing: " in call["user"]
        and "[0] password reset fails" in call["user"]
    )
    assert "not billing" in call["user"]  # it is shown the current labels, other included
    [p] = listing(client, kind="proposal", state="open")["items"]
    assert p["reason"] == "proposal" and p["router"] == "tickets"
    lab = p["proposal"]["labels"][0]
    assert lab["name"] == "security" and lab["member_ids"] == ids[:3]
    assert lab["acceptance"]["passed"] and lab["acceptance"]["on_new"] == 3
    assert p["proposal"]["stats"]["items"] == 4
    assert [review(client, i)["state"] for i in ids] == ["open", "open", "open", "buffered"]
    # nothing is published until a human approves
    assert client.get("/v1/classifiers/tickets").json()["version"] == 1
    assert client.get("/v1/routers/tickets").json()["open_reviews"] == 4


def test_the_recheck_rejects_a_proposal_clef_does_not_confirm(env):
    client, _, _, fake = env
    ids = buffered_security(client)
    fake.responses.append(answer(prop("gibberish", [0, 1, 3], "words that go nowhere")))  # engine has no rule
    out = client.post("/v1/routers/tickets/propose").json()
    assert out["proposals"] == 1 and out["queued"] == 0
    assert listing(client, state="open", kind="proposal")["total"] == 0
    [failed] = listing(client, state="failed")["items"]
    assert failed["proposal"]["labels"][0]["acceptance"]["passed"] is False
    assert failed["proposal"]["labels"][0]["acceptance"]["on_new"] == 0
    rows = [review(client, i) for i in ids]
    assert [r["reroutes"] for r in rows] == [1, 1, 1, 1] and rows[0]["state"] == "buffered"  # all checked
    assert client.get("/v1/classifiers/tickets").json()["version"] == 1


def test_a_duplicate_of_an_existing_label_is_rejected_by_the_recheck(env):
    client, _, _, fake = env
    put_router(client)
    buffer(client, ["vague a", "vague b", "vague c", "vague d"])
    # "payments" would just be billing: the members land on billing, not on the new label
    fake.responses.append(answer(prop("payments", [0, 1], "money paid to us", nearest="billing")))
    client.post("/v1/routers/tickets/propose")
    [failed] = listing(client, state="failed")["items"]
    acc = failed["proposal"]["labels"][0]["acceptance"]
    assert acc["duplicate_of"] == "billing" and not acc["passed"]


def test_invalid_llm_proposals_are_dropped_before_any_check(env):
    client, eng, _, fake = env
    buffered_security(client)
    fake.responses.append(
        answer(prop("billing", [0, 1]), prop("OTHER", [0, 1]), prop("x" * 70, [0, 1]), prop("solo", [0]))
    )
    out = client.post("/v1/routers/tickets/propose").json()
    assert out["proposals"] == 0
    assert listing(client, kind="proposal")["total"] == 0
    assert all(review(client, i["id"])["state"] == "buffered" for i in listing(client)["items"])
    n_before = len(eng.calls)
    assert n_before == 4  # only the four routed items ran: no re-check for dropped proposals


def test_items_the_llm_covers_or_cannot_place_wait_for_a_human(env):
    client, _, _, fake = env
    put_router(client)
    ids = buffer(client, ["warranty one", "warranty two", "warranty three", "warranty four"])
    fake.responses.append(answer(covered=[{"index": 0, "label": "billing"}], unclear=[1]))
    client.post("/v1/routers/tickets/propose")
    a, b, c, d = (review(client, i) for i in ids)
    assert a["state"] == "open" and a["prediction"]["llm_hint"] == "billing"
    assert b["state"] == "open" and b["prediction"]["llm_hint"] == "unclear"
    assert c["state"] == "buffered" and c["reroutes"] == 1 and d["reroutes"] == 1


def test_items_that_never_get_placed_become_permanent_other(env):
    client, _, _, fake = env
    put_router(client)
    ids = buffer(client, ["warranty one", "warranty two", "warranty three", "warranty four"])
    for _ in range(2):
        fake.responses.append(answer())
        client.post("/v1/routers/tickets/propose")
    assert {review(client, i)["state"] for i in ids} == {"permanent_other"}
    assert client.post("/v1/routers/tickets/propose").status_code == 400  # nothing buffered any more


def test_approve_publishes_a_new_classifier_version_and_reroutes_the_buffer(env):
    client, _, _, fake = env
    ids = buffered_security(client)
    fake.responses.append(answer(prop("security", [0, 1, 2], "account access and passwords")))
    client.post("/v1/routers/tickets/propose")
    [p] = listing(client, kind="proposal")["items"]
    r = client.post(f"/v1/review/{p['id']}/resolve", json={"action": "approve"})
    assert r.status_code == 200, r.text
    assert r.json()["state"] == "accepted" and r.json()["label"] == ["security"] and r.json()["resolved_by"]
    rt = client.get("/v1/routers/tickets").json()
    assert rt["active_version"] == 2 and list(rt["labels"]) == ["billing", "technical", "security", "other"]
    cl = client.get("/v1/classifiers/tickets").json()
    assert cl["version"] == 2 and "security" in cl["labels"]
    assert (
        client.get("/v1/classifiers/tickets/versions").json()["versions"][0]["created_by"] == "router:tickets"
    )
    # the three members now classify under the new set: resolved by the router, text purged (D17)
    for i in ids[:3]:
        row = review(client, i)
        assert row["state"] == "resolved" and row["label"] == "security" and row["input"] is None
        assert row["resolved_by"] == "router:tickets"
    last = review(client, ids[3])  # "warranty claim" still has no home: checked twice now (propose, re-route)
    assert last["state"] == "permanent_other" and last["reroutes"] == 2
    assert route(client, "password expired")["reason"] == "classified"
    assert client.post(f"/v1/review/{p['id']}/resolve", json={"action": "approve"}).status_code == 409


def test_approve_an_edited_subset_and_reject_the_rest(env):
    client, _, _, fake = env
    put_router(client)
    buffer(client, ["password a", "password b", "shipping a", "shipping b"])
    fake.responses.append(answer(prop("security", [0, 1]), prop("logistics", [2, 3], "parcels and delivery")))
    assert client.post("/v1/routers/tickets/propose").json()["queued"] == 2
    props = {p["proposal"]["labels"][0]["name"]: p for p in listing(client, kind="proposal")["items"]}
    edit = {"action": "approve", "labels": [{"name": "delivery", "description": "parcels and delivery"}]}
    assert client.post(f"/v1/review/{props['logistics']['id']}/resolve", json=edit).status_code == 200
    labels = client.get("/v1/routers/tickets").json()["labels"]
    assert "delivery" in labels and "logistics" not in labels
    rej = client.post(f"/v1/review/{props['security']['id']}/resolve", json={"action": "reject"})
    assert rej.json()["state"] == "rejected"
    members = [review(client, m)["state"] for m in props["security"]["proposal"]["labels"][0]["member_ids"]]
    assert members == ["buffered", "buffered"]  # back in the buffer, one check closer to permanent_other


def test_labels_are_grow_only_and_capped(env):
    client, _, _, fake = env
    put_router(client, seeds={"billing": "b", "technical": "t"}, max_labels=3)
    buffer(client, ["password a", "password b", "shipping a", "shipping b"])
    fake.responses.append(answer(prop("security", [0, 1]), prop("logistics", [2, 3], "parcels")))
    assert client.post("/v1/routers/tickets/propose").json()["queued"] == 1  # room for one more label only
    [p] = listing(client, kind="proposal")["items"]
    client.post(f"/v1/review/{p['id']}/resolve", json={"action": "approve"})
    assert len(client.get("/v1/routers/tickets").json()["labels"]) == 4  # 3 + other
    buffer(client, ["shipping c", "shipping d"])
    r = client.post("/v1/routers/tickets/propose")
    assert r.status_code == 400 and "cap" in r.json()["detail"]
    assert len(fake.calls) == 1  # no LLM call at the cap


def test_approving_a_label_that_already_exists_is_refused(env):
    client, _, _, fake = env
    ids = buffered_security(client)
    fake.responses.append(answer(prop("security", [0, 1, 2])))
    client.post("/v1/routers/tickets/propose")
    [p] = listing(client, kind="proposal")["items"]
    bad = {"action": "approve", "labels": [{"name": "billings", "description": "dup"}]}
    r = client.post(f"/v1/review/{p['id']}/resolve", json=bad)
    assert r.status_code == 409 and "duplicates" in r.json()["detail"]
    assert review(client, p["id"])["state"] == "open"  # still open: nothing was published
    assert client.get("/v1/routers/tickets").json()["active_version"] == 1 and ids


def test_rollback_publishes_an_earlier_set_as_a_new_version(env):
    client, _, _, fake = env
    buffered_security(client)
    fake.responses.append(answer(prop("security", [0, 1, 2])))
    client.post("/v1/routers/tickets/propose")
    [p] = listing(client, kind="proposal")["items"]
    client.post(f"/v1/review/{p['id']}/resolve", json={"action": "approve"})
    r = client.post("/v1/routers/tickets/versions/1/activate")
    assert r.status_code == 200 and r.json()["active_version"] == 3
    assert list(r.json()["labels"]) == ["billing", "technical", "other"]
    versions = client.get("/v1/routers/tickets/versions").json()["versions"]
    assert [v["source"] for v in versions] == ["rollback", "proposal", "seed"]
    assert client.get("/v1/classifiers/tickets").json()["version"] == 3
    assert client.post("/v1/routers/tickets/versions/9/activate").status_code == 404


# ------------------------------------------------------------------ auto mode


def test_auto_mode_publishes_a_confirmed_proposal_at_once(env):
    client, _, _, fake = env
    put_router(client, mode="auto")
    ids = buffer(client, TEXTS)
    fake.responses.append(answer(prop("security", [0, 1, 2], "account access and passwords")))
    out = client.post("/v1/routers/tickets/propose").json()
    assert out["activated"] == 1 and out["queued"] == 0
    assert client.get("/v1/routers/tickets").json()["active_version"] == 2
    [p] = listing(client, kind="proposal")["items"]
    assert p["state"] == "accepted" and p["resolved_by"] == "auto" and p["label"] == ["security"]
    assert [review(client, i)["state"] for i in ids[:3]] == ["resolved"] * 3
    assert client.get("/v1/routers/tickets").json()["open_reviews"] == 0


def test_auto_mode_failed_recheck_publishes_nothing(env):
    client, _, _, fake = env
    put_router(client, mode="auto")
    buffer(client, TEXTS)
    fake.responses.append(answer(prop("gibberish", [0, 1, 2])))
    assert client.post("/v1/routers/tickets/propose").json()["activated"] == 0
    assert client.get("/v1/routers/tickets").json()["active_version"] == 1


def test_auto_mode_stops_at_the_daily_limit_and_queues_the_rest():
    client, _, _, fake = make(router_max_new_per_day=1)
    with client:
        put_router(client, mode="auto")
        buffer(client, ["password a", "password b", "shipping a", "shipping b"])
        fake.responses.append(answer(prop("security", [0, 1]), prop("logistics", [2, 3], "parcels")))
        out = client.post("/v1/routers/tickets/propose").json()
        assert out["activated"] == 1 and out["queued"] == 1
        [waiting] = listing(client, kind="proposal", state="open")["items"]
        assert waiting["proposal"]["stats"]["waiting"].startswith("daily limit")
        assert client.get("/v1/routers/tickets").json()["active_version"] == 2


# ------------------------------------------------------------------ cold start


COLD = ["password a", "password b", "shipping a", "shipping b"]


def test_a_cold_router_answers_other_and_buffers(env):
    client, eng, _, _ = env
    rt = put_router(client, seeds=None)
    assert rt["active_version"] is None and rt["labels"] == {}
    out = route(client, "password a")
    assert out["label"] == "other" and out["reason"] == "cold" and out["label_set_version"] is None
    assert eng.calls == []  # no forward without a label set
    assert client.get("/v1/classifiers/tickets").status_code == 404


def test_cold_start_proposes_a_starter_set_in_one_bigger_call(env):
    client, _, _, fake = env
    put_router(client, seeds=None)
    ids = buffer(client, COLD)
    fake.responses.append(
        answer(prop("security", [0, 1], "passwords and access"), prop("logistics", [2, 3], "parcels"))
    )
    out = client.post("/v1/routers/tickets/propose").json()
    assert (
        out["queued"] == 1
        and fake.calls[0]["name"] == "bootstrap"
        and "no labels yet" in fake.calls[0]["user"]
    )
    [p] = listing(client, kind="proposal")["items"]
    assert p["reason"] == "bootstrap" and {x["name"] for x in p["proposal"]["labels"]} == {
        "security",
        "logistics",
    }
    client.post(f"/v1/review/{p['id']}/resolve", json={"action": "approve"})
    rt = client.get("/v1/routers/tickets").json()
    assert rt["active_version"] == 1 and list(rt["labels"]) == ["security", "logistics", "other"]
    assert client.get("/v1/routers/tickets/versions").json()["versions"][0]["source"] == "bootstrap"
    assert {review(client, i)["state"] for i in ids} == {"resolved"}
    assert route(client, "shipping c")["label"] == "logistics"


def test_cold_start_starter_labels_are_pruned_to_what_clef_confirms(env):
    client, _, _, fake = env
    put_router(client, seeds=None)
    buffer(client, COLD)
    fake.responses.append(answer(prop("security", [0, 1]), prop("nonsense", [2, 3], "words")))
    client.post("/v1/routers/tickets/propose")
    [p] = listing(client, kind="proposal", state="open")["items"]
    assert [x["name"] for x in p["proposal"]["labels"]] == ["security"]  # nonsense: nothing lands on it


def test_cold_start_that_clef_cannot_confirm_fails(env):
    client, _, _, fake = env
    put_router(client, seeds=None)
    ids = buffer(client, COLD)
    fake.responses.append(answer(prop("nonsense", [0, 1]), prop("junk", [2, 3], "words")))
    client.post("/v1/routers/tickets/propose")
    assert listing(client, kind="proposal", state="open")["total"] == 0
    assert listing(client, state="failed")["total"] == 1
    assert all(review(client, i)["reroutes"] == 1 for i in ids)


def test_cold_start_in_auto_mode_publishes_the_starter_set(env):
    client, _, _, fake = env
    put_router(client, seeds=None, mode="auto")
    buffer(client, COLD)
    fake.responses.append(answer(prop("security", [0, 1]), prop("logistics", [2, 3], "parcels")))
    assert client.post("/v1/routers/tickets/propose").json()["activated"] == 1
    assert client.get("/v1/routers/tickets").json()["active_version"] == 1
    assert client.get("/v1/classifiers/tickets").json()["labels"].keys() >= {"security", "logistics", "other"}


def test_the_batch_size_starts_the_call_without_a_request():
    client, _, app, fake = make(router_cold_batch=4)
    with client:
        put_router(client, seeds=None)
        fake.responses.append(answer(prop("security", [0, 1]), prop("logistics", [2, 3], "parcels")))
        buffer(client, COLD)  # the 4th item reaches CLEF_ROUTER_COLD_BATCH
        for _ in range(100):
            if listing(client, kind="proposal")["total"]:
                break
            time.sleep(0.05)
        assert listing(client, kind="proposal", state="open")["total"] == 1 and len(fake.calls) == 1


# ------------------------------------------------------------------ LLM off / failing


def test_without_a_key_the_router_still_routes_but_never_calls_out():
    client, _, app, _ = make(llm=False)
    with client:
        assert app.state.ctx.extra["llm_client"] is None
        rt = put_router(client)
        assert rt["llm"]["enabled"] is False
        buffer(client, TEXTS)
        r = client.post("/v1/routers/tickets/propose")
        assert r.status_code == 503 and "GEMINI_API_KEY" in r.json()["detail"]


def test_a_failing_llm_is_logged_and_leaves_the_buffer_alone(env):
    client, _, app, fake = env
    ids = buffered_security(client)
    fake.fail = "HTTP 500"
    r = client.post("/v1/routers/tickets/propose")
    assert r.status_code == 502 and "HTTP 500" in r.json()["detail"]
    assert [review(client, i)["state"] for i in ids] == ["buffered"] * 4
    rt = client.get("/v1/routers/tickets").json()
    assert rt["llm"]["calls_today"] == 1 and rt["llm"]["errors_today"] == 1
    assert "tickets" in svc_of(app)._backoff  # the background loop waits before trying again
    fake.fail = None
    fake.responses.append(answer(prop("security", [0, 1, 2])))
    assert client.post("/v1/routers/tickets/propose").status_code == 202  # a manual propose still works
    use = client.get("/v1/routers/tickets").json()["llm"]
    assert (use["calls_today"], use["input_tokens_today"], use["output_tokens_today"]) == (2, 120, 30)


def test_one_call_in_flight_per_router(env):
    client, _, app, _ = env
    buffered_security(client)
    lock = asyncio.Lock()
    lock._locked = True  # as if a call were running
    svc_of(app)._locks["tickets"] = lock
    assert client.post("/v1/routers/tickets/propose").status_code == 409


def test_propose_needs_two_buffered_items(env):
    client, _, _, _ = env
    put_router(client)
    assert client.post("/v1/routers/tickets/propose").status_code == 400


# ------------------------------------------------------------------ human labels


def test_a_human_label_is_rechecked_before_the_text_is_purged(env):
    client, _, _, _ = env
    put_router(client)
    [rid] = buffer(client, ["warranty claim"])
    bad = client.post(f"/v1/review/{rid}/resolve", json={"action": "label", "label": "nonsense"})
    assert bad.status_code == 400 and "technical" in bad.json()["detail"]
    assert review(client, rid)["input"] == "warranty claim"
    ok = client.post(f"/v1/review/{rid}/resolve", json={"action": "label", "label": "technical"})
    row = ok.json()
    assert ok.status_code == 200 and row["state"] == "resolved" and row["label"] == "technical"
    assert row["input"] is None and row["input_sha256"]
    recheck = row["prediction"][
        "recheck"
    ]  # clef said "other" for it: the human disagrees, and it is recorded
    assert recheck["label"] == "other" and recheck["agrees"] is False and recheck["label_set_version"] == 1
    assert (
        client.post(f"/v1/review/{rid}/resolve", json={"action": "label", "label": "billing"}).status_code
        == 409
    )


def test_a_human_label_that_clef_agrees_with_is_recorded_as_agreement(env):
    client, eng, _, _ = env
    put_router(client)
    [rid] = buffer(client, ["vague invoice"])  # billing at 0.4: low confidence
    row = client.post(f"/v1/review/{rid}/resolve", json={"action": "label", "label": "billing"}).json()
    assert row["prediction"]["recheck"]["agrees"] is True and eng.priorities[-1] == "background"


def test_discard_drops_the_text(env):
    client, _, _, _ = env
    put_router(client)
    [rid] = buffer(client, ["warranty claim"])
    row = client.post(f"/v1/review/{rid}/resolve", json={"action": "discard"}).json()
    assert row["state"] == "discarded" and row["input"] is None
    assert client.post(f"/v1/review/{rid}/resolve", json={"action": "approve"}).status_code == 400


def test_resolve_actions_must_fit_the_kind(env):
    client, _, _, fake = env
    ids = buffered_security(client)
    assert client.post(f"/v1/review/{ids[0]}/resolve", json={"action": "reject"}).status_code == 400
    assert client.post(f"/v1/review/{ids[0]}/resolve", json={"action": "label"}).status_code == 400
    assert client.post("/v1/review/rev_nope/resolve", json={"action": "discard"}).status_code == 404
    assert client.post(f"/v1/review/{ids[0]}/resolve", json={"action": "explode"}).status_code == 400


def test_human_labels_count_against_a_proposal_in_the_recheck(env):
    client, _, _, fake = env
    put_router(client)
    [first] = buffer(client, ["password reset"])
    client.post(f"/v1/review/{first}/resolve", json={"action": "label", "label": "technical"})
    # the same text arrives again, and the LLM groups it with one other item under a new label
    ids = buffer(client, ["password reset", "password lost", "shipping a", "shipping b"])
    fake.responses.append(answer(prop("security", [0, 1], "passwords")))
    client.post("/v1/routers/tickets/propose")
    [failed] = listing(client, state="failed")["items"]
    acc = failed["proposal"]["labels"][0]["acceptance"]
    assert acc["human_against"] == 1 and acc["on_new"] == 1 and acc["need"] == 2 and not acc["passed"]
    assert ids


# ------------------------------------------------------------------ purge and TTL


def test_the_ttl_purges_text_and_expires_open_items(env):
    client, _, app, _ = env
    put_router(client)
    [a, b] = buffer(client, ["warranty claim", "vague invoice"])
    client.post(f"/v1/review/{b}/resolve", json={"action": "label", "label": "billing"})
    [c] = buffer(client, ["another warranty"])
    svc = svc_of(app)
    svc.now = lambda: time.time() + 31 * DAY  # 31 days later
    asyncio.run(svc.tick())
    row_a, row_b = review(client, a), review(client, b)
    assert row_a["state"] == "expired" and row_a["input"] is None
    assert row_b["state"] == "resolved" and row_b["label"] == "billing" and row_b["input_sha256"]
    assert review(client, c)["state"] == "expired"
    assert client.get("/v1/routers/tickets").json()["buffered"] == 0


def test_the_ttl_is_configurable():
    client, _, app, _ = make(review_ttl_days=2)
    with client:
        put_router(client)
        [a] = buffer(client, ["warranty claim"])
        svc = svc_of(app)
        svc.now = lambda: time.time() + 1 * DAY
        asyncio.run(svc.tick())
        assert review(client, a)["input"] == "warranty claim"
        svc._last_purge = 0.0
        svc.now = lambda: time.time() + 3 * DAY
        asyncio.run(svc.tick())
        assert review(client, a)["state"] == "expired"


def test_deleting_a_router_drops_its_texts_but_not_its_classifier(env):
    client, _, _, _ = env
    put_router(client)
    [rid] = buffer(client, ["warranty claim"])
    assert client.delete("/v1/routers/tickets").json() == {"deleted": "tickets"}
    row = review(client, rid)
    assert row["state"] == "discarded" and row["input"] is None
    assert client.get("/v1/classifiers/tickets").status_code == 200
    assert client.get("/v1/routers/tickets").status_code == 404


# ------------------------------------------------------------------ low-confidence items of classifiers


def test_review_threshold_queues_low_confidence_classifier_results(env):
    client, _, _, _ = env
    r = client.put("/v1/classifiers/mail", json={"labels": ["billing", "technical"], "review_threshold": 0.8})
    assert r.status_code == 200 and r.json()["review_threshold"] == 0.8
    sure = client.post("/v1/classifiers/mail", json={"input": "invoice 1"}).json()
    assert sure["confidence"] >= 0.9 and listing(client)["total"] == 0
    unsure = client.post("/v1/classifiers/mail", json={"input": "vague invoice"}).json()
    assert unsure["confidence"] < 0.8
    [item] = listing(client)["items"]
    assert item["kind"] == "item" and item["state"] == "open" and item["reason"] == "low_confidence"
    assert item["router"] is None and item["classifier"] == "mail" and item["classifier_version"] == 1
    assert item["input"] == "vague invoice" and item["prediction"]["label"] == unsure["label"]
    batch = client.post("/v1/classifiers/mail/batch", json={"inputs": ["invoice 2", "vague invoice two"]})
    assert batch.status_code == 200 and listing(client)["total"] == 2
    assert listing(client, classifier="mail", state="open")["total"] == 2
    row = client.post(
        f"/v1/review/{item['id']}/resolve", json={"action": "label", "label": "technical"}
    ).json()
    assert row["state"] == "resolved" and row["input"] is None and "recheck" in row["prediction"]


def test_review_threshold_is_off_by_default_and_checked(env):
    client, _, _, _ = env
    client.put("/v1/classifiers/plain", json={"labels": ["a", "b"]})
    client.post("/v1/classifiers/plain", json={"input": "vague"})
    assert listing(client)["total"] == 0
    assert client.get("/v1/classifiers/plain").json().get("review_threshold") is None
    multi = client.put(
        "/v1/classifiers/m", json={"labels": ["a", "b"], "multi_label": True, "review_threshold": 0.5}
    )
    assert multi.status_code == 400
    assert (
        client.put(
            "/v1/classifiers/s", json={"kind": "score", "levels": ["a", "b"], "review_threshold": 0.5}
        ).status_code
        == 400
    )
    assert (
        client.put("/v1/classifiers/plain", json={"labels": ["a", "b"]}).json()["version"] == 1
    )  # idempotent


# ------------------------------------------------------------------ scopes and owners


def test_router_and_review_routes_need_the_classifiers_scope():
    client, _, app, _ = make(api_key="rootkey")
    with client:
        root = {"X-API-Key": "rootkey"}
        classify_only = hdr(add_key(app, "reader", "classify"))
        manager = hdr(add_key(app, "mgr", "classify,classifiers"))
        assert client.put("/v1/routers/t", json={"seeds": SEEDS}, headers=classify_only).status_code == 403
        assert client.put("/v1/routers/t", json={"seeds": SEEDS}, headers=manager).status_code == 200
        for method, path in (
            ("get", "/v1/routers"),
            ("get", "/v1/routers/t"),
            ("delete", "/v1/routers/t"),
            ("post", "/v1/routers/t/propose"),
            ("get", "/v1/routers/t/versions"),
            ("post", "/v1/routers/t/versions/1/activate"),
            ("get", "/v1/review"),
            ("get", "/v1/review/rev_x"),
            ("post", "/v1/review/rev_x/resolve"),
        ):
            assert getattr(client, method)(path, headers=classify_only).status_code == 403, path
            assert getattr(client, method)(path).status_code == 401, path
        jobs_only = hdr(add_key(app, "jobber", "jobs"))
        assert client.post("/v1/routers/t/route", json={"input": "x"}, headers=jobs_only).status_code == 403
        assert (
            client.post("/v1/routers/t/route", json={"input": "invoice"}, headers=manager).status_code == 200
        )
        assert client.get("/v1/routers", headers=root).status_code == 200


def test_routers_and_review_rows_are_private_to_their_owner():
    client, _, app, _ = make(api_key="rootkey")
    with client:
        alice = hdr(add_key(app, "alice", "classify,classifiers"))
        bob = hdr(add_key(app, "bob", "classify,classifiers"))
        root = {"X-API-Key": "rootkey"}
        assert client.put("/v1/routers/t", json={"seeds": SEEDS}, headers=alice).json()["owner"] == "alice"
        rid = client.post("/v1/routers/t/route", json={"input": "warranty"}, headers=alice).json()[
            "review_id"
        ]
        assert client.get("/v1/routers/t", headers=bob).status_code == 404
        assert client.post("/v1/routers/t/route", json={"input": "x"}, headers=bob).status_code == 404
        assert client.get("/v1/routers", headers=bob).json()["routers"] == []
        assert client.put("/v1/routers/t", json={"mode": "auto"}, headers=bob).status_code == 403
        assert client.delete("/v1/routers/t", headers=bob).status_code == 404
        assert client.get(f"/v1/review/{rid}", headers=bob).status_code == 404
        assert client.get("/v1/review", headers=bob).json()["total"] == 0
        assert (
            client.post(f"/v1/review/{rid}/resolve", json={"action": "discard"}, headers=bob).status_code
            == 404
        )
        assert client.get("/v1/review", headers=alice).json()["total"] == 1
        assert client.get("/v1/review", headers=root).json()["total"] == 1  # admins see everything
        # bob may not publish into alice's classifier either
        assert (
            client.put("/v1/routers/t2", json={"classifier": "t", "seeds": SEEDS}, headers=bob).status_code
            == 409
        )
        client.put("/v1/classifiers/mine", json={"labels": ["a", "b"]}, headers=alice)
        assert client.put("/v1/routers/t3", json={"classifier": "mine"}, headers=bob).status_code == 403
