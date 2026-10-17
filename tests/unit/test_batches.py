"""OpenAI Files and Batches (F1): upload, validation, lifecycle, output / error files, cancel, expiry,
ownership, scopes, quota, and the real ``openai`` SDK over a forwarding transport (FakeEngine, no GPU)."""

from __future__ import annotations

import json
import os
import tempfile
import time
from pathlib import Path
from typing import Any

import pytest
from sqlalchemy import text
from test_api import FakeEngine, make
from test_compat import chat
from test_jobs import SlowEngine
from test_keys import _scope_marks, add_key, api_routes, hdr

from clef_server import batches
from clef_server.config import Config
from clef_server.engine import Engine
from clef_server.main import create_app
from clef_server.models.registry import catalog

pytestmark = pytest.mark.postgres  # also run on Postgres when CLEF_TEST_DATABASE_URL is set (conftest.py)

CHAT = "/v1/chat/completions"
CLASSIFY = "/v1/classify"
SCORE = "/v1/score"
DONE = ("completed", "failed", "cancelled", "expired")


class Eng(SlowEngine):
    """SlowEngine (per-call delay, call sizes) that serves clef-flash only: unknown model ids are 404."""

    spec = catalog()["clef-flash"]
    check_model = Engine.check_model


def line(cid: str, url: str = CHAT, body: dict[str, Any] | None = None, **more: Any) -> dict[str, Any]:
    return {"custom_id": cid, "method": "POST", "url": url, "body": chat() if body is None else body, **more}


def jsonl(*items: Any) -> bytes:
    return b"".join((i if isinstance(i, bytes) else json.dumps(i).encode()) + b"\n" for i in items)


def upload(
    client: Any, data: bytes, name: str = "in.jsonl", headers: Any = None, purpose: str = "batch", **form
):
    return client.post(
        "/v1/files", files={"file": (name, data)}, data={"purpose": purpose, **form}, headers=headers
    )


def new_batch(client: Any, fid: str, endpoint: str = CHAT, headers: Any = None, **more: Any):
    body = {"input_file_id": fid, "endpoint": endpoint, "completion_window": "24h", **more}
    return client.post("/v1/batches", json=body, headers=headers)


def run_batch(client: Any, items: list[Any], endpoint: str = CHAT, headers: Any = None, **more: Any):
    fid = upload(client, jsonl(*items), headers=headers).json()["id"]
    r = new_batch(client, fid, endpoint, headers, **more)
    assert r.status_code == 200, r.text
    return r.json()


def wait_batch(
    client: Any, bid: str, headers: Any = None, until: tuple[str, ...] = DONE, timeout: float = 20
):
    t0 = time.monotonic()
    while True:
        b = client.get(f"/v1/batches/{bid}", headers=headers).json()
        if b["status"] in until:
            return b
        assert time.monotonic() - t0 < timeout, f"batch stuck: {b}"
        time.sleep(0.02)


def rows(client: Any, file_id: str | None, headers: Any = None) -> list[dict[str, Any]]:
    assert file_id, "no file id"
    r = client.get(f"/v1/files/{file_id}/content", headers=headers)
    assert r.status_code == 200, r.text
    return [json.loads(x) for x in r.text.splitlines()]


def sql(app: Any, stmt: str, **params: Any) -> list[Any]:
    with app.state.ctx.db.tx() as conn:
        res = conn.execute(text(stmt), params)
        return res.all() if res.returns_rows else []


@pytest.fixture
def api():
    client, eng, app = make(Eng(), max_body_mb=4, max_file_mb=8)
    with client:
        yield client, eng, app


# ------------------------------------------------------------------ files


def test_file_upload_get_content_list_delete(api):
    client, _, app = api
    data = jsonl(line("a"), line("b"))
    r = upload(client, data, name="my batch.jsonl")
    assert r.status_code == 200, r.text
    f = r.json()
    assert f["id"].startswith("file-") and len(f["id"]) == 29
    assert f | {"created_at": 0, "expires_at": 0, "id": ""} == {
        "id": "",
        "object": "file",
        "bytes": len(data),
        "created_at": 0,
        "expires_at": 0,
        "filename": "my batch.jsonl",
        "purpose": "batch",
        "status": "processed",
        "status_details": None,
    }
    assert f["expires_at"] - f["created_at"] == 720 * 3600  # CLEF_FILE_TTL_HOURS
    assert client.get(f"/v1/files/{f['id']}").json() == f
    c = client.get(f"/v1/files/{f['id']}/content")
    assert c.content == data and c.headers["content-type"].startswith("application/jsonl")
    page = client.get("/v1/files").json()
    assert page == {"object": "list", "data": [f], "first_id": f["id"], "last_id": f["id"], "has_more": False}
    path = Path(app.state.ctx.cfg.state_dir) / "files" / f"{f['id']}.jsonl"
    assert path.read_bytes() == data
    d = client.delete(f"/v1/files/{f['id']}")
    assert d.json() == {"id": f["id"], "object": "file", "deleted": True}
    assert client.get(f"/v1/files/{f['id']}").status_code == 404 and not path.exists()
    r = client.get(f"/v1/files/{f['id']}/content")
    assert r.status_code == 404 and r.json()["error"]["type"] == "invalid_request_error"


def test_file_list_paging_order_and_purpose(api):
    client, _, _ = api
    ids = [upload(client, jsonl(line(f"c{i}"))).json()["id"] for i in range(5)]
    got = []
    after = None
    while True:
        r = client.get("/v1/files", params={"limit": 2, **({"after": after} if after else {})}).json()
        got += [f["id"] for f in r["data"]]
        assert r["first_id"] == r["data"][0]["id"] and r["last_id"] == r["data"][-1]["id"]
        if not r["has_more"]:
            break
        after = r["last_id"]
    assert got == ids[::-1]  # newest first
    asc = client.get("/v1/files", params={"order": "asc"}).json()
    assert [f["id"] for f in asc["data"]] == ids
    assert client.get("/v1/files", params={"purpose": "batch_output"}).json()["data"] == []
    bad = client.get("/v1/files", params={"after": "file-" + "0" * 24})
    assert bad.status_code == 400 and bad.json()["error"]["param"] == "after"


def test_upload_validation_errors_are_openai_shaped(api):
    client, _, _ = api
    ok = jsonl(line("a"))

    def bad(r: Any, param: str | None, text_: str) -> None:
        assert r.status_code == 400, r.text
        err = r.json()["error"]
        assert err["type"] == "invalid_request_error" and err["param"] == param and text_ in err["message"]

    bad(upload(client, ok, purpose="assistants"), "purpose", "batch")
    bad(upload(client, ok, name="data.csv"), "file", ".jsonl")
    bad(upload(client, b""), "file", "empty")
    bad(upload(client, b"\n  \n"), "file", "empty")
    bad(client.post("/v1/files", data={"purpose": "batch"}), None, "file")  # no file part
    bad(upload(client, ok, **{"expires_after[anchor]": "created_at"}), "expires_after", "both")
    bad(
        upload(client, ok, **{"expires_after[anchor]": "x", "expires_after[seconds]": "3600"}),
        "expires_after.anchor",
        "created_at",
    )
    bad(
        upload(client, ok, **{"expires_after[anchor]": "created_at", "expires_after[seconds]": "10"}),
        "expires_after.seconds",
        "between",
    )
    assert client.get("/v1/files").json()["data"] == []  # nothing was kept
    assert os.listdir(Path(client.app.state.ctx.cfg.state_dir) / "files") == []
    r = upload(client, ok, **{"expires_after[anchor]": "created_at", "expires_after[seconds]": "7200"}).json()
    assert r["expires_at"] - r["created_at"] == 7200


def test_upload_limit_is_the_file_limit_not_the_body_limit(api):
    client, _, _ = api  # max_body_mb=4, max_file_mb=8
    big = jsonl({"custom_id": "x", "pad": "y" * (5 * 1024 * 1024)})
    r = upload(client, big)
    assert r.status_code == 200, r.text  # 5 MB: over the body limit, under the file limit
    r = client.post(CHAT, content=b"x" * (5 * 1024 * 1024), headers={"content-type": "application/json"})
    assert r.status_code == 413  # every other route keeps CLEF_MAX_BODY_MB
    huge = jsonl({"custom_id": "x", "pad": "y" * (9 * 1024 * 1024)})
    r = upload(client, huge)
    assert r.status_code == 413
    assert r.json()["error"]["type"] == "invalid_request_error" and "8 MB" in r.json()["error"]["message"]


# ------------------------------------------------------------------ lifecycle


def test_chat_batch_lifecycle_output_and_error_files(api):
    client, eng, _ = api
    items = [
        line("ok-1"),
        line("bad-n", body=chat(n=2)),
        line("ok-2"),
        line("no-schema", body={"model": "clef-flash", "messages": [{"role": "user", "content": "hi"}]}),
        line("streamy", body=chat(stream=True)),
        line("wrong-model", body=chat(model="gpt-4o")),
        line("ok-3"),
    ]
    created = run_batch(client, items, metadata={"team": "ops"})
    assert created["object"] == "batch" and created["id"].startswith("batch_") and len(created["id"]) == 26
    assert created["status"] in ("in_progress", "completed")
    assert created["endpoint"] == CHAT and created["completion_window"] == "24h"
    assert created["metadata"] == {"team": "ops"} and created["errors"] is None
    assert created["request_counts"]["total"] == 7 and created["expires_at"] - created["created_at"] == 86400
    b = wait_batch(client, created["id"])
    assert b["status"] == "completed" and b["errors"] is None
    assert b["request_counts"] == {"total": 7, "completed": 3, "failed": 4}
    assert b["usage"]["input_tokens"] == 30 and b["usage"]["total_tokens"] == 30  # FakeEngine: 10 per request
    assert b["model"] == "clef-flash" and b["in_progress_at"] and b["completed_at"] and b["finalizing_at"]
    assert b["expired_at"] is b["failed_at"] is b["cancelled_at"] is b["cancelling_at"] is None

    out = rows(client, b["output_file_id"])
    assert [r["custom_id"] for r in out] == ["ok-1", "ok-2", "ok-3"]  # input order
    for r in out:
        assert r["id"].startswith("batch_req_") and r["error"] is None
        assert r["response"]["status_code"] == 200 and r["response"]["request_id"].startswith("req_")
        body = r["response"]["body"]
        assert body["object"] == "chat.completion" and body["model"] == "clef-flash"
        assert json.loads(body["choices"][0]["message"]["content"])["team"] == "account"  # last enum wins
        assert body["clef"]["request_id"] == r["response"]["request_id"]
    err = rows(client, b["error_file_id"])
    assert [r["custom_id"] for r in err] == ["bad-n", "no-schema", "streamy", "wrong-model"]
    assert [r["response"]["status_code"] for r in err] == [400, 400, 400, 404]
    assert all(r["error"] is None and r["response"]["body"]["error"]["message"] for r in err)
    assert [r["response"]["body"]["error"]["param"] for r in err] == [
        "n",
        "response_format",
        "stream",
        "model",
    ]
    assert err[3]["response"]["body"]["error"]["code"] == "model_not_found"
    assert sum(len(c) for c in eng.calls) == 3

    # the file objects: sizes are known now and match the streamed content
    for key in ("output_file_id", "error_file_id"):
        f = client.get(f"/v1/files/{b[key]}").json()
        content = client.get(f"/v1/files/{b[key]}/content").content
        assert f["purpose"] == "batch_output" and f["bytes"] == len(content) > 0
    # one batch = one job of kind batch
    jobs = client.get("/v1/jobs", params={"kind": "batch"}).json()["jobs"]
    assert len(jobs) == 1 and jobs[0]["status"] == "succeeded"
    assert jobs[0]["id"].removeprefix("job_") == b["id"].removeprefix("batch_")
    listed = client.get("/v1/batches").json()
    assert listed["data"] == [b] and listed["has_more"] is False and listed["first_id"] == b["id"]


def test_batch_bodies_match_the_live_routes(api):
    client, _, _ = api

    def scrub(o: Any) -> Any:
        if isinstance(o, dict):
            return {k: scrub(v) for k, v in o.items() if k not in ("id", "created", "request_id", "timing")}
        return [scrub(v) for v in o] if isinstance(o, list) else o

    cls = {"input": "refund me", "labels": {"billing": "money", "technical": "bugs"}, "instructions": "pick"}
    multi = {"input": "x", "labels": ["a", "b", "c"], "multi_label": True, "threshold": 0.4}
    score = {"input": "meh", "levels": ["bad", "ok", "good"]}
    chat_body = chat(logprobs=True, top_logprobs=2, temperature=0.2)
    items = [
        line("chat", CHAT, chat_body),
        line("cls", CLASSIFY, cls),
        line("multi", CLASSIFY, multi),
        line("score", SCORE, score),
    ]
    out = {}
    for url in (CHAT, CLASSIFY, SCORE):
        sel = [i for i in items if i["url"] == url]
        b = wait_batch(client, run_batch(client, sel, endpoint=url)["id"])
        assert b["request_counts"]["failed"] == 0, rows(client, b["error_file_id"])
        out.update({r["custom_id"]: r["response"]["body"] for r in rows(client, b["output_file_id"])})
    live = {
        "chat": client.post(CHAT, json=chat_body).json(),
        "cls": client.post(CLASSIFY, json=cls).json(),
        "multi": client.post(CLASSIFY, json=multi).json(),
        "score": client.post(SCORE, json=score).json(),
    }
    for name, body in live.items():
        assert scrub(out[name]) == scrub(body), name
    assert out["chat"]["choices"][0]["logprobs"]["content"]  # not just equal-and-empty


def test_all_requests_failing_is_still_a_completed_batch(api):
    client, _, _ = api
    b = run_batch(client, [line(f"c{i}", body=chat(n=2)) for i in range(20)])  # more than the breaker's 16
    b = wait_batch(client, b["id"])
    assert b["status"] == "completed", b
    assert b["request_counts"] == {"total": 20, "completed": 0, "failed": 20}
    assert b["output_file_id"] is None and len(rows(client, b["error_file_id"])) == 20
    assert client.get("/v1/jobs", params={"kind": "batch"}).json()["jobs"][0]["status"] == "succeeded"


def test_structural_problems_fail_the_batch_with_line_numbers(api):
    client, _, app = api
    items = [
        line("dup"),
        line("dup"),  # 2: duplicate custom_id
        b"{not json",  # 3
        line("ok"),
        line("wrong-url", url=CLASSIFY),  # 5
        line("m", method="GET"),  # 6
        {"method": "POST", "url": CHAT, "body": chat()},  # 7: no custom_id
        line("nobody", body=[1]),  # 8
        [1, 2],  # 9
    ]
    # line("m", method="GET") passes method through `more`; the helper's own "method" key is overwritten
    r = run_batch(client, items)
    assert r["status"] == "failed" and r["output_file_id"] is None and r["error_file_id"] is None
    got = {e["line"]: (e["code"], e["param"]) for e in r["errors"]["data"]}
    assert got == {
        2: ("duplicate_custom_id", "custom_id"),
        3: ("invalid_json_line", None),
        5: ("invalid_url", "url"),
        6: ("invalid_method", "method"),
        7: ("missing_custom_id", "custom_id"),
        8: ("invalid_request", "body"),
        9: ("invalid_request", None),
    }
    assert (
        all(e["message"] for e in r["errors"]["data"])
        and r["failed_at"]
        and r["request_counts"]["completed"] == 0
    )
    assert client.get(f"/v1/batches/{r['id']}").json() == r
    assert client.post(f"/v1/batches/{r['id']}/cancel").status_code == 409
    job = client.get("/v1/jobs", params={"kind": "batch"}).json()["jobs"][0]
    assert job["status"] == "failed" and "validation" in job["error"]
    assert sql(app, "SELECT COUNT(*) FROM jobs WHERE status = 'queued'")[0][0] == 0


def test_empty_and_oversized_inputs(api):
    client, _, _ = api
    small, _, _ = make(Eng(), max_batch_lines=2)
    with small:
        fid = upload(small, jsonl(*[line(f"c{i}") for i in range(3)])).json()["id"]
        b = new_batch(small, fid).json()
        assert b["status"] == "failed" and b["errors"]["data"][0]["code"] == "too_many_lines"
        assert b["errors"]["data"][0]["line"] == 3


def test_create_validation_errors(api):
    client, _, _ = api
    fid = upload(client, jsonl(line("a"))).json()["id"]

    def bad(r: Any, status: int, param: str | None):
        assert r.status_code == status, r.text
        err = r.json()["error"]
        assert err["type"] == "invalid_request_error" and err["param"] == param

    bad(new_batch(client, fid, "/v1/embeddings"), 400, "endpoint")
    bad(new_batch(client, fid, completion_window="48h"), 400, "completion_window")
    bad(new_batch(client, fid, metadata={str(i): "x" for i in range(17)}), 400, "metadata")
    bad(new_batch(client, fid, metadata={"k" * 65: "x"}), 400, "metadata")
    bad(new_batch(client, fid, metadata={"k": "v" * 513}), 400, "metadata")
    bad(new_batch(client, fid, metadata={"k": 1}), 400, "metadata")
    bad(new_batch(client, "file-" + "0" * 24), 404, None)
    bad(new_batch(client, "../../etc/passwd"), 404, None)
    bad(client.post("/v1/batches", json={"endpoint": CHAT}), 400, None)
    bad(client.get("/v1/batches/batch_nope"), 404, None)
    bad(client.get("/v1/batches", params={"limit": 101}), 400, None)
    # batch outputs are not inputs
    done = wait_batch(client, run_batch(client, [line("a")])["id"])
    bad(new_batch(client, done["output_file_id"]), 404, None)
    out = client.delete(f"/v1/files/{done['output_file_id']}")
    assert out.status_code == 400 and "DELETE /v1/jobs/job_" in out.json()["error"]["message"]


def test_batches_list_pages_newest_first(api):
    client, _, _ = api
    ids = [wait_batch(client, run_batch(client, [line(f"c{i}")])["id"])["id"] for i in range(3)]
    p1 = client.get("/v1/batches", params={"limit": 2}).json()
    assert [b["id"] for b in p1["data"]] == ids[::-1][:2] and p1["has_more"] is True
    p2 = client.get("/v1/batches", params={"limit": 2, "after": p1["last_id"]}).json()
    assert [b["id"] for b in p2["data"]] == [ids[0]] and p2["has_more"] is False


# ------------------------------------------------------------------ cancel and expiry


def test_cancel_a_running_batch(monkeypatch):
    client, eng, app = make(Eng(delay=0.15), max_microbatch=2)
    with client:
        fid = upload(client, jsonl(*[line(f"c{i}") for i in range(12)])).json()["id"]
        b = new_batch(client, fid).json()
        wait_batch(client, b["id"], until=("in_progress",))
        t0 = time.monotonic()
        while client.get(f"/v1/batches/{b['id']}").json()["request_counts"]["completed"] == 0:
            assert time.monotonic() - t0 < 10
            time.sleep(0.02)
        r = client.post(f"/v1/batches/{b['id']}/cancel")
        assert r.status_code == 200 and r.json()["status"] in ("cancelling", "cancelled")
        assert r.json()["cancelling_at"]
        b = wait_batch(client, b["id"])
        assert b["status"] == "cancelled" and b["cancelled_at"] and b["cancelling_at"]
        done = b["request_counts"]["completed"]
        assert 0 < done < 12 and b["request_counts"]["total"] == 12
        out, err = rows(client, b["output_file_id"]), rows(client, b["error_file_id"])
        assert [r["custom_id"] for r in out] == [f"c{i}" for i in range(done)]
        assert [r["custom_id"] for r in err] == [f"c{i}" for i in range(done, 12)]  # the ones that never ran
        assert {r["error"]["code"] for r in err} == {"batch_cancelled"} and all(
            r["response"] is None for r in err
        )
        assert client.post(f"/v1/batches/{b['id']}/cancel").status_code == 409  # already cancelled
        assert sql(app, "SELECT action FROM audit_log WHERE action = 'batch.cancel'")


def test_cancel_a_queued_batch(monkeypatch):
    client, _, app = make(Eng())
    with client:
        store = app.state.ctx.extra["jobs"].store
        monkeypatch.setattr(store, "claim_next", lambda: None)  # the runner never picks it up
        b = run_batch(client, [line("a"), line("b")])
        assert b["status"] == "in_progress" and b["in_progress_at"] is None
        r = client.post(f"/v1/batches/{b['id']}/cancel").json()
        assert r["status"] == "cancelled" and r["cancelling_at"] and r["cancelled_at"]
        assert r["output_file_id"] is None
        err = rows(client, r["error_file_id"])
        assert [e["custom_id"] for e in err] == ["a", "b"]
        assert err[0]["id"].startswith("batch_req_") and err[0]["error"]["code"] == "batch_cancelled"


def test_an_expired_batch_serves_what_it_did_and_names_the_rest(api):
    client, _, app = api
    fid = upload(client, jsonl(*[line(f"c{i}") for i in range(4)])).json()["id"]
    # make the job look expired after two rows (E1 ends such a job `cancelled` with error "expired")
    store = app.state.ctx.extra["jobs"].store
    store.claim_next = lambda: None  # type: ignore[method-assign]
    b = new_batch(client, fid).json()
    job_id = "job_" + b["id"].removeprefix("batch_")
    items = [
        {"index": i, "custom_id": f"c{i}", "status_code": 200, "request_id": f"req_{i}", "body": {"n": i}}
        for i in range(2)
    ]
    store.add_items(job_id, items)
    store.finish(job_id, "cancelled", error="expired")
    b = client.get(f"/v1/batches/{b['id']}").json()
    assert b["status"] == "expired" and b["expired_at"] and b["cancelled_at"] is None
    assert b["request_counts"] == {"total": 4, "completed": 2, "failed": 0}
    assert [r["response"]["body"] for r in rows(client, b["output_file_id"])] == [{"n": 0}, {"n": 1}]
    err = rows(client, b["error_file_id"])
    assert [(e["custom_id"], e["error"]["code"]) for e in err] == [
        ("c2", "batch_expired"),
        ("c3", "batch_expired"),
    ]


def test_deleting_the_job_deletes_the_batch_and_its_files(api):
    client, _, app = api
    b = wait_batch(client, run_batch(client, [line("a")])["id"])
    assert (
        client.delete(f"/v1/files/{b['input_file_id']}").status_code == 200
    )  # a finished batch no longer holds it
    job_id = "job_" + b["id"].removeprefix("batch_")
    assert client.delete(f"/v1/jobs/{job_id}").status_code == 200
    assert client.get(f"/v1/batches/{b['id']}").status_code == 404
    assert client.get(f"/v1/files/{b['output_file_id']}").status_code == 404
    assert sql(app, "SELECT COUNT(*) FROM files")[0][0] == 0


def test_an_input_file_in_use_is_not_deleted_or_purged(monkeypatch):
    client, _, app = make(Eng())
    with client:
        store = app.state.ctx.extra["jobs"].store
        monkeypatch.setattr(store, "claim_next", lambda: None)
        b = run_batch(client, [line("a")])
        r = client.delete(f"/v1/files/{b['input_file_id']}")
        assert r.status_code == 409 and "cancel" in r.json()["error"]["message"]
        files = app.state.ctx.extra["batches"]
        sql(app, "UPDATE files SET expires_at = 1 WHERE id = :i", i=b["input_file_id"])
        assert files.purge_expired() == 0 and client.get(f"/v1/files/{b['input_file_id']}").status_code == 200
        client.post(f"/v1/batches/{b['id']}/cancel")
        assert files.purge_expired() == 1  # the batch ended: the TTL applies again
        assert client.get(f"/v1/files/{b['input_file_id']}").status_code == 404
        assert not (files.dir / f"{b['input_file_id']}.jsonl").exists()


# ------------------------------------------------------------------ ownership, scopes, quota


def keyed():
    client, eng, app = make(Eng(), api_keys_raw="root:kr")
    return client, eng, app


def test_files_and_batches_are_private_to_their_owner():
    client, _, app = keyed()
    with client:
        alice = hdr(add_key(app, "alice", "jobs"))
        bob = hdr(add_key(app, "bob", "jobs"))
        root = hdr("kr")
        b = run_batch(client, [line("a")], headers=alice)
        b = wait_batch(client, b["id"], headers=alice)
        fids = [b["input_file_id"], b["output_file_id"]]
        for fid in fids:
            assert client.get(f"/v1/files/{fid}", headers=alice).status_code == 200
            for r in (
                client.get(f"/v1/files/{fid}", headers=bob),
                client.get(f"/v1/files/{fid}/content", headers=bob),
                client.delete(f"/v1/files/{fid}", headers=bob),
            ):
                assert r.status_code == 404 and r.json()["error"]["type"] == "invalid_request_error"
        for r in (
            client.get(f"/v1/batches/{b['id']}", headers=bob),
            client.post(f"/v1/batches/{b['id']}/cancel", headers=bob),
        ):
            assert r.status_code == 404
        assert client.get("/v1/files", headers=bob).json()["data"] == []
        assert client.get("/v1/batches", headers=bob).json()["data"] == []
        assert client.get("/v1/batches", params={"after": b["id"]}, headers=bob).status_code == 400
        assert len(client.get("/v1/files", headers=alice).json()["data"]) == 3  # input, output, error
        # another key cannot batch over her file, through the batch route or by submitting the job itself
        assert new_batch(client, b["input_file_id"], headers=bob).status_code == 404
        payload = {"input_file_id": b["input_file_id"], "endpoint": CHAT}
        job = client.post("/v1/jobs", json={"kind": "batch", "payload": payload}, headers=bob).json()
        deadline = time.monotonic() + 10
        while client.get(f"/v1/jobs/{job['id']}", headers=bob).json()["status"] not in (
            "failed",
            "succeeded",
        ):
            assert time.monotonic() < deadline
            time.sleep(0.02)
        stolen = client.get(f"/v1/jobs/{job['id']}", headers=bob).json()
        assert (
            stolen["status"] == "failed"
            and "not found" in stolen["error"]
            and stolen["progress"]["done"] == 0
        )
        # an admin sees everything but cannot start a batch from somebody else's file either
        assert client.get(f"/v1/batches/{b['id']}", headers=root).status_code == 200
        assert len(client.get("/v1/files", headers=root).json()["data"]) == 3
        assert client.get(f"/v1/files/{b['output_file_id']}/content", headers=root).status_code == 200
        assert new_batch(client, b["input_file_id"], headers=root).status_code == 404


def test_scopes_gate_files_and_batches():
    client, _, app = keyed()
    with client:
        classify_only = hdr(add_key(app, "reader", "classify"))
        jobs_key = hdr(add_key(app, "worker", "jobs"))
        no_key = {}
        for call in (
            lambda h: client.get("/v1/files", headers=h),
            lambda h: client.get("/v1/batches", headers=h),
            lambda h: upload(client, jsonl(line("a")), headers=h),
            lambda h: new_batch(client, "file-" + "0" * 24, headers=h),
            lambda h: client.get("/v1/files/file-x/content", headers=h),
            lambda h: client.post("/v1/batches/batch_x/cancel", headers=h),
        ):
            r = call(classify_only)
            assert r.status_code == 403 and r.json()["error"]["type"] == "invalid_request_error", r.text
            assert "jobs" in r.json()["error"]["message"]
            assert call(no_key).status_code == 401
        assert client.get("/v1/files", headers=jobs_key).status_code == 200
        # `urgent` is its own scope
        fid = upload(client, jsonl(line("a")), headers=jobs_key).json()["id"]
        r = new_batch(client, fid, headers=jobs_key, clef={"urgent": True})
        assert r.status_code == 403 and "urgent" in r.json()["error"]["message"]
        assert client.get("/v1/jobs", headers=jobs_key).json()["total"] == 0


def test_the_daily_quota_is_reserved_at_create_and_refused_when_short():
    client, _, app = keyed()
    with client:
        small = hdr(add_key(app, "small", "jobs", daily_quota=2))
        three = jsonl(*[line(f"c{i}") for i in range(3)])
        fid = upload(client, three, headers=small).json()["id"]
        r = new_batch(client, fid, headers=small)
        assert r.status_code == 429 and r.json()["error"]["type"] == "insufficient_quota", r.text
        assert client.get("/v1/jobs", headers=small).json()["total"] == 0 and r.headers["retry-after"]
        assert client.get("/v1/batches", headers=small).json()["data"] == []
        # the same file cannot dodge the quota by being submitted as a job with a smaller claimed size
        payload = {"input_file_id": fid, "endpoint": CHAT, "lines": 1}
        r = client.post("/v1/jobs", json={"kind": "batch", "payload": payload}, headers=small)
        assert r.status_code == 429
        ok = run_batch(client, [line("a"), line("b")], headers=small)
        b = wait_batch(client, ok["id"], headers=small)
        assert b["request_counts"]["completed"] == 2
        assert new_batch(client, fid, headers=small).status_code == 429  # spent: 2 of 2 used


# ------------------------------------------------------------------ resume


def test_a_batch_resumes_after_an_interruption(api):
    client, _, app = api
    store = app.state.ctx.extra["jobs"].store
    real = store.claim_next
    store.claim_next = lambda: None  # type: ignore[method-assign]
    b = run_batch(client, [line(f"c{i}") for i in range(5)])
    job_id = "job_" + b["id"].removeprefix("batch_")
    # two rows were stored before the "crash"; the job is interrupted and resumable
    rows_ = [
        {
            "index": i,
            "custom_id": f"c{i}",
            "status_code": 200,
            "request_id": f"req_{i}",
            "body": {"stored": i},
        }
        for i in range(2)
    ]
    store.add_items(job_id, rows_)
    sql(app, "UPDATE jobs SET status = 'interrupted', resumes = 1 WHERE id = :i", i=job_id)
    store.claim_next = real  # type: ignore[method-assign]
    app.state.ctx.extra["jobs"].poke()
    b = wait_batch(client, b["id"])
    assert b["status"] == "completed" and b["request_counts"] == {"total": 5, "completed": 5, "failed": 0}
    out = rows(client, b["output_file_id"])
    assert [r["custom_id"] for r in out] == [f"c{i}" for i in range(5)]
    assert out[0]["response"]["body"] == {"stored": 0}  # kept, not recomputed
    assert out[4]["response"]["body"]["object"] == "chat.completion"


# ------------------------------------------------------------------ the real openai SDK


def sdk_client(client: Any, headers: dict[str, str] | None = None):
    openai = pytest.importorskip("openai")
    httpx2 = pytest.importorskip("httpx2")

    class Forward(httpx2.BaseTransport):
        """Send the SDK's request through the Starlette TestClient (the app runs in-process)."""

        def handle_request(self, request: Any) -> Any:
            r = client.request(
                request.method, str(request.url), headers=dict(request.headers), content=request.read()
            )
            keep = [
                (k, v)
                for k, v in r.headers.multi_items()
                if k.lower() not in ("content-length", "content-encoding")
            ]
            return httpx2.Response(r.status_code, headers=keep, content=r.content)

    return openai.OpenAI(
        base_url="http://testserver/v1",
        api_key="unused",
        max_retries=0,
        default_headers=headers or {},
        http_client=httpx2.Client(transport=Forward()),
    )


def test_openai_sdk_files_and_batches_end_to_end(api):
    client, _, _ = api
    sdk = sdk_client(client)
    openai = pytest.importorskip("openai")
    data = jsonl(line("one"), line("two"), line("bad", body=chat(n=3)))
    f = sdk.files.create(
        file=("in.jsonl", data), purpose="batch", expires_after={"anchor": "created_at", "seconds": 7200}
    )
    assert f.purpose == "batch" and f.bytes == len(data) and f.expires_at - f.created_at == 7200
    assert sdk.files.retrieve(f.id).filename == "in.jsonl"
    assert [x.id for x in sdk.files.list(purpose="batch").data] == [f.id]
    batch = sdk.batches.create(
        input_file_id=f.id, endpoint=CHAT, completion_window="24h", metadata={"run": "1"}
    )
    assert batch.object == "batch" and batch.input_file_id == f.id and batch.metadata == {"run": "1"}
    done = None
    for _ in range(500):
        done = sdk.batches.retrieve(batch.id)
        if done.status in DONE:
            break
        time.sleep(0.02)
    assert done.status == "completed"
    assert (done.request_counts.total, done.request_counts.completed, done.request_counts.failed) == (3, 2, 1)
    assert done.usage.input_tokens == 20
    out = [json.loads(x) for x in sdk.files.content(done.output_file_id).text.splitlines()]
    assert [o["custom_id"] for o in out] == ["one", "two"]
    err = [json.loads(x) for x in sdk.files.content(done.error_file_id).text.splitlines()]
    assert err[0]["custom_id"] == "bad" and err[0]["response"]["status_code"] == 400
    assert sdk.files.retrieve(done.output_file_id).purpose == "batch_output"
    assert [b.id for b in sdk.batches.list(limit=5).data] == [batch.id]
    with pytest.raises(openai.ConflictError):
        sdk.batches.cancel(batch.id)  # already completed
    with pytest.raises(openai.NotFoundError):
        sdk.batches.retrieve("batch_" + "0" * 20)
    with pytest.raises(openai.BadRequestError) as ei:
        sdk.files.create(file=("in.csv", b"a,b"), purpose="batch")
    assert ".jsonl" in str(ei.value)
    assert sdk.files.delete(f.id).deleted is True
    # the SDK parses a structurally failed batch too
    f2 = sdk.files.create(file=("bad.jsonl", b"nope\n"), purpose="batch")
    failed = sdk.batches.create(input_file_id=f2.id, endpoint=CHAT, completion_window="24h")
    assert failed.status == "failed" and failed.errors.data[0].code == "invalid_json_line"
    assert failed.errors.data[0].line == 1


def test_openai_sdk_with_a_scoped_key():
    client, _, app = keyed()
    with client:
        sdk = sdk_client(client, hdr(add_key(app, "alice", "jobs")))
        openai = pytest.importorskip("openai")
        sdk_other = sdk_client(client, hdr(add_key(app, "bob", "classify")))
        f = sdk.files.create(file=("in.jsonl", jsonl(line("a"))), purpose="batch")
        with pytest.raises(openai.PermissionDeniedError):
            sdk_other.files.retrieve(f.id)
        assert sdk.files.retrieve(f.id).id == f.id


# ------------------------------------------------------------------ the unit pieces


def test_scan_input_and_offsets(tmp_path):
    p = tmp_path / "x.jsonl"
    good = jsonl(line("a"), line("b", CLASSIFY, {"input": "x", "labels": ["a", "b"]}))
    p.write_bytes(good + b"\n\n")
    assert batches.scan_input(p, CHAT, 10)[1][0]["code"] == "invalid_url"  # line 2 is a classify line
    p.write_bytes(jsonl(line("a"), line("b")) + b"\n   \n")
    assert batches.scan_input(p, CHAT, 10) == (2, [])
    offs = batches.line_offsets(p, 10)
    assert len(offs) == 2 and p.read_bytes()[offs[1] :].startswith(b'{"custom_id": "b"')
    assert list(batches.input_custom_ids(p, 1)) == [(1, "b")]
    assert batches.scan_input(p, CHAT, 1)[1][0]["code"] == "too_many_lines"
    p.write_bytes(b"\n\n")
    assert batches.scan_input(p, CHAT, 10)[1][0]["code"] == "empty_file"
    many = jsonl(*[{"custom_id": "x", "method": "GET", "url": CHAT, "body": {}} for _ in range(300)])
    p.write_bytes(many)
    assert len(batches.scan_input(p, CHAT, 1000)[1]) == batches.MAX_STRUCTURE_ERRORS  # capped, not 300


def test_every_batch_route_declares_the_jobs_scope():
    _, _, app = keyed()
    routes = [r for r in api_routes(app) if r.path.startswith(("/v1/files", "/v1/batches"))]
    assert {r.path for r in routes} >= {
        "/v1/files",
        "/v1/files/{file_id}",
        "/v1/files/{file_id}/content",
        "/v1/batches",
        "/v1/batches/{batch_id}",
        "/v1/batches/{batch_id}/cancel",
    }
    assert len(routes) == 9 and all(("jobs",) in _scope_marks(r.dependant) for r in routes)


def test_config_defaults(monkeypatch):
    cfg = Config(state_dir=tempfile.mkdtemp())
    assert (cfg.max_file_mb, cfg.max_batch_lines, cfg.file_ttl_hours) == (200, 50_000, 720.0)
    monkeypatch.setenv("CLEF_MAX_FILE_MB", "0")
    with pytest.raises(ValueError, match="CLEF_MAX_FILE_MB"):
        Config(state_dir=tempfile.mkdtemp())
    assert FakeEngine is not None and create_app is not None
