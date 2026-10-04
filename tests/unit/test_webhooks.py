"""Webhooks: allow-list / SSRF policy, signing, retries, DNS-rebinding pinning, end to end via the job API."""

from __future__ import annotations

import hashlib
import hmac
import json
import tempfile
import time
from typing import Any

import httpx
import pytest
from test_jobs import LABELS, SlowEngine, make, submit, wait

from clef_server.config import Config
from clef_server.jobs import JobStore
from clef_server.paths import jobs_db
from clef_server.webhooks import WebhookDispatcher, WebhookPolicy, WebhookRefused, sign

PUBLIC = "93.184.216.34"


def policy(*entries: str) -> WebhookPolicy:
    return WebhookPolicy(entries)


# ------------------------------------------------------------------ policy


def test_disabled_without_allow_list():
    p = policy()
    assert not p.enabled
    with pytest.raises(WebhookRefused, match="disabled"):
        p.check_url("https://hooks.example.com/x")


@pytest.mark.parametrize(
    "url",
    [
        "https://evil.example.org/x",  # host not listed
        "https://hooks.example.com.evil.org/x",  # suffix trick
        "https://sub.hooks.example.com/x",  # exact entry does not match subdomains
        "ftp://hooks.example.com/x",
        "file:///etc/passwd",
        "https://user:pw@hooks.example.com/x",
        "https://hooks.example.com:notaport/x",
        "http://10.0.0.5/x",  # IP literal outside the listed CIDR
        "http://93.184.216.34/x",  # public IP literals must be listed too
        "http://169.254.169.254/latest/meta-data",  # cloud metadata
        "http://[::1]/x",
        "https:///nohost",
    ],
)
def test_check_url_refuses(url):
    with pytest.raises(WebhookRefused):
        policy("hooks.example.com", "192.168.1.0/24").check_url(url)


def test_check_url_accepts_listed():
    p = policy("hooks.example.com", "*.corp.example.net", "192.168.1.0/24", "::1")
    t = p.check_url("https://Hooks.Example.com./path/x?a=1")
    assert (t.scheme, t.host, t.port, t.path, t.literal) == (
        "https",
        "hooks.example.com",
        443,
        "/path/x?a=1",
        False,
    )
    assert p.check_url("http://a.corp.example.net:8080").port == 8080
    assert p.check_url("http://192.168.1.77:9000/hook").literal
    assert p.check_url("http://[::1]:9/x").host == "::1"
    assert p.check_url("https://hooks.example.com").path == "/"


def test_link_local_is_refused_even_if_listed():
    p = policy("0.0.0.0/0", "169.254.0.0/16", "metadata.internal")
    with pytest.raises(WebhookRefused):
        p.check_url("http://169.254.169.254/")
    t = p.check_url("http://metadata.internal/")
    with pytest.raises(WebhookRefused):
        p.check_addresses(t, ["169.254.169.254"])


def test_resolved_addresses_must_be_public_or_listed():
    p = policy("hooks.example.com", "10.1.0.0/16")
    t = p.check_url("https://hooks.example.com/")
    assert p.check_addresses(t, [PUBLIC]) == [PUBLIC]
    assert p.check_addresses(t, ["10.1.2.3"]) == ["10.1.2.3"]  # private but explicitly listed
    for bad in (["127.0.0.1"], ["10.2.0.1"], ["192.168.0.1"], ["::1"], ["fe80::1"], ["::ffff:127.0.0.1"], []):
        with pytest.raises(WebhookRefused):
            p.check_addresses(t, bad)
    with pytest.raises(WebhookRefused):  # one bad answer poisons the lot (round-robin DNS tricks)
        p.check_addresses(t, [PUBLIC, "10.2.0.1"])


# ------------------------------------------------------------------ signing


def verify(secret: str, headers: dict[str, str], body: bytes, tolerance: float = 300.0) -> bool:
    """The receiver-side check documented in docs/jobs.md."""
    ts = headers["X-Clef-Timestamp"]
    if abs(time.time() - int(ts)) > tolerance:
        return False
    expected = "sha256=" + hmac.new(secret.encode(), ts.encode() + b"." + body, hashlib.sha256).hexdigest()
    return hmac.compare_digest(expected, headers["X-Clef-Signature"])


def test_sign_matches_documented_scheme():
    sig = sign("k", "1700000000", b'{"a":1}')
    assert sig == "sha256=" + hmac.new(b"k", b'1700000000.{"a":1}', hashlib.sha256).hexdigest()


# ------------------------------------------------------------------ dispatcher (no app)


class Recorder:
    def __init__(self, *statuses: int | Exception):
        self.statuses = list(statuses) or [200]
        self.requests: list[httpx.Request] = []

    def __call__(self, req: httpx.Request) -> httpx.Response:
        self.requests.append(req)
        s = self.statuses.pop(0) if len(self.statuses) > 1 else self.statuses[0]
        if isinstance(s, Exception):
            raise s
        return httpx.Response(s, headers={"Location": "http://10.0.0.1/"} if 300 <= s < 400 else {})


def make_dispatcher(rec: Recorder, resolver=None, pol=None, attempts=5, hook=None):
    records: dict[str, dict] = {}
    hook = hook or {"url": "https://hooks.example.com/clef?x=1", "secret": "s3cret"}
    job = {"id": "job_1", "status": "succeeded", "webhook": {"deliveries": {}}}

    async def resolve(host: str, port: int) -> list[str]:
        return [PUBLIC]

    async def nosleep(_: float) -> None:
        return None

    d = WebhookDispatcher(
        pol or policy("hooks.example.com"),
        lambda jid: {"webhook": hook, "job": job},
        lambda jid, ev, rec_: records.__setitem__(ev, dict(rec_)),
        transport=httpx.MockTransport(rec),
        resolver=resolver or resolve,
        sleep=nosleep,
        attempts=attempts,
    )
    return d, records


@pytest.mark.asyncio
async def test_delivery_headers_signature_and_pinning():
    rec = Recorder(200)
    d, records = make_dispatcher(rec)
    await d.deliver("job_1", "job.succeeded")
    req = rec.requests[0]
    assert req.method == "POST"
    assert req.url.host == PUBLIC and req.url.path == "/clef" and req.url.query == b"x=1"  # pinned address
    assert req.headers["Host"] == "hooks.example.com"
    assert req.extensions["sni_hostname"] == "hooks.example.com"
    h = req.headers
    assert h["X-Clef-Event"] == "job.succeeded" and h["Content-Type"] == "application/json"
    assert h["X-Clef-Delivery"] == records["job.succeeded"]["id"]
    assert verify("s3cret", h, req.content)
    assert not verify("wrong", h, req.content)
    assert not verify("s3cret", h, req.content + b" ")
    body = json.loads(req.content)
    assert body["event"] == "job.succeeded" and body["delivery_id"] == h["X-Clef-Delivery"]
    assert body["job"]["id"] == "job_1" and "s3cret" not in req.content.decode()
    assert not {"x-api-key", "authorization", "cookie"} & {k.lower() for k in h}
    assert records["job.succeeded"]["status"] == "delivered" and records["job.succeeded"]["attempts"] == 1


@pytest.mark.asyncio
async def test_no_secret_means_no_signature_headers():
    rec = Recorder(204)
    d, records = make_dispatcher(rec, hook={"url": "https://hooks.example.com/"})
    await d.deliver("job_1", "job.failed")
    h = rec.requests[0].headers
    assert "x-clef-signature" not in h and "x-clef-timestamp" not in h and h["X-Clef-Event"] == "job.failed"
    assert records["job.failed"]["status"] == "delivered"


@pytest.mark.asyncio
async def test_retries_5xx_then_succeeds_with_same_delivery_id():
    rec = Recorder(500, 503, 200)
    d, records = make_dispatcher(rec)
    await d.deliver("job_1", "job.succeeded")
    assert len(rec.requests) == 3 and records["job.succeeded"]["status"] == "delivered"
    assert len({r.headers["X-Clef-Delivery"] for r in rec.requests}) == 1
    assert records["job.succeeded"]["attempts"] == 3


@pytest.mark.asyncio
async def test_gives_up_after_max_attempts():
    rec = Recorder(500)
    d, records = make_dispatcher(rec, attempts=4)
    await d.deliver("job_1", "job.succeeded")
    r = records["job.succeeded"]
    assert len(rec.requests) == 4 and r["status"] == "failed" and r["last_status"] == 500


@pytest.mark.asyncio
async def test_transport_errors_are_retried():
    rec = Recorder(httpx.ConnectError("boom"), httpx.ReadTimeout("slow"), 200)
    d, records = make_dispatcher(rec)
    await d.deliver("job_1", "job.succeeded")
    assert len(rec.requests) == 3 and records["job.succeeded"]["status"] == "delivered"


@pytest.mark.asyncio
@pytest.mark.parametrize("status", [400, 401, 404, 410, 302, 301])
async def test_client_errors_and_redirects_are_final(status):
    rec = Recorder(status)
    d, records = make_dispatcher(rec)
    await d.deliver("job_1", "job.succeeded")
    assert len(rec.requests) == 1  # no retry, and the redirect target is never contacted
    assert records["job.succeeded"]["status"] == "failed"
    assert str(status) in records["job.succeeded"]["last_error"]


@pytest.mark.asyncio
async def test_429_is_retried():
    rec = Recorder(429, 200)
    d, records = make_dispatcher(rec)
    await d.deliver("job_1", "job.succeeded")
    assert len(rec.requests) == 2 and records["job.succeeded"]["status"] == "delivered"


@pytest.mark.asyncio
async def test_dns_rebinding_is_caught_per_attempt():
    answers = [[PUBLIC], ["169.254.169.254"]]

    async def rebinding(host: str, port: int) -> list[str]:
        return answers.pop(0) if len(answers) > 1 else answers[0]

    rec = Recorder(500)
    d, records = make_dispatcher(rec, resolver=rebinding)
    await d.deliver("job_1", "job.succeeded")
    r = records["job.succeeded"]
    assert len(rec.requests) == 1  # second attempt re-resolved to metadata and was refused, not sent
    assert r["status"] == "failed" and "refused" in r["last_error"] and r["attempts"] == 2


@pytest.mark.asyncio
async def test_private_resolution_refused_and_never_sent():
    async def private(host: str, port: int) -> list[str]:
        return ["10.0.0.7"]

    rec = Recorder(200)
    d, records = make_dispatcher(rec, resolver=private)
    await d.deliver("job_1", "job.succeeded")
    assert rec.requests == [] and records["job.succeeded"]["status"] == "failed"


@pytest.mark.asyncio
async def test_allow_list_change_applies_to_pending_deliveries():
    rec = Recorder(200)
    d, records = make_dispatcher(rec, pol=policy("other.example.com"))
    await d.deliver("job_1", "job.succeeded")
    assert rec.requests == [] and "not in CLEF_WEBHOOK_ALLOW" in records["job.succeeded"]["last_error"]


@pytest.mark.asyncio
async def test_progress_events_are_single_attempt_and_throttled():
    rec = Recorder(500)
    d, records = make_dispatcher(rec)
    d.progress("job_1", True)
    d.progress("job_1", True)  # throttled
    d.progress("job_1", False)  # not subscribed
    await d.drain()
    assert len(rec.requests) == 1 and records["job.progress"]["status"] == "failed"


@pytest.mark.asyncio
async def test_literal_ip_receiver_has_no_sni():
    rec = Recorder(200)
    d, _ = make_dispatcher(
        rec, pol=policy("192.168.1.0/24"), hook={"url": "https://192.168.1.5:8443/h", "secret": "x"}
    )
    await d.deliver("job_1", "job.succeeded")
    req = rec.requests[0]
    assert req.url.host == "192.168.1.5" and req.url.port == 8443 and "sni_hostname" not in req.extensions
    assert req.headers["Host"] == "192.168.1.5:8443"


# ------------------------------------------------------------------ end to end through the API


def with_receiver(app: Any, rec: Recorder, ip: str = PUBLIC):
    d = app.state.ctx.extra["jobs"].dispatcher

    async def resolve(host: str, port: int) -> list[str]:
        return [ip]

    async def nosleep(_: float) -> None:
        return None

    d.transport, d.resolver, d._sleep = httpx.MockTransport(rec), resolve, nosleep
    return d


def delivery(client, jid: str, event: str, timeout: float = 10.0) -> dict:
    t0 = time.monotonic()
    while True:
        rec = client.get(f"/v1/jobs/{jid}").json()["webhook"]["deliveries"].get(event, {})
        if rec.get("status") in ("delivered", "failed"):
            return rec
        assert time.monotonic() - t0 < timeout, rec
        time.sleep(0.02)


def hook_cfg(**kw: Any) -> dict[str, Any]:
    return {"webhook_allow": ("hooks.example.com",), **kw}


def test_submit_rejects_webhooks_when_disabled_or_host_not_allowed():
    client, _, _ = make()
    with client:
        r = submit(client, webhook={"url": "https://hooks.example.com/x"})
        assert r.status_code == 400 and "webhooks are disabled" in r.json()["detail"]
    client, _, _ = make(**hook_cfg())
    with client:
        r = submit(client, webhook={"url": "http://169.254.169.254/x"})
        assert r.status_code == 400 and r.json()["detail"].startswith("webhook.url:")
        r = submit(client, webhook={"url": "https://elsewhere.example.org/x"})
        assert r.status_code == 400 and "not in CLEF_WEBHOOK_ALLOW" in r.json()["detail"]
        assert (
            submit(client, webhook={"url": "https://hooks.example.com/x", "events": ["nope"]}).status_code
            == 400
        )
        assert submit(client, webhook={"url": "https://hooks.example.com/x", "events": []}).status_code == 400
        assert submit(client, webhook={"url": "https://hooks.example.com/x", "x": 1}).status_code == 400
        assert client.get("/health").json()["limits"]["webhooks_enabled"] is True


def test_job_succeeded_webhook_end_to_end():
    client, _, app = make(**hook_cfg(), api_keys_raw="alice:KEY-ALICE")
    rec = Recorder(200)
    hdr = {"X-API-Key": "KEY-ALICE"}
    with client:
        with_receiver(app, rec)
        r = submit(
            client, headers=hdr, webhook={"url": "https://hooks.example.com/clef", "secret": "topsecret"}
        )
        assert r.status_code == 202
        hook = r.json()["webhook"]
        assert (
            hook["has_secret"] is True
            and "secret" not in hook
            and hook["events"]
            == [
                "job.succeeded",
                "job.failed",
                "job.cancelled",
            ]
        )
        jid = r.json()["id"]
        wait(client, jid, headers=hdr)
        d = delivery_auth(client, jid, "job.succeeded", hdr)
        assert d["status"] == "delivered" and d["attempts"] == 1
        req = rec.requests[0]
        body = json.loads(req.content)
        assert (
            body["event"] == "job.succeeded"
            and body["job"]["id"] == jid
            and body["job"]["status"] == "succeeded"
        )
        assert body["job"]["progress"]["done"] == 3 and body["job"]["webhook"]["has_secret"] is True
        assert verify("topsecret", req.headers, req.content)
        wire = b"".join([req.content, json.dumps(dict(req.headers)).encode()])
        assert (
            b"KEY-ALICE" not in wire and b"topsecret" not in req.content
        )  # no API key, no secret in the body
        assert "topsecret" not in json.dumps(client.get(f"/v1/jobs/{jid}", headers=hdr).json())
        assert len(rec.requests) == 1  # only subscribed events


def delivery_auth(client, jid, event, headers, timeout=10.0):
    t0 = time.monotonic()
    while True:
        rec = client.get(f"/v1/jobs/{jid}", headers=headers).json()["webhook"]["deliveries"].get(event, {})
        if rec.get("status") in ("delivered", "failed"):
            return rec
        assert time.monotonic() - t0 < timeout, rec
        time.sleep(0.02)


def test_failed_and_cancelled_events():
    eng = SlowEngine(delay=0.1)
    client, _, app = make(eng, **hook_cfg())
    rec = Recorder(200)
    with client:
        with_receiver(app, rec)
        hook = {"url": "https://hooks.example.com/h"}
        eng.raises = RuntimeError("x")
        failed = submit(client, webhook=hook).json()["id"]
        assert delivery(client, failed, "job.failed")["status"] == "delivered"
        eng.raises = None
        slow = submit(
            client, payload={"inputs": [f"t{i}" for i in range(80)], "labels": LABELS}, webhook=hook
        )
        jid = slow.json()["id"]
        queued = submit(client, webhook=hook).json()["id"]
        client.post(f"/v1/jobs/{queued}/cancel")  # cancelled while queued also notifies
        assert delivery(client, queued, "job.cancelled")["status"] == "delivered"
        client.post(f"/v1/jobs/{jid}/cancel")
        assert delivery(client, jid, "job.cancelled")["status"] == "delivered"
        events = sorted(r.headers["X-Clef-Event"] for r in rec.requests)
        assert events == ["job.cancelled", "job.cancelled", "job.failed"]


def test_progress_event_is_opt_in():
    eng = SlowEngine(delay=0.05)
    client, _, app = make(eng, **hook_cfg())
    rec = Recorder(200)
    with client:
        with_receiver(app, rec)
        url = {"url": "https://hooks.example.com/h"}
        jid = submit(
            client, payload={"inputs": [f"t{i}" for i in range(24)], "labels": LABELS}, webhook=url
        ).json()["id"]
        wait(client, jid)
        delivery(client, jid, "job.succeeded")
        assert "job.progress" not in {r.headers["X-Clef-Event"] for r in rec.requests}
        rec.requests.clear()
        url = {"url": "https://hooks.example.com/h", "events": ["job.progress", "job.succeeded"]}
        jid = submit(
            client, payload={"inputs": [f"t{i}" for i in range(24)], "labels": LABELS}, webhook=url
        ).json()["id"]
        wait(client, jid)
        delivery(client, jid, "job.succeeded")
        assert [r.headers["X-Clef-Event"] for r in rec.requests].count("job.progress") == 1  # throttled


def test_pending_delivery_is_retried_after_restart():
    state = tempfile.mkdtemp(prefix="clef-jobs-")
    store = JobStore(jobs_db(Config(state_dir=state)))
    hook = {"url": "https://hooks.example.com/h", "secret": "s"}
    row = store.create("classify", None, {}, None, hook, 10)
    store.mark_running(row["id"], 0)
    store.finish(row["id"], "succeeded")
    store.set_delivery(
        row["id"],
        "job.succeeded",
        {"id": "dlv_1", "event": "job.succeeded", "status": "pending", "attempts": 2},
    )
    store.close()
    client, _, app = make(state_dir=state, **hook_cfg())
    rec = Recorder(200)

    async def swap() -> None:  # runs before the runner's startup hook re-queues the pending delivery
        with_receiver(app, rec)

    app.state.ctx.on_startup.insert(0, swap)
    with client:
        d = delivery(client, row["id"], "job.succeeded")
        assert d["status"] == "delivered" and d["id"] == "dlv_1" and d["attempts"] == 1
        assert rec.requests[0].headers["X-Clef-Delivery"] == "dlv_1"


# ------------------------------------------------------------------ review fixes: SSRF via embedded IPv4


@pytest.mark.parametrize(
    "addr",
    [
        "64:ff9b::a00:5",  # NAT64 -> 10.0.0.5
        "64:ff9b::7f00:1",  # NAT64 -> 127.0.0.1
        "64:ff9b::a9fe:a9fe",  # NAT64 -> 169.254.169.254 (metadata)
        "64:ff9b:1:a00:0:500:0:0",  # RFC 8215 local-use NAT64 /48 -> 10.0.0.5
        "2002:a00:5::1",  # 6to4 -> 10.0.0.5
        "2002:7f00:1::",  # 6to4 -> 127.0.0.1
        "2001:0:4136:e378:8000:63bf:f5ff:fffa",  # Teredo, embedded client 10.0.0.5
        "::ffff:10.0.0.5",
        "fec0::1",  # site-local
        "100::1",  # reserved (discard prefix)
    ],
)
def test_embedded_ipv4_and_reserved_v6_are_refused(addr):
    p = policy("hooks.example.com", "10.1.0.0/16")
    t = p.check_url("https://hooks.example.com/")
    with pytest.raises(WebhookRefused):
        p.check_addresses(t, [addr])


def test_embedded_ipv4_is_judged_by_the_ipv4_policy():
    p = policy("hooks.example.com", "10.1.0.0/16")
    t = p.check_url("https://hooks.example.com/")
    assert p.check_addresses(t, ["64:ff9b::5db8:d822"])  # NAT64 -> 93.184.216.34 (public): fine
    assert p.check_addresses(t, ["64:ff9b::a01:203"])  # NAT64 -> 10.1.2.3, explicitly listed
    assert p.check_addresses(t, ["2002:5db8:d822::1"])  # 6to4 -> public
    with pytest.raises(WebhookRefused):  # link-local is refused even when a CIDR would cover it
        policy("hooks.example.com", "0.0.0.0/0").check_addresses(t, ["64:ff9b::a9fe:a9fe"])
    with pytest.raises(WebhookRefused):  # a literal NAT64 URL needs the embedded address listed
        p.check_url("http://[64:ff9b::a00:5]/x")
    assert p.check_url("http://[64:ff9b::a01:203]/x").literal


def test_loopback_v6_stays_allow_listable():
    assert policy("::1").check_url("http://[::1]:9/x").host == "::1"
    with pytest.raises(WebhookRefused):
        policy("hooks.example.com").check_url("http://[::1]/x")


# ------------------------------------------------------------------ review fixes: dispatcher bookkeeping


def raw_dispatcher(**kw):
    records: dict[str, dict] = {}
    job = {"id": "job_1", "status": "succeeded", "webhook": {"deliveries": {}}}
    hook = {"url": "https://hooks.example.com/clef"}

    async def resolve(host: str, port: int) -> list[str]:
        return [PUBLIC]

    async def nosleep(_: float) -> None:
        return None

    args: dict[str, Any] = {
        "transport": httpx.MockTransport(Recorder(200)),
        "resolver": resolve,
        "sleep": nosleep,
    }
    args.update(kw)
    get_job = args.pop("get_job", lambda jid: {"webhook": hook, "job": job})
    set_delivery = args.pop("set_delivery", lambda jid, ev, rec_: records.__setitem__(ev, dict(rec_)))
    return WebhookDispatcher(policy("hooks.example.com"), get_job, set_delivery, **args), records


@pytest.mark.asyncio
async def test_unexpected_error_marks_the_delivery_failed_instead_of_pending():
    async def boom(host: str, port: int) -> list[str]:
        raise RuntimeError("resolver bug")

    d, records = raw_dispatcher(resolver=boom)
    await d.deliver("job_1", "job.succeeded")
    r = records["job.succeeded"]
    assert r["status"] == "failed" and "RuntimeError" in r["last_error"] and r["attempts"] == 1


@pytest.mark.asyncio
async def test_cancellation_still_propagates_out_of_deliver():
    import asyncio

    gate = asyncio.Event()

    async def slow_sleep(_: float) -> None:
        await gate.wait()

    d, _ = raw_dispatcher(transport=httpx.MockTransport(Recorder(500)), sleep=slow_sleep)
    task = asyncio.ensure_future(d.deliver("job_1", "job.succeeded"))
    await asyncio.sleep(0.05)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task


@pytest.mark.asyncio
async def test_set_delivery_failure_is_logged_not_swallowed_silently(caplog):
    def broken(jid, ev, rec_):
        raise RuntimeError("database is locked")

    d, _ = raw_dispatcher(set_delivery=broken)
    with caplog.at_level("WARNING", logger="clef"):
        await d.deliver("job_1", "job.succeeded")
    assert any(
        "cannot persist delivery state" in r.getMessage() and r.levelname == "WARNING" for r in caplog.records
    )


@pytest.mark.asyncio
async def test_sqlite_calls_run_off_the_event_loop():
    import threading

    main = threading.get_ident()
    threads: dict[str, int] = {}
    hook = {"url": "https://hooks.example.com/clef"}
    job = {"id": "job_1", "status": "succeeded", "webhook": {"deliveries": {}}}

    def get_job(jid):
        threads["get"] = threading.get_ident()
        return {"webhook": hook, "job": job}

    def set_delivery(jid, ev, rec_):
        threads["set"] = threading.get_ident()

    d, _ = raw_dispatcher(get_job=get_job, set_delivery=set_delivery)
    await d.deliver("job_1", "job.succeeded")
    assert threads["get"] != main and threads["set"] != main


@pytest.mark.asyncio
async def test_progress_throttle_entry_is_dropped_on_the_terminal_event():
    d, _ = raw_dispatcher()
    d.progress("job_1", True)
    assert "job_1" in d._last_progress
    d.enqueue("job_1", "job.succeeded")
    assert "job_1" not in d._last_progress
    await d.drain()


@pytest.mark.asyncio
async def test_default_retry_schedule_spans_minutes_and_is_capped():
    sleeps: list[float] = []

    async def record_sleep(s: float) -> None:
        sleeps.append(s)

    d, records = raw_dispatcher(
        transport=httpx.MockTransport(Recorder(503)),
        sleep=record_sleep,
        attempts=10,
        base_delay=2.0,
        max_delay=120.0,
    )
    await d.deliver("job_1", "job.succeeded")
    assert records["job.succeeded"]["status"] == "failed" and len(sleeps) == 9
    assert max(sleeps) <= 120 * 1.25 and sleeps[6] >= 120 * 0.75  # capped at 120 s
    assert 300 < sum(sleeps) < 700  # about eight minutes in all, not the old 30 seconds
    cfg = Config()
    assert (cfg.webhook_attempts, cfg.webhook_backoff_s, cfg.webhook_backoff_cap_s) == (10, 2.0, 120.0)


# ------------------------------------------------------------------ review fixes: through the API


def test_delivery_id_is_minted_with_the_pending_record_and_survives_to_the_receiver():
    client, _, app = make(**hook_cfg())
    rec = Recorder(200)
    runner = app.state.ctx.extra["jobs"]
    seen: dict[str, Any] = {}
    real_enqueue = runner.dispatcher.enqueue

    def spy(job_id: str, event: str) -> None:
        seen["record"] = runner.store.get(job_id)["deliveries"][event]  # persisted BEFORE anything is sent
        real_enqueue(job_id, event)

    with client:
        with_receiver(app, rec)
        runner.dispatcher.enqueue = spy
        jid = submit(client, webhook={"url": "https://hooks.example.com/clef"}).json()["id"]
        d = delivery(client, jid, "job.succeeded")
    assert seen["record"]["status"] == "pending" and seen["record"]["attempts"] == 0
    assert seen["record"]["id"] == d["id"] == rec.requests[0].headers["X-Clef-Delivery"]


def test_failed_delivery_is_visible_and_can_be_redelivered():
    client, _, app = make(**hook_cfg(), api_keys_raw="alice:ka,bob:kb")
    rec = Recorder(404)
    hdr = {"X-API-Key": "ka"}
    with client:
        with_receiver(app, rec)
        jid = submit(client, headers=hdr, webhook={"url": "https://hooks.example.com/clef"}).json()["id"]
        wait(client, jid, headers=hdr)
        failed = delivery_auth(client, jid, "job.succeeded", hdr)
        assert failed["status"] == "failed"
        job = client.get(f"/v1/jobs/{jid}", headers=hdr).json()
        assert job["webhook"]["failed_deliveries"] == 1
        assert any("failed for job.succeeded" in w and "redeliver" in w for w in job["warnings"])
        assert (
            client.post(f"/v1/jobs/{jid}/webhook/redeliver", headers={"X-API-Key": "kb"}).status_code == 404
        )
        rec.statuses = [200]  # the receiver is fixed
        r = client.post(f"/v1/jobs/{jid}/webhook/redeliver", headers=hdr)
        assert r.status_code == 200, r.text
        again = delivery_auth(client, jid, "job.succeeded", hdr)
        assert again["status"] == "delivered" and again["id"] == failed["id"]  # same id: de-duplicates
        assert rec.requests[-1].headers["X-Clef-Delivery"] == failed["id"]
        assert client.get(f"/v1/jobs/{jid}", headers=hdr).json()["webhook"]["failed_deliveries"] == 0
        done = client.post(f"/v1/jobs/{jid}/webhook/redeliver", headers=hdr)  # nothing failed any more
        assert done.status_code == 409 and "delivered" in done.json()["detail"]
        other = client.post(f"/v1/jobs/{jid}/webhook/redeliver?event=job.failed", headers=hdr)
        assert other.status_code == 409


def test_redeliver_rules():
    eng = SlowEngine(delay=0.2)
    client, _, app = make(eng, **hook_cfg())
    with client:
        with_receiver(app, Recorder(200))
        nohook = submit(client).json()["id"]
        assert client.post(f"/v1/jobs/{nohook}/webhook/redeliver").status_code == 409
        running = submit(
            client,
            payload={"inputs": [f"t{i}" for i in range(40)], "labels": LABELS},
            webhook={"url": "https://hooks.example.com/clef"},
        ).json()["id"]
        r = client.post(f"/v1/jobs/{running}/webhook/redeliver")
        assert r.status_code == 409 and "wait until it has finished" in r.json()["detail"]
        client.post(f"/v1/jobs/{running}/cancel")
        wait(client, running)
        assert client.post("/v1/jobs/nope/webhook/redeliver").status_code == 404
        assert client.post(f"/v1/jobs/{running}/webhook/redeliver?event=bogus").status_code == 400
