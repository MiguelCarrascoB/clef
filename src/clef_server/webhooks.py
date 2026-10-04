"""Webhook delivery for async jobs (see docs/jobs.md).

SSRF is the whole risk here, so the policy is strict and enforced on EVERY attempt:

* Webhooks are OFF unless ``CLEF_WEBHOOK_ALLOW`` lists hosts / IPs / CIDRs.
* A hostname must be listed (``hooks.example.com`` or ``*.example.com``); an IP-literal URL must fall in a
  listed IP / CIDR. Resolved addresses must be public, or inside a listed IP / CIDR. Link-local (cloud
  metadata), multicast, unspecified and reserved addresses are refused even when listed.
* IPv6 forms that embed an IPv4 address (IPv4-mapped, NAT64 ``64:ff9b::/96`` and ``64:ff9b:1::/48``, 6to4,
  Teredo) are judged by the embedded IPv4 address, so ``64:ff9b::a00:5`` cannot reach 10.0.0.5. Site-local and
  reserved IPv6 ranges are refused too.
* The hostname is resolved right before each attempt and the request is sent to the address that was
  checked (pinned; Host header and TLS SNI keep the hostname), so DNS rebinding cannot swap it afterwards.
* No redirects, no proxy environment, a hard timeout, and no ``X-API-Key`` / ``Authorization`` is ever sent.

The body is signed with HMAC-SHA256 over ``"<timestamp>.<body>"`` when the job's webhook has a secret.
"""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import ipaddress
import json
import logging
import random
import socket
import time
import uuid
from collections.abc import Awaitable, Callable, Iterable
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any
from urllib.parse import urlsplit

import httpx

from .config import VERSION

log = logging.getLogger("clef")

EVENTS = ("job.succeeded", "job.failed", "job.cancelled", "job.progress")
DEFAULT_EVENTS = ("job.succeeded", "job.failed", "job.cancelled")
PROGRESS_INTERVAL_S = 10.0
MAX_CONCURRENT = 4
BASE_DELAY_S = 2.0  # attempt n waits min(MAX_DELAY_S, BASE_DELAY_S * 2**(n-1)) (+-25 %) before attempt n+1
MAX_DELAY_S = 120.0  # with 10 attempts: 2, 4, 8, 16, 32, 64, 120, 120, 120 s (about 8 minutes in all)

_NAT64_WKP = ipaddress.IPv6Network("64:ff9b::/96")
_NAT64_LOCAL = ipaddress.IPv6Network("64:ff9b:1::/48")

Resolver = Callable[[str, int], Awaitable[list[str]]]


class WebhookRefused(ValueError):
    """The URL or the addresses it resolves to are not allowed by CLEF_WEBHOOK_ALLOW."""


def _unmap(ip: ipaddress.IPv4Address | ipaddress.IPv6Address) -> Any:
    return ip.ipv4_mapped if isinstance(ip, ipaddress.IPv6Address) and ip.ipv4_mapped else ip


def _embedded_v4(ip: ipaddress.IPv6Address) -> list[ipaddress.IPv4Address]:
    """IPv4 addresses a v6 address translates to / tunnels through (empty for ordinary v6)."""
    out: list[ipaddress.IPv4Address] = []
    if ip.ipv4_mapped:
        out.append(ip.ipv4_mapped)
    if ip.sixtofour:
        out.append(ip.sixtofour)
    if ip.teredo:
        out.extend(ip.teredo)  # (server, client)
    raw = ip.packed
    if ip in _NAT64_WKP:  # RFC 6052 /96: the last 32 bits
        out.append(ipaddress.IPv4Address(raw[12:16]))
    elif ip in _NAT64_LOCAL:  # RFC 8215 64:ff9b:1::/48: bits 48-63 and 72-87 (bits 64-71 are reserved)
        out.append(ipaddress.IPv4Address(raw[6:8] + raw[9:11]))
    return out


def sign(secret: str, timestamp: str, body: bytes) -> str:
    """``sha256=<hex>`` of HMAC-SHA256(secret, "<timestamp>.<body>"), what receivers must recompute."""
    mac = hmac.new(secret.encode(), timestamp.encode() + b"." + body, hashlib.sha256)
    return "sha256=" + mac.hexdigest()


@dataclass(frozen=True)
class Target:
    scheme: str
    host: str  # lower-case hostname or IP literal, no brackets
    port: int
    path: str  # path + query, always starts with "/"
    literal: bool  # host is an IP literal

    @property
    def host_header(self) -> str:
        default = 443 if self.scheme == "https" else 80
        host = f"[{self.host}]" if ":" in self.host else self.host
        return host if self.port == default else f"{host}:{self.port}"


class WebhookPolicy:
    """Parsed CLEF_WEBHOOK_ALLOW."""

    def __init__(self, entries: Iterable[str] = ()):
        self.hosts: set[str] = set()
        self.suffixes: list[str] = []
        self.nets: list[Any] = []
        for raw in entries:
            entry = raw.strip().lower()
            if not entry:
                continue
            try:
                self.nets.append(ipaddress.ip_network(entry, strict=False))
                continue
            except ValueError:
                pass
            if entry.startswith("*."):
                self.suffixes.append(entry[1:])  # ".example.com"
            else:
                self.hosts.add(entry.rstrip("."))

    @property
    def enabled(self) -> bool:
        return bool(self.hosts or self.suffixes or self.nets)

    def _net_allows(self, ip: Any) -> bool:
        return any(ip.version == n.version and ip in n for n in self.nets)

    def check_url(self, url: str) -> Target:
        """Syntax + allow-list check (no DNS). Raises WebhookRefused with a message fit for a 400."""
        if not self.enabled:
            raise WebhookRefused("webhooks are disabled on this server (set CLEF_WEBHOOK_ALLOW)")
        parts = urlsplit(url.strip())
        if parts.scheme not in ("http", "https") or not parts.hostname:
            raise WebhookRefused("only http(s) URLs are supported")
        if parts.username or parts.password:
            raise WebhookRefused("URLs with credentials are not allowed")
        try:
            port = parts.port or (443 if parts.scheme == "https" else 80)
        except ValueError as exc:
            raise WebhookRefused("invalid URL port") from exc
        host = parts.hostname.lower().rstrip(".")
        path = (parts.path or "/") + (f"?{parts.query}" if parts.query else "")
        try:
            ip = _unmap(ipaddress.ip_address(host))
        except ValueError:
            if host not in self.hosts and not any(host.endswith(s) for s in self.suffixes):
                raise WebhookRefused(f"host {host!r} is not in CLEF_WEBHOOK_ALLOW") from None
            return Target(parts.scheme, host, port, path, False)
        self._check_ip(ip, literal=True)
        return Target(parts.scheme, str(ip), port, path, True)

    def _check_ip(self, ip: Any, literal: bool) -> None:
        if ip.version == 6:
            embedded = _embedded_v4(ip)
            if embedded:  # a translated / tunnelled address is exactly as safe as the IPv4 inside it
                for v4 in embedded:
                    self._check_ip(v4, literal)
                return
        if (
            ip.is_link_local
            or ip.is_multicast
            or ip.is_unspecified
            or (ip.version == 4 and ip.is_reserved)
            or (ip.version == 6 and (ip.is_site_local or (ip.is_reserved and not ip.is_loopback)))
        ):
            raise WebhookRefused("address is link-local, multicast or reserved; refusing to deliver")
        if self._net_allows(ip):
            return
        if literal:
            raise WebhookRefused(f"address {ip} is not in CLEF_WEBHOOK_ALLOW")
        if not ip.is_global:
            raise WebhookRefused("host resolves to a non-public address that is not in CLEF_WEBHOOK_ALLOW")

    def check_addresses(self, target: Target, addrs: list[str]) -> list[str]:
        """Every resolved address must pass; returns them (the caller connects to the first)."""
        if not addrs:
            raise WebhookRefused(f"cannot resolve host {target.host!r}")
        for a in addrs:
            self._check_ip(_unmap(ipaddress.ip_address(a.split("%")[0])), literal=target.literal)
        return addrs


async def system_resolver(host: str, port: int) -> list[str]:
    loop = asyncio.get_running_loop()
    infos = await loop.getaddrinfo(host, port, type=socket.SOCK_STREAM)
    return list(dict.fromkeys(i[4][0] for i in infos))


def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


class WebhookDispatcher:
    """Delivers job events. State lives in the job row (``deliveries``), so pending ones survive a restart.

    ``get_job(job_id) -> (webhook, public_job) | None`` and ``set_delivery(job_id, event, record)`` are
    injected by jobs.py (keeps this module free of storage details).
    """

    def __init__(
        self,
        policy: WebhookPolicy,
        get_job: Callable[[str], dict[str, Any] | None],
        set_delivery: Callable[[str, str, dict[str, Any]], None],
        *,
        timeout_s: float = 10.0,
        attempts: int = 5,
        transport: httpx.AsyncBaseTransport | None = None,
        resolver: Resolver = system_resolver,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
        base_delay: float = BASE_DELAY_S,
        max_delay: float = MAX_DELAY_S,
    ):
        self.policy, self.get_job, self.set_delivery = policy, get_job, set_delivery
        self.timeout_s, self.attempts = timeout_s, max(1, attempts)
        self.transport, self.resolver, self._sleep, self.base_delay = transport, resolver, sleep, base_delay
        self.max_delay = max_delay
        self._tasks: set[asyncio.Task[None]] = set()
        self._sem: asyncio.Semaphore | None = None
        self._last_progress: dict[str, float] = {}
        self._closed = False

    # ---- scheduling
    def enqueue(self, job_id: str, event: str) -> None:
        if self._closed:
            return
        if event != "job.progress":
            self._last_progress.pop(job_id, None)  # the job is over; do not keep its throttle entry forever
        task = asyncio.get_running_loop().create_task(self.deliver(job_id, event))
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)

    def progress(self, job_id: str, wants: bool) -> None:
        """Throttled ``job.progress`` (single attempt, never retried)."""
        now = time.monotonic()
        if wants and now - self._last_progress.get(job_id, -1e9) >= PROGRESS_INTERVAL_S:
            self._last_progress[job_id] = now
            self.enqueue(job_id, "job.progress")

    async def close(self) -> None:
        self._closed = True
        for t in list(self._tasks):
            t.cancel()
        await asyncio.gather(*self._tasks, return_exceptions=True)

    async def drain(self) -> None:
        """Wait for in-flight deliveries (tests)."""
        while self._tasks:
            await asyncio.gather(*list(self._tasks), return_exceptions=True)

    # ---- delivery
    async def deliver(self, job_id: str, event: str) -> None:
        try:
            await self._deliver(job_id, event)
        except asyncio.CancelledError:
            raise
        except Exception:
            log.exception("webhook delivery crashed (job=%s event=%s)", job_id, event)

    async def _deliver(self, job_id: str, event: str) -> None:
        found = await asyncio.to_thread(self.get_job, job_id)  # sqlite: keep it off the event loop
        if not found:
            return
        hook, job = found["webhook"], found["job"]
        if not hook:
            return
        record = dict((job.get("webhook") or {}).get("deliveries", {}).get(event) or {})
        delivery_id = record.get("id") or uuid.uuid4().hex
        rec = {"id": delivery_id, "event": event, "status": "pending", "attempts": 0, **record}
        rec["id"], rec["status"] = delivery_id, "pending"
        body = json.dumps(
            {"event": event, "delivery_id": delivery_id, "created_at": _now_iso(), "job": job},
            separators=(",", ":"),
            default=str,
        ).encode()
        attempts = 1 if event == "job.progress" else self.attempts
        for attempt in range(1, attempts + 1):
            rec["attempts"] = attempt
            rec["last_attempt_at"] = _now_iso()
            retry = False
            try:
                status = await self._post(hook, event, delivery_id, body)
                rec["last_status"], rec["last_error"] = status, None
                if 200 <= status < 300:
                    rec.update(status="delivered", delivered_at=_now_iso())
                else:
                    rec["last_error"] = f"receiver answered HTTP {status}" + (
                        " (redirects are not followed)" if 300 <= status < 400 else ""
                    )
                    retry = status in (408, 425, 429) or status >= 500
            except WebhookRefused as exc:
                rec["last_status"], rec["last_error"] = None, f"refused: {exc}"
            except (httpx.HTTPError, OSError, asyncio.TimeoutError) as exc:
                rec["last_status"], rec["last_error"] = None, f"{type(exc).__name__}: {str(exc)[:200]}"
                retry = True
            except Exception as exc:  # a bug must not leave the record pending (and retried at every start)
                log.exception("webhook delivery attempt crashed (job=%s event=%s)", job_id, event)
                rec["last_status"], rec["last_error"] = None, f"internal error: {type(exc).__name__}"
            if rec["status"] != "delivered" and (not retry or attempt >= attempts):
                rec["status"] = "failed"
            try:
                await asyncio.to_thread(self.set_delivery, job_id, event, rec)
            except Exception:
                log.warning(
                    "webhook: cannot persist delivery state (job=%s event=%s)", job_id, event, exc_info=True
                )
            if rec["status"] != "pending":
                break
            delay = min(self.max_delay, self.base_delay * 2 ** (attempt - 1))
            await self._sleep(delay * (0.75 + random.random() / 2))
        if rec["status"] == "failed":
            log.warning("webhook %s for job %s failed: %s", event, job_id, rec.get("last_error"))

    async def _post(self, hook: dict[str, Any], event: str, delivery_id: str, body: bytes) -> int:
        target = self.policy.check_url(hook["url"])  # re-checked every attempt (config may have changed)
        addrs = self.policy.check_addresses(target, await self._addresses(target))
        ip = addrs[0]
        netloc = f"[{ip}]" if ":" in ip else ip
        headers = {
            "Host": target.host_header,
            "Content-Type": "application/json",
            "User-Agent": f"clef/{VERSION}",
            "X-Clef-Event": event,
            "X-Clef-Delivery": delivery_id,
        }
        secret = hook.get("secret")
        if secret:
            ts = str(int(time.time()))
            headers["X-Clef-Timestamp"] = ts
            headers["X-Clef-Signature"] = sign(secret, ts, body)
        ext: dict[str, Any] = {}
        if target.scheme == "https" and not target.literal:
            ext["sni_hostname"] = target.host  # certificate is verified against the hostname, not the IP
        async with self._semaphore():
            client = httpx.AsyncClient(
                transport=self.transport, follow_redirects=False, timeout=self.timeout_s, trust_env=False
            )
            async with client:
                resp = await asyncio.wait_for(  # httpx timeouts are per phase; this is the total
                    client.post(
                        f"{target.scheme}://{netloc}:{target.port}{target.path}",
                        content=body,
                        headers=headers,
                        extensions=ext,
                    ),
                    self.timeout_s,
                )
        return resp.status_code

    async def _addresses(self, target: Target) -> list[str]:
        if target.literal:
            return [target.host]
        try:
            return await self.resolver(target.host, target.port)
        except socket.gaierror as exc:
            raise OSError(f"cannot resolve host {target.host!r}") from exc

    def _semaphore(self) -> asyncio.Semaphore:
        if self._sem is None:
            self._sem = asyncio.Semaphore(MAX_CONCURRENT)
        return self._sem
