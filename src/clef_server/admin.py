"""Feature module: ``/v1/whoami``, ``/v1/admin/keys*`` and ``/v1/admin/audit`` (docs/ARCHITECTURE.md, Keys).

Everything under ``/v1/admin`` needs an ``admin`` key, even when auth is off (then: 401 "admin routes need a
key"). Writes reload the key cache before they answer, so a revoked key stops working at once. Secrets are
returned once (create / rotate) and never stored or logged; those responses are ``Cache-Control: no-store``.
"""

from __future__ import annotations

import time
from typing import Annotated, Any

from fastapi import APIRouter, Body, Query, Request, Response
from pydantic import BaseModel, ConfigDict, Field
from starlette.concurrency import run_in_threadpool

from .appctx import AppContext
from .keystore import (
    ALL_SCOPES,
    DEFAULT_GRACE_S,
    Actor,
    KeyStore,
    KeystoreError,
    Principal,
    parse_scopes,
)

DAY_S = 86400.0


class _Base(BaseModel):
    model_config = ConfigDict(extra="forbid")


class KeyInfo(BaseModel):
    id: str
    display: str = Field(description="clef_live_<id>_…")
    created_at: float
    expires_at: float | None
    revoked_at: float | None
    last_used_at: float | None = Field(description="approximate (flushed every 10 s)")
    status: str = Field(description="active | expiring | expired | revoked")


class PrincipalInfo(BaseModel):
    name: str
    source: str = Field(description="db | env")
    scopes: list[str]
    daily_quota: int | None
    rate_limit: int | None
    note: str | None
    created_at: float | None
    created_by: str | None
    disabled_at: float | None
    used_today: int
    keys: list[KeyInfo]


class PrincipalWithKey(PrincipalInfo):
    key: str = Field(description="the new secret: shown once, never stored (only its sha256 is)")


class PrincipalList(BaseModel):
    principals: list[PrincipalInfo]


class KeyCreate(_Base):
    name: str = Field(description="letters, digits, _ . -, max 64; never reused once created")
    scopes: list[str] = Field(description="classify, jobs, classifiers, ops, admin, urgent")
    daily_quota: int | None = Field(None, ge=1, description="records per UTC day; omit for unlimited")
    rate_limit: int | None = Field(None, ge=1, description="requests/min; omit for CLEF_RATE_LIMIT")
    expires_in_days: float | None = Field(None, gt=0, le=3650)
    note: str | None = Field(None, max_length=500)


class KeyUpdate(_Base):
    scopes: list[str] | None = None
    daily_quota: int | None = Field(None, ge=1, description="explicit null clears the quota")
    rate_limit: int | None = Field(None, ge=1, description="explicit null clears the rate limit")
    note: str | None = Field(None, max_length=500)


class RotateRequest(_Base):
    grace_s: float = Field(
        DEFAULT_GRACE_S, ge=0, le=365 * DAY_S, description="how long the old keys keep working"
    )
    expires_in_days: float | None = Field(None, gt=0, le=3650, description="expiry of the new key")


class RevokeRequest(_Base):
    key_id: str | None = Field(
        None, description="one key; omit to disable the principal and revoke all its keys"
    )


class UsageDay(BaseModel):
    day: str
    records: int
    requests: int
    rejected: int


class UsageInfo(BaseModel):
    name: str
    days: list[UsageDay]


class AuditEntry(BaseModel):
    id: int
    ts: float
    actor: str
    action: str
    target: str
    detail: Any | None
    ip: str | None
    request_id: str | None


class AuditPage(BaseModel):
    entries: list[AuditEntry]
    next_before: int | None


class WhoAmI(BaseModel):
    auth: bool
    name: str | None
    source: str | None
    scopes: list[str] | None
    daily_quota: int | None
    used_today: int | None
    key_id: str | None
    expires_at: float | None


def router(ctx: AppContext) -> APIRouter:
    ApiError = ctx.api_error  # noqa: N806
    store = KeyStore(ctx.db)
    quotas, cache = ctx.quotas, ctx.keys
    env_names = ctx.env_key_names
    admin = [*ctx.scope("admin"), *ctx.limited]
    api = APIRouter(prefix="/v1", tags=["admin"], responses=ctx.errors)

    def actor_of(request: Request) -> Actor:
        principal = ctx.principal(request)
        return Actor(
            principal.name if principal else "anonymous",
            request.client.host if request.client else None,
            ctx.request_id(request),
        )

    def view(row: dict[str, Any]) -> dict[str, Any]:
        return {**row, "used_today": quotas.used(row["name"])}

    def env_view(name: str) -> dict[str, Any]:
        return {
            "name": name,
            "source": "env",
            "scopes": sorted(ALL_SCOPES),
            "daily_quota": None,
            "rate_limit": None,
            "note": "CLEF_API_KEY / CLEF_API_KEYS (full admin, no quota; change it in the environment)",
            "created_at": None,
            "created_by": None,
            "disabled_at": None,
            "used_today": 0,
            "keys": [],
        }

    async def write(call: Any, *args: Any, **kw: Any) -> Any:
        """Run a store write in a thread, then reload the cache so it applies before we answer."""
        try:
            out = await run_in_threadpool(call, *args, **kw)
        except KeystoreError as exc:
            raise ApiError(exc.status, str(exc)) from exc
        await run_in_threadpool(cache.reload)
        return out

    def not_env(name: str) -> None:
        if name in env_names:
            raise ApiError(403, f"{name!r} is an env key (CLEF_API_KEY / CLEF_API_KEYS): change it there")

    def own(request: Request, name: str) -> Principal | None:
        principal = ctx.principal(request)
        return (
            principal
            if principal is not None and principal.source == "db" and principal.name == name
            else None
        )

    def no_secret_cache(response: Response) -> None:
        response.headers["Cache-Control"] = "no-store"

    @api.get(
        "/whoami",
        response_model=WhoAmI,
        summary="The calling key's principal, scopes and quota",
        dependencies=[*ctx.scope(), *ctx.limited],
    )
    async def whoami(request: Request) -> dict[str, Any]:
        p = ctx.principal(request)
        if p is None:
            return dict.fromkeys(
                ("name", "source", "scopes", "daily_quota", "used_today", "key_id", "expires_at")
            ) | {"auth": False}
        return {
            "auth": True,
            "name": p.name,
            "source": p.source,
            "scopes": sorted(p.scopes),
            "daily_quota": p.daily_quota,
            "used_today": quotas.used(p.name) if p.source == "db" else None,
            "key_id": p.key_id,
            "expires_at": p.expires_at,
        }

    @api.get(
        "/admin/keys",
        response_model=PrincipalList,
        summary="Principals and their keys (env principals read-only)",
        dependencies=admin,
    )
    async def list_keys(include_disabled: bool = False) -> dict[str, Any]:
        rows = await run_in_threadpool(store.list_principals, include_disabled)
        db_names = {r["name"] for r in rows}
        envs = [env_view(n) for n in sorted(env_names) if n not in db_names]
        return {"principals": [*envs, *map(view, rows)]}

    @api.post(
        "/admin/keys",
        status_code=201,
        response_model=PrincipalWithKey,
        summary="Create a principal and its first key (the secret is shown once)",
        dependencies=admin,
    )
    async def create_key(body: KeyCreate, request: Request, response: Response) -> dict[str, Any]:
        no_secret_cache(response)
        expires = time.time() + body.expires_in_days * DAY_S if body.expires_in_days else None
        row, secret = await write(
            store.create_principal,
            body.name,
            body.scopes,
            actor=actor_of(request),
            daily_quota=body.daily_quota,
            rate_limit=body.rate_limit,
            note=body.note,
            expires_at=expires,
            env_names=env_names,
        )
        return {**view(row), "key": secret}

    @api.patch(
        "/admin/keys/{name}",
        response_model=PrincipalInfo,
        summary="Change scopes, quota, rate limit or note (explicit null clears quota / rate limit)",
        dependencies=admin,
    )
    async def update_key(
        name: str, body: KeyUpdate, request: Request, force: bool = Query(False)
    ) -> dict[str, Any]:
        not_env(name)
        changes = {k: getattr(body, k) for k in body.model_fields_set}
        if "scopes" in changes and changes["scopes"] is None:
            raise ApiError(400, "scopes: cannot be null")
        if "scopes" in changes and own(request, name) and not force:
            try:
                lost_admin = "admin" not in parse_scopes(changes["scopes"])
            except KeystoreError as exc:
                raise ApiError(exc.status, str(exc)) from exc
            if lost_admin:
                raise ApiError(403, "this would remove your own admin scope; add ?force=true to do it anyway")
        row = await write(store.update_principal, name, changes, actor=actor_of(request))
        return view(row)

    @api.post(
        "/admin/keys/{name}/rotate",
        status_code=201,
        response_model=PrincipalWithKey,
        summary="Add a new key; the others keep working for grace_s (default 24 h)",
        dependencies=admin,
    )
    async def rotate_key(
        name: str, request: Request, response: Response, body: Annotated[RotateRequest | None, Body()] = None
    ) -> dict[str, Any]:
        not_env(name)
        no_secret_cache(response)
        body = body or RotateRequest()
        expires = time.time() + body.expires_in_days * DAY_S if body.expires_in_days else None
        row, secret = await write(
            store.rotate, name, actor=actor_of(request), grace_s=body.grace_s, expires_at=expires
        )
        return {**view(row), "key": secret}

    @api.post(
        "/admin/keys/{name}/revoke",
        response_model=PrincipalInfo,
        summary="Revoke one key, or disable the principal (names are never reused)",
        dependencies=admin,
    )
    async def revoke_key(
        name: str,
        request: Request,
        body: Annotated[RevokeRequest | None, Body()] = None,
        force: bool = Query(False),
    ) -> dict[str, Any]:
        not_env(name)
        key_id = body.key_id if body else None
        if own(request, name) and not force:
            current = await run_in_threadpool(store.get, name)
            live = [k["id"] for k in (current or {}).get("keys", []) if k["status"] in ("active", "expiring")]
            if key_id is None or not [k for k in live if k != key_id]:
                raise ApiError(
                    403,
                    "this would lock out your own principal (its last key); add ?force=true to do it anyway",
                )
        row = await write(store.revoke, name, actor=actor_of(request), key_id=key_id)
        return view(row)

    @api.get(
        "/admin/keys/{name}/usage",
        response_model=UsageInfo,
        summary="Daily records / requests / quota rejections, oldest first",
        dependencies=admin,
    )
    async def usage(name: str, days: int = Query(30, ge=1, le=366)) -> dict[str, Any]:
        if name not in env_names and await run_in_threadpool(store.get, name) is None:
            raise ApiError(404, f"principal {name!r} not found")
        return {"name": name, "days": await run_in_threadpool(quotas.history, name, days)}

    @api.get(
        "/admin/audit",
        response_model=AuditPage,
        summary="Audit log of admin actions (newest first)",
        dependencies=admin,
    )
    async def audit_log(
        limit: int = Query(100, ge=1, le=1000),
        before: int | None = Query(None, ge=0),
        action: str | None = Query(None, max_length=64, description="prefix, e.g. key."),
    ) -> dict[str, Any]:
        entries, nxt = await run_in_threadpool(store.audit_entries, limit, before, action)
        return {"entries": entries, "next_before": nxt}

    return api
