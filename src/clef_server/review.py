"""The review queue and the router's LLM-call log, in the database (``review`` and ``llm_calls``).

One table holds router proposals (``kind=proposal``) and items a human should look at (``kind=item``): router
items that failed the gate and classifier results below ``review_threshold``. Input text is stored only here
(D17) and only until the item is resolved or ``CLEF_REVIEW_TTL_DAYS`` pass; the sha256 stays for dedup and for
the router's re-checks. No SQL for these tables lives outside this module. See docs/ARCHITECTURE.md
"Review queue and router" and docs/router.md.
"""

from __future__ import annotations

import hashlib
import json
import secrets
import time
from typing import Any

from sqlalchemy import and_, case, func, select, update

from . import dbschema as t
from .db import Database

R = t.review
OPEN_ITEM = ("open", "buffered")


def new_id() -> str:
    return "rev_" + secrets.token_hex(10)


def canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def input_sha(value: Any) -> str:
    return hashlib.sha256(canonical(value).encode("utf-8")).hexdigest()


def _loads(raw: str | None) -> Any:
    return None if raw is None else json.loads(raw)


def public_row(row: Any) -> dict[str, Any]:
    """The API view of a review row: JSON columns parsed; ``input`` only while it is stored."""
    label: Any = row.label
    if row.kind == "proposal" and label is not None:
        label = json.loads(label)
    return {
        "id": row.id,
        "kind": row.kind,
        "state": row.state,
        "owner": row.owner,
        "router": row.router,
        "classifier": row.classifier,
        "classifier_version": row.classifier_version,
        "reason": row.reason,
        "input": _loads(row.input),
        "input_sha256": row.input_sha256,
        "prediction": _loads(row.prediction),
        "proposal": _loads(row.proposal),
        "label": label,
        "reroutes": row.reroutes,
        "created_at": round(row.created_at, 3),
        "resolved_at": None if row.resolved_at is None else round(row.resolved_at, 3),
        "resolved_by": row.resolved_by,
    }


class ReviewStore:
    def __init__(self, db: Database) -> None:
        self.db = db

    # ---- reads
    def get(self, review_id: str) -> dict[str, Any] | None:
        with self.db.read() as conn:
            row = conn.execute(select(R).where(R.c.id == review_id)).first()
        return None if row is None else public_row(row)

    def list(
        self,
        *,
        kind: str | None = None,
        state: str | None = None,
        router: str | None = None,
        classifier: str | None = None,
        owner: str | None = None,
        scoped: bool = False,
        limit: int = 50,
        offset: int = 0,
    ) -> tuple[list[dict[str, Any]], int]:
        """Newest first. ``scoped`` limits the rows to ``owner`` (auth on, caller not an admin)."""
        conds = [
            col == val
            for col, val in (
                (R.c.kind, kind),
                (R.c.state, state),
                (R.c.router, router),
                (R.c.classifier, classifier),
            )
            if val is not None
        ]
        if scoped:
            conds.append(R.c.owner == owner)
        where = and_(*conds) if conds else True
        with self.db.read() as conn:
            total = conn.execute(select(func.count()).select_from(R).where(where)).scalar_one()
            rows = conn.execute(
                select(R).where(where).order_by(R.c.created_at.desc(), R.c.id).limit(limit).offset(offset)
            ).all()
        return [public_row(r) for r in rows], int(total)

    def buffered(self, router: str, limit: int, with_input: bool = True) -> list[dict[str, Any]]:
        """The oldest buffered items of a router (those whose text is still stored when ``with_input``)."""
        cond = [R.c.router == router, R.c.kind == "item", R.c.state == "buffered"]
        if with_input:
            cond.append(R.c.input.is_not(None))
        with self.db.read() as conn:
            rows = conn.execute(select(R).where(*cond).order_by(R.c.created_at, R.c.id).limit(limit)).all()
        return [public_row(r) for r in rows]

    def by_ids(self, ids: list[str]) -> list[dict[str, Any]]:
        if not ids:
            return []
        with self.db.read() as conn:
            rows = conn.execute(select(R).where(R.c.id.in_(ids))).all()
        order = {i: n for n, i in enumerate(ids)}
        return sorted((public_row(r) for r in rows), key=lambda r: order[r["id"]])

    def counts(self, router: str) -> dict[str, int]:
        """``{"buffered": n, "open_reviews": n}`` (open = items and proposals waiting for a human)."""
        with self.db.read() as conn:
            rows = conn.execute(
                select(R.c.state, func.count()).where(R.c.router == router).group_by(R.c.state)
            ).all()
        by = {state: n for state, n in rows}
        return {"buffered": by.get("buffered", 0), "open_reviews": by.get("open", 0)}

    def oldest_buffered(self, router: str) -> float | None:
        with self.db.read() as conn:
            return conn.execute(
                select(func.min(R.c.created_at)).where(
                    R.c.router == router, R.c.kind == "item", R.c.state == "buffered"
                )
            ).scalar()

    def human_labels(self, router: str, shas: list[str]) -> dict[str, str]:
        """sha256 -> the label a human gave that exact input (resolved items of this router)."""
        if not shas:
            return {}
        with self.db.read() as conn:
            rows = conn.execute(
                select(R.c.input_sha256, R.c.label).where(
                    R.c.router == router,
                    R.c.kind == "item",
                    R.c.state == "resolved",
                    R.c.resolved_by.not_like("router:%"),
                    R.c.resolved_by != "auto",
                    R.c.input_sha256.in_(shas),
                    R.c.label.is_not(None),
                )
            ).all()
        return {sha: label for sha, label in rows}

    # ---- writes
    def add_items(self, rows: list[dict[str, Any]]) -> list[tuple[str, bool]]:
        """Insert items; one ``(id, created)`` per row. A row whose input is already open / buffered for the
        same router (or classifier) is not inserted again: its existing id comes back with ``created=False``.

        Each row: owner, router, classifier, classifier_version, reason, state, input (JSON value),
        prediction."""
        out: list[tuple[str, bool]] = []
        now = time.time()
        with self.db.tx() as conn:
            for r in rows:
                sha = input_sha(r["input"])
                same = [R.c.input_sha256 == sha, R.c.kind == "item", R.c.state.in_(OPEN_ITEM)]
                same.append(R.c.router == r["router"] if r.get("router") else R.c.router.is_(None))
                if r.get("classifier") and not r.get("router"):
                    same.append(R.c.classifier == r["classifier"])
                found = conn.execute(select(R.c.id).where(*same)).first()
                if found is not None:
                    out.append((found[0], False))
                    continue
                rid = new_id()
                conn.execute(
                    R.insert().values(
                        id=rid,
                        kind="item",
                        state=r.get("state", "open"),
                        owner=r.get("owner"),
                        router=r.get("router"),
                        classifier=r.get("classifier"),
                        classifier_version=r.get("classifier_version"),
                        reason=r["reason"],
                        input=canonical(r["input"]),
                        input_sha256=sha,
                        prediction=None if r.get("prediction") is None else canonical(r["prediction"]),
                        created_at=now,
                    )
                )
                out.append((rid, True))
        return out

    def add_proposal(
        self,
        *,
        state: str,
        owner: str | None,
        router: str,
        classifier: str,
        classifier_version: int | None,
        reason: str,
        proposal: dict[str, Any],
    ) -> str:
        rid = new_id()
        with self.db.tx() as conn:
            conn.execute(
                R.insert().values(
                    id=rid,
                    kind="proposal",
                    state=state,
                    owner=owner,
                    router=router,
                    classifier=classifier,
                    classifier_version=classifier_version,
                    reason=reason,
                    proposal=canonical(proposal),
                    created_at=time.time(),
                )
            )
        return rid

    def set_state(
        self,
        ids: list[str],
        state: str,
        *,
        purge: bool = False,
        by: str | None = None,
        only_from: tuple[str, ...] | None = None,
    ) -> None:
        """Move rows to ``state`` (only rows now in ``only_from``, when given); ``purge`` drops the stored
        text; a resolving state stamps who and when."""
        if not ids:
            return
        values: dict[str, Any] = {"state": state}
        if purge:
            values["input"] = None
        if state in ("resolved", "discarded", "accepted", "rejected", "permanent_other"):
            values.update(resolved_at=time.time(), resolved_by=by)
        with self.db.tx() as conn:
            stmt = update(R).where(R.c.id.in_(ids))
            if only_from is not None:
                stmt = stmt.where(R.c.state.in_(only_from))
            conn.execute(stmt.values(**values))

    def bump_attempts(self, ids: list[str], max_attempts: int = 2) -> None:
        """The items were sent through a check and are still unplaced: reroutes += 1; at ``max_attempts`` they
        become ``permanent_other`` (out of future batches) and ``buffered`` otherwise."""
        if not ids:
            return
        with self.db.tx() as conn:
            conn.execute(
                update(R)
                .where(R.c.id.in_(ids), R.c.state.in_(OPEN_ITEM))
                .values(
                    reroutes=R.c.reroutes + 1,
                    state=case((R.c.reroutes + 1 >= max_attempts, "permanent_other"), else_="buffered"),
                )
            )

    def resolve_many(self, rows: list[tuple[str, str, dict[str, Any]]], by: str) -> None:
        """Resolve buffered items with a label: ``(id, label, prediction)``; the text is dropped (D17)."""
        if not rows:
            return
        now = time.time()
        with self.db.tx() as conn:
            for rid, label, prediction in rows:
                conn.execute(
                    update(R)
                    .where(R.c.id == rid, R.c.state.in_(OPEN_ITEM))
                    .values(
                        state="resolved",
                        label=label,
                        input=None,
                        prediction=canonical(prediction),
                        resolved_at=now,
                        resolved_by=by,
                    )
                )

    def update_row(self, review_id: str, **fields: Any) -> None:
        """Set columns on one row; dict values for JSON columns are serialised."""
        for key in ("prediction", "proposal"):
            if isinstance(fields.get(key), (dict, list)):
                fields[key] = canonical(fields[key])
        if isinstance(fields.get("label"), (list, dict)):
            fields["label"] = canonical(fields["label"])
        with self.db.tx() as conn:
            conn.execute(update(R).where(R.c.id == review_id).values(**fields))

    def claim(self, review_id: str, from_states: tuple[str, ...], to_state: str) -> bool:
        """Atomically move one row between states; False when it was not in ``from_states`` (a concurrent
        resolve won). The guard behind every resolve, so a row is resolved once."""
        with self.db.tx() as conn:
            res = conn.execute(
                update(R).where(R.c.id == review_id, R.c.state.in_(from_states)).values(state=to_state)
            )
            return res.rowcount == 1

    def purge(self, ttl_days: float, now: float | None = None) -> int:
        """Drop the text of rows older than ``ttl_days`` (open / buffered items become ``expired``).
        Resolved rows keep their label, sha256 and prediction. Returns the number of rows touched."""
        if ttl_days <= 0:
            return 0
        cutoff = (time.time() if now is None else now) - ttl_days * 86400
        with self.db.tx() as conn:
            expired = conn.execute(
                update(R)
                .where(R.c.kind == "item", R.c.state.in_(OPEN_ITEM), R.c.created_at < cutoff)
                .values(state="expired")
            ).rowcount
            stale = conn.execute(
                update(R).where(R.c.input.is_not(None), R.c.created_at < cutoff).values(input=None)
            ).rowcount
        return max(int(expired or 0), int(stale or 0))

    def drop_router(self, router: str, by: str | None = None) -> None:
        """A router was deleted: its items lose their text at once and its open proposals are rejected."""
        now = time.time()
        with self.db.tx() as conn:
            conn.execute(
                update(R)
                .where(R.c.router == router, R.c.kind == "item", R.c.state.in_(OPEN_ITEM))
                .values(state="discarded", input=None, resolved_at=now, resolved_by=by)
            )
            conn.execute(
                update(R)
                .where(R.c.router == router, R.c.kind == "proposal", R.c.state == "open")
                .values(state="rejected", resolved_at=now, resolved_by=by)
            )
            conn.execute(update(R).where(R.c.router == router, R.c.input.is_not(None)).values(input=None))

    # ---- llm_calls
    def log_call(
        self,
        *,
        router: str | None,
        purpose: str,
        provider_url: str,
        model: str,
        status: str,
        latency_ms: float | None,
        input_tokens: int | None,
        output_tokens: int | None,
        items: int,
        error: str | None = None,
    ) -> int:
        with self.db.tx() as conn:
            res = conn.execute(
                t.llm_calls.insert().values(
                    ts=time.time(),
                    router=router,
                    purpose=purpose,
                    provider_url=provider_url,
                    model=model,
                    status=status,
                    latency_ms=latency_ms,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    items=items,
                    error=None if error is None else error[:500],
                )
            )
            return int(res.inserted_primary_key[0])

    def llm_usage(self, since: float, router: str | None = None) -> dict[str, int]:
        """Calls and tokens since ``since`` (all routers, or one)."""
        c = t.llm_calls.c
        cond = [c.ts >= since] + ([c.router == router] if router else [])
        with self.db.read() as conn:
            row = conn.execute(
                select(
                    func.count(),
                    func.coalesce(func.sum(c.input_tokens), 0),
                    func.coalesce(func.sum(c.output_tokens), 0),
                    func.coalesce(func.sum(case((c.status == "error", 1), else_=0)), 0),
                ).where(*cond)
            ).one()
        return {
            "calls": int(row[0]),
            "input_tokens": int(row[1]),
            "output_tokens": int(row[2]),
            "errors": int(row[3]),
        }


__all__ = ["ReviewStore", "canonical", "input_sha", "new_id", "public_row"]
