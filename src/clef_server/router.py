"""The self-evolving category router and the Review routes (docs/ARCHITECTURE.md "Review queue and router").

A router is its own resource that grows the label set of one single-label saved classifier. Items it cannot
place ("other", or below a threshold that rises with the label count) are buffered. Every
``CLEF_ROUTER_BATCH`` of them go to an LLM (``llm.py``) that proposes new labels. clef re-classifies the
buffered items against the proposed set, and only a proposal that clef itself confirms is shown to a human
(review mode) or published as a new classifier version (auto mode). Labels are only ever added. The router is
OFF unless an LLM key is set.

This file has three parts: pure logic (no FastAPI, no database; unit-tested with a fake ``decide``), the
``RouterStore`` / ``RouterService`` that apply it, and the HTTP routes at the bottom.
"""

from __future__ import annotations

import asyncio
import functools
import json
import logging
import math
import re
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Literal
from urllib.parse import urlsplit

from fastapi import APIRouter, Query, Request
from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy import delete, func, insert, select, update
from sqlalchemy.exc import IntegrityError

from . import dbschema as t
from .appctx import AppContext
from .classifiers import NAME_RE, InvalidName, TooManyClassifiers, validate_name
from .classify import ClassifyResult, classify_result, classify_to_systemone
from .db import Database, audit, is_unavailable
from .llm import LLMError, client_from_config
from .review import ReviewStore

log = logging.getLogger("clef")
OPEN_STATES = ("open", "buffered")

OTHER = "other"
DEFAULT_THRESHOLDS: tuple[tuple[int, float], ...] = ((10, 0.5), (30, 0.7), (64, 0.85))
DEFAULT_MARGIN = 0.15
MAX_NEW_PER_CALL = 3  # labels one propose call may add
MAX_BOOTSTRAP_LABELS = 8
MAX_DESCRIPTION_WORDS = 15
HARD_MAX_LABELS = 63  # plus "other" = 64 options (D8)
MAX_ATTEMPTS = 2  # an item that is still unplaced after two checks becomes permanent_other
ITEM_REASONS = ("other", "low_confidence", "ambiguous", "cold")

# ------------------------------------------------------------------ gate (pure)


def normalise_thresholds(raw: Any) -> tuple[tuple[int, float], ...]:
    """``[[10, 0.5], [30, 0.7], [64, 0.85]]`` -> sorted tuples; raises ValueError on a bad shape."""
    try:
        pairs = tuple((int(n), float(p)) for n, p in raw)
    except (TypeError, ValueError) as exc:
        raise ValueError("gate.thresholds must be a list of [max_options, min_probability] pairs") from exc
    if not pairs or any(n < 2 or not 0.0 < p <= 1.0 for n, p in pairs):
        raise ValueError("gate.thresholds: max_options >= 2 and 0 < min_probability <= 1")
    pairs = tuple(sorted(pairs))
    if any(b[1] < a[1] for a, b in zip(pairs, pairs[1:], strict=False)):
        raise ValueError("gate.thresholds must not fall as the label count grows")
    return pairs


def threshold_for(n: int, thresholds: tuple[tuple[int, float], ...] = DEFAULT_THRESHOLDS) -> float:
    """tau(n): the minimum probability of the top label with ``n`` options (``other`` included). It rises with
    the label count because probability mass spreads over more options; past the last step the last value."""
    for upto, tau in thresholds:
        if n <= upto:
            return tau
    return thresholds[-1][1]


def gate(
    scores: dict[str, float],
    thresholds: tuple[tuple[int, float], ...] = DEFAULT_THRESHOLDS,
    margin: float = DEFAULT_MARGIN,
) -> str:
    """``classified | other | low_confidence | ambiguous`` for one result's ``label -> probability``.

    other and low_confidence are buffered for the LLM; ambiguous (top two within ``margin``) goes to a human
    only."""
    ranked = sorted(scores.items(), key=lambda kv: -kv[1])
    top, p = ranked[0]
    if top == OTHER:
        return "other"
    if p < threshold_for(len(scores), thresholds):
        return "low_confidence"
    if len(ranked) > 1 and p - ranked[1][1] < margin:
        return "ambiguous"
    return "classified"


def top_label(scores: dict[str, float]) -> str:
    return max(scores.items(), key=lambda kv: kv[1])[0]


# ------------------------------------------------------------------ label sets (pure)


def other_description(names: list[str]) -> str:
    """The "other" description: the real labels in negative form (up to 15), else a generic sentence."""
    if not names or len(names) > 15:
        return "The input fits none of the other labels."
    return "The input fits none of the other labels: " + ", ".join(f"not {n}" for n in names) + "."


def build_label_set(real: dict[str, str]) -> dict[str, str]:
    """``{name: description}`` of the real labels plus a freshly described ``other`` (always last)."""
    base = {k: v for k, v in real.items() if k.lower() != OTHER}
    return {**base, OTHER: other_description(list(base))}


def real_labels(labels: dict[str, str]) -> dict[str, str]:
    return {k: v for k, v in labels.items() if k != OTHER}


def labels_of(definition_labels: Any) -> dict[str, str]:
    """A classifier's ``labels`` (list or object) as ``{name: description}``."""
    if isinstance(definition_labels, dict):
        return {str(k): str(v) for k, v in definition_labels.items()}
    return {str(k): str(k) for k in definition_labels}


def capacity(max_labels: int, server_max_labels: int) -> int:
    """How many real labels a router may hold: its own cap, at most the server's ``CLEF_MAX_LABELS`` minus the
    ``other`` option, at most 63."""
    return max(0, min(max_labels, server_max_labels - 1, HARD_MAX_LABELS))


def _stem(token: str) -> str:
    return token[:-1] if len(token) > 3 and token.endswith("s") else token


def lexical_key(name: str) -> str:
    """Case, separators and a plural "s" make no new label: "Refund_Requests" == "refund request"."""
    return " ".join(_stem(tok) for tok in re.sub(r"[^a-z0-9]+", " ", name.lower()).split())


# ------------------------------------------------------------------ LLM proposals (pure)


@dataclass
class Candidate:
    name: str
    description: str
    member_indices: list[int]
    nearest_existing: str = ""
    rationale: str = ""


@dataclass
class Parsed:
    candidates: list[Candidate] = field(default_factory=list)
    covered: dict[int, str] = field(default_factory=dict)  # item index -> existing label the LLM says fits
    unclear: list[int] = field(default_factory=list)
    dropped: list[str] = field(default_factory=list)  # why a proposed label was not kept (for the log)


def _word_count(text: str) -> int:
    return len(text.split())


def check_label_name(name: Any, existing: list[str]) -> str | None:
    """Why ``name`` cannot be added next to ``existing``, or None."""
    if not isinstance(name, str) or not NAME_RE.fullmatch(name):
        return f"{name!r} is not a valid label name (a-z, 0-9, _ and -, at most 64 characters)"
    if name == OTHER:
        return f"{OTHER!r} is reserved"
    keys = {lexical_key(e) for e in existing}
    if lexical_key(name) in keys:
        return f"{name!r} duplicates an existing label"
    return None


def parse_proposals(
    data: dict[str, Any], existing: list[str], n_items: int, room: int, max_new: int
) -> Parsed:
    """The LLM's JSON -> candidates that obey the rules; everything else is dropped with a reason.

    Rules: valid, new, non-duplicate name; description of at most 15 words; at least 2 members, all indices in
    range; not flagged ``duplicate_of`` an existing label; at most ``min(max_new, room)`` labels."""
    out = Parsed()
    taken = list(existing)
    for raw in data.get("proposals", []):
        name = raw.get("name")
        problem = check_label_name(name, taken)
        desc = str(raw.get("description", "")).strip()
        members = [i for i in raw.get("member_indices", []) if isinstance(i, int) and 0 <= i < n_items]
        members = list(dict.fromkeys(members))
        if problem is None and not desc:
            problem = f"{name!r} has no description"
        elif problem is None and _word_count(desc) > MAX_DESCRIPTION_WORDS:
            problem = f"{name!r} description is longer than {MAX_DESCRIPTION_WORDS} words"
        elif problem is None and len(members) < 2:
            problem = f"{name!r} has fewer than 2 valid members"
        elif problem is None and str(raw.get("duplicate_of", "")).strip():
            problem = f"{name!r} was flagged as a duplicate of {raw['duplicate_of']!r}"
        elif problem is None and len(out.candidates) >= min(max_new, room):
            problem = f"{name!r} exceeds the room for new labels"
        if problem:
            out.dropped.append(problem)
            continue
        taken.append(name)
        out.candidates.append(
            Candidate(
                name,
                desc,
                members,
                str(raw.get("nearest_existing", "")).strip(),
                str(raw.get("rationale", "")).strip(),
            )
        )
    for raw in data.get("covered_by", []):
        idx, label = raw.get("index"), raw.get("label")
        if isinstance(idx, int) and 0 <= idx < n_items and label in existing:
            out.covered[idx] = label
    out.unclear = [i for i in data.get("unclear_indices", []) if isinstance(i, int) and 0 <= i < n_items]
    return out


def proposal_schema() -> dict[str, Any]:
    """Response schema for propose and bootstrap. Only type / properties / required / items /
    additionalProperties: the subset every OpenAI-compatible server accepts under strict mode. How many
    labels, names, word limits and index bounds are enforced in code (``parse_proposals``), so a model that
    ignores the prompt loses a proposal, not the whole call."""
    label = {
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "description": {"type": "string"},
            "member_indices": {"type": "array", "items": {"type": "integer"}},
            "nearest_existing": {"type": "string"},
            "duplicate_of": {"type": "string"},
            "rationale": {"type": "string"},
        },
        "required": [
            "name",
            "description",
            "member_indices",
            "nearest_existing",
            "duplicate_of",
            "rationale",
        ],
        "additionalProperties": False,
    }
    covered = {
        "type": "object",
        "properties": {"index": {"type": "integer"}, "label": {"type": "string"}},
        "required": ["index", "label"],
        "additionalProperties": False,
    }
    return {
        "type": "object",
        "properties": {
            "proposals": {"type": "array", "items": label},
            "covered_by": {"type": "array", "items": covered},
            "unclear_indices": {"type": "array", "items": {"type": "integer"}},
        },
        "required": ["proposals", "covered_by", "unclear_indices"],
        "additionalProperties": False,
    }


SYSTEM_PROMPT = (
    "You maintain the label set of a text classifier. You are shown numbered texts the classifier could not "
    "place under its current labels. Group them into NEW labels that will be reused on future texts.\n"
    "Rules:\n"
    "- A label name is lowercase a-z, 0-9, _ or -, at most 40 characters, a category and not a sentence.\n"
    "- Each description has at most 15 words and says what belongs in the label.\n"
    "- Never repeat or reword a current label. If a text already fits one, list it in covered_by.\n"
    "- A new label needs at least 2 of the texts as members (member_indices, the numbers shown).\n"
    "- Texts that fit nothing and no group go in unclear_indices.\n"
    "- Return JSON only, in the requested schema. Do not copy the texts."
)


def _clip(value: Any, limit: int) -> str:
    text = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)
    text = " ".join(text.split())
    return text if len(text) <= limit else text[: limit - 1] + "…"


def build_prompt(
    *,
    current: dict[str, str],
    items: list[Any],
    max_new: int,
    max_chars: int,
    domain_hint: str | None = None,
    instructions: str | None = None,
    bootstrap: bool = False,
) -> tuple[str, str]:
    """``(system, user)`` for a propose call, or for the cold-start bootstrap (3-8 starter labels)."""
    lines = []
    if domain_hint:
        lines.append(f"Domain: {domain_hint.strip()}")
    if instructions:
        lines.append(f"What the classifier decides: {instructions.strip()}")
    if bootstrap:
        lines.append(
            "There are no labels yet. Propose a starter set of 3 to "
            f"{max_new} labels that together cover most of the texts below."
        )
    else:
        lines.append("Current labels (do not duplicate or reword these):")
        lines += [f"- {name}: {desc}" for name, desc in current.items()]
        lines.append(f"Propose at most {max_new} new labels (fewer is fine; none is fine).")
    lines.append("Texts:")
    lines += [f"[{i}] {_clip(x, max_chars)}" for i, x in enumerate(items)]
    return SYSTEM_PROMPT, "\n".join(lines)


# ------------------------------------------------------------------ acceptance test (pure)


def evaluate_proposal(
    name: str,
    member_scores: list[dict[str, float]],
    thresholds: tuple[tuple[int, float], ...] = DEFAULT_THRESHOLDS,
    human_against: int = 0,
) -> dict[str, Any]:
    """Did clef, given the candidate set, put the proposal's members on the new label?

    ``member_scores`` are the members re-classified under current + proposal + other. It passes when at least
    ``max(2, 50 %)`` of the members land on ``name`` with probability >= tau(n'). A member a human already
    labelled with an existing label counts against (``human_against``, not re-classified). It fails as a
    duplicate when most members land on one existing label."""
    n_options = len(member_scores[0]) if member_scores else 0
    tau = threshold_for(n_options, thresholds)
    total = len(member_scores) + human_against
    on_new = sum(1 for s in member_scores if top_label(s) == name and s[name] >= tau)
    need = max(2, math.ceil(total * 0.5))
    landed: dict[str, int] = {}
    for s in member_scores:
        top = top_label(s)
        if top not in (name, OTHER):
            landed[top] = landed.get(top, 0) + 1
    dup, dup_n = max(landed.items(), key=lambda kv: kv[1], default=(None, 0))
    duplicate_of = dup if dup is not None and (dup_n + human_against) * 2 > total else None
    mean = sum(s[name] for s in member_scores) / len(member_scores) if member_scores else 0.0
    return {
        "members": total,
        "on_new": on_new,
        "need": need,
        "tau": tau,
        "n_options": n_options,
        "mean_probability": round(mean, 4),
        "human_against": human_against,
        "duplicate_of": duplicate_of,
        "passed": on_new >= need and duplicate_of is None,
    }


def evaluate_bootstrap(
    names: list[str],
    item_scores: list[dict[str, float]],
    thresholds: tuple[tuple[int, float], ...] = DEFAULT_THRESHOLDS,
) -> dict[str, Any]:
    """Cold start: the starter set is checked on all the buffered items. A label is kept when at least 2 items
    land on it with p >= tau; the set passes when the kept labels cover at least half of the items.

    Returns ``{"kept": {name: [item indices]}, "covered", "total", "tau", "passed"}``."""
    n_options = len(item_scores[0]) if item_scores else 0
    tau = threshold_for(n_options, thresholds)
    landed: dict[str, list[int]] = {n: [] for n in names}
    for i, s in enumerate(item_scores):
        top = top_label(s)
        if top in landed and s[top] >= tau:
            landed[top].append(i)
    kept = {n: idx for n, idx in landed.items() if len(idx) >= 2}
    covered = sum(len(idx) for idx in kept.values())
    total = len(item_scores)
    return {
        "kept": kept,
        "covered": covered,
        "total": total,
        "tau": tau,
        "n_options": n_options,
        "passed": bool(kept) and covered * 2 >= total,
    }


# ------------------------------------------------------------------ storage (routers + label sets)

_RT, _LS = t.routers, t.router_label_sets
BACKOFF_S = 300.0  # after a failed LLM call a router waits this long before the flush loop tries again
PURGE_EVERY_S = 3600.0
TICK_S = 15.0
MIN_AGE_FLUSH = 2  # an age-triggered flush needs at least this many buffered items (5 for a cold start)
MIN_AGE_FLUSH_COLD = 5
DAY_S = 86400.0


class RouterError(Exception):
    """A rule of the router was broken; ``status`` is the HTTP code the routes answer with."""

    def __init__(self, status: int, detail: str) -> None:
        super().__init__(detail)
        self.status = status


def _router_view(row: Any) -> dict[str, Any]:
    return {
        "name": row.name,
        "owner": row.owner,
        "classifier": row.classifier,
        "mode": row.mode,
        "config": json.loads(row.config),
        "active_version": row.active_version,
        "created_at": round(row.created_at, 3),
        "updated_at": round(row.updated_at, 3),
    }


def _label_set_view(row: Any) -> dict[str, Any]:
    return {
        "version": row.version,
        "labels": json.loads(row.labels),
        "source": row.source,
        "classifier_version": row.classifier_version,
        "created_at": round(row.created_at, 3),
        "created_by": row.created_by,
    }


class RouterStore:
    """``routers`` and ``router_label_sets``. Writes that also touch the classifiers live in the service."""

    def __init__(self, db: Database) -> None:
        self.db = db

    def get(self, name: str) -> dict[str, Any] | None:
        with self.db.read() as conn:
            row = conn.execute(select(_RT).where(_RT.c.name == name)).first()
        return None if row is None else _router_view(row)

    def list(self, owner: str | None = None, scoped: bool = False) -> list[dict[str, Any]]:
        stmt = select(_RT).order_by(_RT.c.name)
        if scoped:
            stmt = stmt.where(_RT.c.owner == owner)
        with self.db.read() as conn:
            return [_router_view(r) for r in conn.execute(stmt).all()]

    def by_classifier(self, classifier: str) -> dict[str, Any] | None:
        with self.db.read() as conn:
            row = conn.execute(select(_RT).where(_RT.c.classifier == classifier)).first()
        return None if row is None else _router_view(row)

    def label_sets(self, name: str) -> list[dict[str, Any]]:
        """Every label-set version, newest first."""
        with self.db.read() as conn:
            rows = conn.execute(select(_LS).where(_LS.c.router == name).order_by(_LS.c.version.desc())).all()
        return [_label_set_view(r) for r in rows]

    def label_set(self, name: str, version: int) -> dict[str, Any] | None:
        with self.db.read() as conn:
            row = conn.execute(select(_LS).where(_LS.c.router == name, _LS.c.version == version)).first()
        return None if row is None else _label_set_view(row)

    def active(self, name: str) -> dict[str, Any] | None:
        """The active label set (``version``, ``labels`` with ``other``, ``classifier_version``), or None
        while the router is cold."""
        with self.db.read() as conn:
            version = conn.execute(select(_RT.c.active_version).where(_RT.c.name == name)).scalar()
            if version is None:
                return None
            row = conn.execute(select(_LS).where(_LS.c.router == name, _LS.c.version == version)).first()
        return None if row is None else _label_set_view(row)

    def new_labels_since(self, name: str, since: float) -> int:
        """Labels added by proposals since ``since`` (the daily cap of auto mode counts these)."""
        with self.db.read() as conn:
            rows = conn.execute(
                select(_LS.c.version, _LS.c.labels, _LS.c.source, _LS.c.created_at)
                .where(_LS.c.router == name)
                .order_by(_LS.c.version)
            ).all()
        added, prev = 0, 0
        for r in rows:
            n = len(json.loads(r.labels)) - 1  # without "other"
            if r.source == "proposal" and r.created_at >= since:
                added += max(0, n - prev)
            prev = n
        return added

    def delete(self, name: str) -> bool:
        with self.db.tx() as conn:
            conn.execute(delete(_LS).where(_LS.c.router == name))
            return conn.execute(delete(_RT).where(_RT.c.name == name)).rowcount == 1


# ------------------------------------------------------------------ the service


class RouterService:
    """Routing, buffering, the LLM round trip, acceptance, publishing and the Review actions."""

    def __init__(self, ctx: AppContext) -> None:
        self.ctx, self.cfg, self.db = ctx, ctx.cfg, ctx.db
        self.routers = RouterStore(ctx.db)
        self.reviews = ReviewStore(ctx.db)
        self.now: Callable[[], float] = time.time  # tests move the clock
        self.tick_s = TICK_S
        self._locks: dict[str, asyncio.Lock] = {}  # one LLM call in flight per router
        self._global = asyncio.Lock()  # ... and one overall
        self._backoff: dict[str, float] = {}
        self._kicked: set[str] = set()
        self._tasks: set[asyncio.Task[Any]] = set()
        self._last_purge = 0.0
        self._loop_task: asyncio.Task[Any] | None = None

    @property
    def llm(self) -> Any:
        return self.ctx.extra.get("llm_client")

    # ---- helpers
    @staticmethod
    def thresholds(rt: dict[str, Any]) -> tuple[tuple[int, float], ...]:
        raw = (rt["config"].get("gate") or {}).get("thresholds")
        return normalise_thresholds(raw) if raw else DEFAULT_THRESHOLDS

    @staticmethod
    def margin(rt: dict[str, Any]) -> float:
        return float((rt["config"].get("gate") or {}).get("margin", DEFAULT_MARGIN))

    def room(self, rt: dict[str, Any], labels: dict[str, str]) -> int:
        return capacity(int(rt["config"].get("max_labels", HARD_MAX_LABELS)), self.cfg.max_labels) - len(
            real_labels(labels)
        )

    async def classify(
        self, labels: dict[str, str], instructions: str | None, inputs: list[Any]
    ) -> list[dict[str, float]]:
        """``label -> probability`` for each input under ``labels``, through the background lane."""
        out: list[dict[str, float]] = []
        step = max(1, self.cfg.max_batch)
        for i in range(0, len(inputs), step):
            reqs = [classify_to_systemone(x, labels, instructions) for x in inputs[i : i + step]]
            results = await self.ctx.decide(reqs, batch=True, label="inputs", priority="background")
            out += [classify_result(r["answers"], labels, False)["scores"] for r in results]
        return out

    # ---- publishing
    def activate_sync(
        self,
        name: str,
        added: dict[str, str],
        source: str,
        by: str,
        *,
        replace: dict[str, str] | None = None,
        actor: str | None = None,
        ip: str | None = None,
        request_id: str | None = None,
    ) -> dict[str, Any]:
        """Publish a new label set as a new classifier version, atomically. Grow-only: ``added`` labels must
        be new (``replace`` is for seeds and rollback). Raises ``RouterError`` (400 cap / empty, 409 dup)."""
        with self.db.tx() as conn:
            self.db.serialize(conn, f"router:{name}")
            row = conn.execute(select(_RT).where(_RT.c.name == name)).first()
            if row is None:
                raise RouterError(404, f"router {name!r} not found")
            cfgd = json.loads(row.config)
            cur = self.routers.active(name)
            current = real_labels(cur["labels"]) if cur else {}
            if replace is not None:
                new_real = dict(replace)
            else:
                for label in added:
                    problem = check_label_name(label, list(current))
                    if problem:
                        raise RouterError(409, problem)
                new_real = {**current, **added}
            cap = capacity(int(cfgd.get("max_labels", HARD_MAX_LABELS)), self.cfg.max_labels)
            if not new_real:
                raise RouterError(400, "a router needs at least one label")
            if len(new_real) > cap:
                raise RouterError(400, f"label cap reached ({cap} labels plus {OTHER!r})")
            base = self.ctx.store.get(row.classifier) or {}
            definition = {
                "kind": "classify",
                "labels": build_label_set(new_real),
                "instructions": cfgd.get("instructions") or base.get("instructions"),
                "multi_label": False,
                "threshold": None,
                "description": base.get("description") or f"Maintained by router {name}",
            }
            if base.get("review_threshold") is not None:
                definition["review_threshold"] = base["review_threshold"]
            doc = self.ctx.store.put(
                row.classifier,
                {**definition, "levels": None},
                owner=row.owner,
                created_by=f"router:{name}",
                model=self.ctx.model_ref(),
                actor=actor or by,
                ip=ip,
                request_id=request_id,
            )
            version = (
                conn.execute(select(func.max(_LS.c.version)).where(_LS.c.router == name)).scalar() or 0
            ) + 1
            now = self.now()
            conn.execute(
                insert(_LS).values(
                    router=name,
                    version=version,
                    labels=json.dumps(definition["labels"], ensure_ascii=False),
                    source=source,
                    classifier_version=doc["version"],
                    created_at=now,
                    created_by=by,
                )
            )
            conn.execute(update(_RT).where(_RT.c.name == name).values(active_version=version, updated_at=now))
            audit(
                conn,
                actor or by,
                "router.label_set",
                name,
                {"version": version, "source": source, "added": sorted(added)},
                ip=ip,
                request_id=request_id,
            )
            return {"version": version, "classifier_version": doc["version"]}

    # ---- routing
    async def route(
        self, request: Request, rt: dict[str, Any], inputs: list[Any], batch: bool
    ) -> dict[str, Any]:
        """Classify under the active label set (the request path: logged, counted, quota), gate every result,
        buffer what the router cannot place. A cold router answers ``other`` without a forward."""
        t0 = time.perf_counter()
        name, ctx = rt["name"], self.ctx
        act = await asyncio.to_thread(self.routers.active, name)
        if act is None:
            model = ctx.check_model(None)
            bodies = [
                {
                    "model": model,
                    "multi_label": False,
                    "label": OTHER,
                    "confidence": 0.0,
                    "scores": {},
                    "usage": {"input_tokens": 0, "output_tokens": 0},
                    "timing": {},
                }
                for _ in inputs
            ]
            reasons = ["cold"] * len(inputs)
        else:
            try:
                if len(inputs) > self.cfg.max_batch:
                    raise ValueError(f"inputs: too many inputs (max {self.cfg.max_batch})")
                instr = rt["config"].get("instructions")
                reqs = [classify_to_systemone(x, act["labels"], instr) for x in inputs]
                results = await ctx.infer(request, reqs, batch=batch, label="inputs" if batch else "batch")
            except Exception as exc:
                raise ctx.map_exception(exc) from exc
            ms = (time.perf_counter() - t0) * 1000
            bodies = []
            for res in results:
                body = classify_result(res["answers"], act["labels"], False, None, res.get("model"))
                timing = {**res.get("timing", {}), "total_ms": round(ms, 1)}
                bodies.append({**body, "usage": res.get("usage", {}), "timing": timing})
            thr, margin = self.thresholds(rt), self.margin(rt)
            reasons = [gate(b["scores"], thr, margin) for b in bodies]
        rows, slots = [], []
        for i, (x, body, reason) in enumerate(zip(inputs, bodies, reasons, strict=True)):
            if reason == "classified":
                continue
            rows.append(
                {
                    "owner": rt["owner"],
                    "router": name,
                    "classifier": rt["classifier"],
                    "classifier_version": act["classifier_version"] if act else None,
                    "reason": reason,
                    "state": "open" if reason == "ambiguous" else "buffered",
                    "input": x,
                    "prediction": {
                        "label": body["label"],
                        "confidence": body["confidence"],
                        "scores": body["scores"],
                        "label_set_version": act["version"] if act else None,
                    },
                }
            )
            slots.append(i)
        ids = await asyncio.to_thread(self.reviews.add_items, rows) if rows else []
        out = []
        for i, (body, reason) in enumerate(zip(bodies, reasons, strict=True)):
            review_id = ids[slots.index(i)][0] if i in slots else None
            out.append(
                {
                    **body,
                    "router": name,
                    "reason": reason,
                    "label_set_version": act["version"] if act else None,
                    "review_id": review_id,
                }
            )
        if rows:
            self.maybe_kick(name, cold=act is None)
        return {"results": out, "batch_ms": round((time.perf_counter() - t0) * 1000, 1)}

    # ---- flushing: the LLM round trip
    def maybe_kick(self, name: str, cold: bool) -> None:
        """Start a flush in the background when the buffer reached the batch size (no-op without an LLM)."""
        if self.llm is None or name in self._kicked or self._backoff.get(name, 0.0) > self.now():
            return
        self._kicked.add(name)
        task = asyncio.get_running_loop().create_task(self._kick(name, cold))
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)

    async def _kick(self, name: str, cold: bool) -> None:
        try:
            need = self.cfg.router_cold_batch if cold else self.cfg.router_batch
            if (await asyncio.to_thread(self.reviews.counts, name))["buffered"] >= need:
                await self.flush(name)
        except RouterError as exc:
            log.info("router %s: no flush: %s", name, exc)
        except Exception:
            log.exception("router %s: flush failed", name)
        finally:
            self._kicked.discard(name)

    async def flush(self, name: str) -> dict[str, Any]:
        """One LLM call over the oldest buffered items, then acceptance. Raises ``RouterError``: 503 LLM off,
        409 a call is in flight, 400 nothing to do, 502 the LLM failed."""
        lock = self._locks.setdefault(name, asyncio.Lock())
        if lock.locked():
            raise RouterError(409, "a proposal call is already in flight for this router")
        async with lock, self._global:
            return await self._flush(name)

    async def _flush(self, name: str) -> dict[str, Any]:
        llm = self.llm
        if llm is None:
            raise RouterError(503, "router LLM is off (set GEMINI_API_KEY)")
        rt = await asyncio.to_thread(self.routers.get, name)
        if rt is None:
            raise RouterError(404, f"router {name!r} not found")
        act = await asyncio.to_thread(self.routers.active, name)
        cold = act is None
        current = real_labels(act["labels"]) if act else {}
        room = self.room(rt, act["labels"] if act else {})
        if room <= 0:
            raise RouterError(400, "label cap reached: no more labels can be added")
        limit = self.cfg.router_cold_batch if cold else self.cfg.router_batch
        items = await asyncio.to_thread(self.reviews.buffered, name, limit)
        if len(items) < 2:
            raise RouterError(400, "need at least 2 buffered items for a proposal")
        max_new = min(MAX_BOOTSTRAP_LABELS, room) if cold else min(MAX_NEW_PER_CALL, room)
        cfgd = rt["config"]
        system, user = build_prompt(
            current=act["labels"] if act else {},
            items=[i["input"] for i in items],
            max_new=max_new,
            max_chars=self.cfg.router_max_input_chars,
            domain_hint=cfgd.get("domain_hint"),
            instructions=cfgd.get("instructions"),
            bootstrap=cold,
        )
        purpose = "bootstrap" if cold else "propose"
        try:
            res = await llm.complete_json(
                system=system,
                user=user,
                schema=proposal_schema(),
                name=purpose,
                # Gemini 3 Flash spends ~2-3k of these on hidden reasoning before the JSON (measured)
                max_tokens=8192 if cold else 4096,
            )
        except LLMError as exc:
            self._backoff[name] = self.now() + BACKOFF_S
            await asyncio.to_thread(self._log_call, name, purpose, llm, None, len(items), str(exc))
            raise RouterError(502, f"LLM call failed: {exc}") from exc
        call_id = await asyncio.to_thread(self._log_call, name, purpose, llm, res, len(items), None)
        parsed = parse_proposals(res.data, list(current), len(items), room, max_new)
        for why in parsed.dropped:
            log.info("router %s: dropped a proposal: %s", name, why)
        if cold:
            made = await self._bootstrap(rt, items, parsed, call_id, res.model)
        else:
            made = await self._propose(rt, act, items, parsed, call_id, res.model)
        return {"router": name, "items": len(items), "llm_call_id": call_id, **made}

    def _log_call(self, name: str, purpose: str, llm: Any, res: Any, n_items: int, error: str | None) -> int:
        return self.reviews.log_call(
            router=name,
            purpose=purpose,
            provider_url=llm.base_url,
            model=res.model if res else llm.model,
            status="ok" if res else "error",
            latency_ms=res.latency_ms if res else None,
            input_tokens=res.input_tokens if res else None,
            output_tokens=res.output_tokens if res else None,
            items=n_items,
            error=error,
        )

    def _proposal_row(
        self, rt: dict[str, Any], state: str, reason: str, labels: list[dict[str, Any]], stats: dict[str, Any]
    ) -> str:
        return self.reviews.add_proposal(
            state=state,
            owner=rt["owner"],
            router=rt["name"],
            classifier=rt["classifier"],
            classifier_version=None,
            reason=reason,
            proposal={"labels": labels, "stats": stats},
        )

    async def _auto_allowed(self, rt: dict[str, Any], n_new: int) -> bool:
        since = (self.now() // DAY_S) * DAY_S
        today = await asyncio.to_thread(self.routers.new_labels_since, rt["name"], since)
        return today + n_new <= self.cfg.router_max_new_per_day

    async def _propose(
        self,
        rt: dict[str, Any],
        act: dict[str, Any],
        items: list[dict[str, Any]],
        parsed: Parsed,
        call_id: int,
        model: str,
    ) -> dict[str, Any]:
        """Re-classify each candidate's members against current + candidate + other; publish or queue only
        what clef itself confirms."""
        name, thr = rt["name"], self.thresholds(rt)
        instr = rt["config"].get("instructions")
        current = real_labels(act["labels"])
        humans = await asyncio.to_thread(self.reviews.human_labels, name, [i["input_sha256"] for i in items])
        stats0 = {"items": len(items), "llm_call_id": call_id, "model": model, "mode": rt["mode"]}
        placed: set[int] = set()
        shown = activated = failed = 0
        for cand in parsed.candidates:
            members = [items[i] for i in cand.member_indices]
            against = [m for m in members if humans.get(m["input_sha256"]) in current]
            testable = [m for m in members if m not in against]
            full = build_label_set({**current, cand.name: cand.description})
            scores = await self.classify(full, instr, [m["input"] for m in testable]) if testable else []
            ev = evaluate_proposal(cand.name, scores, thr, len(against))
            placed.update(cand.member_indices)
            ids = [m["id"] for m in members]
            entry = {
                "name": cand.name,
                "description": cand.description,
                "member_ids": ids,
                "rationale": cand.rationale,
                "nearest_existing": cand.nearest_existing,
                "acceptance": ev,
            }
            if not ev["passed"]:
                await asyncio.to_thread(self._proposal_row, rt, "failed", "proposal", [entry], stats0)
                await asyncio.to_thread(self.reviews.bump_attempts, ids)
                failed += 1
                continue
            if rt["mode"] == "auto" and await self._auto_allowed(rt, 1):
                await asyncio.to_thread(
                    self._publish,
                    rt,
                    {cand.name: cand.description},
                    "proposal",
                    "auto",
                    [entry],
                    {**stats0, "label_set_version": None},
                )
                current[cand.name] = cand.description
                activated += 1
                continue
            waiting = {"waiting": "daily limit of new labels reached"} if rt["mode"] == "auto" else {}
            await asyncio.to_thread(
                self._proposal_row, rt, "open", "proposal", [entry], {**stats0, **waiting}
            )
            await asyncio.to_thread(self.reviews.set_state, ids, "open", only_from=("buffered",))
            shown += 1
        rest = [i for n, i in enumerate(items) if n not in placed]
        await self._leftovers(rest, items, parsed)
        if activated:
            await self._reroute_quietly(name)
        return {"proposals": shown + activated + failed, "queued": shown, "activated": activated}

    async def _leftovers(
        self, rest: list[dict[str, Any]], items: list[dict[str, Any]], parsed: Parsed
    ) -> None:
        """Items that no proposal took: the LLM's "covered" and "unclear" ones wait for a human (with a hint);
        the others have been checked once more and move toward permanent_other."""
        hints: dict[str, str] = {}
        for idx, label in parsed.covered.items():
            hints[items[idx]["id"]] = label
        for idx in parsed.unclear:
            hints.setdefault(items[idx]["id"], "unclear")
        keep = [r["id"] for r in rest if r["id"] not in hints]
        marked = [r for r in rest if r["id"] in hints]

        def write() -> None:
            for r in marked:
                pred = {**(r["prediction"] or {}), "llm_hint": hints[r["id"]]}
                self.reviews.update_row(r["id"], state="open", prediction=pred)
            self.reviews.bump_attempts(keep)

        await asyncio.to_thread(write)

    def _publish(
        self,
        rt: dict[str, Any],
        added: dict[str, str],
        source: str,
        by: str,
        labels: list[dict[str, Any]],
        stats: dict[str, Any],
        reason: str = "proposal",
    ) -> str:
        """Auto mode: record the proposal as accepted and publish it, in one transaction."""
        with self.db.tx():
            pub = self.activate_sync(rt["name"], added, source, by)
            row_id = self._proposal_row(
                rt, "accepted", reason, labels, {**stats, "label_set_version": pub["version"]}
            )
            self.reviews.update_row(
                row_id,
                label=sorted(added),
                resolved_at=self.now(),
                resolved_by=by,
                classifier_version=pub["classifier_version"],
            )
        return row_id

    async def _bootstrap(
        self, rt: dict[str, Any], items: list[dict[str, Any]], parsed: Parsed, call_id: int, model: str
    ) -> dict[str, Any]:
        """Cold start: check the whole starter set on the buffered items; keep labels 2+ items land on."""
        name, thr = rt["name"], self.thresholds(rt)
        instr = rt["config"].get("instructions")
        desc = {c.name: c.description for c in parsed.candidates}
        info = {c.name: c for c in parsed.candidates}
        names = list(desc)
        stats0 = {"items": len(items), "llm_call_id": call_id, "model": model, "mode": rt["mode"]}
        ev: dict[str, Any] = {"kept": {}, "passed": False, "covered": 0, "total": len(items)}
        for _ in range(2):  # a second pass re-checks the set that is left after pruning
            if not names:
                break
            scores = await self.classify(
                build_label_set({n: desc[n] for n in names}), instr, [i["input"] for i in items]
            )
            ev = evaluate_bootstrap(names, scores, thr)
            if list(ev["kept"]) == names:
                break
            names = list(ev["kept"])
        ids = [i["id"] for i in items]
        if not ev["passed"]:
            await asyncio.to_thread(
                self._proposal_row,
                rt,
                "failed",
                "bootstrap",
                [],
                {
                    **stats0,
                    "error": "the starter labels did not cover the buffered items",
                    "covered": ev["covered"],
                },
            )
            await asyncio.to_thread(self.reviews.bump_attempts, ids)
            return {"proposals": 1, "queued": 0, "activated": 0}
        labels = [
            {
                "name": n,
                "description": desc[n],
                "member_ids": [items[k]["id"] for k in idx],
                "rationale": info[n].rationale,
                "nearest_existing": "",
                "acceptance": {"landed": len(idx), "tau": ev["tau"], "passed": True},
            }
            for n, idx in ev["kept"].items()
        ]
        landed = {mid for lab in labels for mid in lab["member_ids"]}
        stats = {**stats0, "covered": ev["covered"], "total": ev["total"], "tau": ev["tau"]}
        added = {lab["name"]: lab["description"] for lab in labels}
        rest = [i for i in ids if i not in landed]
        if rt["mode"] == "auto":  # the starter set is exempt from the daily cap: it only starts the loop
            await asyncio.to_thread(self._publish, rt, added, "bootstrap", "auto", labels, stats, "bootstrap")
            await asyncio.to_thread(self.reviews.bump_attempts, rest)
            await self._reroute_quietly(name)
            return {"proposals": 1, "queued": 0, "activated": 1}
        await asyncio.to_thread(self._proposal_row, rt, "open", "bootstrap", labels, stats)
        await asyncio.to_thread(self.reviews.set_state, list(landed), "open", only_from=("buffered",))
        await asyncio.to_thread(self.reviews.bump_attempts, rest)
        return {"proposals": 1, "queued": 1, "activated": 0}

    # ---- re-routing
    async def reroute(self, name: str) -> int:
        """Classify the buffered items again under the active label set: the ones that now pass the gate are
        resolved by the router (text dropped), ambiguous ones wait for a human, the rest move toward
        permanent_other. Returns how many were resolved."""
        rt = await asyncio.to_thread(self.routers.get, name)
        act = await asyncio.to_thread(self.routers.active, name)
        if rt is None or act is None:
            return 0
        thr, margin, instr = self.thresholds(rt), self.margin(rt), rt["config"].get("instructions")
        items = await asyncio.to_thread(self.reviews.buffered, name, 1000)
        resolved = 0
        step = max(1, self.cfg.max_batch)
        for i in range(0, len(items), step):
            chunk = items[i : i + step]
            scores = await self.classify(act["labels"], instr, [c["input"] for c in chunk])
            done, ambiguous, still = [], [], []
            for item, s in zip(chunk, scores, strict=True):
                verdict = gate(s, thr, margin)
                pred = {
                    "label": top_label(s),
                    "confidence": s[top_label(s)],
                    "scores": s,
                    "label_set_version": act["version"],
                }
                if verdict == "classified":
                    done.append((item["id"], top_label(s), pred))
                elif verdict == "ambiguous":
                    ambiguous.append((item["id"], pred))
                else:
                    still.append(item["id"])

            def write(done=done, ambiguous=ambiguous, still=still) -> None:
                self.reviews.resolve_many(done, f"router:{name}")
                for rid, pred in ambiguous:
                    self.reviews.update_row(rid, state="open", prediction=pred, reason="ambiguous")
                self.reviews.bump_attempts(still)

            await asyncio.to_thread(write)
            resolved += len(done)
        return resolved

    async def _reroute_quietly(self, name: str) -> None:
        try:
            await self.reroute(name)
        except Exception as exc:  # the engine may be busy or down: the items stay buffered for the next try
            log.warning("router %s: re-routing deferred: %s", name, exc)

    # ---- the Review actions
    async def resolve_item(
        self, row: dict[str, Any], action: str, label: str | None, by: str
    ) -> dict[str, Any]:
        """Label or discard one item. A label is re-checked against the active label set BEFORE the text is
        purged; the agreement is recorded in ``prediction.recheck``."""
        if action == "discard":
            if not await asyncio.to_thread(self.reviews.claim, row["id"], OPEN_STATES, "discarded"):
                raise RouterError(409, "this review item is already resolved")
            await asyncio.to_thread(
                self.reviews.update_row, row["id"], input=None, resolved_at=self.now(), resolved_by=by
            )
            return {}
        labels, instr, version = await self._labels_for(row)
        if label not in labels:
            raise RouterError(400, f"label {label!r} is not one of the current labels: {', '.join(labels)}")
        if row["input"] is None:
            raise RouterError(409, "the input of this item was already purged")
        if not await asyncio.to_thread(self.reviews.claim, row["id"], OPEN_STATES, "resolved"):
            raise RouterError(409, "this review item is already resolved")
        recheck: dict[str, Any]
        try:
            [scores] = await self.classify(labels, instr, [row["input"]])
            top = top_label(scores)
            recheck = {
                "label": top,
                "confidence": round(scores[top], 4),
                "agrees": top == label,
                "label_set_version": version,
            }
        except Exception as exc:  # the engine may be down: the human label still counts
            recheck = {"error": str(exc)[:200]}
        pred = {**(row["prediction"] or {}), "recheck": recheck}
        await asyncio.to_thread(
            self.reviews.update_row,
            row["id"],
            label=label,
            input=None,
            prediction=pred,
            resolved_at=self.now(),
            resolved_by=by,
        )
        return {}

    async def _labels_for(self, row: dict[str, Any]) -> tuple[dict[str, str], str | None, int | None]:
        if row["router"]:
            rt = await asyncio.to_thread(self.routers.get, row["router"])
            act = await asyncio.to_thread(self.routers.active, row["router"])
            if rt is None or act is None:
                raise RouterError(400, "this router has no labels yet: approve its starter proposal first")
            return act["labels"], rt["config"].get("instructions"), act["version"]
        doc = await asyncio.to_thread(self.ctx.store.get, row["classifier"] or "")
        if doc is None or doc.get("kind") != "classify" or doc.get("multi_label"):
            raise RouterError(400, "the classifier no longer exists or is not single-label")
        return labels_of(doc["labels"]), doc.get("instructions"), doc["version"]

    async def approve(
        self,
        row: dict[str, Any],
        edits: list[tuple[str, str]] | None,
        by: str,
        *,
        ip: str | None = None,
        request_id: str | None = None,
    ) -> None:
        """Publish an open proposal (all its labels, or an edited subset), then re-route the buffer."""
        rt = await asyncio.to_thread(self.routers.get, row["router"] or "")
        if rt is None:
            raise RouterError(404, "the router of this proposal no longer exists")
        entries = {e["name"]: e for e in row["proposal"]["labels"]}
        picked = edits if edits is not None else [(e["name"], e["description"]) for e in entries.values()]
        if not picked:
            raise RouterError(400, "approve at least one label (or reject the proposal)")
        for label, desc in picked:
            if len(desc.split()) > 40 or not desc.strip():
                raise RouterError(400, f"label {label!r}: the description must be 1-40 words")
        if not await asyncio.to_thread(self.reviews.claim, row["id"], ("open",), "accepted"):
            raise RouterError(409, "this proposal is already resolved")
        try:
            pub = await asyncio.to_thread(
                functools.partial(
                    self.activate_sync,
                    rt["name"],
                    dict(picked),
                    row["reason"] if row["reason"] == "bootstrap" else "proposal",
                    by,
                    ip=ip,
                    request_id=request_id,
                )
            )
        except BaseException:
            await asyncio.to_thread(self.reviews.claim, row["id"], ("accepted",), "open")  # nothing published
            raise
        members = [m for e in entries.values() for m in e["member_ids"]]
        await asyncio.to_thread(
            self.reviews.update_row,
            row["id"],
            label=[n for n, _ in picked],
            resolved_at=self.now(),
            resolved_by=by,
            classifier_version=pub["classifier_version"],
        )
        await asyncio.to_thread(self.reviews.set_state, members, "buffered", only_from=("open",))
        await self._reroute_quietly(rt["name"])

    async def reject(self, row: dict[str, Any], by: str) -> None:
        if not await asyncio.to_thread(self.reviews.claim, row["id"], ("open",), "rejected"):
            raise RouterError(409, "this proposal is already resolved")
        members = [m for e in (row["proposal"] or {}).get("labels", []) for m in e["member_ids"]]

        def write() -> None:
            self.reviews.update_row(row["id"], resolved_at=self.now(), resolved_by=by)
            self.reviews.bump_attempts(members)  # back to the buffer, one check closer to permanent_other

        await asyncio.to_thread(write)

    # ---- low-confidence items from saved classifiers
    async def capture(
        self, name: str, doc: dict[str, Any], inputs: list[Any], results: list[dict[str, Any]]
    ) -> None:
        """Results of an interactive single-label run below ``review_threshold`` become open review items.
        Never fails the request."""
        thr = doc.get("review_threshold")
        if thr is None or doc.get("kind") != "classify" or doc.get("multi_label"):
            return
        rows = [
            {
                "owner": doc.get("owner"),
                "router": None,
                "classifier": name,
                "classifier_version": doc.get("version"),
                "reason": "low_confidence",
                "state": "open",
                "input": x,
                "prediction": {
                    "label": r.get("label"),
                    "confidence": r.get("confidence"),
                    "scores": r.get("scores"),
                },
            }
            for x, r in zip(inputs, results, strict=False)
            if r.get("confidence") is not None and r["confidence"] < thr
        ]
        if rows:
            try:
                await asyncio.to_thread(self.reviews.add_items, rows)
            except Exception:
                log.exception("review: could not capture low-confidence items of %s", name)

    # ---- background loop
    async def tick(self) -> None:
        now = self.now()
        if now - self._last_purge >= PURGE_EVERY_S:
            self._last_purge = now
            n = await asyncio.to_thread(self.reviews.purge, self.cfg.review_ttl_days, now)
            if n:
                log.info("review: purged %d row(s) older than %s days", n, self.cfg.review_ttl_days)
        if self.llm is None:
            return
        for rt in await asyncio.to_thread(self.routers.list):
            name = rt["name"]
            if self._backoff.get(name, 0.0) > now or self._locks.get(name, asyncio.Lock()).locked():
                continue
            act = await asyncio.to_thread(self.routers.active, name)
            cold = act is None
            n = (await asyncio.to_thread(self.reviews.counts, name))["buffered"]
            oldest = await asyncio.to_thread(self.reviews.oldest_buffered, name)
            need = self.cfg.router_cold_batch if cold else self.cfg.router_batch
            aged = (
                self.cfg.router_flush_s > 0
                and oldest is not None
                and now - oldest >= self.cfg.router_flush_s
                and n >= (MIN_AGE_FLUSH_COLD if cold else MIN_AGE_FLUSH)
            )
            if n >= need or aged:
                try:
                    await self.flush(name)
                except RouterError as exc:
                    log.info("router %s: no flush: %s", name, exc)

    async def run(self) -> None:
        while True:
            await asyncio.sleep(self.tick_s)
            try:
                await self.tick()
            except asyncio.CancelledError:
                raise
            except Exception:
                log.exception("router: background tick failed")

    async def start(self) -> None:
        self._loop_task = asyncio.create_task(self.run(), name="clef-router")

    async def stop(self) -> None:
        tasks = [self._loop_task, *self._tasks]
        for task in tasks:
            if task is not None:
                task.cancel()
        await asyncio.gather(*(t_ for t_ in tasks if t_ is not None), return_exceptions=True)
        client = self.llm
        if client is not None and hasattr(client, "aclose"):
            await client.aclose()


# ------------------------------------------------------------------ HTTP models


class _Base(BaseModel):
    model_config = ConfigDict(extra="forbid")


class GateIn(_Base):
    thresholds: list[tuple[int, float]] | None = Field(
        None,
        description="[[max_options, min_top_probability], ...], rising; default 10:0.5, 30:0.7, 64:0.85",
    )
    margin: float | None = Field(
        None, ge=0.0, le=1.0, description="top-two margin below which it is ambiguous"
    )


class RouterPut(_Base):
    classifier: str | None = Field(
        None, description="saved classifier to publish to (default: the router name)"
    )
    seeds: dict[str, str] | list[str] | None = Field(
        None, description="starting labels (name -> description); default: the classifier's current labels"
    )
    instructions: str | None = Field(
        None, description="the classifier's instructions (what is being decided)"
    )
    domain_hint: str | None = Field(None, description="one line about the domain, sent to the LLM")
    mode: Literal["review", "auto"] | None = Field(None, description="default: CLEF_ROUTER_MODE (review)")
    max_labels: int | None = Field(
        None, ge=2, le=HARD_MAX_LABELS, description="cap on real labels (other extra)"
    )
    gate: GateIn | None = None


class LlmStatus(BaseModel):
    enabled: bool = Field(description="false until GEMINI_API_KEY / CLEF_LLM_API_KEY is set")
    host: str | None = Field(
        None, description="where buffered inputs are sent, e.g. generativelanguage.googleapis.com"
    )
    model: str | None = None
    calls_today: int = 0
    input_tokens_today: int = 0
    output_tokens_today: int = 0
    errors_today: int = 0


class RouterInfo(BaseModel):
    model_config = ConfigDict(extra="allow")
    name: str
    owner: str | None = None
    classifier: str
    mode: Literal["review", "auto"]
    config: dict[str, Any]
    active_version: int | None = Field(None, description="label-set version; null while cold")
    classifier_version: int | None = None
    labels: dict[str, str] = Field(description="active labels including other; empty while cold")
    buffered: int
    open_reviews: int
    created_at: float
    updated_at: float
    llm: LlmStatus


class RouterList(BaseModel):
    routers: list[RouterInfo]


class RouterDeleted(BaseModel):
    deleted: str


class RouterLabelSet(BaseModel):
    version: int
    labels: dict[str, str]
    source: Literal["seed", "bootstrap", "proposal", "manual", "rollback"]
    classifier_version: int | None = None
    created_at: float
    created_by: str | None = None


class RouterVersions(BaseModel):
    router: str
    versions: list[RouterLabelSet]


class RouteRequest(_Base):
    input: Any = Field(description="Text or any JSON to route")


class RouteBatchRequest(_Base):
    inputs: list[Any] = Field(min_length=1, description="1..max_batch inputs")


class RouteResult(ClassifyResult):
    router: str
    reason: Literal["classified", "other", "low_confidence", "ambiguous", "cold"]
    label_set_version: int | None = Field(None, description="null while the router is cold")
    review_id: str | None = Field(None, description="the review row, when the item was kept for review")


class RouteResponse(RouteResult):
    request_id: str


class RouteBatchResponse(BaseModel):
    batch_ms: float
    results: list[RouteResult]
    request_id: str


class ProposeResponse(BaseModel):
    router: str
    items: int = Field(description="buffered items sent to the LLM")
    llm_call_id: int
    proposals: int = Field(description="proposals the LLM made that passed the re-check or failed it")
    queued: int = Field(description="of those, waiting in the Review queue")
    activated: int = Field(description="of those, published at once (auto mode)")


class ReviewRow(BaseModel):
    model_config = ConfigDict(extra="allow")
    id: str
    kind: Literal["item", "proposal"]
    state: str
    owner: str | None = None
    router: str | None = None
    classifier: str | None = None
    classifier_version: int | None = None
    reason: str
    input: Any = Field(None, description="stored only until the item is resolved or the TTL passes")
    input_sha256: str | None = None
    prediction: dict[str, Any] | None = None
    proposal: dict[str, Any] | None = None
    label: Any = Field(None, description="item: the human label; proposal: the accepted names")
    reroutes: int
    created_at: float
    resolved_at: float | None = None
    resolved_by: str | None = None


class ReviewList(BaseModel):
    items: list[ReviewRow]
    total: int


class LabelEdit(_Base):
    name: str
    description: str


class ResolveRequest(_Base):
    action: Literal["label", "discard", "approve", "reject"]
    label: str | None = Field(None, description="action=label: one of the current labels")
    labels: list[LabelEdit] | None = Field(None, description="action=approve: an edited subset of the labels")


# ------------------------------------------------------------------ routes


def router(ctx: AppContext) -> APIRouter:
    cfg, ApiError = ctx.cfg, ctx.api_error  # noqa: N806
    svc = RouterService(ctx)
    ctx.extra["router_service"] = svc
    ctx.extra.setdefault("llm_client", client_from_config(cfg))  # None = the router's LLM is off
    ctx.extra["review_capture"] = svc.capture  # main.py's saved-classifier routes call this
    ctx.on_startup.append(svc.start)
    ctx.on_shutdown.append(svc.stop)
    manage = ctx.scope("classifiers")
    api = APIRouter(prefix="/v1", tags=["router"], responses=ctx.errors)

    def caller(request: Request) -> str | None:
        return getattr(request.state, "key_name", None)

    def is_admin(request: Request) -> bool:
        principal = ctx.principal(request)
        return True if principal is None else bool(principal.has("admin"))

    def scoped(request: Request) -> bool:
        """Private to the owner when auth is on; admins see everything, owner-less rows included."""
        return ctx.auth_enabled() and not is_admin(request)

    def visible(request: Request, owner: str | None) -> bool:
        return not scoped(request) or owner == caller(request)

    def err(exc: Exception) -> Exception:
        if isinstance(exc, RouterError):
            return ApiError(exc.status, str(exc))
        if isinstance(exc, TooManyClassifiers):
            return ApiError(409, str(exc))
        if isinstance(exc, IntegrityError):
            return ApiError(409, "conflicting write (a router or classifier with that name exists)")
        if is_unavailable(exc):
            log.error("database unavailable: %s", exc)
            return ApiError(503, "database unavailable", {"Retry-After": "5"})
        return ctx.map_exception(exc)

    async def audit_event(request: Request, action: str, target: str, detail: dict[str, Any]) -> None:
        def write() -> None:
            with ctx.db.tx() as conn:
                audit(
                    conn,
                    caller(request) or "anonymous",
                    action,
                    target,
                    detail,
                    request.client.host if request.client else None,
                    ctx.request_id(request),
                )

        await asyncio.to_thread(write)

    async def load_router(request: Request, name: str) -> dict[str, Any]:
        try:
            validate_name(name)
        except InvalidName as exc:
            raise ApiError(400, str(exc)) from exc
        rt = await asyncio.to_thread(svc.routers.get, name)
        if rt is None or not visible(request, rt["owner"]):
            raise ApiError(404, f"router {name!r} not found")
        return rt

    async def view(rt: dict[str, Any]) -> dict[str, Any]:
        act = await asyncio.to_thread(svc.routers.active, rt["name"])
        counts = await asyncio.to_thread(svc.reviews.counts, rt["name"])
        since = (svc.now() // DAY_S) * DAY_S
        use = await asyncio.to_thread(svc.reviews.llm_usage, since, rt["name"])
        client = ctx.extra.get("llm_client")
        llm = {"enabled": client is not None}
        if client is not None:
            llm.update(host=urlsplit(client.base_url).netloc, model=client.model)
        llm.update(
            calls_today=use["calls"],
            input_tokens_today=use["input_tokens"],
            output_tokens_today=use["output_tokens"],
            errors_today=use["errors"],
        )
        return {
            **rt,
            "classifier_version": act["classifier_version"] if act else None,
            "labels": act["labels"] if act else {},
            **counts,
            "llm": llm,
        }

    # ---- routers
    def gate_config(base: dict[str, Any], gate_in: GateIn | None) -> dict[str, Any]:
        out = dict(base)
        if gate_in is not None:
            if gate_in.thresholds is not None:
                try:
                    out["thresholds"] = [list(p) for p in normalise_thresholds(gate_in.thresholds)]
                except ValueError as exc:
                    raise RouterError(400, str(exc)) from exc
            if gate_in.margin is not None:
                out["margin"] = gate_in.margin
        return out

    def create_sync(
        name: str,
        owner: str | None,
        classifier: str,
        config: dict[str, Any],
        mode: str,
        seeds: dict[str, str],
        request: Request,
    ) -> None:
        now = svc.now()
        with ctx.db.tx() as conn:
            conn.execute(
                insert(t.routers).values(
                    name=name,
                    owner=owner,
                    classifier=classifier,
                    config=json.dumps(config),
                    mode=mode,
                    active_version=None,
                    created_at=now,
                    updated_at=now,
                )
            )
            audit(
                conn,
                owner or "anonymous",
                "router.create",
                name,
                {"classifier": classifier, "mode": mode},
                request.client.host if request.client else None,
                ctx.request_id(request),
            )
            if seeds:
                svc.activate_sync(
                    name,
                    {},
                    "seed",
                    owner or "anonymous",
                    replace=seeds,
                    ip=request.client.host if request.client else None,
                    request_id=ctx.request_id(request),
                )

    def clean_seeds(raw: Any) -> dict[str, str]:
        pairs = raw.items() if isinstance(raw, dict) else [(k, k) for k in raw]
        out: dict[str, str] = {}
        for label, desc in pairs:
            if not isinstance(label, str) or not label.strip():
                raise RouterError(400, "seeds: labels must be non-empty strings")
            if label.lower() == OTHER:
                continue  # clef adds and maintains its own "other"
            out[label] = desc
        return out

    @api.put(
        "/routers/{name}",
        response_model=RouterInfo,
        summary="Create a router or change its settings",
        dependencies=manage,
    )
    async def put_router(name: str, body: RouterPut, request: Request) -> dict[str, Any]:
        try:
            validate_name(name)
            caller_name = caller(request)
            existing = await asyncio.to_thread(svc.routers.get, name)
            if existing is not None:
                if not visible(request, existing["owner"]):
                    raise RouterError(
                        403, f"router {name!r} is not yours; only its owner or an admin may change it"
                    )
                if body.classifier is not None and body.classifier != existing["classifier"]:
                    raise RouterError(400, "a router's classifier cannot change")
                config = {
                    **existing["config"],
                    "gate": gate_config(existing["config"].get("gate") or {}, body.gate),
                }
                for key in ("instructions", "domain_hint", "max_labels"):
                    if getattr(body, key) is not None:
                        config[key] = getattr(body, key)
                mode = body.mode or existing["mode"]

                def save() -> None:
                    with ctx.db.tx() as conn:
                        conn.execute(
                            update(t.routers)
                            .where(t.routers.c.name == name)
                            .values(config=json.dumps(config), mode=mode, updated_at=svc.now())
                        )
                        audit(conn, caller_name or "anonymous", "router.update", name, {"mode": mode})
                        if body.seeds is not None:
                            if existing["active_version"] is not None:
                                raise RouterError(
                                    400, "seeds are only for a new or cold router: labels are only added"
                                )
                            seeds = clean_seeds(body.seeds)
                            svc.activate_sync(name, {}, "seed", caller_name or "anonymous", replace=seeds)

                await asyncio.to_thread(save)
                return await view(await asyncio.to_thread(svc.routers.get, name))
            classifier = validate_name(body.classifier or name)
            if await asyncio.to_thread(svc.routers.by_classifier, classifier) is not None:
                raise RouterError(409, f"classifier {classifier!r} already belongs to another router")
            doc = await asyncio.to_thread(ctx.store.get, classifier)
            if doc is not None:
                exists, owner = await asyncio.to_thread(ctx.store.owner_of, classifier)
                if exists and owner != caller_name and not is_admin(request):
                    raise RouterError(403, f"classifier {classifier!r} is not yours")
                if doc.get("kind") != "classify" or doc.get("multi_label"):
                    raise RouterError(400, "a router needs a single-label classify classifier")
            seeds = (
                clean_seeds(body.seeds)
                if body.seeds is not None
                else (clean_seeds(labels_of(doc["labels"])) if doc is not None else {})
            )
            room = capacity(body.max_labels or HARD_MAX_LABELS, cfg.max_labels)
            if len(seeds) > room:
                raise RouterError(400, f"too many seeds (at most {room})")
            config = {
                "instructions": body.instructions
                if body.instructions is not None
                else (doc or {}).get("instructions"),
                "domain_hint": body.domain_hint,
                "max_labels": body.max_labels or capacity(HARD_MAX_LABELS, cfg.max_labels),
                "gate": gate_config(
                    {"thresholds": [list(p) for p in DEFAULT_THRESHOLDS], "margin": DEFAULT_MARGIN}, body.gate
                ),
            }
            await asyncio.to_thread(
                create_sync,
                name,
                caller_name,
                classifier,
                config,
                body.mode or cfg.router_mode,
                seeds,
                request,
            )
            return await view(await asyncio.to_thread(svc.routers.get, name))
        except Exception as exc:
            raise err(exc) from exc

    @api.get("/routers", response_model=RouterList, summary="List routers", dependencies=manage)
    async def list_routers(request: Request) -> dict[str, Any]:
        rows = await asyncio.to_thread(svc.routers.list, caller(request), scoped(request))
        return {"routers": [await view(r) for r in rows]}

    @api.get("/routers/{name}", response_model=RouterInfo, summary="Get a router", dependencies=manage)
    async def get_router(name: str, request: Request) -> dict[str, Any]:
        return await view(await load_router(request, name))

    @api.delete(
        "/routers/{name}",
        response_model=RouterDeleted,
        summary="Delete a router (its classifier stays; buffered texts are dropped)",
        dependencies=manage,
    )
    async def delete_router(name: str, request: Request) -> dict[str, str]:
        await load_router(request, name)
        await asyncio.to_thread(svc.reviews.drop_router, name, caller(request) or "anonymous")
        await asyncio.to_thread(svc.routers.delete, name)
        await audit_event(request, "router.delete", name, {})
        return {"deleted": name}

    # ---- routing
    @api.post(
        "/routers/{name}/route",
        response_model=RouteResponse,
        summary="Route one input; keep what the router cannot place for review",
        dependencies=[*ctx.scope("classify"), *ctx.limited],
    )
    async def route_one(name: str, body: RouteRequest, request: Request) -> dict[str, Any]:
        rt = await load_router(request, name)
        try:
            out = await svc.route(request, rt, [body.input], batch=False)
        except Exception as exc:
            raise err(exc) from exc
        return {**out["results"][0], "request_id": ctx.request_id(request)}

    @api.post(
        "/routers/{name}/route/batch",
        response_model=RouteBatchResponse,
        summary="Route many inputs in one micro-batched pass",
        dependencies=[*ctx.scope("classify"), *ctx.limited],
    )
    async def route_many(name: str, body: RouteBatchRequest, request: Request) -> dict[str, Any]:
        rt = await load_router(request, name)
        try:
            out = await svc.route(request, rt, body.inputs, batch=True)
        except Exception as exc:
            raise err(exc) from exc
        return {**out, "request_id": ctx.request_id(request)}

    @api.post(
        "/routers/{name}/propose",
        status_code=202,
        response_model=ProposeResponse,
        summary="Ask the LLM for new labels now (the call completes before this returns)",
        dependencies=manage,
    )
    async def propose(name: str, request: Request) -> dict[str, Any]:
        await load_router(request, name)
        try:
            out = await svc.flush(name)
        except Exception as exc:
            raise err(exc) from exc
        await audit_event(request, "router.propose", name, {"llm_call_id": out["llm_call_id"]})
        return out

    @api.get(
        "/routers/{name}/versions",
        response_model=RouterVersions,
        summary="Label-set versions, newest first",
        dependencies=manage,
    )
    async def router_versions(name: str, request: Request) -> dict[str, Any]:
        await load_router(request, name)
        return {"router": name, "versions": await asyncio.to_thread(svc.routers.label_sets, name)}

    @api.post(
        "/routers/{name}/versions/{version}/activate",
        response_model=RouterInfo,
        summary="Roll back: publish an earlier label set again as a new classifier version",
        dependencies=manage,
    )
    async def activate_version(name: str, version: int, request: Request) -> dict[str, Any]:
        await load_router(request, name)
        old = await asyncio.to_thread(svc.routers.label_set, name, version)
        if old is None:
            raise ApiError(404, f"router {name!r} has no label-set version {version}")
        try:
            await asyncio.to_thread(
                functools.partial(
                    svc.activate_sync,
                    name,
                    {},
                    "rollback",
                    caller(request) or "anonymous",
                    replace=real_labels(old["labels"]),
                    ip=request.client.host if request.client else None,
                    request_id=ctx.request_id(request),
                )
            )
        except Exception as exc:
            raise err(exc) from exc
        await svc._reroute_quietly(name)
        return await view(await asyncio.to_thread(svc.routers.get, name))

    # ---- review
    async def load_review(request: Request, review_id: str) -> dict[str, Any]:
        row = await asyncio.to_thread(svc.reviews.get, review_id)
        if row is None or not visible(request, row["owner"]):
            raise ApiError(404, f"review item {review_id!r} not found")
        return row

    @api.get(
        "/review",
        response_model=ReviewList,
        summary="Review queue: router proposals and low-confidence items, newest first",
        dependencies=manage,
    )
    async def list_review(
        request: Request,
        kind: Literal["item", "proposal"] | None = Query(None),
        state: str | None = Query(None),
        router: str | None = Query(None),
        classifier: str | None = Query(None),
        limit: int = Query(50, ge=1, le=500),
        offset: int = Query(0, ge=0),
    ) -> dict[str, Any]:
        rows, total = await asyncio.to_thread(
            functools.partial(
                svc.reviews.list,
                kind=kind,
                state=state,
                router=router,
                classifier=classifier,
                owner=caller(request),
                scoped=scoped(request),
                limit=limit,
                offset=offset,
            )
        )
        return {"items": rows, "total": total}

    @api.get("/review/{review_id}", response_model=ReviewRow, summary="One review row", dependencies=manage)
    async def get_review(review_id: str, request: Request) -> dict[str, Any]:
        return await load_review(request, review_id)

    @api.post(
        "/review/{review_id}/resolve",
        response_model=ReviewRow,
        summary="Label or discard an item; approve or reject a proposal",
        dependencies=manage,
    )
    async def resolve_review(review_id: str, body: ResolveRequest, request: Request) -> dict[str, Any]:
        row = await load_review(request, review_id)
        by = caller(request) or "anonymous"
        item_actions, proposal_actions = ("label", "discard"), ("approve", "reject")
        if body.action not in (item_actions if row["kind"] == "item" else proposal_actions):
            raise ApiError(400, f"action {body.action!r} does not apply to a {row['kind']}")
        if row["state"] not in (OPEN_STATES if row["kind"] == "item" else ("open",)):
            raise ApiError(409, f"this {row['kind']} is already {row['state']}")
        try:
            if body.action == "label":
                if not body.label:
                    raise RouterError(400, "action=label needs a label")
                await svc.resolve_item(row, "label", body.label, by)
            elif body.action == "discard":
                await svc.resolve_item(row, "discard", None, by)
            elif body.action == "approve":
                edits = None if body.labels is None else [(e.name, e.description) for e in body.labels]
                await svc.approve(
                    row,
                    edits,
                    by,
                    ip=request.client.host if request.client else None,
                    request_id=ctx.request_id(request),
                )
            else:
                await svc.reject(row, by)
        except Exception as exc:
            raise err(exc) from exc
        await audit_event(request, "review.resolve", review_id, {"action": body.action})
        return await load_review(request, review_id)

    return api


__all__ = ["RouterService", "RouterStore", "router"]
