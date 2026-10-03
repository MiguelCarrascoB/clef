"""Typed results returned by the classification helpers."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class Classification:
    """One classification result.

    Single-label: ``label`` is the chosen label, ``labels == [label]``.
    Multi-label: ``labels`` holds every label scoring >= ``threshold`` (best first, may be empty);
    ``label`` is the top-scoring entry of ``labels`` (None when empty) and ``confidence`` its score.
    ``scores`` always maps every label to its probability.
    """

    label: str | None
    labels: list[str]
    confidence: float | None
    scores: dict[str, float]
    multi_label: bool = False
    threshold: float | None = None
    request_id: str | None = None
    usage: dict[str, Any] = field(default_factory=dict)
    timing: dict[str, Any] = field(default_factory=dict)
    raw: dict[str, Any] = field(default_factory=dict)
    classifier: str | None = None


@dataclass(frozen=True)
class ScoreResult:
    """Ordinal score: ``score`` is the expected level index, ``level`` the most likely level."""

    score: float
    level: str
    level_index: int
    confidence: float | None
    distribution: dict[str, float]
    request_id: str | None = None
    usage: dict[str, Any] = field(default_factory=dict)
    timing: dict[str, Any] = field(default_factory=dict)
    raw: dict[str, Any] = field(default_factory=dict)
    classifier: str | None = None


def parse_classification(body: dict[str, Any], request_id: str | None = None) -> Classification:
    scores = {str(k): float(v) for k, v in (body.get("scores") or {}).items()}
    multi = bool(body.get("multi_label"))
    if multi:
        labels = list(body.get("labels") or [])
        label = labels[0] if labels else None
    else:
        label = body.get("label")
        labels = [label] if label is not None else []
    conf = body.get("confidence")
    if conf is None and label is not None:
        conf = scores.get(label)
    return Classification(
        label=label,
        labels=labels,
        confidence=None if conf is None else float(conf),
        scores=scores,
        multi_label=multi,
        threshold=body.get("threshold"),
        request_id=body.get("request_id") or request_id,
        usage=body.get("usage") or {},
        timing=body.get("timing") or {},
        raw=body,
        classifier=body.get("classifier"),
    )


def parse_score(body: dict[str, Any], request_id: str | None = None) -> ScoreResult:
    conf = body.get("confidence")
    return ScoreResult(
        score=float(body["score"]),
        level=body["level"],
        level_index=int(body["level_index"]),
        confidence=None if conf is None else float(conf),
        distribution={str(k): float(v) for k, v in (body.get("distribution") or {}).items()},
        request_id=body.get("request_id") or request_id,
        usage=body.get("usage") or {},
        timing=body.get("timing") or {},
        raw=body,
        classifier=body.get("classifier"),
    )


def parse_result(body: dict[str, Any]) -> Classification | ScoreResult:
    """Saved classifiers answer with either shape; ``distribution`` marks a score."""
    return parse_score(body) if "distribution" in body else parse_classification(body)


TERMINAL_STATUSES = frozenset({"succeeded", "failed", "cancelled"})


@dataclass(frozen=True)
class Job:
    """An async job (``POST /v1/jobs``).

    ``status``: queued, running, succeeded, failed, cancelled or interrupted. ``result`` is the kind's final
    summary (classify: ``{"items", "ok", "errors", "by_label"}``); ``raw`` is the full server body (timings,
    webhook delivery status, ...).
    """

    id: str
    kind: str
    status: str
    done: int
    total: int | None
    failed: int = 0
    percent: float | None = None
    eta_s: float | None = None
    error: str | None = None
    result: Any = None
    metadata: dict[str, Any] | None = None
    created_at: str | None = None
    started_at: str | None = None
    finished_at: str | None = None
    webhook: dict[str, Any] | None = None
    raw: dict[str, Any] = field(default_factory=dict)

    @property
    def finished(self) -> bool:
        return self.status in TERMINAL_STATUSES

    @property
    def ok(self) -> bool:
        return self.status == "succeeded"


@dataclass(frozen=True)
class JobItem:
    """One result row; ``result`` is typed for classify / score jobs (None for failed rows, other kinds)."""

    index: int
    error: str | None
    result: Classification | ScoreResult | None
    raw: dict[str, Any] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return self.error is None


def parse_job(body: dict[str, Any]) -> Job:
    progress = body.get("progress") or {}
    return Job(
        id=body["id"],
        kind=body.get("kind", ""),
        status=body["status"],
        done=int(progress.get("done", 0)),
        total=progress.get("total"),
        failed=int(progress.get("failed", 0)),
        percent=progress.get("percent"),
        eta_s=body.get("eta_s"),
        error=body.get("error"),
        result=body.get("result"),
        metadata=body.get("metadata"),
        created_at=body.get("created_at"),
        started_at=body.get("started_at"),
        finished_at=body.get("finished_at"),
        webhook=body.get("webhook"),
        raw=body,
    )


def parse_job_item(row: dict[str, Any], kind: str) -> JobItem:
    error = row.get("error")
    result: Classification | ScoreResult | None = None
    if error is None and kind == "classify":
        result = parse_classification({**row, "multi_label": "labels" in row})
    elif error is None and kind == "score":
        result = parse_score(row)
    return JobItem(index=int(row.get("index", -1)), error=error, result=result, raw=row)
