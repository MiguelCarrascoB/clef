"""Classification API models + the (normative) translation classify/score <-> SystemOne records and answers.

Everything here is pure (no FastAPI, no engine): requests become `SystemOneRequest` objects, engine
answers become classify / score response bodies. See docs/ARCHITECTURE.md "Classification API".
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from .config import Config
from .schemas import DEFAULT_MODEL, Question, SystemOneRequest, Timing, Usage

Labels = list[str] | dict[str, str]
SINGLE_QID = "label"
SCORE_QID = "score"


# ------------------------------------------------------------------ validation helpers


def _check_names(v: Any, what: str) -> list[str]:
    """Shape check shared by labels (list or dict) and levels (list): unique, non-empty strings."""
    names = list(v)
    if any(not isinstance(n, str) or not n.strip() for n in names):
        raise ValueError(f"{what} must be non-empty strings")
    if len(set(names)) != len(names):
        raise ValueError(f"{what} must be unique")
    return names


def check_labels_shape(v: Any) -> Any:
    if isinstance(v, dict):
        _check_names(v.keys(), "labels")
        if any(not isinstance(d, str) for d in v.values()):
            raise ValueError("label descriptions must be strings")
    else:
        _check_names(v, "labels")
    return v


def label_names(labels: Labels) -> list[str]:
    return list(labels)


def check_label_count(labels: Labels, multi_label: bool, max_labels: int, prefix: str = "labels") -> None:
    n = len(labels)
    low = 1 if multi_label else 2
    if n < low:
        need = "at least 1 label" if multi_label else "at least 2 labels for single-label classification"
        raise ValueError(f"{prefix}: {need}")
    if n > max_labels:
        raise ValueError(f"{prefix}: too many labels (max {max_labels})")


def check_levels_count(levels: list[str], max_labels: int, prefix: str = "levels") -> None:
    if len(levels) < 2:
        raise ValueError(f"{prefix}: at least 2 levels")
    if len(levels) > max_labels:
        raise ValueError(f"{prefix}: too many levels (max {max_labels})")


def _media_ok(v: list[str] | None) -> list[str] | None:
    if v is not None and any(not s.strip() for s in v):
        raise ValueError("media entries must be non-empty strings")
    return v


# ------------------------------------------------------------------ requests


class _Base(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ClassifyRequest(_Base):
    """Classify one input against a label set (single-label or multi-label)."""

    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "examples": [
                {
                    "input": "Checkout is down",
                    "labels": ["billing", "technical"],
                    "instructions": "Which team should handle the message?",
                }
            ]
        },
    )

    input: Any = Field(description="Text or any JSON to classify")
    labels: Labels = Field(description="list of labels, or object label -> description")
    instructions: str | None = Field(None, description="Optional natural-language hint")
    multi_label: bool = Field(
        False, description="false: pick one label. true: score every label independently"
    )
    threshold: float | None = Field(
        None, ge=0.0, le=1.0, description="multi-label only: labels with score >= threshold are returned"
    )
    images: list[str] | None = Field(None, description="data: URLs (or http(s) when enabled)")
    videos: list[str] | None = Field(None, description="data: URLs (or http(s) when enabled)")
    model: str = Field(DEFAULT_MODEL, description="Model name")

    @field_validator("labels")
    @classmethod
    def _labels(cls, v: Any) -> Any:
        return check_labels_shape(v)

    @field_validator("images", "videos")
    @classmethod
    def _media(cls, v: list[str] | None) -> list[str] | None:
        return _media_ok(v)

    @model_validator(mode="after")
    def _threshold_needs_multi(self) -> ClassifyRequest:
        if self.threshold is not None and not self.multi_label:
            raise ValueError("threshold: only allowed with multi_label=true")
        return self


class ClassifyBatchRequest(_Base):
    """Classify many inputs against the same labels in one micro-batched pass (no media)."""

    inputs: list[Any] = Field(description="1..max_batch inputs (text or any JSON)")
    labels: Labels
    instructions: str | None = None
    multi_label: bool = False
    threshold: float | None = Field(None, ge=0.0, le=1.0)
    model: str = DEFAULT_MODEL

    @field_validator("labels")
    @classmethod
    def _labels(cls, v: Any) -> Any:
        return check_labels_shape(v)

    @field_validator("inputs")
    @classmethod
    def _inputs(cls, v: list[Any]) -> list[Any]:
        if not v:
            raise ValueError("inputs must contain at least one item")
        return v

    @model_validator(mode="after")
    def _threshold_needs_multi(self) -> ClassifyBatchRequest:
        if self.threshold is not None and not self.multi_label:
            raise ValueError("threshold: only allowed with multi_label=true")
        return self


class ScoreRequest(_Base):
    """Place one input on an ordinal scale."""

    input: Any = Field(description="Text or any JSON to score")
    levels: list[str] = Field(description="2..max_labels unique level descriptions, lowest first")
    instructions: str | None = None
    images: list[str] | None = None
    videos: list[str] | None = None
    model: str = DEFAULT_MODEL

    @field_validator("levels")
    @classmethod
    def _levels(cls, v: list[str]) -> list[str]:
        return _check_names(v, "levels")

    @field_validator("images", "videos")
    @classmethod
    def _media(cls, v: list[str] | None) -> list[str] | None:
        return _media_ok(v)


class ClassifierDef(_Base):
    """A saved classifier definition (PUT /v1/classifiers/{name})."""

    kind: Literal["classify", "score"] = "classify"
    labels: Labels | None = None
    levels: list[str] | None = None
    instructions: str | None = None
    multi_label: bool = False
    threshold: float | None = Field(None, ge=0.0, le=1.0)
    description: str | None = None

    @model_validator(mode="after")
    def _shape(self) -> ClassifierDef:
        if self.kind == "classify":
            if self.labels is None:
                raise ValueError("labels: field is required for kind=classify")
            if self.levels is not None:
                raise ValueError("levels: not allowed for kind=classify")
            check_labels_shape(self.labels)
            if self.threshold is not None and not self.multi_label:
                raise ValueError("threshold: only allowed with multi_label=true")
        else:
            if self.levels is None:
                raise ValueError("levels: field is required for kind=score")
            if self.labels is not None:
                raise ValueError("labels: not allowed for kind=score")
            _check_names(self.levels, "levels")
            if self.multi_label or self.threshold is not None:
                raise ValueError("multi_label/threshold: not allowed for kind=score")
        return self


class ClassifierRunRequest(_Base):
    """Run a saved classifier on one input."""

    input: Any = Field(description="Text or any JSON")
    images: list[str] | None = None
    videos: list[str] | None = None
    threshold: float | None = Field(None, ge=0.0, le=1.0, description="multi-label classifiers only")

    @field_validator("images", "videos")
    @classmethod
    def _media(cls, v: list[str] | None) -> list[str] | None:
        return _media_ok(v)


class ClassifierBatchRequest(_Base):
    """Run a saved classifier on many inputs in one micro-batched pass."""

    inputs: list[Any]
    threshold: float | None = Field(None, ge=0.0, le=1.0)

    @field_validator("inputs")
    @classmethod
    def _inputs(cls, v: list[Any]) -> list[Any]:
        if not v:
            raise ValueError("inputs must contain at least one item")
        return v


# ------------------------------------------------------------------ responses


class ClassifyBody(BaseModel):
    """The classify result (shared by single and batch responses)."""

    model_config = ConfigDict(extra="allow")
    model: str
    multi_label: bool
    label: str | None = Field(None, description="single-label: the chosen label")
    confidence: float | None = Field(None, description="single-label: probability of `label`")
    labels: list[str] | None = Field(
        None, description="multi-label: labels with score >= threshold, best first"
    )
    threshold: float | None = Field(None, description="multi-label: the threshold applied")
    scores: dict[str, float] = Field(description="label -> probability")


class ClassifyResult(ClassifyBody):
    usage: Usage
    timing: Timing


class ClassifyResponse(ClassifyResult):
    request_id: str
    classifier: str | None = Field(None, description="name of the saved classifier, when used")


class ClassifyBatchResponse(BaseModel):
    model_config = ConfigDict(extra="allow")
    batch_ms: float
    results: list[ClassifyResult]
    request_id: str
    classifier: str | None = None


class ScoreBody(BaseModel):
    model_config = ConfigDict(extra="allow")
    model: str
    score: float = Field(description="expected level index")
    level: str
    level_index: int
    confidence: float
    distribution: dict[str, float] = Field(description="level text -> probability")


class ScoreResponse(ScoreBody):
    usage: Usage
    timing: Timing
    request_id: str
    classifier: str | None = None


class ClassifierInfo(BaseModel):
    """A stored classifier definition."""

    model_config = ConfigDict(extra="allow")
    name: str
    kind: Literal["classify", "score"]
    labels: Labels | None = None
    levels: list[str] | None = None
    instructions: str | None = None
    multi_label: bool = False
    threshold: float | None = None
    description: str | None = None
    created_at: str
    updated_at: str


class ClassifierList(BaseModel):
    classifiers: list[ClassifierInfo]


class ClassifierDeleted(BaseModel):
    deleted: str


# ------------------------------------------------------------------ translation (normative)


def classify_questions(
    labels: Labels, instructions: str | None, multi_label: bool
) -> dict[str, dict[str, Any]]:
    """labels -> SystemOne `questions` mapping exactly as the contract specifies."""
    if not multi_label:
        criteria = dict(labels) if isinstance(labels, dict) else {name: name for name in labels}
        q: dict[str, Any] = {"type": "choice", "criteria": criteria}
        if instructions is not None:
            q["instructions"] = instructions
        return {SINGLE_QID: q}
    prefix = f"{instructions} " if instructions is not None else ""
    questions: dict[str, dict[str, Any]] = {}
    for name in labels:
        q = {"type": "noul", "instructions": f'{prefix}Does the label "{name}" apply?'}
        if isinstance(labels, dict) and labels[name] != name:
            q["criteria"] = {"true": labels[name]}
        questions[name] = q
    return questions


def score_questions(levels: list[str], instructions: str | None) -> dict[str, dict[str, Any]]:
    q: dict[str, Any] = {"type": "score", "criteria": list(levels)}
    if instructions is not None:
        q["instructions"] = instructions
    return {SCORE_QID: q}


def _to_systemone(
    state: Any,
    questions: dict[str, dict[str, Any]],
    model: str,
    images: list[str] | None = None,
    videos: list[str] | None = None,
) -> SystemOneRequest:
    return SystemOneRequest(
        model=model,
        state=state,
        questions={k: Question(**v) for k, v in questions.items()},
        images=images,
        videos=videos,
    )


def classify_to_systemone(
    state: Any,
    labels: Labels,
    instructions: str | None = None,
    multi_label: bool = False,
    model: str = DEFAULT_MODEL,
    images: list[str] | None = None,
    videos: list[str] | None = None,
) -> SystemOneRequest:
    return _to_systemone(state, classify_questions(labels, instructions, multi_label), model, images, videos)


def score_to_systemone(
    state: Any,
    levels: list[str],
    instructions: str | None = None,
    model: str = DEFAULT_MODEL,
    images: list[str] | None = None,
    videos: list[str] | None = None,
) -> SystemOneRequest:
    return _to_systemone(state, score_questions(levels, instructions), model, images, videos)


def resolve_threshold(threshold: float | None, cfg: Config) -> float:
    return cfg.classify_threshold if threshold is None else threshold


def classify_result(
    answers: dict[str, Any],
    labels: Labels,
    multi_label: bool,
    threshold: float | None = None,
    model: str = DEFAULT_MODEL,
) -> dict[str, Any]:
    """SystemOne answers -> classify result body (no usage/timing/request_id)."""
    if not multi_label:
        ans = answers[SINGLE_QID]
        scores = {k: float(v) for k, v in ans["probabilities"].items()}
        label = ans["choice"]
        conf = ans.get("confidence")
        return {
            "model": model,
            "multi_label": False,
            "label": label,
            "confidence": float(conf) if conf is not None else scores.get(label, 0.0),
            "scores": scores,
        }
    thr = 0.5 if threshold is None else threshold
    scores = {name: float(answers[name]["noul"]) for name in labels}
    picked = sorted((n for n in labels if scores[n] >= thr), key=lambda n: -scores[n])  # stable on ties
    return {"model": model, "multi_label": True, "labels": picked, "scores": scores, "threshold": thr}


def score_result(answers: dict[str, Any], levels: list[str], model: str = DEFAULT_MODEL) -> dict[str, Any]:
    """SystemOne answers -> score result body (no usage/timing/request_id)."""
    ans = answers[SCORE_QID]
    probs = {int(k): float(v) for k, v in ans["probabilities"].items()}
    dist = {level: probs.get(i, 0.0) for i, level in enumerate(levels)}
    best = max(range(len(levels)), key=lambda i: (probs.get(i, 0.0), -i))
    conf = ans.get("confidence")
    return {
        "model": model,
        "score": sum(i * p for i, p in probs.items()),
        "level": levels[best],
        "level_index": best,
        "confidence": float(conf) if conf is not None else probs.get(best, 0.0),
        "distribution": dist,
    }
