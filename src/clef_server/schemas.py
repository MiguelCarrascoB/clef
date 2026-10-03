"""pydantic v2 request/response models + validation helpers for the clef HTTP API."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from .config import Config

DEFAULT_MODEL = "clef-flash"


class Question(BaseModel):
    """One decision question. `criteria` shape depends on `type`."""

    type: Literal["choice", "score", "noul"] = Field(
        description="choice = pick one option, score = ordinal scale, noul = yes/no"
    )
    instructions: str | None = Field(None, description="Optional natural-language hint for this question")
    criteria: Any = Field(
        None,
        description=(
            "choice: non-empty object option->description. score: non-empty array of level descriptions "
            "(index = level). noul: optional object with only 'true'/'false' keys."
        ),
    )

    @model_validator(mode="after")
    def _check_criteria(self) -> Question:
        c = self.criteria
        if self.type == "choice":
            if not isinstance(c, dict) or not c:
                raise ValueError("criteria must be a non-empty object of option -> description")
            if not all(isinstance(k, str) and k and isinstance(v, str) for k, v in c.items()):
                raise ValueError("criteria must map non-empty string options to string descriptions")
        elif self.type == "score":
            if not isinstance(c, list) or not c:
                raise ValueError("criteria must be a non-empty list")
            if not all(isinstance(v, str) for v in c):
                raise ValueError("criteria must be a list of strings")
        elif c is not None:
            if not isinstance(c, dict):
                raise ValueError("criteria must be an object with only 'true'/'false' keys")
            if set(c) - {"true", "false"} or not all(isinstance(v, str) for v in c.values()):
                raise ValueError("criteria may only contain string 'true' and 'false' keys")
        return self


class SystemOneRequest(BaseModel):
    """SystemOne decision request: free-form `state` + questions (+ optional media)."""

    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "examples": [
                {
                    "model": DEFAULT_MODEL,
                    "state": {
                        "ticket": {"text": "Checkout errors, orders blocked.", "customers_affected": 1200}
                    },
                    "questions": {
                        "department": {
                            "type": "choice",
                            "instructions": "Which team should handle it?",
                            "criteria": {"billing": "Payments or invoices", "technical": "Bugs or outages"},
                        },
                        "urgency": {"type": "score", "criteria": ["Can wait", "This week", "Today"]},
                        "outage": {"type": "noul", "instructions": "Is a service down?"},
                    },
                }
            ]
        },
    )

    model: str = Field(DEFAULT_MODEL, description="Model name (only clef-flash is served)")
    state: Any = Field(description="Text or any JSON the decisions are based on")
    questions: dict[str, Question] = Field(description="question id -> question (at least one)")
    images: list[str] | None = Field(
        None, description="data: URLs (or http(s) URLs when CLEF_ALLOW_URL_FETCH=1)"
    )
    videos: list[str] | None = Field(
        None, description="data: URLs (mp4/webm/mov/mkv) or http(s) URLs when enabled"
    )

    @field_validator("questions")
    @classmethod
    def _check_questions(cls, v: dict[str, Question]) -> dict[str, Question]:
        if not v:
            raise ValueError("at least one question is required")
        if any(not k.strip() for k in v):
            raise ValueError("question ids must be non-empty strings")
        return v

    @field_validator("images", "videos")
    @classmethod
    def _check_media(cls, v: list[str] | None) -> list[str] | None:
        if v is not None and any(not s.strip() for s in v):
            raise ValueError("media entries must be non-empty strings")
        return v

    def to_record(self, images: list[Any] | None = None, videos: list[Any] | None = None) -> dict[str, Any]:
        """Sanitized engine record (ALL media of the request in this one record)."""
        questions: dict[str, Any] = {}
        for qid, q in self.questions.items():
            item: dict[str, Any] = {"type": q.type}
            if q.instructions is not None:
                item["instructions"] = q.instructions
            if q.criteria:
                item["criteria"] = q.criteria
            questions[qid] = item
        return {
            "model": self.model,
            "state": self.state,
            "questions": questions,
            "images": images or None,
            "videos": videos or None,
        }


class BatchRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    batch: list[SystemOneRequest] = Field(description="1..max_batch SystemOne requests; media allowed")

    @field_validator("batch")
    @classmethod
    def _check_batch(cls, v: list[SystemOneRequest]) -> list[SystemOneRequest]:
        if not v:
            raise ValueError("batch must contain at least one request")
        return v


def check_limits(req: SystemOneRequest, cfg: Config, prefix: str = "") -> None:
    """Config-dependent limits; raises ValueError with a path-prefixed message."""
    if len(req.questions) > cfg.max_questions:
        raise ValueError(f"{prefix}questions: too many questions (max {cfg.max_questions})")
    if len(req.images or []) > cfg.max_images:
        raise ValueError(f"{prefix}images: too many images (max {cfg.max_images})")
    if len(req.videos or []) > cfg.max_videos:
        raise ValueError(f"{prefix}videos: too many videos (max {cfg.max_videos})")


def check_batch_size(batch: BatchRequest, cfg: Config) -> None:
    if len(batch.batch) > cfg.max_batch:
        raise ValueError(f"batch: too many requests (max {cfg.max_batch})")


def format_errors(errors: list[Any], limit: int = 5) -> str:
    """pydantic/FastAPI error list -> 'batch[3].questions.urgency: criteria must be a non-empty list'."""
    out: list[str] = []
    for err in errors[:limit]:
        loc = list(err.get("loc", ()))
        if loc[:1] == ["body"]:
            loc = loc[1:]
        path = ""
        for part in loc:
            path += f"[{part}]" if isinstance(part, int) else (f".{part}" if path else str(part))
        typ, msg = err.get("type", ""), str(err.get("msg", "invalid value"))
        if typ == "json_invalid":
            path, msg = "", "request body is not valid JSON"
        elif typ in ("model_attributes_type", "dict_type") and not loc:
            path, msg = "", "request body must be a JSON object (Content-Type: application/json)"
        elif typ == "missing":
            msg = "field is required"
        elif typ == "extra_forbidden":
            msg = "unknown field (not allowed)"
        msg = msg.removeprefix("Value error, ")
        out.append(f"{path}: {msg}" if path else msg)
    if len(errors) > limit:
        out.append(f"... and {len(errors) - limit} more")
    return "; ".join(out)


# ------------------------------------------------------------------ responses (OpenAPI docs)


class Usage(BaseModel):
    model_config = ConfigDict(extra="allow")
    input_tokens: int = 0
    output_tokens: int = 0


class Timing(BaseModel):
    model_config = ConfigDict(extra="allow")
    queue_ms: float | None = None
    forward_ms: float | None = None
    total_ms: float | None = None
    batch_size: int | None = None


class SystemOneResponse(BaseModel):
    model_config = ConfigDict(extra="allow")
    model: str
    answers: dict[str, dict[str, Any]] = Field(description="question id -> answer (choice/score/noul object)")
    usage: Usage
    timing: Timing


class BatchResponse(BaseModel):
    batch_ms: float
    results: list[SystemOneResponse]


class ErrorBody(BaseModel):
    detail: str
    request_id: str
