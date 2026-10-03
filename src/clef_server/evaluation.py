"""Built-in evaluation: accuracy, per-label P/R/F1, confusion matrix and calibration for a labelled dataset.

Three layers:

* ``compute_metrics`` and friends: pure functions over ``[{gold, scores}]`` rows (no FastAPI, no engine).
  ``scores`` is the ``scores`` mapping /v1/classify returns. This is the ONE implementation of the metric
  math; the web console posts its collected scores to ``POST /v1/evaluate/metrics`` rather than redoing it.
* ``POST /v1/evaluate`` runs inference (chunks of ``max_batch`` through the normal inference path) and then
  the metrics.
* an ``evaluate`` job kind (``ctx.extra["job_kinds"]``) for datasets too big for one request.

See docs/evaluation.md.
"""

from __future__ import annotations

import json
import math
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from fastapi import APIRouter, Request
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from .appctx import AppContext
from .classify import (
    Labels,
    check_label_count,
    check_labels_shape,
    classify_result,
    classify_to_systemone,
    label_names,
    resolve_threshold,
)
from .schemas import DEFAULT_MODEL

EPS = 1e-12
INPUT_PREVIEW_CHARS = 200
COVERAGE_STEPS = 100  # thresholds 0.00, 0.01 .. 1.00
TARGET_ACCURACIES = (0.8, 0.9, 0.95, 0.98, 0.99)

# ------------------------------------------------------------------ pure metrics


def _prf(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return p, r, (2 * p * r / (p + r) if p + r else 0.0)


def _clip01(x: Any) -> float:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return 0.0
    if math.isnan(v):
        return 0.0
    return min(1.0, max(0.0, v))


def _gold_list(gold: Any, multi_label: bool, names: list[str], idx: int) -> list[str]:
    """Validate one gold value against the label set. Raises ValueError naming the row."""
    if isinstance(gold, str):
        items = [gold]
    elif isinstance(gold, list | tuple) and all(isinstance(g, str) for g in gold):
        items = list(dict.fromkeys(gold))  # de-duplicate, keep order
    else:
        raise ValueError(
            f"rows[{idx}].gold: must be a label string" + (" or list of labels" if multi_label else "")
        )
    if not multi_label and len(items) != 1:
        raise ValueError(f"rows[{idx}].gold: exactly one label expected (multi_label is false)")
    known = set(names)
    for g in items:
        if g not in known:
            raise ValueError(f"rows[{idx}].gold: {g!r} is not in labels")
    return items


def _calibration(pairs: list[tuple[float, bool]], bins: int) -> dict[str, Any]:
    """Equal-width reliability bins over (confidence, hit) pairs; hit = prediction right / event occurred."""
    n = len(pairs)
    cnt = [0] * bins
    conf_sum = [0.0] * bins
    hit_sum = [0] * bins
    for c, hit in pairs:
        b = min(int(c * bins), bins - 1)
        cnt[b] += 1
        conf_sum[b] += c
        hit_sum[b] += 1 if hit else 0
    out_bins = []
    ece = mce = 0.0
    for b in range(bins):
        entry: dict[str, Any] = {
            "lo": b / bins,
            "hi": (b + 1) / bins,
            "count": cnt[b],
            "mean_confidence": conf_sum[b] / cnt[b] if cnt[b] else None,
            "accuracy": hit_sum[b] / cnt[b] if cnt[b] else None,
        }
        if cnt[b]:
            gap = abs(entry["accuracy"] - entry["mean_confidence"])
            ece += cnt[b] / n * gap
            mce = max(mce, gap)
        out_bins.append(entry)
    return {
        "bins": out_bins,
        "ece": ece,
        "mce": mce,
        "mean_confidence": sum(c for c, _ in pairs) / n if n else None,
    }


def _coverage_curve(pairs: list[tuple[float, bool]]) -> list[dict[str, Any]]:
    """For thresholds 0.00..1.00: share of rows with confidence >= t (coverage) and their accuracy."""
    n = len(pairs)
    ordered = sorted(pairs, key=lambda p: -p[0])
    hits_prefix = [0]
    for _, hit in ordered:
        hits_prefix.append(hits_prefix[-1] + (1 if hit else 0))
    curve = []
    k = n  # rows with confidence >= t; shrinks as t grows
    for s in range(COVERAGE_STEPS + 1):
        t = s / COVERAGE_STEPS
        while k > 0 and ordered[k - 1][0] < t - 1e-12:
            k -= 1
        curve.append(
            {
                "threshold": t,
                "n": k,
                "coverage": k / n if n else 0.0,
                "accuracy": hits_prefix[k] / k if k else None,
            }
        )
    return curve


def _auto_route(curve: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """For each target accuracy: the lowest threshold whose accuracy meets it (max coverage), or None."""
    out = []
    for target in TARGET_ACCURACIES:
        hit = next((p for p in curve if p["accuracy"] is not None and p["accuracy"] >= target), None)
        out.append(
            {
                "target_accuracy": target,
                "threshold": hit["threshold"] if hit else None,
                "coverage": hit["coverage"] if hit else None,
                "accuracy": hit["accuracy"] if hit else None,
            }
        )
    return out


def _averages(per_label: list[dict[str, Any]]) -> tuple[float, float]:
    """(macro F1 over labels that occur in gold or predictions, support-weighted F1)."""
    active = [r for r in per_label if r["support"] or r["predicted"]]
    macro = sum(r["f1"] for r in active) / len(active) if active else 0.0
    total = sum(r["support"] for r in per_label)
    weighted = sum(r["f1"] * r["support"] for r in per_label) / total if total else 0.0
    return macro, weighted


def _single_probs(scores: dict[str, Any], names: list[str]) -> list[float]:
    raw = [_clip01(scores.get(name, 0.0)) for name in names]
    s = sum(raw)
    return [v / s for v in raw] if s > 0 else [1.0 / len(names)] * len(names)


def _single(
    names: list[str], rows: list[dict[str, Any]], bins: int
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    pos = {n: i for i, n in enumerate(names)}
    k = len(names)
    cm = [[0] * k for _ in range(k)]
    pairs: list[tuple[float, bool]] = []
    preds: list[dict[str, Any]] = []
    correct = top2 = 0
    brier = nll = 0.0
    for i, row in enumerate(rows):
        gold = _gold_list(row.get("gold"), False, names, i)[0]
        probs = _single_probs(row.get("scores") or {}, names)
        order = sorted(range(k), key=lambda j: (-probs[j], j))  # ties: first label wins
        p_idx, g_idx = order[0], pos[gold]
        cm[g_idx][p_idx] += 1
        ok = p_idx == g_idx
        correct += ok
        top2 += g_idx in order[:2]
        conf = probs[p_idx]
        pairs.append((conf, ok))
        brier += sum((probs[j] - (1.0 if j == g_idx else 0.0)) ** 2 for j in range(k))
        nll -= math.log(max(probs[g_idx], EPS))
        preds.append({"index": i, "gold": gold, "predicted": names[p_idx], "confidence": conf, "correct": ok})
    n = len(rows)
    per_label = []
    for j, name in enumerate(names):
        tp = cm[j][j]
        support = sum(cm[j])
        predicted = sum(cm[r][j] for r in range(k))
        p, r, f = _prf(tp, predicted - tp, support - tp)
        per_label.append(
            {"label": name, "precision": p, "recall": r, "f1": f, "support": support, "predicted": predicted}
        )
    macro, weighted = _averages(per_label)
    curve = _coverage_curve(pairs)
    cal = _calibration(pairs, bins)
    metrics = {
        "accuracy": correct / n,
        "top2_accuracy": top2 / n if k > 1 else 1.0,
        "macro_f1": macro,
        "micro_f1": correct / n,  # single-label: micro P = R = F1 = accuracy
        "weighted_f1": weighted,
        "per_label": per_label,
        "confusion_matrix": {"labels": names, "matrix": cm},
        "calibration": {**cal, "brier": brier / n, "nll": nll / n, "accuracy": correct / n},
        "coverage_curve": curve,
        "auto_route": _auto_route(curve),
        "exact_match": None,
        "hamming_loss": None,
    }
    return metrics, preds


def _multi(
    names: list[str], rows: list[dict[str, Any]], bins: int, threshold: float
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    k = len(names)
    tp = [0] * k
    fp = [0] * k
    fn = [0] * k
    pairs: list[tuple[float, bool]] = []
    preds: list[dict[str, Any]] = []
    exact = 0
    brier = nll = 0.0
    for i, row in enumerate(rows):
        gold = set(_gold_list(row.get("gold"), True, names, i))
        scores = row.get("scores") or {}
        picked = []
        row_ok = True
        for j, name in enumerate(names):
            p = _clip01(scores.get(name, 0.0))
            pred, y = p >= threshold, name in gold
            if pred:
                picked.append((p, name))
            if pred and y:
                tp[j] += 1
            elif pred:
                fp[j] += 1
                row_ok = False
            elif y:
                fn[j] += 1
                row_ok = False
            pairs.append((p, y))
            brier += (p - (1.0 if y else 0.0)) ** 2
            nll -= math.log(max(p if y else 1.0 - p, EPS))
        exact += row_ok
        picked.sort(key=lambda t: -t[0])
        confs = {name: _clip01(scores.get(name, 0.0)) for name in names}
        preds.append(
            {
                "index": i,
                "gold": [n for n in names if n in gold],
                "predicted": [n for _, n in picked],
                "confidence": confs,
                "correct": row_ok,
            }
        )
    n = len(rows)
    per_label = []
    for j, name in enumerate(names):
        p, r, f = _prf(tp[j], fp[j], fn[j])
        per_label.append(
            {
                "label": name,
                "precision": p,
                "recall": r,
                "f1": f,
                "support": tp[j] + fn[j],
                "predicted": tp[j] + fp[j],
                "tp": tp[j],
                "fp": fp[j],
                "fn": fn[j],
                "tn": n - tp[j] - fp[j] - fn[j],
            }
        )
    macro, weighted = _averages(per_label)
    _, _, micro = _prf(sum(tp), sum(fp), sum(fn))
    cal = _calibration(pairs, bins)
    pooled = len(pairs)
    metrics = {
        "accuracy": None,
        "top2_accuracy": None,
        "macro_f1": macro,
        "micro_f1": micro,
        "weighted_f1": weighted,
        "per_label": per_label,
        "confusion_matrix": None,
        "calibration": {
            **cal,
            "brier": brier / pooled,
            "nll": nll / pooled,
            "accuracy": sum(1 for _, y in pairs if y) / pooled,  # base rate of positives
        },
        "coverage_curve": None,
        "auto_route": None,
        "exact_match": exact / n,
        "hamming_loss": (sum(fp) + sum(fn)) / (n * k),
    }
    return metrics, preds


def compute_metrics(
    labels: Labels,
    rows: list[dict[str, Any]],
    multi_label: bool = False,
    threshold: float | None = None,
    bins: int = 10,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """(metrics, per-row predictions). Raises ValueError on bad input (the API maps it to 400).

    rows: ``{"gold": str | list[str], "scores": {label: probability}}``. Single-label scores are
    re-normalised to sum to 1; multi-label scores are independent probabilities (threshold default 0.5).
    """
    names = label_names(labels)
    if not names or len(set(names)) != len(names) or any(not isinstance(x, str) or not x for x in names):
        raise ValueError("labels: need unique non-empty strings")
    if not multi_label and len(names) < 2:
        raise ValueError("labels: at least 2 labels for single-label evaluation")
    if not rows:
        raise ValueError("rows: at least one row is required")
    if not 2 <= bins <= 100:
        raise ValueError("bins: must be between 2 and 100")
    thr = 0.5 if threshold is None else threshold
    if multi_label:
        metrics, preds = _multi(names, rows, bins, thr)
    else:
        metrics, preds = _single(names, rows, bins)
    metrics = {"n": len(rows), "multi_label": multi_label, "labels": names, **metrics}
    if multi_label:
        metrics["threshold"] = thr
    return metrics, preds


# ------------------------------------------------------------------ API models


class _Base(BaseModel):
    model_config = ConfigDict(extra="forbid")


class MetricsRow(_Base):
    gold: str | list[str] = Field(description="gold label (multi_label: a list of labels, possibly empty)")
    scores: dict[str, float] = Field(description="the `scores` object /v1/classify returned for this row")


class MetricsRequest(_Base):
    """Metrics from already-collected scores. No inference."""

    labels: Labels = Field(description="list of labels, or object label -> description")
    rows: list[MetricsRow] = Field(min_length=1)
    multi_label: bool = False
    threshold: float | None = Field(None, ge=0.0, le=1.0, description="multi-label decision threshold")
    bins: int = Field(10, ge=2, le=100, description="reliability-diagram bins")
    include_predictions: bool = Field(False, description="return one prediction object per row (by index)")

    @field_validator("labels")
    @classmethod
    def _labels(cls, v: Any) -> Any:
        return check_labels_shape(v)


class EvalRow(_Base):
    input: Any = Field(description="Text or any JSON to classify")
    gold: str | list[str] = Field(description="gold label (multi_label: a list of labels)")


class EvaluateRequest(_Base):
    """Run inference on labelled rows, then compute the metrics."""

    labels: Labels | None = Field(None, description="label set; omit when `classifier` is given")
    classifier: str | None = Field(None, description="name of a saved classify classifier")
    instructions: str | None = None
    multi_label: bool = False
    threshold: float | None = Field(None, ge=0.0, le=1.0)
    rows: list[EvalRow] = Field(min_length=1)
    bins: int = Field(10, ge=2, le=100)
    include_predictions: bool = Field(False, description="return one prediction object per row")

    @field_validator("labels")
    @classmethod
    def _labels(cls, v: Any) -> Any:
        return None if v is None else check_labels_shape(v)

    @model_validator(mode="after")
    def _shape(self) -> EvaluateRequest:
        if (self.labels is None) == (self.classifier is None):
            raise ValueError("exactly one of labels or classifier is required")
        if self.classifier is not None and (self.instructions is not None or self.multi_label):
            raise ValueError("instructions/multi_label come from the saved classifier; do not send them")
        if self.threshold is not None and not self.multi_label and self.classifier is None:
            raise ValueError("threshold: only allowed with multi_label=true")
        return self


class MetricsResponse(BaseModel):
    model_config = ConfigDict(extra="allow")
    n: int
    multi_label: bool
    labels: list[str]
    macro_f1: float
    micro_f1: float
    weighted_f1: float
    per_label: list[dict[str, Any]]
    calibration: dict[str, Any]


class EvaluateResponse(MetricsResponse):
    predictions: list[dict[str, Any]] | None = None
    classifier: str | None = None
    batch_ms: float | None = None
    request_id: str


# ------------------------------------------------------------------ inference + glue


@dataclass
class Spec:
    """A fully resolved evaluation: the labels/options and the rows to run."""

    labels: Labels
    instructions: str | None
    multi_label: bool
    threshold: float | None
    rows: list[EvalRow]
    bins: int
    include_predictions: bool
    classifier: str | None = None


def preview(value: Any) -> str:
    text = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, default=str)
    return text if len(text) <= INPUT_PREVIEW_CHARS else text[: INPUT_PREVIEW_CHARS - 1] + "…"


def _row_dicts(rows: list[EvalRow], scores: list[dict[str, float]]) -> list[dict[str, Any]]:
    return [{"gold": r.gold, "scores": s} for r, s in zip(rows, scores, strict=True)]


def _attach_inputs(preds: list[dict[str, Any]], rows: list[EvalRow], offset: int = 0) -> list[dict[str, Any]]:
    for p in preds:
        p["index"] += offset
        p["input"] = preview(rows[p["index"]].input)
    return preds


def router(ctx: AppContext) -> APIRouter:
    cfg = ctx.cfg
    r = APIRouter(prefix="/v1", tags=["evaluation"], dependencies=ctx.auth, responses=ctx.errors)

    async def resolve(body: EvaluateRequest) -> Spec:
        """Request -> Spec: load the saved classifier, enforce label/row/gold validity. Raises ValueError."""
        labels, instructions, multi, thr = body.labels, body.instructions, body.multi_label, body.threshold
        if body.classifier is not None:
            doc = await run_in_threadpool(ctx.store.get, body.classifier)
            if doc is None:
                raise LookupError(f"classifier {body.classifier!r} not found")
            if doc.get("kind") != "classify":
                raise ValueError("classifier: only classify classifiers can be evaluated")
            labels, instructions = doc["labels"], doc.get("instructions")
            multi = bool(doc.get("multi_label"))
            thr = body.threshold if body.threshold is not None else doc.get("threshold")
        assert labels is not None
        check_label_count(labels, multi, cfg.max_labels)
        names = label_names(labels)
        for i, row in enumerate(body.rows):  # fail before spending GPU time
            _gold_list(row.gold, multi, names, i)
        return Spec(
            labels, instructions, multi, thr, body.rows, body.bins, body.include_predictions, body.classifier
        )

    async def classify_chunks(
        spec: Spec,
        request: Request | None,
        on_chunk: Callable[[int, list[dict[str, float]]], Any] | None = None,
        cancelled: Callable[[], bool] | None = None,
    ) -> list[dict[str, float]]:
        """Scores per row, inferring max_batch rows per forward through ctx.infer (request) or ctx.decide."""
        thr = resolve_threshold(spec.threshold, cfg) if spec.multi_label else None
        size = max(1, cfg.max_batch)
        out: list[dict[str, float]] = []
        tokens = 0
        for start in range(0, len(spec.rows), size):
            if cancelled is not None and cancelled():
                break
            chunk = spec.rows[start : start + size]
            sreqs = [
                classify_to_systemone(
                    x.input, spec.labels, spec.instructions, spec.multi_label, DEFAULT_MODEL
                )
                for x in chunk
            ]
            if request is not None:
                results = await ctx.infer(request, sreqs, batch=True, label="rows")
            else:
                results = await ctx.decide(sreqs, batch=True, label="rows")
            tokens += sum(int(res.get("usage", {}).get("input_tokens", 0)) for res in results)
            scores = [
                classify_result(res["answers"], spec.labels, spec.multi_label, thr, DEFAULT_MODEL)["scores"]
                for res in results
            ]
            out.extend(scores)
            if on_chunk is not None:
                await on_chunk(start, scores)
        if request is not None:  # infer() records the LAST chunk only; report the whole run
            rec = getattr(request.state, "rec", None)
            if isinstance(rec, dict):
                rec["n_records"] = len(out)
                rec["n_questions"] = len(out) * (len(label_names(spec.labels)) if spec.multi_label else 1)
                rec["input_tokens"] = tokens
        return out

    @r.post(
        "/evaluate/metrics",
        response_model=MetricsResponse,
        summary="Evaluation metrics from collected scores (no inference)",
    )
    async def evaluate_metrics(body: MetricsRequest) -> dict[str, Any]:
        try:
            if len(body.rows) > cfg.max_job_eval_rows:
                raise ValueError(f"rows: too many rows (max {cfg.max_job_eval_rows})")
            check_label_count(body.labels, body.multi_label, cfg.max_labels)
            rows = [{"gold": x.gold, "scores": x.scores} for x in body.rows]
            metrics, preds = await run_in_threadpool(
                compute_metrics, body.labels, rows, body.multi_label, body.threshold, body.bins
            )
        except Exception as exc:
            raise ctx.map_exception(exc) from exc
        if body.include_predictions:
            metrics["predictions"] = preds
        return metrics

    @r.post(
        "/evaluate",
        response_model=EvaluateResponse,
        response_model_exclude_none=True,
        summary="Classify labelled rows and report accuracy, F1, confusion matrix and calibration",
        dependencies=ctx.limited,
    )
    async def evaluate(body: EvaluateRequest, request: Request) -> dict[str, Any]:
        t0 = time.perf_counter()
        try:
            if len(body.rows) > cfg.max_eval_rows:
                raise ValueError(
                    f"rows: too many rows (max {cfg.max_eval_rows}); submit an `evaluate` job for larger sets"
                )
            try:
                spec = await resolve(body)
            except LookupError as exc:
                raise ctx.api_error(404, str(exc)) from exc
            scores = await classify_chunks(spec, request)
            thr = resolve_threshold(spec.threshold, cfg) if spec.multi_label else None
            metrics, preds = await run_in_threadpool(
                compute_metrics, spec.labels, _row_dicts(spec.rows, scores), spec.multi_label, thr, spec.bins
            )
        except Exception as exc:
            raise ctx.map_exception(exc) from exc
        out: dict[str, Any] = {
            **metrics,
            "batch_ms": round((time.perf_counter() - t0) * 1000, 1),
            "request_id": ctx.request_id(request),
        }
        if spec.classifier:
            out["classifier"] = spec.classifier
        if spec.include_predictions:
            out["predictions"] = _attach_inputs(preds, spec.rows)
        return out

    # ---- async job kind (plugs into the jobs API through ctx.extra; neither module imports the other)
    def validate_job(payload: dict[str, Any]) -> EvaluateRequest:
        from pydantic import ValidationError

        try:
            body = EvaluateRequest.model_validate(payload)
        except ValidationError as exc:
            from .schemas import format_errors

            raise ValueError(format_errors(exc.errors())) from exc
        if len(body.rows) > cfg.max_job_eval_rows:
            raise ValueError(f"rows: too many rows (max {cfg.max_job_eval_rows})")
        if body.labels is not None:
            check_label_count(body.labels, body.multi_label, cfg.max_labels)
            names = label_names(body.labels)
            for i, row in enumerate(body.rows):
                _gold_list(row.gold, body.multi_label, names, i)
        return body

    async def run_job(_ctx: AppContext, parsed: EvaluateRequest, job: Any) -> dict[str, Any]:
        spec = await resolve(parsed)  # saved classifiers resolve at run time; ValueError fails the job
        job.set_total(len(spec.rows))
        thr = resolve_threshold(spec.threshold, cfg) if spec.multi_label else None
        names = label_names(spec.labels)
        done: list[dict[str, float]] = []

        async def on_chunk(start: int, scores: list[dict[str, float]]) -> None:
            rows = _row_dicts(spec.rows[start : start + len(scores)], scores)
            _, preds = compute_metrics(spec.labels, rows, spec.multi_label, thr, 2)
            items = _attach_inputs(preds, spec.rows, offset=start)
            for item, sc in zip(items, scores, strict=True):
                item["scores"] = {k: sc.get(k, 0.0) for k in names}
            await job.add_items(items)
            done.extend(scores)

        await classify_chunks(spec, None, on_chunk, lambda: bool(job.cancelled))
        if not done:
            return {"n": 0, "cancelled": bool(job.cancelled)}
        metrics, _ = compute_metrics(
            spec.labels, _row_dicts(spec.rows[: len(done)], done), spec.multi_label, thr, spec.bins
        )
        if job.cancelled:
            metrics["cancelled"] = True
            metrics["partial"] = True
        if spec.classifier:
            metrics["classifier"] = spec.classifier
        return metrics

    ctx.extra.setdefault("job_kinds", {})["evaluate"] = {"validate": validate_job, "run": run_job}
    return r


__all__ = ["compute_metrics", "router", "EvaluateRequest", "MetricsRequest"]
