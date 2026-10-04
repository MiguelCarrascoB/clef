"""Built-in evaluation: accuracy, per-label P/R/F1, confusion matrix and calibration for a labelled dataset.

Three layers:

* ``compute_metrics`` and friends: pure functions over ``[{gold, scores}]`` rows (no FastAPI, no engine).
  ``scores`` is the ``scores`` mapping /v1/classify returns. This is the ONE implementation of the metric
  math; the web console posts its collected scores to ``POST /v1/evaluate/metrics`` rather than redoing it.
* ``POST /v1/evaluate`` runs inference (chunks of ``max_batch`` through the normal inference path) and then
  the metrics.
* an ``evaluate`` job kind (``ctx.extra["job_kinds"]``) for datasets too big for one request: micro-batches
  through ``jobs.decide_resilient`` (engine-not-ready wait, per-row error isolation), resumable.

See docs/evaluation.md.
"""

from __future__ import annotations

import asyncio
import json
import math
import time
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


@dataclass
class _Quality:
    """Counts of scores that were repaired rather than rejected (reported as ``degenerate_rows`` etc.)."""

    rows: int = 0  # rows with at least one defaulted / clipped score (or an all-zero single-label row)
    defaulted: int = 0  # label keys missing from ``scores`` (taken as 0.0)
    clipped: int = 0  # values outside [0, 1] (clamped)

    def row(self, defaulted: int, clipped: int, all_zero: bool = False) -> None:
        self.defaulted += defaulted
        self.clipped += clipped
        if defaulted or clipped or all_zero:
            self.rows += 1

    def fields(self) -> dict[str, int]:
        return {
            "degenerate_rows": self.rows,
            "defaulted_scores": self.defaulted,
            "clipped_scores": self.clipped,
        }


def _row_scores(scores: Any, names: list[str], idx: int) -> tuple[list[float], int, int]:
    """(value per label, n defaulted, n clipped). Raises ValueError naming the row for unusable scores.

    Missing label keys count as 0.0 but a row with NO label key at all (e.g. ``Billing`` vs ``billing``), a
    non-numeric value or a NaN / infinite value is rejected instead of silently becoming a uniform guess.
    """
    where = f"rows[{idx}].scores"
    if not isinstance(scores, dict) or not any(n in scores for n in names):
        got = list(scores)[:5] if isinstance(scores, dict) else type(scores).__name__
        raise ValueError(
            f"{where}: none of the labels {names[:5]} appear as keys (got {got}); "
            "label keys are case-sensitive"
        )
    vals: list[float] = []
    defaulted = clipped = 0
    for name in names:
        if name not in scores:
            vals.append(0.0)
            defaulted += 1
            continue
        raw = scores[name]
        try:
            if isinstance(raw, bool):
                raise TypeError
            v = float(raw)
        except (TypeError, ValueError):
            raise ValueError(f"{where}[{name!r}]: not a number ({raw!r})") from None
        if not math.isfinite(v):
            raise ValueError(f"{where}[{name!r}]: not a finite number ({raw!r})")
        if v < 0.0 or v > 1.0:
            clipped += 1
            v = min(1.0, max(0.0, v))
        vals.append(v)
    return vals, defaulted, clipped


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


def _single_probs(raw: list[float]) -> list[float]:
    s = sum(raw)
    return [v / s for v in raw] if s > 0 else [1.0 / len(raw)] * len(raw)


def _single(
    names: list[str], rows: list[dict[str, Any]], bins: int
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    pos = {n: i for i, n in enumerate(names)}
    k = len(names)
    quality = _Quality()
    cm = [[0] * k for _ in range(k)]
    pairs: list[tuple[float, bool]] = []
    preds: list[dict[str, Any]] = []
    correct = top2 = 0
    brier = nll = 0.0
    for i, row in enumerate(rows):
        gold = _gold_list(row.get("gold"), False, names, i)[0]
        raw, defaulted, clipped = _row_scores(row.get("scores"), names, i)
        quality.row(defaulted, clipped, all_zero=sum(raw) <= 0)
        probs = _single_probs(raw)
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
        **quality.fields(),
    }
    return metrics, preds


def _multi(
    names: list[str], rows: list[dict[str, Any]], bins: int, threshold: float
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    k = len(names)
    quality = _Quality()
    tp = [0] * k
    fp = [0] * k
    fn = [0] * k
    pairs: list[tuple[float, bool]] = []
    preds: list[dict[str, Any]] = []
    exact = 0
    brier = nll = 0.0
    for i, row in enumerate(rows):
        gold = set(_gold_list(row.get("gold"), True, names, i))
        vals, defaulted, clipped = _row_scores(row.get("scores"), names, i)
        quality.row(defaulted, clipped)
        picked = []
        row_ok = True
        for j, name in enumerate(names):
            p = vals[j]
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
        confs = dict(zip(names, vals, strict=True))
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
        **quality.fields(),
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
    degenerate_rows: int = Field(
        0, description="rows with a missing label key (taken as 0), a clipped score or all-zero scores"
    )
    defaulted_scores: int = Field(0, description="label keys missing from `scores`, counted as 0.0")
    clipped_scores: int = Field(0, description="scores outside [0, 1] that were clamped")


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

    def resolve_sync(body: EvaluateRequest) -> Spec:
        """Request -> Spec: load the saved classifier, enforce label/row/gold validity.

        Raises ValueError (bad input) or LookupError (unknown classifier). Blocking: run it in a thread.
        """
        labels, instructions, multi, thr = body.labels, body.instructions, body.multi_label, body.threshold
        if body.classifier is not None:
            doc = ctx.store.get(body.classifier)
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

    async def resolve(body: EvaluateRequest) -> Spec:
        return await run_in_threadpool(resolve_sync, body)

    def build_requests(spec: Spec, rows: list[EvalRow]) -> list[Any]:
        return [
            classify_to_systemone(x.input, spec.labels, spec.instructions, spec.multi_label, DEFAULT_MODEL)
            for x in rows
        ]

    def to_scores(spec: Spec, res: dict[str, Any], thr: float | None) -> dict[str, float]:
        return classify_result(res["answers"], spec.labels, spec.multi_label, thr, DEFAULT_MODEL)["scores"]

    async def classify_chunks(spec: Spec, request: Request) -> list[dict[str, float]]:
        """Scores per row (request path): max_batch rows per forward through ctx.infer; any error fails it."""
        thr = resolve_threshold(spec.threshold, cfg) if spec.multi_label else None
        size = max(1, cfg.max_batch)
        out: list[dict[str, float]] = []
        tokens = 0
        for start in range(0, len(spec.rows), size):
            sreqs = build_requests(spec, spec.rows[start : start + size])
            results = await ctx.infer(request, sreqs, batch=True, label="rows")
            tokens += sum(int(res.get("usage", {}).get("input_tokens", 0)) for res in results)
            out.extend(to_scores(spec, res, thr) for res in results)
        rec = getattr(request.state, "rec", None)
        if isinstance(rec, dict):  # infer() records the LAST chunk only; report the whole run
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
    def validate_job(payload: dict[str, Any]) -> Spec:
        """Payload -> resolved Spec (saved classifier loaded, labels and gold checked). Raises ValueError.

        Runs at submit (400) and again when the job starts, so a classifier deleted in between fails the job
        with a clear message. Blocking: the jobs API calls it in a thread.
        """
        from pydantic import ValidationError

        try:
            body = EvaluateRequest.model_validate(payload)
        except ValidationError as exc:
            from .schemas import format_errors

            raise ValueError(format_errors(exc.errors())) from exc
        if len(body.rows) > cfg.max_job_eval_rows:
            raise ValueError(f"rows: too many rows (max {cfg.max_job_eval_rows})")
        try:
            return resolve_sync(body)
        except LookupError as exc:
            raise ValueError(str(exc)) from exc

    async def run_job(_ctx: AppContext, spec: Spec, job: Any) -> dict[str, Any]:
        """Classify ``spec.rows`` micro-batch by micro-batch through the jobs runner's resilient engine call.

        Rows the engine rejects (too long, OOM) become ``{"index", "error"}`` items and are left out of the
        metrics (``n_errors``). Resumable: items already stored are replayed into the metrics and skipped.
        """
        from .jobs import ItemError, JobCancelled, decide_resilient

        if spec.classifier is not None:  # snapshot is in `spec`; this only catches a deletion since validate
            try:
                await run_in_threadpool(_require_classifier, spec.classifier)
            except LookupError as exc:
                raise ValueError(str(exc)) from exc
        total = len(spec.rows)
        job.set_total(total)
        thr = resolve_threshold(spec.threshold, cfg) if spec.multi_label else None
        names = label_names(spec.labels)
        scored: list[tuple[int, dict[str, float]]] = []  # (row index, scores) of every row that has scores
        n_errors = 0
        start = int(job.resume_from)
        if start:  # resuming: rebuild the collected scores from what is already persisted
            for item in await job.stored_items():
                if not 0 <= item.get("index", -1) < start:
                    continue
                if "error" in item:
                    n_errors += 1
                elif "scores" in item:
                    scored.append((item["index"], item["scores"]))
        size = max(1, int(cfg.max_microbatch))  # job mode shares the engine: keep each forward short
        cancelled = False
        try:
            for lo in range(start, total, size):
                if job.cancelled:
                    cancelled = True
                    break
                hi = min(total, lo + size)
                built: list[tuple[int, Any]] = []
                for i in range(lo, hi):
                    try:
                        built.append((i, build_requests(spec, [spec.rows[i]])[0]))
                    except ValueError as exc:
                        built.append((i, ItemError(str(exc))))
                good = [r for _, r in built if not isinstance(r, ItemError)]
                results = iter(await decide_resilient(_ctx, job, good) if good else [])
                ok_idx: list[int] = []
                ok_scores: list[dict[str, float]] = []
                items: list[dict[str, Any]] = []
                for i, r in built:
                    out = r if isinstance(r, ItemError) else next(results)
                    if isinstance(out, ItemError):
                        n_errors += 1
                        row = spec.rows[i]
                        items.append(
                            {"index": i, "input": preview(row.input), "gold": row.gold, "error": out.message}
                        )
                    else:
                        ok_idx.append(i)
                        ok_scores.append(to_scores(spec, out, thr))
                if ok_idx:
                    pairs = zip(ok_idx, ok_scores, strict=True)
                    rows = [{"gold": spec.rows[i].gold, "scores": s} for i, s in pairs]
                    _, preds = await asyncio.to_thread(
                        compute_metrics, spec.labels, rows, spec.multi_label, thr, 2
                    )
                    by_index = {}
                    for i, p, sc in zip(ok_idx, preds, ok_scores, strict=True):
                        p["index"] = i
                        p["input"] = preview(spec.rows[i].input)
                        p["scores"] = {k: sc.get(k, 0.0) for k in names}
                        by_index[i] = p
                    scored.extend(zip(ok_idx, ok_scores, strict=True))
                    items.extend(by_index.values())
                items.sort(key=lambda it: it["index"])
                await job.add_items(items)
                await asyncio.sleep(0)  # let interactive requests reach the engine queue
        except JobCancelled:
            cancelled = True
        cancelled = cancelled or bool(job.cancelled)
        if not scored:
            return {"n": 0, "n_errors": n_errors, "cancelled": cancelled}
        scored.sort(key=lambda t: t[0])
        rows = [{"gold": spec.rows[i].gold, "scores": s} for i, s in scored]
        metrics, _ = await asyncio.to_thread(
            compute_metrics, spec.labels, rows, spec.multi_label, thr, spec.bins
        )
        metrics["n_errors"] = n_errors
        if cancelled:
            metrics["cancelled"] = True
            metrics["partial"] = True
        if spec.classifier:
            metrics["classifier"] = spec.classifier
        return metrics

    def _require_classifier(name: str) -> None:
        if ctx.store.get(name) is None:
            raise LookupError(f"classifier {name!r} not found (deleted since the job was submitted)")

    ctx.extra.setdefault("job_kinds", {})["evaluate"] = {
        "validate": validate_job,
        "run": run_job,
        "resumable": True,
    }
    return r


__all__ = ["compute_metrics", "router", "EvaluateRequest", "MetricsRequest"]
