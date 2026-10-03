"""Evaluation metrics (pure), /v1/evaluate[/metrics] and the `evaluate` job kind (FakeEngine, no GPU)."""

from __future__ import annotations

import asyncio
import math

import pytest

from clef_server.evaluation import compute_metrics
from tests.unit.test_api import make

LABELS = ["a", "b"]
ROWS = [
    {"gold": "a", "scores": {"a": 0.9, "b": 0.1}},
    {"gold": "a", "scores": {"a": 0.6, "b": 0.4}},
    {"gold": "b", "scores": {"a": 0.7, "b": 0.3}},
    {"gold": "b", "scores": {"a": 0.2, "b": 0.8}},
]
DEPT = ["billing", "technical"]


@pytest.fixture
def api():
    client, eng, app = make()
    with client:
        yield client, eng, app


def test_single_label_metrics_by_hand() -> None:
    m, preds = compute_metrics(LABELS, ROWS)
    assert m["n"] == 4 and m["accuracy"] == 0.75 and m["micro_f1"] == 0.75
    a, b = m["per_label"]
    assert a["precision"] == pytest.approx(2 / 3) and a["recall"] == 1.0 and a["f1"] == pytest.approx(0.8)
    assert b["precision"] == 1.0 and b["recall"] == 0.5 and b["f1"] == pytest.approx(2 / 3)
    assert (a["support"], b["support"]) == (2, 2)
    assert m["macro_f1"] == pytest.approx((0.8 + 2 / 3) / 2)
    assert m["confusion_matrix"] == {"labels": LABELS, "matrix": [[2, 0], [1, 1]]}
    assert m["top2_accuracy"] == 1.0
    cal = m["calibration"]
    assert cal["brier"] == pytest.approx(0.35)
    assert cal["nll"] == pytest.approx(-(math.log(0.9) + math.log(0.6) + math.log(0.3) + math.log(0.8)) / 4)
    assert cal["ece"] == pytest.approx(0.35) and cal["mce"] == pytest.approx(0.7)
    assert len(cal["bins"]) == 10 and sum(b["count"] for b in cal["bins"]) == 4
    assert [p["correct"] for p in preds] == [True, True, False, True]
    assert preds[2]["predicted"] == "a" and preds[2]["confidence"] == pytest.approx(0.7)


def test_coverage_curve_and_auto_route() -> None:
    m, _ = compute_metrics(LABELS, ROWS)
    curve = m["coverage_curve"]
    assert len(curve) == 101
    at = {round(p["threshold"], 2): p for p in curve}
    assert at[0.0]["coverage"] == 1.0 and at[0.0]["accuracy"] == 0.75
    assert at[0.75]["n"] == 2 and at[0.75]["accuracy"] == 1.0  # 0.9 and 0.8 are both right
    assert at[0.95]["n"] == 0 and at[0.95]["accuracy"] is None
    route = {r["target_accuracy"]: r for r in m["auto_route"]}
    assert route[0.99]["threshold"] == pytest.approx(0.71) and route[0.99]["coverage"] == 0.5
    assert route[0.8]["threshold"] <= route[0.99]["threshold"]


def test_label_never_predicted_and_empty_bins() -> None:
    rows = [{"gold": "c", "scores": {"a": 0.5, "b": 0.3, "c": 0.2}}, {"gold": "a", "scores": {"a": 1}}]
    m, _ = compute_metrics(["a", "b", "c"], rows, bins=4)
    c = m["per_label"][2]
    assert c["support"] == 1 and c["predicted"] == 0 and c["precision"] == 0.0 and c["f1"] == 0.0
    b = m["per_label"][1]  # neither gold nor predicted: excluded from the macro average, no division error
    assert b["support"] == b["predicted"] == 0
    assert m["macro_f1"] == pytest.approx((m["per_label"][0]["f1"] + 0.0) / 2)
    assert any(x["count"] == 0 and x["accuracy"] is None for x in m["calibration"]["bins"])


def test_unicode_labels_and_errors() -> None:
    m, preds = compute_metrics(["facturación", "técnico"], [{"gold": "técnico", "scores": {"técnico": 0.8}}])
    assert preds[0]["predicted"] == "técnico" and m["accuracy"] == 1.0
    with pytest.raises(ValueError, match=r"rows\[0\].gold: 'nope' is not in labels"):
        compute_metrics(LABELS, [{"gold": "nope", "scores": {}}])
    with pytest.raises(ValueError, match="at least one row"):
        compute_metrics(LABELS, [])
    with pytest.raises(ValueError, match="at least 2 labels"):
        compute_metrics(["a"], ROWS[:1])
    with pytest.raises(ValueError, match="bins"):
        compute_metrics(LABELS, ROWS, bins=1)
    with pytest.raises(ValueError, match="exactly one label"):
        compute_metrics(LABELS, [{"gold": ["a", "b"], "scores": {}}])


def test_multi_label_metrics() -> None:
    rows = [
        {"gold": ["x"], "scores": {"x": 0.9, "y": 0.2, "z": 0.1}},
        {"gold": ["x", "y"], "scores": {"x": 0.8, "y": 0.4, "z": 0.1}},
        {"gold": [], "scores": {"x": 0.1, "y": 0.1, "z": 0.1}},
        {"gold": ["z"], "scores": {"x": 0.6, "y": 0.1, "z": 0.7}},
    ]
    m, preds = compute_metrics(["x", "y", "z"], rows, multi_label=True)
    assert m["exact_match"] == 0.5 and m["hamming_loss"] == pytest.approx(2 / 12)
    x, y, _z = m["per_label"]
    assert (x["tp"], x["fp"], x["fn"], x["tn"]) == (2, 1, 0, 1)
    assert x["precision"] == pytest.approx(2 / 3) and x["recall"] == 1.0
    assert (y["tp"], y["fn"]) == (0, 1) and y["recall"] == 0.0
    assert m["micro_f1"] == pytest.approx(0.75)
    assert m["confusion_matrix"] is None and m["coverage_curve"] is None and m["accuracy"] is None
    cal = m["calibration"]
    assert sum(b["count"] for b in cal["bins"]) == 12  # pooled over (row, label)
    assert 0 <= cal["ece"] <= 1 and cal["brier"] > 0
    assert preds[1]["predicted"] == ["x"] and preds[1]["gold"] == ["x", "y"] and preds[1]["correct"] is False
    m2, _ = compute_metrics(["x", "y", "z"], rows, multi_label=True, threshold=0.95)
    assert m2["threshold"] == 0.95 and m2["per_label"][0]["tp"] == 0


# ------------------------------------------------------------------ API


def test_metrics_endpoint(api) -> None:
    client, eng, _ = api
    r = client.post("/v1/evaluate/metrics", json={"labels": LABELS, "rows": ROWS, "bins": 5})
    assert r.status_code == 200, r.text
    assert r.json()["accuracy"] == 0.75 and len(r.json()["calibration"]["bins"]) == 5
    assert eng.calls == []  # no inference
    bad = client.post(
        "/v1/evaluate/metrics", json={"labels": LABELS, "rows": [{"gold": "zzz", "scores": {}}]}
    )
    assert bad.status_code == 400 and "not in labels" in bad.json()["detail"] and "request_id" in bad.json()
    assert client.post("/v1/evaluate/metrics", json={"labels": LABELS, "rows": []}).status_code == 400
    # dict labels (label -> description) are accepted
    ok = client.post("/v1/evaluate/metrics", json={"labels": {"a": "first", "b": "second"}, "rows": ROWS})
    assert ok.status_code == 200


def _rows(n: int) -> list[dict]:
    return [{"input": f"ticket {i}", "gold": "technical" if i % 4 else "billing"} for i in range(n)]


def test_evaluate_runs_inference_in_chunks(api) -> None:
    client, eng, _ = api
    r = client.post("/v1/evaluate", json={"labels": DEPT, "rows": _rows(8), "include_predictions": True})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["n"] == 8 and body["accuracy"] == pytest.approx(6 / 8)  # FakeEngine always says "technical"
    assert len(body["predictions"]) == 8 and body["predictions"][0]["input"] == "ticket 0"
    assert body["predictions"][0]["correct"] is False and body["request_id"]
    assert len(eng.calls) == 1


def test_evaluate_chunking_and_input_truncation() -> None:
    client, eng, _ = make(max_batch=3)
    rows = [{"input": "x" * 1000, "gold": "technical"}, *_rows(6)[1:]]
    with client:
        r = client.post("/v1/evaluate", json={"labels": DEPT, "rows": rows, "include_predictions": True})
        assert r.status_code == 200, r.text
        assert [len(c) for c in eng.calls] == [3, 3]
        assert len(r.json()["predictions"][0]["input"]) <= 200
        assert "predictions" not in client.post("/v1/evaluate", json={"labels": DEPT, "rows": rows}).json()


def test_evaluate_validation_and_caps() -> None:
    client, eng, _ = make(max_eval_rows=5)
    with client:
        url = "/v1/evaluate"
        too_many = client.post(url, json={"labels": DEPT, "rows": _rows(6)})
        assert too_many.status_code == 400 and "max 5" in too_many.json()["detail"]
        bad_gold = client.post(url, json={"labels": DEPT, "rows": [{"input": "x", "gold": "other"}]})
        assert bad_gold.status_code == 400 and "not in labels" in bad_gold.json()["detail"]
        assert eng.calls == []  # rejected before any inference
        both = client.post(url, json={"labels": DEPT, "classifier": "c", "rows": _rows(1)})
        assert both.status_code == 400
        missing = client.post(url, json={"classifier": "nope", "rows": _rows(1)})
        assert missing.status_code == 404


def test_evaluate_saved_classifier_and_multi_label(api) -> None:
    client, eng, _ = api
    put = client.put("/v1/classifiers/dept", json={"labels": DEPT, "instructions": "Team?"})
    assert put.status_code == 200
    r = client.post("/v1/evaluate", json={"classifier": "dept", "rows": _rows(4)})
    assert r.status_code == 200 and r.json()["classifier"] == "dept"
    eng.noul = {"billing": 0.9, "technical": 0.2}
    rows = [{"input": "a", "gold": ["billing"]}, {"input": "b", "gold": ["technical"]}]
    m = client.post("/v1/evaluate", json={"labels": DEPT, "multi_label": True, "rows": rows})
    assert m.status_code == 200, m.text
    assert m.json()["exact_match"] == 0.5 and m.json()["multi_label"] is True


# ------------------------------------------------------------------ job kind


class FakeJob:
    def __init__(self, cancel_after: int | None = None) -> None:
        self.total = 0
        self.items: list[dict] = []
        self.cancel_after = cancel_after

    def set_total(self, n: int) -> None:
        self.total = n

    async def add_items(self, items: list[dict]) -> None:
        self.items.extend(items)

    @property
    def cancelled(self) -> bool:
        return self.cancel_after is not None and len(self.items) >= self.cancel_after


def _kind(app):
    return app.state.ctx.extra["job_kinds"]["evaluate"]


def test_job_kind_registered_and_validates(api) -> None:
    _, _, app = api
    kind = _kind(app)
    parsed = kind["validate"]({"labels": DEPT, "rows": _rows(3)})
    assert parsed.rows[0].input == "ticket 0"
    for bad in (
        {"labels": DEPT, "rows": []},
        {"labels": DEPT, "rows": [{"input": "x", "gold": "zzz"}]},
        {"rows": _rows(1)},
        {"labels": DEPT, "rows": _rows(1), "bogus": 1},
    ):
        with pytest.raises(ValueError):
            kind["validate"](bad)


def test_job_row_cap() -> None:
    _, _, app = make(max_job_eval_rows=3)
    with pytest.raises(ValueError, match="max 3"):
        _kind(app)["validate"]({"labels": DEPT, "rows": _rows(4)})


def test_job_run_persists_items_and_returns_metrics() -> None:
    client, eng, app = make(max_batch=3)
    kind = _kind(app)
    parsed = kind["validate"]({"labels": DEPT, "rows": _rows(7)})
    job = FakeJob()
    with client:
        result = asyncio.run(kind["run"](app.state.ctx, parsed, job))
    assert job.total == 7 and len(job.items) == 7
    assert [it["index"] for it in job.items] == list(range(7))
    assert job.items[1]["input"] == "ticket 1" and job.items[1]["predicted"] == "technical"
    assert set(job.items[0]["scores"]) == set(DEPT)
    assert result["n"] == 7 and result["accuracy"] == pytest.approx(5 / 7)
    assert [len(c) for c in eng.calls] == [3, 3, 1]


def test_job_run_cancel_returns_partial() -> None:
    client, _, app = make(max_batch=2)
    kind = _kind(app)
    parsed = kind["validate"]({"labels": DEPT, "rows": _rows(8)})
    job = FakeJob(cancel_after=2)
    with client:
        result = asyncio.run(kind["run"](app.state.ctx, parsed, job))
    assert result["cancelled"] is True and result["partial"] is True and result["n"] == 2
    assert len(job.items) == 2
