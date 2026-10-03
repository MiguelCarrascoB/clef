"""Pure tests for the classification models and the normative classify/score <-> SystemOne translation."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from clef_server.classify import (
    ClassifierDef,
    ClassifyBatchRequest,
    ClassifyRequest,
    ScoreRequest,
    check_label_count,
    check_levels_count,
    classify_questions,
    classify_result,
    classify_to_systemone,
    score_questions,
    score_result,
    score_to_systemone,
)

# ------------------------------------------------------------------ translation


def test_single_label_list():
    q = classify_questions(["billing", "technical"], None, False)
    assert q == {"label": {"type": "choice", "criteria": {"billing": "billing", "technical": "technical"}}}
    assert "instructions" not in q["label"]


def test_single_label_dict_and_instructions():
    labels = {"billing": "Payments or invoices", "technical": "Bugs"}
    q = classify_questions(labels, "Which team?", False)
    assert q == {
        "label": {"type": "choice", "criteria": labels, "instructions": "Which team?"},
    }
    assert q["label"]["criteria"] is not labels  # copied


def test_multi_label_list_without_instructions():
    q = classify_questions(["a", "b"], None, True)
    assert q == {
        "a": {"type": "noul", "instructions": 'Does the label "a" apply?'},
        "b": {"type": "noul", "instructions": 'Does the label "b" apply?'},
    }


def test_multi_label_with_instructions_and_descriptions():
    q = classify_questions({"billing": "Payments", "same": "same"}, "Read the ticket.", True)
    assert q["billing"] == {
        "type": "noul",
        "instructions": 'Read the ticket. Does the label "billing" apply?',
        "criteria": {"true": "Payments"},
    }
    assert q["same"] == {"type": "noul", "instructions": 'Read the ticket. Does the label "same" apply?'}


def test_score_questions():
    assert score_questions(["low", "high"], None) == {"score": {"type": "score", "criteria": ["low", "high"]}}
    assert score_questions(["low", "high"], "How urgent?")["score"]["instructions"] == "How urgent?"


def test_to_systemone_record_matches_hand_built():
    req = classify_to_systemone({"t": 1}, ["a", "b"], "Q?", False, model="m")
    rec = req.to_record()
    assert rec == {
        "model": "m",
        "state": {"t": 1},
        "questions": {
            "label": {"type": "choice", "instructions": "Q?", "criteria": {"a": "a", "b": "b"}},
        },
        "images": None,
        "videos": None,
    }
    sc = score_to_systemone("x", ["l", "h"]).to_record()
    assert sc["questions"] == {"score": {"type": "score", "criteria": ["l", "h"]}} and sc["state"] == "x"
    assert sc["model"] == "clef-flash"


def test_media_passthrough():
    req = classify_to_systemone("x", ["a", "b"], images=["data:image/png;base64,AA"], videos=None)
    assert req.images == ["data:image/png;base64,AA"] and req.videos is None


# ------------------------------------------------------------------ answers -> results


def test_single_result():
    answers = {
        "label": {
            "type": "choice",
            "choice": "technical",
            "confidence": 0.96,
            "probabilities": {"billing": 0.04, "technical": 0.96},
        }
    }
    out = classify_result(answers, ["billing", "technical"], False, model="clef-flash")
    assert out == {
        "model": "clef-flash",
        "multi_label": False,
        "label": "technical",
        "confidence": 0.96,
        "scores": {"billing": 0.04, "technical": 0.96},
    }


def test_single_result_confidence_falls_back_to_probability():
    answers = {"label": {"choice": "a", "probabilities": {"a": 0.7, "b": 0.3}}}
    assert classify_result(answers, ["a", "b"], False)["confidence"] == 0.7


def test_multi_threshold_and_ordering():
    answers = {n: {"type": "noul", "noul": p} for n, p in {"a": 0.5, "b": 0.9, "c": 0.49, "d": 0.7}.items()}
    out = classify_result(answers, ["a", "b", "c", "d"], True, 0.5)
    assert out["labels"] == ["b", "d", "a"]  # >= threshold inclusive, best first
    assert out["scores"] == {"a": 0.5, "b": 0.9, "c": 0.49, "d": 0.7}  # request order kept
    assert out["threshold"] == 0.5 and out["multi_label"] is True
    assert "label" not in out and "confidence" not in out
    assert classify_result(answers, ["a", "b", "c", "d"], True, 0.95)["labels"] == []
    assert classify_result(answers, ["a", "b", "c", "d"], True, 0.0)["labels"] == ["b", "d", "a", "c"]


def test_multi_ties_keep_label_order():
    answers = {n: {"noul": 0.8} for n in ("z", "y", "x")}
    assert classify_result(answers, ["z", "y", "x"], True, 0.5)["labels"] == ["z", "y", "x"]


def test_score_result():
    answers = {
        "score": {
            "type": "score",
            "score": 1.78,
            "confidence": 0.86,
            "legend": {"0": "low", "1": "medium", "2": "high"},
            "probabilities": {"0": 0.07, "1": 0.07, "2": 0.86},
        }
    }
    out = score_result(answers, ["low", "medium", "high"])
    assert out["level"] == "high" and out["level_index"] == 2 and out["confidence"] == 0.86
    assert out["distribution"] == {"low": 0.07, "medium": 0.07, "high": 0.86}
    assert out["score"] == pytest.approx(0.07 + 2 * 0.86)


def test_score_result_tie_picks_lowest_index():
    answers = {"score": {"probabilities": {"0": 0.5, "1": 0.5}}}
    assert score_result(answers, ["a", "b"])["level_index"] == 0


# ------------------------------------------------------------------ validation


@pytest.mark.parametrize(
    "labels",
    [["a", "a"], ["a", ""], ["a", "  "], {"a": "x", "": "y"}, {"a": 5}, [1, 2], "ab"],
)
def test_bad_labels_rejected(labels):
    with pytest.raises(ValidationError):
        ClassifyRequest(input="x", labels=labels)


def test_unknown_field_forbidden():
    with pytest.raises(ValidationError):
        ClassifyRequest(input="x", labels=["a", "b"], bogus=1)
    with pytest.raises(ValidationError):
        ScoreRequest(input="x", levels=["a", "b"], labels=["a"])


def test_input_required():
    with pytest.raises(ValidationError):
        ClassifyRequest(labels=["a", "b"])


def test_threshold_only_with_multi_label():
    with pytest.raises(ValidationError, match="threshold"):
        ClassifyRequest(input="x", labels=["a", "b"], threshold=0.3)
    with pytest.raises(ValidationError, match="threshold"):
        ClassifyBatchRequest(inputs=["x"], labels=["a", "b"], threshold=0.3)
    assert ClassifyRequest(input="x", labels=["a"], multi_label=True, threshold=0.3).threshold == 0.3


@pytest.mark.parametrize("thr", [-0.1, 1.1])
def test_threshold_range(thr):
    with pytest.raises(ValidationError):
        ClassifyRequest(input="x", labels=["a"], multi_label=True, threshold=thr)


def test_label_counts():
    check_label_count(["a", "b"], False, 2)
    check_label_count(["a"], True, 1)
    with pytest.raises(ValueError, match="at least 2"):
        check_label_count(["a"], False, 64)
    with pytest.raises(ValueError, match="at least 1"):
        check_label_count([], True, 64)
    with pytest.raises(ValueError, match="max 2"):
        check_label_count(["a", "b", "c"], True, 2)
    with pytest.raises(ValueError, match="max 2"):
        check_label_count({"a": "", "b": "", "c": ""}, False, 2)


def test_level_counts_and_shape():
    check_levels_count(["a", "b"], 2)
    with pytest.raises(ValueError, match="at least 2"):
        check_levels_count(["a"], 64)
    with pytest.raises(ValueError, match="max 2"):
        check_levels_count(["a", "b", "c"], 2)
    with pytest.raises(ValidationError):
        ScoreRequest(input="x", levels=["a", "a"])


def test_batch_needs_inputs():
    with pytest.raises(ValidationError):
        ClassifyBatchRequest(inputs=[], labels=["a", "b"])


def test_classifier_def_rules():
    d = ClassifierDef(kind="classify", labels=["a", "b"], multi_label=True, threshold=0.4)
    assert d.threshold == 0.4
    with pytest.raises(ValidationError, match="labels"):
        ClassifierDef(kind="classify")
    with pytest.raises(ValidationError, match="levels"):
        ClassifierDef(kind="score")
    with pytest.raises(ValidationError, match="threshold"):
        ClassifierDef(kind="classify", labels=["a", "b"], threshold=0.4)
    with pytest.raises(ValidationError):
        ClassifierDef(kind="score", levels=["a", "b"], labels=["a"])
    with pytest.raises(ValidationError):
        ClassifierDef(kind="score", levels=["a", "b"], multi_label=True)
    with pytest.raises(ValidationError):
        ClassifierDef(kind="other", labels=["a", "b"])
