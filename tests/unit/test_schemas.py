"""Request validation: schemas.py models + error formatting."""

from __future__ import annotations

import copy
from typing import Any

import pytest
from pydantic import ValidationError

from clef_server.config import Config
from clef_server.schemas import BatchRequest, SystemOneRequest, check_batch_size, check_limits, format_errors

GOOD: dict[str, Any] = {
    "state": "hello",
    "questions": {
        "dept": {"type": "choice", "criteria": {"a": "A", "b": "B"}},
        "urg": {"type": "score", "instructions": "how urgent", "criteria": ["low", "high"]},
        "out": {"type": "noul"},
    },
}


def msg(data: dict[str, Any], cls: Any = SystemOneRequest) -> str:
    with pytest.raises(ValidationError) as ei:
        cls.model_validate(data)
    return format_errors(ei.value.errors())


def mutated(**q: Any) -> dict[str, Any]:
    d = copy.deepcopy(GOOD)
    d["questions"].update(q)
    return d


def test_valid_and_model_default():
    r = SystemOneRequest.model_validate(GOOD)
    assert r.model == "clef-flash"
    rec = r.to_record()
    assert rec["images"] is None and rec["videos"] is None
    assert "criteria" not in rec["questions"]["out"]
    assert rec["questions"]["urg"]["instructions"] == "how urgent"


@pytest.mark.parametrize("model", [5, None, ["x"]])
def test_model_must_be_string(model):
    assert "model" in msg({**GOOD, "model": model})


def test_state_required_and_any_json():
    assert "state: field is required" in msg({"questions": GOOD["questions"]})
    for state in ({"a": [1, 2]}, [1], 3, None):
        SystemOneRequest.model_validate({**GOOD, "state": state})


@pytest.mark.parametrize("crit", [{}, [], None, "x", ["a"], {"a": 1}, {"": "x"}])
def test_choice_criteria_rules(crit):
    assert "questions.q: criteria must" in msg(mutated(q={"type": "choice", "criteria": crit}))


@pytest.mark.parametrize("crit", [[], None, {"a": "b"}, "x"])
def test_score_criteria_must_be_nonempty_list(crit):
    assert "questions.q: criteria must be a non-empty list" in msg(
        mutated(q={"type": "score", "criteria": crit})
    )  # noqa: E501


def test_score_criteria_items_are_strings():
    assert "list of strings" in msg(mutated(q={"type": "score", "criteria": ["a", 2]}))


def test_noul_criteria_rules():
    SystemOneRequest.model_validate(mutated(q={"type": "noul", "criteria": {"true": "yes", "false": "no"}}))
    SystemOneRequest.model_validate(mutated(q={"type": "noul", "criteria": {"true": "yes"}}))
    for crit in ({"maybe": "x"}, {"true": 1}, ["a"], "x"):
        assert "questions.q: criteria" in msg(mutated(q={"type": "noul", "criteria": crit}))


def test_question_type_and_instructions():
    assert "questions.q.type" in msg(mutated(q={"type": "rank", "criteria": ["a"]}))
    assert "instructions" in msg(mutated(q={"type": "noul", "instructions": 5}))


def test_question_ids_and_nonempty_questions():
    assert "at least one question" in msg({"state": 1, "questions": {}})
    assert "question ids must be non-empty" in msg({"state": 1, "questions": {"": {"type": "noul"}}})
    assert "questions: field is required" in msg({"state": 1})


def test_extra_top_level_keys_forbidden():
    assert "media_kwargs: unknown field" in msg({**GOOD, "media_kwargs": {"x": 1}})


def test_media_lists():
    assert "images" in msg({**GOOD, "images": [""]})
    assert "images" in msg({**GOOD, "images": "data:x"})
    r = SystemOneRequest.model_validate({**GOOD, "images": ["data:image/png;base64,AA"], "videos": []})
    assert r.videos == []


def test_batch_index_in_message():
    bad = mutated(urgency={"type": "score", "criteria": []})
    out = msg({"batch": [GOOD, GOOD, GOOD, bad]}, BatchRequest)
    assert out == "batch[3].questions.urgency: criteria must be a non-empty list"


def test_batch_empty_and_extra():
    assert "at least one request" in msg({"batch": []}, BatchRequest)
    assert "unknown field" in msg({"batch": [GOOD], "x": 1}, BatchRequest)


def test_limits():
    cfg = Config(max_questions=2, max_images=1, max_videos=0, max_batch=2)
    r = SystemOneRequest.model_validate(GOOD)
    with pytest.raises(ValueError, match=r"batch\[2\]\.questions: too many questions"):
        check_limits(r, cfg, "batch[2].")
    one = {"questions": {"a": {"type": "noul"}}, "state": 1}
    with pytest.raises(ValueError, match="too many images"):
        check_limits(SystemOneRequest.model_validate({**one, "images": ["x", "y"]}), cfg)
    with pytest.raises(ValueError, match="too many videos"):
        check_limits(SystemOneRequest.model_validate({**one, "videos": ["x"]}), cfg)
    with pytest.raises(ValueError, match="too many requests"):
        check_batch_size(BatchRequest.model_validate({"batch": [GOOD] * 3}), cfg)


def test_openapi_schema_has_descriptions():
    schema = SystemOneRequest.model_json_schema()
    assert "examples" in schema and schema["properties"]["questions"]["description"]
