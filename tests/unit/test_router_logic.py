"""The router's pure logic: gate, label sets, proposal parsing, acceptance (no database, no FastAPI)."""

from __future__ import annotations

import pytest

from clef_server.router import (
    DEFAULT_THRESHOLDS,
    OTHER,
    build_label_set,
    build_prompt,
    capacity,
    evaluate_bootstrap,
    evaluate_proposal,
    gate,
    lexical_key,
    normalise_thresholds,
    other_description,
    parse_proposals,
    proposal_schema,
    threshold_for,
)

THR = DEFAULT_THRESHOLDS


# ------------------------------------------------------------------ threshold + gate


@pytest.mark.parametrize(
    "n,tau", [(2, 0.5), (10, 0.5), (11, 0.7), (30, 0.7), (31, 0.85), (64, 0.85), (200, 0.85)]
)
def test_threshold_rises_with_the_label_count(n, tau):
    assert threshold_for(n, THR) == tau


def test_custom_thresholds_are_validated():
    assert normalise_thresholds([[30, 0.7], [10, 0.4]]) == ((10, 0.4), (30, 0.7))
    for bad in ([], [[1, 0.5]], [[10, 0.0]], [[10, 0.9], [20, 0.5]], "x", [[10]]):
        with pytest.raises(ValueError):
            normalise_thresholds(bad)


def test_gate_other_wins_even_when_confident():
    assert gate({"a": 0.05, "b": 0.05, "other": 0.9}) == "other"


def test_gate_low_confidence_uses_tau_of_the_option_count():
    three = {"a": 0.45, "b": 0.35, "other": 0.2}  # tau(3) = 0.5
    assert gate(three) == "low_confidence"
    twelve = {f"l{i}": 0.06 for i in range(11)} | {"top": 0.6, "other": 0.0}  # 13 options -> tau 0.7
    assert gate(twelve) == "low_confidence"
    ok = {"a": 0.8, "b": 0.1, "other": 0.1}
    assert gate(ok) == "classified"


def test_gate_ambiguous_when_top_two_are_close_but_confident():
    assert gate({"a": 0.5, "b": 0.45, "other": 0.05}) == "ambiguous"
    assert gate({"a": 0.5, "b": 0.45, "other": 0.05}, margin=0.01) == "classified"


# ------------------------------------------------------------------ label sets


def test_other_description_names_labels_in_negative_form_up_to_15():
    d = other_description(["billing", "technical"])
    assert "not billing" in d and "not technical" in d
    many = other_description([f"l{i}" for i in range(16)])
    assert "not l0" not in many and many.startswith("The input fits none")


def test_build_label_set_puts_a_fresh_other_last_and_drops_a_stale_one():
    out = build_label_set({"a": "A things", "Other": "stale", "b": "B things"})
    assert list(out) == ["a", "b", OTHER] and "not a" in out[OTHER]


@pytest.mark.parametrize("own,server,want", [(63, 64, 63), (10, 64, 10), (63, 20, 19), (80, 200, 63)])
def test_capacity_is_capped_at_63_and_the_server_limit(own, server, want):
    assert capacity(own, server) == want


def test_lexical_key_ignores_case_separators_and_plurals():
    assert lexical_key("Refund_Requests") == lexical_key("refund-request") == "refund request"
    assert lexical_key("billing") != lexical_key("shipping")


# ------------------------------------------------------------------ proposals


def prop(name, members=(0, 1), desc="things about stuff", dup="", nearest=""):
    return {
        "name": name,
        "description": desc,
        "member_indices": list(members),
        "nearest_existing": nearest,
        "duplicate_of": dup,
        "rationale": "r",
    }


def parse(proposals, **kw):
    data = {
        "proposals": proposals,
        "covered_by": kw.pop("covered", []),
        "unclear_indices": kw.pop("unclear", []),
    }
    return parse_proposals(
        data, kw.pop("existing", ["billing", "other"]), kw.pop("n", 5), kw.pop("room", 10), 3
    )


def test_parse_keeps_good_proposals_and_drops_each_kind_of_bad_one():
    out = parse(
        [
            prop("shipping"),
            prop("Bad Name"),  # not NAME_RE
            prop("other"),  # reserved
            prop("billing"),  # exact duplicate
            prop("Billings_"),  # lexical duplicate (case / plural)
            prop("refunds", members=(0,)),  # a single member
            prop("returns", members=(0, 99)),  # one valid member after the range check
            prop("warranty", desc="w " * 16),  # description too long
            prop("login", dup="billing"),  # LLM flags it a duplicate
            prop("security", members=(2, 3)),
        ]
    )
    assert [c.name for c in out.candidates] == ["shipping", "security"]
    assert len(out.dropped) == 8


def test_parse_respects_room_and_the_per_call_maximum():
    five = [prop(f"label{i}", members=(0, 1)) for i in range(5)]
    assert len(parse(five).candidates) == 3  # MAX_NEW_PER_CALL passed as 3
    assert len(parse(five, room=1).candidates) == 1
    assert parse(five, room=0).candidates == []


def test_parse_dedups_names_within_one_response_and_ignores_bad_covered_entries():
    out = parse(
        [prop("shipping"), prop("shipping"), prop("Shippings")],
        covered=[
            {"index": 4, "label": "billing"},
            {"index": 4, "label": "nope"},
            {"index": 99, "label": "billing"},
        ],
        unclear=[1, "x", 77],
    )
    assert [c.name for c in out.candidates] == ["shipping"]
    assert out.covered == {4: "billing"} and out.unclear == [1]


def test_proposal_schema_uses_only_the_portable_keywords():
    seen = set()

    def walk(node):
        if isinstance(node, dict):
            for k, v in node.items():
                if k in ("properties",):
                    for sub in v.values():
                        walk(sub)
                else:
                    seen.add(k)
                    walk(v)
        elif isinstance(node, list):
            for v in node:
                walk(v)

    walk(proposal_schema(3))
    assert seen <= {"type", "items", "required", "additionalProperties", "maxItems"}


def test_build_prompt_lists_current_labels_and_numbers_and_truncates_items():
    system, user = build_prompt(
        current={"billing": "money", "other": "none"},
        items=["x" * 900, {"k": "v"}],
        max_new=3,
        max_chars=50,
        domain_hint="support tickets",
    )
    assert "Never repeat" in system
    assert "Domain: support tickets" in user and "- billing: money" in user
    assert "[0] " + "x" * 49 + "…" in user and '[1] {"k": "v"}' in user
    cold = build_prompt(current={}, items=["a"], max_new=8, max_chars=50, bootstrap=True)[1]
    assert "no labels yet" in cold and "Current labels" not in cold


# ------------------------------------------------------------------ acceptance


def s3(billing, new, other):
    return {"billing": billing, "new": new, "other": other}


def test_proposal_passes_when_half_the_members_land_on_it():
    members = [s3(0.1, 0.8, 0.1), s3(0.1, 0.7, 0.2), s3(0.6, 0.3, 0.1), s3(0.2, 0.2, 0.6)]
    res = evaluate_proposal("new", members, THR)
    assert res["passed"] and res["on_new"] == 2 and res["need"] == 2 and res["tau"] == 0.5


def test_proposal_fails_below_max_of_two_and_half():
    one = [s3(0.1, 0.8, 0.1), s3(0.5, 0.2, 0.3)]
    assert not evaluate_proposal("new", one, THR)["passed"]  # need is 2
    ten = [s3(0.1, 0.8, 0.1)] * 4 + [s3(0.2, 0.2, 0.6)] * 6
    assert not evaluate_proposal("new", ten, THR)["passed"]  # 4 < 5


def test_proposal_below_tau_does_not_count_as_landing():
    members = [s3(0.1, 0.45, 0.45)] * 3  # top is a tie broken by order; below tau 0.5 anyway
    assert evaluate_proposal("new", members, THR)["on_new"] == 0


def test_proposal_that_lands_on_an_existing_label_is_a_duplicate():
    members = [s3(0.8, 0.1, 0.1), s3(0.7, 0.2, 0.1), s3(0.6, 0.3, 0.1), s3(0.1, 0.8, 0.1)]
    res = evaluate_proposal("new", members, THR)
    assert res["duplicate_of"] == "billing" and not res["passed"]


def test_human_labels_count_against_the_proposal():
    members = [s3(0.1, 0.8, 0.1), s3(0.1, 0.7, 0.2)]
    assert evaluate_proposal("new", members, THR)["passed"]
    res = evaluate_proposal("new", members, THR, human_against=3)
    assert not res["passed"] and res["members"] == 5 and res["duplicate_of"] is None
    assert evaluate_proposal("new", members, THR, human_against=3)["need"] == 3


def test_bootstrap_keeps_labels_with_two_items_and_needs_half_coverage():
    names = ["a", "b"]
    items = (
        [{"a": 0.8, "b": 0.1, "other": 0.1}] * 3
        + [{"a": 0.1, "b": 0.8, "other": 0.1}]
        + [{"a": 0.1, "b": 0.1, "other": 0.8}] * 2
    )
    res = evaluate_bootstrap(names, items, THR)
    assert list(res["kept"]) == ["a"] and res["covered"] == 3 and res["total"] == 6 and res["passed"]
    low = evaluate_bootstrap(
        names, [{"a": 0.8, "b": 0.1, "other": 0.1}] * 2 + [{"a": 0.1, "b": 0.1, "other": 0.8}] * 6, THR
    )
    assert not low["passed"]  # 2 of 8 covered
    assert not evaluate_bootstrap(names, [{"a": 0.4, "b": 0.3, "other": 0.3}] * 4, THR)["passed"]  # < tau
