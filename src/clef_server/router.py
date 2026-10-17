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

import json
import math
import re
from dataclasses import dataclass, field
from typing import Any

from .classifiers import NAME_RE

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


def proposal_schema(max_labels: int) -> dict[str, Any]:
    """Response schema for propose (max 3) and bootstrap (max 8). Only type / properties / required / items /
    additionalProperties / min-maxItems: the subset every OpenAI-compatible server accepts under strict mode.
    Names, word limits and index bounds are enforced in code (``parse_proposals``)."""
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
            "proposals": {"type": "array", "items": label, "maxItems": max_labels},
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
