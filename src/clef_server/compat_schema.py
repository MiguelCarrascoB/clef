"""OpenAI structured-output requests <-> clef questions (pure: no FastAPI, no engine).

A JSON schema (``response_format`` or one forced tool's ``parameters``) becomes a *plan*: one clef question
per top-level property (``choice`` for string enums, ``noul`` for booleans, ``score`` for integer enums /
bounded integers / ordinal string enums, one ``noul`` per option for arrays of string enums), and the
engine's answers are decoded back into the JSON object the caller asked for. See docs/openai-compat.md.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from typing import Any

from .classify import SCORE_QID, SINGLE_QID, classify_questions, score_questions, score_result

LOGPROB_FLOOR = -9999.0  # what OpenAI reports for (near-)zero probability
HINT_ORDINAL = "x-clef-ordinal"  # true on a string enum: the options are an ordered scale (-> score)
HINT_DESCRIPTIONS = "x-clef-descriptions"  # {option: description}; integer options are keyed by their text
_MAX_REF_DEPTH = 16


class SchemaError(ValueError):
    """The request cannot be expressed as clef questions. ``param`` names the offending field."""

    def __init__(self, message: str, param: str | None = None, code: str = "invalid_request_error"):
        super().__init__(message)
        self.param, self.code = param, code


@dataclass
class Prop:
    """One output property and the clef question(s) behind it."""

    name: str
    kind: str  # choice | score | bool | multi
    qids: list[str]
    options: list[Any] = field(default_factory=list)  # values the property can take (multi: the array items)


@dataclass
class Plan:
    questions: dict[str, dict[str, Any]]
    props: list[Prop]


# ------------------------------------------------------------------ schema walking


def _pointer(root: dict[str, Any], ref: str, where: str) -> dict[str, Any]:
    if not ref.startswith("#/"):
        raise SchemaError(f"{where}: only local '#/...' $ref values are supported (got {ref!r})", where)
    node: Any = root
    for part in ref[2:].split("/"):
        part = part.replace("~1", "/").replace("~0", "~")
        if not isinstance(node, dict) or part not in node:
            raise SchemaError(f"{where}: cannot resolve $ref {ref!r}", where)
        node = node[part]
    if not isinstance(node, dict):
        raise SchemaError(f"{where}: $ref {ref!r} does not point to a schema", where)
    return node


def _flatten(node: Any, root: dict[str, Any], where: str, depth: int = 0) -> dict[str, Any]:
    """Resolve $ref / single allOf / Optional (anyOf with null) / enum-of-const unions into one plain node."""
    if not isinstance(node, dict):
        raise SchemaError(f"{where}: schema must be an object", where)
    if depth > _MAX_REF_DEPTH:
        raise SchemaError(f"{where}: schema is nested or recursive too deeply", where)
    out = dict(node)
    if "$ref" in out:
        target = _flatten(_pointer(root, out.pop("$ref"), where), root, where, depth + 1)
        out = {**target, **out}  # siblings (e.g. a property description) win over the target's
    if isinstance(out.get("allOf"), list) and len(out["allOf"]) == 1:
        inner = _flatten(out.pop("allOf")[0], root, where, depth + 1)
        out = {**inner, **out}
    for key in ("anyOf", "oneOf"):
        if key not in out:
            continue
        branches = [_flatten(b, root, where, depth + 1) for b in out.pop(key)]
        branches = [b for b in branches if b.get("type") != "null" and b.get("enum") != [None]]
        if len(branches) == 1:  # Optional[X]: nullability is ignored, clef always decides
            out = {**branches[0], **out}
        elif branches and all("const" in b or _single_enum(b) for b in branches):
            # the "described enum" idiom: oneOf [{const: "a", description: "..."}, ...]
            values, descs = [], dict(out.get(HINT_DESCRIPTIONS) or {})
            for b in branches:
                v = b["const"] if "const" in b else b["enum"][0]
                values.append(v)
                if isinstance(b.get("description"), str):
                    descs.setdefault(str(v), b["description"])
            out["enum"] = values
            if descs:
                out[HINT_DESCRIPTIONS] = descs
        else:
            raise SchemaError(
                f"{where}: unsupported {key} (only Optional[...] and unions of constants)", where
            )
    if isinstance(out.get("type"), list):  # ["string", "null"]
        kinds = [t for t in out["type"] if t != "null"]
        out["type"] = kinds[0] if len(kinds) == 1 else out["type"]
    if "const" in out and "enum" not in out:
        out["enum"] = [out["const"]]
    return out


def _single_enum(node: dict[str, Any]) -> bool:
    return isinstance(node.get("enum"), list) and len(node["enum"]) == 1


def _is_int(v: Any) -> bool:
    return isinstance(v, int) and not isinstance(v, bool)


def _options(node: dict[str, Any], where: str) -> list[Any]:
    vals = [v for v in node["enum"] if v is not None]
    if not vals:
        raise SchemaError(f"{where}: enum has no options", where)
    if all(isinstance(v, str) for v in vals):
        pass
    elif all(_is_int(v) for v in vals):
        vals = sorted(set(vals))
    else:
        raise SchemaError(f"{where}: enum options must be all strings or all integers", where)
    if len(set(vals)) != len(vals):
        raise SchemaError(f"{where}: enum options must be unique", where)
    if any(isinstance(v, str) and not v.strip() for v in vals):
        raise SchemaError(f"{where}: enum options must be non-empty strings", where)
    return vals


def _hint_descriptions(node: dict[str, Any], where: str) -> dict[str, str]:
    d = node.get(HINT_DESCRIPTIONS) or {}
    if not isinstance(d, dict) or not all(isinstance(v, str) for v in d.values()):
        raise SchemaError(f"{where}.{HINT_DESCRIPTIONS}: must be an object of option -> description", where)
    return {str(k): v for k, v in d.items()}


def _join(*parts: str | None) -> str | None:
    text = " ".join(p.strip() for p in parts if isinstance(p, str) and p.strip())
    return text or None


def build_plan(schema: Any, where: str, max_labels: int, intro: str | None = None) -> Plan:
    """JSON schema -> clef questions + decode plan. `where` is the param path used in error messages."""
    if not isinstance(schema, dict):
        raise SchemaError(f"{where} must be a JSON schema object", where)
    root = schema
    top = _flatten(schema, root, where)
    props = top.get("properties")
    if top.get("type") not in (None, "object") or not isinstance(props, dict) or not props:
        raise SchemaError(
            f"{where} must be an object schema with at least one property; each property is one decision "
            "(an enum -> choice, boolean -> yes/no, integer enum -> score). clef picks from options, it "
            "does not write free text.",
            where,
        )
    intro = _join(intro, top.get("description") if isinstance(top.get("description"), str) else None)
    questions: dict[str, dict[str, Any]] = {}
    plan: list[Prop] = []

    def add(qid: str, q: dict[str, Any], at: str) -> None:
        if qid in questions:
            raise SchemaError(f"{at}: question id {qid!r} collides with another property", at)
        questions[qid] = q

    for name, raw in props.items():
        at = f"{where}.properties.{name}"
        node = _flatten(raw, root, at)
        raw_desc = raw.get("description") if isinstance(raw, dict) else None
        desc = raw_desc if isinstance(raw_desc, str) else node.get("description")
        instr = _join(intro, desc if isinstance(desc, str) else None)
        descs = _hint_descriptions(node, at)
        typ = node.get("type")

        if "enum" in node and typ != "array":
            opts = _options(node, at)
            if len(opts) < 2:
                raise SchemaError(f"{at}: needs at least 2 options to decide between (got 1)", at)
            if len(opts) > max_labels:
                raise SchemaError(f"{at}: too many options (max {max_labels})", at)
            if _is_int(opts[0]) or (node.get(HINT_ORDINAL) is True):
                levels = [descs.get(str(o), str(o)) for o in opts]
                add(name, score_questions(levels, instr)[SCORE_QID], at)
                plan.append(Prop(name, "score", [name], opts))
            else:
                labels = {o: descs.get(o, o) for o in opts}
                add(name, classify_questions(labels, instr, False)[SINGLE_QID], at)
                plan.append(Prop(name, "choice", [name], opts))
        elif typ == "boolean":
            q: dict[str, Any] = {"type": "noul"}
            if instr:
                q["instructions"] = instr
            add(name, q, at)
            plan.append(Prop(name, "bool", [name]))
        elif typ == "integer" and _is_int(node.get("minimum")) and _is_int(node.get("maximum")):
            lo, hi = node["minimum"], node["maximum"]
            if hi - lo + 1 < 2 or hi - lo + 1 > max_labels:
                raise SchemaError(f"{at}: integer range must span 2..{max_labels} values", at)
            opts = list(range(lo, hi + 1))
            add(name, score_questions([descs.get(str(o), str(o)) for o in opts], instr)[SCORE_QID], at)
            plan.append(Prop(name, "score", [name], opts))
        elif typ == "array":
            item = _flatten(node.get("items"), root, f"{at}.items")
            if "enum" not in item or not all(isinstance(v, str) for v in item["enum"] if v is not None):
                raise SchemaError(f"{at}: arrays must have items with an enum of strings (multi-label)", at)
            opts = _options(item, f"{at}.items")
            if len(opts) > max_labels:
                raise SchemaError(f"{at}: too many options (max {max_labels})", at)
            idesc = {**_hint_descriptions(item, at), **descs}
            labels = {o: idesc.get(o, o) for o in opts}
            qids = []
            for opt, q in classify_questions(labels, instr, True).items():
                add(f"{name}:{opt}", q, at)
                qids.append(f"{name}:{opt}")
            plan.append(Prop(name, "multi", qids, opts))
        else:
            raise SchemaError(
                f"{at}: unsupported property type. clef decides between options, so use an enum of strings "
                "(choice), an enum of integers or a bounded integer (score), a boolean (yes/no) or an array "
                "of string enums (multi-label). Free text, numbers and nested objects are not supported.",
                at,
            )
    return Plan(questions, plan)


# ------------------------------------------------------------------ answers -> JSON object


def _log(p: float) -> float:
    return math.log(p) if p > 0 else LOGPROB_FLOOR


def _token(text: str, lp: float) -> dict[str, Any]:
    return {"token": text, "logprob": lp, "bytes": list(text.encode())}


@dataclass
class Decoded:
    values: dict[str, Any]  # the JSON object (message content / tool arguments)
    detail: dict[str, Any]  # per property: type, value, confidence, probabilities (the `clef` field)
    probs: dict[str, list[tuple[str, float]]]  # per property: (token text, probability), schema order

    def content(self) -> str:
        return json.dumps(self.values, ensure_ascii=False)

    def logprobs(self, plan: Plan, top: int) -> dict[str, Any]:
        """OpenAI `logprobs.content`: one token-like entry per decision, top_logprobs = best options by p."""
        content: list[dict[str, Any]] = []
        for prop in plan.props:
            pairs = self.probs[prop.name]
            if prop.kind == "multi":  # one entry per option: P(option applies)
                for text, p in pairs:
                    content.append({**_token(text, _log(p)), "top_logprobs": []})
                continue
            chosen = str(
                json.dumps(self.values[prop.name]) if prop.kind == "bool" else self.values[prop.name]
            )
            chosen_p = dict(pairs).get(chosen.strip('"'), 0.0)
            ranked = sorted(pairs, key=lambda kv: -kv[1])[:top]
            content.append(
                {**_token(chosen, _log(chosen_p)), "top_logprobs": [_token(t, _log(p)) for t, p in ranked]}
            )
        return {"content": content, "refusal": None}


def decode(plan: Plan, answers: dict[str, Any], threshold: float) -> Decoded:
    values: dict[str, Any] = {}
    detail: dict[str, Any] = {}
    probs: dict[str, list[tuple[str, float]]] = {}
    for prop in plan.props:
        if prop.kind == "choice":
            ans = answers[prop.name]
            p = {o: float(ans["probabilities"].get(o, 0.0)) for o in prop.options}
            best = ans["choice"]
            values[prop.name] = best
            conf = ans.get("confidence")
            detail[prop.name] = {
                "type": "choice",
                "value": best,
                "confidence": float(conf) if conf is not None else p.get(best, 0.0),
                "probabilities": p,
            }
            probs[prop.name] = list(p.items())
        elif prop.kind == "score":
            ans = answers[prop.name]
            res = score_result({SCORE_QID: ans}, [str(o) for o in prop.options])
            best = prop.options[res["level_index"]]
            by_idx = {int(k): float(v) for k, v in ans["probabilities"].items()}
            p = {str(o): by_idx.get(i, 0.0) for i, o in enumerate(prop.options)}
            values[prop.name] = best
            detail[prop.name] = {
                "type": "score",
                "value": best,
                "confidence": res["confidence"],
                "expected_index": res["score"],
                "probabilities": p,
            }
            probs[prop.name] = list(p.items())
        elif prop.kind == "bool":
            yes = float(answers[prop.name]["noul"])
            values[prop.name] = yes >= 0.5
            detail[prop.name] = {
                "type": "bool",
                "value": yes >= 0.5,
                "confidence": max(yes, 1 - yes),
                "probabilities": {"true": yes, "false": 1 - yes},
            }
            probs[prop.name] = [("true", yes), ("false", 1 - yes)]
        else:  # multi
            p = {o: float(answers[q]["noul"]) for o, q in zip(prop.options, prop.qids, strict=True)}
            picked = sorted((o for o in prop.options if p[o] >= threshold), key=lambda o: -p[o])
            values[prop.name] = picked
            detail[prop.name] = {"type": "multi_label", "value": picked, "probabilities": p}
            probs[prop.name] = list(p.items())
    return Decoded(values, detail, probs)


# ------------------------------------------------------------------ messages -> state + media


def _url_of(obj: Any, key: str, where: str) -> str:
    url = obj.get("url") if isinstance(obj, dict) else obj
    if not isinstance(url, str) or not url.strip():
        raise SchemaError(f"{where}.{key}.url must be a non-empty string", f"{where}.{key}")
    return url


def messages_to_state(messages: Any) -> tuple[str, list[str], list[str]]:
    """OpenAI `messages` -> (state text, image URLs, video URLs).

    A lone user message is passed through verbatim. Otherwise the conversation becomes a readable
    transcript (`system: ...` / `user: ...` / `assistant: ...` / `tool: ...`, blank line between turns),
    in order: assistant and tool turns are kept because they are context for the decision.
    """
    if not isinstance(messages, list) or not messages:
        raise SchemaError("messages must be a non-empty array", "messages")
    turns: list[tuple[str, str]] = []
    images: list[str] = []
    videos: list[str] = []
    for i, msg in enumerate(messages):
        at = f"messages[{i}]"
        if not isinstance(msg, dict) or not isinstance(msg.get("role"), str):
            raise SchemaError(f"{at} must be an object with a string 'role'", at)
        role = {"developer": "system", "function": "tool"}.get(msg["role"], msg["role"])
        content = msg.get("content")
        texts: list[str] = []
        if isinstance(content, str):
            texts.append(content)
        elif isinstance(content, list):
            for j, part in enumerate(content):
                pat = f"{at}.content[{j}]"
                if isinstance(part, str):
                    texts.append(part)
                    continue
                ptype = part.get("type") if isinstance(part, dict) else None
                if ptype in ("text", "input_text", "output_text") and isinstance(part.get("text"), str):
                    texts.append(part["text"])
                elif ptype == "image_url":
                    images.append(_url_of(part.get("image_url"), "image_url", pat))
                elif ptype == "video_url":
                    videos.append(_url_of(part.get("video_url"), "video_url", pat))
                elif ptype == "refusal" and isinstance(part.get("refusal"), str):
                    texts.append(part["refusal"])
                else:
                    raise SchemaError(
                        f"{pat}: unsupported content part type {ptype!r} "
                        "(supported: text, image_url, video_url; audio and files are not)",
                        pat,
                    )
        elif content is not None:
            raise SchemaError(f"{at}.content must be a string or an array of content parts", f"{at}.content")
        for call in msg.get("tool_calls") or []:
            fn = call.get("function", {}) if isinstance(call, dict) else {}
            texts.append(f"[called {fn.get('name', '?')}({fn.get('arguments', '')})]")
        text = "\n".join(t for t in texts if t.strip())
        if text or role == "user":
            turns.append((role, text))
    if not any(r == "user" for r, _ in turns) and not images and not videos:
        raise SchemaError("messages needs at least one user message with content to decide on", "messages")
    if len(turns) == 1 and turns[0][0] == "user":
        return turns[0][1], images, videos
    return "\n\n".join(f"{r}: {t}" for r, t in turns), images, videos
