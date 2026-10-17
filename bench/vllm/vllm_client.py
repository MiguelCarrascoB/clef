"""Client for vLLM's `/v1/systemone` (PR vllm-project/vllm#61308) that speaks clef's record shape.

    python bench/vllm/vllm_client.py dump --url http://127.0.0.1:8000 --out out/dump_vllm.json --n 200 --images 10
    python bench/parity.py --compare out/dump_clef.json out/dump_vllm.json     # unchanged parity tooling

`dump` writes the same JSON as `bench/parity.py --engine-dump` ("singles" one request at a time, "batched" 8
requests in flight at once), so `--compare` / `--noise` work on a vLLM server or on a clef HTTP server alike.

Shapes (source: docs/serving/online_serving/structured_decisions.md at the PR's merge commit 380638af):
  request   {model, state, questions: {id: {type, instructions?, criteria?}}, images?: [url|data URL], videos?}
  response  {answers: {id: {type, choice|noul|score, probabilities, confidence, legend?}}, usage, diagnostics}
  choice    probabilities keyed by option name; noul: `noul` = P(yes), probabilities keyed yes/no;
            score: probabilities keyed by level index "0".."n-1" (str), `legend` maps index -> level name.
Clef-Flash specifics: a top-level `instructions` returns 400 (we never send one); `chat_template_kwargs` has no effect;
a question's own `instructions` is accepted; a question key other than type/instructions/criteria returns 400.

Every shape mismatch raises ShapeError instead of being scored as 0: a wrong key set, a missing answer, a
probability that is not finite or does not sum to 1 (choice / score).
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import io
import json
import math
import statistics
import sys
import time
from pathlib import Path
from typing import Any

MODEL_NAME = "clef-flash"
SUM_TOL = 1e-2  # vLLM rounds nothing, clef rounds to 5 decimals; 1e-2 only catches a wrong shape
QUESTION_KEYS = ("type", "instructions", "criteria")
# UNVERIFIED (docs show no example): whether Clef-Flash adds `legend` to score answers like the generative path. We
# never require it. Also UNVERIFIED: the exact error body of a 400 (we print what the server sends).


class ShapeError(ValueError):
    """The server's response does not have the documented /v1/systemone shape."""


def image_to_data_url(img: Any) -> str:
    """PIL image -> PNG data URL (what vLLM takes for `images`)."""
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def to_request(record: dict, model: str = MODEL_NAME) -> dict:
    """A clef eval record ({id, state, questions, images?: [PIL]}) -> the /v1/systemone request body.

    Drops `id` (not part of the API), keeps only type/instructions/criteria per question and omits `criteria`
    for noul questions that have none. Images (PIL or ready URL strings) become data URLs."""
    questions: dict[str, dict] = {}
    for qid, q in record["questions"].items():
        extra = set(q) - set(QUESTION_KEYS)
        if extra:
            raise ShapeError(f"question {qid!r} has fields vLLM rejects with 400: {sorted(extra)}")
        questions[qid] = {k: q[k] for k in QUESTION_KEYS if q.get(k) is not None}
    body: dict[str, Any] = {"model": model, "state": record["state"], "questions": questions}
    images = record.get("images")
    if images:
        body["images"] = [i if isinstance(i, str) else image_to_data_url(i) for i in images]
    if record.get("videos"):
        body["videos"] = list(record["videos"])
    return body


def _prob(v: Any, where: str) -> float:
    if (
        isinstance(v, bool)
        or not isinstance(v, (int, float))
        or not math.isfinite(v)
        or not -1e-9 <= v <= 1 + 1e-9
    ):
        raise ShapeError(f"{where}: not a probability: {v!r}")
    return float(v)


def flatten_answers(resp: dict, questions: dict) -> dict[str, float]:
    """Response -> {"question.option": prob}, the flat form of bench/parity.py (`flat_answers`):
    choice `q.<option>`, score `q.<level index>`, noul `q.true` / `q.false` (= 1 - p). `questions` is the request's
    `questions` dict. Raises ShapeError on any mismatch with it (works for clef's own response as well)."""
    if not isinstance(resp, dict) or not isinstance(resp.get("answers"), dict):
        raise ShapeError(
            f"no `answers` object in the response (keys: {sorted(resp) if isinstance(resp, dict) else resp!r})"
        )
    answers = resp["answers"]
    if set(answers) != set(questions):
        raise ShapeError(f"answered {sorted(answers)}, asked {sorted(questions)}")
    flat: dict[str, float] = {}
    for qid, q in questions.items():
        ans, kind = answers[qid], q["type"]
        if not isinstance(ans, dict) or ans.get("type") != kind:
            raise ShapeError(f"{qid}: expected an answer of type {kind!r}, got {ans!r:.120}")
        if kind == "noul":
            p = _prob(ans.get("noul"), f"{qid}.noul")
            flat[f"{qid}.true"], flat[f"{qid}.false"] = p, 1.0 - p
            continue
        probs = ans.get("probabilities")
        if kind == "choice":
            expected = list((q.get("criteria") or {}).keys())
        elif kind == "score":
            expected = [str(i) for i in range(len(q.get("criteria") or []))]
        else:
            raise ShapeError(f"{qid}: unknown question type {kind!r}")
        if not isinstance(probs, dict) or {str(k) for k in probs} != set(expected):
            raise ShapeError(
                f"{qid}: probabilities keys {sorted(probs) if isinstance(probs, dict) else probs!r}, expected {expected}"
            )
        vals = {str(k): _prob(v, f"{qid}.{k}") for k, v in probs.items()}
        if abs(sum(vals.values()) - 1.0) > SUM_TOL:
            raise ShapeError(f"{qid}: probabilities sum to {sum(vals.values()):.4f}")
        if kind == "choice" and ans.get("choice") not in expected:
            raise ShapeError(f"{qid}: choice {ans.get('choice')!r} is not one of {expected}")
        for k in expected:
            flat[f"{qid}.{k}"] = vals[k]
    return flat


def check_response(request_body: dict, resp: dict) -> None:
    """Hook for bench/vllm/load.py (httpbench.run_level `check`): raises ShapeError (a ValueError)."""
    flatten_answers(resp, request_body["questions"])


# ---------------------------------------------------------------------------------------------- HTTP / dump
async def _post(client: Any, body: dict) -> tuple[dict, float]:
    t = time.perf_counter()
    r = await client.post("/v1/systemone", json=body)
    ms = (time.perf_counter() - t) * 1000
    if r.status_code != 200:
        raise RuntimeError(f"HTTP {r.status_code} from /v1/systemone: {r.text[:500]}")
    return r.json(), ms


async def dump(
    url: str, records: list[dict], model: str, chunk: int, api_key: str | None = None, backend: str = "vllm"
) -> dict:
    import httpx

    headers = {"X-API-Key": api_key} if api_key else {}
    async with httpx.AsyncClient(base_url=url, headers=headers, timeout=600) as c:
        reqs = [to_request(r, model) for r in records]
        await _post(c, reqs[0])  # warm-up, not scored
        singles, lat, models = [], [], set()
        for body in reqs:
            resp, ms = await _post(c, body)
            models.add(str(resp.get("model")))
            singles.append(flatten_answers(resp, body["questions"]))
            lat.append(ms)
        batched = []
        for i in range(0, len(reqs), chunk):
            part = reqs[i : i + chunk]
            got = await asyncio.gather(*(_post(c, b) for b in part))
            batched.extend(flatten_answers(g[0], b["questions"]) for g, b in zip(got, part, strict=True))
            models.update(str(g[0].get("model")) for g in got)
    return {
        "src": url,
        "model": model,
        "backend": backend,
        "dtype": "served",
        "singles": singles,
        "batched": batched,
        "lat": lat,
        "result_models": sorted(models),
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("dump", help="run the parity eval set through a /v1/systemone server, write a dump")
    d.add_argument("--url", default="http://127.0.0.1:8000")
    d.add_argument("--out", required=True)
    d.add_argument("--n", type=int, default=200)
    d.add_argument("--images", type=int, default=10)
    d.add_argument("--model", default=MODEL_NAME)
    d.add_argument("--chunk", type=int, default=8, help="requests in flight for the `batched` mode")
    d.add_argument("--api-key")
    d.add_argument("--backend", default="vllm", help="label printed by parity.py --compare")
    args = ap.parse_args(argv)

    here = Path(__file__).resolve().parent
    sys.path.insert(0, str(here.parent))  # bench/ for parity.py
    import parity  # noqa: E402

    records = parity.build_eval_set(args.n, images=args.images)
    try:
        result = asyncio.run(dump(args.url, records, args.model, args.chunk, args.api_key, args.backend))
    except (ShapeError, RuntimeError) as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        return 1
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(result))
    print(f"wrote {args.out}: {len(records)} records, singles p50 {statistics.median(result['lat']):.1f} ms")
    return 0


if __name__ == "__main__":
    sys.exit(main())
