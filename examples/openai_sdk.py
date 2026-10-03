"""Triage a support ticket through the official `openai` SDK (pip install openai).

clef is a decision model: it picks between options and returns calibrated probabilities, it does not write
free text. So the request carries a schema whose properties are decisions. Reads CLEF_URL and CLEF_API_KEY.
"""

import json
import os
from typing import Literal

from openai import OpenAI
from pydantic import BaseModel

TICKET = "Checkout is down, orders blocked"

client = OpenAI(
    base_url=os.environ.get("CLEF_URL", "http://127.0.0.1:8910").rstrip("/") + "/v1",
    api_key=os.environ.get("CLEF_API_KEY") or "not-needed",  # sent as Authorization: Bearer
)


class Triage(BaseModel):
    team: Literal["billing", "technical", "account"]
    outage: bool


# 1. Typed output: `parse` with a Pydantic model (Literal / Enum -> choice, bool -> yes/no)
parsed = client.beta.chat.completions.parse(
    model="clef-flash",
    messages=[
        {"role": "system", "content": "You triage support tickets."},
        {"role": "user", "content": TICKET},
    ],
    response_format=Triage,
)
print("parsed :", parsed.choices[0].message.parsed)

# 2. Raw JSON schema, with logprobs: exp(logprob) of each option is clef's calibrated probability
schema = {
    "type": "object",
    "properties": {
        "urgency": {
            "type": "string",
            "enum": ["low", "medium", "high"],
            "x-clef-ordinal": True,  # an ordered scale: asked as a score question
            "description": "How urgent is this ticket?",
        }
    },
}
r = client.chat.completions.create(
    model="clef-flash",
    messages=[{"role": "user", "content": TICKET}],
    response_format={"type": "json_schema", "json_schema": {"name": "urgency", "schema": schema}},
    logprobs=True,
    top_logprobs=3,
)
print("content:", r.choices[0].message.content)
print("probs  :", json.dumps(r.model_extra["clef"]["questions"]["urgency"]["probabilities"]))  # non-standard

# 3. Forced function call
r = client.chat.completions.create(
    model="clef-flash",
    messages=[{"role": "user", "content": TICKET}],
    tools=[{"type": "function", "function": {"name": "route", "parameters": Triage.model_json_schema()}}],
    tool_choice={"type": "function", "function": {"name": "route"}},
)
print("tool   :", r.choices[0].message.tool_calls[0].function.arguments)
