"""Drop-in compatibility endpoints: OpenAI chat completions / models and Hugging Face zero-shot.

clef is a decision model, not a text generator, so these routes accept the familiar request shapes and answer
with calibrated choices: OpenAI structured outputs (enums, booleans, integer scales) and zero-shot labels.
Translation: compat_schema (OpenAI), compat_hf (Hugging Face), compat_errors (client-shaped errors).
See docs/openai-compat.md.
"""

from __future__ import annotations

from fastapi import APIRouter

from . import compat_hf, compat_openai
from .appctx import AppContext


def router(ctx: AppContext) -> APIRouter:
    r = APIRouter()
    r.include_router(compat_openai.router(ctx))
    r.include_router(compat_hf.router(ctx))
    return r
