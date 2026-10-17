"""The `clef-release` adapter: every Cloudflare-style clef release (`joint_schema_model.py` + `joint_head.*` next to
the weights). Serves `clef-flash` (9B) and `clef` (27B); both releases ship the same model code.

Torch is imported lazily (inside `load` and the tensor methods), so the catalog is cheap to read. The model
directory is never modified: its `joint_schema_model.py` is imported under a per-revision module name, and the
offloaded load is assembled in engine.py from the release's own classes.
"""

from __future__ import annotations

import hashlib
import threading
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..config import MODEL_REPO, MODEL_REVISION, QUANTS
from ..paths import MODEL_MARKER, WEIGHTS_GB
from .base import Capabilities, LocalTensorAdapter, ModelSpec, OffloadLayout, Record

# Cloudflare/clef (27B). The sha is the `sha` field of https://huggingface.co/api/models/Cloudflare/clef as
# fetched on 2026-10-17 (lastModified 2026-10-09). Its model code is byte-identical to clef-flash's pin.
CLEF_REPO = "Cloudflare/clef"
CLEF_REVISION = "0b331204bb13fbd2ca93a64df1956f5b55478ce5"
# bf16 safetensors (index metadata.total_size 54,713,457,120 bytes) + joint head: ~55 GB on disk, measured
# the same way as clef-flash's 19.0 (18.82 GB + 0.24 GB).
CLEF_WEIGHTS_GB = 55.0
PINS = {"clef-flash": MODEL_REVISION, "clef": CLEF_REVISION}

FILES = (
    "config.json",
    "joint_head.safetensors",
    "joint_head_config.json",
    MODEL_MARKER,
    "model.safetensors.index.json",
    "tokenizer.json",
    "processor_config.json",
)
# Qwen3.5 checkpoint names (offload.py plans and places the decoder layers with these).
LAYOUT = OffloadLayout(
    layers_prefix="model.language_model.layers.",
    embed_key="model.language_model.embed_tokens",
    head_key="lm_head",
    device_prefixes=("model.visual", "model.language_model.norm", "model.language_model.rotary_emb"),
)
_WARMUP_SENTENCE = "The customer reported that the service was slow yesterday afternoon. "


@dataclass(frozen=True)
class ReleaseEncoded:
    """The release's EncodedRecord plus what the engine needs to know about it."""

    raw: Any  # joint_schema_model.EncodedRecord (or a test stand-in)
    length: int
    is_media: bool

    @property
    def questions(self) -> Sequence[Any]:
        return self.raw.questions

    @property
    def input_ids(self) -> Sequence[int]:  # bench scripts read these two
        return self.raw.input_ids

    @property
    def media(self) -> Any:
        return self.raw.media


class ClefReleaseAdapter(LocalTensorAdapter):
    name = "clef-release"
    files = FILES
    capabilities = Capabilities(
        images=True,
        videos=True,
        supports_offload=True,
        quant=frozenset(QUANTS),
        optional_kernels=("fla", "causal_conv1d"),
    )

    def __init__(self, spec: ModelSpec, cfg: Any, module: Any = None) -> None:
        self.spec = spec
        self.cfg = cfg
        # None = import joint_schema_model.py from the release dir; else the module itself, or a `loader(cfg)`
        # returning it (the engine's test / bench injection), called on the worker thread at load.
        self._module = module
        self._mod: Any = None
        self.path: Path | None = None
        self._model: Any = None
        self._processor: Any = None
        self._pad = 0
        self._lock = threading.Lock()  # HF tokenizers are not safe under concurrent use in all versions

    @classmethod
    def catalog(cls) -> list[ModelSpec]:
        return [
            ModelSpec("clef-flash", cls.name, MODEL_REPO, MODEL_REVISION, WEIGHTS_GB, {"params_b": 9}),
            ModelSpec("clef", cls.name, CLEF_REPO, CLEF_REVISION, CLEF_WEIGHTS_GB, {"params_b": 27}),
        ]

    @classmethod
    def from_module(cls, module: Any, spec: ModelSpec, cfg: Any) -> ClefReleaseAdapter:
        """Wrap a joint_schema_model-like namespace (`load_release_model`, `encode_record`, `collate_records`,
        `systemone_answer`), or a `loader(cfg)` returning one. The path is CLEF_MODEL_PATH as given."""
        return cls(spec, cfg, module=module)

    # ------------------------------------------------------------------ load (worker thread)
    def resolve(self) -> Path | None:
        if self._module is not None:  # injected module: no release dir lookup
            self.path = Path(self.cfg.model_path) if self.cfg.model_path else None
        else:
            from .. import paths

            self.path = paths.resolve_model_path(self.cfg, self.spec)
        return self.path

    def _module_name(self) -> str:
        key = self.spec.revision or hashlib.sha1(str(self.path).encode()).hexdigest()
        return f"clef_release_{key[:8]}"

    def _load_module(self) -> Any:
        from .. import engine as engine_mod

        mod = self._module
        if mod is None:
            mod = engine_mod._import_model_module(self.path, self._module_name())
        elif callable(mod) and not hasattr(mod, "encode_record"):
            mod = mod(self.cfg)
        self._mod = mod
        return mod

    def load(self, backend: Any, dtype: Any) -> Any:
        from .. import engine as engine_mod

        if self.path is None and self._module is None:
            self.resolve()
        mod = self._load_module()
        path = self.path
        model, processor, info = engine_mod.load_model(
            mod, path, backend, dtype, self.cfg, layout=self.offload_layout()
        )
        model.eval()
        self._model, self._processor = model, processor
        pad = getattr(processor.tokenizer, "pad_token_id", None)
        self._pad = 0 if pad is None else int(pad)
        return info

    # ------------------------------------------------------------------ tensors
    def encode(self, record: Record) -> ReleaseEncoded:
        from ..engine import InputTooLarge

        try:
            with self._lock:
                raw = self._mod.encode_record(
                    self._processor.tokenizer,
                    record,
                    max_length=self.cfg.max_tokens,
                    processor=self._processor,
                )
        except ValueError as exc:
            if "schema requires" in str(exc):  # the release's own "too long" message
                raise InputTooLarge(str(exc)) from exc
            raise
        return ReleaseEncoded(raw, len(raw.input_ids), raw.media is not None)

    def collate(self, encs: list[Any], pad_to: int | None) -> dict:
        import torch

        from ..engine import pad_batch_right

        batch = self._mod.collate_records([e.raw for e in encs], self._pad, torch.device("cpu"))
        if pad_to is not None:
            batch = pad_batch_right(batch, pad_to, self._pad)
        return {**batch, "lengths": [e.length for e in encs]}

    def to_device(self, batch: dict, device: Any) -> dict:
        out = dict(batch)
        for key in ("input_ids", "attention_mask"):
            out[key] = batch[key].to(device, non_blocking=True)
        out["media"] = {k: v.to(device, non_blocking=True) for k, v in (batch.get("media") or {}).items()}
        return out

    def forward(self, batch: dict) -> list[list[Any]]:
        return self._model(batch)

    def answer(self, q: Any, question: dict, logits: Any) -> dict:
        probs = dict(zip(q.option_ids, logits.float().softmax(-1).tolist(), strict=True))
        return self._mod.systemone_answer(question, probs)

    def pad_id(self) -> int:
        return self._pad

    def warm_record(self, repeats: int) -> Record:
        return {
            "id": "warmup",
            "state": _WARMUP_SENTENCE * repeats,
            "questions": {
                "q": {"type": "choice", "instructions": "warmup", "criteria": {"a": "first", "b": "second"}}
            },
        }

    def offload_layout(self) -> OffloadLayout:
        return LAYOUT


__all__ = ["CLEF_REPO", "CLEF_REVISION", "FILES", "LAYOUT", "PINS", "ClefReleaseAdapter", "ReleaseEncoded"]
