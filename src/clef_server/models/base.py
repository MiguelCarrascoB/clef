"""The decision model interface: what the engine drives and what an adapter (built-in or third-party)
implements.

Torch-free at import. One adapter instance per process (D6). An adapter is found through the ``clef.models``
entry-point group (see registry.py); `DecisionModel` is the generic part (identity, capabilities, files,
load),
`LocalTensorAdapter` adds the tensor steps the engine's worker thread calls for every micro-batch:

    encode (any thread) -> collate (CPU batch) -> to_device -> forward (worker, inference mode) -> answer

The engine owns the threads, the priority lanes, micro-batching, bucketing / padding policy, warmup, OOM
handling and stats. The adapter owns everything model-specific. See docs/ARCHITECTURE.md (Decision model
interface).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar, Protocol

if TYPE_CHECKING:  # pragma: no cover
    from ..backend import Backend
    from ..config import Config
    from ..engine import LoadInfo

Record = dict


class UnknownModel(LookupError):
    """A model id this server does not serve (-> 404; `model_not_found` on the OpenAI routes)."""


@dataclass(frozen=True)
class Capabilities:
    question_types: frozenset[str] = frozenset({"choice", "score", "noul"})
    joint_questions: bool = True  # all questions of a record in ONE forward
    images: bool = False
    videos: bool = False
    max_tokens: int = 16384  # hard input limit
    max_options: int | None = None  # None = CLEF_MAX_LABELS
    calibrated: str = "trained"  # "trained" | "temperature-fitted" | "raw" | "unknown"
    supports_offload: bool = False
    quant: frozenset[str] = frozenset({"none"})
    optional_kernels: tuple[str, ...] = ()  # ("fla", "causal_conv1d") for Qwen3.5 hybrids

    def to_json(self) -> dict[str, Any]:
        return {
            "question_types": sorted(self.question_types),
            "joint_questions": self.joint_questions,
            "images": self.images,
            "videos": self.videos,
            "max_tokens": self.max_tokens,
            "max_options": self.max_options,
            "calibrated": self.calibrated,
            "supports_offload": self.supports_offload,
            "quant": sorted(self.quant),
            "optional_kernels": list(self.optional_kernels),
        }


@dataclass(frozen=True)
class ModelSpec:
    id: str  # public id: "clef-flash", "clef"
    adapter: str  # entry-point name: "clef-release"
    repo: str | None  # HF repo, e.g. "Cloudflare/clef-flash"
    revision: str | None  # full 40-char sha (pinned), or None for a local dir
    weights_gb: float  # catalog estimate; the adapter reads the real size at load
    options: dict = field(default_factory=dict)

    @property
    def ref(self) -> str:
        return f"{self.id}@{self.revision or 'local'}"


@dataclass(frozen=True)
class OffloadLayout:
    layers_prefix: str  # "model.language_model.layers."
    embed_key: str  # "model.language_model.embed_tokens"
    head_key: str  # "lm_head"
    device_prefixes: tuple[str, ...]  # kept on the device: vision tower, final norm, rotary


class Encoded(Protocol):
    """One encoded record. Opaque to the engine except these three attributes."""

    length: int  # real tokens (bucketing, usage.input_tokens)
    is_media: bool  # media records run alone and unpadded
    questions: Sequence[Any]  # each has .question_id and .option_ids


def safetensors_gb(path: Path) -> float:
    """On-disk size (GiB, as the engine has always measured it) of the safetensors in `path`; 0.0 if none."""
    try:
        return sum(f.stat().st_size for f in Path(path).glob("*.safetensors")) / 1024**3
    except OSError:
        return 0.0


class DecisionModel(Protocol):
    """Identity, capabilities and loading of one decision model. Subclass it (or `LocalTensorAdapter`)."""

    name: ClassVar[str]  # adapter (entry-point) name
    files: ClassVar[tuple[str, ...]] = ()  # files a complete local release has (`clef doctor` checks them)
    spec: ModelSpec
    capabilities: Capabilities

    @classmethod
    def catalog(cls) -> list[ModelSpec]:
        """The ids this adapter serves out of the box."""
        return []

    def __init__(self, spec: ModelSpec, cfg: Config) -> None:
        self.spec = spec
        self.cfg = cfg

    def resolve(self) -> Path:
        """Local weights dir; never downloads. Raises paths.ModelNotFound with a hint."""
        raise NotImplementedError

    def weights_gb(self, path: Path) -> float:
        """Weights size from the safetensors headers; the catalog estimate when nothing is on disk."""
        gb = safetensors_gb(path) if path else 0.0
        return gb if gb > 1.0 else self.spec.weights_gb

    def load(self, backend: Backend, dtype: Any) -> LoadInfo:
        """Worker thread: load the weights on `backend` (standard / quantized / offloaded)."""
        raise NotImplementedError

    def validate(self, record: Record) -> None:
        """Capability check of one sanitized record; ValueError (400) naming the field."""
        caps = self.capabilities
        if record.get("images") and not caps.images:
            raise ValueError(f"images: model {self.spec.id!r} does not accept images")
        if record.get("videos") and not caps.videos:
            raise ValueError(f"videos: model {self.spec.id!r} does not accept videos")
        max_options = caps.max_options or int(getattr(self.cfg, "max_labels", 0) or 0) or None
        for qid, q in (record.get("questions") or {}).items():
            qtype = q.get("type")
            if qtype not in caps.question_types:
                allowed = ", ".join(sorted(caps.question_types))
                raise ValueError(
                    f"questions.{qid}.type: model {self.spec.id!r} does not support {qtype!r} ({allowed})"
                )
            criteria = q.get("criteria")
            if max_options and qtype in ("choice", "score") and criteria and len(criteria) > max_options:
                raise ValueError(f"questions.{qid}.criteria: too many options (max {max_options})")

    def info(self) -> dict[str, Any]:
        return {
            "id": self.spec.id,
            "adapter": self.name,
            "revision": self.spec.revision,
            "ref": self.spec.ref,
            "capabilities": self.capabilities.to_json(),
        }


class LocalTensorAdapter(DecisionModel, Protocol):
    """A model that runs in this process as tensors. What the engine's worker drives."""

    def encode(self, record: Record) -> Encoded:
        """Any thread (holds its own tokenizer lock). Raises engine.InputTooLarge when it cannot fit."""
        raise NotImplementedError

    def collate(self, encs: list[Encoded], pad_to: int | None) -> dict:
        """CPU batch {"input_ids", "attention_mask", "media", "lengths": list[int]}, right-padded to `pad_to`
        (None for media records). "lengths" are the real lengths, known on the CPU (no .item())."""
        raise NotImplementedError

    def to_device(self, batch: dict, device: Any) -> dict:
        raise NotImplementedError

    def forward(self, batch: dict) -> list[list[Any]]:
        """Worker thread, inside torch.inference_mode(): per record, per question, the option logits."""
        raise NotImplementedError

    def answer(self, q: Any, question: dict, logits: Any) -> dict:
        """One question: q = an element of enc.questions, question = the record's question dict, logits =
        that question's option logits from forward(). Returns the SystemOne answer (softmax inside)."""
        raise NotImplementedError

    def pad_id(self) -> int:
        raise NotImplementedError

    def warm_record(self, repeats: int) -> Record:
        raise NotImplementedError

    def offload_layout(self) -> OffloadLayout | None:
        return None


__all__ = [
    "Capabilities",
    "DecisionModel",
    "Encoded",
    "LocalTensorAdapter",
    "ModelSpec",
    "OffloadLayout",
    "Record",
    "UnknownModel",
    "safetensors_gb",
]
