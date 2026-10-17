"""The `clef-release` adapter: every Cloudflare-style clef release (`joint_schema_model.py` + `joint_head.*`
next to the weights). Serves `clef-flash` (9B) and `clef` (27B); both releases ship the same model code.

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
        # D2 perf flags (default off; docs/performance.md). Read once: the worker thread is the only caller.
        self._mask_free = bool(getattr(cfg, "mask_free", False))
        self._fast_head = bool(getattr(cfg, "fast_head", False))
        self._lock = threading.Lock()  # HF tokenizers are not safe under concurrent use in all versions

    @classmethod
    def catalog(cls) -> list[ModelSpec]:
        return [
            ModelSpec("clef-flash", cls.name, MODEL_REPO, MODEL_REVISION, WEIGHTS_GB, {"params_b": 9}),
            ModelSpec(
                "clef",
                cls.name,
                CLEF_REPO,
                CLEF_REVISION,
                CLEF_WEIGHTS_GB,
                {"params_b": 27, "fits_24gb": "nf4"},
            ),
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
        flags = [
            n for n, on in (("CLEF_MASK_FREE", self._mask_free), ("CLEF_FAST_HEAD", self._fast_head)) if on
        ]
        if flags and hasattr(getattr(info, "notes", None), "append"):
            info.notes.append("perf (opt-in, measure with bench/perf_check.sh): " + ", ".join(flags))
        pad = getattr(processor.tokenizer, "pad_token_id", None)
        self._pad = 0 if pad is None else int(pad)
        return info

    def validate(self, record: Record) -> None:
        super().validate(record)
        # An empty span makes the release span mean NaN (the fast head gives 0): refuse it up front. The release
        # renders `instructions` (or the question id when it is missing or "") as the question span.
        for qid, q in (record.get("questions") or {}).items():
            if not str(qid).strip():
                raise ValueError("questions: question ids must not be empty")
            text = q.get("instructions")
            if not (str(qid) if text in (None, "") else str(text)).strip():
                raise ValueError(f"questions.{qid}.instructions: must not be blank")

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
        if (self._mask_free or self._fast_head) and self._splittable():
            return self._forward_split(batch)
        return self._model(batch)  # flags off (or a foreign model): exactly the release's own forward

    # ------------------------------------------------------------------ D2 perf paths (opt-in)
    def _splittable(self) -> bool:
        """True for the release's ClefModel (backbone + head attributes); test stand-ins fall back."""
        m = self._model
        return m is not None and hasattr(m, "language_model") and hasattr(m, "head")

    def _forward_split(self, batch: dict) -> list[list[Any]]:
        """`ClefModel.forward` re-expressed so each flag can change one step (same calls, same arguments).

        CLEF_MASK_FREE: a text-only, right-padded batch calls the backbone with attention_mask=None. That is
        exact for a causal model: position t only reads positions <= t, and padding sits after every real
        token, so the hidden states at real positions do not depend on the mask. This removes the host syncs
        in transformers' mask helpers (`torch.all(mask == 1)` in the linear-attention path, the SDPA causal
        check), the blocker for graph capture. The head never reads padded positions (it slices
        [:length] and takes hidden[length-1] as the global vector), so it only needs the lengths, which we
        already have on the CPU (`batch["lengths"]`). Not exact (so not used): media batches (rope positions
        and the visual-token scatter read the mask), left padding or holes in the mask, a non-causal layer.
        Numerically the real positions are equal up to kernel choice (is_causal flash vs masked SDPA), hence
        the bench/parity.py gate.
        """
        m = self._model
        lm = m.language_model
        base = lm.get_base_model() if hasattr(lm, "get_base_model") else lm
        media = batch.get("media") or {}
        text_model = base.model
        if not media and hasattr(text_model, "language_model"):
            text_model = text_model.language_model
        input_ids = batch["input_ids"]
        lengths = batch.get("lengths")
        have_lengths = (
            lengths is not None
            and len(lengths) == input_ids.shape[0]
            and len(lengths) == len(batch["records"])
            and min(lengths) > 0
            and max(lengths) <= input_ids.shape[1]
        )
        free = self._mask_free and not media and have_lengths
        outputs = text_model(
            input_ids=input_ids,
            attention_mask=None if free else batch["attention_mask"],
            use_cache=False,
            return_dict=True,
            **media,
        )
        hidden = outputs.last_hidden_state
        weight = base.get_output_embeddings().weight
        if self._fast_head and have_lengths:
            return fast_joint_head(m.head, hidden, input_ids, lengths, batch["records"], weight)
        mask = _cpu_mask(lengths, input_ids.shape[1]) if free else batch["attention_mask"]
        # a CPU mask makes the release head's `attention_mask[b].sum().item()` a host op, not a device sync
        return m.head(hidden, input_ids, mask, batch["records"], weight)

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


# ---------------------------------------------------------------------------------------------- D2: fast head
def _cpu_mask(lengths: Sequence[int], width: int) -> Any:
    """[B, width] long mask (1 for t < length) built on the CPU from the known lengths."""
    import torch

    lens = torch.tensor(list(lengths), dtype=torch.long)
    return (torch.arange(width)[None, :] < lens[:, None]).long()


def fast_joint_head(
    head: Any,
    hidden_states: Any,
    input_ids: Any,
    lengths: Sequence[int],
    records: Sequence[Any],
    output_embedding_weight: Any,
) -> list[list[Any]]:
    """The release's `JointSchemaHead.forward`, vectorised over the whole batch and free of host syncs.

    Same parameters, same maths; only the loop structure changes (CLEF_FAST_HEAD=1):
      * the release does `int(attention_mask[b].sum().item())` (a device sync) and then loops over records,
        questions and options in Python, launching a few hundred tiny kernels with host-side slicing. Here
        every span / index table is built on the CPU from the encodings (`lengths`, `question_span`,
        `option_spans`), shipped in ONE transfer, and the whole batch runs as a fixed number of kernels:
        span means are `index_add_` segment sums (the `segment_mean` formulation of vLLM's Apache-2.0
        ClefFlashForJointSchema head, vllm-project/vllm#61308; this code is an independent rewrite),
        ragged option / question sets are scattered into padded grids with key-padding masks, and the
        per-question option softmax is taken over a masked [question, option] grid;
      * padded positions are masked out of every attention, so results do not depend on the padding;
      * the logits come back in one device->host copy, as CPU views per record and question, which also
        makes `adapter.answer()`'s softmax / tolist a host op.
    Span means are accumulated in float32 and rounded to the activation dtype (the release's `mean` also
    accumulates in float32), so the difference to the release head is the usual bf16 reassociation noise
    (~1e-6 in fp32). Needs `head` to be the release's JointSchemaHead (attribute names are the contract).
    """
    import math

    import torch
    import torch.nn.functional as functional

    device = hidden_states.device
    bsz, width_l, hid = hidden_states.shape
    # ---- CPU index tables
    q_tok: list[int] = []  # flat (b * L + t) token index of every question-span token
    q_seg: list[int] = []  # ... and the question it belongs to
    o_tok: list[int] = []
    o_seg: list[int] = []
    rec_of_q: list[int] = []
    types: list[int] = []
    q_of_o: list[int] = []
    o_slot: list[int] = []  # b * Omax + j: option slot in the [B, Omax] grid
    k_slot: list[int] = []  # q * Kmax + j: option slot in the [Q, Kmax] grid
    q_slot: list[int] = []  # b * Qmax + j: question slot in the [B, Qmax] grid
    counts_q: list[int] = []
    counts_o: list[int] = []
    n_questions: list[int] = []
    n_options: list[int] = []
    for rec in records:
        n_questions.append(len(rec.questions))
        n_options.append(sum(len(q.option_spans) for q in rec.questions))
    qmax, omax = max(n_questions), max(n_options)
    kmax = max(len(q.option_spans) for rec in records for q in rec.questions)
    for b, rec in enumerate(records):
        base = b * width_l
        j_opt = 0
        for j_q, q in enumerate(rec.questions):
            qi = len(rec_of_q)
            rec_of_q.append(b)
            types.append(q.question_type)
            q_slot.append(b * qmax + j_q)
            start, end = q.question_span
            q_tok.extend(range(base + start, base + end))
            q_seg.extend([qi] * (end - start))
            counts_q.append(end - start)
            for k, (start, end) in enumerate(q.option_spans):
                oi = len(q_of_o)
                q_of_o.append(qi)
                o_slot.append(b * omax + j_opt)
                k_slot.append(qi * kmax + k)
                j_opt += 1
                o_tok.extend(range(base + start, base + end))
                o_seg.extend([oi] * (end - start))
                counts_o.append(end - start)
    nq, no = len(rec_of_q), len(q_of_o)
    last = [b * width_l + n - 1 for b, n in enumerate(lengths)]
    # ---- one host->device copy for every index table
    tables = [
        q_tok, q_seg, o_tok, o_seg, rec_of_q, types, q_of_o, o_slot, k_slot, q_slot,
        counts_q, counts_o, n_questions, list(lengths), last,
    ]  # fmt: skip
    packed = torch.tensor([v for t in tables for v in t], dtype=torch.long)
    if device.type != "cpu":
        packed = packed.pin_memory()
    packed = packed.to(device, non_blocking=True)
    parts = list(packed.split([len(t) for t in tables]))
    (t_qtok, t_qseg, t_otok, t_oseg, t_recq, t_types, t_qo, t_oslot, t_kslot, t_qslot,
     t_cq, t_co, t_nq, t_len, t_last) = parts  # fmt: skip
    key_pad = torch.arange(width_l, device=device)[None, :] >= t_len[:, None]  # [B, L] True = padding
    q_pad = torch.arange(qmax, device=device)[None, :] >= t_nq[:, None]  # [B, Qmax]

    def seg_mean(rows: Any, seg: Any, counts: Any, n: int) -> Any:
        acc = torch.zeros(n, rows.shape[-1], dtype=torch.float32, device=device)
        acc.index_add_(0, seg, rows.float())
        return (acc / counts.clamp(min=1).unsqueeze(-1)).to(rows.dtype)

    def scatter(rows: Any, slot: Any, n: int, fill: float = 0.0) -> Any:
        grid = torch.full((n,) + tuple(rows.shape[1:]), fill, dtype=rows.dtype, device=device)
        return grid.index_copy_(0, slot, rows)

    normalized = head.hidden_norm(hidden_states)
    flat = normalized.reshape(bsz * width_l, hid)
    memory = head.memory_projection(normalized).masked_fill(key_pad.unsqueeze(-1), 0)
    global_vec = flat[t_last]  # [B, D]: hidden at the last REAL token
    q_vec = seg_mean(flat[t_qtok], t_qseg, t_cq, nq)
    o_ctx = seg_mean(flat[t_otok], t_oseg, t_co, no)
    lexical = seg_mean(
        output_embedding_weight[input_ids.reshape(-1)[t_otok]], t_oseg, t_co, no
    )  # [O, D], mean of the output-embedding rows of each option's tokens

    queries = (
        head.option_context_projection(o_ctx)
        + head.option_lexical_projection(lexical)
        + head.option_question_projection(q_vec)[t_qo]
    )
    routed = scatter(queries, t_oslot, bsz * omax).view(bsz, omax, -1)
    for layer in head.evidence_layers:  # EvidenceRoutingLayer.forward + a key-padding mask
        normalized_memory = layer.memory_norm(memory)
        attended, _ = layer.attention(
            layer.query_norm(routed),
            normalized_memory,
            normalized_memory,
            key_padding_mask=key_pad,
            need_weights=False,
        )
        routed = routed + layer.attention_dropout(attended)
        routed = routed + layer.feedforward(layer.feedforward_norm(routed))
    options = routed.reshape(bsz * omax, -1)[t_oslot]  # [O, W]

    base_fields = head.question_projection(q_vec)  # [Q, W]
    scores = torch.einsum("ow,ow->o", options, base_fields[t_qo]) / math.sqrt(options.shape[-1])
    grid = scatter(scores, t_kslot, nq * kmax, fill=float("-inf")).view(nq, kmax)
    weights = torch.softmax(grid, dim=-1)  # per-question softmax over its own options
    option_grid = scatter(options, t_kslot, nq * kmax).view(nq, kmax, -1)
    summaries = torch.sum(weights.unsqueeze(-1) * option_grid, dim=1)
    fields = (
        base_fields
        + head.option_summary_norm(summaries)
        + head.global_projection(global_vec)[t_recq]
        + head.type_embedding(t_types)
    )
    fields = scatter(fields, t_qslot, bsz * qmax).view(bsz, qmax, -1)
    for layer in head.layers:
        fields = layer(fields, memory, tgt_key_padding_mask=q_pad, memory_key_padding_mask=key_pad)
    fields = head.field_norm(fields.reshape(bsz * qmax, -1)[t_qslot])  # [Q, W]

    anchor = functional.normalize(q_vec + global_vec[t_recq], dim=-1)
    lexical_anchor = functional.normalize(lexical, dim=-1)
    prior_scale = head.prior_logit_scale.clamp(max=math.log(100.0)).exp()
    prior = prior_scale * torch.einsum("od,od->o", lexical_anchor, anchor[t_qo])
    option_n = head.option_norm(options)
    field_o = fields[t_qo]
    cosine = functional.cosine_similarity(field_o, option_n, dim=-1)
    features = torch.cat([field_o, option_n, field_o * option_n, torch.abs(field_o - option_n)], dim=-1)
    residual = head.residual_scorer(features).squeeze(-1)
    joint_scale = head.joint_logit_scale.clamp(max=math.log(100.0)).exp()
    logits = (prior + torch.sigmoid(head.residual_gate) * (joint_scale * cosine + residual)).cpu()

    out: list[list[Any]] = []
    sizes = [len(q.option_spans) for rec in records for q in rec.questions]
    flat_logits = list(logits.split(sizes))
    pos = 0
    for rec in records:
        out.append(flat_logits[pos : pos + len(rec.questions)])
        pos += len(rec.questions)
    return out


__all__ = [
    "CLEF_REPO",
    "CLEF_REVISION",
    "FILES",
    "LAYOUT",
    "PINS",
    "ClefReleaseAdapter",
    "ReleaseEncoded",
    "fast_joint_head",
]
