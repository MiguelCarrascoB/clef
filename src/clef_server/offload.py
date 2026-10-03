"""CPU offload: keep what does not fit in device memory on the host and stream it in per forward.

Device-agnostic (torch tensors + a `copier` supplied by backend.py): all device specifics (streams, events,
pinned memory) live in `backend.HostCopier`. Three levers, applied in this order:

1. Token embeddings and the output embedding stay on the host for good (~4 GB in bf16). The model only ever
   *gathers* rows from them (a few thousand tokens per forward), so they are looked up on the CPU and the
   result is moved to the device: lossless, a few hundred microseconds per forward.
2. Decoder layers that do not fit under the device budget are held in pinned host memory and copied to the
   device just before they run, on a side stream, one or two layers ahead of the compute (the copy of layer
   i+1 overlaps the compute of layer i). The streamed layers are spread evenly through the stack so resident
   layers sit between them and hide the transfers.
3. The vision tower, final norm, rotary embedding and the joint head always stay on the device.

`plan_layers` and `LayerStreamer` are pure logic and unit-tested without a GPU.
"""

from __future__ import annotations

import json
import logging
import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch

log = logging.getLogger("clef.offload")

GB = 1024**3
LAYER_PREFIX = "model.language_model.layers."
EMBED_KEY = "model.language_model.embed_tokens"
HEAD_KEY = "lm_head"
_LAYER_RE = re.compile(r"^model\.language_model\.layers\.(\d+)\.")
_DTYPE_BYTES = {"BF16": 2, "F16": 2, "F32": 4, "F64": 8, "I8": 1, "U8": 1, "I32": 4, "I64": 8, "BOOL": 1}


@dataclass(frozen=True)
class OffloadPlan:
    """Which parts of the model live where. Sizes in bytes."""

    n_layers: int
    streamed: tuple[int, ...]  # decoder layer indices held on the host
    layer_bytes: tuple[int, ...]
    resident_bytes: int  # device weights: resident layers + visual + head + norms
    host_bytes: int  # host weights: embeddings + streamed layers
    stream_buffer_bytes: int  # transient device buffers for the layers in flight
    feasible: bool = True
    note: str = ""

    @property
    def device_weights_gb(self) -> float:
        return (self.resident_bytes + self.stream_buffer_bytes) / GB

    @property
    def host_gb(self) -> float:
        return self.host_bytes / GB

    def device_map(self, device: str) -> dict[str, str]:
        """from_pretrained `device_map`: streamed layers and embeddings on the CPU, the rest on `device`."""
        dm = {
            "model.visual": device,
            "model.language_model.norm": device,
            "model.language_model.rotary_emb": device,
            EMBED_KEY: "cpu",
            HEAD_KEY: "cpu",
        }
        streamed = set(self.streamed)
        for i in range(self.n_layers):
            dm[f"{LAYER_PREFIX}{i}"] = "cpu" if i in streamed else device
        return dm


def checkpoint_sizes(model_path: str | Path) -> dict[str, int]:
    """Parameter bytes per tensor name, from the safetensors headers only (no weights are read)."""
    path = Path(model_path)
    index = json.loads((path / "model.safetensors.index.json").read_text(encoding="utf-8"))
    from safetensors import safe_open

    sizes: dict[str, int] = {}
    for shard in sorted(set(index["weight_map"].values())):
        with safe_open(str(path / shard), framework="pt") as fh:
            for name in fh.keys():  # noqa: SIM118 - safe_open handle is not a mapping
                sl = fh.get_slice(name)
                n = 1
                for dim in sl.get_shape():
                    n *= dim
                sizes[name] = n * _DTYPE_BYTES.get(sl.get_dtype(), 2)
    return sizes


def split_sizes(sizes: dict[str, int], dtype_scale: float = 1.0) -> tuple[list[int], int, int, int]:
    """(per-layer bytes, embedding bytes (embed + lm_head), other device-bound bytes, head bytes)."""
    layers: dict[int, int] = {}
    embed = other = 0
    for name, size in sizes.items():
        size = int(size * dtype_scale)
        m = _LAYER_RE.match(name)
        if m:
            layers[int(m.group(1))] = layers.get(int(m.group(1)), 0) + size
        elif name.startswith((EMBED_KEY, HEAD_KEY + ".")):
            embed += size
        else:
            other += size
    if not layers:
        raise ValueError(
            "checkpoint has no model.language_model.layers.* tensors: offload does not support it"
        )
    return [layers[i] for i in range(max(layers) + 1)], embed, other, 0


def spread(n: int, k: int) -> tuple[int, ...]:
    """k indices out of range(n), evenly spaced."""
    if k <= 0:
        return ()
    if k >= n:
        return tuple(range(n))
    return tuple(sorted({(i * n + n // 2) // k for i in range(k)}))


def plan_layers(
    layer_bytes: Sequence[int],
    embed_bytes: int,
    other_bytes: int,
    budget_bytes: int,
    lookahead: int = 2,
    head_bytes: int = 0,
) -> OffloadPlan:
    """Choose the layers to stream so device weights (+ in-flight buffers) fit `budget_bytes`.

    `other_bytes` = everything device-bound that is not a decoder layer (vision tower, norms, joint head).
    Embeddings are always on the host. Returns feasible=False when even streaming every layer does not fit.
    """
    layers = list(layer_bytes)
    n = len(layers)
    fixed = other_bytes + head_bytes
    biggest = max(layers)
    total_layers = sum(layers)
    if fixed + total_layers <= budget_bytes:  # everything fits on the device: nothing to stream
        return OffloadPlan(n, (), tuple(layers), fixed + total_layers, embed_bytes, 0)
    chosen: tuple[int, ...] = ()
    for k in range(1, n + 1):
        cand = spread(n, k)
        streamed = set(cand)
        resident = fixed + sum(b for i, b in enumerate(layers) if i not in streamed)
        buffers = min(k, lookahead + 1) * biggest
        if resident + buffers <= budget_bytes:
            chosen = cand
            break
    else:
        streamed = set(range(n))
        resident = fixed
        buffers = min(n, lookahead + 1) * biggest
        feasible = resident + buffers <= budget_bytes
        return OffloadPlan(
            n,
            tuple(range(n)),
            tuple(layers),
            resident,
            embed_bytes + total_layers,
            buffers,
            feasible=feasible,
            note=""
            if feasible
            else "the budget does not even fit the vision tower, head and streaming buffers",
        )
    streamed = set(chosen)
    resident = fixed + sum(b for i, b in enumerate(layers) if i not in streamed)
    host = embed_bytes + sum(layers[i] for i in chosen)
    return OffloadPlan(n, chosen, tuple(layers), resident, host, min(len(chosen), lookahead + 1) * biggest)


# ---------------------------------------------------------------------------------------------- modules
class HostEmbedding(torch.nn.Module):
    """nn.Embedding whose table stays on the host: gather there, then move the (small) result to `device`."""

    def __init__(self, embedding: torch.nn.Embedding, device: Any):
        super().__init__()
        self.weight = embedding.weight
        self.padding_idx = embedding.padding_idx
        self.embedding_dim = embedding.embedding_dim
        self.num_embeddings = embedding.num_embeddings
        self._device = device

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        out = torch.nn.functional.embedding(ids.to("cpu"), self.weight, self.padding_idx)
        return out.to(self._device, non_blocking=False)


class _HostRows:
    """Stands in for `lm_head.weight`: the joint head only ever does `weight[token_ids]`."""

    def __init__(self, weight: torch.Tensor, device: Any):
        self._w = weight
        self._device = device

    def __getitem__(self, ids: Any) -> torch.Tensor:
        if isinstance(ids, torch.Tensor):
            ids = ids.to("cpu")
        return self._w[ids].to(self._device)

    @property
    def shape(self) -> torch.Size:
        return self._w.shape

    @property
    def dtype(self) -> torch.dtype:
        return self._w.dtype

    @property
    def device(self) -> Any:
        return self._device


class HostOutputEmbedding(torch.nn.Module):
    """lm_head stand-in: the vocabulary projection never runs (the joint head only gathers rows)."""

    def __init__(self, linear: torch.nn.Linear, device: Any):
        super().__init__()
        self._host_weight = linear.weight
        self.weight = _HostRows(linear.weight.data, device)  # plain attribute: not a Parameter

    def forward(self, *args: Any, **kwargs: Any) -> Any:  # pragma: no cover - guard
        raise RuntimeError("the output embedding is host-resident and only supports row lookup")


@dataclass
class _Streamed:
    index: int
    slots: list[tuple[torch.nn.Module, str]]  # (owner module, parameter name)
    host: list[torch.Tensor]  # pinned host tensors, same order
    originals: list[Any]  # the Parameter objects, restored after the layer ran
    nbytes: int = 0


@dataclass
class LayerStreamer:
    """Forward hooks that copy host-resident decoder layers to the device right before they run.

    `copier.fetch(list_of_host_tensors)` returns an object whose `.wait()` yields the device tensors (see
    backend.HostCopier). At most `lookahead + 1` streamed layers are on the device at once.
    """

    layers: Sequence[torch.nn.Module]
    streamed: Sequence[int]
    copier: Any
    lookahead: int = 2
    pin: Callable[[torch.Tensor], torch.Tensor] | None = None  # backend.pin_host: page-locked host copy
    _info: dict[int, _Streamed] = field(default_factory=dict, init=False)
    _inflight: dict[int, Any] = field(default_factory=dict, init=False)
    _handles: list[Any] = field(default_factory=list, init=False)
    _order: list[int] = field(default_factory=list, init=False)

    def install(self) -> None:
        self._order = sorted(set(self.streamed))
        for i in self._order:
            layer = self.layers[i]
            slots, host, originals = [], [], []
            for _, mod in layer.named_modules():
                for pname, param in list(mod._parameters.items()):
                    if param is None:
                        continue
                    slots.append((mod, pname))
                    originals.append(param)
                    host_t = param.data.to("cpu")
                    param.data = self.pin(host_t) if self.pin else host_t
                    host.append(param.data)
                if any(b is not None for b in mod._buffers.values()):
                    raise RuntimeError("offloaded layers with buffers are not supported")
            self._info[i] = _Streamed(
                i, slots, host, originals, sum(t.numel() * t.element_size() for t in host)
            )
        for i in range(len(self.layers)):
            self._handles.append(self.layers[i].register_forward_pre_hook(self._make_pre(i)))
            if i in self._info:
                self._handles.append(self.layers[i].register_forward_hook(self._make_post(i)))

    def remove(self) -> None:
        self.reset()
        for h in self._handles:
            h.remove()
        self._handles.clear()

    def reset(self) -> None:
        """Put the host parameters back (after an exception mid-forward) and drop prefetched copies."""
        self._inflight.clear()
        for info in self._info.values():
            self._restore(info)

    # hooks
    def _make_pre(self, i: int) -> Callable:
        def pre(_module: torch.nn.Module, _args: Any) -> None:
            self._prefetch(i)
            if i in self._info:
                self._install_layer(self._info[i])

        return pre

    def _make_post(self, i: int) -> Callable:
        def post(_module: torch.nn.Module, _args: Any, _out: Any) -> None:
            self._restore(self._info[i])

        return post

    def _prefetch(self, i: int) -> None:
        """Start the copies for the next `lookahead + 1` streamed layers at or after layer i."""
        started = 0
        for j in self._order:
            if j < i:
                continue
            if started > self.lookahead:
                break
            started += 1
            if j not in self._inflight:
                self._inflight[j] = self.copier.fetch(self._info[j].host)

    def _install_layer(self, info: _Streamed) -> None:
        pending = self._inflight.pop(info.index, None) or self.copier.fetch(info.host)
        for (mod, pname), tensor in zip(info.slots, pending.wait(), strict=True):
            mod._parameters[pname] = tensor  # plain tensor: only read by forward

    def _restore(self, info: _Streamed) -> None:
        for (mod, pname), original in zip(info.slots, info.originals, strict=True):
            mod._parameters[pname] = original
