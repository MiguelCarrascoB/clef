"""Inference engine: background model load, one GPU worker thread, micro-batching, length bucketing, warmup.

Performance knobs (environment variables, set by the launcher BEFORE starting the process; the engine
itself never sets them). Recommended for ROCm 7.2 / RX 7900 XTX under WSL2:

  PYTORCH_TUNABLEOP_ENABLED=1                  enable TunableOp GEMM autotuning
  PYTORCH_TUNABLEOP_FILENAME=<path>/tunableop_%d.csv   persist tuned kernels across restarts
  TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1    AOTriton flash/SDPA attention kernels
  (PYTORCH_HIP_ALLOC_CONF=expandable_segments:True is unsupported on ROCm-on-WSL)
  TRITON_CACHE_DIR=<path>                      persistent Triton kernel cache (faster warm restarts)

Threading model: exactly one thread (the "worker") touches the GPU. It loads the model, runs warmup
interleaved with request serving, and then processes the queue. `decide()` tokenizes on the caller side
(thread pool), enqueues encoded records, and awaits futures that the worker resolves.

Priority lanes: the queue has two FIFO lanes, "interactive" (requests, the default) and "background"
(jobs, through ctx.decide). After the batch window the worker fills the micro-batch from ONE lane:
interactive whenever any is queued, else background. After cfg.interactive_burst interactive micro-batches
in a row while background work waited, one background micro-batch goes next (starvation bound). Classes are
never mixed in a forward: the model is compute-bound, so an interactive record riding in a full background
batch would pay for all of it.

Bucketing: text-only records are grouped by length bucket (see `bucket_for`) so one forward carries similar
lengths, then right-padded to the next cfg.pad_multiple (default 64): hipBLASLt picks faster GEMM tiles for
aligned lengths, and the waste stays under 64 tokens. Full padding to the bucket is opt-in
(cfg.pad_to_bucket): the model is compute-bound, so it costs real GPU time, while a never-seen length only
costs ~10-20% extra once. Media records run alone and are never padded (their media token-type tensors and
rope positions depend on the exact sequence layout).

Graph capture / torch.compile: blocked today by a device sync in transformers' SDPA masking
(masking_utils._ignore_causal_mask_sdpa calls .all() on the mask); single-record forwards are ~50% host
overhead, so removing that sync is the next big latency lever.
"""

from __future__ import annotations

import asyncio
import contextlib
import copy
import importlib
import importlib.metadata
import importlib.util
import logging
import os
import sys
import threading
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch

from . import backend as backend_mod
from .backend import Backend, BackendError, PreflightError
from .cache import ResultCache
from .models import registry
from .models.base import Capabilities, DecisionModel, ModelSpec, OffloadLayout, UnknownModel, safetensors_gb
from .paths import MODEL_MARKER, WEIGHTS_GB, ModelNotFound, resolve_model_path

log = logging.getLogger("clef.engine")

Record = dict
Result = dict

INTERACTIVE = "interactive"
BACKGROUND = "background"
PRIORITIES = (INTERACTIVE, BACKGROUND)
_EMPTY_TELEMETRY = {
    "available": False,
    "source": None,
    "util_pct": None,
    "temp_c": None,
    "power_w": None,
    "mem_used_gb": None,
    "mem_total_gb": None,
}
_WARMUP_SENTENCE = "The customer reported that the service was slow yesterday afternoon. "


class EngineNotReady(RuntimeError):
    """Model not loaded yet, failed to load, or engine shut down (-> HTTP 503)."""


class InputTooLarge(ValueError):
    """Schema alone exceeds max_tokens (-> HTTP 413)."""


class GpuOutOfMemory(RuntimeError):
    """Device out of memory during a forward (-> HTTP 503)."""


def _dtype_name(dtype: Any) -> str:
    return str(dtype).replace("torch.", "")


def bucket_for(length: int, buckets: tuple[int, ...] | list[int]) -> int:
    """Smallest bucket >= length; beyond the largest bucket, round up to a multiple of 512."""
    for b in sorted(buckets):
        if length <= b:
            return b
    return -(-length // 512) * 512


def pad_batch_right(batch: dict[str, Any], target: int, pad_token_id: int) -> dict[str, Any]:
    """Right-pad a collated (CPU) text batch to `target` columns (input_ids with pad id, mask with 0)."""
    ids, mask = batch["input_ids"], batch["attention_mask"]
    n, length = ids.shape
    if target < length:
        raise ValueError(f"target {target} shorter than batch length {length}")
    if target == length:
        return batch
    new_ids = torch.full((n, target), pad_token_id, dtype=ids.dtype, device=ids.device)
    new_mask = torch.zeros((n, target), dtype=mask.dtype, device=mask.device)
    new_ids[:, :length] = ids
    new_mask[:, :length] = mask
    return {**batch, "input_ids": new_ids, "attention_mask": new_mask}


def _import_model_module(path: Any, name: str = "joint_schema_model") -> Any:
    """Import `<path>/joint_schema_model.py` as module `name` (per revision, so releases never collide)."""
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, Path(path) / MODEL_MARKER)
    if spec is None or spec.loader is None:
        raise ModelNotFound(f"cannot import {Path(path) / MODEL_MARKER}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod  # before exec: the release's dataclasses look their module up
    try:
        spec.loader.exec_module(mod)
    except BaseException:
        sys.modules.pop(name, None)
        raise
    return mod


def _default_loader(cfg: Any) -> Any:
    return _import_model_module(resolve_model_path(cfg))


def _weights_gb(path: Any) -> float:
    """On-disk size of the release weights (falls back to the measured 19 GB)."""
    gb = safetensors_gb(path)
    return gb if gb > 1.0 else WEIGHTS_GB


@dataclass
class LoadInfo:
    """What `load_model` did, for logs, /health warnings and the memory benchmark."""

    mode: str = "standard"  # standard | offload
    streamed_layers: int = 0
    n_layers: int = 0
    host_weights_gb: float = 0.0
    device_weights_gb: float = 0.0
    notes: list[str] = field(default_factory=list)  # informational, logged at INFO
    warnings: list[str] = field(default_factory=list)  # things the operator should fix; shown in /health
    streamer: Any = None

    def warn(self, message: str) -> None:
        if message not in self.warnings:
            self.warnings.append(message)


def _file_gb(path: Path) -> float:
    try:
        return path.stat().st_size / 1024**3
    except OSError:
        return 0.0


def device_budget_gb(backend: Backend, cfg: Any) -> float | None:
    """Device memory for the weights: the explicit cap, else what is free now, minus activation headroom."""
    cap = float(getattr(cfg, "max_device_memory_gb", 0.0) or 0.0)
    free = backend.memory().get("free_gb")
    if cap > 0:
        # a cap above what is free cannot be honoured: plan with the smaller number (preflight warns)
        total = cap if free is None else min(cap, float(free))
    else:
        if free is None:
            return None
        total = float(free)
    return max(total - backend_mod.ACTIVATION_RESERVE_GB, 0.0)


def _is_device_map_refusal(exc: BaseException) -> bool:
    """transformers' ValueError when a quantizer refuses host entries in the device_map (the one failure the
    fallback load can fix). Anything else (corrupt checkpoint, host OOM, a bug) must surface."""
    if not isinstance(exc, ValueError):
        return False
    text = str(exc).lower()
    return any(k in text for k in ("device_map", "dispatched on the cpu", "cpu_offload", "offload"))


def load_model(
    mod: Any, path: Any, backend: Backend, dtype: Any, cfg: Any, layout: OffloadLayout | None = None
) -> tuple[Any, Any, LoadInfo]:
    """Load (model, processor, info) honouring cfg.quant / cfg.offload / cfg.max_device_memory_gb.

    Called by the clef-release adapter's `load()` with its OffloadLayout (None = the Qwen3.5 names).

    The standard path (no offload) is the model's own `load_release_model`, optionally with a quantization
    config. With CLEF_OFFLOAD=cpu the backbone is loaded with a per-module device map (embeddings and the
    layers that do not fit under the cap on the host) and the clef model is assembled here from the release's
    own classes, so the model directory is never touched. See offload.py.
    """
    info = LoadInfo()
    quant = getattr(cfg, "quant", "none")
    offload = getattr(cfg, "offload", "none")
    cap = float(getattr(cfg, "max_device_memory_gb", 0.0) or 0.0)
    getattr(backend, "reset_load_state", lambda: None)()
    if cap > 0:
        if backend.apply_memory_cap(cap):
            info.notes.append(f"device memory capped at {cap:g} GB")
        else:
            why = getattr(backend, "cap_failure", lambda: None)() or "it could not be applied"
            info.warn(f"CLEF_MAX_DEVICE_MEMORY_GB={cap:g} is not enforced: {why}")
    if offload == "cpu" and not backend.has_discrete_memory:
        info.warn(f"CLEF_OFFLOAD=cpu has no effect on {backend.name} (device and host share one memory pool)")
        offload = "none"
    qc = backend_mod.quantization_config(backend, quant, dtype, getattr(cfg, "quant_backend", "auto"))
    caveat = backend_mod.quant_caveat(backend, quant, getattr(cfg, "quant_backend", "auto"))
    if caveat:
        info.warn(caveat)
    if offload == "cpu":
        try:
            return _load_offloaded(mod, path, backend, dtype, cfg, info, qc, layout=layout)
        except (PreflightError, BackendError):
            raise
        except Exception as exc:
            if qc is None or not _is_device_map_refusal(exc):
                raise
            # transformers refused host entries in the device_map for this quantizer: load, then move them
            msg = (
                f"CLEF_OFFLOAD=cpu with CLEF_QUANT={quant}: direct host placement was refused ({exc}); "
                "loaded the quantized model on the device and moved the embeddings afterwards (peak memory "
                "during the load is higher)"
            )
            log.warning("%s", " ".join(msg.split()))
            info.warn(" ".join(msg.split()))
            backend.empty_cache()
    kwargs: dict[str, Any] = {}
    if qc is not None:
        kwargs["quantization_config"] = qc
    model, processor = mod.load_release_model(
        None if path is None else str(path), device=backend.device, dtype=dtype, **kwargs
    )
    if offload == "cpu":
        _embeddings_to_host(model, backend, info)
    return model, processor, info


def _plan_budget_gb(backend: Backend, cfg: Any, qc: Any, info: LoadInfo) -> float:
    """Device budget for the layer plan; 1e9 (stream nothing, only the vocabulary moves) warns."""
    budget = device_budget_gb(backend, cfg)
    cap = float(getattr(cfg, "max_device_memory_gb", 0.0) or 0.0)
    free = backend.memory().get("free_gb")
    if cap > 0 and free is not None and cap > float(free):
        info.warn(
            f"CLEF_MAX_DEVICE_MEMORY_GB={cap:g} is above the {float(free):.1f} GB free; "
            "the layer plan uses the free memory"
        )
    if qc is not None:
        if cap > 0:
            info.warn(
                f"CLEF_MAX_DEVICE_MEMORY_GB={cap:g} does not shape the offload plan with CLEF_QUANT: "
                "quantized layers stay on the device and only the embeddings move to the host"
            )
        return 1e9
    if budget is None:
        info.warn(
            "device memory is unknown, so no decoder layers are streamed (only the embeddings move to the "
            "host); set CLEF_MAX_DEVICE_MEMORY_GB to plan a split"
        )
        return 1e9
    return budget


def _attr(obj: Any, dotted: str) -> Any:
    for part in dotted.split("."):
        obj = getattr(obj, part)
    return obj


def _load_offloaded(
    mod: Any,
    path: Any,
    backend: Backend,
    dtype: Any,
    cfg: Any,
    info: LoadInfo,
    qc: Any = None,
    layout: OffloadLayout | None = None,
):
    import json

    from safetensors.torch import load_file
    from transformers import AutoProcessor, Qwen3_5ForConditionalGeneration

    from . import offload as off

    layout = layout or off.DEFAULT_LAYOUT
    root = Path(path)
    sizes = off.checkpoint_sizes(root)
    layer_bytes, embed_bytes, other_bytes, _ = off.split_sizes(sizes, _bytes_scale(dtype), layout=layout)
    head_bytes = int(_file_gb(root / "joint_head.safetensors") * 1024**3 * _bytes_scale(dtype))
    budget_gb = _plan_budget_gb(backend, cfg, qc, info)
    plan = off.plan_layers(
        layer_bytes, embed_bytes, other_bytes, int(budget_gb * 1024**3), head_bytes=head_bytes
    )
    if not plan.feasible:
        raise PreflightError(
            f"CLEF_MAX_DEVICE_MEMORY_GB leaves {budget_gb:.1f} GB for weights: not enough even with every "
            "layer "
            f"streamed from host memory ({plan.note}). Raise the cap or use a quantized model (CLEF_QUANT)."
        )
    host_free = backend_mod._cpu_memory_bytes()[2]
    if cfg.preflight and host_free is not None and plan.host_gb + 2.0 > host_free / 1024**3:
        raise PreflightError(
            f"not enough host RAM for CLEF_OFFLOAD=cpu: {plan.host_gb:.1f} GB of weights stay in host "
            "memory, "
            f"{host_free / 1024**3:.1f} GB free. Raise CLEF_MAX_DEVICE_MEMORY_GB or free memory."
        )
    device = str(backend.device)
    extra = {"quantization_config": qc} if qc is not None else {}
    backbone = Qwen3_5ForConditionalGeneration.from_pretrained(
        root, dtype=dtype, device_map=plan.device_map(device, layout=layout), **extra
    )
    _strip_accelerate_hooks(backbone)
    backbone.config.use_cache = False
    # the layout's checkpoint names are attribute paths on the backbone
    embed_owner, _, embed_attr = layout.embed_key.rpartition(".")
    lm = _attr(backbone, embed_owner)
    setattr(lm, embed_attr, off.HostEmbedding(getattr(lm, embed_attr), backend.device))
    head = _attr(backbone, layout.head_key)
    setattr(backbone, layout.head_key, off.HostOutputEmbedding(head, backend.device))
    streamer = None
    if plan.streamed:
        layers = _attr(backbone, layout.layers_prefix.rstrip("."))
        streamer = off.LayerStreamer(
            list(layers), plan.streamed, backend.host_copier(), lookahead=2, pin=backend.pin_host
        )
        streamer.install()
        pin_err = getattr(backend, "pin_failure", lambda: None)()
        if pin_err:
            info.warn(
                f"pinned host memory is unavailable ({pin_err}); streamed layers use pageable copies, which "
                "is much slower"
            )
    head = mod.JointSchemaHead(**json.loads((root / "joint_head_config.json").read_text(encoding="utf-8")))
    head.load_state_dict(load_file(root / "joint_head.safetensors"), strict=True)
    head = head.to(device=backend.device, dtype=dtype)
    processor = AutoProcessor.from_pretrained(root)
    model = mod.ClefModel(backbone, head).eval()
    info.mode = "offload"
    info.streamed_layers = len(plan.streamed)
    info.n_layers = plan.n_layers
    info.host_weights_gb = round(plan.host_gb, 2)
    info.device_weights_gb = round(plan.device_weights_gb, 2)
    info.streamer = streamer
    if qc is not None:
        info.notes.append(f"offload: token and output embeddings on the host ({plan.host_gb:.1f} GB)")
    else:
        info.notes.append(
            f"offload: {len(plan.streamed)}/{plan.n_layers} decoder layers + embeddings on the host "
            f"({plan.host_gb:.1f} GB host, ~{plan.device_weights_gb:.1f} GB device weights)"
        )
    return model, processor, info


def _embeddings_to_host(model: Any, backend: Backend, info: LoadInfo) -> None:
    """Move the token / output embeddings (~4 GB in bf16) to the host; they are only ever gathered from."""
    from . import offload as off

    backbone = model.language_model
    lm = backbone.model.language_model
    lm.embed_tokens = off.HostEmbedding(lm.embed_tokens.to("cpu"), backend.device)
    backbone.lm_head = off.HostOutputEmbedding(backbone.lm_head.to("cpu"), backend.device)
    backend.empty_cache()
    info.mode = "offload"
    info.notes.append("offload: token and output embeddings on the host (~4 GB less device memory)")


def _bytes_scale(dtype: Any) -> float:
    """Checkpoint tensors are bf16 (2 bytes); float32 doubles them."""
    return 2.0 if _dtype_name(dtype) == "float32" else 1.0


def _strip_accelerate_hooks(model: Any) -> None:
    """transformers dispatches a device_map with accelerate hooks that leave host params on `meta`; remove
    them (this restores real host tensors) so offload.py owns data movement."""
    from accelerate.hooks import remove_hook_from_module

    remove_hook_from_module(model, recurse=True)
    if hasattr(model, "hf_device_map"):
        with contextlib.suppress(AttributeError):
            del model.hf_device_map


def _as_hit(result: Result) -> Result:
    """A private copy of a stored / shared result marked as served without a forward (`timing.cached`)."""
    out = copy.deepcopy(result)
    out["timing"] = {"queue_ms": 0, "forward_ms": 0, "batch_size": 0, "cached": True}
    return out


@dataclass
class Prepared:
    """Records checked and encoded ahead of `Engine.submit` (tokenize-ahead, D3 cache lookups)."""

    records: list[Record]
    keys: list[str | None]  # cache keys; None = not cacheable (filled by D3)
    hits: dict[int, Result]  # index -> cached result (filled by D3)
    encoded: list[Any]  # Encoded, None at hit positions (a duplicate shares its first occurrence's Encoded)
    dups: dict[int, int] = field(default_factory=dict)  # index -> index of the identical record computed once

    def slice(self, start: int, stop: int) -> Prepared:
        """For bisection in decide_resilient: the same records [start, stop), hits re-indexed."""
        hits = {i - start: r for i, r in self.hits.items() if start <= i < stop}
        dups = {i - start: s - start for i, s in self.dups.items() if start <= i < stop and start <= s < stop}
        return Prepared(self.records[start:stop], self.keys[start:stop], hits, self.encoded[start:stop], dups)

    def __len__(self) -> int:
        return len(self.records)


@dataclass
class _Item:
    enc: Any
    questions: dict[str, Any]
    model: str
    loop: asyncio.AbstractEventLoop
    future: asyncio.Future
    t_enq: float
    is_media: bool = False
    length: int = 0
    extra: dict = field(default_factory=dict)


def _resolve(item: _Item, result: Any = None, exc: BaseException | None = None) -> None:
    def _set() -> None:
        if item.future.done():
            return
        if exc is not None:
            item.future.set_exception(exc)
        else:
            item.future.set_result(result)

    with contextlib.suppress(RuntimeError):  # event loop closed
        item.loop.call_soon_threadsafe(_set)


_KERNEL_PACKAGES = {"fla": "flash-linear-attention", "causal_conv1d": "causal-conv1d"}


class Engine:
    def __init__(
        self,
        cfg: Any,
        stats: Any,
        loader: Callable | None = None,
        backend: Backend | None = None,
        model: DecisionModel | None = None,
    ):
        self.cfg = cfg
        self.stats = stats
        # loader: the old test / bench injection (loader(cfg) -> a joint_schema_model-like namespace),
        # wrapped in ClefReleaseAdapter.from_module at load. model: an adapter instance. Both skip the
        # release dir lookup.
        self._loader = loader
        self._injected_model = model
        self._custom_loader = loader is not None or model is not None
        self._backend: Backend | None = backend
        self._injected_backend = backend is not None
        self._dtype: Any = None
        self._model_path: str | None = cfg.model_path
        self._warnings: list[str] = []
        self._telemetry: dict = {}
        self._versions: dict | None = None
        self.status = "loading"
        self.error: str | None = None
        self.load_seconds: float | None = None
        self.warmup_seconds: float | None = None
        # The served model: resolved now (torch-free), loaded on the worker. An unknown CLEF_MODEL is a load
        # error (status "error"), not a crash.
        self._spec_error: str | None = None
        if model is not None:
            self.spec: ModelSpec = model.spec
        else:
            try:
                self.spec = registry.resolve_spec(cfg)
            except (UnknownModel, ValueError) as exc:
                self._spec_error = str(exc)
                requested = (getattr(cfg, "model", None) or "").partition("@")[0] or registry.DEFAULT_MODEL_ID
                self.spec = ModelSpec(requested, "unknown", None, None, WEIGHTS_GB)
        self.model: DecisionModel | None = None  # set on the worker thread once loaded
        self.cache = ResultCache(
            int(getattr(cfg, "cache_size", 10000)), int(getattr(cfg, "cache_mb", 256)) * 1024 * 1024
        )  # D3
        register = getattr(stats, "set_cache_source", None)
        if register is not None:
            register(self.cache_stats)
        # Two FIFO lanes guarded by one condition; _closed replaces the old shutdown sentinel and _drained
        # refuses late enqueues once the worker has failed the leftovers.
        self._lanes: dict[str, deque[_Item]] = {p: deque() for p in PRIORITIES}
        self._cv = threading.Condition()
        self._closed = False
        self._drained = False
        self._burst = 0  # interactive micro-batches in a row while background work waited
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._started = False
        self._load_info: LoadInfo | None = None
        self._device = torch.device("cpu")

    # ------------------------------------------------------------------ lifecycle
    def start(self) -> None:
        if self._started:
            return
        self._started = True
        self._thread = threading.Thread(target=self._run, name="clef-gpu-worker", daemon=True)
        self._thread.start()

    def shutdown(self, timeout: float = 10.0) -> None:
        self._stop.set()
        with self._cv:
            self._closed = True
            self._cv.notify_all()
        if self._thread is not None:
            self._thread.join(timeout)

    # ------------------------------------------------------------------ model identity
    def model_ref(self) -> str:
        """`<id>@<revision>` (or `<id>@local`) of the served model, e.g. for jobs.model."""
        return self.spec.ref

    def check_model(self, requested: str | None) -> str:
        """The served id when `requested` names it (id, id@<sha or a >= 7-char prefix>, the HF repo), else
        UnknownModel (-> 404). None / "" mean the served model."""
        spec = self.spec
        if not requested:
            return spec.id
        if requested == spec.id or (spec.repo is not None and requested == spec.repo):
            return spec.id
        model_id, sep, rev = requested.partition("@")
        rev = rev.lower()
        if sep and model_id == spec.id and spec.revision and len(rev) >= 7 and spec.revision.startswith(rev):
            return spec.id
        raise UnknownModel(f"model {requested!r} is not served here (this server serves {spec.id!r})")

    def capabilities(self) -> Capabilities:
        if self.model is not None:
            return self.model.capabilities
        if self._injected_model is not None:
            return self._injected_model.capabilities
        cls = registry.adapters().get(self.spec.adapter)
        return getattr(cls, "capabilities", None) or Capabilities()

    def model_info(self) -> dict[str, Any]:
        """/health `model`: the loaded adapter's info(), or the spec before (and after a failed) load."""
        if self.model is not None:
            out = dict(self.model.info())
        else:
            spec = self.spec
            out = {"id": spec.id, "adapter": spec.adapter, "revision": spec.revision, "ref": spec.ref}
            with contextlib.suppress(Exception):
                out["capabilities"] = self.capabilities().to_json()
        out["distribution"] = registry.adapter_origin(self.spec.adapter)
        return out

    # ------------------------------------------------------------------ public API
    def _check_priority(self, priority: str) -> None:
        if priority not in PRIORITIES:
            raise ValueError(f"priority must be one of {', '.join(PRIORITIES)}, got {priority!r}")

    def _check_ready(self) -> None:
        if self.status not in ("ready", "warming") or self._stop.is_set() or self.model is None:
            raise EngineNotReady(self.error or f"engine is {self.status}")

    async def prepare(self, records: list[Record], cache: bool = True) -> Prepared:
        """Check the model and capabilities of every record, then encode them in threads (nothing is queued).

        `cache`: False = `Cache-Control: no-cache`: no lookup (the fresh result is still stored). Hits are
        returned with `timing.cached` true; records that are not cacheable (media) are always computed.
        """
        self._check_ready()
        if not records:
            return Prepared([], [], {}, [])
        model = self.model
        checked: list[Record] = []
        for r in records:
            rec = {**r, "model": self.check_model(r.get("model"))}
            model.validate(rec)
            checked.append(rec)
        n = len(checked)
        keys: list[str | None] = [None] * n
        hits: dict[int, Result] = {}
        dups: dict[int, int] = {}
        if self.cache.enabled:
            ref = self.spec.ref
            first: dict[str, int] = {}
            for i, rec in enumerate(checked):
                key = keys[i] = self.cache.key(ref, rec)
                if key is None or not cache:
                    self.cache.note_bypass()
                    continue
                hit = self.cache.get(key)
                if hit is not None:
                    hits[i] = _as_hit(hit)
                elif key in first:
                    dups[i] = first[key]
                    self.cache.note_dup()
                else:
                    first[key] = i
        todo = [i for i in range(n) if i not in hits and i not in dups]
        done = await asyncio.gather(*(asyncio.to_thread(self._encode, checked[i]) for i in todo))
        self._check_ready()
        encoded: list[Any] = [None] * n
        for i, enc in zip(todo, done, strict=True):
            encoded[i] = enc
        for i, src in dups.items():
            encoded[i] = encoded[src]
        return Prepared(checked, keys, hits, encoded, dups)

    async def submit(self, prepared: Prepared, priority: str = INTERACTIVE) -> list[Result]:
        """Enqueue the prepared misses in the `priority` lane, await them, merge the hits back in order."""
        self._check_priority(priority)
        self._check_ready()
        n = len(prepared.records)
        if not n:
            return []
        loop = asyncio.get_running_loop()
        now = time.perf_counter()
        items: dict[int, _Item] = {}
        for i in range(n):
            if i in prepared.hits or i in prepared.dups:
                continue
            enc = prepared.encoded[i]
            items[i] = _Item(
                enc=enc,
                questions={str(k): v for k, v in prepared.records[i]["questions"].items()},
                model=self.spec.id,
                loop=loop,
                future=loop.create_future(),
                t_enq=now,
                is_media=bool(enc.is_media),
                length=int(enc.length),
            )
        if items:
            with self._cv:
                if self._drained:
                    raise EngineNotReady(self.error or "engine shut down")
                self._lanes[priority].extend(items.values())
                self._cv.notify()
            self._report_depth()
        results = await asyncio.gather(*(it.future for it in items.values()), return_exceptions=True)
        # Store every successful forward, even when a sibling failed; errors and OOM are never cached.
        for i, res in zip(items, results, strict=True):
            key = prepared.keys[i]
            if key is not None and not isinstance(res, BaseException):
                self.cache.put(key, res)
        for res in results:
            if isinstance(res, BaseException):
                raise res
        computed = dict(zip(items, results, strict=True))
        out: list[Result] = []
        for i in range(n):
            if i in prepared.hits:
                out.append(prepared.hits[i])
            elif i in prepared.dups:
                out.append(_as_hit(computed[prepared.dups[i]]))
            else:
                out.append(computed[i])
        return out

    def cache_stats(self) -> dict[str, Any]:
        """Result cache counters (-> /v1/stats `cache`)."""
        return self.cache.stats()

    async def decide(
        self, records: list[Record], priority: str = INTERACTIVE, cache: bool = True
    ) -> list[Result]:
        """= submit(await prepare(records, cache), priority)."""
        self._check_priority(priority)
        return await self.submit(await self.prepare(records, cache=cache), priority)

    def info(self) -> dict:
        """Static facts + live memory. Fast (< 5 ms): telemetry is whatever the last sample() cached."""
        b = self._backend
        gpu: dict[str, Any] = {
            "available": False,
            "name": None,
            "memory_kind": None,
            "vram_total_gb": None,
            "vram_allocated_gb": None,
            "vram_reserved_gb": None,
            "vram_free_gb": None,
        }
        versions: dict[str, Any] = {
            "torch": torch.__version__,
            "cuda": None,
            "hip": None,
            "driver": None,
            "macos": None,
        }
        fast_path: dict[str, Any] = {"causal_conv1d": False, "fla": False, "expected": False}
        if b is not None:
            try:
                mem = b.memory()
                gpu = {
                    "available": b.name != "cpu",
                    "name": b.device_name(),
                    "memory_kind": mem.get("kind"),
                    "vram_total_gb": mem.get("total_gb"),
                    "vram_allocated_gb": mem.get("allocated_gb"),
                    "vram_reserved_gb": mem.get("reserved_gb"),
                    "vram_free_gb": mem.get("free_gb"),
                }
                if self._versions is None:
                    self._versions = b.versions()
                versions = self._versions
                fast_path = self._fast_path(b)
            except Exception as exc:  # never let /health crash
                log.warning("backend info failed: %s", exc)
        # Read the version from package metadata: importing transformers here (from /health) while the loader
        # thread is mid-import races transformers' lazy module and breaks the load with an ImportError.
        try:
            tf_version = importlib.metadata.version("transformers")
        except Exception:
            tf_version = None
        return {
            "memory": self._memory_info(),
            "backend": b.name if b is not None else None,
            "device": str(b.device) if b is not None else self.cfg.device,
            "dtype": _dtype_name(self._dtype) if self._dtype is not None else self.cfg.dtype,
            "quant": self.cfg.quant,
            "model_path": self._model_path,
            "torch": torch.__version__,
            "transformers": tf_version,
            "gpu": gpu,
            "telemetry": {**_EMPTY_TELEMETRY, **self._telemetry},
            "fast_path": fast_path,
            "versions": versions,
            "warnings": list(self._warnings),
        }

    def _fast_path(self, b: Backend) -> dict[str, Any]:
        """backend.fast_path() limited to the kernels the model can use (capabilities.optional_kernels)."""
        fp = dict(b.fast_path())
        kernels = self.capabilities().optional_kernels
        out = {k: v for k, v in fp.items() if k in kernels}
        out["expected"] = bool(fp.get("expected")) and bool(kernels)
        return out

    def _memory_info(self) -> dict[str, Any]:
        """Memory mode in effect (offload, device cap, quant) and what sits on the host."""
        cfg, li, b = self.cfg, self._load_info, self._backend
        quant = getattr(cfg, "quant", "none")
        quant_backend = None
        if quant != "none" and b is not None:
            with contextlib.suppress(Exception):
                quant_backend = backend_mod.quant_method(b, quant, getattr(cfg, "quant_backend", "auto"))
        cap = float(getattr(cfg, "max_device_memory_gb", 0.0) or 0.0)
        mode = li.mode if li is not None else "standard"  # "offload" only when it was actually applied
        host_layers = li.streamed_layers if li is not None else 0
        return {
            "offload": getattr(cfg, "offload", "none"),
            "mode": mode,
            "max_device_memory_gb": cap if cap > 0 else None,
            "quant": quant,
            "quant_backend": quant_backend,
            "embeddings_on_host": mode == "offload",
            "layers_on_host": host_layers,
            "layers_total": li.n_layers if li is not None else 0,
            "host_weights_gb": li.host_weights_gb if li is not None else 0.0,
            "device_weights_gb": li.device_weights_gb if li is not None else 0.0,
        }

    def sample(self) -> dict:
        """Cheap gauges for the time series. Never raises; any thread except the worker's hot path."""
        out: dict[str, Any] = {
            "mem_used_gb": None,
            "mem_total_gb": None,
            "gpu_util_pct": None,
            "gpu_temp_c": None,
            "gpu_power_w": None,
        }
        b = self._backend
        if b is None:
            return out
        if self.status == "loading" and b.name in ("cuda", "rocm"):
            # The loader thread is initialising the GPU runtime (torch calls amdsmi_init / NVML itself).
            # A concurrent amdsmi_init from here segfaulted the process on ROCm / WSL2 (measured, with
            # CLEF_MAX_DEVICE_MEMORY_GB set); the gauges are empty during the load anyway.
            return out
        try:
            mem = b.memory()
            out["mem_used_gb"] = mem.get("allocated_gb")
            out["mem_total_gb"] = mem.get("total_gb")
            if self.cfg.telemetry:
                tel = b.telemetry()
                self._telemetry = tel
                out["gpu_util_pct"] = tel.get("util_pct")
                out["gpu_temp_c"] = tel.get("temp_c")
                out["gpu_power_w"] = tel.get("power_w")
        except Exception as exc:
            log.debug("sample failed: %s", exc)
        return out

    # ------------------------------------------------------------------ caller side
    def _encode(self, record: Record) -> Any:
        """Any thread: the adapter encodes (and maps its own "too long" error to InputTooLarge)."""
        return self.model.encode(record)

    def _report_depth(self) -> None:
        try:
            by_class = {p: len(lane) for p, lane in self._lanes.items()}
            self.stats.set_queue_depth(sum(by_class.values()), by_class=by_class)
        except Exception:
            log.exception("stats.set_queue_depth failed")

    # ------------------------------------------------------------------ worker thread
    def _fail(self, message: str) -> None:
        self.error = message
        self.status = "error"
        log.error("engine error: %s", message)

    def _run(self) -> None:
        if not self._load():
            self._drain_fail()
            return
        warm = self._warmup_steps() if self.cfg.warmup else iter(())
        if self.cfg.warmup:
            self.status = "warming"
        warm_t0 = time.perf_counter()
        warm_done = not self.cfg.warmup
        if warm_done:
            self.status = "ready"
        while not self._stop.is_set():
            timeout = 0.0 if not warm_done else 0.5
            batch = self._collect(timeout)
            if batch is None:  # sentinel
                break
            if batch:
                self._process(batch)
                continue
            if not warm_done:  # idle: advance warmup one step
                try:
                    next(warm)
                except StopIteration:
                    warm_done = True
                    self.warmup_seconds = round(time.perf_counter() - warm_t0, 2)
                    self.status = "ready"
                    log.info("warmup done in %.1fs", self.warmup_seconds)
                    # a final empty-queue check happens on the next loop turn
        self._drain_fail()

    def _make_model(self) -> DecisionModel:
        if self._injected_model is not None:
            return self._injected_model
        if self._loader is not None:
            from .models.clef_release import ClefReleaseAdapter

            return ClefReleaseAdapter.from_module(self._loader, self.spec, self.cfg)
        return registry.create(self.cfg, self.spec)

    def _kernel_warnings(self, b: Backend, kernels: tuple[str, ...]) -> None:
        if not kernels:
            return
        fp = b.fast_path()
        if b.name == "cuda" and not all(fp.get(k) for k in kernels):
            names = " ".join(_KERNEL_PACKAGES.get(k, k) for k in kernels)
            self._warnings.append(
                f"fast path kernels missing (pip install {names}); the slow torch fallback is used"
            )
        elif b.name == "rocm" and "fla" in kernels and not fp.get("fla"):
            self._warnings.append("flash-linear-attention is not installed; the slow torch fallback is used")

    def _load(self) -> bool:
        """Detect backend -> dtype -> adapter -> model path -> preflight -> adapter.load. One-line errors."""
        cfg = self.cfg
        t0 = time.perf_counter()
        try:
            if self._spec_error:
                raise UnknownModel(self._spec_error)
            b = self._backend or backend_mod.detect(cfg.device)
            self._backend = b
            with contextlib.suppress(Exception):
                b.telemetry_enabled = bool(cfg.telemetry)
            dtype, warn = b.resolve_dtype(cfg.dtype)
            self._dtype = dtype
            if warn:
                self._warnings.append(warn)
                log.warning("%s", warn)
            self._device = b.device
            tuned = os.environ.get("PYTORCH_TUNABLEOP_FILENAME")  # per model (backend.prepare_environment)
            if tuned and os.environ.get("PYTORCH_TUNABLEOP_ENABLED") == "1":
                with contextlib.suppress(OSError):
                    Path(tuned).parent.mkdir(parents=True, exist_ok=True)
            adapter = self._make_model()
            path = adapter.resolve()
            self._model_path = None if path is None else str(path)
            self._kernel_warnings(b, adapter.capabilities.optional_kernels)
            # Preflight needs real memory numbers; with an injected loader (tests) only run it if the backend
            # is injected too, so unit tests never depend on the host's RAM.
            if cfg.preflight and (not self._custom_loader or self._injected_backend):
                self._warnings.extend(
                    backend_mod.preflight(
                        b,
                        dtype,
                        cfg.quant,
                        adapter.weights_gb(path) if path else self.spec.weights_gb,
                        cap_gb=getattr(cfg, "max_device_memory_gb", 0.0),
                        offload=getattr(cfg, "offload", "none"),
                    )
                )
            load_info = adapter.load(b, dtype)
            self._load_info = load_info
            for note in load_info.notes:
                log.info("%s", note)
            for warning in load_info.warnings:
                log.warning("%s", warning)
                if warning not in self._warnings:
                    self._warnings.append(warning)
            self.model = adapter
            self.cache.clear()  # D3: results of any earlier model are never served
        except (BackendError, PreflightError, ModelNotFound, UnknownModel) as exc:
            self._fail(" ".join(str(exc).split()))
            return False
        except Exception as exc:
            log.exception("model load failed")
            self._fail(f"{type(exc).__name__}: {exc}")
            return False
        self.load_seconds = round(time.perf_counter() - t0, 2)
        log.info(
            "model %s (adapter %s from %s) loaded in %.1fs on %s/%s (%s)",
            self.spec.ref,
            self.spec.adapter,
            registry.adapter_origin(self.spec.adapter) or "?",
            self.load_seconds,
            b.name,
            b.device,
            _dtype_name(self._dtype),
        )
        return True

    def _drain_fail(self) -> None:
        with self._cv:
            self._drained = True
            left = [it for lane in self._lanes.values() for it in lane]
            for lane in self._lanes.values():
                lane.clear()
        for it in left:
            _resolve(it, exc=EngineNotReady(self.error or "engine shut down"))
        self._report_depth()

    def _pending(self) -> int:
        return sum(len(lane) for lane in self._lanes.values())

    def _collect(self, first_timeout: float) -> list[_Item] | None:
        """Block up to first_timeout for a first item, let the batch window run, then take one lane's items.

        Returns [] if nothing arrived, None on shutdown with nothing pending. The lane is chosen after the
        window, so interactive work that arrives during a window opened by background work still goes first.
        """
        cap = max(1, int(self.cfg.max_microbatch))
        with self._cv:
            if not self._pending() and not self._closed and first_timeout > 0:
                self._cv.wait_for(lambda: self._pending() or self._closed, first_timeout)
            if not self._pending():
                return None if self._closed else []
            deadline = time.perf_counter() + self.cfg.batch_window_ms / 1000.0
            while self._pending() < cap and not self._closed:
                remaining = deadline - time.perf_counter()
                if remaining <= 0:
                    break
                self._cv.wait(remaining)
            lane = self._pick_lane()
            batch = [lane.popleft() for _ in range(min(cap, len(lane)))]
        self._report_depth()
        return batch

    def _pick_lane(self) -> deque[_Item]:
        """Interactive first; one background micro-batch after cfg.interactive_burst interactive in a row."""
        inter, back = self._lanes[INTERACTIVE], self._lanes[BACKGROUND]
        burst = max(1, int(getattr(self.cfg, "interactive_burst", 8)))
        if inter and (not back or self._burst < burst):
            self._burst = self._burst + 1 if back else 0
            return inter
        self._burst = 0
        return back

    def _group(self, items: list[_Item]) -> list[list[_Item]]:
        groups: list[list[_Item]] = [[it] for it in items if it.is_media]
        text = sorted((it for it in items if not it.is_media), key=lambda i: i.length)
        by_bucket: dict[int, list[_Item]] = {}
        for it in text:
            by_bucket.setdefault(bucket_for(it.length, self.cfg.buckets), []).append(it)
        for b in sorted(by_bucket):
            groups.append(by_bucket[b])
        return groups

    def _process(self, items: list[_Item]) -> None:
        live = [it for it in items if not it.future.done()]
        for group in self._group(live):
            self._run_group(group)

    def _run_group(self, group: list[_Item]) -> None:
        t_start = time.perf_counter()
        model = self.model
        try:
            logits, ms, n_tokens, padded = self._forward([it.enc for it in group], media=group[0].is_media)
            for it, rec_logits in zip(group, logits, strict=True):
                try:
                    answers = {}
                    for q, ql in zip(it.enc.questions, rec_logits, strict=True):
                        answers[q.question_id] = model.answer(q, it.questions[q.question_id], ql)
                    _resolve(
                        it,
                        result={
                            "model": it.model,
                            "answers": answers,
                            "usage": {"input_tokens": it.length, "output_tokens": 0},
                            "timing": {
                                "queue_ms": round((t_start - it.t_enq) * 1000, 3),
                                "forward_ms": round(ms, 3),
                                "batch_size": len(group),
                                "cached": False,
                            },
                        },
                    )
                except Exception as exc:
                    log.exception("answer build failed")
                    _resolve(it, exc=exc)
            try:
                self.stats.record_forward(len(group), n_tokens, padded, ms)
            except Exception:
                log.exception("stats.record_forward failed")
        except Exception as exc:
            self._reset_streamer()
            if self._backend.is_oom(exc):
                log.error("device OOM on group of %d (tokens=%d)", len(group), sum(i.length for i in group))
                self._backend.empty_cache()
                for it in group:
                    _resolve(it, exc=GpuOutOfMemory(f"GPU out of memory: {exc}"))
            else:
                log.exception("forward failed")
                for it in group:
                    _resolve(it, exc=exc)

    def _reset_streamer(self) -> None:
        """After a failed forward, put offloaded layers back on the host (their hooks did not finish)."""
        streamer = self._load_info.streamer if self._load_info else None
        if streamer is not None:
            try:
                streamer.reset()
            except Exception:
                log.exception("offload reset failed")

    def padded_length(self, length: int) -> int:
        """Text-batch sequence length after padding: the bucket (opt-in) or the next pad_multiple."""
        if getattr(self.cfg, "pad_to_bucket", False):
            return bucket_for(length, self.cfg.buckets)
        multiple = max(1, int(getattr(self.cfg, "pad_multiple", 1)))
        return -(-length // multiple) * multiple

    def _forward(self, encs: list[Any], media: bool) -> tuple[list, float, int, int]:
        """One batched forward: collate -> to_device -> forward. Returns (per-record logits, ms, real tokens,
        padded tokens). Text batches are right-padded per the engine's policy; media records are not."""
        model = self.model
        pad_to = None if media else self.padded_length(max(e.length for e in encs))
        batch = model.collate(encs, pad_to)
        padded_len = batch["input_ids"].shape[1]
        batch = model.to_device(batch, self._device)
        t0 = time.perf_counter()
        with torch.inference_mode():
            logits = model.forward(batch)
            self._backend.synchronize()
        ms = (time.perf_counter() - t0) * 1000
        return logits, ms, sum(e.length for e in encs), padded_len * len(encs)

    # ------------------------------------------------------------------ warmup
    def _warm_encode(self, repeats: int) -> Any:
        model = self.model
        return model.encode(model.warm_record(repeats))

    def _warmup_steps(self):
        """Generator: each next() runs one warmup forward (so queued requests can interleave)."""
        buckets = [b for b in sorted(self.cfg.buckets) if b <= 1024]
        sizes = sorted({1, max(1, self.cfg.max_microbatch)})
        try:
            base = self._warm_encode(0).length
            per = max(1, self._warm_encode(1).length - base)
        except Exception:
            log.exception("warmup disabled: could not synthesize record")
            return
        seen: set[int] = set()
        total = len(buckets) * len(sizes)
        done = 0
        for b in buckets:
            if base > b:
                continue
            repeats = (b - base) // per
            try:
                enc = self._warm_encode(repeats)
                while enc.length > b and repeats > 0:
                    repeats -= 1
                    enc = self._warm_encode(repeats)
            except Exception:
                log.exception("warmup encode failed for bucket %d", b)
                continue
            real = bucket_for(enc.length, self.cfg.buckets)
            if real in seen:
                continue
            seen.add(real)
            for n in sizes:
                done += 1
                try:
                    _, ms, _, _ = self._forward([enc] * n, media=False)
                    log.info("warmup %d/%d bucket=%d batch=%d %.0f ms", done, total, real, n, ms)
                except Exception as exc:
                    if self._backend.is_oom(exc):
                        self._backend.empty_cache()
                    log.warning("warmup bucket=%d batch=%d failed: %s", real, n, exc)
                yield
