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
import importlib
import importlib.metadata
import logging
import queue
import sys
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch

from . import backend as backend_mod
from .backend import Backend, BackendError, PreflightError
from .paths import WEIGHTS_GB, ModelNotFound, resolve_model_path

log = logging.getLogger("clef.engine")

Record = dict
Result = dict

_SENTINEL = object()
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


def _import_model_module(path: Any) -> Any:
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)
    return importlib.import_module("joint_schema_model")


def _default_loader(cfg: Any) -> Any:
    return _import_model_module(resolve_model_path(cfg))


def _weights_gb(path: Any) -> float:
    """On-disk size of the release weights (falls back to the measured 19 GB)."""
    try:
        total = sum(f.stat().st_size for f in Path(path).glob("*.safetensors"))
        return total / 1024**3 if total > 1024**3 else WEIGHTS_GB
    except Exception:
        return WEIGHTS_GB


@dataclass
class LoadInfo:
    """What `load_model` did, for logs, /health warnings and the memory benchmark."""

    mode: str = "standard"  # standard | offload
    streamed_layers: int = 0
    n_layers: int = 0
    host_weights_gb: float = 0.0
    device_weights_gb: float = 0.0
    notes: list[str] = field(default_factory=list)
    streamer: Any = None


def _file_gb(path: Path) -> float:
    try:
        return path.stat().st_size / 1024**3
    except OSError:
        return 0.0


def device_budget_gb(backend: Backend, cfg: Any) -> float | None:
    """Device memory for the weights: the explicit cap, else what is free now, minus activation headroom."""
    cap = float(getattr(cfg, "max_device_memory_gb", 0.0) or 0.0)
    if cap > 0:
        total = cap
    else:
        free = backend.memory().get("free_gb")
        if free is None:
            return None
        total = float(free)
    return max(total - backend_mod.ACTIVATION_RESERVE_GB, 0.0)


def load_model(mod: Any, path: Any, backend: Backend, dtype: Any, cfg: Any) -> tuple[Any, Any, LoadInfo]:
    """Load (model, processor, info) honouring cfg.quant / cfg.offload / cfg.max_device_memory_gb.

    The standard path (no offload) is the model's own `load_release_model`, optionally with a quantization
    config. With CLEF_OFFLOAD=cpu the backbone is loaded with a per-module device map (embeddings and the
    layers that do not fit under the cap on the host) and the clef model is assembled here from the release's
    own classes, so the model directory is never touched. See offload.py.
    """
    info = LoadInfo()
    quant = getattr(cfg, "quant", "none")
    offload = getattr(cfg, "offload", "none")
    cap = float(getattr(cfg, "max_device_memory_gb", 0.0) or 0.0)
    if cap > 0 and backend.apply_memory_cap(cap):
        info.notes.append(f"device memory capped at {cap:g} GB")
    if offload == "cpu" and not backend.has_discrete_memory:
        info.notes.append(
            f"CLEF_OFFLOAD=cpu has no effect on {backend.name} (device and host share one memory pool)"
        )
        offload = "none"
    qc = backend_mod.quantization_config(backend, quant, dtype, getattr(cfg, "quant_backend", "auto"))
    if offload == "cpu":
        try:
            return _load_offloaded(mod, path, backend, dtype, cfg, info, qc)
        except (PreflightError, BackendError):
            raise
        except Exception as exc:
            if qc is None:
                raise
            # transformers refused host entries in the device_map for this quantizer: load, then move them
            log.warning(
                "direct host placement failed with %s (%s); moving the embeddings after load", quant, exc
            )
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


def _load_offloaded(
    mod: Any, path: Any, backend: Backend, dtype: Any, cfg: Any, info: LoadInfo, qc: Any = None
):
    import json

    from safetensors.torch import load_file
    from transformers import AutoProcessor, Qwen3_5ForConditionalGeneration

    from . import offload as off

    root = Path(path)
    sizes = off.checkpoint_sizes(root)
    layer_bytes, embed_bytes, other_bytes, _ = off.split_sizes(sizes, _bytes_scale(dtype))
    head_bytes = int(_file_gb(root / "joint_head.safetensors") * 1024**3 * _bytes_scale(dtype))
    budget_gb = device_budget_gb(backend, cfg)
    if budget_gb is None or qc is not None:  # unknown memory, or quantized layers: only the vocabulary moves
        budget_gb = 1e9
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
        root, dtype=dtype, device_map=plan.device_map(device), **extra
    )
    _strip_accelerate_hooks(backbone)
    backbone.config.use_cache = False
    lm = backbone.model.language_model
    lm.embed_tokens = off.HostEmbedding(lm.embed_tokens, backend.device)
    backbone.lm_head = off.HostOutputEmbedding(backbone.lm_head, backend.device)
    streamer = None
    if plan.streamed:
        streamer = off.LayerStreamer(
            list(lm.layers), plan.streamed, backend.host_copier(), lookahead=2, pin=backend.pin_host
        )
        streamer.install()
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
        with contextlib.suppress(Exception):
            del model.hf_device_map


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


class Engine:
    def __init__(self, cfg: Any, stats: Any, loader: Callable | None = None, backend: Backend | None = None):
        self.cfg = cfg
        self.stats = stats
        self._custom_loader = loader is not None
        self._loader = loader or _default_loader
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
        self._q: queue.Queue = queue.Queue()
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._started = False
        self._encode_lock = threading.Lock()
        self._mod: Any = None
        self._load_info: LoadInfo | None = None
        self._model: Any = None
        self._processor: Any = None
        self._pad_id = 0
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
        self._q.put(_SENTINEL)
        if self._thread is not None:
            self._thread.join(timeout)

    # ------------------------------------------------------------------ public API
    async def decide(self, records: list[Record]) -> list[Result]:
        if self.status not in ("ready", "warming") or self._stop.is_set():
            raise EngineNotReady(self.error or f"engine is {self.status}")
        if not records:
            return []
        loop = asyncio.get_running_loop()
        encs = await asyncio.gather(*(asyncio.to_thread(self._encode, r) for r in records))
        if self.status not in ("ready", "warming") or self._stop.is_set():
            raise EngineNotReady(self.error or f"engine is {self.status}")
        now = time.perf_counter()
        items = [
            _Item(
                enc=enc,
                questions={str(k): v for k, v in r["questions"].items()},
                model=r.get("model") or "clef-flash",
                loop=loop,
                future=loop.create_future(),
                t_enq=now,
                is_media=enc.media is not None,
                length=len(enc.input_ids),
            )
            for r, enc in zip(records, encs, strict=True)
        ]
        for it in items:
            self._q.put(it)
        self._report_depth()
        results = await asyncio.gather(*(it.future for it in items), return_exceptions=True)
        for res in results:
            if isinstance(res, BaseException):
                raise res
        return list(results)

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
                fast_path = b.fast_path()
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
        try:
            with self._encode_lock:  # HF tokenizers are not safe under concurrent use in all versions
                return self._mod.encode_record(
                    self._processor.tokenizer,
                    record,
                    max_length=self.cfg.max_tokens,
                    processor=self._processor,
                )
        except ValueError as exc:
            if "schema requires" in str(exc):
                raise InputTooLarge(str(exc)) from exc
            raise

    def _report_depth(self) -> None:
        try:
            self.stats.set_queue_depth(self._q.qsize())
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

    def _load(self) -> bool:
        """Detect backend -> dtype -> model path -> preflight -> quant config -> load. One-line errors."""
        cfg = self.cfg
        t0 = time.perf_counter()
        try:
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
            if self._custom_loader:
                self._mod = self._loader(cfg)
                path = cfg.model_path
            else:
                path = resolve_model_path(cfg)
                self._mod = _import_model_module(path)
            self._model_path = None if path is None else str(path)
            fp = b.fast_path()
            if b.name == "cuda" and not (fp["fla"] and fp["causal_conv1d"]):
                self._warnings.append(
                    "fast path kernels missing (pip install flash-linear-attention causal-conv1d); "
                    "the slow torch fallback is used"
                )
            elif b.name == "rocm" and not fp["fla"]:
                self._warnings.append(
                    "flash-linear-attention is not installed; the slow torch fallback is used"
                )
            # Preflight needs real memory numbers; with an injected loader (tests) only run it if the backend
            # is injected too, so unit tests never depend on the host's RAM.
            if cfg.preflight and (not self._custom_loader or self._injected_backend):
                self._warnings.extend(
                    backend_mod.preflight(
                        b,
                        dtype,
                        cfg.quant,
                        _weights_gb(path) if path else WEIGHTS_GB,
                        cap_gb=getattr(cfg, "max_device_memory_gb", 0.0),
                        offload=getattr(cfg, "offload", "none"),
                    )
                )
            model, processor, load_info = load_model(self._mod, path, b, dtype, cfg)
            self._load_info = load_info
            for note in load_info.notes:
                log.info("%s", note)
                if "no effect" in note:
                    self._warnings.append(note)
            model.eval()
            self._model, self._processor = model, processor
            pad = getattr(processor.tokenizer, "pad_token_id", None)
            self._pad_id = 0 if pad is None else int(pad)
        except (BackendError, PreflightError, ModelNotFound) as exc:
            self._fail(" ".join(str(exc).split()))
            return False
        except Exception as exc:
            log.exception("model load failed")
            self._fail(f"{type(exc).__name__}: {exc}")
            return False
        self.load_seconds = round(time.perf_counter() - t0, 2)
        log.info(
            "model loaded in %.1fs on %s/%s (%s)",
            self.load_seconds,
            b.name,
            b.device,
            _dtype_name(self._dtype),
        )
        return True

    def _drain_fail(self) -> None:
        while True:
            try:
                it = self._q.get_nowait()
            except queue.Empty:
                break
            if it is not _SENTINEL:
                _resolve(it, exc=EngineNotReady(self.error or "engine shut down"))
        self._report_depth()

    def _collect(self, first_timeout: float) -> list[_Item] | None:
        """Block up to first_timeout for a first item, then drain for the batch window.

        Returns [] if nothing arrived, None if the shutdown sentinel was seen with nothing pending.
        """
        try:
            first = self._q.get(timeout=first_timeout) if first_timeout > 0 else self._q.get_nowait()
        except queue.Empty:
            return []
        if first is _SENTINEL:
            return None
        batch = [first]
        deadline = time.perf_counter() + self.cfg.batch_window_ms / 1000.0
        while len(batch) < self.cfg.max_microbatch:
            remaining = deadline - time.perf_counter()
            try:
                nxt = self._q.get(timeout=remaining) if remaining > 0 else self._q.get_nowait()
            except queue.Empty:
                break
            if nxt is _SENTINEL:
                self._stop.set()
                break
            batch.append(nxt)
        self._report_depth()
        return batch

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
        try:
            logits, ms, n_tokens, padded = self._forward([it.enc for it in group], media=group[0].is_media)
            for it, rec_logits in zip(group, logits, strict=True):
                try:
                    answers = {}
                    for q, ql in zip(it.enc.questions, rec_logits, strict=True):
                        probs = dict(zip(q.option_ids, ql.float().softmax(-1).tolist(), strict=True))
                        answers[q.question_id] = self._mod.systemone_answer(
                            it.questions[q.question_id], probs
                        )
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
        """One batched forward. Returns (per-record logits, ms, real tokens, padded tokens)."""
        batch = self._mod.collate_records(encs, self._pad_id, torch.device("cpu"))
        length = batch["input_ids"].shape[1]
        if not media:
            batch = pad_batch_right(batch, self.padded_length(length), self._pad_id)
        padded_len = batch["input_ids"].shape[1]
        batch = self._to_device(batch)
        t0 = time.perf_counter()
        with torch.inference_mode():
            logits = self._model(batch)
            self._backend.synchronize()
        ms = (time.perf_counter() - t0) * 1000
        return logits, ms, sum(len(e.input_ids) for e in encs), padded_len * len(encs)

    def _to_device(self, batch: dict[str, Any]) -> dict[str, Any]:
        dev = self._device
        out = dict(batch)
        for key in ("input_ids", "attention_mask"):
            out[key] = batch[key].to(dev, non_blocking=True)
        out["media"] = {k: v.to(dev, non_blocking=True) for k, v in (batch.get("media") or {}).items()}
        return out

    # ------------------------------------------------------------------ warmup
    def _warm_record(self, repeats: int) -> Record:
        return {
            "model": "clef-flash",
            "id": "warmup",
            "state": _WARMUP_SENTENCE * repeats,
            "questions": {
                "q": {"type": "choice", "instructions": "warmup", "criteria": {"a": "first", "b": "second"}}
            },
        }

    def _warm_encode(self, repeats: int) -> Any:
        with self._encode_lock:
            return self._mod.encode_record(
                self._processor.tokenizer,
                self._warm_record(repeats),
                max_length=self.cfg.max_tokens,
                processor=self._processor,
            )

    def _warmup_steps(self):
        """Generator: each next() runs one warmup forward (so queued requests can interleave)."""
        buckets = [b for b in sorted(self.cfg.buckets) if b <= 1024]
        sizes = sorted({1, max(1, self.cfg.max_microbatch)})
        try:
            base = len(self._warm_encode(0).input_ids)
            per = max(1, len(self._warm_encode(1).input_ids) - base)
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
                while len(enc.input_ids) > b and repeats > 0:
                    repeats -= 1
                    enc = self._warm_encode(repeats)
            except Exception:
                log.exception("warmup encode failed for bucket %d", b)
                continue
            real = bucket_for(len(enc.input_ids), self.cfg.buckets)
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
