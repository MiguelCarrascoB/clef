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
from typing import Any

import torch

log = logging.getLogger("clef.engine")

Record = dict
Result = dict

_SENTINEL = object()
_WARMUP_SENTENCE = "The customer reported that the service was slow yesterday afternoon. "


class EngineNotReady(RuntimeError):
    """Model not loaded yet, failed to load, or engine shut down (-> HTTP 503)."""


class InputTooLarge(ValueError):
    """Schema alone exceeds max_tokens (-> HTTP 413)."""


class GpuOutOfMemory(RuntimeError):
    """CUDA/HIP out of memory during a forward (-> HTTP 503)."""


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


def _default_loader(cfg: Any) -> Any:
    if cfg.model_path not in sys.path:
        sys.path.insert(0, cfg.model_path)
    return importlib.import_module("joint_schema_model")


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
    def __init__(self, cfg: Any, stats: Any, loader: Callable | None = None):
        self.cfg = cfg
        self.stats = stats
        self._loader = loader or _default_loader
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
        gpu: dict[str, Any] = {
            "available": False,
            "name": None,
            "vram_total_gb": None,
            "vram_allocated_gb": None,
            "vram_reserved_gb": None,
        }
        try:
            if str(self.cfg.device).startswith("cuda") and torch.cuda.is_available():
                dev = torch.device(self.cfg.device)
                gb = 1024**3
                gpu = {
                    "available": True,
                    "name": torch.cuda.get_device_name(dev),
                    "vram_total_gb": round(torch.cuda.get_device_properties(dev).total_memory / gb, 2),
                    "vram_allocated_gb": round(torch.cuda.memory_allocated(dev) / gb, 2),
                    "vram_reserved_gb": round(torch.cuda.memory_reserved(dev) / gb, 2),
                }
        except Exception as exc:  # never let /health crash
            log.warning("gpu info failed: %s", exc)
        # Read the version from package metadata: importing transformers here (from /health) while the loader
        # thread is mid-import races transformers' lazy module and breaks the load with an ImportError.
        try:
            tf_version = importlib.metadata.version("transformers")
        except Exception:
            tf_version = None
        return {
            "device": self.cfg.device,
            "dtype": self.cfg.dtype,
            "model_path": self.cfg.model_path,
            "torch": torch.__version__,
            "transformers": tf_version,
            "gpu": gpu,
        }

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
        dtype = getattr(torch, str(self.cfg.dtype), None)
        if not isinstance(dtype, torch.dtype):
            self._fail(f"unknown dtype {self.cfg.dtype!r} (use e.g. bfloat16, float16, float32)")
            return False
        t0 = time.perf_counter()
        try:
            self._mod = self._loader(self.cfg)
            model, processor = self._mod.load_release_model(
                self.cfg.model_path, device=self.cfg.device, dtype=dtype
            )
            model.eval()
            self._model, self._processor = model, processor
            pad = getattr(processor.tokenizer, "pad_token_id", None)
            self._pad_id = 0 if pad is None else int(pad)
            self._device = torch.device(self.cfg.device)
        except Exception as exc:
            log.exception("model load failed")
            self._fail(f"{type(exc).__name__}: {exc}")
            return False
        self.load_seconds = round(time.perf_counter() - t0, 2)
        log.info("model loaded in %.1fs on %s (%s)", self.load_seconds, self.cfg.device, self.cfg.dtype)
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
        except torch.cuda.OutOfMemoryError as exc:
            log.error("GPU OOM on group of %d (tokens=%d)", len(group), sum(i.length for i in group))
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            for it in group:
                _resolve(it, exc=GpuOutOfMemory(f"GPU out of memory: {exc}"))
        except Exception as exc:
            log.exception("forward failed")
            for it in group:
                _resolve(it, exc=exc)

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
        is_cuda = self._device.type == "cuda"
        t0 = time.perf_counter()
        with torch.inference_mode():
            logits = self._model(batch)
            if is_cuda:
                torch.cuda.synchronize()
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
                    if isinstance(exc, torch.cuda.OutOfMemoryError) and torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    log.warning("warmup bucket=%d batch=%d failed: %s", real, n, exc)
                yield
