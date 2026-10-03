"""Device backends: the only module with device-specific code (see docs/ARCHITECTURE.md).

Importing this module must NOT import torch: `clef_server/__init__.py` calls `prepare_environment()` before
torch is loaded so per-platform env vars (ROCm / MPS) take effect. Anything needing torch imports it lazily.
"""

from __future__ import annotations

import contextlib
import functools
import importlib.metadata
import importlib.util
import json
import logging
import os
import platform
import shutil
import subprocess
import sys
import threading
import time
from collections.abc import MutableMapping
from dataclasses import dataclass, field
from typing import Any

log = logging.getLogger("clef.backend")

GB = 1024**3
BACKENDS = ("cuda", "rocm", "mps", "cpu")

# Weights are ~19 GB in bf16. Multiplier of the bf16 size each dtype / quantization needs resident.
_SIZE_FACTOR = {"bfloat16": 1.0, "float16": 1.0, "float32": 2.0}
_QUANT_FACTOR = {"int8": 0.55, "nf4": 0.32}
_HEADROOM_GB = 3.0  # activations + CUDA context + allocator slack; below need+headroom is a warning only


TELEMETRY_RETRY_S = 60.0  # back-off after a telemetry runtime error


class BackendError(RuntimeError):
    """Unavailable explicit device, or an unsupported quantization / dtype combination."""


class PreflightError(RuntimeError):
    """Not enough free memory for the chosen dtype / quantization."""


# ---------------------------------------------------------------------------------------------- environment
def _installed_torch_version() -> str | None:
    try:
        return importlib.metadata.version("torch")
    except Exception:
        return None


def _is_wsl() -> bool:
    return os.path.exists("/dev/dxg")


def prepare_environment(environ: MutableMapping[str, str] = os.environ) -> list[str]:
    """Torch-free. setdefault() the per-platform env BEFORE torch is imported. Returns the names it set.

    ROCm variables are applied only when the installed torch is a +rocm build, DXG detection only on WSL
    (/dev/dxg), MPS fallback only on macOS. HF_HUB_OFFLINE is deliberately not set (`clef download` needs it).
    """
    wanted: dict[str, str] = {}
    version = _installed_torch_version() or ""
    if "+rocm" in version:
        wanted["TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL"] = "1"
        if _is_wsl():
            wanted["HSA_ENABLE_DXG_DETECTION"] = "1"
        if environ.get("CLEF_TUNABLEOP") == "1":
            wanted["PYTORCH_TUNABLEOP_ENABLED"] = "1"
            try:
                from .config import default_state_dir

                wanted["PYTORCH_TUNABLEOP_FILENAME"] = str(default_state_dir() / "tunableop_%d.csv")
            except Exception:  # pragma: no cover - cosmetic
                pass
    if sys.platform == "darwin":
        wanted["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
    applied = []
    for key, value in wanted.items():
        if key not in environ:
            environ[key] = value
            applied.append(key)
    return applied


# ---------------------------------------------------------------------------------------------- helpers
def _torch() -> Any:
    import torch

    return torch


def _gb(value: float | int | None) -> float | None:
    return None if value is None else round(value / GB, 2)


def _has_module(name: str) -> bool:
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError):
        return False


def _macos_version() -> tuple[int, ...] | None:
    if sys.platform != "darwin":
        return None
    raw = platform.mac_ver()[0]
    try:
        return tuple(int(p) for p in raw.split(".") if p)
    except ValueError:
        return None


def _run(cmd: list[str], timeout: float = 3.0) -> str | None:
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, check=False)
    except Exception:
        return None
    return out.stdout if out.returncode == 0 else None


@functools.lru_cache(maxsize=1)
def _cpu_brand() -> str:
    if sys.platform == "darwin":
        out = _run(["sysctl", "-n", "machdep.cpu.brand_string"])
        if out and out.strip():
            return out.strip()
    try:
        with open("/proc/cpuinfo", encoding="utf-8") as fh:
            for line in fh:
                if line.lower().startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or platform.machine() or "CPU"


def _cpu_memory_bytes() -> tuple[int | None, int | None, int | None]:
    """(rss, total, available) in bytes. psutil when installed, /proc on Linux, else None."""
    try:
        import psutil

        vm = psutil.virtual_memory()
        return psutil.Process().memory_info().rss, vm.total, vm.available
    except Exception:
        pass
    try:
        total = avail = None
        with open("/proc/meminfo", encoding="utf-8") as fh:
            for line in fh:
                if line.startswith("MemTotal:"):
                    total = int(line.split()[1]) * 1024
                elif line.startswith("MemAvailable:"):
                    avail = int(line.split()[1]) * 1024
        with open("/proc/self/statm", encoding="utf-8") as fh:
            rss = int(fh.read().split()[1]) * os.sysconf("SC_PAGE_SIZE")
        return rss, total, avail
    except Exception:
        return None, None, None


def _empty_telemetry() -> dict:
    return {
        "available": False,
        "source": None,
        "util_pct": None,
        "temp_c": None,
        "power_w": None,
        "mem_used_gb": None,
        "mem_total_gb": None,
    }


# ---------------------------------------------------------------------------------------------- Backend
@dataclass
class Backend:
    name: str  # "cuda" | "rocm" | "mps" | "cpu"
    device: Any  # torch.device: cuda:N for cuda AND rocm, mps, cpu
    index: int = 0
    telemetry_enabled: bool = True
    _cache: dict = field(default_factory=dict, repr=False, compare=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False, compare=False)

    # ---- device operations (hot path: no extra work)
    def synchronize(self) -> None:
        if self.name in ("cuda", "rocm"):
            _torch().cuda.synchronize(self.device)
        elif self.name == "mps":
            _torch().mps.synchronize()

    def empty_cache(self) -> None:
        try:
            torch = _torch()
            if self.name in ("cuda", "rocm"):
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            elif self.name == "mps":
                torch.mps.empty_cache()
        except Exception as exc:  # never mask the original OOM
            log.debug("empty_cache failed: %s", exc)

    def is_oom(self, exc: BaseException) -> bool:
        torch = _torch()
        oom_types: list[type] = [MemoryError]
        for owner, attr in ((torch, "OutOfMemoryError"), (torch.cuda, "OutOfMemoryError")):
            t = getattr(owner, attr, None)
            if isinstance(t, type):
                oom_types.append(t)
        if isinstance(exc, tuple(oom_types)):
            return True
        # MPS raises a plain RuntimeError("MPS backend out of memory ...")
        return self.name == "mps" and isinstance(exc, RuntimeError) and "out of memory" in str(exc).lower()

    # ---- identity
    def device_name(self) -> str:
        if "name" in self._cache:
            return self._cache["name"]
        name: str
        try:
            if self.name in ("cuda", "rocm"):
                name = _torch().cuda.get_device_name(self.device)
            elif self.name == "mps":
                brand = _cpu_brand()
                name = brand if brand.startswith("Apple") else "Apple GPU"
            else:
                name = _cpu_brand()
        except Exception as exc:
            log.debug("device_name failed: %s", exc)
            name = self.name
        self._cache["name"] = name
        return name

    def versions(self) -> dict:
        if "versions" not in self._cache:
            torch = _torch()
            macos = platform.mac_ver()[0] if sys.platform == "darwin" else None
            self._cache["versions"] = {
                "torch": torch.__version__,
                "cuda": getattr(torch.version, "cuda", None),
                "hip": getattr(torch.version, "hip", None),
                "driver": self._driver_version(),
                "macos": macos or None,
            }
        return dict(self._cache["versions"])

    def _driver_version(self) -> str | None:
        if self.name != "cuda":
            return None
        try:
            import pynvml

            pynvml.nvmlInit()
            try:
                drv = pynvml.nvmlSystemGetDriverVersion()
            finally:
                with contextlib.suppress(Exception):
                    pynvml.nvmlShutdown()
            return drv.decode() if isinstance(drv, bytes) else str(drv)
        except Exception:
            out = _run(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"])
            return out.strip().splitlines()[0] if out and out.strip() else None

    def fast_path(self) -> dict:
        """Optional linear-attention kernels. `expected`: this backend should have them."""
        conv = _has_module("causal_conv1d")
        fla = _has_module("fla")
        return {"causal_conv1d": conv, "fla": fla, "expected": self.name in ("cuda", "rocm")}

    # ---- dtype
    def _bf16_supported(self) -> bool:
        torch = _torch()
        if self.name == "cuda":
            try:
                return bool(torch.cuda.is_bf16_supported())
            except Exception:
                return True
        if self.name == "rocm":
            return True
        if self.name == "mps":
            ver = _macos_version()
            return ver is None or ver >= (14,)
        try:  # cpu
            return bool(torch.backends.mkldnn.is_available() and torch.ops.mkldnn._is_mkldnn_bf16_supported())
        except Exception:
            return False

    def resolve_dtype(self, requested: str) -> tuple[Any, str | None]:
        """(dtype, warning). `auto` picks the best supported dtype; an unsupported explicit one falls back."""
        torch = _torch()
        requested = (requested or "auto").lower()
        if requested not in ("auto", "bfloat16", "float16", "float32"):
            raise BackendError(f"unknown dtype {requested!r} (use auto, bfloat16, float16 or float32)")
        bf16 = self._bf16_supported()
        if requested == "float32":
            return torch.float32, None
        if requested == "float16":
            if self.name == "cpu":
                return torch.float32, "float16 is slow and poorly supported on CPU; using float32"
            return torch.float16, None
        # auto or bfloat16
        if bf16:
            return torch.bfloat16, None
        if self.name == "cpu":
            return torch.float32, "this CPU has no bfloat16 support; using float32 (slower, 2x memory)"
        if self.name == "mps":
            why = "bfloat16 on MPS needs macOS 14+; using float16 (validate with bench/parity.py)"
        else:
            why = f"this {self.name} device has no bfloat16 support; using float16 (see bench/parity.py)"
        return torch.float16, why

    # ---- memory
    def memory(self) -> dict:
        out = {
            "kind": {"cuda": "vram", "rocm": "vram", "mps": "unified", "cpu": "system"}[self.name],
            "total_gb": None,
            "allocated_gb": None,
            "reserved_gb": None,
            "free_gb": None,
        }
        try:
            torch = _torch()
            if self.name in ("cuda", "rocm"):
                free, total = torch.cuda.mem_get_info(self.device)
                out.update(
                    total_gb=_gb(total),
                    free_gb=_gb(free),
                    allocated_gb=_gb(torch.cuda.memory_allocated(self.device)),
                    reserved_gb=_gb(torch.cuda.memory_reserved(self.device)),
                )
            elif self.name == "mps":
                alloc = torch.mps.current_allocated_memory()
                driver = torch.mps.driver_allocated_memory()
                total = torch.mps.recommended_max_memory()
                out.update(
                    total_gb=_gb(total),
                    allocated_gb=_gb(alloc),
                    reserved_gb=_gb(driver),
                    free_gb=_gb(max(total - driver, 0)),
                )
            else:
                rss, total, avail = _cpu_memory_bytes()
                out.update(
                    total_gb=_gb(total), allocated_gb=_gb(rss), reserved_gb=_gb(rss), free_gb=_gb(avail)
                )
        except Exception as exc:
            log.debug("memory() failed: %s", exc)
        return out

    # ---- telemetry
    def telemetry(self) -> dict:
        """GPU utilisation / temperature / power via pynvml, amdsmi or rocm-smi. Never raises."""
        result = _empty_telemetry()
        if not self.telemetry_enabled or self.name not in ("cuda", "rocm"):
            return result
        try:
            with self._lock:
                data = self._nvml() if self.name == "cuda" else self._amd()
            if data:
                result.update(data)
                result["available"] = True
        except Exception as exc:
            self._disable_telemetry(f"telemetry failed: {exc}")
        return result

    def _disable_telemetry(self, why: str, permanent: bool = False) -> None:
        """Missing library -> off for good; a runtime error -> retry after TELEMETRY_RETRY_S.

        Measured on the 7900 XTX under WSL: amdsmi calls fail transiently (e.g. while the model loads), and a
        permanent switch-off after the first failure left the console without telemetry for the whole run.
        """
        if not self._telemetry_off():
            log.debug("%s", why)
        until = float("inf") if permanent else time.monotonic() + TELEMETRY_RETRY_S
        self._cache["tel_off_until"] = until
        if not permanent:  # re-initialise the handle on the next attempt
            self._cache.pop("nvml_handle", None)
            self._cache.pop("amdsmi_handle", None)

    def _telemetry_off(self) -> bool:
        return time.monotonic() < self._cache.get("tel_off_until", 0.0)

    def _nvml(self) -> dict | None:
        if self._telemetry_off():
            return None
        handle = self._cache.get("nvml_handle")
        try:
            import pynvml
        except ImportError as exc:
            self._disable_telemetry(f"pynvml unavailable: {exc}", permanent=True)
            return None
        try:
            if handle is None:
                pynvml.nvmlInit()
                handle = pynvml.nvmlDeviceGetHandleByIndex(self.index)
                self._cache["nvml_handle"] = handle
            util = pynvml.nvmlDeviceGetUtilizationRates(handle)
            mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
            data: dict[str, Any] = {
                "source": "nvml",
                "util_pct": float(util.gpu),
                "mem_used_gb": _gb(mem.used),
                "mem_total_gb": _gb(mem.total),
            }
            for key, fn, scale in (
                ("temp_c", lambda: pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU), 1),
                ("power_w", lambda: pynvml.nvmlDeviceGetPowerUsage(handle), 1000),
            ):
                try:
                    data[key] = round(float(fn()) / scale, 1)
                except Exception:
                    data[key] = None
            return data
        except Exception as exc:
            self._disable_telemetry(f"pynvml failed: {exc}")
            return None

    def _amd(self) -> dict | None:
        if self.name != "rocm" or self._telemetry_off():
            return None
        data = self._amdsmi()
        if data is None and not _is_wsl():  # no SMI over /dev/dxg: do not even try
            data = self._rocm_smi()
        if data is None:
            permanent = self._cache.get("amdsmi_missing", False) and _is_wsl()
            self._disable_telemetry("no AMD telemetry source (amdsmi / rocm-smi)", permanent=permanent)
        return data

    def _amdsmi(self) -> dict | None:
        if self._cache.get("amdsmi_missing"):
            return None
        try:
            import amdsmi
        except ImportError:
            self._cache["amdsmi_missing"] = True
            return None
        try:
            handle = self._cache.get("amdsmi_handle")
            if handle is None:
                amdsmi.amdsmi_init()
                handles = amdsmi.amdsmi_get_processor_handles()
                handle = handles[self.index]
                self._cache["amdsmi_handle"] = handle
            act = amdsmi.amdsmi_get_gpu_activity(handle)
            data: dict[str, Any] = {"source": "amdsmi", "util_pct": _num(act.get("gfx_activity"))}
            try:
                temp = amdsmi.amdsmi_get_temp_metric(
                    handle,
                    amdsmi.AmdSmiTemperatureType.EDGE,
                    amdsmi.AmdSmiTemperatureMetric.CURRENT,
                )
                data["temp_c"] = _num(temp)
            except Exception:
                data["temp_c"] = None
            try:
                power = amdsmi.amdsmi_get_power_info(handle)
                data["power_w"] = _num(power.get("average_socket_power", power.get("current_socket_power")))
            except Exception:
                data["power_w"] = None
            try:
                vram = amdsmi.amdsmi_get_gpu_vram_usage(handle)
                used, total = float(vram["vram_used"]), float(vram["vram_total"])  # reported in MB
                data["mem_total_gb"] = round(total / 1024, 2)
                # Under WSL (/dev/dxg) amdsmi reports nonsense usage (e.g. 863707 MB used of 24560 MB)
                data["mem_used_gb"] = round(used / 1024, 2) if 0 <= used <= total else None
            except Exception:
                pass
            return data
        except Exception as exc:
            self._cache.pop("amdsmi_handle", None)
            log.debug("amdsmi failed: %s", exc)
            return None

    def _rocm_smi(self) -> dict | None:
        """rocm-smi subprocess; results cached for >= 2 s, failures for 5 min."""
        now = time.monotonic()
        cached = self._cache.get("rocm_smi")
        if cached and now < cached[0]:
            return cached[1]
        exe = shutil.which("rocm-smi")
        data = None
        if exe:
            raw = _run(
                [
                    exe,
                    "-d",
                    str(self.index),
                    "--showuse",
                    "--showtemp",
                    "--showpower",
                    "--showmeminfo",
                    "vram",
                    "--json",
                ],
                timeout=5.0,
            )
            data = _parse_rocm_smi(raw) if raw else None
        self._cache["rocm_smi"] = (now + (2.0 if data else 300.0), data)
        return data


def _num(value: Any) -> float | None:
    try:
        return round(float(str(value).split()[0]), 1)
    except Exception:
        return None


def _parse_rocm_smi(raw: str) -> dict | None:
    """Parse `rocm-smi --json` output (key names vary between ROCm releases, so match by substring)."""
    try:
        payload = json.loads(raw)
        card = next(v for k, v in payload.items() if k.lower().startswith("card") and isinstance(v, dict))
    except Exception:
        return None

    def find(*needles: str) -> Any:
        for key, val in card.items():
            low = key.lower()
            if all(n in low for n in needles):
                return val
        return None

    used, total = _num(find("vram", "used")), _num(find("vram", "total memory"))
    data = {
        "source": "rocm-smi",
        "util_pct": _num(find("gpu use")),
        "temp_c": _num(find("temperature", "edge")),
        "power_w": _num(find("power")),
        "mem_used_gb": None if used is None else round(used / GB, 2),
        "mem_total_gb": None if total is None else round(total / GB, 2),
    }
    return data if any(data[k] is not None for k in ("util_pct", "temp_c", "power_w")) else None


# ---------------------------------------------------------------------------------------------- detection
def _split_request(requested: str) -> tuple[str, int | None]:
    kind, _, idx = (requested or "auto").strip().lower().partition(":")
    kind = kind or "auto"
    if kind not in ("auto", *BACKENDS):
        raise BackendError(
            f"unknown device {requested!r} (use auto, cuda, rocm, mps or cpu; cuda:1 selects an index)"
        )
    if idx:
        if kind in ("auto", "mps", "cpu") or not idx.isdigit():
            raise BackendError(f"invalid device {requested!r}: an index is only valid as cuda:N or rocm:N")
        return kind, int(idx)
    return kind, None


def _unavailable_hint(torch: Any, kind: str) -> str:
    hip = getattr(torch.version, "hip", None)
    cuda_built = getattr(torch.version, "cuda", None)
    if kind == "cuda":
        if hip:
            return (
                "the installed torch is a ROCm build; use CLEF_DEVICE=rocm or install a CUDA build of torch"
            )
        if not cuda_built:
            return "the installed torch is a CPU-only build; install a CUDA build (see https://pytorch.org/get-started)"
        return "no NVIDIA GPU visible: check the driver with nvidia-smi and CUDA_VISIBLE_DEVICES"
    if kind == "rocm":
        if not hip:
            return "the installed torch is not a ROCm build; install the +rocm wheel for your GPU"
        hint = "no AMD GPU visible: check the ROCm install (rocminfo) and your user's render/video group"
        if _is_wsl():
            hint += "; on WSL set HSA_ENABLE_DXG_DETECTION=1 and update the Windows AMD driver"
        return hint
    if kind == "mps":
        built = getattr(torch.backends.mps, "is_built", lambda: False)()
        return (
            "MPS needs Apple Silicon and macOS 12.3+"
            if built
            else "this torch build has no MPS support (needs macOS arm64)"
        )
    return ""


def detect(requested: str = "auto") -> Backend:
    """auto: cuda > rocm > mps > cpu. An explicit request that is unavailable raises BackendError."""
    torch = _torch()
    kind, index = _split_request(requested)
    hip = getattr(torch.version, "hip", None)

    def cuda_ok() -> bool:
        return bool(torch.cuda.is_available())

    def check_index(k: str) -> int:
        idx = index or 0
        if idx:
            count = torch.cuda.device_count()
            if idx >= count:
                raise BackendError(f"CLEF_DEVICE={requested}: only {count} {k} device(s) visible")
        return idx

    if kind == "auto":
        if cuda_ok() and not hip:
            kind = "cuda"
        elif cuda_ok() and hip:
            kind = "rocm"
        elif torch.backends.mps.is_available():
            kind = "mps"
        else:
            kind = "cpu"
    elif kind == "cuda":
        if hip or not cuda_ok():
            raise BackendError(
                f"CLEF_DEVICE={requested} is not available: {_unavailable_hint(torch, 'cuda')}"
            )
    elif kind == "rocm":
        if not hip or not cuda_ok():
            raise BackendError(
                f"CLEF_DEVICE={requested} is not available: {_unavailable_hint(torch, 'rocm')}"
            )
    elif kind == "mps" and not torch.backends.mps.is_available():
        raise BackendError(f"CLEF_DEVICE=mps is not available: {_unavailable_hint(torch, 'mps')}")

    if kind in ("cuda", "rocm"):
        idx = check_index(kind)
        return Backend(kind, torch.device("cuda", idx), index=idx)
    return Backend(kind, torch.device(kind))


# ---------------------------------------------------------------------------------- preflight / quant
def _dtype_name(dtype: Any) -> str:
    return str(dtype).replace("torch.", "")


def required_gb(dtype: Any, quant: str, weights_gb: float) -> float:
    """Resident memory the weights need for this dtype / quantization (weights_gb is the bf16 size)."""
    if quant in _QUANT_FACTOR:
        return weights_gb * _QUANT_FACTOR[quant]
    return weights_gb * _SIZE_FACTOR.get(_dtype_name(dtype), 1.0)


def preflight(backend: Backend, dtype: Any, quant: str, weights_gb: float) -> list[str]:
    """Raise PreflightError when free memory is clearly below what the load needs; warn when close."""
    need = required_gb(dtype, quant, weights_gb)
    mem = backend.memory()
    free, total, kind = mem.get("free_gb"), mem.get("total_gb"), mem.get("kind")
    if free is None:
        return []
    label = {"vram": "GPU memory", "unified": "unified memory", "system": "system memory"}.get(kind, "memory")
    what = f"{quant}" if quant in _QUANT_FACTOR else _dtype_name(dtype)
    if free < need:
        tips = ["free memory (stop other GPU processes)" if kind != "system" else "free system memory"]
        if _dtype_name(dtype) == "float32":
            tips.append("use CLEF_DTYPE=bfloat16")
        if backend.name == "cuda" and quant == "none":
            tips.append("use CLEF_QUANT=nf4 or int8 (bitsandbytes)")
        if backend.name == "mps":
            tips.append("close other apps or use a Mac with more unified memory")
        raise PreflightError(
            f"not enough {label} to load the model ({what}): need ~{need:.1f} GB, {free:.1f} GB free"
            + (f" of {total:.1f} GB" if total else "")
            + f" on {backend.device_name()}. Try: {'; '.join(tips)}. Set CLEF_PREFLIGHT=0 to skip this check."
        )
    if free < need + _HEADROOM_GB:
        return [
            f"low {label}: {free:.1f} GB free, the model needs ~{need:.1f} GB plus activations; "
            "long inputs or large batches may run out of memory"
        ]
    return []


def quantization_config(backend: Backend, quant: str, dtype: Any) -> Any | None:
    """None for "none". int8 / nf4 -> transformers BitsAndBytesConfig, ONLY on cuda."""
    quant = (quant or "none").lower()
    if quant == "none":
        return None
    if quant not in _QUANT_FACTOR:
        raise BackendError(f"unknown CLEF_QUANT={quant!r} (use none, int8 or nf4)")
    if backend.name != "cuda":
        raise BackendError(
            f"CLEF_QUANT={quant} requires an NVIDIA GPU (bitsandbytes); backend is {backend.name}"
        )
    if not _has_module("bitsandbytes"):
        raise BackendError(f"CLEF_QUANT={quant} needs bitsandbytes: pip install bitsandbytes")
    from transformers import BitsAndBytesConfig

    if quant == "int8":
        return BitsAndBytesConfig(load_in_8bit=True)
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=dtype,
        bnb_4bit_use_double_quant=True,
    )
