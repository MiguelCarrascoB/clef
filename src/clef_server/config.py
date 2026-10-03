"""Runtime configuration, read once from CLEF_* environment variables (see docs/ARCHITECTURE.md)."""

from __future__ import annotations

import os
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

VERSION = "3.0.0"

# Hugging Face release the server is pinned to (`clef download` fetches exactly this revision).
MODEL_REPO = "Cloudflare/clef-flash"
MODEL_REVISION = "17f0b0ad64efb65d273590632833508766b2aae6"

DEVICES = ("auto", "cuda", "rocm", "mps", "cpu")
DTYPES = ("auto", "bfloat16", "float16", "float32")
QUANTS = ("none", "int8", "nf4")
KEY_NAME_RE = re.compile(r"^[A-Za-z0-9_.-]{1,64}$")


def _int(name: str, default: int) -> int:
    return int(os.environ.get(name, default))


def _float(name: str, default: float) -> float:
    return float(os.environ.get(name, default))


def _bool(name: str, default: bool) -> bool:
    return os.environ.get(name, "1" if default else "0").strip().lower() in ("1", "true", "yes", "on")


def _str(name: str, default: str) -> str:
    return os.environ.get(name, default).strip()


def _buckets(raw: str) -> tuple[int, ...]:
    return tuple(sorted({int(x) for x in raw.split(",") if x.strip()}))


def _csv(raw: str) -> tuple[str, ...]:
    return tuple(x.strip() for x in raw.split(",") if x.strip())


def default_state_dir() -> Path:
    """Per-OS state dir (logs, pidfile, saved classifiers). CLEF_STATE_DIR (or legacy CLEF_LOG_DIR) wins."""
    override = os.environ.get("CLEF_STATE_DIR") or os.environ.get("CLEF_LOG_DIR")
    if override:
        return Path(override).expanduser()
    if os.name == "nt":
        base = os.environ.get("LOCALAPPDATA") or str(Path.home() / "AppData" / "Local")
        return Path(base) / "clef"
    if sys.platform == "darwin":
        return Path.home() / "Library" / "Application Support" / "clef"
    base = os.environ.get("XDG_STATE_HOME") or str(Path.home() / ".local" / "state")
    return Path(base) / "clef"


def parse_api_keys(single: str | None, multi: str | None) -> dict[str, str]:
    """name -> key from CLEF_API_KEY (named "default") and CLEF_API_KEYS.

    CLEF_API_KEYS is either a comma-separated list of `name:key` entries, or the path of a file with one
    `name:key` per line (blank lines and `#` comments ignored). A bare `key` is named key1, key2, ...
    Raises ValueError on a malformed name, an empty key or a duplicate name.
    """
    keys: dict[str, str] = {}
    if single:
        keys["default"] = single
    if not multi or not multi.strip():
        return keys
    raw = multi.strip()
    path = Path(raw).expanduser()
    if path.is_file():
        lines = (ln.strip() for ln in path.read_text(encoding="utf-8").splitlines())
        entries = [ln for ln in lines if ln and not ln.startswith("#")]
    else:
        entries = list(_csv(raw))
    for i, entry in enumerate(entries, 1):
        name, sep, key = entry.partition(":")
        if not sep:
            name, key = f"key{i}", entry
        name, key = name.strip(), key.strip()
        if not KEY_NAME_RE.match(name):
            raise ValueError(f"CLEF_API_KEYS: invalid key name {name!r} (letters, digits, _ . -, max 64)")
        if not key:
            raise ValueError(f"CLEF_API_KEYS: empty key for {name!r}")
        if name in keys:
            raise ValueError(f"CLEF_API_KEYS: duplicate key name {name!r}")
        keys[name] = key
    return keys


@dataclass(frozen=True)
class Config:
    # model / device
    # None = resolve: ~/models/clef-flash if it holds a release, else the pinned HF cache snapshot (paths.py)
    model_path: str | None = field(default_factory=lambda: os.environ.get("CLEF_MODEL_PATH") or None)
    model_repo: str = field(default_factory=lambda: _str("CLEF_MODEL_REPO", MODEL_REPO))
    model_revision: str = field(default_factory=lambda: _str("CLEF_MODEL_REVISION", MODEL_REVISION))
    device: str = field(default_factory=lambda: _str("CLEF_DEVICE", "auto").lower())  # see DEVICES
    dtype: str = field(default_factory=lambda: _str("CLEF_DTYPE", "auto").lower())  # see DTYPES
    quant: str = field(default_factory=lambda: _str("CLEF_QUANT", "none").lower())  # see QUANTS (CUDA only)
    preflight: bool = field(default_factory=lambda: _bool("CLEF_PREFLIGHT", True))  # memory check before load
    telemetry: bool = field(default_factory=lambda: _bool("CLEF_TELEMETRY", True))  # pynvml / amdsmi
    max_tokens: int = field(default_factory=lambda: _int("CLEF_MAX_TOKENS", 16384))

    # network / auth
    host: str = field(default_factory=lambda: _str("CLEF_HOST", "127.0.0.1"))
    port: int = field(default_factory=lambda: _int("CLEF_PORT", 8910))
    api_key: str | None = field(default_factory=lambda: os.environ.get("CLEF_API_KEY") or None)
    api_keys_raw: str | None = field(default_factory=lambda: os.environ.get("CLEF_API_KEYS") or None)
    rate_limit: int = field(default_factory=lambda: _int("CLEF_RATE_LIMIT", 0))  # requests/min per key, 0=off
    cors_origins: tuple[str, ...] = field(
        default_factory=lambda: _csv(os.environ.get("CLEF_CORS_ORIGINS", ""))
    )

    # state (logs, pidfile, saved classifiers)
    state_dir: str = field(default_factory=lambda: str(default_state_dir()))

    # request limits
    max_body_mb: int = field(default_factory=lambda: _int("CLEF_MAX_BODY_MB", 64))
    max_images: int = field(default_factory=lambda: _int("CLEF_MAX_IMAGES", 8))
    max_videos: int = field(default_factory=lambda: _int("CLEF_MAX_VIDEOS", 2))
    max_pixels: int = field(default_factory=lambda: _int("CLEF_MAX_PIXELS", 1024 * 1024))
    max_frames: int = field(default_factory=lambda: _int("CLEF_MAX_FRAMES", 32))
    video_fps: float = field(default_factory=lambda: _float("CLEF_VIDEO_FPS", 2.0))
    max_batch: int = field(default_factory=lambda: _int("CLEF_MAX_BATCH", 64))
    max_questions: int = field(default_factory=lambda: _int("CLEF_MAX_QUESTIONS", 64))
    max_labels: int = field(default_factory=lambda: _int("CLEF_MAX_LABELS", 64))
    max_classifiers: int = field(default_factory=lambda: _int("CLEF_MAX_CLASSIFIERS", 1000))
    classify_threshold: float = field(default_factory=lambda: _float("CLEF_CLASSIFY_THRESHOLD", 0.5))
    max_eval_rows: int = field(default_factory=lambda: _int("CLEF_MAX_EVAL_ROWS", 500))  # POST /v1/evaluate
    max_job_eval_rows: int = field(default_factory=lambda: _int("CLEF_MAX_JOB_EVAL_ROWS", 100000))  # job
    allow_url_fetch: bool = field(default_factory=lambda: _bool("CLEF_ALLOW_URL_FETCH", False))
    url_fetch_max_mb: int = field(default_factory=lambda: _int("CLEF_URL_FETCH_MAX_MB", 32))

    # engine / performance
    max_microbatch: int = field(default_factory=lambda: _int("CLEF_MAX_MICROBATCH", 8))
    batch_window_ms: float = field(default_factory=lambda: _float("CLEF_BATCH_WINDOW_MS", 4.0))
    buckets: tuple[int, ...] = field(
        default_factory=lambda: _buckets(
            os.environ.get("CLEF_BUCKETS", "128,192,256,384,512,768,1024,1536,2048,3072,4096")
        )
    )
    # Buckets group similar lengths into one forward. Text batches are padded to the next pad_multiple: on the
    # 7900 XTX hipBLASLt picks faster GEMM tiles for aligned lengths (e.g. 330 tokens: 155 ms raw vs ~148 ms
    # padded to 384), while full bucket padding (opt-in) wastes up to ~50% compute on every forward.
    # 64 is measured on ROCm only; re-measure on CUDA / MPS (bench/http_bench.py) before changing the default.
    pad_multiple: int = field(default_factory=lambda: _int("CLEF_PAD_MULTIPLE", 64))
    pad_to_bucket: bool = field(default_factory=lambda: _bool("CLEF_PAD_TO_BUCKET", False))
    warmup: bool = field(default_factory=lambda: _bool("CLEF_WARMUP", True))

    # observability
    log_state: bool = field(default_factory=lambda: _bool("CLEF_LOG_STATE", False))
    log_buffer: int = field(default_factory=lambda: _int("CLEF_LOG_BUFFER", 1000))
    sample_interval_s: float = field(default_factory=lambda: _float("CLEF_SAMPLE_INTERVAL_S", 2.0))

    def __post_init__(self) -> None:
        for name, value, allowed in (
            ("CLEF_DEVICE", self.device.split(":")[0], DEVICES),
            ("CLEF_DTYPE", self.dtype, DTYPES),
            ("CLEF_QUANT", self.quant, QUANTS),
        ):
            if value not in allowed:
                raise ValueError(f"{name}={value!r} is not one of {', '.join(allowed)}")
        if not 0.0 <= self.classify_threshold <= 1.0:
            raise ValueError("CLEF_CLASSIFY_THRESHOLD must be within [0, 1]")
        if self.rate_limit < 0:
            raise ValueError("CLEF_RATE_LIMIT must be >= 0 (0 disables it)")
        self.api_keys()  # fail fast on a malformed CLEF_API_KEYS

    def api_keys(self) -> dict[str, str]:
        """name -> key (empty dict = auth disabled)."""
        return parse_api_keys(self.api_key, self.api_keys_raw)

    @property
    def auth_required(self) -> bool:
        return bool(self.api_key or self.api_keys_raw)

    @property
    def is_loopback(self) -> bool:
        return self.host in ("127.0.0.1", "localhost", "::1") or self.host.startswith("127.")

    def public_limits(self) -> dict[str, object]:
        return {
            "max_body_mb": self.max_body_mb,
            "max_images": self.max_images,
            "max_videos": self.max_videos,
            "max_pixels": self.max_pixels,
            "max_frames": self.max_frames,
            "video_fps": self.video_fps,
            "max_batch": self.max_batch,
            "max_questions": self.max_questions,
            "max_labels": self.max_labels,
            "max_tokens": self.max_tokens,
            "classify_threshold": self.classify_threshold,
            "allow_url_fetch": self.allow_url_fetch,
            "auth_required": self.auth_required,
            "rate_limit_per_min": self.rate_limit,
        }


def load() -> Config:
    return Config()
