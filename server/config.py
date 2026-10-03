"""Runtime configuration, read once from CLEF_* environment variables."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

VERSION = "2.0.0"


def _int(name: str, default: int) -> int:
    return int(os.environ.get(name, default))


def _float(name: str, default: float) -> float:
    return float(os.environ.get(name, default))


def _bool(name: str, default: bool) -> bool:
    return os.environ.get(name, "1" if default else "0").strip().lower() in ("1", "true", "yes", "on")


def _buckets(raw: str) -> tuple[int, ...]:
    return tuple(sorted({int(x) for x in raw.split(",") if x.strip()}))


@dataclass(frozen=True)
class Config:
    # model / device
    model_path: str = field(
        default_factory=lambda: os.environ.get("CLEF_MODEL_PATH", str(Path.home() / "models" / "clef-flash"))
    )
    device: str = field(default_factory=lambda: os.environ.get("CLEF_DEVICE", "cuda"))
    dtype: str = field(default_factory=lambda: os.environ.get("CLEF_DTYPE", "bfloat16"))
    max_tokens: int = field(default_factory=lambda: _int("CLEF_MAX_TOKENS", 16384))

    # network / auth
    host: str = field(default_factory=lambda: os.environ.get("CLEF_HOST", "127.0.0.1"))
    port: int = field(default_factory=lambda: _int("CLEF_PORT", 8910))
    api_key: str | None = field(default_factory=lambda: os.environ.get("CLEF_API_KEY") or None)

    # request limits
    max_body_mb: int = field(default_factory=lambda: _int("CLEF_MAX_BODY_MB", 64))
    max_images: int = field(default_factory=lambda: _int("CLEF_MAX_IMAGES", 8))
    max_videos: int = field(default_factory=lambda: _int("CLEF_MAX_VIDEOS", 2))
    max_pixels: int = field(default_factory=lambda: _int("CLEF_MAX_PIXELS", 1024 * 1024))
    max_frames: int = field(default_factory=lambda: _int("CLEF_MAX_FRAMES", 32))
    video_fps: float = field(default_factory=lambda: _float("CLEF_VIDEO_FPS", 2.0))
    max_batch: int = field(default_factory=lambda: _int("CLEF_MAX_BATCH", 64))
    max_questions: int = field(default_factory=lambda: _int("CLEF_MAX_QUESTIONS", 64))
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
    pad_multiple: int = field(default_factory=lambda: _int("CLEF_PAD_MULTIPLE", 64))
    pad_to_bucket: bool = field(default_factory=lambda: _bool("CLEF_PAD_TO_BUCKET", False))
    warmup: bool = field(default_factory=lambda: _bool("CLEF_WARMUP", True))

    # observability
    log_state: bool = field(default_factory=lambda: _bool("CLEF_LOG_STATE", False))
    log_buffer: int = field(default_factory=lambda: _int("CLEF_LOG_BUFFER", 1000))

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
            "max_tokens": self.max_tokens,
            "allow_url_fetch": self.allow_url_fetch,
            "auth_required": self.api_key is not None,
        }


def load() -> Config:
    return Config()
