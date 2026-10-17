"""Runtime configuration, read once from CLEF_* environment variables (see docs/ARCHITECTURE.md)."""

from __future__ import annotations

import os
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

VERSION = "3.1.0"

# Hugging Face release the server is pinned to (`clef download` fetches exactly this revision).
MODEL_REPO = "Cloudflare/clef-flash"
MODEL_REVISION = "17f0b0ad64efb65d273590632833508766b2aae6"

# E2: the router's LLM defaults to Gemini's OpenAI-compatible endpoint (ollama, LM Studio, OpenAI: set URL).
DEFAULT_LLM_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"
DEFAULT_LLM_MODEL = "gemini-3.8-flash"

DEVICES = ("auto", "cuda", "rocm", "mps", "cpu")
DTYPES = ("auto", "bfloat16", "float16", "float32")
QUANTS = ("none", "int8", "nf4")
OFFLOADS = ("none", "cpu")
LOG_FORMATS = ("text", "json")  # CLEF_LOG_FORMAT, see logs.py
KEY_NAME_RE = re.compile(r"^[A-Za-z0-9_.-]{1,64}$")


def _env(name: str) -> str | None:
    """The variable's value, or None when unset or blank (WSLENV passes unset Windows variables as "")."""
    value = os.environ.get(name, "").strip()
    return value or None


def _int(name: str, default: int) -> int:
    value = _env(name)
    return default if value is None else int(value)


def _float(name: str, default: float) -> float:
    value = _env(name)
    return default if value is None else float(value)


def _bool(name: str, default: bool) -> bool:
    value = _env(name)
    return default if value is None else value.lower() in ("1", "true", "yes", "on")


def _str(name: str, default: str) -> str:
    return _env(name) or default


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
    # Aliases of CLEF_MODEL (models/registry.py resolve_spec): None = the selected catalog entry's repo / pin.
    model_repo: str | None = field(default_factory=lambda: _env("CLEF_MODEL_REPO"))
    model_revision: str | None = field(default_factory=lambda: _env("CLEF_MODEL_REVISION"))
    device: str = field(default_factory=lambda: _str("CLEF_DEVICE", "auto").lower())  # see DEVICES
    dtype: str = field(default_factory=lambda: _str("CLEF_DTYPE", "auto").lower())  # see DTYPES
    quant: str = field(
        default_factory=lambda: _str("CLEF_QUANT", "none").lower()
    )  # see QUANTS (backend rules)
    quant_backend: str = field(
        default_factory=lambda: _str("CLEF_QUANT_BACKEND", "auto").lower()
    )  # auto|bnb|torchao
    # Smaller-memory options (docs/memory.md): cap the device memory the process may use and, with
    # CLEF_OFFLOAD=cpu, keep whatever does not fit under the cap in host RAM and stream it in per forward.
    offload: str = field(default_factory=lambda: _str("CLEF_OFFLOAD", "none").lower())  # see OFFLOADS
    max_device_memory_gb: float = field(
        default_factory=lambda: _float("CLEF_MAX_DEVICE_MEMORY_GB", 0.0)
    )  # 0 = auto
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
    # Seconds to keep serving after the first SIGTERM / Ctrl-C with /readyz 503 "draining", so a load balancer
    # stops routing first (main.run only; 0 = stop at once). A second signal skips the rest.
    drain_s: float = field(default_factory=lambda: _float("CLEF_DRAIN_S", 0.0))

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

    # async jobs + webhooks (see docs/jobs.md)
    max_job_items: int = field(default_factory=lambda: _int("CLEF_MAX_JOB_ITEMS", 100_000))
    job_ttl_hours: float = field(
        default_factory=lambda: _float("CLEF_JOB_TTL_HOURS", 168.0)
    )  # 0 = keep forever
    max_jobs: int = field(default_factory=lambda: _int("CLEF_MAX_JOBS", 100))  # queued + running jobs
    job_engine_wait_s: float = field(default_factory=lambda: _float("CLEF_JOB_ENGINE_WAIT_S", 600.0))
    # A job found still running at startup (a crash, not a clean stop) more often than this is failed
    # instead of resumed (0 = no limit).
    job_max_resumes: int = field(default_factory=lambda: _int("CLEF_JOB_MAX_RESUMES", 3))
    # Hosts / IPs / CIDRs webhooks may be delivered to. Empty (default) = webhooks are OFF.
    webhook_allow: tuple[str, ...] = field(
        default_factory=lambda: _csv(os.environ.get("CLEF_WEBHOOK_ALLOW", ""))
    )
    webhook_timeout_s: float = field(default_factory=lambda: _float("CLEF_WEBHOOK_TIMEOUT_S", 10.0))
    webhook_attempts: int = field(default_factory=lambda: _int("CLEF_WEBHOOK_ATTEMPTS", 10))
    # Retry delays: min(cap, base * 2**(n-1)) seconds (+-25 %): 2, 4, 8, ... 120 s, about 8 minutes in all.
    webhook_backoff_s: float = field(default_factory=lambda: _float("CLEF_WEBHOOK_BACKOFF_S", 2.0))
    webhook_backoff_cap_s: float = field(default_factory=lambda: _float("CLEF_WEBHOOK_BACKOFF_CAP_S", 120.0))

    # engine / performance
    max_microbatch: int = field(default_factory=lambda: _int("CLEF_MAX_MICROBATCH", 8))
    batch_window_ms: float = field(default_factory=lambda: _float("CLEF_BATCH_WINDOW_MS", 4.0))
    # Interactive micro-batches in a row before waiting background (job) work gets one (starvation bound).
    interactive_burst: int = field(default_factory=lambda: _int("CLEF_INTERACTIVE_BURST", 8))
    buckets: tuple[int, ...] = field(
        default_factory=lambda: _buckets(
            _env("CLEF_BUCKETS") or "128,192,256,384,512,768,1024,1536,2048,3072,4096"
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
    log_format: str = field(
        default_factory=lambda: _str("CLEF_LOG_FORMAT", "text").lower()
    )  # see LOG_FORMATS

    # C1: database (docs/jobs.md, ARCHITECTURE Storage). Unset = sqlite:///<state_dir>/clef.db.
    database_url: str | None = field(default_factory=lambda: _env("CLEF_DATABASE_URL"))
    database_pool_size: int = field(default_factory=lambda: _int("CLEF_DATABASE_POOL_SIZE", 5))
    # C2: the decision model. CLEF_MODEL=<id>[@<revision>] from the catalog (`clef models`); None = clef-flash
    # (or the catalog entry whose repo is CLEF_MODEL_REPO). Resolved by models.registry.resolve_spec.
    model: str | None = field(default_factory=lambda: _env("CLEF_MODEL"))

    # D3: in-memory result cache entries (docs/ARCHITECTURE.md, Result cache); 0 = off.
    cache_size: int = field(default_factory=lambda: _int("CLEF_CACHE_SIZE", 10000))
    # Byte budget of the same cache (estimated from the stored JSON); LRU entries go until it fits.
    cache_mb: int = field(default_factory=lambda: _int("CLEF_CACHE_MB", 256))
    # D2: opt-in speedups, default OFF until bench/perf_check.sh has measured them (docs/performance.md).
    # mask_free: text batches call the backbone with attention_mask=None (right-padded + causal = exact);
    # fast_head: sync-free, batch-vectorised joint head. Both only apply to the clef-release adapter.
    mask_free: bool = field(default_factory=lambda: _bool("CLEF_MASK_FREE", False))
    fast_head: bool = field(default_factory=lambda: _bool("CLEF_FAST_HEAD", False))

    # E2: the router's LLM (llm.py) and the review queue (router.py). The router is OFF unless a key is set in
    # the process environment (GEMINI_API_KEY or CLEF_LLM_API_KEY): the key is the consent to send buffered
    # inputs. clef reads no .env file itself; docker compose / the systemd EnvironmentFile set the variable.
    llm_api_key: str | None = field(
        default_factory=lambda: _env("CLEF_LLM_API_KEY") or _env("GEMINI_API_KEY"), repr=False
    )
    llm_base_url: str = field(default_factory=lambda: _str("CLEF_LLM_BASE_URL", DEFAULT_LLM_BASE_URL))
    llm_model: str = field(default_factory=lambda: _str("CLEF_LLM_MODEL", DEFAULT_LLM_MODEL))
    llm_timeout_s: float = field(default_factory=lambda: _float("CLEF_LLM_TIMEOUT_S", 60.0))
    router_batch: int = field(default_factory=lambda: _int("CLEF_ROUTER_BATCH", 20))  # items per LLM call
    router_cold_batch: int = field(default_factory=lambda: _int("CLEF_ROUTER_COLD_BATCH", 20))
    router_mode: str = field(
        default_factory=lambda: _str("CLEF_ROUTER_MODE", "review").lower()
    )  # review|auto
    router_flush_s: float = field(default_factory=lambda: _float("CLEF_ROUTER_FLUSH_S", 600.0))
    router_max_input_chars: int = field(default_factory=lambda: _int("CLEF_ROUTER_MAX_INPUT_CHARS", 500))
    router_max_new_per_day: int = field(default_factory=lambda: _int("CLEF_ROUTER_MAX_NEW_PER_DAY", 10))
    review_ttl_days: float = field(default_factory=lambda: _float("CLEF_REVIEW_TTL_DAYS", 30.0))  # 0 = keep

    def __post_init__(self) -> None:
        for name, value, allowed in (
            ("CLEF_DEVICE", self.device.split(":")[0], DEVICES),
            ("CLEF_DTYPE", self.dtype, DTYPES),
            ("CLEF_QUANT", self.quant, QUANTS),
            ("CLEF_QUANT_BACKEND", self.quant_backend, ("auto", "bnb", "torchao")),
            ("CLEF_LOG_FORMAT", self.log_format, LOG_FORMATS),
        ):
            if value not in allowed:
                raise ValueError(f"{name}={value!r} is not one of {', '.join(allowed)}")
        if self.offload not in OFFLOADS:
            raise ValueError(f"CLEF_OFFLOAD={self.offload!r} is not one of {', '.join(OFFLOADS)}")
        if self.max_device_memory_gb < 0:
            raise ValueError("CLEF_MAX_DEVICE_MEMORY_GB must be >= 0 (0 = no cap)")
        if not 0.0 <= self.classify_threshold <= 1.0:
            raise ValueError("CLEF_CLASSIFY_THRESHOLD must be within [0, 1]")
        if self.rate_limit < 0:
            raise ValueError("CLEF_RATE_LIMIT must be >= 0 (0 disables it)")
        if self.interactive_burst < 1:
            raise ValueError("CLEF_INTERACTIVE_BURST must be >= 1")
        if self.drain_s < 0:
            raise ValueError("CLEF_DRAIN_S must be >= 0 (0 = no drain)")
        if self.cache_size < 0:
            raise ValueError("CLEF_CACHE_SIZE must be >= 0 (0 = off)")
        if self.cache_mb < 0:
            raise ValueError("CLEF_CACHE_MB must be >= 0 (0 = off)")
        if self.router_mode not in ("review", "auto"):
            raise ValueError("CLEF_ROUTER_MODE must be review or auto")
        if self.llm_timeout_s <= 0:
            raise ValueError("CLEF_LLM_TIMEOUT_S must be > 0")
        if not 2 <= self.router_batch <= 100 or not 2 <= self.router_cold_batch <= 100:
            raise ValueError("CLEF_ROUTER_BATCH and CLEF_ROUTER_COLD_BATCH must be within [2, 100]")
        if self.router_flush_s < 0 or self.review_ttl_days < 0 or self.router_max_new_per_day < 0:
            raise ValueError(
                "CLEF_ROUTER_FLUSH_S, CLEF_REVIEW_TTL_DAYS, CLEF_ROUTER_MAX_NEW_PER_DAY must be >= 0"
            )
        if self.router_max_input_chars < 20:
            raise ValueError("CLEF_ROUTER_MAX_INPUT_CHARS must be >= 20")
        self.api_keys()  # fail fast on a malformed CLEF_API_KEYS
        # C2: CLEF_MODEL syntax and the @rev / CLEF_MODEL_REVISION alias (an unknown id is a load error)
        if self.model:
            model_id, sep, rev = self.model.strip().partition("@")
            if sep and (not model_id or not rev):
                raise ValueError(f"CLEF_MODEL={self.model!r} must be <id> or <id>@<revision>")
            if rev and self.model_revision and rev != self.model_revision:
                raise ValueError(
                    f"CLEF_MODEL={self.model!r} and CLEF_MODEL_REVISION={self.model_revision!r} disagree; "
                    "set only one"
                )

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
            "max_job_items": self.max_job_items,
            "webhooks_enabled": bool(self.webhook_allow),
            "auth_required": self.auth_required,
            "rate_limit_per_min": self.rate_limit,
        }


def load() -> Config:
    return Config()
