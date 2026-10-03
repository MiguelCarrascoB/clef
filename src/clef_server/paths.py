"""Filesystem locations: state dir (logs, pidfile, saved classifiers) and model weights resolution.

Torch-free and cheap to import (the CLI uses it before any heavy import).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

MODEL_MARKER = "joint_schema_model.py"  # every clef release ships its model code next to the weights
WEIGHTS_GB = 19.0  # bf16 safetensors + joint head, measured on disk


class ModelNotFound(FileNotFoundError):
    """No usable model directory (-> engine status "error" with a `clef download` hint)."""


def state_dir(cfg: Any) -> Path:
    path = Path(cfg.state_dir)
    path.mkdir(parents=True, exist_ok=True)
    return path


def log_file(cfg: Any) -> Path:
    return state_dir(cfg) / "server.log"


def pid_file(cfg: Any) -> Path:
    return state_dir(cfg) / "server.pid"


def classifiers_dir(cfg: Any) -> Path:
    path = state_dir(cfg) / "classifiers"
    path.mkdir(parents=True, exist_ok=True)
    return path


def legacy_model_dir() -> Path:
    """The v1/v2 location (~/models/clef-flash); still honoured when it holds a release."""
    return Path.home() / "models" / "clef-flash"


def is_model_dir(path: Path) -> bool:
    return (path / MODEL_MARKER).is_file() and (path / "config.json").is_file()


def resolve_model_path(cfg: Any) -> Path:
    """CLEF_MODEL_PATH > ~/models/clef-flash > pinned snapshot in the HF cache. Never downloads."""
    if cfg.model_path:
        path = Path(cfg.model_path).expanduser()
        if not is_model_dir(path):
            raise ModelNotFound(
                f"CLEF_MODEL_PATH={path} is not a clef release ({MODEL_MARKER} or config.json missing)"
            )
        return path
    legacy = legacy_model_dir()
    if is_model_dir(legacy):
        return legacy
    try:
        from huggingface_hub import snapshot_download

        path = Path(snapshot_download(cfg.model_repo, revision=cfg.model_revision, local_files_only=True))
    except Exception as exc:  # LocalEntryNotFoundError, offline, missing package...
        raise ModelNotFound(
            f"model weights not found (looked in CLEF_MODEL_PATH, {legacy} and the Hugging Face cache for "
            f"{cfg.model_repo}@{cfg.model_revision[:12]}). Run `clef download` (~{WEIGHTS_GB:.0f} GB) or set "
            "CLEF_MODEL_PATH."
        ) from exc
    if not is_model_dir(path):
        raise ModelNotFound(f"incomplete snapshot at {path}; re-run `clef download`")
    return path
