"""Filesystem locations: state dir (logs, pidfile, database) and model weights resolution.

Torch-free and cheap to import (the CLI uses it before any heavy import).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

MODEL_MARKER = "joint_schema_model.py"  # every clef release ships its model code next to the weights
WEIGHTS_GB = 19.0  # clef-flash bf16 safetensors + joint head on disk (other models: ModelSpec.weights_gb)


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


def jobs_db(cfg: Any) -> Path:
    """The v3.1 SQLite file (jobs only); copied to clef.db once at startup (db.migrate_legacy_file)."""
    return state_dir(cfg) / "jobs.db"


def database_path(cfg: Any) -> Path:
    """Default database file (CLEF_DATABASE_URL unset): jobs, classifiers, keys, ... (see db.py)."""
    return state_dir(cfg) / "clef.db"


def legacy_model_dir() -> Path:
    """The v1/v2 location (~/models/clef-flash); still honoured when it holds a release."""
    return Path.home() / "models" / "clef-flash"


def is_model_dir(path: Path) -> bool:
    return (path / MODEL_MARKER).is_file() and (path / "config.json").is_file()


def resolve_model_path(cfg: Any, spec: Any = None) -> Path:
    """Local release dir of `spec` (default: the selected model, registry.resolve_spec). Never downloads.

    CLEF_MODEL_PATH > ~/models/clef-flash (clef-flash only) > the spec's pinned snapshot in the HF cache.
    Raises ModelNotFound with a `clef download` hint (models.UnknownModel for an unknown CLEF_MODEL).
    """
    if cfg.model_path:
        path = Path(cfg.model_path).expanduser()
        if not is_model_dir(path):
            raise ModelNotFound(
                f"CLEF_MODEL_PATH={path} is not a clef release ({MODEL_MARKER} or config.json missing)"
            )
        return path
    if spec is None:
        from .models.registry import resolve_spec

        spec = resolve_spec(cfg)
    flash = spec.id == "clef-flash"
    legacy = legacy_model_dir()
    if flash and is_model_dir(legacy):  # the v1/v2 dir only ever held clef-flash
        return legacy
    looked = f"CLEF_MODEL_PATH, {legacy} and" if flash else "CLEF_MODEL_PATH and"
    fetch = "clef download" if flash else f"clef download --model {spec.id}"
    try:
        from huggingface_hub import snapshot_download

        path = Path(snapshot_download(spec.repo, revision=spec.revision, local_files_only=True))
    except Exception as exc:  # LocalEntryNotFoundError, offline, missing package...
        raise ModelNotFound(
            f"model weights not found (looked in {looked} the Hugging Face cache for "
            f"{spec.repo}@{(spec.revision or 'main')[:12]}). Run `{fetch}` (~{spec.weights_gb:.0f} GB) or "
            "set CLEF_MODEL_PATH."
        ) from exc
    if not is_model_dir(path):
        raise ModelNotFound(f"incomplete snapshot at {path}; re-run `{fetch}`")
    return path


def config_dir() -> Path:
    """The user's clef config directory, ``~/.config/clef`` (``env``, ``keys``). Not created here.

    The same place ``deploy/clef.service`` reads as its EnvironmentFile (``%h/.config/clef/env``).
    """
    return Path.home() / ".config" / "clef"
