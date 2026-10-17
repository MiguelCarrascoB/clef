"""Model registry: adapters from the ``clef.models`` entry points, the model catalog, `CLEF_MODEL` resolution.

Torch-free at import. Built-in adapters register through the same entry-point group as third-party ones
(pyproject.toml ``[project.entry-points."clef.models"]``). A third-party package adds an adapter with::

    [project.entry-points."clef.models"]
    my-adapter = "my_pkg.adapter:MyAdapter"     # a DecisionModel / LocalTensorAdapter subclass

and its ``catalog()`` ids become selectable with ``CLEF_MODEL=<id>[@<rev>]`` (after a restart).
"""

from __future__ import annotations

import dataclasses
import functools
import importlib
import importlib.metadata
import logging
import os
import re
from collections.abc import Mapping
from typing import Any

from .base import DecisionModel, ModelSpec, UnknownModel

log = logging.getLogger("clef.models")

GROUP = "clef.models"
DEFAULT_MODEL_ID = "clef-flash"
# name -> "module:attr" of the adapters this distribution ships (same strings as in pyproject.toml)
BUILTIN: dict[str, str] = {"clef-release": "clef_server.models.clef_release:ClefReleaseAdapter"}


def _load_ref(ref: str) -> Any:
    module, _, attr = ref.partition(":")
    obj: Any = importlib.import_module(module)
    for part in attr.split("."):
        obj = getattr(obj, part)
    return obj


def _dist_label(ep: Any) -> str:
    dist = getattr(ep, "dist", None)
    if dist is None:
        return "unknown distribution"
    try:
        return f"{dist.metadata['Name']} {dist.version}"
    except Exception:
        return "unknown distribution"


@functools.lru_cache(maxsize=1)
def _discover() -> tuple[dict[str, type], dict[str, str]]:
    """(adapter name -> class, adapter name -> origin). Cached; `cache_clear()` re-reads the entry points."""
    found: dict[str, type] = {}
    origin: dict[str, str] = {}
    try:
        eps = list(importlib.metadata.entry_points(group=GROUP))
    except Exception as exc:  # broken metadata must not stop the server
        log.warning("could not read the %s entry points: %s", GROUP, exc)
        eps = []
    # built-ins first, so a third-party entry point reusing a built-in name never replaces it
    eps.sort(key=lambda ep: (BUILTIN.get(ep.name) != ep.value, ep.name))
    for ep in eps:
        if ep.name in found:
            log.warning(
                "model adapter %r from %s ignored: the name is already registered (%s)",
                ep.name,
                _dist_label(ep),
                origin[ep.name],
            )
            continue
        try:
            cls = ep.load()
        except Exception as exc:
            log.warning("model adapter %r from %s failed to import: %s", ep.name, _dist_label(ep), exc)
            continue
        found[ep.name] = cls
        origin[ep.name] = _dist_label(ep)
    for name, ref in BUILTIN.items():
        if name not in found:  # a source checkout / an install from before the entry point existed
            log.warning(
                "the %r entry point is not installed (running from a source checkout?); using the built-in "
                "adapter. Re-install clef-local to silence this.",
                name,
            )
            found[name] = _load_ref(ref)
            origin[name] = "built-in (no entry point)"
    return found, origin


def cache_clear() -> None:
    _discover.cache_clear()


def adapters() -> dict[str, type[DecisionModel]]:
    """Adapter name -> class, from importlib.metadata.entry_points(group="clef.models")."""
    return dict(_discover()[0])


def adapter_origin(name: str) -> str | None:
    """`<distribution> <version>` the adapter was loaded from (for /health and the startup log)."""
    return _discover()[1].get(name)


def _is_builtin(name: str) -> bool:
    return name in BUILTIN


def catalog() -> dict[str, ModelSpec]:
    """Union of the adapter catalogs. A duplicate id: the built-in wins (else the first adapter) + WARNING."""
    out: dict[str, ModelSpec] = {}
    names = sorted(adapters(), key=lambda n: (not _is_builtin(n), n))
    for name in names:
        cls = adapters()[name]
        try:
            specs = list(cls.catalog())
        except Exception as exc:
            log.warning("model adapter %r: catalog() failed: %s", name, exc)
            continue
        for spec in specs:
            if spec.adapter != name:
                spec = dataclasses.replace(spec, adapter=name)
            if spec.id in out:
                log.warning(
                    "model id %r from adapter %r ignored: already served by adapter %r",
                    spec.id,
                    name,
                    out[spec.id].adapter,
                )
                continue
            out[spec.id] = spec
    return out


def parse_model(value: str | None) -> tuple[str | None, str | None]:
    """`id[@rev]` -> (id, rev); blanks are None. Raises ValueError on `@rev` / `id@`."""
    if not value or not value.strip():
        return None, None
    model_id, sep, rev = value.strip().partition("@")
    if sep and (not model_id or not rev):
        raise ValueError(f"CLEF_MODEL={value!r} must be <id> or <id>@<revision>")
    return model_id, (rev or None)


def resolve_spec(cfg: Any) -> ModelSpec:
    """The ModelSpec this server serves: CLEF_MODEL plus the legacy aliases. Raises UnknownModel.

    1. CLEF_MODEL=<id>[@<rev>] picks the catalog entry; unset = clef-flash, unless CLEF_MODEL_REPO names a
       catalog entry's repo (then that entry).
    2. @<rev> (alias CLEF_MODEL_REVISION; Config refuses both set and different) overrides the pin.
    3. CLEF_MODEL_REPO overrides the entry's repo.
    4. CLEF_MODEL_PATH is the local release dir; the ref is <id>@<rev> when a revision was given, else
       <id>@local.
    """
    model_id, rev = parse_model(getattr(cfg, "model", None))
    repo = getattr(cfg, "model_repo", None) or None
    rev = rev or getattr(cfg, "model_revision", None) or None
    cat = catalog()
    if model_id is None:
        model_id = next((s.id for s in cat.values() if repo and s.repo == repo), DEFAULT_MODEL_ID)
    spec = cat.get(model_id)
    if spec is None:
        known = ", ".join(sorted(cat)) or "none"
        raise UnknownModel(f"CLEF_MODEL={model_id!r} is not a known model (known: {known}; `clef models`)")
    changes: dict[str, Any] = {}
    if repo:
        changes["repo"] = repo
    if getattr(cfg, "model_path", None):
        changes["revision"] = rev  # a local dir: only a revision the user gave
    elif rev:
        changes["revision"] = rev
    return dataclasses.replace(spec, **changes) if changes else spec


def create(cfg: Any, spec: ModelSpec | None = None) -> DecisionModel:
    """Instantiate the adapter for `spec` (default: resolve_spec(cfg)). Cheap: nothing is loaded yet."""
    spec = spec or resolve_spec(cfg)
    cls = adapters().get(spec.adapter)
    if cls is None:
        raise UnknownModel(f"model {spec.id!r} needs adapter {spec.adapter!r}, which is not installed")
    return cls(spec, cfg)


def served_model(engine: Any, requested: str | None) -> str:
    """`engine.check_model(requested)`; engines without it (test fakes) accept anything."""
    fn = getattr(engine, "check_model", None)
    if callable(fn):
        return fn(requested)
    return requested or DEFAULT_MODEL_ID


def model_slug(environ: Mapping[str, str] = os.environ) -> str:
    """`<id>@<rev8>` for per-model state dirs (TunableOp), from the environment alone. Never raises.

    Same rules as `resolve_spec`, over the built-in catalog only (no entry-point scan): CLEF_MODEL unset and
    CLEF_MODEL_REPO naming a built-in entry's repo picks that entry."""
    try:
        from .clef_release import PINS, REPOS
    except Exception:
        PINS, REPOS = {}, {}
    try:
        model_id, rev = parse_model(environ.get("CLEF_MODEL"))
    except ValueError:
        model_id, rev = None, None
    repo = (environ.get("CLEF_MODEL_REPO") or "").strip()
    if model_id is None:
        model_id = next((mid for mid, r in REPOS.items() if repo and r == repo), DEFAULT_MODEL_ID)
    rev = rev or (environ.get("CLEF_MODEL_REVISION") or "").strip() or None
    if rev is None and not (environ.get("CLEF_MODEL_PATH") or "").strip():
        rev = PINS.get(model_id)  # the built-in pins
    slug = f"{model_id}@{(rev or 'local')[:8]}"
    return re.sub(r"[^A-Za-z0-9@_.-]", "_", slug)


__all__ = [
    "DEFAULT_MODEL_ID",
    "GROUP",
    "adapter_origin",
    "adapters",
    "cache_clear",
    "catalog",
    "create",
    "model_slug",
    "parse_model",
    "resolve_spec",
    "served_model",
]
