#!/usr/bin/env bash
# Environment for the Linux/WSL wrapper scripts. Source it; never execute it.
# It only locates the repo and activates the venv. Everything else (device, dtype, state dir, ROCm/WSL variables)
# is decided by the `clef` CLI and clef_server.backend.prepare_environment(); every CLEF_* value can still be
# overridden from the calling environment (see docs/ARCHITECTURE.md, "Configuration").

_clef_env_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export CLEF_HOME="${CLEF_HOME:-$(dirname "$_clef_env_dir")}"
export CLEF_VENV="${CLEF_VENV:-$HOME/venvs/clef}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-$HOME/.triton/cache}"
mkdir -p "$TRITON_CACHE_DIR"

# Activate the venv when present (scripts that only need system tools still work without it)
if [ -f "$CLEF_VENV/bin/activate" ] && [ -z "${VIRTUAL_ENV:-}" ]; then
  # shellcheck disable=SC1091
  . "$CLEF_VENV/bin/activate"
fi

# clef_run ARGS...: the installed `clef`, or the checkout's sources when the package is not installed.
clef_run() {
  if command -v clef >/dev/null 2>&1; then
    clef "$@"
  else
    PYTHONPATH="$CLEF_HOME/src${PYTHONPATH:+:$PYTHONPATH}" python -m clef_server.cli "$@"
  fi
}
unset _clef_env_dir
