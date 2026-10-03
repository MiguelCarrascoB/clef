#!/usr/bin/env bash
# Single source of truth for the clef environment. Source it; never execute it.
# Every value is overridable from the calling environment.

_clef_env_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export CLEF_HOME="${CLEF_HOME:-$(dirname "$_clef_env_dir")}"
export CLEF_VENV="${CLEF_VENV:-$HOME/venvs/clef}"
export CLEF_MODEL_PATH="${CLEF_MODEL_PATH:-$HOME/models/clef-flash}"
export CLEF_HOST="${CLEF_HOST:-127.0.0.1}"
export CLEF_PORT="${CLEF_PORT:-8910}"
export CLEF_LOG_DIR="${CLEF_LOG_DIR:-$HOME/.local/state/clef}"

# ROCm on WSL2 (ROCDXG)
export HSA_ENABLE_DXG_DETECTION="${HSA_ENABLE_DXG_DETECTION:-1}"
export HSA_OVERRIDE_GFX_VERSION="${HSA_OVERRIDE_GFX_VERSION:-11.0.0}"

# Weights are local; never touch the network at runtime
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"

# Performance defaults
# PYTORCH_HIP_ALLOC_CONF=expandable_segments:True is not supported on ROCm-on-WSL (warns, no effect)
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-$HOME/.triton/cache}"
export TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL="${TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL:-1}"
# TunableOp is off until measured (CLEF_TUNABLEOP=1 turns it on; results are cached across runs)
export PYTORCH_TUNABLEOP_ENABLED="${PYTORCH_TUNABLEOP_ENABLED:-${CLEF_TUNABLEOP:-0}}"
export PYTORCH_TUNABLEOP_FILENAME="${PYTORCH_TUNABLEOP_FILENAME:-$HOME/.cache/clef/tunableop_results%d.csv}"

mkdir -p "$CLEF_LOG_DIR" "$TRITON_CACHE_DIR" "$HOME/.cache/clef"

# Activate the venv when present (scripts that only need system tools still work without it)
if [ -f "$CLEF_VENV/bin/activate" ] && [ -z "${VIRTUAL_ENV:-}" ]; then
  # shellcheck disable=SC1091
  . "$CLEF_VENV/bin/activate"
fi
unset _clef_env_dir
