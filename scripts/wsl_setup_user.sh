#!/usr/bin/env bash
# User-space setup: Python venv + clef (requirements/rocm.txt, pip install -e) + model weights (clef download).
# Idempotent and non-destructive: never deletes existing weights or venv.
set -euo pipefail
# shellcheck source=env.sh
. "$(dirname "${BASH_SOURCE[0]}")/env.sh"

echo "== [1/3] Python virtual environment -> $CLEF_VENV =="
if [ -x "$CLEF_VENV/bin/python" ]; then
  echo "venv exists"
else
  mkdir -p "$(dirname "$CLEF_VENV")"
  python3 -m venv "$CLEF_VENV"
fi

echo "== [2/3] Install clef (ROCm 7.2 lock file, then the package in editable mode) =="
"$CLEF_VENV/bin/python" -m pip install --upgrade pip -q
"$CLEF_VENV/bin/python" -m pip install -r "$CLEF_HOME/requirements/rocm.txt"
"$CLEF_VENV/bin/python" -m pip install -e "$CLEF_HOME[server,rocm]"     --extra-index-url https://download.pytorch.org/whl/rocm7.2
if [ "${CLEF_INSTALL_DEV:-0}" = 1 ]; then
  "$CLEF_VENV/bin/python" -m pip install -e "$CLEF_HOME[dev]"
fi

echo "== [3/3] Model weights (pinned revision, ~19 GB, resumable) =="
if [ -f "$HOME/models/clef-flash/joint_schema_model.py" ] || [ -n "${CLEF_MODEL_PATH:-}" ]; then
  echo "weights already configured (${CLEF_MODEL_PATH:-$HOME/models/clef-flash}), leaving them untouched"
else
  "$CLEF_VENV/bin/clef" download --yes
fi

echo "DONE: user setup complete. Verify with: $CLEF_VENV/bin/clef doctor"
