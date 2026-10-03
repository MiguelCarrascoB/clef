#!/usr/bin/env bash
# User-space setup: model weights (pinned revision) + Python venv from requirements/server.txt.
# Idempotent and non-destructive: never deletes existing weights or venv.
set -euo pipefail
# shellcheck source=env.sh
. "$(dirname "${BASH_SOURCE[0]}")/env.sh"

CLEF_MODEL_REPO="${CLEF_MODEL_REPO:-https://huggingface.co/Cloudflare/clef-flash}"
CLEF_MODEL_REV="${CLEF_MODEL_REV:-17f0b0ad64efb65d273590632833508766b2aae6}"

echo "== [1/3] Model weights -> $CLEF_MODEL_PATH (rev ${CLEF_MODEL_REV:0:12}) =="
if [ -d "$CLEF_MODEL_PATH/.git" ] || [ -f "$CLEF_MODEL_PATH/joint_head.safetensors" ]; then
  echo "weights already present, leaving them untouched"
else
  mkdir -p "$(dirname "$CLEF_MODEL_PATH")"
  git lfs install --skip-repo >/dev/null 2>&1 || true
  git clone "$CLEF_MODEL_REPO" "$CLEF_MODEL_PATH"
  git -C "$CLEF_MODEL_PATH" checkout "$CLEF_MODEL_REV"
fi
echo "checked out: $(git -C "$CLEF_MODEL_PATH" rev-parse HEAD 2>/dev/null || echo 'n/a (not a git checkout)')"

echo "== [2/3] Python virtual environment -> $CLEF_VENV =="
if [ -x "$CLEF_VENV/bin/python" ]; then
  echo "venv exists"
else
  mkdir -p "$(dirname "$CLEF_VENV")"
  python3 -m venv "$CLEF_VENV"
fi

echo "== [3/3] Install pinned dependencies (torch 2.11.0+rocm7.2, transformers 5.10.2, ...) =="
"$CLEF_VENV/bin/python" -m pip install --upgrade pip -q
"$CLEF_VENV/bin/python" -m pip install -r "$CLEF_HOME/requirements/server.txt"
if [ "${CLEF_INSTALL_DEV:-0}" = 1 ]; then
  "$CLEF_VENV/bin/python" -m pip install -r "$CLEF_HOME/requirements/dev.txt"
fi

echo "DONE: user setup complete. Verify with: $CLEF_VENV/bin/python $CLEF_HOME/scripts/doctor.py"
