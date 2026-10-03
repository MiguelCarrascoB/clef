#!/bin/bash
# User-space setup: download clef-flash weights + build the Python AI environment
set -euo pipefail

echo "== [1/3] Download model weights (git-lfs full checkout, ~19 GB) =="
mkdir -p "$HOME/models"
cd "$HOME/models"
rm -rf clef-flash
git clone https://huggingface.co/Cloudflare/clef-flash ~/models/clef-flash
ls -la ~/models/clef-flash

echo "== [2/3] Create Python virtual environment =="
python3 -m venv "$HOME/venvs/clef"
PIP="$HOME/venvs/clef/bin/pip"
"$PIP" install --upgrade pip -q

echo "== [3/3] Install PyTorch 2.11.0 (ROCm 7.2 build) + transformers 5.10.2 + deps =="
"$PIP" install --index-url https://download.pytorch.org/whl/rocm7.2 "torch==2.11.0+rocm7.2" -q
"$PIP" install "transformers==5.10.2" "accelerate" "pillow" "imageio" "imageio-ffmpeg" \
    "fastapi" "uvicorn[standard]" "huggingface-hub" -q

echo "DONE: user setup complete"
"$HOME/venvs/clef/bin/python" -c "import torch, transformers, safetensors, PIL; print('torch', torch.__version__, '| transformers', transformers.__version__)"
