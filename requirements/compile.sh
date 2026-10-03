#!/usr/bin/env bash
# Regenerate the per-backend lock files. Needs uv >= 0.8 (--torch-backend); run from anywhere.
#   bash requirements/compile.sh [rocm|cuda|cpu|macos|dev ...]     (default: all)
# Order matters: dev.txt is resolved against cpu.txt.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
UV="${UV:-uv}"
PY=3.10
# uv emits only the default index; the torch wheels live on download.pytorch.org, so add it to the lock file.
add_torch_index() {  # file, backend (rocm7.2 | cu128 | cpu)
  awk -v u="https://download.pytorch.org/whl/$2" '{print} /^--index-url/{print "--extra-index-url " u}' "$1" > "$1.tmp"
  mv "$1.tmp" "$1"
}

targets=("$@"); [ ${#targets[@]} -gt 0 ] || targets=(rocm cuda cpu macos dev)

for t in "${targets[@]}"; do
  case "$t" in
    rocm)  # ROCm 7.2 wheels (WSL2 / native Linux), torch pinned to what is measured to work
      $UV pip compile pyproject.toml --extra server --extra rocm --python-platform x86_64-unknown-linux-gnu \
        --python-version $PY --torch-backend rocm7.2 --emit-index-url -c requirements/constraints-rocm.txt \
        -o requirements/rocm.txt
      add_torch_index requirements/rocm.txt rocm7.2 ;;
    cuda)  # NVIDIA, Linux x86_64, CUDA 12.8 wheels (causal-conv1d is the optional cuda-fast extra: sdist only)
      $UV pip compile pyproject.toml --extra server --extra cuda --python-platform x86_64-unknown-linux-gnu \
        --python-version $PY --torch-backend cu128 --emit-index-url -o requirements/cuda.txt
      add_torch_index requirements/cuda.txt cu128 ;;
    cpu)   # CPU wheels, Linux + Windows (CI)
      $UV pip compile pyproject.toml --extra server --extra cpu --universal --python-version $PY \
        --torch-backend cpu --emit-index-url -o requirements/cpu.txt
      add_torch_index requirements/cpu.txt cpu ;;
    macos) # Apple Silicon (MPS), default PyPI wheels
      $UV pip compile pyproject.toml --extra server --extra mps --python-platform aarch64-apple-darwin \
        --python-version $PY -o requirements/macos.txt ;;
    dev)   # CPU + pytest/ruff, same pins as cpu.txt (its index lines are stripped from the constraints copy)
      tmp="$(mktemp)"; grep -v '^--' requirements/cpu.txt > "$tmp"
      $UV pip compile pyproject.toml --extra server --extra cpu --extra dev --universal --python-version $PY \
        --torch-backend cpu --emit-index-url -c "$tmp" -o requirements/dev.txt
      rm -f "$tmp"
      sed -i -E 's#[A-Za-z:/_.0-9-]*/tmp\.[A-Za-z0-9]+#requirements/cpu.txt#g' requirements/dev.txt
      add_torch_index requirements/dev.txt cpu ;;
    *) echo "unknown target $t" >&2; exit 2 ;;
  esac
done
