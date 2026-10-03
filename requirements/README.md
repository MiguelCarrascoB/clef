# Requirements and install matrix

`pyproject.toml` is the source of truth. The server needs a PyTorch build that depends on the platform and the GPU,
so it is split into extras:

| Extra | Contents |
| --- | --- |
| *(base)* | `httpx`: enough for `clef_client` |
| `server` | fastapi, uvicorn, pydantic, pillow, imageio(+ffmpeg), numpy, transformers==5.10.2, safetensors, huggingface_hub, einops, psutil. **No torch.** |
| `cuda` | torch, flash-linear-attention, nvidia-ml-py, bitsandbytes (`CLEF_QUANT=int8/nf4` on CUDA; also works on ROCm, see docs/memory.md) |
| `quant` | torchao (weight-only int8 for `CLEF_QUANT=int8`; already part of `rocm`, `mps` and `cpu`; optional on CUDA via `CLEF_QUANT_BACKEND=torchao`) |
| `cuda-fast` | causal-conv1d (sdist only: needs `nvcc` / the CUDA toolkit; optional speed-up, `clef doctor` reports the fast path) |
| `rocm` | torch, flash-linear-attention, torchao (torch must come from the ROCm index) |
| `mps` | torch, torchao (Apple Silicon) |
| `cpu` | torch, torchao (CPU wheels) |
| `dev` | pytest, pytest-asyncio, ruff, httpx |

Always combine `server` with exactly one backend: `clef-local[server,cuda]`, `clef-local[server,rocm]`, ...

## Lock files (exact pins, Python 3.10 resolution)

| File | Platform | torch |
| --- | --- | --- |
| `rocm.txt` | Linux x86_64 (also WSL2) | `2.11.0+rocm7.2` from `download.pytorch.org/whl/rocm7.2` (+ `triton-rocm 3.6.0`); the set measured on the RX 7900 XTX |
| `cuda.txt` | Linux x86_64 | `2.11.0+cu128` from `.../whl/cu128` |
| `cpu.txt` | Linux + Windows (universal; also resolves macOS) | `2.14.1+cpu` from `.../whl/cpu` |
| `macos.txt` | macOS arm64 | `2.11.0` default PyPI wheels (MPS) |
| `dev.txt` | CPU + pytest/ruff; same pins as `cpu.txt` (used by CI) | `2.14.1+cpu` |

Every file starts with `--index-url` and `--extra-index-url` lines, so `pip install -r` and `uv pip install -r` fetch
torch from the right index. `constraints-rocm.txt` holds the hand-picked ROCm pins used when compiling `rocm.txt`.

Known gaps: `causal-conv1d` has no wheels for any platform, so it is left out of every lock (install it with the
`cuda-fast` extra on a machine with the CUDA toolkit). `bitsandbytes` is only in `cuda.txt` (its nf4 also loads on ROCm: `pip install bitsandbytes`, not recommended); `torchao` is in `rocm.txt`, `macos.txt`, `cpu.txt` and `dev.txt`.
No lock is provided for native Windows + CUDA: use `--torch-backend` (below) or install CUDA torch first.

### Regenerate

Needs uv >= 0.8 (for `--torch-backend`; the system uv on Windows is 0.4 and too old: `pip install -U uv`).

```bash
bash requirements/compile.sh              # all lock files
bash requirements/compile.sh rocm cuda    # a subset (dev.txt is resolved against cpu.txt: recompile it after cpu)
```

The commands it runs (in this order):

```bash
uv pip compile pyproject.toml --extra server --extra rocm --python-platform x86_64-unknown-linux-gnu --python-version 3.10 --torch-backend rocm7.2 --emit-index-url -c requirements/constraints-rocm.txt -o requirements/rocm.txt
uv pip compile pyproject.toml --extra server --extra cuda --python-platform x86_64-unknown-linux-gnu --python-version 3.10 --torch-backend cu128 --emit-index-url -o requirements/cuda.txt
uv pip compile pyproject.toml --extra server --extra cpu --universal --python-version 3.10 --torch-backend cpu --emit-index-url -o requirements/cpu.txt
uv pip compile pyproject.toml --extra server --extra mps --python-platform aarch64-apple-darwin --python-version 3.10 -o requirements/macos.txt
uv pip compile pyproject.toml --extra server --extra cpu --extra dev --universal --python-version 3.10 --torch-backend cpu --emit-index-url -c <cpu.txt without option lines> -o requirements/dev.txt
```

uv emits only the default index, so the script appends the `--extra-index-url https://download.pytorch.org/whl/<backend>`
line to each file (`add_torch_index`).

## Install recipes

Clone (reproducible, exact pins), then the package itself without re-resolving:

```bash
git clone https://github.com/MiguelCarrascoB/clef && cd clef
uv venv && source .venv/bin/activate            # Windows: .venv\Scripts\activate
uv pip install -r requirements/cuda.txt         # or rocm.txt / cpu.txt / macos.txt
uv pip install -e . --no-deps
clef doctor --no-gpu && clef download && clef serve
```

One-liner, no clone (not on PyPI: install from the git URL). `uv >= 0.8` can pick the torch wheel for the machine
(`--torch-backend=auto` inspects the NVIDIA driver / ROCm install; `cpu`, `cu128`, `rocm7.2`, ... force one):

```bash
uv tool install "clef-local[server,cuda] @ git+https://github.com/MiguelCarrascoB/clef" --torch-backend=auto
uv tool install "clef-local[server,rocm] @ git+https://github.com/MiguelCarrascoB/clef" --torch-backend=rocm7.2
uv tool install "clef-local[server,mps]  @ git+https://github.com/MiguelCarrascoB/clef"      # macOS
uv tool install "clef-local[server,cpu]  @ git+https://github.com/MiguelCarrascoB/clef" --torch-backend=cpu
```

Notes on how this works (verified with uv 0.12):

- `--torch-backend` exists on `uv pip install` and `uv tool install` (also `UV_TORCH_BACKEND`). It only redirects the
  PyTorch-ecosystem packages (torch, torchvision, triton, ...) to the matching `download.pytorch.org/whl/<backend>` index.
- `[tool.uv.sources]` is deliberately not used: it is honoured only by `uv sync`/`uv lock` inside the project, not for
  users installing the package from git or a wheel, and pip ignores it.
- On native Windows the PyPI torch wheel is CPU only. For CUDA use `--torch-backend=auto` (or `cu128`).
- With plain pip: `pip install "clef-local[server,cuda] @ git+https://..." --extra-index-url https://download.pytorch.org/whl/cu128`.
- `clef doctor` reports which torch build and fast paths were found; if it says CPU on a GPU machine, the torch wheel is wrong.
