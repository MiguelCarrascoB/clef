# Install

Every path below ends with the same three commands: `clef download` (one-time, ~19 GB), `clef serve`, then open
<http://127.0.0.1:8910/>. The recommended installer is [uv](https://docs.astral.sh/uv/); plain `pip` in a venv works
too. Pick the lock file that matches your hardware from `requirements/`.

## Hardware requirements

The weights are ~19 GB in bf16 and peak device memory is ~19.5 GB (measured on ROCm), so out of the box you need a
**24 GB GPU** or a **Mac with 32 GB or more of unified memory** (macOS 14+ for bf16). Smaller GPUs work with one
extra setting. `clef doctor` reads your memory and prints the matching line.

| You have | Use | Cost (measured on the RX 7900 XTX) |
| --- | --- | --- |
| 24 GB GPU, 32 GB+ Mac | nothing | none |
| 16 GB GPU | `CLEF_OFFLOAD=cpu CLEF_MAX_DEVICE_MEMORY_GB=15` | identical probabilities; p50 152 to 175 ms; ~6 GB host RAM |
| 12 GB GPU | `CLEF_QUANT=int8 CLEF_OFFLOAD=cpu` | p50 ~206 ms; mean probability error 0.006, 2 of 399 top choices flipped |
| 12 GB GPU, lossless | `CLEF_OFFLOAD=cpu CLEF_MAX_DEVICE_MEMORY_GB=11` | identical probabilities; p50 343 ms; ~11 GB host RAM |
| 8 GB GPU | `CLEF_OFFLOAD=cpu CLEF_MAX_DEVICE_MEMORY_GB=7` | identical probabilities; p50 540 ms; ~15 GB host RAM |
| 24 GB Mac | `CLEF_QUANT=int8` | ~12 GB; **not validated on Apple Silicon** |
| CPU | ~40 GB RAM | works, but slow. Meant for development and tests |

These numbers come from one AMD card; the NVIDIA and Apple Silicon paths run the same code but are untested on
hardware. Set them as environment variables or with `clef serve --offload cpu --max-device-memory-gb 15`. Details,
caveats and the full measurement table: [Smaller GPUs](memory.md).

Disk: ~19 GB for the weights plus ~21 GB free during `clef download`.

!!! note "Nothing is published to PyPI"
    Install from a checkout, from the wheel on the
    [GitHub Releases](https://github.com/MiguelCarrascoB/clef/releases) page, or straight from git (below).

## One-line install

With uv >= 0.8, no clone needed. `--torch-backend=auto` picks the CUDA / ROCm / CPU torch build for the machine; swap
the extra for `rocm`, `mps` or `cpu`.

```bash
uv tool install "clef-local[server,cuda] @ git+https://github.com/MiguelCarrascoB/clef" --torch-backend=auto
clef doctor && clef download && clef serve
```

## Per platform

=== "macOS (Apple Silicon)"

    ```bash
    git clone https://github.com/MiguelCarrascoB/clef && cd clef
    uv venv --python 3.12 && source .venv/bin/activate
    uv pip install -r requirements/macos.txt
    uv pip install -e ".[server]"
    clef doctor            # checks MPS, macOS version, memory, disk
    clef download
    clef serve
    ```

    Needs macOS 14+ for bf16. `PYTORCH_ENABLE_MPS_FALLBACK=1` is set automatically; the fla Triton kernels do not run
    on Mac, so the (fast enough) torch fallback is expected. **Untested on hardware.**

=== "Ubuntu + NVIDIA"

    Requires a current NVIDIA driver (`nvidia-smi` works). No CUDA toolkit is needed for the default install.

    ```bash
    git clone https://github.com/MiguelCarrascoB/clef && cd clef
    uv venv --python 3.12 && source .venv/bin/activate
    uv pip install -r requirements/cuda.txt
    uv pip install -e ".[server]"
    clef doctor
    clef download
    clef serve
    ```

    Less than 24 GB: see the table above, for example `CLEF_OFFLOAD=cpu CLEF_MAX_DEVICE_MEMORY_GB=15 clef serve`
    on a 16 GB card. [Smaller GPUs](memory.md) and [troubleshooting](troubleshooting.md#cuda) have the rest.
    **Untested on hardware.**

=== "Windows + WSL2, AMD ROCm"

    The verified setup. Windows side: current Adrenalin driver, WSL2 with Ubuntu, `.wslconfig` from
    `deploy/wslconfig.sample` (`memory=26GB` on a 32 GB host). Inside WSL:

    ```bash
    sudo bash scripts/wsl_setup_root.sh      # ROCm + librocdxg (once)
    bash scripts/wsl_setup_user.sh           # venv, model download
    ```

    Then from PowerShell:

    ```powershell
    .\clef.ps1 doctor
    .\clef.ps1 start         # detached in WSL, keeps a hidden keep-alive session, waits until ready
    .\clef.ps1 open          # http://localhost:8910/
    .\clef.ps1 stop
    ```

    Details and failure modes: [troubleshooting](troubleshooting.md#rocm-under-wsl2).

=== "Windows + WSL2, NVIDIA CUDA"

    Install the NVIDIA Windows driver (it exposes the GPU to WSL; do not install a Linux driver inside WSL). Then
    follow the **Ubuntu + NVIDIA** tab inside the WSL distro, and use `.\clef.ps1 start` from Windows if you want the
    keep-alive wrapper. **Untested on hardware.**

=== "Docker"

    ```bash
    docker compose --profile cuda up -d     # NVIDIA, needs the NVIDIA container toolkit
    docker compose --profile rocm up -d     # AMD, Linux host only
    ```

    The compose file mounts a volume for the weights; download them once with
    `docker compose --profile cuda run --rm clef-cuda clef download --yes` (or `--profile rocm ... clef-rocm`).
    Ports are published on `127.0.0.1`; for LAN access set `CLEF_BIND=0.0.0.0` together with `CLEF_API_KEYS`.

    !!! warning
        GPU containers on Docker Desktop for Windows/macOS are not supported. Use the WSL2 or native paths.

=== "CPU only"

    ```bash
    uv venv --python 3.12 && source .venv/bin/activate        # Windows: .venv\Scripts\activate
    uv pip install -r requirements/cpu.txt
    uv pip install -e ".[server]"
    clef download && CLEF_DEVICE=cpu clef serve
    ```

    Expect seconds per record and ~40 GB of RAM. Use it for development, not for serving.

## Next

Once `clef serve` reports ready, send your [first request](first-request.md).
