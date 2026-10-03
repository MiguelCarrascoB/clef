# Troubleshooting

Run `clef doctor` first (`clef doctor --json` for a bug report). On Windows + WSL use `.\clef.ps1 doctor`.

Contents: [CUDA](#cuda) - [ROCm under WSL2](#rocm-under-wsl2) - [MPS (macOS)](#mps-macos) - [CPU](#cpu) -
[Common](#common)

## CUDA

Untested on hardware at the time of writing; these are the known failure modes of the stack.

- **`CUDA not available` / backend falls to cpu**: the installed torch is a CPU build, or the driver is missing. Check
  `nvidia-smi`, then `python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"`.
  Reinstall from `requirements/cuda.txt`.
- **Driver / toolkit mismatch** (`CUDA driver version is insufficient`, `no kernel image is available`): the driver
  must support the CUDA version of the torch wheel (`nvidia-smi` shows the maximum supported CUDA version at the top
  right). Update the driver; you do not need a system CUDA toolkit for the wheels.
- **"The fast path is not available" warning** (transformers): `flash-linear-attention` and `causal-conv1d` are
  missing. The model still runs through a slower torch fallback. Install `flash-linear-attention` (part of the CUDA
  lock file) and `causal-conv1d`; `clef doctor` shows which one is missing.
- **`causal-conv1d` fails to build**: it compiles CUDA code and needs `nvcc` matching the torch CUDA version, a C++
  compiler and `ninja`. Prefer a prebuilt wheel matching your torch/CUDA/Python, or install the CUDA toolkit of the same
  major.minor as `torch.version.cuda`. If it cannot be built, run without it; correctness is unchanged.
- **`CLEF_QUANT` errors**: quantization is CUDA only and needs `bitsandbytes` (`pip install bitsandbytes`). On other
  backends it is refused with a clear error. If bitsandbytes cannot find the CUDA libraries, check
  `python -m bitsandbytes`. Quantized probabilities differ slightly from bf16 (numbers pending hardware).
- **Out of memory on load or under load**: bf16 needs ~19.5 GB peak. Close other GPU users, use a 24 GB card, or
  `CLEF_QUANT=int8|nf4`. Preflight (`CLEF_PREFLIGHT=1`) reports this before loading. At runtime an OOM returns 503 and
  the server keeps running; lower `CLEF_MAX_MICROBATCH` or `CLEF_MAX_TOKENS`.
- **bf16 on older GPUs** (pre-Ampere): bf16 is not supported; clef falls back to fp16 with a warning. Check parity with
  `bench/parity.py`.

## ROCm under WSL2

This is the verified setup (RX 7900 XTX, ROCm 7.2).

- **`No CUDA GPUs are available` / torch sees no device**: `HSA_ENABLE_DXG_DETECTION=1` must be set (clef sets it
  automatically when `/dev/dxg` exists; required until ROCm 7.13+ in WSL). If you run torch yourself, export it.
- **`/dev/dxg` missing**: update the Windows AMD Adrenalin driver, make sure WSL2 (not WSL1) is used
  (`wsl -l -v`), then `wsl --shutdown`.
- **Permission denied on `/dev/dxg` or `/dev/kfd`**: add your user to the `render` group
  (`sudo usermod -aG render $USER`), then restart the distro.
- **WSL stops the server a few seconds after you close the terminal**: WSL shuts an idle distro down ~15 s after the
  last `wsl.exe` session exits. `clef.ps1 start` keeps a hidden keep-alive session for this reason; use it (or the
  systemd user service with `systemd=true` in `/etc/wsl.conf`, see `deploy/wsl.conf.sample`) instead of a bare
  `wsl -e clef serve --detach`.
- **MSYS path conversion** (Git Bash): `wsl ... /mnt/c/...` arguments get rewritten to `C:/Program Files/Git/...`.
  Prefix the command with `MSYS_NO_PATHCONV=1`, or use PowerShell.
- **`expandable_segments` is not supported** on ROCm/WSL: do not set `PYTORCH_ALLOC_CONF=expandable_segments:True`
  (it logs a warning and is ignored at best).
- **First forwards take seconds to minutes**: one-time Triton/SDPA kernel compilation per input shape. The startup
  warmup covers the length buckets; caches live in `~/.triton/cache`. The first warmup after a fresh install takes
  ~110 s, later ones ~12 s.
- **Out of memory in WSL**: set `memory=26GB` in `%USERPROFILE%\.wslconfig` (for a 32 GB host); it applies after
  `wsl --shutdown`.
- **causal-conv1d**: its HIP build does not work on WSL-ROCm yet. `flash-linear-attention` is active and the torch
  fallback for the convolution is already fast; the fast-path warning is expected for `causal_conv1d` only.
- **bench refuses to run**: the server holds ~19.6 GB of VRAM; stop it first (`clef stop`) or pass `--force`.
- **AMD telemetry under WSL is partial**: measured on the RX 7900 XTX (ROCm 7.2), `amdsmi` reports GPU utilization and temperature, but no power, and its VRAM usage is nonsense (clef drops it and charts the allocator figure instead). `rocm-smi` is not tried under WSL. Without the `amdsmi` Python package, telemetry shows "unavailable"; that is harmless.

## MPS (macOS)

Untested on hardware at the time of writing.

- **bf16 on macOS < 14**: MPS bf16 needs macOS 14+. clef falls back to fp16 with a warning; update macOS or check
  parity with `bench/parity.py`.
- **Memory pressure / the Mac becomes unresponsive**: the model takes ~19.5 GB of unified memory; 32 GB is the
  realistic floor. Close other apps, lower `CLEF_MAX_MICROBATCH`, keep images small (`CLEF_MAX_PIXELS`). Preflight
  warns when free memory looks too low.
- **`NotImplementedError: ... not currently implemented for the MPS device`**: clef sets
  `PYTORCH_ENABLE_MPS_FALLBACK=1` automatically so unsupported ops run on the CPU. If you import torch yourself before
  clef, export it in your shell.
- **fla Triton kernels not available**: expected. Triton does not run on macOS, so the torch fallback is used and the
  transformers fast-path warning is normal.
- **`CLEF_QUANT` is refused**: bitsandbytes quantization is CUDA only.
- **Slow first request**: MPS compiles kernels lazily; the startup warmup covers the common shapes.
- **Closing the lid / sleep stops the server**: use the launchd agent in `deploy/` and keep the Mac awake
  (`caffeinate -i`).

## CPU

- **Out of memory / the machine swaps**: the model needs ~40 GB of RAM on CPU (weights plus activations). Use a GPU
  backend if you have one.
- **Very slow**: expected, seconds per record. CPU mode is for development and CI (the unit tests never load the
  model). Set `CLEF_WARMUP=0` to skip the warmup.
- **Backend detected as cpu on a machine with a GPU**: you installed a CPU torch build. Reinstall from the matching
  lock file, or force a device with `CLEF_DEVICE=cuda|rocm|mps` to get an explicit error with a hint.

## Common

- **Port already in use**: something else answers on 8910. `clef status` shows whether it is clef. Stop it
  (`clef stop`), or use `CLEF_PORT=8911` / `clef serve --port 8911`. `clef serve --detach` refuses to start when the port
  answers.
- **401 Unauthorized**: keys are configured; send `X-API-Key: <key>` or `Authorization: Bearer <key>`. Check the key
  name in the keys file (`name:key`). See [remote-access.md](remote-access.md).
- **429 Too Many Requests**: `CLEF_RATE_LIMIT` is exceeded for that key; wait `Retry-After` seconds or raise the limit.
- **503 Service Unavailable**: the model is still loading/warming (`clef status`, `/health`) or the device ran out of
  memory. The clients retry with backoff.
- **Model not found**: run `clef download`. Path resolution: `CLEF_MODEL_PATH` if set, else `~/models/clef-flash` if it
  holds a release, else the pinned snapshot in the Hugging Face cache. The server never downloads on its own
  (`/health` shows `status: error` with the message).
- **Download fails or stalls**: `clef download` is resumable, just run it again. It needs ~21 GB free; behind a proxy
  set `HTTPS_PROXY`. Set `HF_HOME` to put the cache on another disk.
- **Not enough disk space**: the weights are ~19 GB; `clef doctor` and `clef download` check this up front.
- **Where are the logs / state?** `clef logs -f`. The state directory is `~/.local/state/clef` on Linux,
  `~/Library/Application Support/clef` on macOS, `%LOCALAPPDATA%\clef` on Windows (`CLEF_STATE_DIR` overrides;
  `CLEF_LOG_DIR` is a legacy alias). It holds `server.log`, `server.pid` and `classifiers/`.
- **Upgrading from v2**: see the migration notes in the [CHANGELOG](../CHANGELOG.md).
