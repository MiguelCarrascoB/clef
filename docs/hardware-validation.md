# Hardware validation checklist

For maintainers with a real Mac (Apple Silicon) or NVIDIA machine. The unit tests mock the backends and never load
the model, so they cannot prove these backends. Until this checklist has been run and its results pasted back, the
README marks CUDA and MPS as "untested on hardware".

Time: ~30-60 min plus the ~19 GB download. Needs: 24 GB NVIDIA GPU or Mac with >= 32 GB (or a 16 GB NVIDIA for the
quantized rows).

## 1. Install

Follow the quick start for your platform in the README (lock file: `requirements/cuda.txt` or `requirements/macos.txt`),
with the dev extras:

```bash
git clone https://github.com/MiguelCarrascoB/clef && cd clef
uv venv --python 3.12 && source .venv/bin/activate
uv pip install -r requirements/<cuda|macos>.txt
uv pip install -e ".[server,dev]"
```

## 2. Environment and weights

```bash
clef doctor --json > doctor.json     # keep this file
clef download                        # ~19 GB, resumable
```

`doctor` must report the expected backend (`cuda` or `mps`). On NVIDIA the "fast path is not available" warning should
be gone once `flash-linear-attention` and `causal-conv1d` are installed; on Mac the torch fallback is expected.

## 3. Smoke test (loads the model)

```bash
clef doctor --smoke
```

Passes when one fixed record returns probabilities that sum to 1. Note the printed latency and load time.

## 4. Serve

```bash
clef serve --detach
clef status                          # backend, device, status=ready
curl -s http://127.0.0.1:8910/health
```

## 5. Integration tests against the live server

```bash
pytest -m gpu tests/integration
```

Includes the check that `/v1/classify` matches the equivalent `/v1/systemone` call within 1e-3.

## 6. Benchmarks

```bash
clef bench --requests 100            # HTTP: concurrency 1/2/4/8/16
clef stop
python bench/parity.py               # dtype / quantization parity vs bf16 (needs the GPU, server stopped)
```

Smaller-memory modes ([memory.md](memory.md)): run `bench/memory_bench.py` for bf16, `--offload cpu --max-device-gb 15`
(16 GB cards), `--quant int8 --offload cpu` (12 GB cards) and, on NVIDIA, `--quant int8 --quant-backend bnb` and
`--quant nf4` (`pip install bitsandbytes`). Save the bf16 run with `--save-reference` and pass it to the others with
`--reference` (commands in memory.md), then paste the JSON results.

Optionally compare padding: `CLEF_PAD_MULTIPLE=64` (default) vs `8` vs `128`, since the best value was only measured on ROCm.

## 7. What to paste back

In an issue or PR (use the bug report template if something failed):

- `doctor.json` (the `clef doctor --json` output)
- the `clef bench` table
- `bench/parity.py` output (max abs prob diff per mode)
- peak memory (from `/health` or `nvidia-smi` / Activity Monitor) and load time
- pass/fail of `pytest -m gpu tests/integration`, with the failing test output if any

Do not paste API keys.

## 8. Flip the status

When steps 2-6 pass:

1. Add a row to `docs/hardware-results.md` with the numbers, device, date and clef version.
2. In `README.md`, change the platform's support-matrix status from "untested on hardware" to
   "verified (<device>)", e.g. "verified (RTX 4090)" or "verified (M3 Max, 64 GB)".
3. Add a line to `CHANGELOG.md` under the next release.

If something failed, do not flip the status; open an issue with the pasted output.
