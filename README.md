# clef-flash local (AMD Radeon RX 7900 XTX / WSL2 ROCm)

[Cloudflare/clef-flash](https://huggingface.co/Cloudflare/clef-flash) running **100% locally** on this PC,
with GPU acceleration on the RX 7900 XTX via **ROCm 7.2 in WSL2** (ROCDXG) and the model's exact tested
stack: **torch 2.11.0+rocm7.2 + transformers 5.10.2**.

## What this model does

Clef-flash is a 9B **multimodal decision model**: it reads a `state` (any text/JSON **or images/videos**)
plus a schema of typed questions, and returns **a calibrated probability for every allowed option of every
question in a single forward pass** - no free-form text generation, no output parsing.

- Question types: `choice` (named options), `score` (ordered levels), `noul` (true/false).
- Batch the records for near-linear throughput.
- Served through the Jev / SystemOne-compatible API (`POST /v1/systemone`).

## Where things live

| What | Where |
| --- | --- |
| Model weights (18.9 GB, git-lfs) | WSL: `~/models/clef-flash` (Ubuntu) |
| Python env (torch 2.11.0+rocm7.2) | WSL: `~/venvs/clef` |
| GPU runtime (ROCm 7.2.4 + librocdxg 1.2.2) | WSL: `/opt/rocm` + dpkg |
| Server / tests / setup scripts | Windows: `C:\path\to\clef` (this repo) |

## Quick start

```powershell
# 1) Start the server (Windows -> launches detached inside WSL on port 8910)
wsl -d Ubuntu -- bash -c "bash /path/to/clef/scripts/launch_server.sh"

# 2) Watch it come up (loads the model in ~45 s)
wsl -d Ubuntu -- bash -c "tail -f /tmp/server.log"

# 3) Health check
powershell:  Invoke-RestMethod http://localhost:8910/health

# 4) Make a decision (examples below), or run the full suite:
powershell -NoProfile -ExecutionPolicy Bypass -File server\test_api.ps1        # text/JSON + batch
wsl -d Ubuntu -- bash /path/to/clef/server/test_image.sh # image capability
wsl -d Ubuntu -- bash /path/to/clef/server/test_video.sh # video capability
```

## API examples

### Text / JSON state (choice + score + noul)

```powershell
$body = @{
  model     = 'clef-flash'
  state     = 'Our checkout started returning errors and orders are blocked.'
  questions = @{
    department = @{ type = 'choice'; instructions = 'Which team should handle the message?'
                    criteria = @{ billing = 'Payments or invoices'; technical = 'Bugs or outages' } }
    urgency    = @{ type = 'score';  criteria = @('Can wait', 'This week', 'Today') }
    outage     = @{ type = 'noul';   instructions = 'Is a service down?' }
  }
} | ConvertTo-Json -Depth 6
Invoke-RestMethod http://localhost:8910/v1/systemone -Method Post -Body $body -ContentType 'application/json'
```

Verified response shape:

```json
{
  "model": "clef-flash",
  "answers": {
    "department": { "type": "choice",  "choice": "technical", "confidence": 0.9602,
                     "probabilities": { "billing": 0.0398, "technical": 0.9602 } },
    "urgency":    { "type": "score",   "score": 1.783, "confidence": 0.8561,
                     "legend": { "0": "Can wait", "1": "This week", "2": "Today" },
                     "probabilities": { "0": 0.0731, "1": 0.0708, "2": 0.8561 } },
    "outage":     { "type": "noul",    "noul": 0.8379 }
  },
  "usage": { "input_tokens": 300, "output_tokens": 0 }
}
```

### Images / video

`images` accepts `data:image/...;base64,` or plain `http(s)://` URLs (PIL handles jpeg/png/webp/tiff...).
`videos` accepts `data:video/*;base64,` mp4/webm/mov URLs or http(s) URLs - they're decoded with
imageio/ffmpeg and sampled at 2 fps (matching the model card defaults; tweak via `media_kwargs`).

### Batch (multiple records, ONE forward pass)

```json
POST /v1/batch
{ "batch": [ {model, state, questions}, {model, state, questions}, ... ] }
```

Benchmark on the RX 7900 XTX (`bench/bench.py`): single-record median **162 ms**, p95 **222 ms**
(H200 reference: 38.8 ms / 122.4 ms); batch-of-8 ~ 9 records/s; peak VRAM 19.6 GB of 25.7 GB.

## Direct Python (no server)

```bash
wsl -d Ubuntu -- bash -lc 'source ~/venvs/clef/bin/activate && export HSA_ENABLE_DXG_DETECTION=1 && \
  python /path/to/clef/scripts/model_smoke.py'
```

## Reinstalling from scratch

Run as root inside WSL: `bash /path/to/clef/scripts/wsl_setup_root.sh`, then as
user: `bash .../scripts/wsl_setup_user.sh` (downloads weights + builds the venv).

## Troubleshooting

- **`No CUDA GPUs are available`**: `HSA_ENABLE_DXG_DETECTION=1` must be set - `launch_server.sh` and
  the profile script (`/etc/profile.d/rocm-wsl.sh`) do this; it is required until ROCm 7.13+ in WSL.
- **First forwards are slow (seconds ~ minutes)**: one-time Triton/SDPA kernel compilation per input
  shape; subsequent runs hit warm caches (~160 ms). Registries live in `~/.triton/cache`.
- **OOM in WSL**: `%USERPROFILE%\.wslconfig` sets `memory=26GB` (host has 32 GB); applies after
  `wsl --shutdown`.
- **Optional linear-attn fast path**: `pip install causal-conv1d` builds a CUDA/HIP extension that does
  not build cleanly on WSL-ROCm yet; `flash-linear-attention` (installed, fla 0.5.2) is active and the
  torch fallback is already fast. Retry periodically as AMD improves HIP builds.
- Windows-side duplicate clone at `models\clef-flash` was removed intentionally; weights live only in
  the WSL filesystem (fast safetensors I/O, no double storage).

## License

Model: Apache-2.0 (Cloudflare clef-flash, base model Qwen/Qwen3.5-9B).
