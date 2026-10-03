# Configuration

All settings are `CLEF_*` environment variables; the flags of `clef serve` set the same variables. The full table,
including the v2 limits (`CLEF_MAX_TOKENS`, `CLEF_MAX_BODY_MB`, `CLEF_MAX_BATCH`, batching and padding knobs), is in
[Architecture](ARCHITECTURE.md#configuration-configpy). The ones you will touch first:

| Variable | Default | Meaning |
| --- | --- | --- |
| `CLEF_DEVICE` | `auto` | `auto`, `cuda`, `rocm`, `mps`, `cpu` (`cuda:1` / `rocm:1` select an index) |
| `CLEF_DTYPE` | `auto` | `auto`, `bfloat16`, `float16`, `float32` |
| `CLEF_QUANT` | `none` | `none`, `int8`, `nf4` (bitsandbytes, CUDA only; refused elsewhere with a clear error) |
| `CLEF_MODEL_PATH` | unset | release dir. Unset: `~/models/clef-flash` if it holds a release, else the pinned HF cache snapshot |
| `CLEF_HOST` / `CLEF_PORT` | `127.0.0.1` / `8910` | bind address. A non-loopback host without keys logs a loud warning |
| `CLEF_API_KEY` / `CLEF_API_KEYS` | unset | one key, or `name:key,name2:key2` / a path to a file with one `name:key` per line |
| `CLEF_RATE_LIMIT` | `0` | requests per minute per key (per client IP when auth is off); 0 = off |
| `CLEF_CORS_ORIGINS` | empty | comma-separated allowed origins (`*` allowed); empty = CORS off |
| `CLEF_STATE_DIR` | per OS | logs, pidfile and saved classifiers |
| `CLEF_MAX_LABELS` | `64` | labels per classify request / saved classifier |
| `CLEF_CLASSIFY_THRESHOLD` | `0.5` | default multi-label threshold |
| `CLEF_PAD_MULTIPLE` | `64` | pad text batches to a multiple of this (measured on ROCm only) |

Example for LAN access:

```bash
CLEF_HOST=0.0.0.0 CLEF_API_KEYS=~/.config/clef/keys CLEF_RATE_LIMIT=120 clef serve
```

See [remote access](remote-access.md) before exposing the server beyond the machine.
