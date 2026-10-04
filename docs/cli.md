# CLI

`clef` is installed with the package. Flags of `clef serve` set the matching `CLEF_*` environment variables, see
[configuration](configuration.md).

| Command | What it does |
| --- | --- |
| `clef serve [--host --port --device --dtype --quant --offload --max-device-memory-gb --model-path --detach --timeout]` | start the server (foreground; `--detach` writes a pidfile and waits until healthy) |
| `clef stop` / `clef status [--json]` / `clef logs [-f] [-n N]` | manage a detached server |
| `clef doctor [--no-gpu] [--smoke] [--json]` | environment checks per backend; `--smoke` loads the model and runs one record |
| `clef download [--revision SHA] [--dir DIR] [--yes]` | pinned, resumable download of the weights with a disk-space check |
| `clef bench [args]` | HTTP load test against the running server |
| `clef open` | open the console in the browser |
| `clef version` | print the version |

## Details

`--quant`, `--offload`, `--max-device-memory-gb`
:   The smaller-memory settings, equal to `CLEF_QUANT`, `CLEF_OFFLOAD` (`none` or `cpu`) and
    `CLEF_MAX_DEVICE_MEMORY_GB` (`0` = no cap). For example `clef serve --offload cpu --max-device-memory-gb 15` on a
    16 GB card. See [Smaller GPUs](memory.md).

`--detach`
:   Spawns the server as a detached process, writes a pidfile, logs to the state dir (rotating the previous log), then
    waits for `/livez` and for `/health` to report `ready` or `warming` (`--timeout`, default 600 s). On failure it
    prints the last 20 log lines and exits 1. Refuses to start when something already answers on the port.

`stop`
:   SIGTERM (Windows: `taskkill /PID /T`), waits 30 s, then kills; removes the pidfile.

`status`
:   Process plus `/health` summary (backend, device, status); exit 0 when healthy.

`doctor`
:   Exit 0 on ok or warn, 1 on any FAIL. `--no-gpu` skips everything that needs a GPU or the weights (CI smoke).
    `--smoke` loads the model, runs one fixed record, checks that probabilities sum to 1 and prints latency. When
    the detected memory is too small for bf16 it prints a `memory setting` line with the option to use
    (offload, int8) and a preflight check that honours your `--offload` / `--max-device-memory-gb` choice.

`download`
:   Fetches the pinned revision into the Hugging Face cache (or `--dir` / `CLEF_MODEL_PATH`). Resumable. Checks for
    >= 21 GB of free disk first and warns about the ~19 GB size.

On Windows with WSL2, `clef.ps1` wraps the CLI (`doctor`, `start`, `open`, `stop`) and keeps a hidden WSL keep-alive
session. More in [Architecture](ARCHITECTURE.md#cli-clef-clipy).
