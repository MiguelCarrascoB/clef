# Performance

Measured on the AMD RX 7900 XTX (ROCm 7.2, WSL2, bf16), the only GPU the project has been run on. Records of ~330
tokens, `clef bench`, 100 requests per level. v2 and v3 were run back to back on 2026-10-03 under the same conditions
(details: [hardware results](hardware-results.md)).

| | v2 | v3 |
| --- | --- | --- |
| Single record p50 / p95 | 150.1 / 155.4 ms | 149.0 / 159.2 ms (2nd run 149.3 / 153.6) |
| Throughput, concurrency 1 | 6.6 req/s | 6.6-6.7 req/s |
| Throughput, concurrency 8 | 8.3 req/s | 8.1-8.2 req/s |
| Throughput, concurrency 16 (micro-batching) | 8.1 req/s | 8.8-8.9 req/s |
| Peak VRAM | ~19.5 GB | ~19.5 GB |

!!! info "Compute-bound"
    GEMMs are ~83% of GPU kernel time at ~95 TFLOPS, so batching adds only ~30% throughput. Single-record forwards
    spend ~50% of the time on host overhead; graph capture would remove most of it but is blocked by a device sync in
    transformers' SDPA masking. Padding text batches to a multiple of 64 beats both raw lengths and full buckets
    (`CLEF_PAD_MULTIPLE`); that was measured on ROCm only. NVIDIA, Mac and CPU numbers are pending hardware.

## Sharing the GPU with jobs

Requests and jobs share one model and one worker thread. The engine queue has two lanes: requests are
interactive, job chunks are background, and the next micro-batch is interactive whenever a request waits (one
class per forward; after `CLEF_INTERACTIVE_BURST` interactive micro-batches in a row, one job micro-batch runs).
A request during a job still waits for the job micro-batch already being computed (one forward per length
bucket or media record in it). Before the lane, interactive p50 was
422 ms during a job vs 109 ms idle ([Jobs](jobs.md)).

## Opt-in speedups (not yet measured on a GPU)

Two flags and one existing switch, all **off by default** until `bench/perf_check.sh` has been run on the GPU box.
With the flags off the code path is the release's own `ClefModel.forward` (CPU results are byte-identical).
The flags only apply to the `clef-release` adapter (`clef-flash`, `clef`).

| Flag | What it does | Exact? |
| --- | --- | --- |
| `CLEF_MASK_FREE=1` | Text-only batches call the backbone with `attention_mask=None`; the head gets a CPU-built mask, so it takes the lengths without a device sync. Removes the syncs in transformers' mask helpers (`torch.all(mask == 1)` in the linear-attention path, the SDPA causal check), the blocker for graph capture. | Yes in exact arithmetic: the model is causal, position `t` reads only `<= t`, and the engine pads on the right, so real positions never see padding. The head slices `[:length]` and takes `hidden[length-1]` as the global vector, so it never reads padded positions either. Not applied (and not exact) for media batches (rope positions and the visual-token scatter read the mask), left padding or masks with holes. Real positions can still differ in the last bits (a different SDPA kernel for `is_causal`), hence the parity gate. |
| `CLEF_FAST_HEAD=1` | A batch-vectorised, sync-free joint head (`fast_joint_head` in `models/clef_release.py`): index tables are built on the CPU from the encodings and sent in one transfer, span means are segment sums, ragged question and option sets are padded grids with key-padding masks, logits come back in one device-to-host copy. It reads the release head's weights; the model directory is untouched. The segment-mean formulation follows vLLM's Apache-2.0 ClefFlash head (vllm-project/vllm#61308); the code is an independent rewrite. | Same maths as the release head, different loop structure. Tests compare it with the release head on random tensors: max difference below 1e-5 in fp32 (and about one bf16 ulp in bf16). |

Both are expected to help mostly single-record latency (the forward is ~50% host overhead) and, if step 0 of the
check shows batched serving is not GEMM-bound, batched throughput too. Nothing here is a claim until measured.

### TunableOp (ROCm)

`CLEF_TUNABLEOP=1` makes PyTorch autotune each GEMM shape once and remember the winner. The tuned kernels are
written to `<state_dir>/tunableop/<model>@<rev8>/tunableop_0.csv` and reloaded on the next start.

1. `CLEF_TUNABLEOP=1 clef serve --detach` with warmup on (the default): startup tunes every bucket shape, so the
   first start is slow (minutes) and the file grows.
2. Run `clef bench` once, so shapes the warmup did not visit (odd pad lengths, batch sizes) are tuned too.
3. Restart with the same variable and measure the second run: that is the number to compare with TunableOp off.
4. Delete the CSV to re-tune (after a ROCm, PyTorch or model change).

Anything not tuned yet is tuned on its first request (a latency spike), so do not judge the first run.

### The GPU check

```bash
ssh gpu; cd ~/src/clef && git checkout upg/d2-perf
PY=~/venvs/clef/bin/python bash bench/perf_check.sh        # QUICK=1 for a ~15 min smoke
```

It runs the batch-8 profile (is batched serving GEMM-bound?), engine parity for each flag against the flags-off
run (PASS = 0 top-1 flips and max |dp| <= 2e-3), `clef bench` at c=1/8/16 for each flag and both, and TunableOp
on vs off. Output is in `bench/out/perf/` (`summary.md` has the PASS/FAIL and speed tables). Copy the numbers
into [hardware results](hardware-results.md) and flip a default only after parity passes and the gain is outside
the ~5% same-day drift.

## Job throughput: length sort and tokenize ahead (pending measurement)

Batch jobs feed the engine in micro-batches of `CLEF_MAX_MICROBATCH` (default 8) in input order, so one forward
can mix a 60-token and a 900-token record and pays the padding of the longest. Two opt-in flags change how a
job is fed. Both are **off by default** and nothing here is a claim until `bench/sort_check.sh` has run on the GPU.
The result rows, their order and the cache behaviour are the same either way.

| Flag | What it does |
| --- | --- |
| `CLEF_JOB_SORT_WINDOW=N` (`0` = off; 256-1024 recommended) | The job takes N items at a time, encodes them all on CPU threads, sorts the ones that need a forward by token length and sends them to the engine in micro-batches of `CLEF_MAX_MICROBATCH`, so each forward carries similar lengths. Results go back to input order and the window is persisted in one write. |
| `CLEF_TOKENIZE_AHEAD=1` | While a window is on the GPU, the next one is encoded on CPU threads (one micro-batch when the sort is off). Still ONE GPU worker and ONE job micro-batch in flight; interactive requests keep going first. |

What to expect, and what changes:

- Progress and `job.progress` webhooks move once per window instead of once per micro-batch, and a clean stop,
  crash or cancel inside a window recomputes (or drops) at most that window. Rows are never persisted out of order,
  so results stay gap-free and resume works as before.
- Sorting only changes padding, so probabilities can differ in the last bits (a different batch composition);
  labels should not change. `sort_check.sh` reports label agreement with the flags-off run.
- With `CLEF_TOKENIZE_AHEAD=1` the next window's cache lookup happens before the current window's results are
  stored, so a record repeated in the next window is computed again (same result, different cache statistics).
- A record the engine rejects, a GPU OOM or a failed encode falls back to the usual per-record isolation.

### The GPU run (pending)

```bash
ssh gpu; cd ~/src/clef && git checkout upg/f2-sort     # push the branch from the devbox first
PY=~/venvs/clef/bin/python bash bench/sort_check.sh    # ~20 min; QUICK=1 for a ~5 min smoke, SWEEP=1 adds 256/1024
```

It starts a fresh server per variant (base, sort with window 512, tokenize ahead, both) with the result cache
off, submits one classify job of 2000 long-tailed-length inputs three times after a warm-up job, probes
interactive latency every 2 s, and prints items/s, gain over base, label agreement and a verdict
(`bench/out/sort/summary.md`). Flip a default only if the median gain is over 10% and above the base spread, labels
agree and interactive p95 did not get worse. Try `MICROBATCH=16` as well: the gain depends on how many records a
forward holds.

| Variant | Items/s (median) | Gain vs base | Label agreement | Interactive p95 | Verdict |
| --- | --- | --- | --- | --- | --- |
| base (flags off) | pending | - | - | pending | - |
| `CLEF_JOB_SORT_WINDOW=512` | pending | pending | pending | pending | pending |
| `CLEF_TOKENIZE_AHEAD=1` | pending | pending | pending | pending | pending |
| both | pending | pending | pending | pending | pending |

## Smaller-memory modes

Same GPU, same session, 200 mixed records (`bench/memory_bench.py`). Offload is lossless: the probabilities are
bit-identical to bf16. Full table, method and caveats: [Smaller GPUs](memory.md).

| Setting | Device GB (peak allocated) | single p50 | req/s at concurrency 8 | Max / mean abs probability difference |
| --- | --- | --- | --- | --- |
| bf16 (default) | 18.4 | 152 ms | 4.72 | reference |
| `CLEF_OFFLOAD=cpu`, cap 15 (16 GB card) | 13.1 | 175 ms | 4.15 | 0 / 0 |
| `CLEF_OFFLOAD=cpu`, cap 11 (12 GB card) | 9.0 | 343 ms | 3.37 | 0 / 0 |
| `CLEF_OFFLOAD=cpu`, cap 7 (8 GB card) | 5.1 | 540 ms | 2.59 | 0 / 0 |
| `CLEF_QUANT=int8` + `CLEF_OFFLOAD=cpu` (12 GB card) | 8.3 | 206 ms | 4.10 | 0.062 / 0.0058 |

Offload latency is the PCIe copy of the layers kept in host RAM, partly hidden behind compute, and it needs host RAM
(about 6, 11 and 15 GB for the three caps). On NVIDIA and Apple Silicon these modes are untested on hardware.

## Benchmarks

| Tool | Use |
| --- | --- |
| `clef bench` | HTTP load test against the running server (realistic) |
| `bench/bench.py` | in-process, needs the GPU |
| `bench/profile_forward.py` | `torch.profiler` |
| `bench/parity.py` | probability parity across dtypes and quantization, and engine parity (`--engine-dump` / `--compare`) |
| `bench/perf_check.sh` | the D2 GPU check: profile, parity per flag, `clef bench` A/B, TunableOp on/off; PASS/FAIL table |
| `bench/memory_bench.py` | one smaller-memory setting per process: device and host memory, latency, parity against a saved bf16 reference |
