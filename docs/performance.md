# Performance

Measured on the AMD RX 7900 XTX (ROCm 7.2, WSL2, bf16), 2026-10-18: records of ~330 tokens, `clef bench`, 100
requests per level, one session (details: [hardware results](hardware-results.md)).

| | flags off | ROCm defaults (TunableOp + `MASK_FREE` + `FAST_HEAD`) |
| --- | --- | --- |
| Single record p50 / p95 | 137 / 145 ms | 104 / 108 ms |
| Throughput, concurrency 1 | 7.2 req/s | 9.5 req/s |
| Throughput, concurrency 8 | 8.8 req/s | 10.9-11.3 req/s |
| Throughput, concurrency 16 (micro-batching) | 9.0 req/s | 11.1-11.8 req/s |
| Mixed-length traffic, concurrency 1-16 | 5.9-6.2 req/s | 7.6-7.9 req/s |
| Peak VRAM | ~21.7 GB | ~21.0 GB |

!!! info "Compute-bound"
    GEMMs are 81-84% of GPU kernel time, so batching adds only ~25% throughput, and the fastest lever on ROCm is
    better GEMM kernels (TunableOp below). A single-record forward is mostly host overhead (kernel/wall 0.22);
    graph capture would remove most of it but is blocked by a device sync in transformers' SDPA masking. Padding
    text batches to a multiple of 64 beats both raw lengths and full buckets (`CLEF_PAD_MULTIPLE`); that was
    measured on ROCm only. NVIDIA, Mac and CPU numbers are pending hardware.

## Sharing the GPU with jobs

Requests and jobs share one model and one worker thread. The engine queue has two lanes: requests are
interactive, job chunks are background, and the next micro-batch is interactive whenever a request waits (one
class per forward; after `CLEF_INTERACTIVE_BURST` interactive micro-batches in a row, one job micro-batch runs).
A request during a job still waits for the job micro-batch already being computed (one forward per length
bucket or media record in it). Measured on the 7900 XTX (unique requests, cache off): interactive p50 140 ms idle
and ~625 ms during a 3000-item job, with one job or with 2 keys x 2 jobs: about one batch-8 forward
([Jobs](jobs.md)).

## Speedups: on by default on ROCm

Two flags and TunableOp. With the ROCm backend all three are **on by default** (measured on the RX 7900 XTX:
TunableOp +27-32% req/s on mixed-length traffic, the two flags another +5-12% on top); on CUDA, MPS and CPU the two
flags stay off until someone measures them there (`bench/perf_check.sh`), and TunableOp only exists on ROCm builds
of PyTorch. `0` turns each one off, `1` forces a flag on. With both flags off the code path is the release's own
`ClefModel.forward` (CPU results are byte-identical). The flags only apply to the `clef-release` adapter
(`clef-flash`, `clef`).

With all three on, probabilities move by up to ~0.03 (different kernels sum in a different order); in the 399-question
parity set no top answer changed, and with TunableOp alone the 3 that did were near coin flips (top-2 margin below
0.02). Accuracy on the 47 labelled tickets: 0.872 either way.

| Flag | What it does | Exact? |
| --- | --- | --- |
| `CLEF_MASK_FREE=1` | Text-only batches call the backbone with `attention_mask=None`; the head gets a CPU-built mask, so it takes the lengths without a device sync. Removes the syncs in transformers' mask helpers (`torch.all(mask == 1)` in the linear-attention path, the SDPA causal check), the blocker for graph capture. | Yes in exact arithmetic: the model is causal, position `t` reads only `<= t`, and the engine pads on the right, so real positions never see padding. The head slices `[:length]` and takes `hidden[length-1]` as the global vector, so it never reads padded positions either. Not applied (and not exact) for media batches (rope positions and the visual-token scatter read the mask), left padding or masks with holes. Real positions can still differ in the last bits (a different SDPA kernel for `is_causal`), hence the parity gate. |
| `CLEF_FAST_HEAD=1` | A batch-vectorised, sync-free joint head (`fast_joint_head` in `models/clef_release.py`): index tables are built on the CPU from the encodings and sent in one transfer, span means are segment sums, ragged question and option sets are padded grids with key-padding masks, logits come back in one device-to-host copy. It reads the release head's weights; the model directory is untouched. The segment-mean formulation follows vLLM's Apache-2.0 ClefFlash head (vllm-project/vllm#61308); the code is an independent rewrite. | Same maths as the release head, different loop structure. Tests compare it with the release head on random tensors: max difference below 1e-5 in fp32 (and about one bf16 ulp in bf16). |

Measured (fixed-length records): the two flags +2-4% req/s alone, +5-12% with TunableOp, and a lower p95 under load
(c=8 p95 776 vs 1186 ms with TunableOp). Full table: [hardware results](hardware-results.md).

### TunableOp (ROCm)

`CLEF_TUNABLEOP=1` makes PyTorch autotune each GEMM shape once and remember the winner. The tuned kernels are
written to `<state_dir>/tunableop/<model>@<rev8>/tunableop_0.csv` and reloaded on the next start.

TunableOp picks, for every GEMM shape clef runs, the fastest kernel this GPU, driver, ROCm and PyTorch offer
(on the 7900 XTX mostly rocBLAS solutions that rocBLAS's own default does not pick: ~40% faster GEMMs, +27-32%
req/s on mixed-length traffic). It does not change the model. The results are tied to those versions: torch checks
them and tunes again after an upgrade.

1. `CLEF_TUNABLEOP=1 clef serve`. The server loads the tuned file, runs the normal short warmup and is **ready in
   seconds**, with the default kernels for anything not tuned yet. A live request never tunes.
2. In the background it records the GEMMs it produces (it runs every text-batch shape up to 1024 tokens, each
   padded length at every batch size, once it has been quiet for `CLEF_TUNABLEOP_IDLE_S`, 2 s) and starts a
   separate **tuner process** (`python -m clef_server.tuner`, log in `tuner.log` next to the tuned file). The tuner
   is paused (SIGSTOP) the moment a request arrives and resumed once it is quiet again, so a request never waits
   for it; the server reloads the tuned file whenever it changes, so each tuned GEMM speeds up the next forward.
   The tuner writes `<state_dir>/tunableop/<model>@<rev8>/tunableop_0.csv` (it owns the file) and lists the GEMMs
   it cannot tune in `tunableop_skip.csv`. All of it takes about an hour of idle time on the 7900 XTX (~620 GEMMs:
   rocBLAS has many candidate kernels per GEMM, some take 30 s); progress is in `/v1/stats` (`tunableop`: phase
   discovering / tuning / done, tuned, pending, tuner running / paused). Later starts only read the file: keep the
   state dir (a Docker volume) and this happens once per GPU / driver / ROCm / PyTorch version.
3. GEMMs that live traffic ran untuned (a long text, a record with images) are recorded in
   `tunableop_untuned0.csv` and tuned the same way; `clef stop && clef tune` does it offline.
4. Delete the directory to re-tune from scratch.

clef sets a small budget per candidate kernel (`PYTORCH_TUNABLEOP_MAX_TUNING_DURATION_MS=5`,
`PYTORCH_TUNABLEOP_MAX_TUNING_ITERATIONS=20`); the time is mostly a fixed cost per candidate, so a smaller budget
barely helps and torch's own 30 ms / 100 picks the same kernels. `CLEF_TUNABLEOP_ONLINE=1` keeps the old
behaviour (every new shape is tuned on its first request, a latency spike of seconds).

### The GPU check

```bash
PY=~/venvs/clef/bin/python bash bench/perf_check.sh        # QUICK=1 for a ~15 min smoke
```

It runs the batch-8 profile (is batched serving GEMM-bound?), engine parity for each flag against the flags-off
run (PASS = 0 top-1 flips and max |dp| <= 2e-3), `clef bench` at c=1/8/16 for each flag and both, and TunableOp
on vs off (the baseline sets every flag to 0, so the defaults do not leak in). Output is in `bench/out/perf/`
(`summary.md` has the PASS/FAIL and speed tables). Run it on another GPU before turning the flags on there, and copy
the numbers into [hardware results](hardware-results.md). On the 7900 XTX every variant "fails" the strict 2e-3 gate
with 0-3 coin-flip changes; the drift was accepted for the speed.

## Job throughput: length sort and tokenize ahead (measured: both stay off)

Batch jobs feed the engine in micro-batches of `CLEF_MAX_MICROBATCH` (default 8) in input order, so one forward
can mix a 60-token and a 900-token record and pays the padding of the longest. Two opt-in flags change how a
job is fed. Both are **off by default**: on the RX 7900 XTX the window gave +13-14% job throughput but a worse
interactive p95 (926 vs ~750 ms), and tokenize-ahead gave nothing (+0.3%).
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

### The GPU run

```bash
PY=~/venvs/clef/bin/python bash bench/sort_check.sh    # QUICK=1 for a ~5 min smoke, SWEEP=1 adds 256/1024
```

It starts a fresh server per variant (base, sort with window 512, tokenize ahead, both; TunableOp and the speed flags
off so only the feeding changes) with the result cache off, submits one classify job of 2000 long-tailed-length
inputs three times after a warm-up job, probes interactive latency every 2 s, and prints items/s, gain over base,
label agreement and a verdict (`bench/out/sort/summary.md`). Flip a default only if the median gain is over 10% and
above the base spread, labels agree and interactive p95 did not get worse.

| Variant | Items/s | Gain vs base | Label agreement | Interactive p95 | Verdict |
| --- | --- | --- | --- | --- | --- |
| base (flags off) | 9.5-9.7 | - | - | 734-805 ms | - |
| `CLEF_JOB_SORT_WINDOW=512` | 10.7-11.0 | +13-14% | 100% | 786-926 ms | off (p95) |
| `CLEF_TOKENIZE_AHEAD=1` | 9.5 | +0.3% | 100% | 789 ms | off |
| both | 10.6 | +12% | 100% | 1005 ms | off (p95) |

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
| `bench/vllm/vllm_check.sh` | vLLM vs clef (measured on the 7900 XTX, see below; re-run on other GPUs) |

## vLLM benchmark

vLLM serves clef-flash natively through its own `/v1/systemone` (vllm-project/vllm#61308, merged 2026-10-13, nightly
images only; not in the v0.32.0 release). `bench/vllm/vllm_check.sh` runs clef (flags off, the two flags, TunableOp)
and a pinned `vllm/vllm-openai-rocm` nightly container on the same records at concurrency 1, 8, 16 and 32, checks
parity against clef and prints a decision table. The proposed rule: parity, at least 1.3x req/s at concurrency 16,
and p95 no worse.

On the RX 7900 XTX (2026-10-18) vLLM reached 12.2 req/s at concurrency 8-32 against 10.9-11.1 for tuned clef
(+10%), but single requests were slower (p95 157 against 108 ms) and its probabilities differ more (max |dp|
0.034). **Not adopted on this card.** The check stays in the repo because another GPU (CUDA, MI300) may behave
differently: run it there before deciding. On ROCm under WSL the container needs `ROCPROFILER_REGISTER_ENABLED=0`
(set by the script) and, with snap Docker, the WSL files copied under `$HOME` (`DXCORE`, `ROCDXG`, `DIDS`);
`TUN_STATE` reuses an already tuned clef state dir for the TunableOp line. Details and sources:
`bench/vllm/README.md`; full table: [hardware results](hardware-results.md).
