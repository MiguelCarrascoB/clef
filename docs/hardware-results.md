# Hardware results

Measured numbers per platform. "pending" means nobody has run it on that hardware yet; run
[hardware-validation.md](hardware-validation.md) and paste the results here. Nothing in the pending rows is an
estimate.

Columns: single = one request at a time (HTTP, ~330-token records); req/s at concurrency 1 and 16 (micro-batching on
the server); parity = max absolute probability difference against the bf16 reference on the fixed eval set
(`python bench/parity.py`).

| Platform | Device | Backend | dtype / quant | Single p50 / p95 | req/s c=1 | req/s c=16 | Peak memory | Load time | Parity vs bf16 | Run |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Windows 11 + WSL2 | AMD RX 7900 XTX (24 GB) | rocm 7.2 | bf16 | 140.7 / 147.3 ms | 7.0 | 9.0 | ~19.5 GB | ~45-100 s (page cache) | reference | v2, 2026-10-03 |
| Windows 11 + WSL2 | AMD RX 7900 XTX (24 GB) | rocm 7.2 | bf16 | 150.1 / 155.4 ms | 6.6 | 8.1 | ~19.5 GB | 66-79 s | reference | v2 A/B, 2026-10-03 |
| Windows 11 + WSL2 | AMD RX 7900 XTX (24 GB) | rocm 7.2 | bf16 | 149.0 / 159.2 ms (149.3 / 153.6) | 6.6 / 6.7 | 8.8 / 8.9 | ~19.5 GB | 76-110 s | reference | v3, 2026-10-03 (two runs) |
| Windows 11 + WSL2 | NVIDIA RTX 4070 Ti (12 GB) | cuda 12.8 | int8 (torchao) + `OFFLOAD=cpu` | 160.1 / 164.3 ms | 6.2 | 8.7 | 8.1 GB allocated, 11.3 GB device used (1.2 GB is the desktop) | 12.8 s + 9.5 s warmup | mean 0.0057, max 0.066, 3 of 399 flips (vs lossless offload) | v3.1.0, 2026-10-12 |
| Windows 11 + WSL2 | NVIDIA RTX 4070 Ti (12 GB) | cuda 12.8 | int8 (bitsandbytes, the CUDA default) + `OFFLOAD=cpu` | 228.2 / 259.4 ms | 4.4 | 9.4 | 7.9 GB allocated, 12.0 GB device used | 26-38 s + 10 s warmup | mean 0.0172, max 0.209, 22 of 399 flips (vs lossless offload) | v3.1.0, 2026-10-12 |
| Ubuntu | NVIDIA 24 GB class | cuda | bf16 | pending (run docs/hardware-validation.md) | pending | pending | pending | pending | pending | - |
| Ubuntu | NVIDIA 16 GB class | cuda | int8 | pending | pending | pending | pending | pending | pending | - |
| Ubuntu | NVIDIA 16 GB class | cuda | nf4 | pending | pending | pending | pending | pending | pending | - |
| Windows + WSL2 | NVIDIA | cuda | bf16 | pending | pending | pending | pending | pending | pending | - |
| macOS | Apple Silicon, >= 32 GB | mps | bf16 | pending | pending | pending | pending | pending | pending | - |
| any | CPU, ~40 GB RAM | cpu | bf16 or fp32 | pending | pending | pending | pending | pending | pending | - |

## v2 vs v3 on ROCm (same conditions)

The first row is the v2 baseline from the morning. In the afternoon both versions measured ~6% slower at c=1, so the
v2/v3 comparison was re-run back to back: v2 code (commit 5910c31) and v3 each served from a fresh start, then
`clef bench --concurrency 1,4,8,16 --requests 100`. v3 also ran 300-request c=1 runs: p50 148.0-150.3 ms, the same
as v2's 149.1-149.3 ms in the same window.

| conc | v2 req/s | v3 req/s (run 1 / run 2) | v2 p50 ms | v3 p50 ms |
| --- | --- | --- | --- | --- |
| 1 | 6.6 | 6.6 / 6.7 | 150.1 | 149.0 / 149.3 |
| 4 | 7.2 | 7.0 / 7.0 | 547.9 | 566.3 / 569.3 |
| 8 | 8.3 | 8.2 / 8.1 | 945.2 | 956.5 / 965.7 |
| 16 | 8.1 | 8.8 / 8.9 | 1766.5 | 1746.5 / 1754.3 |

Every level is within the 5% regression budget (worst: c=4, -2.8% throughput, +3.9% p50). The v2 c=16 run had a
noisy tail (p95 2873 ms).

## Notes on the ROCm v2 run

- 100 requests per level; avg server batch 1.0 at c=1 and ~7.8 at c=16. Concurrency 8: 8.2 req/s.
- Compute-bound: GEMMs ~83% of GPU kernel time at ~95 TFLOPS. Batching adds ~30% throughput, not more.
- About half of a single forward is host overhead; HIP graph capture is blocked by a device sync in transformers'
  `masking_utils._ignore_causal_mask_sdpa`.
- The joint head is ~5% of the forward. Padding to a multiple of 64 beat raw lengths and full buckets
  (`CLEF_PAD_MULTIPLE`); re-measure on other backends before assuming the same.
- Warm restart: load ~45-100 s, warmup ~12 s with a warm Triton cache (~110 s the very first time).

## Quantization accuracy (CUDA only)

Measured on the RTX 4070 Ti (12 GB) against the **lossless offload run** (`OFFLOAD=cpu CLEF_MAX_DEVICE_MEMORY_GB=10`),
not against plain bf16, because bf16 does not fit in 12 GB. Offload is documented to give identical probabilities to
bf16 (0 / 0 on ROCm); that was not re-checked on CUDA.

| Mode | Max abs prob diff | Mean abs diff | Argmax agreement | Latency vs reference |
| --- | --- | --- | --- | --- |
| int8, torchao (`CLEF_QUANT_BACKEND=torchao`) | 0.066 | 0.0057 | 396 / 399 (99.2%) | p50 170 ms vs 572 ms |
| int8, bitsandbytes (CUDA default) | 0.209 | 0.0172 | 377 / 399 (94.5%) | p50 232 ms vs 572 ms |
| nf4 (bitsandbytes) | 0.444 | 0.0333 | 363 / 399 (91.0%) | p50 153 ms vs 572 ms |

The numbers match the ROCm run of the same two backends (torchao 0.0058 / 2 flips, bitsandbytes 0.0175 / 22 flips).
`nf4` also matches ROCm (0.0329 / 35 flips) and stays not recommended. Plain bf16 needs a 24 GB card and was not run.

## Smaller-memory options on ROCm (offload, int8, nf4), 2026-10-03

Setup: RX 7900 XTX 24 GB, ROCm 7.2, torch 2.11.0+rocm7.2, WSL2 (25 GB RAM), 2026-10-03, one configuration per process,
bf16 measured first and last in the same session (152 / 154 ms p50, 4.72 / 4.67 req/s: 1% drift). Workload: 200
records (choice 2-6 options, score 3-5 levels, noul; 1-3 questions each; states of ~15-1500 tokens; 10 with an
image). "single" = 100 text records one at a time through `Engine.decide`; "c=8" = 160 requests, 8 in flight, through
the engine's micro-batching. Parity = every answer option's probability against the bf16 run of the same session,
one record per forward (399 questions).

| Setting | Device GB, peak allocated / reserved | Host RAM after load | single p50 / p95 | req/s c=8 | Max / mean abs prob diff | Top-1 flips |
| --- | --- | --- | --- | --- | --- | --- |
| bf16 (default) | 18.4 / 19.4 | 1.5 GB | 152 / 560 ms | 4.72 | reference | - |
| `OFFLOAD=cpu` cap 16 | 13.8 / 15.6 | 5.1 GB | 169 / 600 ms | 4.36 | 0 / 0 | 0 / 399 |
| `OFFLOAD=cpu` cap 15 | 13.1 / 15.0 | 6.2 GB | 175 / 597 ms | 4.15 | 0 / 0 | 0 / 399 |
| `OFFLOAD=cpu` cap 12 | 9.8 / 12.0 | 9.9 GB | 327 / 648 ms | 3.47 | 0 / 0 | 0 / 399 |
| `OFFLOAD=cpu` cap 11 | 9.0 / 11.0 | 11.1 GB | 343 / 634 ms | 3.37 | 0 / 0 | 0 / 399 |
| `OFFLOAD=cpu` cap 8 | 5.8 / 8.0 | 14.3 GB | 498 / 667 ms | 2.80 | 0 / 0 | 0 / 399 |
| `OFFLOAD=cpu` cap 7 | 5.1 / 7.0 | 15.0 GB | 540 / 671 ms | 2.59 | 0 / 0 | 0 / 399 |
| `OFFLOAD=cpu` cap 6 | crashed (segfault while loading; 28 layers pinned on WSL2) | | | | | |
| `QUANT=int8` (torchao) | 12.0 / 12.9 | 1.5 GB | 198 / 595 ms | 4.23 | 0.062 / 0.0058 | 2 / 399 |
| `QUANT=int8 OFFLOAD=cpu` (embeddings on host) | 8.3 / 9.2 | 2.5 GB | 206 / 611 ms | 4.10 | 0.062 / 0.0058 | 2 / 399 |
| `QUANT=int8 QUANT_BACKEND=bnb` (bitsandbytes) | 11.6 / 12.7 | 2.5 GB | 420 / 680 ms | 3.01 | 0.262 / 0.0175 | 22 / 399 |
| `QUANT=nf4` (bitsandbytes) | 8.5 / 9.2 | 1.5 GB | 249 / 637 ms | 3.66 | 0.449 / 0.0329 | 35 / 399 |
| *fp16 (noise floor, separate run, 200 text records)* | *18.2 / 18.4* | | *159 ms (bf16 that run: 176)* | | *0.021 / 0.0019* | *0 / 399* |

- The cap is enforced on the PyTorch allocator. The whole process (reserved memory + driver context) measured about
  0.65 GB above the cap, so use `N - 0.5` GB for an `N` GB card: 15 for 16 GB, 11 for 12 GB, 7 for 8 GB.
- Host RAM after load is the process RSS once the model is in place: the pinned copies of everything that was moved
  off the GPU. During loading RSS peaks higher (up to ~19 GB) because the safetensors files are memory-mapped; those
  pages are reclaimable page cache.
- p95 is dominated by the longest records of the eval set (~1500 tokens), which are compute-bound and unaffected by
  the settings; p50 is the number that moves.
- Offload latency is the PCIe copy of the streamed layers (about 0.43 GB per layer at ~26 GB/s) partly hidden behind
  compute; batching amortises it (req/s at c=8 falls less than single latency rises).
- int8 flips: 2 of 399 questions changed their top option, against 0 for fp16 vs bf16. The mean
  probability error 0.006 is ~3x the fp16/bf16 noise floor.

Background: [docs/memory.md](memory.md). Reproduce with `bench/memory_bench.py`. CUDA and Apple Silicon rows are still pending hardware.

## RTX 4070 Ti (12 GB) on WSL2, CUDA 12.8, 2026-10-12

Setup: i5-13600K, 32 GB host (WSL2 limited to 21 GB; 15 GB is not enough for the lossless modes), RTX 4070 Ti 12 GB
shared with the Windows desktop (1.2 GB in use before the model loads), torch 2.11.0+cu128, `requirements/cuda.txt`
plus the two dependencies added in this change, clef 3.1.0. Same workload and method as the ROCm section above
(`bench/memory_bench.py`, one configuration per process). The parity reference is the lossless offload run, not bf16.

| Setting | Device GB, peak allocated / reserved | Peak device used | Host RSS after load / peak | single p50 / p95 | req/s c=8 | Max / mean abs prob diff | Top-1 flips |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `OFFLOAD=cpu` cap 10 (reference) | 7.7 / 10.0 | 11.4 GB | 12.3 / 17.1 GB | 572 / 631 ms | 2.67 | reference | - |
| `QUANT=int8 OFFLOAD=cpu` (torchao) | 8.1 / 9.8 | 11.3 GB | 2.9 / 15.5 GB | 170 / 489 ms | 5.03 | 0.066 / 0.0057 | 3 / 399 |
| `QUANT=int8 OFFLOAD=cpu` (bitsandbytes) | 7.9 / 10.9 | 12.0 GB | 5.8 / 19.4 GB | 232 / 446 ms | 4.81 | 0.209 / 0.0172 | 22 / 399 |
| `QUANT=nf4 OFFLOAD=cpu` (bitsandbytes) | 4.4 / 7.6 | 8.9 GB | 5.8 / 19.3 GB | 153 / 445 ms | 5.50 | 0.444 / 0.0333 | 36 / 399 |

HTTP, `clef bench --requests 100` (500 requests per mode, no errors):

| Mode | c=1 req/s, p50 / p95 | c=4 req/s, p50 | c=8 req/s, p50 | c=16 req/s, p50 / p95 |
| --- | --- | --- | --- | --- |
| int8 torchao + offload | 6.2, 160 / 164 ms | 8.7, 458 ms | 8.7, 814 ms | 8.7, 1830 / 2084 ms |
| int8 bitsandbytes + offload | 4.4, 228 / 259 ms | 8.8, 428 ms | 8.4, 968 ms | 9.4, 1674 / 1897 ms |

- `pytest -m gpu tests/integration`: 22 passed with torchao. With bitsandbytes, 21 passed and
  `test_batch_matches_single` failed (`noul` 0.7469 batched against 0.7931 single, tolerance 0.02): bitsandbytes int8
  gives different results for the same record in a batch.
- On CUDA, bitsandbytes is the `auto` choice for int8, and it is the worse one here: 3x the error, 7x the flipped
  answers and 36% slower than torchao. See the note in [memory.md](memory.md).
- The lossless modes need more host RAM than a default 15 GB WSL2: `OFFLOAD=cpu` was refused at caps of 10 and 11 GB
  ("11.0 GB of weights stay in host memory, 13.0 GB free"), and the cap 10 run peaked at 17.1 GB of RSS while loading.
- `causal-conv1d` 1.7.0, built from source (the `cuda-fast` extra, ~23 min): `clef doctor` no longer warns about the fast
  path. `memory_bench` int8 torchao + offload went from p50 170.3 / p95 488.7 ms and 5.03 req/s to 167.5 / 481.2 ms and
  5.11 req/s (about 1.6%, inside the ~5% same-day drift), parity unchanged (mean 0.0054, 2 of 399 flips), and the
  integration tests still pass (22). `clef bench`: c=1 6.2 req/s, p50 161 ms; c=16 8.9 req/s.
- Not run: plain bf16 (needs 24 GB) and native Ubuntu.

### `CLEF_PAD_MULTIPLE` on CUDA (RTX 4070 Ti, int8 torchao + offload, 2026-10-12)

The default of 64 was chosen on ROCm. `clef bench --requests 100 --concurrency 1,8`, ~330-token records, a fresh server
per run; the default (64) was measured twice and a multiple of 8 three times.

| `CLEF_PAD_MULTIPLE` | c=1 req/s, p50 | c=8 req/s, p50 |
| --- | --- | --- |
| 8 | 6.0, 165.5 / 166.5 / 166.1 ms | 9.5, 880 / 866 / 869 ms |
| 64 (default) | 5.9, 169.3 / 168.9 ms | 8.5, 942 / 949 ms |
| 128 | 5.9, 169.6 ms | 8.5, 956 ms |

A multiple of 8 was about 12% faster at concurrency 8 and 2% faster at concurrency 1, consistently across the three runs.
Only one mode and one record length were measured; the bf16 and mixed-length workloads are untested, so the default is unchanged.


## Speed flags and TunableOp on ROCm (RX 7900 XTX, 2026-10-18)

`bench/perf_check.sh`, one session, result cache off, fixed-length records (`clef bench`). Parity is against the
flags-off engine (200 records, 10 with images; the flags-off engine run twice differs by 0.0, so the 2e-3 gate is
the floor). Every top-1 change was a near coin-flip (reference top-2 margin 0.0015-0.017, the lowest 1.5% of the
399 questions).

| Variant | Parity: max \|dp\| / top-1 flips | c=1 p50 / p95 ms | c=1 req/s | c=8 req/s | c=16 req/s |
| --- | --- | --- | --- | --- | --- |
| flags off (base) | reference | 133.8 / 683.7 | 4.5 | 8.5 | 9.1 |
| `CLEF_MASK_FREE=1` | 0.0235 / 1 | 137.5 / 610.0 | 4.5 | 8.6 | 9.0 |
| `CLEF_FAST_HEAD=1` | 0.0038 / 0 | 133.3 / 588.8 | 4.6 | 8.6 | 9.5 |
| both | 0.0235 / 1 | 133.8 / 585.6 | 4.6 | 8.7 | 9.5 |
| `CLEF_TUNABLEOP=1` (tuned) | 0.0219 / 3 | 104.2 / 502.5 | 5.7 | 10.1 | 10.8 |
| both + TunableOp (**the ROCm default**) | 0.0273 / 0 | 98.6 / 452.3 | 6.0 | 11.3 | 11.8 |

Step 0 (`profile_forward.py --batch 8`): single record kernel/wall 0.22 (host overhead dominates), batch 8 0.68;
GEMM 81-84% of kernel time. Accuracy on the 47 labelled tickets: 0.872 with TunableOp off and on.

Mixed-length traffic (`clef bench --mixed`), TunableOp off vs on: c=1 5.9 vs 7.6 req/s (p50/p95/max 151/241/249 vs
111/199/204 ms), c=8 6.0 vs 7.9, c=16 6.2 vs 7.9 (+27-32%).

**TunableOp cost and design.** On gfx1100 PyTorch already uses rocBLAS; TunableOp's gain is choosing better rocBLAS
solutions (1054 of 1181 tuned GEMMs are rocBLAS, 66 hipBLASLt; tuned GEMMs ~40% faster than rocBLAS's own pick).
Tuning costs a fixed ~4-8 s per GEMM whatever the per-candidate budget (30 ms / 100 runs, 5 ms / 20 and 1 ms / 3
pick equally fast kernels), so the ~535 GEMMs of the serving grid take about an hour. Tuning on the request instead
(the old behaviour) gave single requests p50 2.7 s, p95 39 s until every shape had been seen. clef now tunes in the
background while idle (a step only after 2 s without requests), never on a live request.

## vLLM vs clef (RX 7900 XTX, 2026-10-18)

`bench/vllm/vllm_check.sh`, vLLM nightly image `vllm/vllm-openai-rocm:nightly-39983a5e…` (clef-flash natively,
`max-num-seqs 8`), 100 requests per level, same session. In-container ROCm on WSL needed
`ROCPROFILER_REGISTER_ENABLED=0` (the image's torch links rocprofiler-sdk, which is unsupported on WSL).

| Line | c=1 p50 / p95 ms, req/s | c=8 req/s | c=16 p50 / p95 ms, req/s | c=32 req/s |
| --- | --- | --- | --- | --- |
| clef, flags off | 137 / 145, 7.2 | 8.8 | 1748 / 1797, 9.0 | 9.0 |
| clef, `MASK_FREE` + `FAST_HEAD` | 134 / 138, 7.4 | 8.8 | 1663 / 1684, 9.6 | 9.5 |
| clef, TunableOp | 104 / 108, 9.5 | 10.9 | 1434 / 1443, 11.1 | 10.8 |
| vLLM | 136 / 157, 7.3 | 12.2 | 1304 / 1312, 12.2 | 12.1 |

Parity vLLM vs clef flags off: max |dp| 0.034, 1 flip (near tie). Decision on this card: **not adopted** (1.10x at
c=16 against a 1.3x bar, 1.45x the p95 at c=1). Re-run the script on other GPUs before deciding there.

## Job feeding (length sort, tokenize ahead) and sharing the GPU (RX 7900 XTX, 2026-10-18)

`bench/sort_check.sh` (2000 long-tailed items x 3 for base and sort; quick run 600 x 2 for all four):

| Variant | Items/s | Gain | Labels | Interactive p95 |
| --- | --- | --- | --- | --- |
| base | 9.5-9.7 | - | - | 734-805 ms |
| `CLEF_JOB_SORT_WINDOW=512` | 10.7-11.0 | +13-14% | 100% | 786 (quick) / 926 (full) ms |
| `CLEF_TOKENIZE_AHEAD=1` | 9.5 | +0.3% | 100% | 789 ms |
| both | 10.6 | +12% | 100% | 1005 ms |

Both stay **off**: the window's gain costs interactive latency. Interactive latency during a 3000-item job
(unique requests, cache off): idle p50 140 ms; one job 622 ms (before the priority lane) vs 627 ms (with it);
2 keys x 2 jobs 624 ms, about one batch-8 forward.

