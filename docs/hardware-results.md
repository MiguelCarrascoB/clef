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

| Mode | Max abs prob diff vs bf16 | Mean abs diff | Argmax agreement | Latency vs bf16 |
| --- | --- | --- | --- | --- |
| int8 | pending hardware | pending | pending | pending |
| nf4 | pending hardware | pending | pending | pending |

These rows are still pending: the bitsandbytes-on-CUDA path was not measured (no NVIDIA GPU). On the AMD GPU,
int8 / nf4 (bitsandbytes and torchao) and CPU offload were measured; see the last section.

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
