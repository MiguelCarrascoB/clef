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

The only local GPU is AMD, and bitsandbytes quantization is CUDA only, so these were not measured.
