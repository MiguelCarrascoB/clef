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

## Benchmarks

| Tool | Use |
| --- | --- |
| `clef bench` | HTTP load test against the running server (realistic) |
| `bench/bench.py` | in-process, needs the GPU |
| `bench/profile_forward.py` | `torch.profiler` |
| `bench/parity.py` | probability parity across dtypes and quantization |
