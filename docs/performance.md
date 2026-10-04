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
| `bench/parity.py` | probability parity across dtypes and quantization |
| `bench/memory_bench.py` | one smaller-memory setting per process: device and host memory, latency, parity against a saved bf16 reference |
