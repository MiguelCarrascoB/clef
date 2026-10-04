# Running clef-flash in less memory

The release is ~19 GB in bf16 and a default `clef serve` peaks at ~19.5 GB of device memory, so out of the box the
floor is a 24 GB GPU or a 32 GB Mac. Three settings lower that floor. Everything below was measured on one machine
(RX 7900 XTX, ROCm 7.2, Windows 11 + WSL2, 2026-10-03); see [Measured results](#measured-results) for the numbers and
[Caveats](#caveats) for what was not tested.

| You have | Use | Cost (measured, 7900 XTX) |
| --- | --- | --- |
| 24 GB GPU, 32 GB Mac | nothing | none |
| 16 GB GPU | `CLEF_OFFLOAD=cpu CLEF_MAX_DEVICE_MEMORY_GB=15` | lossless (identical probabilities); p50 +15%, throughput -12%; ~6 GB host RAM |
| 12 GB GPU | `CLEF_QUANT=int8 CLEF_OFFLOAD=cpu` | mean prob error 0.006, 2 of 399 top choices flipped; p50 +36%, throughput -13%; ~2.5 GB host RAM |
| 12 GB GPU, lossless | `CLEF_OFFLOAD=cpu CLEF_MAX_DEVICE_MEMORY_GB=11` | identical probabilities; p50 2.3x, throughput -29%; ~11 GB host RAM |
| 8 GB GPU | `CLEF_OFFLOAD=cpu CLEF_MAX_DEVICE_MEMORY_GB=7` | identical probabilities; p50 3.6x, throughput -45%; ~15 GB host RAM (a cap of 6 crashed) |
| 24 GB Mac | `CLEF_QUANT=int8` | ~12 GB; **not validated on Apple Silicon** |

`clef doctor` reads the detected memory and prints the matching line ("memory setting").

## The three settings

### `CLEF_OFFLOAD=cpu` and `CLEF_MAX_DEVICE_MEMORY_GB` (lossless, CUDA and ROCm)

Keeps part of the model in host RAM and streams it to the GPU per forward. Nothing is approximated: the weights are
the same bf16 tensors, so the probabilities are bit-identical to a normal run (measured: max |dp| 0.0 on 200 records).

1. The token embedding and the output embedding (~4 GB) never leave the host. The model only gathers rows from them
   (a few thousand tokens per forward), so the lookup runs on the CPU and the result is copied over. This alone
   frees 4 GB at a cost of microseconds.
2. Decoder layers that do not fit under the budget stay in pinned host memory and are copied to the GPU just before
   they run, on a side stream, up to two layers ahead of the compute, then dropped. Streamed layers are spread
   evenly through the stack so resident layers hide the transfer (measured copy bandwidth ~26 GB/s on the 7900 XTX,
   and copies overlap compute).
3. The vision tower, the joint head, the final norm and rotary tables always stay on the GPU.

`CLEF_MAX_DEVICE_MEMORY_GB` is the memory the *process* may allocate on the device (weights + activations). The
server plans the split from it (2.5 GB is kept for activations), applies it as a hard allocator limit, and fails the
load with a one-line error if even streaming every layer does not fit. `0` (default) means "whatever is free". The
driver context and the desktop take about 1 GB on top, so on an `N` GB card use `N - 1`.

Only the layers you cannot fit are streamed, so the cost grows smoothly with how much you cut. Host RAM needed is
the size of what was moved off the GPU (listed in the table) plus a couple of GB. `CLEF_PREFLIGHT=1` checks this.

macOS and CPU: device and host share one memory pool, so offload does nothing there (a warning is logged).

### Warnings you may see

Settings that cannot take full effect no longer fail silently. They are logged at WARNING and listed in
`/health` `warnings` (and by `clef doctor`):

- `CLEF_MAX_DEVICE_MEMORY_GB` is not enforced (macOS / CPU share one pool, or the allocator refused the limit), or is
  above the free memory (the layer plan then uses the free memory), or does not shape the plan because `CLEF_QUANT`
  keeps the quantized layers on the device.
- Pinned host memory is unavailable: streamed layers fall back to pageable copies, which is much slower.
- `CLEF_OFFLOAD=cpu` with `CLEF_QUANT`: the direct host placement was refused by transformers and the embeddings were
  moved after the load. Only that specific refusal falls back; a corrupt checkpoint, host out-of-memory or any other
  error fails the load with its own message.
- `CLEF_QUANT=nf4` on ROCm (lossy) and `CLEF_QUANT=int8` on MPS (not validated): doctor shows WARN, not OK.
- `CLEF_OFFLOAD=cpu` with a cap below 7 GB: a 6 GB cap segfaulted at load on WSL2, so the preflight (and doctor)
  refuses it there and warns elsewhere. `CLEF_PREFLIGHT=0` skips the check.

### `CLEF_QUANT=int8` (weight-only, all backends)

| Backend | Library | Notes |
| --- | --- | --- |
| CUDA | bitsandbytes (unchanged) | `CLEF_QUANT_BACKEND=torchao` selects the torchao path instead (untested on CUDA) |
| ROCm | **torchao** | measured: 2x faster and 3x closer to bf16 than bitsandbytes int8 on the same GPU |
| MPS, CPU | torchao | loads through the same code; **not validated on hardware** |

Linear layers of the language model (~13 GB -> ~6.5 GB) become int8 with per-channel scales (weight-only: activations stay bf16). The
embeddings, the output embedding and the vision tower stay bf16. Needs `pip install torchao` (it is part of the
`rocm`, `mps` and `cpu` extras; `pip install "clef-local[quant]"` adds it elsewhere). Combine with
`CLEF_OFFLOAD=cpu` to also move the 4 GB of embeddings to the host: that is the 12 GB setting.

`CLEF_QUANT=nf4` (bitsandbytes 4-bit) loads on ROCm too, but is **not recommended**: it flipped the top choice on
35 of 399 questions (~9%) in our set. Use it only if nothing else fits.

### Why not 4-bit, GGUF, AWQ, GPTQ

- **torchao int4**: needs the `mslk` kernel package (CUDA); it failed to load on ROCm. Not offered. The unpacked
  int4 variants store int8 and save nothing.
- **bitsandbytes nf4**: works on ROCm, but the accuracy cost is large (table). Offered only as `nf4`.
- **GGUF / llama.cpp**: the quantized backbone would have to be driven as an encoder whose hidden states feed the
  joint schema head (cross-attention over every token, per-span means, option-embedding lookups), which GGUF has no
  notion of. The head would have to be a separate PyTorch module fed with llama.cpp embeddings, the vision tower
  needs the separate mmproj path, and batching / padding semantics change. Also depends on llama.cpp's support for
  this Qwen3.5 Gated DeltaNet hybrid, which we did not verify. Not feasible cheaply; no parity could be measured.
- **AWQ / GPTQ**: need a calibration run with a task-appropriate objective (the logits come from the joint head, not
  from next-token prediction, so stock calibration optimises the wrong thing), kernels that are mostly CUDA-only
  (Marlin/ExLlama; weak ROCm and none for MPS), and support for the hybrid linear-attention layers. Weight-only int8
  already gets 12 GB with one-digit-percent probability error; the 4-bit tools would trade much more accuracy for a
  smaller gain. Not attempted.

## Measured results

RX 7900 XTX 24 GB, ROCm 7.2, torch 2.11.0+rocm7.2, WSL2 (25 GB RAM), 2026-10-03, one configuration per process,
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
  0.65 GB above the cap, so use `N - 1` GB for an `N` GB card: 15 for 16 GB, 11 for 12 GB, 7 for 8 GB.
- Host RAM after load is the process RSS once the model is in place: the pinned copies of everything that was moved
  off the GPU. During loading RSS peaks higher (up to ~19 GB) because the safetensors files are memory-mapped; those
  pages are reclaimable page cache.
- p95 is dominated by the longest records of the eval set (~1500 tokens), which are compute-bound and unaffected by
  the settings; p50 is the number that moves.
- Offload latency is the PCIe copy of the streamed layers (about 0.43 GB per layer at ~26 GB/s) partly hidden behind
  compute; batching amortises it (req/s at c=8 falls less than single latency rises).
- int8 flips: 2 of 399 questions changed their top option, against 0 for fp16 vs bf16. The mean
  probability error 0.006 is ~3x the fp16/bf16 noise floor.

## Caveats

- One machine, one day, one model. Same-day drift is ~5%; every row was measured back to back in one session
  against the bf16 reference of that session.
- CUDA and Apple Silicon paths were written to the same interfaces but could **not be run**: the bitsandbytes
  behaviour on CUDA is unchanged code, torchao on CUDA / MPS / CPU and the cap / pinned-memory code on CUDA are
  untested on hardware. Please run `bench/memory_bench.py` there and add the rows to `docs/hardware-results.md`.
- The cap limits what PyTorch allocates, not the driver context (~0.5-1 GB) or other processes.
- Offload needs host RAM (table). On WSL2, size the VM (`.wslconfig` `memory=`) accordingly.
- Offload and long inputs: activations for a very long record (thousands of tokens, batch 8) can exceed the 2.5 GB
  reserve; the request then fails with a 503 `GpuOutOfMemory` and the server stays up. Lower `CLEF_MAX_MICROBATCH` or raise the cap.
- Media requests are served the same way. The vision tower stays on the GPU (0.9 GB).
- The server's text/media parity above comes from `bench/memory_bench.py` (200 records: choice 2-6 options, score
  3-5 levels, noul; 1-3 questions each; states of ~15-1500 tokens; 10 image records), one record per forward.

## Reproduce

```bash
python bench/memory_bench.py --label bf16 --dtype bfloat16 --save-reference bench/out/ref_bf16.json
python bench/memory_bench.py --label off15 --offload cpu --max-device-gb 15 --reference bench/out/ref_bf16.json
python bench/memory_bench.py --label int8off --quant int8 --offload cpu --reference bench/out/ref_bf16.json
python bench/parity.py --quant int8 --dtype bfloat16     # same eval set, loads bf16 itself as the reference
```

One configuration per process (device memory is only meaningful that way). Stop any running clef server first.
