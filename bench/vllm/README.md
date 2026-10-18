# vLLM vs clef benchmark (Wave G)

> **Result on the RX 7900 XTX (2026-10-18):** vLLM 12.2 req/s at concurrency 8-32 vs 10.9-11.1 for tuned clef
> (+10%), slower single requests (p95 157 vs 108 ms), max |dp| 0.034 → not adopted on this card; keep this check
> for other GPUs. On WSL the image needs `ROCPROFILER_REGISTER_ENABLED=0` (now in `DOCKER_GPU_FLAGS`; its torch links
> rocprofiler-sdk, unsupported on WSL) and snap Docker needs the WSL files under `$HOME` (`DXCORE`, `ROCDXG`, `DIDS`).
> Full table: [docs/hardware-results.md](../../docs/hardware-results.md).

Question (UPGRADES.md D12): does vLLM serve clef-flash enough faster than clef to justify an adapter? vLLM is adopted
only if this benchmark shows a real gain. Prepared on 2026-10-17 without a GPU; nothing here has been run yet.

## Tomorrow: one command (on `gpu`, in WSL)

```bash
ssh gpu; cd ~/src/clef && repo-sync && git checkout upg/g-vllm-bench
QUICK=1 bash bench/vllm/vllm_check.sh        # ~20 min smoke: 20 records, c=1,8, 20 requests. Run this first.
bash bench/vllm/vllm_check.sh                # full: 200 records, c=1,8,16,32, 100 requests per level (1-2 h cold)
```

Everything lands in `bench/out/vllm/`; the last thing printed is `summary.md`, which ends in the decision table. The script
refuses to start (exit 2) if a clef server or a container is up, a port is taken, the device already uses more than
`IDLE_MAX_GB` (4) of VRAM, or the VRAM cannot be read at all (`FORCE=1` overrides). A refused run touches nothing.
Once preflight passed, the run cleans up on any exit, but only what it started: the container id it wrote and the clef
servers of its own state dirs (`bench/out/vllm/run-<timestamp>-<pid>/`), never another run's. Exit 0 means the
verdict is ADOPT.

Before running, check no one else uses the card: `run-on gpu --ls` and `/opt/rocm-wsl/bin/amd-smi`.

## What the script does (the manual steps, if you want to do one by hand)

1. **Preflight.** `docker` present and usable; >= 45 GB free for the image under the Docker root and >= 25 GB in
   `$HF_HOME` for the model; `/dev/dxg`, `/usr/lib/wsl/lib/libdxcore.so`, `librocdxg.so` and `dids.conf` exist;
   `HSA_ENABLE_DXG_DETECTION=1` exported; device VRAM in use measured as the idle baseline.
2. **Image.** `docker pull vllm/vllm-openai-rocm:nightly-39983a5e7a5a575a7829f5d5ada50a33d9337ec1` (see "Image" below),
   then a GPU smoke test inside it (`torch.cuda.is_available()` and a bf16 matmul). If that fails, vLLM-on-WSL does not work here
   and the run reports only the clef lines (verdict INCONCLUSIVE; that is a result, see "If the GPU smoke test fails").
3. **clef parity reference.** `bench/parity.py --engine-dump` (flags off), the same eval set as the other parity checks.
4. **clef speed.** Two lines, each a fresh `clef serve --detach` + `bench/vllm/load.py`: flags off (baseline), and
   `CLEF_MASK_FREE=1 CLEF_FAST_HEAD=1`.
5. **vLLM.** Container serving `Cloudflare/clef-flash --revision <PINS["clef-flash"]>` (read from
   `src/clef_server/models/clef_release.py`), then `vllm_client.py dump` (the same records through `/v1/systemone`),
   `bench/parity.py --compare` against step 3, then `load.py` on the same levels. Request bodies are `clef bench`'s, so
   both servers get identical requests.
6. **Teardown** (`docker rm -f <this run's container id>`; the image stays, `docker rmi` it to get ~25 GB back) and `vllm_table.py` -> `summary.md`.

Load: concurrency 1 / 8 / 16 / 32 (override `LEVELS=`), p50 / p95 / p99, req/s, and device VRAM peak (`torch.cuda.mem_get_info`,
device-wide, minus the idle value from before the server started). `MIXED=1` uses mixed-length states.
VRAM caveat: vLLM reserves `GPU_MEM_UTIL` (0.90) of the card for its KV cache whatever it needs, so its peak says little
about need; clef's peak is real.

## Decision rule (PROPOSED: please confirm or change)

Adopt vLLM (build the adapter) only if all hold. Hard rules decide DO NOT ADOPT; a failing or missing soft rule gives
INCONCLUSIVE (a person reads the table).

| # | Rule | Kind |
|---|---|---|
| 1 | Parity: 0 top-1 flips against clef (near-ties with a reference top-2 margin < 1e-3 reported apart, as `parity.py` does) and max abs probability difference <= **2e-3**, singles and batched | hard |
| 2 | Speed: req/s at **c=16 >= 1.3x** the better of the two clef lines (flags off, flags on) | hard |
| 3 | Latency: p95 at c=16 <= 1.0x the better clef p95 (no worse), and p95 at c=1 <= 1.1x | soft |
| 4 | Stability: 0 failed or mis-shaped requests at any level | soft |
| 5 | Memory: >= 1 GB of the 24 GB left free at vLLM's peak | soft |

Why these numbers: 1.3x is above the ~5% same-day drift and the ~30% that batching already gives clef; the baseline is
the better clef line because the flags may be adopted anyway. For 2e-3: the PR's own parity run measured max |dp| 0.0015
eager and 0.00086 compiled (57/57 top-1), and the project gate in `perf_check.sh` is 2e-3, so a 1e-3 gate would likely fail
eager mode on noise. Alternative: gate on labels only (100% top-1) and report |dp|. Thresholds can be changed without code:
`MIN_SPEEDUP=1.5 LEVEL=8 MAX_P95_RATIO=1.2 TOL=1e-3 bash bench/vllm/vllm_check.sh`
or `python bench/vllm/vllm_table.py bench/out/vllm --min-speedup 1.5` on existing results.

## Verified facts and sources (checked 2026-10-17)

- **PR** [vllm-project/vllm#61308](https://github.com/vllm-project/vllm/pull/61308) "Add Clef-Flash support", merged
  2026-10-13 (merge commit `380638af`). Files: `vllm/model_executor/models/clef_flash.py`, `clef_joint_schema.py`,
  `vllm/entrypoints/generate/structured_decisions/strategies.py`, `docs/serving/online_serving/structured_decisions.md`.
- **Request/response format**: the docs file at the merge commit
  ([structured_decisions.md](https://github.com/vllm-project/vllm/blob/380638afbb2c5658f22124f6d1d38fcd464ea7d9/docs/serving/online_serving/structured_decisions.md)).
  Request `{model, state, questions: {id: {type, instructions, criteria}}, images?, videos?}`; response
  `{answers: {id: {type, choice|noul|score, probabilities, confidence}}, usage, diagnostics}`. choice probabilities are keyed
  by option name, noul has `noul` (P yes) plus `probabilities` keyed `yes`/`no`, score probabilities are keyed by level index.
  Clef-Flash: a top-level `instructions` returns 400 (a question's own `instructions` is fine), `chat_template_kwargs`
  has no effect, state truncated to 16,384 tokens, `usage.output_tokens` is 0. clef's own response (from the release's
  `systemone_answer`) has the same keys, except noul has only `noul`. `bench/vllm/vllm_client.py` maps both to the flat
  `question.option` form of `parity.py` and raises on any other shape.
- **Serve command** from the PR: `vllm serve Cloudflare/clef-flash --served-model-name clef-flash --dtype bfloat16
  --max-model-len 16384 --max-num-batched-tokens 512 --max-num-seqs 8` (+ `--enforce-eager` for the eager comparison).
- **Image**: Docker Hub [`vllm/vllm-openai-rocm`](https://hub.docker.com/r/vllm/vllm-openai-rocm), tag
  `nightly-39983a5e7a5a575a7829f5d5ada50a33d9337ec1`, pushed 2026-10-17 05:51 UTC (Docker Hub API). Its vllm commit `39983a5e` is
  306 commits ahead of the PR's merge commit (GitHub compare API), so it contains clef-flash. The official docs call
  `vllm/vllm-openai-rocm:nightly` the successor of the deprecated `rocm/vllm-dev` ([docs.vllm.ai deployment/docker](https://docs.vllm.ai/en/v0.20.0/deployment/docker/)).
  Base: `rocm/dev-ubuntu-22.04:7.2.3-complete`, PyTorch ROCm fork release/2.13, `PYTORCH_ROCM_ARCH` includes gfx1100
  ([Dockerfile.rocm_base](https://github.com/vllm-project/vllm/blob/main/docker/Dockerfile.rocm_base)); entrypoint `vllm serve`
  ([Dockerfile.rocm](https://github.com/vllm-project/vllm/blob/main/docker/Dockerfile.rocm)). The container brings its own
  glibc and torch, which is the way around gpu's glibc 2.35 / torch 2.11 blockers. ~12 GB compressed. The floating `nightly` tag
  moves daily; the pinned tag does not. If it was deleted, list again:
  `curl -s 'https://hub.docker.com/v2/repositories/vllm/vllm-openai-rocm/tags?page_size=20&ordering=last_updated'`
  and pick the newest `nightly-<40 hex>` (not `-rocm100`, `base-`), then check it is ahead of `380638af` with
  `gh api repos/vllm-project/vllm/compare/380638af...<sha> --jq .status` (`ahead`).
  `rocm/vllm-dev` also has dated `nightly_main_YYYYMMDD` tags but is deprecated; I did not pin one.
- **ROCm in Docker on WSL**: AMD documents it in the [librocdxg README](https://github.com/ROCm/librocdxg) ("Container Launch -
  WSL-Specific Flags"): `--device /dev/dxg`; mount `/usr/lib/wsl/lib/libdxcore.so` -> `/usr/lib/libdxcore.so`,
  `/opt/rocm/lib/librocdxg.so` -> `/usr/lib/librocdxg.so`, `/opt/rocm/share/rocdxg/dids.conf` -> `/usr/share/rocdxg/dids.conf`;
  `-e HSA_ENABLE_DXG_DETECTION=1` for ROCm < 7.13; plus `--cap-add=SYS_PTRACE --security-opt seccomp=unconfined --ipc=host
  --shm-size 8G`. librocdxg 1.2.0 matches ROCm 7.2.x; RX 7900 XTX is on AMD's [WSL support matrix](https://rocm.docs.amd.com/projects/radeon-ryzen/en/latest/docs/compatibility/compatibilityrad/wsl/wsl_compatibility.html)
  (ROCm 7.2.1 page). No `/dev/kfd` or `/dev/dri`. It needs a Docker engine inside the WSL distro (Docker Desktop's engine does not
  expose `/dev/dxg`, per the unsloth-rocm notes). The README's known issues list amd-smi as limited on WSL2, hence torch is
  used for VRAM here.
- **vLLM on WSL2**: pinned memory is off by default there and the V2 model runner fails with `RuntimeError: UVA is not
  available`. The fix is [PR #41496](https://github.com/vllm-project/vllm/pull/41496) (merged 2026-06-16), which adds
  `VLLM_WSL2_ENABLE_PIN_MEMORY=1`; the script sets it. Related report: [issue #43381](https://github.com/vllm-project/vllm/issues/43381).

## UNVERIFIED (could fail tomorrow)

- That the vLLM image's ROCm runtime (7.2.3, with its own pinned `ROCM_RUNTIME_COMMIT`) works with the host's librocdxg 1.2.0
  (7.2.4 on the host). AMD's example uses `rocm/pytorch`, not this image. The smoke test in step 2 exists for this.
- That `VLLM_WSL2_ENABLE_PIN_MEMORY` is honoured by the ROCm platform class (the PR touched the generic and CUDA paths; I did
  not read `platforms/rocm.py`). If vLLM dies with `UVA is not available`, read `vllm_server.log` and issue #43381.
- Whether `librocdxg.so` is installed on the host at `/opt/rocm/lib` (COMPUTE.md only says ROCm 7.2.4 works on WSL). The script
  also looks in `/opt/rocm-wsl` and falls back to `find`; set `ROCDXG=` and `DIDS=` if it cannot find them.
- The error body vLLM returns for a 400 and whether Clef-Flash adds `legend` to score answers (docs show no example). The
  client does not need `legend`; load.py prints the first error body per status.
- A single vLLM config is measured (`--max-num-seqs 8`, the PR's). At c=16/32 requests queue beyond 8 sequences; if vLLM is
  close, rerun with `VLLM_MAX_SEQS=32` (and consider `--enforce-eager` to separate graph compile from scheduling).
- Anything about speed: the PR reports no performance numbers.

## If the GPU smoke test fails

The container cannot see the card: then ROCm-in-Docker on this WSL setup is unsupported or mis-mounted, and vLLM is blocked on
gpu until one of: a native Ubuntu 24.04 install (glibc 2.39 for the `manylinux_2_39` wheel), a newer librocdxg/ROCm, or an NVIDIA
box (research-5-perf.md: CUDA is where vLLM is most likely to win, on the 12 GB 4070 Ti with fp8). Copy
`container_smoke.txt` and the last lines of `vllm_server.log` into UPGRADES.md and close Wave G as "blocked, no adapter".

## Files

- `vllm_check.sh` the single command; `vllm_client.py` request/response mapping + parity dump (`dump`); `load.py` load test
  (reuses `clef_server.httpbench`, plus shape checks and VRAM); `vllm_table.py` summary + decision; tests in
  `tests/unit/test_vllm_client.py`, `tests/unit/test_vllm_table.py`.
- Output: `dump_clef.json`, `dump_vllm.json`, `parity_vllm.txt/.rc`, `bench_{clef_base,clef_both,vllm}.txt`,
  `vram_*.json`, `vllm_server.log`, `container_smoke.txt`, `summary.md/.json`.
