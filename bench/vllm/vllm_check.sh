#!/usr/bin/env bash
# Wave G: vLLM vs clef on the gpu box (RX 7900 XTX, WSL2, ROCm). One command, from the root of a checkout:
#
#   ssh gpu; cd ~/src/clef && git checkout upg/g-vllm-bench
#   bash bench/vllm/vllm_check.sh            # ~1-2 h with a cold image/model download; QUICK=1 for a ~20 min smoke
#
# Steps (results in $OUT, default bench/out/vllm; the last thing printed is summary.md; this run's clef state dirs and
# container id in $OUT/run-<timestamp>-<pid>/):
#   0. preflight: GPU free (no clef server, no vLLM container, device VRAM < IDLE_MAX_GB), amd-smi, disk, docker,
#      the WSL files for the container (/dev/dxg, libdxcore.so, librocdxg.so, dids.conf)
#   1. docker pull $IMAGE, then a GPU smoke test INSIDE the image (torch sees the 7900 XTX?) before anything long
#   2. clef parity reference: Engine.decide dump, flags off (bench/parity.py --engine-dump)
#   3. clef speed, baseline (flags off) and CLEF_MASK_FREE=1 CLEF_FAST_HEAD=1: clef serve + vllm/load.py
#   4. vLLM: container with the pinned revision, parity dump over HTTP vs step 2, then the same load.py
#   5. teardown (always once preflight passed, also on error/Ctrl-C; only THIS run's container id and clef state
#      dirs, never another run's), bench/vllm/vllm_table.py -> summary.md with the D12 decision table
#
# Env: PY (venv python, ~/venvs/clef/bin/python), IMAGE (pinned tag below), N (200 records) IMAGES (10)
# REQUESTS (100 per level) LEVELS (1,8,16,32) TOL (2e-3) OUT, QUICK=1 (N=20 IMAGES=2 REQUESTS=20 LEVELS=1,8),
# MIXED=1 (mixed-length states), VLLM_PORT (8000), VLLM_MAX_SEQS (8, the PR's flag), VLLM_TIMEOUT (2400 s to start),
# GPU_MEM_UTIL (0.90), IDLE_MAX_GB (4), SKIP_PULL=1, FORCE=1 (ignore a busy or unreadable GPU, low disk), DXG /
# DXCORE / ROCDXG / DIDS (the WSL files mounted into the container), thresholds: MIN_SPEEDUP LEVEL ...
# Exit 0 = ADOPT verdict, 1 = DO NOT ADOPT / INCONCLUSIVE / a failed step, 2 = setup problem or the GPU is busy.
set -euo pipefail

cd "$(dirname "$0")/../.."
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
PY=${PY:-$HOME/venvs/clef/bin/python}
# Verified 2026-10-17 on Docker Hub (vllm/vllm-openai-rocm): built from vllm commit 39983a5e, which is 306 commits
# AFTER the merge commit of PR #61308 (380638af), ROCm 7.2.3 base image, entrypoint `vllm serve`. If it was
# deleted, list again: curl -s 'https://hub.docker.com/v2/repositories/vllm/vllm-openai-rocm/tags?page_size=20&ordering=last_updated'
IMAGE=${IMAGE:-vllm/vllm-openai-rocm:nightly-39983a5e7a5a575a7829f5d5ada50a33d9337ec1}
if [ "${QUICK:-0}" = 1 ]; then
  N=${N:-20}; IMAGES=${IMAGES:-2}; REQUESTS=${REQUESTS:-20}; LEVELS=${LEVELS:-1,8}
fi
N=${N:-200}; IMAGES=${IMAGES:-10}; REQUESTS=${REQUESTS:-100}; LEVELS=${LEVELS:-1,8,16,32}
TOL=${TOL:-2e-3}
OUT=${OUT:-bench/out/vllm}
case $OUT in /*) ;; *) OUT=$PWD/$OUT ;; esac
RUN_DIR=$OUT/run-$(date +%Y%m%d-%H%M%S)-$$ # this run's clef state dirs and container id (never another run's)
CLEF_PORT_=${CLEF_PORT:-8910}
VLLM_PORT=${VLLM_PORT:-8000}
VLLM_MAX_SEQS=${VLLM_MAX_SEQS:-8}
VLLM_TIMEOUT=${VLLM_TIMEOUT:-2400}
GPU_MEM_UTIL=${GPU_MEM_UTIL:-0.90}
IDLE_MAX_GB=${IDLE_MAX_GB:-4}
CONTAINER=clef-vllm-bench
CLEF=("$PY" -m clef_server.cli)
MIXED_ARG=(); [ "${MIXED:-0}" = 1 ] && MIXED_ARG=(--mixed)
mkdir -p "$OUT"
status=0
note() { printf '\n== %s\n' "$*"; }
fail() { echo "FAIL: $*"; status=1; }
die() { echo "$*" >&2; exit 2; }

# The model revision comes from the repo's registry, so the container serves exactly what clef serves.
read -r REPO REV < <("$PY" - <<'EOF'
from clef_server.models.clef_release import PINS, REPOS
print(REPOS["clef-flash"], PINS["clef-flash"])
EOF
)
[ -n "$REV" ] || die "could not read the clef-flash pin from clef_server.models.clef_release"

# shellcheck disable=SC2317  # called by the EXIT trap, which shellcheck does not follow
cleanup() { # installed only once preflight passed: a refused run must not touch another run's container / clef
  set +e
  local cid
  cid=$(cat "$RUN_DIR/vllm_container_id.txt" 2>/dev/null)
  [ -n "$cid" ] && docker rm -f "$cid" >/dev/null 2>&1
  for v in base both; do
    [ -d "$RUN_DIR/state-$v" ] && env CLEF_STATE_DIR="$RUN_DIR/state-$v" "${CLEF[@]}" stop >/dev/null 2>&1
  done
}
trap 'exit 130' INT TERM

vrun() { # vrun <variant> <cmd...>: clef variant environment, its own state dir
  local v=$1; shift
  local extra=""
  [ "$v" = both ] && extra="CLEF_MASK_FREE=1 CLEF_FAST_HEAD=1"
  # shellcheck disable=SC2086
  env -u CLEF_MASK_FREE -u CLEF_FAST_HEAD -u CLEF_TUNABLEOP -u PYTORCH_TUNABLEOP_ENABLED \
    CLEF_STATE_DIR="$RUN_DIR/state-$v" CLEF_PORT="$CLEF_PORT_" $extra "$@"
}

# ---------------------------------------------------------------- step 0: preflight
note "step 0: preflight"
export HSA_ENABLE_DXG_DETECTION=1 # WSL: ROCm sees no GPU without it (COMPUTE.md)
if curl -fsS "http://127.0.0.1:${CLEF_PORT_}/livez" >/dev/null 2>&1; then
  die "a clef server answers on :${CLEF_PORT_} and holds the GPU; stop it first (clef stop)"
fi
if curl -fsS "http://127.0.0.1:${VLLM_PORT}/health" >/dev/null 2>&1; then
  die "something answers on :${VLLM_PORT} (a vLLM server?); stop it first or set VLLM_PORT"
fi
command -v docker >/dev/null || die "docker not found. Install a Docker ENGINE inside this WSL distro (Docker Desktop's engine does not expose /dev/dxg): https://docs.docker.com/engine/install/ubuntu/"
docker info >/dev/null 2>&1 || die "docker is installed but not usable by this user (daemon down? sudo usermod -aG docker \$USER, then re-login)"
if docker ps -a --format '{{.Names}}' | grep -qx "$CONTAINER"; then
  die "a container named $CONTAINER exists (a previous run?): docker rm -f $CONTAINER"
fi
"$PY" -c "import torch, clef_server; print('torch', torch.__version__, 'gpu', torch.cuda.is_available(), \
'clef_server from', clef_server.__file__)" || die "venv $PY cannot import torch / clef_server"
[ -x /opt/rocm-wsl/bin/amd-smi ] && /opt/rocm-wsl/bin/amd-smi list 2>/dev/null | head -5 || echo "(amd-smi not found or not usable on WSL; AMD documents it as limited there)"
IDLE_GB=$("$PY" bench/vllm/load.py --print-vram)
echo "device VRAM in use before we start: ${IDLE_GB} GB"
if [ "${FORCE:-0}" != 1 ]; then
  [ "$IDLE_GB" != nan ] || die "cannot read the GPU's VRAM (torch sees no GPU), so a busy GPU cannot be ruled out; FORCE=1 to run anyway"
  "$PY" -c "import sys; sys.exit(0 if float('$IDLE_GB') < float('$IDLE_MAX_GB') else 1)" ||
    die "the GPU is busy: ${IDLE_GB} GB in use (limit ${IDLE_MAX_GB}). Check 'run-on gpu --ls' and tmux; FORCE=1 to ignore"
fi
if [ "$IDLE_GB" = nan ]; then echo "WARNING: torch cannot see the GPU, so VRAM cannot be measured and the clef steps will fail"; fi
# disk: the image is ~12 GB compressed (~25 GB unpacked) and the model ~19 GB if it is not in the HF cache yet
DOCKER_ROOT=$(docker info --format '{{.DockerRootDir}}' 2>/dev/null || echo /var/lib/docker)
HF_HOME_=${HF_HOME:-$HOME/.cache/huggingface}
mkdir -p "$HF_HOME_"
free_gb() { # GB free under a path (falls back to / when the path is not readable, e.g. a root-only docker dir)
  local g
  g=$(df --output=avail -BG "$1" 2>/dev/null | tail -1 | tr -dc '0-9')
  [ -n "$g" ] || g=$(df --output=avail -BG / | tail -1 | tr -dc '0-9')
  echo "$g"
}
echo "free disk: docker root ${DOCKER_ROOT}: $(free_gb "$DOCKER_ROOT") GB, HF cache $HF_HOME_: $(free_gb "$HF_HOME_") GB"
[ "$(free_gb "$DOCKER_ROOT")" -ge 45 ] || [ "${FORCE:-0}" = 1 ] || die "less than 45 GB free for the vLLM image under $DOCKER_ROOT (FORCE=1 to ignore)"
[ "$(free_gb "$HF_HOME_")" -ge 25 ] || [ "${FORCE:-0}" = 1 ] || die "less than 25 GB free in $HF_HOME_ for the model (FORCE=1 to ignore)"
# WSL files that go into the container (AMD librocdxg README, "Container Launch - WSL-Specific Flags")
DXG=${DXG:-/dev/dxg}
[ -e "$DXG" ] || die "$DXG is missing: this is not WSL2 with GPU paravirtualisation"
DXCORE=${DXCORE:-/usr/lib/wsl/lib/libdxcore.so}
[ -e "$DXCORE" ] || die "$DXCORE missing"
find_one() { for p in "$@"; do [ -e "$p" ] && { readlink -f "$p"; return 0; }; done; return 1; }
ROCDXG=${ROCDXG:-$(find_one /opt/rocm/lib/librocdxg.so /opt/rocm-wsl/lib/librocdxg.so /usr/local/lib/librocdxg.so || find /opt /usr -name 'librocdxg.so*' 2>/dev/null | head -1)}
DIDS=${DIDS:-$(find_one /opt/rocm/share/rocdxg/dids.conf /opt/rocm-wsl/share/rocdxg/dids.conf || find /opt /usr -name dids.conf -path '*rocdxg*' 2>/dev/null | head -1)}
{ [ -n "$ROCDXG" ] && [ -e "$ROCDXG" ]; } || die "librocdxg.so not found: install ROCm's librocdxg on the host (set ROCDXG=/path), https://github.com/ROCm/librocdxg"
{ [ -n "$DIDS" ] && [ -e "$DIDS" ]; } || die "dids.conf of librocdxg not found (set DIDS=/path)"
echo "mounting: $DXCORE, $ROCDXG, $DIDS"

# Preflight passed: from here on this run owns its state dirs and (once started) its container, and cleans them up.
mkdir -p "$RUN_DIR"
trap cleanup EXIT

DOCKER_GPU_FLAGS=(
  --device="$DXG"
  -v "$DXCORE:/usr/lib/libdxcore.so"
  -v "$ROCDXG:/usr/lib/librocdxg.so"
  -v "$DIDS:/usr/share/rocdxg/dids.conf"
  -e HSA_ENABLE_DXG_DETECTION=1
  --cap-add=SYS_PTRACE --security-opt seccomp=unconfined --ipc=host --shm-size 8G
)

# ---------------------------------------------------------------- step 1: image + in-container GPU smoke test
note "step 1: image $IMAGE"
if [ "${SKIP_PULL:-0}" != 1 ]; then docker pull "$IMAGE"; fi
docker image inspect "$IMAGE" --format 'image id {{.Id}} created {{.Created}}' | tee "$OUT/image.txt"
echo "$IMAGE" >>"$OUT/image.txt"
note "GPU smoke test inside the image (torch must see the card, or vLLM cannot work on this WSL setup)"
if ! docker run --rm --entrypoint python3 "${DOCKER_GPU_FLAGS[@]}" "$IMAGE" -c \
  "import torch; print('torch', torch.__version__, 'hip', torch.version.hip, 'cuda_available', torch.cuda.is_available()); \
assert torch.cuda.is_available(), 'torch cannot see the GPU in the container'; print(torch.cuda.get_device_name(0)); \
x = torch.randn(2048, 2048, device='cuda', dtype=torch.bfloat16); print('matmul ok', float((x @ x).float().abs().mean()))" 2>&1 | tee "$OUT/container_smoke.txt"; then
  echo "The container cannot use the GPU. vLLM-on-WSL is unsupported or mis-mounted here: that IS the result of this run." >&2
  echo "See bench/vllm/README.md, section 'If the GPU smoke test fails'. Only the clef lines can be measured." >&2
  status=1
  SKIP_VLLM=1
fi
SKIP_VLLM=${SKIP_VLLM:-0}

# ---------------------------------------------------------------- step 2: clef parity reference
note "step 2: clef parity reference (flags off): Engine.decide on $N records ($IMAGES with images)"
vrun base "$PY" bench/parity.py --engine-dump "$OUT/dump_clef.json" --n "$N" --images "$IMAGES" \
  || die "clef engine dump failed (weights missing? run: clef download)"

# ---------------------------------------------------------------- step 3: clef speed
clef_speed() { # clef_speed <base|both>
  local v=$1
  note "step 3: clef speed, $v"
  if ! vrun "$v" "${CLEF[@]}" serve --detach --timeout 1800; then
    fail "clef server did not start for $v"; vrun "$v" "${CLEF[@]}" stop >/dev/null 2>&1 || true; return
  fi
  "$PY" bench/vllm/load.py --url "http://127.0.0.1:${CLEF_PORT_}" --concurrency "$LEVELS" --requests "$REQUESTS" \
    "${MIXED_ARG[@]}" --label "clef_$v" --idle-gb "$IDLE_GB" --vram-json "$OUT/vram_clef_$v.json" \
    2>&1 | tee "$OUT/bench_clef_$v.txt" || fail "load test clef $v (see $OUT/bench_clef_$v.txt)"
  vrun "$v" "${CLEF[@]}" stop
}
clef_speed base
clef_speed both

# ---------------------------------------------------------------- step 4: vLLM
if [ "$SKIP_VLLM" = 1 ]; then
  note "step 4: SKIPPED (container smoke test failed)"
else
  note "step 4: vLLM ($REPO @ $REV, max-num-seqs $VLLM_MAX_SEQS)"
  docker run -d --name "$CONTAINER" "${DOCKER_GPU_FLAGS[@]}" \
    -e VLLM_WSL2_ENABLE_PIN_MEMORY=1 \
    -e HF_HOME=/hf -v "$HF_HOME_:/hf" \
    -p "127.0.0.1:${VLLM_PORT}:8000" \
    "$IMAGE" "$REPO" --revision "$REV" --served-model-name clef-flash --dtype bfloat16 \
    --max-model-len 16384 --max-num-batched-tokens 512 --max-num-seqs "$VLLM_MAX_SEQS" \
    --gpu-memory-utilization "$GPU_MEM_UTIL" >"$RUN_DIR/vllm_container_id.txt"
  CID=$(cat "$RUN_DIR/vllm_container_id.txt")
  cp "$RUN_DIR/vllm_container_id.txt" "$OUT/vllm_container_id.txt"
  up=0
  for ((t = 0; t < VLLM_TIMEOUT; t += 5)); do
    if curl -fsS "http://127.0.0.1:${VLLM_PORT}/health" >/dev/null 2>&1; then up=1; break; fi
    if [ "$(docker inspect -f '{{.State.Running}}' "$CID" 2>/dev/null)" != true ]; then break; fi
    sleep 5
  done
  docker logs "$CID" >"$OUT/vllm_server.log" 2>&1 || true
  if [ "$up" != 1 ]; then
    fail "vLLM did not become healthy within ${VLLM_TIMEOUT}s (see $OUT/vllm_server.log)"
    tail -40 "$OUT/vllm_server.log" || true
  else
    echo "vLLM is up after ~${t}s"
    # The exact request/response shape is checked here, loudly, before any number is trusted.
    if "$PY" bench/vllm/vllm_client.py dump --url "http://127.0.0.1:${VLLM_PORT}" --out "$OUT/dump_vllm.json" \
      --n "$N" --images "$IMAGES" 2>&1 | tee "$OUT/parity_vllm_dump.txt"; then
      set +e
      "$PY" bench/parity.py --compare "$OUT/dump_clef.json" "$OUT/dump_vllm.json" --tol "$TOL" | tee "$OUT/parity_vllm.txt"
      echo "${PIPESTATUS[0]}" >"$OUT/parity_vllm.rc"
      set -e
    else
      fail "vLLM parity dump (shape mismatch or HTTP error: see $OUT/parity_vllm_dump.txt)"
    fi
    "$PY" bench/vllm/load.py --url "http://127.0.0.1:${VLLM_PORT}" --concurrency "$LEVELS" --requests "$REQUESTS" \
      "${MIXED_ARG[@]}" --label vllm --idle-gb "$IDLE_GB" --vram-json "$OUT/vram_vllm.json" \
      2>&1 | tee "$OUT/bench_vllm.txt" || fail "load test vLLM (see $OUT/bench_vllm.txt)"
    docker logs "$CID" >"$OUT/vllm_server.log" 2>&1 || true
  fi
  # ---------------------------------------------------------------- teardown
  note "teardown"
  docker rm -f "$CID" >/dev/null && : >"$RUN_DIR/vllm_container_id.txt" &&
    echo "container removed (the image stays; docker rmi $IMAGE to free ~25 GB)"
fi

# ---------------------------------------------------------------- summary
note "summary"
set +e
"$PY" bench/vllm/vllm_table.py "$OUT" ${MIN_SPEEDUP:+--min-speedup "$MIN_SPEEDUP"} ${LEVEL:+--level "$LEVEL"} \
  ${MAX_P95_RATIO:+--max-p95-ratio "$MAX_P95_RATIO"}
verdict=$?
set -e
echo "results: $OUT/summary.md  (raw: $OUT/*.txt, *.json, vllm_server.log)"
[ "$status" = 0 ] && exit "$verdict"
exit 1
