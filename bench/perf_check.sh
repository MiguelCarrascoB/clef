#!/usr/bin/env bash
# GPU check for the D2 perf work (CLEF_MASK_FREE, CLEF_FAST_HEAD, CLEF_TUNABLEOP). Run it on the gpu box
# (RX 7900 XTX, ROCm, venv ~/venvs/clef) from the root of a checkout of the branch, no clef server running:
#
#   ssh gpu; cd ~/src/clef && git checkout upg/d2-perf
#   PY=~/venvs/clef/bin/python bash bench/perf_check.sh           # ~1 h; QUICK=1 for a ~15 min smoke
#
# Steps (results in $OUT, default bench/out/perf):
#   0. bench/profile_forward.py --batch 8     kernel/wall at batch 1 and 8: is batched serving GEMM-bound?
#   1. parity: Engine.decide dumps (bench/parity.py --engine-dump) for flags off (base), a repeat of base
#      (noise floor), CLEF_MASK_FREE=1 (mask), CLEF_FAST_HEAD=1 (head) and both; each compared with base.
#      PASS = 0 top-1 flips (near-ties apart) and max |dp| <= TOL (max(2e-3, 2 x the measured noise)), near-tie flips (ref top-2 margin < 1e-3) reported apart, singles and micro-batched.
#   2. speed: for each variant a fresh `clef serve --detach` + `clef bench` (single c=1, concurrent c=8,16).
#   3. TunableOp on vs off: base and both with CLEF_TUNABLEOP=1. A first pass tunes (CSV in the variant's own
#      state dir, so your real tuned files are untouched); the parity dump and the timed bench use the tuned
#      CSV. Warmup must stay on (CLEF_WARMUP=1, the default) so every bucket is tuned during startup.
# Then bench/perf_table.py prints the PASS/FAIL + speed table (also in $OUT/summary.md).
#
# Env: PY (python of the venv), N (200 eval records), IMAGES (10), REQUESTS (100 per level), LEVELS (1,8,16),
# TOL (2e-3), OUT, QUICK=1 (N=40 IMAGES=2 REQUESTS=30), SKIP_PROFILE=1, SKIP_NOISE=1, SKIP_TUNABLEOP=1,
# ONLY="mask head" (variants to run besides base), CLEF_* for device/dtype (same for every variant).
# Exit 0 = every parity check passed; 1 = a parity FAIL or a failed step; 2 = setup problem.
set -uo pipefail

cd "$(dirname "$0")/.." || exit 2
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
PY=${PY:-$HOME/venvs/clef/bin/python}
if [ "${QUICK:-0}" = 1 ]; then
  N=${N:-40}
  IMAGES=${IMAGES:-2}
  REQUESTS=${REQUESTS:-30}
fi
N=${N:-200}
IMAGES=${IMAGES:-10}
REQUESTS=${REQUESTS:-100}
LEVELS=${LEVELS:-1,8,16}
TOL=${TOL:-2e-3}
OUT=${OUT:-bench/out/perf}
PORT=${CLEF_PORT:-8910}
CLEF=("$PY" -m clef_server.cli)
mkdir -p "$OUT"
status=0
note() { printf '\n== %s\n' "$*"; }
fail() { echo "FAIL: $*"; status=1; }

if curl -fsS "http://127.0.0.1:${PORT}/livez" >/dev/null 2>&1; then
  echo "a clef server answers on :${PORT} and holds the GPU; stop it first (clef stop)"
  exit 2
fi
"$PY" -c "import torch, clef_server; print('torch', torch.__version__, 'gpu', torch.cuda.is_available(), \
'clef_server from', clef_server.__file__)" || exit 2
[ -x /opt/rocm-wsl/bin/amd-smi ] && /opt/rocm-wsl/bin/amd-smi list 2>/dev/null | head -5

# variant -> extra environment. Everything else is pinned off so a stray CLEF_* in your shell cannot leak in.
declare -A VENV=(
  [base]=""
  [mask]="CLEF_MASK_FREE=1"
  [head]="CLEF_FAST_HEAD=1"
  [both]="CLEF_MASK_FREE=1 CLEF_FAST_HEAD=1"
  [tun]="CLEF_TUNABLEOP=1"
  [tun_both]="CLEF_MASK_FREE=1 CLEF_FAST_HEAD=1 CLEF_TUNABLEOP=1"
)
vrun() { # vrun <variant> <cmd...>: run with that variant's environment and its own state dir
  local v=$1
  shift
  # shellcheck disable=SC2086
  env -u CLEF_MASK_FREE -u CLEF_FAST_HEAD -u CLEF_TUNABLEOP -u PYTORCH_TUNABLEOP_ENABLED \
    -u PYTORCH_TUNABLEOP_FILENAME CLEF_STATE_DIR="$PWD/$OUT/state-$v" ${VENV[$v]} "$@"
}
dump() { # dump <variant> <out json>
  vrun "$1" "$PY" bench/parity.py --engine-dump "$2" --n "$N" --images "$IMAGES"
}
compare() { # compare <name> <ref json> <new json> [informational]
  "$PY" bench/parity.py --compare "$2" "$3" --tol "$TOL" | tee "$OUT/parity_$1.txt"
  local rc=${PIPESTATUS[0]}
  echo "$rc" >"$OUT/parity_$1.rc"
  if [ "$rc" != 0 ] && [ -z "${4:-}" ]; then fail "parity $1 (see $OUT/parity_$1.txt)"; fi
}
serve_bench() { # serve_bench <variant> <requests> <out txt>
  local v=$1
  if ! vrun "$v" "${CLEF[@]}" serve --detach --timeout 1800; then
    fail "server did not start for $v"
    vrun "$v" "${CLEF[@]}" stop >/dev/null 2>&1 || true # a half-started server must not keep the GPU
    return 1
  fi
  vrun "$v" "${CLEF[@]}" bench --concurrency "$LEVELS" --requests "$2" 2>&1 | tee "$3"
  local rc=${PIPESTATUS[0]}
  [ "$rc" = 0 ] || fail "bench $v exited $rc (see $3)"
  vrun "$v" "${CLEF[@]}" stop
  return "$rc"
}

# ---------------------------------------------------------------- step 0
if [ "${SKIP_PROFILE:-0}" = 1 ]; then
  note "step 0: SKIP"
else
  note "step 0: profile_forward.py --batch 8 (flags off)"
  vrun base "$PY" bench/profile_forward.py --batch 8 --out "$OUT/profile" 2>&1 | tee "$OUT/profile.txt"
  [ "${PIPESTATUS[0]}" = 0 ] || fail "profile_forward"
fi

# ---------------------------------------------------------------- step 1: parity
note "parity reference: base (flags off), $N records ($IMAGES with images)"
if ! dump base "$OUT/dump_base.json"; then
  echo "base dump failed (weights missing? clef download)"
  exit 2
fi
variants=${ONLY:-"mask head both"}
if [ "${SKIP_NOISE:-0}" != 1 ]; then
  note "parity noise floor: base twice (informational)"
  if dump base "$OUT/dump_base2.json"; then
    compare noise "$OUT/dump_base.json" "$OUT/dump_base2.json" info
    # threshold = max(TOL, 2 x measured noise): the flags are not judged tighter than the run-to-run jitter
    if line=$("$PY" bench/parity.py --noise "$OUT/dump_base.json" "$OUT/dump_base2.json" --tol "$TOL"); then
      echo "$line" | tee "$OUT/noise_floor.txt"
      TOL=$(echo "$line" | awk '{print $4}')
    else
      echo "could not compute the noise floor; keeping TOL=$TOL"
    fi
  fi
fi
for v in $variants; do
  note "parity: $v (${VENV[$v]})"
  if dump "$v" "$OUT/dump_$v.json"; then compare "$v" "$OUT/dump_base.json" "$OUT/dump_$v.json"; else fail "dump $v"; fi
done

# ---------------------------------------------------------------- step 2: speed
note "speed: base"
serve_bench base "$REQUESTS" "$OUT/bench_base.txt"
for v in $variants; do
  note "speed: $v"
  serve_bench "$v" "$REQUESTS" "$OUT/bench_$v.txt"
done

# ---------------------------------------------------------------- step 3: TunableOp
if [ "${SKIP_TUNABLEOP:-0}" = 1 ]; then
  note "TunableOp: SKIP"
else
  for v in tun tun_both; do
    note "TunableOp tuning pass: $v (slow: tunes each GEMM shape once; CSV in $OUT/state-$v/tunableop)"
    serve_bench "$v" "$((REQUESTS / 3 + 1))" "$OUT/tune_$v.txt"
    ls -la "$OUT/state-$v"/tunableop/*/ 2>/dev/null | head
    note "parity: $v (tuned)"
    if dump "$v" "$OUT/dump_$v.json"; then compare "$v" "$OUT/dump_base.json" "$OUT/dump_$v.json"; else fail "dump $v"; fi
    note "speed: $v (tuned)"
    serve_bench "$v" "$REQUESTS" "$OUT/bench_$v.txt"
  done
fi

note "summary"
"$PY" bench/perf_table.py "$OUT" || status=1
echo "results: $OUT/summary.md  (raw: $OUT/*.txt, *.json)"
exit "$status"
