#!/usr/bin/env bash
# GPU check for F2 (CLEF_JOB_SORT_WINDOW, CLEF_TOKENIZE_AHEAD): job throughput with the flags off vs on. Run it on
# the gpu box (RX 7900 XTX, ROCm, venv ~/venvs/clef) from the root of a checkout of the branch, no clef server
# running. The branch is local to the devbox until pushed: get it there first (push a WIP branch, then
# `repo-sync` on gpu), or `run-on gpu bash bench/sort_check.sh` from the devbox, which copies the working tree.
#
#   ssh gpu; cd ~/src/clef && git checkout upg/f2-sort
#   PY=~/venvs/clef/bin/python bash bench/sort_check.sh            # ~20 min; QUICK=1 for a ~5 min smoke
#
# For each variant it starts a fresh `clef serve --detach` (own state dir, result cache OFF so repeats are not
# served from it), submits ONE classify job of ITEMS mixed-length inputs (long-tailed lengths, bench/jobs_bench.py)
# REPEAT times after one untimed warm-up job, with an interactive /v1/classify probe every PROBE seconds, and stops
# the server. Variants: base (flags off), sort (window 512), ahead (tokenize ahead), both; SWEEP=1 adds windows
# 256 and 1024. bench/sort_table.py then prints items/s, gain over base, label agreement and a verdict.
#
# Decision rule: flip a flag's default only if the median gain over base is above 10% (and above the spread
# of the base repeats), the label agreement is ~100% and the interactive p95 did not get worse.
#
# Env: PY, ITEMS (2000), REPEAT (3), PROBE (2 s; 0 = no interactive probe), MICROBATCH (CLEF_MAX_MICROBATCH,
# default 8 = the server default; also try 16), VARIANTS ("base sort ahead both"), SWEEP=1, OUT (bench/out/sort),
# QUICK=1 (ITEMS=600 REPEAT=2), CLEF_* for device/dtype (same for every variant).
# Exit 0 = every job succeeded; 1 = a failed step or job; 2 = setup problem.
set -uo pipefail

cd "$(dirname "$0")/.." || exit 2
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
PY=${PY:-$HOME/venvs/clef/bin/python}
if [ "${QUICK:-0}" = 1 ]; then
  ITEMS=${ITEMS:-600}
  REPEAT=${REPEAT:-2}
fi
ITEMS=${ITEMS:-2000}
REPEAT=${REPEAT:-3}
PROBE=${PROBE:-2}
OUT=${OUT:-bench/out/sort}
PORT=${CLEF_PORT:-8910}
VARIANTS=${VARIANTS:-"base sort ahead both"}
[ "${SWEEP:-0}" = 1 ] && VARIANTS="$VARIANTS sort256 sort1024"
CLEF=("$PY" -m clef_server.cli)
mkdir -p "$OUT"
rm -f "$OUT/runs.jsonl"
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
  [sort]="CLEF_JOB_SORT_WINDOW=512"
  [ahead]="CLEF_TOKENIZE_AHEAD=1"
  [both]="CLEF_JOB_SORT_WINDOW=512 CLEF_TOKENIZE_AHEAD=1"
  [sort256]="CLEF_JOB_SORT_WINDOW=256"
  [sort1024]="CLEF_JOB_SORT_WINDOW=1024"
)
vrun() { # vrun <variant> <cmd...>: run with that variant's environment and its own state dir
  local v=$1
  shift
  # shellcheck disable=SC2086
  env -u CLEF_JOB_SORT_WINDOW -u CLEF_TOKENIZE_AHEAD -u CLEF_MASK_FREE -u CLEF_FAST_HEAD \
    CLEF_MASK_FREE=0 CLEF_FAST_HEAD=0 CLEF_TUNABLEOP=0 CLEF_STATE_DIR="$PWD/$OUT/state-$v" CLEF_CACHE_SIZE=0 \
    CLEF_MAX_MICROBATCH="${MICROBATCH:-8}" \
    CLEF_MAX_JOB_ITEMS=100000 ${VENV[$v]} "$@"
}

for v in $VARIANTS; do
  if [ -z "${VENV[$v]+x}" ]; then
    echo "unknown variant: $v"
    exit 2
  fi
  note "variant: $v (${VENV[$v]:-flags off}), $ITEMS items x $REPEAT"
  if ! vrun "$v" "${CLEF[@]}" serve --detach --timeout 1800; then
    fail "server did not start for $v"
    vrun "$v" "${CLEF[@]}" stop >/dev/null 2>&1 || true # a half-started server must not keep the GPU
    continue
  fi
  vrun "$v" "$PY" bench/jobs_bench.py --tag "$v" --items "$ITEMS" --repeat "$REPEAT" --interactive "$PROBE" \
    --out "$OUT/runs.jsonl" 2>&1 | tee -a "$OUT/log_$v.txt"
  [ "${PIPESTATUS[0]}" = 0 ] || fail "jobs_bench $v"
  vrun "$v" "${CLEF[@]}" stop
done

note "summary"
"$PY" bench/sort_table.py "$OUT" || status=1
echo "results: $OUT/summary.md  (raw: $OUT/runs.jsonl, $OUT/log_*.txt)"
exit "$status"
