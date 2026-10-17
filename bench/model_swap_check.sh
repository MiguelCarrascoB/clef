#!/usr/bin/env bash
# GPU check for the decision-model refactor (branch upg/c2-models). Run it on the GPU box from the root of a
# checkout of the branch, with no clef server running:
#
#   PY=~/venvs/clef/bin/python bash bench/model_swap_check.sh
#
# 1. Engine parity on clef-flash: the eval set through Engine.decide with the pre-refactor engine (REF, default
#    the merge-base with main) and with this tree's engine, same weights and settings; then compare
#    (0 top-1 flips, max |dp| <= TOL). Needs the clef-flash weights (`clef download`).
# 2. A smoke load + one record for each catalog id (`clef doctor --smoke` with CLEF_MODEL=<id>). The 27B runs with
#    CLEF_QUANT=nf4 (it fits 24 GB only that way) and needs bitsandbytes and its weights (~55 GB; set
#    DOWNLOAD_27B=1 to fetch them). Otherwise it is skipped with the reason.
#
# Env: PY (python with torch), REF (git commit for the reference engine), REF_SRC (a ready src/ dir instead of
# REF), N (eval records, 200), IMAGES (image records, 10), TOL (2e-3), OUT (bench/out/model_swap),
# DOWNLOAD_27B=1, SKIP_27B=1. Exit 0 = every check passed or was skipped with a reason.
set -uo pipefail

cd "$(dirname "$0")/.." || exit 2
PY=${PY:-python}
N=${N:-200}
IMAGES=${IMAGES:-10}
TOL=${TOL:-2e-3}
OUT=${OUT:-bench/out/model_swap}
PORT=${CLEF_PORT:-8910}
mkdir -p "$OUT"
status=0
note() { printf '\n== %s\n' "$*"; }
fail() { echo "FAIL: $*"; status=1; }

if curl -fsS "http://127.0.0.1:${PORT}/livez" >/dev/null 2>&1; then
  echo "a clef server answers on :${PORT} and holds the GPU; stop it first (clef stop)"
  exit 2
fi
"$PY" -c "import torch; print('torch', torch.__version__, 'gpu', torch.cuda.is_available())" || exit 2
command -v amd-smi >/dev/null 2>&1 && amd-smi list 2>/dev/null | head -5
[ -x /opt/rocm-wsl/bin/amd-smi ] && /opt/rocm-wsl/bin/amd-smi list 2>/dev/null | head -5

# ---------------------------------------------------------------- 1. engine parity (clef-flash)
note "reference engine source"
ref_src=${REF_SRC:-}
if [ -z "$ref_src" ]; then
  REF=${REF:-$(git merge-base HEAD main 2>/dev/null || git merge-base HEAD origin/main 2>/dev/null || true)}
  if [ -z "$REF" ]; then
    echo "no reference: set REF=<pre-refactor commit> (the branch started from 7a855c8) or REF_SRC=<src dir>"
    exit 2
  fi
  ref_dir="$OUT/ref-$(git rev-parse --short "$REF")"
  rm -rf "$ref_dir" && mkdir -p "$ref_dir"
  git archive "$REF" src | tar -x -C "$ref_dir" || { echo "git archive $REF failed"; exit 2; }
  ref_src="$ref_dir/src"
  echo "REF=$REF -> $ref_src"
fi

note "engine parity: clef-flash, $N records ($IMAGES with images), CLEF_DEVICE=${CLEF_DEVICE:-auto} CLEF_DTYPE=${CLEF_DTYPE:-auto}"
# both runs: the same settings, clef-flash, no warmup (parity.py turns it off), one engine per process
run_dump() {  # $1 out file, $2.. extra args
  local out=$1; shift
  env -u CLEF_MODEL CLEF_QUANT=none "$PY" bench/parity.py --engine-dump "$out" --n "$N" --images "$IMAGES" "$@"
}
if run_dump "$OUT/ref.json" --src "$ref_src" && run_dump "$OUT/new.json"; then
  "$PY" bench/parity.py --compare "$OUT/ref.json" "$OUT/new.json" --tol "$TOL" | tee "$OUT/parity.txt" \
    || fail "engine parity (see $OUT/parity.txt)"
  "$PY" - "$OUT/new.json" <<'EOF' || fail "results do not report model=clef-flash"
import json, sys
d = json.load(open(sys.argv[1]))
print("new engine: model_ref", d["model"], "result models", d["result_models"])
sys.exit(0 if d["result_models"] == ["clef-flash"] else 1)
EOF
else
  fail "an engine dump failed (weights missing? run: clef download)"
fi

# ---------------------------------------------------------------- 2. smoke load per model id
note "catalog"
"$PY" -m clef_server.cli models || fail "clef models"

note "smoke: clef-flash"
env CLEF_MODEL=clef-flash "$PY" -m clef_server.cli doctor --smoke | tee "$OUT/smoke_clef-flash.txt" \
  || fail "clef-flash smoke"

note "smoke: clef (27B, CLEF_QUANT=nf4)"
if [ "${SKIP_27B:-0}" = 1 ]; then
  echo "SKIP: SKIP_27B=1"
elif ! "$PY" -c "import bitsandbytes" 2>/dev/null; then
  echo "SKIP: bitsandbytes is not installed in $PY (nf4 needs it; the 27B does not fit 24 GB in bf16 or int8)."
  echo "      pip install bitsandbytes (ROCm: the multi-backend wheel), then re-run with SKIP_27B=0."
else
  have_27b=$(env CLEF_MODEL=clef "$PY" -c "
from clef_server.config import Config
from clef_server.paths import resolve_model_path
try:
    print(resolve_model_path(Config()))
except Exception:
    pass" 2>/dev/null)
  if [ -z "$have_27b" ] && [ "${DOWNLOAD_27B:-0}" = 1 ]; then
    env CLEF_MODEL=clef "$PY" -m clef_server.cli download --model clef --yes || fail "27B download"
    have_27b=yes
  fi
  if [ -z "$have_27b" ]; then
    echo "SKIP: the 27B weights are not downloaded (~55 GB): DOWNLOAD_27B=1 or clef download --model clef"
  else
    env CLEF_MODEL=clef CLEF_QUANT=nf4 CLEF_QUANT_BACKEND=bnb "$PY" -m clef_server.cli doctor --smoke \
      | tee "$OUT/smoke_clef.txt" || fail "clef (27B) nf4 smoke"
  fi
fi

note "unknown id is a load error, not a crash"
bad=$(env CLEF_MODEL=gpt-4o "$PY" -m clef_server.cli doctor --no-gpu 2>&1 || true)  # exits 1 on purpose
grep -E "\[FAIL\] model" <<<"$bad" || fail "CLEF_MODEL=gpt-4o should FAIL the doctor model check"

note "done: $([ $status = 0 ] && echo ALL OK || echo FAILURES) (outputs in $OUT)"
exit $status
