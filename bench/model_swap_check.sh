#!/usr/bin/env bash
# GPU check for the decision-model refactor (branch upg/c2-models). Run it on the GPU box from the root of a
# checkout of the branch, with no clef server running:
#
#   PY=~/venvs/clef/bin/python REF=7a855c8 bash bench/model_swap_check.sh
#
# This tree's src/ is put first on PYTHONPATH, so the installed clef-local (any version) does not matter.
# 1. Engine parity on clef-flash: the eval set through Engine.decide with the pre-refactor engine (REF; the
#    branch starts from 7a855c8) and with this tree's engine, same weights and settings (TunableOp off); then
#    compare (0 top-1 flips, max |dp| <= TOL). Then the same with CLEF_OFFLOAD=cpu (the offload load was
#    rewritten to take its layer names from the adapter). Needs the clef-flash weights (`clef download`).
# 2. A smoke load + one record per catalog id (`clef doctor --smoke` with CLEF_MODEL=<id>). The 27B runs with
#    CLEF_QUANT=nf4 (it fits 24 GB only that way); it needs bitsandbytes and its weights (~55 GB; DOWNLOAD_27B=1
#    fetches them), else it is skipped with the reason. CLEF_MODEL_PATH / _REPO / _REVISION are unset for it
#    (they would redirect it), unless CLEF_27B_PATH names a local 27B release dir.
# 3. `clef serve` (warmup on) + /v1/models, one /v1/systemone, the unknown-model 404 and tests/integration.
#
# Env: PY, REF (git commit of the reference engine) or REF_SRC (a ready src/ dir), N (200), IMAGES (10),
# OFFLOAD_N (40), TOL (2e-3), OUT (bench/out/model_swap), DOWNLOAD_27B=1, SKIP_27B=1, SKIP_OFFLOAD=1,
# SKIP_SERVE=1, CLEF_27B_PATH. Exit 0 = every check passed or was skipped with a reason.
set -uo pipefail

cd "$(dirname "$0")/.." || exit 2
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
PY=${PY:-python}
N=${N:-200}
IMAGES=${IMAGES:-10}
OFFLOAD_N=${OFFLOAD_N:-40}
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
"$PY" -c "import torch, clef_server; print('torch', torch.__version__, 'gpu', torch.cuda.is_available(), \
'clef_server from', clef_server.__file__)" || exit 2
[ -x /opt/rocm-wsl/bin/amd-smi ] && /opt/rocm-wsl/bin/amd-smi list 2>/dev/null | head -5

# ---------------------------------------------------------------- 1. engine parity (clef-flash)
note "reference engine source"
ref_src=${REF_SRC:-}
if [ -z "$ref_src" ]; then
  REF=${REF:-$(git merge-base HEAD main 2>/dev/null || git merge-base HEAD origin/main 2>/dev/null || true)}
  if [ -z "$REF" ]; then
    echo "no reference: set REF=7a855c8 (the commit the branch starts from) or REF_SRC=<src dir>"
    exit 2
  fi
  ref_dir="$OUT/ref-$(git rev-parse --short "$REF")"
  rm -rf "$ref_dir" && mkdir -p "$ref_dir"
  git archive "$REF" src | tar -x -C "$ref_dir" || { echo "git archive $REF failed"; exit 2; }
  ref_src="$ref_dir/src"
  echo "REF=$REF -> $ref_src"
fi

# clef-flash with the user's device / dtype settings; no CLEF_MODEL, no quant, TunableOp off (main and this
# branch keep their tuned GEMMs in different files, which would show up as false differences)
run_dump() {  # $1 out file, $2.. extra args
  local out=$1
  shift
  env -u CLEF_MODEL CLEF_MASK_FREE=0 CLEF_FAST_HEAD=0 CLEF_TUNABLEOP=0 PYTORCH_TUNABLEOP_ENABLED=0 CLEF_QUANT=none \
    "$PY" bench/parity.py --engine-dump "$out" "$@"
}
check_parity() {  # $1 label, $2 ref json, $3 new json
  "$PY" bench/parity.py --compare "$2" "$3" --tol "$TOL" | tee "$OUT/parity_$1.txt" \
    || fail "engine parity $1 (see $OUT/parity_$1.txt)"
}

note "engine parity: clef-flash, $N records ($IMAGES with images), CLEF_DEVICE=${CLEF_DEVICE:-auto}"
if run_dump "$OUT/ref.json" --n "$N" --images "$IMAGES" --src "$ref_src" \
  && run_dump "$OUT/new.json" --n "$N" --images "$IMAGES"; then
  check_parity standard "$OUT/ref.json" "$OUT/new.json"
  "$PY" - "$OUT/new.json" <<'EOF' || fail "results do not report model=clef-flash"
import json, sys
d = json.load(open(sys.argv[1]))
print("new engine: model_ref", d["model"], "result models", d["result_models"])
sys.exit(0 if d["result_models"] == ["clef-flash"] else 1)
EOF
else
  fail "an engine dump failed (weights missing? run: clef download)"
fi

if [ "${SKIP_OFFLOAD:-0}" = 1 ]; then
  note "offload parity: SKIP (SKIP_OFFLOAD=1)"
else
  note "offload parity: CLEF_OFFLOAD=cpu CLEF_MAX_DEVICE_MEMORY_GB=14, $OFFLOAD_N records"
  export CLEF_OFFLOAD=cpu CLEF_MAX_DEVICE_MEMORY_GB=14
  if run_dump "$OUT/ref_off.json" --n "$OFFLOAD_N" --images 2 --src "$ref_src" \
    && run_dump "$OUT/new_off.json" --n "$OFFLOAD_N" --images 2; then
    check_parity offload "$OUT/ref_off.json" "$OUT/new_off.json"
  else
    fail "an offload engine dump failed"
  fi
  unset CLEF_OFFLOAD CLEF_MAX_DEVICE_MEMORY_GB
fi

# ---------------------------------------------------------------- 2. smoke load per model id
note "catalog"
"$PY" -m clef_server.cli models || fail "clef models"

note "smoke: clef-flash"
env CLEF_TUNABLEOP=0 CLEF_MODEL=clef-flash "$PY" -m clef_server.cli doctor --smoke \
  | tee "$OUT/smoke_clef-flash.txt" || fail "clef-flash smoke"

note "smoke: clef (27B, CLEF_QUANT=nf4)"
# the aliases must not redirect the 27B to other weights
if [ -n "${CLEF_27B_PATH:-}" ]; then
  big=(env -u CLEF_MODEL_REPO -u CLEF_MODEL_REVISION CLEF_TUNABLEOP=0 CLEF_MODEL=clef
    "CLEF_MODEL_PATH=$CLEF_27B_PATH")
else
  big=(env -u CLEF_MODEL_PATH -u CLEF_MODEL_REPO -u CLEF_MODEL_REVISION CLEF_TUNABLEOP=0 CLEF_MODEL=clef)
fi
if [ "${SKIP_27B:-0}" = 1 ]; then
  echo "SKIP: SKIP_27B=1"
elif ! "$PY" -c "import bitsandbytes" 2>/dev/null; then
  echo "SKIP: bitsandbytes is not installed in $PY (nf4 needs it; the 27B does not fit 24 GB in bf16 or int8)."
  echo "      Install a ROCm build of bitsandbytes, then re-run."
else
  where=$("${big[@]}" "$PY" -c "
from clef_server.config import Config
from clef_server.models.registry import resolve_spec
from clef_server.paths import resolve_model_path
cfg = Config()
print('ref', resolve_spec(cfg).ref)
try:
    print('path', resolve_model_path(cfg))
except Exception as exc:
    print('missing', exc)" 2>&1)
  echo "$where"
  if grep -q '^missing' <<<"$where" && [ "${DOWNLOAD_27B:-0}" = 1 ]; then
    "${big[@]}" "$PY" -m clef_server.cli download --model clef --yes || fail "27B download"
    where="path downloaded"
  fi
  if grep -q '^path' <<<"$where"; then
    "${big[@]}" CLEF_QUANT=nf4 CLEF_QUANT_BACKEND=bnb "$PY" -m clef_server.cli doctor --smoke \
      | tee "$OUT/smoke_clef.txt" \
      || fail "27B nf4 smoke (if the memory preflight refused it: retry with CLEF_OFFLOAD=cpu or CLEF_PREFLIGHT=0)"
  else
    echo "SKIP: the 27B weights are not downloaded (~55 GB): DOWNLOAD_27B=1, or clef download --model clef"
  fi
fi

note "unknown CLEF_MODEL is a load error, not a crash"
bad=$(env CLEF_MODEL=gpt-4o "$PY" -m clef_server.cli doctor --no-gpu 2>&1 || true)  # exits 1 on purpose
grep -E "\[FAIL\] model" <<<"$bad" || fail "CLEF_MODEL=gpt-4o should FAIL the doctor model check"

# ---------------------------------------------------------------- 3. the server (warmup on) over HTTP
if [ "${SKIP_SERVE:-0}" = 1 ]; then
  note "serve: SKIP (SKIP_SERVE=1)"
else
  note "serve: clef-flash with warmup, then HTTP checks"
  base="http://127.0.0.1:${PORT}"
  if env -u CLEF_MODEL CLEF_TUNABLEOP=0 CLEF_WARMUP=1 "$PY" -m clef_server.cli serve --detach --timeout 900; then
    curl -fsS "$base/readyz" && echo
    curl -fsS "$base/v1/models" | tee "$OUT/v1_models.json" && echo
    if curl -fsS "$base/v1/systemone" -H 'content-type: application/json' \
      -d '{"state":"Checkout is down","questions":{"outage":{"type":"noul","instructions":"Is a service down?"}}}' \
      | tee "$OUT/systemone.json"; then echo; else fail "/v1/systemone"; fi
    code=$(curl -s -o "$OUT/unknown_model.json" -w '%{http_code}' "$base/v1/systemone" \
      -H 'content-type: application/json' -d '{"model":"gpt-4o","state":"x","questions":{"o":{"type":"noul"}}}')
    [ "$code" = 404 ] || fail "unknown model on /v1/systemone gave HTTP $code, want 404"
    cat "$OUT/unknown_model.json" && echo
    curl -fsS "$base/health" | "$PY" -c "import json, sys; d = json.load(sys.stdin); \
print('health: status', d.get('status'), 'model', d.get('model'))"
    CLEF_URL="$base" "$PY" -m pytest -q -m gpu tests/integration -p no:cacheprovider || fail "tests/integration"
    "$PY" -m clef_server.cli stop || fail "clef stop"
  else
    fail "clef serve --detach did not become healthy (see: clef logs)"
    "$PY" -m clef_server.cli stop >/dev/null 2>&1
  fi
fi

note "done: $([ $status = 0 ] && echo ALL OK || echo FAILURES) (outputs in $OUT)"
exit $status
