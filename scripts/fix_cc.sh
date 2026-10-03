#!/bin/bash
set -x
source ~/venvs/clef/bin/activate
pip install -q wheel ninja
pip install -q --no-build-isolation causal-conv1d > /tmp/cc_build.log 2>&1
RC=$?
tail -5 /tmp/cc_build.log
python -c 'import causal_conv1d; print("causal-conv1d IMPORT OK")'
echo "CAUSAL_RC=$RC"
