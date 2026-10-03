#!/usr/bin/env bash
# Foreground server (what systemd runs): `clef serve`. Extra arguments are passed through.
set -euo pipefail
# shellcheck source=env.sh
. "$(dirname "${BASH_SOURCE[0]}")/env.sh"
if command -v clef >/dev/null 2>&1; then
  exec clef serve "$@"
fi
export PYTHONPATH="$CLEF_HOME/src${PYTHONPATH:+:$PYTHONPATH}"
exec python -m clef_server.cli serve "$@"
