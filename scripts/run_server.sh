#!/usr/bin/env bash
# Foreground server entrypoint (used by systemd; launch_server.sh is the detached/dev variant).
set -euo pipefail
# shellcheck source=env.sh
. "$(dirname "${BASH_SOURCE[0]}")/env.sh"
exec "$CLEF_VENV/bin/python" "$CLEF_HOME/server/main.py"
