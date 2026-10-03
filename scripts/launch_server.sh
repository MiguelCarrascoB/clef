#!/usr/bin/env bash
# Thin wrapper over the `clef` CLI for WSL / Linux (used by clef.ps1).
# Usage: launch_server.sh [start|stop|restart|status|logs] [port]     (a bare numeric arg = start on that port)
set -euo pipefail
# shellcheck source=env.sh
. "$(dirname "${BASH_SOURCE[0]}")/env.sh"

CMD="${1:-start}"
if [[ "$CMD" =~ ^[0-9]+$ ]]; then
  CLEF_PORT="$CMD"; CMD=start
elif [ -n "${2:-}" ]; then
  CLEF_PORT="$2"
fi
export CLEF_PORT="${CLEF_PORT:-8910}"
TIMEOUT="${CLEF_START_TIMEOUT:-600}"

case "$CMD" in
  start)   clef_run serve --detach --timeout "$TIMEOUT" ;;
  stop)    clef_run stop ;;
  restart) clef_run stop; clef_run serve --detach --timeout "$TIMEOUT" ;;
  status)  clef_run status ;;
  logs)    clef_run logs -f -n 100 ;;
  *) echo "usage: $0 [start|stop|restart|status|logs] [port]" >&2; exit 2 ;;
esac
