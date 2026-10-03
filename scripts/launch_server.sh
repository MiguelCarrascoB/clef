#!/usr/bin/env bash
# Manage the clef-flash server inside WSL.
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
export CLEF_PORT

PIDFILE="$CLEF_LOG_DIR/server.pid"
LOG="$CLEF_LOG_DIR/server.log"
URL="http://127.0.0.1:$CLEF_PORT"
START_TIMEOUT="${CLEF_START_TIMEOUT:-300}"

is_alive() { [ -f "$PIDFILE" ] && kill -0 "$(cat "$PIDFILE")" 2>/dev/null; }

rotate_logs() {
  [ -s "$LOG" ] || return 0
  local i
  for i in 2 1; do
    [ -f "$LOG.$i" ] && mv -f "$LOG.$i" "$LOG.$((i + 1))"
  done
  mv -f "$LOG" "$LOG.1"
  rm -f "$LOG.4"
}

health_status() {
  curl -s --max-time 3 "$URL/health" 2>/dev/null |
    python3 -c 'import json,sys; print(json.load(sys.stdin).get("status",""))' 2>/dev/null || true
}

do_stop() {
  if ! is_alive; then
    echo "not running"; rm -f "$PIDFILE"; return 0
  fi
  local pid; pid="$(cat "$PIDFILE")"
  echo "stopping pid $pid"
  kill "$pid" 2>/dev/null || true
  for _ in $(seq 1 30); do
    kill -0 "$pid" 2>/dev/null || break
    sleep 1
  done
  if kill -0 "$pid" 2>/dev/null; then
    echo "still alive after 30 s, sending SIGKILL"; kill -9 "$pid" 2>/dev/null || true
  fi
  rm -f "$PIDFILE"
  echo "stopped"
}

do_status() {
  if is_alive; then
    echo "process: running (pid $(cat "$PIDFILE"))"
  else
    echo "process: not running"
  fi
  local st; st="$(health_status)"
  echo "health:  ${st:-unreachable} ($URL)"
  is_alive
}

do_start() {
  if is_alive; then
    echo "already running (pid $(cat "$PIDFILE")); use restart"; do_status || true; return 0
  fi
  if curl -fs --max-time 2 "$URL/livez" >/dev/null 2>&1; then
    echo "ERROR: something already answers on $URL (not managed by $PIDFILE). Stop it first." >&2
    return 1
  fi
  rotate_logs
  echo "starting clef server on port $CLEF_PORT (log: $LOG)"
  setsid nohup "$CLEF_VENV/bin/python" "$CLEF_HOME/server/main.py" > "$LOG" 2>&1 < /dev/null &
  echo $! > "$PIDFILE"
  local waited=0 st=""
  # phase 1: process up (/livez)
  until curl -fs --max-time 2 "$URL/livez" >/dev/null 2>&1; do
    if ! is_alive; then
      echo "SERVER EXITED during startup; last log lines:" >&2; tail -n 20 "$LOG" >&2; return 1
    fi
    if [ "$waited" -ge "$START_TIMEOUT" ]; then
      echo "TIMEOUT waiting for /livez" >&2; tail -n 20 "$LOG" >&2; return 1
    fi
    sleep 1; waited=$((waited + 1))
  done
  # phase 2: model loaded (ready|warming)
  until st="$(health_status)"; [ "$st" = ready ] || [ "$st" = warming ]; do
    if ! is_alive; then
      echo "SERVER EXITED while loading; last log lines:" >&2; tail -n 20 "$LOG" >&2; return 1
    fi
    if [ "$st" = error ]; then
      echo "SERVER REPORTS status=error; last log lines:" >&2; tail -n 20 "$LOG" >&2; return 1
    fi
    if [ "$waited" -ge "$START_TIMEOUT" ]; then
      echo "TIMEOUT (${START_TIMEOUT}s) waiting for model; status=${st:-unknown}" >&2; return 1
    fi
    sleep 2; waited=$((waited + 2))
  done
  echo "SERVER UP (pid $(cat "$PIDFILE"), status=$st, ${waited}s)"
}

case "$CMD" in
  start)   do_start ;;
  stop)    do_stop ;;
  restart) do_stop; do_start ;;
  status)  do_status ;;
  logs)    exec tail -n 100 -F "$LOG" ;;
  *) echo "usage: $0 [start|stop|restart|status|logs] [port]" >&2; exit 2 ;;
esac
