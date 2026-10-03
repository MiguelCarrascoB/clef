#!/usr/bin/env bash
# Install the clef systemd --user unit (idempotent). Does not start the service.
# Works on native Linux and on WSL2 with systemd enabled. Uses the `clef` found on PATH / in the venv.
set -euo pipefail
# shellcheck source=env.sh
. "$(dirname "${BASH_SOURCE[0]}")/env.sh"

if [ "$(ps -p 1 -o comm= 2>/dev/null)" != systemd ]; then
  cat <<MSG
systemd is not running here. On WSL2, enable it:
  1. add to /etc/wsl.conf (sample: $CLEF_HOME/deploy/wsl.conf.sample):
       [boot]
       systemd=true
  2. from Windows run: wsl --shutdown
  3. re-run this script.
MSG
  exit 1
fi

CLEF_BIN="$(command -v clef || true)"
if [ -z "$CLEF_BIN" ]; then
  echo "clef is not installed (pip install -e . or uv tool install ...); see requirements/README.md" >&2
  exit 1
fi

UNIT_DIR="$HOME/.config/systemd/user"
mkdir -p "$UNIT_DIR" "$HOME/.config/clef"
sed "s#__CLEF_BIN__#$CLEF_BIN#g" "$CLEF_HOME/deploy/clef.service" > "$UNIT_DIR/clef.service"
[ -f "$HOME/.config/clef/env" ] || printf '# CLEF_API_KEYS=name:key\n# CLEF_PORT=8910\n# CLEF_DEVICE=auto\n' > "$HOME/.config/clef/env"
systemctl --user daemon-reload
systemctl --user enable clef.service
cat <<MSG
Installed $UNIT_DIR/clef.service (enabled, not started; ExecStart=$CLEF_BIN serve).
  start:   systemctl --user start clef
  logs:    journalctl --user -u clef -f
  overrides: $HOME/.config/clef/env   (one KEY=value per line)
  start at boot without a login session: sudo loginctl enable-linger $USER
MSG
