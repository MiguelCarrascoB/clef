#!/usr/bin/env bash
# Install the clef systemd --user unit (idempotent). Does not start the service.
set -euo pipefail
# shellcheck source=env.sh
. "$(dirname "${BASH_SOURCE[0]}")/env.sh"

if [ "$(ps -p 1 -o comm= 2>/dev/null)" != systemd ]; then
  cat <<MSG
systemd is not running in this WSL distro. Enable it:
  1. add to /etc/wsl.conf (sample: $CLEF_HOME/deploy/wsl.conf.sample):
       [boot]
       systemd=true
  2. from Windows run: wsl --shutdown
  3. re-run this script.
MSG
  exit 1
fi

UNIT_DIR="$HOME/.config/systemd/user"
mkdir -p "$UNIT_DIR" "$HOME/.config/clef"
sed "s#__CLEF_HOME__#$CLEF_HOME#g" "$CLEF_HOME/deploy/clef.service" > "$UNIT_DIR/clef.service"
[ -f "$HOME/.config/clef/env" ] || printf '# CLEF_API_KEY=...\n# CLEF_PORT=8910\n' > "$HOME/.config/clef/env"
systemctl --user daemon-reload
systemctl --user enable clef.service
cat <<MSG
Installed $UNIT_DIR/clef.service (enabled, not started).
  start:   systemctl --user start clef
  logs:    journalctl --user -u clef -f
  overrides: $HOME/.config/clef/env
  start at WSL boot without a login session: sudo loginctl enable-linger $USER
MSG
