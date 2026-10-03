#!/bin/bash
# Launch (or relaunch) the clef-flash server detached inside WSL.
pkill -f "clef/server/main.py" 2>/dev/null
sleep 1
export HSA_ENABLE_DXG_DETECTION=1
export HF_HUB_OFFLINE=1
export CLEF_PORT="${1:-8910}"
setsid nohup ~/venvs/clef/bin/python /path/to/clef/server/main.py \
    > /tmp/server.log 2>&1 < /dev/null &
sleep 2
if pgrep -f "clef/server/main.py" > /dev/null; then
  echo "SERVER LAUNCHED (pid $(pgrep -f 'clef/server/main.py' | head -1)), log: /tmp/server.log"
else
  echo "SERVER FAILED TO START; log:"
  tail -5 /tmp/server.log
fi
