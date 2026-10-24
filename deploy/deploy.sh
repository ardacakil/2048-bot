#!/usr/bin/env bash
# Update the live 2048 bot on the VPS: pull, install the pinned server dependencies, restart,
# wait until /api/health reports ready. Run as root from anywhere. Never touches nginx.
set -euo pipefail
REPO=/opt/2048-bot
UV=${UV:-uv}
PORT=$(. /etc/bot2048.env && echo "$BOT2048_PORT")

cd "$REPO"
git pull --ff-only
"$UV" pip install --python "$REPO/.venv" -r deploy/requirements-server.txt
systemctl restart bot2048

echo -n "waiting for /api/health"
for _ in $(seq 1 120); do
    if body=$(curl -fsS "http://127.0.0.1:$PORT/api/health" 2>/dev/null); then
        echo; echo "ready: $body"
        exit 0
    fi
    echo -n "."
    sleep 1
done
echo; echo "not ready after 120 s" >&2
journalctl -u bot2048 -n 40 --no-pager >&2
exit 1
