#!/usr/bin/env bash
# Update the live 2048 bot on the VPS: pull, install the pinned server dependencies, restart,
# wait until /api/health reports ready, then publish web/ to /var/www/2048-bot. Run as root from
# anywhere. Never touches nginx.
set -euo pipefail
REPO=/opt/2048-bot
WWW=/var/www/2048-bot
UV=${UV:-uv}
PORT=$(. /etc/bot2048.env && echo "$BOT2048_PORT")

cd "$REPO"
git pull --ff-only
"$UV" pip install --python "$REPO/.venv" -r deploy/requirements-server.txt
systemctl restart bot2048

echo -n "waiting for /api/health"
ready=
for _ in $(seq 1 120); do
    if body=$(curl -fsS "http://127.0.0.1:$PORT/api/health" 2>/dev/null); then
        echo; echo "ready: $body"
        ready=1
        break
    fi
    echo -n "."
    sleep 1
done
if [ -z "$ready" ]; then
    echo; echo "not ready after 120 s; web/ not published" >&2
    journalctl -u bot2048 -n 40 --no-pager >&2
    exit 1
fi

# the static page: copy web/ next to the live one, then swap directories (no half-copied site)
rm -rf "$WWW.new" "$WWW.old"
cp -R "$REPO/web" "$WWW.new"
chmod -R u=rwX,go=rX "$WWW.new"
if [ -d "$WWW" ]; then mv "$WWW" "$WWW.old"; fi
mv "$WWW.new" "$WWW"
rm -rf "$WWW.old"
echo "published $WWW ($(find "$WWW" -type f | wc -l | tr -d ' ') files)"
