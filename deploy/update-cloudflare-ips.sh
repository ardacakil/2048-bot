#!/usr/bin/env bash
# Regenerate /etc/nginx/snippets/bot2048-cloudflare-realip.conf from Cloudflare's published edge
# ranges, check the nginx config and reload. Run as root when you switch the 2048 record to
# proxied, and again whenever Cloudflare changes its ranges (rare; a monthly cron job is fine).
set -euo pipefail
out=/etc/nginx/snippets/bot2048-cloudflare-realip.conf
tmp=$(mktemp)
trap 'rm -f "$tmp"' EXIT
{
    echo "# Cloudflare edge ranges, generated $(date -u +%Y-%m-%d) by update-cloudflare-ips.sh from"
    echo "# https://www.cloudflare.com/ips-v4 and https://www.cloudflare.com/ips-v6"
    for url in https://www.cloudflare.com/ips-v4 https://www.cloudflare.com/ips-v6; do
        curl -fsS --max-time 20 "$url" | while read -r net || [ -n "$net" ]; do
            [ -n "$net" ] && echo "set_real_ip_from $net;"
        done
    done
    echo "real_ip_header CF-Connecting-IP;"
} > "$tmp"
count=$(grep -c '^set_real_ip_from' "$tmp" || true)
if [ "$count" -lt 10 ]; then
    echo "only $count ranges downloaded; keeping the existing file" >&2
    exit 1
fi
mkdir -p "$(dirname "$out")"
install -m 644 "$tmp" "$out"
echo "wrote $out ($count ranges)"
nginx -t
systemctl reload nginx
echo "nginx reloaded"
