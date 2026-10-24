# Deploying the live bot (VPS)

Everything here runs on the VPS as root (or with `sudo`), in order. Placeholders to fill in:

| Placeholder | Meaning |
|---|---|
| `PORT` | local port for uvicorn, e.g. `8200`; must not clash with Chuckle (`8100`) or the RUL predictor |
| `OTHER_SITE_1`, `OTHER_SITE_2` | URLs of your existing sites, to check they still respond |

**Prerequisite (main-site repo).** The `ardacakil.com` server block, with HTTPS from certbot and
`include /etc/nginx/snippets/ardacakil/*.conf;` (or an include of `2048-bot.conf`), comes from
the main-site repo and must exist before section 5. This repo installs no server block and runs
no certbot. Once the domain is proxied by Cloudflare, this setup also **requires the main site's
real-IP config** (section 7).

What runs: one uvicorn worker on `127.0.0.1:PORT` as the system user `bot2048`, behind nginx.
nginx serves the page at `https://ardacakil.com/2048-bot/` (a copy of `web/` in
`/var/www/2048-bot`) and proxies `/2048-bot/api/` to uvicorn, stripping the `/2048-bot` prefix
(FastAPI's routes stay `/api/...`). The repo at
`/opt/2048-bot` and the checkpoint stay owned by root; `bot2048` only reads them and writes its
Numba cache in `/var/cache/bot2048`.

## 1. Code and dependencies

```bash
cd /opt/2048-bot
git pull --ff-only
uv pip install --python /opt/2048-bot/.venv -r deploy/requirements-server.txt
/opt/2048-bot/.venv/bin/python -c "import fastapi, uvicorn, numba; print('ok', numba.__version__)"
```

`deploy/requirements-server.txt` pins numpy, numba, fastapi, uvicorn and their dependencies to
the versions on the Mac (pandas, matplotlib and jupyter are never installed here). Same versions
on both machines also keep games reproducible from their seed.

## 2. Service user and permissions

```bash
useradd --system --no-create-home --shell /usr/sbin/nologin bot2048

# read access is all it needs: check it (each line should print "ok")
sudo -u bot2048 test -r /opt/2048-bot/src/server/app.py && echo ok
sudo -u bot2048 test -r /opt/2048-bot/checkpoints/overnight-a0025/meta.json && echo ok
sudo -u bot2048 sh -c 'test -r /opt/2048-bot/checkpoints/overnight-a0025/weights-*.npy' && echo ok
sudo -u bot2048 /opt/2048-bot/.venv/bin/python -c "print('ok')"
```

If a check fails, grant read access without changing ownership, e.g.
`chmod -R o+rX /opt/2048-bot/src /opt/2048-bot/checkpoints /opt/2048-bot/.venv` (and `o+x` on
`/opt/2048-bot` itself).

## 3. Settings and systemd unit

```bash
cp /opt/2048-bot/deploy/bot2048.env.example /etc/bot2048.env
nano /etc/bot2048.env                       # set BOT2048_PORT=PORT
chmod 644 /etc/bot2048.env

cp /opt/2048-bot/deploy/bot2048.service /etc/systemd/system/bot2048.service
systemctl daemon-reload
systemctl enable --now bot2048
```

Verify (the first start compiles the Numba code into `/var/cache/bot2048`, give it ~30 s):

```bash
systemctl status bot2048 --no-pager
journalctl -u bot2048 -n 30 --no-pager        # expect "ready: {...} depth 2, load ... s, warm-up ... s"
curl -s http://127.0.0.1:PORT/api/health; echo
systemctl show bot2048 -p MemoryCurrent -p MemoryMax
ls -la /var/cache/bot2048 | head
```

`/api/health` returns 503 with `"ready": false` until the weights are loaded and warmed up.

## 4. Web root

```bash
mkdir -p /var/www/2048-bot
cp -R /opt/2048-bot/web/. /var/www/2048-bot/
chmod -R u=rwX,go=rX /var/www/2048-bot
ls /var/www/2048-bot        # index.html  css  img  js
```

Later updates publish it with `deploy.sh` (see Updating).

## 5. nginx: one http-level file and one snippet (no other file is touched)

```bash
# http level: rate-limit zones and the address-free log format
cp /opt/2048-bot/deploy/nginx/conf.d-2048-bot.conf /etc/nginx/conf.d/2048-bot.conf

# location blocks, included by the main site's ardacakil.com server block
mkdir -p /etc/nginx/snippets/ardacakil
cp /opt/2048-bot/deploy/nginx/2048-bot.conf /etc/nginx/snippets/ardacakil/2048-bot.conf
sed -i 's/PORT/<your port>/g' /etc/nginx/snippets/ardacakil/2048-bot.conf   # or edit by hand
grep -n proxy_pass /etc/nginx/snippets/ardacakil/2048-bot.conf               # check both lines

grep -rn "snippets/ardacakil" /etc/nginx/sites-enabled/   # the main site's server block includes it
nginx -t && systemctl reload nginx
```

The snippet:

- redirects `/2048-bot` to `/2048-bot/`;
- serves `/var/www/2048-bot`;
- proxies `/2048-bot/api/play/` (SSE: no buffering, no gzip, 120 s timeouts) and `/2048-bot/api/`;
- sets cache headers: `no-cache` (revalidate) for HTML, JS and CSS, 30 days for fonts and images;
- sets a strict Content-Security-Policy (same origin only), `nosniff` and a referrer policy.

**The limits.** 6 game starts per minute per address with a burst of 4, at most 2 concurrent
streams, 2 req/s for the other API paths. They are a safety net: the page itself allows one game
per target per page load.

**Logging.** The access log `/var/log/nginx/2048-bot.access.log` uses the `bot2048_noip` format:
request line, status, size and timing only. nginx's **error log** can still contain client
addresses when errors occur; that is nginx's behaviour and not configurable per field.

**Headers.** `add_header` inside a location replaces every `add_header` of the server block for
that location. If the main site sets security headers at server level (HSTS, ...), repeat them
in the snippet's two static locations.

## 6. Verify

```bash
curl -sI https://ardacakil.com/2048-bot | grep -iE "^(HTTP|location)"       # 301 -> /2048-bot/
curl -sI https://ardacakil.com/2048-bot/ | grep -iE "^(HTTP|cache-control|content-security)"
curl -s  https://ardacakil.com/2048-bot/api/health; echo
curl -s  https://ardacakil.com/2048-bot/api/stats; echo
curl -s -o /dev/null -w "%{http_code}\n" https://ardacakil.com/2048-bot/api/play/1000   # 422

# one streamed game, each line timestamped (python3 instead of `ts`, which needs moreutils)
curl -sN https://ardacakil.com/2048-bot/api/play/2048 | python3 -u -c '
import sys, time
t0 = time.monotonic()
for line in sys.stdin:
    if line.strip():
        print(f"{time.monotonic() - t0:7.3f}s  {line.rstrip()[:100]}")'
```

Expected: `event: meta` at ~0 s, `event: steps` lines spread over roughly the compute time
(~0.7 s for 2048 on this VPS; for 16384 about 4 s), then `event: end`. If every line arrives at
the same moment at the end, something is buffering (check `proxy_buffering off` in the
`/2048-bot/api/play/` location).

Then open `https://ardacakil.com/2048-bot/` in a browser, play one target, and check the
browser console for Content-Security-Policy errors (there should be none).

```bash
journalctl -u bot2048 -n 5 --no-pager      # one "game id=... status=reached ..." line, no addresses
systemctl show bot2048 -p MemoryCurrent
curl -sI OTHER_SITE_1 | head -1             # your other sites still respond
curl -sI OTHER_SITE_2 | head -1
```

Rate limit (each request starts a game that is cancelled when curl gives up after 1 s): the first 5
quick starts get 200, later ones 429:

```bash
for i in $(seq 1 8); do curl -s -o /dev/null -m 1 -w "%{http_code} " https://ardacakil.com/2048-bot/api/play/2048; done; echo
```

Reproduce a game: take `seed` from a stream's `meta` and the move count from its `end`, then on
the Mac (or here with `PYTHONPATH=src .venv/bin/python`):

```bash
~/.local/bin/uv run python -m ntuple.timing --bot overnight-a0025 --games 1 --base-seed SEED --targets 2048:0
```

The move count must match (same seed, same search settings, same library versions).

From the Mac you can also watch a game in the terminal:
`~/.local/bin/uv run python -m server.watch --url https://ardacakil.com/2048-bot --target 8192`.

## 7. When ardacakil.com switches to proxied (Cloudflare orange cloud)

ardacakil.com is DNS only for now. When the main site switches to proxied:

**7.1 Real visitor addresses (mandatory, main-site repo).** The Cloudflare real-IP config
(`set_real_ip_from` for Cloudflare's ranges plus `real_ip_header CF-Connecting-IP`, and the script
that refreshes the ranges) is server-wide and owned by the main-site repo. Without it, nginx sees
Cloudflare's addresses, and the per-address limits (6 starts/min, 2 streams) would apply to all
visitors together. Install it from there first.

**7.2 Verify it with the temporary `/2048-bot/api/_ip` location.** It answers each caller with the
address nginx sees for that caller, nothing else (it reveals no one else's address and is not
logged with addresses). It ships commented out:

```bash
nano /etc/nginx/snippets/ardacakil/2048-bot.conf
#   uncomment   location = /2048-bot/api/_ip { default_type text/plain; return 200 "$remote_addr\n"; }
nginx -t && systemctl reload nginx
```

From the Mac, and from a phone on mobile data (browser: `https://ardacakil.com/2048-bot/api/_ip`):

```bash
curl -s https://ardacakil.com/2048-bot/api/_ip
```

Each must show **its own** public address (compare with what your router or phone reports). If
you see an address from Cloudflare's ranges (e.g. 172.64.x.x, 162.158.x.x, 104.x.x.x or an
`2a06:98c0:`/`2400:cb00:` IPv6), the real-IP config is not active: go back to 7.1.

**7.3 Remove the temporary location again** (comment the line out), then:

```bash
nginx -t && systemctl reload nginx
curl -s -o /dev/null -w "%{http_code}\n" https://ardacakil.com/2048-bot/api/_ip    # 404 (now handled by the app)
grep -n '^ *location = /2048-bot/api/_ip' /etc/nginx/snippets/ardacakil/2048-bot.conf || echo "removed"
```

**7.4 Stream check through the proxy:** repeat the timestamped stream from section 6; lines must
still arrive spread out. Queued clients get a message every 15 s, well within Cloudflare's ~100 s
idle limit.

## Leftovers from the subdomain plan

The earlier plan used `2048.ardacakil.com`. If any of it was created, remove it: the `2048` DNS
record in Cloudflare, `/etc/nginx/sites-enabled/2048.ardacakil.com` (and `sites-available`), its
certificate (`certbot delete --cert-name 2048.ardacakil.com`), and `/var/www/2048.ardacakil.com`.

## Updating

```bash
/opt/2048-bot/deploy/deploy.sh      # git pull, pinned deps, restart, wait for /api/health, publish web/
```

nginx changes are never made by `deploy.sh`; copy the files by hand and `nginx -t` first.

## Troubleshooting

| Symptom | Check |
|---|---|
| health stays 503 | `journalctl -u bot2048 -n 50`: checkpoint path (`BOT2048_CHECKPOINT`), read permissions, `"error"` in the health JSON |
| 502 from nginx | service down or wrong `PORT` in the snippet vs `/etc/bot2048.env` |
| `/2048-bot/` is 404 | snippet not included by the main site's server block (`nginx -T \| grep 2048-bot`), or `/var/www/2048-bot/index.html` missing |
| `nginx -t`: zone "bot2048_play" unknown | `/etc/nginx/conf.d/2048-bot.conf` missing, or conf.d not included in `nginx.conf`'s http block |
| page loads but no styles/scripts | console CSP errors, or the main site's regex locations catching `.js`/`.css` (the snippet's `^~` prevents this; check it is still there) |
| stream arrives all at once | `proxy_buffering off` missing in `/2048-bot/api/play/`; through Cloudflare, compare with the same check against `http://127.0.0.1:PORT` on the VPS to see which hop buffers |
| 429 for everyone after proxying | the main site's real-IP config is not active (section 7.1) |
| killed / restarted | `MemoryMax=800M` reached: `journalctl -u bot2048 \| grep -i memory` |
