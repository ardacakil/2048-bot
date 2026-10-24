# Deploying the live bot API (VPS)

Everything here runs on the VPS as root (or with `sudo`), in order. Placeholders to fill in:

| Placeholder | Meaning |
|---|---|
| `PORT` | local port for uvicorn, e.g. `8200`; must not clash with Chuckle (`8100`) or the RUL predictor |
| `OTHER_SITE_1`, `OTHER_SITE_2` | URLs of your existing sites, to check they still respond |

What runs: one uvicorn worker on `127.0.0.1:PORT` as the system user `bot2048`, behind nginx,
which serves `https://2048.ardacakil.com/` (static files) and proxies `/api/` to it. The repo at
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
mkdir -p /var/www/2048.ardacakil.com
cp /opt/2048-bot/deploy/www/index.html /var/www/2048.ardacakil.com/index.html
```

## 5. nginx (a new file only; your other sites are not touched)

```bash
cp /opt/2048-bot/deploy/nginx/2048.ardacakil.com.conf /etc/nginx/sites-available/2048.ardacakil.com
sed -i 's/PORT/<your port>/g' /etc/nginx/sites-available/2048.ardacakil.com   # or edit by hand
grep -n proxy_pass /etc/nginx/sites-available/2048.ardacakil.com               # check both lines
ln -s /etc/nginx/sites-available/2048.ardacakil.com /etc/nginx/sites-enabled/2048.ardacakil.com
nginx -t && systemctl reload nginx
curl -sI http://2048.ardacakil.com/ | head -1                  # 200
curl -s  http://2048.ardacakil.com/api/health; echo
```

The file also defines the rate limits (6 game starts per minute per address with a burst of 4,
at most 2 concurrent streams per address, 2 req/s for other API paths) and an access log format
without client addresses (`bot2048_noip`: request line, status, size and timing only). Note that
nginx's **error log** can still contain client addresses when errors occur; that is nginx's
behaviour and not configurable per field.

## 6. HTTPS with certbot (while the DNS record is still "DNS only")

```bash
certbot --version || apt install certbot python3-certbot-nginx
certbot --nginx -d 2048.ardacakil.com --redirect
nginx -t && systemctl reload nginx
```

certbot adds `listen 443 ssl` and the certificate to our server block, and a new port-80 server
block that redirects to HTTPS. That new block has no `access_log` line, so redirect hits would be
logged in the default format (with addresses): add
`access_log /var/log/nginx/2048.ardacakil.com.access.log bot2048_noip;` to it, then
`nginx -t && systemctl reload nginx`. Renewal runs automatically (`systemctl list-timers | grep certbot`;
test with `certbot renew --dry-run`).

## 7. Verify

```bash
curl -s https://2048.ardacakil.com/api/health; echo
curl -s https://2048.ardacakil.com/api/stats; echo
curl -s -o /dev/null -w "%{http_code}\n" https://2048.ardacakil.com/api/play/1000   # 422

# one streamed game, each line timestamped (python3 instead of `ts`, which needs moreutils)
curl -sN https://2048.ardacakil.com/api/play/2048 | python3 -u -c '
import sys, time
t0 = time.monotonic()
for line in sys.stdin:
    if line.strip():
        print(f"{time.monotonic() - t0:7.3f}s  {line.rstrip()[:100]}")'
```

Expected: `event: meta` at ~0 s, `event: steps` lines spread over roughly the compute time
(~0.7 s for 2048 on this VPS), then `event: end`. If every line arrives at the same moment at the
end, something is buffering (check `proxy_buffering off` in the `/api/play/` location).

```bash
journalctl -u bot2048 -n 5 --no-pager      # one "game id=... status=reached ..." line, no addresses
systemctl show bot2048 -p MemoryCurrent
curl -sI OTHER_SITE_1 | head -1             # your other sites still respond
curl -sI OTHER_SITE_2 | head -1
```

Rate limit (each request starts a game that is cancelled when curl gives up after 1 s): the first 5
quick starts get 200, later ones 429:

```bash
for i in $(seq 1 8); do curl -s -o /dev/null -m 1 -w "%{http_code} " https://2048.ardacakil.com/api/play/2048; done; echo
```

Reproduce a game: take `seed` from a stream's `meta` and the move count from its `end`, then on
the Mac (or here with `PYTHONPATH=src .venv/bin/python`):

```bash
~/.local/bin/uv run python -m ntuple.timing --bot overnight-a0025 --games 1 --base-seed SEED --targets 2048:0
```

The move count must match (same seed, same search settings, same library versions).

From the Mac you can also watch a game in the terminal:
`~/.local/bin/uv run python -m server.watch --url https://2048.ardacakil.com --target 8192`.

## 8. Switch to proxied (Cloudflare orange cloud)

All steps are required, in this order.

**8.1 Real visitor addresses (mandatory).** Without this, nginx sees Cloudflare's addresses, and
the per-address limits (6 starts/min, 2 streams) would apply to all visitors together.

```bash
/opt/2048-bot/deploy/update-cloudflare-ips.sh     # writes /etc/nginx/snippets/bot2048-cloudflare-realip.conf, tests, reloads
nano /etc/nginx/sites-available/2048.ardacakil.com
#   uncomment   include /etc/nginx/snippets/bot2048-cloudflare-realip.conf;
#   in the HTTPS server block (and in certbot's port-80 block, so redirects are limited per visitor too)
nginx -t && systemctl reload nginx
```

**8.2 Cloudflare dashboard:** SSL/TLS mode **Full (strict)** (the Let's Encrypt certificate is
valid; renewal keeps working through the proxy), then switch the `2048` record to **Proxied**.

**8.3 Verify the real-IP setup with the temporary `/api/_ip` location.** It answers each caller
with the address nginx sees for that caller, nothing else (it reveals no one else's address and
is not logged with addresses). It ships commented out:

```bash
nano /etc/nginx/sites-available/2048.ardacakil.com
#   uncomment   location = /api/_ip { default_type text/plain; return 200 "$remote_addr\n"; }
nginx -t && systemctl reload nginx
```

From the Mac, and from a phone on mobile data (browser: `https://2048.ardacakil.com/api/_ip`):

```bash
curl -s https://2048.ardacakil.com/api/_ip
```

Each must show **its own** public address (compare with what your router or phone reports). If
you see an address from Cloudflare's ranges (e.g. 172.64.x.x, 162.158.x.x, 104.x.x.x or an
`2a06:98c0:`/`2400:cb00:` IPv6), the include is not active: go back to 8.1.

**8.4 Remove the temporary location again** (comment the line out), then:

```bash
nginx -t && systemctl reload nginx
curl -s -o /dev/null -w "%{http_code}\n" https://2048.ardacakil.com/api/_ip    # 404 (now handled by the app)
grep -n '^ *location = /api/_ip' /etc/nginx/sites-available/2048.ardacakil.com || echo "removed"
```

**8.5 Stream check through the proxy:** repeat the timestamped stream from section 7; lines must
still arrive spread out. Queued clients get a message every 15 s, well within Cloudflare's ~100 s
idle limit.

Cloudflare's ranges change rarely; rerun `update-cloudflare-ips.sh` when they announce a change,
or add a monthly cron job: `0 4 1 * * /opt/2048-bot/deploy/update-cloudflare-ips.sh`.

## Updating

```bash
/opt/2048-bot/deploy/deploy.sh      # git pull, pinned deps, restart, wait for /api/health
```

nginx changes are never made by `deploy.sh`; copy the file by hand and `nginx -t` first.

## Troubleshooting

| Symptom | Check |
|---|---|
| health stays 503 | `journalctl -u bot2048 -n 50`: checkpoint path (`BOT2048_CHECKPOINT`), read permissions, `"error"` in the health JSON |
| 502 from nginx | service down or wrong `PORT` in the nginx file vs `/etc/bot2048.env` |
| stream arrives all at once | `proxy_buffering off` missing in `/api/play/`; through Cloudflare, compare with the same check against `http://127.0.0.1:PORT` on the VPS to see which hop buffers |
| 429 for everyone after proxying | the real-IP include is not enabled (section 8, step 1) |
| killed / restarted | `MemoryMax=800M` reached: `journalctl -u bot2048 | grep -i memory` |
