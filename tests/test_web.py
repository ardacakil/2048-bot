"""Phase 5 checks on the Python side: the SSE fixtures the JavaScript tests use, the dev server's
/2048-bot/ mount, the static site's no-third-party / relative-URL / CSP rules, and the nginx
snippet. The frontend logic itself is tested with node --test tests/web/*.test.mjs."""

import asyncio
import importlib.util
import re
from pathlib import Path

import httpx
import pytest

from game import engine as E
from ntuple import expectimax as X
from server import dev
from server.config import Settings
from server.runner import Runner

ROOT = Path(__file__).resolve().parents[1]
WEB = ROOT / "web"
SNIPPET = ROOT / "deploy" / "nginx" / "2048-bot.conf"
CONFD = ROOT / "deploy" / "nginx" / "conf.d-2048-bot.conf"


def load_make_fixtures():
    spec = importlib.util.spec_from_file_location("make_fixtures", ROOT / "tests" / "web" / "make_fixtures.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# ---------------------------------------------------------------- fixtures


@pytest.mark.parametrize("name", ["reached-2048", "missed-16384"])
def test_committed_fixtures_match_the_current_wire_format(name):
    mf = load_make_fixtures()
    committed = (mf.OUT / f"{name}.sse").read_bytes()
    assert mf.build(name) == committed, "run: uv run python tests/web/make_fixtures.py"


# ---------------------------------------------------------------- dev server


class FirstLegal:
    """Deterministic stand-in for Expectimax (as in test_server.py)."""

    prune = 1e-4

    def __init__(self):
        self.table = X.new_table(4)

    def best_move(self, board):
        for a in (E.LEFT, E.UP, E.RIGHT, E.DOWN):
            if E.is_legal(board, a):
                return a, 0.0
        return -1, 0.0


def dev_get(*paths):
    """Responses from a fresh dev app (its runner started by hand: httpx's ASGI transport sends
    no lifespan events), one GET per path, redirects not followed."""
    settings = Settings(lag_interval_s=0)
    model = {"run": "test", "checkpoint": "final", "games_trained": 0}
    runner = Runner(settings, loader=lambda s: (FirstLegal(), model))
    app = dev.create_dev_app(settings, runner)

    async def run():
        await runner.start()
        assert runner.ready
        try:
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://dev") as c:
                return [await c.get(p) for p in paths]
        finally:
            runner.shutdown()

    return asyncio.run(run())


def test_dev_redirects_the_bare_prefix():
    [r] = dev_get("/2048-bot")
    assert r.status_code == 301 and r.headers["location"] == "/2048-bot/"


def test_dev_serves_the_page_with_the_production_headers():
    r, js = dev_get("/2048-bot/", "/2048-bot/js/main.js")
    assert r.status_code == 200 and "<title>2048 bot</title>" in r.text
    for name, value in dev.STATIC_HEADERS.items():
        assert r.headers[name] == value
    assert js.status_code == 200 and "javascript" in js.headers["content-type"]
    assert js.headers["content-security-policy"] == dev.CSP


def test_dev_strips_the_prefix_for_the_api_like_nginx():
    health, bad_target, unprefixed = dev_get("/2048-bot/api/health", "/2048-bot/api/play/1000", "/api/health")
    assert health.status_code == 200 and health.json()["ready"] is True
    assert bad_target.status_code == 422
    assert unprefixed.status_code == 404  # only the prefixed paths exist, as on the site


def test_dev_404s():
    assert [r.status_code for r in dev_get("/2048-bot/nope.js", "/")] == [404, 404]


# ---------------------------------------------------------------- static site rules


def web_files(*suffixes):
    return [p for p in WEB.rglob("*") if p.is_file() and p.suffix in suffixes]


def test_no_third_party_requests_and_only_relative_urls():
    allowed_absolute = {"https://ardacakil.com/"}  # the link back, not a request
    for path in web_files(".html", ".css", ".js", ".svg"):
        text = path.read_text()
        name = path.relative_to(ROOT)
        for url in re.findall(r"https?://[^\s\"')]+", text):
            if path.suffix == ".svg" and url == "http://www.w3.org/2000/svg":
                continue  # the SVG namespace, never fetched
            assert url in allowed_absolute, f"{name}: {url}"
        assert not re.search(r"""(?:src|href)\s*=\s*["']/""", text), f"{name}: root-absolute path"
        assert not re.search(r"""url\(\s*["']?/""", text), f"{name}: root-absolute url()"
        assert not re.search(r"""fetch\(\s*[`"']/""", text), f"{name}: root-absolute fetch"
        assert "data:" not in text, f"{name}: data: URL (not allowed by the CSP)"


def test_html_has_no_inline_styles_scripts_or_handlers():
    html = (WEB / "index.html").read_text()
    assert not re.search(r"\sstyle\s*=", html)
    assert not re.search(r"\son[a-z]+\s*=", html)
    assert "<style" not in html
    for tag in re.findall(r"<script\b[^>]*>", html):
        assert 'src="' in tag and 'type="module"' in tag
    assert not re.search(r"<script\b[^>]*>\s*[^<\s]", html)  # no inline script bodies


def test_js_never_sets_style_attributes():
    for path in web_files(".js"):
        text = path.read_text()
        assert not re.search(r"setAttribute\(\s*[\"']style|\.style\s*=|cssText|innerHTML", text), path.name


def test_api_urls_in_js_are_relative():
    main = (WEB / "js" / "main.js").read_text()
    assert "`api/play/${target}`" in main


# ---------------------------------------------------------------- nginx files


def snippet_headers():
    return re.findall(r'add_header\s+(\S+)\s+"([^"]*)"\s+always;', SNIPPET.read_text())


def test_nginx_static_headers_match_the_dev_server():
    headers = snippet_headers()
    for name, value in headers:
        if name != "Cache-Control":
            assert dev.STATIC_HEADERS[name] == value, name
    names = {n for n, _ in headers}
    assert names == set(dev.STATIC_HEADERS)


def test_nginx_snippet_strips_the_prefix_and_keeps_the_sse_settings():
    text = SNIPPET.read_text()
    assert "server {" not in text and "server_name" not in text and "listen" not in text
    assert "proxy_pass http://127.0.0.1:PORT/api/play/;" in text
    assert "proxy_pass http://127.0.0.1:PORT/api/;" in text
    play = text[text.index("location ^~ /2048-bot/api/play/"):]
    play = play[:play.index("\n}")]
    for line in ("proxy_buffering off;", "gzip off;", "proxy_read_timeout 120s;", "limit_conn bot2048_conn 2;",
                 "limit_req zone=bot2048_play burst=4 nodelay;", "limit_req_status 429;"):
        assert line in play
    assert "location = /2048-bot {" in text and "return 301 /2048-bot/;" in text
    assert re.search(r"^# location = /2048-bot/api/_ip", text, re.M)  # ships commented out


def test_nginx_http_level_file_has_only_the_zones_and_log_format():
    lines = [ln.strip() for ln in CONFD.read_text().splitlines() if ln.strip() and not ln.strip().startswith("#")]
    directives = {ln.split()[0] for ln in lines if not ln.startswith("'")}
    assert directives == {"limit_req_zone", "limit_conn_zone", "log_format"}
    for zone in ("bot2048_play", "bot2048_api", "bot2048_conn"):
        assert f"zone={zone}:" in CONFD.read_text()
    assert "remote_addr" not in CONFD.read_text().split("log_format", 1)[1].replace("$binary_remote_addr", "")
