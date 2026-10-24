"""Local development only: the page and the API under /2048-bot/, laid out as nginx serves them.

    uv run uvicorn --factory server.dev:create_dev_app --reload --port 8200
    open http://127.0.0.1:8200/2048-bot/

Production never imports this module (nginx serves web/ and proxies /2048-bot/api/). Like
deploy/nginx/2048-bot.conf, it redirects /2048-bot to /2048-bot/, strips the prefix from
/2048-bot/api/... before the unchanged app sees the request, and serves web/ with the same
Content-Security-Policy and security headers (no-cache, so edits show on reload), so CSP
violations show up locally.
"""

from pathlib import Path

from starlette.exceptions import HTTPException
from starlette.staticfiles import StaticFiles

from server.app import create_app

PREFIX = "/2048-bot"
WEB = Path(__file__).resolve().parents[2] / "web"
# keep identical to the add_header lines of deploy/nginx/2048-bot.conf (a test compares them)
CSP = "default-src 'self'; base-uri 'none'; object-src 'none'; form-action 'none'; frame-ancestors 'none'"
STATIC_HEADERS = {
    "Content-Security-Policy": CSP,
    "X-Content-Type-Options": "nosniff",
    "Referrer-Policy": "strict-origin-when-cross-origin",
    "Cache-Control": "no-cache",
}


def create_dev_app(settings=None, runner=None, web=WEB):
    api = create_app(settings, runner)
    static = StaticFiles(directory=web, html=True)
    extra = [(k.lower().encode(), v.encode()) for k, v in STATIC_HEADERS.items()]

    async def plain(send, status, body, headers=()):
        await send({"type": "http.response.start", "status": status,
                    "headers": [(b"content-type", b"text/plain; charset=utf-8"), *headers]})
        await send({"type": "http.response.body", "body": body.encode()})

    async def app(scope, receive, send):
        if scope["type"] == "lifespan":
            return await api(scope, receive, send)
        path = scope["path"]
        if path == PREFIX:
            query = scope.get("query_string", b"").decode()
            location = PREFIX + "/" + (f"?{query}" if query else "")
            return await plain(send, 301, "moved\n", [(b"location", location.encode())])
        if path == PREFIX + "/api" or path.startswith(PREFIX + "/api/"):
            stripped = path[len(PREFIX):]
            raw = scope.get("raw_path") or path.encode()
            return await api({**scope, "path": stripped, "raw_path": raw[len(PREFIX):]}, receive, send)
        if path.startswith(PREFIX + "/"):
            async def send_with_headers(message):
                if message["type"] == "http.response.start":
                    message = {**message, "headers": [*message.get("headers", []), *extra]}
                await send(message)

            try:
                return await static({**scope, "root_path": PREFIX}, receive, send_with_headers)
            except HTTPException as e:  # 404/405: no exception middleware out here
                return await plain(send, e.status_code, f"{e.detail}\n", extra)
        return await plain(send, 404, f"dev server: the page is at {PREFIX}/\n")

    app.api = api  # for tests
    return app
