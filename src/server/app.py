"""FastAPI app: GET /api/play/{target} (SSE), /api/health, /api/stats.

    uvicorn --factory server.app:create_app --host 127.0.0.1 --port PORT --workers 1 --no-access-log

One worker only: the queue lives in this process. A request for a valid target gets a fresh random
seed and a place in the queue (or 503 when the queue is full); its stream sends meta, queued
messages while waiting, then init, batches of steps and end. Games stream as fast as they are
computed; the browser paces playback with meta.delay_ms. A client that disconnects leaves the
queue or has its game cancelled. One log line per game; no IP addresses are logged.
"""

import asyncio
import collections
import contextlib
import logging
import secrets
import statistics
import sys
import threading
import time
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse

from server import build_config, wire
from server.config import Settings
from server.runner import Runner
from server.slot import ComputeSlot, QueueFull

log = logging.getLogger("bot2048")
ROOT = Path(__file__).resolve().parents[2]
MAX_SEED = 2**53  # seeds stay exact as JavaScript numbers


def code_version(root=ROOT):
    """Short commit hash read from .git (no subprocess: the service user cannot run git in a
    root-owned checkout), or "unknown"."""
    git = Path(root) / ".git"
    try:
        head = (git / "HEAD").read_text().strip()
        if not head.startswith("ref: "):
            return head[:7]
        ref = head[5:]
        if (git / ref).exists():
            return (git / ref).read_text().strip()[:7]
        for line in (git / "packed-refs").read_text().splitlines():
            if line.endswith(" " + ref):
                return line.split()[0][:7]
    except OSError:
        pass
    return "unknown"


class Stats:
    """Counters since start (in memory, reset on restart). No client data."""

    FINISHED = ("reached", "game_over", "max_moves")

    def __init__(self, targets):
        self.started = time.monotonic()
        self.games_served = self.cancelled = self.errors = 0
        self.per_target = {t: {"games": 0, "reached": 0, "compute": collections.deque(maxlen=50)}
                           for t in targets}

    def record(self, target, status, compute_s=None):
        if status in self.FINISHED:
            self.games_served += 1
            t = self.per_target[target]
            t["games"] += 1
            t["reached"] += status == "reached"
            if compute_s is not None:
                t["compute"].append(compute_s)
        elif status.startswith("cancelled"):
            self.cancelled += 1
        else:
            self.errors += 1

    def median_compute(self, target):
        c = self.per_target[target]["compute"]
        return statistics.median(c) if c else None

    def snapshot(self, slot, lag):
        return {
            "uptime_s": round(time.monotonic() - self.started, 1),
            "games_served": self.games_served, "games_cancelled": self.cancelled, "errors": self.errors,
            "running": int(slot.running is not None), "queue_length": len(slot.waiting),
            "queue_cap": slot.cap,
            "per_target": {str(t): {"games": v["games"], "reached": v["reached"],
                                    "median_compute_s": (round(self.median_compute(t), 3)
                                                         if v["compute"] else None)}
                           for t, v in self.per_target.items()},
            "event_loop_lag_ms": lag.snapshot(),
        }


class LagMonitor:
    """How late a periodic timer fires: the event loop's responsiveness (last minute + overall)."""

    def __init__(self, interval_s):
        self.interval = interval_s
        self.samples = collections.deque(maxlen=max(1, int(60 / interval_s)) if interval_s else 1)
        self.max_total = 0.0

    async def run(self):
        loop = asyncio.get_running_loop()
        while True:
            t0 = loop.time()
            await asyncio.sleep(self.interval)
            lag = max(0.0, loop.time() - t0 - self.interval)
            self.samples.append(lag)
            self.max_total = max(self.max_total, lag)

    def snapshot(self):
        if not self.interval or not self.samples:
            return None
        s = sorted(self.samples)
        ms = lambda v: round(v * 1e3, 2)
        return {"interval_ms": ms(self.interval), "samples_last_min": len(s), "mean_ms": ms(sum(s) / len(s)),
                "p99_ms": ms(s[min(len(s) - 1, int(0.99 * len(s)))]), "max_ms": ms(s[-1]),
                "max_since_start_ms": ms(self.max_total)}


class EventStream(StreamingResponse):
    """Streams an async iterator of SSE bytes. A client disconnect is noticed at once, even while
    nothing is being sent (e.g. waiting in the queue), and cancels the stream; `on_close` always
    runs, also when the stream never started."""

    def __init__(self, events, on_close):
        super().__init__(events, media_type=wire.MEDIA_TYPE, headers=wire.HEADERS)
        self.on_close = on_close

    async def __call__(self, scope, receive, send):
        async def pump():
            await send({"type": "http.response.start", "status": self.status_code, "headers": self.raw_headers})
            async for chunk in self.body_iterator:
                await send({"type": "http.response.body", "body": chunk, "more_body": True})
            await send({"type": "http.response.body", "body": b"", "more_body": False})

        async def watch():
            while (await receive())["type"] != "http.disconnect":
                pass

        tasks = [asyncio.ensure_future(pump()), asyncio.ensure_future(watch())]
        try:
            await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
        finally:
            self.on_close()
            for t in tasks:
                t.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            with contextlib.suppress(Exception):
                await self.body_iterator.aclose()


class LiveGame:
    """One request: its ticket, seed and game; events() streams it, close() cleans up once."""

    def __init__(self, app_state, ticket, target):
        self.s = app_state
        self.ticket, self.target = ticket, target
        self.seed = secrets.randbelow(MAX_SEED)
        self.game_id = secrets.token_hex(6)
        self.cancel = threading.Event()
        self.future = None
        self.status, self.moves, self.score, self.compute_s = "cancelled_queued", 0, 0, None
        self.closed = False

    async def events(self):
        st, ticket = self.s, self.ticket
        try:
            yield wire.sse("meta", wire.meta(
                self.game_id, self.seed, self.target, st.settings.depth, st.runner.bot.prune,
                int(st.runner.bot.table[0].shape[0]).bit_length() - 1, st.runner.model, st.code,
                st.settings.delays_ms[self.target], build_config.reach_rate(st.reach, st.settings.depth, self.target)))
            last_pos = None
            while not ticket.granted.is_set():
                pos = st.slot.position(ticket)
                if pos != last_pos:
                    yield wire.sse("queued", wire.queued(pos, len(st.slot.waiting), self._eta()))
                    last_pos = pos
                if not await st.slot.wait_change(st.settings.keepalive_s) and not ticket.granted.is_set():
                    yield wire.KEEPALIVE
                    last_pos = None  # repeat the queued message as well
            self.status = "cancelled"
            self.future, queue = st.runner.submit(self.seed, self.target, self.cancel)
            first, score = 1, 0
            while True:
                item = await queue.get()
                if item[0] == "init":
                    yield wire.sse("init", wire.init(item[1], item[2]))
                elif item[0] == "steps":
                    data, score = wire.steps(first, item[1], score)
                    first += len(item[1])
                    self.moves, self.score = first - 1, score
                    yield wire.sse("steps", data)
                elif item[0] == "end":
                    r = item[1]
                    data = wire.end(r.reached, r.game_over, r.moves, r.score, r.board, r.compute_s, self.wait_s)
                    self.status, self.compute_s = data["status"], r.compute_s
                    yield wire.sse("end", data)
                    return
                else:
                    self.status = "error"
                    yield wire.sse("error", wire.error("internal", "the game could not be computed"))
                    return
        finally:
            self.close()

    @property
    def wait_s(self):
        t = self.ticket
        return (t.granted_at or time.monotonic()) - t.created

    def _eta(self):
        medians = [self.s.stats.median_compute(t.target) for t in self.s.slot.ahead(self.ticket)]
        return sum(medians) if medians and None not in medians else None

    def close(self):
        """Leave the queue, or cancel the running game and release the slot once its thread stops."""
        if self.closed:
            return
        self.closed = True
        self.ticket.closed = True
        if not self.ticket.granted.is_set():
            self.s.slot.leave(self.ticket)
            self._finish()
        elif self.future is not None and not self.future.done():
            self.cancel.set()
            self.future.add_done_callback(lambda _: self._finish())
        else:
            self._finish()

    def _finish(self):
        self.s.slot.release(self.ticket)
        self.s.stats.record(self.target, self.status, self.compute_s)
        compute = f"{self.compute_s:.2f}" if self.compute_s is not None else "-"
        log.info(f"game id={self.game_id} target={self.target} seed={self.seed} status={self.status} "
                 f"reached={str(self.status == 'reached').lower()} moves={self.moves} compute_s={compute} "
                 f"wait_s={self.wait_s:.2f}")


def _setup_logging(level):
    if not log.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter("%(levelname)s %(message)s"))
        log.addHandler(handler)
        log.propagate = False
    log.setLevel(level.upper())


def create_app(settings=None, runner=None):
    settings = settings or Settings.from_env()
    _setup_logging(settings.log_level)
    state = type("State", (), {})()
    state.settings = settings
    state.runner = runner or Runner(settings)
    state.slot = ComputeSlot(settings.queue_cap)
    state.stats = Stats(settings.targets)
    state.lag = LagMonitor(settings.lag_interval_s)
    state.code = code_version()
    try:
        state.reach = build_config.load()
    except OSError:
        state.reach = None

    @contextlib.asynccontextmanager
    async def lifespan(app):
        tasks = [asyncio.create_task(state.runner.start())]
        if settings.lag_interval_s:
            tasks.append(asyncio.create_task(state.lag.run()))
        try:
            yield
        finally:
            for t in tasks:
                t.cancel()
            state.runner.shutdown()

    app = FastAPI(title="2048 bot", lifespan=lifespan, docs_url=None, redoc_url=None, openapi_url=None)
    app.state.bot2048 = state
    if settings.cors_origins:
        app.add_middleware(CORSMiddleware, allow_origins=list(settings.cors_origins), allow_methods=["GET"])

    @app.get("/api/health")
    async def health():
        r = state.runner
        body = {"ready": r.ready, "model": r.model, "depth": settings.depth, "code": state.code,
                "load_s": r.load_s and round(r.load_s, 3), "warmup_s": r.warmup_s and round(r.warmup_s, 3)}
        if r.error:
            body["error"] = r.error
        return JSONResponse(body, status_code=200 if r.ready else 503)

    @app.get("/api/stats")
    async def stats():
        return JSONResponse(state.stats.snapshot(state.slot, state.lag))

    @app.get("/api/play/{target}")
    async def play(target: int):
        if target not in settings.delays_ms:
            raise HTTPException(422, f"target must be one of {', '.join(map(str, settings.targets))}")
        if not state.runner.ready:
            return JSONResponse({"detail": "warming up"}, 503, headers={"Retry-After": "10"})
        try:
            ticket = state.slot.join(target)
        except QueueFull:
            known = [state.stats.median_compute(t) for t in settings.targets]
            per_game = max([m for m in known if m] or [5.0])
            retry = max(5, round(per_game * (len(state.slot.waiting) + 1)))
            return JSONResponse({"detail": "busy", "retry_after_s": retry}, 503,
                                headers={"Retry-After": str(retry)})
        game = LiveGame(state, ticket, target)
        return EventStream(game.events(), on_close=game.close)

    return app
