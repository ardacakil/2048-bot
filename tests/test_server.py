"""Live API tests. Streams are driven through a minimal ASGI client (lifespan, incremental body,
explicit disconnects); games use scripted or gated bots so nothing waits on real playback."""

import asyncio
import collections
import json
import logging
import threading

import pytest

from game import engine as E
from ntuple import expectimax as X
from ntuple import timing as TI
from ntuple.train import CHECKPOINT_ROOT
from server import wire
from server.app import MAX_SEED, code_version, create_app
from server.config import Settings
from server.runner import Runner
from server.watch import Checker, Desync

Z = [0, 0, 0, 0]
MODEL = {"run": "test", "checkpoint": "final", "games_trained": 0}
TIMEOUT = 10


# ---------------------------------------------------------------- bots


class FirstLegal:
    """Deterministic stand-in for Expectimax: the first legal move of LEFT, UP, RIGHT, DOWN."""

    prune = 1e-4

    def __init__(self):
        self.table = X.new_table(4)

    def best_move(self, board):
        for a in (E.LEFT, E.UP, E.RIGHT, E.DOWN):
            if E.is_legal(board, a):
                return a, 0.0
        return -1, 0.0


class Gated:
    """Wraps a bot; while the gate is closed every best_move waits for a permit."""

    def __init__(self, inner):
        self.inner, self.prune, self.table = inner, inner.prune, inner.table
        self.permits = threading.Semaphore(0)
        self.open = True
        self.calls = 0

    def close(self):
        """Close the gate and reset the call count (the warm-up move went through while open)."""
        self.open = False
        self.calls = 0

    def allow(self, n=1):
        for _ in range(n):
            self.permits.release()

    def release_all(self):
        self.open = True
        self.permits.release()

    def best_move(self, board):
        if not self.open:
            self.permits.acquire()
        self.calls += 1
        return self.inner.best_move(board)


def two_moves_to(target):
    """Row 0 = target/4, target/4, target/2: LEFT, LEFT makes `target` on move 2."""
    return E.from_values([[target // 4, target // 4, target // 2, 0], Z, Z, Z])


def make_app(bot, start_board=None, **kw):
    settings = Settings(**{"keepalive_s": 5.0, "lag_interval_s": 0.01, **kw})
    runner = Runner(settings, loader=lambda s: (bot, MODEL), start_board=start_board)
    return create_app(settings, runner)


# ---------------------------------------------------------------- minimal ASGI client


class Conn:
    def __init__(self, task, inbox, outbox):
        self.task, self.inbox, self.outbox = task, inbox, outbox
        self.parser = wire.Parser()
        self.pending = collections.deque()
        self.seen = []  # every event handed out so far, so events() can return the whole stream
        self.status, self.headers, self.body, self.ended = None, {}, b"", False

    async def start(self):
        msg = await asyncio.wait_for(self.outbox.get(), TIMEOUT)
        assert msg["type"] == "http.response.start", msg
        self.status = msg["status"]
        self.headers = {k.decode().lower(): v.decode() for k, v in msg["headers"]}
        return self

    async def next(self, timeout=TIMEOUT):
        """Next (event, data), comments included; None once the body has ended."""
        while not self.pending:
            if self.ended:
                return None
            msg = await asyncio.wait_for(self.outbox.get(), timeout)
            self.body += msg.get("body", b"")
            self.pending.extend(self.parser.feed(msg.get("body", b"").decode()))
            self.ended = not msg.get("more_body", False)
        self.seen.append(self.pending.popleft())
        return self.seen[-1]

    async def until(self, kind, timeout=TIMEOUT):
        while (ev := await self.next(timeout)) is not None:
            if ev[0] == kind:
                return ev[1]
        raise AssertionError(f"stream ended without {kind}")

    async def events(self):
        """The whole stream (including events already read with next/until), once it has ended."""
        while await self.next() is not None:
            pass
        await asyncio.wait_for(self.task, TIMEOUT)
        return list(self.seen)

    def json(self):
        return json.loads(self.body)

    async def disconnect(self):
        await self.inbox.put({"type": "http.disconnect"})
        await asyncio.wait_for(self.task, TIMEOUT)


class Client:
    def __init__(self, app):
        self.app = app

    async def __aenter__(self):
        self.ls_in, self.ls_out = asyncio.Queue(), asyncio.Queue()
        scope = {"type": "lifespan", "asgi": {"version": "3.0"}, "state": {}}
        self.ls_task = asyncio.create_task(self.app(scope, self.ls_in.get, self.ls_out.put))
        await self.ls_in.put({"type": "lifespan.startup"})
        assert (await asyncio.wait_for(self.ls_out.get(), TIMEOUT))["type"] == "lifespan.startup.complete"
        return self

    async def __aexit__(self, *exc):
        await self.ls_in.put({"type": "lifespan.shutdown"})
        await asyncio.wait_for(self.ls_out.get(), TIMEOUT)

    async def open(self, path):
        inbox, outbox = asyncio.Queue(), asyncio.Queue()
        await inbox.put({"type": "http.request", "body": b"", "more_body": False})
        scope = {"type": "http", "asgi": {"version": "3.0", "spec_version": "2.3"}, "http_version": "1.1",
                 "method": "GET", "scheme": "http", "path": path, "raw_path": path.encode(), "query_string": b"",
                 "root_path": "", "headers": [(b"host", b"test")], "client": ("127.0.0.1", 50000),
                 "server": ("test", 80), "state": {}}
        return await Conn(asyncio.create_task(self.app(scope, inbox.get, outbox.put)), inbox, outbox).start()

    async def get(self, path):
        conn = await self.open(path)
        await conn.events()
        return conn

    async def ready(self):
        await until(lambda: self.app.state.bot2048.runner.ready or self.app.state.bot2048.runner.error)


async def until(cond, timeout=TIMEOUT):
    """Wait for a condition set by the compute thread (polls; no fixed sleeps)."""
    loop = asyncio.get_running_loop()
    end = loop.time() + timeout
    while not cond():
        if loop.time() > end:
            raise AssertionError("condition not met in time")
        await asyncio.sleep(0.005)


def keys(obj):
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield k
            yield from keys(v)


def run(coro):
    return asyncio.run(coro)


def replay(events):
    """Check a finished stream against the engine; returns (meta, end, boards)."""
    kinds = [k for k, _ in events if k != "comment"]
    assert kinds[0] == "meta" and kinds[-1] == "end" and "init" in kinds
    data = {k: d for k, d in events if k in ("meta", "init", "end")}
    checker = Checker()
    checker.init(data["init"])
    boards = [checker.board]
    expected_first = 1
    for kind, d in events:
        if kind == "steps":
            assert d["first"] == expected_first
            for row in d["steps"]:
                checker.step(row)
                boards.append(checker.board)
            expected_first += len(d["steps"])
    end = data["end"]
    assert end["moves"] == checker.moves and end["score"] == checker.score
    assert end["final_board"] == f"0x{checker.board:016x}"
    return data["meta"], end, boards


# ---------------------------------------------------------------- streams


@pytest.mark.parametrize("target", [2048, 4096, 8192, 16384])
def test_stream_stops_exactly_when_the_target_first_appears(target):
    async def go():
        app = make_app(FirstLegal(), start_board=two_moves_to(target))
        async with Client(app) as c:
            await c.ready()
            conn = await c.open(f"/api/play/{target}")
            assert conn.status == 200
            assert conn.headers["content-type"].startswith("text/event-stream")
            assert conn.headers["x-accel-buffering"] == "no" and "no-transform" in conn.headers["cache-control"]
            meta, end, boards = replay(await conn.events())
        assert end["reached"] and end["status"] == "reached" and end["moves"] == 2 and end["max_tile"] == target
        assert [E.has_tile(b, target) for b in boards] == [False, False, True]
        assert meta["target"] == target and meta["delay_ms"] == Settings().delays_ms[target]
        assert 0 <= meta["seed"] < MAX_SEED and meta["depth"] == 2 and meta["model"] == MODEL
        assert meta["step_fields"] == list(wire.STEP_FIELDS) and "never via Number" in meta["encoding"]["board_hex"]
        assert meta["encoding"]["actions"] == list(E.ACTION_NAMES)
        assert meta["reach_rate"]["policy"] == "expectimax-d2" and meta["reach_rate"]["games"] > 0
        assert end["queue_wait_s"] >= 0 and end["compute_s"] >= 0

    run(go())


def test_game_over_before_the_target(W_eval):
    async def go():
        bot = X.Expectimax(W_eval, depth=1, prune=1e-4, tt_bits=18)  # random weights: never reaches 2048
        async with Client(make_app(bot, batch_moves=40)) as c:
            await c.ready()
            events = await (await c.open("/api/play/2048")).events()
        meta, end, boards = replay(events)
        assert not end["reached"] and end["status"] == "game_over" and E.is_game_over(boards[-1])
        assert sum(k == "steps" for k, _ in events) >= end["moves"] // 40  # batched
        # the same game as the timing tool / evaluation for this seed
        same = TI.play_until(bot, meta["seed"], 2048)
        assert (same.moves, same.score) == (end["moves"], end["score"])

    run(go())


def test_max_moves_cap(W_eval):
    async def go():
        bot = X.Expectimax(W_eval, depth=1, prune=1e-4, tt_bits=18)
        async with Client(make_app(bot, max_moves=7)) as c:
            await c.ready()
            _, end, _ = replay(await (await c.open("/api/play/2048")).events())
        assert end["status"] == "max_moves" and end["moves"] == 7 and not end["reached"]

    run(go())


@pytest.mark.skipif(not (CHECKPOINT_ROOT / "overnight-a0025" / "meta.json").exists(), reason="no checkpoint")
def test_real_model_every_target():
    async def go():
        app = create_app(Settings(checkpoint="overnight-a0025", lag_interval_s=0.01))
        async with Client(app) as c:
            await c.ready()
            assert app.state.bot2048.runner.ready
            for target in (2048, 4096, 8192, 16384):
                meta, end, boards = replay(await (await c.open(f"/api/play/{target}")).events())
                # random seed: d2 reaches 16384 in ~90% of games, so both outcomes are valid
                assert not any(E.has_tile(b, target) for b in boards[:-1])
                if end["reached"]:
                    assert E.has_tile(boards[-1], target) and end["status"] == "reached"
                else:
                    assert E.is_game_over(boards[-1]) and end["status"] == "game_over"
                same = TI.play_until(app.state.bot2048.runner.bot, meta["seed"], target)
                assert same.moves == end["moves"] and same.score == end["score"]
            assert meta["model"]["run"] == "overnight-a0025" and meta["tt_bits"] == 20

    run(go())


# ---------------------------------------------------------------- queue and disconnects


def test_queue_positions_keepalives_cap_and_leaving():
    async def go():
        bot = Gated(FirstLegal())
        app = make_app(bot, start_board=two_moves_to(2048), queue_cap=1, keepalive_s=0.05)
        st = app.state.bot2048
        try:
            stats = await queue_scenario(app, bot, st)
        finally:
            bot.release_all()  # never leave the compute thread blocked (it would hang interpreter exit)
        assert stats["games_served"] == 2 and stats["games_cancelled"] == 1 and stats["queue_length"] == 0

    async def queue_scenario(app, bot, st):
        async with Client(app) as c:
            await c.ready()
            bot.close()
            a = await c.open("/api/play/2048")
            await a.until("meta")
            await until(lambda: st.slot.running is not None)
            b = await c.open("/api/play/4096")
            assert (await b.until("queued"))["position"] == 1
            # keep-alive comment, then the queued message again
            assert await b.until("comment") == "ka"
            assert (await b.until("queued")) == {"position": 1, "queue_length": 1}
            full = await c.get("/api/play/8192")
            assert full.status == 503 and full.json()["detail"] == "busy"
            assert int(full.headers["retry-after"]) >= 5
            await b.disconnect()  # leaves the queue at once
            await until(lambda: not st.slot.waiting)
            d = await c.open("/api/play/2048")
            assert (await d.until("queued"))["position"] == 1
            bot.release_all()
            _, end_a, _ = replay(await a.events())
            events_d = await d.events()
            _, end_d, _ = replay(events_d)
            assert end_a["reached"] and end_d["reached"] and end_d["queue_wait_s"] >= 0
            assert [k for k, _ in events_d if k != "comment"][:3] == ["meta", "queued", "init"]  # it waited
            assert bot.calls == 4  # two games of two moves each
            return (await c.get("/api/stats")).json()

    run(go())


def test_disconnect_cancels_the_running_game():
    async def go():
        bot = Gated(FirstLegal())
        app = make_app(bot, start_board=two_moves_to(2048), queue_cap=2)
        st = app.state.bot2048
        records = []
        handler = logging.Handler()
        handler.emit = lambda r: records.append(r.getMessage())
        logging.getLogger("bot2048").addHandler(handler)
        try:
            async with Client(app) as c:
                await c.ready()
                bot.close()
                a = await c.open("/api/play/2048")
                await a.until("meta")
                waiting = await c.open("/api/play/2048")
                assert (await waiting.until("queued"))["position"] == 1
                await a.disconnect()
                assert st.slot.running is not None  # the thread is still inside best_move
                bot.allow(1)  # let that move finish: the thread sees the cancel and stops
                await waiting.until("init")  # the next game got the slot ...
                assert bot.calls == 1  # ... and the cancelled game played no further move
                bot.release_all()
                await waiting.events()
        finally:
            bot.release_all()
            logging.getLogger("bot2048").removeHandler(handler)
        assert st.stats.cancelled == 1 and st.stats.games_served == 1
        cancelled = [r for r in records if "status=cancelled" in r]
        assert len(cancelled) == 1 and "target=2048" in cancelled[0] and "seed=" in cancelled[0]
        assert not any("127.0.0.1" in r for r in records)  # no client addresses in the logs

    run(go())


# ---------------------------------------------------------------- endpoints


def test_invalid_targets_are_422():
    async def go():
        async with Client(make_app(FirstLegal())) as c:
            await c.ready()
            for path in ("/api/play/1000", "/api/play/32768", "/api/play/abc", "/api/play/-2048"):
                assert (await c.get(path)).status == 422, path

    run(go())


def test_health_and_play_wait_for_warm_up():
    async def go():
        gate = threading.Event()

        def slow_loader(settings):
            gate.wait(TIMEOUT)
            return FirstLegal(), MODEL

        settings = Settings(lag_interval_s=0.01)
        app = create_app(settings, Runner(settings, loader=slow_loader))
        async with Client(app) as c:
            h = await c.get("/api/health")
            assert h.status == 503 and h.json()["ready"] is False
            p = await c.get("/api/play/2048")
            assert p.status == 503 and p.json()["detail"] == "warming up"
            gate.set()
            await c.ready()
            h = await c.get("/api/health")
        body = h.json()
        assert h.status == 200 and body["ready"] and body["model"] == MODEL and body["warmup_s"] >= 0

    run(go())


def test_health_reports_a_failed_load():
    async def go():
        def broken(settings):
            raise FileNotFoundError("no checkpoint at /nowhere")

        settings = Settings(lag_interval_s=0.01)
        app = create_app(settings, Runner(settings, loader=broken))
        async with Client(app) as c:
            await c.ready()
            h = await c.get("/api/health")
        assert h.status == 503 and "no checkpoint" in h.json()["error"]

    run(go())


def test_stats():
    async def go():
        async with Client(make_app(FirstLegal(), start_board=two_moves_to(2048))) as c:
            await c.ready()
            await (await c.open("/api/play/2048")).events()
            await asyncio.sleep(0.05)  # a few lag samples (lag_interval_s = 0.01)
            s = await c.get("/api/stats")
        body = s.json()
        assert body["games_served"] == 1 and body["per_target"]["2048"] == {
            "games": 1, "reached": 1, "median_compute_s": body["per_target"]["2048"]["median_compute_s"]}
        assert body["per_target"]["16384"]["games"] == 0 and body["running"] == 0
        lag = body["event_loop_lag_ms"]
        assert lag["samples_last_min"] > 0 and lag["max_ms"] >= lag["mean_ms"] >= 0
        assert not [k for k in keys(body) if any(w in k for w in ("ip", "addr", "client", "host"))]

    run(go())


def test_code_version_reads_git_files(tmp_path):
    git = tmp_path / ".git"
    (git / "refs" / "heads").mkdir(parents=True)
    (git / "HEAD").write_text("ref: refs/heads/main\n")
    (git / "refs" / "heads" / "main").write_text("0123456789abcdef\n")
    assert code_version(tmp_path) == "0123456"
    (git / "refs" / "heads" / "main").unlink()
    (git / "packed-refs").write_text("# pack-refs\nfedcba9876543210 refs/heads/main\n")
    assert code_version(tmp_path) == "fedcba9"
    assert code_version(tmp_path / "missing") == "unknown"


# ---------------------------------------------------------------- client-side checks


def test_checker_catches_a_tampered_stream():
    async def go():
        async with Client(make_app(FirstLegal(), start_board=two_moves_to(2048))) as c:
            await c.ready()
            return await (await c.open("/api/play/2048")).events()

    events = run(go())
    replay(events)
    steps = [d for k, d in events if k == "steps"][0]["steps"]
    bad = [row[:] for row in steps]
    bad[0][4] = "0x0000000000000000"
    checker = Checker()
    checker.init([d for k, d in events if k == "init"][0])
    with pytest.raises(Desync, match="move 1: board"):
        checker.step(bad[0])
