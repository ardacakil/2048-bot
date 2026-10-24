"""Watch a live game from the server in the terminal.

    uv run python -m server.watch --url http://127.0.0.1:8200 --target 8192 [--delay-ms N] [--plain]

Reads /api/play/{target} (standard library only), checks every move against the engine (legal
move, reward, empty spawn cell, board and score) and plays the game back at the server's delay
(meta.delay_ms, or --delay-ms) on game.play's curses board, like the browser will. While queued
it shows the position. --plain prints progress lines instead of drawing (still paced), for
terminals without curses or for logs. q quits.
"""

import argparse
import collections
import queue
import threading
import time
import urllib.error
import urllib.request

from game import engine as E
from server import wire


class Desync(Exception):
    pass


class Checker:
    """Replays the stream through the engine; raises Desync at the first disagreement."""

    def __init__(self):
        self.board, self.score, self.moves = None, 0, 0

    def init(self, data):
        board = 0
        for cell, exp in data["spawns"]:
            if (board >> (4 * cell)) & 0xF:
                raise Desync(f"initial spawn on occupied cell {cell}")
            board |= exp << (4 * cell)
        if data["spawns"] and board != int(data["board"], 16):
            raise Desync("initial board does not match its spawns")
        self.board = int(data["board"], 16)

    def step(self, row):
        action, reward, cell, exp, board_hex, score = row
        after, r = E.move(self.board, action)
        after = int(after)
        n = self.moves + 1
        if after == self.board:
            raise Desync(f"move {n}: illegal move {E.ACTION_NAMES[action]}")
        if r != reward:
            raise Desync(f"move {n}: reward {reward}, engine says {r}")
        if (after >> (4 * cell)) & 0xF:
            raise Desync(f"move {n}: spawn on occupied cell {cell}")
        board = after | (exp << (4 * cell))
        if board != int(board_hex, 16):
            raise Desync(f"move {n}: board {board_hex}, engine says {board:#018x}")
        if self.score + reward != score:
            raise Desync(f"move {n}: score {score}, expected {self.score + reward}")
        self.board, self.score, self.moves = board, score, n


def read_stream(url, target, out, timeout=120):
    """Put every (event, data) of the stream on queue `out`, then ("closed", None)."""
    req = urllib.request.Request(f"{url.rstrip('/')}/api/play/{target}", headers={"Accept": "text/event-stream"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            parser = wire.Parser()
            while line := resp.readline():
                for ev in parser.feed(line.decode()):
                    out.put(ev)
    except urllib.error.HTTPError as e:
        out.put(("http_error", {"status": e.code, "body": e.read().decode(errors="replace")[:200]}))
    except OSError as e:
        out.put(("http_error", {"status": None, "body": str(e)}))
    out.put(("closed", None))


class Playback:
    """Buffers streamed moves and hands them out at the playback pace."""

    def __init__(self, events, delay_ms=None):
        self.events = events
        self.delay_override = delay_ms
        self.checker = Checker()
        self.buffer = collections.deque()
        self.meta = self.end = self.problem = None
        self.queued = None
        self.closed = False

    @property
    def delay(self):
        ms = self.delay_override if self.delay_override is not None else (self.meta or {}).get("delay_ms", 20)
        return ms / 1000

    def poll(self):
        while True:
            try:
                kind, data = self.events.get_nowait()
            except queue.Empty:
                return
            if kind == "meta":
                self.meta = data
            elif kind == "queued":
                self.queued = data
            elif kind == "init":
                self.queued = None
                self.checker.init(data)
            elif kind == "steps":
                self.buffer.extend(data["steps"])
            elif kind == "end":
                self.end = data
            elif kind in ("error", "http_error"):
                self.problem = data
            elif kind == "closed":
                self.closed = True

    def next_move(self):
        """Apply the next buffered move (checked); False if none is buffered."""
        if not self.buffer:
            return False
        self.checker.step(self.buffer.popleft())
        return True

    @property
    def finished(self):
        return not self.buffer and (self.end is not None or self.problem is not None or self.closed)

    def summary(self):
        if self.problem:
            return f"error: {self.problem}"
        if self.end is None:
            return "stream closed before the game ended"
        e, m = self.end, self.meta
        what = f"reached {m['target']}" if e["reached"] else f"game over before {m['target']} (max {e['max_tile']})"
        return (f"{what} in {e['moves']:,} moves, score {e['score']:,}  (seed {m['seed']}, compute "
                f"{e['compute_s']:.2f} s, queue {e['queue_wait_s']:.2f} s)")


class View:
    """What game.play.draw needs."""

    def __init__(self, until):
        self.board, self.score, self.moves, self.until = 0, 0, 0, until
        self.over = self.reached = False

    @property
    def done(self):
        return self.over or self.reached


def run_plain(play, target, progress_every=500):
    t0 = None
    last_queued = None
    while not play.finished:
        play.poll()
        if play.queued and play.queued != last_queued:
            print(f"queued: position {play.queued['position']} of {play.queued['queue_length']}", flush=True)
            last_queued = play.queued
        if play.meta and t0 is None:
            m = play.meta
            rate = m["reach_rate"]
            print(f"game {m['game_id']} seed {m['seed']}: target {target}, {m['policy']} depth {m['depth']}, "
                  f"{m['delay_ms']} ms/move" + (f", reaches it in {rate['pct']}% of evaluation games" if rate else ""),
                  flush=True)
            t0 = time.monotonic()
        if play.next_move():
            c = play.checker
            if c.moves % progress_every == 0:
                print(f"  move {c.moves:,}  score {c.score:,}  max tile {1 << E.max_tile(c.board)}", flush=True)
            time.sleep(play.delay)
        else:
            time.sleep(0.005)
    print(play.summary())
    if t0 is not None:
        print(f"playback {time.monotonic() - t0:.1f} s, every board checked against the engine")


def run_curses(scr, play, target):
    import curses

    from game import play as P

    curses.curs_set(0)
    has_color = P.init_colors()
    view = View(target)
    scr.nodelay(True)
    while True:
        play.poll()
        moved = play.next_move()
        c = play.checker
        if c.board is not None:
            view.board, view.score, view.moves = c.board, c.score, c.moves
            view.reached = E.has_tile(c.board, target)
            view.over = play.finished and not view.reached
        m = play.meta or {}
        title = f"2048 bot live: seed {m.get('seed', '?')}, target {target}"
        if play.queued:
            footer = f"queued: position {play.queued['position']}   q quit"
        elif play.finished:
            footer = play.summary()[:70] + "   q quit"
        else:
            footer = f"playing at {play.delay * 1000:g} ms/move   q quit"
        if c.board is not None or play.queued or play.finished:
            P.draw(scr, view, has_color, title, footer)
        if scr.getch() in (ord("q"), ord("Q")):
            return
        if play.finished:
            scr.nodelay(False)
        time.sleep(play.delay if moved else 0.01)


def main(argv=None):
    p = argparse.ArgumentParser(description="Watch a live game from the 2048 bot server.")
    p.add_argument("--url", default="http://127.0.0.1:8200", help="server base URL")
    p.add_argument("--target", type=int, default=2048)
    p.add_argument("--delay-ms", type=float, default=None, help="override the server's playback delay")
    p.add_argument("--plain", action="store_true", help="print progress lines instead of drawing")
    args = p.parse_args(argv)
    events = queue.Queue()
    threading.Thread(target=read_stream, args=(args.url, args.target, events), daemon=True).start()
    play = Playback(events, args.delay_ms)
    try:
        if args.plain:
            run_plain(play, args.target)
        else:
            import curses

            curses.wrapper(run_curses, play, args.target)
            print(play.summary())
    except Desync as e:
        raise SystemExit(f"stream does not match the engine: {e}")


if __name__ == "__main__":
    main()
