"""The model and the single compute thread.

Everything that touches Numba runs in one dedicated thread: the model load (importing
ntuple.expectimax compiles or loads the cached machine code), the warm-up move and every game.
The event loop stays free: each best_move call holds the GIL for well under a millisecond, then
the loop gets its turn. A game runs ntuple.timing.moves_until, the same code as the timing tool,
and hands batches of moves to the event loop with call_soon_threadsafe. A cancel event is checked
between moves, so a disconnected client stops its game within one move.
"""

import asyncio
import concurrent.futures
import dataclasses
import logging
import time

log = logging.getLogger("bot2048")


@dataclasses.dataclass
class GameResult:
    reached: bool
    game_over: bool
    moves: int
    score: int
    board: int
    compute_s: float  # wall time of the whole game computation in the compute thread


def default_loader(settings):
    """(bot, model info): weights read fully into RAM, evaluation search settings."""
    from ntuple import expectimax as X

    bot, meta, ckpt = X.load(settings.checkpoint, settings.depth, mmap=False)
    if ckpt.parent.name == "milestones":
        model = {"run": ckpt.parent.parent.name, "checkpoint": ckpt.name}
    else:
        model = {"run": ckpt.name, "checkpoint": "final"}
    return bot, {**model, "games_trained": int(meta["games_done"])}


class Runner:
    def __init__(self, settings, loader=default_loader, start_board=None):
        """`start_board` is a test hook: every game starts from that board instead of a new game."""
        self.settings = settings
        self.loader = loader
        self.start_board = start_board
        self.executor = concurrent.futures.ThreadPoolExecutor(1, thread_name_prefix="bot2048-compute")
        self.bot = self.model = None
        self.ready = False
        self.error = None
        self.load_s = self.warmup_s = None

    async def start(self):
        """Load the model and play one warm-up move in the compute thread."""
        try:
            await asyncio.get_running_loop().run_in_executor(self.executor, self._load)
        except Exception as e:  # reported by /api/health; the process stays up for inspection
            self.error = f"{type(e).__name__}: {e}"
            log.exception("model load failed")

    def _load(self):
        from game import engine as E

        t0 = time.perf_counter()
        self.bot, self.model = self.loader(self.settings)
        self.load_s = time.perf_counter() - t0
        t0 = time.perf_counter()
        self.bot.best_move(E.new_game(E.make_rng(0)))
        self.warmup_s = time.perf_counter() - t0
        self.ready = True
        log.info(f"ready: {self.model} depth {self.settings.depth}, load {self.load_s:.2f} s, "
                 f"warm-up {self.warmup_s:.2f} s")

    def submit(self, seed, target, cancel):
        """Start a game in the compute thread. Returns (future, queue); the queue receives
        ("init", board, spawns), ("steps", [Move, ...])..., then ("end", GameResult) or
        ("error", message). Nothing more is sent once `cancel` is set."""
        loop = asyncio.get_running_loop()
        queue = asyncio.Queue()

        def put(item):
            loop.call_soon_threadsafe(queue.put_nowait, item)

        return loop.run_in_executor(self.executor, self._compute, seed, target, cancel, put), queue

    def _compute(self, seed, target, cancel, put):
        from game import engine as E
        from ntuple import timing as T

        try:
            t0 = last = time.perf_counter()
            rng, board, spawns = T.start(seed, self.start_board)
            put(("init", board, spawns))
            batch, moves, score = [], 0, 0
            batch_s = self.settings.batch_ms / 1000
            for m in T.moves_until(self.bot, rng, board, target, self.settings.max_moves):
                if cancel.is_set():
                    return None
                batch.append(m)
                moves += 1
                score += m.reward
                board = m.board
                now = time.perf_counter()
                if len(batch) >= self.settings.batch_moves or now - last >= batch_s:
                    put(("steps", batch))
                    batch, last = [], now
            if cancel.is_set():
                return None
            if batch:
                put(("steps", batch))
            result = GameResult(E.has_tile(board, target), bool(E.is_game_over(board)), moves, score, board,
                                time.perf_counter() - t0)
            put(("end", result))
            return result
        except Exception as e:
            log.exception("game failed")
            put(("error", f"{type(e).__name__}: {e}"))
            return None

    def shutdown(self):
        self.executor.shutdown(wait=False, cancel_futures=True)
