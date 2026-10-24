"""How long does the bot take to reach a tile? A timing tool for the server machine (VPS).

    python -m ntuple.timing --bot overnight-a0025 [--depth 2] [--games 5] [--base-seed 0]
                            [--targets 2048:0.02,4096:0.02,8192:0.01,16384:0.005] [--sleep]

For each target tile, plays --games games (seeds base_seed + i) with expectimax at --depth until
a tile of at least that value first appears, or the game ends. Per game it measures the wall time
of the move computations only (no sleeping) and computes the playback time a viewer would wait:
moves x delay, the delay being the target's seconds per move. With --sleep it really sleeps the
delay after each move and reports the measured wall time (compute + sleep) instead, to confirm
the calculation.

Only the standard library, numpy, numba and this repo's engine/expectimax/train modules are
imported: the VPS has nothing else installed (no pandas, matplotlib or jupyter).

The weights are read fully into memory (no memory map), so page faults do not count as compute
time. Startup is reported separately: imports (functions with explicit Numba signatures are
compiled, or loaded from the on-disk cache, at import, so a first run on a new machine pays the
compilation here), checkpoint load, and one warm-up move.
Each game starts with an empty transposition table, as in the evaluation, so seed i plays the
same game as row i of the eval CSVs with the same depth.
"""

import argparse
import dataclasses
import resource
import sys
import time

_t0 = time.perf_counter()
import numpy as np  # noqa: E402

from game import engine as E  # noqa: E402
from ntuple import expectimax as X  # noqa: E402

IMPORT_S = time.perf_counter() - _t0  # includes Numba compilation / cache loading

DEFAULT_TARGETS = "2048:0.02,4096:0.02,8192:0.01,16384:0.005"


@dataclasses.dataclass
class Timing:
    seed: int
    target: int
    reached: bool
    moves: int  # moves until the target first appeared (or until game over)
    compute_s: float  # wall time of best_move calls only
    wall_s: float  # whole loop, including sleeps (only meaningful with sleep=True)
    score: int
    max_tile: int

    @property
    def ms_per_move(self):
        return self.compute_s / self.moves * 1e3 if self.moves else 0.0


def play_until(bot, seed, target, delay=0.0, sleep=False, start_board=None):
    """Play one game with `bot.best_move` until `target` first appears or no move is legal."""
    bot.table[1][:] = X.NO_ENTRY  # empty transposition table: the game depends only on the seed
    rng = E.make_rng(seed)
    board = E.new_game(rng) if start_board is None else start_board
    moves = score = 0
    compute = 0.0
    t_start = time.perf_counter()
    while not E.has_tile(board, target):
        t0 = time.perf_counter()
        action, _ = bot.best_move(board)
        compute += time.perf_counter() - t0
        if action < 0:
            break
        after, reward = E.move(board, action)
        board = E.spawn_tile(after, rng)
        score += int(reward)
        moves += 1
        if sleep:
            time.sleep(delay)
    return Timing(seed, target, E.has_tile(board, target), moves, compute, time.perf_counter() - t_start,
                  score, 1 << E.max_tile(board))


def parse_targets(text):
    """'2048:0.02,4096:0.02' -> [(2048, 0.02), (4096, 0.02)]."""
    out = []
    for item in text.split(","):
        tile, _, delay = item.strip().partition(":")
        tile, delay = int(tile), float(delay or 0)
        if tile < 4 or tile & (tile - 1):
            raise ValueError(f"target {tile} is not a power of two >= 4")
        out.append((tile, delay))
    return out


def peak_memory_mb():
    """Peak resident memory of this process: ru_maxrss is KB on Linux, bytes on macOS."""
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return rss / (1 << 20) if sys.platform == "darwin" else rss / (1 << 10)


STATS = {"min": min, "median": lambda v: float(np.median(v)), "max": max}


def _stats(values, names, fmt):
    """e.g. '120 / 150 / 200' for names ('min', 'median', 'max'); '–' if there are no values."""
    if not values:
        return "–"
    return " / ".join(fmt(STATS[n](values)) for n in names)


def summarize(timings, delay, sleep):
    """Summary dict over one target's games; moves/times are over the games that reached it."""
    ok = [t for t in timings if t.reached]
    playback = [t.wall_s if sleep else t.moves * delay for t in ok]
    total_moves = sum(t.moves for t in timings)
    return {
        "games": len(timings),
        "reached_pct": 100 * len(ok) / len(timings) if timings else 0.0,
        "moves": [t.moves for t in ok],
        "compute_s": [t.compute_s for t in ok],
        "ms_per_move": sum(t.compute_s for t in timings) / total_moves * 1e3 if total_moves else 0.0,
        "playback_s": playback,
    }


def format_target(target, delay, timings, sleep):
    s = summarize(timings, delay, sleep)
    kind = "measured, compute + sleep" if sleep else "moves x delay"
    lines = [f"target {target}  (delay {delay * 1000:g} ms/move)",
             f"  {'seed':>6} {'reached':>8} {'moves':>8} {'compute s':>10} {'ms/move':>9} {'playback s':>11}"]
    for t in timings:
        play = t.wall_s if sleep else t.moves * delay
        lines.append(f"  {t.seed:>6} {'yes' if t.reached else 'no':>8} {t.moves:>8,} {t.compute_s:>10.2f} "
                     f"{t.ms_per_move:>9.3f} {play:>11.1f}" + ("" if t.reached else f"   (game over, max {t.max_tile})"))
    lines += [
        f"  games {s['games']}, reached {s['reached_pct']:.0f}%",
        f"  moves      min/median/max  {_stats(s['moves'], ('min', 'median', 'max'), lambda v: f'{v:,.0f}')}",
        f"  compute s  median/max      {_stats(s['compute_s'], ('median', 'max'), lambda v: f'{v:.2f}')}",
        f"  compute    {s['ms_per_move']:.3f} ms/move (all games)",
        f"  playback s median/max      {_stats(s['playback_s'], ('median', 'max'), lambda v: f'{v:.1f}')}"
        f"   ({kind})",
    ]
    calc = [t.moves * delay + t.compute_s for t in timings if t.reached]
    if sleep and calc:
        lines.append(f"  for comparison, moves x delay + compute: median {np.median(calc):.1f} s")
    return "\n".join(lines)


def main(argv=None):
    p = argparse.ArgumentParser(description="Time how long the bot needs to reach target tiles.")
    p.add_argument("--bot", required=True, metavar="RUN", help="run name under checkpoints/ or a directory")
    p.add_argument("--depth", type=int, default=2, help="expectimax depth (0 = greedy)")
    p.add_argument("--targets", default=DEFAULT_TARGETS, help="tile:delay_seconds,... (default %(default)s)")
    p.add_argument("--games", type=int, default=5, help="games per target")
    p.add_argument("--base-seed", type=int, default=0, help="game i uses seed base_seed + i")
    p.add_argument("--sleep", action="store_true", help="really sleep the delay after each move")
    args = p.parse_args(argv)
    targets = parse_targets(args.targets)

    t0 = time.perf_counter()
    bot, meta, ckpt = X.load(args.bot, args.depth, mmap=False)
    load_s = time.perf_counter() - t0
    t0 = time.perf_counter()
    bot.best_move(E.new_game(E.make_rng(0)))  # compile / load cached machine code: not timed below
    warm_s = time.perf_counter() - t0
    search = f"expectimax depth {args.depth}" if args.depth else "greedy"
    print(f"{ckpt.name} ({meta['games_done']:,} games trained), {search} "
          f"(prune {bot.prune:g}, table 2^{bot.table[0].shape[0].bit_length() - 1} entries)")
    print(f"startup (not included below): imports + Numba compile/cache {IMPORT_S:.2f} s, "
          f"checkpoint load {load_s:.2f} s, warm-up move {warm_s:.2f} s\n")
    for target, delay in targets:
        timings = [play_until(bot, args.base_seed + i, target, delay, args.sleep) for i in range(args.games)]
        print(format_target(target, delay, timings, args.sleep) + "\n", flush=True)
    print(f"peak memory {peak_memory_mb():,.0f} MB")


if __name__ == "__main__":
    main()
