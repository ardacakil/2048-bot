"""Evaluate a checkpoint: play N games with greedy (depth 0) or expectimax, in parallel.

    uv run python -m ntuple.evaluate --run NAME --games 200 --depth 2 [--workers 8]

Workers memory-map the checkpoint's weights copy-on-write, so all processes share one
physical copy of the 268 MB. Game i always uses seed (seed, i) and starts with an empty
transposition table, so results do not depend on the number of workers.
"""

import argparse
import multiprocessing as mp
import os
import time
from pathlib import Path

import numpy as np

from game import engine as E
from ntuple import expectimax as X
from ntuple.train import CHECKPOINT_ROOT, REACH_TILES, load_checkpoint

_worker = {}


def _init_worker(ckpt_dir, depth, prune, tt_bits):
    W, _ = load_checkpoint(ckpt_dir, mmap=True)
    _worker["ex"] = X.Expectimax(W, depth=depth, prune=prune, tt_bits=tt_bits, use_tt=depth > 0)


def game_seed(seed, i):
    return seed * 1_000_003 + i


def _play(args):
    i, seed = args
    ex = _worker["ex"]
    ex.table[1][:] = X.NO_ENTRY  # empty table per game, for reproducibility
    t0 = time.perf_counter()
    score, tile, n = ex.play_game(E.make_rng(game_seed(seed, i)))
    return i, score, tile, n, time.perf_counter() - t0


def evaluate(ckpt_dir, games=100, depth=0, workers=None, seed=0, prune=1e-4, tt_bits=20,
             verbose=True):
    """Play `games` games; returns a dict of summary stats plus a per-game DataFrame."""
    import pandas as pd

    ckpt_dir = Path(ckpt_dir)
    workers = workers or min(8, os.cpu_count())
    ctx = mp.get_context("spawn")
    t0 = time.perf_counter()
    rows = []
    with ctx.Pool(workers, _init_worker, (str(ckpt_dir), depth, prune, tt_bits)) as pool:
        for k, r in enumerate(pool.imap_unordered(_play, [(i, seed) for i in range(games)]), 1):
            rows.append(r)
            if verbose and (k % max(1, games // 10) == 0 or k == games):
                print(f"  {k}/{games} games, {time.perf_counter() - t0:.1f} s", flush=True)
    wall = time.perf_counter() - t0
    df = pd.DataFrame(rows, columns=["game", "score", "max_exp", "moves", "seconds"]).sort_values("game")
    df["max_tile"] = np.left_shift(1, df["max_exp"].to_numpy())
    df = df.drop(columns="max_exp").reset_index(drop=True)

    tiles = df["max_tile"].value_counts().sort_index(ascending=False)
    summary = {
        "checkpoint": str(ckpt_dir),
        "depth": depth,
        "games": games,
        "avg_score": float(df["score"].mean()),
        "std_score": float(df["score"].std()),
        "median_score": float(df["score"].median()),
        "max_score": int(df["score"].max()),
        **{f"reach_{t}": float((df["max_tile"] >= t).mean() * 100) for t in REACH_TILES},
        "max_tile_pct": {int(t): float(c / games * 100) for t, c in tiles.items()},
        "ms_per_move": float(df["seconds"].sum() / df["moves"].sum() * 1e3),
        "wall_s": wall,
        "workers": workers,
    }
    if verbose:
        print(format_summary(summary))
    return summary, df


def format_summary(s):
    lines = [
        f"{s['checkpoint']}  depth {s['depth']}  {s['games']} games  ({s['workers']} workers, {s['wall_s']:.1f} s)",
        f"  score   avg {s['avg_score']:,.0f}  ± {s['std_score']:,.0f}   median {s['median_score']:,.0f}   max {s['max_score']:,}",
        "  reach   " + "   ".join(f"{t}: {s[f'reach_{t}']:.1f}%" for t in REACH_TILES),
        "  max tile " + "  ".join(f"{t}: {p:.1f}%" for t, p in s["max_tile_pct"].items()),
        f"  time    {s['ms_per_move']:.3f} ms/move (per worker)",
    ]
    return "\n".join(lines)


def main(argv=None):
    p = argparse.ArgumentParser(description="Evaluate an n-tuple checkpoint.")
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--run", help="run name under checkpoints/")
    src.add_argument("--checkpoint", type=Path, help="checkpoint directory")
    p.add_argument("--games", type=int, default=100)
    p.add_argument("--depth", type=int, default=0, help="0 = greedy, >0 = expectimax chance layers")
    p.add_argument("--workers", type=int, default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--prune", type=float, default=1e-4, help="reach-probability pruning threshold")
    p.add_argument("--tt-bits", type=int, default=20, help="transposition table size, log2 entries")
    args = p.parse_args(argv)
    ckpt = args.checkpoint or CHECKPOINT_ROOT / args.run
    evaluate(ckpt, args.games, args.depth, args.workers, args.seed, args.prune, args.tt_bits)


if __name__ == "__main__":
    main()
