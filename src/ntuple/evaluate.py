"""Evaluate policies on n-tuple checkpoints: one CSV row per game, resumable, in parallel.

    uv run python -m ntuple.evaluate --policy random --games 10000
    uv run python -m ntuple.evaluate --runs RUN1,RUN2 [--checkpoints final|all|games-0001000000,...] \\
                                     --policy greedy --games 10000
    uv run python -m ntuple.evaluate --run RUN --policy expectimax --depth 2 --games 2000
    uv run python -m ntuple.evaluate --checkpoint checkpoints/RUN/milestones/games-0001000000 --games 1000
    uv run python -m ntuple.evaluate --bench --run RUN [--bench-policies greedy,d1,d2,d3]

Policies: random, greedy (= expectimax depth 0, the training policy) and expectimax --depth N
(N chance layers; see expectimax.py). As before, --depth 0 alone means greedy and --depth N >= 1
alone means expectimax.

Game i uses seed base_seed + i for every policy and checkpoint (common random numbers, player.py).
Rows go to results/eval/<run>/<checkpoint>/<label>.csv (random: results/eval/random/random.csv),
appended as games finish. Rerunning skips seeds already in the file, so an interrupted evaluation
resumes; a file never mixes settings (games_trained, prune, tt_bits). Afterwards
results/eval_summary.csv is rebuilt from every CSV under results/eval/.

Workers memory-map the checkpoint copy-on-write, so all processes share one physical copy of the
268 MB weights; each worker adds ~0.2 GB plus its transposition table (24 B x 2**tt_bits).

--bench times policies in this single process, which macOS keeps on a performance core (parallel
per-game timings depend on P/E-core placement): positions from one greedy game at 0/25/50/75/95%
of its length, `--bench-moves` moves from each with a fresh table. Written to
results/eval/<run>/<checkpoint>/bench.csv.
"""

import argparse
import csv
import dataclasses
import json
import multiprocessing as mp
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd

from analysis import stats as S
from game import engine as E
from game.replay import board_hex, trajectory
from ntuple.player import Player, parse_label, policy_label, resolve_settings
from ntuple.train import CHECKPOINT_ROOT, REACH_TILES, ROOT, load_checkpoint

EVAL_ROOT = ROOT / "results" / "eval"
SUMMARY_PATH = ROOT / "results" / "eval_summary.csv"
CSV_FIELDS = ["run", "checkpoint", "games_trained", "policy", "depth", "seed", "score", "max_tile",
              "max_tile_pos", "max_tile_in_corner", "corner_frac", "corner_stable_frac", "moves",
              "wall_s", "ms_per_move", "final_board", "n_workers", "prune", "tt_bits"]
COLUMN_TYPES = {"run": str, "checkpoint": str, "games_trained": np.int64, "policy": str,
                "depth": np.int64, "seed": np.int64, "score": np.int64, "max_tile": np.int64,
                "max_tile_pos": np.int64, "max_tile_in_corner": bool, "corner_frac": np.float64,
                "corner_stable_frac": np.float64, "moves": np.int64,
                "wall_s": np.float64, "ms_per_move": np.float64, "final_board": str,
                "n_workers": np.int64, "prune": np.float64, "tt_bits": np.int64}
SETTINGS = ("run", "checkpoint", "games_trained", "policy", "depth", "prune", "tt_bits")
CORNERS = (0, 3, 12, 15)
DEFAULT_WORKERS = 8
MEMORY_LIMIT_GB = 8.0
BENCH_STAGES = (0.0, 0.25, 0.5, 0.75, 0.95)
FLUSH_SECONDS = 2.0


# ---------------------------------------------------------------- targets


@dataclasses.dataclass(frozen=True)
class Target:
    """What to evaluate: a checkpoint directory, or nothing for the random policy."""
    run: str
    checkpoint: str  # "final", a milestone directory name ("games-0001000000"), or "-"
    path: Path | None
    games_trained: int


RANDOM_TARGET = Target("-", "-", None, 0)


def target_from_path(path):
    """Target for a run directory (checkpoint "final") or a milestones/games-* directory."""
    path = Path(path)
    meta = load_checkpoint_meta(path)
    if path.parent.name == "milestones":
        return Target(path.parent.parent.name, path.name, path, meta["games_done"])
    return Target(path.name, "final", path, meta["games_done"])


def load_checkpoint_meta(path):
    return json.loads((Path(path) / "meta.json").read_text())


def list_checkpoints(run, spec="final", root=CHECKPOINT_ROOT):
    """Targets for `run`: spec "final", "all" (milestones in order, then final) or a comma list."""
    run_dir = Path(root) / run
    if spec == "all":
        names = sorted(p.name for p in (run_dir / "milestones").glob("games-*")) + ["final"]
    else:
        names = [s.strip() for s in spec.split(",") if s.strip()]
    return [target_from_path(run_dir if n == "final" else run_dir / "milestones" / n) for n in names]


def csv_path(target, label, out=EVAL_ROOT):
    if target.path is None:
        return Path(out) / "random" / "random.csv"
    return Path(out) / target.run / target.checkpoint / f"{label}.csv"


# ---------------------------------------------------------------- per-game rows


def max_tile_position(board):
    """(cell of the max tile, whether a max tile is in a corner). Ties prefer a corner, then the
    lowest cell index."""
    cells = E.decode(board).ravel()
    best = np.flatnonzero(cells == cells.max())
    corners = [c for c in best if c in CORNERS]
    return int(corners[0] if corners else best[0]), bool(corners)


def _repair_tail(path):
    """Drop a torn last line (a crash in the middle of a write)."""
    data = path.read_bytes()
    if data and not data.endswith(b"\n"):
        path.write_bytes(data[:data.rfind(b"\n") + 1])


def typed(df):
    """Per-game rows with the column types of COLUMN_TYPES, whether read from CSV or just played."""
    df = df.copy()
    if df["max_tile_in_corner"].dtype != bool:
        df["max_tile_in_corner"] = df["max_tile_in_corner"].astype(str).isin(["True", "true", "1"])
    return df.astype(COLUMN_TYPES)


def read_results(path):
    """Per-game rows of one CSV (empty DataFrame with CSV_FIELDS if the file does not exist)."""
    path = Path(path)
    if not path.exists() or path.stat().st_size == 0:
        return typed(pd.DataFrame(columns=CSV_FIELDS))
    _repair_tail(path)
    return typed(pd.read_csv(path, dtype=str, keep_default_na=False))


def estimated_memory_gb(workers, tt_bits):
    return 0.27 + workers * (0.2 + 24 * 2**tt_bits / 1e9)


def check_memory(workers, tt_bits, limit=MEMORY_LIMIT_GB):
    need = estimated_memory_gb(workers, tt_bits)
    if need > limit:
        raise ValueError(f"{workers} workers with tt_bits {tt_bits} need ~{need:.1f} GB "
                         f"(limit {limit:g} GB): use fewer workers or a smaller --tt-bits")


_worker = {}


def _init_worker(ckpt_dir, policy, depth, prune, tt_bits):
    W = load_checkpoint(ckpt_dir, mmap=True)[0] if ckpt_dir else None
    _worker["player"] = Player(W, policy, depth, prune, tt_bits)


def _play(seed):
    rec = _worker["player"].play(seed, record=False)
    pos, corner = max_tile_position(rec.final_board)
    return {"seed": seed, "score": rec.score, "max_tile": rec.max_tile, "max_tile_pos": pos,
            "max_tile_in_corner": corner, "corner_frac": rec.corner_frac,
            "corner_stable_frac": rec.corner_stable_frac, "moves": rec.moves, "wall_s": rec.wall_s,
            "ms_per_move": rec.wall_s / rec.moves * 1e3 if rec.moves else 0.0,
            "final_board": board_hex(rec.final_board)}


def evaluate_target(target, policy, depth=0, games=100, base_seed=0, workers=None, prune=None,
                    tt_bits=None, out_csv=None, verbose=True):
    """Play seeds base_seed .. base_seed + games - 1 that `out_csv` does not have yet.

    Returns the rows of all requested seeds (old and new), sorted by seed.
    """
    prune, tt_bits = resolve_settings(policy, depth, prune, tt_bits)
    if policy != "random" and target.path is None:
        raise ValueError(f"{policy} needs a checkpoint")
    workers = workers or min(DEFAULT_WORKERS, os.cpu_count())
    fixed = {"run": target.run, "checkpoint": target.checkpoint, "games_trained": target.games_trained,
             "policy": policy, "depth": depth, "prune": prune, "tt_bits": tt_bits}
    old = read_results(out_csv) if out_csv else typed(pd.DataFrame(columns=CSV_FIELDS))
    for k in SETTINGS:
        found = set(old[k].tolist())
        if found and found != {fixed[k]}:
            raise ValueError(f"{out_csv} has {k} = {sorted(found)}, this evaluation uses {fixed[k]!r}: "
                             "move the file away or use the same settings")
    seeds = range(base_seed, base_seed + games)
    have = set(old["seed"].tolist())
    todo = [s for s in seeds if s not in have]
    name = policy_label(policy, depth)
    if target.path is not None:
        name = f"{target.run} {target.checkpoint} {name}"
    if verbose:
        print(f"{name}: {len(todo):,} of {games:,} games to play"
              + (f" ({len(have & set(seeds)):,} already in {out_csv})" if out_csv else ""), flush=True)
    new = []
    if todo:
        check_memory(workers, tt_bits)
        new = _run_pool(target, policy, depth, prune, tt_bits, todo, workers, fixed, out_csv, verbose)
    if new:
        old = pd.concat([old, typed(pd.DataFrame(new, columns=CSV_FIELDS))], ignore_index=True)
    return old[old["seed"].isin(seeds)].sort_values("seed").reset_index(drop=True)


def _run_pool(target, policy, depth, prune, tt_bits, todo, workers, fixed, out_csv, verbose):
    workers = min(workers, len(todo))
    # cheap games: batch tasks to keep the IPC overhead small; expectimax: one game per task
    chunk = 1 if policy == "expectimax" else max(1, min(64, len(todo) // (workers * 8)))
    ckpt = str(target.path) if target.path else None
    rows, pending = [], []
    t0 = last_flush = time.perf_counter()
    f = writer = None
    if out_csv:
        out_csv = Path(out_csv)
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        new_file = not out_csv.exists() or out_csv.stat().st_size == 0
        f = open(out_csv, "a", newline="")
        writer = csv.DictWriter(f, CSV_FIELDS)
        if new_file:
            writer.writeheader()
    try:
        ctx = mp.get_context("spawn")
        with ctx.Pool(workers, _init_worker, (ckpt, policy, depth, prune, tt_bits)) as pool:
            for k, r in enumerate(pool.imap_unordered(_play, todo, chunksize=chunk), 1):
                row = {**fixed, **r, "n_workers": workers}
                rows.append(row)
                pending.append(row)
                now = time.perf_counter()
                if writer and (now - last_flush >= FLUSH_SECONDS or k == len(todo)):
                    writer.writerows(pending)
                    f.flush()
                    pending, last_flush = [], now
                if verbose and (k % max(1, len(todo) // 10) == 0 or k == len(todo)):
                    print(f"  {k:,}/{len(todo):,} games, {now - t0:.1f} s", flush=True)
    finally:
        if writer and pending:
            writer.writerows(pending)
        if f:
            f.close()
    return rows


def format_per_move(ms):
    """Time per move with useful precision: µs below 0.1 ms."""
    return f"{ms * 1e3:.2f} µs/move" if ms < 0.1 else f"{ms:.3f} ms/move"


def summary_line(df):
    s = S.summarize(df)
    return (f"  {s['games']:,} games  score {s['mean_score']:,.0f} [{s['mean_lo']:,.0f}, {s['mean_hi']:,.0f}]"
            f"  median {s['median_score']:,.0f}  max {s['max_score']:,}  "
            + "  ".join(f"{t}: {s[f'reach_{t}']:.1f}%" for t in S.REACH_TILES)
            + f"  corner {100 * s['corner_frac']:.1f}% of moves  {format_per_move(s['ms_per_move'])}")


# ---------------------------------------------------------------- notebook API (unchanged)


def evaluate(ckpt_dir, games=100, depth=0, workers=None, seed=0, prune=1e-4, tt_bits=20,
             verbose=True, policy=None, out_csv=None):
    """Play `games` games; returns (summary dict, per-game DataFrame).

    Kept for notebook 02. Nothing is written unless out_csv is given.
    """
    policy = policy or ("greedy" if depth == 0 else "expectimax")
    target = RANDOM_TARGET if policy == "random" else target_from_path(ckpt_dir)
    t0 = time.perf_counter()
    df = evaluate_target(target, policy, depth, games, seed, workers, prune, tt_bits, out_csv, verbose)
    wall = time.perf_counter() - t0
    tiles = df["max_tile"].value_counts().sort_index(ascending=False)
    summary = {
        "checkpoint": str(ckpt_dir),
        "depth": depth,
        "games": len(df),
        "avg_score": float(df["score"].mean()),
        "std_score": float(df["score"].std()),
        "median_score": float(df["score"].median()),
        "max_score": int(df["score"].max()),
        **{f"reach_{t}": float((df["max_tile"] >= t).mean() * 100) for t in REACH_TILES},
        "max_tile_pct": {int(t): float(c / len(df) * 100) for t, c in tiles.items()},
        "ms_per_move": float(df["wall_s"].sum() / df["moves"].sum() * 1e3),
        "wall_s": wall,
        "workers": int(df["n_workers"].max()),
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
        f"  time    {format_per_move(s['ms_per_move'])} (per worker)",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------- single-process benchmark


def benchmark(target, labels=("greedy", "d1", "d2", "d3"), moves=None, seed=0, stages=BENCH_STAGES,
              prune=None, tt_bits=None, out=EVAL_ROOT, verbose=True, ref_games=21):
    """ms/move per policy and game stage, single process. Returns (and writes) the rows.

    Positions come from a typical greedy game: the median-length one of seeds seed .. seed +
    ref_games - 1 (a single seed can die early and give only early-game positions). Each segment
    plays `moves` moves; by default 100 for depth >= 2 and 2,000 for the cheap policies, whose
    100 moves would take ~50 µs and be dominated by per-call overhead.
    """
    W, _ = load_checkpoint(target.path)  # in RAM, so page faults do not count as search time
    greedy = Player(W, "greedy")
    lengths = sorted((greedy.play(s, record=False).moves, s) for s in range(seed, seed + ref_games))
    ref = greedy.play(lengths[len(lengths) // 2][1])
    boards = trajectory(ref.init, ref.actions, ref.spawns)
    rows = []
    for label in labels:
        policy, depth = parse_label(label)
        player = Player(W, policy, depth, prune, tt_bits)
        player.play(ref.seed, max_moves=1)  # load the compiled code before timing
        n_moves = moves or (100 if depth >= 2 else 2000)
        for stage in stages:
            k = int(stage * ref.moves)
            rec = player.play(ref.seed, start_board=int(boards[k]), max_moves=n_moves, record=False)
            rows.append({"run": target.run, "checkpoint": target.checkpoint,
                         "games_trained": target.games_trained, "policy": policy, "depth": depth,
                         "prune": player.prune, "tt_bits": player.tt_bits, "stage": stage,
                         "start_move": k, "moves": rec.moves, "wall_s": rec.wall_s,
                         "ms_per_move": rec.wall_s / max(1, rec.moves) * 1e3})
            if verbose:
                r = rows[-1]
                print(f"  {policy_label(policy, depth):14s} stage {stage:4.0%} (move {k:>6,})  "
                      f"{r['moves']:>4} moves  {format_per_move(r['ms_per_move']):>14s}", flush=True)
    df = pd.DataFrame(rows)
    if out:
        path = Path(out) / target.run / target.checkpoint / "bench.csv"
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(path, index=False)
    if verbose:
        print(f"reference greedy game: seed {ref.seed} (median length of {ref_games}), "
              f"{ref.moves:,} moves, score {ref.score:,}")
        for (policy, depth), g in df.groupby(["policy", "depth"], sort=False):
            print(f"  {policy_label(policy, depth):14s} mean {format_per_move(g['wall_s'].sum() / g['moves'].sum() * 1e3):>14s}")
    return df


# ---------------------------------------------------------------- CLI


def _targets(args):
    if args.policy == "random":
        return [RANDOM_TARGET]
    if args.checkpoint:
        return [target_from_path(args.checkpoint)]
    runs = [r.strip() for r in (args.runs or args.run or "").split(",") if r.strip()]
    if not runs:
        raise SystemExit("give --run, --runs or --checkpoint (not needed for --policy random)")
    return [t for run in runs for t in list_checkpoints(run, args.checkpoints)]


def main(argv=None):
    p = argparse.ArgumentParser(description="Evaluate policies on n-tuple checkpoints.")
    p.add_argument("--run", help="run name under checkpoints/")
    p.add_argument("--runs", help="comma-separated run names")
    p.add_argument("--checkpoint", type=Path, help="checkpoint directory (run dir or milestone dir)")
    p.add_argument("--checkpoints", default="final",
                   help="for --run/--runs: final (default), all, or milestone names games-...,...")
    p.add_argument("--policy", choices=["random", "greedy", "expectimax"])
    p.add_argument("--depth", type=int, help="expectimax chance layers (0 = greedy)")
    p.add_argument("--games", type=int, default=100)
    p.add_argument("--base-seed", "--seed", dest="base_seed", type=int, default=0,
                   help="game i uses seed base_seed + i")
    p.add_argument("--workers", type=int, default=None, help=f"default {DEFAULT_WORKERS}")
    p.add_argument("--prune", type=float, default=None, help="expectimax pruning threshold (1e-4)")
    p.add_argument("--tt-bits", type=int, default=None,
                   help="log2 transposition-table entries (default d1 18, d2 20, d3 22; max 24)")
    p.add_argument("--out", type=Path, default=EVAL_ROOT, help="results directory")
    p.add_argument("--no-summary", action="store_true", help=f"do not rebuild {SUMMARY_PATH.name}")
    p.add_argument("--bench", action="store_true", help="single-process ms/move benchmark")
    p.add_argument("--bench-policies", default="greedy,d1,d2,d3")
    p.add_argument("--bench-moves", type=int, default=None,
                   help="moves per position (default 100 for depth >= 2, 2000 otherwise)")
    args = p.parse_args(argv)

    if args.bench:
        for target in _targets(args):
            benchmark(target, args.bench_policies.split(","), args.bench_moves, args.base_seed,
                      prune=args.prune, tt_bits=args.tt_bits, out=args.out)
    else:
        if args.policy is None:
            args.policy = "greedy" if not args.depth else "expectimax"
        depth = args.depth or 0
        if args.policy == "expectimax" and depth < 1:
            raise SystemExit("--policy expectimax needs --depth >= 1")
        for target in _targets(args):
            path = csv_path(target, policy_label(args.policy, depth), args.out)
            t0 = time.perf_counter()
            df = evaluate_target(target, args.policy, depth, args.games, args.base_seed, args.workers,
                                 args.prune, args.tt_bits, path)
            print(summary_line(df) + f"  ({time.perf_counter() - t0:.1f} s)", flush=True)
    if not args.no_summary:
        from analysis import report

        print(f"summary: {report.write_summary(args.out)}")


if __name__ == "__main__":
    main()
