"""TD(0) afterstate training of the n-tuple network.

    uv run python -m ntuple.train --run NAME --games N [--alpha 0.1] [--seed 0]
                                  [--update-order online|backward] [--hours H] [--resume]

Move choice: argmax over legal moves of reward + V(afterstate).
TD(0) on afterstates: delta = r_next + V(s'_next) - V(s'), terminal target 0, each of the
32 weights used by s' += alpha * delta / 32.

update order
  online    update the previous afterstate after every move (Szubert & Jaskowski 2014)
  backward  play the whole game, then apply the same TD(0) updates from the last afterstate
            to the first, so each target uses the freshly updated V(s'_next) (moporgic/TDL2048)

Games run inside Numba in chunks of at most CHUNK_GAMES. Chunks end exactly on log,
checkpoint and alpha-decay boundaries, and weights + RNG carry over between chunks, so a run
that is stopped and resumed is bit-identical to an uninterrupted one. A checkpoint is saved
every checkpoint_every games or checkpoint_minutes minutes, whichever comes first. Ctrl+C /
SIGTERM / SIGHUP (closing the terminal window) finish the current chunk, save a checkpoint
and exit.

Checkpoint (checkpoints/NAME/): weights-<games>.npy + meta.json (config, game count, RNG
state, pending log window). The weights file is written first, then meta.json is atomically
replaced to point at it, so a crash mid-save always leaves a consistent checkpoint.
Log (logs/NAME.csv): one row per log_every games.
"""

import argparse
import csv
import dataclasses
import json
import os
import platform
import resource
import signal
import sys
import time
from pathlib import Path

import numpy as np
from numba import njit

from game import engine as E
from ntuple import model as M

ROOT = Path(__file__).resolve().parents[2]
CHECKPOINT_ROOT = ROOT / "checkpoints"
LOG_ROOT = ROOT / "logs"
CHUNK_GAMES = 100
# every move spawns >= 2 points of tile value and a board holds < 16 * 2**15, so no game is longer
MAX_GAME_MOVES = 1 << 18
CHECKPOINT_VERSION = 1
REACH_TILES = (2048, 4096, 8192, 16384)
LOG_FIELDS = (
    ["games", "moves", "alpha", "avg_score", "max_score", "avg_moves"]
    + [f"reach_{t}" for t in REACH_TILES]
    + ["games_per_s", "moves_per_s", "elapsed_s"]
)
TIMING_FIELDS = ("games_per_s", "moves_per_s", "elapsed_s")


def _say(*args):
    """print() that never raises: once the terminal window is closed (SIGHUP) stdout is gone,
    and a failing print must not stop the final checkpoint from being saved."""
    try:
        print(*args, flush=True)
    except (OSError, ValueError):
        sys.stdout = open(os.devnull, "w")  # also avoids a failing flush at interpreter exit


@dataclasses.dataclass
class Config:
    run: str
    games: int
    alpha: float = 0.1
    seed: int = 0
    update_order: str = "online"  # tied with backward per wall-clock second; see TRAINING.md
    checkpoint_every: int = 20_000  # save every this many games ...
    checkpoint_minutes: float = 20.0  # ... or after this many minutes, whichever comes first (0 = off)
    log_every: int = 1_000
    decay_factor: float = 1.0
    decay_every: int = 0  # 0 = constant alpha
    keep_every: int = 0  # keep a weights snapshot every this many games (0 = only the latest)

    def __post_init__(self):
        if self.update_order not in ("online", "backward"):
            raise ValueError(f"update_order must be 'online' or 'backward', got {self.update_order!r}")
        for name in ("games", "checkpoint_every", "log_every"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")

    def alpha_at(self, games_done):
        """Step decay: alpha * decay_factor ** (games_done // decay_every)."""
        if self.decay_every <= 0:
            return self.alpha
        return self.alpha * self.decay_factor ** (games_done // self.decay_every)


# ---------------------------------------------------------------- Numba hot loop


@njit(
    f"void({M.W_TYPE}, uint64[:], float64, boolean, int64[:], int64[:], int64[:])",
    cache=True,
)
def train_games(W, rng, alpha, backward, scores, max_tiles, moves):
    """Play len(scores) training games, updating W in place. Writes per-game stats."""
    step = alpha / M.NUM_LOOKUPS
    afters = np.empty(MAX_GAME_MOVES, np.uint64)
    rewards = np.empty(MAX_GAME_MOVES, np.int64)
    for g in range(scores.shape[0]):
        board = E.new_game(rng)
        score = 0
        n = 0
        while True:
            a, after, r, v = M.greedy_move(W, board)
            if a < 0:
                break
            if not backward and n > 0:
                prev = afters[n - 1]
                M.update(W, prev, step * (r + v - M.value(W, prev)))
            afters[n] = after
            rewards[n] = r
            n += 1
            score += r
            board = E.spawn_tile(after, rng)
        if backward:
            target = 0.0
            for t in range(n - 1, -1, -1):
                s = afters[t]
                M.update(W, s, step * (target - M.value(W, s)))
                target = rewards[t] + M.value(W, s)
        elif n > 0:
            prev = afters[n - 1]
            M.update(W, prev, step * (0.0 - M.value(W, prev)))
        scores[g] = score
        max_tiles[g] = E.max_tile(board)
        moves[g] = n


# ---------------------------------------------------------------- checkpoints


def _write_atomic(path, write):
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "wb") as f:
        write(f)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def save_checkpoint(ckpt_dir, W, meta, keep_every=0):
    """Write weights-<games>.npy, then atomically point meta.json at it, then prune old weights."""
    ckpt_dir = Path(ckpt_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    games = meta["games_done"]
    weights_file = f"weights-{games:010d}.npy"
    _write_atomic(ckpt_dir / weights_file, lambda f: np.save(f, W))
    meta = {**meta, "weights_file": weights_file, "version": CHECKPOINT_VERSION}
    _write_atomic(ckpt_dir / "meta.json", lambda f: f.write(json.dumps(meta, indent=1).encode()))
    for old in ckpt_dir.glob("weights-*.npy"):
        g = int(old.stem.split("-")[1])
        if old.name != weights_file and not (keep_every and g % keep_every == 0):
            old.unlink()


def load_checkpoint(ckpt_dir, mmap=False):
    """Returns (weights, meta). mmap=True maps the file copy-on-write: processes share pages."""
    ckpt_dir = Path(ckpt_dir)
    meta = json.loads((ckpt_dir / "meta.json").read_text())
    W = np.load(ckpt_dir / meta["weights_file"], mmap_mode="c" if mmap else None)
    if mmap:
        W = np.ndarray(W.shape, W.dtype, buffer=W)  # plain ndarray view, still file-backed
    assert W.dtype == np.float32 and W.shape == (M.NUM_TUPLES, M.TABLE_SIZE) and W.flags.c_contiguous
    return W, meta


def read_log(path):
    """Training log as a pandas DataFrame."""
    import pandas as pd

    return pd.read_csv(path)


def _truncate_log(log_path, games_done):
    """Drop rows written after the checkpoint we resume from."""
    if not log_path.exists():
        return
    with open(log_path, newline="") as f:
        rows = [r for r in csv.DictReader(f) if int(r["games"]) <= games_done]
    with open(log_path, "w", newline="") as f:
        w = csv.DictWriter(f, LOG_FIELDS)
        w.writeheader()
        w.writerows(rows)


# ---------------------------------------------------------------- training loop


class _StopFlag:
    """First signal: finish the chunk, checkpoint, exit. Second Ctrl+C: abort immediately."""

    SIGNALS = (signal.SIGINT, signal.SIGTERM, signal.SIGHUP)

    def __init__(self):
        self.requested = False
        self._old = {}

    def _handle(self, signum, frame):
        if self.requested:
            # only a deliberate second Ctrl+C aborts; closing the window (SIGHUP) or SIGTERM
            # while the checkpoint is being written must not interrupt the save
            if signum == signal.SIGINT:
                raise KeyboardInterrupt
            return
        self.requested = True
        _say(f"\n{signal.Signals(signum).name}: saving checkpoint after the current chunk "
             "(press Ctrl+C again to abort without saving)")

    def __enter__(self):
        try:
            for s in self.SIGNALS:
                self._old[s] = signal.signal(s, self._handle)
        except ValueError:  # not in the main thread: no signal handling
            pass
        return self

    def __exit__(self, *exc):
        for s, h in self._old.items():
            signal.signal(s, h)


def _next_boundary(done, every):
    return every - done % every if every > 0 else sys.maxsize


def train(cfg=None, *, resume_run=None, games=None, hours=None,
          checkpoint_root=CHECKPOINT_ROOT, log_root=LOG_ROOT, verbose=True):
    """Train a new run (cfg) or resume one (resume_run=NAME, optional new total `games`).

    Returns a summary dict. Always saves a checkpoint before returning.
    """
    if resume_run is not None:
        ckpt_dir = Path(checkpoint_root) / resume_run
        W, meta = load_checkpoint(ckpt_dir)
        W = np.ascontiguousarray(W)
        cfg = Config(**meta["config"])
        if games is not None:
            cfg.games = games
        state = meta
        rng = np.array([state["rng_state"]], dtype=np.uint64)
        log_path = Path(log_root) / f"{cfg.run}.csv"
        _truncate_log(log_path, state["games_done"])
    else:
        ckpt_dir = Path(checkpoint_root) / cfg.run
        log_path = Path(log_root) / f"{cfg.run}.csv"
        if (ckpt_dir / "meta.json").exists() or log_path.exists():
            raise FileExistsError(f"run {cfg.run!r} already exists; resume it or pick another name")
        W = M.new_weights()
        rng = E.make_rng(cfg.seed)
        state = {"games_done": 0, "total_moves": 0, "elapsed_s": 0.0,
                 "window": {"scores": [], "max_tiles": [], "moves": [], "train_s": 0.0}}
    log_path.parent.mkdir(parents=True, exist_ok=True)
    if not log_path.exists():
        with open(log_path, "w", newline="") as f:
            csv.DictWriter(f, LOG_FIELDS).writeheader()

    games_done = state["games_done"]
    total_moves = state["total_moves"]
    window = state["window"]
    elapsed0 = state["elapsed_s"]
    session_start = time.perf_counter()
    last_save = session_start
    stop_reason = "done"
    backward = cfg.update_order == "backward"

    def make_meta():
        return {
            "config": dataclasses.asdict(cfg),
            "games_done": games_done,
            "total_moves": total_moves,
            "rng_state": int(rng[0]),
            "elapsed_s": elapsed0 + time.perf_counter() - session_start,
            "window": window,
        }

    if verbose:
        _say(f"run {cfg.run}: {games_done:,} -> {cfg.games:,} games, alpha {cfg.alpha_at(games_done)}, "
              f"{cfg.update_order} updates, weights {W.nbytes / 2**20:.0f} MiB")

    with _StopFlag() as stop:
        while games_done < cfg.games:
            if stop.requested:
                stop_reason = "interrupted"
                break
            if hours is not None and time.perf_counter() - session_start > hours * 3600:
                stop_reason = "time limit"
                break
            n = min(CHUNK_GAMES, cfg.games - games_done,
                    _next_boundary(games_done, cfg.log_every),
                    _next_boundary(games_done, cfg.checkpoint_every),
                    _next_boundary(games_done, cfg.decay_every))
            scores = np.zeros(n, np.int64)
            tiles = np.zeros(n, np.int64)
            moves = np.zeros(n, np.int64)
            t0 = time.perf_counter()
            train_games(W, rng, cfg.alpha_at(games_done), backward, scores, tiles, moves)
            window["train_s"] += time.perf_counter() - t0
            window["scores"] += scores.tolist()
            window["max_tiles"] += tiles.tolist()
            window["moves"] += moves.tolist()
            games_done += n
            total_moves += int(moves.sum())

            if games_done % cfg.log_every == 0:
                row = _log_row(cfg, games_done, total_moves, window,
                               elapsed0 + time.perf_counter() - session_start)
                with open(log_path, "a", newline="") as f:
                    csv.DictWriter(f, LOG_FIELDS).writerow(row)
                if verbose:
                    _print_row(row)
                window = {"scores": [], "max_tiles": [], "moves": [], "train_s": 0.0}
            # chunking never changes the result, so saving at a time-triggered chunk boundary
            # resumes exactly like saving on a game-count boundary
            now = time.perf_counter()
            due_by_time = cfg.checkpoint_minutes > 0 and now - last_save >= cfg.checkpoint_minutes * 60
            if games_done % cfg.checkpoint_every == 0 or due_by_time:
                save_checkpoint(ckpt_dir, W, make_meta(), cfg.keep_every)
                last_save = time.perf_counter()

        meta = make_meta()
        save_checkpoint(ckpt_dir, W, meta, cfg.keep_every)

    peak_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss  # bytes on macOS
    summary = {"run": cfg.run, "stop_reason": stop_reason, "games_done": games_done,
               "total_moves": total_moves, "elapsed_s": meta["elapsed_s"],
               "session_s": time.perf_counter() - session_start, "peak_rss_mb": peak_rss / 2**20,
               "checkpoint": str(ckpt_dir), "log": str(log_path)}
    if verbose:
        _say(f"{stop_reason}: {games_done:,} games, checkpoint saved to {ckpt_dir}")
    return summary


def _log_row(cfg, games_done, total_moves, window, elapsed_s):
    scores = np.array(window["scores"])
    tiles = np.array(window["max_tiles"])
    moves = np.array(window["moves"])
    train_s = max(window["train_s"], 1e-9)
    row = {
        "games": games_done,
        "moves": total_moves,
        "alpha": cfg.alpha_at(games_done - 1),
        "avg_score": round(float(scores.mean()), 1),
        "max_score": int(scores.max()),
        "avg_moves": round(float(moves.mean()), 1),
    }
    for t in REACH_TILES:
        row[f"reach_{t}"] = round(100.0 * float((tiles >= t.bit_length() - 1).mean()), 2)
    row["games_per_s"] = round(len(scores) / train_s, 1)
    row["moves_per_s"] = round(float(moves.sum()) / train_s)
    row["elapsed_s"] = round(elapsed_s, 1)
    return row


def _print_row(row):
    _say(f"games {row['games']:>10,}  avg {row['avg_score']:>9,.0f}  max {row['max_score']:>7,}  "
          f"2048 {row['reach_2048']:5.1f}%  4096 {row['reach_4096']:5.1f}%  8192 {row['reach_8192']:5.1f}%  "
          f"{row['games_per_s']:>6,.0f} g/s  {row['moves_per_s'] / 1e3:>5,.0f}k mv/s  "
          f"{row['elapsed_s'] / 3600:5.2f} h")


# ---------------------------------------------------------------- CLI


def main(argv=None):
    p = argparse.ArgumentParser(description="TD(0) afterstate training of the 4x6-tuple network.")
    p.add_argument("--run", required=True, help="run name: checkpoints/RUN/ and logs/RUN.csv")
    p.add_argument("--games", type=int, help="total number of training games (target)")
    p.add_argument("--alpha", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--update-order", choices=["online", "backward"], default="online")
    p.add_argument("--checkpoint-every", type=int, default=20_000,
                   help="save a checkpoint every N games ...")
    p.add_argument("--checkpoint-minutes", type=float, default=20.0,
                   help="... or every M minutes, whichever comes first (0 = only by games)")
    p.add_argument("--log-every", type=int, default=1_000)
    p.add_argument("--decay-factor", type=float, default=1.0, help="multiply alpha by this ...")
    p.add_argument("--decay-every", type=int, default=0, help="... every this many games (0 = off)")
    p.add_argument("--keep-every", type=int, default=0, help="keep a weights snapshot every N games")
    p.add_argument("--hours", type=float, help="stop (with checkpoint) after this many hours")
    p.add_argument("--resume", action="store_true",
                   help="continue RUN from its checkpoint; only --games and --hours apply")
    p.add_argument("--checkpoint-root", type=Path, default=CHECKPOINT_ROOT)
    p.add_argument("--log-root", type=Path, default=LOG_ROOT)
    args = p.parse_args(argv)

    if platform.machine() != "arm64":
        sys.exit(f"refusing to train on {platform.machine()}: run native arm64 Python (uv run)")
    roots = {"checkpoint_root": args.checkpoint_root, "log_root": args.log_root}
    if args.resume:
        summary = train(resume_run=args.run, games=args.games, hours=args.hours, **roots)
    else:
        if args.games is None:
            p.error("--games is required for a new run")
        cfg = Config(run=args.run, games=args.games, alpha=args.alpha, seed=args.seed,
                     update_order=args.update_order, checkpoint_every=args.checkpoint_every,
                     checkpoint_minutes=args.checkpoint_minutes, log_every=args.log_every, decay_factor=args.decay_factor,
                     decay_every=args.decay_every, keep_every=args.keep_every)
        summary = train(cfg, hours=args.hours, **roots)
    _say(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
