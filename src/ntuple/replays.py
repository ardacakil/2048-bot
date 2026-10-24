"""Replay files from evaluation results: pick games, regenerate them from their seed, write JSON.

    uv run python -m ntuple.replays --run RUN [--checkpoint final]     # the default selection
    uv run python -m ntuple.replays --csv results/eval/RUN/final/greedy.csv --select best|median|seed=N
    uv run python -m ntuple.replays --verify results/replays/*.json

Default selection for a run's checkpoint: the best game by score for random and for every policy
evaluated on it, plus the median-score game for greedy and for the expectimax policy with the
highest mean score. "Median game" = the game whose score is closest to the median (lowest seed on
ties); "best" = highest score (lowest seed on ties).

A game is regenerated with the same policy, depth, prune, tt_bits and seed; it must reproduce the
score, move count and final board in the CSV exactly, and the written file is then verified by
running it through the engine (game.replay.replay).
"""

import argparse
import datetime
from pathlib import Path

from game import replay as R
from ntuple import evaluate as V
from ntuple.player import Player, policy_label
from ntuple.train import CHECKPOINT_ROOT, ROOT, load_checkpoint

REPLAY_ROOT = ROOT / "results" / "replays"


def pick(df, how):
    """Row of `df` for how = "best", "median" or "seed=N"."""
    if how.startswith("seed="):
        rows = df[df["seed"] == int(how[5:])]
        if not len(rows):
            raise ValueError(f"seed {how[5:]} not in results")
        return rows.iloc[0]
    if how == "best":
        key = -df["score"]
    elif how == "median":
        key = (df["score"] - df["score"].median()).abs()
    else:
        raise ValueError(f"unknown selection {how!r}")
    return df.loc[df.assign(_key=key).sort_values(["_key", "seed"]).index[0]]


def checkpoint_dir(row, root=CHECKPOINT_ROOT):
    if row["policy"] == "random":
        return None
    run_dir = Path(root) / row["run"]
    return run_dir if row["checkpoint"] == "final" else run_dir / "milestones" / row["checkpoint"]


def regenerate(row, ckpt_root=CHECKPOINT_ROOT):
    """Replay the game of a CSV row from its seed; raises if it differs from the recorded result."""
    ckpt = checkpoint_dir(row, ckpt_root)
    W = load_checkpoint(ckpt, mmap=True)[0] if ckpt else None
    player = Player(W, row["policy"], int(row["depth"]), float(row["prune"]), int(row["tt_bits"]))
    rec = player.play(int(row["seed"]))
    got = (rec.score, rec.moves, R.board_hex(rec.final_board))
    want = (int(row["score"]), int(row["moves"]), row["final_board"])
    if got != want:
        raise RuntimeError(f"seed {row['seed']} did not reproduce: (score, moves, board) {got} != {want}")
    return rec


def replay_name(row, selection):
    name = f"{policy_label(row['policy'], int(row['depth']))}_{selection}_seed{int(row['seed'])}.json"
    return name if row["policy"] == "random" else f"{row['run']}_{row['checkpoint']}_{name}"


def write_replay(row, selection, out=REPLAY_ROOT, ckpt_root=CHECKPOINT_ROOT):
    """Regenerate, write and verify one replay; returns its path."""
    rec = regenerate(row, ckpt_root)
    meta = {"run": row["run"], "checkpoint": row["checkpoint"], "games_trained": int(row["games_trained"]),
            "policy": row["policy"], "depth": int(row["depth"]), "prune": float(row["prune"]),
            "tt_bits": int(row["tt_bits"]), "seed": rec.seed, "score": rec.score,
            "max_tile": rec.max_tile, "moves": rec.moves, "final_board": rec.final_board,
            "selection": selection,
            "created": datetime.datetime.now().isoformat(timespec="seconds")}
    doc = R.make_doc(meta, rec.init, rec.actions, rec.rewards, rec.spawns)
    path = R.write(doc, Path(out) / replay_name(row, selection))
    R.replay(R.load(path))
    return path


def default_selection(run, checkpoint="final", eval_root=V.EVAL_ROOT):
    """[(row, selection)] per the module docstring."""
    folder = Path(eval_root) / run / checkpoint
    results = {p.stem: V.read_results(p) for p in sorted(folder.glob("*.csv")) if p.name != "bench.csv"}
    results = {k: v for k, v in results.items() if len(v)}
    random = V.read_results(Path(eval_root) / "random" / "random.csv")
    if len(random):
        results = {"random": random, **results}
    picks = [(pick(df, "best"), "best") for df in results.values()]
    if "greedy" in results:
        picks.append((pick(results["greedy"], "median"), "median"))
    deep = {k: v for k, v in results.items() if k.startswith("expectimax-")}
    if deep:
        best = max(deep, key=lambda k: deep[k]["score"].mean())
        picks.append((pick(deep[best], "median"), "median"))
    return picks


def main(argv=None):
    p = argparse.ArgumentParser(description="Write replay JSON files from evaluation results.")
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--run", help="default selection for this run's checkpoint")
    src.add_argument("--csv", type=Path, help="one results CSV")
    src.add_argument("--verify", nargs="+", type=Path, help="replay files to check")
    p.add_argument("--checkpoint", default="final")
    p.add_argument("--select", default="best", help="with --csv: best, median or seed=N")
    p.add_argument("--out", type=Path, default=REPLAY_ROOT)
    p.add_argument("--eval-root", type=Path, default=V.EVAL_ROOT)
    args = p.parse_args(argv)
    if args.verify:
        for path in args.verify:
            board, score = R.replay(R.load(path))
            print(f"ok  {path}  score {score:,}  final {R.board_hex(board)}")
        return
    picks = (default_selection(args.run, args.checkpoint, args.eval_root) if args.run
             else [(pick(V.read_results(args.csv), args.select), args.select.replace("=", ""))])
    for row, selection in picks:
        path = write_replay(row, selection, args.out)
        print(f"{path}  score {int(row['score']):,}  {int(row['moves']):,} moves  "
              f"{path.stat().st_size / 1024:.0f} KB")


if __name__ == "__main__":
    main()
