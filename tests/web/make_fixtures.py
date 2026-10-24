"""SSE fixtures for the frontend tests: committed replays streamed exactly as /api/play would.

    uv run python tests/web/make_fixtures.py        # rewrites tests/web/fixtures/*.sse

Each fixture is built with server.wire (meta, queued, init, steps in batches of the runner's
256 moves, end) from a game in results/replays/, cut where the target first appears, like
ntuple.timing.moves_until. tests/test_web.py checks that the committed files regenerate
byte for byte, so the JavaScript tests always run against the current wire format.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from game import engine as E  # noqa: E402
from game import replay as R  # noqa: E402
from ntuple.timing import Move  # noqa: E402
from server import build_config, wire  # noqa: E402
from server.config import DEFAULT_DELAYS_MS, Settings  # noqa: E402

OUT = Path(__file__).resolve().parent / "fixtures"
REPLAYS = ROOT / "results" / "replays"

# name: (replay file, target, queued events before the game starts)
FIXTURES = {
    "reached-2048": ("overnight-a0025_final_expectimax-d2_best_seed312.json", 2048, True),
    "missed-16384": ("overnight-a0025_final_greedy_median_seed127.json", 16384, False),
}


def build(name):
    file, target, queued = FIXTURES[name]
    doc = R.load(REPLAYS / file)
    m = doc["meta"]
    actions = [s[0] for s in doc["steps"]]
    spawns = [(s[2], s[3]) for s in doc["steps"]]
    boards = R.trajectory(doc["init"], actions, spawns)
    n = next((k for k, b in enumerate(boards) if E.has_tile(int(b), target)), len(actions))
    moves = [Move(a, r, c, e, int(boards[k + 1]), 0.0)
             for k, (a, r, c, e) in enumerate(doc["steps"][:n])]
    model = {"run": m["run"], "checkpoint": m["checkpoint"], "games_trained": m["games_trained"]}
    reach = build_config.reach_rate(build_config.load(), m["depth"], target)

    out = [wire.sse("meta", wire.meta(f"fixture-{name}", m["seed"], target, m["depth"], m["prune"],
                                      m["tt_bits"], model, "fixture", DEFAULT_DELAYS_MS[target], reach))]
    if queued:
        out += [wire.sse("queued", wire.queued(2, 2, 3.5)), wire.KEEPALIVE,
                wire.sse("queued", wire.queued(1, 1, 1.2))]
    out.append(wire.sse("init", wire.init(int(boards[0]), doc["init"])))
    first, score, size = 1, 0, Settings().batch_moves
    for i in range(0, n, size):
        data, score = wire.steps(first, moves[i:i + size], score)
        first += len(moves[i:i + size])
        out.append(wire.sse("steps", data))
    board = int(boards[n])
    out.append(wire.sse("end", wire.end(E.has_tile(board, target), bool(E.is_game_over(board)), n, score,
                                        board, 0.25, 1.5 if queued else 0.0)))
    return b"".join(out)


def main():
    OUT.mkdir(exist_ok=True)
    for name in FIXTURES:
        path = OUT / f"{name}.sse"
        path.write_bytes(build(name))
        print(f"wrote {path.relative_to(ROOT)} ({path.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()
