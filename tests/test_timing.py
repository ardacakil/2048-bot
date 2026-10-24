import subprocess
import sys

import pytest

from game import engine as E
from ntuple import expectimax as X
from ntuple import timing as TI
from ntuple.player import Player

Z = [0, 0, 0, 0]


class Scripted:
    """Stand-in for Expectimax: plays a fixed list of actions (then -1 = no move)."""

    def __init__(self, actions):
        self.actions = list(actions)
        self.table = X.new_table(4)

    def best_move(self, board):
        return (self.actions.pop(0) if self.actions else -1), 0.0


def test_target_detected_on_the_move_it_appears():
    # 512 512 1024 -> LEFT -> 1024 1024 (no 2048 yet) -> LEFT -> 2048 on move 2
    start = E.from_values([[512, 512, 1024, 0], Z, Z, Z])
    t = TI.play_until(Scripted([E.LEFT, E.LEFT, E.LEFT]), seed=3, target=2048, start_board=start)
    assert t.reached and t.moves == 2 and t.max_tile == 2048
    assert t.score == 1024 + 2048


def test_target_already_on_the_board_needs_no_move():
    t = TI.play_until(Scripted([E.LEFT]), seed=0, target=2048,
                      start_board=E.from_values([[4096, 0, 0, 0], Z, Z, Z]))
    assert t.reached and t.moves == 0 and t.compute_s == 0.0 and t.ms_per_move == 0.0


def test_game_over_before_target():
    full = E.from_values([[2, 4, 2, 4], [4, 2, 4, 2], [2, 4, 2, 4], [4, 2, 4, 2]])
    t = TI.play_until(Scripted([]), seed=0, target=2048, start_board=full)
    assert not t.reached and t.moves == 0 and t.max_tile == 4


def test_play_until_plays_the_eval_game_and_stops_at_its_target(W_eval):
    bot = X.Expectimax(W_eval, depth=1, prune=1e-4, tt_bits=18)
    full = TI.play_until(bot, seed=2, target=1 << 15)  # random weights: never reached, plays to the end
    rec = Player(W_eval, "expectimax", 1).play(2)  # same seed, same settings: the eval game
    assert not full.reached and (full.score, full.moves, full.max_tile) == (rec.score, rec.moves, rec.max_tile)
    part = TI.play_until(bot, seed=2, target=full.max_tile)
    assert part.reached and part.moves <= full.moves and part.max_tile == full.max_tile
    assert part.compute_s > 0 and part.wall_s >= part.compute_s


def test_sleep_measures_real_wall_time():
    start = E.from_values([[512, 512, 1024, 0], Z, Z, Z])
    t = TI.play_until(Scripted([E.LEFT, E.LEFT]), seed=3, target=2048, delay=0.02, sleep=True,
                      start_board=start)
    assert t.moves == 2 and t.wall_s >= 2 * 0.02


def test_parse_targets():
    assert TI.parse_targets("2048:0.02,16384:0.005") == [(2048, 0.02), (16384, 0.005)]
    assert TI.parse_targets(TI.DEFAULT_TARGETS)[2] == (8192, 0.01)
    assert TI.parse_targets("4096") == [(4096, 0.0)]
    for bad in ("2000:0.1", "2:0.1", "x:1"):
        with pytest.raises(ValueError):
            TI.parse_targets(bad)


def test_summary_and_table():
    timings = [TI.Timing(0, 2048, True, 1000, 0.1, 0.1, 0, 2048),
               TI.Timing(1, 2048, True, 1200, 0.2, 0.2, 0, 2048),
               TI.Timing(2, 2048, False, 500, 0.05, 0.05, 0, 1024)]
    s = TI.summarize(timings, delay=0.02, sleep=False)
    assert s["games"] == 3 and s["reached_pct"] == pytest.approx(200 / 3)
    assert s["moves"] == [1000, 1200] and s["playback_s"] == pytest.approx([20.0, 24.0])
    assert s["ms_per_move"] == pytest.approx(0.35 / 2700 * 1e3)  # over all games
    text = TI.format_target(2048, 0.02, timings, sleep=False)
    assert "1,000 / 1,100 / 1,200" in text and "22.0 / 24.0" in text and "game over, max 1024" in text


def test_peak_memory_is_plausible():
    assert 10 < TI.peak_memory_mb() < 64_000


def test_imports_nothing_the_vps_lacks():
    code = ("import sys, ntuple.timing; "
            "print(sorted(m for m in ('pandas', 'matplotlib', 'IPython', 'jupyter', 'scipy') if m in sys.modules))")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "[]"


def test_cli(ckpt_root, capsys):
    TI.main(["--bot", str(ckpt_root / "runx"), "--depth", "1", "--games", "2", "--targets", "64:0.01,128:0.02"])
    out = capsys.readouterr().out
    assert "target 64" in out and "target 128" in out and "warm-up" in out and "peak memory" in out
