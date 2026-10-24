import numpy as np
import pandas as pd
import pytest

from game import engine as E
from game.replay import board_hex
from ntuple import evaluate as V

from conftest import EVAL_RUN


@pytest.fixture(scope="module")
def final(ckpt_root):
    return V.target_from_path(ckpt_root / EVAL_RUN)


def test_max_tile_position_prefers_corners_then_lowest_cell():
    assert V.max_tile_position(E.from_values([[2, 0, 0, 0], [0, 64, 0, 0], [0] * 4, [0] * 4])) == (5, False)
    assert V.max_tile_position(E.from_values([[0, 64, 0, 0], [0] * 4, [0] * 4, [0, 0, 0, 64]])) == (15, True)
    assert V.max_tile_position(E.from_values([[64, 0, 0, 64], [0] * 4, [0] * 4, [0] * 4])) == (0, True)
    assert V.max_tile_position(E.from_values([[0] * 4, [0, 8, 8, 0], [0] * 4, [0] * 4])) == (5, False)
    assert V.max_tile_position(E.from_values([[0] * 4, [0] * 4, [0] * 4, [4096, 0, 0, 0]])) == (12, True)


def test_targets_and_csv_paths(ckpt_root, final, tmp_path):
    targets = V.list_checkpoints(EVAL_RUN, "all", ckpt_root)
    assert [(t.checkpoint, t.games_trained) for t in targets] == [("games-0000000500", 500), ("final", 1000)]
    assert targets[0].run == targets[1].run == EVAL_RUN
    assert V.list_checkpoints(EVAL_RUN, "games-0000000500", ckpt_root) == targets[:1]
    assert final == targets[1] and final.path == ckpt_root / EVAL_RUN
    assert V.csv_path(final, "expectimax-d2", tmp_path) == tmp_path / EVAL_RUN / "final" / "expectimax-d2.csv"
    assert V.csv_path(V.RANDOM_TARGET, "random", tmp_path) == tmp_path / "random" / "random.csv"


def test_csv_columns_types_and_settings(eval_root):
    path = eval_root / EVAL_RUN / "final" / "expectimax-d1.csv"
    header = path.read_text().splitlines()[0].split(",")
    assert header == V.CSV_FIELDS
    df = V.read_results(path)
    assert df["max_tile_in_corner"].dtype == bool
    assert len(df) == 8 and sorted(df["seed"]) == list(range(8))
    row = df.iloc[0]
    assert (row["run"], row["checkpoint"], row["games_trained"]) == (EVAL_RUN, "final", 1000)
    assert (row["policy"], row["depth"], row["prune"], row["tt_bits"], row["n_workers"]) == \
        ("expectimax", 1, 1e-4, 18, 2)
    for _, r in df.iterrows():
        board = int(r["final_board"], 16)
        assert r["max_tile"] == 1 << E.max_tile(board) and E.is_game_over(board)
        assert (r["max_tile_pos"], r["max_tile_in_corner"]) == V.max_tile_position(board)
        assert r["ms_per_move"] == pytest.approx(r["wall_s"] / r["moves"] * 1e3)
    greedy = V.read_results(eval_root / EVAL_RUN / "final" / "greedy.csv")
    assert (greedy["prune"] == 0).all() and (greedy["tt_bits"] == 0).all()
    random = V.read_results(eval_root / "random" / "random.csv")
    assert (random["run"] == "-").all() and (random["games_trained"] == 0).all()


def test_rows_match_single_process_player(eval_root, W_eval):
    from ntuple.player import Player

    df = V.read_results(eval_root / EVAL_RUN / "final" / "greedy.csv")
    player = Player(W_eval, "greedy")
    for seed in (0, 7, 19):
        rec = player.play(seed)
        row = df[df["seed"] == seed].iloc[0]
        assert (row["score"], row["moves"], row["final_board"]) == (rec.score, rec.moves, board_hex(rec.final_board))


def test_resume_skips_existing_seeds(final, tmp_path):
    out = tmp_path / "greedy.csv"
    first = V.evaluate_target(final, "greedy", games=6, workers=2, out_csv=out, verbose=False)
    before = out.read_text()
    full = V.evaluate_target(final, "greedy", games=10, workers=2, out_csv=out, verbose=False)
    assert out.read_text().startswith(before)  # old rows untouched, new ones appended
    assert full["seed"].tolist() == list(range(10))
    pd.testing.assert_frame_equal(full.iloc[:6].reset_index(drop=True), first)
    again = V.evaluate_target(final, "greedy", games=10, workers=2, out_csv=out, verbose=False)
    assert out.read_text().count("\n") == 11  # header + 10 rows: nothing replayed
    pd.testing.assert_frame_equal(again, full)
    shifted = V.evaluate_target(final, "greedy", games=4, base_seed=8, workers=2, out_csv=out, verbose=False)
    assert shifted["seed"].tolist() == [8, 9, 10, 11]


def test_results_do_not_depend_on_worker_count(final, tmp_path):
    cols = ["seed", "score", "moves", "max_tile", "final_board"]
    one = V.evaluate_target(final, "expectimax", 1, games=6, workers=1, out_csv=tmp_path / "a.csv", verbose=False)
    three = V.evaluate_target(final, "expectimax", 1, games=6, workers=3, out_csv=tmp_path / "b.csv", verbose=False)
    pd.testing.assert_frame_equal(one[cols], three[cols])
    assert set(one["n_workers"]) == {1} and set(three["n_workers"]) == {3}


def test_mismatched_settings_are_refused(final, tmp_path):
    out = tmp_path / "d1.csv"
    V.evaluate_target(final, "expectimax", 1, games=2, workers=1, tt_bits=16, out_csv=out, verbose=False)
    with pytest.raises(ValueError, match="tt_bits"):
        V.evaluate_target(final, "expectimax", 1, games=4, workers=1, tt_bits=17, out_csv=out, verbose=False)
    with pytest.raises(ValueError, match="prune"):
        V.evaluate_target(final, "expectimax", 1, games=4, workers=1, tt_bits=16, prune=0.0, out_csv=out,
                          verbose=False)


def test_torn_last_line_is_dropped_and_resumed(final, tmp_path):
    out = tmp_path / "greedy.csv"
    V.evaluate_target(final, "greedy", games=3, workers=1, out_csv=out, verbose=False)
    with open(out, "a") as f:
        f.write(f"{EVAL_RUN},final,1000,greedy,0,3,12")  # crash in the middle of a row
    df = V.evaluate_target(final, "greedy", games=5, workers=1, out_csv=out, verbose=False)
    assert df["seed"].tolist() == [0, 1, 2, 3, 4]
    assert len(V.read_results(out)) == 5


def test_random_policy_needs_no_checkpoint(tmp_path):
    df = V.evaluate_target(V.RANDOM_TARGET, "random", games=50, workers=2,
                           out_csv=tmp_path / "random.csv", verbose=False)
    assert len(df) == 50 and df["score"].mean() > 500
    with pytest.raises(ValueError, match="checkpoint"):
        V.evaluate_target(V.RANDOM_TARGET, "greedy", games=1, verbose=False)


def test_memory_guard():
    assert V.estimated_memory_gb(8, 20) == pytest.approx(0.27 + 8 * (0.2 + 24 * 2**20 / 1e9))
    V.check_memory(8, 22)
    with pytest.raises(ValueError, match="workers"):
        V.check_memory(16, 24)  # ~9.9 GB; 10 workers at tt_bits 24 fit (~6.3 GB)


def test_notebook_api_writes_nothing(ckpt_root, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    summary, df = V.evaluate(ckpt_root / EVAL_RUN, games=4, depth=0, workers=2, verbose=False)
    for key in ("avg_score", "median_score", "reach_2048", "reach_4096", "reach_8192", "ms_per_move",
                "max_tile_pct", "workers"):
        assert key in summary
    assert summary["games"] == 4 and summary["workers"] == 2 and len(df) == 4
    deep, _ = V.evaluate(ckpt_root / EVAL_RUN, games=2, depth=2, workers=2, verbose=False)
    assert deep["depth"] == 2
    assert not list(tmp_path.iterdir())


def test_cli_writes_csv_and_summary(ckpt_root, tmp_path):
    out = tmp_path / "results" / "eval"
    milestone = ckpt_root / EVAL_RUN / "milestones" / "games-0000000500"
    V.main(["--checkpoint", str(milestone), "--games", "3", "--workers", "1", "--out", str(out)])
    V.main(["--checkpoint", str(ckpt_root / EVAL_RUN), "--depth", "1", "--games", "2", "--workers", "1",
            "--out", str(out)])
    assert len(V.read_results(out / EVAL_RUN / "games-0000000500" / "greedy.csv")) == 3
    assert len(V.read_results(out / EVAL_RUN / "final" / "expectimax-d1.csv")) == 2
    summary = pd.read_csv(tmp_path / "results" / "eval_summary.csv")
    assert len(summary) == 2 and set(summary["policy"]) == {"greedy", "expectimax"}
    with pytest.raises(SystemExit):
        V.main(["--policy", "greedy", "--games", "1", "--out", str(out)])


def test_benchmark(final, tmp_path):
    df = V.benchmark(final, labels=("greedy", "d1"), moves=5, stages=(0.0, 0.5), out=tmp_path, verbose=False)
    assert len(df) == 4 and set(df["policy"]) == {"greedy", "expectimax"}
    assert (df["moves"] <= 5).all() and (df["ms_per_move"] > 0).all()
    assert df[df["stage"] == 0.5]["start_move"].iloc[0] > 0
    saved = pd.read_csv(tmp_path / EVAL_RUN / "final" / "bench.csv")
    assert len(saved) == 4 and saved["tt_bits"].tolist() == [0, 0, 18, 18]
