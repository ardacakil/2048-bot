import pandas as pd
import pytest

from game import replay as R
from ntuple import evaluate as V
from ntuple import replays as RP

from conftest import EVAL_RUN


def test_pick_best_median_and_seed():
    df = pd.DataFrame({"seed": [0, 1, 2, 3, 4], "score": [50, 300, 120, 300, 100]})
    assert RP.pick(df, "best")["seed"] == 1  # tie on 300: lowest seed
    assert RP.pick(df, "median")["seed"] == 2  # median 120
    assert RP.pick(df.iloc[:4], "median")["seed"] == 1  # median 210: seeds 1 and 2 are 90 away, lowest wins
    assert RP.pick(df, "seed=4")["score"] == 100
    with pytest.raises(ValueError):
        RP.pick(df, "seed=9")
    with pytest.raises(ValueError):
        RP.pick(df, "worst")


@pytest.mark.parametrize("label", ["random", "greedy", "expectimax-d1"])
def test_write_replay_regenerates_and_verifies(eval_root, ckpt_root, tmp_path, label):
    folder = "random" if label == "random" else f"{EVAL_RUN}/final"
    df = V.read_results(eval_root / folder / f"{label}.csv")
    row = RP.pick(df, "best")
    path = RP.write_replay(row, "best", tmp_path, ckpt_root)
    prefix = "" if label == "random" else f"{EVAL_RUN}_final_"
    assert path.name == f"{prefix}{label}_best_seed{row['seed']}.json"
    doc = R.load(path)
    m = doc["meta"]
    assert (m["seed"], m["score"], m["moves"], m["final_board"]) == \
        (row["seed"], row["score"], row["moves"], row["final_board"])
    assert (m["policy"], m["depth"], m["tt_bits"], m["selection"]) == (row["policy"], row["depth"], row["tt_bits"], "best")
    board, score = R.replay(doc)
    assert R.board_hex(board) == row["final_board"] and score == row["score"]


def test_regenerate_detects_a_mismatch(eval_root, ckpt_root):
    row = V.read_results(eval_root / EVAL_RUN / "final" / "greedy.csv").iloc[0].copy()
    row["score"] += 4
    with pytest.raises(RuntimeError, match="did not reproduce"):
        RP.regenerate(row, ckpt_root)


def test_default_selection(eval_root):
    picks = RP.default_selection(EVAL_RUN, "final", eval_root)
    got = [(r["policy"], int(r["depth"]), sel) for r, sel in picks]
    assert got == [("random", 0, "best"), ("expectimax", 1, "best"), ("greedy", 0, "best"),
                   ("greedy", 0, "median"), ("expectimax", 1, "median")]


def test_cli_verify(eval_root, ckpt_root, tmp_path, capsys):
    row = RP.pick(V.read_results(eval_root / "random" / "random.csv"), "median")
    path = RP.write_replay(row, "median", tmp_path, ckpt_root)
    RP.main(["--verify", str(path)])
    assert capsys.readouterr().out.startswith("ok")
