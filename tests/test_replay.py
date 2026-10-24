import copy
import json

import numpy as np
import pytest

from game import engine as E
from game import replay as R
from ntuple import player as P


def doc_for(rec, **meta):
    meta = {"run": "-", "checkpoint": "-", "games_trained": 0, "policy": "random", "depth": 0,
            "prune": 0.0, "tt_bits": 0, "seed": rec.seed, "score": rec.score, "max_tile": rec.max_tile,
            "moves": rec.moves, "final_board": rec.final_board, "selection": "test", **meta}
    return R.make_doc(meta, rec.init, rec.actions, rec.rewards, rec.spawns, check_every=20)


@pytest.fixture(scope="module")
def game():
    """A random-policy game (~100 moves) and its replay document."""
    rec = P.Player(None, "random").play(4)
    assert rec.moves > 60
    return rec, doc_for(rec)


def test_encoding_action_order_matches_engine_constants():
    names = R.ENCODING["actions"]
    assert names[E.UP] == "up" and names[E.RIGHT] == "right"
    assert names[E.DOWN] == "down" and names[E.LEFT] == "left"
    assert tuple(names) == E.ACTION_NAMES
    assert "up, right, down, left" in R.ENCODING["action_index"]


def test_encoding_documents_js_decoding():
    text = R.ENCODING["board_hex"]
    assert "never via Number" in text and "BigInt" in text and "nibble by nibble" in text
    assert "never via Number" in R.__doc__ and "engine.UP, RIGHT, DOWN, LEFT" in R.__doc__


def test_board_hex_layout():
    assert R.board_hex(E.from_values([[32, 0, 0, 0]] + [[0] * 4] * 3)) == "0x0000000000000005"
    assert R.board_hex(E.from_values([[0] * 4] * 3 + [[0, 0, 0, 32768]])) == "0xf000000000000000"
    high = 0xFEDCBA9876543210  # > 2**63, beyond a double's 53-bit integer precision
    assert R.parse_board_hex(R.board_hex(high)) == high
    with pytest.raises(ValueError):
        R.parse_board_hex("0x123")


def test_replay_reproduces_final_board_and_score(game, tmp_path):
    rec, doc = game
    path = R.write(doc, tmp_path / "g.json")
    loaded = R.load(path)
    assert loaded == json.loads(json.dumps(doc))
    board, score = R.replay(loaded)
    assert (board, score) == (rec.final_board, rec.score)
    assert loaded["meta"]["final_board"] == R.board_hex(rec.final_board)
    assert len(loaded["steps"]) == rec.moves and len(loaded["checks"]["boards"]) == rec.moves // 20


def test_replay_of_greedy_and_expectimax_games(W_eval):
    for policy, depth in (("greedy", 0), ("expectimax", 1)):
        rec = P.Player(W_eval, policy, depth).play(1)
        assert R.replay(doc_for(rec, policy=policy, depth=depth)) == (rec.final_board, rec.score)


def test_trajectory_matches_the_game(game):
    rec, doc = game
    boards = R.trajectory(rec.init, rec.actions, rec.spawns)
    assert len(boards) == rec.moves + 1 and int(boards[-1]) == rec.final_board
    assert R.board_hex(boards[20]) == doc["checks"]["boards"][0]
    assert E.count_empty(int(boards[0])) == 14


def corrupt(doc, fn):
    bad = copy.deepcopy(doc)
    fn(bad)
    return bad


def test_desync_detection(game):
    _, doc = game
    board4 = int(R.trajectory(doc["init"], [s[0] for s in doc["steps"]], [s[2:] for s in doc["steps"]])[4])
    after5 = int(E.move(board4, doc["steps"][4][0])[0])
    occupied = next(c for c in range(16) if (after5 >> (4 * c)) & 0xF)
    illegal = next((a for a in range(4) if not E.is_legal(board4, a)), None)

    def expect(bad, step, what):
        with pytest.raises(R.ReplayDesync) as e:
            R.replay(bad)
        assert e.value.step == step and what in e.value.what

    expect(corrupt(doc, lambda d: d["steps"][4].__setitem__(2, occupied)), 5, "occupied")
    expect(corrupt(doc, lambda d: d["steps"][4].__setitem__(1, d["steps"][4][1] + 2)), 5, "reward")
    if illegal is not None:
        expect(corrupt(doc, lambda d: d["steps"][4].__setitem__(0, illegal)), 5, "illegal")
    expect(corrupt(doc, lambda d: d["checks"]["boards"].__setitem__(1, "0x0000000000000000")), 40, "board check")
    expect(corrupt(doc, lambda d: d["meta"].__setitem__("score", d["meta"]["score"] + 4)),
           len(doc["steps"]), "final score")
    expect(corrupt(doc, lambda d: d["steps"].pop()), len(doc["steps"]) - 1, "")
    expect(corrupt(doc, lambda d: d["init"].__setitem__(1, list(d["init"][0]))), "init 1", "occupied")


def test_load_rejects_other_versions(game, tmp_path):
    _, doc = game
    R.write({**doc, "version": 2}, tmp_path / "v2.json")
    with pytest.raises(ValueError, match="version"):
        R.load(tmp_path / "v2.json")


def test_10k_move_game_stays_under_1mb(tmp_path):
    """Size bound with worst-case-ish numbers: 4-digit rewards and 2-digit cells on every step."""
    n = 12_000
    rng = np.random.default_rng(0)
    actions = rng.integers(0, 4, n)
    rewards = rng.choice([0, 4, 16, 1024, 8192], n)
    spawns = np.column_stack([rng.integers(10, 16, n), rng.choice([1, 2], n)])
    doc = {"schema": R.SCHEMA, "version": R.VERSION, "encoding": R.ENCODING,
           "meta": {"score": int(rewards.sum()), "moves": n, "final_board": "0x" + "f" * 16},
           "init": [[0, 1], [5, 2]], "step_fields": list(R.STEP_FIELDS),
           "steps": [[int(a), int(r), int(c), int(e)] for a, r, (c, e) in zip(actions, rewards, spawns)],
           "checks": {"every": 100, "boards": ["0x" + "a" * 16] * (n // 100),
                      "scores": [10**6] * (n // 100)}}
    size = R.write(doc, tmp_path / "big.json").stat().st_size
    assert size < 1_000_000, size
