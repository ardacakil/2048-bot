import json
import os
import subprocess
import sys

import numpy as np
import pytest
from numba import types

from game import engine as E
from game import reference as R

UP, RIGHT, DOWN, LEFT = E.UP, E.RIGHT, E.DOWN, E.LEFT


def exps(values):
    """Tile values -> exponents, e.g. [2, 4, 0] -> [1, 2, 0]."""
    return [0 if v == 0 else int(v).bit_length() - 1 for v in values]


def row_from_values(values):
    """Tile values (column 0 first) -> 16-bit row."""
    return sum(e << (4 * c) for c, e in enumerate(exps(values)))


def row_to_values(row):
    return [0 if (t := (int(row) >> (4 * c)) & 0xF) == 0 else 1 << t for c in range(4)]


def grid(board):
    return E.decode(board).tolist()


def random_board(rng, max_exp=15, p_empty=0.3):
    g = rng.integers(1, max_exp + 1, size=(4, 4))
    g[rng.random((4, 4)) < p_empty] = 0
    return E.encode(g.astype(np.int64))


# ---------------------------------------------------------------- row tables

LEFT_CASES = [
    # (row, expected after move-left, merge score)
    ([2, 2, 2, 2], [4, 4, 0, 0], 8),
    ([4, 4, 8, 0], [8, 8, 0, 0], 8),  # the new 8 must not merge with the old 8
    ([2, 2, 4, 0], [4, 4, 0, 0], 4),  # no double merge
    ([2, 2, 4, 4], [4, 8, 0, 0], 12),
    ([8, 4, 4, 0], [8, 8, 0, 0], 8),
    ([2, 2, 2, 0], [4, 2, 0, 0], 4),  # leftmost pair merges first
    ([2, 0, 0, 2], [4, 0, 0, 0], 4),
    ([0, 0, 0, 2], [2, 0, 0, 0], 0),
    ([0, 2, 0, 4], [2, 4, 0, 0], 0),
    ([2, 4, 2, 4], [2, 4, 2, 4], 0),
    ([0, 0, 0, 0], [0, 0, 0, 0], 0),
    ([1024, 1024, 0, 0], [2048, 0, 0, 0], 2048),
    ([2048, 2048, 2048, 2048], [4096, 4096, 0, 0], 8192),
]


@pytest.mark.parametrize("row, expected, score", LEFT_CASES)
def test_row_left(row, expected, score):
    r = row_from_values(row)
    assert row_to_values(E.ROW_LEFT[r]) == expected
    assert E.SCORE_LEFT[r] == score


@pytest.mark.parametrize("row, expected, score", LEFT_CASES)
def test_row_right_is_mirror(row, expected, score):
    r = row_from_values(row[::-1])
    assert row_to_values(E.ROW_RIGHT[r]) == expected[::-1]
    assert E.SCORE_RIGHT[r] == score


def test_max_exponent_does_not_merge():
    r = 15 | (15 << 4)
    assert E.ROW_LEFT[r] == r
    assert E.SCORE_LEFT[r] == 0
    r = 14 | (14 << 4)
    assert E.ROW_LEFT[r] == 15
    assert E.SCORE_LEFT[r] == 1 << 15


def test_tables_match_reference_for_all_rows():
    for r in range(65536):
        cells = [(r >> (4 * c)) & 0xF for c in range(4)]
        out, score = R.slide_row_left(cells)
        assert E.ROW_LEFT[r] == sum(t << (4 * c) for c, t in enumerate(out)), r
        assert E.SCORE_LEFT[r] == score, r
        out, score = R.slide_row_left(cells[::-1])
        out = out[::-1]
        assert E.ROW_RIGHT[r] == sum(t << (4 * c) for c, t in enumerate(out)), r
        assert E.SCORE_RIGHT[r] == score, r


def test_table_dtypes():
    assert E.ROW_LEFT.dtype == np.uint16 and E.ROW_RIGHT.dtype == np.uint16
    assert E.SCORE_LEFT.dtype == np.uint32 and E.SCORE_RIGHT.dtype == np.uint32
    assert E.ROW_LEFT.shape == (65536,)


# ---------------------------------------------------------------- uint64 typing

BOARD_FUNCS = ["transpose", "afterstate", "encode", "spawn_tile", "new_game"]
MOVE_FUNCS = ["move", "move_left", "move_right", "move_up", "move_down"]
# functions that should contain no int<->float conversion at all
INT_ONLY_FUNCS = MOVE_FUNCS + [
    "transpose", "afterstate", "is_legal", "legal_moves", "count_empty",
    "max_tile", "is_game_over", "encode", "decode", "make_rng", "rng_next", "rng_below",
    "spawned_cell",
]
# these use rng_uniform (one intended uint64 -> float64 conversion), but must still never
# treat a board as signed
SPAWN_FUNCS = ["spawn_tile", "new_game", "play_random_game"]


def test_move_returns_uint64():
    for name in MOVE_FUNCS:
        (sig,) = getattr(E, name).nopython_signatures
        assert sig.return_type == types.Tuple((types.uint64, types.int64)), name
        assert sig.args[0] == types.uint64, name


def test_board_functions_return_uint64():
    for name in BOARD_FUNCS:
        (sig,) = getattr(E, name).nopython_signatures
        assert sig.return_type == types.uint64, name


def test_one_signature_per_function():
    # explicit signatures: Python ints must be coerced, never compile an int64 variant
    E.move(5, 0)
    E.move(np.int64(5), 0)
    E.move(2**64 - 1, 0)
    for name in INT_ONLY_FUNCS:
        assert len(getattr(E, name).nopython_signatures) == 1, name


# Mixing uint64 with int64 makes Numba produce int64 (0.6x) or float64 (older versions).
# In LLVM IR the first shows up as `ashr` (sign-extending shift), the second as int<->float
# conversions. Positive controls prove the check detects both.
_IR_CHECK = """
import json, re, sys
import numpy as np
from numba import njit
from game import engine as E

OPS = re.compile(r"\\b(uitofp|sitofp|fptoui|fptosi|ashr)\\b")

def ops(fn):
    return sorted(set(OPS.findall(fn.inspect_llvm(fn.signatures[0]))))

signed = njit("uint64(uint64)")(lambda b: (b + 1) >> 4)  # int literal -> signed shift
floaty = njit("uint64(uint64)")(lambda b: np.uint64(b * 0.5))
out = {"__signed__": ops(signed), "__float__": ops(floaty)}
out.update({name: ops(getattr(E, name)) for name in sys.argv[1:]})
print(json.dumps(out))
"""


def test_no_signed_or_float_promotion_in_compiled_code(tmp_path):
    # LLVM inspection is disabled for cache-loaded functions, so compile fresh in a subprocess
    env = {**os.environ, "NUMBA_CACHE_DIR": str(tmp_path)}
    proc = subprocess.run(
        [sys.executable, "-c", _IR_CHECK, *INT_ONLY_FUNCS, *SPAWN_FUNCS],
        env=env, capture_output=True, text=True, check=True,
    )
    found = json.loads(proc.stdout.strip().splitlines()[-1])
    assert "ashr" in found.pop("__signed__"), "IR check missed a known signed promotion"
    assert "uitofp" in found.pop("__float__"), "IR check missed a known float promotion"
    for name in INT_ONLY_FUNCS:
        assert found[name] == [], name
    for name in SPAWN_FUNCS:
        assert "ashr" not in found[name], name


def test_top_bit_board_from_python_int():
    # cell (3, 3) = exponent 8 sets bit 63; Python int must arrive as uint64 intact
    board = 8 << 60
    assert board >= 2**63
    after, reward = E.move(board, LEFT)
    assert after == 8 << 48
    assert reward == 0
    assert E.move(after, RIGHT)[0] == board


# ---------------------------------------------------------------- board moves

BOARD = [
    [2, 2, 2, 2],
    [4, 4, 8, 0],
    [2, 0, 0, 2],
    [0, 4, 0, 0],
]


@pytest.mark.parametrize(
    "action, expected, score",
    [
        (LEFT, [[4, 4, 0, 0], [8, 8, 0, 0], [4, 0, 0, 0], [4, 0, 0, 0]], 8 + 8 + 4),
        (RIGHT, [[0, 0, 4, 4], [0, 0, 8, 8], [0, 0, 0, 4], [0, 0, 0, 4]], 8 + 8 + 4),
        (UP, [[2, 2, 2, 4], [4, 8, 8, 0], [2, 0, 0, 0], [0, 0, 0, 0]], 8 + 4),
        (DOWN, [[0, 0, 0, 0], [2, 0, 0, 0], [4, 2, 2, 0], [2, 8, 8, 4]], 8 + 4),
    ],
)
def test_each_direction(action, expected, score):
    board = E.from_values(BOARD)
    after, reward = E.move(board, action)
    assert E.to_values(after).tolist() == expected
    assert reward == score
    assert E.afterstate(board, action) == after


def test_move_does_not_spawn():
    board = E.from_values(BOARD)
    for a in range(4):
        after, _ = E.move(board, a)
        assert sum(E.to_values(after).flatten()) == sum(map(sum, BOARD))


def test_illegal_move_returns_same_board():
    board = E.from_values([[2, 0, 0, 0], [4, 0, 0, 0], [8, 0, 0, 0], [16, 0, 0, 0]])
    for a in (UP, DOWN, LEFT):
        after, reward = E.move(board, a)
        assert after == board and reward == 0
        assert not E.is_legal(board, a)
    assert E.is_legal(board, RIGHT)


def test_moves_match_reference_on_random_boards():
    rng = np.random.default_rng(0)
    for _ in range(5000):
        board = random_board(rng, p_empty=rng.random())
        g = grid(board)
        for a in range(4):
            after, reward = E.move(board, a)
            ref_after, ref_score = R.move(g, a)
            assert grid(after) == ref_after
            assert reward == ref_score
        assert E.legal_moves(board).tolist() == R.legal_moves(g)
        assert E.is_game_over(board) == R.is_game_over(g)


# ---------------------------------------------------------------- primitives


def test_transpose():
    rng = np.random.default_rng(1)
    for _ in range(1000):
        board = random_board(rng, p_empty=0.0)
        assert grid(E.transpose(board)) == E.decode(board).T.tolist()
        assert E.transpose(E.transpose(board)) == board


def test_encode_decode_roundtrip():
    rng = np.random.default_rng(2)
    for _ in range(1000):
        g = rng.integers(0, 16, size=(4, 4)).astype(np.int64)
        assert (E.decode(E.encode(g)) == g).all()


def test_cell_layout():
    # cell (r, c) lives at bit 16r + 4c
    g = np.zeros((4, 4), np.int64)
    g[2, 1] = 5
    assert E.encode(g) == 5 << (16 * 2 + 4 * 1)


def test_from_to_values():
    board = E.from_values(BOARD)
    assert E.to_values(board).tolist() == BOARD


def test_count_empty_and_max_tile():
    board = E.from_values(BOARD)
    assert E.count_empty(board) == 6
    assert E.max_tile(board) == 3  # 8
    assert E.count_empty(0) == 16
    assert E.max_tile(0) == 0
    assert E.max_tile(15 << 60) == 15


# ---------------------------------------------------------------- legal moves / game over


def test_legal_moves_mask():
    # one tile in the top-left corner: only right and down
    board = E.from_values([[2, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]])
    mask = E.legal_moves(board)
    assert mask.dtype == np.bool_ and mask.shape == (4,)
    assert mask.tolist() == [False, True, True, False]


def test_full_board_with_horizontal_merge_is_alive():
    board = E.from_values([[2, 2, 4, 8], [4, 8, 16, 32], [8, 16, 32, 64], [16, 32, 64, 128]])
    assert E.legal_moves(board).tolist() == [False, True, False, True]
    assert not E.is_game_over(board)


def test_full_board_with_vertical_merge_is_alive():
    board = E.from_values([[2, 4, 2, 4], [2, 8, 4, 2], [4, 2, 8, 4], [2, 4, 2, 8]])
    assert E.legal_moves(board).tolist() == [True, False, True, False]
    assert not E.is_game_over(board)


def test_game_over_checkerboard():
    board = E.from_values([[2, 4, 2, 4], [4, 2, 4, 2], [2, 4, 2, 4], [4, 2, 4, 2]])
    assert E.legal_moves(board).tolist() == [False] * 4
    assert E.is_game_over(board)


def test_board_with_empty_cell_is_not_over():
    board = E.from_values([[2, 4, 2, 4], [4, 2, 4, 2], [2, 4, 2, 4], [4, 2, 4, 0]])
    assert not E.is_game_over(board)


def test_empty_board_has_no_legal_moves():
    # unreachable in play, but is_game_over must mean exactly "no legal move"
    assert E.legal_moves(0).tolist() == [False] * 4
    assert E.is_game_over(0)


# ---------------------------------------------------------------- RNG


def test_rng_seeded_reproducible():
    a, b, c = E.make_rng(42), E.make_rng(42), E.make_rng(43)
    xs = [E.rng_next(a) for _ in range(100)]
    assert xs == [E.rng_next(b) for _ in range(100)]
    assert xs != [E.rng_next(c) for _ in range(100)]
    assert a.dtype == np.uint64 and a.shape == (1,)


def test_rng_zero_seed_is_valid():
    rng = E.make_rng(0)
    assert rng[0] != 0
    assert len({E.rng_next(rng) for _ in range(1000)}) == 1000


def test_rng_ranges():
    rng = E.make_rng(7)
    us = [E.rng_uniform(rng) for _ in range(20000)]
    assert 0.0 <= min(us) and max(us) < 1.0
    assert abs(np.mean(us) - 0.5) < 0.01
    ks = [E.rng_below(rng, 5) for _ in range(20000)]
    assert set(ks) == {0, 1, 2, 3, 4}


def test_rng_state_can_be_restored():
    rng = E.make_rng(3)
    for _ in range(10):
        E.rng_next(rng)
    saved = rng.copy()
    xs = [E.rng_next(rng) for _ in range(10)]
    assert [E.rng_next(saved) for _ in range(10)] == xs


# ---------------------------------------------------------------- spawning


def test_spawn_probabilities():
    rng = E.make_rng(123)
    n = 200_000
    fours = 0
    for _ in range(n):
        fours += E.max_tile(E.spawn_tile(0, rng)) == 2
    sigma = np.sqrt(n * 0.1 * 0.9)
    assert abs(fours - 0.1 * n) < 4 * sigma


def test_spawn_uniform_over_empty_cells():
    # 5 empty cells: (0, 1), (1, 2), (2, 0), (3, 3), (3, 0)
    values = [[2, 0, 2, 2], [2, 2, 0, 2], [0, 2, 2, 2], [0, 2, 2, 0]]
    board = E.from_values(values)
    empty = [4 * r + c for r in range(4) for c in range(4) if values[r][c] == 0]
    rng = E.make_rng(9)
    n = 100_000
    counts = dict.fromkeys(empty, 0)
    for _ in range(n):
        new = E.spawn_tile(board, rng)
        added = new ^ board
        cell = (added.bit_length() - 1) // 4
        assert added >> (4 * cell) in (1, 2)  # exactly one new 2 or 4 ...
        assert cell in counts  # ... on an empty cell
        counts[cell] += 1
    p = 1 / len(empty)
    sigma = np.sqrt(n * p * (1 - p))
    for cell, k in counts.items():
        assert abs(k - n * p) < 5 * sigma, (cell, k)


def test_spawn_on_full_board_is_noop():
    board = E.from_values([[2, 4, 2, 4], [4, 2, 4, 2], [2, 4, 2, 4], [4, 2, 4, 2]])
    assert E.spawn_tile(board, E.make_rng(0)) == board


def test_new_game_has_two_tiles():
    rng = E.make_rng(5)
    for _ in range(1000):
        g = E.decode(E.new_game(rng))
        assert (g > 0).sum() == 2
        assert set(g[g > 0].tolist()) <= {1, 2}


# ---------------------------------------------------------------- full games


def test_random_games_agree_with_reference():
    """Lockstep: engine plays, reference checks every move, legal mask and game-over flag."""
    for seed in range(200):
        rng = E.make_rng(seed)
        moves = np.random.default_rng(seed)
        board = E.new_game(rng)
        score = 0
        while True:
            g = grid(board)
            legal = E.legal_moves(board).tolist()
            assert legal == R.legal_moves(g)
            assert E.is_game_over(board) == R.is_game_over(g) == (not any(legal))
            if not any(legal):
                break
            for a in range(4):
                after, reward = E.move(board, a)
                assert (grid(after), reward) == R.move(g, a)
            a = int(moves.choice(np.flatnonzero(legal)))
            after, reward = E.move(board, a)
            score += reward
            board = E.spawn_tile(after, rng)
            # spawn adds exactly one 2 or 4 to the afterstate
            diff = E.decode(board) - E.decode(after)
            assert (diff != 0).sum() == 1 and diff.max() in (1, 2)
        assert score > 0


def test_play_random_game_reproducible():
    a = E.play_random_game(E.make_rng(11))
    b = E.play_random_game(E.make_rng(11))
    assert a == b
    score, tile, n_moves = a
    assert score > 0 and 3 <= tile <= 15 and n_moves > 0


def test_play_random_games_matches_single_games():
    scores, tiles, moves = E.play_random_games(17, 20)
    rng = E.make_rng(17)
    for i in range(20):
        assert E.play_random_game(rng) == (scores[i], tiles[i], moves[i])
    # score is a sum of merges, so it is always even and at least 4 per merge
    assert (scores % 2 == 0).all()


# ---------------------------------------------------------------- spawn location


def test_spawned_cell_finds_the_new_tile():
    rng = np.random.default_rng(3)
    spawn_rng = E.make_rng(7)
    checked = 0
    for _ in range(500):
        after = random_board(rng, p_empty=0.5)
        if E.count_empty(after) == 0:
            continue
        board = E.spawn_tile(after, spawn_rng)
        cell, exp = E.spawned_cell(after, board)
        before, now = E.decode(after).ravel(), E.decode(board).ravel()
        assert before[cell] == 0 and now[cell] == exp and exp in (1, 2)
        assert np.array_equal(np.delete(before, cell), np.delete(now, cell))
        checked += 1
    assert checked > 400


def test_spawned_cell_corners_and_equal_boards():
    empty = E.ZERO
    assert E.spawned_cell(empty, E.from_values([[2, 0, 0, 0]] + [[0] * 4] * 3)) == (0, 1)
    assert E.spawned_cell(empty, E.from_values([[0] * 4] * 3 + [[0, 0, 0, 4]])) == (15, 2)
    b = E.from_values([[2, 4, 8, 16], [0] * 4, [0] * 4, [0] * 4])
    assert E.spawned_cell(b, b) == (-1, 0)
