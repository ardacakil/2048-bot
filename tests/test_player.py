import numpy as np
import pytest

from game import engine as E
from game import replay as R
from ntuple import expectimax as X
from ntuple import model as M
from ntuple import player as P


def python_greedy_game(W, seed):
    rng = E.make_rng(seed)
    board, score, n = E.new_game(rng), 0, 0
    while True:
        a, after, r, _ = M.greedy_move(W, board)
        if a < 0:
            return score, E.max_tile(board), n
        score, n = score + r, n + 1
        board = E.spawn_tile(after, rng)


@pytest.mark.parametrize("seed", range(4))
def test_greedy_matches_python_loop_and_expectimax_depth0(W_eval, seed):
    rec = P.Player(W_eval, "greedy").play(seed)
    got = (rec.score, rec.max_tile.bit_length() - 1, rec.moves)
    assert got == python_greedy_game(W_eval, seed)
    ex = X.Expectimax(W_eval, depth=0, use_tt=False)
    assert got == ex.play_game(E.make_rng(seed))


@pytest.mark.parametrize("depth", [1, 2])
def test_expectimax_matches_existing_play_game(W_eval, depth):
    player = P.Player(W_eval, "expectimax", depth)
    for seed in range(3):
        rec = player.play(seed)
        ex = X.Expectimax(W_eval, depth=depth, prune=player.prune, tt_bits=player.tt_bits)  # fresh table
        assert (rec.score, rec.max_tile.bit_length() - 1, rec.moves) == ex.play_game(E.make_rng(seed))


def test_every_policy_starts_from_the_same_board(W_eval):
    seed = 11
    rng = E.make_rng(seed)
    first = E.spawn_tile(E.ZERO, rng)
    expected = [list(E.spawned_cell(E.ZERO, first)), list(E.spawned_cell(first, E.spawn_tile(first, rng)))]
    players = [P.Player(None, "random"), P.Player(W_eval, "greedy"), P.Player(M.new_weights(), "greedy"),
               P.Player(W_eval, "expectimax", 1)]
    for player in players:
        assert player.play(seed).init.tolist() == expected


@pytest.mark.parametrize("policy, depth", [("random", 0), ("greedy", 0), ("expectimax", 1)])
def test_spawns_come_only_from_the_seed_stream(W_eval, policy, depth):
    """Replaying the recorded moves with a fresh make_rng(seed) reproduces every recorded spawn,
    so the random policy's move choices never consume spawn randomness."""
    rec = P.Player(W_eval, policy, depth).play(5)
    rng = E.make_rng(5)
    board = E.new_game(rng)
    for a, (cell, exp) in zip(rec.actions, rec.spawns):
        after = E.move(board, int(a))[0]
        board = E.spawn_tile(after, rng)
        assert E.spawned_cell(after, board) == (cell, exp)
    assert board == rec.final_board


def test_random_policy_moves_are_legal_and_vary_with_seed():
    player = P.Player(None, "random")
    recs = [player.play(s) for s in range(20)]
    for rec in recs:
        boards = R.trajectory(rec.init, rec.actions, rec.spawns)
        for board, a in zip(boards[:-1], rec.actions):
            assert E.is_legal(int(board), int(a))
        assert E.is_game_over(int(boards[-1])) and int(boards[-1]) == rec.final_board
    assert len({r.score for r in recs}) > 10


@pytest.mark.parametrize("policy, depth", [("random", 0), ("greedy", 0), ("expectimax", 2)])
def test_same_seed_gives_identical_record(W_eval, policy, depth):
    a = P.Player(W_eval, policy, depth)
    first = a.play(3)
    a.play(4)  # a different game in between: the table is cleared per game
    again = a.play(3)
    other = P.Player(W_eval, policy, depth).play(3)
    for rec in (again, other):
        assert (rec.score, rec.moves, rec.final_board) == (first.score, first.moves, first.final_board)
        for k in ("init", "actions", "rewards", "spawns"):
            assert np.array_equal(getattr(rec, k), getattr(first, k))


def test_record_is_consistent(W_eval):
    rec = P.Player(W_eval, "expectimax", 1).play(2)
    assert rec.rewards.sum() == rec.score
    assert len(rec.actions) == len(rec.rewards) == len(rec.spawns) == rec.moves
    assert rec.max_tile == 1 << E.max_tile(rec.final_board)
    assert set(rec.spawns[:, 1].tolist()) <= {1, 2}


def test_start_board_and_max_moves(W_eval):
    start = E.from_values([[2, 4, 8, 16], [0, 0, 0, 0], [0, 0, 2, 0], [0, 0, 0, 0]])
    player = P.Player(W_eval, "greedy")
    rec = player.play(9, start_board=start, max_moves=10)
    assert rec.moves == 10 and rec.init is None and len(rec.actions) == 10
    board = start
    for a, (cell, exp) in zip(rec.actions, rec.spawns):
        board = int(E.move(board, int(a))[0]) | (int(exp) << (4 * int(cell)))
    assert board == rec.final_board
    assert player.play(9, max_moves=0).moves == 0


def test_resolve_settings_and_validation(W_eval):
    assert P.resolve_settings("greedy", 0, 0.5, 20) == (0.0, 0)
    assert P.resolve_settings("random", 0) == (0.0, 0)
    assert P.resolve_settings("expectimax", 3) == (1e-4, 22)
    assert P.resolve_settings("expectimax", 1, 0.01, 12) == (0.01, 12)
    for policy, depth in (("expectimax", 0), ("greedy", 1), ("random", 2)):
        with pytest.raises(ValueError):
            P.resolve_settings(policy, depth)
    with pytest.raises(ValueError):
        P.resolve_settings("expectimax", 2, tt_bits=P.MAX_TT_BITS + 1)
    with pytest.raises(ValueError):
        P.Player(None, "greedy")
    with pytest.raises(ValueError):
        P.resolve_settings("minimax", 1)


def test_labels_round_trip():
    for policy, depth in (("random", 0), ("greedy", 0), ("expectimax", 1), ("expectimax", 3)):
        assert P.parse_label(P.policy_label(policy, depth)) == (policy, depth)
    assert P.policy_label("expectimax", 2) == "expectimax-d2"
    assert P.parse_label("d3") == ("expectimax", 3)
    with pytest.raises(ValueError):
        P.parse_label("expectimax")


def reference_corner_counts(boards):
    """Plain-Python corner metric over boards[0] (start) .. boards[n] (after move n)."""
    def corner(b):
        cells = E.decode(int(b)).ravel()
        tops = [c for c in (0, 3, 12, 15) if cells[c] == cells.max()]
        return tops[0] if tops else -1
    in_corner = stable = 0
    prev = corner(boards[0])
    for b in boards[1:]:
        c = corner(b)
        in_corner += c >= 0
        stable += c >= 0 and c == prev
        prev = c
    return in_corner, stable


def test_reference_corner_counts_on_a_hand_built_game():
    V = E.from_values
    z = [0] * 4
    boards = [V([[64, 2, 0, 0], z, z, z]),      # start: 64 in corner 0
              V([[64, 2, 2, 0], z, z, z]),      # 1: corner 0, same corner     -> corner, stable
              V([[0, 0, 4, 64], z, z, z]),      # 2: corner 3, moved           -> corner
              V([[0, 0, 4, 64], z, z, [2, 0, 0, 0]]),  # 3: still corner 3     -> corner, stable
              V([[0, 0, 4, 0], [0, 0, 0, 64], z, z]),  # 4: off the corner     -> -
              V([[0, 0, 4, 0], z, z, [2, 0, 0, 128]])]  # 5: corner 15, new    -> corner
    assert reference_corner_counts(boards) == (4, 2)


@pytest.mark.parametrize("policy, depth", [("random", 0), ("greedy", 0), ("expectimax", 1)])
def test_corner_fractions_match_reference(W_eval, policy, depth):
    player = P.Player(W_eval, policy, depth)
    for seed in range(3):
        rec = player.play(seed)
        corner, stable = reference_corner_counts(R.trajectory(rec.init, rec.actions, rec.spawns))
        assert rec.corner_frac == corner / rec.moves and rec.corner_stable_frac == stable / rec.moves
        assert 0 <= rec.corner_stable_frac <= rec.corner_frac <= 1


def test_corner_fractions_from_a_start_board(W_eval):
    start = E.from_values([[256, 128, 64, 32], [0, 0, 0, 8], [0, 0, 2, 0], [0, 0, 0, 0]])
    rec = P.Player(W_eval, "greedy").play(1, start_board=start, max_moves=5)
    boards = [start]
    for a, (cell, exp) in zip(rec.actions, rec.spawns):
        boards.append(int(E.move(boards[-1], int(a))[0]) | (int(exp) << (4 * int(cell))))
    corner, stable = reference_corner_counts(boards)
    assert (rec.corner_frac, rec.corner_stable_frac) == (corner / 5, stable / 5)
    assert P.Player(W_eval, "greedy").play(1, max_moves=0).corner_frac == 0.0
