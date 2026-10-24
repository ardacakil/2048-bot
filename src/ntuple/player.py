"""One game loop for every evaluation policy, recording every move and spawn.

    random      uniform over legal moves
    greedy      argmax over legal moves of r + V(afterstate): the training policy, which is
                exactly expectimax at depth 0
    expectimax  expectimax.best_move with `depth` chance layers (depth >= 1)

Common random numbers: spawns always come from make_rng(seed) and nothing else draws from that
stream (the random policy picks its moves from a second stream, make_rng(-1 - seed)). So game
`seed` starts from the same board for every policy and checkpoint, and greedy/expectimax games
are identical to expectimax.play_game on the same seed.

Every game starts with an empty transposition table, so a game depends only on (weights, policy,
depth, prune, tt_bits, seed): evaluation and replay regeneration give bit-identical games.

Corner metrics, per move (board after the move and its spawn): corner_moves counts moves after
which a largest tile is in a corner, stable_moves those after which it is in the same corner as
after the previous move (engine.max_tile_corner: lowest corner index on ties).

Record (filled in place, valid up to the number of moves played):
    spawns[0], spawns[1]   the two initial tiles as (cell, exponent)
    actions[k], rewards[k] move k and its merge reward
    spawns[k + 2]          the tile spawned after move k
"""

import dataclasses
import time

import numpy as np
from numba import njit

from game import engine as E
from ntuple import expectimax as X
from ntuple import model as M
from ntuple.train import MAX_GAME_MOVES

RANDOM, GREEDY, EXPECTIMAX = 0, 1, 2
POLICIES = {"random": RANDOM, "greedy": GREEDY, "expectimax": EXPECTIMAX}
NO_LIMIT = -1
DEFAULT_PRUNE = 1e-4
DEFAULT_TT_BITS = {1: 18, 2: 20, 3: 22}  # per expectimax depth; deeper searches reuse more entries
MAX_TT_BITS = 24


def policy_label(policy, depth=0):
    """'random', 'greedy' or 'expectimax-d<depth>'; also the eval CSV file stem."""
    return f"expectimax-d{depth}" if policy == "expectimax" else policy


def parse_label(label):
    """Inverse of policy_label; also accepts the short forms 'd1', 'd2', ... -> (policy, depth)."""
    if label in ("random", "greedy"):
        return label, 0
    for prefix in ("expectimax-d", "d"):
        if label.startswith(prefix) and label[len(prefix):].isdigit():
            return "expectimax", int(label[len(prefix):])
    raise ValueError(f"unknown policy {label!r}: random, greedy, expectimax-d<N> or d<N>")


def resolve_settings(policy, depth, prune=None, tt_bits=None):
    """(prune, tt_bits) actually used. They only matter for expectimax; 0 for random and greedy."""
    if policy not in POLICIES:
        raise ValueError(f"unknown policy {policy!r}")
    if (policy == "expectimax") != (depth >= 1):
        raise ValueError("expectimax needs depth >= 1; random and greedy use depth 0")
    if policy != "expectimax":
        return 0.0, 0
    prune = DEFAULT_PRUNE if prune is None else float(prune)
    tt_bits = DEFAULT_TT_BITS.get(depth, 22) if tt_bits is None else int(tt_bits)
    if tt_bits > MAX_TT_BITS:
        raise ValueError(f"tt_bits {tt_bits} > {MAX_TT_BITS} ({24 << tt_bits >> 20} MB per process)")
    return prune, tt_bits


@njit(
    f"Tuple((int64, int64, int64, uint64, int64, int64))({M.W_TYPE}, int64, int64, int64, float64, "
    "uint64[:], int64[:], float64[:], boolean, uint64, int64, int64[:], int64[:], int64[:, :])",
    cache=True,
)
def play_game(W, seed, policy, depth, prune, tt_keys, tt_depths, tt_vals, use_tt,
              start_board, max_moves, actions, rewards, spawns):
    """Play one game (or max_moves moves from start_board).

    Returns (score, max_exp, moves, board, corner_moves, stable_moves).

    start_board = 0 starts a new game; max_moves = NO_LIMIT plays until no move is legal.
    """
    rng = E.make_rng(seed)
    act_rng = E.make_rng(-1 - seed)
    if start_board == E.ZERO:
        first = E.spawn_tile(E.ZERO, rng)  # the same two draws as E.new_game
        board = E.spawn_tile(first, rng)
        spawns[0, 0], spawns[0, 1] = E.spawned_cell(E.ZERO, first)
        spawns[1, 0], spawns[1, 1] = E.spawned_cell(first, board)
    else:
        board = start_board
        spawns[0, 0], spawns[0, 1] = -1, 0
        spawns[1, 0], spawns[1, 1] = -1, 0
    legal = np.empty(4, np.int64)
    score = 0
    n = 0
    corner_moves = 0
    stable_moves = 0
    prev_corner = E.max_tile_corner(board)
    while n != max_moves:
        if policy == RANDOM:
            k = 0
            for a in range(4):
                if E.move(board, a)[0] != board:
                    legal[k] = a
                    k += 1
            a = legal[E.rng_below(act_rng, k)] if k > 0 else -1
        elif policy == GREEDY:
            a = M.greedy_move(W, board)[0]
        else:
            a = X.best_move(W, board, depth, 1.0, prune, tt_keys, tt_depths, tt_vals, use_tt)[0]
        if a < 0:
            break
        after, r = E.move(board, a)
        board = E.spawn_tile(after, rng)
        actions[n] = a
        rewards[n] = r
        spawns[n + 2, 0], spawns[n + 2, 1] = E.spawned_cell(after, board)
        corner = E.max_tile_corner(board)
        if corner >= 0:
            corner_moves += 1
            if corner == prev_corner:
                stable_moves += 1
        prev_corner = corner
        score += r
        n += 1
    return score, E.max_tile(board), n, board, corner_moves, stable_moves


@dataclasses.dataclass
class GameRecord:
    seed: int
    score: int
    max_tile: int  # tile value, e.g. 16384
    moves: int
    final_board: int
    wall_s: float
    corner_frac: float  # share of moves after which a largest tile is in a corner
    corner_stable_frac: float  # ... in the same corner as after the previous move
    init: np.ndarray = None  # (2, 2) initial (cell, exponent); None for games from start_board
    actions: np.ndarray = None
    rewards: np.ndarray = None
    spawns: np.ndarray = None  # (moves, 2) (cell, exponent) after each move


_NO_WEIGHTS = np.zeros((1, 1), np.float32)  # the random policy never reads W


class Player:
    """A policy plus its preallocated record buffers and transposition table."""

    def __init__(self, W=None, policy="greedy", depth=0, prune=None, tt_bits=None):
        self.prune, self.tt_bits = resolve_settings(policy, depth, prune, tt_bits)
        if policy != "random" and W is None:
            raise ValueError(f"{policy} needs weights")
        self.policy, self.depth = policy, depth
        self.label = policy_label(policy, depth)
        self.W = _NO_WEIGHTS if policy == "random" else W
        self.use_tt = policy == "expectimax"
        self.table = X.new_table(self.tt_bits)
        self.actions = np.zeros(MAX_GAME_MOVES, np.int64)
        self.rewards = np.zeros(MAX_GAME_MOVES, np.int64)
        self.spawns = np.zeros((MAX_GAME_MOVES + 2, 2), np.int64)

    def play(self, seed, start_board=0, max_moves=NO_LIMIT, record=True):
        """Play game `seed` with an empty transposition table; returns a GameRecord."""
        self.table[1][:] = X.NO_ENTRY
        t0 = time.perf_counter()
        score, max_exp, n, board, corner, stable = play_game(
            self.W, seed, POLICIES[self.policy], self.depth, self.prune, *self.table, self.use_tt,
            start_board, max_moves, self.actions, self.rewards, self.spawns)
        wall = time.perf_counter() - t0
        rec = GameRecord(int(seed), int(score), 1 << int(max_exp), int(n), int(board), wall,
                         corner / n if n else 0.0, stable / n if n else 0.0)
        if record:
            if start_board == 0:
                rec.init = self.spawns[:2].copy()
            rec.actions = self.actions[:n].copy()
            rec.rewards = self.rewards[:n].copy()
            rec.spawns = self.spawns[2:n + 2].copy()
        return rec
