"""Depth-limited expectimax over afterstates, with V(afterstate) at the leaves.

    max node    (board):      max over legal a of  r(a) + chance(afterstate(a), depth)
                              (no legal move: 0)
    chance node (afterstate): depth 0 -> V(afterstate); otherwise the average over every
                              empty cell of 0.9 * max(cell=2) + 0.1 * max(cell=4), at depth-1

depth = number of chance layers expanded; depth 0 is exactly the greedy policy used in
training (same tie-breaking: lowest action index wins).

Speedups (both optional, so tests can compare against the exact search):
  * probability pruning: a chance node whose probability of being reached is below
    `prune` is evaluated with V instead of expanded (mostly cuts deep 4-spawn branches).
  * transposition table: fixed-size array keyed by (afterstate, depth), never cleared.
    For fixed weights the exact value of (afterstate, depth) does not depend on the path,
    so entries stay valid across moves and games. With pruning on, a hit may reuse a value
    computed at a different reach probability, a small approximation.
"""

import numpy as np
from numba import njit

from game import engine as E
from ntuple import model as M

NO_ENTRY = np.int64(-1)
_HASH_MUL = np.uint64(0x9E3779B97F4A7C15)
_TWO = np.uint64(2)
_SIG_ARGS = f"{M.W_TYPE}, uint64, int64, float64, float64, uint64[:], int64[:], float64[:], boolean"


def new_table(bits=20):
    """Empty transposition table with 2**bits entries (24 bytes each)."""
    size = 1 << bits
    return (np.zeros(size, np.uint64), np.full(size, NO_ENTRY, np.int64), np.zeros(size, np.float64))


@njit("int64(uint64, int64, int64)", cache=True)
def _slot(after, depth, mask):
    h = (after ^ np.uint64(depth)) * _HASH_MUL
    return np.int64(h >> np.uint64(32)) & mask  # table size <= 2**32


@njit(f"float64({_SIG_ARGS})", cache=True)
def chance_value(W, after, depth, prob, prune, tt_keys, tt_depths, tt_vals, use_tt):
    """Expected value of an afterstate searched `depth` chance layers deep."""
    if depth <= 0 or prob < prune:
        return M.value(W, after)
    slot = 0
    if use_tt:
        slot = _slot(after, depth, tt_keys.shape[0] - 1)
        if tt_keys[slot] == after and tt_depths[slot] == depth:
            return tt_vals[slot]
    n = E.count_empty(after)  # an afterstate of a legal move always has an empty cell
    p2 = prob * 0.9 / n
    p4 = prob * 0.1 / n
    total = 0.0
    shift = E.ZERO
    for _ in range(16):
        if ((after >> shift) & E.NIBBLE) == E.ZERO:
            for tile, p_tile, p_reach in ((E.ONE, 0.9, p2), (_TWO, 0.1, p4)):
                child = after | (tile << shift)
                best = 0.0
                found = False
                for a in range(4):
                    a2, r = E.move(child, a)
                    if a2 != child:
                        v = r + chance_value(W, a2, depth - 1, p_reach, prune,
                                             tt_keys, tt_depths, tt_vals, use_tt)
                        if not found or v > best:
                            best = v
                            found = True
                total += p_tile * best
        shift += E.S4
    v = total / n
    if use_tt:
        tt_keys[slot] = after
        tt_depths[slot] = depth
        tt_vals[slot] = v
    return v


@njit(f"Tuple((int64, float64))({_SIG_ARGS})", cache=True)
def best_move(W, board, depth, prob, prune, tt_keys, tt_depths, tt_vals, use_tt):
    """(action, value) maximising r + chance_value(afterstate). action = -1 if none is legal.

    `prob` is the reach probability of the root (pass 1.0).
    """
    best_a = -1
    best_v = 0.0
    for a in range(4):
        after, r = E.move(board, a)
        if after == board:
            continue
        v = r + chance_value(W, after, depth, prob, prune, tt_keys, tt_depths, tt_vals, use_tt)
        if best_a < 0 or v > best_v:
            best_a = a
            best_v = v
    return best_a, best_v


@njit(
    f"Tuple((int64, int64, int64))({M.W_TYPE}, uint64[:], int64, float64, "
    "uint64[:], int64[:], float64[:], boolean)",
    cache=True,
)
def play_game(W, rng, depth, prune, tt_keys, tt_depths, tt_vals, use_tt):
    """Play one game with expectimax at `depth`. Returns (score, max_tile, n_moves)."""
    board = E.new_game(rng)
    score = 0
    n = 0
    while True:
        a, _ = best_move(W, board, depth, 1.0, prune, tt_keys, tt_depths, tt_vals, use_tt)
        if a < 0:
            break
        after, r = E.move(board, a)
        score += r
        n += 1
        board = E.spawn_tile(after, rng)
    return score, E.max_tile(board), n


class Expectimax:
    """Convenience wrapper holding the weights, settings and transposition table."""

    def __init__(self, W, depth=2, prune=1e-4, tt_bits=20, use_tt=True):
        self.W = W
        self.depth = depth
        self.prune = prune
        self.use_tt = use_tt
        self.table = new_table(tt_bits if use_tt else 0)

    def best_move(self, board):
        """(action, value); action = -1 on a finished board."""
        return best_move(self.W, board, self.depth, 1.0, self.prune, *self.table, self.use_tt)

    def play_game(self, rng):
        return play_game(self.W, rng, self.depth, self.prune, *self.table, self.use_tt)
