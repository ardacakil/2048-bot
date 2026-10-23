"""N-tuple network: four 6-tuples with 8-way symmetry sharing.

Tuple configuration from Wu, Yeh, Liang et al., "Multi-Stage Temporal Difference Learning
for 2048" (2014), the standard "4x6-tuple" network (also used by moporgic/TDL2048). Cells are
numbered row-major, 0 = top-left, matching the engine's nibble order (cell i at bit 4i):

    A {0,1,2,3,4,5}   row 0 + two cells of row 1   (bent)
    B {4,5,6,7,8,9}   row 1 + two cells of row 2   (bent)
    C {0,1,2,4,5,6}   2x3 rectangle, rows 0-1
    D {4,5,6,8,9,10}  2x3 rectangle, rows 1-2

A tuple's index puts its k-th cell (ascending cell order) in nibble k, so every index is
< 16**6. V(board) sums the 4 tuples over all 8 symmetric versions of the board: 32 lookups.

Because the tuple cells are (nearly) contiguous nibbles, each index is 1-3 shift/mask ops on a
transformed board, so the 8 symmetries are applied to the board (bit tricks) rather than via
precomputed per-symmetry cell lists (measured on M4 with random weights and boards from real
games: 28 ns vs 44 ns per V(board)).

Weights: one C-contiguous float32 array of shape (4, 16**6), 268 MB, always passed in as an
argument (never a Numba global), so it can be backed by np.zeros, shared memory (Hogwild
later) or a copy-on-write memmap of a checkpoint.
"""

import numpy as np
from numba import njit

from game.engine import S4, S16, ZERO, move, transpose

TUPLES = (
    (0, 1, 2, 3, 4, 5),
    (4, 5, 6, 7, 8, 9),
    (0, 1, 2, 4, 5, 6),
    (4, 5, 6, 8, 9, 10),
)
NUM_TUPLES = 4
NUM_SYMMETRIES = 8
NUM_LOOKUPS = NUM_TUPLES * NUM_SYMMETRIES
TABLE_SIZE = 16**6

_M12 = np.uint64(0xFFF)
_M12_HI = np.uint64(0xFFF000)
_M24 = np.uint64(0xFFFFFF)
_S12 = np.uint64(12)
_S20 = np.uint64(20)
_S48 = np.uint64(48)
# mirror left-right: swap nibbles c0<->c3, c1<->c2 within every row
_H0 = np.uint64(0x000F000F000F000F)
_H1 = np.uint64(0x00F000F000F000F0)
_H2 = np.uint64(0x0F000F000F000F00)
_H3 = np.uint64(0xF000F000F000F000)
# flip top-bottom: swap rows r0<->r3, r1<->r2
_V1 = np.uint64(0x00000000FFFF0000)
_V2 = np.uint64(0x0000FFFF00000000)

W_TYPE = "float32[:, ::1]"


def new_weights():
    """Zero-initialised weight tables, shape (4, 16**6) float32."""
    return np.zeros((NUM_TUPLES, TABLE_SIZE), np.float32)


@njit("uint64(uint64)", cache=True)
def mirror(board):
    """Mirror left-right."""
    return (
        ((board & _H0) << _S12)
        | ((board & _H1) << S4)
        | ((board & _H2) >> S4)
        | ((board & _H3) >> _S12)
    )


@njit("uint64(uint64)", cache=True)
def flip(board):
    """Flip top-bottom."""
    return (
        (board >> _S48)
        | ((board & _V2) >> S16)
        | ((board & _V1) << S16)
        | (board << _S48)
    )


@njit("uint64[:](uint64)", cache=True)
def symmetries(board):
    """The 8 rotations/reflections of a board: {id, mirror, flip, rot180} x {id, transpose}."""
    out = np.empty(NUM_SYMMETRIES, np.uint64)
    out[0] = board
    out[1] = mirror(board)
    out[2] = flip(board)
    out[3] = flip(out[1])
    for i in range(4):
        out[4 + i] = transpose(out[i])
    return out


@njit("uint64[:](uint64)", cache=True)
def tuple_indices(b):
    """Indices of the 4 tuples on one (already transformed) board."""
    out = np.empty(NUM_TUPLES, np.uint64)
    out[0] = b & _M24
    out[1] = (b >> S16) & _M24
    out[2] = (b & _M12) | ((b >> S4) & _M12_HI)
    out[3] = ((b >> S16) & _M12) | ((b >> _S20) & _M12_HI)
    return out


@njit("int64[:, :](uint64)", cache=True)
def features(board):
    """All 32 lookup indices, shape (4 tuples, 8 symmetries). For tests and inspection."""
    syms = symmetries(board)
    out = np.empty((NUM_TUPLES, NUM_SYMMETRIES), np.int64)
    for s in range(NUM_SYMMETRIES):
        idx = tuple_indices(syms[s])
        for t in range(NUM_TUPLES):
            out[t, s] = np.int64(idx[t])
    return out


@njit(cache=True, inline="always")
def _lookup(W, b):
    return (
        np.float64(W[0, b & _M24])
        + np.float64(W[1, (b >> S16) & _M24])
        + np.float64(W[2, (b & _M12) | ((b >> S4) & _M12_HI)])
        + np.float64(W[3, ((b >> S16) & _M12) | ((b >> _S20) & _M12_HI)])
    )


@njit(cache=True, inline="always")
def _add(W, b, step):
    W[0, b & _M24] += step
    W[1, (b >> S16) & _M24] += step
    W[2, (b & _M12) | ((b >> S4) & _M12_HI)] += step
    W[3, ((b >> S16) & _M12) | ((b >> _S20) & _M12_HI)] += step


@njit(f"float64({W_TYPE}, uint64)", cache=True)
def value(W, board):
    """V(board): sum of 32 weights (4 tuples x 8 symmetries)."""
    m = mirror(board)
    f = flip(board)
    mf = flip(m)
    return (
        _lookup(W, board) + _lookup(W, m) + _lookup(W, f) + _lookup(W, mf)
        + _lookup(W, transpose(board)) + _lookup(W, transpose(m))
        + _lookup(W, transpose(f)) + _lookup(W, transpose(mf))
    )


@njit(f"void({W_TYPE}, uint64, float64)", cache=True)
def update(W, board, step):
    """Add `step` to each of the 32 weights V(board) uses (callers pass alpha * delta / 32)."""
    s = np.float32(step)
    m = mirror(board)
    f = flip(board)
    mf = flip(m)
    _add(W, board, s)
    _add(W, m, s)
    _add(W, f, s)
    _add(W, mf, s)
    _add(W, transpose(board), s)
    _add(W, transpose(m), s)
    _add(W, transpose(f), s)
    _add(W, transpose(mf), s)


@njit(f"Tuple((int64, uint64, int64, float64))({W_TYPE}, uint64)", cache=True)
def greedy_move(W, board):
    """argmax over legal moves of reward + V(afterstate).

    Returns (action, afterstate, reward, V(afterstate)); action = -1 if no move is legal.
    Ties go to the lowest action index.
    """
    best_a = -1
    best_total = 0.0
    best_after = ZERO
    best_r = 0
    best_v = 0.0
    for a in range(4):
        after, r = move(board, a)
        if after == board:
            continue
        v = value(W, after)
        if best_a < 0 or r + v > best_total:
            best_a, best_total, best_after, best_r, best_v = a, r + v, after, r, v
    return best_a, best_after, best_r, best_v
