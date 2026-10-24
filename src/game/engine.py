"""Fast 2048 engine, Numba-compiled.

Board layout: one uint64 = 4 rows x 16 bits. Row ``r`` occupies bits ``16r .. 16r+15``
(row 0 = top), cell ``(r, c)`` is the nibble at bit ``16r + 4c`` (column 0 = left).
Tiles are exponents: 0 = empty, 1 = 2, 2 = 4, ..., 11 = 2048, max 15.

Actions: 0 = up, 1 = right, 2 = down, 3 = left.

Typing rule: every bit operation on a board uses np.uint64 operands only. Mixing uint64
with signed ints makes Numba promote to int64 or float64, so all constants and shift
amounts below are np.uint64, and public functions have explicit signatures (which also
coerces Python ints passed in from notebooks/tests to uint64).

RNG: each game carries an explicit xorshift64* state, a ``uint64[1]`` array made by
``make_rng(seed)``. The state is a single number, so it can be checkpointed.
"""

import numpy as np
from numba import njit

UP, RIGHT, DOWN, LEFT = 0, 1, 2, 3
ACTION_NAMES = ("up", "right", "down", "left")
MAX_EXPONENT = 15

# ---------------------------------------------------------------- uint64 constants

ZERO = np.uint64(0)
ONE = np.uint64(1)
NIBBLE = np.uint64(0xF)
ROW_MASK = np.uint64(0xFFFF)
S4 = np.uint64(4)
S16 = np.uint64(16)
S32 = np.uint64(32)
S48 = np.uint64(48)

# transpose masks
_T1 = np.uint64(0xF0F00F0FF0F00F0F)
_T2 = np.uint64(0x0000F0F00000F0F0)
_T3 = np.uint64(0x0F0F00000F0F0000)
_T4 = np.uint64(0xFF00FF0000FF00FF)
_T5 = np.uint64(0x00FF00FF00000000)
_T6 = np.uint64(0x00000000FF00FF00)
_S12 = np.uint64(12)
_S24 = np.uint64(24)
_S60 = np.uint64(60)

# RNG constants
_XS_MUL = np.uint64(0x2545F4914F6CDD1D)
_SM_GAMMA = np.uint64(0x9E3779B97F4A7C15)
_SM_MUL1 = np.uint64(0xBF58476D1CE4E5B9)
_SM_MUL2 = np.uint64(0x94D049BB133111EB)
_S11 = np.uint64(11)
_S25 = np.uint64(25)
_S27 = np.uint64(27)
_S30 = np.uint64(30)
_S31 = np.uint64(31)
_INV_2_53 = 1.0 / 9007199254740992.0  # 2**-53

# ---------------------------------------------------------------- row lookup tables


@njit(cache=True)
def _reverse_row(row):
    return (
        ((row & 0xF) << 12)
        | ((row & 0xF0) << 4)
        | ((row & 0xF00) >> 4)
        | ((row & 0xF000) >> 12)
    )


@njit(cache=True)
def _build_tables():
    """Slide-left result and merge score for every 16-bit row, plus the mirrored right tables.

    Pure int64 arithmetic on values < 2**16, no uint64 involved.
    """
    left = np.zeros(65536, np.uint16)
    right = np.zeros(65536, np.uint16)
    score_left = np.zeros(65536, np.uint32)
    score_right = np.zeros(65536, np.uint32)
    vals = np.zeros(4, np.int64)
    out = np.zeros(4, np.int64)
    for row in range(65536):
        # compact non-empty tiles toward column 0
        m = 0
        for c in range(4):
            t = (row >> (4 * c)) & 0xF
            if t != 0:
                vals[m] = t
                m += 1
        # merge equal neighbours, each tile at most once
        out[:] = 0
        k = 0
        j = 0
        score = 0
        while j < m:
            if j + 1 < m and vals[j] == vals[j + 1] and vals[j] < MAX_EXPONENT:
                out[k] = vals[j] + 1
                score += 1 << (vals[j] + 1)
                j += 2
            else:
                out[k] = vals[j]
                j += 1
            k += 1
        left[row] = out[0] | (out[1] << 4) | (out[2] << 8) | (out[3] << 12)
        score_left[row] = score
    for row in range(65536):
        rev = _reverse_row(row)
        right[row] = _reverse_row(np.int64(left[rev]))
        score_right[row] = score_left[rev]
    return left, right, score_left, score_right


ROW_LEFT, ROW_RIGHT, SCORE_LEFT, SCORE_RIGHT = _build_tables()

# ---------------------------------------------------------------- board primitives


@njit("uint64(uint64)", cache=True)
def transpose(board):
    """Swap cell (r, c) with (c, r)."""
    a = (board & _T1) | ((board & _T2) << _S12) | ((board & _T3) >> _S12)
    return (a & _T4) | ((a & _T5) >> _S24) | ((a & _T6) << _S24)


@njit("Tuple((uint64, int64))(uint64)", cache=True)
def move_left(board):
    r0 = board & ROW_MASK
    r1 = (board >> S16) & ROW_MASK
    r2 = (board >> S32) & ROW_MASK
    r3 = (board >> S48) & ROW_MASK
    after = (
        np.uint64(ROW_LEFT[r0])
        | (np.uint64(ROW_LEFT[r1]) << S16)
        | (np.uint64(ROW_LEFT[r2]) << S32)
        | (np.uint64(ROW_LEFT[r3]) << S48)
    )
    reward = (
        np.int64(SCORE_LEFT[r0])
        + np.int64(SCORE_LEFT[r1])
        + np.int64(SCORE_LEFT[r2])
        + np.int64(SCORE_LEFT[r3])
    )
    return after, reward


@njit("Tuple((uint64, int64))(uint64)", cache=True)
def move_right(board):
    r0 = board & ROW_MASK
    r1 = (board >> S16) & ROW_MASK
    r2 = (board >> S32) & ROW_MASK
    r3 = (board >> S48) & ROW_MASK
    after = (
        np.uint64(ROW_RIGHT[r0])
        | (np.uint64(ROW_RIGHT[r1]) << S16)
        | (np.uint64(ROW_RIGHT[r2]) << S32)
        | (np.uint64(ROW_RIGHT[r3]) << S48)
    )
    reward = (
        np.int64(SCORE_RIGHT[r0])
        + np.int64(SCORE_RIGHT[r1])
        + np.int64(SCORE_RIGHT[r2])
        + np.int64(SCORE_RIGHT[r3])
    )
    return after, reward


@njit("Tuple((uint64, int64))(uint64)", cache=True)
def move_up(board):
    # after transposing, column c is row c with the top cell in nibble 0, so up = left
    after, reward = move_left(transpose(board))
    return transpose(after), reward


@njit("Tuple((uint64, int64))(uint64)", cache=True)
def move_down(board):
    after, reward = move_right(transpose(board))
    return transpose(after), reward


@njit("Tuple((uint64, int64))(uint64, int64)", cache=True)
def move(board, action):
    """Slide the board. Returns (afterstate, reward); afterstate == board means illegal.

    The afterstate is the board after the slide, before the random spawn.
    """
    if action == UP:
        return move_up(board)
    elif action == RIGHT:
        return move_right(board)
    elif action == DOWN:
        return move_down(board)
    else:
        return move_left(board)


@njit("uint64(uint64, int64)", cache=True)
def afterstate(board, action):
    """Board after the slide, before the spawn."""
    return move(board, action)[0]


@njit("boolean(uint64, int64)", cache=True)
def is_legal(board, action):
    return move(board, action)[0] != board


@njit("boolean[:](uint64)", cache=True)
def legal_moves(board):
    """Boolean mask of shape (4,), indexed by action."""
    mask = np.zeros(4, np.bool_)
    for a in range(4):
        mask[a] = move(board, a)[0] != board
    return mask


@njit("int64(uint64)", cache=True)
def count_empty(board):
    n = 0
    b = board
    for _ in range(16):
        if (b & NIBBLE) == ZERO:
            n += 1
        b >>= S4
    return n


@njit("int64(uint64)", cache=True)
def max_tile(board):
    """Largest exponent on the board (11 = 2048)."""
    best = ZERO
    b = board
    for _ in range(16):
        t = b & NIBBLE
        if t > best:
            best = t
        b >>= S4
    return np.int64(best)


@njit("int64(uint64)", cache=True)
def max_tile_corner(board):
    """Corner cell (0, 3, 12 or 15, lowest first) that holds a largest tile; -1 if none does."""
    top = np.uint64(max_tile(board))
    if (board & NIBBLE) == top:
        return 0
    if ((board >> _S12) & NIBBLE) == top:
        return 3
    if ((board >> S48) & NIBBLE) == top:
        return 12
    if (board >> _S60) == top:
        return 15
    return -1


@njit("boolean(uint64)", cache=True)
def is_game_over(board):
    """True iff no move is legal."""
    # a non-empty board with an empty cell always has a legal move
    if board != ZERO and count_empty(board) > 0:
        return False
    # on a full board a move is legal iff some neighbours are equal, which is
    # symmetric: left covers horizontal pairs, up covers vertical pairs
    return move_left(board)[0] == board and move_up(board)[0] == board


@njit("uint64(int64[:, :])", cache=True)
def encode(grid):
    """4x4 array of exponents -> board."""
    board = ZERO
    shift = ZERO
    for r in range(4):
        for c in range(4):
            board |= (np.uint64(grid[r, c]) & NIBBLE) << shift
            shift += S4
    return board


@njit("int64[:, :](uint64)", cache=True)
def decode(board):
    """Board -> 4x4 array of exponents."""
    grid = np.zeros((4, 4), np.int64)
    b = board
    for r in range(4):
        for c in range(4):
            grid[r, c] = np.int64(b & NIBBLE)
            b >>= S4
    return grid


# ---------------------------------------------------------------- RNG


@njit("uint64[:](int64)", cache=True)
def make_rng(seed):
    """New RNG state (uint64[1]) from a seed, scrambled with splitmix64."""
    z = np.uint64(seed) + _SM_GAMMA
    z = (z ^ (z >> _S30)) * _SM_MUL1
    z = (z ^ (z >> _S27)) * _SM_MUL2
    z = z ^ (z >> _S31)
    if z == ZERO:  # xorshift state must be non-zero
        z = _SM_GAMMA
    state = np.empty(1, np.uint64)
    state[0] = z
    return state


@njit("uint64(uint64[:])", cache=True)
def rng_next(state):
    """xorshift64*: advance the state, return 64 random bits."""
    x = state[0]
    x ^= x >> _S12
    x ^= x << _S25
    x ^= x >> _S27
    state[0] = x
    return x * _XS_MUL


@njit("float64(uint64[:])", cache=True)
def rng_uniform(state):
    """Uniform float in [0, 1) with 53 random bits."""
    return np.float64(rng_next(state) >> _S11) * _INV_2_53


@njit("int64(uint64[:], int64)", cache=True)
def rng_below(state, n):
    """Uniform integer in [0, n) for 1 <= n < 2**32 (multiply-shift, bias < n / 2**32)."""
    hi = rng_next(state) >> S32
    return np.int64((hi * np.uint64(n)) >> S32)


# ---------------------------------------------------------------- spawning and games


@njit("uint64(uint64, uint64[:])", cache=True)
def spawn_tile(board, rng):
    """Put a 2 (p=0.9) or 4 (p=0.1) on a uniformly random empty cell. Full board: unchanged."""
    n = count_empty(board)
    if n == 0:
        return board
    k = rng_below(rng, n)
    tile = np.uint64(2) if rng_uniform(rng) < 0.1 else ONE
    shift = ZERO
    for _ in range(16):
        if ((board >> shift) & NIBBLE) == ZERO:
            if k == 0:
                return board | (tile << shift)
            k -= 1
        shift += S4
    return board  # unreachable


@njit("Tuple((int64, int64))(uint64, uint64)", cache=True)
def spawned_cell(after, board):
    """(cell, exponent) of the tile that spawn_tile added to `after` to give `board`.

    Returns (-1, 0) if the boards are equal. Only valid when `board` is `after` plus one tile.
    """
    diff = after ^ board
    if diff == ZERO:
        return -1, 0
    cell = 0
    while (diff & NIBBLE) == ZERO:
        diff >>= S4
        cell += 1
    return cell, np.int64(diff & NIBBLE)


@njit("uint64(uint64[:])", cache=True)
def new_game(rng):
    """Empty board with two spawned tiles."""
    return spawn_tile(spawn_tile(ZERO, rng), rng)


@njit("Tuple((int64, int64, int64))(uint64[:])", cache=True)
def play_random_game(rng):
    """Play uniformly random legal moves until game over. Returns (score, max_tile, n_moves)."""
    board = new_game(rng)
    afters = np.zeros(4, np.uint64)
    rewards = np.zeros(4, np.int64)
    score = 0
    n_moves = 0
    while True:
        k = 0
        for a in range(4):
            after, reward = move(board, a)
            if after != board:
                afters[k] = after
                rewards[k] = reward
                k += 1
        if k == 0:
            break
        i = rng_below(rng, k)
        score += rewards[i]
        board = spawn_tile(afters[i], rng)
        n_moves += 1
    return score, max_tile(board), n_moves


@njit("Tuple((int64[:], int64[:], int64[:]))(int64, int64)", cache=True)
def play_random_games(seed, n_games):
    """Play n_games random games from one seeded RNG. Returns (scores, max_tiles, n_moves) arrays."""
    rng = make_rng(seed)
    scores = np.zeros(n_games, np.int64)
    tiles = np.zeros(n_games, np.int64)
    moves = np.zeros(n_games, np.int64)
    for g in range(n_games):
        scores[g], tiles[g], moves[g] = play_random_game(rng)
    return scores, tiles, moves


# ---------------------------------------------------------------- Python-side helpers (not jitted)


def has_tile(board, tile):
    """True once a tile of at least `tile` (a value such as 2048) is on the board."""
    return (1 << max_tile(board)) >= tile


def from_values(rows):
    """4x4 nested list of tile values (0, 2, 4, ...) -> board."""
    grid = np.zeros((4, 4), np.int64)
    for r in range(4):
        for c in range(4):
            v = int(rows[r][c])
            grid[r, c] = 0 if v == 0 else v.bit_length() - 1
    return encode(grid)


def to_values(board):
    """Board -> 4x4 int64 array of tile values (0, 2, 4, ...)."""
    grid = decode(board)
    return np.where(grid == 0, 0, 1 << grid)


def format_board(board, width=6):
    """Text grid of tile values, for printing."""
    sep = "+" + "+".join(["-" * width] * 4) + "+"
    lines = [sep]
    for row in to_values(board):
        cells = [(str(v) if v else ".").center(width) for v in row]
        lines.append("|" + "|".join(cells) + "|")
        lines.append(sep)
    return "\n".join(lines)
