"""Slow, obviously-correct 2048 rules in plain Python, used only to test the engine.

Boards are 4x4 lists of exponents (0 = empty, 1 = 2, ...). Shares no code with engine.py.
Actions: 0 = up, 1 = right, 2 = down, 3 = left.
"""

MAX_EXPONENT = 15


def slide_row_left(row):
    """Compress, merge equal neighbours once, compress again. Returns (new_row, score)."""
    tiles = [t for t in row if t != 0]
    score = 0
    for i in range(len(tiles) - 1):
        if tiles[i] != 0 and tiles[i] == tiles[i + 1] and tiles[i] < MAX_EXPONENT:
            tiles[i] += 1
            tiles[i + 1] = 0
            score += 2 ** tiles[i]
    tiles = [t for t in tiles if t != 0]
    return tiles + [0] * (4 - len(tiles)), score


def _transpose(grid):
    return [list(col) for col in zip(*grid)]


def _mirror(grid):
    return [row[::-1] for row in grid]


def _left(grid):
    rows, scores = zip(*(slide_row_left(row) for row in grid))
    return [list(r) for r in rows], sum(scores)


def move(grid, action):
    """Returns (new_grid, score). Input is not modified."""
    if action == 3:  # left
        return _left(grid)
    if action == 1:  # right
        out, score = _left(_mirror(grid))
        return _mirror(out), score
    if action == 0:  # up
        out, score = _left(_transpose(grid))
        return _transpose(out), score
    if action == 2:  # down
        out, score = _left(_mirror(_transpose(grid)))
        return _transpose(_mirror(out)), score
    raise ValueError(f"bad action {action}")


def legal_moves(grid):
    return [move(grid, a)[0] != grid for a in range(4)]


def is_game_over(grid):
    return not any(legal_moves(grid))
