"""Play 2048 in the terminal on the Numba engine.

    uv run python -m game.play [--seed N]

Arrow keys or WASD to move, r to restart, q to quit.
"""

import argparse
import curses
import time

from game import engine as E

CELL_W, CELL_H = 8, 3

KEYS = {
    curses.KEY_UP: E.UP, ord("w"): E.UP, ord("W"): E.UP,
    curses.KEY_RIGHT: E.RIGHT, ord("d"): E.RIGHT, ord("D"): E.RIGHT,
    curses.KEY_DOWN: E.DOWN, ord("s"): E.DOWN, ord("S"): E.DOWN,
    curses.KEY_LEFT: E.LEFT, ord("a"): E.LEFT, ord("A"): E.LEFT,
}

# exponent -> (foreground, background) in the xterm 256-colour palette, close to the
# original game's colours. 0 = empty cell, 12+ = everything past 2048.
COLORS_256 = {
    0: (243, 250),
    1: (238, 255),
    2: (238, 230),
    3: (231, 215),
    4: (231, 209),
    5: (231, 203),
    6: (231, 196),
    7: (238, 222),
    8: (238, 221),
    9: (238, 220),
    10: (231, 214),
    11: (231, 208),
    12: (231, 237),
}
# fallback for 8-colour terminals
COLORS_8 = {
    0: (curses.COLOR_WHITE, curses.COLOR_BLACK),
    1: (curses.COLOR_BLACK, curses.COLOR_WHITE),
    2: (curses.COLOR_BLACK, curses.COLOR_YELLOW),
    3: (curses.COLOR_WHITE, curses.COLOR_RED),
    4: (curses.COLOR_WHITE, curses.COLOR_MAGENTA),
    5: (curses.COLOR_WHITE, curses.COLOR_RED),
    6: (curses.COLOR_WHITE, curses.COLOR_MAGENTA),
    7: (curses.COLOR_BLACK, curses.COLOR_YELLOW),
    8: (curses.COLOR_BLACK, curses.COLOR_GREEN),
    9: (curses.COLOR_BLACK, curses.COLOR_CYAN),
    10: (curses.COLOR_WHITE, curses.COLOR_BLUE),
    11: (curses.COLOR_BLACK, curses.COLOR_GREEN),
    12: (curses.COLOR_WHITE, curses.COLOR_BLACK),
}


class Game:
    def __init__(self, seed):
        self.seed = seed
        self.rng = E.make_rng(seed)
        self.restart()

    def restart(self):
        self.board = E.new_game(self.rng)
        self.score = 0
        self.moves = 0
        self.over = False

    def step(self, action):
        after, reward = E.move(self.board, action)
        if after == self.board:  # illegal: ignored
            return
        self.board = E.spawn_tile(after, self.rng)
        self.score += reward
        self.moves += 1
        self.over = E.is_game_over(self.board)


def init_colors():
    if not curses.has_colors():
        return False
    curses.start_color()
    try:
        curses.use_default_colors()
    except curses.error:
        pass
    palette = COLORS_256 if curses.COLORS >= 256 else COLORS_8
    for exp, (fg, bg) in palette.items():
        curses.init_pair(exp + 1, fg, bg)
    return True


def draw(scr, game, has_color):
    scr.erase()
    rows, cols = scr.getmaxyx()
    board_w, board_h = 4 * CELL_W + 5, 4 * CELL_H + 5
    if rows < board_h + 4 or cols < board_w:
        scr.addstr(0, 0, f"Terminal too small: need {board_w}x{board_h + 4}")
        scr.refresh()
        return

    best = 1 << E.max_tile(game.board)
    scr.addstr(0, 0, "2048", curses.A_BOLD)
    scr.addstr(1, 0, f"Score {game.score:<8d} Moves {game.moves:<6d} Best {best}")

    grid = E.decode(game.board)
    top = 3
    for r in range(4):
        for c in range(4):
            exp = int(grid[r, c])
            y = top + 1 + r * (CELL_H + 1)
            x = 1 + c * (CELL_W + 1)
            label = str(1 << exp) if exp else ""
            if has_color:
                attr = curses.color_pair(min(exp, 12) + 1) | (curses.A_BOLD if exp else 0)
            else:
                attr = curses.A_REVERSE if exp else curses.A_NORMAL
            for dy in range(CELL_H):
                text = label if dy == CELL_H // 2 else ""
                scr.addstr(y + dy, x, text.center(CELL_W), attr)

    y = top + board_h
    if game.over:
        scr.addstr(y, 0, "Game over!  r = restart, q = quit", curses.A_BOLD)
    else:
        scr.addstr(y, 0, "arrows/WASD move   r restart   q quit")
    scr.refresh()


ESC_ARROWS = {ord("A"): curses.KEY_UP, ord("B"): curses.KEY_DOWN,
              ord("C"): curses.KEY_RIGHT, ord("D"): curses.KEY_LEFT}


def read_key(scr):
    """getch(), plus decoding of raw ESC [ A / ESC O A arrow sequences that curses did not
    translate (otherwise the trailing letter would be read as WASD)."""
    key = scr.getch()
    if key != 27:
        return key
    scr.nodelay(True)
    try:
        second, third = scr.getch(), scr.getch()
    finally:
        scr.nodelay(False)
    if second in (ord("["), ord("O")) and third in ESC_ARROWS:
        return ESC_ARROWS[third]
    return -1


def main(scr, seed):
    curses.curs_set(0)
    scr.keypad(True)
    curses.set_escdelay(25)
    has_color = init_colors()
    game = Game(seed)
    while True:
        draw(scr, game, has_color)
        key = read_key(scr)
        if key in (ord("q"), ord("Q")):
            return game
        if key in (ord("r"), ord("R")):
            game.restart()
        elif key in KEYS and not game.over:
            game.step(KEYS[key])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Play 2048 in the terminal.")
    parser.add_argument("--seed", type=int, default=None, help="RNG seed (default: time-based)")
    args = parser.parse_args()
    seed = args.seed if args.seed is not None else time.time_ns() % (2**63)
    game = curses.wrapper(main, seed)
    print(f"Final score {game.score} in {game.moves} moves, best tile {1 << E.max_tile(game.board)} (seed {seed})")
