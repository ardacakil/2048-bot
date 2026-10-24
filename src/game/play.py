"""Play 2048 in the terminal on the Numba engine.

    uv run python -m game.play [--seed N] [--until TILE]
    uv run python -m game.play --bot RUN [--depth 2] [--delay 0.05] [--seed N] [--until TILE]

Arrow keys or WASD to move, r to restart, q to quit. With --bot, watch the n-tuple
checkpoint checkpoints/RUN play one game with expectimax; q quits. --until stops the game as
soon as a tile of at least that value first appears.
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
    def __init__(self, seed, until=None):
        self.seed = seed
        self.until = until
        self.rng = E.make_rng(seed)
        self.restart()

    def restart(self):
        self.board = E.new_game(self.rng)
        self.score = 0
        self.moves = 0
        self.over = False
        self.reached = self.until is not None and E.has_tile(self.board, self.until)

    @property
    def done(self):
        """No more moves: game over, or the --until tile has appeared."""
        return self.over or self.reached

    def step(self, action):
        if self.done:
            return
        after, reward = E.move(self.board, action)
        if after == self.board:  # illegal: ignored
            return
        self.board = E.spawn_tile(after, self.rng)
        self.score += reward
        self.moves += 1
        self.over = E.is_game_over(self.board)
        self.reached = self.until is not None and E.has_tile(self.board, self.until)


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


def draw(scr, game, has_color, title="2048", footer=None):
    scr.erase()
    rows, cols = scr.getmaxyx()
    board_w, board_h = 4 * CELL_W + 5, 4 * CELL_H + 5
    if rows < board_h + 4 or cols < board_w:
        scr.addstr(0, 0, f"Terminal too small: need {board_w}x{board_h + 4}")
        scr.refresh()
        return

    best = 1 << E.max_tile(game.board)
    scr.addstr(0, 0, title, curses.A_BOLD)
    scr.addstr(1, 0, f"Score {game.score:<8d} Moves {game.moves:<6d} Max tile {best}")

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
    if footer is not None:
        scr.addstr(y, 0, footer, curses.A_BOLD if game.done else curses.A_NORMAL)
    elif game.reached:
        scr.addstr(y, 0, f"Reached {game.until}!  r = restart, q = quit", curses.A_BOLD)
    elif game.over:
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


def main(scr, seed, until=None):
    curses.curs_set(0)
    scr.keypad(True)
    curses.set_escdelay(25)
    has_color = init_colors()
    game = Game(seed, until)
    while True:
        draw(scr, game, has_color)
        key = read_key(scr)
        if key in (ord("q"), ord("Q")):
            return game
        if key in (ord("r"), ord("R")):
            game.restart()
        elif key in KEYS and not game.done:
            game.step(KEYS[key])


def bot_main(scr, seed, bot, title, delay, until=None):
    """Watch the bot play one game (until game over or the --until tile); q quits at any time."""
    curses.curs_set(0)
    has_color = init_colors()
    game = Game(seed, until)
    game.over = E.is_game_over(game.board)
    scr.timeout(max(1, int(delay * 1000)))  # getch() doubles as the delay between moves
    while True:
        if game.reached:
            footer = f"Reached {until} in {game.moves:,} moves!  q = quit"
        else:
            footer = "Game over!  q = quit" if game.over else "bot playing   q quit"
        draw(scr, game, has_color, title, footer)
        if game.done:
            scr.timeout(-1)  # nothing left to animate: wait for q
        if scr.getch() in (ord("q"), ord("Q")):
            return game
        if not game.done:
            action, _ = bot.best_move(game.board)
            game.step(action)


def load_bot(name, depth):
    """Expectimax player on a checkpoint: a run name under checkpoints/ or a directory path.

    Weights are memory-mapped copy-on-write, so the checkpoint file is never modified.
    """
    from ntuple import expectimax as X

    bot, meta, ckpt = X.load(name, depth)
    return bot, f"{ckpt.name} ({meta['games_done']:,} games)"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Play 2048 in the terminal, or watch the bot.")
    parser.add_argument("--seed", type=int, default=None, help="RNG seed (default: time-based)")
    parser.add_argument("--bot", metavar="RUN", help="watch a checkpoint play (run name or directory)")
    parser.add_argument("--depth", type=int, default=2, help="bot expectimax depth (0 = greedy)")
    parser.add_argument("--delay", type=float, default=0.05, help="bot seconds between moves")
    parser.add_argument("--until", type=int, default=None, metavar="TILE",
                        help="stop when a tile of at least this value first appears (e.g. 2048)")
    args = parser.parse_args()
    seed = args.seed if args.seed is not None else time.time_ns() % (2**63)
    if args.bot:
        bot, name = load_bot(args.bot, args.depth)
        bot.best_move(E.new_game(E.make_rng(0)))  # load compiled code before the screen opens
        title = f"2048 bot: {name}, expectimax depth {args.depth}"
        game = curses.wrapper(bot_main, seed, bot, title, args.delay, args.until)
    else:
        game = curses.wrapper(main, seed, args.until)
    status = f", reached {args.until}" if game.reached else ""
    print(f"Final score {game.score} in {game.moves} moves, max tile {1 << E.max_tile(game.board)}"
          f"{status} (seed {seed})")
