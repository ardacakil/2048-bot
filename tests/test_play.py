from game import engine as E
from game.play import Game


def test_game_step_and_restart():
    game = Game(seed=1)
    assert E.count_empty(game.board) == 14
    for a in [E.UP, E.RIGHT, E.DOWN, E.LEFT] * 10:
        game.step(a)
    assert game.moves > 0 and game.score > 0
    game.restart()
    assert (game.score, game.moves, game.over) == (0, 0, False)
    assert E.count_empty(game.board) == 14


def test_illegal_move_is_ignored():
    game = Game(seed=0)
    game.board = E.from_values([[2, 0, 0, 0], [4, 0, 0, 0], [8, 0, 0, 0], [16, 0, 0, 0]])
    game.step(E.LEFT)
    assert game.moves == 0
    assert game.board == E.from_values([[2, 0, 0, 0], [4, 0, 0, 0], [8, 0, 0, 0], [16, 0, 0, 0]])


def test_game_over_detected():
    game = Game(seed=0)
    while not game.over:
        before = game.moves
        for a in range(4):
            game.step(a)
        assert game.moves > before or game.over
    assert E.is_game_over(game.board)


def test_until_stops_when_the_tile_first_appears():
    game = Game(seed=0, until=2048)
    game.board = E.from_values([[512, 512, 1024, 0], [0] * 4, [0] * 4, [0] * 4])
    game.reached = False
    game.step(E.LEFT)  # 1024 1024: not yet
    assert not game.reached and not game.done and game.moves == 1
    game.step(E.LEFT)  # 2048 appears on move 2
    assert game.reached and game.done and game.moves == 2
    board = game.board
    game.step(E.RIGHT)  # ignored once the target is reached
    assert game.moves == 2 and game.board == board


def test_without_until_the_game_never_counts_as_reached():
    game = Game(seed=0)
    game.board = E.from_values([[1024, 1024, 0, 0], [0] * 4, [0] * 4, [0] * 4])
    game.step(E.LEFT)
    assert not game.reached and not game.done and game.moves == 1


def test_engine_has_tile():
    b = E.from_values([[2048, 2, 0, 0], [0] * 4, [0] * 4, [0] * 4])
    assert E.has_tile(b, 2048) and E.has_tile(b, 1024) and not E.has_tile(b, 4096)
