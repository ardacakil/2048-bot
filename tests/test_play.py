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
