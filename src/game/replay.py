"""Replay files: one finished game as JSON, enough for a browser to replay and animate it from an
empty board. The phase-4 server streams the same steps as SSE events.

    {"schema": "2048-replay", "version": 1,
     "encoding": {...},                       # ENCODING below, so a file explains itself
     "meta": {"run", "checkpoint", "games_trained", "policy", "depth", "prune", "tt_bits",
              "seed", "score", "max_tile", "moves", "final_board", "selection", "created"},
     "init": [[cell, exp], [cell, exp]],      # the two initial tiles, in spawn order
     "step_fields": ["action", "reward", "spawn_cell", "spawn_exp"],
     "steps": [[3, 0, 14, 1], ...],           # one per move; every legal move is followed by a spawn
     "checks": {"every": 100, "boards": [...], "scores": [...]}}   # after steps 100, 200, ...

Boards are "0x" + 16 hex digits of the engine's uint64: cell i (row-major, 0 = top-left) is bits
4i..4i+3, i.e. hex digit 15 - i after the "0x" (cell 0 is the last digit). A JS client must
decode it nibble by nibble from the string (or with BigInt), never via Number: a double holds
only 53 bits, so the low cells would be silently corrupted.

Action indices 0-3 are up, right, down, left, the same as engine.UP, RIGHT, DOWN, LEFT.

replay(doc) runs a file through the engine and raises ReplayDesync at the first step where the
file disagrees with it (illegal move, wrong reward, spawn on an occupied cell, check mismatch).
"""

import json
from pathlib import Path

import numpy as np

from game import engine as E

SCHEMA = "2048-replay"
VERSION = 1
CHECK_EVERY = 100
STEP_FIELDS = ("action", "reward", "spawn_cell", "spawn_exp")
ENCODING = {
    "cells": "row-major index 0-15: 0 = top-left, 3 = top-right, 12 = bottom-left, 15 = bottom-right",
    "tiles": "exponent e: 0 = empty, otherwise the tile value is 2**e",
    "board_hex": ("'0x' + 16 hex digits of a 64-bit board; cell i is bits 4i..4i+3, i.e. hex digit "
                  "15 - i after the '0x' (cell 0 is the last digit). JS: decode nibble by nibble from "
                  "the string (or with BigInt), never via Number, which keeps only 53 bits"),
    "actions": ["up", "right", "down", "left"],
    "action_index": "step action 0-3 = up, right, down, left (engine.UP, RIGHT, DOWN, LEFT)",
    "steps": "per move: action, merge reward, then the tile spawned after it (cell, exponent)",
    "checks": "board and score after steps every, 2*every, ... for desync detection",
}


class ReplayDesync(ValueError):
    def __init__(self, step, what, expected, got):
        self.step, self.what, self.expected, self.got = step, what, expected, got
        super().__init__(f"step {step}: {what}: expected {expected}, got {got}")


def board_hex(board):
    return f"0x{int(board):016x}"


def parse_board_hex(text):
    if not (isinstance(text, str) and text.startswith("0x") and len(text) == 18):
        raise ValueError(f"not a board: {text!r}")
    return int(text, 16)


def _place(board, cell, exp, step):
    """Put tile `exp` on `cell`, which must be empty."""
    if not (0 <= cell < 16 and 1 <= exp <= E.MAX_EXPONENT):
        raise ReplayDesync(step, "spawn", "cell 0-15 and exponent 1-15", (cell, exp))
    if (board >> (4 * cell)) & 0xF:
        raise ReplayDesync(step, f"spawn on occupied cell {cell}", "empty", board_hex(board))
    return board | (exp << (4 * cell))


def trajectory(init, actions, spawns):
    """Boards after the initial spawns and after each step: array of len(actions) + 1 uint64.

    Trusts its input (no legality checks); use replay() to verify a file.
    """
    board = 0
    for cell, exp in init:
        board |= int(exp) << (4 * int(cell))
    out = np.empty(len(actions) + 1, np.uint64)
    out[0] = board
    for k, (a, (cell, exp)) in enumerate(zip(actions, spawns)):
        board = int(E.move(board, int(a))[0]) | (int(exp) << (4 * int(cell)))
        out[k + 1] = board
    return out


def make_doc(meta, init, actions, rewards, spawns, check_every=CHECK_EVERY):
    """Replay document for a recorded game. `meta` must contain score, moves and final_board."""
    boards = trajectory(init, actions, spawns)
    scores = np.cumsum(rewards)
    idx = range(check_every, len(actions) + 1, check_every)
    return {
        "schema": SCHEMA,
        "version": VERSION,
        "encoding": ENCODING,
        "meta": {**meta, "final_board": board_hex(meta["final_board"])},
        "init": [[int(c), int(e)] for c, e in init],
        "step_fields": list(STEP_FIELDS),
        "steps": [[int(a), int(r), int(c), int(e)] for a, r, (c, e) in zip(actions, rewards, spawns)],
        "checks": {"every": check_every,
                   "boards": [board_hex(boards[i]) for i in idx],
                   "scores": [int(scores[i - 1]) for i in idx]},
    }


def write(doc, path):
    """Compact JSON (no whitespace): ~12 bytes per step."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(doc, separators=(",", ":"), ensure_ascii=False))
    return path


def load(path):
    doc = json.loads(Path(path).read_text())
    if doc.get("schema") != SCHEMA or doc.get("version") != VERSION:
        raise ValueError(f"{path}: not a {SCHEMA} v{VERSION} file "
                         f"(schema {doc.get('schema')!r}, version {doc.get('version')!r})")
    return doc


def replay(doc):
    """Run a replay document through the engine; returns (final_board, score).

    Raises ReplayDesync at the first disagreement, including with the recorded meta.
    """
    if list(doc["step_fields"]) != list(STEP_FIELDS):
        raise ValueError(f"unknown step_fields {doc['step_fields']}")
    board = 0
    for k, (cell, exp) in enumerate(doc["init"]):
        board = _place(board, cell, exp, f"init {k}")
    every = doc["checks"]["every"]
    checks = dict(zip(range(every, every * (len(doc["checks"]["boards"]) + 1), every),
                      zip(doc["checks"]["boards"], doc["checks"]["scores"])))
    score = 0
    for k, (a, r, cell, exp) in enumerate(doc["steps"], 1):
        if not 0 <= a < 4:
            raise ReplayDesync(k, "action", "0-3", a)
        after, reward = E.move(board, a)
        after = int(after)
        if after == board:
            raise ReplayDesync(k, f"illegal move {E.ACTION_NAMES[a]}", "a legal move", board_hex(board))
        if reward != r:
            raise ReplayDesync(k, "reward", int(reward), r)
        board = _place(after, cell, exp, k)
        score += r
        if k in checks:
            want_board, want_score = checks.pop(k)
            if board_hex(board) != want_board:
                raise ReplayDesync(k, "board check", want_board, board_hex(board))
            if score != want_score:
                raise ReplayDesync(k, "score check", want_score, score)
    if checks:
        raise ReplayDesync(len(doc["steps"]), "checks past the last step", "none", sorted(checks))
    meta = doc["meta"]
    n = len(doc["steps"])
    for what, want, got in (("move count", meta["moves"], n), ("final score", meta["score"], score),
                            ("final board", meta["final_board"], board_hex(board)),
                            ("max tile", meta["max_tile"], 1 << E.max_tile(board))):
        if want != got:
            raise ReplayDesync(n, what, want, got)
    if not E.is_game_over(board):
        raise ReplayDesync(n, "game over", "no legal move", board_hex(board))
    return board, score
