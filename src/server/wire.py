"""Wire format of /api/play: Server-Sent Events, one JSON object per event (standard library only).

    event: meta     game id, seed, target, policy, playback delay, reach rate, encoding notes
    event: queued   position in the queue (repeated while waiting, at least every keepalive_s)
    event: init     initial board and the two spawns
    event: steps    a batch of moves: [action, reward, spawn_cell, spawn_exp, board, score] each
    event: end      reached, moves, score, max tile, compute and queue-wait seconds
    event: error    something failed; the stream ends
    : ka            comment line, keeps proxies (Cloudflare ~100 s idle limit) from closing

Every `data:` line is a single line of compact JSON. Boards are "0x" + 16 hex digits (see
ENCODING): clients must decode them nibble by nibble or with BigInt, never via Number. The same
stream is meant to be read with fetch() + a stream reader (EventSource would reconnect on a
dropped connection, which here would start a new game).
"""

import json

from game import replay as R

VERSION = 1
STEP_FIELDS = ("action", "reward", "spawn_cell", "spawn_exp", "board", "score")
ENCODING = {
    **{k: R.ENCODING[k] for k in ("cells", "tiles", "board_hex", "actions", "action_index")},
    "steps": "per move: action, merge reward, spawned tile (cell, exponent), board after the move and "
             "its spawn, cumulative score",
}
KEEPALIVE = b": ka\n\n"
HEADERS = {"Cache-Control": "no-cache, no-transform", "X-Accel-Buffering": "no"}
MEDIA_TYPE = "text/event-stream; charset=utf-8"


def sse(event, data):
    return f"event: {event}\ndata: {json.dumps(data, separators=(',', ':'))}\n\n".encode()


def meta(game_id, seed, target, depth, prune, tt_bits, model, code, delay_ms, reach_rate):
    return {"v": VERSION, "game_id": game_id, "seed": seed, "target": target,
            "policy": "expectimax" if depth else "greedy", "depth": depth, "prune": prune,
            "tt_bits": tt_bits, "model": model, "code": code, "delay_ms": delay_ms,
            "reach_rate": reach_rate, "step_fields": list(STEP_FIELDS), "encoding": ENCODING}


def queued(position, queue_length, eta_s=None):
    out = {"position": position, "queue_length": queue_length}
    if eta_s is not None:
        out["eta_s"] = round(eta_s, 1)
    return out


def init(board, spawns):
    return {"board": R.board_hex(board), "spawns": [list(s) for s in spawns], "score": 0}


def steps(first, moves, score):
    """Batch of Move objects starting at move number `first`; returns (data, score after the batch)."""
    rows = []
    for m in moves:
        score += m.reward
        rows.append([m.action, m.reward, m.spawn_cell, m.spawn_exp, R.board_hex(m.board), score])
    return {"first": first, "steps": rows}, score


def end(reached, game_over, moves, score, board, compute_s, queue_wait_s):
    status = "reached" if reached else "game_over" if game_over else "max_moves"
    return {"reached": reached, "status": status, "moves": moves, "score": score,
            "max_tile": max_tile(board), "final_board": R.board_hex(board),
            "compute_s": round(compute_s, 3), "queue_wait_s": round(queue_wait_s, 3)}


def max_tile(board):
    return 1 << max((int(board) >> (4 * i)) & 0xF for i in range(16))


def error(code, message):
    return {"code": code, "message": message}


class Parser:
    """Incremental SSE parser: feed() text chunks, get back (event, data) pairs; comments are
    returned as ("comment", text) so clients and tests can see keep-alives."""

    def __init__(self):
        self.buffer = ""
        self.event, self.data = None, []

    def feed(self, text):
        self.buffer += text
        out = []
        while "\n" in self.buffer:
            line, self.buffer = self.buffer.split("\n", 1)
            line = line.rstrip("\r")
            if not line:
                if self.data:
                    out.append((self.event or "message", json.loads("\n".join(self.data))))
                self.event, self.data = None, []
            elif line.startswith(":"):
                out.append(("comment", line[1:].strip()))
            else:
                field, _, value = line.partition(":")
                value = value[1:] if value.startswith(" ") else value
                if field == "event":
                    self.event = value
                elif field == "data":
                    self.data.append(value)
        return out
