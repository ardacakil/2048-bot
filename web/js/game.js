// One streamed game, buffered for playback and "watch again" (pure, no DOM).
//
// handle(event, data) takes the /api/play events in order. Every step is replayed with
// board.js and must match the server's board after the move and its spawn, its merge reward and
// its running score; the first disagreement throws a Desync, so the page never draws a board
// the bot did not play.

import { decodeBoard, encodeBoard, initialBoard, maxExponent, slide } from "./board.js";

export const WIRE_VERSION = 1;
const STEP_FIELDS = ["action", "reward", "spawn_cell", "spawn_exp", "board", "score"];

export class Desync extends Error {
  constructor(move, what, expected, got) {
    super(`move ${move}: ${what}: expected ${expected}, got ${got}`);
    this.name = "Desync";
    this.move = move;
  }
}

export class GameBuffer {
  constructor() {
    this.meta = null;
    this.queue = null; // last queued event: {position, queue_length, eta_s?}
    this.boards = []; // boards[k] = cells after move k (boards[0] = the initial board)
    this.steps = []; // steps[k - 1] = {action, reward, spawnCell, spawnExp, score} of move k
    this.result = null; // end event data
    this.reachedAt = null; // first move whose board holds the target tile
  }

  get target() {
    return this.meta?.target ?? null;
  }

  /** Moves received so far. */
  get length() {
    return this.steps.length;
  }

  /** Total moves once the end event arrived, else null. */
  get total() {
    return this.result ? this.result.moves : null;
  }

  get started() {
    return this.boards.length > 0;
  }

  get complete() {
    return this.result !== null;
  }

  board(k) {
    return this.boards[k];
  }

  step(k) {
    return this.steps[k - 1];
  }

  score(k) {
    return k ? this.steps[k - 1].score : 0;
  }

  handle(event, data) {
    if (event === "meta") {
      if (data.v !== WIRE_VERSION) throw new Desync(0, "wire version", WIRE_VERSION, data.v);
      if (data.step_fields && data.step_fields.join() !== STEP_FIELDS.join()) {
        throw new Desync(0, "step fields", STEP_FIELDS.join(), data.step_fields.join());
      }
      this.meta = data;
    } else if (event === "queued") {
      this.queue = data;
    } else if (event === "init") {
      const board = decodeBoard(data.board);
      const fromSpawns = encodeBoard(initialBoard(data.spawns));
      if (fromSpawns !== data.board) throw new Desync(0, "initial board", data.board, fromSpawns);
      this.queue = null;
      this.boards = [board];
      this.noteTarget(0);
    } else if (event === "steps") {
      this.addSteps(data);
    } else if (event === "end") {
      this.finish(data);
    }
  }

  addSteps({ first, steps }) {
    if (!this.started) throw new Desync(first, "steps before init", "init", "steps");
    if (first !== this.steps.length + 1) throw new Desync(first, "first move of batch", this.steps.length + 1, first);
    let k = this.steps.length;
    for (const [action, reward, spawnCell, spawnExp, hex, score] of steps) {
      k += 1;
      const prev = this.boards[k - 1];
      const moved = slide(prev, action);
      if (!moved.legal) throw new Desync(k, "legal move", "a legal move", action);
      if (moved.reward !== reward) throw new Desync(k, "reward", moved.reward, reward);
      if (moved.cells[spawnCell] !== 0) throw new Desync(k, "spawn cell", "empty", spawnCell);
      moved.cells[spawnCell] = spawnExp;
      const expected = encodeBoard(moved.cells);
      if (expected !== hex) throw new Desync(k, "board", expected, hex);
      const runningScore = this.score(k - 1) + reward;
      if (runningScore !== score) throw new Desync(k, "score", runningScore, score);
      this.boards.push(moved.cells);
      this.steps.push({ action, reward, spawnCell, spawnExp, score });
      this.noteTarget(k);
    }
  }

  finish(data) {
    const n = this.steps.length;
    if (data.moves !== n) throw new Desync(n, "move count", n, data.moves);
    if (data.score !== this.score(n)) throw new Desync(n, "final score", this.score(n), data.score);
    const last = this.started ? encodeBoard(this.boards[n]) : null;
    if (data.final_board !== last) throw new Desync(n, "final board", last, data.final_board);
    this.result = data;
  }

  noteTarget(k) {
    if (this.reachedAt === null && this.target && 2 ** maxExponent(this.boards[k]) >= this.target) {
      this.reachedAt = k;
    }
  }
}
