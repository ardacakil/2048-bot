import assert from "node:assert/strict";
import { test } from "node:test";

import { decodeBoard, isGameOver, maxTile } from "../../web/js/board.js";
import { Desync, GameBuffer } from "../../web/js/game.js";
import { SseParser } from "../../web/js/sse.js";
import { decode, fixture } from "./helpers.mjs";

function events(name) {
  return new SseParser().feed(decode(fixture(name))).filter(([e]) => e !== "comment");
}

function load(list) {
  const g = new GameBuffer();
  for (const [e, d] of list) g.handle(e, d);
  return g;
}

test("reached-2048: every board checks out and the target appears on the last move", () => {
  const g = load(events("reached-2048"));
  assert.ok(g.complete);
  assert.equal(g.target, 2048);
  assert.equal(g.length, g.total);
  assert.equal(g.reachedAt, g.total);
  assert.equal(maxTile(g.board(g.total)), 2048);
  assert.equal(maxTile(g.board(g.total - 1)), 1024);
  assert.deepEqual(g.board(g.total), decodeBoard(g.result.final_board));
  assert.equal(g.score(g.total), g.result.score);
  assert.equal(g.result.reached, true);
  assert.equal(g.queue, null); // cleared by init
});

test("missed-16384: a full game that dies at 8192", () => {
  const g = load(events("missed-16384"));
  assert.equal(g.result.status, "game_over");
  assert.equal(g.reachedAt, null);
  assert.equal(maxTile(g.board(g.total)), 8192);
  assert.ok(isGameOver(g.board(g.total)));
  assert.equal(g.meta.reach_rate.policy, "greedy");
});

test("the queued event is kept until init", () => {
  const list = events("reached-2048");
  const g = new GameBuffer();
  g.handle(...list[0]);
  g.handle(...list[1]);
  assert.deepEqual(g.queue, { position: 2, queue_length: 2, eta_s: 3.5 });
  assert.equal(g.started, false);
});

function tampered(edit) {
  const list = structuredClone(events("reached-2048"));
  edit(list);
  return list;
}

function firstSteps(list) {
  return list.find(([e]) => e === "steps")[1];
}

function desync(list) {
  try {
    load(list);
  } catch (e) {
    if (e instanceof Desync) return e;
    throw e;
  }
  assert.fail("no Desync");
}

for (const [what, edit, move] of [
  ["board", (l) => { firstSteps(l).steps[9][4] = "0x0000000000000001"; }, 10],
  ["reward", (l) => { firstSteps(l).steps[4][1] += 4; }, 5],
  ["score", (l) => { firstSteps(l).steps[2][5] += 2; }, 3],
  ["spawn cell", (l) => { firstSteps(l).steps[0][2] = 99; }, 1],
  ["batch start", (l) => { l.filter(([e]) => e === "steps")[1][1].first += 1; }, 258],
]) {
  test(`a wrong ${what} stops the game with a Desync at move ${move}`, () => {
    assert.equal(desync(tampered(edit)).move, move);
  });
}

test("end must match the buffered game", () => {
  const list = tampered((l) => { l.at(-1)[1].moves += 1; });
  assert.throws(() => load(list), /move count/);
  const list2 = tampered((l) => { l.at(-1)[1].final_board = "0x0000000000000000"; });
  assert.throws(() => load(list2), /final board/);
});

test("an unknown wire version is refused", () => {
  const list = tampered((l) => { l[0][1].v = 2; });
  assert.throws(() => load(list), /wire version/);
});
