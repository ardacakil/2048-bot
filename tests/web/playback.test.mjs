import assert from "node:assert/strict";
import { test } from "node:test";

import { FINAL_MS, MAX_FRAME_MS, Player, RAMP_MAX_MS, RAMP_MOVES, START_HOLD_MS } from "../../web/js/playback.js";
import { rng } from "./helpers.mjs";

const FRAME = 1000 / 60;

function player(delayMs, { available = 1e9, total = null, slowdown = true } = {}) {
  const box = { available, total };
  const p = new Player({ delayMs, available: () => box.available, total: () => box.total, slowdown });
  p.box = box;
  return p;
}

/** Time needed before board `pos` may show: the sum of showFor(0 .. pos-1). */
function due(p, pos) {
  let t = 0;
  for (let k = 0; k < pos; k++) t += p.showFor(k);
  return t;
}

test("steady pace with everything buffered: never ahead of the schedule, at most a frame behind", () => {
  for (const delay of [20, 10, 5]) {
    const p = player(delay, { available: 3000, total: 3000 });
    let t = 0;
    while (!p.finished) {
      t += FRAME;
      const { pos } = p.tick(FRAME);
      assert.ok(due(p, pos) <= t + 1e-6, `ahead at t=${t}`);
      if (pos < 3000) assert.ok(due(p, pos + 1) > t - FRAME - 1e-6, `behind at t=${t}`);
    }
    assert.ok(Math.abs(t - due(p, 3000)) <= FRAME + 1e-6);
  }
});

test("bursty batches: a late batch never plays back as a burst", () => {
  const random = rng(11);
  const p = player(20);
  p.box.available = 0;
  let t = 0, nextBatch = 0, maxAdvance = 0;
  while (p.pos < 2000) {
    t += FRAME;
    if (t >= nextBatch) { // a 256-move batch, then a random gap of 0-3 s
      p.box.available = Math.min(2000, p.box.available + 256);
      nextBatch = t + random() * 3000;
    }
    const { advanced } = p.tick(FRAME);
    maxAdvance = Math.max(maxAdvance, advanced);
  }
  assert.ok(maxAdvance <= Math.ceil(FRAME / 20) + 1, `advanced ${maxAdvance} moves in one frame`);
});

test("an underrun holds the clock: no catch-up when the data arrives", () => {
  const p = player(10, { available: 5 });
  for (let i = 0; i < 300; i++) p.tick(FRAME); // 5 s, only 5 moves buffered
  assert.equal(p.pos, 5);
  p.box.available = 1000;
  const { advanced } = p.tick(FRAME);
  assert.ok(advanced <= 2, `advanced ${advanced}`);
});

test("a long frame (hidden tab) is capped", () => {
  const p = player(5, { available: 1000 });
  p.tick(START_HOLD_MS);
  const { advanced } = p.tick(60_000);
  assert.ok(advanced <= MAX_FRAME_MS / 5 + 1);
});

test("the initial board holds, then the base delay applies", () => {
  const p = player(20, { available: 100 });
  for (let t = 0; t < START_HOLD_MS - 50; t += 50) p.tick(50);
  assert.equal(p.tick(49).pos, 0);
  assert.equal(p.tick(1).pos, 1);
  assert.equal(p.tick(19).pos, 1);
  assert.equal(p.tick(1).pos, 2);
});

test("slowdown: the last boards stay longer, up to a slow final move", () => {
  const n = 500;
  const p = player(5, { total: n });
  assert.equal(p.showFor(n - RAMP_MOVES - 1), 5);
  const ramp = Array.from({ length: RAMP_MOVES }, (_, i) => p.showFor(n - RAMP_MOVES + i));
  ramp.forEach((d, i) => i && assert.ok(d > ramp[i - 1]));
  assert.ok(Math.abs(p.showFor(n - 1) - RAMP_MAX_MS) < 1e-9);
  assert.equal(p.showFor(n), FINAL_MS);
  // without slowdown (reduced motion) the pace stays flat
  const flat = player(5, { total: n, slowdown: false });
  assert.equal(flat.showFor(n - 1), 5);
});

test("unknown length: no slowdown yet", () => {
  const p = player(20);
  assert.equal(p.showFor(5000), 20);
});

test("skip jumps to the buffered moves and follows the stream to its end", () => {
  const p = player(20, { available: 300 });
  p.skip();
  assert.deepEqual(p.tick(FRAME).pos, 300);
  p.box.available = 400;
  p.box.total = 400;
  assert.equal(p.tick(FRAME).pos, 400);
  assert.ok(p.finished);
  p.restart();
  assert.equal(p.pos, 0);
  assert.equal(p.tick(FRAME).pos, 0);
});

test("ESTIMATED playback times at the server's delays (MEASURED move count for 2048: 1,189)", () => {
  const p = player(20, { total: 1189 });
  const seconds = due(p, 1189) / 1000;
  assert.ok(seconds > 23.5 && seconds < 27, `${seconds} s`);
});
