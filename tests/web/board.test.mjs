import assert from "node:assert/strict";
import { test } from "node:test";

import {
  DOWN, LEFT, RIGHT, UP, decodeBoard, encodeBoard, initialBoard, isGameOver, maxTile, slide,
} from "../../web/js/board.js";
import { replays } from "./helpers.mjs";

function row(cells, r) {
  return cells.slice(4 * r, 4 * r + 4);
}

test("hex boards decode nibble by nibble: cell 0 is the last digit, cell 15 the first", () => {
  const cells = decodeBoard("0xf00000000000000b");
  assert.equal(cells[0], 11);
  assert.equal(cells[15], 15);
  assert.equal(cells.filter(Boolean).length, 2);
  // the high cells survive although the number is far beyond 2**53
  const hex = "0xfedcba9876543210";
  const viaBigInt = Array.from({ length: 16 }, (_, i) => Number((BigInt(hex) >> BigInt(4 * i)) & 15n));
  assert.deepEqual(decodeBoard(hex), viaBigInt);
  assert.equal(encodeBoard(decodeBoard(hex)), hex);
});

test("malformed boards are rejected", () => {
  for (const bad of ["0x123", "123456789abcdef01", "0xg000000000000000", 42, null]) {
    assert.throws(() => decodeBoard(bad));
  }
});

test("slide merges each tile once, toward the wall, like the engine's row tables", () => {
  const cells = [1, 1, 1, 1, 2, 1, 1, 0, 3, 0, 0, 3, 15, 15, 0, 0];
  const left = slide(cells, LEFT);
  assert.deepEqual(row(left.cells, 0), [2, 2, 0, 0]);
  assert.deepEqual(row(left.cells, 1), [2, 2, 0, 0]);
  assert.deepEqual(row(left.cells, 2), [4, 0, 0, 0]);
  assert.deepEqual(row(left.cells, 3), [15, 15, 0, 0]); // 32768 is the largest tile: no merge
  assert.equal(left.reward, 4 + 4 + 4 + 16);
  assert.equal(left.legal, true);
  const right = slide(cells, RIGHT);
  assert.deepEqual(row(right.cells, 0), [0, 0, 2, 2]);
  assert.deepEqual(row(right.cells, 1), [0, 0, 2, 2]);
});

test("columns: up and down", () => {
  const cells = [1, 0, 0, 0, 1, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0];
  assert.deepEqual(slide(cells, UP).cells.filter((_, i) => i % 4 === 0), [2, 2, 0, 0]);
  assert.deepEqual(slide(cells, DOWN).cells.filter((_, i) => i % 4 === 0), [0, 0, 2, 2]);
  assert.equal(slide(cells, LEFT).legal, false);
});

test("paths cover every tile once and conserve the tile sum", () => {
  const cells = [1, 1, 2, 2, 0, 3, 0, 3, 4, 0, 4, 4, 1, 2, 3, 4];
  for (const a of [UP, RIGHT, DOWN, LEFT]) {
    const { cells: out, paths } = slide(cells, a);
    assert.deepEqual(paths.map((p) => p.from).sort((x, y) => x - y), cells.flatMap((e, i) => (e ? [i] : [])));
    const sum = (cs) => cs.reduce((s, e) => s + (e ? 2 ** e : 0), 0);
    assert.equal(sum(out), sum(cells));
    for (const p of paths) assert.equal(out[p.to], p.merged ? p.exp + 1 : p.exp);
  }
});

for (const [name, doc] of replays()) {
  test(`replay ${name}: every check, the final board, score and max tile match`, () => {
    let cells = initialBoard(doc.init);
    let score = 0;
    const { every, boards, scores } = doc.checks;
    doc.steps.forEach(([action, reward, cell, exp], i) => {
      const moved = slide(cells, action);
      assert.ok(moved.legal, `move ${i + 1} must be legal`);
      assert.equal(moved.reward, reward, `reward of move ${i + 1}`);
      assert.equal(moved.cells[cell], 0, `spawn cell of move ${i + 1}`);
      moved.cells[cell] = exp;
      cells = moved.cells;
      score += reward;
      if ((i + 1) % every === 0) {
        assert.equal(encodeBoard(cells), boards[(i + 1) / every - 1], `board after move ${i + 1}`);
        assert.equal(score, scores[(i + 1) / every - 1], `score after move ${i + 1}`);
      }
    });
    assert.equal(encodeBoard(cells), doc.meta.final_board);
    assert.equal(score, doc.meta.score);
    assert.equal(maxTile(cells), doc.meta.max_tile);
    assert.ok(isGameOver(cells));
  });
}
