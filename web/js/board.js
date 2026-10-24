// 2048 board logic, a port of src/game/engine.py for the browser (pure, no DOM).
//
// A board is an array of 16 exponents, row-major: cell 0 = top-left, 3 = top-right,
// 12 = bottom-left. 0 = empty, otherwise the tile is 2 ** e. The wire format sends boards as
// "0x" + 16 hex digits of a 64-bit integer where cell i is hex digit 15 - i after the "0x".
// They are decoded digit by digit from the string: a Number keeps only 53 bits.

export const UP = 0, RIGHT = 1, DOWN = 2, LEFT = 3;
export const ACTION_NAMES = ["up", "right", "down", "left"];
export const MAX_EXPONENT = 15;

const HEX = /^0x[0-9a-f]{16}$/;

export function decodeBoard(hex) {
  if (typeof hex !== "string" || !HEX.test(hex)) throw new Error(`not a board: ${hex}`);
  const cells = new Array(16);
  for (let i = 0; i < 16; i++) cells[i] = parseInt(hex[17 - i], 16);
  return cells;
}

export function encodeBoard(cells) {
  let out = "0x";
  for (let i = 15; i >= 0; i--) out += cells[i].toString(16);
  return out;
}

export function emptyBoard() {
  return new Array(16).fill(0);
}

export function maxExponent(cells) {
  return Math.max(...cells);
}

export function maxTile(cells) {
  const e = maxExponent(cells);
  return e ? 2 ** e : 0;
}

// For each action, the four lines of cells in the order tiles slide: the first cell is the wall.
const LINES = [UP, RIGHT, DOWN, LEFT].map((action) => {
  const lines = [];
  for (let k = 0; k < 4; k++) {
    if (action === LEFT) lines.push([4 * k, 4 * k + 1, 4 * k + 2, 4 * k + 3]);
    if (action === RIGHT) lines.push([4 * k + 3, 4 * k + 2, 4 * k + 1, 4 * k]);
    if (action === UP) lines.push([k, 4 + k, 8 + k, 12 + k]);
    if (action === DOWN) lines.push([12 + k, 8 + k, 4 + k, k]);
  }
  return lines;
});

/**
 * Slide every line towards the wall, merging equal neighbours once each (as the engine's row
 * tables do). Returns {cells, reward, legal, paths}; paths lists every tile as
 * {from, to, exp, merged}, both halves of a merge included, for the slide animation.
 */
export function slide(cells, action) {
  const out = new Array(16).fill(0);
  const paths = [];
  let reward = 0;
  for (const line of LINES[action]) {
    const tiles = line.filter((i) => cells[i]);
    let k = 0;
    for (let j = 0; j < tiles.length; k++) {
      const from = tiles[j], e = cells[from], to = line[k];
      if (j + 1 < tiles.length && cells[tiles[j + 1]] === e && e < MAX_EXPONENT) {
        out[to] = e + 1;
        reward += 2 ** (e + 1);
        paths.push({ from, to, exp: e, merged: true }, { from: tiles[j + 1], to, exp: e, merged: true });
        j += 2;
      } else {
        out[to] = e;
        paths.push({ from, to, exp: e, merged: false });
        j += 1;
      }
    }
  }
  const legal = out.some((e, i) => e !== cells[i]);
  return { cells: out, reward, legal, paths };
}

export function isGameOver(cells) {
  return [UP, RIGHT, DOWN, LEFT].every((a) => !slide(cells, a).legal);
}

/** Board after the initial spawns, each [cell, exponent], placed on an empty board. */
export function initialBoard(spawns) {
  const cells = emptyBoard();
  for (const [cell, exp] of spawns) cells[cell] = exp;
  return cells;
}
