// Playback clock: which move to show, at a fixed pace independent of how batches arrive
// (pure, no DOM; the caller passes frame times).
//
// Board k (the board after move k; 0 = the initial board) stays on screen for showFor(k) ms:
// the base delay (meta.delay_ms), longer for the initial board, and stretched over the last
// RAMP_MOVES moves once the game's length is known, so the final merge is visible. tick(dt)
// adds the frame time, capped at MAX_FRAME_MS so a hidden tab or a long frame never causes a
// jump, and advances over every buffered move whose time has come. When the next move is not
// buffered yet the clock holds instead of banking time, so a late batch never plays back as a
// burst.

export const START_HOLD_MS = 400; // the initial board
export const MAX_FRAME_MS = 100;
export const RAMP_MOVES = 12; // boards before the last that stay longer and longer
export const RAMP_MAX_MS = 180; // the board before the last
export const FINAL_MS = 450; // the last move's animation

export class Player {
  /**
   * `available()` = moves buffered so far, `total()` = total moves or null while unknown.
   * `slowdown: false` keeps the base delay up to the end.
   */
  constructor({ delayMs, available, total, slowdown = true }) {
    this.delay = Math.max(1, delayMs);
    this.available = available;
    this.total = total;
    this.slowdown = slowdown;
    this.restart();
  }

  restart() {
    this.pos = 0; // board shown = moves played back
    this.elapsed = 0; // time board `pos` has been shown
    this.skipping = false;
  }

  /** Jump to the last buffered move, and keep following the stream until its end. */
  skip() {
    this.skipping = true;
  }

  get finished() {
    const n = this.total();
    return n !== null && this.pos >= n;
  }

  showFor(k) {
    if (k === 0) return Math.max(this.delay, START_HOLD_MS);
    const n = this.total();
    if (!this.slowdown || n === null || k < n - RAMP_MOVES) return this.delay;
    if (k >= n) return Math.max(this.delay, FINAL_MS);
    const r = (k - (n - RAMP_MOVES) + 1) / RAMP_MOVES; // 1/12 .. 1 for boards n-12 .. n-1
    return Math.max(this.delay, this.delay * (RAMP_MAX_MS / this.delay) ** r);
  }

  /**
   * Advance by `dtMs`. Returns {pos, advanced, duration}: the board now shown, how many moves
   * this frame passed, and how long the new board will stay (its animation time).
   */
  tick(dtMs) {
    const start = this.pos;
    const available = this.available();
    if (this.skipping) {
      this.pos = Math.max(this.pos, available);
      this.elapsed = 0;
    } else {
      this.elapsed += Math.min(Math.max(dtMs, 0), MAX_FRAME_MS);
      while (this.pos < available) {
        const d = this.showFor(this.pos);
        if (this.elapsed < d) break;
        this.elapsed -= d;
        this.pos += 1;
      }
      if (this.pos >= available) this.elapsed = Math.min(this.elapsed, this.showFor(this.pos));
    }
    const advanced = this.pos - start;
    return { pos: this.pos, advanced, duration: this.showFor(this.pos) };
  }
}
