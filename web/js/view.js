// DOM side: the board's tiles and animations, the target buttons, stats, status and the main
// button. Holds no game state; main.js tells it what to show. Styles go through classes, data
// attributes and CSS custom properties (CSSOM), never inline style attributes, so the page
// works under the strict Content-Security-Policy.

import { maxTile, slide } from "./board.js";

const fmt = new Intl.NumberFormat("en-US");
export const formatNumber = (n) => fmt.format(n);

const SLIDE_MAX_MS = 110;
const FINAL_SLIDE_MS = 160;
const POP_MIN_MS = 60; // shorter moves skip the merge/spawn pops: they would not be visible

export class BoardView {
  constructor(root) {
    this.layer = root.querySelector(".tiles");
    this.tiles = [];
    for (let i = 0; i < 16; i++) {
      const tile = document.createElement("div");
      tile.className = "tile";
      const inner = document.createElement("span");
      inner.className = "tile-inner";
      tile.append(inner);
      this.layer.append(tile);
      this.tiles.push(tile);
    }
    this.pending = null; // {timer, next}: an animation whose final board is not drawn yet
    this.render(new Array(16).fill(0));
  }

  /** Draw `cells` at once (no transition). */
  render(cells) {
    if (this.pending) {
      clearTimeout(this.pending.timer);
      this.pending = null;
    }
    this.tiles.forEach((tile, i) => {
      tile.classList.remove("moving", "goal");
      place(tile, i);
      setExponent(tile, cells[i]);
    });
  }

  /**
   * Slide from `prev` to `next` (move `action`, then the spawn on `spawnCell`) within `ms`;
   * `final` marks the move that made the target tile.
   */
  animate(prev, action, next, spawnCell, ms, { final = false } = {}) {
    if (this.pending) {
      this.render(prev);
      void this.layer.offsetWidth; // flush styles so the next transition starts from `prev`
    }
    const { paths } = slide(prev, action);
    const slideMs = final ? FINAL_SLIDE_MS : Math.min(0.75 * ms, SLIDE_MAX_MS);
    this.layer.style.setProperty("--slide-ms", `${Math.round(slideMs)}ms`);
    for (const p of paths) {
      if (p.from === p.to) continue;
      const tile = this.tiles[p.from];
      tile.classList.add("moving");
      place(tile, p.to);
    }
    const merged = new Set(paths.filter((p) => p.merged).map((p) => p.to));
    const timer = setTimeout(() => {
      this.render(next);
      const popMs = final ? 360 : Math.min(ms - slideMs, 140);
      if (popMs < POP_MIN_MS && !final) return;
      for (const cell of merged) {
        const isGoal = final && 2 ** next[cell] === maxTile(next);
        if (isGoal) this.tiles[cell].classList.add("goal");
        pop(this.tiles[cell], isGoal ? 1.25 : 1.12, popMs);
      }
      appear(this.tiles[spawnCell], Math.max(popMs, POP_MIN_MS));
    }, slideMs);
    this.pending = { timer };
  }
}

function place(tile, cell) {
  tile.style.setProperty("--row", String(cell >> 2));
  tile.style.setProperty("--col", String(cell & 3));
}

function setExponent(tile, e) {
  const key = String(e);
  if (tile.dataset.exp === key) return;
  tile.dataset.exp = key;
  const value = e ? String(2 ** e) : "";
  tile.dataset.digits = String(value.length);
  tile.firstChild.textContent = value;
}

function pop(tile, scale, ms) {
  tile.firstChild.animate?.([{ scale: "1" }, { scale: String(scale) }, { scale: "1" }],
    { duration: ms, easing: "ease-out" });
}

function appear(tile, ms) {
  tile.firstChild.animate?.([{ scale: "0.3", opacity: 0 }, { scale: "1", opacity: 1 }],
    { duration: ms, easing: "ease-out" });
}

/** Everything around the board. */
export class PanelView {
  constructor(doc, onSelect, onPrimary) {
    this.buttons = new Map();
    for (const button of doc.querySelectorAll("[data-target]")) {
      const target = Number(button.dataset.target);
      this.buttons.set(target, button);
      button.addEventListener("click", () => onSelect(target));
    }
    this.primary = doc.getElementById("primary");
    this.primary.addEventListener("click", () => onPrimary());
    this.score = doc.getElementById("score");
    this.moves = doc.getElementById("moves");
    this.max = doc.getElementById("max-tile");
    this.status = doc.getElementById("status");
    this.progress = doc.getElementById("progress");
    this.fill = this.progress.querySelector(".progress-fill");
    this.last = {};
  }

  /** states: Map target -> {state, used}; `passed` = largest tile on the board shown. */
  setTargets(states, selected, locked, passed) {
    for (const [target, button] of this.buttons) {
      const { state, used } = states.get(target);
      button.setAttribute("aria-pressed", String(target === selected));
      button.disabled = locked && target !== selected;
      button.dataset.state = state;
      button.classList.toggle("passed", passed >= target);
      button.querySelector(".target-note").textContent = used ? "watch again" : "";
    }
  }

  setStats(score, moves, max) {
    this.text(this.score, "score", formatNumber(score));
    this.text(this.moves, "moves", formatNumber(moves));
    this.text(this.max, "max", max ? formatNumber(max) : "–");
  }

  /** fraction 0..1, or null to hide the bar. */
  setProgress(fraction) {
    const value = fraction === null ? null : Math.max(0, Math.min(1, fraction));
    if (this.last.progress === value) return;
    this.last.progress = value;
    this.progress.classList.toggle("idle", value === null);
    this.fill.style.setProperty("--progress", String(value ?? 0));
    this.progress.setAttribute("aria-valuenow", String(Math.round((value ?? 0) * 100)));
  }

  setStatus(text, tone = "") {
    this.text(this.status, "status", text);
    this.status.dataset.tone = tone;
  }

  setPrimary(label, disabled = false) {
    this.text(this.primary, "primary", label);
    this.primary.disabled = disabled;
  }

  text(el, key, value) {
    if (this.last[key] === value) return;
    this.last[key] = value;
    el.textContent = value;
  }
}
