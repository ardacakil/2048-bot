// Page controller: one state machine per target, the stream, and the playback loop.
//
//   idle -> connecting -> queued -> streaming -> playing -> done | missed
//   any failure before the end event -> error (Start again by hand; never retried in a loop)
//
// Product rule: a target is used once its end event arrived; from then on its button offers
// "watch again", which replays the buffered game without a server call. At most one stream is
// open; while it is connecting, queued or streaming the other targets are locked. Leaving the
// page aborts the stream.

import { emptyBoard, maxTile } from "./board.js";
import { GameBuffer } from "./game.js";
import { statusText } from "./messages.js";
import { Player } from "./playback.js";
import { streamGame } from "./stream.js";
import { BoardView, PanelView } from "./view.js";

const TARGETS = [2048, 4096, 8192, 16384];
const ANIMATE_MIN_MS = 16; // shorter moves are redrawn, not animated
const OPEN = new Set(["connecting", "queued", "streaming"]);

const reducedMotion = matchMedia("(prefers-reduced-motion: reduce)");
const games = new Map(TARGETS.map((t) => [t, { state: "idle", buffer: null, error: null, retryUntil: 0 }]));
let selected = TARGETS[0];
let stream = null; // {target, controller} while a stream is open
let playback = null; // {target, player, buffer, last, raf}
let countdown = 0;

const boardView = new BoardView(document.getElementById("board"));
const panel = new PanelView(document, select, primaryAction);

function used(game) {
  return game.buffer?.complete ?? false;
}

function setState(target, state) {
  games.get(target).state = state;
  update();
}

// ---------------------------------------------------------------- what is on screen

function update() {
  const game = games.get(selected);
  const states = new Map([...games].map(([t, g]) => [t, { state: g.state, used: used(g) }]));
  const shown = shownBoard();
  panel.setTargets(states, selected, stream !== null, maxTile(shown.cells));
  const [text, tone] = statusText(selected, game);
  panel.setStatus(text, tone);
  panel.setStats(shown.score, shown.moves, maxTile(shown.cells));
  const total = game.buffer?.total;
  panel.setProgress(total ? shown.moves / total : game.state === "streaming" ? 0 : null);
  setPrimary(game);
}

/** The board of the selected target as currently shown. */
function shownBoard() {
  const game = games.get(selected);
  const b = game.buffer;
  let k = null;
  if (playback?.target === selected) k = playback.player.pos;
  else if (b?.complete) k = b.total;
  if (k === null || !b?.started) return { cells: emptyBoard(), score: 0, moves: 0 };
  return { cells: b.board(k), score: b.score(k), moves: k };
}

function setPrimary(game) {
  switch (game.state) {
    case "connecting":
    case "queued": return panel.setPrimary("Cancel");
    case "streaming":
    case "playing": return panel.setPrimary("Skip to end");
    case "done":
    case "missed": return panel.setPrimary("Watch again");
    default: {
      const wait = Math.ceil((game.retryUntil - Date.now()) / 1000);
      return wait > 0 ? panel.setPrimary(`Start (${wait} s)`, true) : panel.setPrimary("Start");
    }
  }
}

function startCountdown() {
  if (countdown) return;
  countdown = setInterval(() => {
    if (![...games.values()].some((g) => g.retryUntil > Date.now())) {
      clearInterval(countdown);
      countdown = 0;
    }
    update();
  }, 1000);
}

// ---------------------------------------------------------------- user actions

function select(target) {
  if (stream && target !== stream.target) return; // locked while a stream is open
  if (playback && playback.target !== target) stopPlayback();
  selected = target;
  const game = games.get(target);
  const b = game.buffer;
  boardView.render(b?.complete ? b.board(b.total) : emptyBoard());
  update();
}

function primaryAction() {
  const game = games.get(selected);
  switch (game.state) {
    case "idle":
    case "error":
      if (game.retryUntil <= Date.now()) start(selected);
      break;
    case "connecting":
    case "queued":
      stream?.controller.abort("cancel");
      break;
    case "streaming":
    case "playing":
      playback?.player.skip();
      break;
    case "done":
    case "missed":
      beginPlayback(selected);
      setState(selected, "playing");
      break;
  }
}

// ---------------------------------------------------------------- the stream

async function start(target) {
  stream?.controller.abort("replaced");
  stopPlayback();
  const game = games.get(target);
  const buffer = new GameBuffer();
  Object.assign(game, { buffer, error: null, retryUntil: 0 });
  const controller = new AbortController();
  stream = { target, controller };
  boardView.render(emptyBoard());
  setState(target, "connecting");
  try {
    await streamGame(`api/play/${target}`, {
      signal: controller.signal,
      onEvent(event, data) {
        buffer.handle(event, data);
        if (event === "queued") setState(target, "queued");
        else if (event === "init") {
          setState(target, "streaming");
          beginPlayback(target);
        } else if (event === "end") {
          stream = null;
          setState(target, playback?.target === target ? "playing" : buffer.result.reached ? "done" : "missed");
        }
      },
    });
  } catch (err) {
    if (stream?.controller === controller) stream = null;
    if (playback?.target === target) stopPlayback();
    if (controller.signal.aborted) {
      if (controller.signal.reason === "pagehide") fail(target, { kind: "left" });
      else {
        game.buffer = null; // cancelled by the user: the target stays unused
        setState(target, "idle");
      }
      return;
    }
    fail(target, err);
  }
}

function fail(target, error) {
  const game = games.get(target);
  game.error = error;
  game.buffer = game.buffer?.complete ? game.buffer : null;
  if (error.retryAfter) {
    game.retryUntil = Date.now() + error.retryAfter * 1000;
    startCountdown();
  }
  if (target === selected) boardView.render(emptyBoard());
  setState(target, "error");
}

// ---------------------------------------------------------------- playback

function beginPlayback(target) {
  stopPlayback();
  const buffer = games.get(target).buffer;
  const player = new Player({
    delayMs: buffer.meta.delay_ms,
    available: () => buffer.length,
    total: () => buffer.total,
    slowdown: !reducedMotion.matches,
  });
  playback = { target, player, buffer, last: null, raf: 0 };
  boardView.render(buffer.board(0));
  schedule();
}

function schedule() {
  if (playback && !playback.raf && !document.hidden) playback.raf = requestAnimationFrame(frame);
}

function frame(now) {
  const p = playback;
  if (!p) return;
  p.raf = 0;
  const dt = p.last === null ? 0 : now - p.last;
  p.last = now;
  const prev = p.player.pos;
  const { pos, advanced, duration } = p.player.tick(dt);
  if (advanced) {
    const b = p.buffer;
    const step = b.step(pos);
    if (advanced === 1 && !reducedMotion.matches && duration >= ANIMATE_MIN_MS) {
      const final = pos === b.total && b.result.reached;
      boardView.animate(b.board(prev), step.action, b.board(pos), step.spawnCell, duration, { final });
    } else {
      boardView.render(b.board(pos));
    }
    update();
  }
  if (p.player.finished) finishPlayback();
  else schedule();
}

/** Stop the loop; a complete game is then shown as finished. */
function stopPlayback() {
  const p = playback;
  if (!p) return;
  if (p.raf) cancelAnimationFrame(p.raf);
  playback = null;
  const game = games.get(p.target);
  if (p.buffer.complete && game.state === "playing") game.state = p.buffer.result.reached ? "done" : "missed";
}

function finishPlayback() {
  stopPlayback();
  update();
}

document.addEventListener("visibilitychange", () => {
  if (!playback) return;
  if (document.hidden) {
    if (playback.raf) cancelAnimationFrame(playback.raf);
    playback.raf = 0;
  }
  playback.last = null;
  schedule();
});

addEventListener("pagehide", () => stream?.controller.abort("pagehide"));

update();
