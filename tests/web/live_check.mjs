// Play real games through the frontend's own stream and validation code, against a running
// server (not part of `node --test`: it needs the dev server or the live site).
//
//   node tests/web/live_check.mjs http://127.0.0.1:8200/2048-bot/ [2048 4096 ...]
//   node tests/web/live_check.mjs https://ardacakil.com/2048-bot/ 2048
//
// The API URL is resolved against the page URL exactly like the page's relative "api/play/N".
// Each game streams through stream.js into a GameBuffer, which re-checks every board.

import { maxTile } from "../../web/js/board.js";
import { GameBuffer } from "../../web/js/game.js";
import { streamGame } from "../../web/js/stream.js";

const [page = "http://127.0.0.1:8200/2048-bot/", ...rest] = process.argv.slice(2);
const targets = rest.length ? rest.map(Number) : [2048, 4096, 8192, 16384];
let failed = false;

for (const target of targets) {
  const url = new URL(`api/play/${target}`, page).href;
  const game = new GameBuffer();
  const t0 = performance.now();
  let firstStep = null;
  try {
    const end = await streamGame(url, {
      onEvent(event, data) {
        if (event === "steps" && firstStep === null) firstStep = performance.now() - t0;
        if (event !== "comment") game.handle(event, data);
      },
    });
    const ms = performance.now() - t0;
    const playback = game.meta.delay_ms * end.moves / 1000;
    console.log(`${target}: ${end.status}, ${end.moves} moves, score ${end.score}, max tile ` +
                `${maxTile(game.board(end.moves))}, all boards checked; stream ${(ms / 1000).toFixed(2)} s ` +
                `(first steps after ${(firstStep / 1000).toFixed(2)} s, server compute ${end.compute_s} s), ` +
                `playback at ${game.meta.delay_ms} ms/move ≈ ${playback.toFixed(1)} s; seed ${game.meta.seed}`);
  } catch (err) {
    failed = true;
    console.log(`${target}: FAILED ${err.name} ${err.kind ?? ""} ${err.message}`);
  }
}
process.exit(failed ? 1 : 0);
