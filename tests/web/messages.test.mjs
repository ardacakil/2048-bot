import assert from "node:assert/strict";
import { test } from "node:test";

import { GameBuffer } from "../../web/js/game.js";
import { errorText, statusText } from "../../web/js/messages.js";
import { SseParser } from "../../web/js/sse.js";
import { StreamError } from "../../web/js/stream.js";
import { decode, fixture } from "./helpers.mjs";

function buffer(name) {
  const g = new GameBuffer();
  for (const [e, d] of new SseParser().feed(decode(fixture(name)))) if (e !== "comment") g.handle(e, d);
  return g;
}

test("reached: says so, with moves and score", () => {
  const g = buffer("reached-2048");
  const [text, tone] = statusText(2048, { state: "done", buffer: g });
  assert.equal(tone, "good");
  assert.match(text, new RegExp(`^Reached 2,048 in ${g.total.toLocaleString("en-US")} moves`));
});

test("missed: honest, with the largest tile and the reach rate from meta", () => {
  const g = buffer("missed-16384");
  const [text, tone] = statusText(16384, { state: "missed", buffer: g });
  assert.equal(tone, "bad");
  assert.match(text, /^Didn't reach 16,384: no move was left after 7,426 moves, with 8,192 as the largest tile\./);
  assert.match(text, /greedy this bot reaches it in 48\.3% of games \(10,000 evaluation games\)\./);
});

test("missed without a reach rate in meta: no made-up number", () => {
  const g = buffer("missed-16384");
  g.meta = { ...g.meta, reach_rate: null };
  const [text] = statusText(16384, { state: "missed", buffer: g });
  assert.doesNotMatch(text, /%/);
});

test("queued: position, length and estimate", () => {
  const g = new GameBuffer();
  g.queue = { position: 2, queue_length: 3, eta_s: 4.2 };
  assert.equal(statusText(2048, { state: "queued", buffer: g })[0], "In the queue: position 2 of 3, about 5 s.");
});

test("errors: every kind has its own wording, with the retry countdown", () => {
  const now = 1_000_000;
  const busy = errorText(new StreamError("busy", "", { retryAfter: 12 }), null, now, now + 12_000);
  assert.equal(busy, "The server is busy: the queue is full. Try again in 12 s.");
  assert.match(errorText(new StreamError("rate_limited", ""), null), /Too many games/);
  assert.match(errorText(new StreamError("unavailable", "", { status: 502 }), null), /HTTP 502/);
  const started = buffer("reached-2048");
  started.result = null;
  assert.match(errorText(new StreamError("network", ""), started), /^Connection lost at move [\d,]+\./);
  assert.match(errorText(new StreamError("network", ""), new GameBuffer()), /^Couldn't reach the server/);
  assert.match(errorText({ kind: "left" }, null), /left the page/);
  const kinds = ["busy", "warming", "rate_limited", "invalid", "unavailable", "protocol", "network", "incomplete", "server"];
  assert.equal(new Set(kinds.map((k) => errorText(new StreamError(k, ""), null))).size, kinds.length);
});
