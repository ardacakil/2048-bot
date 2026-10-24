import assert from "node:assert/strict";
import { test } from "node:test";

import { Desync, GameBuffer } from "../../web/js/game.js";
import { StreamError, streamGame } from "../../web/js/stream.js";
import { chunks, encode, fakeFetch, fixture, rng, sseResponse } from "./helpers.mjs";

const json = (status, body, headers = {}) =>
  new Response(JSON.stringify(body), { status, headers: { "Content-Type": "application/json", ...headers } });
const html = (status, headers = {}) =>
  new Response("<html><body><h1>Error</h1></body></html>", { status, headers: { "Content-Type": "text/html", ...headers } });

async function rejection(promise) {
  try {
    await promise;
  } catch (e) {
    return e;
  }
  assert.fail("expected a rejection");
}

test("a whole game in random chunks: every event reaches onEvent, the end data is returned", async () => {
  const random = rng(3);
  const fetch = fakeFetch(() => sseResponse(chunks(fixture("reached-2048"), () => 1 + Math.floor(random() * 2000))));
  const g = new GameBuffer();
  const seen = [];
  const end = await streamGame("api/play/2048", {
    fetch,
    onEvent(e, d) {
      seen.push(e);
      if (e !== "comment") g.handle(e, d);
    },
  });
  assert.equal(fetch.calls[0].url, "api/play/2048"); // relative: resolved under /2048-bot/
  assert.equal(fetch.calls[0].init.cache, "no-store");
  assert.equal(end.reached, true);
  assert.equal(g.total, end.moves);
  assert.deepEqual(seen.slice(0, 5), ["meta", "queued", "comment", "queued", "init"]);
});

test("UTF-8 split inside a character survives (TextDecoder in stream mode)", async () => {
  const bytes = encode('event: error\ndata: {"code":"x","message":"Größe ünd ç"}\n\n');
  const at = bytes.indexOf(0xc3) + 1; // the middle of "ö"
  const fetch = fakeFetch(sseResponse([bytes.subarray(0, at), bytes.subarray(at)]));
  const err = await rejection(streamGame("u", { fetch }));
  assert.equal(err.kind, "server");
  assert.equal(err.message, "Größe ünd ç");
});

for (const [name, response, kind, retryAfter] of [
  ["503 busy", () => json(503, { detail: "busy", retry_after_s: 12 }, { "Retry-After": "12" }), "busy", 12],
  ["503 warming up", () => json(503, { detail: "warming up" }, { "Retry-After": "10" }), "warming", 10],
  ["429 from nginx (HTML body)", () => html(429), "rate_limited", 60],
  ["422 unknown target", () => json(422, { detail: "target must be one of ..." }), "invalid", null],
  ["502 service down", () => html(502), "unavailable", null],
  ["200 that is not an event stream", () => html(200), "protocol", null],
]) {
  test(`${name} -> ${kind}`, async () => {
    const err = await rejection(streamGame("u", { fetch: fakeFetch(response) }));
    assert.ok(err instanceof StreamError, String(err));
    assert.equal(err.kind, kind);
    assert.equal(err.retryAfter, retryAfter);
  });
}

test("a failed request -> network", async () => {
  const fetch = async () => { throw new TypeError("Failed to fetch"); };
  assert.equal((await rejection(streamGame("u", { fetch }))).kind, "network");
});

test("a drop mid-stream -> network, after the events that did arrive", async () => {
  const parts = chunks(fixture("reached-2048"), 5000).slice(0, 3);
  const fetch = fakeFetch(sseResponse(parts, { fail: new TypeError("network error") }));
  const g = new GameBuffer();
  const err = await rejection(streamGame("u", { fetch, onEvent: (e, d) => e !== "comment" && g.handle(e, d) }));
  assert.equal(err.kind, "network");
  assert.ok(g.started && g.length > 0 && !g.complete);
});

test("a stream that closes before end -> incomplete", async () => {
  const parts = chunks(fixture("reached-2048"), 5000).slice(0, 2);
  const err = await rejection(streamGame("u", { fetch: fakeFetch(sseResponse(parts)) }));
  assert.equal(err.kind, "incomplete");
});

test("abort rejects with the abort reason, not a StreamError", async () => {
  const controller = new AbortController();
  const parts = chunks(fixture("reached-2048"), 4000).slice(0, 2);
  const fetch = fakeFetch((url, init) => sseResponse(parts, { signal: init.signal, hold: true }));
  const seen = [];
  const done = streamGame("u", { fetch, signal: controller.signal, onEvent: (e) => seen.push(e) });
  setTimeout(() => controller.abort("cancel"), 20); // the stream is open and silent by then
  const err = await rejection(done);
  assert.equal(err, "cancel");
  assert.ok(seen.includes("meta"));
});

test("an onEvent error (a desync) propagates and cancels the body", async () => {
  const res = sseResponse(chunks(fixture("reached-2048"), 3000));
  const err = await rejection(streamGame("u", {
    fetch: fakeFetch(res),
    onEvent(e) {
      if (e === "init") throw new Desync(0, "test", 1, 2);
    },
  }));
  assert.ok(err instanceof Desync);
  assert.equal(res.state.cancelled, true);
});
