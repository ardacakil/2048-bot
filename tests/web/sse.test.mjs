import assert from "node:assert/strict";
import { test } from "node:test";

import { SseParser } from "../../web/js/sse.js";
import { FIXTURES, chunks, decode, fixture, rng } from "./helpers.mjs";

function parseAll(texts) {
  const p = new SseParser();
  return texts.flatMap((t) => p.feed(t));
}

test("same cases as the Python wire.Parser test: split chunks, comments, multi-line data", () => {
  const stream = 'event: meta\ndata: {"a":1}\n\n: ka\n\nevent: x\ndata: [1,\ndata: 2]\n\nevent: end\ndata: {"ok":true}\n\n';
  for (const size of [1, 2, 3, 7, 64, stream.length]) {
    const parts = [];
    for (let i = 0; i < stream.length; i += size) parts.push(stream.slice(i, i + size));
    assert.deepEqual(parseAll(parts), [["meta", { a: 1 }], ["comment", "ka"], ["x", [1, 2]], ["end", { ok: true }]]);
  }
});

test("CRLF line endings, a CR split from its LF, no space after the colon, unnamed events", () => {
  const out = parseAll(["event:meta\r", "\ndata:{\"v\":1}\r\n\r\n", "data: 5\n\n"]);
  assert.deepEqual(out, [["meta", { v: 1 }], ["message", 5]]);
});

test("an event without data is dropped; unknown fields are ignored", () => {
  assert.deepEqual(parseAll(["event: lonely\n\nid: 3\nretry: 10\nevent: a\ndata: 1\n\n"]), [["a", 1]]);
});

test("a partial event waits for its blank line", () => {
  const p = new SseParser();
  assert.deepEqual(p.feed("event: end\ndata: {}"), []);
  assert.deepEqual(p.feed("\n"), []);
  assert.deepEqual(p.feed("\n"), [["end", {}]]);
});

for (const name of FIXTURES) {
  test(`${name}: any chunking gives the same events`, () => {
    const bytes = fixture(name);
    const whole = parseAll([decode(bytes)]);
    const kinds = whole.map(([e]) => e);
    assert.equal(kinds[0], "meta");
    assert.equal(kinds.at(-1), "end");
    assert.ok(kinds.includes("init") && kinds.includes("steps"));
    const random = rng(7);
    for (const size of [1, 5, 4096, () => 1 + Math.floor(random() * 300)]) {
      const p = new SseParser();
      const decoder = new TextDecoder();
      const out = chunks(bytes, size).flatMap((c) => p.feed(decoder.decode(c, { stream: true })));
      assert.deepEqual(out, whole);
    }
  });
}

test("the queued fixture carries a keep-alive comment between queued events", () => {
  const kinds = parseAll([decode(fixture("reached-2048"))]).map(([e]) => e).slice(0, 5);
  assert.deepEqual(kinds, ["meta", "queued", "comment", "queued", "init"]);
});
