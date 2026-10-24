// Shared helpers for the frontend tests (node --test tests/web/*.test.mjs).

import { readFileSync, readdirSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const HERE = dirname(fileURLToPath(import.meta.url));
export const ROOT = join(HERE, "..", "..");

/** Bytes of tests/web/fixtures/<name>.sse (built by make_fixtures.py from server.wire). */
export function fixture(name) {
  return new Uint8Array(readFileSync(join(HERE, "fixtures", `${name}.sse`)));
}

export const FIXTURES = ["reached-2048", "missed-16384"];

export function replays() {
  const dir = join(ROOT, "results", "replays");
  return readdirSync(dir).filter((f) => f.endsWith(".json"))
    .map((f) => [f, JSON.parse(readFileSync(join(dir, f), "utf8"))]);
}

/** Deterministic PRNG (mulberry32) for reproducible random chunk boundaries. */
export function rng(seed) {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/** Split bytes into chunks of `size`, or of random sizes 1..max when size is a function. */
export function chunks(bytes, size) {
  const out = [];
  for (let i = 0; i < bytes.length;) {
    const n = typeof size === "function" ? size() : size;
    out.push(bytes.subarray(i, i + n));
    i += n;
  }
  return out;
}

/** A fetch stand-in: returns `response` (or calls it with (url, init)). */
export function fakeFetch(response) {
  const calls = [];
  const fn = async (url, init) => {
    calls.push({ url, init });
    return typeof response === "function" ? response(url, init) : response;
  };
  fn.calls = calls;
  return fn;
}

/**
 * Response whose body delivers `parts` (Uint8Array chunks), then closes, or errors with
 * `fail` if given. Aborting `signal` errors the body with the abort reason, as real fetch does.
 * `state.cancelled` turns true when the reader cancels the body.
 */
export function sseResponse(parts, { fail = null, signal = null, hold = false, status = 200,
                                      contentType = "text/event-stream; charset=utf-8" } = {}) {
  const state = { cancelled: false };
  let i = 0;
  const body = new ReadableStream({
    start(controller) {
      signal?.addEventListener("abort", () => controller.error(signal.reason));
    },
    pull(controller) {
      if (i < parts.length) controller.enqueue(parts[i++]);
      else if (fail) controller.error(fail);
      else if (!hold) controller.close();
      else return new Promise(() => {}); // stays open until aborted
    },
    cancel() {
      state.cancelled = true;
    },
  });
  const res = new Response(body, { status, headers: { "Content-Type": contentType } });
  res.state = state;
  return res;
}

export const encode = (text) => new TextEncoder().encode(text);
export const decode = (bytes) => new TextDecoder().decode(bytes);
