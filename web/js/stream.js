// One game stream: fetch + ReadableStream reader + SseParser (no DOM; fetch is injectable).
//
// EventSource is not used on purpose: it reconnects after a drop, and here every new
// connection starts a new game. Failures become a StreamError with a `kind` the UI can word:
//   busy          503 {"detail": "busy"}: the queue is full (retryAfter from the server)
//   warming       503 {"detail": "warming up"}: the model is still loading
//   rate_limited  429 from nginx (its body is HTML, not JSON)
//   invalid       422: unknown target
//   unavailable   any other HTTP error (502/504 when the service is down, ...)
//   protocol      200 but not an event stream, or a malformed event
//   network       the request failed or the connection dropped mid-stream
//   incomplete    the stream closed before its end event
//   server        an error event from the server
// An aborted request (signal) rejects with the AbortError itself, which callers ignore.

import { SseParser } from "./sse.js";

export class StreamError extends Error {
  constructor(kind, message, { status = null, retryAfter = null } = {}) {
    super(message);
    this.name = "StreamError";
    this.kind = kind;
    this.status = status;
    this.retryAfter = retryAfter;
  }
}

function retryAfterHeader(res) {
  const s = Number.parseInt(res.headers.get("Retry-After") ?? "", 10);
  return Number.isFinite(s) && s >= 0 ? s : null;
}

async function jsonBody(res) {
  try {
    return await res.json();
  } catch {
    return null;
  }
}

async function httpError(res) {
  const status = res.status;
  if (status === 503) {
    const body = await jsonBody(res);
    if (body?.detail === "busy") {
      const retryAfter = body.retry_after_s ?? retryAfterHeader(res);
      return new StreamError("busy", "queue full", { status, retryAfter });
    }
    if (body?.detail === "warming up") {
      return new StreamError("warming", "warming up", { status, retryAfter: retryAfterHeader(res) ?? 10 });
    }
  } else if (status === 429) {
    return new StreamError("rate_limited", "rate limited", { status, retryAfter: retryAfterHeader(res) ?? 60 });
  } else if (status === 422) {
    return new StreamError("invalid", "invalid target", { status });
  }
  return new StreamError("unavailable", `HTTP ${status}`, { status, retryAfter: retryAfterHeader(res) });
}

function isAbort(err, signal) {
  return signal?.aborted || err?.name === "AbortError";
}

/**
 * Stream one game from `url`, calling onEvent(event, data) for every event (comments
 * included). Resolves with the end event's data; rejects with a StreamError, or with the abort
 * reason / AbortError when `signal` aborts. Errors thrown by onEvent close the connection and
 * propagate unchanged.
 */
export async function streamGame(url, { signal, onEvent, fetch = globalThis.fetch } = {}) {
  let res;
  try {
    res = await fetch(url, { signal, cache: "no-store", headers: { Accept: "text/event-stream" } });
  } catch (err) {
    if (isAbort(err, signal)) throw err;
    throw new StreamError("network", "request failed");
  }
  if (!res.ok) throw await httpError(res);
  if (!(res.headers.get("Content-Type") ?? "").startsWith("text/event-stream") || !res.body) {
    throw new StreamError("protocol", "not an event stream", { status: res.status });
  }

  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  const parser = new SseParser();
  try {
    for (;;) {
      let chunk;
      try {
        chunk = await reader.read();
      } catch (err) {
        if (isAbort(err, signal)) throw err;
        throw new StreamError("network", "connection lost");
      }
      if (chunk.done) throw new StreamError("incomplete", "stream closed before the end of the game");
      let events;
      try {
        events = parser.feed(decoder.decode(chunk.value, { stream: true }));
      } catch {
        throw new StreamError("protocol", "malformed event");
      }
      for (const [event, data] of events) {
        if (event === "error") throw new StreamError("server", data?.message ?? "server error");
        onEvent?.(event, data);
        if (event === "end") return data;
      }
    }
  } finally {
    reader.cancel().catch(() => {});
  }
}
