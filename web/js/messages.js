// Status-line wording for each state of a target (pure, no DOM).

const fmt = new Intl.NumberFormat("en-US");
const n = (v) => fmt.format(v);

export function policyLabel(meta) {
  if (!meta) return "the bot";
  return meta.policy === "greedy" ? "greedy policy" : `expectimax depth ${meta.depth}`;
}

function reachRateLabel(rate) {
  if (!rate) return "";
  const policy = rate.policy.replace(/^expectimax-d(\d)$/, "expectimax depth $1");
  return ` With ${policy} this bot reaches it in ${rate.pct}% of games (${n(rate.games)} evaluation games).`;
}

function resultText(target, buffer) {
  const r = buffer.result;
  if (r.reached) return [`Reached ${n(target)} in ${n(r.moves)} moves, score ${n(r.score)}.`, "good"];
  const why = r.status === "max_moves" ? "the game hit the move limit" : "no move was left";
  return [`Didn't reach ${n(target)}: ${why} after ${n(r.moves)} moves, with ${n(r.max_tile)} as the ` +
          `largest tile.${reachRateLabel(buffer.meta?.reach_rate)}`, "bad"];
}

function queueText(q) {
  if (!q) return "Waiting for the server…";
  const eta = q.eta_s != null ? `, about ${Math.max(1, Math.ceil(q.eta_s))} s` : "";
  return `In the queue: position ${q.position} of ${q.queue_length}${eta}.`;
}

export function errorText(err, buffer, now = Date.now(), retryUntil = 0) {
  const wait = retryUntil > now ? Math.ceil((retryUntil - now) / 1000) : 0;
  const later = wait ? ` Try again in ${wait} s.` : " Try again.";
  const at = buffer?.started ? ` at move ${n(buffer.length)}` : "";
  switch (err?.kind ?? err?.name) {
    case "busy": return `The server is busy: the queue is full.${later}`;
    case "warming": return `The bot is still starting up.${later}`;
    case "rate_limited": return `Too many games started from your network in the last minute.${later}`;
    case "invalid": return "The server doesn't offer this target.";
    case "unavailable": return `The server isn't available right now (HTTP ${err.status}).${later}`;
    case "protocol": return "Unexpected response from the server.";
    case "network": return buffer?.started
      ? `Connection lost${at}. Start again for a new game.`
      : "Couldn't reach the server. Check your connection and try again.";
    case "incomplete": return `The stream ended early${at}. Start again for a new game.`;
    case "server": return "The server couldn't finish this game. Try again.";
    case "Desync": return `The streamed game broke the rules at move ${n(err.move)}, so playback stopped.`;
    case "left": return "The game stopped when you left the page. Start again for a new game.";
    default: return "Something went wrong. Try again.";
  }
}

/** [text, tone] for the status line. */
export function statusText(target, game, now = Date.now()) {
  const { state, buffer } = game;
  switch (state) {
    case "idle": return [`Press Start: the server plays a new game until a ${n(target)} tile appears, ` +
                         "and your browser replays it.", ""];
    case "connecting": return ["Connecting…", ""];
    case "queued": return [queueText(buffer?.queue), ""];
    case "streaming":
    case "playing": return [`Playing: ${policyLabel(buffer.meta)}, ${buffer.meta.delay_ms} ms per move.`, ""];
    case "done":
    case "missed": return resultText(target, buffer);
    case "error": return [errorText(game.error, buffer, now, game.retryUntil), "bad"];
    default: return ["", ""];
  }
}
