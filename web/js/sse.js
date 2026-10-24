// Incremental Server-Sent Events parser, a port of server.wire.Parser (pure, no DOM).
//
// feed() takes text chunks split anywhere and returns the complete events so far as
// [event, data] pairs with data parsed as JSON. Comment lines (": ka" keep-alives) come back as
// ["comment", text] so callers and tests can see them.

export class SseParser {
  constructor() {
    this.buffer = "";
    this.event = null;
    this.data = [];
  }

  feed(text) {
    this.buffer += text;
    const out = [];
    let nl;
    while ((nl = this.buffer.indexOf("\n")) >= 0) {
      let line = this.buffer.slice(0, nl);
      this.buffer = this.buffer.slice(nl + 1);
      if (line.endsWith("\r")) line = line.slice(0, -1);
      if (!line) {
        if (this.data.length) out.push([this.event || "message", JSON.parse(this.data.join("\n"))]);
        this.event = null;
        this.data = [];
      } else if (line.startsWith(":")) {
        out.push(["comment", line.slice(1).trim()]);
      } else {
        const colon = line.indexOf(":");
        const field = colon < 0 ? line : line.slice(0, colon);
        let value = colon < 0 ? "" : line.slice(colon + 1);
        if (value.startsWith(" ")) value = value.slice(1);
        if (field === "event") this.event = value;
        else if (field === "data") this.data.push(value);
      }
    }
    return out;
  }
}
