import { describe, expect, it } from "vitest";
import { parseEvent, sleep } from "./api";

describe("parseEvent (agent SSE discriminated union)", () => {
  it("parses a tool_call event", () => {
    const e = parseEvent(JSON.stringify({ type: "tool_call", data: { id: "1", name: "bash", args: {} } }));
    expect(e?.type).toBe("tool_call");
  });
  it("parses an assistant event", () => {
    const e = parseEvent(JSON.stringify({ type: "assistant", data: { content: "hi" } }));
    expect(e && e.type === "assistant" && e.data.content).toBe("hi");
  });
  it("rejects an unknown type", () => {
    expect(parseEvent(JSON.stringify({ type: "nope", data: {} }))).toBeNull();
  });
  it("rejects malformed JSON", () => {
    expect(parseEvent("{not json")).toBeNull();
  });
  it("rejects when data is missing", () => {
    expect(parseEvent(JSON.stringify({ type: "done" }))).toBeNull();
  });
});

describe("sleep (abort-aware)", () => {
  it("resolves immediately if the signal is already aborted", async () => {
    const ac = new AbortController();
    ac.abort();
    const t0 = Date.now();
    await sleep(5000, ac.signal);
    expect(Date.now() - t0).toBeLessThan(200);
  });
  it("resolves when aborted mid-wait", async () => {
    const ac = new AbortController();
    const p = sleep(5000, ac.signal);
    ac.abort();
    const t0 = Date.now();
    await p;
    expect(Date.now() - t0).toBeLessThan(200);
  });
  it("resolves normally without a signal", async () => {
    await sleep(1);
    expect(true).toBe(true);
  });
});
