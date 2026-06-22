import { describe, expect, it } from "vitest";
import { parseEvent } from "./api";

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
