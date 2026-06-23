import { describe, expect, it } from "vitest";
import { reduce } from "./components/AgentConsole";
import type { Turn } from "./components/AgentConsole";
import type { AgentEvent } from "./api";

// deterministic id generator for the reducer
function ids(): () => string {
  let n = 0;
  return () => `t${++n}`;
}

const delta = (s: string): AgentEvent => ({ type: "assistant_delta", data: { delta: s } });
const assistant = (content: string, streamed: boolean): AgentEvent => ({
  type: "assistant",
  data: { content, streamed },
});

describe("reduce — BYO token streaming", () => {
  it("accumulates deltas into one streaming turn, then finalizes on assistant", () => {
    const nextId = ids();
    let turns: readonly Turn[] = [];
    turns = reduce(turns, delta("Hel"), nextId);
    turns = reduce(turns, delta("lo"), nextId);

    expect(turns).toHaveLength(1);
    const live = turns[0];
    expect(live?.kind).toBe("assistant");
    expect(live && live.kind === "assistant" && live.text).toBe("Hello");
    expect(live && live.kind === "assistant" && live.streaming).toBe(true);

    // authoritative final replaces text in-place and clears the streaming flag
    turns = reduce(turns, assistant("Hello", true), nextId);
    expect(turns).toHaveLength(1);
    const done = turns[0];
    expect(done && done.kind === "assistant" && done.text).toBe("Hello");
    expect(done && done.kind === "assistant" && done.streaming).toBe(false);
  });

  it("non-streamed assistant event appends a fresh turn (no regression)", () => {
    const nextId = ids();
    const turns = reduce([], assistant("done", false), nextId);
    expect(turns).toHaveLength(1);
    expect(turns[0]?.kind).toBe("assistant");
    expect(turns[0] && turns[0].kind === "assistant" && turns[0].streaming).toBeUndefined();
  });

  it("a second streamed message after a finalized one opens a new turn", () => {
    const nextId = ids();
    let turns: readonly Turn[] = [];
    turns = reduce(turns, delta("a"), nextId);
    turns = reduce(turns, assistant("a", true), nextId);
    turns = reduce(turns, delta("b"), nextId);
    expect(turns).toHaveLength(2);
    expect(turns[1] && turns[1].kind === "assistant" && turns[1].streaming).toBe(true);
    expect(turns[1] && turns[1].kind === "assistant" && turns[1].text).toBe("b");
  });
});
