import { describe, expect, it } from "vitest";
import { toModelHistory } from "./chatHistory";

// #31 — animation context must stay OUT of chat context. These lock in that the model only ever sees the
// human<->model dialogue, so animating the companion (many render frames, moods, tool activity) can never
// inflate the context window.
describe("toModelHistory — animation/chat context separation (#31)", () => {
  it("keeps only user + assistant dialogue, in order", () => {
    const out = toModelHistory([
      { kind: "user", text: "hi" },
      { kind: "assistant", text: "hello" },
      { kind: "tool", text: "ran search()" },      // tool-use → excluded (lives in the ToolFeed, not context)
      { kind: "notice", text: "compacted" },
      { kind: "memory" },
      { kind: "permission", text: "approve?" },
      { kind: "assistant", text: "done" },
    ]);
    expect(out).toEqual([
      { role: "user", content: "hi" },
      { role: "assistant", content: "hello" },
      { role: "assistant", content: "done" },
    ]);
  });

  it("excludes ANY avatar/animation-flavoured turn, whatever fields it carries", () => {
    const out = toModelHistory([
      { kind: "avatar", text: "mood:happy" },
      { kind: "animation", text: "blink" },
      { kind: "companion", text: "talking" },
      { kind: "user", text: "question" },
    ]);
    expect(out).toEqual([{ role: "user", content: "question" }]);
  });

  it("drops conversation turns with no text (defensive)", () => {
    expect(toModelHistory([{ kind: "assistant" }, { kind: "user", text: "q" }]))
      .toEqual([{ role: "user", content: "q" }]);
  });

  it("empty in → empty out", () => {
    expect(toModelHistory([])).toEqual([]);
  });
});
