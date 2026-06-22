import { describe, expect, it } from "vitest";
import { chatEndpoint, describeService, extractModels } from "./services";

describe("BYO-AI service layer", () => {
  it("extracts Ollama models", () => {
    expect(extractModels("ollama", { models: [{ name: "llama3" }, { model: "qwen2" }] })).toEqual(["llama3", "qwen2"]);
  });
  it("extracts OpenAI-style models", () => {
    expect(extractModels("openai", { data: [{ id: "gpt" }, { id: "mistral" }] })).toEqual(["gpt", "mistral"]);
  });
  it("returns [] for junk", () => {
    expect(extractModels("openai", null)).toEqual([]);
  });
  it("marks crucible full, ollama chat-only", () => {
    expect(describeService({ type: "crucible", name: "C", baseUrl: "x" }, []).full).toBe(true);
    const o = describeService({ type: "ollama", name: "O", baseUrl: "x" }, []);
    expect(o.full).toBe(false);
    expect(o.chat).toBe(true);
    expect(o.note).toContain("write access");
  });
  it("comfyui is not a chat backend", () => {
    expect(describeService({ type: "comfyui", name: "CF", baseUrl: "x" }, []).chat).toBe(false);
  });
  it("builds the OpenAI-compatible chat endpoint", () => {
    expect(chatEndpoint(describeService({ type: "ollama", name: "O", baseUrl: "http://localhost:11434/" }, [])))
      .toBe("http://localhost:11434/v1/chat/completions");
  });
});
