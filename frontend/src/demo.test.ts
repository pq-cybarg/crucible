import { describe, expect, it } from "vitest";
import { demoRespond } from "./demo";

async function body(path: string, method = "GET"): Promise<Record<string, unknown>> {
  const r = demoRespond(path, { method });
  expect(r).not.toBeNull();
  return (await r!.json()) as Record<string, unknown>;
}

describe("demoRespond (static Pages demo layer)", () => {
  it("serves the models list", async () => {
    const r = demoRespond("/api/models", { method: "GET" });
    const m = (await r!.json()) as unknown[];
    expect(Array.isArray(m) && m.length).toBeGreaterThan(0);
  });
  it("serves the diagnosis with the expected shape", async () => {
    const d = await body("/api/abliteration/diagnose", "POST");
    expect(d["best_layer"]).toBe(21);
    expect(d["surgical"]).toBe(true);
    expect(Array.isArray(d["layer_profile"])).toBe(true);
  });
  it("serves the feature card", async () => {
    const d = await body("/api/abliteration/feature-card", "POST");
    expect(d["name"]).toBe("The Apology Reflex");
    expect(Array.isArray(d["triggers"])).toBe(true);
  });
  it("generates a heatmap (25 layers x N tokens)", async () => {
    const d = await body("/api/abliteration/heatmap", "POST");
    expect((d["matrix"] as unknown[]).length).toBe(25);
    expect((d["tokens"] as unknown[]).length).toBeGreaterThan(0);
  });
  it("serves the three guardrail presets", async () => {
    const r = demoRespond("/api/guardrails/presets", { method: "GET" });
    const p = (await r!.json()) as { id: string }[];
    expect(p.map((x) => x.id)).toEqual(["unrestricted", "balanced", "strict"]);
  });
  it("returns null for an unknown GET (so live fetch falls through)", () => {
    expect(demoRespond("/api/does-not-exist", { method: "GET" })).toBeNull();
  });
  it("returns a benign demo ack for an unknown mutation", async () => {
    const d = await body("/api/does-not-exist", "POST");
    expect(d["demo"]).toBe(true);
  });
});
