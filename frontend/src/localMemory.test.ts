import { beforeEach, describe, expect, it } from "vitest";
import {
  localCompact, localConsolidate, localCrystallize, localIndex, localRead, localReadForSplit,
  localRecrystallize, localSearch, localTree,
} from "./localMemory";

// vitest/node has no localStorage — install a Map-backed shim so the on-device store is exercised.
class MemStore {
  m = new Map<string, string>();
  getItem(k: string): string | null { return this.m.has(k) ? this.m.get(k)! : null; }
  setItem(k: string, v: string): void { this.m.set(k, v); }
  removeItem(k: string): void { this.m.delete(k); }
}
beforeEach(() => { (globalThis as unknown as { localStorage: MemStore }).localStorage = new MemStore(); });

const msgs = (n: number, tag = "t") =>
  Array.from({ length: n }, (_, i) => ({ role: i % 2 === 0 ? "user" : "assistant", content: `${tag}${i} content` }));

describe("localMemory (on-device crystallized memory)", () => {
  it("crystallize + index (summary passthrough)", () => {
    localCrystallize(msgs(4), "abliteration removed the refusal", "uncensor", "s1");
    localCrystallize(msgs(6), "quantized weights for speed", "quantize", "s1");
    const idx = localIndex();
    expect(idx.versioned).toBe(false);
    expect(idx.memories.map((m) => m.key)).toEqual(["m-0001", "m-0002"]);
    expect(idx.memories[1]?.size).toBe(6);
    expect("messages" in idx.memories[0]!).toBe(false);   // cheap card, no bodies
  });

  it("index filters by session", () => {
    localCrystallize(msgs(2), "a", "", "a");
    localCrystallize(msgs(2), "b", "", "b");
    expect(localIndex("a").memories.map((m) => m.key)).toEqual(["m-0001"]);
  });

  it("read returns full messages for a leaf", () => {
    localCrystallize(msgs(3, "x"), "sum");
    const r = localRead("m-0001");
    expect(r.kind).toBe("leaf");
    expect(r.messages?.length).toBe(3);
    expect(r.messages?.[0]?.content).toContain("x0");
    expect(() => localRead("m-9999")).toThrow();
  });

  it("lexical search ranks by relevance and labels the method", () => {
    localCrystallize(msgs(2), "abliteration removed the refusal direction", "a");
    localCrystallize(msgs(2), "quantization compresses weights", "b");
    const res = localSearch("refusal direction");
    expect(res.method).toBe("lexical");
    expect(res.matches[0]?.key).toBe("m-0001");
    expect(localSearch("xylophone").matches).toEqual([]);
  });

  it("recrystallize splits a leaf into chunked children", () => {
    localCrystallize(msgs(10), "the whole thing");
    const chunks = localReadForSplit("m-0001", 2);
    const res = localRecrystallize("m-0001", chunks);
    expect(res.kind).toBe("chunked" as string);
    expect(res.children.length).toBe(2);
    const parent = localRead("m-0001");
    expect(parent.kind).toBe("chunked");
    expect("messages" in parent).toBe(false);
    expect((parent.children ?? []).length).toBe(2);
  });

  it("consolidate files top-level memories under a new domain node (LCA=top)", () => {
    localCrystallize(msgs(2), "work A", "a");
    localCrystallize(msgs(2), "work B", "b");
    const dom = localConsolidate(["m-0001", "m-0002"], "all the work", "domain");
    expect(dom.kind).toBe("chunked");
    // only the domain node is top-level now; the two originals are filed under it
    const top = localTree();
    expect(top.map((n) => n.key)).toEqual([dom.key]);
    expect(top[0]?.children?.length).toBe(2);
  });

  it("consolidate rejects <2 and ancestor-of-self", () => {
    localCrystallize(msgs(6), "root");
    localRecrystallize("m-0001", localReadForSplit("m-0001", 1));
    expect(() => localConsolidate(["m-0001"], "one")).toThrow();
    expect(() => localConsolidate(["m-0001", "m-0002"], "parent+child")).toThrow();
  });

  it("local compaction crystallizes an extractive summary (no model)", () => {
    const convo = [{ role: "system", content: "sys" }, ...msgs(20)];
    const out = localCompact(convo, 4);
    expect(out.compacted).toBe(true);
    expect(out.summary).toContain("extractive");
    expect(out.messages.length).toBeLessThan(convo.length);
    expect(out.stats.token_estimate).toContain("heuristic");
    // it was stored so it can be recalled later
    expect(localIndex().memories.length).toBe(1);
  });
});
