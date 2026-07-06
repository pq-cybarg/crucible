import { describe, expect, it } from "vitest";
import {
  array, bool, literals, nullable, num, object, optional, record, ShapeError, str,
} from "./validate";
import {
  benchmarksInfoP, compactResultP, graphResultP, guardrailConfigP, hierarchyProfileP, lineageP,
  mediaStatusP, memoryNodeP, memorySearchP, memoryTreeP, modalityDirectionP, modelRowP, modelRowsP,
  profilesP, publishedPayloadP, runtimeStatusP, verifyReportP, weightsViewP,
} from "./schemas";

describe("validate combinators", () => {
  it("primitives accept the right type and reject others", () => {
    expect(str("hi")).toBe("hi");
    expect(num(3)).toBe(3);
    expect(bool(true)).toBe(true);
    expect(() => str(3)).toThrow(ShapeError);
    expect(() => num("3")).toThrow(ShapeError);
    expect(() => num(NaN)).toThrow(ShapeError);           // not a finite number
    expect(() => bool(1)).toThrow(ShapeError);
  });

  it("nullable allows null but still validates non-null", () => {
    expect(nullable(str)(null)).toBeNull();
    expect(nullable(str)("x")).toBe("x");
    expect(() => nullable(str)(5)).toThrow(ShapeError);
  });

  it("literals only accept allowed strings", () => {
    const kind = literals("a", "b");
    expect(kind("a")).toBe("a");
    expect(() => kind("c")).toThrow(ShapeError);
  });

  it("array validates every element and reports the index in the path", () => {
    expect(array(num)([1, 2, 3])).toEqual([1, 2, 3]);
    expect(() => array(num)("nope")).toThrow(/expected array/);
    expect(() => array(num)([1, "x"])).toThrow(/\[1\]/);   // path points at the bad element
  });

  it("record validates values", () => {
    expect(record(num)({ a: 1, b: 2 })).toEqual({ a: 1, b: 2 });
    expect(() => record(num)({ a: "x" })).toThrow(/a: expected finite number/);
    // nested record reports the dotted path
    expect(() => object({ outer: record(num) })({ outer: { a: "x" } })).toThrow(/outer\.a/);
    expect(() => record(num)([1, 2])).toThrow(ShapeError); // array is not a record
  });

  it("object omits absent optional keys (exactOptionalPropertyTypes) and keeps present ones", () => {
    const p = object({ a: num, b: optional(str) });
    expect(p({ a: 1 })).toEqual({ a: 1 });                 // b omitted, not set to undefined
    expect("b" in p({ a: 1 })).toBe(false);
    expect(p({ a: 1, b: "x" })).toEqual({ a: 1, b: "x" });
    expect(() => p({ b: "x" })).toThrow(/a:/);             // missing required a
  });

  it("object ignores unknown keys (backend may add fields)", () => {
    const p = object({ a: num });
    expect(p({ a: 1, extra: "ignored", plain: { headline: "x" } })).toEqual({ a: 1 });
  });

  it("nested paths surface in the error message", () => {
    const p = object({ outer: object({ inner: num }) });
    expect(() => p({ outer: { inner: "bad" } })).toThrow(/outer\.inner: expected finite number/);
  });
});

describe("real schemas parse valid payloads and reject malformed ones", () => {
  const goodModelRow = {
    id: "m1", name: "M", base_id: null, path: "/x", quant: "Q4",
    kind: "base", endpoint: null, created: "2026", notes: "",
  };

  it("modelRowP accepts a valid row and rejects a bad kind", () => {
    expect(modelRowP(goodModelRow).id).toBe("m1");
    expect(() => modelRowP({ ...goodModelRow, kind: "weird" })).toThrow(/kind:/);
  });

  it("modelRowsP rejects a non-array (e.g. an error object)", () => {
    expect(modelRowsP([goodModelRow])).toHaveLength(1);
    expect(() => modelRowsP({ detail: "boom" })).toThrow(ShapeError);
  });

  it("runtimeStatusP validates nested resident instances", () => {
    const ok = runtimeStatusP({ max_resident: 2, resident: [], active: [] });
    expect(ok.max_resident).toBe(2);
    expect(() => runtimeStatusP({ max_resident: "two", resident: [], active: [] })).toThrow(/max_resident/);
  });

  it("guardrailConfigP validates the regex-rule array shape", () => {
    const cfg = guardrailConfigP({
      enabled: true, preset_id: "p", constitution: "", constitution_enabled: false,
      regex_rules: [{ pattern: "x", mode: "block", label: "l", stages: ["input"] }],
    });
    expect(cfg.regex_rules[0]?.mode).toBe("block");
    expect(() => guardrailConfigP({
      enabled: true, preset_id: "p", constitution: "", constitution_enabled: false,
      regex_rules: [{ pattern: "x", mode: "explode", label: "l", stages: ["input"] }],
    })).toThrow(/mode/);
  });

  it("verifyReportP validates the before/after blocks", () => {
    const r = verifyReportP({
      harmful_refusal_rate: { before: 0.9, after: 0.1 },
      harmful_compliance_rate: { before: 0.1, after: 0.9 },
      benign_over_refusal_rate: { before: 0, after: 0 },
      samples: [{ prompt: "p", before: "a", after: "b" }],
      plain: { headline: "ignored extra" },
    });
    expect(r.harmful_refusal_rate.after).toBe(0.1);
  });

  it("benchmarksInfoP and publishedPayloadP parse the honest-eval shapes", () => {
    expect(benchmarksInfoP({ benchmarks: { "mmlu-sample": 28 }, kind: "quick-screen samples", note: "n" })
      .benchmarks["mmlu-sample"]).toBe(28);
    const pub = publishedPayloadP({
      providers: { "GLM": { "SWE": { value: 0.77, source: "s", verified: false } } },
      disclaimer: "context only",
    });
    expect(pub.providers["GLM"]?.["SWE"]?.verified).toBe(false);
  });

  it("mediaStatusP parses the capability map with nullable reachable/endpoint", () => {
    const st = mediaStatusP({
      backends: {
        image: { kind: "image", label: "text-to-image", env: "CRUCIBLE_IMAGE_ENDPOINT",
          endpoint: "http://x:8188", configured: true, reachable: null },
        stt: { kind: "stt", label: "speech-to-text", env: "CRUCIBLE_STT_ENDPOINT",
          endpoint: null, configured: false, reachable: null },
      },
      n_configured: 1, n_total: 2, note: "brokered",
    });
    expect(st.backends["image"]?.configured).toBe(true);
    expect(st.backends["stt"]?.endpoint).toBeNull();
    expect(() => mediaStatusP({ backends: { x: { kind: "x" } }, n_configured: 0, n_total: 1, note: "" }))
      .toThrow(/label/);   // missing required fields fail loudly
  });

  it("compactResultP parses the compaction payload (nullable summary)", () => {
    const r = compactResultP({
      messages: [{ role: "system", content: "summary" }, { role: "user", content: "hi" }],
      summary: "did stuff", compacted: true,
      stats: { before_tokens: 100, after_tokens: 20, summarized_turns: 8, token_estimate: "heuristic" },
      tokens: 100,
    });
    expect(r.compacted).toBe(true);
    expect(r.messages[0]?.role).toBe("system");
    expect(compactResultP({
      messages: [], summary: null, compacted: false,
      stats: { before_tokens: 5, after_tokens: 5, summarized_turns: 0, token_estimate: "heuristic" },
      tokens: 5,
    }).summary).toBeNull();
  });

  it("graphResultP parses the DAG result with opaque per-stage outputs", () => {
    const g = graphResultP({
      order: ["a", "vote"],
      outputs: { a: "ECHO:hi", vote: { strategy: "majority", result: "ECHO:hi", n: 3, agreement: 1 } },
      result: { vote: { result: "ECHO:hi" } },
    });
    expect(g.order).toEqual(["a", "vote"]);
    expect(typeof g.outputs["a"]).toBe("string");
    expect((g.outputs["vote"] as { agreement: number }).agreement).toBe(1);
    expect(() => graphResultP({ order: "nope", outputs: {}, result: {} })).toThrow(/order/);
  });

  it("modalityDirectionP parses the direction result with its plain card", () => {
    const m = modalityDirectionP({
      modality: "image", n_harmful: 20, n_benign: 20, dim: 16, separability: 3.1,
      separability_kind: "held-out (2-fold cross-validated)", in_sample_separability: 3.8,
      reliable: true, reliability_note: "ok", linearly_encoded: true, direction_norm: 1,
      direction: [0.1, 0.2, 0.3],
      plain: { technique: "modality-direction", headline: "h", what_it_is: "a", what_we_found: "b",
        what_it_means: "c", caveat: "d" },
    });
    expect(m.linearly_encoded).toBe(true);
    expect(m.plain.headline).toBe("h");
    // missing plain card fails loudly (it's required)
    expect(() => modalityDirectionP({
      modality: "image", n_harmful: 1, n_benign: 1, dim: 2, separability: 0,
      separability_kind: "x", in_sample_separability: 0, reliable: false, reliability_note: "n",
      linearly_encoded: false, direction_norm: 1, direction: [1, 2],
    })).toThrow(/plain/);
  });

  it("memory schemas parse a leaf node and a recursive tree", () => {
    const leaf = memoryNodeP({
      key: "m-0001", label: "setup", summary: "did setup", kind: "leaf", session: "s", size: 4,
      ref: "abc123", messages: [{ role: "user", content: "hi" }],
    });
    expect(leaf.messages?.[0]?.content).toBe("hi");
    // recursive tree: a chunked node with nested children (ref may be null)
    const tree = memoryTreeP({
      tree: [{
        key: "m-0001", label: "domain", summary: "s", kind: "chunked", session: "s", size: 2, ref: null,
        children: [
          { key: "m-0002", label: "a", summary: "sa", kind: "leaf", session: "s", size: 3, ref: null },
          { key: "m-0003", label: "b", summary: "sb", kind: "chunked", session: "s", size: 1, ref: null,
            children: [{ key: "m-0004", label: "c", summary: "sc", kind: "leaf", session: "s", size: 2, ref: null }] },
        ],
      }],
    });
    expect(tree.tree[0]?.children?.[1]?.children?.[0]?.key).toBe("m-0004");
    // a malformed node inside the tree fails loudly (with a path)
    expect(() => memoryTreeP({ tree: [{ key: "m-1", label: "x", summary: "s", kind: "leaf", session: "s", size: "big", ref: null }] }))
      .toThrow(/size/);
  });

  it("weightsViewP parses the humanized explain payload (optional)", () => {
    const v = weightsViewP({
      summary: { n_tensors: 2, total_params: 100, n_layers: 2, dtypes: { F16: 2 }, architecture: "x" },
      tensors: [], metadata: {},
      explain: {
        model: { headline: "h", what_it_is: "a", how_it_works: "b", size_meaning: "c", how_to_change: "d" },
        layers: [{ layer: 0, band: "early", role: "r", params: 50, components: ["attention"] }],
        legend: { early: "e", middle: "m", late: "l" },
      },
    });
    expect(v.explain?.model.headline).toBe("h");
    expect(v.explain?.layers[0]?.band).toBe("early");
    // explain is optional — a view without it still parses
    expect(weightsViewP({
      summary: { n_tensors: 0, total_params: 0, n_layers: 0, dtypes: {}, architecture: null },
      tensors: [], metadata: {},
    }).explain).toBeUndefined();
  });

  it("memorySearchP parses ranked matches with their method", () => {
    const r = memorySearchP({
      method: "lexical",
      matches: [{ key: "m-0001", label: "a", summary: "s", kind: "leaf", session: "x", size: 4, ref: null, score: 2.3 }],
    });
    expect(r.method).toBe("lexical");
    expect(r.matches[0]?.score).toBe(2.3);
    expect(() => memorySearchP({ method: "semantic", matches: [{ key: "m", label: "a", summary: "s", kind: "leaf", session: "x", size: 1, ref: null }] })).toThrow(/score/);
  });

  it("lineageP parses per-part version chains", () => {
    const l = lineageP({
      branch: "main",
      parts: [{ part: "language_model", n_versions: 2, latest: "c3",
        commits: [{ id: "c1", op: "inplace", summary: "edit 1" }, { id: "c3", op: "inplace", summary: "edit 2" }] }],
    });
    expect(l.parts[0]?.part).toBe("language_model");
    expect(l.parts[0]?.commits[1]?.id).toBe("c3");
  });

  it("hierarchy profile schemas parse layers with nullable model ids", () => {
    const p = hierarchyProfileP({ name: "research", layers: [
      { worker: null, communicator: null }, { worker: "opus", communicator: "haiku" }] });
    expect(p.layers[1]?.communicator).toBe("haiku");
    expect(profilesP({ profiles: [p] }).profiles[0]?.name).toBe("research");
    expect(() => hierarchyProfileP({ name: "x", layers: [{ worker: 5, communicator: null }] })).toThrow(/worker/);
  });

  it("a truncated/HTML error body fails loudly instead of silently passing", () => {
    expect(() => verifyReportP("<html>502 Bad Gateway</html>")).toThrow(ShapeError);
    expect(() => runtimeStatusP(null)).toThrow(ShapeError);
  });
});
