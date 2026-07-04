import { describe, expect, it } from "vitest";
import {
  array, bool, literals, nullable, num, object, optional, record, ShapeError, str,
} from "./validate";
import {
  benchmarksInfoP, guardrailConfigP, modelRowP, modelRowsP, publishedPayloadP,
  runtimeStatusP, verifyReportP,
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

  it("a truncated/HTML error body fails loudly instead of silently passing", () => {
    expect(() => verifyReportP("<html>502 Bad Gateway</html>")).toThrow(ShapeError);
    expect(() => runtimeStatusP(null)).toThrow(ShapeError);
  });
});
