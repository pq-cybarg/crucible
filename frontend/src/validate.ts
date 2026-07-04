// Dependency-free runtime validation for the backend boundary. Every fetch helper in api.ts used
// to `as`-cast the parsed JSON, which lies to the compiler: a 404 HTML page, an error body, a
// null, or a renamed field would sail past the cast and blow up somewhere deep in a component.
// These combinators validate the shape AT THE SEAM and throw a precise ShapeError (with a field
// path) the instant the backend returns something unexpected.
//
// Type safety: a parser built with object({...}) infers its result type from the shape, so
// annotating `const p: Parser<Foo> = object({...})` makes the COMPILER prove the schema matches
// the Foo interface — drift (missing field, wrong field type) is a build error, not a runtime
// surprise. optional() maps to a truly-optional key (respecting exactOptionalPropertyTypes);
// unknown keys in the payload are ignored so the backend can add fields without breaking the GUI.

export class ShapeError extends Error {
  constructor(message: string) {
    super(message);
    this.name = "ShapeError";
  }
}

export type Parser<T> = (value: unknown, path?: string) => T;

// Brand so object() can tell an optional field from a required one at the type level.
declare const OPTIONAL: unique symbol;
export interface OptionalParser<T> {
  (value: unknown, path?: string): T | undefined;
  readonly [OPTIONAL]: true;
}

function typeName(v: unknown): string {
  if (v === null) return "null";
  if (Array.isArray(v)) return "array";
  return typeof v;
}

function fail(path: string, expected: string, got: unknown): never {
  throw new ShapeError(`${path || "value"}: expected ${expected}, got ${typeName(got)}`);
}

export const str: Parser<string> = (v, p = "") => (typeof v === "string" ? v : fail(p, "string", v));

export const num: Parser<number> = (v, p = "") =>
  typeof v === "number" && Number.isFinite(v) ? v : fail(p, "finite number", v);

export const bool: Parser<boolean> = (v, p = "") => (typeof v === "boolean" ? v : fail(p, "boolean", v));

// Pass-through for genuinely opaque values (e.g. metadata records of mixed type).
export const unknown: Parser<unknown> = (v) => v;

export function literals<L extends string>(...allowed: readonly L[]): Parser<L> {
  const set = new Set<string>(allowed);
  return (v, p = "") => (typeof v === "string" && set.has(v) ? (v as L) : fail(p, allowed.join("|"), v));
}

export function nullable<T>(inner: Parser<T>): Parser<T | null> {
  return (v, p = "") => (v === null ? null : inner(v, p));
}

export function optional<T>(inner: Parser<T>): OptionalParser<T> {
  const f = (v: unknown, p = ""): T | undefined => (v === undefined ? undefined : inner(v, p));
  return f as OptionalParser<T>;
}

export function array<T>(inner: Parser<T>): Parser<readonly T[]> {
  return (v, p = "") => {
    if (!Array.isArray(v)) return fail(p, "array", v);
    return v.map((el, i) => inner(el, `${p}[${i}]`));
  };
}

export function record<T>(inner: Parser<T>): Parser<Readonly<Record<string, T>>> {
  return (v, p = "") => {
    if (typeof v !== "object" || v === null || Array.isArray(v)) return fail(p, "object", v);
    const out: Record<string, T> = {};
    for (const [k, val] of Object.entries(v)) out[k] = inner(val, p ? `${p}.${k}` : k);
    return out;
  };
}

type Shape = Record<string, Parser<unknown>>;
type OptionalKeys<S extends Shape> = { [K in keyof S]: S[K] extends OptionalParser<unknown> ? K : never }[keyof S];
type RequiredKeys<S extends Shape> = Exclude<keyof S, OptionalKeys<S>>;
// Required keys carry their parser's return type; optional keys become truly-optional (`?`).
type Infer<S extends Shape> =
  { [K in RequiredKeys<S>]: ReturnType<S[K]> } &
  { [K in OptionalKeys<S>]?: S[K] extends OptionalParser<infer U> ? U : never };

export function object<S extends Shape>(shape: S): Parser<Infer<S>> {
  const entries = Object.entries(shape);
  return (v, p = "") => {
    if (typeof v !== "object" || v === null || Array.isArray(v)) return fail(p, "object", v);
    const rec = v as Record<string, unknown>;
    const out: Record<string, unknown> = {};
    for (const [key, parser] of entries) {
      const parsed = parser(rec[key], p ? `${p}.${key}` : key);
      if (parsed !== undefined) out[key] = parsed;   // omit (don't set undefined) for exactOptionalPropertyTypes
    }
    return out as Infer<S>;
  };
}
