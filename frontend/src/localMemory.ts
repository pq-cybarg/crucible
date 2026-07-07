// Client-side crystallized memory for the static (GitHub Pages) build, where there's no backend.
// Mirrors the server MemoryStore shapes and behaviour (crystallize / index / tree / read / search /
// consolidate / recrystallize) but persists to LocalStorage — so memory lookup works fully on-device
// in the browser. When a real Crucible node is connected the api layer uses the server store instead;
// this is the offline fallback. Search is BM25 lexical (honest — labeled "lexical", not semantic);
// compaction here uses an extractive heuristic summary (no model in the browser), also labeled.
import type {
  CompactMessage, CompactResult, MemoryCard, MemoryMatch, MemoryNode, MemorySearchResult, MemoryTreeNode,
} from "./api";

type Stored = {
  key: string; kind: "leaf" | "chunked"; label: string; summary: string; session: string;
  parent: string | null; n_messages: number; stats: Record<string, unknown>;
  messages?: readonly CompactMessage[]; children?: string[];
  priority?: number; links?: { to: string; type: string }[];
};
type Db = { seq: number; nodes: Record<string, Stored> };

const KEY = "crucible_memory";

function load(): Db {
  try {
    const raw = typeof localStorage !== "undefined" ? localStorage.getItem(KEY) : null;
    if (raw) return JSON.parse(raw) as Db;
  } catch { /* corrupt/absent → fresh */ }
  return { seq: 0, nodes: {} };
}
function save(db: Db): void {
  try { localStorage.setItem(KEY, JSON.stringify(db)); } catch { /* quota / private mode — best effort */ }
}
function nextKey(db: Db): string {
  db.seq += 1;
  return `m-${String(db.seq).padStart(4, "0")}`;
}
function deriveLabel(summary: string): string {
  const w = (summary || "").replace(/\n/g, " ").split(" ").filter(Boolean).slice(0, 6).join(" ").replace(/[.,:;—-]+$/, "");
  return w || "memory";
}
function card(n: Stored): MemoryCard & { priority: number; degree: number } {
  const size = n.kind === "chunked" ? (n.children?.length ?? 0) : n.n_messages;
  return { key: n.key, label: n.label, summary: n.summary, kind: n.kind, session: n.session, size, ref: null,
    priority: n.priority ?? 0, degree: (n.links ?? []).length };
}

// Configurable ordering, mirroring the backend sorting module (recency / priority / size / degree /
// label / balanced). "balanced" blends recency + priority (salience) the way human recall does —
// privilege what's recent AND what matters — instead of pure positional bias.
function seq(key: string): number { const d = key.replace(/\D/g, ""); return d ? Number(d) : 0; }
export const BALANCED_RECENCY_WEIGHT = 0.5;
function normBy<T>(items: T[], val: (x: T) => number): (i: number) => number {
  const vs = items.map(val); const lo = Math.min(...vs), hi = Math.max(...vs); const span = hi - lo;
  return (i) => (span <= 0 ? 0.5 : (vs[i]! - lo) / span);   // flat set → neutral 0.5, not dominant
}
function sortCards<T extends { key: string; label?: string; size?: number; priority?: number; degree?: number; score?: number }>(
  items: T[], by: string, recencyWeight = BALANCED_RECENCY_WEIGHT): T[] {
  if (by === "balanced") {
    const w = Math.max(0, Math.min(1, recencyWeight));
    const rec = normBy(items, (x) => seq(x.key)), pri = normBy(items, (x) => x.priority ?? 0);
    const score = items.map((_, i) => w * rec(i) + (1 - w) * pri(i));
    return items.map((_, i) => i).sort((a, b) => score[b]! - score[a]!).map((i) => items[i]!);
  }
  const keyed: Record<string, [(x: T) => number | string, boolean]> = {
    relevance: [(x) => x.score ?? 0, true], priority: [(x) => (x.priority ?? 0) * 1e7 + seq(x.key), true],
    size: [(x) => x.size ?? 0, true], degree: [(x) => x.degree ?? 0, true],
    recency: [(x) => seq(x.key), true], oldest: [(x) => seq(x.key), false],
    label: [(x) => (x.label ?? "").toLowerCase(), false],
  };
  const spec = keyed[by];
  if (!spec) return items;
  const [fn, desc] = spec;
  return [...items].sort((a, b) => { const av = fn(a), bv = fn(b); const c = av < bv ? -1 : av > bv ? 1 : 0; return desc ? -c : c; });
}

// --- distance/similarity metrics (mirror backend crucible.metrics — the OFFLINE families only) ------
// The statistical + lexical families run fully in-browser; embedding (needs an embed backend) and
// llm-judged (needs a processing model) are backend-only and reported as unavailable here.
export const OFFLINE_METRICS = ["bm25", "jaccard", "dice", "overlap", "tfidf", "edit"] as const;
export const METRIC_LABELS: Record<string, string> = {
  bm25: "lexical-bm25", jaccard: "statistical-jaccard", dice: "statistical-dice",
  overlap: "statistical-overlap", tfidf: "statistical-tfidf-cosine", edit: "statistical-edit-distance",
  embedding: "semantic-embedding", llm: "llm-judged",
};
function tokenize(t: string): string[] { return (t.toLowerCase().match(/[a-z0-9]+/g) ?? []); }
function tset(t: string): Set<string> { return new Set(tokenize(t)); }
function inter(a: Set<string>, b: Set<string>): number { let n = 0; for (const x of a) if (b.has(x)) n++; return n; }

function jaccard(q: string, docs: string[]): number[] {
  const qs = tset(q);
  return docs.map((d) => { const ds = tset(d); const u = new Set([...qs, ...ds]).size; return u ? inter(qs, ds) / u : 0; });
}
function dice(q: string, docs: string[]): number[] {
  const qs = tset(q);
  return docs.map((d) => { const ds = tset(d); const s = qs.size + ds.size; return s ? (2 * inter(qs, ds)) / s : 0; });
}
function overlap(q: string, docs: string[]): number[] {
  const qs = tset(q);
  return docs.map((d) => { const ds = tset(d); const m = Math.min(qs.size, ds.size); return m ? inter(qs, ds) / m : 0; });
}
function tfidfCosine(q: string, docs: string[]): number[] {
  const toks = docs.map(tokenize); const n = docs.length || 1;
  const df = new Map<string, number>();
  for (const t of toks) for (const w of new Set(t)) df.set(w, (df.get(w) ?? 0) + 1);
  const idf = (w: string): number => Math.log((1 + n) / (1 + (df.get(w) ?? 0))) + 1;
  const vec = (t: string[]): Map<string, number> => {
    const tf = new Map<string, number>(); for (const w of t) tf.set(w, (tf.get(w) ?? 0) + 1);
    const v = new Map<string, number>(); for (const [w, f] of tf) v.set(w, f * idf(w)); return v;
  };
  const cos = (a: Map<string, number>, b: Map<string, number>): number => {
    if (!a.size || !b.size) return 0;
    let dot = 0; for (const [w, x] of a) dot += x * (b.get(w) ?? 0);
    const na = Math.sqrt([...a.values()].reduce((s, x) => s + x * x, 0));
    const nb = Math.sqrt([...b.values()].reduce((s, x) => s + x * x, 0));
    return na && nb ? dot / (na * nb) : 0;
  };
  const qv = vec(tokenize(q));
  return toks.map((t) => cos(qv, vec(t)));
}
function levenshtein(a: string, b: string): number {
  if (a === b) return 0; if (!a) return b.length; if (!b) return a.length;
  let prev = Array.from({ length: b.length + 1 }, (_, i) => i);
  for (let i = 1; i <= a.length; i++) {
    const cur = [i];
    for (let j = 1; j <= b.length; j++) cur[j] = Math.min(prev[j]! + 1, cur[j - 1]! + 1, prev[j - 1]! + (a[i - 1] === b[j - 1] ? 0 : 1));
    prev = cur;
  }
  return prev[b.length]!;
}
function editSim(q: string, docs: string[]): number[] {
  const ql = q.toLowerCase();
  return docs.map((d) => { const t = d.toLowerCase(); const m = Math.max(ql.length, t.length); return m ? 1 - levenshtein(ql, t) / m : 1; });
}
function minmax(vs: number[]): number[] {
  if (!vs.length) return []; const lo = Math.min(...vs), hi = Math.max(...vs); const span = hi - lo;
  return span <= 0 ? vs.map(() => 0) : vs.map((v) => (v - lo) / span);
}
// Score docs by the named offline metric, normalized to [0,1]. Unknown/backend-only metrics fall
// back to BM25 (honestly — the caller reports the actual method used).
function metricScores(query: string, docs: string[], metric: string): { method: string; scores: number[] } {
  const fn: Record<string, (q: string, d: string[]) => number[]> = {
    bm25, jaccard, dice, overlap, tfidf: tfidfCosine, edit: editSim,
  };
  const chosen = fn[metric] ? metric : "bm25";
  return { method: METRIC_LABELS[chosen]!, scores: minmax(fn[chosen]!(query, docs)) };
}

function bm25(query: string, docs: string[], k1 = 1.5, b = 0.75): number[] {
  const toks = docs.map(tokenize);
  const N = docs.length || 1;
  const avgdl = (toks.reduce((s, t) => s + t.length, 0) / N) || 1;
  const df = new Map<string, number>();
  for (const t of toks) for (const term of new Set(t)) df.set(term, (df.get(term) ?? 0) + 1);
  const q = tokenize(query);
  return toks.map((t) => {
    const tf = new Map<string, number>();
    for (const w of t) tf.set(w, (tf.get(w) ?? 0) + 1);
    const dl = t.length || 1;
    let s = 0;
    for (const term of q) {
      const f = tf.get(term); if (!f) continue;
      const idf = Math.log(1 + (N - (df.get(term) ?? 0) + 0.5) / ((df.get(term) ?? 0) + 0.5));
      s += idf * (f * (k1 + 1)) / (f + k1 * (1 - b + b * dl / avgdl));
    }
    return s;
  });
}

// --- public API (matches the server functions the app calls) -------------------------------
export function localCrystallize(messages: readonly CompactMessage[], summary: string,
                                 label = "", session = ""): MemoryCard {
  const db = load();
  const key = nextKey(db);
  db.nodes[key] = {
    key, kind: "leaf", label: label || deriveLabel(summary), summary, session, parent: null,
    n_messages: messages.length, stats: {}, messages: [...messages],
  };
  save(db);
  return card(db.nodes[key]!);
}

export function localIndex(session?: string, sort = "recency"): { memories: readonly MemoryCard[]; versioned: boolean } {
  const db = load();
  const memories = sortCards(Object.values(db.nodes)
    .filter((n) => n.parent === null && (session === undefined || n.session === session))
    .map(card), sort);
  return { memories, versioned: false };
}

export function localTree(session?: string): readonly MemoryTreeNode[] {
  const db = load();
  const node = (key: string): MemoryTreeNode => {
    const n = db.nodes[key]!;
    const base = card(n) as MemoryTreeNode;
    if (n.kind === "chunked") return { ...base, children: (n.children ?? []).map(node) };
    return base;
  };
  return localIndex(session).memories.map((c) => node(c.key));
}

export function localRead(key: string): MemoryNode {
  const db = load();
  const n = db.nodes[key];
  if (!n) throw new Error(`no memory ${key}`);
  const base = card(n);
  if (n.kind === "chunked") return { ...base, children: (n.children ?? []).map((c) => card(db.nodes[c]!)) };
  return { ...base, messages: n.messages ?? [] };
}

export function localSearch(query: string, session?: string, sort = "relevance", metric?: string): MemorySearchResult {
  const db = load();
  const cards = Object.values(db.nodes)
    .filter((n) => session === undefined || n.session === session)
    .map((n) => ({ n, c: card(n) }));
  // embedding/llm metrics can't run in-browser — fall back to bm25. With no explicit metric we report
  // the plain "lexical" method (parity with the server default); an explicit metric gets its honest label.
  const useMetric = metric && (OFFLINE_METRICS as readonly string[]).includes(metric) ? metric : "bm25";
  const label = metric ? (METRIC_LABELS[useMetric] ?? "lexical-bm25") : "lexical";
  if (cards.length === 0) return { method: label, matches: [] };
  const { scores } = metricScores(query, cards.map(({ c }) => `${c.label} ${c.summary}`), useMetric);
  const method = label;
  let matches: MemoryMatch[] = cards
    .map(({ c }, i) => ({ ...c, score: Math.round(scores[i]! * 1e4) / 1e4 }))
    .filter((m) => m.score > 0);
  matches = (sort === "relevance" ? matches.sort((a, b) => b.score - a.score) : sortCards(matches, sort)).slice(0, 5);
  return { method, matches };
}

// DAG parity: priority + typed cross-links + a graph view, matching the server store.
export function localSetPriority(key: string, priority: number): MemoryCard {
  const db = load();
  const n = db.nodes[key];
  if (!n) throw new Error(`no memory ${key}`);
  n.priority = priority; save(db);
  return card(n);
}
export function localLink(src: string, dst: string, type = "relates"): { from: string; to: string; type: string } {
  if (src === dst) throw new Error("a memory cannot link to itself");
  const db = load();
  if (!db.nodes[src] || !db.nodes[dst]) throw new Error("both memories must exist");
  const links = (db.nodes[src]!.links ??= []);
  if (!links.some((e) => e.to === dst && e.type === type)) { links.push({ to: dst, type }); save(db); }
  return { from: src, to: dst, type };
}
export function localGraph(session?: string): { nodes: MemoryCard[]; edges: { from: string; to: string; type: string; kind: string }[]; n_nodes: number; n_edges: number } {
  const db = load();
  const nodes: MemoryCard[] = [];
  const edges: { from: string; to: string; type: string; kind: string }[] = [];
  for (const n of Object.values(db.nodes)) {
    if (session !== undefined && n.session !== session) continue;
    nodes.push({ ...card(n), parent: n.parent } as MemoryCard);
    for (const c of n.children ?? []) edges.push({ from: n.key, to: c, type: "child", kind: "parent" });
    for (const e of n.links ?? []) edges.push({ from: n.key, to: e.to, type: e.type, kind: "link" });
  }
  return { nodes, edges, n_nodes: nodes.length, n_edges: edges.length };
}

export function localConsolidate(keys: readonly string[], summary: string, label = ""): MemoryCard {
  const db = load();
  const ks = [...new Set(keys)];
  if (ks.length < 2) throw new Error("consolidate needs at least two memories");
  const ancestors = (k: string): string[] => {
    const chain: string[] = []; let cur: string | null = k; const seen = new Set<string>();
    while (cur && !seen.has(cur)) { seen.add(cur); chain.push(cur); cur = db.nodes[cur]?.parent ?? null; }
    return chain;
  };
  for (const k of ks) if (ancestors(k).slice(1).some((a) => ks.includes(a))) throw new Error("cannot consolidate a memory with its own ancestor");
  // lowest common ancestor (excluding the keys themselves) → placement
  let common = new Set(ancestors(ks[0]!));
  for (const k of ks.slice(1)) common = new Set(ancestors(k).filter((a) => common.has(a)));
  for (const k of ks) common.delete(k);
  const target = ancestors(ks[0]!).find((a) => common.has(a)) ?? null;
  const key = nextKey(db);
  db.nodes[key] = {
    key, kind: "chunked", label: label || deriveLabel(summary), summary,
    session: db.nodes[ks[0]!]?.session ?? "", parent: target, children: [...ks],
    n_messages: ks.reduce((s, k) => s + (db.nodes[k]?.n_messages ?? 0), 0), stats: {},
  };
  for (const k of ks) {
    const n = db.nodes[k]!; const old = n.parent; n.parent = key;
    if (old && old !== target && db.nodes[old]?.children) db.nodes[old]!.children = db.nodes[old]!.children!.filter((c) => c !== k);
  }
  if (target && db.nodes[target]) db.nodes[target]!.children = [...(db.nodes[target]!.children ?? []).filter((c) => !ks.includes(c)), key];
  save(db);
  return card(db.nodes[key]!);
}

export function localRecrystallize(key: string, subchunks: readonly { label?: string; summary: string; messages: readonly CompactMessage[] }[]):
  { key: string; children: readonly string[]; kind: string; ref: string | null } {
  const db = load();
  const n = db.nodes[key];
  if (!n) throw new Error(`no memory ${key}`);
  if (!subchunks.length) throw new Error("recrystallize needs at least one subchunk");
  const childKeys: string[] = [];
  for (const sc of subchunks) {
    const ck = nextKey(db);
    db.nodes[ck] = { key: ck, kind: "leaf", label: sc.label || deriveLabel(sc.summary), summary: sc.summary,
      session: n.session, parent: key, n_messages: sc.messages.length, stats: {}, messages: [...sc.messages] };
    childKeys.push(ck);
  }
  n.kind = "chunked"; n.children = childKeys; delete n.messages;
  n.n_messages = subchunks.reduce((s, sc) => s + sc.messages.length, 0);
  save(db);
  return { key, children: childKeys, kind: "chunked", ref: null };
}

// Split a leaf memory's messages into N contiguous parts with extractive summaries (no model
// available in-browser) — the material for a local re-crystallization.
export function localReadForSplit(key: string, chunks: number): { label: string; summary: string; messages: readonly CompactMessage[] }[] {
  const db = load();
  const n = db.nodes[key];
  const msgs = n?.messages ?? [];
  if (!n || n.kind !== "leaf" || msgs.length === 0) throw new Error("can only re-crystallize a leaf memory with messages");
  const k = Math.max(1, Math.min(chunks, msgs.length));
  const size = Math.ceil(msgs.length / k);
  const out: { label: string; summary: string; messages: readonly CompactMessage[] }[] = [];
  for (let i = 0; i < msgs.length; i += size) {
    const grp = msgs.slice(i, i + size);
    const first = (grp[0]?.content ?? "").split(/(?<=[.!?])\s/)[0]?.slice(0, 120) ?? "part";
    out.push({ label: deriveLabel(first), summary: `${first} … (${grp.length} turns)`, messages: grp });
  }
  return out;
}

// Extractive, model-free compaction for the browser: keep the system prompt + last keepRecent turns,
// summarise the older ones by pulling their leading sentences. Honest — labeled as a heuristic.
export function localCompact(messages: readonly CompactMessage[], keepRecent = 6, session = ""): CompactResult {
  const est = (ms: readonly CompactMessage[]): number => ms.reduce((s, m) => s + Math.floor((m.content ?? "").length / 4), 0);
  const system = messages.filter((m) => m.role === "system");
  const convo = messages.filter((m) => m.role !== "system");
  const recent = keepRecent > 0 ? convo.slice(-keepRecent) : [];
  const old = convo.slice(0, convo.length - recent.length);
  const before = est(messages);
  if (old.length === 0) {
    return { messages, summary: null, compacted: false,
      stats: { before_tokens: before, after_tokens: before, summarized_turns: 0, token_estimate: "heuristic (chars/4), not a tokenizer" }, tokens: before };
  }
  // extractive summary: first ~200 chars of each old turn, joined
  const summary = "Earlier conversation (extractive summary — no model available in-browser):\n" +
    old.map((m) => `${m.role}: ${(m.content ?? "").split(/(?<=[.!?])\s/)[0]?.slice(0, 200) ?? ""}`).join("\n");
  localCrystallize(old, summary, "", session);
  const summaryMsg: CompactMessage = { role: "system", content: summary };
  const out = [...system, summaryMsg, ...recent];
  return { messages: out, summary, compacted: true,
    stats: { before_tokens: before, after_tokens: est(out), summarized_turns: old.length, token_estimate: "heuristic (chars/4), not a tokenizer" }, tokens: before };
}
