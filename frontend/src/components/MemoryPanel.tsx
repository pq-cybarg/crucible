import type { JSX } from "react";
import { useEffect, useMemo, useState } from "react";
import { motion } from "framer-motion";
import {
  consolidateMemory, getMemoryTree, readMemory, recrystallizeMemory, searchMemory,
} from "../api";
import type { MemoryMatch, MemoryNode, MemoryTreeNode } from "../api";
import { getActiveModelId } from "../services";
import MemoryMap from "./MemoryMap";

// Crystallized-memory browser. Compaction files old context into a git-versioned tree; here you
// scan summaries (the cheap passthrough), drill into a memory to read its full context, RE-
// crystallize a leaf into finer labelled subchunks, and CONSOLIDATE a set of memories under a new
// parent (siblings bubble to their shared parent; top-level sets form a new domain node).
function TreeNode({ node, depth, selected, onToggle, onOpen }: {
  readonly node: MemoryTreeNode;
  readonly depth: number;
  readonly selected: ReadonlySet<string>;
  readonly onToggle: (key: string) => void;
  readonly onOpen: (key: string) => void;
}): JSX.Element {
  return (
    <div className="mem-node" style={{ marginLeft: depth * 16 }}>
      <div className="mem-row">
        <input type="checkbox" checked={selected.has(node.key)} onChange={() => onToggle(node.key)}
          title="select for consolidation" />
        <code className="mem-key">{node.key}</code>
        <span className={`mem-kind ${node.kind}`}>{node.kind === "chunked" ? `▸ ${node.size}` : `${node.size} msg`}</span>
        <button className="mem-label" onClick={() => onOpen(node.key)} title="open this memory">{node.label}</button>
      </div>
      <div className="mem-summary">{node.summary}</div>
      {node.children?.map((c) => (
        <TreeNode key={c.key} node={c} depth={depth + 1} selected={selected} onToggle={onToggle} onOpen={onOpen} />
      ))}
    </div>
  );
}

export default function MemoryPanel(): JSX.Element {
  const [tree, setTree] = useState<readonly MemoryTreeNode[]>([]);
  const [selected, setSelected] = useState<ReadonlySet<string>>(new Set());
  const [opened, setOpened] = useState<MemoryNode | null>(null);
  const [consLabel, setConsLabel] = useState("");
  const [consSummary, setConsSummary] = useState("");
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState<string | null>(null);
  const [note, setNote] = useState<string | null>(null);
  const [view, setView] = useState<"tree" | "map">("map");
  const [searchQ, setSearchQ] = useState("");
  const [searchRes, setSearchRes] = useState<{ method: string; matches: readonly MemoryMatch[] } | null>(null);

  async function runSearch(): Promise<void> {
    const q = searchQ.trim();
    if (q.length === 0) { setSearchRes(null); return; }
    setErr(null);
    try { setSearchRes(await searchMemory(q)); }
    catch (e: unknown) { setErr(e instanceof Error ? e.message : "search failed"); }
  }

  async function refresh(): Promise<void> {
    setErr(null);
    try { setTree(await getMemoryTree()); }
    catch (e: unknown) { setErr(e instanceof Error ? e.message : "failed to load memory"); }
  }
  useEffect(() => { void refresh(); }, []);

  const count = useMemo(() => {
    let n = 0;
    const walk = (nodes: readonly MemoryTreeNode[]): void => { for (const x of nodes) { n += 1; if (x.children) walk(x.children); } };
    walk(tree);
    return n;
  }, [tree]);

  function toggle(key: string): void {
    setSelected((prev) => {
      const next = new Set(prev);
      if (next.has(key)) next.delete(key); else next.add(key);
      return next;
    });
  }

  async function open(key: string): Promise<void> {
    setErr(null);
    try { setOpened(await readMemory(key)); }
    catch (e: unknown) { setErr(e instanceof Error ? e.message : "failed"); }
  }

  async function consolidate(): Promise<void> {
    if (selected.size < 2 || consSummary.trim().length === 0) return;
    setBusy(true); setErr(null); setNote(null);
    try {
      const card = await consolidateMemory([...selected], consSummary.trim(), consLabel.trim());
      setNote(`consolidated ${selected.size} into ${card.key} (${card.label})`);
      setSelected(new Set()); setConsLabel(""); setConsSummary("");
      await refresh();
    } catch (e: unknown) { setErr(e instanceof Error ? e.message : "failed"); } finally { setBusy(false); }
  }

  async function recrystallize(key: string): Promise<void> {
    setBusy(true); setErr(null); setNote(null);
    try {
      const res = await recrystallizeMemory(key, { chunks: 3, modelId: getActiveModelId() ?? undefined });
      setNote(`re-crystallized ${key} into ${res.children.length} subchunks`);
      setOpened(null);
      await refresh();
    } catch (e: unknown) { setErr(e instanceof Error ? e.message : "failed"); } finally { setBusy(false); }
  }

  return (
    <div className="panel">
      <div className="panel-head">
        <h1>crystallized <em>memory</em></h1>
        <p>Every compaction is kept as a <b>git-versioned</b> memory — never discarded. Scan the summaries,
          open one to read its full context, <b>re-crystallize</b> a memory into finer labelled subchunks,
          or <b>consolidate</b> a set under a new parent to file domain knowledge. Agents recall these via
          the <code>recall_memory</code> tool.</p>
      </div>

      <div className="mem-toolbar">
        <button className="btn ghost" onClick={() => void refresh()}>refresh</button>
        <span className="seg">
          <button className={view === "map" ? "on" : ""} onClick={() => setView("map")}>map</button>
          <button className={view === "tree" ? "on" : ""} onClick={() => setView("tree")}>tree</button>
        </span>
        <span className="hint" style={{ margin: 0 }}>{count} memories · {selected.size} selected</span>
        {err && <span className="runtime-err">{err}</span>}
        {note && <span className="mem-note">{note}</span>}
      </div>

      <div className="mem-search">
        <input className="in" placeholder="search memories (semantic if an embed backend is set, else keyword)…"
          value={searchQ}
          onChange={(e) => setSearchQ(e.target.value)}
          onKeyDown={(e) => { if (e.key === "Enter") void runSearch(); }} />
        <button className="btn ghost" onClick={() => void runSearch()}>search</button>
        {searchRes && <button className="btn ghost" onClick={() => { setSearchQ(""); setSearchRes(null); }}>clear</button>}
      </div>
      {searchRes && (
        <div className="mem-search-results">
          <div className="engrave" style={{ margin: "4px 0" }}>
            {searchRes.matches.length} match{searchRes.matches.length === 1 ? "" : "es"} · {searchRes.method}
          </div>
          {searchRes.matches.length === 0 && <div className="hint">no matches.</div>}
          {searchRes.matches.map((m) => (
            <div key={m.key} className="mem-child" onClick={() => void open(m.key)} style={{ cursor: "pointer" }}>
              <code className="mem-key">{m.key}</code> <b className="mem-label">{m.label}</b>
              <span className="mem-kind">score {m.score}</span>
              <div className="mem-summary">{m.summary}</div>
            </div>
          ))}
        </div>
      )}

      {selected.size >= 2 && (
        <div className="mem-consolidate">
          <span className="engrave" style={{ margin: 0 }}>consolidate {selected.size} → new parent</span>
          <input className="in" placeholder="label (optional)" value={consLabel} onChange={(e) => setConsLabel(e.target.value)} />
          <input className="in" placeholder="summary for the new parent memory" value={consSummary} onChange={(e) => setConsSummary(e.target.value)} />
          <button className="btn" disabled={busy || consSummary.trim().length === 0} onClick={() => void consolidate()}>consolidate</button>
        </div>
      )}

      {view === "map" ? (
        <MemoryMap tree={tree} onOpen={(k) => void open(k)} />
      ) : (
        <div className="mem-tree">
          {tree.length === 0 && <div className="hint">no crystallized memories yet — compact a conversation in the forge to create one.</div>}
          {tree.map((n) => (
            <TreeNode key={n.key} node={n} depth={0} selected={selected} onToggle={toggle} onOpen={(k) => void open(k)} />
          ))}
        </div>
      )}

      {opened && (
        <motion.div className="mem-open" initial={{ opacity: 0, y: 8 }} animate={{ opacity: 1, y: 0 }}>
          <div className="mem-open-head">
            <code className="mem-key">{opened.key}</code> <b>{opened.label}</b>
            <span className="mem-kind">{opened.kind}</span>
            {opened.kind === "leaf" && (
              <button className="btn ghost" disabled={busy} onClick={() => void recrystallize(opened.key)}
                title="split this memory into finer labelled/summarized subchunks (uses the active model)">
                re-crystallize
              </button>
            )}
            <button className="mem-close" onClick={() => setOpened(null)}>✕</button>
          </div>
          <div className="mem-summary">{opened.summary}</div>
          {opened.children && opened.children.map((c) => (
            <div key={c.key} className="mem-child">
              <code className="mem-key">{c.key}</code> <button className="mem-label" onClick={() => void open(c.key)}>{c.label}</button>
              <div className="mem-summary">{c.summary}</div>
            </div>
          ))}
          {opened.messages && (
            <div className="mem-messages">
              {opened.messages.map((m, i) => (
                <div key={i} className="mem-msg"><span className="mem-role">{m.role}</span>{m.content}</div>
              ))}
            </div>
          )}
        </motion.div>
      )}
    </div>
  );
}
