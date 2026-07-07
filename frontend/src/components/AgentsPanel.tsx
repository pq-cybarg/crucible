import type { JSX } from "react";
import { useCallback, useEffect, useMemo, useState } from "react";
import {
  attachSlot, createAgentSession, deleteAgentSession, detachSlot, getAgentSession,
  getAgentSessionContext, getMemoryIndex, listAgentSessions, toggleSlot,
} from "../api";
import type { AgentSessionCard, AgentSessionFull, ChatMessage, MemoryCard } from "../api";
import { getActiveModelId } from "../services";

// Agent workbench: TABS, each an agent bound to a working DIRECTORY (different dirs, the same dir, or a
// SUBAGENT of another), and SLOTS — crystallized memories and other agents' contexts you can LOAD or
// UNLOAD into a tab's live context at will. The right rail browses everything loadable; the panel
// previews the assembled live context so you can see exactly what the agent will be given.
export default function AgentsPanel(): JSX.Element {
  const [sessions, setSessions] = useState<readonly AgentSessionCard[]>([]);
  const [activeId, setActiveId] = useState<string | null>(null);
  const [active, setActive] = useState<AgentSessionFull | null>(null);
  const [memories, setMemories] = useState<readonly MemoryCard[]>([]);
  const [context, setContext] = useState<readonly ChatMessage[]>([]);
  const [err, setErr] = useState<string | null>(null);

  const refreshList = useCallback(async (): Promise<void> => {
    try {
      const all = await listAgentSessions();
      setSessions(all);
      if (activeId === null && all.length > 0) setActiveId(all[0]!.id);
    } catch (e: unknown) { setErr(e instanceof Error ? e.message : "failed to load sessions"); }
  }, [activeId]);

  const refreshActive = useCallback(async (id: string): Promise<void> => {
    try {
      setActive(await getAgentSession(id));
      setContext(await getAgentSessionContext(id));
    } catch (e: unknown) { setErr(e instanceof Error ? e.message : "failed to load session"); }
  }, []);

  useEffect(() => { void refreshList(); void getMemoryIndex().then((r) => setMemories(r.memories)).catch(() => undefined); }, [refreshList]);
  useEffect(() => { if (activeId) void refreshActive(activeId); else setActive(null); }, [activeId, refreshActive]);

  // group tabs: top-level sessions, each followed by its subagents
  const tree = useMemo(() => {
    const top = sessions.filter((s) => !s.parent_id);
    const kids = (id: string): readonly AgentSessionCard[] => sessions.filter((s) => s.parent_id === id);
    return top.flatMap((t) => [{ s: t, depth: 0 }, ...kids(t.id).map((k) => ({ s: k, depth: 1 }))]);
  }, [sessions]);

  async function newSession(parentId?: string): Promise<void> {
    const title = window.prompt(parentId ? "Subagent title:" : "New agent tab — title:", parentId ? "helper" : "work");
    if (title == null) return;
    const cwd = window.prompt("Working directory for this agent:", active?.cwd ?? ".");
    if (cwd == null) return;
    try {
      const card = await createAgentSession({ title: title || "agent", cwd: cwd || ".", model_id: getActiveModelId(), parent_id: parentId ?? null });
      await refreshList(); setActiveId(card.id);
    } catch (e: unknown) { setErr(e instanceof Error ? e.message : "create failed"); }
  }
  async function closeSession(id: string): Promise<void> {
    if (!window.confirm("Close this agent tab (and any subagents)?")) return;
    try { await deleteAgentSession(id); if (activeId === id) setActiveId(null); await refreshList(); }
    catch (e: unknown) { setErr(e instanceof Error ? e.message : "close failed"); }
  }
  async function load(kind: "memory" | "context", ref: string, label: string): Promise<void> {
    if (!activeId) return;
    try { setActive(await attachSlot(activeId, kind, ref, label)); await refreshActive(activeId); await refreshList(); }
    catch (e: unknown) { setErr(e instanceof Error ? e.message : "load failed"); }
  }
  async function toggle(kind: "memory" | "context", ref: string, enabled: boolean): Promise<void> {
    if (!activeId) return;
    try { setActive(await toggleSlot(activeId, kind, ref, enabled)); await refreshActive(activeId); }
    catch (e: unknown) { setErr(e instanceof Error ? e.message : "toggle failed"); }
  }
  async function unload(kind: "memory" | "context", ref: string): Promise<void> {
    if (!activeId) return;
    try { setActive(await detachSlot(activeId, kind, ref)); await refreshActive(activeId); await refreshList(); }
    catch (e: unknown) { setErr(e instanceof Error ? e.message : "unload failed"); }
  }

  const loadedRefs = useMemo(() => new Set((active?.slots ?? []).map((s) => `${s.kind}:${s.ref}`)), [active]);
  const otherSessions = sessions.filter((s) => s.id !== activeId);

  return (
    <div className="panel">
      <div className="panel-head">
        <h1>agent <em>tabs</em></h1>
        <p>Each tab is an agent in a working <b>directory</b> (or a <b>subagent</b> of another). <b>Load</b> or
          <b> unload</b> crystallized memories and other agents' contexts into a tab's live context — the
          preview shows exactly what the agent will be given.</p>
      </div>

      <div className="agents-tabs">
        {tree.map(({ s, depth }) => (
          <div key={s.id} className={`atab ${s.id === activeId ? "on" : ""}`} style={{ marginLeft: depth * 14 }}>
            <button className="atab-name" onClick={() => setActiveId(s.id)} title={s.cwd}>
              {depth > 0 && <span className="atab-sub">↳ </span>}{s.title}
              <span className="atab-meta">{s.n_loaded}/{s.n_slots} loaded · {s.cwd}</span>
            </button>
            <button className="atab-x" title="close tab" onClick={() => void closeSession(s.id)}>✕</button>
          </div>
        ))}
        <button className="btn ghost" onClick={() => void newSession()}>+ agent</button>
        {activeId && <button className="btn ghost" onClick={() => void newSession(activeId)}>+ subagent</button>}
      </div>
      {err && <div className="runtime-err">{err}</div>}
      {sessions.length === 0 && <div className="hint">no agent tabs yet — “+ agent” opens one bound to a working directory.</div>}

      {active && (
        <div className="agents-body">
          <div className="agents-main">
            <div className="engrave">context slots · {active.title}</div>
            <div className="slot-list">
              {active.slots.length === 0 && <div className="hint">nothing loaded — load a memory or context from the right.</div>}
              {active.slots.map((sl) => (
                <div key={`${sl.kind}:${sl.ref}`} className={`slot ${sl.enabled ? "on" : "off"}`}>
                  <label className="slot-toggle" title="load / unload this slot">
                    <input type="checkbox" checked={sl.enabled} onChange={(e) => void toggle(sl.kind, sl.ref, e.target.checked)} />
                  </label>
                  <span className={`slot-kind ${sl.kind}`}>{sl.kind}</span>
                  <code className="mem-key">{sl.ref}</code>
                  <span className="slot-label">{sl.label}</span>
                  <button className="slot-x" title="remove slot" onClick={() => void unload(sl.kind, sl.ref)}>✕</button>
                </div>
              ))}
            </div>

            <div className="engrave" style={{ marginTop: 12 }}>live context preview <span className="hint" style={{ margin: 0 }}>— what the agent will be given ({context.length} msgs)</span></div>
            <div className="ctx-preview">
              {context.length === 0 && <div className="hint">empty — load slots or add conversation.</div>}
              {context.map((m, i) => (
                <div key={i} className="ctx-msg"><span className="mem-role">{m.role}</span>{m.content.slice(0, 400)}</div>
              ))}
            </div>
          </div>

          <div className="agents-browse">
            <div className="engrave">load a memory</div>
            <div className="browse-list">
              {memories.length === 0 && <div className="hint">no crystallized memories yet.</div>}
              {memories.map((m) => {
                const on = loadedRefs.has(`memory:${m.key}`);
                return (
                  <div key={m.key} className="browse-row">
                    <button className="browse-add" disabled={on} onClick={() => void load("memory", m.key, m.label)}>{on ? "loaded" : "load"}</button>
                    <code className="mem-key">{m.key}</code> <span className="slot-label">{m.label}</span>
                  </div>
                );
              })}
            </div>
            <div className="engrave" style={{ marginTop: 12 }}>load another context</div>
            <div className="browse-list">
              {otherSessions.length === 0 && <div className="hint">open another tab to load it as context.</div>}
              {otherSessions.map((s) => {
                const on = loadedRefs.has(`context:${s.id}`);
                return (
                  <div key={s.id} className="browse-row">
                    <button className="browse-add" disabled={on} onClick={() => void load("context", s.id, s.title)}>{on ? "loaded" : "load"}</button>
                    <code className="mem-key">{s.id}</code> <span className="slot-label">{s.title}</span>
                  </div>
                );
              })}
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
