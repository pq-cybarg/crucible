import type { JSX } from "react";
import { useCallbackRef } from "../useCallbackRef";
import { useEffect, useMemo, useRef, useState } from "react";
import type { FormEvent, KeyboardEvent } from "react";
import { AnimatePresence, motion } from "framer-motion";
import { approveAgent, cancelAgent, compactConversation, getModelToolSupport, getPreferences, getProfiles, runAgent } from "../api";
import type { AgentEvent, ChatMessage, CompactMessage, HierarchyProfile, PathRuleConfig, PermissionMode } from "../api";

// heuristic client-side token estimate (chars/4), mirrors the backend meter — for the UI only.
const CONTEXT_LIMIT = 4000;
function estimateTokens(msgs: readonly { readonly content: string }[]): number {
  return msgs.reduce((sum, m) => sum + Math.floor((m.content ?? "").length / 4), 0);
}
import { chatDirectStream, getActiveChatModel, getActiveChatService, getActiveModelId, getChatMode } from "../services";
import ContextExplorer from "./ContextExplorer";
import ChatAvatar from "./ChatAvatar";
import ToolFeed from "./ToolFeed";
import { toModelHistory } from "../chatHistory";
import { moodFromText } from "../avatar/sentiment";

export type Turn =
  | { readonly id: string; readonly kind: "user"; readonly text: string }
  | { readonly id: string; readonly kind: "assistant"; readonly text: string; readonly streaming?: boolean }
  | { readonly id: string; readonly kind: "notice"; readonly text: string }
  | {
      readonly id: string;
      readonly kind: "permission";
      readonly name: string;
      readonly args: Readonly<Record<string, unknown>>;
      readonly resolved?: "approved" | "denied";
    }
  | {
      readonly id: string;
      readonly kind: "tool";
      readonly name: string;
      readonly args: Readonly<Record<string, unknown>>;
      readonly status: "running" | "ok" | "fail";
      readonly output: string;
    }
  // housekeeping (memory) ops collapsed into ONE rolling summary chip, not one turn per call
  | { readonly id: string; readonly kind: "memory"; readonly counts: Readonly<Record<string, number>> };

const PERMS: readonly PermissionMode[] = ["allow", "ask", "deny"];
const MEMORY_VERB: Readonly<Record<string, string>> = {
  recall_memory: "recalled", crystallize_memory: "crystallized", recrystallize_memory: "recrystallized",
  consolidate_memory: "consolidated", link_memory: "linked", prioritize_memory: "prioritized",
};
function memorySummary(counts: Readonly<Record<string, number>>): string {
  const parts = Object.entries(counts).filter(([, n]) => n > 0).map(([k, n]) => `${MEMORY_VERB[k] ?? k} ${n}`);
  return parts.length > 0 ? parts.join(" · ") : "no changes";
}

export function reduce(turns: readonly Turn[], event: AgentEvent, nextId: () => string): readonly Turn[] {
  switch (event.type) {
    case "assistant_delta": {
      // token-level streaming: extend the open streaming turn, or open a new one
      const last = turns[turns.length - 1];
      if (last !== undefined && last.kind === "assistant" && last.streaming === true) {
        return turns.map((t) =>
          t.id === last.id && t.kind === "assistant" ? { ...t, text: t.text + event.data.delta } : t,
        );
      }
      return [...turns, { id: nextId(), kind: "assistant", text: event.data.delta, streaming: true }];
    }
    case "assistant": {
      // finalize an open streaming turn with the authoritative content; else append a new turn
      const last = turns[turns.length - 1];
      if (last !== undefined && last.kind === "assistant" && last.streaming === true) {
        return turns.map((t) =>
          t.id === last.id && t.kind === "assistant"
            ? { ...t, text: event.data.content, streaming: false }
            : t,
        );
      }
      return [...turns, { id: nextId(), kind: "assistant", text: event.data.content }];
    }
    case "permission_request":
      return [
        ...turns,
        { id: event.data.id, kind: "permission", name: event.data.name, args: event.data.args },
      ];
    case "tool_call": {
      if (event.data.quiet === true) {          // memory upkeep → fold into a single rolling summary turn
        const name = event.data.name;
        const last = turns[turns.length - 1];
        if (last !== undefined && last.kind === "memory") {
          return turns.map((t) =>
            t.id === last.id && t.kind === "memory"
              ? { ...t, counts: { ...t.counts, [name]: (t.counts[name] ?? 0) + 1 } }
              : t);
        }
        return [...turns, { id: nextId(), kind: "memory", counts: { [name]: 1 } }];
      }
      return [
        ...turns,
        { id: event.data.id, kind: "tool", name: event.data.name, args: event.data.args, status: "running", output: "" },
      ];
    }
    case "tool_result":
      if (event.data.quiet === true) return turns;   // already tallied on the tool_call
      return turns.map((turn) =>
        turn.kind === "tool" && turn.id === event.data.id
          ? {
              ...turn,
              status: event.data.ok ? "ok" : "fail",
              output: event.data.output.length > 0 ? event.data.output : (event.data.error ?? ""),
            }
          : turn,
      );
    case "done":
      return turns;
    case "error":
      return [...turns, { id: nextId(), kind: "notice", text: `forge halted — ${event.data.reason}` }];
  }
}

export default function AgentConsole(): JSX.Element {
  const [turns, setTurns] = useState<readonly Turn[]>([]);
  // The in-chat companion reacts to the mood of each finished reply (client-side sentiment, context-free) —
  // a temporary emotional beat (heart/star eyes on strong moments) that decays back to neutral.
  const [chatMood, setChatMood] = useState<Record<string, number>>({ neutral: 1 });
  const moodTimer = useRef<number | undefined>(undefined);
  const lastMoodId = useRef<string>("");
  const [draft, setDraft] = useState("");
  const [perm, setPerm] = useState<PermissionMode>("ask");
  const [react, setReact] = useState(false);
  // Auto-detected: does the active model support native tool-calling? false → we quietly switch on
  // compatibility mode and tell the user in plain words (no "ReAct" jargon).
  const [toolSupport, setToolSupport] = useState<boolean | null>(null);
  const [autoCompact, setAutoCompact] = useState(false);
  const [compacting, setCompacting] = useState(false);
  const [profile, setProfile] = useState("");
  const [profiles, setProfiles] = useState<readonly HierarchyProfile[]>([]);
  // Tool-permission defaults come from the Preferences panel: seed the per-tool modes + path rules,
  // and adopt the saved default mode so the forge honors what the user configured centrally.
  const [permModes, setPermModes] = useState<Readonly<Record<string, PermissionMode>>>({});
  const [pathRules, setPathRules] = useState<readonly PathRuleConfig[]>([]);
  useEffect(() => { getProfiles().then(setProfiles).catch(() => undefined); }, []);
  useEffect(() => { void getPreferences().then((p) => {
    setPerm(p.preferences.permissions.default);
    setPermModes(p.preferences.permissions.modes);
    setPathRules(p.preferences.permissions.path_rules ?? []);
  }).catch(() => undefined); }, []);
  // Watch the active model; when it can't do native tool use, quietly turn on compatibility mode.
  useEffect(() => {
    let last = "";
    const check = (): void => {
      const id = getActiveModelId();
      if (id && id !== last) {
        last = id;
        void getModelToolSupport(id).then((s) => { setToolSupport(s); if (s === false) setReact(true); });
      }
    };
    check();
    const h = window.setInterval(check, 3000);
    return () => window.clearInterval(h);
  }, []);
  const [busy, setBusy] = useState(false);
  const counter = useRef(0);
  const abortRef = useRef<AbortController | null>(null);
  const runIdRef = useRef<string | null>(null);
  const [liveRate, setLiveRate] = useState<number | null>(null);
  const tokCount = useRef(0);
  const tokStart = useRef(0);

  function countToken(): void {
    if (tokCount.current === 0) tokStart.current = performance.now();
    tokCount.current += 1;
    const elapsed = (performance.now() - tokStart.current) / 1000;
    if (elapsed > 0) setLiveRate(tokCount.current / elapsed);
  }
  function resetRate(): void {
    tokCount.current = 0;
    tokStart.current = 0;
    setLiveRate(null);
  }
  const nextId = useCallbackRef(() => {
    counter.current += 1;
    return `t${counter.current}`;
  });

  // close any open streaming turn (used when the operator stops a run)
  function finalizeStreaming(): void {
    setTurns((prev) =>
      prev.map((t) => (t.kind === "assistant" && t.streaming === true ? { ...t, streaming: false } : t)),
    );
  }

  function stop(): void {
    if (runIdRef.current) void cancelAgent(runIdRef.current);   // halt generation server-side
    abortRef.current?.abort();                                  // and stop the client stream
  }

  function decide(callId: string, approved: boolean): void {
    if (runIdRef.current) void approveAgent(runIdRef.current, callId, approved);
    setTurns((prev) =>
      prev.map((t) =>
        t.kind === "permission" && t.id === callId ? { ...t, resolved: approved ? "approved" : "denied" } : t,
      ),
    );
  }

  // The model context = ONLY the human<->model dialogue. toModelHistory enforces that tool-use, notices,
  // and all avatar ANIMATION state stay OUT of the context window (animation context != chat context, #31).
  const history = useMemo<readonly ChatMessage[]>(() => toModelHistory(turns), [turns]);

  const estTokens = useMemo(() => estimateTokens(history), [history]);
  const [copied, setCopied] = useState(false);

  function transcriptText(): string {
    return history.map((m) => `## ${m.role}\n\n${m.content}`).join("\n\n");
  }
  // Copy the full context to the clipboard — grab it BEFORE compacting.
  async function copyTranscript(): Promise<void> {
    try {
      await navigator.clipboard.writeText(transcriptText());
      setCopied(true);
      setTimeout(() => setCopied(false), 1500);
    } catch { /* clipboard blocked — export instead */ }
  }
  // Download the full context as a markdown file (nothing leaves the browser).
  function exportTranscript(): void {
    const blob = new Blob([transcriptText()], { type: "text/markdown" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `crucible-context-${history.length}turns.md`;
    a.click();
    URL.revokeObjectURL(url);
  }

  // "compact now": summarize the old turns server-side, then rebuild the transcript from the
  // returned messages (the summary lands as a notice turn; recent turns stay verbatim).
  async function compactNow(): Promise<void> {
    if (compacting || busy || history.length < 4) return;
    setCompacting(true);
    try {
      const _mid = getActiveModelId();
      const res = await compactConversation(history as readonly CompactMessage[], {
        force: true, keepRecent: 6, ...(_mid ? { modelId: _mid } : {}),
      });
      if (!res.compacted) return;
      const rebuilt: Turn[] = res.messages.flatMap((m): Turn[] => {
        if (m.role === "user") return [{ id: nextId(), kind: "user", text: m.content }];
        if (m.role === "assistant") return [{ id: nextId(), kind: "assistant", text: m.content }];
        return [{ id: nextId(), kind: "notice", text: `context compacted · ${res.stats.summarized_turns} old turns summarized (${res.stats.before_tokens}→${res.stats.after_tokens} tok)` }];
      });
      setTurns(rebuilt);
    } catch (e: unknown) {
      setTurns((prev) => [...prev, { id: nextId(), kind: "notice", text: `compaction failed — ${e instanceof Error ? e.message : "error"}` }]);
    } finally {
      setCompacting(false);
    }
  }

  async function send(): Promise<void> {
    const text = draft.trim();
    if (text.length === 0 || busy) return;
    const userTurn: Turn = { id: nextId(), kind: "user", text };
    const messages: readonly ChatMessage[] = [...history, { role: "user", content: text }];
    setTurns((prev) => [...prev, userTurn]);
    setDraft("");
    setBusy(true);
    const controller = new AbortController();
    abortRef.current = controller;
    resetRate();
    const runId = `run-${counter.current}-${Math.floor(performance.now())}`;
    runIdRef.current = runId;
    const aborted = (): boolean => controller.signal.aborted;

    // BYO-AI: a non-Crucible chat backend can be driven two ways.
    const byo = getActiveChatService();
    const mode = byo && !byo.full ? getChatMode() : null;
    const chosenModel = getActiveChatModel() ?? undefined;

    // "direct": browser → service /v1, plain chat (no tool loop). Works from the static page.
    if (byo && mode === "direct") {
      setTurns((prev) => [
        ...prev,
        {
          id: nextId(),
          kind: "notice",
          text: `chat → ${byo.name} (${byo.baseUrl})${chosenModel ? ` · ${chosenModel}` : ""} · direct, no tool loop`,
        },
      ]);
      try {
        const reply = await chatDirectStream(
          byo,
          messages,
          (delta) => { countToken(); setTurns((prev) => reduce(prev, { type: "assistant_delta", data: { delta } }, nextId)); },
          chosenModel,
          512,
          controller.signal,
        );
        // finalize the streamed turn with the authoritative text (or a fallback notice)
        setTurns((prev) =>
          reduce(prev, { type: "assistant", data: { content: reply || "(empty reply)", streamed: true } }, nextId),
        );
        if (aborted()) {
          setTurns((prev) => [...prev, { id: nextId(), kind: "notice", text: "stopped by operator" }]);
        }
      } catch (err: unknown) {
        finalizeStreaming();
        const why = err instanceof Error ? err.message : "request failed";
        setTurns((prev) => [
          ...prev,
          {
            id: nextId(),
            kind: "notice",
            text: `${byo.name} unreachable — ${why}. For browser access Ollama needs OLLAMA_ORIGINS set.`,
          },
        ]);
      }
      abortRef.current = null;
      runIdRef.current = null;
      setBusy(false);
      return;
    }

    // "tools": browser → Crucible backend → service. Full agent tool-loop, Crucible runs the tools.
    const byoTools = byo && mode === "tools" ? byo : null;
    const upstream = byoTools
      ? { endpoint: byoTools.baseUrl, model: chosenModel ?? byoTools.models[0] ?? "local" }
      : undefined;
    if (byoTools) {
      setTurns((prev) => [
        ...prev,
        { id: nextId(), kind: "notice", text: `forge → ${byoTools.name} via Crucible · full tool loop` },
      ]);
    }

    // a registry model selected in the Models tab (mutually exclusive with a BYO service)
    const modelId = !byo ? getActiveModelId() : null;
    if (modelId) {
      setTurns((prev) => [
        ...prev,
        { id: nextId(), kind: "notice", text: `forge → model "${modelId}" (registry)` },
      ]);
    }

    const status = await runAgent({
      messages,
      permissions: { default: perm, modes: permModes, path_rules: pathRules },
      onEvent: (event) => {
        if (event.type === "assistant_delta") countToken();
        setTurns((prev) => reduce(prev, event, nextId));
      },
      signal: controller.signal,
      runId,
      ...(upstream ? { upstream } : {}),
      ...(modelId ? { modelId } : {}),
      ...(react ? { react: true } : {}),
      ...(autoCompact ? { autoCompact: true, contextLimit: CONTEXT_LIMIT } : {}),
      ...(profile ? { profile } : {}),
    });
    if (aborted()) {
      finalizeStreaming();
      setTurns((prev) => [...prev, { id: nextId(), kind: "notice", text: "stopped by operator" }]);
    } else if (status === "no-model") {
      setTurns((prev) => [
        ...prev,
        { id: nextId(), kind: "notice", text: "no inference node is loaded — bring the forge online to get a reply" },
      ]);
    } else if (status === "offline") {
      setTurns((prev) => [
        ...prev,
        { id: nextId(), kind: "notice", text: "backend offline — start the Crucible API on :8400" },
      ]);
    }
    abortRef.current = null;
    setBusy(false);
  }

  function onSubmit(e: FormEvent): void {
    e.preventDefault();
    void send();
  }

  function onKey(e: KeyboardEvent<HTMLTextAreaElement>): void {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      void send();
    }
  }

  // React the companion to each FINISHED reply once: map its text → a mood, hold it ~4s, then ease back to
  // neutral. Only the latest finalized assistant turn triggers (guarded by id), so it's a beat, not a loop.
  useEffect(() => {
    const last = [...turns].reverse().find((t) => t.kind === "assistant");
    if (!last || last.kind !== "assistant" || last.streaming === true || last.id === lastMoodId.current) return;
    lastMoodId.current = last.id;
    const hit = moodFromText(last.text);
    if (!hit) return;
    setChatMood({ [hit.mood]: hit.weight });
    window.clearTimeout(moodTimer.current);
    moodTimer.current = window.setTimeout(() => setChatMood({ neutral: 1 }), 4200);
  }, [turns]);
  useEffect(() => () => window.clearTimeout(moodTimer.current), []);

  const lastTurn = turns[turns.length - 1];
  const streaming = lastTurn?.kind === "assistant" && lastTurn.streaming === true;
  // Tool-use is relegated to a compact feed UNDER the companion so it no longer floods the thread; the
  // thread stays the human↔model dialogue (+ interactive permission cards, notices, memory tallies).
  const toolItems = turns.filter((t) => t.kind === "tool");
  const mainTurns = turns.filter((t) => t.kind !== "tool");
  return (
    <div className="console">
      {/* the companion + her tool-activity feed — she talks while the agent streams, idles otherwise */}
      <div className="chat-avatar-dock">
        <ChatAvatar talking={streaming || busy} mood={chatMood} size={112} />
        <ToolFeed items={toolItems} />
      </div>
      <div className="thread">
        <AnimatePresence initial={false}>
          {turns.length === 0 && (
            <motion.div
              key="cold"
              className="notice"
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
            >
              the forge is lit — issue an instruction and the harness will drive the model with tools
            </motion.div>
          )}
          {mainTurns.map((turn) => (
            <motion.div
              key={turn.id}
              initial={{ opacity: 0, y: 8 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.22 }}
              className={
                turn.kind === "tool" ? "toolcard" : turn.kind === "permission" ? "permcard"
                  : turn.kind === "notice" ? "notice" : turn.kind === "memory" ? "memcard" : `msg ${turn.kind}`
              }
            >
              {turn.kind === "permission" ? (
                <>
                  <div className="perm-req">
                    <span className="perm-name">approve <b>{turn.name}</b>?</span>
                    <code style={{ color: "var(--ash)", fontSize: 11 }}>{JSON.stringify(turn.args)}</code>
                  </div>
                  {turn.resolved ? (
                    <span className={`perm-done ${turn.resolved}`}>{turn.resolved}</span>
                  ) : (
                    <div className="perm-actions">
                      <button className="btn perm-yes" onClick={() => decide(turn.id, true)}>approve</button>
                      <button className="btn perm-no" onClick={() => decide(turn.id, false)}>deny</button>
                    </div>
                  )}
                </>
              ) : turn.kind === "tool" ? (
                <>
                  <div className="tc-head">
                    <span className="tc-name">{turn.name}</span>
                    <code style={{ color: "var(--ash)", fontSize: 11 }}>{JSON.stringify(turn.args)}</code>
                    <span className={`tc-tag ${turn.status === "ok" ? "ok" : turn.status === "fail" ? "fail" : ""}`}>
                      {turn.status}
                    </span>
                  </div>
                  {turn.output.length > 0 && <pre>{turn.output}</pre>}
                </>
              ) : turn.kind === "notice" ? (
                turn.text
              ) : turn.kind === "memory" ? (
                <span className="mem-summary">🧠 memory · {memorySummary(turn.counts)}</span>
              ) : (
                <>
                  <div className="who">{turn.kind === "user" ? "operator" : "model"}</div>
                  <div className="bubble">
                    {turn.text}
                    {turn.kind === "assistant" && turn.streaming === true && (
                      <span className="caret" aria-hidden="true" />
                    )}
                  </div>
                </>
              )}
            </motion.div>
          ))}
        </AnimatePresence>
      </div>

      <ContextExplorer turns={history} limit={CONTEXT_LIMIT} />
      <form className="composer" onSubmit={onSubmit}>
        <div className="row">
          <textarea
            value={draft}
            placeholder="instruct the forge…  (Enter to send · Shift+Enter for newline)"
            onChange={(e) => setDraft(e.target.value)}
            onKeyDown={onKey}
          />
          {liveRate !== null && (
            <span className="tokrate" title="live generation throughput (≈ tokens/second)">
              {liveRate.toFixed(1)} tok/s
            </span>
          )}
          {busy ? (
            <button type="button" className="btn stop" onClick={stop} title="abort this run">
              stop
            </button>
          ) : (
            <button type="submit" className="btn" disabled={draft.trim().length === 0}>
              send
            </button>
          )}
        </div>
        <div className="perm">
          <span>tool permission</span>
          <span className="seg">
            {PERMS.map((mode) => (
              <button
                type="button"
                key={mode}
                className={perm === mode ? "on" : ""}
                onClick={() => setPerm(mode)}
              >
                {mode}
              </button>
            ))}
          </span>
          <label className="react-toggle" title="Some models can't use tools natively. Turn this on to let them use tools through a text-based workaround. The forge switches it on automatically when it detects a model needs it.">
            <input type="checkbox" checked={react} onChange={(e) => setReact(e.target.checked)} />
            tool-use compatibility{toolSupport === false ? " (auto)" : ""}
          </label>
          {toolSupport === false && (
            <span className="compat-hint" title="This model can't call tools natively; compatibility mode lets it use them via a text protocol.">
              this model needs compatibility mode for tools — turned on for you
            </span>
          )}
          <span className="ctx-controls">
            <span className={`ctx-meter ${estTokens > CONTEXT_LIMIT ? "over" : ""}`}
              title={`estimated context ≈ ${estTokens} tokens (heuristic, chars/4). Limit ${CONTEXT_LIMIT}.`}>
              ~{estTokens} tok
            </span>
            <button type="button" className="btn ctx-compact" disabled={history.length === 0}
              onClick={() => void copyTranscript()} title="copy the full context to the clipboard (before compacting)">
              {copied ? "copied ✓" : "copy"}
            </button>
            <button type="button" className="btn ctx-compact" disabled={history.length === 0}
              onClick={exportTranscript} title="download the full context as markdown (before compacting)">
              export
            </button>
            <button type="button" className="btn ctx-compact" disabled={compacting || busy || history.length < 4}
              onClick={() => void compactNow()}
              title="summarize the old turns and keep the recent ones — frees context (kept as versioned memory)">
              {compacting ? "compacting…" : "compact"}
            </button>
            <label className="react-toggle" title={`When on, the forge auto-summarizes old turns before a run once the context passes ~${CONTEXT_LIMIT} tokens.`}>
              <input type="checkbox" checked={autoCompact} onChange={(e) => setAutoCompact(e.target.checked)} />
              auto
            </label>
            {profiles.length > 0 && (
              <select className="byo-modelsel" value={profile} onChange={(e) => setProfile(e.target.value)}
                title="agent hierarchy profile — per-layer worker + communicator models for spawned sub-agents">
                <option value="">no hierarchy</option>
                {profiles.map((p) => <option key={p.name} value={p.name}>{p.name}</option>)}
              </select>
            )}
          </span>
        </div>
      </form>
    </div>
  );
}
