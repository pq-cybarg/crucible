import type { JSX } from "react";
import { useCallbackRef } from "../useCallbackRef";
import { useMemo, useRef, useState } from "react";
import type { FormEvent, KeyboardEvent } from "react";
import { AnimatePresence, motion } from "framer-motion";
import { cancelAgent, runAgent } from "../api";
import type { AgentEvent, ChatMessage, PermissionMode } from "../api";
import { chatDirectStream, getActiveChatModel, getActiveChatService, getActiveModelId, getChatMode } from "../services";

export type Turn =
  | { readonly id: string; readonly kind: "user"; readonly text: string }
  | { readonly id: string; readonly kind: "assistant"; readonly text: string; readonly streaming?: boolean }
  | { readonly id: string; readonly kind: "notice"; readonly text: string }
  | {
      readonly id: string;
      readonly kind: "tool";
      readonly name: string;
      readonly args: Readonly<Record<string, unknown>>;
      readonly status: "running" | "ok" | "fail";
      readonly output: string;
    };

const PERMS: readonly PermissionMode[] = ["allow", "ask", "deny"];

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
    case "tool_call":
      return [
        ...turns,
        { id: event.data.id, kind: "tool", name: event.data.name, args: event.data.args, status: "running", output: "" },
      ];
    case "tool_result":
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
  const [draft, setDraft] = useState("");
  const [perm, setPerm] = useState<PermissionMode>("ask");
  const [react, setReact] = useState(false);
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

  const history = useMemo<readonly ChatMessage[]>(
    () =>
      turns.flatMap((turn): readonly ChatMessage[] =>
        turn.kind === "user" || turn.kind === "assistant"
          ? [{ role: turn.kind, content: turn.text }]
          : [],
      ),
    [turns],
  );

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
      permissions: { default: perm, modes: {} },
      onEvent: (event) => {
        if (event.type === "assistant_delta") countToken();
        setTurns((prev) => reduce(prev, event, nextId));
      },
      signal: controller.signal,
      runId,
      ...(upstream ? { upstream } : {}),
      ...(modelId ? { modelId } : {}),
      ...(react ? { react: true } : {}),
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

  return (
    <div className="console">
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
          {turns.map((turn) => (
            <motion.div
              key={turn.id}
              initial={{ opacity: 0, y: 8 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.22 }}
              className={
                turn.kind === "tool" ? "toolcard" : turn.kind === "notice" ? "notice" : `msg ${turn.kind}`
              }
            >
              {turn.kind === "tool" ? (
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
          <label className="react-toggle" title="ReAct text tool-loop — use for models without native function-calling (most small GGUF)">
            <input type="checkbox" checked={react} onChange={(e) => setReact(e.target.checked)} />
            ReAct
          </label>
        </div>
      </form>
    </div>
  );
}
