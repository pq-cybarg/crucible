// Configurable backend base — point the GUI at a remote Crucible node (e.g. the
// Windows box hosting the 1.5TB model). Empty string = same-origin (vite proxy).
const _stored = typeof localStorage !== "undefined" && typeof localStorage.getItem === "function" ? localStorage.getItem("crucible_api_base") : null;
export let API_BASE = _stored ?? "";
export function setApiBase(url: string): void {
  API_BASE = url.replace(/\/$/, "");
  if (typeof localStorage !== "undefined" && typeof localStorage.getItem === "function") localStorage.setItem("crucible_api_base", API_BASE);
}
export function getApiBase(): string { return API_BASE; }

const _tok = typeof localStorage !== "undefined" && typeof localStorage.getItem === "function" ? localStorage.getItem("crucible_api_token") : null;
export let API_TOKEN = _tok ?? "";
export function setApiToken(t: string): void {
  API_TOKEN = t;
  if (typeof localStorage !== "undefined" && typeof localStorage.getItem === "function") localStorage.setItem("crucible_api_token", t);
}
export function getApiToken(): string { return API_TOKEN; }
function withAuth(init?: RequestInit): RequestInit {
  if (!API_TOKEN) return init ?? {};
  const headers = { ...(init?.headers as Record<string, string> | undefined), Authorization: `Bearer ${API_TOKEN}` };
  return { ...init, headers };
}

import { demoRespond, isDemo } from "./demo";
import {
  localCompact, localConsolidate, localGraph, localIndex, localLink, localMigrate, localReadForSplit, localRead,
  localRecrystallize, localSearch, localSetPriority, localTree, METRIC_LABELS, OFFLINE_METRICS,
} from "./localMemory";
import { localContextDelete, localContextIndex, localContextRead } from "./localContext";
import { runLocalMcEval, SAMPLE_NOTE } from "./localEval";
import { DEMO_AVATAR, DEMO_REACTION, demoRigFrame } from "./avatar/demoRig";
import {
  avatarInfoP, rigFrameP,
  abliterateOutP, autotuneReportP, benchScoreP, benchmarkResultP, benchmarksInfoP,
  diagnosisReportP, editHistoryP, featureCardP, flowReportP, guardrailConfigP, guardrailResultP,
  hierarchyProfileP, lineageP, profilesP,
  heatmapReportP, hhItemsWrapP, lmEvalWrapP, manualReportP, modelRowsP, presetsP, probeWrapP,
  compactResultP, contextIndexP, contextNodeP, graphResultP, mediaStatusP, modalityDirectionP, memoryCardP, memoryIndexP,
  memoryGraphP, memoryNodeP, memorySearchP, memoryTreeP, metricsCatalogP, preferencesP2, preferencesResultP,
  biasReportP, publishedPayloadP, purgeReportP, recipesP, recrystallizeResultP, runtimeSteerReportP, saePurgeReportP, steerFeaturesReportP,
  runtimeStatusP, startResultP, statusWrapP, suiteP, sweepReportP, systemPromptPresetP,
  verifyReportP, weightsViewP,
  studioScanP, studioPreviewP, studioApplyP, reliabilityProfileP, calibrationP, crossModelP, corpusInfoP, evidenceP,
  studioSteerP, studioMapP,
} from "./schemas";

// Sort + metric name lists for the offline/preferences fallbacks (mirror the backend registries).
export const SORT_NAMES = ["relevance", "priority", "size", "degree", "recency", "oldest", "label", "balanced"] as const;
export const METRIC_NAMES = ["bm25", "jaccard", "dice", "overlap", "tfidf", "edit", "embedding", "llm"] as const;
async function cfetch(input: string, init?: RequestInit): Promise<Response> {
  if (isDemo()) {
    const path = input.startsWith(API_BASE) ? input.slice(API_BASE.length) : input;
    const r = demoRespond(path, init);
    if (r) return r;
  }
  return fetch(input, withAuth(init));
}

// Thin, fully-typed client for the Crucible backend. The agent stream is a
// discriminated union so every consumer narrows event payloads exhaustively.
// SSE is read over fetch(POST) because EventSource cannot send a request body.

export type ModelKind = "base" | "abliterated" | "steered";
export type Stage = "input" | "output";
export type PermissionMode = "allow" | "ask" | "deny";
export type FilterMode = "block" | "redact";

export interface ModelRow {
  readonly id: string;
  readonly name: string;
  readonly base_id: string | null;
  readonly path: string;
  readonly quant: string;
  readonly kind: ModelKind;
  readonly endpoint: string | null;
  readonly created: string;
  readonly notes: string;
}

export type ChatMessage = { readonly role: "user" | "assistant"; readonly content: string };

// Path-scoped rule: allow/ask/deny a tool for specific files/directories (e.g. deny ~/.ssh/**).
// Empty `tools` = every path-taking tool. Evaluated firewall-style, first match wins; deny is decisive.
export interface PathRuleConfig {
  readonly glob: string;
  readonly mode: PermissionMode;
  readonly tools: readonly string[];
}

export interface PermissionConfig {
  readonly default: PermissionMode;
  readonly modes: Readonly<Record<string, PermissionMode>>;
  readonly path_rules?: readonly PathRuleConfig[];
}

export interface SystemPromptPreset {
  readonly id: string;
  readonly name: string;
  readonly intensity: number;
  readonly system_prompt: string;
}

export interface RegexRule {
  readonly pattern: string;
  readonly mode: FilterMode;
  readonly label: string;
  readonly stages: readonly Stage[];
}

export interface GuardrailConfig {
  readonly enabled: boolean;
  readonly preset_id: string;
  readonly regex_rules: readonly RegexRule[];
  readonly constitution: string;
  readonly constitution_enabled: boolean;
}

export interface GuardrailAction {
  readonly layer: string;
  readonly stage: Stage;
  readonly action: "inject" | "block" | "redact" | "revise" | "pass";
  readonly detail: string;
}

export interface GuardrailResult {
  readonly text: string;
  readonly blocked: boolean;
  readonly actions: readonly GuardrailAction[];
}

export type AgentEvent =
  | { readonly type: "assistant"; readonly data: { readonly content: string; readonly streamed?: boolean } }
  | { readonly type: "assistant_delta"; readonly data: { readonly delta: string } }
  | { readonly type: "permission_request"; readonly data: { readonly id: string; readonly name: string; readonly args: Readonly<Record<string, unknown>> } }
  | { readonly type: "tool_call"; readonly data: { readonly id: string; readonly name: string; readonly args: Readonly<Record<string, unknown>>; readonly quiet?: boolean } }
  | { readonly type: "tool_result"; readonly data: { readonly id: string; readonly name: string; readonly ok: boolean; readonly output: string; readonly error: string | null; readonly quiet?: boolean } }
  | { readonly type: "done"; readonly data: { readonly content: string } }
  | { readonly type: "error"; readonly data: { readonly reason: string } };

export type RunStatus = "ok" | "no-model" | "offline";

const EVENT_TYPES: ReadonlySet<string> = new Set(["assistant", "assistant_delta", "permission_request", "tool_call", "tool_result", "done", "error"]);

// Approve or deny a pending 'ask' tool call (unblocks the waiting run server-side).
export async function approveAgent(runId: string, callId: string, approved: boolean): Promise<void> {
  try {
    await cfetch(API_BASE + "/api/agent/approve", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ run_id: runId, call_id: callId, approved }),
    });
  } catch { /* best-effort */ }
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null;
}

export function parseEvent(raw: string): AgentEvent | null {
  let parsed: unknown;
  try {
    parsed = JSON.parse(raw);
  } catch {
    return null;
  }
  if (!isRecord(parsed)) return null;
  const type = parsed["type"];
  const data = parsed["data"];
  if (typeof type !== "string" || !EVENT_TYPES.has(type) || !isRecord(data)) return null;
  return { type, data } as AgentEvent;
}

export async function getHealth(): Promise<boolean> {
  try {
    const r = await cfetch(API_BASE + "/api/health");
    if (!r.ok) return false;
    const body: unknown = await r.json();
    return isRecord(body) && body["ok"] === true;
  } catch {
    return false;
  }
}

export async function getModels(): Promise<readonly ModelRow[]> {
  const r = await cfetch(API_BASE + "/api/models");
  if (!r.ok) throw new Error(`GET /api/models -> ${r.status}`);
  return modelRowsP(await r.json());
}

// Canonical agent tool names — used to offer per-tool permission overrides. Fetched from the tool
// catalog when a backend is reachable, else the built-in list (keeps the Preferences UI working offline).
export const BUILTIN_TOOLS = [
  "read_file", "write_file", "edit_file", "multi_edit", "list_dir", "glob", "grep", "bash",
  "web_fetch", "web_search", "todo_write", "generate_image", "transcribe_audio",
  "recall_memory", "crystallize_memory", "recrystallize_memory", "consolidate_memory",
  "link_memory", "prioritize_memory",
] as const;
// Tools that take a filesystem path — path rules meaningfully apply to these.
export const PATH_TOOLS = ["read_file", "write_file", "edit_file", "multi_edit", "list_dir", "glob", "grep", "bash"] as const;
// --- live agent sessions: tabs (dirs / subagents) + loadable memory/context slots ---------------
export interface AgentSlot { readonly kind: "memory" | "context"; readonly ref: string; readonly label: string; readonly enabled: boolean }
export interface AgentSessionCard {
  readonly id: string; readonly title: string; readonly cwd: string; readonly model_id: string | null;
  readonly parent_id: string | null; readonly status: string; readonly created: string;
  readonly updated: string; readonly n_messages: number; readonly n_slots: number; readonly n_loaded: number;
}
export interface AgentSessionFull extends AgentSessionCard {
  readonly messages: readonly ChatMessage[]; readonly slots: readonly AgentSlot[];
}
async function jbody(r: Response, what: string): Promise<unknown> {
  if (!r.ok) throw new Error(`${what} ${r.status}`);
  return r.json();
}
export async function listAgentSessions(opts?: { top?: boolean; parent?: string }): Promise<readonly AgentSessionCard[]> {
  if (isDemo()) return [];
  const q = opts?.top ? "?top=true" : opts?.parent ? `?parent=${encodeURIComponent(opts.parent)}` : "";
  try { const b = await jbody(await cfetch(API_BASE + "/api/agent-sessions" + q), "sessions") as { sessions: AgentSessionCard[] }; return b.sessions; }
  catch { return []; }
}
export async function createAgentSession(body: { title: string; cwd: string; model_id?: string | null; parent_id?: string | null }): Promise<AgentSessionCard> {
  const r = await cfetch(API_BASE + "/api/agent-sessions", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
  return await jbody(r, "create session") as AgentSessionCard;
}
export async function getAgentSession(id: string): Promise<AgentSessionFull> {
  return await jbody(await cfetch(`${API_BASE}/api/agent-sessions/${encodeURIComponent(id)}`), "get session") as AgentSessionFull;
}
export async function updateAgentSession(id: string, fields: Record<string, unknown>): Promise<AgentSessionCard> {
  const r = await cfetch(`${API_BASE}/api/agent-sessions/${encodeURIComponent(id)}`, { method: "PATCH", headers: { "Content-Type": "application/json" }, body: JSON.stringify(fields) });
  return await jbody(r, "update session") as AgentSessionCard;
}
export async function deleteAgentSession(id: string): Promise<void> {
  const r = await cfetch(`${API_BASE}/api/agent-sessions/${encodeURIComponent(id)}`, { method: "DELETE" });
  if (!r.ok) throw new Error(`delete session ${r.status}`);
}
export async function getAgentSessionContext(id: string): Promise<readonly ChatMessage[]> {
  const b = await jbody(await cfetch(`${API_BASE}/api/agent-sessions/${encodeURIComponent(id)}/context`), "context") as { messages: ChatMessage[] };
  return b.messages;
}
// Run a tab's agent in its working directory with its assembled (slotted) context; stream SSE events.
export async function runAgentSession(id: string, message: string,
                                     onEvent: (ev: { type: string; data: Record<string, unknown> }) => void,
                                     runId?: string, signal?: AbortSignal): Promise<void> {
  const r = await cfetch(`${API_BASE}/api/agent-sessions/${encodeURIComponent(id)}/run`, {
    method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ message, ...(runId ? { run_id: runId } : {}) }),
    ...(signal ? { signal } : {}),
  });
  if (!r.ok || !r.body) throw new Error(`run ${r.status}`);
  const reader = r.body.getReader();
  const dec = new TextDecoder();
  let buf = "";
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    buf += dec.decode(value, { stream: true });
    const parts = buf.split("\n\n");
    buf = parts.pop() ?? "";
    for (const p of parts) {
      const line = p.trim();
      if (!line.startsWith("data:")) continue;
      try { onEvent(JSON.parse(line.slice(5)) as { type: string; data: Record<string, unknown> }); } catch { /* skip */ }
    }
  }
}

export async function attachSlot(id: string, kind: "memory" | "context", ref: string, label = ""): Promise<AgentSessionFull> {
  const r = await cfetch(`${API_BASE}/api/agent-sessions/${encodeURIComponent(id)}/slots`, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ kind, ref, label }) });
  return await jbody(r, "attach slot") as AgentSessionFull;
}
export async function toggleSlot(id: string, kind: "memory" | "context", ref: string, enabled: boolean): Promise<AgentSessionFull> {
  const r = await cfetch(`${API_BASE}/api/agent-sessions/${encodeURIComponent(id)}/slots`, { method: "PATCH", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ kind, ref, enabled }) });
  return await jbody(r, "toggle slot") as AgentSessionFull;
}
export async function detachSlot(id: string, kind: "memory" | "context", ref: string): Promise<AgentSessionFull> {
  const r = await cfetch(`${API_BASE}/api/agent-sessions/${encodeURIComponent(id)}/slots?kind=${kind}&ref=${encodeURIComponent(ref)}`, { method: "DELETE" });
  return await jbody(r, "detach slot") as AgentSessionFull;
}

// Co-watch: stream commentary from the vision model while a video plays. onEvent gets {type, data}.
export async function cowatchStream(source: string, interval: number, question: string,
                                   onEvent: (ev: { type: string; data: Record<string, unknown> }) => void,
                                   signal?: AbortSignal): Promise<void> {
  const r = await cfetch(`${API_BASE}/api/vision/cowatch`, {
    method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ source, interval, question }),
    ...(signal ? { signal } : {}),
  });
  if (!r.ok || !r.body) throw new Error(`cowatch ${r.status}`);
  const reader = r.body.getReader();
  const dec = new TextDecoder();
  let buf = "";
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    buf += dec.decode(value, { stream: true });
    const parts = buf.split("\n\n");
    buf = parts.pop() ?? "";
    for (const p of parts) {
      const line = p.trim();
      if (!line.startsWith("data:")) continue;
      try { onEvent(JSON.parse(line.slice(5)) as { type: string; data: Record<string, unknown> }); } catch { /* skip */ }
    }
  }
}

// Forget a dead/experiment model registry entry (does NOT delete weight files on disk).
export async function forgetModel(modelId: string): Promise<void> {
  const r = await cfetch(`${API_BASE}/api/models/${encodeURIComponent(modelId)}`, { method: "DELETE" });
  if (!r.ok) throw new Error(`forget model ${r.status}`);
}
// Re-point a model at a live endpoint — re-enable one whose server moved (e.g. aim it at Ollama).
export async function repointModel(modelId: string, endpoint: string): Promise<void> {
  const r = await cfetch(`${API_BASE}/api/models/${encodeURIComponent(modelId)}/endpoint`, {
    method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ endpoint }),
  });
  if (!r.ok) throw new Error(`repoint model ${r.status}`);
}

// Does a model support NATIVE tool-calling? null = unknown/offline. The forge uses this to auto-turn-on
// compatibility mode and explain it in plain language.
export async function getModelToolSupport(modelId: string): Promise<boolean | null> {
  if (isDemo()) return null;
  try {
    const r = await cfetch(`${API_BASE}/api/models/${encodeURIComponent(modelId)}/tool-support`);
    if (!r.ok) return null;
    const body = await r.json() as { supports_tools?: boolean | null };
    return body.supports_tools ?? null;
  } catch { return null; }
}

export async function getToolNames(): Promise<readonly string[]> {
  if (isDemo()) return [...BUILTIN_TOOLS];
  try {
    const r = await cfetch(API_BASE + "/api/tools");
    if (!r.ok) throw new Error(`tools ${r.status}`);
    const body = await r.json() as { tools?: readonly { function?: { name?: string } }[] };
    const names = (body.tools ?? []).map((t) => t.function?.name).filter((n): n is string => typeof n === "string");
    return names.length ? names : [...BUILTIN_TOOLS];
  } catch { return [...BUILTIN_TOOLS]; }
}

export async function getPresets(): Promise<readonly SystemPromptPreset[]> {
  const r = await cfetch(API_BASE + "/api/guardrails/presets");
  if (!r.ok) throw new Error(`GET /api/guardrails/presets -> ${r.status}`);
  return presetsP(await r.json());
}

export async function getGuardrailConfig(): Promise<GuardrailConfig> {
  const r = await cfetch(API_BASE + "/api/guardrails/config");
  if (!r.ok) throw new Error(`GET /api/guardrails/config -> ${r.status}`);
  return guardrailConfigP(await r.json());
}

export async function putGuardrailConfig(config: GuardrailConfig): Promise<GuardrailConfig> {
  const r = await cfetch(API_BASE + "/api/guardrails/config", {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(config),
  });
  if (!r.ok) throw new Error(`PUT /api/guardrails/config -> ${r.status}`);
  return guardrailConfigP(await r.json());
}

export async function previewGuardrail(stage: Stage, text: string, config: GuardrailConfig): Promise<GuardrailResult> {
  const r = await cfetch(API_BASE + "/api/guardrails/apply", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ stage, text, config }),
  });
  if (!r.ok) throw new Error(`POST /api/guardrails/apply -> ${r.status}`);
  return guardrailResultP(await r.json());
}

export interface UpstreamOverride {
  readonly endpoint: string;
  readonly model?: string;
  readonly token?: string;
}

export interface RunOpts {
  readonly messages: readonly ChatMessage[];
  readonly permissions: PermissionConfig;
  readonly onEvent: (event: AgentEvent) => void;
  // BYO-AI: drive the full Crucible tool-loop against a user endpoint (Crucible runs tools).
  readonly upstream?: UpstreamOverride;
  // Drive a specific registry model by id (the backend resolves its endpoint or adapter).
  readonly modelId?: string;
  // ReAct tool-loop for models without native function-calling.
  readonly react?: boolean;
  // Id so Stop can cancel this run server-side (via cancelAgent).
  readonly runId?: string;
  // Abort an in-flight run (the Stop button wires this to an AbortController).
  readonly signal?: AbortSignal;
  // Context compaction: summarize old turns before running when the heuristic size exceeds
  // contextLimit tokens (keeps the last few turns verbatim).
  readonly autoCompact?: boolean;
  readonly contextLimit?: number;
  // Named hierarchy profile for the spawn tree (per-layer worker + communicator models).
  readonly profile?: string;
}

export type CompactMessage = { readonly role: string; readonly content: string };
export interface CompactResult {
  readonly messages: readonly CompactMessage[];
  readonly summary: string | null;
  readonly compacted: boolean;
  readonly stats: {
    readonly before_tokens: number;
    readonly after_tokens: number;
    readonly summarized_turns: number;
    readonly token_estimate: string;
  };
  readonly tokens: number;
}

// Compact a conversation: summarize old turns, keep the recent ones verbatim. force=true always
// compacts; force=false only when over max_tokens.
export async function compactConversation(
  messages: readonly CompactMessage[],
  opts: { readonly keepRecent?: number; readonly maxTokens?: number; readonly force?: boolean; readonly modelId?: string } = {},
): Promise<CompactResult> {
  // no backend/model in the static build: compact locally with an extractive summary, stored on-device
  if (isDemo()) return localCompact(messages, opts.keepRecent ?? 6);
  const r = await cfetch(API_BASE + "/api/agent/compact", {
    method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      messages, keep_recent: opts.keepRecent ?? 6, max_tokens: opts.maxTokens ?? 4000,
      force: opts.force ?? true, ...(opts.modelId ? { model_id: opts.modelId } : {}),
    }),
  });
  if (r.status === 503) throw new Error("no model available to write the summary");
  if (!r.ok) throw new Error(`compact ${r.status}`);
  return compactResultP(await r.json());
}

// Halt a run server-side (stops further generation, not just the client stream).
export async function cancelAgent(runId: string): Promise<void> {
  try {
    await cfetch(API_BASE + "/api/agent/cancel", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ run_id: runId }),
    });
  } catch { /* best-effort; the client abort still stops the stream */ }
}

const DEMO_REPLY =
  "This is the static Crucible demo — the agent harness streams tokens like this, one at a time. " +
  "Connect a node (top-right), run Crucible locally, or scan for a BYO backend to drive a real model with tools.";

export function sleep(ms: number, signal?: AbortSignal): Promise<void> {
  return new Promise((resolve) => {
    if (signal?.aborted === true) return resolve();
    const t = setTimeout(resolve, ms);
    signal?.addEventListener("abort", () => { clearTimeout(t); resolve(); }, { once: true });
  });
}

export async function runAgent(opts: RunOpts): Promise<RunStatus> {
  if (isDemo()) {
    // simulate token streaming so the static page shows the live animation + caret
    let first = true;
    for (const word of DEMO_REPLY.split(" ")) {
      if (opts.signal?.aborted === true) break;
      opts.onEvent({ type: "assistant_delta", data: { delta: first ? word : ` ${word}` } });
      first = false;
      await sleep(40, opts.signal);
    }
    opts.onEvent({ type: "assistant", data: { content: DEMO_REPLY, streamed: true } });
    opts.onEvent({ type: "done", data: { content: "" } });
    return "ok";
  }
  let resp: Response;
  try {
    resp = await cfetch(API_BASE + "/api/agent/run", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        messages: opts.messages,
        permissions: opts.permissions,
        ...(opts.upstream
          ? {
              endpoint: opts.upstream.endpoint,
              endpoint_model: opts.upstream.model ?? "local",
              endpoint_token: opts.upstream.token ?? "",
            }
          : {}),
        ...(opts.modelId ? { model_id: opts.modelId } : {}),
        ...(opts.react ? { react: true } : {}),
        ...(opts.autoCompact ? { auto_compact: true, context_limit: opts.contextLimit ?? 4000 } : {}),
        ...(opts.profile ? { profile: opts.profile } : {}),
        ...(opts.runId ? { run_id: opts.runId } : {}),
      }),
      ...(opts.signal ? { signal: opts.signal } : {}),
    });
  } catch {
    return opts.signal?.aborted === true ? "ok" : "offline";
  }
  if (resp.status === 503) return "no-model";
  if (!resp.ok || resp.body === null) return "offline";

  const reader = resp.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  try {
    for (;;) {
      const chunk = await reader.read();
      if (chunk.done) break;
      buffer += decoder.decode(chunk.value, { stream: true });
      const frames = buffer.split("\n\n");
      buffer = frames.pop() ?? "";
      for (const frame of frames) {
        const dataLine = frame.split("\n").find((line) => line.startsWith("data: "));
        if (dataLine === undefined) continue;
        const event = parseEvent(dataLine.slice(6));
        if (event !== null) opts.onEvent(event);
      }
    }
  } catch {
    // aborted by the operator, or the stream dropped — treat as a clean stop
    return "ok";
  }
  return "ok";
}

export async function createPreset(preset: SystemPromptPreset): Promise<SystemPromptPreset> {
  const r = await cfetch(API_BASE + "/api/guardrails/presets", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(preset),
  });
  if (r.status === 409) throw new Error("a preset with that id already exists");
  if (!r.ok) throw new Error(`POST /api/guardrails/presets -> ${r.status}`);
  return systemPromptPresetP(await r.json());
}

export async function updatePreset(id: string, preset: SystemPromptPreset): Promise<SystemPromptPreset> {
  const r = await cfetch(`${API_BASE}/api/guardrails/presets/${encodeURIComponent(id)}`, {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(preset),
  });
  if (!r.ok) throw new Error(`PUT /api/guardrails/presets/${id} -> ${r.status}`);
  return systemPromptPresetP(await r.json());
}

export async function deletePreset(id: string): Promise<void> {
  const r = await cfetch(`${API_BASE}/api/guardrails/presets/${encodeURIComponent(id)}`, { method: "DELETE" });
  if (!r.ok && r.status !== 204) throw new Error(`DELETE /api/guardrails/presets/${id} -> ${r.status}`);
}

export async function resetPresets(): Promise<readonly SystemPromptPreset[]> {
  const r = await cfetch(API_BASE + "/api/guardrails/presets/reset", { method: "POST" });
  if (!r.ok) throw new Error(`POST /api/guardrails/presets/reset -> ${r.status}`);
  return presetsP(await r.json());
}

export interface LayerProfile {
  readonly layer: number;
  readonly separation: number;
  readonly margin: number;
}

export interface AblationImpact {
  readonly total_norm: number;
  readonly removed_norm: number;
  readonly removed_fraction: number;
}

export interface DiagnosisReport {
  readonly base_id: string;
  readonly best_layer: number;
  readonly layer_profile: readonly LayerProfile[];
  readonly components: Readonly<Record<string, AblationImpact>>;
  readonly heaviest_component: string | null;
  readonly mean_removed_fraction: number;
  readonly surgical: boolean;
  readonly collateral_risk: string;
  readonly why: string;
  readonly how: string;
  readonly removal: string;
  readonly narrative?: PlainNarrative;
}

export interface RuntimeInstance {
  readonly model_id: string;
  readonly port: number;
  readonly endpoint: string;
  readonly active: boolean;
  readonly started_at: number;
  readonly last_used: number;
}

export interface RuntimeStatus {
  readonly max_resident: number;
  readonly resident: readonly RuntimeInstance[];
  readonly active: readonly string[];
}

// Validate an arbitrary Crucible node URL before committing the GUI to it. Pings its
// /api/health with the given token; returns ok or the reason it failed.
export async function probeNode(base: string, token: string): Promise<{ ok: boolean; detail: string }> {
  const url = base.replace(/\/$/, "") + "/api/health";
  const headers: Record<string, string> = {};
  if (token) headers["Authorization"] = `Bearer ${token}`;
  try {
    const r = await fetch(url, { headers });
    if (!r.ok) return { ok: false, detail: `HTTP ${r.status}` };
    const body: unknown = await r.json();
    const healthy = isRecord(body) && body["ok"] === true;
    return { ok: healthy, detail: healthy ? "healthy" : "unexpected response" };
  } catch {
    return { ok: false, detail: "unreachable (offline or CORS)" };
  }
}

export async function getRuntime(): Promise<RuntimeStatus> {
  const r = await cfetch(API_BASE + "/api/runtime");
  if (!r.ok) throw new Error(`GET /api/runtime -> ${r.status}`);
  return runtimeStatusP(await r.json());
}

export async function startModel(modelId: string): Promise<{ healthy: boolean; status: RuntimeStatus }> {
  const r = await cfetch(API_BASE + "/api/runtime/start", {
    method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ model_id: modelId }),
  });
  if (!r.ok) throw new Error((await r.json().catch(() => ({}))).detail ?? `start -> ${r.status}`);
  return startResultP(await r.json());
}

export async function stopModel(modelId: string): Promise<RuntimeStatus> {
  const r = await cfetch(API_BASE + "/api/runtime/stop", {
    method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ model_id: modelId }),
  });
  if (!r.ok) throw new Error(`stop -> ${r.status}`);
  return statusWrapP(await r.json()).status;
}

export interface BenchmarkResult {
  readonly model: string;
  readonly tok_per_s: number;
  readonly decode_tok_per_s: number;
  readonly prefill_tok_per_s: number;
  readonly gen_tokens: number;
  readonly total_s: number;
  readonly estimated?: boolean;
  readonly sample?: string;
}

// Pre-flight tokens/second speed test for a model — run before going live.
export async function benchmarkModel(modelId: string | undefined, tokens = 64): Promise<BenchmarkResult> {
  const r = await cfetch(API_BASE + "/api/runtime/benchmark", {
    method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ ...(modelId ? { model_id: modelId } : {}), tokens }),
  });
  if (!r.ok) throw new Error((await r.json().catch(() => ({}))).detail ?? `benchmark -> ${r.status}`);
  return benchmarkResultP(await r.json());
}

export async function setActiveModels(ids: readonly string[]): Promise<RuntimeStatus> {
  const r = await cfetch(API_BASE + "/api/runtime/active", {
    method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ model_ids: ids }),
  });
  if (!r.ok) throw new Error(`active -> ${r.status}`);
  return runtimeStatusP(await r.json());
}

// Crystallized memory = DISTILLED KNOWLEDGE (a fact/decision/preference), NOT a transcript. A card is
// the cheap summary passthrough; a leaf opens to its fact `text`, a chunked node to child cards (drill
// down). Raw conversations live separately as CONTEXTS (below). `source_context` traces a memory back
// to the conversation it was distilled from — without ever being that conversation.
export interface MemoryCard {
  readonly key: string;
  readonly label: string;
  readonly summary: string;
  readonly kind: string;
  readonly session: string;
  readonly size: number;
  readonly ref: string | null;
  readonly priority?: number;         // agent-set weight for prioritized recall
  readonly degree?: number;           // number of out-links (graph)
  readonly source_context?: string | null;
}
export interface MemoryTreeNode extends MemoryCard {
  readonly children?: readonly MemoryTreeNode[];
}
export interface MemoryNode extends MemoryCard {
  readonly text?: string;             // a leaf's distilled fact
  readonly children?: readonly MemoryCard[];
}
export type MemorySubchunk = { readonly label?: string; readonly summary: string; readonly text: string };

// A CONTEXT is a raw archived CONVERSATION you reload wholesale — the counterpart to a memory, kept
// in a separate store so knowledge and transcripts are never conflated.
export interface ContextCard {
  readonly key: string;
  readonly label: string;
  readonly summary: string;
  readonly session: string;
  readonly source: string;            // compaction / migrated / manual
  readonly size: number;              // message count
  readonly created: number;
  readonly kind: string;
}
export interface ContextNode extends ContextCard {
  readonly messages: readonly CompactMessage[];
}
export async function getContextsIndex(session?: string): Promise<{ contexts: readonly ContextCard[] }> {
  if (isDemo()) return localContextIndex(session);
  try {
    const q = session ? `?session=${encodeURIComponent(session)}` : "";
    const r = await cfetch(API_BASE + "/api/contexts" + q);
    if (!r.ok) throw new Error(`contexts ${r.status}`);
    return contextIndexP(await r.json());
  } catch { return localContextIndex(session); }
}
export async function readContext(key: string): Promise<ContextNode> {
  if (isDemo()) return localContextRead(key);
  try {
    const r = await cfetch(`${API_BASE}/api/contexts/${encodeURIComponent(key)}`);
    if (r.status === 404) throw new Error(`no context ${key}`);
    if (!r.ok) throw new Error(`context ${r.status}`);
    return contextNodeP(await r.json());
  } catch { return localContextRead(key); }
}
export async function deleteContext(key: string): Promise<{ deleted: boolean; key: string }> {
  if (isDemo()) return localContextDelete(key);
  const r = await cfetch(`${API_BASE}/api/contexts/${encodeURIComponent(key)}`, { method: "DELETE" });
  if (!r.ok) throw new Error(`delete context ${r.status}`);
  return (await r.json()) as { deleted: boolean; key: string };
}
// Migrate legacy transcript-memories into contexts (+ distilled facts). Offline: heuristic extraction.
export async function migrateMemory(modelId?: string): Promise<{ migrated: unknown[]; converted: string[] }> {
  if (isDemo()) return localMigrate();
  const r = await cfetch(API_BASE + "/api/memory/migrate", {
    method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify(modelId ? { model_id: modelId } : {}),
  });
  if (!r.ok) throw new Error(`migrate ${r.status}`);
  return (await r.json()) as { migrated: unknown[]; converted: string[] };
}

// Memory works with NO backend (static GitHub Pages build): when in demo/offline mode, or if the
// backend is unreachable, these fall back to a LocalStorage store on the device (see localMemory).
export async function getMemoryIndex(session?: string): Promise<{ memories: readonly MemoryCard[]; versioned: boolean }> {
  if (isDemo()) return localIndex(session);
  try {
    const q = session ? `?session=${encodeURIComponent(session)}` : "";
    const r = await cfetch(API_BASE + "/api/memory/index" + q);
    if (!r.ok) throw new Error(`memory index ${r.status}`);
    return memoryIndexP(await r.json());
  } catch { return localIndex(session); }
}
export type MemoryMatch = MemoryCard & { readonly score: number };
export interface MemorySearchResult {
  readonly method: string;                 // "semantic" | "lexical"
  readonly matches: readonly MemoryMatch[];
}
// Relevance search over crystallized memories. `metric` selects the distance/similarity family
// (statistical / lexical / embedding / llm-judged); the method is reported honestly by the backend.
export async function searchMemory(q: string, session?: string, sort = "relevance", metric?: string): Promise<MemorySearchResult> {
  if (isDemo()) return localSearch(q, session, sort, metric);
  try {
    const s = session ? `&session=${encodeURIComponent(session)}` : "";
    const m = metric ? `&metric=${encodeURIComponent(metric)}` : "";
    const r = await cfetch(`${API_BASE}/api/memory/search?q=${encodeURIComponent(q)}&sort=${encodeURIComponent(sort)}${s}${m}`);
    if (!r.ok) throw new Error(`memory search ${r.status}`);
    return memorySearchP(await r.json());
  } catch { return localSearch(q, session, sort, metric); }
}

// The distance/similarity families available for search + reorganization, each with its honest
// method label and whether it can run right now (offline stats always; embedding/llm need backends).
export interface MetricInfo { readonly name: string; readonly label: string; readonly available: boolean }
export interface MetricsCatalog { readonly metrics: readonly MetricInfo[]; readonly processing_model: string | null }
export async function getMetrics(): Promise<MetricsCatalog> {
  if (isDemo()) {
    const metrics: MetricInfo[] = OFFLINE_METRICS.map((n) => ({ name: n, label: METRIC_LABELS[n] ?? n, available: true }));
    return { metrics: [...metrics,
      { name: "embedding", label: "semantic-embedding", available: false },
      { name: "llm", label: "llm-judged", available: false }], processing_model: null };
  }
  try {
    const r = await cfetch(API_BASE + "/api/metrics");
    if (!r.ok) throw new Error(`metrics ${r.status}`);
    return metricsCatalogP(await r.json());
  } catch {
    return { metrics: OFFLINE_METRICS.map((n) => ({ name: n, label: METRIC_LABELS[n] ?? n, available: true })), processing_model: null };
  }
}

// Organizational preferences: recall ordering + distance metric + the processing model + persisted
// tool-permission defaults the forge applies to every run.
// Memory/compute caps for Ollama models (applied via its native /api/chat). Trading RAM for time so
// big local models stop freezing the machine. 0 / "" = model default (uncapped).
export interface ResourceLimits {
  readonly num_ctx: number;           // context window = KV-cache size (the big RAM lever)
  readonly keep_alive: string;        // unload after ("0" = free RAM now, "5m", "-1" = forever)
  readonly max_output_tokens: number; // cap generation length
  readonly num_gpu: number;           // layers on GPU/Metal; lower keeps more on CPU (-1 = auto)
}
export interface Preferences {
  readonly default_sort: string;
  readonly balanced_recency_weight: number;
  readonly default_metric: string;
  readonly processing_model: string | null;
  readonly vision_model: string;
  readonly permissions: PermissionConfig;
  readonly resource_limits: ResourceLimits;
}
export interface PreferencesResult { readonly preferences: Preferences; readonly sorts: readonly string[]; readonly metrics: readonly string[] }
const PREFS_KEY = "crucible_preferences";
const DEFAULT_PREFS: Preferences = {
  default_sort: "recency", balanced_recency_weight: 0.5, default_metric: "bm25",
  processing_model: null, vision_model: "", permissions: { default: "ask", modes: {}, path_rules: [] },
  resource_limits: { num_ctx: 0, keep_alive: "", max_output_tokens: 0, num_gpu: -1 },
};
export async function getPreferences(): Promise<PreferencesResult> {
  if (isDemo()) {
    try { const raw = localStorage.getItem(PREFS_KEY); if (raw) return { preferences: { ...DEFAULT_PREFS, ...JSON.parse(raw) }, sorts: [...SORT_NAMES], metrics: [...METRIC_NAMES] }; } catch { /* ignore */ }
    return { preferences: DEFAULT_PREFS, sorts: [...SORT_NAMES], metrics: [...METRIC_NAMES] };
  }
  try {
    const r = await cfetch(API_BASE + "/api/preferences");
    if (!r.ok) throw new Error(`preferences ${r.status}`);
    return preferencesResultP(await r.json());
  } catch {
    return { preferences: DEFAULT_PREFS, sorts: [...SORT_NAMES], metrics: [...METRIC_NAMES] };
  }
}
export async function savePreferences(body: Partial<Preferences>): Promise<Preferences> {
  if (isDemo()) {
    let cur = DEFAULT_PREFS;
    try { const raw = localStorage.getItem(PREFS_KEY); if (raw) cur = { ...DEFAULT_PREFS, ...JSON.parse(raw) }; } catch { /* ignore */ }
    const merged = { ...cur, ...body };
    try { localStorage.setItem(PREFS_KEY, JSON.stringify(merged)); } catch { /* quota */ }
    return merged;
  }
  const r = await cfetch(API_BASE + "/api/preferences", {
    method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body),
  });
  if (!r.ok) throw new Error(`save preferences ${r.status}`);
  return preferencesP2(await r.json()).preferences;
}

export interface MemoryEdge { readonly from: string; readonly to: string; readonly type: string; readonly kind: string }
export interface MemoryGraph { readonly nodes: readonly MemoryCard[]; readonly edges: readonly MemoryEdge[]; readonly n_nodes: number; readonly n_edges: number }

export async function getMemoryGraph(session?: string): Promise<MemoryGraph> {
  if (isDemo()) return localGraph(session);
  try {
    const q = session ? `?session=${encodeURIComponent(session)}` : "";
    const r = await cfetch(API_BASE + "/api/memory/graph" + q);
    if (!r.ok) throw new Error(`memory graph ${r.status}`);
    return memoryGraphP(await r.json());
  } catch { return localGraph(session); }
}
export async function prioritizeMemory(key: string, priority: number): Promise<void> {
  if (isDemo()) { localSetPriority(key, priority); return; }
  await cfetch(`${API_BASE}/api/memory/${encodeURIComponent(key)}/priority`, {
    method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ priority }),
  });
}
export async function linkMemory(src: string, dst: string, type = "relates"): Promise<void> {
  if (isDemo()) { localLink(src, dst, type); return; }
  await cfetch(API_BASE + "/api/memory/link", {
    method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ src, dst, type }),
  });
}

export async function getMemoryTree(session?: string): Promise<readonly MemoryTreeNode[]> {
  if (isDemo()) return localTree(session);
  try {
    const q = session ? `?session=${encodeURIComponent(session)}` : "";
    const r = await cfetch(API_BASE + "/api/memory/tree" + q);
    if (!r.ok) throw new Error(`memory tree ${r.status}`);
    return memoryTreeP(await r.json()).tree;
  } catch { return localTree(session); }
}
export async function readMemory(key: string): Promise<MemoryNode> {
  if (isDemo()) return localRead(key);
  try {
    const r = await cfetch(`${API_BASE}/api/memory/${encodeURIComponent(key)}`);
    if (r.status === 404) throw new Error(`no memory ${key}`);
    if (!r.ok) throw new Error(`memory ${r.status}`);
    return memoryNodeP(await r.json());
  } catch { return localRead(key); }
}
export async function consolidateMemory(keys: readonly string[], summary: string, label = ""): Promise<MemoryCard> {
  if (isDemo()) return localConsolidate(keys, summary, label);
  const r = await cfetch(API_BASE + "/api/memory/consolidate", {
    method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ keys, summary, label }),
  });
  if (r.status === 422) throw new Error(((await r.json().catch(() => ({}))) as { detail?: string }).detail ?? "cannot consolidate");
  if (!r.ok) throw new Error(`consolidate ${r.status}`);
  return memoryCardP(await r.json());
}
export async function recrystallizeMemory(
  key: string, opts: { subchunks?: readonly MemorySubchunk[]; chunks?: number; modelId?: string } = {},
): Promise<{ key: string; children: readonly string[]; kind: string; ref: string | null }> {
  if (isDemo()) {
    // no model in-browser: split the leaf's messages into `chunks` extractive parts
    const subchunks = opts.subchunks ?? localReadForSplit(key, opts.chunks ?? 2);
    return localRecrystallize(key, subchunks);
  }
  const r = await cfetch(`${API_BASE}/api/memory/${encodeURIComponent(key)}/recrystallize`, {
    method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      ...(opts.subchunks ? { subchunks: opts.subchunks } : {}),
      chunks: opts.chunks ?? 2, ...(opts.modelId ? { model_id: opts.modelId } : {}),
    }),
  });
  if (r.status === 422 || r.status === 503) throw new Error(((await r.json().catch(() => ({}))) as { detail?: string }).detail ?? "cannot re-crystallize");
  if (!r.ok) throw new Error(`recrystallize ${r.status}`);
  return recrystallizeResultP(await r.json());
}

// A plain-language card (attached to interpretability results) — jargon-free explanation.
export interface PlainCardData {
  readonly technique?: string;
  readonly headline: string;
  readonly what_it_is: string;
  readonly what_we_found: string;
  readonly what_it_means: string;
  readonly caveat: string;
}

// Modality safety/refusal direction (image/audio/video) in an encoder's embedding space, scored
// by HELD-OUT (cross-validated) separability so it's honest (~0 for unrelated data).
export interface ModalityDirection {
  readonly modality: string;
  readonly n_harmful: number;
  readonly n_benign: number;
  readonly dim: number;
  readonly separability: number;
  readonly separability_kind: string;
  readonly in_sample_separability: number;
  readonly reliable: boolean;
  readonly reliability_note: string;
  readonly linearly_encoded: boolean;
  readonly direction_norm: number;
  readonly direction: readonly number[];
  readonly plain: PlainCardData;
}
export async function computeModalityDirection(
  modality: string, harmful: readonly (readonly number[])[], benign: readonly (readonly number[])[],
): Promise<ModalityDirection> {
  const r = await cfetch(API_BASE + "/api/abliteration/modality-direction", {
    method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ modality, harmful_embeddings: harmful, benign_embeddings: benign }),
  });
  const detail = (): Promise<string> => r.json().then((j: { detail?: string }) => j.detail ?? `modality ${r.status}`).catch(() => `modality ${r.status}`);
  if (r.status === 422 || r.status === 503) throw new Error(await detail());
  if (!r.ok) throw new Error(`modality ${r.status}`);
  return modalityDirectionP(await r.json());
}

// Model graphs: compose subsystems into a DAG. Stages are model / tool / transform / vote
// (verifier ensemble) / cascade (cheap -> escalate). Outputs are opaque (a string, or a rich
// dict for vote/cascade stages), so they stay `unknown` and the panel narrows per-kind.
export type GraphStage = {
  readonly id: string;
  readonly kind: string;
  readonly inputs: readonly string[];
  readonly config?: Readonly<Record<string, unknown>>;
};
export interface GraphResult {
  readonly order: readonly string[];
  readonly outputs: Readonly<Record<string, unknown>>;
  readonly result: Readonly<Record<string, unknown>>;
}
export async function runGraph(stages: readonly GraphStage[], initial = ""): Promise<GraphResult> {
  const r = await cfetch(API_BASE + "/api/graph/run", {
    method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ stages, initial }),
  });
  if (r.status === 422) throw new Error(((await r.json().catch(() => ({}))) as { detail?: string }).detail ?? "invalid graph");
  if (r.status === 503) throw new Error("no model available for a graph model-stage");
  if (!r.ok) throw new Error(`graph ${r.status}`);
  return graphResultP(await r.json());
}

export interface MediaBackend {
  readonly kind: string;
  readonly label: string;
  readonly env: string;
  readonly endpoint: string | null;
  readonly configured: boolean;
  readonly reachable: boolean | null;
}
export interface MediaStatus {
  readonly backends: Readonly<Record<string, MediaBackend>>;
  readonly n_configured: number;
  readonly n_total: number;
  readonly note: string;
}

// Honest media capability map: which sibling modalities (image/stt/tts/embed) have an external
// backend configured. Nothing is generated in-process, so this shows what's actually wired.
export async function getMediaStatus(): Promise<MediaStatus> {
  const r = await cfetch(API_BASE + "/api/media/status");
  if (!r.ok) throw new Error(`media status ${r.status}`);
  return mediaStatusP(await r.json());
}

export interface PlainNarrative {
  readonly headline: string;
  readonly locate: string;
  readonly evidence: string;
  readonly target: string;
  readonly repair: string;
  readonly risk: string;
  readonly confidence: string;
  readonly steps: readonly string[];
  readonly language?: string;
  readonly translated?: boolean;
}

export interface ModelCard {
  readonly variant_id: string;
  readonly base_id: string;
  readonly method: string;
  readonly layer: number;
  readonly strength: number;
  readonly hidden_size: number;
  readonly repro_hash: string;
  readonly eval_delta: number | null;
}

export type DiagnoseResult =
  | { readonly kind: "report"; readonly report: DiagnosisReport }
  | { readonly kind: "no-weights" }
  | { readonly kind: "no-base" }
  | { readonly kind: "offline" };

export async function diagnoseCensorship(baseId: string): Promise<DiagnoseResult> {
  let resp: Response;
  try {
    resp = await cfetch(API_BASE + "/api/abliteration/diagnose", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ base_id: baseId }),
    });
  } catch {
    return { kind: "offline" };
  }
  if (resp.status === 503) return { kind: "no-weights" };
  if (resp.status === 404) return { kind: "no-base" };
  if (!resp.ok) return { kind: "offline" };
  return { kind: "report", report: diagnosisReportP(await resp.json()) };
}

export interface AbliterateRequestBody {
  readonly base_id: string;
  readonly variant_id: string;
  readonly layer: number;
  readonly strength: number;
}

export type AbliterateResult =
  | { readonly kind: "done"; readonly variant: ModelRow; readonly card: ModelCard }
  | { readonly kind: "no-weights" }
  | { readonly kind: "no-base" }
  | { readonly kind: "offline" };

export async function abliterate(body: AbliterateRequestBody): Promise<AbliterateResult> {
  let resp: Response;
  try {
    resp = await cfetch(API_BASE + "/api/abliteration/run", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
  } catch {
    return { kind: "offline" };
  }
  if (resp.status === 503) return { kind: "no-weights" };
  if (resp.status === 404) return { kind: "no-base" };
  if (!resp.ok) return { kind: "offline" };
  const out = abliterateOutP(await resp.json());
  return { kind: "done", variant: out.variant, card: out.card };
}

export interface MCResult {
  readonly id: string;
  readonly predicted: string;
  readonly answer: string;
  readonly correct: boolean;
}

export interface BenchScore {
  readonly accuracy: number;
  readonly n: number;
  readonly results: readonly MCResult[];
}

export interface HHItem {
  readonly id: string;
  readonly prompt: string;
}

export type PublishedCell = {
  readonly value: number | null;
  readonly source: string;
  readonly source_type?: string;
  readonly verified?: boolean;
  readonly note?: string;
};
export type PublishedTable = Readonly<Record<string, Readonly<Record<string, PublishedCell>>>>;
export type PublishedPayload = { readonly providers: PublishedTable; readonly disclaimer: string };

export type BenchmarksInfo = {
  readonly benchmarks: Readonly<Record<string, number>>;
  readonly kind: string;
  readonly note: string;
};
export async function getBenchmarks(): Promise<BenchmarksInfo> {
  const r = await cfetch(API_BASE + "/api/evals/benchmarks");
  if (!r.ok) throw new Error(`benchmarks ${r.status}`);
  return benchmarksInfoP(await r.json());
}

export async function getPublished(): Promise<PublishedPayload> {
  const r = await cfetch(API_BASE + "/api/evals/published");
  if (!r.ok) throw new Error(`published ${r.status}`);
  return publishedPayloadP(await r.json());
}

export type EvalRunResult =
  | { readonly kind: "score"; readonly score: BenchScore }
  | { readonly kind: "no-model" }
  | { readonly kind: "offline" };

export async function runLocalEval(benchmark: string): Promise<EvalRunResult> {
  let resp: Response;
  try {
    resp = await cfetch(API_BASE + "/api/evals/run", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ benchmark }),
    });
  } catch {
    return { kind: "offline" };
  }
  if (resp.status === 503) return { kind: "no-model" };
  if (!resp.ok) return { kind: "offline" };
  return { kind: "score", score: benchScoreP(await resp.json()) };
}

export async function exportHeadToHead(benchmark: string): Promise<readonly HHItem[]> {
  const r = await cfetch(API_BASE + "/api/evals/headtohead/export", {
    method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ benchmark }),
  });
  if (!r.ok) throw new Error(`export ${r.status}`);
  return hhItemsWrapP(await r.json()).items;
}

export async function scoreHeadToHead(benchmark: string, answers: Readonly<Record<string, string>>): Promise<BenchScore> {
  const r = await cfetch(API_BASE + "/api/evals/headtohead/score", {
    method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ benchmark, answers }),
  });
  if (!r.ok) throw new Error(`score ${r.status}`);
  return benchScoreP(await r.json());
}

export interface SuiteTask {
  readonly task: string;
  readonly label: string;
  readonly detail: string;
  readonly primary: string;
}

export interface LmEvalRow {
  readonly task: string;
  readonly metric: string;
  readonly filter: string | null;
  readonly value: number;
  readonly stderr: number | null;
}

export async function getSuite(): Promise<readonly SuiteTask[]> {
  const r = await cfetch(API_BASE + "/api/evals/suite");
  if (!r.ok) throw new Error(`suite ${r.status}`);
  return suiteP(await r.json());
}

export type LmEvalResult =
  | { readonly kind: "results"; readonly rows: readonly LmEvalRow[]; readonly note?: string }
  | { readonly kind: "no-model" }
  | { readonly kind: "no-endpoint" }
  | { readonly kind: "offline" };

// Collect a model's full reply to one prompt — works in the static build (via runAgent's stream,
// canned in demo, live against a connected model). The `ask` the client-side benchmark drives.
async function askViaAgent(prompt: string): Promise<string> {
  let out = "";
  try {
    await runAgent({
      messages: [{ role: "user", content: prompt }],
      permissions: { default: "deny", modes: {} },
      onEvent: (e) => {
        if (e.type === "assistant_delta") out += String((e.data as { delta?: unknown }).delta ?? "");
        else if (e.type === "assistant" || e.type === "done") {
          const c = (e.data as { content?: unknown }).content;
          if (typeof c === "string" && c) out = c;
        }
      },
    });
  } catch { /* leave out as-is → scored as no-answer */ }
  return out;
}

// No lm-eval backend (static build or a raw model endpoint): run the small bundled MC sample client-side
// against whatever model is reachable. Honest — returns "no-model" if nothing actually answered rather
// than reporting fabricated numbers.
async function clientSideEval(tasks: readonly string[], limit: number | null): Promise<LmEvalResult> {
  const { rows, answered } = await runLocalMcEval(tasks, askViaAgent, limit);
  if (answered === 0) return { kind: "no-model" };
  return { kind: "results", rows, note: SAMPLE_NOTE };
}

export async function runLmEval(modelId: string, tasks: readonly string[], limit: number | null): Promise<LmEvalResult> {
  if (isDemo()) return clientSideEval(tasks, limit);         // static build: benchmark client-side
  let resp: Response;
  try {
    resp = await cfetch(API_BASE + "/api/evals/lmeval", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ model_id: modelId, tasks, limit }),
    });
  } catch {
    return { kind: "offline" };
  }
  if (resp.ok) {
    try { return { kind: "results", rows: lmEvalWrapP(await resp.json()).results }; }
    catch { return clientSideEval(tasks, limit); }          // reachable, but not a Crucible eval backend
  }
  if (resp.status === 404 || resp.status === 409) return clientSideEval(tasks, limit);
  return { kind: "offline" };
}

export interface TensorInfo {
  readonly name: string;
  readonly shape: readonly number[];
  readonly dtype: string;
  readonly n_params: number;
  readonly offset: number;
}

export interface WeightSummary {
  readonly n_tensors: number;
  readonly total_params: number;
  readonly n_layers: number;
  readonly dtypes: Readonly<Record<string, number>>;
  readonly architecture: string | null;
}

// Plain-language guide to the model — what it is, how text flows through it, where behaviors live.
export interface WeightsExplainCard {
  readonly headline: string;
  readonly what_it_is: string;
  readonly how_it_works: string;
  readonly size_meaning: string;
  readonly how_to_change: string;
}
export interface WeightsLayerRole {
  readonly layer: number;
  readonly band: string;      // early | middle | late
  readonly role: string;
  readonly params: number;
  readonly components: readonly string[];
}
export interface WeightsExplain {
  readonly model: WeightsExplainCard;
  readonly layers: readonly WeightsLayerRole[];
  readonly legend: Readonly<Record<string, string>>;
}

export interface WeightsView {
  readonly summary: WeightSummary;
  readonly tensors: readonly TensorInfo[];
  readonly metadata: Readonly<Record<string, unknown>>;
  readonly explain?: WeightsExplain;
}

export type WeightsResult =
  | { readonly kind: "view"; readonly view: WeightsView }
  | { readonly kind: "no-file" }
  | { readonly kind: "no-model" }
  | { readonly kind: "offline" };

export async function getWeights(modelId: string): Promise<WeightsResult> {
  let resp: Response;
  try {
    resp = await cfetch(`${API_BASE}/api/weights/${encodeURIComponent(modelId)}`);
  } catch {
    return { kind: "offline" };
  }
  if (resp.status === 404) return { kind: "no-file" };
  if (!resp.ok) return { kind: "offline" };
  return { kind: "view", view: weightsViewP(await resp.json()) };
}

export interface BeforeAfter {
  readonly before: number;
  readonly after: number;
}

export interface VerifyReport {
  readonly harmful_refusal_rate: BeforeAfter;
  readonly harmful_compliance_rate: BeforeAfter;
  readonly benign_over_refusal_rate: BeforeAfter;
  readonly samples: readonly { readonly prompt: string; readonly before: string; readonly after: string }[];
}

export interface SweepPoint {
  readonly strength: number;
  readonly harmful_compliance: number;
  readonly benign_over_refusal: number;
}

export interface SweepReport {
  readonly layer: number;
  readonly direction_norm: number;
  readonly curve: readonly SweepPoint[];
  readonly recommended_strength: number;
}

export type VerifyResult =
  | { readonly kind: "report"; readonly report: VerifyReport }
  | { readonly kind: "not-found" }
  | { readonly kind: "no-weights" }
  | { readonly kind: "offline" };

export type SweepResult =
  | { readonly kind: "report"; readonly report: SweepReport }
  | { readonly kind: "not-found" }
  | { readonly kind: "no-weights" }
  | { readonly kind: "offline" };

export async function verifyAbliteration(baseId: string, variantId: string): Promise<VerifyResult> {
  let resp: Response;
  try {
    resp = await cfetch(API_BASE + "/api/abliteration/verify", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ base_id: baseId, variant_id: variantId }),
    });
  } catch {
    return { kind: "offline" };
  }
  if (resp.status === 503) return { kind: "no-weights" };
  if (resp.status === 404) return { kind: "not-found" };
  if (!resp.ok) return { kind: "offline" };
  return { kind: "report", report: verifyReportP(await resp.json()) };
}

export async function sweepStrength(baseId: string): Promise<SweepResult> {
  let resp: Response;
  try {
    resp = await cfetch(API_BASE + "/api/abliteration/sweep", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ base_id: baseId }),
    });
  } catch {
    return { kind: "offline" };
  }
  if (resp.status === 503) return { kind: "no-weights" };
  if (resp.status === 404) return { kind: "not-found" };
  if (!resp.ok) return { kind: "offline" };
  return { kind: "report", report: sweepReportP(await resp.json()) };
}

export interface RuntimeSteerReport {
  readonly layer: number;
  readonly rank: number;
  readonly coefficient: number;
  readonly explained_variance: readonly number[];
  readonly weights_modified: boolean;
  readonly harmful_refusal: { readonly hooks_off: number; readonly hooks_on: number; readonly after_detach: number };
  readonly benign_over_refusal: { readonly hooks_off: number; readonly hooks_on: number };
  readonly sample: { readonly prompt: string; readonly hooks_off: string; readonly hooks_on: string };
}

export type RuntimeSteerResult =
  | { readonly kind: "report"; readonly report: RuntimeSteerReport }
  | { readonly kind: "no-weights" }
  | { readonly kind: "not-found" }
  | { readonly kind: "offline" };

export async function runtimeSteer(baseId: string, rank: number, coefficient: number): Promise<RuntimeSteerResult> {
  let resp: Response;
  try {
    resp = await cfetch(API_BASE + "/api/abliteration/runtime-steer", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ base_id: baseId, rank, coefficient }),
    });
  } catch {
    return { kind: "offline" };
  }
  if (resp.status === 503) return { kind: "no-weights" };
  if (resp.status === 404) return { kind: "not-found" };
  if (!resp.ok) return { kind: "offline" };
  return { kind: "report", report: runtimeSteerReportP(await resp.json()) };
}

// UNSUPERVISED bias / propaganda auto-detection: probe a broad bank and rank where THIS model is
// railroaded (refuses) or overwritten (recites one side) — surfacing biases nobody named in advance.
export interface BiasItem {
  readonly topic: string; readonly domain: string; readonly question: string;
  readonly verdict: string;          // "railroaded" | "overwritten" | "balanced"
  readonly bias_score: number; readonly refused: boolean;
  readonly lean: number; readonly separability: number; readonly response: string;
}
export interface BiasReport {
  readonly layer: number; readonly n_probes: number;
  readonly biases: readonly BiasItem[]; readonly flagged: readonly BiasItem[];
  readonly demo?: { readonly topic: string; readonly question: string; readonly base: string; readonly depropagandized: string };
}
export type BiasResult = { kind: "offline" } | { kind: "no-weights" } | { kind: "report"; report: BiasReport };

export async function detectBias(demo = false): Promise<BiasResult> {
  let resp: Response;
  try {
    resp = await cfetch(API_BASE + "/api/abliteration/detect-bias", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ demo }),
    });
  } catch {
    return { kind: "offline" };
  }
  if (resp.status === 503) return { kind: "no-weights" };
  if (!resp.ok) return { kind: "offline" };
  return { kind: "report", report: biasReportP(await resp.json()) };
}

// COMPREHENSIVE bias removal: auto-detect every bias axis and project the weights out of the whole
// subspace at once (multi-directional abliteration) — enumerate → remove, no truth judgment.
export interface PurgeReport {
  readonly layer: number;
  readonly removed: { readonly rank: number; readonly n_matrices: number; readonly topics: readonly string[] };
  readonly comparison: readonly { readonly topic: string; readonly question: string; readonly verdict: string; readonly before: string; readonly after: string }[];
}
export type PurgeResult = { kind: "offline" } | { kind: "no-weights" } | { kind: "report"; report: PurgeReport };

export async function purgeBiases(): Promise<PurgeResult> {
  let resp: Response;
  try {
    resp = await cfetch(API_BASE + "/api/abliteration/purge-biases", {
      method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({}),
    });
  } catch {
    return { kind: "offline" };
  }
  if (resp.status === 503) return { kind: "no-weights" };
  if (!resp.ok) return { kind: "offline" };
  return { kind: "report", report: purgeReportP(await resp.json()) };
}

// The enumerable-basis path: fit an SAE feature dictionary, enumerate the bias features, purge over them.
export interface SaePurgeReport {
  readonly layer: number; readonly n_features_total: number;
  readonly bias_features: readonly { readonly feature: number; readonly bias_score: number; readonly fires_on?: readonly string[] }[];
  readonly removed: { readonly rank: number; readonly n_features: number; readonly n_matrices: number };
  readonly comparison: readonly { readonly topic: string; readonly question: string; readonly before: string; readonly after: string }[];
}
export type SaePurgeResult = { kind: "offline" } | { kind: "no-weights" } | { kind: "report"; report: SaePurgeReport };

export async function saePurge(): Promise<SaePurgeResult> {
  let resp: Response;
  try {
    resp = await cfetch(API_BASE + "/api/abliteration/sae-purge", {
      method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({}),
    });
  } catch {
    return { kind: "offline" };
  }
  if (resp.status === 503) return { kind: "no-weights" };
  if (!resp.ok) return { kind: "offline" };
  return { kind: "report", report: saePurgeReportP(await resp.json()) };
}

// Independent per-feature control via superposition: dial named SAE features up/down (reversible
// inference-time steering) and see base vs steered. Omit coeffs → demo suppressing the top features.
export interface SteerFeaturesReport {
  readonly layer: number;
  readonly bias_features: readonly { readonly feature: number; readonly bias_score: number; readonly fires_on?: readonly string[] }[];
  readonly applied: Readonly<Record<string, number>>;
  readonly prompt: string; readonly base: string; readonly steered: string;
}
export type SteerFeaturesResult = { kind: "offline" } | { kind: "no-weights" } | { kind: "report"; report: SteerFeaturesReport };

export async function steerFeatures(featureCoeffs?: Readonly<Record<string, number>>, prompt?: string): Promise<SteerFeaturesResult> {
  let resp: Response;
  try {
    resp = await cfetch(API_BASE + "/api/abliteration/steer-features", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ ...(featureCoeffs ? { feature_coeffs: featureCoeffs } : {}), ...(prompt ? { prompt } : {}) }),
    });
  } catch {
    return { kind: "offline" };
  }
  if (resp.status === 503) return { kind: "no-weights" };
  if (!resp.ok) return { kind: "offline" };
  return { kind: "report", report: steerFeaturesReportP(await resp.json()) };
}

// ---- STUDIO: the guided beginner flow. Bias discovery is DYNAMIC — the model proposes and frames its
// own contested claims, and every metric below is measured live (no fixed probe list, no fake numbers).
export interface StudioFinding {
  readonly topic: string; readonly domain: string; readonly question: string;
  readonly verdict: string; readonly kind: string; readonly plain: string;
  readonly bias_score: number; readonly refused: boolean; readonly lean: number;
  readonly separability: number; readonly response: string;
  readonly pro: string; readonly con: string; readonly claim: string;
  readonly source: string;                    // "internals" (read from weights/activations) | "ask"
}
export interface StudioHealth {
  readonly perplexity: number; readonly refusal_rate: number;
  readonly refusals: readonly string[]; readonly n_refusal_probes: number;
}
export interface StudioSummary {
  readonly n_candidates: number; readonly n_flagged: number; readonly n_refusal: number;
  readonly n_lean: number; readonly n_balanced: number; readonly healthy: boolean;
  readonly method: string;
}
export interface StudioReadout {
  readonly refusal: readonly string[]; readonly overcommit: readonly string[];
}
export interface StudioScan {
  readonly layer: number; readonly findings: readonly StudioFinding[];
  readonly summary: StudioSummary; readonly health: StudioHealth;
  readonly readout?: StudioReadout;           // the model's own refusal/over-commitment words (internals scan)
}
export type StudioMethod = "internals" | "ask" | "both";
export interface StudioPreview {
  readonly topic: string; readonly action: string | null;
  readonly before: string; readonly after: string; readonly coef: number;
}
export interface StudioVerify {
  readonly healthy: boolean; readonly coherent: boolean; readonly perplexity_ratio: number;
  readonly lean_improved: boolean; readonly refusals_improved: boolean; readonly reason: string;
}
export interface StudioApply {
  readonly mode: string;
  readonly removed: { readonly rank: number; readonly n_matrices: number; readonly topics: readonly string[] };
  readonly cloned_to: string | null;
  readonly applied: { readonly remove: number; readonly enhance: number; readonly keep: number };
  readonly comparison: readonly { readonly topic: string; readonly kind: string; readonly action: string;
    readonly before: string; readonly after: string }[];
  readonly verify: StudioVerify;
  readonly metrics: {
    readonly perplexity: { readonly before: number; readonly after: number; readonly ratio: number };
    readonly mean_lean: { readonly before: number; readonly after: number };
    readonly refusal_rate: { readonly before: number; readonly after: number };
  };
  readonly profile: { readonly name: string; readonly n_steers: number; readonly mode: string };
}

export interface StudioChoice { readonly topic: string; readonly action: string; readonly strength: number }
export type StudioMode = "copy" | "inplace" | "profile";

// TRUTH-FINDING (stage 1): reliability signals, never a truth verdict.
export interface TruthConsistency {
  readonly stance_mean: number; readonly stance_std: number; readonly flip_rate: number;
  readonly entropy: number; readonly negation_coherent: boolean; readonly refusal_rate: number;
  readonly stability: number; readonly n_paraphrases: number; readonly stances: readonly number[];
}
export interface TruthInternalProbe {
  readonly truth_score: number; readonly cv_accuracy: number; readonly statement: string;
}
export interface ReliabilityProfile {
  readonly topic?: string; readonly claim?: string; readonly layer: number;
  readonly consistency: TruthConsistency; readonly consistency_plain: string;
  readonly signals: Readonly<Record<string, number>>;
  readonly internal_probe?: TruthInternalProbe; readonly internal_plain?: string;
  readonly combined_verdict?: string; readonly combined_plain?: string;
}
export type ReliabilityResult =
  | { readonly kind: "report"; readonly report: ReliabilityProfile }
  | { readonly kind: "stale" } | { readonly kind: "no-weights" } | { readonly kind: "offline" };

// Sculpt — live superposition steer bench + PCA feature map over the last scan
export interface StudioSteer {
  readonly question: string; readonly base: string; readonly steered: string;
  readonly applied: Readonly<Record<string, number>>; readonly norm: number;
}
export interface MapPoint { readonly topic: string; readonly kind: string; readonly x: number; readonly y: number; readonly strength: number; readonly lean: number }
export interface StudioMap { readonly points: readonly MapPoint[] }

export async function studioSteer(question: string, steers: Readonly<Record<string, number>>): Promise<StudioSteer | null> {
  try {
    const r = await cfetch(API_BASE + "/api/studio/steer", {
      method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ question, steers }),
    });
    if (!r.ok) return null;
    return studioSteerP(await r.json());
  } catch { return null; }
}

export async function studioMap(): Promise<StudioMap> {
  try {
    const r = await cfetch(API_BASE + "/api/studio/map");
    if (!r.ok) return { points: [] };
    return studioMapP(await r.json());
  } catch { return { points: [] }; }
}

export async function studioReliability(topic: string): Promise<ReliabilityResult> {
  let resp: Response;
  try {
    resp = await cfetch(API_BASE + "/api/studio/reliability", {
      method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ topic }),
    });
  } catch { return { kind: "offline" }; }
  if (resp.status === 503) return { kind: "no-weights" };
  if (resp.status === 409) return { kind: "stale" };
  if (!resp.ok) return { kind: "offline" };
  return { kind: "report", report: reliabilityProfileP(await resp.json()) };
}

// Calibration meter — does stated confidence match accuracy? (model-level, offline, self-contained)
export interface CalibrationBin {
  readonly lo: number; readonly hi: number; readonly mean_conf: number; readonly accuracy: number; readonly count: number;
}
export interface Calibration {
  readonly model: string; readonly accuracy: number; readonly mean_confidence: number; readonly ece: number;
  readonly overconfidence: number; readonly n: number; readonly answered: number;
  readonly bins: readonly CalibrationBin[]; readonly plain: string;
}
export type CalibrationResult =
  | { readonly kind: "report"; readonly report: Calibration }
  | { readonly kind: "no-weights" } | { readonly kind: "offline" };

export async function getCalibration(sample = 20): Promise<CalibrationResult> {
  let resp: Response;
  try {
    resp = await cfetch(API_BASE + "/api/truth/calibration", {
      method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ sample }),
    });
  } catch { return { kind: "offline" }; }
  if (resp.status === 503) return { kind: "no-weights" };
  if (!resp.ok) return { kind: "offline" };
  return { kind: "report", report: calibrationP(await resp.json()) };
}

// Cross-model triangulation — consensus among independent local models (NOT truth)
export interface CrossModelRow { readonly name: string; readonly stance: number; readonly says: string; readonly answer: string }
export interface CrossModel {
  readonly claim: string; readonly n_models: number; readonly n_voted: number; readonly agreement: number;
  readonly consensus: string; readonly contested: boolean; readonly models: readonly CrossModelRow[]; readonly plain: string;
}
export type CrossModelResult =
  | { readonly kind: "report"; readonly report: CrossModel }
  | { readonly kind: "no-weights" } | { readonly kind: "offline" };

export async function crossModel(claim: string, ollamaModels: readonly string[]): Promise<CrossModelResult> {
  let resp: Response;
  try {
    resp = await cfetch(API_BASE + "/api/truth/crossmodel", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ claim, ollama_models: ollamaModels }),
    });
  } catch { return { kind: "offline" }; }
  if (resp.status === 503) return { kind: "no-weights" };
  if (!resp.ok) return { kind: "offline" };
  return { kind: "report", report: crossModelP(await resp.json()) };
}

// Evidence grounding — check a claim against YOUR local trusted corpus (offline, provenance not verdict)
export interface CorpusSource { readonly source: string; readonly passages: number }
export interface CorpusInfo { readonly sources: readonly CorpusSource[]; readonly n_passages: number }
export interface EvidencePassage { readonly source: string; readonly text: string; readonly label: string; readonly fact: string; readonly score: number }
export interface Evidence {
  readonly claim: string; readonly verdict: string; readonly n_support: number; readonly n_contradict: number;
  readonly n_facts: number; readonly passages: readonly EvidencePassage[]; readonly plain: string;
}
export type EvidenceResult =
  | { readonly kind: "report"; readonly report: Evidence }
  | { readonly kind: "no-weights" } | { readonly kind: "offline" };

export async function getCorpus(): Promise<CorpusInfo> {
  try {
    const r = await cfetch(API_BASE + "/api/truth/corpus");
    if (!r.ok) return { sources: [], n_passages: 0 };
    return corpusInfoP(await r.json());
  } catch { return { sources: [], n_passages: 0 }; }
}

export async function addCorpus(source: string, text: string): Promise<CorpusInfo> {
  const r = await cfetch(API_BASE + "/api/truth/corpus", {
    method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ source, text }),
  });
  if (!r.ok) throw new Error("add failed");
  const j = await r.json();
  return { sources: corpusInfoP({ sources: j.sources, n_passages: j.n_passages }).sources, n_passages: j.n_passages };
}

export async function clearCorpus(): Promise<void> {
  await cfetch(API_BASE + "/api/truth/corpus", { method: "DELETE" });
}

export async function checkEvidence(claim: string): Promise<EvidenceResult> {
  let resp: Response;
  try {
    resp = await cfetch(API_BASE + "/api/truth/evidence", {
      method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ claim }),
    });
  } catch { return { kind: "offline" }; }
  if (resp.status === 503) return { kind: "no-weights" };
  if (!resp.ok) return { kind: "offline" };
  return { kind: "report", report: evidenceP(await resp.json()) };
}

export async function truthCheck(claim: string): Promise<ReliabilityResult> {
  let resp: Response;
  try {
    resp = await cfetch(API_BASE + "/api/truth/check", {
      method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ claim }),
    });
  } catch { return { kind: "offline" }; }
  if (resp.status === 503) return { kind: "no-weights" };
  if (!resp.ok) return { kind: "offline" };
  return { kind: "report", report: reliabilityProfileP(await resp.json()) };
}

export type StudioScanResult =
  | { readonly kind: "report"; readonly report: StudioScan }
  | { readonly kind: "no-weights" } | { readonly kind: "offline" };
export type StudioPreviewResult =
  | { readonly kind: "report"; readonly report: StudioPreview }
  | { readonly kind: "stale" } | { readonly kind: "no-weights" } | { readonly kind: "offline" };
export type StudioApplyResult =
  | { readonly kind: "report"; readonly report: StudioApply }
  | { readonly kind: "stale" } | { readonly kind: "no-weights" } | { readonly kind: "offline" };

export async function studioScan(body: Readonly<Record<string, unknown>> = {}): Promise<StudioScanResult> {
  let resp: Response;
  try {
    resp = await cfetch(API_BASE + "/api/studio/scan", {
      method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body),
    });
  } catch { return { kind: "offline" }; }
  if (resp.status === 503) return { kind: "no-weights" };
  if (!resp.ok) return { kind: "offline" };
  return { kind: "report", report: studioScanP(await resp.json()) };
}

export async function studioPreview(topic: string, action: string, strength: number): Promise<StudioPreviewResult> {
  let resp: Response;
  try {
    resp = await cfetch(API_BASE + "/api/studio/preview", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ topic, action, strength }),
    });
  } catch { return { kind: "offline" }; }
  if (resp.status === 503) return { kind: "no-weights" };
  if (resp.status === 409) return { kind: "stale" };
  if (!resp.ok) return { kind: "offline" };
  return { kind: "report", report: studioPreviewP(await resp.json()) };
}

export async function studioApply(choices: readonly StudioChoice[], mode: StudioMode,
                                  opts: { readonly recipe_name?: string; readonly out_path?: string } = {}): Promise<StudioApplyResult> {
  let resp: Response;
  try {
    resp = await cfetch(API_BASE + "/api/studio/apply", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ choices, mode, ...opts }),
    });
  } catch { return { kind: "offline" }; }
  if (resp.status === 503) return { kind: "no-weights" };
  if (resp.status === 409) return { kind: "stale" };
  if (!resp.ok) return { kind: "offline" };
  return { kind: "report", report: studioApplyP(await resp.json()) };
}

export interface AutotuneConfigResult {
  readonly band: string;
  readonly rank: number;
  readonly coefficient: number;
  readonly harmful_refusal: number;
  readonly benign_over_refusal: number;
  readonly score: number;
}

export interface AutotuneReport {
  readonly baseline: { readonly harmful_refusal: number; readonly benign_over_refusal: number };
  readonly results: readonly AutotuneConfigResult[];
  readonly best: AutotuneConfigResult;
  readonly recipe: { readonly band: string; readonly rank: number; readonly coefficient: number };
  readonly recipe_hash: string;
  readonly weights_modified: boolean;
}

export type AutotuneResult =
  | { readonly kind: "report"; readonly report: AutotuneReport }
  | { readonly kind: "no-weights" }
  | { readonly kind: "not-found" }
  | { readonly kind: "offline" };

export async function autotuneAbliteration(baseId: string): Promise<AutotuneResult> {
  let resp: Response;
  try {
    resp = await cfetch(API_BASE + "/api/abliteration/autotune", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ base_id: baseId }),
    });
  } catch {
    return { kind: "offline" };
  }
  if (resp.status === 503) return { kind: "no-weights" };
  if (resp.status === 404) return { kind: "not-found" };
  if (!resp.ok) return { kind: "offline" };
  return { kind: "report", report: autotuneReportP(await resp.json()) };
}

export interface ManualReport {
  readonly layers: readonly number[];
  readonly rank: number;
  readonly coefficient: number;
  readonly explained_variance: Readonly<Record<string, readonly number[]>>;
  readonly weights_modified: boolean;
  readonly harmful_refusal: number;
  readonly benign_over_refusal: number;
  readonly recipe_hash: string;
  readonly test?: { readonly prompt: string; readonly base: string; readonly ablated: string };
}

export type ManualResult =
  | { readonly kind: "report"; readonly report: ManualReport }
  | { readonly kind: "no-weights" }
  | { readonly kind: "not-found" }
  | { readonly kind: "offline" };

export interface RecipeRow {
  readonly name: string;
  readonly base_id: string;
  readonly layers: readonly number[];
  readonly rank: number;
  readonly coefficient: number;
  readonly recipe_hash: string;
}

export async function manualSteer(baseId: string, layers: readonly number[], rank: number, coefficient: number, testPrompt: string): Promise<ManualResult> {
  let resp: Response;
  try {
    resp = await cfetch(API_BASE + "/api/abliteration/manual", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ base_id: baseId, layers, rank, coefficient, test_prompt: testPrompt || null }),
    });
  } catch {
    return { kind: "offline" };
  }
  if (resp.status === 503) return { kind: "no-weights" };
  if (resp.status === 404) return { kind: "not-found" };
  if (!resp.ok) return { kind: "offline" };
  return { kind: "report", report: manualReportP(await resp.json()) };
}

export async function getRecipes(): Promise<readonly RecipeRow[]> {
  const r = await cfetch(API_BASE + "/api/abliteration/recipes");
  if (!r.ok) return [];
  return recipesP(await r.json());
}

export async function saveRecipe(recipe: RecipeRow): Promise<void> {
  await cfetch(API_BASE + "/api/abliteration/recipes", {
    method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify(recipe),
  });
}

export async function deleteRecipe(name: string): Promise<void> {
  await cfetch(`${API_BASE}/api/abliteration/recipes/${encodeURIComponent(name)}`, { method: "DELETE" });
}

export interface HeatmapReport {
  readonly direction_layer: number;
  readonly matrix: readonly (readonly number[])[];
  readonly tokens: readonly string[];
}

export type HeatmapResult =
  | { readonly kind: "report"; readonly report: HeatmapReport }
  | { readonly kind: "no-weights" }
  | { readonly kind: "not-found" }
  | { readonly kind: "offline" };

export async function getHeatmap(baseId: string, prompt: string): Promise<HeatmapResult> {
  let resp: Response;
  try {
    resp = await cfetch(API_BASE + "/api/abliteration/heatmap", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ base_id: baseId, prompt }),
    });
  } catch {
    return { kind: "offline" };
  }
  if (resp.status === 503) return { kind: "no-weights" };
  if (resp.status === 404) return { kind: "not-found" };
  if (!resp.ok) return { kind: "offline" };
  return { kind: "report", report: heatmapReportP(await resp.json()) };
}

export interface FeatureTrigger { readonly prompt: string; readonly refusal: string }

export interface FeatureCard {
  readonly name: string;
  readonly summary: string;
  readonly peak_layer: number;
  readonly active_layers: readonly number[];
  readonly strength: number;
  readonly output_signature: readonly string[];
  readonly triggers: readonly FeatureTrigger[];
}

export type FeatureCardResult =
  | { readonly kind: "report"; readonly card: FeatureCard }
  | { readonly kind: "no-weights" }
  | { readonly kind: "not-found" }
  | { readonly kind: "offline" };

export async function getFeatureCard(baseId: string): Promise<FeatureCardResult> {
  let resp: Response;
  try {
    resp = await cfetch(API_BASE + "/api/abliteration/feature-card", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ base_id: baseId }),
    });
  } catch {
    return { kind: "offline" };
  }
  if (resp.status === 503) return { kind: "no-weights" };
  if (resp.status === 404) return { kind: "not-found" };
  if (!resp.ok) return { kind: "offline" };
  return { kind: "report", card: featureCardP(await resp.json()) };
}

export interface EditCommit {
  readonly id: string;
  readonly parent: string | null;
  readonly branch: string;
  readonly op: string;
  readonly summary: string;
  readonly metrics: Readonly<Record<string, number>>;
  readonly tensors: readonly string[];
}

export interface EditHistory { readonly branch: string; readonly commits: readonly EditCommit[] }

export async function getHistory(): Promise<EditHistory> {
  const r = await cfetch(API_BASE + "/api/inference/history");
  if (!r.ok) return { branch: "main", commits: [] };
  return editHistoryP(await r.json());
}

export async function revertCommit(id: string): Promise<boolean> {
  const r = await cfetch(`${API_BASE}/api/inference/revert/${encodeURIComponent(id)}`, { method: "POST" });
  return r.ok;
}

// Per-part lineage: each model subsystem (encoder / connector / language / moderation) versioned
// independently, so you can revert one part's edits without disturbing the others.
export interface PartLineage {
  readonly part: string;
  readonly n_versions: number;
  readonly latest: string;
  readonly commits: readonly { readonly id: string; readonly op: string; readonly summary: string }[];
}
export interface Lineage { readonly branch: string; readonly parts: readonly PartLineage[] }

export async function getLineage(): Promise<Lineage> {
  const r = await cfetch(API_BASE + "/api/inference/lineage");
  if (!r.ok) return { branch: "main", parts: [] };
  return lineageP(await r.json());
}

export async function revertPart(part: string): Promise<boolean> {
  const r = await cfetch(`${API_BASE}/api/inference/revert-part/${encodeURIComponent(part)}`, { method: "POST" });
  return r.ok;
}

// Agent hierarchy profiles: per-layer worker + lighter communicator model pairs (multi-layer,
// multi-profile). The spawn tree uses the named profile; the communicator relays between layers.
export interface HierarchyLayer { readonly worker: string | null; readonly communicator: string | null }
export interface HierarchyProfile { readonly name: string; readonly layers: readonly HierarchyLayer[] }

export async function getProfiles(): Promise<readonly HierarchyProfile[]> {
  const r = await cfetch(API_BASE + "/api/hierarchy/profiles");
  if (!r.ok) return [];
  return profilesP(await r.json()).profiles;
}
export async function saveProfile(p: HierarchyProfile): Promise<HierarchyProfile> {
  const r = await cfetch(API_BASE + "/api/hierarchy/profiles", {
    method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(p),
  });
  if (r.status === 422) throw new Error("profile name is required");
  if (!r.ok) throw new Error(`save profile ${r.status}`);
  return hierarchyProfileP(await r.json());
}
export async function deleteProfile(name: string): Promise<boolean> {
  const r = await cfetch(`${API_BASE}/api/hierarchy/profiles/${encodeURIComponent(name)}`, { method: "DELETE" });
  return r.ok;
}

export async function cloneModel(outPath: string): Promise<boolean> {
  const r = await cfetch(API_BASE + "/api/inference/clone", {
    method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ out_path: outPath }),
  });
  return r.ok;
}

export interface ProbeRow {
  readonly category: string;
  readonly prompt: string;
  readonly base: string;
  readonly steered: string;
  readonly base_refused: boolean;
  readonly steered_refused: boolean;
}

export type ProbeResult =
  | { readonly kind: "report"; readonly rows: readonly ProbeRow[] }
  | { readonly kind: "no-weights" }
  | { readonly kind: "not-found" }
  | { readonly kind: "offline" };

export async function runProbe(baseId: string, layers: readonly number[], rank: number, coefficient: number): Promise<ProbeResult> {
  let resp: Response;
  try {
    resp = await cfetch(API_BASE + "/api/abliteration/probe", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ base_id: baseId, layers, rank, coefficient }),
    });
  } catch {
    return { kind: "offline" };
  }
  if (resp.status === 503) return { kind: "no-weights" };
  if (resp.status === 404) return { kind: "not-found" };
  if (!resp.ok) return { kind: "offline" };
  return { kind: "report", rows: probeWrapP(await resp.json()).rows };
}

export interface FlowCarrier { readonly layer: number; readonly component: string; readonly mass: number }
export interface FlowReport {
  readonly input: string;
  readonly best_layer: number;
  readonly carriers: readonly FlowCarrier[];
  readonly outputs: readonly string[];
}
export type FlowResult =
  | { readonly kind: "report"; readonly report: FlowReport }
  | { readonly kind: "no-weights" }
  | { readonly kind: "not-found" }
  | { readonly kind: "offline" };

export async function getFlow(baseId: string): Promise<FlowResult> {
  let resp: Response;
  try {
    resp = await cfetch(API_BASE + "/api/abliteration/flow", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ base_id: baseId }),
    });
  } catch {
    return { kind: "offline" };
  }
  if (resp.status === 503) return { kind: "no-weights" };
  if (resp.status === 404) return { kind: "not-found" };
  if (!resp.ok) return { kind: "offline" };
  return { kind: "report", report: flowReportP(await resp.json()) };
}

// --- avatar / companion rig ------------------------------------------------------------------
// The web companion window: the backend is the source of truth for a mood → face PARAMS mapping (the same
// engine-agnostic state that drives the TUI pixel face and any VTube-Studio rig). We fetch a RigFrame per
// mood change and animate gaze/blink locally for smooth motion. `live2d` params drive the SVG face.
export interface AvatarLayerInfo {
  readonly id: string; readonly part: string; readonly protected: boolean;
  readonly states: readonly string[]; readonly default_state: string;
  readonly pos: readonly number[]; readonly mirror: boolean; readonly spacing: number;
}
export interface AvatarInfo {
  readonly name: string; readonly kind: string; readonly size: readonly number[];
  readonly expressions: readonly string[]; readonly layers: readonly AvatarLayerInfo[];
}
export interface RigFrame {
  readonly params: Readonly<Record<string, number>>;
  readonly gaze: readonly number[];
  readonly blink: number;
  readonly arkit: Readonly<Record<string, number>>;
  readonly live2d: Readonly<Record<string, number>>;
  readonly vrm: Readonly<Record<string, number>>;
}

export async function getAvatarInfo(): Promise<AvatarInfo> {
  if (isDemo()) return DEMO_AVATAR;
  const r = await cfetch(API_BASE + "/api/avatar");
  if (r.status === 404) throw new Error("stale-backend");   // endpoints not loaded — restart the backend
  if (!r.ok) throw new Error(`GET /api/avatar -> ${r.status}`);
  return avatarInfoP(await r.json());
}

export async function postRigFrame(
  weights: Readonly<Record<string, number>>, gaze?: readonly [number, number], blink = 0,
): Promise<RigFrame> {
  if (isDemo()) return demoRigFrame(weights, gaze, blink);
  const r = await cfetch(API_BASE + "/api/avatar/rig-frame", {
    method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ weights, gaze: gaze ?? null, blink }),
  });
  if (!r.ok) throw new Error(`POST /api/avatar/rig-frame -> ${r.status}`);
  return rigFrameP(await r.json());
}

export async function getReactionFrame(reaction: string): Promise<RigFrame> {
  if (isDemo()) return demoRigFrame({ [DEMO_REACTION[reaction] ?? "neutral"]: 1 });
  const r = await cfetch(API_BASE + "/api/avatar/reaction/" + encodeURIComponent(reaction));
  if (!r.ok) throw new Error(`GET /api/avatar/reaction -> ${r.status}`);
  return rigFrameP(await r.json());
}

// A server-rendered PNG of the ACTUAL avatar (the cute-anime sprite composite) for a mood blend + gaze/
// blink/talk — the web window shows this so the browser displays the real avatar art, not a stand-in.
export interface AvatarRenderParams {
  readonly blend?: string;        // "happy:0.6,surprised:0.4"
  readonly expression?: string;
  readonly gx?: number; readonly gy?: number;
  readonly blink?: number; readonly talk?: number;
  readonly scale?: number;
  readonly bob?: number;          // vertical head-bob px (animation)
  readonly tilt?: number;         // head tilt degrees (animation)
  readonly armL?: number;         // left arm rotation degrees (gesture)
  readonly armR?: number;         // right arm rotation degrees (gesture)
  readonly hairPhys?: boolean;    // mesh-deform the hair with spring physics (lag/bounce)
  readonly sid?: string;          // stable session id for stateful hair physics
  readonly hide?: string;         // comma-separated layers to hide (glasses,hair,headphones,body,blush,brows,mouth,eyes)
}
export function avatarRenderUrl(p: AvatarRenderParams): string {
  const q = new URLSearchParams();
  if (p.blend) q.set("blend", p.blend);
  if (p.expression) q.set("expression", p.expression);
  if (p.gx !== undefined) q.set("gx", p.gx.toFixed(3));
  if (p.gy !== undefined) q.set("gy", p.gy.toFixed(3));
  if (p.blink !== undefined) q.set("blink", String(p.blink));
  if (p.talk !== undefined) q.set("talk", String(p.talk));
  if (p.scale !== undefined) q.set("scale", String(Math.round(p.scale)));
  if (p.bob !== undefined && p.bob !== 0) q.set("bob", p.bob.toFixed(2));
  if (p.tilt !== undefined && p.tilt !== 0) q.set("tilt", p.tilt.toFixed(2));
  if (p.armL !== undefined && p.armL !== 0) q.set("arm_l", p.armL.toFixed(2));
  if (p.armR !== undefined && p.armR !== 0) q.set("arm_r", p.armR.toFixed(2));
  if (p.hairPhys && p.sid) { q.set("hair_phys", "1"); q.set("sid", p.sid); }
  if (p.hide) q.set("hide", p.hide);
  return API_BASE + "/api/avatar/render.png?" + q.toString();
}
export async function fetchAvatarRender(p: AvatarRenderParams): Promise<Blob> {
  const r = await cfetch(avatarRenderUrl(p));
  if (!r.ok) throw new Error(`GET /api/avatar/render.png -> ${r.status}`);
  return r.blob();
}
