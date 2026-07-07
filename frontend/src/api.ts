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
  localCompact, localConsolidate, localGraph, localIndex, localLink, localReadForSplit, localRead,
  localRecrystallize, localSearch, localSetPriority, localTree, METRIC_LABELS, OFFLINE_METRICS,
} from "./localMemory";
import {
  abliterateOutP, autotuneReportP, benchScoreP, benchmarkResultP, benchmarksInfoP,
  diagnosisReportP, editHistoryP, featureCardP, flowReportP, guardrailConfigP, guardrailResultP,
  hierarchyProfileP, lineageP, profilesP,
  heatmapReportP, hhItemsWrapP, lmEvalWrapP, manualReportP, modelRowsP, presetsP, probeWrapP,
  compactResultP, graphResultP, mediaStatusP, modalityDirectionP, memoryCardP, memoryIndexP,
  memoryGraphP, memoryNodeP, memorySearchP, memoryTreeP, metricsCatalogP, preferencesP2, preferencesResultP,
  publishedPayloadP, recipesP, recrystallizeResultP, runtimeSteerReportP,
  runtimeStatusP, startResultP, statusWrapP, suiteP, sweepReportP, systemPromptPresetP,
  verifyReportP, weightsViewP,
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
  | { readonly type: "tool_call"; readonly data: { readonly id: string; readonly name: string; readonly args: Readonly<Record<string, unknown>> } }
  | { readonly type: "tool_result"; readonly data: { readonly id: string; readonly name: string; readonly ok: boolean; readonly output: string; readonly error: string | null } }
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

// Crystallized memory: compaction keeps old context as a git-versioned tree. A card is the cheap
// summary passthrough; a node opens to full messages (leaf) or child cards (chunked, drill down).
export interface MemoryCard {
  readonly key: string;
  readonly label: string;
  readonly summary: string;
  readonly kind: string;
  readonly session: string;
  readonly size: number;
  readonly ref: string | null;
  readonly priority?: number;   // agent-set weight for prioritized recall
  readonly degree?: number;     // number of out-links (graph)
}
export interface MemoryTreeNode extends MemoryCard {
  readonly children?: readonly MemoryTreeNode[];
}
export interface MemoryNode extends MemoryCard {
  readonly messages?: readonly { readonly role: string; readonly content: string }[];
  readonly children?: readonly MemoryCard[];
}
export type MemorySubchunk = { readonly label?: string; readonly summary: string; readonly messages: readonly { readonly role: string; readonly content: string }[] };

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
    return { metrics: OFFLINE_METRICS.map((n) => ({ name: n, label: METRIC_LABELS[n] ?? n, available: true }))
      .concat([{ name: "embedding", label: "semantic-embedding", available: false },
               { name: "llm", label: "llm-judged", available: false }]), processing_model: null };
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
  readonly permissions: PermissionConfig;
  readonly resource_limits: ResourceLimits;
}
export interface PreferencesResult { readonly preferences: Preferences; readonly sorts: readonly string[]; readonly metrics: readonly string[] }
const PREFS_KEY = "crucible_preferences";
const DEFAULT_PREFS: Preferences = {
  default_sort: "recency", balanced_recency_weight: 0.5, default_metric: "bm25",
  processing_model: null, permissions: { default: "ask", modes: {}, path_rules: [] },
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
  | { readonly kind: "results"; readonly rows: readonly LmEvalRow[] }
  | { readonly kind: "no-model" }
  | { readonly kind: "no-endpoint" }
  | { readonly kind: "offline" };

export async function runLmEval(modelId: string, tasks: readonly string[], limit: number | null): Promise<LmEvalResult> {
  let resp: Response;
  try {
    resp = await cfetch(API_BASE + "/api/evals/lmeval", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ model_id: modelId, tasks, limit }),
    });
  } catch {
    return { kind: "offline" };
  }
  if (resp.status === 404) return { kind: "no-model" };
  if (resp.status === 409) return { kind: "no-endpoint" };
  if (!resp.ok) return { kind: "offline" };
  const out = lmEvalWrapP(await resp.json());
  return { kind: "results", rows: out.results };
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
