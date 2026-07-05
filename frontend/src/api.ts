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
  abliterateOutP, autotuneReportP, benchScoreP, benchmarkResultP, benchmarksInfoP,
  diagnosisReportP, editHistoryP, featureCardP, flowReportP, guardrailConfigP, guardrailResultP,
  heatmapReportP, hhItemsWrapP, lmEvalWrapP, manualReportP, modelRowsP, presetsP, probeWrapP,
  compactResultP, mediaStatusP, publishedPayloadP, recipesP, runtimeSteerReportP, runtimeStatusP,
  startResultP, statusWrapP, suiteP, sweepReportP, systemPromptPresetP, verifyReportP, weightsViewP,
} from "./schemas";
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

export interface PermissionConfig {
  readonly default: PermissionMode;
  readonly modes: Readonly<Record<string, PermissionMode>>;
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

export interface WeightsView {
  readonly summary: WeightSummary;
  readonly tensors: readonly TensorInfo[];
  readonly metadata: Readonly<Record<string, unknown>>;
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
