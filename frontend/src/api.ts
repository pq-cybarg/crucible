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
  | { readonly type: "assistant"; readonly data: { readonly content: string } }
  | { readonly type: "tool_call"; readonly data: { readonly id: string; readonly name: string; readonly args: Readonly<Record<string, unknown>> } }
  | { readonly type: "tool_result"; readonly data: { readonly id: string; readonly name: string; readonly ok: boolean; readonly output: string; readonly error: string | null } }
  | { readonly type: "done"; readonly data: { readonly content: string } }
  | { readonly type: "error"; readonly data: { readonly reason: string } };

export type RunStatus = "ok" | "no-model" | "offline";

const EVENT_TYPES: ReadonlySet<string> = new Set(["assistant", "tool_call", "tool_result", "done", "error"]);

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null;
}

function parseEvent(raw: string): AgentEvent | null {
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
    const r = await fetch("/api/health");
    if (!r.ok) return false;
    const body: unknown = await r.json();
    return isRecord(body) && body["ok"] === true;
  } catch {
    return false;
  }
}

export async function getModels(): Promise<readonly ModelRow[]> {
  const r = await fetch("/api/models");
  if (!r.ok) throw new Error(`GET /api/models -> ${r.status}`);
  const body: unknown = await r.json();
  return Array.isArray(body) ? (body as readonly ModelRow[]) : [];
}

export async function getPresets(): Promise<readonly SystemPromptPreset[]> {
  const r = await fetch("/api/guardrails/presets");
  if (!r.ok) throw new Error(`GET /api/guardrails/presets -> ${r.status}`);
  const body: unknown = await r.json();
  return Array.isArray(body) ? (body as readonly SystemPromptPreset[]) : [];
}

export async function getGuardrailConfig(): Promise<GuardrailConfig> {
  const r = await fetch("/api/guardrails/config");
  if (!r.ok) throw new Error(`GET /api/guardrails/config -> ${r.status}`);
  return (await r.json()) as GuardrailConfig;
}

export async function putGuardrailConfig(config: GuardrailConfig): Promise<GuardrailConfig> {
  const r = await fetch("/api/guardrails/config", {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(config),
  });
  if (!r.ok) throw new Error(`PUT /api/guardrails/config -> ${r.status}`);
  return (await r.json()) as GuardrailConfig;
}

export async function previewGuardrail(stage: Stage, text: string, config: GuardrailConfig): Promise<GuardrailResult> {
  const r = await fetch("/api/guardrails/apply", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ stage, text, config }),
  });
  if (!r.ok) throw new Error(`POST /api/guardrails/apply -> ${r.status}`);
  return (await r.json()) as GuardrailResult;
}

export interface RunOpts {
  readonly messages: readonly ChatMessage[];
  readonly permissions: PermissionConfig;
  readonly onEvent: (event: AgentEvent) => void;
}

export async function runAgent(opts: RunOpts): Promise<RunStatus> {
  let resp: Response;
  try {
    resp = await fetch("/api/agent/run", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ messages: opts.messages, permissions: opts.permissions }),
    });
  } catch {
    return "offline";
  }
  if (resp.status === 503) return "no-model";
  if (!resp.ok || resp.body === null) return "offline";

  const reader = resp.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
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
  return "ok";
}
