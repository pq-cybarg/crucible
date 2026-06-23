// BYO-AI: detect local/remote AI services and use them as a chat backend from the
// (even static) Crucible page. Chat-capable services power the Agent console directly;
// editing/abliteration still needs a Crucible backend with WRITE access to the weights.
export type ServiceType = "crucible" | "openai" | "ollama" | "comfyui";

export interface DetectedService {
  readonly type: ServiceType;
  readonly name: string;
  readonly baseUrl: string;
  readonly models: readonly string[];
  readonly chat: boolean;   // usable as a chat backend
  readonly full: boolean;   // full Crucible (agent tools + diagnose + edit)
  readonly note: string;
}

interface Probe { type: ServiceType; name: string; baseUrl: string; test: string }

export const PROBES: readonly Probe[] = [
  { type: "crucible", name: "Crucible", baseUrl: "http://localhost:8400", test: "/api/health" },
  { type: "ollama", name: "Ollama", baseUrl: "http://localhost:11434", test: "/api/tags" },
  { type: "openai", name: "llama.cpp / OpenAI-compatible", baseUrl: "http://localhost:8080", test: "/v1/models" },
  { type: "openai", name: "llama.cpp (Crucible default)", baseUrl: "http://localhost:8081", test: "/v1/models" },
  { type: "openai", name: "vLLM / OpenAI-compatible", baseUrl: "http://localhost:8000", test: "/v1/models" },
  { type: "comfyui", name: "ComfyUI", baseUrl: "http://localhost:8188", test: "/system_stats" },
];

export function extractModels(type: ServiceType, body: unknown): string[] {
  if (typeof body !== "object" || body === null) return [];
  const b = body as Record<string, unknown>;
  if (type === "ollama" && Array.isArray(b["models"])) {
    return (b["models"] as Record<string, unknown>[]).map((m) => String(m["name"] ?? m["model"] ?? "")).filter(Boolean);
  }
  if (Array.isArray(b["data"])) {
    return (b["data"] as Record<string, unknown>[]).map((m) => String(m["id"] ?? "")).filter(Boolean);
  }
  return [];
}

export function describeService(p: { type: ServiceType; name: string; baseUrl: string }, models: string[]): DetectedService {
  const full = p.type === "crucible";
  const chat = p.type === "crucible" || p.type === "ollama" || p.type === "openai";
  const note = full
    ? "full — chat, agent tools, diagnose & edit"
    : p.type === "comfyui"
      ? "image generation — detected, not a chat backend"
      : "chat only — to abliterate/edit, run Crucible (it can wrap this model) or give it write access to the weights";
  return { type: p.type, name: p.name, baseUrl: p.baseUrl, models, chat, full, note };
}

async function probeOne(p: Probe, timeoutMs = 1200): Promise<DetectedService | null> {
  const ctrl = new AbortController();
  const timer = setTimeout(() => ctrl.abort(), timeoutMs);
  try {
    const r = await fetch(p.baseUrl + p.test, { signal: ctrl.signal });
    if (!r.ok) return null;
    let models: string[] = [];
    try { models = extractModels(p.type, await r.json()); } catch { /* non-JSON ok */ }
    return describeService(p, models);
  } catch {
    return null;
  } finally {
    clearTimeout(timer);
  }
}

export async function detectServices(extraUrls: readonly string[] = []): Promise<DetectedService[]> {
  const extra: Probe[] = extraUrls.map((u) => ({ type: "openai", name: "custom", baseUrl: u.replace(/\/$/, ""), test: "/v1/models" }));
  const all = await Promise.all([...PROBES, ...extra].map((p) => probeOne(p)));
  return all.filter((x): x is DetectedService => x !== null);
}

export function chatEndpoint(svc: DetectedService): string {
  // Ollama, llama.cpp, vLLM and Crucible all expose an OpenAI-compatible /v1.
  return svc.baseUrl.replace(/\/$/, "") + "/v1/chat/completions";
}

export async function chatDirect(svc: DetectedService, messages: readonly { role: string; content: string }[],
                                 model?: string, maxTokens = 512): Promise<string> {
  const r = await fetch(chatEndpoint(svc), {
    method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ model: model ?? svc.models[0] ?? "local", messages, max_tokens: maxTokens, stream: false }),
  });
  if (!r.ok) throw new Error(`chat ${r.status}`);
  const body = (await r.json()) as { choices?: { message?: { content?: string } }[] };
  return body.choices?.[0]?.message?.content ?? "";
}

// Parse a single OpenAI SSE "data: {...}" line into its content delta (or null).
// Pure + exported so the streaming loop stays trivial and this stays unit-tested.
export function sseContentDelta(dataLine: string): string | null {
  const line = dataLine.trim();
  if (!line.startsWith("data:")) return null;
  const payload = line.slice(5).trim();
  if (payload === "" || payload === "[DONE]") return null;
  try {
    const obj = JSON.parse(payload) as { choices?: { delta?: { content?: string } }[] };
    return obj.choices?.[0]?.delta?.content ?? null;
  } catch {
    return null;
  }
}

// Direct streaming chat: browser → service /v1 with stream:true. Calls onToken for each
// content delta and resolves with the full text. Falls back to chatDirect if the body
// isn't a readable stream (some servers ignore stream:true). No tool-loop.
export async function chatDirectStream(
  svc: DetectedService,
  messages: readonly { role: string; content: string }[],
  onToken: (delta: string) => void,
  model?: string,
  maxTokens = 512,
): Promise<string> {
  const r = await fetch(chatEndpoint(svc), {
    method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ model: model ?? svc.models[0] ?? "local", messages, max_tokens: maxTokens, stream: true }),
  });
  if (!r.ok) throw new Error(`chat ${r.status}`);
  if (r.body === null) return chatDirect(svc, messages, model, maxTokens);

  const reader = r.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  let full = "";
  for (;;) {
    const chunk = await reader.read();
    if (chunk.done) break;
    buffer += decoder.decode(chunk.value, { stream: true });
    const frames = buffer.split("\n\n");
    buffer = frames.pop() ?? "";
    for (const frame of frames) {
      for (const line of frame.split("\n")) {
        const delta = sseContentDelta(line);
        if (delta !== null && delta.length > 0) {
          full += delta;
          onToken(delta);
        }
      }
    }
  }
  return full;
}

// How a BYO chat service is driven from the forge console:
//  - "direct": browser → service /v1 (plain chat, no tools; works from the static page)
//  - "tools":  browser → Crucible backend → service (full agent tool-loop; Crucible runs the tools)
export type ChatMode = "direct" | "tools";

// active BYO chat service (persisted) — null means use Crucible's own agent path
const KEY = "crucible_chat_service";
const MODE_KEY = "crucible_chat_mode";
export function getActiveChatService(): DetectedService | null {
  if (typeof localStorage === "undefined" || typeof localStorage.getItem !== "function") return null;
  const raw = localStorage.getItem(KEY);
  if (!raw) return null;
  try { return JSON.parse(raw) as DetectedService; } catch { return null; }
}
export function setActiveChatService(svc: DetectedService | null, mode: ChatMode = "direct"): void {
  if (typeof localStorage === "undefined" || typeof localStorage.setItem !== "function") return;
  if (svc) {
    localStorage.setItem(KEY, JSON.stringify(svc));
    localStorage.setItem(MODE_KEY, mode);
  } else {
    localStorage.removeItem(KEY);
    localStorage.removeItem(MODE_KEY);
  }
}
export function getChatMode(): ChatMode {
  if (typeof localStorage === "undefined" || typeof localStorage.getItem !== "function") return "direct";
  return localStorage.getItem(MODE_KEY) === "tools" ? "tools" : "direct";
}

// Register a BYO endpoint as a first-class Crucible registry model (needs a Crucible
// backend reachable at apiBase). Returns the registered id, or throws.
export async function connectService(
  apiBase: string, svc: DetectedService, token = "",
): Promise<string> {
  const id = `${svc.type}-${svc.baseUrl.replace(/^https?:\/\//, "").replace(/[^a-z0-9]+/gi, "-")}`;
  const headers: Record<string, string> = { "Content-Type": "application/json" };
  if (token) headers["Authorization"] = `Bearer ${token}`;
  const r = await fetch(apiBase.replace(/\/$/, "") + "/api/models/connect", {
    method: "POST", headers,
    body: JSON.stringify({ id, name: svc.name, endpoint: svc.baseUrl, notes: svc.note }),
  });
  if (!r.ok && r.status !== 409) throw new Error(`connect ${r.status}`);
  return id;
}
