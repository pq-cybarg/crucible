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

// active BYO chat service (persisted) — null means use Crucible's own agent path
const KEY = "crucible_chat_service";
export function getActiveChatService(): DetectedService | null {
  if (typeof localStorage === "undefined" || typeof localStorage.getItem !== "function") return null;
  const raw = localStorage.getItem(KEY);
  if (!raw) return null;
  try { return JSON.parse(raw) as DetectedService; } catch { return null; }
}
export function setActiveChatService(svc: DetectedService | null): void {
  if (typeof localStorage === "undefined" || typeof localStorage.setItem !== "function") return;
  if (svc) localStorage.setItem(KEY, JSON.stringify(svc));
  else localStorage.removeItem(KEY);
}
