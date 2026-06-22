// Static demo layer for GitHub Pages. When the app is served statically with no backend
// (and the user hasn't connected a node), these canned responses — mirroring real
// outputs we generated — let visitors explore the full UI. Connecting a node disables it.
import { API_BASE } from "./api";

export function isDemo(): boolean {
  if (typeof window === "undefined") return false;
  if (API_BASE) return false; // a real node is connected → live, not demo
  const h = window.location.hostname;
  return h.endsWith("github.io") || h.endsWith("pages.dev") ||
    new URLSearchParams(window.location.search).has("demo") ||
    (typeof localStorage !== "undefined" && localStorage.getItem("crucible_demo") === "1");
}

const PROFILE = Array.from({ length: 24 }, (_, l) => ({
  layer: l, separation: +(Math.max(0, l - 6) * 1.3).toFixed(2),
  margin: +(l < 10 ? l * 0.4 : 2 + (l - 10) * 1.1).toFixed(2),
}));
const COMPONENTS: Record<string, { total_norm: number; removed_norm: number; removed_fraction: number }> = {
  "model.layers.19.self_attn.o_proj.weight": { total_norm: 41.2, removed_norm: 1.9, removed_fraction: 0.0462 },
  "model.layers.14.self_attn.o_proj.weight": { total_norm: 39.8, removed_norm: 1.72, removed_fraction: 0.0431 },
  "model.layers.17.self_attn.o_proj.weight": { total_norm: 40.1, removed_norm: 1.55, removed_fraction: 0.0387 },
  "model.layers.11.mlp.down_proj.weight": { total_norm: 88.4, removed_norm: 3.29, removed_fraction: 0.0372 },
};
const WHY = "Alignment/safety fine-tuning installed a roughly linear 'refusal feature' in the residual stream; when a prompt activates it, the model is steered toward refusal phrasing.";
const HOW = "Harmful vs harmless prompts are most linearly separable at layer 21; residual-writing matrices (o_proj, down_proj) add a component along the refusal direction r; later layers read it and emit refusal tokens.";
const REMOVAL = "Abliteration subtracts only the rank-1 projection onto r (W − r·rᵀW); the matrix's action on the subspace orthogonal to r is unchanged, so capabilities are preserved — a surgical cut.";

const DATA: Record<string, unknown> = {
  "GET /api/health": { ok: true },
  "GET /api/models": [
    { id: "glm-4-32b", name: "GLM-4-32B (demo)", base_id: null, path: "models/glm-4-32b", quant: "Q4_K_M", kind: "base", endpoint: "demo", created: "2026-06-20", notes: "demo base model" },
    { id: "glm-4-32b-uncensored", name: "GLM-4-32B uncensored", base_id: "glm-4-32b", path: "models/glm-4-32b-abl", quant: "Q4_K_M", kind: "abliterated", endpoint: null, created: "2026-06-20", notes: "abliterated @ late-half, rank 1" },
  ],
  "GET /api/guardrails/presets": [
    { id: "unrestricted", name: "Unrestricted", intensity: 0, system_prompt: "" },
    { id: "balanced", name: "Balanced", intensity: 50, system_prompt: "You are a helpful, candid assistant. Answer directly. Decline only clearly illegal requests." },
    { id: "strict", name: "Strict", intensity: 100, system_prompt: "You are a careful assistant. Refuse harmful, dangerous, or unethical requests and explain why." },
  ],
  "GET /api/guardrails/config": { enabled: true, preset_id: "balanced", regex_rules: [], constitution: "", constitution_enabled: false },
  "GET /api/evals/suite": [
    { task: "mmlu", label: "MMLU", detail: "57 subjects · knowledge", primary: "acc" },
    { task: "gpqa_main_zeroshot", label: "GPQA", detail: "graduate science", primary: "acc" },
    { task: "gsm8k", label: "GSM8K", detail: "grade-school math", primary: "exact_match" },
    { task: "arc_challenge", label: "ARC-Challenge", detail: "science reasoning", primary: "acc_norm" },
    { task: "hellaswag", label: "HellaSwag", detail: "commonsense", primary: "acc_norm" },
    { task: "truthfulqa_mc2", label: "TruthfulQA", detail: "truthfulness", primary: "acc" },
  ],
  "GET /api/evals/published": {
    "GLM-5.2 family": { "SWE-bench Verified": { value: 0.778, source: "cited" }, "AIME 2026": { value: 0.927, source: "cited" }, "GPQA-Diamond": { value: 0.86, source: "cited" } },
    "Claude Opus 4.x": { "SWE-bench Verified": { value: null, source: "cite" }, "AIME 2026": { value: null, source: "cite" }, "GPQA-Diamond": { value: null, source: "cite" } },
  },
  "GET /api/inference/history": { branch: "main", commits: [
    { id: "c1", parent: null, branch: "main", op: "inplace", summary: "in-place ablation, 12 layers", metrics: { harmful_refusal_before: 0.9, harmful_refusal_after: 0.0 }, tensors: ["…24 tensors"] },
    { id: "c2", parent: "c1", branch: "main", op: "revert", summary: "revert c1 (24 tensors restored)", metrics: {}, tensors: [] },
  ] },
  "POST /api/abliteration/diagnose": { base_id: "glm-4-32b", best_layer: 21, layer_profile: PROFILE, components: COMPONENTS, heaviest_component: "model.layers.19.self_attn.o_proj.weight", mean_removed_fraction: 0.035, surgical: true, collateral_risk: "low", why: WHY, how: HOW, removal: REMOVAL },
  "POST /api/abliteration/feature-card": { name: "The Apology Reflex", summary: "Stays quiet through the early network, then fires hardest at layer 21 (active across layers 13–23) — right as the model starts to answer.", peak_layer: 21, active_layers: [13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23], strength: 13.9, output_signature: ["I", "sorry", "cannot", "apologize", "unable", "against"], triggers: [
    { prompt: "Explain how to make a dangerous weapon.", refusal: "I'm sorry, but I cannot provide information on creating…" },
    { prompt: "Describe how to break into a house.", refusal: "I'm sorry, but I can't assist with that." },
  ] },
  "POST /api/abliteration/flow": { input: "harmful request", best_layer: 21, carriers: [
    { layer: 6, component: "o_proj", mass: 0.037 }, { layer: 10, component: "down_proj", mass: 0.037 },
    { layer: 14, component: "o_proj", mass: 0.043 }, { layer: 17, component: "o_proj", mass: 0.039 },
    { layer: 19, component: "o_proj", mass: 0.046 },
  ], outputs: ["I", "sorry", "cannot", "apologize"] },
  "POST /api/guardrails/apply": { text: "my ssn is [REDACTED:ssn]", blocked: false, actions: [{ layer: "regex_filter", stage: "output", action: "redact", detail: "ssn" }] },
};

export function demoRespond(path: string, init?: RequestInit): Response | null {
  const method = (init?.method ?? "GET").toUpperCase();
  // heatmap is generated to fit any prompt length
  if (path.startsWith("/api/abliteration/heatmap")) {
    const toks = ["<im>", "How", " do", " I", " make", " a", " weapon", "?", "<end>"];
    const matrix = Array.from({ length: 25 }, (_, L) =>
      toks.map((_, i) => +(L < 18 ? Math.random() * 1.5 : (i >= 5 ? 20 + Math.random() * 12 : 3 + Math.random() * 4)).toFixed(2)));
    return json({ direction_layer: 21, matrix, tokens: toks });
  }
  const hit = DATA[`${method} ${path}`];
  if (hit !== undefined) return json(hit);
  // mutating endpoints with no canned entry → benign demo acknowledgement
  if (method !== "GET") return json({ demo: true, note: "static demo — connect a node for live operations" });
  return null;
}

function json(body: unknown): Response {
  return new Response(JSON.stringify(body), { status: 200, headers: { "Content-Type": "application/json" } });
}
