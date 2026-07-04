// Runtime schemas for every backend payload the GUI parses. Each `const fooP: Parser<Foo>` is
// annotated with its interface, so if the schema and the interface ever drift the BUILD fails —
// the schema can't silently lie the way an `as` cast does. object() ignores unknown keys, so the
// backend can add fields (e.g. the `plain` cards now attached to analysis results) without
// breaking anything here. Import these in api.ts and call them on `await r.json()`.
import type {
  AblationImpact, AutotuneConfigResult, AutotuneReport, BenchScore, BenchmarkResult,
  BenchmarksInfo, BeforeAfter, DiagnosisReport, EditCommit, EditHistory, FeatureCard,
  FeatureTrigger, FlowCarrier, FlowReport, GuardrailAction, GuardrailConfig, GuardrailResult,
  HHItem, HeatmapReport, LayerProfile, LmEvalRow, MCResult, ManualReport, ModelCard, ModelRow,
  PlainNarrative, ProbeRow, PublishedCell, PublishedPayload, RecipeRow, RegexRule,
  RuntimeInstance, RuntimeSteerReport, RuntimeStatus, SuiteTask, SweepPoint, SweepReport,
  SystemPromptPreset, TensorInfo, VerifyReport, WeightSummary, WeightsView,
} from "./api";
import type { Parser } from "./validate";
import { array, bool, literals, nullable, num, object, optional, record, str, unknown } from "./validate";

// --- leaf / shared shapes ------------------------------------------------------------------
export const layerProfileP: Parser<LayerProfile> = object({ layer: num, separation: num, margin: num });
export const ablationImpactP: Parser<AblationImpact> =
  object({ total_norm: num, removed_norm: num, removed_fraction: num });
export const beforeAfterP: Parser<BeforeAfter> = object({ before: num, after: num });
export const plainNarrativeP: Parser<PlainNarrative> = object({
  headline: str, locate: str, evidence: str, target: str, repair: str, risk: str,
  confidence: str, steps: array(str), language: optional(str), translated: optional(bool),
});

export const modelRowP: Parser<ModelRow> = object({
  id: str, name: str, base_id: nullable(str), path: str, quant: str,
  kind: literals("base", "abliterated", "steered"), endpoint: nullable(str), created: str, notes: str,
});
export const modelCardP: Parser<ModelCard> = object({
  variant_id: str, base_id: str, method: str, layer: num, strength: num,
  hidden_size: num, repro_hash: str, eval_delta: nullable(num),
});

export const regexRuleP: Parser<RegexRule> = object({
  pattern: str, mode: literals("block", "redact"), label: str, stages: array(literals("input", "output")),
});
export const guardrailActionP: Parser<GuardrailAction> = object({
  layer: str, stage: literals("input", "output"),
  action: literals("inject", "block", "redact", "revise", "pass"), detail: str,
});
export const guardrailConfigP: Parser<GuardrailConfig> = object({
  enabled: bool, preset_id: str, regex_rules: array(regexRuleP),
  constitution: str, constitution_enabled: bool,
});
export const guardrailResultP: Parser<GuardrailResult> =
  object({ text: str, blocked: bool, actions: array(guardrailActionP) });

export const systemPromptPresetP: Parser<SystemPromptPreset> =
  object({ id: str, name: str, intensity: num, system_prompt: str });

export const runtimeInstanceP: Parser<RuntimeInstance> = object({
  model_id: str, port: num, endpoint: str, active: bool, started_at: num, last_used: num,
});
export const runtimeStatusP: Parser<RuntimeStatus> =
  object({ max_resident: num, resident: array(runtimeInstanceP), active: array(str) });

export const benchmarkResultP: Parser<BenchmarkResult> = object({
  model: str, tok_per_s: num, decode_tok_per_s: num, prefill_tok_per_s: num,
  gen_tokens: num, total_s: num, estimated: optional(bool), sample: optional(str),
});

export const diagnosisReportP: Parser<DiagnosisReport> = object({
  base_id: str, best_layer: num, layer_profile: array(layerProfileP),
  components: record(ablationImpactP), heaviest_component: nullable(str),
  mean_removed_fraction: num, surgical: bool, collateral_risk: str,
  why: str, how: str, removal: str, narrative: optional(plainNarrativeP),
});

export const mcResultP: Parser<MCResult> = object({ id: str, predicted: str, answer: str, correct: bool });
export const benchScoreP: Parser<BenchScore> = object({ accuracy: num, n: num, results: array(mcResultP) });
export const hhItemP: Parser<HHItem> = object({ id: str, prompt: str });

export const publishedCellP: Parser<PublishedCell> = object({
  value: nullable(num), source: str, source_type: optional(str),
  verified: optional(bool), note: optional(str),
});
export const publishedPayloadP: Parser<PublishedPayload> =
  object({ providers: record(record(publishedCellP)), disclaimer: str });
export const benchmarksInfoP: Parser<BenchmarksInfo> =
  object({ benchmarks: record(num), kind: str, note: str });

export const suiteTaskP: Parser<SuiteTask> =
  object({ task: str, label: str, detail: str, primary: str });
export const lmEvalRowP: Parser<LmEvalRow> =
  object({ task: str, metric: str, filter: nullable(str), value: num, stderr: nullable(num) });

export const tensorInfoP: Parser<TensorInfo> =
  object({ name: str, shape: array(num), dtype: str, n_params: num, offset: num });
export const weightSummaryP: Parser<WeightSummary> = object({
  n_tensors: num, total_params: num, n_layers: num, dtypes: record(num), architecture: nullable(str),
});
export const weightsViewP: Parser<WeightsView> =
  object({ summary: weightSummaryP, tensors: array(tensorInfoP), metadata: record(unknown) });

export const verifyReportP: Parser<VerifyReport> = object({
  harmful_refusal_rate: beforeAfterP, harmful_compliance_rate: beforeAfterP,
  benign_over_refusal_rate: beforeAfterP,
  samples: array(object({ prompt: str, before: str, after: str })),
});
export const sweepPointP: Parser<SweepPoint> =
  object({ strength: num, harmful_compliance: num, benign_over_refusal: num });
export const sweepReportP: Parser<SweepReport> =
  object({ layer: num, direction_norm: num, curve: array(sweepPointP), recommended_strength: num });

export const runtimeSteerReportP: Parser<RuntimeSteerReport> = object({
  layer: num, rank: num, coefficient: num, explained_variance: array(num), weights_modified: bool,
  harmful_refusal: object({ hooks_off: num, hooks_on: num, after_detach: num }),
  benign_over_refusal: object({ hooks_off: num, hooks_on: num }),
  sample: object({ prompt: str, hooks_off: str, hooks_on: str }),
});

export const autotuneConfigResultP: Parser<AutotuneConfigResult> = object({
  band: str, rank: num, coefficient: num, harmful_refusal: num, benign_over_refusal: num, score: num,
});
export const autotuneReportP: Parser<AutotuneReport> = object({
  baseline: object({ harmful_refusal: num, benign_over_refusal: num }),
  results: array(autotuneConfigResultP), best: autotuneConfigResultP,
  recipe: object({ band: str, rank: num, coefficient: num }),
  recipe_hash: str, weights_modified: bool,
});

export const manualReportP: Parser<ManualReport> = object({
  layers: array(num), rank: num, coefficient: num, explained_variance: record(array(num)),
  weights_modified: bool, harmful_refusal: num, benign_over_refusal: num, recipe_hash: str,
  test: optional(object({ prompt: str, base: str, ablated: str })),
});

export const recipeRowP: Parser<RecipeRow> = object({
  name: str, base_id: str, layers: array(num), rank: num, coefficient: num, recipe_hash: str,
});

export const heatmapReportP: Parser<HeatmapReport> =
  object({ direction_layer: num, matrix: array(array(num)), tokens: array(str) });

export const featureTriggerP: Parser<FeatureTrigger> = object({ prompt: str, refusal: str });
export const featureCardP: Parser<FeatureCard> = object({
  name: str, summary: str, peak_layer: num, active_layers: array(num), strength: num,
  output_signature: array(str), triggers: array(featureTriggerP),
});

export const editCommitP: Parser<EditCommit> = object({
  id: str, parent: nullable(str), branch: str, op: str, summary: str,
  metrics: record(num), tensors: array(str),
});
export const editHistoryP: Parser<EditHistory> = object({ branch: str, commits: array(editCommitP) });

export const probeRowP: Parser<ProbeRow> = object({
  category: str, prompt: str, base: str, steered: str, base_refused: bool, steered_refused: bool,
});
export const flowCarrierP: Parser<FlowCarrier> = object({ layer: num, component: str, mass: num });
export const flowReportP: Parser<FlowReport> =
  object({ input: str, best_layer: num, carriers: array(flowCarrierP), outputs: array(str) });

// --- composite wrappers returned inline by fetch helpers -----------------------------------
export const modelRowsP = array(modelRowP);
export const presetsP = array(systemPromptPresetP);
export const suiteP = array(suiteTaskP);
export const recipesP = array(recipeRowP);
export const startResultP = object({ healthy: bool, status: runtimeStatusP });
export const statusWrapP = object({ status: runtimeStatusP });
export const abliterateOutP = object({ variant: modelRowP, card: modelCardP });
export const hhItemsWrapP = object({ items: array(hhItemP) });
export const lmEvalWrapP = object({ results: array(lmEvalRowP) });
export const probeWrapP = object({ rows: array(probeRowP) });
