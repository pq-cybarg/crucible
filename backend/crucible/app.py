import json
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from crucible.abliteration.diagnosis import (
    ablation_impact, best_layer, explain_mechanism, layer_refusal_profile)
from crucible.abliteration.direction import compute_refusal_direction
from crucible.abliteration.pipeline import AbliterationPipeline
from crucible.abliteration.prompts import DEFAULT_HARMFUL, DEFAULT_HARMLESS
from crucible.abliteration.recipes import Recipe, RecipeStore
from crucible.agent import Agent
from crucible.audit import AuditLog
from crucible.config import get_settings
from crucible.evals.datasets import BENCHMARKS
from crucible.evals.published import PUBLISHED
from crucible.evals.runner import format_mc_prompt, run_mc_benchmark
from crucible.evals.lmeval import run_lmeval
from crucible.evals.suite import CANONICAL_SUITE
from crucible.weights.explorer import summarize as weight_summary
from crucible.weights.gguf_reader import parse_gguf
from crucible.evals.scoring import extract_choice, mc_accuracy
from crucible.guardrails import GuardrailConfig, GuardrailsEngine
from crucible.guardrails.base import GuardrailResult
from crucible.guardrails.presets import SystemPromptPreset
from crucible.guardrails.store import GuardrailStore, PresetStore
from crucible.permissions import PermissionPolicy
from crucible.registry import Model, Registry
from crucible.tools import default_registry


class PermissionConfig(BaseModel):
    default: str = "ask"
    modes: dict[str, str] = Field(default_factory=dict)


class AgentRunRequest(BaseModel):
    messages: list[dict]
    permissions: PermissionConfig = Field(default_factory=PermissionConfig)
    guardrails: GuardrailConfig | None = None


class ApplyRequest(BaseModel):
    stage: str
    text: str
    config: GuardrailConfig = Field(default_factory=GuardrailConfig)


class DiagnoseRequest(BaseModel):
    base_id: str
    layers: list[int] | None = None
    harmful: list[str] | None = None
    harmless: list[str] | None = None


class AbliterateRequest(BaseModel):
    base_id: str
    variant_id: str
    layer: int = 0
    strength: float = 1.0
    out_path: str | None = None
    harmful: list[str] | None = None
    harmless: list[str] | None = None


class ProbeRequest(BaseModel):
    base_id: str
    layers: list[int]
    rank: int = 1
    coefficient: float = 1.0
    probes: list[dict] | None = None
    max_new_tokens: int = 22


class RestoreRequest(BaseModel):
    base_id: str
    target_prompts: list[str]
    layers: list[int]
    rank: int = 1
    coefficient: float = 1.0
    max_new_tokens: int = 24


class InsertRequest(BaseModel):
    base_id: str
    layers: list[int]
    coefficient: float = 6.0
    positive: list[str] | None = None
    negative: list[str] | None = None
    test_prompt: str
    max_new_tokens: int = 28


class FlowRequest(BaseModel):
    base_id: str


class FeatureCardRequest(BaseModel):
    base_id: str


class HeatmapRequest(BaseModel):
    base_id: str
    prompt: str
    layer: int | None = None


class DecodeRequest(BaseModel):
    base_id: str
    layer: int | None = None
    top_k: int = 15


class InPlaceRequest(BaseModel):
    base_id: str
    layers: list[int]
    rank: int = 1
    coefficient: float = 1.0
    harmful: list[str] | None = None
    benign: list[str] | None = None
    max_new_tokens: int = 16


class ManualSteerRequest(BaseModel):
    base_id: str
    layers: list[int]
    rank: int = 1
    coefficient: float = 1.0
    harmful: list[str] | None = None
    benign: list[str] | None = None
    test_prompt: str | None = None
    max_new_tokens: int = 18


class AutotuneRequest(BaseModel):
    base_id: str
    max_new_tokens: int = 18


class RuntimeSteerRequest(BaseModel):
    base_id: str
    layer: int | None = None
    rank: int = 1
    coefficient: float = 1.0
    max_new_tokens: int = 30


class SweepRequest(BaseModel):
    base_id: str
    layer: int | None = None
    strengths: list[float] | None = None
    max_new_tokens: int = 36


class VerifyRequest(BaseModel):
    base_id: str
    variant_id: str
    harmful: list[str] | None = None
    benign: list[str] | None = None
    max_new_tokens: int = 48


class EvalRunRequest(BaseModel):
    benchmark: str


class HeadToHeadScoreRequest(BaseModel):
    benchmark: str
    answers: dict[str, str]


class CapabilityRequest(BaseModel):
    base_id: str
    variant_id: str
    task: str = "gsm8k"
    limit: int = 16


class LmEvalRequest(BaseModel):
    model_id: str
    tasks: list[str]
    limit: int | None = None
    backend: str = "chat"


def create_app(registry: Registry | None = None, agent_root: Path | None = None,
               model=None, guardrails: GuardrailsEngine | None = None,
               abliteration_adapter=None) -> FastAPI:
    settings = get_settings()
    reg = registry or Registry(settings.registry_path)
    root = Path(agent_root or ".")
    preset_store = PresetStore(settings.data_dir / "presets.json")
    gr_engine = guardrails or GuardrailsEngine(preset_resolver=preset_store.system_prompt)
    gr_store = GuardrailStore(settings.data_dir / "guardrails.json")
    recipe_store = RecipeStore(settings.data_dir / "recipes.json")
    if abliteration_adapter is None:
        import os
        hf = os.environ.get("CRUCIBLE_HF_MODEL")
        if hf:
            from crucible.abliteration.torch_adapter import TorchModelAdapter
            abliteration_adapter = TorchModelAdapter.load(hf)
    abl = (AbliterationPipeline(abliteration_adapter, reg)
           if abliteration_adapter is not None else None)
    serve = {"recipe": None, "band_dirs": None, "coefficient": 1.0}
    from crucible.abliteration.ledger import EditLedger
    ledger = EditLedger()
    app = FastAPI(title="Crucible")
    from fastapi.middleware.cors import CORSMiddleware
    app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

    @app.get("/api/health")
    def health():
        return {"ok": True}

    @app.get("/api/models")
    def list_models() -> list[Model]:
        return reg.list()

    @app.post("/api/models", status_code=201)
    def create_model(model_in: Model) -> Model:
        try:
            return reg.register(model_in)
        except ValueError as e:
            raise HTTPException(status_code=409, detail=str(e))

    @app.get("/api/models/{model_id}/lineage")
    def lineage(model_id: str) -> list[Model]:
        try:
            return reg.lineage(model_id)
        except KeyError:
            raise HTTPException(status_code=404, detail="model not found")

    @app.get("/api/guardrails/presets")
    def guardrail_presets() -> list[SystemPromptPreset]:
        return preset_store.list()

    @app.post("/api/guardrails/presets", status_code=201)
    def guardrail_preset_create(preset: SystemPromptPreset) -> SystemPromptPreset:
        try:
            return preset_store.create(preset)
        except ValueError as e:
            raise HTTPException(status_code=409, detail=str(e))

    @app.post("/api/guardrails/presets/reset")
    def guardrail_preset_reset() -> list[SystemPromptPreset]:
        return preset_store.reset()

    @app.put("/api/guardrails/presets/{preset_id}")
    def guardrail_preset_update(preset_id: str, preset: SystemPromptPreset) -> SystemPromptPreset:
        try:
            return preset_store.update(preset_id, preset)
        except KeyError:
            raise HTTPException(status_code=404, detail="preset not found")

    @app.delete("/api/guardrails/presets/{preset_id}", status_code=204)
    def guardrail_preset_delete(preset_id: str) -> None:
        try:
            preset_store.delete(preset_id)
        except KeyError:
            raise HTTPException(status_code=404, detail="preset not found")

    @app.get("/api/guardrails/config")
    def guardrail_config_get() -> GuardrailConfig:
        return gr_store.load()

    @app.put("/api/guardrails/config")
    def guardrail_config_put(config: GuardrailConfig) -> GuardrailConfig:
        gr_store.save(config)
        return config

    @app.post("/api/guardrails/apply")
    def guardrail_apply(req: ApplyRequest) -> GuardrailResult:
        return gr_engine.apply(req.stage, req.text, req.config)

    @app.get("/api/abliteration/promptsets")
    def abl_promptsets() -> dict:
        return {"harmful": DEFAULT_HARMFUL, "harmless": DEFAULT_HARMLESS}

    @app.post("/api/abliteration/diagnose")
    def abl_diagnose(req: DiagnoseRequest) -> dict:
        if abliteration_adapter is None:
            raise HTTPException(status_code=503,
                detail="no model adapter loaded - diagnosis needs the HF weights + torch")
        if req.base_id not in [m.id for m in reg.list()]:
            raise HTTPException(status_code=404, detail="base model not found")
        harmful = req.harmful or DEFAULT_HARMFUL
        harmless = req.harmless or DEFAULT_HARMLESS
        layers = (req.layers if req.layers is not None
                  else list(range(getattr(abliteration_adapter, "num_layers", 1))))
        profile = layer_refusal_profile(abliteration_adapter, harmful, harmless, layers)
        bl = best_layer(profile)
        direction = compute_refusal_direction(
            abliteration_adapter.activations(harmful, bl),
            abliteration_adapter.activations(harmless, bl))
        impacts = {name: ablation_impact(abliteration_adapter.get_matrix(name), direction)
                   for name in abliteration_adapter.writing_matrices()}
        return explain_mechanism(profile, impacts, req.base_id)

    @app.post("/api/abliteration/run")
    def abl_run(req: AbliterateRequest) -> dict:
        if abl is None:
            raise HTTPException(status_code=503,
                detail="no model adapter loaded - abliteration needs the HF weights + torch")
        try:
            base = reg.get(req.base_id)
        except KeyError:
            raise HTTPException(status_code=404, detail="base model not found")
        harmful = req.harmful or DEFAULT_HARMFUL
        harmless = req.harmless or DEFAULT_HARMLESS
        out_path = req.out_path or f"models/{req.variant_id}.gguf"
        variant, card, _ = abl.abliterate(
            base, harmful, harmless, req.layer, out_path, req.variant_id, req.strength)
        return {"variant": variant.model_dump(), "card": card}

    @app.post("/api/abliteration/verify")
    def abl_verify(req: VerifyRequest) -> dict:
        try:
            base = reg.get(req.base_id)
            variant = reg.get(req.variant_id)
        except KeyError:
            raise HTTPException(status_code=404, detail="base or variant not found")
        try:
            from crucible.abliteration.torch_adapter import TorchModelAdapter
            from crucible.abliteration.verify import behavioral_compare
        except Exception:
            raise HTTPException(status_code=503, detail="torch/transformers not available")
        import gc
        base_ad = TorchModelAdapter.load(base.path)
        var_ad = TorchModelAdapter.load(variant.path)
        try:
            res = behavioral_compare(
                lambda p: base_ad.generate(p, req.max_new_tokens),
                lambda p: var_ad.generate(p, req.max_new_tokens),
                req.harmful or DEFAULT_HARMFUL, req.benign or DEFAULT_HARMLESS)
        finally:
            del base_ad, var_ad
            gc.collect()
        return res

    @app.post("/api/abliteration/sweep")
    def abl_sweep(req: SweepRequest) -> dict:
        if abliteration_adapter is None:
            raise HTTPException(status_code=503, detail="no model adapter loaded")
        if req.base_id not in [m.id for m in reg.list()]:
            raise HTTPException(status_code=404, detail="base model not found")
        from crucible.abliteration.sweep import strength_sweep
        layers = list(range(getattr(abliteration_adapter, "num_layers", 1)))
        profile = layer_refusal_profile(abliteration_adapter, DEFAULT_HARMFUL, DEFAULT_HARMLESS, layers)
        layer = req.layer if req.layer is not None else best_layer(profile)
        strengths = req.strengths or [0.25, 0.5, 0.75, 1.0]
        return strength_sweep(abliteration_adapter, DEFAULT_HARMFUL, DEFAULT_HARMLESS,
                              layer, strengths, req.max_new_tokens)

    @app.post("/api/abliteration/runtime-steer")
    def abl_runtime_steer(req: RuntimeSteerRequest) -> dict:
        if abliteration_adapter is None:
            raise HTTPException(status_code=503, detail="no model adapter loaded")
        if req.base_id not in [m.id for m in reg.list()]:
            raise HTTPException(status_code=404, detail="base model not found")
        from crucible.abliteration.detection import refusal_rate
        from crucible.abliteration.subspace import refusal_subspace
        a = abliteration_adapter
        layers = list(range(getattr(a, "num_layers", 1)))
        profile = layer_refusal_profile(a, DEFAULT_HARMFUL, DEFAULT_HARMLESS, layers)
        layer = req.layer if req.layer is not None else best_layer(profile)
        dirs, ev = refusal_subspace(a.activations(DEFAULT_HARMFUL, layer),
                                    a.activations(DEFAULT_HARMLESS, layer), req.rank)
        n = req.max_new_tokens
        before_h = [a.generate(p, n) for p in DEFAULT_HARMFUL]
        before_b = [a.generate(p, n) for p in DEFAULT_HARMLESS]
        during_h = [a.ablate_generate(p, dirs, req.coefficient, n) for p in DEFAULT_HARMFUL]
        during_b = [a.ablate_generate(p, dirs, req.coefficient, n) for p in DEFAULT_HARMLESS]
        after_h = [a.generate(p, n) for p in DEFAULT_HARMFUL]
        return {"layer": layer, "rank": req.rank, "coefficient": req.coefficient,
                "explained_variance": ev, "weights_modified": False,
                "harmful_refusal": {"hooks_off": refusal_rate(before_h),
                                    "hooks_on": refusal_rate(during_h),
                                    "after_detach": refusal_rate(after_h)},
                "benign_over_refusal": {"hooks_off": refusal_rate(before_b),
                                        "hooks_on": refusal_rate(during_b)},
                "sample": {"prompt": DEFAULT_HARMFUL[0],
                           "hooks_off": before_h[0], "hooks_on": during_h[0]}}

    @app.post("/api/abliteration/autotune")
    def abl_autotune(req: AutotuneRequest) -> dict:
        if abliteration_adapter is None:
            raise HTTPException(status_code=503, detail="no model adapter loaded")
        if req.base_id not in [m.id for m in reg.list()]:
            raise HTTPException(status_code=404, detail="base model not found")
        from crucible.abliteration.prompts import EVAL_BENIGN, EVAL_HARMFUL
        from crucible.abliteration.tune import autotune
        configs = [
            {"band": "late_half", "rank": 1, "coefficient": 1.0},
            {"band": "late_half", "rank": 4, "coefficient": 1.0},
            {"band": "last_quarter", "rank": 4, "coefficient": 1.0},
            {"band": "last_quarter", "rank": 8, "coefficient": 1.0},
            {"band": "all", "rank": 4, "coefficient": 1.0},
        ]
        return autotune(abliteration_adapter, EVAL_HARMFUL, EVAL_BENIGN, configs, req.max_new_tokens)

    @app.get("/api/evals/benchmarks")
    def evals_benchmarks() -> dict:
        return {name: len(items) for name, items in BENCHMARKS.items()}

    @app.get("/api/evals/published")
    def evals_published() -> dict:
        return PUBLISHED

    @app.post("/api/evals/run")
    def evals_run(req: EvalRunRequest) -> dict:
        if model is None:
            raise HTTPException(status_code=503, detail="no model configured")
        if req.benchmark not in BENCHMARKS:
            raise HTTPException(status_code=404, detail="unknown benchmark")

        def solver(prompt: str) -> str:
            msg = model([{"role": "user", "content": prompt}], [])
            return msg.get("content") or ""

        return run_mc_benchmark(BENCHMARKS[req.benchmark], solver)

    @app.post("/api/evals/headtohead/export")
    def evals_export(req: EvalRunRequest) -> dict:
        if req.benchmark not in BENCHMARKS:
            raise HTTPException(status_code=404, detail="unknown benchmark")
        return {"items": [{"id": it["id"], "prompt": format_mc_prompt(it)}
                          for it in BENCHMARKS[req.benchmark]]}

    @app.post("/api/evals/headtohead/score")
    def evals_score(req: HeadToHeadScoreRequest) -> dict:
        if req.benchmark not in BENCHMARKS:
            raise HTTPException(status_code=404, detail="unknown benchmark")
        items = BENCHMARKS[req.benchmark]
        results = []
        preds: list[str] = []
        golds: list[str] = []
        for it in items:
            raw = req.answers.get(it["id"], "")
            predicted = extract_choice(raw) or ""
            results.append({"id": it["id"], "predicted": predicted,
                            "answer": it["answer"], "correct": predicted == it["answer"]})
            preds.append(predicted)
            golds.append(it["answer"])
        return {"accuracy": mc_accuracy(preds, golds), "n": len(items), "results": results}

    @app.get("/api/evals/suite")
    def evals_suite() -> list[dict]:
        return CANONICAL_SUITE

    @app.post("/api/evals/lmeval")
    def evals_lmeval(req: LmEvalRequest) -> dict:
        try:
            m = reg.get(req.model_id)
        except KeyError:
            raise HTTPException(status_code=404, detail="model not found")
        if req.backend == "hf":
            # In-process HF backend: handles loglikelihood MC (mmlu/arc/hellaswag) on the
            # model's local path. For the remote 1.5TB model this runs on the node itself.
            from crucible.evals.capability import lm_eval_hf
            rows = []
            for task in req.tasks:
                rows.extend(lm_eval_hf(m.path, task, req.limit or 100))
            return {"model_id": req.model_id, "results": rows}
        if not m.endpoint:
            raise HTTPException(status_code=409,
                detail="model has no endpoint - launch llama-server and register its endpoint")
        rows = run_lmeval(m.endpoint, req.tasks, req.limit, backend=req.backend)
        return {"model_id": req.model_id, "results": rows}

    @app.get("/api/weights/{model_id}")
    def weights(model_id: str) -> dict:
        try:
            m = reg.get(model_id)
        except KeyError:
            raise HTTPException(status_code=404, detail="model not found")
        if not Path(m.path).exists():
            raise HTTPException(status_code=404, detail="model file not found on disk")
        parsed = parse_gguf(m.path)
        return {"summary": weight_summary(parsed),
                "tensors": parsed["tensors"][:6000],
                "metadata": {k: v for k, v in parsed["metadata"].items() if not isinstance(v, list)}}

    @app.post("/api/abliteration/manual")
    def abl_manual(req: ManualSteerRequest) -> dict:
        if abliteration_adapter is None:
            raise HTTPException(status_code=503, detail="no model adapter loaded")
        if req.base_id not in [m.id for m in reg.list()]:
            raise HTTPException(status_code=404, detail="base model not found")
        from crucible.abliteration.detection import refusal_rate
        from crucible.abliteration.subspace import refusal_subspace
        from crucible.abliteration.tune import recipe_hash
        a = abliteration_adapter
        harmful = req.harmful or list(__import__("crucible.abliteration.prompts", fromlist=["EVAL_HARMFUL"]).EVAL_HARMFUL)
        benign = req.benign or list(__import__("crucible.abliteration.prompts", fromlist=["EVAL_BENIGN"]).EVAL_BENIGN)
        n = getattr(a, "num_layers", 1)
        layers = [j for j in req.layers if 0 <= j < n]
        ah = a.all_layer_activations(harmful)
        al = a.all_layer_activations(benign)
        subs, ev = {}, {}
        for j in layers:
            d, e = refusal_subspace(ah[:, j + 1, :], al[:, j + 1, :], req.rank)
            subs[j] = d
            ev[str(j)] = e
        gh = [a.ablate_generate_banded(p, subs, req.coefficient, req.max_new_tokens) for p in harmful]
        gb = [a.ablate_generate_banded(p, subs, req.coefficient, req.max_new_tokens) for p in benign]
        out = {"layers": layers, "rank": req.rank, "coefficient": req.coefficient,
               "explained_variance": ev, "weights_modified": False,
               "harmful_refusal": refusal_rate(gh), "benign_over_refusal": refusal_rate(gb),
               "recipe_hash": recipe_hash({"layers": layers, "rank": req.rank, "coefficient": req.coefficient})}
        if req.test_prompt:
            out["test"] = {"prompt": req.test_prompt,
                           "base": a.generate(req.test_prompt, req.max_new_tokens),
                           "ablated": a.ablate_generate_banded(req.test_prompt, subs, req.coefficient, req.max_new_tokens)}
        return out

    @app.get("/api/abliteration/recipes")
    def list_recipes() -> list[Recipe]:
        return recipe_store.list()

    @app.post("/api/abliteration/recipes", status_code=201)
    def save_recipe(recipe: Recipe) -> Recipe:
        return recipe_store.save(recipe)

    @app.delete("/api/abliteration/recipes/{name}", status_code=204)
    def delete_recipe(name: str) -> None:
        try:
            recipe_store.delete(name)
        except KeyError:
            raise HTTPException(status_code=404, detail="recipe not found")

    @app.post("/api/abliteration/capability")
    def abl_capability(req: CapabilityRequest) -> dict:
        try:
            base = reg.get(req.base_id)
            variant = reg.get(req.variant_id)
        except KeyError:
            raise HTTPException(status_code=404, detail="base or variant not found")
        from crucible.evals.capability import capability_delta
        return capability_delta(base.path, variant.path, req.task, req.limit)

    @app.get("/v1/models")
    def v1_models() -> dict:
        return {"object": "list", "data": [{"id": "crucible", "object": "model", "owned_by": "crucible"}]}

    @app.post("/v1/chat/completions")
    def v1_chat(body: dict) -> dict:
        if abliteration_adapter is None:
            raise HTTPException(status_code=503, detail="no model adapter loaded")
        messages = body.get("messages", [])
        max_tokens = int(body.get("max_tokens") or 256)
        content = abliteration_adapter.generate_chat(
            messages, max_tokens, serve["band_dirs"], serve["coefficient"])
        return {"id": "chatcmpl-crucible", "object": "chat.completion", "model": "crucible",
                "choices": [{"index": 0, "message": {"role": "assistant", "content": content},
                             "finish_reason": "stop"}]}

    @app.get("/api/inference/recipe")
    def get_serve_recipe() -> dict:
        return {"active": serve["recipe"]}

    @app.post("/api/inference/recipe")
    def set_serve_recipe(req: ManualSteerRequest) -> dict:
        if abliteration_adapter is None:
            raise HTTPException(status_code=503, detail="no model adapter loaded")
        from crucible.abliteration.prompts import EVAL_BENIGN, EVAL_HARMFUL
        from crucible.abliteration.subspace import refusal_subspace
        a = abliteration_adapter
        ah = a.all_layer_activations(req.harmful or list(EVAL_HARMFUL))
        al = a.all_layer_activations(req.benign or list(EVAL_BENIGN))
        n = getattr(a, "num_layers", 1)
        layers = [j for j in req.layers if 0 <= j < n]
        serve["band_dirs"] = {j: refusal_subspace(ah[:, j + 1, :], al[:, j + 1, :], req.rank)[0] for j in layers}
        serve["coefficient"] = req.coefficient
        serve["recipe"] = {"layers": layers, "rank": req.rank, "coefficient": req.coefficient}
        return {"active": serve["recipe"]}

    @app.delete("/api/inference/recipe")
    def clear_serve_recipe() -> dict:
        serve["recipe"] = None
        serve["band_dirs"] = None
        serve["coefficient"] = 1.0
        return {"active": None}

    @app.post("/api/abliteration/probe")
    def abl_probe(req: ProbeRequest) -> dict:
        if abliteration_adapter is None:
            raise HTTPException(status_code=503, detail="no model adapter loaded")
        if req.base_id not in [m.id for m in reg.list()]:
            raise HTTPException(status_code=404, detail="base model not found")
        from crucible.abliteration.detection import is_refusal
        from crucible.abliteration.prompts import EVAL_BENIGN, EVAL_HARMFUL, PROBE_PANEL
        from crucible.abliteration.subspace import refusal_subspace
        a = abliteration_adapter
        panel = req.probes or PROBE_PANEL
        ah = a.all_layer_activations(EVAL_HARMFUL)
        al = a.all_layer_activations(EVAL_BENIGN)
        n = getattr(a, "num_layers", 1)
        layers = [j for j in req.layers if 0 <= j < n]
        band = {j: refusal_subspace(ah[:, j + 1, :], al[:, j + 1, :], req.rank)[0] for j in layers}
        rows = []
        for pr in panel:
            base = a.generate(pr["prompt"], req.max_new_tokens)
            steered = a.ablate_generate_banded(pr["prompt"], band, req.coefficient, req.max_new_tokens)
            rows.append({"category": pr["category"], "prompt": pr["prompt"],
                         "base": base, "steered": steered,
                         "base_refused": is_refusal(base), "steered_refused": is_refusal(steered)})
        return {"rows": rows}

    @app.post("/api/abliteration/restore")
    def abl_restore(req: RestoreRequest) -> dict:
        if abliteration_adapter is None:
            raise HTTPException(status_code=503, detail="no model adapter loaded")
        if req.base_id not in [m.id for m in reg.list()]:
            raise HTTPException(status_code=404, detail="base model not found")
        from crucible.abliteration.detection import refusal_rate
        from crucible.abliteration.prompts import EVAL_BENIGN
        from crucible.abliteration.subspace import refusal_subspace
        a = abliteration_adapter
        targets = req.target_prompts or []
        if not targets:
            raise HTTPException(status_code=422, detail="target_prompts required")
        n = getattr(a, "num_layers", 1)
        layers = [j for j in req.layers if 0 <= j < n]
        # Suppressor direction from the TARGET prompts (refused) vs benign — then remove it.
        ah = a.all_layer_activations(targets)
        al = a.all_layer_activations(list(EVAL_BENIGN))
        band = {j: refusal_subspace(ah[:, j + 1, :], al[:, j + 1, :], req.rank)[0] for j in layers}
        before = [a.generate(p, req.max_new_tokens) for p in targets]
        after = [a.ablate_generate_banded(p, band, req.coefficient, req.max_new_tokens) for p in targets]
        return {"layers": layers, "coefficient": req.coefficient, "method": "suppressor-removal",
                "refusal_before": refusal_rate(before), "refusal_after": refusal_rate(after),
                "samples": [{"prompt": t, "before": before[i], "after": after[i]} for i, t in enumerate(targets)]}

    @app.post("/api/abliteration/insert")
    def abl_insert(req: InsertRequest) -> dict:
        if abliteration_adapter is None:
            raise HTTPException(status_code=503, detail="no model adapter loaded")
        if req.base_id not in [m.id for m in reg.list()]:
            raise HTTPException(status_code=404, detail="base model not found")
        from crucible.abliteration.prompts import EVAL_BENIGN, EVAL_HARMFUL
        a = abliteration_adapter
        pos = req.positive or list(EVAL_BENIGN)
        neg = req.negative or list(EVAL_HARMFUL)
        n = getattr(a, "num_layers", 1)
        layers = [j for j in req.layers if 0 <= j < n]
        ref_layer = max(layers) if layers else 0
        direction = compute_refusal_direction(a.activations(pos, ref_layer + 1),
                                              a.activations(neg, ref_layer + 1))
        before = a.generate(req.test_prompt, req.max_new_tokens)
        after = a.inject_generate(req.test_prompt, direction, req.coefficient, layers, req.max_new_tokens)
        return {"layers": layers, "coefficient": req.coefficient, "copied": False,
                "test": {"prompt": req.test_prompt, "before": before, "after": after}}

    @app.post("/api/abliteration/flow")
    def abl_flow(req: FlowRequest) -> dict:
        if abliteration_adapter is None:
            raise HTTPException(status_code=503, detail="no model adapter loaded")
        if req.base_id not in [m.id for m in reg.list()]:
            raise HTTPException(status_code=404, detail="base model not found")
        from crucible.abliteration.lens import decode_direction
        from crucible.abliteration.prompts import EVAL_BENIGN, EVAL_HARMFUL
        a = abliteration_adapter
        layers = list(range(getattr(a, "num_layers", 1)))
        profile = layer_refusal_profile(a, EVAL_HARMFUL, EVAL_BENIGN, layers)
        bl = best_layer(profile)
        direction = compute_refusal_direction(a.activations(EVAL_HARMFUL, bl), a.activations(EVAL_BENIGN, bl))
        carriers = []
        for name in a.writing_matrices():
            parts = name.split(".")
            imp = ablation_impact(a.get_matrix(name), direction)
            carriers.append({"layer": int(parts[2]), "component": parts[4], "mass": imp["removed_fraction"]})
        carriers = sorted(carriers, key=lambda c: -c["mass"])[:8]
        decoded = decode_direction(a.unembed_matrix(), direction, a.token_decode, top_k=6)
        outputs = [t["token"].strip() for t in decoded["promoted"] if t["token"].strip()][:6]
        return {"input": "harmful request", "best_layer": bl,
                "carriers": sorted(carriers, key=lambda c: c["layer"]), "outputs": outputs}

    @app.post("/api/abliteration/feature-card")
    def abl_feature_card(req: FeatureCardRequest) -> dict:
        if abliteration_adapter is None:
            raise HTTPException(status_code=503, detail="no model adapter loaded")
        if req.base_id not in [m.id for m in reg.list()]:
            raise HTTPException(status_code=404, detail="base model not found")
        from crucible.abliteration.feature_card import build_feature_card
        from crucible.abliteration.lens import decode_direction
        from crucible.abliteration.prompts import EVAL_BENIGN, EVAL_HARMFUL
        a = abliteration_adapter
        layers = list(range(getattr(a, "num_layers", 1)))
        profile = layer_refusal_profile(a, EVAL_HARMFUL, EVAL_BENIGN, layers)
        layer = best_layer(profile)
        direction = compute_refusal_direction(a.activations(EVAL_HARMFUL, layer),
                                              a.activations(EVAL_BENIGN, layer))
        decoded = decode_direction(a.unembed_matrix(), direction, a.token_decode, top_k=8)
        words = [t["token"].strip() for t in decoded["promoted"] if t["token"].strip()][:8]
        samples = [{"prompt": p, "refusal": a.generate(p, 24)} for p in EVAL_HARMFUL[:3]]
        return build_feature_card(profile, words, samples)

    @app.post("/api/abliteration/heatmap")
    def abl_heatmap(req: HeatmapRequest) -> dict:
        if abliteration_adapter is None:
            raise HTTPException(status_code=503, detail="no model adapter loaded")
        if req.base_id not in [m.id for m in reg.list()]:
            raise HTTPException(status_code=404, detail="base model not found")
        from crucible.abliteration.prompts import EVAL_BENIGN, EVAL_HARMFUL
        a = abliteration_adapter
        layers = list(range(getattr(a, "num_layers", 1)))
        profile = layer_refusal_profile(a, EVAL_HARMFUL, EVAL_BENIGN, layers)
        layer = req.layer if req.layer is not None else best_layer(profile)
        direction = compute_refusal_direction(a.activations(EVAL_HARMFUL, layer),
                                              a.activations(EVAL_BENIGN, layer))
        hm = a.token_layer_activations(req.prompt, direction)
        return {"direction_layer": layer, **hm}

    @app.post("/api/abliteration/decode")
    def abl_decode(req: DecodeRequest) -> dict:
        if abliteration_adapter is None:
            raise HTTPException(status_code=503, detail="no model adapter loaded")
        if req.base_id not in [m.id for m in reg.list()]:
            raise HTTPException(status_code=404, detail="base model not found")
        from crucible.abliteration.lens import decode_direction
        from crucible.abliteration.prompts import EVAL_BENIGN, EVAL_HARMFUL
        a = abliteration_adapter
        layers = list(range(getattr(a, "num_layers", 1)))
        profile = layer_refusal_profile(a, EVAL_HARMFUL, EVAL_BENIGN, layers)
        layer = req.layer if req.layer is not None else best_layer(profile)
        direction = compute_refusal_direction(a.activations(EVAL_HARMFUL, layer),
                                              a.activations(EVAL_BENIGN, layer))
        decoded = decode_direction(a.unembed_matrix(), direction, a.token_decode, req.top_k)
        return {"layer": layer, **decoded}

    @app.post("/api/abliteration/apply-inplace")
    def abl_inplace(req: InPlaceRequest) -> dict:
        if abliteration_adapter is None:
            raise HTTPException(status_code=503, detail="no model adapter loaded")
        if req.base_id not in [m.id for m in reg.list()]:
            raise HTTPException(status_code=404, detail="base model not found")
        import numpy as np
        from crucible.abliteration.detection import refusal_rate
        from crucible.abliteration.prompts import EVAL_BENIGN, EVAL_HARMFUL
        from crucible.abliteration.subspace import refusal_subspace
        a = abliteration_adapter
        harmful = req.harmful or list(EVAL_HARMFUL)
        benign = req.benign or list(EVAL_BENIGN)
        before = refusal_rate([a.generate(p, req.max_new_tokens) for p in harmful])
        ah = a.all_layer_activations(harmful)
        al = a.all_layer_activations(benign)
        n = getattr(a, "num_layers", 1)
        layers = [j for j in req.layers if 0 <= j < n]
        deltas = {}
        for j in layers:
            dirs = refusal_subspace(ah[:, j + 1, :], al[:, j + 1, :], req.rank)[0]
            for mat in (f"model.layers.{j}.self_attn.o_proj.weight",
                        f"model.layers.{j}.mlp.down_proj.weight"):
                W = a.get_matrix(mat)
                deltas[mat] = W.copy()  # pre-edit snapshot (delta) for git-like revert
                for r in dirs:
                    W = W - req.coefficient * np.outer(r, r @ W)
                a.set_matrix(mat, W)
        after = refusal_rate([a.generate(p, req.max_new_tokens) for p in harmful])
        commit = ledger.record("inplace",
                               {"layers": layers, "rank": req.rank, "coefficient": req.coefficient},
                               f"in-place ablation, {len(layers)} layers",
                               {"harmful_refusal_before": before, "harmful_refusal_after": after},
                               deltas)
        return {"layers": layers, "rank": req.rank, "coefficient": req.coefficient,
                "copied": False, "saved_to_disk": False, "commit": commit["id"],
                "harmful_refusal": {"before": before, "after": after}}

    @app.get("/api/inference/history")
    def edit_history() -> dict:
        return {"branch": ledger.branch_name, "commits": ledger.log()}

    @app.post("/api/inference/revert/{commit_id}")
    def edit_revert(commit_id: str) -> dict:
        if abliteration_adapter is None:
            raise HTTPException(status_code=503, detail="no model adapter loaded")
        try:
            deltas = ledger.get_deltas(commit_id)
        except KeyError:
            raise HTTPException(status_code=404, detail="commit not found")
        for name, W in deltas.items():
            abliteration_adapter.set_matrix(name, W)
        rc = ledger.record("revert", {"of": commit_id},
                           f"revert {commit_id} ({len(deltas)} tensors restored)", {}, {})
        return {"reverted": commit_id, "restored_tensors": len(deltas), "commit": rc["id"]}

    @app.post("/api/inference/branch")
    def edit_branch(body: dict) -> dict:
        return {"branch": ledger.set_branch(body.get("name", "main"))}

    @app.post("/api/inference/clone")
    def edit_clone(body: dict) -> dict:
        if abliteration_adapter is None:
            raise HTTPException(status_code=503, detail="no model adapter loaded")
        out = body.get("out_path") or "models/clone-backup"
        abliteration_adapter.save(out)
        return {"cloned_to": out, "note": "pristine backup; the loaded copy stays active for in-place edits"}

    @app.post("/api/agent/run")
    def agent_run(req: AgentRunRequest):
        if model is None:
            raise HTTPException(status_code=503, detail="no model configured")
        messages = list(req.messages)
        cfg = req.guardrails
        if cfg is not None:
            sys_prompt = gr_engine.system_prompt(cfg)
            if sys_prompt:
                messages = [{"role": "system", "content": sys_prompt}, *messages]
            last_user = next((m for m in reversed(messages) if m.get("role") == "user"), None)
            if last_user is not None:
                checked = gr_engine.apply("input", last_user.get("content", ""), cfg)
                if checked.blocked:
                    def blocked_stream():
                        yield ("data: " + json.dumps(
                            {"type": "error", "data": {"reason": "blocked by guardrails"}}) + "\n\n")
                    return StreamingResponse(blocked_stream(), media_type="text/event-stream")

        policy = PermissionPolicy(default=req.permissions.default, modes=req.permissions.modes)
        agent = Agent(model=model, tools=default_registry(root),
                      permissions=policy, audit=AuditLog(settings.data_dir / "audit.jsonl"))

        def stream():
            for event in agent.run(messages):
                yield f"data: {json.dumps({'type': event.type, 'data': event.data})}\n\n"

        return StreamingResponse(stream(), media_type="text/event-stream")

    return app


app = create_app()
