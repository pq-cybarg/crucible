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


class EvalRunRequest(BaseModel):
    benchmark: str


class HeadToHeadScoreRequest(BaseModel):
    benchmark: str
    answers: dict[str, str]


class LmEvalRequest(BaseModel):
    model_id: str
    tasks: list[str]
    limit: int | None = None


def create_app(registry: Registry | None = None, agent_root: Path | None = None,
               model=None, guardrails: GuardrailsEngine | None = None,
               abliteration_adapter=None) -> FastAPI:
    settings = get_settings()
    reg = registry or Registry(settings.registry_path)
    root = Path(agent_root or ".")
    preset_store = PresetStore(settings.data_dir / "presets.json")
    gr_engine = guardrails or GuardrailsEngine(preset_resolver=preset_store.system_prompt)
    gr_store = GuardrailStore(settings.data_dir / "guardrails.json")
    abl = (AbliterationPipeline(abliteration_adapter, reg)
           if abliteration_adapter is not None else None)
    app = FastAPI(title="Crucible")

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
        if not m.endpoint:
            raise HTTPException(status_code=409,
                detail="model has no endpoint - launch llama-server and register its endpoint")
        rows = run_lmeval(m.endpoint, req.tasks, req.limit)
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
