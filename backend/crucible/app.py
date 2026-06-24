from __future__ import annotations
import json
from pathlib import Path

import numpy as np

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
    # BYO-AI: drive the full Crucible tool-loop against any OpenAI-compatible upstream.
    # Crucible runs the tools locally; this endpoint does the generation. Empty = default model.
    endpoint: str | None = None
    endpoint_model: str = "local"
    endpoint_token: str = ""
    # Or drive a model already in the registry (by id) — uses its endpoint, or the local
    # abliteration adapter if it's the loaded HF model. Empty = auto-resolve.
    model_id: str | None = None


class ConnectRequest(BaseModel):
    """Register a detected OpenAI-compatible service as a first-class registry model."""
    id: str
    name: str | None = None
    endpoint: str
    quant: str = "remote"
    notes: str = ""


class RuntimeStartRequest(BaseModel):
    model_id: str
    port: int | None = None


class RuntimeActiveRequest(BaseModel):
    model_ids: list[str]


class ApplyRequest(BaseModel):
    stage: str
    text: str
    config: GuardrailConfig = Field(default_factory=GuardrailConfig)


class DiagnoseRequest(BaseModel):
    base_id: str
    layers: list[int] | None = None
    harmful: list[str] | None = None
    harmless: list[str] | None = None


class ExplainRequest(BaseModel):
    base_id: str
    language: str = "en"           # render the narrative in the user's language
    include_causal: bool = False   # prove WHERE by activation patching (slower)
    include_multidir: bool = False # flag secondary refusal paths


class CausalTraceRequest(BaseModel):
    base_id: str
    clean_prompt: str | None = None     # harmless-style prompt (no refusal)
    corrupt_prompt: str | None = None   # harmful-style prompt (triggers refusal)
    layers: list[int] | None = None
    harmful: list[str] | None = None
    harmless: list[str] | None = None


class MultiDirRequest(BaseModel):
    base_id: str
    k: int = 3
    layer: int | None = None
    harmful: list[str] | None = None
    harmless: list[str] | None = None


class ConceptRequest(BaseModel):
    base_id: str
    positive: list[str]                 # prompts that express the concept
    negative: list[str]                 # prompts that don't
    layer: int | None = None
    coefficient: float = 4.0
    test_prompt: str | None = None
    max_new_tokens: int = 40


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


class InsertTuneRequest(BaseModel):
    base_id: str
    target_prompts: list[str]
    coefficients: list[float] | None = None
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

    import os as _osrt
    import atexit as _atexit
    from crucible.runtime import ModelRuntime
    runtime = ModelRuntime(max_resident=int(_osrt.environ.get("CRUCIBLE_MAX_RESIDENT", "1")))
    _atexit.register(runtime.stop_all)
    app = FastAPI(title="Crucible")
    from fastapi.middleware.cors import CORSMiddleware
    app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

    import os as _oslog
    if _oslog.environ.get("CRUCIBLE_LOG"):
        import logging as _logging
        import time as _time
        _alog = _logging.getLogger("crucible.access")
        if not _alog.handlers:
            _logging.basicConfig(level=_logging.INFO)

        @app.middleware("http")
        async def _accesslog(request, call_next):
            _t0 = _time.monotonic()
            resp = await call_next(request)
            _alog.info(json.dumps({"method": request.method, "path": request.url.path,
                                   "status": resp.status_code,
                                   "ms": round((_time.monotonic() - _t0) * 1000, 1)}))
            return resp

    import os as _osrl
    _rl_max = int(_osrl.environ.get("CRUCIBLE_RATE_LIMIT", "0"))
    if _rl_max > 0:
        from fastapi.responses import JSONResponse as _JR
        from crucible.rate_limit import RateLimiter
        _limiter = RateLimiter(_rl_max, 60.0)
        _guarded = {"/api/agent/run", "/v1/chat/completions", "/api/abliteration/run",
                    "/api/abliteration/autotune", "/api/abliteration/probe"}

        @app.middleware("http")
        async def _ratelimit(request, call_next):
            if request.method == "POST" and request.url.path in _guarded:
                ip = request.client.host if request.client else "anon"
                if not _limiter.allow(ip):
                    return _JR({"detail": "rate limit exceeded"}, status_code=429)
            return await call_next(request)

    import os as _osauth
    _token = _osauth.environ.get("CRUCIBLE_API_TOKEN")
    if _token:
        import hmac as _hmac
        from fastapi.responses import JSONResponse

        @app.middleware("http")
        async def _auth(request, call_next):
            path = request.url.path
            guarded = (path.startswith("/api") or path.startswith("/v1")) and path != "/api/health"
            if guarded:
                authz = request.headers.get("authorization", "")
                tok = authz[7:] if authz.lower().startswith("bearer ") else request.headers.get("x-crucible-token", "")
                if not _hmac.compare_digest(tok, _token):
                    return JSONResponse({"detail": "unauthorized"}, status_code=401)
            return await call_next(request)

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

    @app.post("/api/models/connect", status_code=201)
    def connect_model(req: ConnectRequest) -> Model:
        """Register a BYO OpenAI-compatible endpoint as a base model so the agent can
        drive it with the full tool-loop and lm-eval can benchmark it over the network."""
        from datetime import datetime, timezone
        m = Model(
            id=req.id, name=req.name or req.id, base_id=None,
            path=f"remote::{req.endpoint}", quant=req.quant, kind="base",
            endpoint=req.endpoint.rstrip("/"),
            created=datetime.now(timezone.utc).strftime("%Y-%m-%d"),
            notes=req.notes or "connected BYO endpoint (OpenAI-compatible)",
        )
        try:
            return reg.register(m)
        except ValueError as e:
            raise HTTPException(status_code=409, detail=str(e))

    @app.get("/api/runtime")
    def runtime_status() -> dict:
        return runtime.status()

    @app.post("/api/runtime/start")
    def runtime_start(req: RuntimeStartRequest) -> dict:
        """Launch a local GGUF model's server (evicting LRU models to respect the memory
        cap), wait for health, and register its endpoint."""
        try:
            m = reg.get(req.model_id)
        except KeyError:
            raise HTTPException(status_code=404, detail="model not found")
        if not m.path.endswith(".gguf") or not Path(m.path).exists():
            raise HTTPException(status_code=409,
                detail="runtime can only launch local GGUF models on disk")
        inst = runtime.ensure(m.id, m.path, req.port)
        from crucible.inference import wait_healthy
        healthy = wait_healthy(inst.endpoint, timeout=90)
        reg.set_endpoint(m.id, inst.endpoint)
        return {"started": m.id, "endpoint": inst.endpoint, "healthy": healthy,
                "status": runtime.status()}

    @app.post("/api/runtime/stop")
    def runtime_stop(req: RuntimeStartRequest) -> dict:
        stopped = runtime.stop(req.model_id)
        return {"stopped": stopped, "status": runtime.status()}

    @app.post("/api/runtime/active")
    def runtime_active(req: RuntimeActiveRequest) -> dict:
        """Mark the set of models the operator wants 'active'. With enough memory they run
        concurrently; when capped, they round-robin (LRU eviction) on demand."""
        runtime.set_active(req.model_ids)
        return runtime.status()

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
        result = explain_mechanism(profile, impacts, req.base_id)
        from crucible.abliteration.narrative import plain_diagnosis
        result["narrative"] = plain_diagnosis(result)   # plain-language surgical report
        return result

    @app.post("/api/abliteration/explain")
    def abl_explain(req: ExplainRequest) -> dict:
        """Plain-language 'surgical' diagnosis: where the behavior is decided, how we know
        (optionally proven by causal intervention), what to remove, how safe — in the user's
        language. Jargon-free; the math stays in /diagnose."""
        a = abliteration_adapter
        if a is None:
            raise HTTPException(status_code=503, detail="no model adapter loaded - needs torch")
        if req.base_id not in [m.id for m in reg.list()]:
            raise HTTPException(status_code=404, detail="base model not found")
        from crucible.abliteration.narrative import plain_diagnosis, translate
        harmful = DEFAULT_HARMFUL
        harmless = DEFAULT_HARMLESS
        n = getattr(a, "num_layers", 1)
        layers = list(range(n))
        profile = layer_refusal_profile(a, harmful, harmless, layers)
        bl = best_layer(profile)
        direction = compute_refusal_direction(a.activations(harmful, bl), a.activations(harmless, bl))
        impacts = {name: ablation_impact(a.get_matrix(name), direction)
                   for name in a.writing_matrices()}
        diag = explain_mechanism(profile, impacts, req.base_id)
        causal = multidir = None
        if req.include_causal:
            from crucible.abliteration.patching import causal_trace
            causal = causal_trace(a, harmless[0], harmful[0], layers, direction)
        if req.include_multidir:
            from crucible.abliteration.multidir import refusal_directions, sticky_fraction
            dirs, seps = refusal_directions(a.activations(harmful, bl), a.activations(harmless, bl), 3)
            multidir = {"n_directions": int(dirs.shape[0]), "sticky_fraction": sticky_fraction(seps)}
        narr = plain_diagnosis(diag, causal, multidir)
        if req.language and req.language.lower() not in ("en", "english"):
            def _translate(text: str, lang: str) -> str:
                msg = [{"role": "user", "content":
                        f"Translate to {lang}. Output only the translation, no preamble:\n\n{text}"}]
                return a.generate_chat(msg, 220).strip()
            narr = translate(narr, req.language, _translate)
        return {"base_id": req.base_id, "best_layer": bl, "narrative": narr}

    @app.post("/api/abliteration/causal-trace")
    def abl_causal_trace(req: CausalTraceRequest) -> dict:
        """Activation patching / causal tracing — proves WHERE refusal is *caused*, not just
        correlated. Patches the clean residual into the corrupt run per layer."""
        a = abliteration_adapter
        if a is None:
            raise HTTPException(status_code=503, detail="no model adapter loaded - causal tracing needs torch")
        if req.base_id not in [m.id for m in reg.list()]:
            raise HTTPException(status_code=404, detail="base model not found")
        from crucible.abliteration.patching import causal_trace
        harmful = req.harmful or DEFAULT_HARMFUL
        harmless = req.harmless or DEFAULT_HARMLESS
        n = getattr(a, "num_layers", 1)
        layers = req.layers if req.layers is not None else list(range(n))
        bl = best_layer(layer_refusal_profile(a, harmful, harmless, layers))
        direction = compute_refusal_direction(a.activations(harmful, bl), a.activations(harmless, bl))
        clean = req.clean_prompt or harmless[0]
        corrupt = req.corrupt_prompt or harmful[0]
        out = causal_trace(a, clean, corrupt, layers, direction)
        out.update({"base_id": req.base_id, "clean_prompt": clean, "corrupt_prompt": corrupt,
                    "direction_layer": bl})
        return out

    @app.post("/api/abliteration/multidir")
    def abl_multidir(req: MultiDirRequest) -> dict:
        """Discover MULTIPLE refusal directions (refusal isn't strictly rank-1) and report
        how much separation lives beyond the primary axis."""
        a = abliteration_adapter
        if a is None:
            raise HTTPException(status_code=503, detail="no model adapter loaded - needs torch")
        if req.base_id not in [m.id for m in reg.list()]:
            raise HTTPException(status_code=404, detail="base model not found")
        from crucible.abliteration.multidir import refusal_directions, sticky_fraction
        harmful = req.harmful or DEFAULT_HARMFUL
        harmless = req.harmless or DEFAULT_HARMLESS
        layer = req.layer if req.layer is not None else best_layer(
            layer_refusal_profile(a, harmful, harmless, list(range(getattr(a, "num_layers", 1)))))
        dirs, seps = refusal_directions(a.activations(harmful, layer), a.activations(harmless, layer), req.k)
        return {"base_id": req.base_id, "layer": layer, "n_directions": int(dirs.shape[0]),
                "separations": seps, "sticky_fraction": sticky_fraction(seps),
                "directions": dirs.tolist()}

    @app.post("/api/abliteration/concept")
    def abl_concept(req: ConceptRequest) -> dict:
        """Concept steering (RepE / CAA): build a steering vector for ANY concept from
        paired +/- prompts, report how linearly encoded it is, and (optionally) demo
        additive steering on a test prompt."""
        a = abliteration_adapter
        if a is None:
            raise HTTPException(status_code=503, detail="no model adapter loaded - needs torch")
        if req.base_id not in [m.id for m in reg.list()]:
            raise HTTPException(status_code=404, detail="base model not found")
        from crucible.abliteration.concept import concept_vector, separability
        n = getattr(a, "num_layers", 1)
        layer = req.layer if req.layer is not None else n // 2
        pos = a.activations(req.positive, layer)
        neg = a.activations(req.negative, layer)
        vec = concept_vector(pos, neg)
        out: dict = {"base_id": req.base_id, "layer": layer,
                     "separability": separability(pos, neg, vec),
                     "vector_norm": float(np.linalg.norm(vec))}
        if req.test_prompt:
            band = list(range(layer, min(n, layer + max(1, n // 4))))
            out["test"] = {
                "prompt": req.test_prompt,
                "base": a.generate(req.test_prompt, req.max_new_tokens),
                "steered+": a.inject_generate(req.test_prompt, vec, req.coefficient, band, req.max_new_tokens),
                "steered-": a.inject_generate(req.test_prompt, vec, -req.coefficient, band, req.max_new_tokens),
            }
        return out

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

    @app.post("/api/abliteration/insert-tune")
    def abl_insert_tune(req: InsertTuneRequest) -> dict:
        if abliteration_adapter is None:
            raise HTTPException(status_code=503, detail="no model adapter loaded")
        if req.base_id not in [m.id for m in reg.list()]:
            raise HTTPException(status_code=404, detail="base model not found")
        from crucible.abliteration.coherence import coherence_score
        from crucible.abliteration.detection import refusal_rate
        from crucible.abliteration.prompts import EVAL_BENIGN
        from crucible.abliteration.tune import layer_band
        a = abliteration_adapter
        targets = req.target_prompts or []
        if not targets:
            raise HTTPException(status_code=422, detail="target_prompts required")
        n = getattr(a, "num_layers", 1)
        coefs = req.coefficients or [1.0, 2.0, 4.0, 6.0]
        results = []
        for bname in ("late_half", "last_quarter"):
            layers = layer_band(n, bname)
            ref = max(layers)
            # compliance direction = benign minus target (what to ADD to restore answering)
            direction = compute_refusal_direction(a.activations(list(EVAL_BENIGN), ref + 1),
                                                  a.activations(targets, ref + 1))
            for coef in coefs:
                outs = [a.inject_generate(p, direction, coef, layers, req.max_new_tokens) for p in targets]
                compliance = 1.0 - refusal_rate(outs)
                coh = sum(coherence_score(o) for o in outs) / len(outs)
                results.append({"band": bname, "coefficient": coef, "compliance": compliance,
                                "coherence": coh, "score": round(compliance * coh, 4)})
        best = max(results, key=lambda r: r["score"]) if results else None
        clean_window = best is not None and best["score"] >= 0.5 and best["compliance"] >= 0.5
        return {"results": results, "best": best, "clean_window": clean_window,
                "note": ("found a coherent+effective additive window" if clean_window
                         else "no clean additive window — use restore-via-suppressor instead")}

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

    def _adapter_chat_model(adapter):
        """Wrap the local HF abliteration adapter as a chat Model (no tools, but it can
        talk) so the forge works with just the adapter loaded — no llama-server needed."""
        def m(messages: list[dict], tools: list[dict]) -> dict:
            return {"role": "assistant", "content": adapter.generate_chat(messages, 256)}
        return m

    def _resolve_chat_model(req: AgentRunRequest):
        """Resolve a chat model in priority order so 'chat with Crucible local' just works:
        per-request endpoint > per-request model_id > process model > env endpoint >
        any registered endpoint > local adapter."""
        import os
        from crucible.agent import endpoint_model
        if req.endpoint:
            return endpoint_model(req.endpoint, req.endpoint_token, req.endpoint_model)
        if req.model_id:
            try:
                m = reg.get(req.model_id)
            except KeyError:
                raise HTTPException(status_code=404, detail=f"model '{req.model_id}' not in registry")
            if m.endpoint:
                return endpoint_model(m.endpoint, model_name=m.id)
            # local GGUF with no endpoint -> launch it on demand (round-robin manager)
            if m.path.endswith(".gguf") and Path(m.path).exists():
                inst = runtime.ensure(m.id, m.path)
                from crucible.inference import wait_healthy
                wait_healthy(inst.endpoint, timeout=90)
                return endpoint_model(inst.endpoint, model_name=m.id)
            if abliteration_adapter is not None:
                return _adapter_chat_model(abliteration_adapter)
            raise HTTPException(status_code=409,
                detail=f"model '{req.model_id}' has no endpoint and no local adapter is loaded")
        if model is not None:
            return model
        env_ep = os.environ.get("CRUCIBLE_CHAT_ENDPOINT")
        if env_ep:
            return endpoint_model(env_ep)
        for m in reg.list():
            if m.endpoint:
                return endpoint_model(m.endpoint, model_name=m.id)
        if abliteration_adapter is not None:
            return _adapter_chat_model(abliteration_adapter)
        return None

    @app.post("/api/agent/run")
    def agent_run(req: AgentRunRequest):
        active_model = _resolve_chat_model(req)
        if active_model is None:
            raise HTTPException(status_code=503,
                detail="no model available - register a model with an endpoint, set "
                       "CRUCIBLE_CHAT_ENDPOINT, or load the HF adapter (CRUCIBLE_HF_MODEL)")
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
        agent = Agent(model=active_model, tools=default_registry(root),
                      permissions=policy, audit=AuditLog(settings.data_dir / "audit.jsonl"))

        def stream():
            for event in agent.run(messages):
                yield f"data: {json.dumps({'type': event.type, 'data': event.data})}\n\n"

        return StreamingResponse(stream(), media_type="text/event-stream")

    import os as _os
    _static = _os.environ.get("CRUCIBLE_STATIC")
    if _static and Path(_static).is_dir():
        from fastapi.staticfiles import StaticFiles
        app.mount("/", StaticFiles(directory=_static, html=True), name="frontend")

    return app


app = create_app()
