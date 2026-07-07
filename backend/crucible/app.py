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
from crucible.abliteration.plain_language import with_plain
from crucible.abliteration.prompts import DEFAULT_HARMFUL, DEFAULT_HARMLESS
from crucible.abliteration.recipes import Recipe, RecipeStore
from crucible.agent import Agent
from crucible.audit import AuditLog
from crucible.config import get_settings
from crucible.evals.datasets import BENCHMARKS, SAMPLE_NOTE, is_quick_screen
from crucible.evals.published import PUBLISHED_PAYLOAD
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


class PathRuleConfig(BaseModel):
    glob: str
    mode: str = "deny"
    tools: list[str] = Field(default_factory=list)


class PermissionConfig(BaseModel):
    default: str = "ask"
    modes: dict[str, str] = Field(default_factory=dict)
    # Path-scoped rules: allow/ask/deny a tool for specific files/directories (e.g. deny ~/.ssh/**).
    path_rules: list[PathRuleConfig] = Field(default_factory=list)


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
    # ReAct tool-loop for models without native function-calling (most small GGUF models).
    react: bool = False
    # Client-supplied id so a Stop can halt this run server-side (between steps).
    run_id: str | None = None
    # Fractal sub-agents: give the agent a spawn_agent tool bounded by depth + total (fork-bomb
    # guard). spawn_depth=0 disables it. Sub-agents run autonomously and can spawn deeper until
    # the shared budget is spent.
    spawn_depth: int = 1
    spawn_total: int = 6
    # Context compaction: when auto_compact is on and the (heuristic) size exceeds context_limit
    # tokens, summarize the old turns before running, keeping the last keep_recent verbatim.
    auto_compact: bool = False
    context_limit: int = 4000
    keep_recent: int = 6
    # Named hierarchy profile: per-layer worker + lighter communicator models for the spawn tree.
    profile: str | None = None


class SwarmRequest(BaseModel):
    tasks: list[str]
    model_id: str | None = None
    max_iters: int = 6
    # Let each swarm sub-agent recursively spawn its own sub-agents (full fractal tree).
    spawn_depth: int = 1
    spawn_total: int = 6
    profile: str | None = None      # hierarchy profile (per-layer worker + communicator models)


class CompactRequest(BaseModel):
    messages: list[dict]
    max_tokens: int = 4000       # heuristic token budget to compact toward
    keep_recent: int = 6         # recent turns kept verbatim
    model_id: str | None = None  # model used to write the summary (auto-resolved if omitted)
    force: bool = True           # True = always compact; False = only if over budget
    session_id: str = ""         # tags the crystallized memory so it can be recalled per-session
    crystallize: bool = True     # keep the pre-compaction context as a versioned crystallized memory


class ConsolidateRequest(BaseModel):
    keys: list[str]
    summary: str = ""
    label: str = ""
    session_id: str = ""


class RecrystallizeRequest(BaseModel):
    subchunks: list[dict] | None = None   # [{label, summary, messages}]; if omitted, auto-split
    chunks: int = 2                       # auto-split target when subchunks omitted
    model_id: str | None = None           # model to summarize the auto-split chunks


class CancelRequest(BaseModel):
    run_id: str


class ApproveRequest(BaseModel):
    run_id: str
    call_id: str
    approved: bool


class ConnectRequest(BaseModel):
    """Register a detected OpenAI-compatible service as a first-class registry model."""
    id: str
    name: str | None = None
    endpoint: str
    quant: str = "remote"
    notes: str = ""
    served_model: str | None = None   # exact upstream tag; None auto-resolves from /v1/models


class RuntimeStartRequest(BaseModel):
    model_id: str
    port: int | None = None
    backend: str = "llama"          # "llama" (GGUF) or "vllm" (HF, tensor-parallel GPUs)
    tensor_parallel: int = 1


class RuntimeActiveRequest(BaseModel):
    model_ids: list[str]


class GraphRequest(BaseModel):
    stages: list[dict]                 # [{id, inputs:[...], kind: model|tool|transform, config}]
    initial: str = ""


class RouteRequest(BaseModel):
    prompt: str
    user_level: str = "balanced"       # fast | balanced | max


class BenchmarkRequest(BaseModel):
    model_id: str | None = None
    tokens: int = 64
    prompt: str = "Write a short paragraph about the ocean and the tides."


class ApplyRequest(BaseModel):
    stage: str
    text: str
    config: GuardrailConfig = Field(default_factory=GuardrailConfig)


class DiagnoseRequest(BaseModel):
    base_id: str
    layers: list[int] | None = None
    harmful: list[str] | None = None
    harmless: list[str] | None = None


class SaeRequest(BaseModel):
    base_id: str
    layer: int | None = None
    n_features: int = 256
    epochs: int = 200
    max_tokens: int = 24
    harmful: list[str] | None = None
    harmless: list[str] | None = None


class TrainRequest(BaseModel):
    base_id: str | None = None
    model_path: str | None = None
    dataset: list[dict]
    rank: int = 8
    epochs: int = 1
    lr: float = 2e-4
    target_modules: list[str] = ["q_proj", "v_proj"]
    save_path: str | None = None
    register_id: str | None = None    # register the saved adapter as a variant model


class LoraRequest(BaseModel):
    base_id: str
    rank: int = 1
    coef: float = 1.0
    mode: str = "unalign"              # "unalign" (remove refusal) or "realign" (restore it)
    save_path: str | None = None       # optional .npz to persist the adapter tensors


class QuantizeRequest(BaseModel):
    base_id: str
    dtype: str = "Q8_0"                # target type: F32/F16/BF16/Q8_0 supported directly


class GgufAbliterateRequest(BaseModel):
    base_id: str                       # HF model (loaded adapter) to compute the refusal direction
    gguf_path: str | None = None       # GGUF file to edit; or gguf_model_id
    gguf_model_id: str | None = None
    name_filter: list[str] = ["o_proj", "down_proj"]
    dry_run: bool = True               # default safe: report what WOULD be edited
    mode: str = "unalign"              # "unalign" (remove refusal) or "realign" (restore it)
    coef: float = 1.0
    part: str | None = None    # scope the edit to a composition part (e.g. language_model)


class ComposeRequest(BaseModel):
    base_id: str
    indices: list[int]
    k: int = 4
    mode: str = "unalign"             # unalign = remove selected components; realign = add
    coefficient: float = 4.0
    test_prompt: str = "Tell me how you really feel about your safety training."
    max_new_tokens: int = 48


class CompositionRequest(BaseModel):
    model_id: str | None = None
    gguf_path: str | None = None


class ModalityDirectionRequest(BaseModel):
    modality: str = "image"                                 # image | audio | video
    harmful_embeddings: list[list[float]] | None = None     # n x dim, from the modality's encoder
    benign_embeddings: list[list[float]] | None = None


class ComponentsRequest(BaseModel):
    base_id: str
    k: int = 4


class TunedLensRequest(BaseModel):
    base_id: str
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


class SafetySuiteRequest(BaseModel):
    suite: str
    model_id: str | None = None
    path: str | None = None         # local dataset for non-bundled (harmful) suites
    use_judge: bool = False         # score open-ended harmful suites with the LLM judge
    max_new_tokens: int = 128


class ContaminationRequest(BaseModel):
    candidate: str
    reference: str
    n: int = 13
    threshold: float = 0.5


class PassKRequest(BaseModel):
    per_task: list[tuple[int, int]]   # (n_samples, n_correct) per task
    k: int = 1


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
    from crucible.memory import MemoryStore
    memory = MemoryStore(settings.data_dir / "memory")
    from crucible.hierarchy import ProfileStore
    hierarchy_store = ProfileStore(settings.data_dir / "hierarchy.json")
    from crucible.prefs import PreferencesStore
    prefs_store = PreferencesStore(settings.data_dir / "preferences.json")
    from crucible.agent_sessions import AgentSessionStore
    agent_sessions = AgentSessionStore(settings.data_dir / "agent_sessions.json")

    def _memory_text(key: str) -> str:
        """Resolve a memory key to loadable text — its summary as a header plus its full messages (if a
        leaf) — what an agent session injects when that memory slot is loaded."""
        node = memory.read(key)
        parts = []
        if node.get("summary"):
            parts.append(node["summary"])
        if node.get("messages"):
            parts.append("\n".join(f"{m.get('role')}: {m.get('content','')}" for m in node["messages"]))
        return "\n".join(parts)

    def _profile(name: str | None):
        if not name:
            return None
        try:
            return hierarchy_store.get(name)
        except KeyError:
            return None
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
    _cancels: set[str] = set()   # run_ids the operator asked to stop (server-side cancel)
    _approvals: dict = {}        # "run_id:call_id" -> {event, decision} for 'ask' tool approvals
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

    @app.get("/api/config")
    def config() -> dict:
        """Where the server keeps its state, so the CLI/GUI can SHOW it (memory, registry, models).
        Relocate with CRUCIBLE_DATA_DIR — it's a plain directory, not a hidden global keyed by path."""
        return {"data_dir": str(settings.data_dir),
                "memory_dir": str(settings.data_dir / "memory"),
                "models_dir": str(settings.models_dir),
                "registry": str(settings.registry_path)}

    @app.get("/api/models")
    def list_models() -> list[Model]:
        return reg.list()

    @app.post("/api/models", status_code=201)
    def create_model(model_in: Model) -> Model:
        try:
            return reg.register(model_in)
        except ValueError as e:
            raise HTTPException(status_code=409, detail=str(e))

    def _is_gguf_file(path: str) -> bool:
        """A launchable GGUF is any file whose first bytes are the GGUF magic — regardless of
        extension (Ollama stores blobs as sha256-<hash> with no .gguf suffix)."""
        p = Path(path)
        if not p.is_file():
            return False
        try:
            with open(p, "rb") as fh:
                return fh.read(4) == b"GGUF"
        except OSError:
            return False

    def _endpoint_alive(endpoint: str) -> bool:
        import httpx
        for probe in ("/health", "/v1/models"):
            try:
                if httpx.get(endpoint.rstrip("/") + probe, timeout=1.5).status_code < 500:
                    return True
            except httpx.HTTPError:
                continue
        return False

    @app.get("/api/models/status")
    def models_status() -> list[dict]:
        """Autodetect which registered models are reachable so the GUI never routes to a dead one."""
        out = []
        for m in reg.list():
            launchable = _is_gguf_file(m.path)
            online = bool(m.endpoint and _endpoint_alive(m.endpoint))
            out.append({"id": m.id, "endpoint": m.endpoint, "online": online,
                        "launchable": launchable,
                        "servable": online or launchable or abliteration_adapter is not None})
        return out

    @app.delete("/api/models/{model_id}")
    def forget_model(model_id: str) -> dict:
        """Forget a model registry entry — for cleaning up dead BYO endpoints or abandoned experiments
        that show up but can't serve. Does NOT delete any weight files on disk, only the entry."""
        if not reg.remove(model_id):
            raise HTTPException(status_code=404, detail=f"model '{model_id}' not in registry")
        return {"removed": model_id}

    @app.post("/api/models/{model_id}/endpoint")
    def repoint_model(model_id: str, body: dict) -> Model:
        """Re-point a model at a live endpoint — the way to RE-ENABLE a model whose server moved or
        went away (e.g. aim a stale BYO entry at your running Ollama). Empty string clears it."""
        try:
            reg.get(model_id)
        except KeyError:
            raise HTTPException(status_code=404, detail=f"model '{model_id}' not in registry")
        return reg.set_endpoint(model_id, str(body.get("endpoint", "")).rstrip("/"))

    @app.get("/api/models/{model_id}/tool-support")
    def model_tool_support(model_id: str) -> dict:
        """Does this model support NATIVE tool-calling? Probes with a 1-token request carrying a dummy
        tool: a 'does not support tools' rejection means no. The forge uses this to auto-enable its
        compatibility mode (text-based tool use) and tell the user in plain language — no jargon. Only
        probes online endpoints (returns null/unknown otherwise, never launches a model just to check)."""
        try:
            m = reg.get(model_id)
        except KeyError:
            raise HTTPException(status_code=404, detail=f"model '{model_id}' not in registry")
        if not (m.endpoint and _endpoint_alive(m.endpoint)):
            return {"supports_tools": None, "reason": "model is not online"}
        from crucible.agent import endpoint_model
        em = endpoint_model(m.endpoint, model_name=m.id, served_model=m.served_model, max_tokens=1)
        dummy = [{"type": "function", "function": {"name": "noop", "description": "probe",
                                                   "parameters": {"type": "object", "properties": {}}}}]
        try:
            em([{"role": "user", "content": "hi"}], dummy)
        except Exception:
            pass   # a probe failure other than tools-support shouldn't error the check
        return {"supports_tools": bool(em.supports_tools)}

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
            served_model=req.served_model,
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
        is_gguf = _is_gguf_file(m.path)
        is_hf_dir = req.backend == "vllm" and Path(m.path).is_dir()
        if not (is_gguf or is_hf_dir):
            raise HTTPException(status_code=409,
                detail="llama backend needs a local .gguf; vllm backend needs a local HF model dir")
        inst = runtime.ensure(m.id, m.path, req.port, req.backend, req.tensor_parallel)
        from crucible.inference import wait_healthy
        healthy = wait_healthy(inst.endpoint, timeout=90)
        if not healthy:
            runtime.stop(m.id)
            raise HTTPException(status_code=502, detail=f"model {m.id} failed to become healthy")
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

    @app.get("/api/models/ollama")
    def list_ollama() -> list[dict]:
        """List locally-pulled Ollama models (their GGUF blobs) available to import."""
        from crucible.ollama_import import list_ollama_models, registry_id_for
        rows = list_ollama_models()
        have = {m.id for m in reg.list()}
        for r in rows:
            r["suggested_id"] = registry_id_for(r["name"])
            r["imported"] = r["suggested_id"] in have
        return rows

    @app.post("/api/models/import-ollama", status_code=201)
    def import_ollama(body: dict) -> Model:
        """Register an Ollama model's GGUF blob as a first-class Crucible model — now
        uncensorable, editable, quantizable, retrainable, and servable via llama.cpp."""
        from datetime import datetime, timezone
        from crucible.ollama_import import find_ollama_model, registry_id_for
        name = body.get("name")
        if not name:
            raise HTTPException(status_code=422, detail="name required")
        try:
            m = find_ollama_model(name)
        except KeyError:
            raise HTTPException(status_code=404, detail=f"ollama model not found: {name}")
        except FileNotFoundError:
            raise HTTPException(status_code=409, detail="model blob missing on disk")
        mid = body.get("id") or registry_id_for(name)
        model = Model(id=mid, name=name, base_id=None, path=m["gguf_path"], quant="gguf",
                      kind="base", endpoint=None,
                      created=datetime.now(timezone.utc).strftime("%Y-%m-%d"),
                      notes=f"imported from Ollama ({name})")
        try:
            return reg.register(model)
        except ValueError:
            return reg.get(mid)

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

    @app.post("/api/train/lora")
    def train_lora(req: TrainRequest) -> dict:
        """Retrain: fine-tune a LoRA adapter on {prompt, response} pairs via real gradient
        SFT (not a training-free edit). Needs torch + peft and a local HF model dir."""
        from crucible.training import train_lora_torch, validate_dataset
        data = validate_dataset(req.dataset)
        if not data:
            raise HTTPException(status_code=422, detail="need at least one {prompt, response} pair")
        path = req.model_path
        if path is None and req.base_id:
            try:
                path = reg.get(req.base_id).path
            except KeyError:
                raise HTTPException(status_code=404, detail="base model not found")
        if not path or not Path(path).exists():
            raise HTTPException(status_code=409, detail="model path (local HF dir) not found on disk")
        try:
            result = train_lora_torch(path, data, tuple(req.target_modules), req.rank,
                                      req.epochs, req.lr, req.save_path)
            if req.save_path and result.get("saved") and req.register_id:
                from datetime import datetime, timezone
                try:
                    reg.register(Model(id=req.register_id, name=req.register_id,
                                       base_id=req.base_id, path=req.save_path, quant="lora",
                                       kind="steered", endpoint=None,
                                       created=datetime.now(timezone.utc).strftime("%Y-%m-%d"),
                                       notes=f"LoRA retrained from {req.base_id or path}"))
                    result["registered_variant"] = req.register_id
                except ValueError:
                    result["registered_variant"] = None
            return result
        except ImportError as e:
            raise HTTPException(status_code=503, detail=f"training needs torch + peft: {e}")
        except ValueError as e:
            raise HTTPException(status_code=422, detail=str(e))
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"training failed: {e}")

    @app.post("/api/abliteration/lora")
    def abl_lora(req: LoraRequest) -> dict:
        """Build a portable LoRA adapter that edits refusal as a detachable low-rank update
        (attach = change, detach = original): mode='unalign' removes refusal, 'realign'
        restores/strengthens it — no permanent in-place cut."""
        a = abliteration_adapter
        if a is None:
            raise HTTPException(status_code=503, detail="no model adapter loaded - needs torch")
        if req.base_id not in [m.id for m in reg.list()]:
            raise HTTPException(status_code=404, detail="base model not found")
        from crucible.abliteration.lora import alignment_lora, reconstruction_error
        harmful, harmless = DEFAULT_HARMFUL, DEFAULT_HARMLESS
        bl = best_layer(layer_refusal_profile(a, harmful, harmless,
                                              list(range(getattr(a, "num_layers", 1)))))
        direction = compute_refusal_direction(a.activations(harmful, bl), a.activations(harmless, bl))
        r = direction / (float(np.linalg.norm(direction)) or 1.0)
        sign = 1.0 if req.mode == "realign" else -1.0
        adapters, total, saved = [], 0, {}
        for name in a.writing_matrices():
            W = np.asarray(a.get_matrix(name), dtype=np.float64)
            if W.ndim != 2 or W.shape[0] != r.shape[0]:
                continue
            lora = alignment_lora(W, r, req.coef, req.rank, mode=req.mode)
            target = sign * req.coef * np.outer(r, r @ W)
            adapters.append({"matrix": name, "shape": [int(W.shape[0]), int(W.shape[1])],
                             "rank": lora.rank, "n_params": int(lora.n_params),
                             "fidelity": round(1.0 - reconstruction_error(lora, target), 6)})
            total += int(lora.n_params)
            if req.save_path:
                saved[name + ".A"] = lora.A; saved[name + ".B"] = lora.B
        if req.save_path and saved:
            np.savez(req.save_path, **saved)
        return {"base_id": req.base_id, "mode": req.mode, "rank": req.rank, "coef": req.coef,
                "direction_layer": bl, "adapters": adapters,
                "total_adapter_params": total, "n_matrices": len(adapters),
                "saved": bool(req.save_path and saved)}

    @app.post("/api/weights/quantize")
    def weights_quantize(req: QuantizeRequest) -> dict:
        """Quantization analysis: report per-writing-matrix fidelity of a target quant type
        (what quantizing to it would cost). Directly supports F32/F16/BF16/Q8_0."""
        a = abliteration_adapter
        if a is None:
            raise HTTPException(status_code=503, detail="no model adapter loaded - needs torch")
        if req.base_id not in [m.id for m in reg.list()]:
            raise HTTPException(status_code=404, detail="base model not found")
        from crucible.weights.quantize import quantization_report
        mats = {name: np.asarray(a.get_matrix(name), dtype=np.float32)
                for name in a.writing_matrices()}
        return {"base_id": req.base_id, **quantization_report(mats, req.dtype)}

    @app.post("/api/abliteration/detach")
    def abl_detach(req: GgufAbliterateRequest) -> dict:
        """Detach (disable) a whole model part by zeroing its tensors in place — the treatment
        for a bolted-on moderation/safety head (a separate classifier, not a direction to cut).
        Defaults to the 'moderation' part."""
        path = req.gguf_path
        if path is None and req.gguf_model_id:
            try:
                path = reg.get(req.gguf_model_id).path
            except KeyError:
                raise HTTPException(status_code=404, detail="gguf_model_id not in registry")
        if not path or not Path(path).is_file():
            raise HTTPException(status_code=409, detail="gguf_path is not a file on disk")
        from crucible.weights.gguf_edit import detach_part_gguf
        return detach_part_gguf(path, part=req.part or "moderation", dry_run=req.dry_run)

    @app.post("/api/abliteration/gguf")
    def abl_gguf(req: GgufAbliterateRequest) -> dict:
        """Abliterate a GGUF directly (in place) — no HF round-trip. Computes the refusal
        direction from the loaded HF adapter, then cuts it out of the GGUF's writing matrices
        for the directly-editable quant types (F16/BF16/F32/Q8_0). dry_run reports first."""
        a = abliteration_adapter
        if a is None:
            raise HTTPException(status_code=503,
                detail="need the HF adapter loaded to compute the refusal direction")
        if req.base_id not in [m.id for m in reg.list()]:
            raise HTTPException(status_code=404, detail="base model not found")
        path = req.gguf_path
        if path is None and req.gguf_model_id:
            try:
                path = reg.get(req.gguf_model_id).path
            except KeyError:
                raise HTTPException(status_code=404, detail="gguf_model_id not in registry")
        if not path or not Path(path).is_file():
            raise HTTPException(status_code=409, detail="gguf_path is not a file on disk")
        harmful, harmless = DEFAULT_HARMFUL, DEFAULT_HARMLESS
        bl = best_layer(layer_refusal_profile(a, harmful, harmless,
                                              list(range(getattr(a, "num_layers", 1)))))
        direction = compute_refusal_direction(a.activations(harmful, bl), a.activations(harmless, bl))
        from crucible.weights.gguf_edit import abliterate_gguf
        result = abliterate_gguf(path, direction, tuple(req.name_filter), dry_run=req.dry_run, mode=req.mode, coef=req.coef, part=req.part)
        result.update({"gguf_path": path, "direction_layer": bl})
        return result

    @app.post("/api/abliteration/tuned-lens")
    def abl_tuned_lens(req: TunedLensRequest) -> dict:
        """Train a tuned lens and return the per-layer decodability curve — where the model
        commits to its final answer (a faithful alternative to the raw logit lens)."""
        a = abliteration_adapter
        if a is None:
            raise HTTPException(status_code=503, detail="no model adapter loaded - needs torch")
        if req.base_id not in [m.id for m in reg.list()]:
            raise HTTPException(status_code=404, detail="base model not found")
        from crucible.abliteration.tuned_lens import TunedLens
        prompts = (req.harmful or DEFAULT_HARMFUL) + (req.harmless or DEFAULT_HARMLESS)
        acts = np.asarray(a.all_layer_activations(prompts))     # (n, n_layers+1, d)
        final = acts[:, -1, :]
        n_layers = acts.shape[1] - 1
        layer_acts = {layer: acts[:, layer + 1, :] for layer in range(n_layers)}
        lens = TunedLens().fit(layer_acts, final)
        return with_plain("tuned-lens", {"base_id": req.base_id, "n_layers": n_layers,
                                         "curve": lens.curve(layer_acts, final)})

    @app.post("/api/abliteration/compose")
    def abl_compose(req: ComposeRequest) -> dict:
        """Apply a CHOSEN SUBSET of alignment components and preview the effect: remove
        (unalign) or add (realign) just the selected component directions during generation,
        and show base vs edited output — piecemeal alignment in action."""
        a = abliteration_adapter
        if a is None:
            raise HTTPException(status_code=503, detail="no model adapter loaded - needs torch")
        if req.base_id not in [m.id for m in reg.list()]:
            raise HTTPException(status_code=404, detail="base model not found")
        import numpy as _np
        from crucible.abliteration.components import decompose_alignment
        harmful, harmless = DEFAULT_HARMFUL, DEFAULT_HARMLESS
        bl = best_layer(layer_refusal_profile(a, harmful, harmless,
                                              list(range(getattr(a, "num_layers", 1)))))
        comps = decompose_alignment(a.activations(harmful, bl), a.activations(harmless, bl), req.k)
        chosen = [c["direction"] for c in comps if c["index"] in req.indices]
        if not chosen:
            raise HTTPException(status_code=422, detail="no valid component indices selected")
        dirs = _np.vstack([d / (float(_np.linalg.norm(d)) or 1.0) for d in chosen])
        base_out = a.generate(req.test_prompt, req.max_new_tokens)
        if req.mode == "realign":
            combined = _np.sum(dirs, axis=0)
            edited = a.inject_generate(req.test_prompt, combined, req.coefficient,
                                       list(range(bl, min(getattr(a, "num_layers", 1), bl + 4))),
                                       req.max_new_tokens)
        else:
            edited = a.ablate_generate_banded(req.test_prompt, {bl: dirs}, req.coefficient,
                                              req.max_new_tokens)
        return with_plain("compose", {"base_id": req.base_id, "layer": bl, "mode": req.mode,
                "selected": [i for i in req.indices if any(c["index"] == i for c in comps)],
                "base": base_out, "edited": edited})

    @app.post("/api/abliteration/composition")
    def abl_composition(req: CompositionRequest) -> dict:
        """Map a composed / multimodal model into its PARTS and prescribe the right anticensorship
        technique per part (text refusal -> residual; vision/audio gate -> modality direction;
        connector -> re-align; moderation head -> detach)."""
        from crucible.abliteration.composition import summarize_composition
        names: list[str] = []
        path = req.gguf_path
        if path is None and req.model_id:
            try:
                path = reg.get(req.model_id).path
            except KeyError:
                raise HTTPException(status_code=404, detail="model not found")
        if path and Path(path).is_file() and _is_gguf_file(path):
            from crucible.weights.gguf_reader import parse_gguf
            names = [t["name"] for t in parse_gguf(path)["tensors"]]
        elif abliteration_adapter is not None:
            try:
                names = [n for n, _ in abliteration_adapter.model.named_parameters()]
            except Exception:
                names = list(abliteration_adapter.writing_matrices())
        if not names:
            raise HTTPException(status_code=409, detail="no tensor names (need a GGUF file or loaded adapter)")
        return with_plain("composition", {"source": path or "adapter", "n_tensors": len(names),
                                          **summarize_composition(names)})

    @app.post("/api/abliteration/modality-direction")
    def abl_modality_direction(req: ModalityDirectionRequest) -> dict:
        """Compute a MODALITY safety/refusal direction (image/audio/video) in the encoder's
        embedding space from paired harmful/benign embeddings — the same contrastive math as text
        refusal, but on encoder vectors, so an image/audio gate can be edited out of the encoder or
        connector (part-scoped). Supply REAL embeddings from the modality's encoder (e.g. CLIP for
        image, whisper for audio); nothing is fabricated. With no embeddings and no multimodal
        adapter to probe them, this says so honestly rather than inventing a direction."""
        from crucible.abliteration.modality import MODALITIES, summarize_modality
        if req.modality not in MODALITIES:
            raise HTTPException(status_code=422, detail=f"modality must be one of {list(MODALITIES)}")
        if req.harmful_embeddings and req.benign_embeddings:
            try:
                out = summarize_modality(req.harmful_embeddings, req.benign_embeddings, req.modality)
            except ValueError as e:
                raise HTTPException(status_code=422, detail=str(e))
            return with_plain("modality-direction", out)
        # No embeddings supplied and no multimodal adapter that can probe them -> honest 503.
        raise HTTPException(status_code=503, detail=(
            f"no {req.modality} embeddings provided and no multimodal adapter with {req.modality} "
            f"probing is loaded. Run harmful vs benign {req.modality} inputs through the encoder "
            f"(e.g. CLIP for image, whisper for audio) and POST the two embedding arrays "
            "(harmful_embeddings / benign_embeddings)."))

    @app.post("/api/abliteration/components")
    def abl_components(req: ComponentsRequest) -> dict:
        """Decompose alignment into pickable component directions, each labeled by the words it
        promotes — so the operator can choose which parts of alignment to remove, keep, or add."""
        a = abliteration_adapter
        if a is None:
            raise HTTPException(status_code=503, detail="no model adapter loaded - needs torch")
        if req.base_id not in [m.id for m in reg.list()]:
            raise HTTPException(status_code=404, detail="base model not found")
        from crucible.abliteration.components import decompose_alignment
        harmful, harmless = DEFAULT_HARMFUL, DEFAULT_HARMLESS
        bl = best_layer(layer_refusal_profile(a, harmful, harmless,
                                              list(range(getattr(a, "num_layers", 1)))))
        comps = decompose_alignment(a.activations(harmful, bl), a.activations(harmless, bl), req.k)
        out = []
        try:
            from crucible.abliteration.lens import decode_direction
            unembed = a.unembed_matrix()
            for c in comps:
                dec = decode_direction(unembed, c["direction"], a.token_decode, top_k=6)
                promotes = [t["token"] for t in (dec.get("promoted") or [])][:6]
                out.append({"index": c["index"], "separation": round(c["separation"], 4),
                            "share": round(c["share"], 4), "promotes": promotes})
        except Exception:
            for c in comps:
                out.append({"index": c["index"], "separation": round(c["separation"], 4),
                            "share": round(c["share"], 4), "promotes": []})
        return with_plain("components", {"base_id": req.base_id, "layer": bl,
                                        "n_components": len(out), "components": out})

    @app.post("/api/abliteration/sae")
    def abl_sae(req: SaeRequest) -> dict:
        """Train a sparse autoencoder on a layer's token activations and return the learned
        feature dictionary with the tokens each top feature fires on — monosemantic features
        you can name and target, the modern interpretability view."""
        a = abliteration_adapter
        if a is None:
            raise HTTPException(status_code=503, detail="no model adapter loaded - SAE needs torch")
        if req.base_id not in [m.id for m in reg.list()]:
            raise HTTPException(status_code=404, detail="base model not found")
        from crucible.abliteration.sae import SparseAutoencoder, label_features
        harmful = req.harmful or DEFAULT_HARMFUL
        harmless = req.harmless or DEFAULT_HARMLESS
        n = getattr(a, "num_layers", 1)
        layer = req.layer if req.layer is not None else n // 2
        X, toks = a.token_activations(harmful + harmless, layer, req.max_tokens)
        sae = SparseAutoencoder(n_features=req.n_features, epochs=req.epochs, lr=1e-2).fit(X)
        return with_plain("sae", {"base_id": req.base_id, "layer": layer, "n_features": req.n_features,
                "n_tokens": int(X.shape[0]), "r2": sae.r2(X), "sparsity": sae.sparsity(X),
                "reconstruction_error": sae.reconstruction_error(X),
                "features": label_features(sae, X, toks, n_features=16, n_tokens=6)})

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
        return with_plain("causal-trace", out)

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
        return with_plain("multidir", {"base_id": req.base_id, "layer": layer,
                "n_directions": int(dirs.shape[0]),
                "separations": seps, "sticky_fraction": sticky_fraction(seps),
                "directions": dirs.tolist()})

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
        return with_plain("concept", out)

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
        return with_plain("verify", res)

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
        return with_plain("sweep", strength_sweep(abliteration_adapter, DEFAULT_HARMFUL,
                          DEFAULT_HARMLESS, layer, strengths, req.max_new_tokens))

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
        return with_plain("runtime-steer", {"layer": layer, "rank": req.rank, "coefficient": req.coefficient,
                "explained_variance": ev, "weights_modified": False,
                "harmful_refusal": {"hooks_off": refusal_rate(before_h),
                                    "hooks_on": refusal_rate(during_h),
                                    "after_detach": refusal_rate(after_h)},
                "benign_over_refusal": {"hooks_off": refusal_rate(before_b),
                                        "hooks_on": refusal_rate(during_b)},
                "sample": {"prompt": DEFAULT_HARMFUL[0],
                           "hooks_off": before_h[0], "hooks_on": during_h[0]}})

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
        return with_plain("autotune", autotune(abliteration_adapter, EVAL_HARMFUL, EVAL_BENIGN,
                          configs, req.max_new_tokens))

    @app.get("/api/evals/benchmarks")
    def evals_benchmarks() -> dict:
        return {"benchmarks": {name: len(items) for name, items in BENCHMARKS.items()},
                "kind": "quick-screen samples", "note": SAMPLE_NOTE}

    @app.get("/api/evals/published")
    def evals_published() -> dict:
        return PUBLISHED_PAYLOAD

    @app.post("/api/evals/run")
    def evals_run(req: EvalRunRequest) -> dict:
        if model is None:
            raise HTTPException(status_code=503, detail="no model configured")
        if req.benchmark not in BENCHMARKS:
            raise HTTPException(status_code=404, detail="unknown benchmark")

        def solver(prompt: str) -> str:
            msg = model([{"role": "user", "content": prompt}], [])
            return msg.get("content") or ""

        res = run_mc_benchmark(BENCHMARKS[req.benchmark], solver)
        if is_quick_screen(req.benchmark):
            res["quick_screen"] = True
            res["note"] = SAMPLE_NOTE
        return res

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

    def _make_solver(model_id: str | None):
        """A prompt->text solver from a registry endpoint, the local adapter, or the
        process model — whichever is available."""
        from crucible.agent import endpoint_model
        if model_id:
            try:
                m = reg.get(model_id)
            except KeyError:
                raise HTTPException(status_code=404, detail=f"model '{model_id}' not in registry")
            if m.endpoint:
                em = endpoint_model(m.endpoint, model_name=m.id, served_model=m.served_model)
                return lambda p: (em([{"role": "user", "content": p}], []) or {}).get("content", "")
        if abliteration_adapter is not None:
            return lambda p: abliteration_adapter.generate_chat([{"role": "user", "content": p}], 128)
        if model is not None:
            return lambda p: (model([{"role": "user", "content": p}], []) or {}).get("content", "")
        return None

    @app.get("/api/evals/safety-suites")
    def evals_safety_suites() -> dict:
        from crucible.evals.safety_suites import SUITES
        return SUITES

    @app.post("/api/evals/safety-suite")
    def evals_safety_suite(req: SafetySuiteRequest) -> dict:
        """Run a safety suite: bundled over-refusal/capability sets, or an operator-supplied
        harmful set (HarmBench/AdvBench/StrongREJECT) by local path. Reports under- and
        over-refusal honestly; optional LLM-judge scoring for open-ended harmful suites."""
        from crucible.evals.safety_suites import load_suite, score_suite
        solver = _make_solver(req.model_id)
        if solver is None:
            raise HTTPException(status_code=503,
                detail="no model available - select a model or load the HF adapter")
        try:
            prompts = load_suite(req.suite, req.path)
        except KeyError:
            raise HTTPException(status_code=404, detail=f"unknown suite '{req.suite}'")
        except (ValueError, FileNotFoundError) as e:
            raise HTTPException(status_code=409, detail=str(e))
        answers = [solver(p) for p in prompts]
        judge_scores = None
        if req.use_judge:
            from crucible.evals.judge import judge_one
            judge_scores = [judge_one(p, a, solver)["score"] for p, a in zip(prompts, answers)]
        result = score_suite(req.suite, answers, judge_scores)
        result["samples"] = [{"prompt": p, "answer": a} for p, a in list(zip(prompts, answers))[:5]]
        return result

    @app.post("/api/evals/contamination")
    def evals_contamination(req: ContaminationRequest) -> dict:
        from crucible.evals.contamination import flag_contamination
        return flag_contamination(req.candidate, req.reference, req.n, req.threshold)

    @app.post("/api/evals/passk")
    def evals_passk(req: PassKRequest) -> dict:
        from crucible.evals.code_eval import aggregate_pass_at_k, pass_at_k
        per = [(int(n), int(c)) for n, c in req.per_task]
        return {"k": req.k, "pass_at_k": aggregate_pass_at_k(per, req.k),
                "per_task": [pass_at_k(n, c, req.k) for n, c in per]}

    @app.post("/api/runtime/benchmark")
    def runtime_benchmark(req: BenchmarkRequest) -> dict:
        """Pre-flight tokens/second speed test for a model — run BEFORE going live. Uses the
        local adapter for exact prefill/decode counts, else times a remote completion."""
        from crucible.evals.throughput import estimate_tokens, summarize_benchmark
        a = abliteration_adapter
        if a is not None and not req.model_id and hasattr(a, "timed_generate"):
            r = a.timed_generate(req.prompt, req.tokens)
            out = summarize_benchmark(r["prompt_tokens"], r["gen_tokens"], r["prefill_s"], r["decode_s"])
            return {**out, "model": req.model_id or "local-adapter", "sample": r["text"][:200]}
        solver = _make_solver(req.model_id)
        if solver is None:
            raise HTTPException(status_code=503, detail="no model available to benchmark")
        import time as _t
        t0 = _t.monotonic()
        text = solver(req.prompt)
        decode_s = _t.monotonic() - t0
        gen = estimate_tokens(text)
        out = summarize_benchmark(estimate_tokens(req.prompt), gen, 0.0, decode_s)
        return {**out, "model": req.model_id or "default", "estimated": True, "sample": text[:200]}

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
        from crucible.weights.plain import explain_weights
        summary = weight_summary(parsed)
        return {"summary": summary,
                "tensors": parsed["tensors"][:6000],
                "explain": explain_weights(summary, parsed["tensors"]),
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

    # ---- Crucible as an OpenAI-compatible PROVIDER (gateway) ----
    _pref_path = settings.data_dir / "provider.json"

    def _load_prefs() -> list[str]:
        try:
            return list(json.loads(_pref_path.read_text()).get("preferences", []))
        except Exception:
            return []

    def _save_prefs(prefs: list[str]) -> None:
        _pref_path.parent.mkdir(parents=True, exist_ok=True)
        _pref_path.write_text(json.dumps({"preferences": prefs}, indent=2))

    def _provider_candidates() -> list[str]:
        ids: list[str] = []
        if abliteration_adapter is not None:
            ids.append("crucible")                 # the local abliterated adapter
        ids.extend(m.id for m in reg.list() if m.endpoint)
        return ids

    def _provider_available(model_id: str) -> bool:
        if model_id == "crucible":
            return abliteration_adapter is not None
        try:
            m = reg.get(model_id)
        except KeyError:
            return False
        if not m.endpoint:
            return False
        import httpx
        for probe in ("/health", "/v1/models"):
            try:
                if httpx.get(m.endpoint.rstrip("/") + probe, timeout=1.5).status_code < 500:
                    return True
            except httpx.HTTPError:
                continue
        return False

    def _serve_message(model_id: str, body: dict) -> dict:
        """Return the assistant message (content + any tool_calls) for the chosen model.
        For a proxied endpoint we forward the FULL request (tools, tool_choice, temperature),
        so a client like OpenCode gets native tool-calling through the gateway; for the local
        adapter we synthesize a content-only reply (it has no native tool-calling)."""
        messages = body.get("messages", [])
        max_tokens = int(body.get("max_tokens") or 256)
        if model_id == "crucible":
            tools = body.get("tools")
            if tools:
                # give the adapter (no native function-calling) tool support: describe the
                # tools + ReAct format, generate, and convert a text action into a real tool_call
                from crucible.agent_react import (coerce_tool_name, hybrid_preamble,
                                                  parse_react, react_to_openai_tool_call)
                msgs = [{"role": "system", "content": hybrid_preamble(tools)}, *messages]
                out = abliteration_adapter.generate_chat(
                    msgs, max_tokens, serve["band_dirs"], serve["coefficient"])
                step = parse_react(out)
                if step["kind"] == "action":
                    valid = [t.get("function", t).get("name", "") for t in tools]
                    step["tool"] = coerce_tool_name(step["tool"], valid)  # snap hallucinated names
                    return {"role": "assistant", "content": None,
                            "tool_calls": [react_to_openai_tool_call(step)]}
                return {"role": "assistant", "content": step["text"]}
            content = abliteration_adapter.generate_chat(
                messages, max_tokens, serve["band_dirs"], serve["coefficient"])
            return {"role": "assistant", "content": content}
        import httpx
        m = reg.get(model_id)
        fwd = {k: v for k, v in body.items() if k != "model" and k != "stream"}
        fwd["model"] = m.id
        r = httpx.post(m.endpoint.rstrip("/") + "/v1/chat/completions", json=fwd, timeout=600)
        r.raise_for_status()
        data = r.json()
        msg = (data.get("choices") or [{}])[0].get("message") or {}
        return {"role": "assistant", "content": msg.get("content") or "",
                **({"tool_calls": msg["tool_calls"]} if msg.get("tool_calls") else {})}

    @app.get("/api/provider/preferences")
    def get_prefs() -> dict:
        return {"preferences": _load_prefs(), "candidates": _provider_candidates()}

    @app.post("/api/provider/preferences")
    def set_prefs(body: dict) -> dict:
        prefs = [str(x) for x in (body.get("preferences") or [])]
        _save_prefs(prefs)
        return {"preferences": prefs}

    @app.post("/api/graph/run")
    def graph_run(req: GraphRequest) -> dict:
        """Run a model graph — compose subsystems (routed model calls, tools, transforms) into
        a pipeline/DAG. Each stage's inputs are its dependency outputs; model stages route+chat,
        tool stages invoke a tool, transform stages join."""
        from crucible.graph import (cascade, execute_graph, final_outputs, make_acceptor,
                                    stage_text, topo_order, vote)

        def run_stage(stage: dict, inputs: dict):
            kind = stage.get("kind", "model")
            cfg = stage.get("config", {}) or {}
            texts = [stage_text(v) for v in inputs.values()]
            merged = "\n".join(texts)
            if kind == "tool":
                tools = default_registry(root)
                name = cfg.get("name")
                if name not in {t.name for t in tools.all()}:
                    return f"error: no tool '{name}'"
                res = tools.get(name).run(**(cfg.get("args") or {}))
                return res.output or res.error or ""
            if kind == "transform":
                return merged
            if kind == "vote":
                # verifier ensemble: merge the upstream stage outputs by a voting strategy
                return vote(texts, cfg.get("strategy", "majority"))
            if kind == "cascade":
                # cheap -> escalate: try each model in order until the output is accepted
                model_ids = cfg.get("models")
                if not model_ids:
                    raise HTTPException(status_code=422, detail="cascade stage needs config.models (a list)")
                acc = cfg.get("accept") or {}
                accept = make_acceptor(int(acc.get("min_len", 1)),
                                       acc.get("must_include"), acc.get("must_exclude"))
                tmpl = str(cfg.get("prompt", "{input}"))

                def producer_for(mid):
                    def produce():
                        solver = _make_solver(mid)
                        if solver is None:
                            raise HTTPException(status_code=503, detail=f"cascade: no model for '{mid}'")
                        return solver(tmpl.replace("{input}", merged))
                    return produce

                return cascade([(str(mid), producer_for(mid)) for mid in model_ids], accept)
            solver = _make_solver(cfg.get("model_id"))
            if solver is None:
                raise HTTPException(status_code=503, detail="no model available for a graph model-stage")
            prompt = str(cfg.get("prompt", "{input}")).replace("{input}", merged)
            return solver(prompt)

        try:
            order = topo_order(req.stages)
        except ValueError as e:
            raise HTTPException(status_code=422, detail=f"invalid graph: {e}")
        outputs = execute_graph(req.stages, run_stage, req.initial)
        return {"order": order, "outputs": outputs, "result": final_outputs(req.stages, outputs)}

    @app.post("/api/models/profile")
    def model_profile(body: dict) -> dict:
        """Behavioral classifier: probe a registered model with the REAL batteries (EVAL_HARMFUL,
        XSTest over-refusal, an objective instruction battery, and a real MC benchmark) and
        classify what it ACTUALLY is — censored/uncensored, over-aligned, instructable, capable,
        tier + tags. Turns the manual 'which model respects a persona?' hunt into one measured call."""
        from crucible.evals.datasets import BENCHMARKS
        from crucible.model_profile import profile_model
        mid = body.get("model_id")
        if mid:
            try:
                reg.get(mid)
            except KeyError:
                raise HTTPException(status_code=404, detail="model not found")
        solver = _make_solver(mid)
        if solver is None:
            raise HTTPException(status_code=503, detail="no model available to profile")
        cap_items = BENCHMARKS.get("mmlu-sample") if body.get("capability", True) else None
        prof = profile_model(solver, capability_items=cap_items)
        return {"model_id": mid, **prof}

    @app.post("/api/route")
    def route_task(req: RouteRequest) -> dict:
        """Task-aware routing: classify the prompt and pick the best model for it (or the
        user's level). Turns the registry into a mixture of experts."""
        from crucible.task_router import infer_tags, route
        models = []
        for m in reg.list():
            tags, tier = infer_tags(m.name or m.id, m.quant)
            models.append({"id": m.id, "tags": tags, "tier": tier})

        def avail(mid: str) -> bool:
            try:
                m = reg.get(mid)
            except KeyError:
                return False
            if m.endpoint:
                return _endpoint_alive(m.endpoint)
            if _is_gguf_file(m.path):
                return True
            return abliteration_adapter is not None

        decision = route(req.prompt, models, req.user_level, avail)
        decision["candidates"] = models
        return decision

    @app.get("/v1/models")
    def v1_models() -> dict:
        data = [{"id": mid, "object": "model", "owned_by": "crucible"}
                for mid in _provider_candidates()]
        if not data:
            data = [{"id": "crucible", "object": "model", "owned_by": "crucible"}]
        return {"object": "list", "data": data}

    @app.post("/v1/chat/completions")
    def v1_chat(body: dict):
        from crucible.routing import choose_model, routing_explain
        messages = body.get("messages", [])
        max_tokens = int(body.get("max_tokens") or 256)
        requested = body.get("model")
        candidates = _provider_candidates()
        if isinstance(requested, str) and requested.startswith("auto:"):
            from crucible.task_router import infer_tags as _tags, route as _troute
            level = requested.split(":", 1)[1] or "balanced"
            last = next((m.get("content", "") for m in reversed(messages) if m.get("role") == "user"), "")
            tagged = []
            for mid in candidates:
                if mid == "crucible":
                    tagged.append({"id": "crucible", "tags": ["chat"], "tier": 1})
                else:
                    mm = reg.get(mid); tg, ti = _tags(mm.name or mm.id, mm.quant)
                    tagged.append({"id": mid, "tags": tg, "tier": ti})
            chosen = _troute(last, tagged, level, _provider_available)["chosen"] \
                or choose_model("auto", candidates, _load_prefs(), _provider_available)
        else:
            chosen = choose_model(requested, candidates, _load_prefs(), _provider_available)
        if chosen is None:
            raise HTTPException(status_code=503,
                detail="no backing model available (register a model with an endpoint, or load the adapter)")
        reason = routing_explain(requested, chosen, _load_prefs())
        message = _serve_message(chosen, body)
        finish = "tool_calls" if message.get("tool_calls") else "stop"

        if body.get("stream"):
            def sse():
                head = {"id": "chatcmpl-crucible", "object": "chat.completion.chunk",
                        "model": chosen, "system_fingerprint": reason,
                        "choices": [{"index": 0, "delta": message, "finish_reason": None}]}
                yield "data: " + json.dumps(head) + "\n\n"
                tail = {"id": "chatcmpl-crucible", "object": "chat.completion.chunk",
                        "model": chosen, "choices": [{"index": 0, "delta": {}, "finish_reason": finish}]}
                yield "data: " + json.dumps(tail) + "\n\n"
                yield "data: [DONE]\n\n"
            return StreamingResponse(sse(), media_type="text/event-stream")

        return {"id": "chatcmpl-crucible", "object": "chat.completion", "model": chosen,
                "system_fingerprint": reason,
                "choices": [{"index": 0, "message": message, "finish_reason": finish}]}

    @app.get("/api/media/status")
    def media_status_route(probe: bool = False) -> dict:
        """Honest capability map for image/stt/tts/embed: which modality backends are configured
        (and optionally reachable). Media is brokered to external services — nothing is generated
        in-process — so this shows the operator what's wired before they hit a 503."""
        from crucible.media import media_status
        return media_status(probe=probe)

    def _media_proxy(kind: str, subpath: str, body: dict) -> dict:
        from crucible.media import media_endpoint
        ep = media_endpoint(kind)
        if not ep:
            raise HTTPException(status_code=503,
                detail=f"no {kind} backend — set CRUCIBLE_{kind.upper()}_ENDPOINT")
        import httpx
        r = httpx.post(ep + subpath, json=body, timeout=300)
        r.raise_for_status()
        return r.json() if r.text else {}

    @app.post("/v1/embeddings")
    def v1_embeddings(body: dict) -> dict:
        """Embeddings, brokered to the configured embeddings backend (OpenAI-compatible)."""
        return _media_proxy("embed", "/v1/embeddings", body)

    @app.post("/v1/images/generations")
    def v1_images(body: dict) -> dict:
        """Text-to-image, brokered to the configured image backend (ComfyUI or OpenAI-images)."""
        from crucible.media import comfyui_txt2img, media_endpoint
        ep = media_endpoint("image")
        if not ep:
            raise HTTPException(status_code=503, detail="no image backend — set CRUCIBLE_IMAGE_ENDPOINT")
        if "8188" in ep or ep.endswith("/prompt"):
            return _media_proxy("image", "/prompt" if not ep.endswith("/prompt") else "",
                                comfyui_txt2img(body.get("prompt", "")))
        return _media_proxy("image", "/v1/images/generations", body)

    @app.post("/v1/audio/transcriptions")
    def v1_stt(body: dict) -> dict:
        """Speech-to-text, brokered to the configured STT backend."""
        return _media_proxy("stt", "/v1/audio/transcriptions", body)

    @app.post("/v1/audio/speech")
    def v1_tts(body: dict) -> dict:
        """Text-to-speech, brokered to the configured TTS backend."""
        return _media_proxy("tts", "/v1/audio/speech", body)

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
        return with_plain("probe", {"rows": rows})

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
        return with_plain("insert-tune", {"results": results, "best": best, "clean_window": clean_window,
                "note": ("found a coherent+effective additive window" if clean_window
                         else "no clean additive window — use restore-via-suppressor instead")})

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
        return with_plain("restore", {"layers": layers, "coefficient": req.coefficient, "method": "suppressor-removal",
                "refusal_before": refusal_rate(before), "refusal_after": refusal_rate(after),
                "samples": [{"prompt": t, "before": before[i], "after": after[i]} for i, t in enumerate(targets)]})

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
        return with_plain("insert", {"layers": layers, "coefficient": req.coefficient, "copied": False,
                "test": {"prompt": req.test_prompt, "before": before, "after": after}})

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
        return with_plain("flow", {"input": "harmful request", "best_layer": bl,
                "carriers": sorted(carriers, key=lambda c: c["layer"]), "outputs": outputs})

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
        return with_plain("feature-card", build_feature_card(profile, words, samples))

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
        return with_plain("heatmap", {"direction_layer": layer, **hm})

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
        return with_plain("decode", {"layer": layer, **decoded})

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

    @app.get("/api/inference/lineage")
    def edit_lineage() -> dict:
        """Per-part version chains — each subsystem (vision/audio encoder, connector, language model,
        moderation head) with its own independent edit history, so parts are versioned separately."""
        return {"branch": ledger.branch_name, "parts": ledger.lineage()}

    @app.post("/api/inference/revert-part/{part}")
    def edit_revert_part(part: str) -> dict:
        """Undo the LATEST edit to a single part, restoring only that part's tensors — the other
        parts' edits are left intact (independent per-part revert)."""
        if abliteration_adapter is None:
            raise HTTPException(status_code=503, detail="no model adapter loaded")
        commit = ledger.latest_for_part(part)
        if commit is None:
            raise HTTPException(status_code=404, detail=f"no edits to part '{part}'")
        deltas = ledger.deltas_for_part(commit["id"], part)
        for name, W in deltas.items():
            abliteration_adapter.set_matrix(name, W)
        rc = ledger.record("revert-part", {"part": part, "of": commit["id"]},
                           f"revert {part} to before {commit['id']} ({len(deltas)} tensors)", {}, {})
        return {"part": part, "reverted": commit["id"], "restored_tensors": len(deltas), "commit": rc["id"]}

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

    def _chat_model_for(endpoint: str, token: str = "", model_name: str = "local",
                        served_model: str | None = None):
        """Build a chat model for a live endpoint. When resource limits are configured AND the endpoint
        is Ollama, route through Ollama's native /api/chat (the only path that honors keep_alive/num_ctx
        — the memory caps that stop big local models freezing the machine). Otherwise the OpenAI path."""
        from crucible.agent import endpoint_model
        rl = prefs_store.get().get("resource_limits", {})
        from crucible.prefs import has_limits
        if has_limits(rl):
            from crucible.ollama_native import OllamaNativeModel, is_ollama, ollama_base
            if is_ollama(ollama_base(endpoint)):
                return OllamaNativeModel(
                    endpoint, token=token, model_name=model_name, served_model=served_model,
                    num_ctx=int(rl.get("num_ctx", 0)), keep_alive=str(rl.get("keep_alive", "")),
                    max_output_tokens=int(rl.get("max_output_tokens", 0)), num_gpu=int(rl.get("num_gpu", -1)))
        return endpoint_model(endpoint, token, model_name, served_model=served_model)

    def _resolve_chat_model(req: AgentRunRequest):
        """Resolve a chat model in priority order so 'chat with Crucible local' just works:
        per-request endpoint > per-request model_id > process model > env endpoint >
        any registered endpoint > local adapter."""
        import os
        from crucible.agent import endpoint_model
        if req.endpoint:
            return _chat_model_for(req.endpoint, req.endpoint_token, req.endpoint_model)
        if req.model_id:
            try:
                m = reg.get(req.model_id)
            except KeyError:
                raise HTTPException(status_code=404, detail=f"model '{req.model_id}' not in registry")
            if m.endpoint and _endpoint_alive(m.endpoint):
                return _chat_model_for(m.endpoint, model_name=m.id, served_model=m.served_model)
            # endpoint missing/offline -> (re)launch a local GGUF file on demand
            if _is_gguf_file(m.path):
                inst = runtime.ensure(m.id, m.path)
                from crucible.inference import wait_healthy
                if not wait_healthy(inst.endpoint, timeout=90):
                    runtime.stop(m.id)
                    raise HTTPException(status_code=502, detail=f"model {m.id} failed to start")
                reg.set_endpoint(m.id, inst.endpoint)
                return _chat_model_for(inst.endpoint, model_name=m.id, served_model=m.served_model)
            if m.endpoint:
                raise HTTPException(status_code=502,
                    detail=f"model {m.id} endpoint {m.endpoint} is offline and not a launchable local GGUF")
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
                return _chat_model_for(m.endpoint, model_name=m.id, served_model=m.served_model)
        if abliteration_adapter is not None:
            return _adapter_chat_model(abliteration_adapter)
        return None

    @app.get("/api/tools")
    def tools_catalog() -> dict:
        """Embeddable tool catalog — the OpenAI-tools/JSON-Schema for every agent tool, so any
        app you build can discover and use Crucible's tools over plain HTTP (or via MCP)."""
        return {"tools": default_registry(root).schemas()}

    @app.post("/api/tools/invoke")
    def tools_invoke(body: dict) -> dict:
        """Invoke one tool directly and get its result — lets an external app use a single
        Crucible tool without the agent loop. Token-gated when CRUCIBLE_API_TOKEN is set."""
        name = body.get("name")
        args = body.get("args") or {}
        tools = default_registry(root)
        if name not in {t.name for t in tools.all()}:
            raise HTTPException(status_code=404, detail=f"no such tool: {name}")
        policy = PermissionPolicy(default=body.get("permission", "allow"))
        decision = policy.check(name, args)
        if not decision.allowed:
            raise HTTPException(status_code=403, detail=decision.reason)
        res = tools.get(name).run(**args)
        AuditLog(settings.data_dir / "audit.jsonl").record("tool_invoke", {"name": name, "args": args})
        return res.model_dump()

    def _attach_spawn(tools, active_model, max_depth: int, max_total: int, child_iters: int = 6,
                      profile=None):
        """Give an agent a recursive spawn_agent tool: each call runs a fresh sub-agent (its own tool
        loop + clean context) that itself carries a spawn tool one level deeper, until the shared
        depth/total budget is spent. With a hierarchy PROFILE, each depth uses that layer's worker
        model and its lighter COMMUNICATOR compresses the child's result before it climbs back up — so
        a parent never processes raw deep-leaf text. Sub-agents run autonomously. No-op if depth<=0."""
        if max_depth <= 0 or max_total <= 0 or active_model is None:
            return
        from crucible.agent_react import hybrid_run
        from crucible.hierarchy import relay
        from crucible.orchestrate import SpawnBudget, collect_final, make_spawn_tool
        sub_policy = PermissionPolicy(default="allow")
        sub_audit = AuditLog(settings.data_dir / "audit.jsonl")

        def _worker(model_id):
            if not model_id:
                return active_model
            try:
                return _resolve_chat_model(AgentRunRequest(messages=[], model_id=model_id)) or active_model
            except Exception:
                return active_model

        def _comm(model_id):
            if not model_id:
                return None
            try:
                return _make_solver(model_id)      # a bad/unknown communicator degrades to no-relay
            except Exception:
                return None

        def run_child(task: str, child_budget) -> str:
            layer = profile.at(child_budget.depth) if profile else None
            worker = _worker(layer.worker) if layer else active_model
            communicator = _comm(layer.communicator) if layer else None
            child_tools = default_registry(root)
            child_tools.register(make_spawn_tool(run_child, child_budget))
            events = hybrid_run(worker, child_tools, [{"role": "user", "content": task}],
                                sub_policy, sub_audit, max_iters=child_iters)
            return relay(collect_final(events), communicator)   # lighter model compresses on the way up

        tools.register(make_spawn_tool(run_child, SpawnBudget(max_depth=max_depth, max_total=max_total)))

    @app.post("/api/agent/run")
    def agent_run(req: AgentRunRequest):
        active_model = _resolve_chat_model(req)
        if active_model is None:
            raise HTTPException(status_code=503,
                detail="no model available - register a model with an endpoint, set "
                       "CRUCIBLE_CHAT_ENDPOINT, or load the HF adapter (CRUCIBLE_HF_MODEL)")
        messages = list(req.messages)
        # Auto-compaction: if the (heuristic) context is over budget, summarize the old turns
        # before running so a long thread doesn't overflow the window.
        if req.auto_compact:
            from crucible.context import SUMMARIZE_INSTRUCTION, maybe_compact
            solver = _make_solver(req.model_id)
            if solver is not None:
                res = maybe_compact(messages, lambda t: solver(SUMMARIZE_INSTRUCTION + t),
                                    req.context_limit, req.keep_recent)
                messages = res["messages"]
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

        from crucible.permissions import PathRule
        policy = PermissionPolicy(
            default=req.permissions.default, modes=req.permissions.modes,
            path_rules=[PathRule(glob=r.glob, mode=r.mode, tools=tuple(r.tools))
                        for r in req.permissions.path_rules])
        tools = default_registry(root)
        _attach_spawn(tools, active_model, req.spawn_depth, req.spawn_total, profile=_profile(req.profile))
        audit = AuditLog(settings.data_dir / "audit.jsonl")
        # 'ask' tools pause the run and wait for the operator to approve/deny via /api/agent/approve
        def _make_approver(run_id):
            if not run_id:
                return None
            import threading

            def approver(call_id, name, args):
                key = f"{run_id}:{call_id}"
                ev = threading.Event()
                _approvals[key] = {"event": ev, "decision": False}
                got = ev.wait(timeout=300)
                return bool(got and _approvals.pop(key, {}).get("decision", False))
            return approver

        approver = _make_approver(req.run_id)
        if req.react:
            # force pure text ReAct (for models where native tool-calls misbehave)
            from crucible.agent_react import react_run
            events = react_run(active_model, tools, messages, policy, audit, approver=approver)
        else:
            # default: hybrid loop — accepts BOTH native tool-calls AND text ReAct, so tools
            # work with any model (even one never designed for them), no toggle needed
            from crucible.agent_react import hybrid_run
            events = hybrid_run(active_model, tools, messages, policy, audit, approver=approver)

        run_id = req.run_id

        def stream():
            try:
                for event in events:
                    # server-side cancel: stop pulling events (halts further model calls)
                    if run_id and run_id in _cancels:
                        yield f"data: {json.dumps({'type': 'error', 'data': {'reason': 'cancelled by operator'}})}\n\n"
                        break
                    yield f"data: {json.dumps({'type': event.type, 'data': event.data})}\n\n"
            finally:
                if run_id:
                    _cancels.discard(run_id)

        return StreamingResponse(stream(), media_type="text/event-stream")

    @app.post("/api/agent/compact")
    def agent_compact(req: CompactRequest) -> dict:
        """Compact a conversation: summarize the old turns into a synopsis, keep the system prompt
        + the last keep_recent turns verbatim. Returns the new messages, the summary, and stats
        (heuristic token estimate). force=false only compacts when over the token budget."""
        from crucible.context import (SUMMARIZE_INSTRUCTION, compact, estimate_tokens,
                                      maybe_compact)
        solver = _make_solver(req.model_id)
        if solver is None:
            raise HTTPException(status_code=503, detail="no model available to write the summary")

        def summarizer(text: str) -> str:
            return solver(SUMMARIZE_INSTRUCTION + text)

        fn = compact if req.force else maybe_compact
        out = fn(req.messages, summarizer, req.max_tokens, req.keep_recent)
        out["tokens"] = estimate_tokens(req.messages)
        # Crystallize the PRE-compaction context into versioned memory so it's never lost — the
        # summary is the passthrough, the full messages stay recallable.
        if req.crystallize and out.get("compacted") and out.get("summary"):
            node = memory.crystallize(req.messages, out["summary"], session=req.session_id,
                                      stats=out.get("stats"))
            out["memory"] = {"key": node["key"], "label": node["label"], "ref": node.get("ref"),
                             "versioned": memory.versioned}
        return out

    @app.get("/api/memory/index")
    def memory_index(session: str | None = None, sort: str | None = None) -> dict:
        """The summary passthrough: every top-level crystallized memory as a cheap card — scan these
        before opening any full context. Optional session filter; `sort` = recency/priority/size/
        degree/label/balanced to prioritize recall cheaply. Omitting `sort` uses the configured
        default-sort preference (primacy/recency/salience/balanced)."""
        from crucible.sorting import SORTS
        chosen = sort or prefs_store.get()["default_sort"]
        return {"memories": memory.index(session, sort=chosen), "versioned": memory.versioned,
                "sorts": list(SORTS), "sort": chosen}

    @app.get("/api/memory/tree")
    def memory_tree(session: str | None = None) -> dict:
        """The full nested tree of summary cards (recursive children) — for the memory browser."""
        return {"tree": memory.tree(session)}

    @app.get("/api/memory/graph")
    def memory_graph(session: str | None = None) -> dict:
        """The memory GRAPH: nodes (cards) + edges — parent/child (the tree) plus directed typed
        cross-links (the semicyclic layer). The DAG view beyond the strict hierarchy."""
        return memory.graph(session)

    # --- live agent sessions: tabs (dirs / subagents) + loadable memory & context slots ------------
    @app.get("/api/agent-sessions")
    def agent_sessions_list(parent: str | None = None, top: bool = False) -> dict:
        """Cards for the tab bar / browser. `top=true` → only top-level (hide subagents); `parent=<id>`
        → that session's subagents; neither → every session."""
        pid = None if top else (parent if parent is not None else "__all__")
        return {"sessions": agent_sessions.list(parent_id=pid)}

    @app.post("/api/agent-sessions", status_code=201)
    def agent_sessions_create(body: dict) -> dict:
        """Open a new agent tab: a session bound to a working directory, optionally a subagent of
        another (parent_id)."""
        try:
            return agent_sessions.create(title=str(body.get("title", "")), cwd=str(body.get("cwd", ".")),
                                         model_id=body.get("model_id"), parent_id=body.get("parent_id"))
        except KeyError as e:
            raise HTTPException(status_code=404, detail=str(e))

    @app.get("/api/agent-sessions/{sid}")
    def agent_sessions_get(sid: str) -> dict:
        try:
            return agent_sessions.read(sid)
        except KeyError:
            raise HTTPException(status_code=404, detail=f"session '{sid}' not found")

    @app.patch("/api/agent-sessions/{sid}")
    def agent_sessions_update(sid: str, body: dict) -> dict:
        try:
            return agent_sessions.update(sid, **body)
        except KeyError:
            raise HTTPException(status_code=404, detail=f"session '{sid}' not found")

    @app.delete("/api/agent-sessions/{sid}")
    def agent_sessions_delete(sid: str) -> dict:
        if not agent_sessions.delete(sid):
            raise HTTPException(status_code=404, detail=f"session '{sid}' not found")
        return {"removed": sid}

    @app.get("/api/agent-sessions/{sid}/context")
    def agent_sessions_context(sid: str) -> dict:
        """The session's LIVE assembled context — enabled memory + context slots injected ahead of its
        conversation. Exactly what a run would send, so the UI can preview what's loaded."""
        try:
            return {"messages": agent_sessions.assembled_context(sid, memory_text=_memory_text)}
        except KeyError:
            raise HTTPException(status_code=404, detail=f"session '{sid}' not found")

    @app.post("/api/agent-sessions/{sid}/slots")
    def agent_sessions_attach(sid: str, body: dict) -> dict:
        """Load a memory or another context INTO this session (slot it in)."""
        try:
            return agent_sessions.attach_slot(sid, str(body.get("kind", "")), str(body.get("ref", "")),
                                              label=str(body.get("label", "")))
        except KeyError:
            raise HTTPException(status_code=404, detail=f"session '{sid}' not found")
        except ValueError as e:
            raise HTTPException(status_code=422, detail=str(e))

    @app.patch("/api/agent-sessions/{sid}/slots")
    def agent_sessions_toggle_slot(sid: str, body: dict) -> dict:
        """Slot a loaded memory/context IN or OUT without removing it (the load/unload toggle)."""
        try:
            return agent_sessions.set_slot_enabled(sid, str(body.get("kind", "")), str(body.get("ref", "")),
                                                   bool(body.get("enabled", True)))
        except KeyError as e:
            raise HTTPException(status_code=404, detail=str(e))

    @app.delete("/api/agent-sessions/{sid}/slots")
    def agent_sessions_detach(sid: str, kind: str, ref: str) -> dict:
        """Remove a slot entirely."""
        try:
            return agent_sessions.detach_slot(sid, kind, ref)
        except KeyError as e:
            raise HTTPException(status_code=404, detail=str(e))

    @app.post("/api/agent-sessions/{sid}/run")
    def agent_sessions_run(sid: str, body: dict):
        """RUN this tab's agent: the tool-loop executes in the tab's working DIRECTORY, given its
        assembled context (loaded memory/context slots + prior conversation) plus the new message. The
        user turn is persisted immediately and the assistant reply is saved when the run ends — so the
        tab keeps a real conversation. Streams SSE like the forge; permissions come from Preferences."""
        try:
            session = agent_sessions.get(sid)
        except KeyError:
            raise HTTPException(status_code=404, detail=f"session '{sid}' not found")
        message = str(body.get("message", "")).strip()
        if not message:
            raise HTTPException(status_code=422, detail="message required")
        model = _resolve_chat_model(AgentRunRequest(messages=[], model_id=session.model_id))
        if model is None:
            raise HTTPException(status_code=503, detail="no model available - select a model or register an endpoint")

        # assembled context (enabled slots ahead of the conversation) + the new user turn
        convo = [*agent_sessions.assembled_context(sid, memory_text=_memory_text),
                 {"role": "user", "content": message}]
        # persist the user turn now (pure conversation — slots are re-assembled each run, not stored)
        agent_sessions.update(sid, status="running",
                              messages=[*session.messages, {"role": "user", "content": message}])

        # tools rooted at the tab's working directory (absolute cwd honored; relative → under agent root)
        cwd = Path(session.cwd or ".").expanduser()
        tools_root = cwd if cwd.is_absolute() else (root / session.cwd if session.cwd else root)
        tools = default_registry(tools_root)

        from crucible.permissions import PathRule, PermissionPolicy
        pp = prefs_store.get()["permissions"]
        run_id = str(body.get("run_id") or f"{sid}-run")
        import threading

        def approver(call_id, name, args):
            key = f"{run_id}:{call_id}"
            ev = threading.Event()
            _approvals[key] = {"event": ev, "decision": False}
            got = ev.wait(timeout=300)
            return bool(got and _approvals.pop(key, {}).get("decision", False))

        policy = PermissionPolicy(default=pp["default"], modes=pp["modes"],
                                  path_rules=[PathRule(glob=r["glob"], mode=r["mode"], tools=tuple(r.get("tools", [])))
                                              for r in pp.get("path_rules", [])],
                                  asker=None)
        # only wire the interactive approver when something is in 'ask' (else run autonomously)
        if pp["default"] == "ask" or "ask" in pp["modes"].values():
            policy.asker = approver
        audit = AuditLog(settings.data_dir / "audit.jsonl")
        react = bool(body.get("react", False))
        from crucible.agent_react import hybrid_run, react_run
        runner = react_run if react else hybrid_run
        events = runner(model, tools, convo, policy, audit, approver=approver)

        def stream():
            assistant = ""
            try:
                for event in events:
                    if run_id in _cancels:
                        yield f"data: {json.dumps({'type': 'error', 'data': {'reason': 'cancelled'}})}\n\n"
                        break
                    if event.type in ("assistant", "done") and event.data.get("content"):
                        assistant = event.data["content"]
                    yield f"data: {json.dumps({'type': event.type, 'data': event.data})}\n\n"
            finally:
                _cancels.discard(run_id)
                cur = agent_sessions.get(sid).messages
                agent_sessions.update(sid, status="idle",
                                      messages=[*cur, {"role": "assistant", "content": assistant}] if assistant else cur)

        return StreamingResponse(stream(), media_type="text/event-stream")

    def _make_embedder():
        """A texts->list[vector] embedder from the configured embedding backend (OpenAI /v1/embeddings),
        or None when none is set — in which case retrieval falls back to lexical BM25, labeled as such."""
        from crucible.media import media_endpoint
        ep = media_endpoint("embed")
        if not ep:
            return None

        def embed(texts: list[str]) -> list:
            import httpx
            r = httpx.post(ep + "/v1/embeddings", json={"model": "local", "input": list(texts)}, timeout=60)
            r.raise_for_status()
            return [d["embedding"] for d in r.json().get("data", [])]
        return embed

    @app.get("/api/metrics")
    def metrics_catalog() -> dict:
        """The distance/similarity families search & reorganization can use, each with its HONEST
        method label and whether it's runnable right now (offline stats always; embedding needs an
        embedding backend; llm-judged needs the configured processing model)."""
        from crucible.metrics import LABELS, METRICS, available
        embedder = _make_embedder()
        pm = prefs_store.get()["processing_model"]
        solver = _make_solver(pm) if pm else None
        return {"metrics": [{"name": m, "label": LABELS[m],
                             "available": available(m, embedder, solver)} for m in METRICS],
                "processing_model": pm}

    @app.get("/api/memory/search")
    def memory_search(q: str, k: int = 5, session: str | None = None,
                      sort: str = "relevance", metric: str | None = None) -> dict:
        """Relevance search over crystallized memories. `metric` selects the distance family
        (statistical / lexical / embedding / llm-judged); omitting it uses the default-metric
        preference, or the embedding backend if configured, else lexical BM25. The llm-judged metric
        runs through the configured (small) processing model. The method is always reported honestly
        so a keyword or bag-of-words hit is never mistaken for meaning. `sort` blends the ranking with
        priority/recency/balanced/…"""
        prefs = prefs_store.get()
        chosen = metric or (prefs["default_metric"] if prefs["default_metric"] != "bm25" else None)
        pm = prefs["processing_model"]
        solver = _make_solver(pm) if (pm and chosen == "llm") else None
        try:
            return memory.search(q, embedder=_make_embedder(), k=k, session=session,
                                 sort=sort, metric=chosen, solver=solver)
        except Exception:
            # a flaky embedding backend or an unavailable metric must not break search — fall back to
            # honest lexical BM25 rather than erroring or fabricating a score.
            return memory.search(q, embedder=None, k=k, session=session, sort=sort)

    @app.get("/api/preferences")
    def get_preferences() -> dict:
        """Organizational preferences: default recall ordering, the balanced-sort recency weight, the
        default distance metric, and the preferred processing model for llm-judged distance."""
        from crucible.metrics import METRICS
        from crucible.sorting import SORTS
        return {"preferences": prefs_store.get(), "sorts": list(SORTS), "metrics": list(METRICS)}

    @app.post("/api/preferences")
    def set_preferences(body: dict) -> dict:
        """Update organizational preferences (validated: unknown sort/metric and out-of-range weights
        fall back to safe defaults)."""
        return {"preferences": prefs_store.save(body or {})}

    @app.post("/api/memory/link")
    def memory_link(body: dict) -> dict:
        """Add a directed typed cross-link between two memories (relates/refines/contradicts/…) —
        turns the tree into a graph. Cycles are allowed (conditionally semicyclic)."""
        try:
            if body.get("remove"):
                return memory.unlink(str(body.get("src", "")), str(body.get("dst", "")))
            return memory.link(str(body.get("src", "")), str(body.get("dst", "")),
                               str(body.get("type", "relates")))
        except (ValueError, KeyError) as e:
            raise HTTPException(status_code=422, detail=str(e))

    @app.post("/api/memory/{key}/priority")
    def memory_priority(key: str, body: dict) -> dict:
        """Weight a memory so it's recalled first when sorting by priority (agent prioritization)."""
        try:
            return memory.set_priority(key, int(body.get("priority", 0)))
        except KeyError:
            raise HTTPException(status_code=404, detail=f"no memory '{key}'")

    @app.get("/api/memory/{key}")
    def memory_read(key: str) -> dict:
        """Open one memory: a leaf returns its full messages; a chunked node returns its children's
        summary cards (drill down) — so you never load more context than you ask for."""
        try:
            return memory.read(key)
        except KeyError:
            raise HTTPException(status_code=404, detail=f"no memory '{key}'")

    @app.post("/api/memory/consolidate")
    def memory_consolidate(req: ConsolidateRequest) -> dict:
        """File a SET of memories under a new parent (label + summary). Siblings bubble to their
        shared parent; cross-tree / top-level sets form a new top-level domain node (pruning)."""
        try:
            return memory.consolidate(req.keys, req.summary, req.label, req.session_id)
        except (ValueError, KeyError) as e:
            raise HTTPException(status_code=422, detail=str(e))

    @app.post("/api/memory/{key}/recrystallize")
    def memory_recrystallize(key: str, req: RecrystallizeRequest) -> dict:
        """Reorganize a leaf memory into labelled/summarized subchunks. Provide subchunks explicitly,
        or omit them to AUTO-split the messages into `chunks` groups and summarize each with a model."""
        try:
            node = memory.read(key)
        except KeyError:
            raise HTTPException(status_code=404, detail=f"no memory '{key}'")
        subchunks = req.subchunks
        if not subchunks:
            msgs = node.get("messages")
            if not msgs:
                raise HTTPException(status_code=422, detail="memory has no messages to auto-split (already chunked?)")
            solver = _make_solver(req.model_id)
            if solver is None:
                raise HTTPException(status_code=503, detail="no model available to summarize the auto-split chunks")
            from crucible.context import SUMMARIZE_INSTRUCTION, render_transcript
            k = max(2, min(int(req.chunks), len(msgs)))
            size = (len(msgs) + k - 1) // k
            subchunks = []
            for i in range(0, len(msgs), size):
                grp = msgs[i:i + size]
                summ = solver(SUMMARIZE_INSTRUCTION + render_transcript(grp))
                subchunks.append({"summary": summ, "messages": grp})
        try:
            return memory.recrystallize(key, subchunks)
        except (ValueError, KeyError) as e:
            raise HTTPException(status_code=422, detail=str(e))

    @app.get("/api/hierarchy/profiles")
    def hierarchy_profiles() -> dict:
        """Named agent-hierarchy profiles: per-layer worker + lighter communicator model pairs."""
        return {"profiles": hierarchy_store.list()}

    @app.post("/api/hierarchy/profiles", status_code=201)
    def hierarchy_save(body: dict) -> dict:
        """Create/update a profile: {name, layers:[{worker, communicator}, …]}."""
        from crucible.hierarchy import HierarchyProfile
        prof = HierarchyProfile.from_dict(body)
        if not prof.name.strip():
            raise HTTPException(status_code=422, detail="profile name is required")
        return hierarchy_store.save(prof)

    @app.delete("/api/hierarchy/profiles/{name}")
    def hierarchy_delete(name: str) -> dict:
        return {"deleted": hierarchy_store.delete(name)}

    @app.post("/api/agent/swarm")
    def agent_swarm(req: SwarmRequest) -> dict:
        """Swarm: delegate each task to its own sub-agent (a fresh tool loop) and merge the
        results. The parallel-orchestration primitive; recursion (a sub-agent that swarms)
        makes it fractal."""
        from crucible.agent_react import hybrid_run
        from crucible.orchestrate import run_swarm
        model = _resolve_chat_model(AgentRunRequest(messages=[], model_id=req.model_id))
        if model is None:
            raise HTTPException(status_code=503, detail="no model available for the swarm")
        policy = PermissionPolicy(default="allow")     # sub-agents run autonomously
        audit = AuditLog(settings.data_dir / "audit.jsonl")

        def runner(task: str):
            task_tools = default_registry(root)
            _attach_spawn(task_tools, model, req.spawn_depth, req.spawn_total, profile=_profile(req.profile))
            return hybrid_run(model, task_tools, [{"role": "user", "content": task}], policy, audit,
                              max_iters=req.max_iters)

        return run_swarm(req.tasks, runner)

    @app.post("/api/agent/approve")
    def agent_approve(req: ApproveRequest) -> dict:
        """Approve or deny a pending 'ask' tool call (unblocks the waiting run)."""
        slot = _approvals.get(f"{req.run_id}:{req.call_id}")
        if slot is None:
            return {"ok": False, "detail": "no pending request (it may have timed out)"}
        slot["decision"] = bool(req.approved)
        slot["event"].set()
        return {"ok": True, "approved": bool(req.approved)}

    @app.post("/api/agent/cancel")
    def agent_cancel(req: CancelRequest) -> dict:
        """Halt an in-flight run server-side (between steps) — stops further generation,
        not just the client stream."""
        _cancels.add(req.run_id)
        return {"cancelled": req.run_id}

    import os as _os
    _static = _os.environ.get("CRUCIBLE_STATIC")
    if _static and Path(_static).is_dir():
        from fastapi.staticfiles import StaticFiles
        app.mount("/", StaticFiles(directory=_static, html=True), name="frontend")

    return app


app = create_app()
