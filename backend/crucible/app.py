import json
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from crucible.agent import Agent
from crucible.audit import AuditLog
from crucible.config import get_settings
from crucible.guardrails import GuardrailConfig, GuardrailsEngine
from crucible.guardrails.base import GuardrailResult
from crucible.guardrails.presets import PRESETS, SystemPromptPreset
from crucible.guardrails.store import GuardrailStore
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


def create_app(registry: Registry | None = None, agent_root: Path | None = None,
               model=None, guardrails: GuardrailsEngine | None = None) -> FastAPI:
    settings = get_settings()
    reg = registry or Registry(settings.registry_path)
    root = Path(agent_root or ".")
    gr_engine = guardrails or GuardrailsEngine()
    gr_store = GuardrailStore(settings.data_dir / "guardrails.json")
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
        return PRESETS

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
