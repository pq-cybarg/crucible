from fastapi import FastAPI, HTTPException

from crucible.config import get_settings
from crucible.registry import Model, Registry


def create_app(registry: Registry | None = None) -> FastAPI:
    reg = registry or Registry(get_settings().registry_path)
    app = FastAPI(title="Crucible")

    @app.get("/api/health")
    def health():
        return {"ok": True}

    @app.get("/api/models")
    def list_models() -> list[Model]:
        return reg.list()

    @app.post("/api/models", status_code=201)
    def create_model(model: Model) -> Model:
        try:
            return reg.register(model)
        except ValueError as e:
            raise HTTPException(status_code=409, detail=str(e))

    @app.get("/api/models/{model_id}/lineage")
    def lineage(model_id: str) -> list[Model]:
        try:
            return reg.lineage(model_id)
        except KeyError:
            raise HTTPException(status_code=404, detail="model not found")

    return app


app = create_app()
