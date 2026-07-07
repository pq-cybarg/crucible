from __future__ import annotations
import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel


class Model(BaseModel):
    id: str
    name: str
    base_id: str | None
    path: str
    quant: str
    kind: Literal["base", "abliterated", "steered"]
    endpoint: str | None
    created: str
    notes: str = ""
    # The exact model tag the upstream endpoint expects (e.g. Ollama's "llama3.2:latest"). None lets
    # EndpointModel auto-resolve it from /v1/models — so the registry id can be a friendly label.
    served_model: str | None = None


class Registry:
    def __init__(self, path: Path):
        self.path = Path(path)
        self._models: dict[str, Model] = {}
        self._load()

    def _load(self) -> None:
        if self.path.exists():
            data = json.loads(self.path.read_text())
            self._models = {m["id"]: Model(**m) for m in data}

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(
            json.dumps([m.model_dump() for m in self._models.values()], indent=2)
        )

    def list(self) -> list[Model]:
        return list(self._models.values())

    def get(self, id: str) -> Model:
        return self._models[id]

    def register(self, model: Model) -> Model:
        if model.id in self._models:
            raise ValueError(f"duplicate id: {model.id}")
        if model.kind != "base":
            for existing in self._models.values():
                if existing.path == model.path:
                    raise ValueError(f"path reuses an existing model file: {model.path}")
        self._models[model.id] = model
        self._save()
        return model

    def set_endpoint(self, id: str, endpoint: str) -> Model:
        m = self._models[id]
        updated = m.model_copy(update={"endpoint": endpoint})
        self._models[id] = updated
        self._save()
        return updated

    def remove(self, id: str) -> bool:
        """Forget a model (e.g. a dead BYO endpoint or an abandoned experiment). Returns False if the
        id was unknown. Only touches the registry entry — it never deletes weight files on disk."""
        if id not in self._models:
            return False
        del self._models[id]
        self._save()
        return True

    def lineage(self, id: str) -> list[Model]:
        chain: list[Model] = []
        cur: str | None = id
        while cur is not None:
            m = self._models[cur]
            chain.append(m)
            cur = m.base_id
        return list(reversed(chain))
