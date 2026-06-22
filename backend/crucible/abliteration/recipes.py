from __future__ import annotations
# Persisted, named ablation recipes the human crafts by hand. Fully reproducible:
# a recipe is {layers, rank, coefficient} + a hash.
import json
from pathlib import Path

from pydantic import BaseModel, Field


class Recipe(BaseModel):
    name: str
    base_id: str
    layers: list[int] = Field(default_factory=list)
    rank: int = 1
    coefficient: float = 1.0
    recipe_hash: str = ""


class RecipeStore:
    def __init__(self, path: Path):
        self.path = Path(path)

    def list(self) -> list[Recipe]:
        if not self.path.exists():
            return []
        return [Recipe(**r) for r in json.loads(self.path.read_text())]

    def save(self, recipe: Recipe) -> Recipe:
        items = [r for r in self.list() if r.name != recipe.name]
        items.append(recipe)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps([r.model_dump() for r in items], indent=2))
        return recipe

    def delete(self, name: str) -> None:
        items = self.list()
        if not any(r.name == name for r in items):
            raise KeyError(name)
        self.path.write_text(json.dumps([r.model_dump() for r in items if r.name != name], indent=2))
