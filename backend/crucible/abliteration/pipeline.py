from __future__ import annotations
from typing import Protocol

import numpy as np

from crucible.abliteration.cards import build_model_card
from crucible.abliteration.direction import compute_refusal_direction
from crucible.registry import Model, Registry


class ModelAdapter(Protocol):
    hidden_size: int
    def activations(self, prompts: list[str], layer: int) -> np.ndarray: ...
    def writing_matrices(self) -> list[str]: ...
    def get_matrix(self, name: str) -> np.ndarray: ...
    def set_matrix(self, name: str, W: np.ndarray) -> None: ...
    def save(self, path: str) -> None: ...


class AbliterationPipeline:
    def __init__(self, adapter: ModelAdapter, registry: Registry):
        self.adapter = adapter
        self.registry = registry

    def compute_direction(self, harmful: list[str], harmless: list[str], layer: int) -> np.ndarray:
        return compute_refusal_direction(
            self.adapter.activations(harmful, layer),
            self.adapter.activations(harmless, layer))

    def abliterate(self, base: Model, harmful: list[str], harmless: list[str], layer: int,
                   out_path: str, variant_id: str, strength: float = 1.0
                   ) -> tuple[Model, dict, np.ndarray]:
        direction = self.compute_direction(harmful, harmless, layer)
        for name in self.adapter.writing_matrices():
            W = self.adapter.get_matrix(name)
            ablated = W - strength * np.outer(direction, direction @ W)
            self.adapter.set_matrix(name, ablated)
        self.adapter.save(out_path)
        variant = self.registry.register(Model(
            id=variant_id, name=variant_id, base_id=base.id, path=out_path,
            quant=base.quant, kind="abliterated", endpoint=None,
            created="2026-06-20",
            notes=f"abliterated from {base.id} @ layer {layer} (strength {strength})"))
        card = build_model_card(base.id, variant_id, "abliteration", layer, strength,
                                {"harmful": len(harmful), "harmless": len(harmless)},
                                self.adapter.hidden_size)
        return variant, card, direction
