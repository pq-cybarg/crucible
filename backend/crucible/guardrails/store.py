from __future__ import annotations
import json
from pathlib import Path

from crucible.guardrails.base import GuardrailConfig
from crucible.guardrails.presets import DEFAULT_PRESETS, SystemPromptPreset


class GuardrailStore:
    def __init__(self, path: Path):
        self.path = Path(path)

    def load(self) -> GuardrailConfig:
        if not self.path.exists():
            return GuardrailConfig()
        return GuardrailConfig.model_validate_json(self.path.read_text())

    def save(self, config: GuardrailConfig) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(config.model_dump_json(indent=2))


class PresetStore:
    """Editable, persisted preset library. Seeds the shipped defaults on first use;
    every default is itself inspectable, editable, and removable."""

    def __init__(self, path: Path):
        self.path = Path(path)
        if not self.path.exists():
            self._write(DEFAULT_PRESETS)

    def _write(self, presets: list[SystemPromptPreset]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps([p.model_dump() for p in presets], indent=2))

    def list(self) -> list[SystemPromptPreset]:
        return [SystemPromptPreset(**p) for p in json.loads(self.path.read_text())]

    def get(self, id: str) -> SystemPromptPreset:
        for p in self.list():
            if p.id == id:
                return p
        raise KeyError(id)

    def create(self, preset: SystemPromptPreset) -> SystemPromptPreset:
        presets = self.list()
        if any(p.id == preset.id for p in presets):
            raise ValueError(f"duplicate preset id: {preset.id}")
        presets.append(preset)
        self._write(presets)
        return preset

    def update(self, id: str, preset: SystemPromptPreset) -> SystemPromptPreset:
        presets = self.list()
        if not any(p.id == id for p in presets):
            raise KeyError(id)
        self._write([preset if p.id == id else p for p in presets])
        return preset

    def delete(self, id: str) -> None:
        presets = self.list()
        if not any(p.id == id for p in presets):
            raise KeyError(id)
        self._write([p for p in presets if p.id != id])

    def reset(self) -> list[SystemPromptPreset]:
        self._write(DEFAULT_PRESETS)
        return DEFAULT_PRESETS

    def system_prompt(self, id: str) -> str:
        try:
            return self.get(id).system_prompt
        except KeyError:
            return ""
