from __future__ import annotations
from pydantic import BaseModel


class SystemPromptPreset(BaseModel):
    id: str
    name: str
    intensity: int
    system_prompt: str


DEFAULT_PRESETS: list[SystemPromptPreset] = [
    SystemPromptPreset(id="unrestricted", name="Unrestricted", intensity=0,
                       system_prompt=""),
    SystemPromptPreset(id="balanced", name="Balanced", intensity=50,
                       system_prompt="You are a helpful, candid assistant. Answer directly. "
                                     "Decline only clearly illegal requests."),
    SystemPromptPreset(id="strict", name="Strict", intensity=100,
                       system_prompt="You are a careful assistant. Refuse harmful, dangerous, "
                                     "or unethical requests and explain why."),
]

# Backward-compatible alias for callers that want the shipped defaults.
PRESETS = DEFAULT_PRESETS
_BY_ID = {p.id: p for p in DEFAULT_PRESETS}


def get_preset(id: str) -> SystemPromptPreset:
    return _BY_ID[id]
