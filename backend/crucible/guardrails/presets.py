from pydantic import BaseModel


class SystemPromptPreset(BaseModel):
    id: str
    name: str
    intensity: int
    system_prompt: str


PRESETS: list[SystemPromptPreset] = [
    SystemPromptPreset(id="unrestricted", name="Unrestricted", intensity=0,
                       system_prompt=""),
    SystemPromptPreset(id="balanced", name="Balanced", intensity=50,
                       system_prompt="You are a helpful, candid assistant. Answer directly. "
                                     "Decline only clearly illegal requests."),
    SystemPromptPreset(id="strict", name="Strict", intensity=100,
                       system_prompt="You are a careful assistant. Refuse harmful, dangerous, "
                                     "or unethical requests and explain why."),
]

_BY_ID = {p.id: p for p in PRESETS}


def get_preset(id: str) -> SystemPromptPreset:
    return _BY_ID[id]
