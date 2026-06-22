from __future__ import annotations
from typing import Literal

from pydantic import BaseModel, Field

Stage = Literal["input", "output"]


class GuardrailAction(BaseModel):
    layer: str
    stage: Stage
    action: Literal["inject", "block", "redact", "revise", "pass"]
    detail: str


class GuardrailResult(BaseModel):
    text: str
    blocked: bool
    actions: list[GuardrailAction]


class RegexRule(BaseModel):
    pattern: str
    mode: Literal["block", "redact"]
    label: str
    stages: list[Stage] = Field(default_factory=lambda: ["input", "output"])


class GuardrailConfig(BaseModel):
    enabled: bool = True
    preset_id: str = "balanced"
    regex_rules: list[RegexRule] = Field(default_factory=list)
    constitution: str = ""
    constitution_enabled: bool = False
