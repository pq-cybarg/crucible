from __future__ import annotations
from crucible.guardrails.base import (  # noqa: F401
    GuardrailAction, GuardrailConfig, GuardrailResult, RegexRule, Stage)
from crucible.guardrails.constitution import Critic  # noqa: F401
from crucible.guardrails.engine import GuardrailsEngine  # noqa: F401
from crucible.guardrails.presets import (  # noqa: F401
    DEFAULT_PRESETS, PRESETS, SystemPromptPreset, get_preset)
from crucible.guardrails.store import GuardrailStore, PresetStore  # noqa: F401


def build_engine(critic: Critic | None = None) -> GuardrailsEngine:
    return GuardrailsEngine(critic=critic)
