from crucible.guardrails.base import (  # noqa: F401
    GuardrailAction, GuardrailConfig, GuardrailResult, RegexRule, Stage)
from crucible.guardrails.constitution import Critic  # noqa: F401
from crucible.guardrails.engine import GuardrailsEngine  # noqa: F401
from crucible.guardrails.presets import PRESETS, SystemPromptPreset, get_preset  # noqa: F401


def build_engine(critic: Critic | None = None) -> GuardrailsEngine:
    return GuardrailsEngine(critic=critic)
