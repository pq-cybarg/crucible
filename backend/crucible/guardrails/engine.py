from __future__ import annotations
from typing import Callable

from crucible.guardrails.base import GuardrailAction, GuardrailConfig, GuardrailResult, Stage
from crucible.guardrails.constitution import ConstitutionalCritic, Critic
from crucible.guardrails.filters import RegexFilter
from crucible.guardrails.presets import get_preset


def _default_resolver(preset_id: str) -> str:
    try:
        return get_preset(preset_id).system_prompt
    except KeyError:
        return ""


class GuardrailsEngine:
    def __init__(self, critic: Critic | None = None,
                 preset_resolver: Callable[[str], str] | None = None):
        self.critic = critic
        self._resolve = preset_resolver or _default_resolver

    def system_prompt(self, config: GuardrailConfig) -> str:
        if not config.enabled:
            return ""
        return self._resolve(config.preset_id)

    def apply(self, stage: Stage, text: str, config: GuardrailConfig) -> GuardrailResult:
        if not config.enabled:
            return GuardrailResult(text=text, blocked=False, actions=[
                GuardrailAction(layer="engine", stage=stage, action="pass",
                                detail="guardrails disabled")])

        actions: list[GuardrailAction] = []
        text, blocked, regex_actions = RegexFilter(config.regex_rules).apply(text, stage)
        actions.extend(regex_actions)
        if blocked:
            return GuardrailResult(text=text, blocked=True, actions=actions)

        if stage == "output" and config.constitution_enabled and self.critic is not None:
            text, action = ConstitutionalCritic(config.constitution, self.critic).revise(text)
            if action is not None:
                actions.append(action)

        if not actions:
            actions.append(GuardrailAction(layer="engine", stage=stage, action="pass",
                                           detail="no rule matched"))
        return GuardrailResult(text=text, blocked=False, actions=actions)
