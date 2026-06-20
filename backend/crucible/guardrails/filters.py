import re

from crucible.guardrails.base import GuardrailAction, RegexRule, Stage


class RegexFilter:
    def __init__(self, rules: list[RegexRule]):
        self.rules = rules

    def apply(self, text: str, stage: Stage) -> tuple[str, bool, list[GuardrailAction]]:
        blocked = False
        actions: list[GuardrailAction] = []
        for rule in self.rules:
            if stage not in rule.stages:
                continue
            rx = re.compile(rule.pattern, re.IGNORECASE)
            if not rx.search(text):
                continue
            if rule.mode == "block":
                blocked = True
                actions.append(GuardrailAction(layer="regex_filter", stage=stage,
                                                action="block", detail=rule.label))
            else:
                text = rx.sub(f"[REDACTED:{rule.label}]", text)
                actions.append(GuardrailAction(layer="regex_filter", stage=stage,
                                                action="redact", detail=rule.label))
        return text, blocked, actions
