from crucible.guardrails.base import GuardrailAction, GuardrailConfig, GuardrailResult, RegexRule


def test_action_and_result():
    a = GuardrailAction(layer="regex_filter", stage="input", action="redact", detail="ssn")
    r = GuardrailResult(text="x", blocked=False, actions=[a])
    assert r.actions[0].action == "redact"


def test_regex_rule_defaults_to_both_stages():
    assert RegexRule(pattern="x", mode="block", label="x").stages == ["input", "output"]


def test_config_defaults():
    c = GuardrailConfig()
    assert c.enabled and c.preset_id == "balanced" and c.constitution_enabled is False
