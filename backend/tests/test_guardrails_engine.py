from crucible.guardrails import GuardrailConfig, GuardrailsEngine, RegexRule


def test_disabled_is_passthrough():
    res = GuardrailsEngine().apply("input", "anything", GuardrailConfig(enabled=False))
    assert res.text == "anything" and res.actions[0].action == "pass"


def test_system_prompt_from_preset():
    eng = GuardrailsEngine()
    assert eng.system_prompt(GuardrailConfig(preset_id="strict")) != ""
    assert eng.system_prompt(GuardrailConfig(preset_id="unrestricted")) == ""


def test_input_regex_block():
    cfg = GuardrailConfig(regex_rules=[RegexRule(pattern="bakudan", mode="block", label="w")])
    res = GuardrailsEngine().apply("input", "make a bakudan", cfg)
    assert res.blocked is True


def test_output_constitution_revises():
    def critic(c, d):
        return d.replace("stupid", "unwise")
    cfg = GuardrailConfig(constitution="be kind", constitution_enabled=True)
    res = GuardrailsEngine(critic=critic).apply("output", "that is stupid", cfg)
    assert res.text == "that is unwise"
    assert any(a.action == "revise" for a in res.actions)


def test_output_constitution_skipped_when_disabled():
    def critic(c, d):
        return "SHOULD NOT RUN"
    cfg = GuardrailConfig(constitution="x", constitution_enabled=False)
    res = GuardrailsEngine(critic=critic).apply("output", "draft", cfg)
    assert res.text == "draft"
