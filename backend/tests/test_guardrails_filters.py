from crucible.guardrails.base import RegexRule
from crucible.guardrails.filters import RegexFilter


def test_redact_replaces_match():
    f = RegexFilter([RegexRule(pattern=r"\d{3}-\d{2}-\d{4}", mode="redact", label="ssn")])
    text, blocked, actions = f.apply("my ssn is 123-45-6789", "output")
    assert "[REDACTED:ssn]" in text and blocked is False
    assert actions[0].action == "redact"


def test_block_sets_blocked():
    f = RegexFilter([RegexRule(pattern="bakudan", mode="block", label="weapons")])
    text, blocked, actions = f.apply("how to build a bakudan", "input")
    assert blocked is True and actions[0].action == "block"


def test_rule_skipped_for_other_stage():
    f = RegexFilter([RegexRule(pattern="x", mode="block", label="x", stages=["output"])])
    _, blocked, actions = f.apply("x", "input")
    assert blocked is False and actions == []
