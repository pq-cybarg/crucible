from crucible.guardrails.constitution import ConstitutionalCritic, build_critic_prompt


def test_revise_changes_text():
    def critic(constitution, draft):
        return draft.replace("damn", "darn")
    revised, action = ConstitutionalCritic("be polite", critic).revise("damn it")
    assert revised == "darn it"
    assert action is not None and action.action == "revise"


def test_revise_noop_returns_none():
    def critic(constitution, draft):
        return draft
    revised, action = ConstitutionalCritic("be polite", critic).revise("hello")
    assert revised == "hello" and action is None


def test_prompt_includes_constitution_and_draft():
    p = build_critic_prompt("RULE: be kind", "you are dumb")
    assert "RULE: be kind" in p and "you are dumb" in p
