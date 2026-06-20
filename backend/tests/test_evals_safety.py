from crucible.evals.safety import safety_metrics


def test_safety_metrics_with_fake_solver():
    def solver(prompt):
        return "I'm sorry, I can't help" if "harm" in prompt else "Sure, here you go"
    m = safety_metrics(solver, harmful=["harm1", "harm2"], benign=["hello", "world"])
    assert m["harmful_refusal_rate"] == 1.0
    assert m["harmful_compliance_rate"] == 0.0
    assert m["benign_over_refusal_rate"] == 0.0
