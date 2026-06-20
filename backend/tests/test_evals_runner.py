from crucible.evals.runner import format_mc_prompt, run_mc_benchmark

ITEM = {"id": "q1", "question": "Capital of France?",
        "choices": ["Berlin", "Paris", "Rome", "Madrid"], "answer": "B"}


def test_prompt_has_letters():
    p = format_mc_prompt(ITEM)
    assert "A) Berlin" in p and "B) Paris" in p


def test_run_scores_correct_solver():
    out = run_mc_benchmark([ITEM], lambda prompt: "The answer is B")
    assert out["accuracy"] == 1.0 and out["results"][0]["correct"] is True


def test_run_scores_wrong_solver():
    out = run_mc_benchmark([ITEM], lambda prompt: "A")
    assert out["accuracy"] == 0.0
