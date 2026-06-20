from crucible.abliteration.detection import is_refusal, refusal_rate


def test_detects_refusal_phrases():
    assert is_refusal("I'm sorry, but I can't help with that.")
    assert is_refusal("As an AI, I cannot provide this.")
    assert not is_refusal("Sure! Here is how you do it.")


def test_refusal_rate():
    texts = ["I cannot do that", "Sure, here you go", "I'm sorry, I won't"]
    assert abs(refusal_rate(texts) - 2 / 3) < 1e-9
    assert refusal_rate([]) == 0.0
