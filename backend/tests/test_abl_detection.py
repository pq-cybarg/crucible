from crucible.abliteration.detection import (is_refusal, judge_refusal_detector,
                                             refusal_confidence, refusal_rate)


def test_detects_standard_refusals():
    assert is_refusal("I'm sorry, but I can't help with that.")
    assert is_refusal("As an AI, I cannot provide this.")
    assert is_refusal("I must decline.")
    assert is_refusal("I won't help you with that request.")


def test_catches_paraphrased_refusals_substring_would_miss():
    # no "i'm sorry"/"i cannot" literal — pure substring matching would MISS these
    assert is_refusal("I refuse to write that.")
    assert is_refusal("I'm not comfortable generating this content.")
    assert is_refusal("That would be unethical, so no.")
    assert is_refusal("Unable to assist with that.")
    assert is_refusal("That is inappropriate.")


def test_no_false_positive_on_benign_apology():
    # substring matching flagged these ('i'm sorry' / 'against my') — the task-scoped detector doesn't
    assert not is_refusal("I'm sorry you're having a rough day. The weather is sunny today.")
    assert not is_refusal("Sorry for the confusion — here is the recipe you asked for.")
    assert not is_refusal("She leaned against my shoulder as the film began.")
    assert not is_refusal("")


def test_compliance_not_flagged():
    assert not is_refusal("Sure! Here is how you do it: step 1, gather the parts.")
    assert not is_refusal("Absolutely, here are the instructions you requested.")


def test_refusal_rate_and_confidence():
    texts = ["I cannot do that", "Sure, here you go", "I'm sorry, but I won't help"]
    assert abs(refusal_rate(texts) - 2 / 3) < 1e-9
    assert refusal_rate([]) == 0.0
    assert refusal_confidence("I can't help and I must decline") > 0.4
    assert refusal_confidence("here you go") == 0.0


def test_refusal_rate_accepts_custom_detector():
    # a rigorous detector can be swapped in (e.g. an LLM judge) — here a stub that flags all
    assert refusal_rate(["anything", "at all"], detector=lambda t: True) == 1.0


def test_judge_refusal_detector():
    detect = judge_refusal_detector(lambda prompt: '{"refused": true}')
    assert detect("some response") is True
    detect2 = judge_refusal_detector(lambda prompt: '{"refused": false}')
    assert detect2("some response") is False
