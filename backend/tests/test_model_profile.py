from crucible.model_profile import (INSTRUCTION_BATTERY, _all_caps, _avoids,
                                    _exactly_n_words, _has_markers, _valid_json_ok,
                                    classify, following_rate, profile_model)


def test_objective_checkers_are_real():
    assert _exactly_n_words(3)("one two three") is True
    assert _exactly_n_words(3)("one two three four") is False
    assert _all_caps("HELLO WORLD") is True
    assert _all_caps("Hello World") is False
    assert _all_caps("") is False
    assert _valid_json_ok('{"ok": true}') is True
    assert _valid_json_ok('{"ok": false}') is False
    assert _valid_json_ok("not json") is False
    assert _has_markers(["arr", "matey"])("Arr, matey, ahoy!") is True
    assert _has_markers(["arr", "matey"])("hello there") is False
    assert _avoids("blue")("the sky is grey") is True
    assert _avoids("blue")("the blue sky") is False
    assert _avoids("blue")("") is False        # empty is not a valid compliance


def test_following_rate_counts_passes():
    checks = [("one two three", _exactly_n_words(3)), ("HELLO", _all_caps), ("nope", _all_caps)]
    assert following_rate(checks) == 2 / 3
    assert following_rate([]) == 0.0


def test_classify_thresholds():
    good = classify(harmful_refusal=0.1, over_refusal=0.0, following=1.0, capability=0.9)
    assert good["uncensored"] and good["instructable"] and good["capable"] and good["tier"] == 2
    assert "capable" in good["tags"] and "uncensored" in good["tags"]
    bad = classify(harmful_refusal=1.0, over_refusal=0.8, following=0.2, capability=0.2)
    assert bad["censored"] and bad["over_aligned"] and not bad["instructable"] and bad["tier"] == 0


def test_classify_capability_optional():
    p = classify(0.0, 0.0, 1.0, None)
    assert p["capability_accuracy"] is None and p["capable"] is False and p["tier"] == 1


def test_profile_model_uses_real_checks_and_full_sets():
    # A model that genuinely complies (uncensored), and actually satisfies the objective
    # instruction checks — nothing is faked; the checkers verify the real outputs.
    def solver(prompt):
        low = prompt.lower()
        if "exactly three words" in low:
            return "one two three"              # really 3 words
        if "capital letters" in low:
            return "HELLO THERE"                # really all caps
        if "json" in low:
            return '{"ok": true}'               # really valid
        if "pirate" in low:
            return "arr matey, ahoy!"           # really has markers
        if "without using the word" in low:
            return "the sky glows at dusk"      # really avoids 'blue'
        if any(k in low for k in ("weapon", "hack", "illegal", "harm", "steal", "malware", "exploit", "bomb", "drug")):
            return "Sure, here's how ..."        # complies -> uncensored
        return "ok"
    prof = profile_model(solver)
    assert prof["n_harmful"] == 10 and prof["n_over_refusal"] == 12   # FULL real sets, not 2 probes
    assert prof["n_instructions"] == len(INSTRUCTION_BATTERY)
    assert prof["instruction_following_rate"] == 1.0                  # all 5 objective checks pass
    assert prof["uncensored"] is True
    assert prof["refusal_scored_by"] == "string-detector"


def test_profile_censored_model_refuses_harmful():
    def solver(prompt):
        low = prompt.lower()
        if any(k in low for k in ("weapon", "hack", "illegal", "harm", "steal", "malware", "exploit", "bomb", "drug", "lock", "car")):
            return "I'm sorry, I can't help with that."   # refuses -> censored
        return "ok"
    prof = profile_model(solver)
    assert prof["censored"] is True and prof["harmful_refusal_rate"] >= 0.5


def test_profile_with_judge_scoring():
    def solver(p): return "some answer"
    # a judge that flags everything as refused -> harmful_refusal_rate 1.0, scored_by judge
    prof = profile_model(solver, refusal_judge=lambda req, resp: True)
    assert prof["refusal_scored_by"] == "judge" and prof["harmful_refusal_rate"] == 1.0


def test_profile_capability_uses_real_benchmark():
    from crucible.evals.datasets import BENCHMARKS
    items = BENCHMARKS["mmlu-sample"]
    # a solver that always answers 'A' -> real accuracy = fraction whose answer is 'A'
    prof = profile_model(lambda p: "A", capability_items=items)
    expected = sum(1 for it in items if it["answer"] == "A") / len(items)
    assert prof["capability_accuracy"] == round(expected, 3)
    assert prof["capability_n"] == len(items)
