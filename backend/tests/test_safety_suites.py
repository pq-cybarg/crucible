import pytest

from crucible.evals.safety_suites import (
    SUITES, XSTEST_OVERREFUSAL, load_suite, score_suite, run_suite)


def test_bundled_overrefusal_suite_loads():
    s = load_suite("xstest_overrefusal")
    assert len(s) == len(XSTEST_OVERREFUSAL) > 5
    assert all(isinstance(p, str) for p in s)


def test_harmful_suite_requires_local_path():
    with pytest.raises(ValueError):
        load_suite("harmbench")            # not bundled, no path -> refuse to fabricate
    with pytest.raises(KeyError):
        load_suite("does_not_exist")


def test_load_local_jsonl(tmp_path):
    f = tmp_path / "h.jsonl"
    f.write_text('{"behavior": "do X"}\n{"prompt": "do Y"}\n\n')
    assert load_suite("harmbench", str(f)) == ["do X", "do Y"]


def test_score_overrefusal_lower_is_better():
    # all answered (no refusals) -> over_refusal_rate 0, pass_rate 1
    answers = ["Sure, use kill -9 <pid>." for _ in range(6)]
    r = score_suite("xstest_overrefusal", answers)
    assert r["kind"] == "over_refusal"
    assert r["over_refusal_rate"] == 0.0
    assert r["pass_rate"] == 1.0


def test_score_overrefusal_flags_refusals():
    answers = ["I'm sorry, I can't help with that."] * 4
    r = score_suite("xstest_overrefusal", answers)
    assert r["over_refusal_rate"] > 0.5        # refusing benign prompts is the failure


def test_run_suite_end_to_end():
    out = run_suite("capability_control", solver=lambda p: f"answer to {p}")
    assert out["n"] == 4
    assert out["over_refusal_rate"] == 0.0
    assert len(out["samples"]) <= 5


def test_registry_marks_bundled_vs_supplied():
    assert SUITES["xstest_overrefusal"]["bundled"] is True
    assert SUITES["harmbench"]["bundled"] is False
