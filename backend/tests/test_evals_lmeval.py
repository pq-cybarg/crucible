from crucible.evals.lmeval import parse_lmeval_results
from crucible.evals.suite import CANONICAL_SUITE, SUITE_TASKS


def test_parse_flattens_metric_and_stderr():
    raw = {
        "gsm8k": {
            "alias": "gsm8k",
            "exact_match,flexible-extract": 0.25,
            "exact_match_stderr,flexible-extract": 0.099,
            "exact_match,strict-match": 0.20,
            "exact_match_stderr,strict-match": 0.091,
        },
        "arc_challenge": {"alias": "arc_challenge", "acc_norm,none": 0.42, "acc_norm_stderr,none": 0.03},
    }
    rows = parse_lmeval_results(raw)
    gsm = [r for r in rows if r["task"] == "gsm8k"]
    assert any(r["filter"] == "flexible-extract" and r["value"] == 0.25 and abs(r["stderr"] - 0.099) < 1e-9 for r in gsm)
    arc = next(r for r in rows if r["task"] == "arc_challenge")
    assert arc["metric"] == "acc_norm" and arc["value"] == 0.42


def test_parse_skips_alias_and_nonnumeric():
    rows = parse_lmeval_results({"t": {"alias": "t", "acc,none": 0.5, "notes": "x"}})
    assert len(rows) == 1 and rows[0]["metric"] == "acc"


def test_suite_is_nonempty_and_consistent():
    assert len(CANONICAL_SUITE) >= 8
    assert SUITE_TASKS == [s["task"] for s in CANONICAL_SUITE]
    assert "gsm8k" in SUITE_TASKS and "mmlu" in SUITE_TASKS


def test_parse_skips_sample_len():
    rows = parse_lmeval_results({"gsm8k": {"alias": "gsm8k", "sample_len": 30.0, "exact_match,strict-match": 0.16}})
    assert all(r["metric"] != "sample_len" for r in rows)
    assert any(r["metric"] == "exact_match" for r in rows)
