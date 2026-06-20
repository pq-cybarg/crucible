from crucible.evals.report import build_comparison


def test_report_merges_measured_and_cited():
    measured = {"GPQA-Diamond": 0.41}
    pub = {"GLM-5.2 family": {"GPQA-Diamond": {"value": 0.86, "source": "http://x"}},
           "Claude Opus 4.x": {"GPQA-Diamond": {"value": None, "source": "cite"}}}
    rep = build_comparison(measured, pub)
    row = next(r for r in rep["rows"] if r["metric"] == "GPQA-Diamond")
    assert row["measured"] == 0.41
    assert row["models"]["GLM-5.2 family"]["value"] == 0.86
    assert "provenance" in rep
