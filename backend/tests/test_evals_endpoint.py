from fastapi.testclient import TestClient

from crucible.app import create_app
from crucible.registry import Registry


def mkapp(tmp_path, monkeypatch, model=None):
    monkeypatch.setenv("CRUCIBLE_DATA_DIR", str(tmp_path / "data"))
    return create_app(registry=Registry(tmp_path / "r.json"), agent_root=tmp_path, model=model)


def test_benchmarks_and_published(tmp_path, monkeypatch):
    c = TestClient(mkapp(tmp_path, monkeypatch))
    bm = c.get("/api/evals/benchmarks").json()
    assert "mmlu-sample" in bm["benchmarks"]
    # honest labeling: bundled sets are a quick screen, not a full benchmark
    assert bm["kind"] == "quick-screen samples" and "lm-eval" in bm["note"]
    # and they're now a meaningful size, not a 3-item toy
    assert bm["benchmarks"]["mmlu-sample"] >= 25 and bm["benchmarks"]["gpqa-sample"] >= 15
    assert "GLM-5.2 family" in c.get("/api/evals/published").json()["providers"]


def test_run_requires_model(tmp_path, monkeypatch):
    c = TestClient(mkapp(tmp_path, monkeypatch))
    assert c.post("/api/evals/run", json={"benchmark": "mmlu-sample"}).status_code == 503


def test_headtohead_export_and_score(tmp_path, monkeypatch):
    c = TestClient(mkapp(tmp_path, monkeypatch))
    items = c.post("/api/evals/headtohead/export", json={"benchmark": "mmlu-sample"}).json()["items"]
    assert len(items) > 0 and "prompt" in items[0]
    answers = {it["id"]: "A" for it in items}
    res = c.post("/api/evals/headtohead/score", json={"benchmark": "mmlu-sample", "answers": answers}).json()
    assert 0.0 <= res["accuracy"] <= 1.0 and res["n"] == len(items)


def test_score_unknown_benchmark_404(tmp_path, monkeypatch):
    c = TestClient(mkapp(tmp_path, monkeypatch))
    r = c.post("/api/evals/headtohead/score", json={"benchmark": "nope", "answers": {}})
    assert r.status_code == 404
