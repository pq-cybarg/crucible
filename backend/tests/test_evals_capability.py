from fastapi.testclient import TestClient

from crucible.app import create_app
from crucible.evals.capability import primary_metric
from crucible.registry import Registry


def test_primary_metric_prefers_exact_match():
    rows = [{"metric": "acc", "value": 0.5}, {"metric": "exact_match", "value": 0.3}]
    assert primary_metric(rows)["metric"] == "exact_match"
    assert primary_metric([]) is None


def test_capability_unknown_ids_404(tmp_path, monkeypatch):
    monkeypatch.setenv("CRUCIBLE_DATA_DIR", str(tmp_path / "data"))
    c = TestClient(create_app(registry=Registry(tmp_path / "r.json")))
    r = c.post("/api/abliteration/capability", json={"base_id": "a", "variant_id": "b"})
    assert r.status_code == 404
