from fastapi.testclient import TestClient
from crucible.app import create_app
from crucible.registry import Registry


def mkapp(tmp_path):
    return TestClient(create_app(registry=Registry(tmp_path / "r.json"), agent_root=tmp_path))


def test_train_rejects_empty_dataset(tmp_path):
    r = mkapp(tmp_path).post("/api/train/lora", json={"model_path": str(tmp_path), "dataset": []})
    assert r.status_code == 422


def test_train_unknown_base_404(tmp_path):
    r = mkapp(tmp_path).post("/api/train/lora",
                             json={"base_id": "ghost", "dataset": [{"prompt": "a", "response": "b"}]})
    assert r.status_code == 404


def test_train_missing_path_409(tmp_path):
    r = mkapp(tmp_path).post("/api/train/lora",
                             json={"model_path": "/no/such/dir", "dataset": [{"prompt": "a", "response": "b"}]})
    assert r.status_code == 409


def test_train_reports_missing_peft_503(tmp_path):
    # valid dataset + existing path -> reaches train_lora_torch -> peft not installed -> 503
    r = mkapp(tmp_path).post("/api/train/lora",
                             json={"model_path": str(tmp_path), "dataset": [{"prompt": "a", "response": "b"}]})
    assert r.status_code == 503
    assert "peft" in r.json()["detail"]
