"""Per-part lineage + per-part revert endpoints. The ledger's part-tagging/lineage math is covered
by test_abl_ledger; here we check the routes' shape and guards."""
from fastapi.testclient import TestClient
from crucible.app import create_app
from crucible.registry import Registry


class StubAdapter:
    num_layers = 2
    def set_matrix(self, name, W): pass


def mkapp(tmp_path, adapter=None):
    return TestClient(create_app(registry=Registry(tmp_path / "r.json"),
                                 agent_root=tmp_path, abliteration_adapter=adapter))


def test_lineage_empty_initially(tmp_path):
    r = mkapp(tmp_path).get("/api/inference/lineage").json()
    assert r["branch"] == "main" and r["parts"] == []


def test_revert_part_requires_adapter(tmp_path):
    assert mkapp(tmp_path).post("/api/inference/revert-part/language_model").status_code == 503


def test_revert_part_404_when_part_never_edited(tmp_path):
    c = mkapp(tmp_path, StubAdapter())
    r = c.post("/api/inference/revert-part/vision_encoder")
    assert r.status_code == 404 and "no edits" in r.json()["detail"]
