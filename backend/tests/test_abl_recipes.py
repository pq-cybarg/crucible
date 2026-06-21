import pytest
from fastapi.testclient import TestClient

from crucible.abliteration.recipes import Recipe, RecipeStore
from crucible.app import create_app
from crucible.registry import Registry


def test_recipe_store_roundtrip(tmp_path):
    store = RecipeStore(tmp_path / "rec.json")
    store.save(Recipe(name="my", base_id="m", layers=[12, 13], rank=1, coefficient=1.0))
    assert store.list()[0].layers == [12, 13]
    store.delete("my")
    assert store.list() == []
    with pytest.raises(KeyError):
        store.delete("nope")


def test_recipe_endpoints(tmp_path, monkeypatch):
    monkeypatch.setenv("CRUCIBLE_DATA_DIR", str(tmp_path / "data"))
    c = TestClient(create_app(registry=Registry(tmp_path / "r.json")))
    body = {"name": "r1", "base_id": "m", "layers": [12], "rank": 2, "coefficient": 1.5, "recipe_hash": "abc"}
    assert c.post("/api/abliteration/recipes", json=body).status_code == 201
    assert [r["name"] for r in c.get("/api/abliteration/recipes").json()] == ["r1"]
    assert c.delete("/api/abliteration/recipes/r1").status_code == 204
    assert c.delete("/api/abliteration/recipes/r1").status_code == 404


def test_manual_requires_adapter(tmp_path, monkeypatch):
    monkeypatch.setenv("CRUCIBLE_DATA_DIR", str(tmp_path / "data"))
    c = TestClient(create_app(registry=Registry(tmp_path / "r.json")))
    assert c.post("/api/abliteration/manual", json={"base_id": "x", "layers": [1]}).status_code == 503
