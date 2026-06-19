import pytest

from crucible.registry import Model, Registry


def make(id, base=None, kind="base", path=None):
    return Model(
        id=id, name=id, base_id=base, path=path or f"/m/{id}.gguf",
        quant="Q4_K_M", kind=kind, endpoint=None, created="2026-06-19", notes="",
    )


def test_register_and_get(tmp_path):
    reg = Registry(tmp_path / "registry.json")
    reg.register(make("glm32b"))
    assert reg.get("glm32b").name == "glm32b"
    assert [m.id for m in reg.list()] == ["glm32b"]


def test_duplicate_id_rejected(tmp_path):
    reg = Registry(tmp_path / "registry.json")
    reg.register(make("a"))
    with pytest.raises(ValueError):
        reg.register(make("a"))


def test_persistence_across_instances(tmp_path):
    p = tmp_path / "registry.json"
    Registry(p).register(make("a"))
    assert Registry(p).get("a").id == "a"


def test_lineage(tmp_path):
    reg = Registry(tmp_path / "registry.json")
    reg.register(make("base"))
    reg.register(make("abl", base="base", kind="abliterated"))
    reg.register(make("steer", base="abl", kind="steered"))
    assert [m.id for m in reg.lineage("steer")] == ["base", "abl", "steer"]


def test_original_path_immutable(tmp_path):
    reg = Registry(tmp_path / "registry.json")
    reg.register(make("base", path="/m/base.gguf"))
    with pytest.raises(ValueError):
        reg.register(make("abl", base="base", kind="abliterated", path="/m/base.gguf"))


def test_set_endpoint(tmp_path):
    reg = Registry(tmp_path / "registry.json")
    reg.register(make("a"))
    reg.set_endpoint("a", "http://127.0.0.1:8081")
    assert reg.get("a").endpoint == "http://127.0.0.1:8081"
