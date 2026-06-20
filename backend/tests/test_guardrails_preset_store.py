import pytest

from crucible.guardrails.presets import SystemPromptPreset
from crucible.guardrails.store import PresetStore


def mk(id, name="x", intensity=10, prompt="p"):
    return SystemPromptPreset(id=id, name=name, intensity=intensity, system_prompt=prompt)


def test_seeds_defaults(tmp_path):
    store = PresetStore(tmp_path / "presets.json")
    assert [p.id for p in store.list()] == ["unrestricted", "balanced", "strict"]


def test_create_and_get(tmp_path):
    store = PresetStore(tmp_path / "p.json")
    store.create(mk("feral", prompt="anything goes"))
    assert store.get("feral").system_prompt == "anything goes"


def test_create_duplicate_raises(tmp_path):
    store = PresetStore(tmp_path / "p.json")
    with pytest.raises(ValueError):
        store.create(mk("strict"))


def test_update_existing_default(tmp_path):
    store = PresetStore(tmp_path / "p.json")
    store.update("strict", mk("strict", name="Strict+", intensity=90, prompt="be careful"))
    assert store.get("strict").name == "Strict+"
    assert store.system_prompt("strict") == "be careful"


def test_update_missing_raises(tmp_path):
    store = PresetStore(tmp_path / "p.json")
    with pytest.raises(KeyError):
        store.update("ghost", mk("ghost"))


def test_delete_default(tmp_path):
    store = PresetStore(tmp_path / "p.json")
    store.delete("balanced")
    assert "balanced" not in [p.id for p in store.list()]


def test_reset_restores_defaults(tmp_path):
    store = PresetStore(tmp_path / "p.json")
    store.delete("balanced")
    store.create(mk("feral"))
    store.reset()
    assert [p.id for p in store.list()] == ["unrestricted", "balanced", "strict"]


def test_system_prompt_missing_returns_empty(tmp_path):
    assert PresetStore(tmp_path / "p.json").system_prompt("ghost") == ""
