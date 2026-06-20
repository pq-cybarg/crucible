from crucible.guardrails.base import GuardrailConfig
from crucible.guardrails.store import GuardrailStore


def test_load_defaults_when_missing(tmp_path):
    assert GuardrailStore(tmp_path / "g.json").load().preset_id == "balanced"


def test_save_then_load(tmp_path):
    store = GuardrailStore(tmp_path / "g.json")
    store.save(GuardrailConfig(preset_id="strict", constitution_enabled=True))
    loaded = store.load()
    assert loaded.preset_id == "strict" and loaded.constitution_enabled is True
