"""Organizational preferences: validated defaults for recall ordering + distance metric."""
from crucible.prefs import DEFAULTS, PreferencesStore


def test_defaults_when_absent(tmp_path):
    store = PreferencesStore(tmp_path / "preferences.json")
    assert store.get() == DEFAULTS


def test_save_and_reload_roundtrip(tmp_path):
    store = PreferencesStore(tmp_path / "preferences.json")
    store.save({"default_sort": "balanced", "processing_model": "tiny", "balanced_recency_weight": 0.3})
    reread = PreferencesStore(tmp_path / "preferences.json").get()
    assert reread["default_sort"] == "balanced"
    assert reread["processing_model"] == "tiny"
    assert reread["balanced_recency_weight"] == 0.3


def test_invalid_values_fall_back(tmp_path):
    store = PreferencesStore(tmp_path / "preferences.json")
    out = store.save({"default_sort": "telepathy", "default_metric": "vibes",
                      "balanced_recency_weight": 9.0})
    assert out["default_sort"] == DEFAULTS["default_sort"]      # unknown sort ignored
    assert out["default_metric"] == DEFAULTS["default_metric"]  # unknown metric ignored
    assert out["balanced_recency_weight"] == 1.0               # clamped into [0,1]


def test_partial_update_preserves_other_fields(tmp_path):
    store = PreferencesStore(tmp_path / "preferences.json")
    store.save({"processing_model": "tiny"})
    store.save({"default_sort": "priority"})
    out = store.get()
    assert out["processing_model"] == "tiny" and out["default_sort"] == "priority"
