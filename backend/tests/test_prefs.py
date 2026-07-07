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


def test_resource_limits_persist_and_validate(tmp_path):
    from crucible.prefs import has_limits
    store = PreferencesStore(tmp_path / "preferences.json")
    assert store.get()["resource_limits"] == {"num_ctx": 0, "keep_alive": "", "max_output_tokens": 0, "num_gpu": -1}
    assert has_limits(store.get()["resource_limits"]) is False       # all defaults → OpenAI path
    out = store.save({"resource_limits": {"num_ctx": 4096, "keep_alive": "0",
                                          "max_output_tokens": -5, "num_gpu": "bad"}})["resource_limits"]
    assert out["num_ctx"] == 4096 and out["keep_alive"] == "0"
    assert out["max_output_tokens"] == 0                              # clamped up from -5
    assert out["num_gpu"] == -1                                       # invalid → default
    assert has_limits(out) is True                                   # a real limit is set → native path


def test_permission_defaults_persist_and_validate(tmp_path):
    store = PreferencesStore(tmp_path / "preferences.json")
    assert store.get()["permissions"] == {"default": "ask", "modes": {}, "path_rules": []}
    out = store.save({"permissions": {
        "default": "allow",
        "modes": {"bash": "deny", "read_file": "vibes"},          # invalid mode dropped
        "path_rules": [
            {"glob": "~/.ssh/**", "mode": "deny", "tools": ["read_file", ""]},
            {"glob": "", "mode": "deny"},                          # empty glob dropped
            {"glob": "/x", "mode": "telepathy"},                   # invalid mode dropped
        ],
    }})["permissions"]
    assert out["default"] == "allow"
    assert out["modes"] == {"bash": "deny"}                        # only the valid override kept
    assert out["path_rules"] == [{"glob": "~/.ssh/**", "mode": "deny", "tools": ["read_file"]}]
