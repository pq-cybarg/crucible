from crucible.config import get_settings


def test_settings_defaults(tmp_path, monkeypatch):
    monkeypatch.setenv("CRUCIBLE_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("CRUCIBLE_MODELS_DIR", str(tmp_path / "models"))
    s = get_settings()
    assert s.data_dir == tmp_path / "data"
    assert s.models_dir == tmp_path / "models"
    assert s.registry_path == tmp_path / "data" / "registry.json"
    assert s.host == "127.0.0.1"


def test_data_dir_created(tmp_path, monkeypatch):
    monkeypatch.setenv("CRUCIBLE_DATA_DIR", str(tmp_path / "d"))
    s = get_settings()
    assert s.data_dir.is_dir()
