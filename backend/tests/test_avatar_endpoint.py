"""Avatar rig HTTP API — the engine-agnostic face state a web VRM/Live2D window or VTS bridge consumes."""
from fastapi.testclient import TestClient

from crucible.app import create_app
from crucible.registry import Registry


def _client(tmp_path, monkeypatch):
    monkeypatch.setenv("CRUCIBLE_DATA_DIR", str(tmp_path / "data"))    # isolate the active avatar
    return TestClient(create_app(Registry(tmp_path / "registry.json")))


def test_avatar_info_reports_the_rig(tmp_path, monkeypatch):
    c = _client(tmp_path, monkeypatch)
    info = c.get("/api/avatar").json()
    assert info["kind"] == "sprites" and info["name"]
    parts = {l["part"] for l in info["layers"]}
    assert {"eyes", "pupils", "mouth"} <= parts                        # part-based rig incl. movable pupils
    assert "neutral" in info["expressions"]


def test_rig_frame_maps_a_blend_to_every_engine(tmp_path, monkeypatch):
    c = _client(tmp_path, monkeypatch)
    r = c.post("/api/avatar/rig-frame", json={
        "weights": {"happy": 0.6, "surprised": 0.4}, "gaze": [0.5, 0.0], "extra": {"blush": 0.4}})
    assert r.status_code == 200
    f = r.json()
    assert set(f) >= {"params", "gaze", "arkit", "live2d", "vrm", "vtube_studio"}
    assert f["arkit"]["jawOpen"] > 0                                   # surprised opens the jaw
    assert f["live2d"]["ParamEyeBallX"] == 0.5                         # gaze rides along
    assert f["vtube_studio"]["messageType"] == "InjectParameterDataRequest"
    assert f["params"]["blush"] > 0                                    # the extra overlay applied


def test_rig_frame_defaults_to_neutral(tmp_path, monkeypatch):
    c = _client(tmp_path, monkeypatch)
    f = c.post("/api/avatar/rig-frame", json={}).json()
    assert f["params"]["smile"] == 0.0 and f["blink"] == 0.0


def test_reaction_frame_resolves_expression(tmp_path, monkeypatch):
    c = _client(tmp_path, monkeypatch)
    f = c.get("/api/avatar/reaction/funny").json()
    assert f["expression"] == "laughing" and f["arkit"]["mouthSmileLeft"] > 0
    # an unknown reaction word falls back to neutral, not an error
    assert c.get("/api/avatar/reaction/zzz").json()["expression"] == "neutral"
