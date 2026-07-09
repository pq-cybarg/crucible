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


def test_render_png_returns_the_actual_avatar_image(tmp_path, monkeypatch):
    c = _client(tmp_path, monkeypatch)
    r = c.get("/api/avatar/render.png", params={"expression": "happy", "scale": 200})
    assert r.status_code == 200 and r.headers["content-type"] == "image/png"
    from PIL import Image
    import io
    img = Image.open(io.BytesIO(r.content))
    # upscaled by an INTEGER factor (crisp uniform pixels, no distortion) → a multiple of the native width,
    # near the requested size
    info = c.get("/api/avatar").json()
    native_w = info["size"][0]
    assert img.width % native_w == 0 and 0 < abs(img.width - 200) < native_w and img.height > 0
    # gaze/blink/talk params render distinct frames (the pupils move, the lids shut)
    a = c.get("/api/avatar/render.png", params={"gx": 1.0}).content
    b = c.get("/api/avatar/render.png", params={"gx": -1.0}).content
    blinked = c.get("/api/avatar/render.png", params={"blink": 1.0}).content
    assert a != b and blinked != a


def test_render_png_accepts_a_blend(tmp_path, monkeypatch):
    c = _client(tmp_path, monkeypatch)
    r = c.get("/api/avatar/render.png", params={"blend": "happy:0.6,surprised:0.4"})
    assert r.status_code == 200 and r.content[:8] == b"\x89PNG\r\n\x1a\n"


def test_live_driver_mood_react_talk(tmp_path, monkeypatch):
    c = _client(tmp_path, monkeypatch)
    assert c.post("/api/avatar/mood", json={"weights": {"happy": 1.0}}).status_code == 200
    rj = c.post("/api/avatar/react", json={"reaction": "funny"}).json()
    assert rj["expression"] == "laughing"
    assert c.post("/api/avatar/talk", json={"talking": True}).json()["ok"] is True
    assert c.post("/api/avatar/talk", json={"level": 0.8}).status_code == 200
