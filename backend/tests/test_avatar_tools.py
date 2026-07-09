"""Part-by-part avatar design tools — the agent builds/tunes the rig, protected imports stay immutable."""
import json
import os


def _png(path, size=(8, 8), color=(0, 0, 0, 255)):
    from PIL import Image
    Image.new("RGBA", size, color).save(path)


def _tools(tmp_path, monkeypatch):
    monkeypatch.setenv("CRUCIBLE_DATA_DIR", str(tmp_path / "data"))
    from crucible.tools.avatar_tools import (AvatarInspect, AvatarRender, AvatarSetExpression,
                                             AvatarSetPart, AvatarTune)
    root = tmp_path / "work"; root.mkdir()
    return {"inspect": AvatarInspect(root), "set": AvatarSetPart(root), "tune": AvatarTune(root),
            "expr": AvatarSetExpression(root), "render": AvatarRender(root)}, root


def test_design_loop_set_part_tune_expression_render(tmp_path, monkeypatch):
    t, root = _tools(tmp_path, monkeypatch)
    # inspect the auto-created default avatar
    info = json.loads(t["inspect"].run().output)
    assert info["name"] and any(l["part"] == "eyes" for l in info["layers"])

    # design a NEW eye sprite part-by-part: draw it, set it as a mirrored pair with a spacing
    _png(root / "eye.png", (6, 6))
    r = t["set"].run(part="eyes", state="open", image="eye.png", mirror=True, spacing=12)
    assert r.ok and "eyes.open" in r.output
    # the sprite was copied into the avatar's OWN dir (not referencing the work dir)
    info = json.loads(t["inspect"].run().output)
    eyes = next(l for l in info["layers"] if l["part"] == "eyes")
    assert eyes["mirror"] is True and eyes["spacing"] == 12 and "open" in eyes["states"]
    # the sprite was copied into the avatar's OWN dir (data_dir/avatars/active), not the work dir
    from crucible.config import get_settings
    owned = os.path.join(str(get_settings().data_dir), "avatars", "active", "eyes_open.png")
    assert os.path.exists(owned)

    # tune the eye distance (the sync knob) without new art
    assert t["tune"].run(part="eyes", spacing=20).ok
    assert next(l for l in json.loads(t["inspect"].run().output)["layers"] if l["part"] == "eyes")["spacing"] == 20

    # define an expression + render a preview PNG the agent could see_image
    _png(root / "m.png", (6, 4), (180, 40, 40, 255))
    t["set"].run(part="mouth", state="smile", image="m.png", pos=[21, 32])
    assert t["expr"].run(name="happy", mapping={"eyes": "open", "mouth": "smile"}).ok
    rr = t["render"].run(expression="happy", out="prev.png")
    assert rr.ok and os.path.exists(root / "prev.png")

    # render a BLENDSHAPE-STYLE mix (layered emotion between presets) the agent can preview
    t["expr"].run(name="neutral", mapping={"eyes": "open", "mouth": "smile"})
    rb = t["render"].run(blend={"happy": 0.6, "neutral": 0.4}, out="blend.png")
    assert rb.ok and "blend" in rb.output and os.path.exists(root / "blend.png")


def test_protected_import_rejects_design_edits(tmp_path, monkeypatch):
    t, root = _tools(tmp_path, monkeypatch)
    # import a protected custom face, then try to edit that part → refused
    from crucible.avatar_gen import import_portrait
    from crucible.config import get_settings
    _png(root / "custom.png", (40, 50), (200, 180, 160, 255))
    active = os.path.join(str(get_settings().data_dir), "avatars", "active")
    import_portrait(str(root / "custom.png"), "mine", active)   # replaces active with a protected import

    _png(root / "newface.png")
    r = t["set"].run(part="face", state="base", image="newface.png")
    assert r.ok is False and "PROTECTED" in r.error
    assert t["tune"].run(part="face", pos=[1, 1]).ok is False


def test_registered_in_default_registry(tmp_path):
    from crucible.tools import default_registry
    names = {tool.name for tool in default_registry(tmp_path).all()}
    assert {"avatar_inspect", "avatar_set_part", "avatar_tune", "avatar_set_expression",
            "avatar_render"} <= names
