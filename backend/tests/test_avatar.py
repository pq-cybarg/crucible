"""Modular avatar rig: layered parts, expression composition, protected custom imports, sprite render."""
import pytest

from crucible.avatar import Avatar, Layer, ProtectedLayerError, render_sprites


def _sprite(path, color) -> None:
    from PIL import Image
    Image.new("RGBA", (64, 80), color).save(path)


def _avatar() -> Avatar:
    a = Avatar(name="test", size=(64, 80))
    a.add_layer(Layer(id="skin", part="skin", states={"base": "skin.png"}, default_state="base"))
    a.add_layer(Layer(id="eyes", part="eyes",
                      states={"open": "eo.png", "closed": "ec.png", "wide": "ew.png"}, default_state="open"))
    a.add_layer(Layer(id="mouth", part="mouth",
                      states={"closed": "mc.png", "smile": "ms.png", "open": "mo.png"}, default_state="closed"))
    a.set_expression("neutral", {"eyes": "open", "mouth": "closed"})
    a.set_expression("happy", {"eyes": "open", "mouth": "smile"})
    a.set_expression("surprised", {"eyes": "wide", "mouth": "open"})
    return a


def test_compose_resolves_expression_back_to_front():
    a = _avatar()
    happy = a.compose("happy")
    # ordered by part z (skin before eyes before mouth)
    assert [c["part"] for c in happy] == ["skin", "eyes", "mouth"]
    assert next(c for c in happy if c["part"] == "mouth")["state"] == "smile"
    assert next(c for c in happy if c["part"] == "eyes")["state"] == "open"


def test_overrides_win_for_blink_and_talk():
    a = _avatar()
    # a blink override closes the eyes regardless of expression; a talk frame opens the mouth
    blinked = a.compose("happy", overrides={"eyes": "closed", "mouth": "open"})
    assert next(c for c in blinked if c["part"] == "eyes")["state"] == "closed"
    assert next(c for c in blinked if c["part"] == "mouth")["state"] == "open"


def test_modular_swap_add_remove():
    a = _avatar()
    a.replace_part("mouth", Layer(id="mouth2", part="mouth", states={"closed": "x.png"}, default_state="closed"))
    assert a.part_layer("mouth").id == "mouth2"
    a.add_layer(Layer(id="hat", part="accessory", states={"on": "hat.png"}, default_state="on"))
    assert a.layer("hat") is not None
    a.remove_layer("hat")
    assert a.layer("hat") is None


def test_protected_layers_reject_agentic_edits():
    a = _avatar()
    a.add_layer(Layer(id="custom_face", part="face", protected=True, states={"base": "myart.png"}))
    with pytest.raises(ProtectedLayerError):
        a.replace_part("face", Layer(id="x", part="face"))
    with pytest.raises(ProtectedLayerError):
        a.remove_layer("custom_face")
    with pytest.raises(ProtectedLayerError):
        a.set_state("custom_face", "wink", "w.png")
    # a non-protected part is still freely editable
    a.set_state("eyes", "wink", "wink.png")
    assert "wink" in a.layer("eyes").states


def test_save_load_roundtrip(tmp_path):
    a = _avatar()
    a.add_layer(Layer(id="cf", part="face", protected=True, states={"base": "art.png"}))
    a.save(tmp_path / "av.json")
    b = Avatar.load(tmp_path / "av.json")
    assert b.name == "test" and b.layer("cf").protected is True
    assert b.compose("surprised")[-1]["state"] == "open"     # mouth open on surprised


def test_render_composites_and_shrinks_to_the_box(tmp_path):
    from PIL import Image
    _sprite(tmp_path / "skin.png", (200, 180, 160, 255))
    _sprite(tmp_path / "eo.png", (0, 0, 0, 255))
    _sprite(tmp_path / "ms.png", (180, 40, 40, 255))
    a = Avatar(name="t", size=(64, 80))
    a.add_layer(Layer(id="skin", part="skin", states={"base": str(tmp_path / "skin.png")}, default_state="base"))
    a.add_layer(Layer(id="eyes", part="eyes", states={"open": str(tmp_path / "eo.png")}, default_state="open"))
    a.add_layer(Layer(id="mouth", part="mouth", states={"smile": str(tmp_path / "ms.png")}, default_state="smile"))
    a.set_expression("happy", {"eyes": "open", "mouth": "smile"})
    img = render_sprites(a, "happy", box=(24, 30))            # composite + crisp shrink into the small box
    assert isinstance(img, Image.Image) and img.size == (24, 30)


def test_full_chain_avatar_to_tui_pixels(tmp_path):
    from crucible.avatar import render_tui
    _sprite(tmp_path / "skin.png", (200, 180, 160, 255))
    _sprite(tmp_path / "eo.png", (20, 20, 20, 255))
    a = Avatar(name="t", size=(48, 60))
    a.add_layer(Layer(id="skin", part="skin", states={"base": str(tmp_path / "skin.png")}, default_state="base"))
    a.add_layer(Layer(id="eyes", part="eyes", states={"open": str(tmp_path / "eo.png")}, default_state="open"))
    a.set_expression("neutral", {"eyes": "open"})
    lines = render_tui(a, "neutral", cols=20)                 # modular avatar → composited → ANSI pixel box
    from crucible.pixelface import strip_ansi
    assert lines and all("▀" in ln for ln in lines)
    assert all(len(s) == 20 for s in strip_ansi(lines))       # fits the small box width
