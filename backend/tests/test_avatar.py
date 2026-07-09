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


def test_part_positioning_and_mirror_pair(tmp_path):
    from PIL import Image
    import numpy as np
    from crucible.avatar import render_sprites
    # a small single-eye sprite (a solid square), mirrored into a pair with a spacing gap
    eye = Image.new("RGBA", (6, 6), (0, 0, 0, 255))
    eye.save(tmp_path / "eye.png")
    a = Avatar(name="t", size=(48, 60))
    a.add_layer(Layer(id="eyes", part="eyes", mirror=True, spacing=10, pos=(0, 24),
                      states={"open": str(tmp_path / "eye.png")}, default_state="open"))
    a.set_expression("neutral", {"eyes": "open"})
    arr = np.asarray(render_sprites(a, "neutral").convert("RGBA"))
    opaque_x = np.where(arr[..., 3].max(axis=0) > 0)[0]        # columns that have any eye pixel
    cx = 24
    assert opaque_x.min() < cx and opaque_x.max() > cx         # eyes on BOTH sides of centre
    # tuning the spacing wider pushes the pair further apart (eye-distance knob)
    a.layer("eyes").spacing = 24
    arr2 = np.asarray(render_sprites(a, "neutral").convert("RGBA"))
    ox2 = np.where(arr2[..., 3].max(axis=0) > 0)[0]
    assert (cx - ox2.min()) > (cx - opaque_x.min())           # left eye moved further from centre


def test_full_chain_avatar_to_tui_pixels(tmp_path):
    from crucible.avatar import render_tui
    _sprite(tmp_path / "skin.png", (200, 180, 160, 255))
    _sprite(tmp_path / "eo.png", (20, 20, 20, 255))
    a = Avatar(name="t", size=(48, 60))
    a.add_layer(Layer(id="skin", part="skin", states={"base": str(tmp_path / "skin.png")}, default_state="base"))
    a.add_layer(Layer(id="eyes", part="eyes", states={"open": str(tmp_path / "eo.png")}, default_state="open"))
    a.set_expression("neutral", {"eyes": "open"})
    lines = render_tui(a, "neutral", cols=20)                 # modular avatar → composited → ANSI pixel box (quad)
    from crucible.pixelface import strip_ansi
    assert lines and all("\x1b[" in ln for ln in lines)       # ANSI color codes present
    assert any(g in "".join(lines) for g in "▀▘▝▖▗█▌▐▚▞")     # block glyphs (quad renderer)
    assert all(len(s) == 20 for s in strip_ansi(lines))       # fits the small box width


def _blend_avatar(tmp_path) -> Avatar:
    """A face part with three distinctly-colored expression states so a blend is measurably 'between'."""
    from PIL import Image
    Image.new("RGBA", (48, 60), (40, 40, 40, 255)).save(tmp_path / "neutral.png")   # dark
    Image.new("RGBA", (48, 60), (240, 240, 240, 255)).save(tmp_path / "happy.png")  # bright
    Image.new("RGBA", (48, 60), (240, 40, 40, 255)).save(tmp_path / "surprised.png")
    a = Avatar(name="b", size=(48, 60))
    a.add_layer(Layer(id="face", part="face", default_state="neutral", states={
        "neutral": str(tmp_path / "neutral.png"),
        "happy": str(tmp_path / "happy.png"),
        "surprised": str(tmp_path / "surprised.png")}))
    for e in ("neutral", "happy", "surprised"):
        a.set_expression(e, {"face": e})
    return a


def test_blend_is_between_the_two_pure_expressions(tmp_path):
    import numpy as np
    from crucible.avatar import blend_expressions, render_sprites
    a = _blend_avatar(tmp_path)
    dark = np.asarray(render_sprites(a, "neutral").convert("RGB")).astype(float)
    bright = np.asarray(render_sprites(a, "happy").convert("RGB")).astype(float)
    mix = np.asarray(blend_expressions(a, {"neutral": 0.5, "happy": 0.5}).convert("RGB")).astype(float)
    # a 50/50 blend sits strictly between the two pure faces (neither preset), ≈ their average
    assert dark.mean() < mix.mean() < bright.mean()
    assert abs(mix.mean() - (dark.mean() + bright.mean()) / 2) < 3.0


def test_blend_weight_shifts_toward_the_heavier_expression(tmp_path):
    import numpy as np
    from crucible.avatar import blend_expressions
    a = _blend_avatar(tmp_path)
    mostly_dark = np.asarray(blend_expressions(a, {"neutral": 0.8, "happy": 0.2}).convert("RGB")).mean()
    mostly_bright = np.asarray(blend_expressions(a, {"neutral": 0.2, "happy": 0.8}).convert("RGB")).mean()
    assert mostly_dark < mostly_bright                        # heavier weight dominates the mix


def test_blend_is_order_independent_and_normalized(tmp_path):
    import numpy as np
    from crucible.avatar import blend_expressions
    a = _blend_avatar(tmp_path)
    ab = np.asarray(blend_expressions(a, {"neutral": 1, "happy": 1, "surprised": 1}).convert("RGB")).astype(int)
    ba = np.asarray(blend_expressions(a, {"surprised": 1, "happy": 1, "neutral": 1}).convert("RGB")).astype(int)
    assert np.abs(ab - ba).max() <= 1                         # true average — order doesn't matter
    # un-normalized weights == normalized weights (only the RATIO matters)
    scaled = np.asarray(blend_expressions(a, {"neutral": 20, "happy": 20, "surprised": 20}).convert("RGB")).astype(int)
    assert np.abs(ab - scaled).max() <= 1


def test_blend_degenerate_cases(tmp_path):
    import numpy as np
    from crucible.avatar import blend_expressions, render_sprites
    a = _blend_avatar(tmp_path)
    # single-entry / empty / all-zero weights fall back sanely (no crash, no NaN)
    one = np.asarray(blend_expressions(a, {"happy": 1.0}).convert("RGB"))
    pure = np.asarray(render_sprites(a, "happy").convert("RGB"))
    assert np.array_equal(one, pure)                          # one expression == that expression
    empty = blend_expressions(a, {}).convert("RGB")           # empty → neutral fallback
    zeros = blend_expressions(a, {"happy": 0.0, "neutral": 0.0}).convert("RGB")
    assert empty.size == zeros.size == (48, 60)


def test_blend_overrides_apply_through_the_mix(tmp_path):
    import numpy as np
    from PIL import Image
    from crucible.avatar import blend_expressions
    # eyes as a separate part so a blink override is visible; face states carry the mood color
    Image.new("RGBA", (48, 60), (40, 40, 40, 255)).save(tmp_path / "neutral.png")
    Image.new("RGBA", (48, 60), (240, 240, 240, 255)).save(tmp_path / "happy.png")
    Image.new("RGBA", (12, 6), (0, 0, 0, 255)).save(tmp_path / "eopen.png")     # black eye bar
    Image.new("RGBA", (12, 6), (0, 0, 0, 0)).save(tmp_path / "eclosed.png")     # transparent (blink)
    a = Avatar(name="b", size=(48, 60))
    a.add_layer(Layer(id="face", part="face", default_state="neutral", pos=(0, 0), states={
        "neutral": str(tmp_path / "neutral.png"), "happy": str(tmp_path / "happy.png")}))
    a.add_layer(Layer(id="eyes", part="eyes", pos=(18, 20), default_state="open",
                      states={"open": str(tmp_path / "eopen.png"), "closed": str(tmp_path / "eclosed.png")}))
    for e in ("neutral", "happy"):
        a.set_expression(e, {"face": e, "eyes": "open"})
    with_eyes = np.asarray(blend_expressions(a, {"neutral": 0.5, "happy": 0.5}).convert("RGBA"))
    blinked = np.asarray(blend_expressions(a, {"neutral": 0.5, "happy": 0.5},
                                           overrides={"eyes": "closed"}).convert("RGBA"))
    # a blink override applies to every layer of the blend, so closing the eyes changes the composite
    assert not np.array_equal(with_eyes, blinked)


def _gaze_avatar(tmp_path):
    """A rig with a full-canvas eyes layer (red) and a separate pupils layer (blue dot) so we can watch
    the gaze axis move the PUPILS while the eyes stay put."""
    from PIL import Image
    skin = Image.new("RGBA", (48, 60), (200, 180, 160, 255))
    skin.save(tmp_path / "skin.png")
    eyes = Image.new("RGBA", (48, 60), (0, 0, 0, 0))
    for x in range(16, 20):                                    # a red eye-white marker (static)
        for y in range(22, 26):
            eyes.putpixel((x, y), (255, 0, 0, 255))
    eyes.save(tmp_path / "eyes.png")
    pupils = Image.new("RGBA", (48, 60), (0, 0, 0, 0))
    for x in range(22, 26):                                    # a blue pupil dot (moves with gaze)
        for y in range(22, 26):
            pupils.putpixel((x, y), (0, 0, 255, 255))
    pupils.save(tmp_path / "pupils.png")
    a = Avatar(name="g", size=(48, 60))
    a.add_layer(Layer(id="skin", part="skin", states={"base": str(tmp_path / "skin.png")}, default_state="base"))
    a.add_layer(Layer(id="eyes", part="eyes", states={"open": str(tmp_path / "eyes.png")}, default_state="open"))
    a.add_layer(Layer(id="pupils", part="pupils", states={"on": str(tmp_path / "pupils.png")}, default_state="on"))
    a.set_expression("neutral", {"eyes": "open", "pupils": "on"})
    return a


def _centroid_x(arr, chan):
    import numpy as np
    # x-centroid of pixels where the given color channel dominates (red=0, blue=2)
    other = [c for c in (0, 1, 2) if c != chan]
    mask = (arr[..., chan] > 150) & (arr[..., other[0]] < 120) & (arr[..., other[1]] < 120) & (arr[..., 3] > 0)
    xs = np.where(mask.any(axis=0))[0]
    return xs.mean() if len(xs) else None


def test_gaze_shifts_pupils_not_the_eye_whites(tmp_path):
    import numpy as np
    from crucible.avatar import render_sprites
    a = _gaze_avatar(tmp_path)
    center = np.asarray(render_sprites(a, "neutral").convert("RGBA"))
    right = np.asarray(render_sprites(a, "neutral", gaze=(1.0, 0.0)).convert("RGBA"))
    left = np.asarray(render_sprites(a, "neutral", gaze=(-1.0, 0.0)).convert("RGBA"))
    # the blue PUPIL moves with gaze…
    assert _centroid_x(right, 2) > _centroid_x(center, 2) > _centroid_x(left, 2)
    # …but the red eye-white marker does NOT (gaze only drives the pupils layer)
    assert abs(_centroid_x(right, 0) - _centroid_x(center, 0)) < 1e-6


def test_gaze_none_is_identity_and_mixes_with_expression(tmp_path):
    import numpy as np
    from crucible.avatar import render_sprites, blend_expressions
    a = _gaze_avatar(tmp_path)
    a.set_expression("happy", {"eyes": "open", "pupils": "on"})
    assert np.array_equal(np.asarray(render_sprites(a, "neutral")),
                          np.asarray(render_sprites(a, "neutral", gaze=(0.0, 0.0))))   # zero gaze = no-op
    # gaze layers on top of a blendshape mix (look-direction is independent of emotion)
    straight = np.asarray(blend_expressions(a, {"neutral": 0.5, "happy": 0.5}).convert("RGBA"))
    glancing = np.asarray(blend_expressions(a, {"neutral": 0.5, "happy": 0.5}, gaze=(1.0, 0.0)).convert("RGBA"))
    assert not np.array_equal(straight, glancing)


def test_generate_avatar_customization(tmp_path):
    import numpy as np
    from crucible.avatar_gen import generate_avatar, HAIRSTYLES, PALETTES
    from crucible.avatar import render_sprites
    # ART STYLE: different palettes → visibly different pixels
    sky = np.asarray(render_sprites(generate_avatar("a", str(tmp_path / "sky"), style="sky"), "neutral").convert("RGB"))
    rose = np.asarray(render_sprites(generate_avatar("a", str(tmp_path / "rose"), style="rose"), "neutral").convert("RGB"))
    assert not np.array_equal(sky, rose) and set(PALETTES) >= {"sky", "rose", "mint"}
    # HAIRSTYLE: kept as states, default follows the param — swappable
    a = generate_avatar("a", str(tmp_path / "hair"), hairstyle="long")
    hair = a.part_layer("hair")
    assert hair.default_state == "long" and set(hair.states) == set(HAIRSTYLES)
    # EYE DISTANCE: the eyes are a mirror pair; a bigger `spacing` pushes them apart
    close = generate_avatar("a", str(tmp_path / "close"), spacing=1)
    wide = generate_avatar("a", str(tmp_path / "wide"), spacing=16)
    assert close.part_layer("eyes").mirror and wide.part_layer("eyes").spacing == 16
    # different eye distance → the eyes land in different places → a different composite
    ci = np.asarray(render_sprites(close, "neutral").convert("RGBA"))
    wi = np.asarray(render_sprites(wide, "neutral").convert("RGBA"))
    assert not np.array_equal(ci, wi)
    # the iris (pupils) pair tracks the eye distance so it stays centred in the whites
    assert wide.part_layer("pupils").spacing > close.part_layer("pupils").spacing


def test_procedural_avatar_has_pupils_and_gaze_moves_them(tmp_path):
    import numpy as np
    from crucible.avatar_gen import generate_avatar
    from crucible.avatar import render_sprites
    a = generate_avatar("kiri", str(tmp_path / "av"))
    assert a.part_layer("pupils") is not None                 # eyes split into whites + movable pupils
    straight = np.asarray(render_sprites(a, "neutral").convert("RGBA"))
    glancing = np.asarray(render_sprites(a, "neutral", gaze=(1.0, 0.2)).convert("RGBA"))
    assert not np.array_equal(straight, glancing)             # the real default avatar can glance around
    # a closed-eye expression hides the pupils (they don't float over shut lids)
    assert a.expressions["laughing"]["pupils"] == "off"
