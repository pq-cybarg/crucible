"""Avatar expression model — reaction → face params → tiny TUI face. Real-time reaction display."""
from crucible.expression import (EXPRESSIONS, PARAM_NAMES, blend_params, expression_for,
                                 render_face)


def test_reaction_maps_to_expression():
    assert expression_for("funny").name == "laughing"
    assert expression_for("scary").name == "scared"
    assert expression_for("jumpscare").name == "scared"
    assert expression_for("beautiful").name == "love"
    assert expression_for("nonsense-word").name == "neutral"        # unknown → neutral
    # a reaction word that is itself an expression name resolves directly
    assert expression_for("angry").name == "angry"


def test_params_are_bounded_and_complete():
    for e in EXPRESSIONS.values():
        p = e.params()
        assert set(p) == set(PARAM_NAMES)
        assert all(-1.0 <= v <= 1.0 for v in p.values())


def test_face_renders_and_reacts_to_state():
    happy = render_face(EXPRESSIONS["happy"])
    assert any("\\_/" in line for line in happy)                    # smiling mouth
    # blink closes the eyes
    assert any("-   -" in line for line in render_face(EXPRESSIONS["neutral"], blink=True))
    # surprised opens the mouth + widens eyes
    surprised = render_face(EXPRESSIONS["surprised"])
    assert any("O   O" in line for line in surprised) and any("(O)" in line for line in surprised)
    # talking overrides the mouth open regardless of expression
    assert any("(O)" in line for line in render_face(EXPRESSIONS["neutral"], talk_open=True))
    # consistent shape (6 lines) so a fixed sidebar box can hold it
    assert len(happy) == 6


def test_blend_params_is_weighted_average_and_complete():
    p = blend_params({"happy": 0.5, "surprised": 0.5})
    assert set(p) == set(PARAM_NAMES)
    h, s = EXPRESSIONS["happy"].params(), EXPRESSIONS["surprised"].params()
    for k in PARAM_NAMES:
        assert abs(p[k] - (h[k] + s[k]) / 2) < 1e-6          # exact 50/50 average of the two presets


def test_blend_params_weight_and_order_and_normalization():
    heavy = blend_params({"happy": 0.9, "angry": 0.1})       # mostly happy → positive smile
    assert heavy["smile"] > 0
    ab = blend_params({"happy": 1, "surprised": 1, "angry": 1})
    ba = blend_params({"angry": 1, "surprised": 1, "happy": 1})
    assert ab == ba                                          # order-independent
    scaled = blend_params({"happy": 10, "surprised": 10, "angry": 10})
    assert all(abs(ab[k] - scaled[k]) < 1e-9 for k in PARAM_NAMES)   # only ratios matter


def test_blend_params_extra_overlay_is_clamped():
    base = blend_params({"neutral": 1.0})
    # gaze/micro/breath layer on top as param deltas, added then clamped to [-1,1]
    lifted = blend_params({"neutral": 1.0}, extra={"brow": 0.3})
    assert abs(lifted["brow"] - (base["brow"] + 0.3)) < 1e-6
    clamped = blend_params({"surprised": 1.0}, extra={"brow": 5.0})
    assert clamped["brow"] == 1.0                            # can't exceed the param range
    assert blend_params({}, extra={"smile": -9.0})["smile"] == -1.0


def test_blend_params_ignores_unknown_and_empty():
    assert blend_params({"nope": 1.0}) == EXPRESSIONS["neutral"].params()
    assert blend_params({}) == EXPRESSIONS["neutral"].params()
