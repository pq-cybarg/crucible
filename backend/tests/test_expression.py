"""Avatar expression model — reaction → face params → tiny TUI face. Real-time reaction display."""
from crucible.expression import EXPRESSIONS, PARAM_NAMES, expression_for, render_face


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
