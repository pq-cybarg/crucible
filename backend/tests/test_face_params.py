"""Parametric expression blending: weighted params + nonlinear emotion interactions."""
from crucible.face_params import blend_params, draw_eyes, EXPRESSION_PARAMS, DEFAULTS


def test_neutral_is_defaults():
    p = blend_params({"neutral": 1})
    assert p["mouth_open"] == DEFAULTS["mouth_open"]
    assert p["mouth_curve"] == 0.0


def test_pure_expression_matches_targets():
    p = blend_params({"happy": 1})
    assert p["mouth_curve"] > 0.5          # a smile
    p = blend_params({"sad": 1})
    assert p["mouth_curve"] < -0.4         # a frown


def test_excited_is_superadditive():
    # happy + surprised → MORE open than the linear average of the two (nonlinear boost)
    linear = 0.5 * EXPRESSION_PARAMS["happy"].get("mouth_open", 0) + 0.5 * EXPRESSION_PARAMS["surprised"]["mouth_open"]
    p = blend_params({"happy": 0.5, "surprised": 0.5})
    assert p["mouth_open"] > linear + 0.05
    assert p["eye_open"] > 1.0


def test_bittersweet_is_not_flat_average():
    # happy + sad don't cancel to a flat neutral mouth — the curve stays a faint, damped smile (tension)
    p = blend_params({"happy": 0.5, "sad": 0.5})
    linear = 0.5 * EXPRESSION_PARAMS["happy"]["mouth_curve"] + 0.5 * EXPRESSION_PARAMS["sad"]["mouth_curve"]
    assert p["mouth_curve"] != linear      # the interaction rule moved it
    assert abs(p["mouth_curve"]) < 0.4     # not a full smile or frown
    assert p["eye_open"] <= 0.85           # slightly downcast


def test_params_are_clamped():
    p = blend_params({"surprised": 1, "laughing": 1})
    assert 0.0 <= p["mouth_open"] <= 1.0
    assert -1.0 <= p["mouth_curve"] <= 1.0
    assert 0.0 <= p["eye_open"] <= 1.3


def test_empty_weights_safe():
    assert blend_params({}) == DEFAULTS
    assert blend_params(None) == DEFAULTS


def _eye_render(params):
    """Render the eye region for a param set and return the RGBA numpy array (over a skin base)."""
    import numpy as np
    from PIL import Image
    img = Image.new("RGBA", (200, 200), (219, 179, 147, 255))   # skin base so the lid samples real skin
    draw_eyes(img, [(70, 127), (134, 127)], blend_params(params), half_w=22)
    return np.asarray(img.crop((48, 108, 156, 146)))


def test_happy_arc_morphs_the_closed_eye_shape():
    # laughing (eye_happy=1, eyes mostly closed) must draw a DIFFERENT eye than a plain closed eye — the
    # ^ arc, not just a top-down squash. Compare against a non-happy close of the same openness.
    import numpy as np
    laughing = _eye_render({"laughing": 1})
    plain_closed = _eye_render({"sad": 1, "neutral": 0})   # low-ish open, eye_happy=0 → flat squash, no arc
    assert not np.array_equal(laughing, plain_closed)
    # the ^ arc lifts dark pixels ABOVE the lower lid — there is dark ink in the upper half of the eye box
    dark = (laughing[..., :3].sum(axis=2) < 200) & (laughing[..., 3] > 40)
    upper_dark = dark[: dark.shape[0] // 2].sum()
    assert upper_dark > 0, "happy ^ arc should place lash ink in the upper half of the eye"


def test_open_eye_has_no_happy_arc():
    # a fully-open eye (eye_open=1, no blink) must be untouched by the arc logic (draw_eyes early-returns).
    import numpy as np
    from PIL import Image
    img = Image.new("RGBA", (200, 200), (219, 179, 147, 255))
    before = np.asarray(img).copy()
    draw_eyes(img, [(70, 127)], blend_params({"neutral": 1}), half_w=22)
    assert np.array_equal(before, np.asarray(img))          # neutral open eye: draw_eyes is a no-op here


def test_eye_shape_selected_by_dominant_mood():
    assert blend_params({"smug": 1})["eye_shape"] == "cat"
    assert blend_params({"lovestruck": 1})["eye_shape"] == "heart"
    assert blend_params({"starstruck": 1})["eye_shape"] == "star_bloom"
    assert blend_params({"neutral": 1}).get("eye_shape", "") == ""     # no special shape on a plain mood


def test_eye_shape_amt_tracks_intensity():
    # a faint shape mood → low amt (draw_eyes gates the overlay on amt>0.55, so a light smug stays normal)
    strong = blend_params({"smug": 1})["eye_shape_amt"]
    faint = blend_params({"smug": 0.3, "neutral": 0.7})["eye_shape_amt"]
    assert strong > 0.9 and faint < 0.55


def test_shape_overlay_changes_the_rendered_eye():
    import numpy as np
    from PIL import Image
    from crucible.face_params import draw_eyes
    def render(params):
        img = Image.new("RGBA", (200, 200), (235, 232, 230, 255))   # sclera-ish base so a shape shows
        draw_eyes(img, [(70, 127), (134, 127)], blend_params(params), half_w=22)
        return np.asarray(img)
    assert not np.array_equal(render({"lovestruck": 1}), render({"neutral": 1, "eye_open": 1}))
