"""Parametric expression blending: weighted params + nonlinear emotion interactions."""
from crucible.face_params import blend_params, EXPRESSION_PARAMS, DEFAULTS


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
