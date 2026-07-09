"""Rig driver bridge — continuous face params/gaze → ARKit-VRM blendshapes, Live2D params, VTube Studio."""
from crucible.expression import EXPRESSIONS, blend_params
from crucible.rigmap import (VTS_PARAMS, rig_frame, to_arkit, to_live2d, to_vrm_expressions,
                             to_vtube_studio)


def _p(name):
    return EXPRESSIONS[name].params()


def test_arkit_maps_smile_frown_and_brow_signs():
    happy = to_arkit(_p("happy"))
    assert happy["mouthSmileLeft"] > 0 and happy["mouthFrownLeft"] == 0     # smiling, not frowning
    angry = to_arkit(_p("angry"))
    assert angry["browDownLeft"] > 0 and angry["browInnerUp"] == 0          # furrowed brow (brow<0)
    surprised = to_arkit(_p("surprised"))
    assert surprised["browInnerUp"] > 0 and surprised["eyeWideLeft"] > 0 and surprised["jawOpen"] > 0
    # every weight is a normalized blendshape in [0,1]
    assert all(0.0 <= v <= 1.0 for v in happy.values())


def test_arkit_blink_forces_lids_shut():
    p = _p("surprised")                                    # eye_open = 1.0
    assert to_arkit(p)["eyeBlinkLeft"] == 0.0
    assert to_arkit(p, blink=1.0)["eyeBlinkLeft"] == 1.0   # a blink frame overrides eye_open
    assert to_arkit(p, blink=1.0)["eyeBlinkRight"] == 1.0


def test_arkit_gaze_uses_per_eye_look_shapes_and_is_symmetric():
    right = to_arkit(_p("neutral"), gaze=(1.0, 0.0))
    assert right["eyeLookOutLeft"] == 1.0 and right["eyeLookInRight"] == 1.0
    assert right["eyeLookInLeft"] == 0.0 and right["eyeLookOutRight"] == 0.0
    left = to_arkit(_p("neutral"), gaze=(-1.0, 0.0))
    assert left["eyeLookInLeft"] == 1.0 and left["eyeLookOutRight"] == 1.0
    down = to_arkit(_p("neutral"), gaze=(0.0, 1.0))
    assert down["eyeLookDownLeft"] == 1.0 and down["eyeLookUpLeft"] == 0.0


def test_live2d_native_ranges():
    l = to_live2d(_p("happy"), gaze=(0.5, -0.5))
    assert 0.0 <= l["ParamEyeLOpen"] <= 1.0
    assert -1.0 <= l["ParamMouthForm"] <= 1.0 and l["ParamMouthForm"] > 0      # smiling form
    assert l["ParamEyeBallX"] == 0.5 and l["ParamEyeBallY"] == 0.5             # EyeBallY: +up = -gy
    # head tilt maps to degrees within ±30
    tilted = to_live2d({**_p("neutral"), "head_tilt": 1.0})
    assert tilted["ParamAngleZ"] == 30.0
    assert abs(to_live2d({**_p("neutral"), "head_tilt": -1.0})["ParamAngleZ"]) == 30.0


def test_live2d_blink_closes_eyes():
    assert to_live2d(_p("neutral"), blink=1.0)["ParamEyeLOpen"] == 0.0


def test_vrm_expression_presets_from_blend_weights():
    v = to_vrm_expressions({"happy": 1.0})
    assert v["happy"] == 1.0 and v.get("angry", 0.0) == 0.0
    mix = to_vrm_expressions({"happy": 0.5, "angry": 0.5})
    assert 0.0 < mix["happy"] <= 1.0 and 0.0 < mix["angry"] <= 1.0             # both presets present
    assert all(0.0 <= w <= 1.0 for w in mix.values())                         # normalized, in range
    # gaze + mouth ride along as look/viseme presets
    look = to_vrm_expressions({"neutral": 1.0}, gaze=(1.0, 0.0))
    assert look["lookRight"] == 1.0 and look["lookLeft"] == 0.0


def test_vtube_studio_payload_shape():
    msg = to_vtube_studio(_p("happy"), gaze=(0.3, -0.2))
    assert msg["messageType"] == "InjectParameterDataRequest"
    ids = {pv["id"] for pv in msg["data"]["parameterValues"]}
    assert ids == set(VTS_PARAMS)                                             # exactly our custom params
    smile = next(pv for pv in msg["data"]["parameterValues"] if pv["id"] == "CrucibleSmile")
    assert smile["value"] > 0
    eyex = next(pv for pv in msg["data"]["parameterValues"] if pv["id"] == "CrucibleEyeX")
    assert eyex["value"] == 0.3


def test_rig_frame_bundles_all_targets_and_layers_extra():
    f = rig_frame({"happy": 0.6, "surprised": 0.4}, gaze=(0.5, 0.0),
                  extra={"blush": 0.5}, blink=0.0)
    assert set(f) == {"params", "gaze", "blink", "arkit", "live2d", "vrm", "vtube_studio"}
    # params are the weighted blend + the extra overlay (blush lifted)
    base = blend_params({"happy": 0.6, "surprised": 0.4})
    assert f["params"]["blush"] > base["blush"]
    # the blend is a real mix (surprised contributes an open jaw)
    assert f["arkit"]["jawOpen"] > 0 and f["live2d"]["ParamEyeBallX"] == 0.5
    assert f["vtube_studio"]["data"]["parameterValues"]                       # payload populated
