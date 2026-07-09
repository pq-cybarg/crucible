from __future__ import annotations
# The RIG DRIVER bridge — turn Crucible's engine-agnostic face state (the continuous `expression` params
# + the gaze axis) into the ACTUAL parameter names each rig engine speaks, so the SAME reaction/expression
# stream that animates the TUI pixel face can drive a web VRM, a Live2D model, or an external VTube Studio
# rig. Three targets, all pure data (no sockets here — a bridge/frontend sends them):
#
#   • ARKit / VRM blendshapes — the 52-blendshape names Apple face-tracking + most VRM face rigs expose
#     (browInnerUp, eyeBlink*, jawOpen, mouthSmile*, eyeLook*…), each a weight in [0,1].
#   • Live2D Cubism standard parameters — ParamEyeLOpen, ParamMouthOpenY/Form, ParamBrow*, ParamEyeBallX/Y,
#     ParamAngleZ… in their native ranges.
#   • VTube Studio — an InjectParameterData payload of custom params (what the WS API consumes).
#
# `rig_frame()` ties it together: a blend of expressions (+ gaze/blink/talk) → params → all three targets
# in one call, so a driver just picks its engine's dict and pushes it.
from typing import Optional

from crucible.expression import PARAM_NAMES, blend_params


def _c(v: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return round(max(lo, min(hi, float(v))), 4)


def _gaze(gaze: Optional[tuple]) -> tuple:
    if not gaze:
        return (0.0, 0.0)
    return (max(-1.0, min(1.0, float(gaze[0]))), max(-1.0, min(1.0, float(gaze[1]))))


def to_arkit(params: dict, gaze: Optional[tuple] = None, blink: float = 0.0) -> dict:
    """Continuous params + gaze → ARKit/VRM blendshape weights in [0,1] (the names face-tracking rigs and
    most VRM avatars bind). Symmetric L/R shapes get the same weight; gaze uses the per-eye look shapes.
    `blink` (0..1) forces the lids shut (a blink frame) over whatever eye_open says. Convention: gaze +x =
    the viewer's right, +y = down."""
    p = {k: params.get(k, 0.0) for k in PARAM_NAMES}
    gx, gy = _gaze(gaze)
    lids = _c(max(blink, 1.0 - p["eye_open"]))              # 0 open … 1 shut
    up = _c(max(0.0, -gy)); down = _c(max(0.0, gy))
    out = {
        "browInnerUp": _c(max(0.0, p["brow"])),
        "browOuterUpLeft": _c(max(0.0, p["brow"])),
        "browOuterUpRight": _c(max(0.0, p["brow"])),
        "browDownLeft": _c(max(0.0, -p["brow"])),
        "browDownRight": _c(max(0.0, -p["brow"])),
        "eyeBlinkLeft": lids, "eyeBlinkRight": lids,
        "eyeWideLeft": _c(p["eye_wide"]), "eyeWideRight": _c(p["eye_wide"]),
        "jawOpen": _c(p["mouth_open"]),
        "mouthSmileLeft": _c(max(0.0, p["smile"])), "mouthSmileRight": _c(max(0.0, p["smile"])),
        "mouthFrownLeft": _c(max(0.0, -p["smile"])), "mouthFrownRight": _c(max(0.0, -p["smile"])),
        "cheekSquintLeft": _c(p["blush"]), "cheekSquintRight": _c(p["blush"]),
        # gaze: both eyes rotate together — right eye looks "in", left eye looks "out" for a rightward glance
        "eyeLookOutLeft": _c(max(0.0, gx)), "eyeLookInRight": _c(max(0.0, gx)),
        "eyeLookInLeft": _c(max(0.0, -gx)), "eyeLookOutRight": _c(max(0.0, -gx)),
        "eyeLookUpLeft": up, "eyeLookUpRight": up,
        "eyeLookDownLeft": down, "eyeLookDownRight": down,
    }
    return out


def to_live2d(params: dict, gaze: Optional[tuple] = None, blink: float = 0.0) -> dict:
    """Continuous params + gaze → Live2D Cubism standard parameters in their native ranges (eye/mouth open
    0..1, forms/brows/eyeball -1..1, head angle in degrees)."""
    p = {k: params.get(k, 0.0) for k in PARAM_NAMES}
    gx, gy = _gaze(gaze)
    eye = _c((1.0 - blink) * min(1.0, p["eye_open"] + 0.4 * p["eye_wide"]))   # wide → opened further
    return {
        "ParamEyeLOpen": eye, "ParamEyeROpen": eye,
        "ParamEyeBallX": round(gx, 4), "ParamEyeBallY": round(-gy, 4),        # Live2D EyeBallY: +up
        "ParamBrowLY": round(p["brow"], 4), "ParamBrowRY": round(p["brow"], 4),
        "ParamBrowLAngle": round(min(0.0, p["brow"]), 4), "ParamBrowRAngle": round(min(0.0, p["brow"]), 4),
        "ParamMouthOpenY": _c(p["mouth_open"]),
        "ParamMouthForm": round(max(-1.0, min(1.0, p["smile"])), 4),
        "ParamCheek": _c(p["blush"]),
        "ParamAngleZ": round(max(-1.0, min(1.0, p["head_tilt"])) * 30.0, 2),  # degrees, ±30
        "ParamAngleX": round(gx * 30.0, 2), "ParamAngleY": round(-gy * 30.0, 2),
    }


# our expression names → VRM 1.0 expression preset(s) they contribute to (with a weight factor)
_VRM_PRESET = {
    "happy": [("happy", 1.0)], "laughing": [("happy", 1.0)], "love": [("happy", 0.7), ("relaxed", 0.3)],
    "smug": [("happy", 0.4)], "sad": [("sad", 1.0)], "angry": [("angry", 1.0)], "tense": [("angry", 0.5)],
    "surprised": [("surprised", 1.0)], "scared": [("surprised", 0.7), ("sad", 0.3)],
    "curious": [("relaxed", 0.6)], "bored": [("relaxed", 0.5)], "neutral": [("neutral", 1.0)],
}


def to_vrm_expressions(weights: dict, gaze: Optional[tuple] = None, params: Optional[dict] = None,
                       blink: float = 0.0) -> dict:
    """Expression BLEND weights → VRM 1.0 expression preset weights (happy/angry/sad/relaxed/surprised),
    plus mouth viseme (aa), blink and look presets from the params/gaze. Weights are normalized to the
    total emotion weight so presets stay in [0,1]. This is the high-level VRM path (emotion presets)
    complementing the low-level ARKit blendshape path."""
    items = [(n, float(w)) for n, w in (weights or {}).items() if w and w > 0]
    total = sum(w for _, w in items) or 1.0
    presets: dict[str, float] = {}
    for name, w in items:
        for preset, factor in _VRM_PRESET.get(name, [("neutral", 1.0)]):
            presets[preset] = presets.get(preset, 0.0) + (w / total) * factor
    out = {k: _c(v) for k, v in presets.items()}
    p = params if params is not None else blend_params(weights)
    gx, gy = _gaze(gaze)
    out["aa"] = _c(p.get("mouth_open", 0.0))
    out["blink"] = _c(max(blink, 1.0 - p.get("eye_open", 1.0)))
    out["lookLeft"] = _c(max(0.0, -gx)); out["lookRight"] = _c(max(0.0, gx))
    out["lookUp"] = _c(max(0.0, -gy)); out["lookDown"] = _c(max(0.0, gy))
    return out


# custom VTube Studio parameter ids Crucible injects (register these once in VTS, then bind them in the rig)
VTS_PARAMS = ("CrucibleBrow", "CrucibleEyeOpen", "CrucibleEyeWide", "CrucibleSmile", "CrucibleMouthOpen",
              "CrucibleBlush", "CrucibleHeadTilt", "CrucibleEyeX", "CrucibleEyeY")


def to_vtube_studio(params: dict, gaze: Optional[tuple] = None, blink: float = 0.0,
                    face_found: bool = True, mode: str = "set") -> dict:
    """Continuous params + gaze → a VTube Studio `InjectParameterData` request payload (custom params).
    A bridge sends this over the VTS websocket API; here it's just the message data."""
    p = {k: params.get(k, 0.0) for k in PARAM_NAMES}
    gx, gy = _gaze(gaze)
    values = {
        "CrucibleBrow": round(max(-1.0, min(1.0, p["brow"])), 4),
        "CrucibleEyeOpen": _c((1.0 - blink) * p["eye_open"]),
        "CrucibleEyeWide": _c(p["eye_wide"]),
        "CrucibleSmile": round(max(-1.0, min(1.0, p["smile"])), 4),
        "CrucibleMouthOpen": _c(p["mouth_open"]),
        "CrucibleBlush": _c(p["blush"]),
        "CrucibleHeadTilt": round(max(-1.0, min(1.0, p["head_tilt"])), 4),
        "CrucibleEyeX": round(gx, 4), "CrucibleEyeY": round(gy, 4),
    }
    return {
        "apiName": "VTubeStudioPublicAPI", "apiVersion": "1.0", "messageType": "InjectParameterDataRequest",
        "data": {"faceFound": bool(face_found), "mode": mode,
                 "parameterValues": [{"id": k, "value": v} for k, v in values.items()]},
    }


def rig_frame(weights: dict, gaze: Optional[tuple] = None, extra: Optional[dict] = None,
              blink: float = 0.0) -> dict:
    """One call → a full driver frame for every target. Blend the expression `weights` into continuous
    params (with an optional `extra` param overlay: micro-expression/breath deltas), then map to ARKit/VRM
    blendshapes, Live2D params, VRM expression presets, and a VTube Studio payload. A rig driver just reads
    the field for its engine. `gaze` and `blink` (0..1) layer on top, independent of emotion."""
    params = blend_params(weights, extra=extra)
    return {
        "params": params, "gaze": _gaze(gaze), "blink": _c(blink),
        "arkit": to_arkit(params, gaze, blink),
        "live2d": to_live2d(params, gaze, blink),
        "vrm": to_vrm_expressions(weights, gaze, params, blink),
        "vtube_studio": to_vtube_studio(params, gaze, blink),
    }
