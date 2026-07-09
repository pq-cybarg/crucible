from __future__ import annotations
# Avatar EXPRESSION model — the bridge between the AI's emotional state and a face you can see in
# REAL TIME (not the slow STT→LLM→TTS round-trip). Two layers, mirroring how VTuber rigs work:
#
#  1. PARAMETERS: a small, standard-ish set of continuous face params (brow / eye-open / smile / mouth-
#     open / …) in [-1,1] — the same idea as Live2D parameters or ARKit blendshapes. An expression is a
#     preset of these; downstream renderers (web Live2D/VRM, VTube Studio via InjectParameterData, or the
#     TUI face below) interpolate toward them. Procedural idle (blink, breath) layers on top.
#  2. RENDER: a tiny two-color, outline-only, low-res ASCII face for the TUI sidebar so you see live
#     reactions while you code — eyes blink, mouth talks, the expression shifts the moment a reaction fires.
#
# The reaction vocabulary (crucible.reactions) maps onto expressions, so the co-watch / chat reaction
# stream drives the face directly.
from dataclasses import dataclass

# Continuous face parameters (Live2D/ARKit-style), each in [-1, 1]. Renderers map these however they like.
PARAM_NAMES = ("brow", "eye_open", "eye_wide", "smile", "mouth_open", "blush", "head_tilt")


@dataclass(frozen=True)
class Expression:
    name: str
    brow: float = 0.0       # -1 furrowed/angry … +1 raised/surprised
    eye_open: float = 1.0   #  0 closed … 1 open
    eye_wide: float = 0.0   #  0 normal … 1 wide (shock/fear)
    smile: float = 0.0      # -1 frown … +1 big smile
    mouth_open: float = 0.0 #  0 closed … 1 open
    blush: float = 0.0
    head_tilt: float = 0.0

    def params(self) -> dict:
        return {k: getattr(self, k) for k in PARAM_NAMES}


# Named expression presets — a starting set (extensible).
EXPRESSIONS: dict[str, Expression] = {
    "neutral":   Expression("neutral"),
    "happy":     Expression("happy", brow=0.2, smile=0.9, blush=0.2, eye_open=0.8),
    "laughing":  Expression("laughing", brow=0.3, smile=1.0, eye_open=0.3, mouth_open=0.6, blush=0.3),
    "sad":       Expression("sad", brow=0.4, smile=-0.7, eye_open=0.7, head_tilt=0.3),
    "surprised": Expression("surprised", brow=1.0, eye_wide=0.9, mouth_open=0.7, eye_open=1.0),
    "scared":    Expression("scared", brow=0.6, eye_wide=1.0, mouth_open=0.5, smile=-0.4),
    "angry":     Expression("angry", brow=-1.0, smile=-0.5, eye_open=0.9),
    "tense":     Expression("tense", brow=-0.6, smile=-0.2, eye_open=0.8),
    "curious":   Expression("curious", brow=0.5, head_tilt=0.5, eye_open=0.9),
    "love":      Expression("love", smile=0.8, blush=0.9, eye_open=0.6, head_tilt=0.2),
    "bored":     Expression("bored", brow=-0.2, eye_open=0.5, smile=-0.2),
    "smug":      Expression("smug", brow=0.2, smile=0.5, eye_open=0.6, head_tilt=0.2),
}

# Reaction word (crucible.reactions) → expression preset. Unknown → neutral.
REACTION_TO_EXPRESSION: dict[str, str] = {
    "funny": "laughing", "cute": "happy", "wholesome": "happy", "beautiful": "love", "romantic": "love",
    "exciting": "surprised", "epic": "surprised", "surprising": "surprised", "shocking": "surprised",
    "scary": "scared", "jumpscare": "scared", "tense": "tense", "sad": "sad", "gross": "scared",
    "boring": "bored", "calm": "neutral", "confusing": "curious", "sus": "smug", "cringe": "bored",
    "action": "surprised", "awkward": "tense", "dialogue": "neutral", "wtf": "surprised",
}


def expression_for(reaction: str) -> Expression:
    return EXPRESSIONS.get(REACTION_TO_EXPRESSION.get(reaction.lower(), reaction.lower()), EXPRESSIONS["neutral"])


def blend_params(weights: dict, extra: dict | None = None) -> dict:
    """Continuous, parameter-level blend of expression presets — the analog of `avatar.blend_expressions`
    but at the PARAM layer (what a VRM/Live2D/VTube-Studio driver consumes). Mix e.g.
    {"happy": 0.6, "surprised": 0.4} → a single {brow, eye_open, smile, …} param dict that's the weighted
    average of those presets. Weights are normalized and order-independent. `extra` overlays param deltas
    (added then clamped) — this is where GAZE / micro-expression jitter / breath layer on top of emotion.
    Unknown expression names are ignored. An empty/zero mix → neutral."""
    items = [(EXPRESSIONS[n], float(w)) for n, w in (weights or {}).items()
             if w and w > 0 and n in EXPRESSIONS]
    if not items:
        params = dict(EXPRESSIONS["neutral"].params())
    else:
        total = sum(w for _, w in items)
        params = {k: sum(e.params()[k] * w for e, w in items) / total for k in PARAM_NAMES}
    if extra:
        for k, dv in extra.items():
            if k in params:
                params[k] = max(-1.0, min(1.0, params[k] + float(dv)))
    return params


# --- TUI face: two-color, outline-only, low-res ------------------------------------------------------
def _eyes(e: Expression, blink: bool) -> str:
    if blink or e.eye_open < 0.15:
        return "-   -"
    if e.eye_wide > 0.6:
        return "O   O"
    if e.smile > 0.7 and e.eye_open < 0.5:      # happy squint
        return "^   ^"
    if e.eye_open < 0.6:
        return "•   •"
    return "o   o"


def _brows(e: Expression) -> str:
    if e.brow <= -0.5:
        return "＼   ／"      # angry, drawn in
    if e.brow >= 0.6:
        return "‾   ‾"       # raised
    return "     "


def _mouth(e: Expression, talk_open: bool) -> str:
    if talk_open or e.mouth_open > 0.6:
        return " (O) "
    if e.mouth_open > 0.3:
        return " (o) "
    if e.smile >= 0.6:
        return " \\_/ "       # smile
    if e.smile <= -0.5:
        return " /‾\\ "       # frown
    return " --- "


def render_face(expression: Expression, blink: bool = False, talk_open: bool = False,
                blush: bool | None = None) -> list[str]:
    """A tiny outline face for the TUI. Returns text lines; a renderer colors the outline in one color and
    the eyes/mouth (the 'accent') in another for the two-color look. Low-res on purpose."""
    b = expression.blush > 0.4 if blush is None else blush
    cheek = "*" if b else " "
    return [
        "  .-----.  ",
        _brows(expression).center(11),
        f" | {_eyes(expression, blink)} | ",
        f" |{cheek}     {cheek}| ",
        f" | {_mouth(expression, talk_open)} | ",
        "  '-----'  ",
    ]
