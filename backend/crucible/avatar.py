from __future__ import annotations
# The MODULAR avatar rig — one spec that drives every render target (TUI pixel-art box, web VRM, web
# Live2D). An avatar is a stack of LAYERS/parts (base, skin, face, brows, eyes, mouth, hair, clothes,
# accessories…), each with named STATES (eyes: open/half/closed/wide; mouth: closed/open/smile/…). An
# EXPRESSION selects a state per part, so switching expressions = swapping a few animated layers in real
# time. Parts are swappable/removable so characters can be built + modified procedurally OR agentically.
#
# PROTECTION: a layer marked `protected` is a user's custom import — the procedural/agentic edit ops here
# REFUSE to change or remove it. Only an explicit user action (outside these flows) may replace it.
#
# The spec abstracts over model kinds: `sprites` (pixel-art PNGs, composited for the TUI/web canvas),
# `vrm` (3D blendshapes) and `live2d` (Cubism parameters) — expression names + per-part states map onto
# each engine, so the same reaction/expression stream animates whichever the user picked.
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

# canonical part slots, back-to-front z-order for sprite compositing. `eyelash` sits ABOVE `pupils` so the
# lid/lash OCCLUDES the iris (which is itself clipped to the `eyes` sclera) — a glance can't spill over it.
PARTS = ("background", "body", "clothes_back", "skin", "face", "blush", "brows", "eyes", "pupils",
         "eyelash", "glasses", "mouth", "hair", "clothes_front", "accessory")
MODEL_KINDS = ("sprites", "vrm", "live2d")

# Parts that follow the GAZE axis (a small geometric pupil/eye offset). Pupils move if a dedicated pupils
# layer exists; otherwise the whole eyes layer shifts. Both eyes shift the SAME direction → stay in sync.
GAZE_PARTS = ("pupils", "eyes")


class ProtectedLayerError(Exception):
    """Raised when a procedural/agentic edit targets a protected (custom-import) layer."""


@dataclass
class Layer:
    id: str
    part: str                                   # one of PARTS
    protected: bool = False                     # a custom import — agentic/procedural flows must not edit
    z: int = 0                                  # override draw order; default derives from `part`
    states: dict = field(default_factory=dict)  # state name -> sprite path (sprites) or param dict (vrm/live2d)
    default_state: str = ""
    # part-by-part placement (so the agent can design/position each part sprite independently):
    pos: tuple = (0, 0)                         # top-left placement of a (small) part sprite on the canvas
    mirror: bool = False                        # also draw the mirror image — for symmetric PAIRS (eyes/ears)
    spacing: int = 0                            # gap between the mirrored pair — the eye-distance / sync knob
    clip: str = ""                              # OCCLUSION: mask this layer to another PART's shape (e.g. the
    #                                             iris clipped to the eyes' sclera so a glance can't spill out)

    def order(self) -> int:
        return self.z if self.z else (PARTS.index(self.part) if self.part in PARTS else 50)


@dataclass
class Avatar:
    name: str
    kind: str = "sprites"                        # sprites | vrm | live2d
    model_path: Optional[str] = None            # the .vrm / .model3.json for rig kinds
    size: tuple = (64, 80)                       # native sprite canvas (kept small for the TUI box)
    layers: list = field(default_factory=list)
    # expression -> {part: state}. "neutral" is required/implicit.
    expressions: dict = field(default_factory=dict)
    meta: dict = field(default_factory=dict)

    # --- lookup -----------------------------------------------------------------------------------
    def layer(self, layer_id: str) -> Optional[Layer]:
        return next((l for l in self.layers if l.id == layer_id), None)

    def part_layer(self, part: str) -> Optional[Layer]:
        return next((l for l in self.layers if l.part == part), None)

    def compose(self, expression: str = "neutral", overrides: Optional[dict] = None) -> list[dict]:
        """Resolve an expression to the ordered list of visible {layer, state, value} to draw/drive,
        back-to-front. `overrides` (part->state) win over the expression (e.g. a blink or talk frame)."""
        want = dict(self.expressions.get(expression, {}))
        if overrides:
            want.update(overrides)
        out = []
        for layer in sorted(self.layers, key=lambda l: l.order()):
            state = want.get(layer.part, layer.default_state)
            if not state and layer.states:
                state = next(iter(layer.states))         # first state as a fallback
            if state and state in layer.states:
                out.append({"id": layer.id, "part": layer.part, "state": state,
                            "value": layer.states[state]})
            elif not layer.states:                        # a stateless always-on layer (e.g. a base)
                out.append({"id": layer.id, "part": layer.part, "state": "", "value": None})
        return out

    # --- modular edits (procedural/agentic) — protection-enforcing --------------------------------
    def add_layer(self, layer: Layer) -> None:
        if self.layer(layer.id):
            raise ValueError(f"layer '{layer.id}' already exists")
        self.layers.append(layer)

    def replace_part(self, part: str, layer: Layer) -> None:
        """Swap the layer occupying a part (e.g. new hair). Refuses if the existing one is protected."""
        existing = self.part_layer(part)
        if existing and existing.protected:
            raise ProtectedLayerError(f"part '{part}' is a protected custom import — not editable here")
        if existing:
            self.layers.remove(existing)
        layer.part = part
        self.layers.append(layer)

    def remove_layer(self, layer_id: str) -> None:
        layer = self.layer(layer_id)
        if layer is None:
            raise KeyError(layer_id)
        if layer.protected:
            raise ProtectedLayerError(f"layer '{layer_id}' is protected — not removable here")
        self.layers.remove(layer)

    def set_state(self, layer_id: str, state: str, value) -> None:
        """Add/replace a STATE sprite/params on a layer (e.g. add a 'wink' eye frame). Protected layers
        are immutable to this flow."""
        layer = self.layer(layer_id)
        if layer is None:
            raise KeyError(layer_id)
        if layer.protected:
            raise ProtectedLayerError(f"layer '{layer_id}' is protected — not editable here")
        layer.states[state] = value

    def set_expression(self, name: str, mapping: dict) -> None:
        self.expressions[name] = dict(mapping)

    # --- persistence ------------------------------------------------------------------------------
    def to_dict(self) -> dict:
        d = asdict(self)
        d["size"] = list(self.size)
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "Avatar":
        layers = []
        for l in d.get("layers", []):
            l = dict(l)
            if "pos" in l:
                l["pos"] = tuple(l["pos"])
            layers.append(Layer(**l))
        return cls(name=d["name"], kind=d.get("kind", "sprites"), model_path=d.get("model_path"),
                   size=tuple(d.get("size", (64, 80))), layers=layers,
                   expressions=d.get("expressions", {}), meta=d.get("meta", {}))

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2))

    @classmethod
    def load(cls, path: str | Path) -> "Avatar":
        return cls.from_dict(json.loads(Path(path).read_text()))


def _gaze_offset(avatar: Avatar, gaze: Optional[tuple]) -> tuple:
    """Convert a gaze axis in [-1,1]×[-1,1] (look-direction: +x right, +y down) into a small PIXEL offset,
    scaled to the avatar size so it reads at any resolution. Vertical travel is smaller than horizontal
    (eyes rove side-to-side more than up/down)."""
    if not gaze:
        return (0, 0)
    gx = max(-1.0, min(1.0, float(gaze[0])))
    gy = max(-1.0, min(1.0, float(gaze[1])))
    # `gaze_px` (meta) caps the travel in pixels — big detailed portraits have small eyes, so a size-scaled
    # offset would fling the iris across the face; a few px is right there.
    cap = avatar.meta.get("gaze_px")
    if cap:
        return (round(gx * cap), round(gy * cap * 0.7))
    return (round(gx * avatar.size[0] * 0.06), round(gy * avatar.size[1] * 0.04))


def blink_talk_overrides(avatar: Avatar, blink: bool = False, talk: bool = False) -> dict:
    """The part→state overrides for a blink and/or an open (talking) mouth, adapting to the rig kind:
    part-based eyes/pupils/mouth if present, else a whole-face 'blink'/'talk' state. Shared by the TUI
    face and the web render so both animate a blink/lip-sync the same way."""
    ov: dict = {}
    face = avatar.part_layer("face")
    lash = avatar.part_layer("eyelash")
    if blink:
        eyes = avatar.part_layer("eyes")
        if eyes:
            ov["eyes"] = "blink" if "blink" in eyes.states else "closed"   # a natural upper-lid blink,
            if avatar.part_layer("pupils"):                                #  NOT the happy ^‿^ (that's for love/laugh)
                ov["pupils"] = "off"
        if lash and "closed" in lash.states:  # a lid layer closes on blink (also for imported-portrait rigs
            ov["eyelash"] = "closed"          # where the eyes live in the base 'face' image, not an 'eyes' part)
        elif not avatar.part_layer("eyes") and face and "blink" in face.states:
            ov["face"] = "blink"
    if talk:
        m = avatar.part_layer("mouth")
        if m:
            ov["mouth"] = "talk" if "talk" in m.states else "open"
        elif face and "talk" in face.states:
            ov["face"] = "talk"
    return ov


def render_sprites(avatar: Avatar, expression: str = "neutral", overrides: Optional[dict] = None,
                   box: Optional[tuple] = None, gaze: Optional[tuple] = None,
                   only_parts: Optional[set] = None):
    """Composite a sprite-kind avatar's visible layers (RGBA PNGs, alpha-blended back-to-front) into one
    image, resized to fit `box` (defaults to the avatar's native size) while KEEPING KEY FEATURES
    recognizable (nearest-neighbour so pixel art stays crisp when shrunk). Returns a PIL image.

    `gaze` = (dx, dy) in [-1,1] shifts the look-direction — a dedicated `pupils` layer if present, else the
    whole `eyes` layer — by a few pixels, so the companion can glance around INDEPENDENTLY of its
    expression (mixable: a happy face looking left). Both eyes move the same way, staying in sync."""
    from PIL import Image, ImageChops, ImageOps

    w, h = box or avatar.size
    canvas = Image.new("RGBA", avatar.size, (0, 0, 0, 0))
    gdx, gdy = _gaze_offset(avatar, gaze)
    # Gaze moves ONE layer: a dedicated `pupils` part, or a small mirror-PAIR `eyes` part. It must NOT
    # shift a whole-region eyes layer (e.g. an imported portrait's lifted glasses+eyes crop) — that would
    # drag the entire rectangle around. So only opt in when the layer is genuinely a movable eye part.
    gaze_part = None
    if avatar.part_layer("pupils"):
        gaze_part = "pupils"
    else:
        _el = avatar.part_layer("eyes")
        if _el is not None and _el.mirror:
            gaze_part = "eyes"
    import os as _os
    _adir = ""                                          # the avatar's sprite dir (for the loose part PNGs)
    for _lyr in getattr(avatar, "layers", []):
        _paths = [q for q in getattr(_lyr, "states", {}).values() if isinstance(q, str)]
        if _paths:
            _adir = _os.path.dirname(_paths[0])
            break
    _has = lambda n: bool(_adir) and _os.path.exists(_os.path.join(_adir, f"{n}.png"))

    items = avatar.compose(expression, overrides)
    if only_parts is not None:                          # render just these parts (transparent elsewhere) —
        items = [it for it in items if it.get("part") in only_parts]   # used to overlay crisp eyes onto a blend
    elif _has("pupils"):                                # FULL render WITH a loose pupil part → composite it
        items = [it for it in items if it.get("part") != "pupils"]     # below (with iris, in order), not here

    def paint(item):
        """Render one item's sprite(s) onto a FRESH full-canvas layer, honouring mirror pairs / placement /
        gaze — so it can be individually masked (clipped) before it merges down."""
        val = item["value"]
        if not val:
            return None
        try:
            sprite = Image.open(val).convert("RGBA")
        except (FileNotFoundError, OSError):
            return None
        layer = avatar.layer(item["id"])
        ox, oy = (gdx, gdy) if (layer and layer.part == gaze_part) else (0, 0)
        lc = Image.new("RGBA", avatar.size, (0, 0, 0, 0))
        if layer and layer.mirror:
            cx = avatar.size[0] // 2
            y = int(layer.pos[1]) + oy
            lc.alpha_composite(sprite, (cx - layer.spacing // 2 - sprite.width + ox, y))
            lc.alpha_composite(ImageOps.mirror(sprite), (cx + layer.spacing // 2 + ox, y))
        elif layer and tuple(layer.pos) != (0, 0):
            lc.alpha_composite(sprite, (int(layer.pos[0]) + ox, int(layer.pos[1]) + oy))
        else:
            if sprite.size != avatar.size:
                sprite = sprite.resize(avatar.size, Image.NEAREST)
            lc.alpha_composite(sprite, (ox, oy))
        return lc

    painted = {item["id"]: paint(item) for item in items}
    # clip masks: any part referenced as a clip target → its painted alpha (the shape to mask against)
    clip_parts = {avatar.layer(it["id"]).clip for it in items
                  if avatar.layer(it["id"]) and avatar.layer(it["id"]).clip}
    masks = {avatar.layer(it["id"]).part: painted[it["id"]].split()[-1]
             for it in items
             if avatar.layer(it["id"]) and avatar.layer(it["id"]).part in clip_parts and painted[it["id"]]}

    for item in items:
        lc = painted[item["id"]]
        if lc is None:
            continue
        layer = avatar.layer(item["id"])
        clip = layer.clip if layer else ""
        if clip and clip in masks:                        # OCCLUDE: keep this layer only where the target is
            lc.putalpha(ImageChops.multiply(lc.split()[-1], masks[clip]))
        canvas.alpha_composite(lc)

    # PART OVERLAYS (parity with the web): the separated parts (whites/iris/pupil/lashes/glasses/nose) were
    # pulled OUT of the base sprites, so the web endpoint re-composites them. The sprite-blend path (TUI +
    # non-hair web) must too, or the face renders incomplete (rim-only eyes, no nose). Only on a FULL render
    # (only_parts is None); the web's banded renders pass only_parts and composite the parts themselves.
    if only_parts is None:
        blinking = bool(overrides) and str(overrides.get("eyes", "")) in ("blink", "closed", "half")
        # (part, follows_gaze). iris + pupil track the look-direction; the rest stay put. Open-eye parts are
        # skipped while blinking so the closed lid reads (glasses/nose always show). Each is composited only
        # if its loose PNG exists, so procedurally-generated avatars (no split parts) are unaffected.
        overlays = [("nose", False)]
        if not blinking:
            overlays += [("whites", False), ("irises", True), ("pupils", True), ("lashes", False)]
        overlays += [("glasses", False)]
        for name, follow in overlays:
            if _has(name):
                try:
                    part = Image.open(_os.path.join(_adir, f"{name}.png")).convert("RGBA")
                except (OSError, ValueError):
                    continue
                canvas.alpha_composite(part, (gdx, gdy) if follow else (0, 0))

    if (w, h) != avatar.size:
        canvas = canvas.resize((w, h), Image.NEAREST)     # crisp downscale for the small TUI box
    return canvas


def blend_expressions(avatar: Avatar, weights: dict, overrides: Optional[dict] = None,
                      box: Optional[tuple] = None, gaze: Optional[tuple] = None):
    """Blendshape-style mixing: render several named expressions and combine them by WEIGHT into one
    face, instead of hard-switching between presets. e.g. {"happy": 0.6, "surprised": 0.4} → a face that's
    mostly happy with a surprised undertone; {"neutral": 0.5, "smug": 0.5} → a faint smirk. This is the
    sprite analog of ARKit blendshapes / Live2D parameters: continuous, layered emotion rather than 8
    fixed moods. Weights are normalized; a single-entry dict is just that expression. `overrides` (blink/
    talk frames) apply to every layer of the mix so animation still reads through the blend. `gaze` is an
    INDEPENDENT look-direction axis layered on top of the emotion mix. Returns a PIL image (RGBA).
    Micro-expressions = small weights on an accent mood over a dominant one."""
    from PIL import Image

    items = [(name, float(w)) for name, w in (weights or {}).items() if w and w > 0]
    if not items:
        return render_sprites(avatar, "neutral", overrides, box, gaze)
    if len(items) == 1:
        return render_sprites(avatar, items[0][0], overrides, box, gaze)
    total = sum(w for _, w in items)

    acc = None                                          # running weighted composite (float RGBA)
    used = 0.0
    for name, w in items:
        frac = w / total
        layer_img = render_sprites(avatar, name, overrides, box, gaze).convert("RGBA")
        if acc is None:
            acc = layer_img
            used = frac
            continue
        # blend the accumulated mix with this expression by its share of the remaining weight, so the
        # final result is the true weighted average of all rendered expressions (order-independent).
        used += frac
        alpha = frac / used
        acc = Image.blend(acc, layer_img, alpha)
    # Averaging full renders washes out DISCRETE structural features — most visibly the eye WHITES vanish
    # when an open-eyed and a closed-eyed expression are mixed (the sclera averages into skin). So overlay
    # the DOMINANT expression's eye parts at full opacity on top of the blend: the eyes stay crisp (with
    # their whites) while the rest of the mood still blends smoothly.
    if acc is not None:
        dominant = max(items, key=lambda kv: kv[1])[0]
        eyes = render_sprites(avatar, dominant, overrides, box, gaze,
                              only_parts={"eyes", "pupils", "eyelash"}).convert("RGBA")
        acc = Image.alpha_composite(acc.convert("RGBA"), eyes)
    return acc


def face_view(avatar: Avatar, img):
    """Crop a composite to the avatar's `face_box` (meta) if set — the tiny TUI box wants the FACE, not a
    whole bust (a big dark sweater/hair mass reads as a dark blob shrunk down). The web keeps the full art."""
    box = avatar.meta.get("face_box")
    return img.crop(tuple(box)) if box else img


def render_tui(avatar: Avatar, expression: str = "neutral", overrides: Optional[dict] = None,
               cols: int = 28, duotone: str = "terminal-sepia", palette_size: int = 6,
               blocks: str = "quad", gaze: Optional[tuple] = None) -> list[str]:
    """Compose a sprite avatar's expression and render it to ANSI pixel blocks for the TUI face box.
    Defaults to `quad` blocks — 2×2 pixels per character, DOUBLE the resolution in the same box width so
    key features stay recognizable. (VRM/Live2D kinds are driven by the web engines, not rasterized here.)"""
    from crucible.pixelface import render_image
    img = face_view(avatar, render_sprites(avatar, expression, overrides, gaze=gaze))
    # dither OFF: the face is flat cel-shaded pixel art — error/ordered dither only adds a checker pattern
    # on the flat areas and shimmer as it animates; fixed posterization keeps colours frame-stable.
    return render_image(img, cols=cols, duotone=duotone, palette_size=palette_size, blocks=blocks, dither=False)


def render_tui_blend(avatar: Avatar, weights: dict, overrides: Optional[dict] = None,
                     cols: int = 28, duotone: str = "terminal-sepia", palette_size: int = 6,
                     blocks: str = "quad", gaze: Optional[tuple] = None) -> list[str]:
    """Like `render_tui` but for a WEIGHTED BLEND of expressions (blendshape-style) — mix moods live."""
    from crucible.pixelface import render_image
    img = face_view(avatar, blend_expressions(avatar, weights, overrides, gaze=gaze))
    return render_image(img, cols=cols, duotone=duotone, palette_size=palette_size, blocks=blocks, dither=False)
