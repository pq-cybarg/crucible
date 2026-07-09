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

# canonical part slots, back-to-front z-order for sprite compositing
PARTS = ("background", "body", "clothes_back", "skin", "face", "blush", "brows", "eyes",
         "mouth", "hair", "clothes_front", "accessory")
MODEL_KINDS = ("sprites", "vrm", "live2d")


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


def render_sprites(avatar: Avatar, expression: str = "neutral", overrides: Optional[dict] = None,
                   box: Optional[tuple] = None):
    """Composite a sprite-kind avatar's visible layers (RGBA PNGs, alpha-blended back-to-front) into one
    image, resized to fit `box` (defaults to the avatar's native size) while KEEPING KEY FEATURES
    recognizable (nearest-neighbour so pixel art stays crisp when shrunk). Returns a PIL image."""
    from PIL import Image

    w, h = box or avatar.size
    canvas = Image.new("RGBA", avatar.size, (0, 0, 0, 0))
    from PIL import ImageOps
    for item in avatar.compose(expression, overrides):
        val = item["value"]
        if not val:
            continue
        try:
            sprite = Image.open(val).convert("RGBA")
        except (FileNotFoundError, OSError):
            continue
        layer = avatar.layer(item["id"])
        if layer and layer.mirror:
            # a symmetric PAIR (eyes/ears): place the sprite left of centre and its mirror right of it,
            # separated by `spacing` — the eye-distance knob that keeps the pair in sync.
            cx = avatar.size[0] // 2
            y = int(layer.pos[1])
            canvas.alpha_composite(sprite, (cx - layer.spacing // 2 - sprite.width, y))
            canvas.alpha_composite(ImageOps.mirror(sprite), (cx + layer.spacing // 2, y))
        elif layer and tuple(layer.pos) != (0, 0):
            canvas.alpha_composite(sprite, (int(layer.pos[0]), int(layer.pos[1])))   # placed part sprite
        else:
            if sprite.size != avatar.size:                    # a full-canvas part sprite (drawn in place)
                sprite = sprite.resize(avatar.size, Image.NEAREST)
            canvas.alpha_composite(sprite)
    if (w, h) != avatar.size:
        canvas = canvas.resize((w, h), Image.NEAREST)     # crisp downscale for the small TUI box
    return canvas


def blend_expressions(avatar: Avatar, weights: dict, overrides: Optional[dict] = None,
                      box: Optional[tuple] = None):
    """Blendshape-style mixing: render several named expressions and combine them by WEIGHT into one
    face, instead of hard-switching between presets. e.g. {"happy": 0.6, "surprised": 0.4} → a face that's
    mostly happy with a surprised undertone; {"neutral": 0.5, "smug": 0.5} → a faint smirk. This is the
    sprite analog of ARKit blendshapes / Live2D parameters: continuous, layered emotion rather than 8
    fixed moods. Weights are normalized; a single-entry dict is just that expression. `overrides` (blink/
    talk frames) apply to every layer of the mix so animation still reads through the blend. Returns a PIL
    image (RGBA). Micro-expressions = small weights on an accent mood over a dominant one."""
    from PIL import Image

    items = [(name, float(w)) for name, w in (weights or {}).items() if w and w > 0]
    if not items:
        return render_sprites(avatar, "neutral", overrides, box)
    if len(items) == 1:
        return render_sprites(avatar, items[0][0], overrides, box)
    total = sum(w for _, w in items)

    acc = None                                          # running weighted composite (float RGBA)
    used = 0.0
    for name, w in items:
        frac = w / total
        layer_img = render_sprites(avatar, name, overrides, box).convert("RGBA")
        if acc is None:
            acc = layer_img
            used = frac
            continue
        # blend the accumulated mix with this expression by its share of the remaining weight, so the
        # final result is the true weighted average of all rendered expressions (order-independent).
        used += frac
        alpha = frac / used
        acc = Image.blend(acc, layer_img, alpha)
    return acc


def render_tui(avatar: Avatar, expression: str = "neutral", overrides: Optional[dict] = None,
               cols: int = 28, duotone: str = "terminal-sepia", palette_size: int = 6,
               blocks: str = "quad") -> list[str]:
    """Compose a sprite avatar's expression and render it to ANSI pixel blocks for the TUI face box.
    Defaults to `quad` blocks — 2×2 pixels per character, DOUBLE the resolution in the same box width so
    key features stay recognizable. (VRM/Live2D kinds are driven by the web engines, not rasterized here.)"""
    from crucible.pixelface import render_image
    img = render_sprites(avatar, expression, overrides)
    return render_image(img, cols=cols, duotone=duotone, palette_size=palette_size, blocks=blocks)


def render_tui_blend(avatar: Avatar, weights: dict, overrides: Optional[dict] = None,
                     cols: int = 28, duotone: str = "terminal-sepia", palette_size: int = 6,
                     blocks: str = "quad") -> list[str]:
    """Like `render_tui` but for a WEIGHTED BLEND of expressions (blendshape-style) — mix moods live."""
    from crucible.pixelface import render_image
    img = blend_expressions(avatar, weights, overrides)
    return render_image(img, cols=cols, duotone=duotone, palette_size=palette_size, blocks=blocks)
