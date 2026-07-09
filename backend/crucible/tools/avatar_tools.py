from __future__ import annotations
# Part-by-part avatar DESIGN tools — the agent builds/modifies the companion rig one part at a time
# (the middle ground: not whole-character generation). It draws or generates an individual part sprite
# (transparent PNG per state), places + sizes it on the rig, tunes eye-distance / positioning / variants,
# and renders a preview it can `see_image` to check its work. All edits respect PROTECTED custom imports.
import json
import os

from crucible.tools.base import ToolResult


def _active():
    """The active companion avatar + its owned directory (in the data dir, shared with the TUI face)."""
    from crucible.avatar_gen import ensure_default_avatar
    from crucible.config import get_settings
    data = str(get_settings().data_dir)
    return ensure_default_avatar(data), os.path.join(data, "avatars", "active")


def _guard(layer) -> str | None:
    if layer is not None and layer.protected:
        return f"layer '{layer.id}' is a PROTECTED custom import — the design tools cannot edit it"
    return None


class AvatarInspect:
    name = "avatar_inspect"
    description = "Inspect the companion avatar rig: its parts (layers), each part's states, positioning " \
                  "(pos), eye-distance (spacing), mirror pairs, protected flag, and the expression map."
    parameters = {"type": "object", "properties": {}}

    def __init__(self, root=None):
        pass

    def run(self) -> ToolResult:
        a, _ = _active()
        layers = [{"id": l.id, "part": l.part, "protected": l.protected, "states": list(l.states),
                   "default_state": l.default_state, "pos": list(l.pos), "mirror": l.mirror,
                   "spacing": l.spacing} for l in a.layers]
        return ToolResult(ok=True, output=json.dumps(
            {"name": a.name, "kind": a.kind, "size": list(a.size), "layers": layers,
             "expressions": a.expressions}, indent=2))


class AvatarSetPart:
    name = "avatar_set_part"
    description = ("Add or replace a PART sprite for one state (transparent PNG), e.g. the 'eyes' part's "
                  "'open' state, or 'hair' 'base'. Draw/generate the sprite first (any tool) and pass its "
                  "path. Optional pos [x,y] to place a small sprite; mirror+spacing for symmetric pairs "
                  "(eyes). Creates the part layer if missing. Refuses protected parts.")
    parameters = {"type": "object", "properties": {
        "part": {"type": "string"}, "state": {"type": "string"}, "image": {"type": "string"},
        "pos": {"type": "array", "items": {"type": "number"}},
        "mirror": {"type": "boolean"}, "spacing": {"type": "number"}}, "required": ["part", "state", "image"]}

    def __init__(self, root=None):
        self.root = str(root) if root else "."

    def run(self, part="", state="", image="", pos=None, mirror=None, spacing=None) -> ToolResult:
        from crucible.avatar import Layer, PARTS
        from PIL import Image
        if part not in PARTS:
            return ToolResult(ok=False, output="", error=f"part must be one of: {', '.join(PARTS)}")
        a, adir = _active()
        layer = a.part_layer(part)
        g = _guard(layer)
        if g:
            return ToolResult(ok=False, output="", error=g)
        src = image if os.path.isabs(image) else os.path.join(self.root, image)
        if not os.path.exists(src):
            return ToolResult(ok=False, output="", error=f"image not found: {src}")
        dst = os.path.join(adir, f"{part}_{state}.png")
        try:
            Image.open(src).convert("RGBA").save(dst)      # validate + own a PNG copy (keeps transparency)
        except Exception as e:
            return ToolResult(ok=False, output="", error=f"not a valid image: {e}")
        if layer is None:
            layer = Layer(id=part, part=part, states={}, default_state=state)
            a.add_layer(layer)
        layer.states[state] = dst
        if not layer.default_state:
            layer.default_state = state
        if pos is not None:
            layer.pos = (int(pos[0]), int(pos[1]))
        if mirror is not None:
            layer.mirror = bool(mirror)
        if spacing is not None:
            layer.spacing = int(spacing)
        a.save(os.path.join(adir, "avatar.json"))
        return ToolResult(ok=True, output=f"set {part}.{state} ← {os.path.basename(dst)}"
                          + (f", pos={layer.pos}" if pos is not None else "")
                          + (f", mirror spacing={layer.spacing}" if mirror else ""))


class AvatarTune:
    name = "avatar_tune"
    description = ("Tune a part without new art: eye-distance (spacing), positioning (pos [x,y]), mirror " \
                  "on/off, or pick a variant as the default state (hairstyle/nose swap). Refuses protected.")
    parameters = {"type": "object", "properties": {
        "part": {"type": "string"}, "spacing": {"type": "number"},
        "pos": {"type": "array", "items": {"type": "number"}},
        "mirror": {"type": "boolean"}, "default_state": {"type": "string"}}, "required": ["part"]}

    def __init__(self, root=None):
        pass

    def run(self, part="", spacing=None, pos=None, mirror=None, default_state=None) -> ToolResult:
        a, adir = _active()
        layer = a.part_layer(part)
        if layer is None:
            return ToolResult(ok=False, output="", error=f"no '{part}' part")
        g = _guard(layer)
        if g:
            return ToolResult(ok=False, output="", error=g)
        if spacing is not None:
            layer.spacing = int(spacing)
        if pos is not None:
            layer.pos = (int(pos[0]), int(pos[1]))
        if mirror is not None:
            layer.mirror = bool(mirror)
        if default_state is not None:
            if default_state not in layer.states:
                return ToolResult(ok=False, output="", error=f"'{part}' has no state '{default_state}'")
            layer.default_state = default_state
        a.save(os.path.join(adir, "avatar.json"))
        return ToolResult(ok=True, output=f"tuned {part}: spacing={layer.spacing}, pos={layer.pos}, "
                          f"mirror={layer.mirror}, default={layer.default_state}")


class AvatarSetExpression:
    name = "avatar_set_expression"
    description = "Define an EXPRESSION as a part→state mapping, e.g. happy = {eyes: open, mouth: smile}. " \
                  "These are what the companion face switches between when reacting."
    parameters = {"type": "object", "properties": {
        "name": {"type": "string"}, "mapping": {"type": "object"}}, "required": ["name", "mapping"]}

    def __init__(self, root=None):
        pass

    def run(self, name="", mapping=None) -> ToolResult:
        if not name or not isinstance(mapping, dict):
            return ToolResult(ok=False, output="", error="need name + mapping {part: state}")
        a, adir = _active()
        a.set_expression(name, {str(k): str(v) for k, v in mapping.items()})
        a.save(os.path.join(adir, "avatar.json"))
        return ToolResult(ok=True, output=f"expression '{name}' = {a.expressions[name]}")


class AvatarGeneratePart:
    name = "avatar_generate_part"
    description = ("GENERATE a part sprite from a text prompt (local anime image model), auto-remove its "
                  "background for transparency, and place it on the rig as part.state. Optional negative "
                  "prompt, pos [x,y], mirror+spacing (eyes), size. This is how you draw cute-anime parts "
                  "one at a time. Refuses protected parts.")
    parameters = {"type": "object", "properties": {
        "part": {"type": "string"}, "state": {"type": "string"}, "prompt": {"type": "string"},
        "negative": {"type": "string"}, "size": {"type": "string"},
        "pos": {"type": "array", "items": {"type": "number"}},
        "mirror": {"type": "boolean"}, "spacing": {"type": "number"}},
        "required": ["part", "state", "prompt"]}

    def __init__(self, root=None):
        self.root = str(root) if root else "."

    def run(self, part="", state="", prompt="", negative="", size="384x384",
            pos=None, mirror=None, spacing=None) -> ToolResult:
        from crucible.avatar import PARTS
        if part not in PARTS:
            return ToolResult(ok=False, output="", error=f"part must be one of: {', '.join(PARTS)}")
        a, adir = _active()
        g = _guard(a.part_layer(part))
        if g:
            return ToolResult(ok=False, output="", error=g)
        try:
            from crucible import imagegen
        except Exception:
            return ToolResult(ok=False, output="", error="image generation not available (diffusers/torch)")
        if not imagegen.available():
            return ToolResult(ok=False, output="", error="no local image model available")
        try:
            w, h = (int(x) for x in size.lower().split("x"))
        except Exception:
            w, h = 384, 384
        neg = negative or "blurry, extra limbs, text, watermark, multiple, background clutter"
        try:
            from crucible.bgremove import remove_background
            img = imagegen.generate(prompt + ", single, centered, plain flat background",
                                    negative=neg, size=(w, h))
            img = remove_background(img)                  # knock out the flat bg → transparent part
        except Exception as e:
            return ToolResult(ok=False, output="", error=f"generation failed: {e}")
        finally:
            imagegen.unload()                            # free the model right after (memory safety)
        tmp = os.path.join(self.root, f"_gen_{part}_{state}.png")
        img.save(tmp)
        return AvatarSetPart(self.root).run(part=part, state=state, image=tmp,
                                            pos=pos, mirror=mirror, spacing=spacing)


class AvatarRender:
    name = "avatar_render"
    description = ("Render the companion avatar to a PNG file and return its path — then use see_image on "
                  "it to CHECK your work and iterate part by part. Pass a single `expression`, OR a `blend` "
                  "map of expression→weight for a BLENDSHAPE-STYLE mix (e.g. {\"happy\":0.6,\"surprised\":"
                  "0.4}) → layered emotion between presets, the same continuous mixing the live face uses.")
    parameters = {"type": "object", "properties": {
        "expression": {"type": "string"},
        "blend": {"type": "object", "description": "expression name → weight (mixed & normalized)"},
        "out": {"type": "string"}}, "required": []}

    def __init__(self, root=None):
        self.root = str(root) if root else "."

    def run(self, expression="neutral", blend=None, out="") -> ToolResult:
        from crucible.avatar import render_sprites, blend_expressions
        a, _ = _active()
        if isinstance(blend, dict) and blend:
            weights = {str(k): float(v) for k, v in blend.items()}
            img = blend_expressions(a, weights)
            label = "blend " + "+".join(f"{k}:{v:g}" for k, v in weights.items())
        else:
            img = render_sprites(a, expression)
            label = f"'{expression}'"
        path = out if out else os.path.join(self.root, "avatar_preview.png")
        if not os.path.isabs(path):
            path = os.path.join(self.root, path)
        img.convert("RGBA").save(path)
        return ToolResult(ok=True, output=f"rendered {label} → {path} ({a.size[0]}x{a.size[1]}). "
                          "Use see_image on it to review.")
