from __future__ import annotations
# PROCEDURAL avatar creation — draw a simple, modular pixel-art face as separate part sprites (each on
# its own transparent layer) and assemble them into an Avatar spec with expression states. This gives a
# real, shippable default avatar (NOT the reference samples) and demonstrates procedural character
# creation: params (colors) → an editable rig the agentic flow can then modify (swap hair, add accessory,
# add expression frames). Eyes are drawn large so key features stay recognizable when shrunk to the box.
import os
from dataclasses import dataclass

from crucible.avatar import Avatar, Layer

W, H = 48, 60


@dataclass
class Palette:
    skin: tuple = (236, 206, 180, 255)
    hair: tuple = (60, 46, 40, 255)
    eye: tuple = (70, 50, 120, 255)
    line: tuple = (40, 30, 28, 255)
    cloth: tuple = (44, 40, 52, 255)
    blush: tuple = (232, 150, 150, 180)


def _canvas():
    from PIL import Image
    return Image.new("RGBA", (W, H), (0, 0, 0, 0))


def _draw(fn) -> "object":
    from PIL import ImageDraw
    img = _canvas()
    fn(ImageDraw.Draw(img))
    return img


def _skin(p: Palette):
    def d(dr):
        dr.ellipse([10, 6, 38, 40], fill=p.skin, outline=p.line)          # head
        dr.rectangle([20, 38, 28, 44], fill=p.skin)                       # neck
        dr.polygon([(8, 60), (14, 46), (34, 46), (40, 60)], fill=p.cloth, outline=p.line)  # shoulders/cloth
    return _draw(d)


def _hair(p: Palette):
    def d(dr):
        dr.pieslice([8, 2, 40, 36], 180, 360, fill=p.hair)               # top hair
        dr.rectangle([8, 16, 13, 34], fill=p.hair)                       # side bangs
        dr.rectangle([35, 16, 40, 34], fill=p.hair)
    return _draw(d)


def _eyes(p: Palette, state: str):
    def d(dr):
        for cx in (18, 30):
            if state == "closed":
                dr.line([cx - 4, 24, cx + 4, 24], fill=p.line, width=1)
            else:
                pad = 5 if state == "wide" else 4
                dr.ellipse([cx - pad, 22 - (pad - 4), cx + pad, 26 + (pad - 4)], fill=(255, 255, 255, 255), outline=p.line)
                dr.ellipse([cx - 2, 22, cx + 2, 26], fill=p.eye)          # iris/pupil
    return _draw(d)


def _mouth(p: Palette, state: str):
    def d(dr):
        if state == "open":
            dr.ellipse([22, 32, 26, 37], fill=(120, 40, 40, 255), outline=p.line)
        elif state == "smile":
            dr.arc([20, 30, 28, 38], 20, 160, fill=p.line, width=2)
        elif state == "frown":
            dr.arc([20, 34, 28, 42], 200, 340, fill=p.line, width=2)
        else:  # closed
            dr.line([22, 34, 26, 34], fill=p.line, width=1)
    return _draw(d)


def _blush(p: Palette):
    def d(dr):
        dr.ellipse([13, 28, 18, 31], fill=p.blush)
        dr.ellipse([30, 28, 35, 31], fill=p.blush)
    return _draw(d)


# expression -> (eyes state, mouth state, blush?)
_EXPR = {
    "neutral":   ("open", "closed", False),
    "happy":     ("open", "smile", True),
    "laughing":  ("closed", "open", True),
    "sad":       ("open", "frown", False),
    "surprised": ("wide", "open", False),
    "scared":    ("wide", "open", False),
    "angry":     ("open", "frown", False),
    "love":      ("closed", "smile", True),
}


def generate_avatar(name: str, out_dir: str, palette: Palette | None = None) -> Avatar:
    """Draw the part sprites into out_dir and return an assembled, editable Avatar. Procedural creation:
    colors in → a modular rig out. The agentic flow can then modify non-protected parts."""
    p = palette or Palette()
    os.makedirs(out_dir, exist_ok=True)

    def save(img, fn) -> str:
        path = os.path.join(out_dir, fn)
        img.save(path)
        return path

    a = Avatar(name=name, kind="sprites", size=(W, H))
    a.add_layer(Layer(id="skin", part="skin", states={"base": save(_skin(p), "skin.png")}, default_state="base"))
    a.add_layer(Layer(id="blush", part="blush", states={"on": save(_blush(p), "blush.png")}, default_state=""))
    a.add_layer(Layer(id="eyes", part="eyes", default_state="open", states={
        s: save(_eyes(p, s), f"eyes_{s}.png") for s in ("open", "closed", "wide")}))
    a.add_layer(Layer(id="mouth", part="mouth", default_state="closed", states={
        s: save(_mouth(p, s), f"mouth_{s}.png") for s in ("closed", "smile", "open", "frown")}))
    a.add_layer(Layer(id="hair", part="hair", states={"base": save(_hair(p), "hair.png")}, default_state="base"))

    for expr, (eyes, mouth, blush) in _EXPR.items():
        mapping = {"eyes": eyes, "mouth": mouth}
        mapping["blush"] = "on" if blush else ""
        a.set_expression(expr, mapping)
    a.save(os.path.join(out_dir, "avatar.json"))
    return a


# The intended art style: cute anime girl — but FLAT and BOLD so it stays legible at the low resolution
# of the terminal face box (soft/detailed shading turns to mud when shrunk). Flat colours, clean thick
# lineart, big eyes, high contrast, minimal shading.
STYLE = ("cute anime girl, portrait, big expressive eyes, flat color, cel shading, bold clean thick "
         "lineart, simple, high contrast, minimal shading, plain background, upper body")
NEG = ("blurry, lowres, soft shading, gradient, noise, realistic, detailed background, bad anatomy, "
       "deformed, extra limbs, text, watermark, signature, multiple people")

# expression -> (prompt tweak, img2img strength). Same base + seed → the SAME character, just emoting.
# Richer than jumpscares: a range of moods incl. smug/teasing/wink. blink/talk are animation frames.
_COMPANION_EXPR = {
    "neutral":   ("neutral calm expression, eyes open, mouth closed", 0.0),
    "happy":     ("happy smiling, eyes open, gentle smile, blushing", 0.4),
    "laughing":  ("laughing, closed happy eyes, open mouth smile, blush", 0.5),
    "surprised": ("surprised, wide open eyes, open mouth, raised eyebrows", 0.5),
    "sad":       ("sad, teary downcast eyes, frown", 0.45),
    "angry":     ("angry, furrowed brows, pout, glaring", 0.45),
    "curious":   ("curious, head slightly tilted, looking, one eyebrow raised", 0.4),
    "love":      ("loving, half-closed eyes, soft smile, heavy blush, hearts", 0.45),
    "smug":      ("smug smirk, half-lidded confident eyes, one eyebrow raised", 0.45),
    "teasing":   ("teasing playful wink, tongue out, closed one eye, grin", 0.5),
    "shy":       ("shy embarrassed, looking away, heavy blush, small frown", 0.45),
    "blink":     ("eyes closed, relaxed, mouth closed", 0.32),   # for blinking
    "talk":      ("mouth open speaking, eyes open", 0.32),        # for talk animation
}


def build_anime_companion(name: str, out_dir: str, seed: int = 7, size: int = 384,
                          extra_style: str = "") -> Avatar:
    """Generate a REAL cute-anime companion in the intended style: a base face (txt2img) plus consistent
    expression variants (img2img from the base, same seed → same character). Assembled as a single 'face'
    part with those states + the expression map. Requires the local image model. Keeps the model loaded
    across the batch, then unloads (memory safety)."""
    import os

    from crucible import imagegen
    if not imagegen.available():
        raise RuntimeError("local image model not available (install the [avatar] extra)")
    os.makedirs(out_dir, exist_ok=True)
    style = STYLE + (", " + extra_style if extra_style else "")
    try:
        base = imagegen.generate(f"{style}, neutral calm expression", negative=NEG,
                                 size=(size, size), steps=26, seed=seed)
        a = Avatar(name=name, kind="sprites", size=base.size, meta={"generated": True, "style": style})
        states: dict[str, str] = {}
        for expr, (tweak, strength) in _COMPANION_EXPR.items():
            if strength <= 0.0:
                img = base
            else:
                img = imagegen.img2img(base, f"{style}, {tweak}", negative=NEG,
                                       strength=strength, steps=24, seed=seed)
            path = os.path.join(out_dir, f"face_{expr}.png")
            img.convert("RGBA").save(path)
            states[expr] = path
        a.add_layer(Layer(id="face", part="face", states=states, default_state="neutral"))
        # expressions + blink/talk map onto the single face part's states
        for expr in _COMPANION_EXPR:
            if expr not in ("blink", "talk"):
                a.set_expression(expr, {"face": expr})
        a.save(os.path.join(out_dir, "avatar.json"))
        return a
    finally:
        imagegen.unload()


def ensure_default_avatar(data_dir: str) -> Avatar:
    """Load the active avatar from <data_dir>/avatars/active, generating a default procedural one the
    first time so the TUI face box always has something to show."""
    active = os.path.join(data_dir, "avatars", "active")
    spec = os.path.join(active, "avatar.json")
    if os.path.exists(spec):
        try:
            return Avatar.load(spec)
        except (OSError, ValueError):
            pass
    return generate_avatar("kiri", active)


def import_portrait(image_path: str, name: str, out_dir: str, max_w: int = 128) -> Avatar:
    """Import a custom character image (e.g. a cute anime portrait) as a PROTECTED avatar. The image is
    COPIED into the avatar's own directory (so it's owned, and the original is never touched), then wrapped
    as a single protected 'base' layer — the agentic/procedural edit ops refuse to modify it. A one-image
    import has no part layers, so it renders the portrait but can't animate eyes/mouth until it's rigged
    (slice into parts or generate expression variants). Great for the cute-anime look in the box now."""
    from PIL import Image

    os.makedirs(out_dir, exist_ok=True)
    img = Image.open(image_path).convert("RGBA")
    if img.width > max_w:                                   # keep the sprite small for the TUI box
        img = img.resize((max_w, round(img.height * max_w / img.width)), Image.LANCZOS)
    dst = os.path.join(out_dir, "base.png")
    img.save(dst)
    a = Avatar(name=name, kind="sprites", size=img.size,
               meta={"imported_from": os.path.basename(image_path)})
    a.add_layer(Layer(id="base", part="face", protected=True, states={"base": dst}, default_state="base"))
    a.set_expression("neutral", {})
    a.save(os.path.join(out_dir, "avatar.json"))
    return a
