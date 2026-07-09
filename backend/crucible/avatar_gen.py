from __future__ import annotations
# PROCEDURAL avatar creation — draw a simple, modular pixel-art face as separate part sprites (each on
# its own transparent layer) and assemble them into an Avatar spec with expression states. This gives a
# real, shippable default avatar (NOT the reference samples) and demonstrates procedural character
# creation: params (colors) → an editable rig the agentic flow can then modify (swap hair, add accessory,
# add expression frames). Eyes are drawn large so key features stay recognizable when shrunk to the box.
import os
from dataclasses import dataclass

from crucible.avatar import Avatar, Layer

# PIXEL-ART authoring: the parts are drawn at NATIVE low resolution (nothing is shrunk, so nothing turns
# to mud) as a cute-anime face. The KEY feature — the eyes — is hand-authored as a pixel grid (crisp,
# intentional, sparkly) rather than crude vector circles; the rest are bold flat shapes at native res.
# Diffusion is NOT in this path: it only authors part sprites offline (once); realtime = compositing these.
W, H = 64, 78
EYE_Y = 31                          # top of the eye sprites on the canvas
EYE_W = 12                          # eye-white sprite width (for the mirror-pair spacing maths)
IRIS_W = 9


@dataclass
class Palette:
    skin: tuple = (253, 229, 214, 255)
    hair: tuple = (78, 52, 62, 255)
    hair_hi: tuple = (128, 94, 108, 255)
    iris: tuple = (74, 150, 214, 255)              # bright anime iris (mid)
    iris_hi: tuple = (164, 214, 247, 255)          # iris light rim
    line: tuple = (58, 40, 44, 255)                # stark dark outline (also the pupil core)
    cloth: tuple = (74, 100, 156, 255)
    blush: tuple = (247, 162, 162, 235)
    mouth: tuple = (182, 90, 94, 255)


# ART-STYLE presets (palette swaps) — pick the whole look; user/agent can tune further. Tunable proof.
PALETTES: dict[str, Palette] = {
    "sky":  Palette(),
    "rose": Palette(hair=(150, 70, 96, 255), hair_hi=(196, 118, 142, 255), iris=(206, 92, 132, 255),
                    iris_hi=(246, 168, 194, 255), cloth=(150, 84, 112, 255)),
    "mint": Palette(hair=(66, 108, 92, 255), hair_hi=(120, 162, 142, 255), iris=(74, 178, 154, 255),
                    iris_hi=(168, 228, 210, 255), cloth=(92, 132, 112, 255)),
    "noir": Palette(skin=(240, 230, 226, 255), hair=(46, 42, 52, 255), hair_hi=(100, 96, 110, 255),
                    iris=(150, 152, 172, 255), iris_hi=(212, 214, 228, 255),
                    cloth=(64, 64, 76, 255), blush=(232, 172, 172, 220)),
}
HAIRSTYLES = ("bangs", "short", "long")


def _canvas():
    from PIL import Image
    return Image.new("RGBA", (W, H), (0, 0, 0, 0))


def _draw(fn) -> "object":
    from PIL import ImageDraw
    img = _canvas()
    fn(ImageDraw.Draw(img))
    return img


def _grid(rows: list[str], pal: dict):
    """Author a small sprite from a pixel grid: each char → an RGBA colour (space/'.' = transparent)."""
    from PIL import Image
    h = len(rows)
    w = max(len(r) for r in rows)
    img = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    px = img.load()
    for y, row in enumerate(rows):
        for x, ch in enumerate(row):
            c = pal.get(ch)
            if c and c[3] != 0:
                px[x, y] = c
    return img


def _dark(c: tuple, f: float = 0.68) -> tuple:
    return (int(c[0] * f), int(c[1] * f), int(c[2] * f), 255)


# Hand-authored pixel anime eye WHITE (big, round, thick rounded lash) — one eye, mirrored into a PAIR so
# the eye-DISTANCE is the `spacing` knob. 'K' lash/outline, '.' sclera, 'l' soft top-lid shade.
_EYE_OPEN = [
    "   KKKKKK   ",
    "  KKKKKKKK  ",
    " KKKKKKKKKK ",
    "KKl......lKK",
    "K..........K",
    "K..........K",
    "K..........K",
    "K..........K",
    " K........K ",
    "  K......K  ",
]
_EYE_WIDE = [
    "   KKKKKK   ",
    "  KKKKKKKK  ",
    " KKKKKKKKKK ",
    "KKl......lKK",
    "K..........K",
    "K..........K",
    "K..........K",
    "K..........K",
    "K..........K",
    " K........K ",
    "  K......K  ",
]
# the BIG sparkly iris (its own layer → gaze moves it): 'a' light rim, 'b' iris, 'p' iris-dark, 'K' pupil
# core, '*' highlight. A large iris + a big shine is what makes anime eyes read as cute.
_IRIS = [
    "  aaaaa  ",
    " aabbbaa ",
    "abbpppbba",
    "abpK*Kpba",
    "abpp**ppa",
    "abpppppba",
    " abbbba  ",
    "  aaaaa  ",
]


def _eye_pal(p: Palette) -> dict:
    return {"K": p.line, ".": (255, 253, 250, 255), "l": (236, 222, 224, 255),
            "a": p.iris_hi, "b": p.iris, "p": _dark(p.iris), "*": (255, 255, 255, 255)}


def _eye_sprite(p: Palette, state: str):
    if state == "closed":                                   # a gentle downward lash (happy closed eye)
        return _draw(lambda dr: dr.arc([0, 0, EYE_W - 1, 12], 198, 342, fill=p.line, width=3))
    return _grid(_EYE_WIDE if state == "wide" else _EYE_OPEN, _eye_pal(p))


def _iris_sprite(p: Palette):
    return _grid(_IRIS, _eye_pal(p))


def _skin(p: Palette):
    def d(dr):
        dr.polygon([(18, 78), (24, 66), (40, 66), (46, 78)], fill=p.cloth, outline=p.line)  # shoulders
        dr.rectangle([27, 60, 37, 68], fill=p.skin, outline=p.line)          # neck
        dr.ellipse([9, 6, 55, 64], fill=p.skin, outline=p.line, width=2)     # round chibi head
        dr.line([(30, 47), (32, 50)], fill=(200, 154, 136, 255))             # tiny nose
        dr.line([(32, 50), (34, 47)], fill=(200, 154, 136, 255))
    return _draw(d)


def _mouth(p: Palette, state: str):
    def d(dr):
        if state == "open":
            dr.ellipse([29, 54, 35, 61], fill=p.mouth, outline=p.line)
            dr.chord([30, 58, 34, 61], 0, 180, fill=(214, 128, 128, 255))
        elif state == "smile":
            dr.chord([27, 52, 37, 60], 20, 160, fill=p.mouth, outline=p.line)
        elif state == "frown":
            dr.arc([27, 57, 37, 66], 194, 346, fill=p.line, width=2)
        else:
            dr.line([29, 55, 35, 55], fill=p.line, width=2)
    return _draw(d)


def _blush(p: Palette):
    def d(dr):
        dr.ellipse([14, 44, 23, 49], fill=p.blush)
        dr.ellipse([41, 44, 50, 49], fill=p.blush)
    return _draw(d)


def _hair(p: Palette, style: str):
    # a rounded scalp + a SMOOTH fill-only fringe (no dark outline notches) that clears the eyes, framed by
    # side locks. `style` swaps the silhouette (short crop / long side locks).
    def d(dr):
        dr.pieslice([7, 3, 57, 53], 180, 360, fill=p.hair, outline=p.line, width=1)   # scalp dome
        if style == "short":
            dr.polygon([(11, 15), (11, 27), (22, 23), (32, 27), (42, 23), (53, 27), (53, 15)], fill=p.hair)
        else:                                                # soft bangs (fill only → notches show forehead)
            dr.polygon([(11, 15), (11, 29), (20, 25), (26, 30), (32, 24), (38, 30), (44, 25), (53, 29),
                        (53, 15)], fill=p.hair)
            top = 70 if style == "long" else 58
            dr.polygon([(9, 24), (7, top), (19, top - 6), (17, 29)], fill=p.hair, outline=p.line)
            dr.polygon([(55, 24), (57, top), (45, top - 6), (47, 29)], fill=p.hair, outline=p.line)
        dr.line([(26, 19), (23, 31)], fill=p.hair_hi, width=2)              # shine strands
        dr.line([(38, 19), (41, 31)], fill=p.hair_hi, width=2)
    return _draw(d)


# expression -> (eyes state, mouth state, blush?). Pupils hide when the eyes shut.
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


def generate_avatar(name: str, out_dir: str, style: str = "sky", spacing: int = 6,
                    hairstyle: str = "bangs", palette: Palette | None = None) -> Avatar:
    """Author the pixel-art part sprites into out_dir and assemble an editable, MODULAR rig. Customization
    is first-class: `style` picks a colour palette (art style), `spacing` sets the eye distance, `hairstyle`
    picks the default hair — and every part stays swappable/tunable by the agent/user afterwards (the eyes
    are a mirror PAIR so `spacing` moves them apart; the iris is its own layer so gaze moves it)."""
    p = palette or PALETTES.get(style, PALETTES["sky"])
    os.makedirs(out_dir, exist_ok=True)

    def save(img, fn) -> str:
        path = os.path.join(out_dir, fn)
        img.save(path)
        return path

    a = Avatar(name=name, kind="sprites", size=(W, H), meta={"style": style, "hairstyle": hairstyle})
    a.add_layer(Layer(id="skin", part="skin", states={"base": save(_skin(p), "skin.png")}, default_state="base"))
    a.add_layer(Layer(id="blush", part="blush", states={"on": save(_blush(p), "blush.png")}, default_state=""))
    # eyes: a mirror PAIR separated by `spacing` (the eye-distance knob), placed at EYE_Y
    a.add_layer(Layer(id="eyes", part="eyes", default_state="open", mirror=True, spacing=spacing,
                      pos=(0, EYE_Y), states={
                          s: save(_eye_sprite(p, s), f"eyes_{s}.png") for s in ("open", "closed", "wide")}))
    # iris: its own mirror pair, spaced to sit centred inside each eye; gaze shifts this layer
    pupil_spacing = spacing + (EYE_W - IRIS_W)
    from PIL import Image as _Img
    a.add_layer(Layer(id="pupils", part="pupils", default_state="on", mirror=True, spacing=pupil_spacing,
                      pos=(0, EYE_Y + 3), states={
                          "on": save(_iris_sprite(p), "iris.png"),
                          "off": save(_Img.new("RGBA", (IRIS_W, 8), (0, 0, 0, 0)), "iris_off.png")}))
    a.add_layer(Layer(id="mouth", part="mouth", default_state="closed", states={
        s: save(_mouth(p, s), f"mouth_{s}.png") for s in ("closed", "smile", "open", "frown")}))
    # hair: all styles kept as STATES so the user/agent can swap hairstyle by re-picking the default state
    a.add_layer(Layer(id="hair", part="hair", default_state=hairstyle, states={
        h: save(_hair(p, h), f"hair_{h}.png") for h in HAIRSTYLES}))

    for expr, (eyes, mouth, blush) in _EXPR.items():
        mapping = {"eyes": eyes, "mouth": mouth, "blush": "on" if blush else "",
                   "pupils": "off" if eyes == "closed" else "on"}
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
