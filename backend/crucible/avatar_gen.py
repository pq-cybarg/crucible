from __future__ import annotations
# PROCEDURAL avatar creation — draw a simple, modular pixel-art face as separate part sprites (each on
# its own transparent layer) and assemble them into an Avatar spec with expression states. This gives a
# real, shippable default avatar (NOT the reference samples) and demonstrates procedural character
# creation: params (colors) → an editable rig the agentic flow can then modify (swap hair, add accessory,
# add expression frames). Eyes are drawn large so key features stay recognizable when shrunk to the box.
import os
from dataclasses import dataclass

from crucible.avatar import Avatar, Layer

# A larger native canvas (more px to spend on the KEY anime features) that still shrinks crisply into the
# tiny TUI box. The face is drawn CHIBI + BOLD: an oversized head, huge sparkly eyes with thick lashes, a
# tiny nose/mouth, and framing bangs — high-contrast shapes chosen so they survive the low-res render.
W, H = 96, 120
EYE_CY = 66
EYE_CX = (34, 62)                                  # left, right eye centres


@dataclass
class Palette:
    # High LUMINANCE contrast so features survive the low-res sepia render: a LIGHT face vs DARK hair /
    # outlines / mouth (in a two-tone ramp, only luminance differences read — colour alone vanishes).
    skin: tuple = (252, 228, 210, 255)
    hair: tuple = (58, 40, 38, 255)                # dark — a strong mass against the light face
    hair_hi: tuple = (96, 68, 60, 255)
    eye: tuple = (74, 140, 200, 255)               # bright anime iris (blue)
    eye_dk: tuple = (28, 34, 52, 255)              # near-black pupil (reads dark in sepia)
    line: tuple = (36, 26, 26, 255)                # stark dark outline (bold, high-contrast)
    cloth: tuple = (70, 92, 138, 255)
    blush: tuple = (240, 150, 150, 210)
    mouth: tuple = (150, 60, 62, 255)


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
        dr.polygon([(30, 118), (36, 96), (60, 96), (66, 118)], fill=p.cloth, outline=p.line)  # shoulders
        dr.rectangle([42, 86, 54, 100], fill=p.skin, outline=p.line)      # neck
        dr.ellipse([14, 14, 82, 96], fill=p.skin, outline=p.line, width=3)  # big chibi head, bold outline
        # a small but DEFINITE nose mark (a soft chevron) so a feature reads between eyes and mouth
        dr.line([(46, 76), (49, 79)], fill=(188, 138, 120, 255), width=2)
        dr.line([(49, 79), (52, 76)], fill=(188, 138, 120, 255), width=2)
    return _draw(d)


def _hair(p: Palette):
    def d(dr):
        # side locks framing the cheeks
        dr.polygon([(12, 40), (9, 86), (24, 80), (22, 46)], fill=p.hair, outline=p.line)
        dr.polygon([(84, 40), (87, 86), (72, 80), (74, 46)], fill=p.hair, outline=p.line)
        # top dome / scalp
        dr.pieslice([10, 2, 86, 78], 180, 360, fill=p.hair, outline=p.line, width=3)
        # bangs: a soft fringe with points, dipping over the forehead but clearing the eyes
        dr.polygon([(20, 44), (30, 40), (38, 56), (48, 42), (58, 56), (66, 40), (76, 44),
                    (74, 30), (22, 30)], fill=p.hair, outline=p.line, width=2)
        dr.line([(40, 24), (34, 40)], fill=p.hair_hi, width=3)            # a couple of shine strands
        dr.line([(56, 24), (60, 40)], fill=p.hair_hi, width=3)
    return _draw(d)


def _eyes(p: Palette, state: str):
    """The eye WHITES + bold upper LASHES only — the irises/pupils are a separate layer so gaze can move
    them independently. Big and bold so they read as anime eyes even in the tiny terminal box."""
    def d(dr):
        for cx in EYE_CX:
            if state == "closed":                                        # happy closed eyes (upward arc)
                dr.arc([cx - 11, EYE_CY - 2, cx + 11, EYE_CY + 16], 200, 340, fill=p.line, width=3)
                continue
            grow = 3 if state == "wide" else 0
            box = [cx - 11 - grow, EYE_CY - 13 - grow, cx + 11 + grow, EYE_CY + 11]
            dr.ellipse(box, fill=(255, 255, 255, 255), outline=p.line, width=2)   # eye white
            dr.arc([box[0] - 1, box[1] - 2, box[2] + 1, box[1] + 18], 198, 342, fill=p.line, width=4)  # lash
    return _draw(d)


def _pupils(p: Palette, state: str):
    """The big sparkly irises on their OWN transparent layer — the gaze axis shifts this layer a few px so
    the eyes glance around while the whites stay put. `off` = hidden (closed-eye expressions / blinks)."""
    def d(dr):
        if state == "off":
            return
        for cx in EYE_CX:
            dr.ellipse([cx - 8, EYE_CY - 9, cx + 8, EYE_CY + 7], fill=p.eye, outline=p.line, width=1)  # iris
            dr.ellipse([cx - 4, EYE_CY - 3, cx + 4, EYE_CY + 5], fill=p.eye_dk)                         # pupil
            dr.ellipse([cx - 6, EYE_CY - 8, cx - 1, EYE_CY - 3], fill=(255, 255, 255, 255))             # sparkle
            dr.ellipse([cx + 2, EYE_CY + 1, cx + 5, EYE_CY + 4], fill=(255, 255, 255, 220))             # 2nd shine
    return _draw(d)


def _mouth(p: Palette, state: str):
    # bold + DARK so the mouth reads at low res (a thin/reddish mouth washes out in the sepia ramp)
    def d(dr):
        if state == "open":
            dr.ellipse([40, 82, 56, 98], fill=p.mouth, outline=p.line, width=3)      # open/talking
            dr.chord([43, 90, 53, 97], 0, 180, fill=(210, 120, 120, 255))           # tongue hint
        elif state == "smile":
            dr.chord([38, 78, 58, 94], 20, 160, fill=p.mouth, outline=p.line, width=2)   # filled smile (reads)
        elif state == "frown":
            dr.arc([38, 88, 58, 106], 192, 348, fill=p.line, width=4)                # downward frown
        else:  # closed — a short bold dark bar so a mouth is still visible
            dr.line([42, 85, 54, 85], fill=p.line, width=3)
    return _draw(d)


def _blush(p: Palette):
    def d(dr):
        dr.ellipse([22, 74, 36, 83], fill=p.blush)
        dr.ellipse([60, 74, 74, 83], fill=p.blush)
    return _draw(d)


# expression -> (eyes state, mouth state, blush?). Pupils follow the eyes: hidden when the eyes are closed.
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
    a.add_layer(Layer(id="pupils", part="pupils", default_state="on", states={
        s: save(_pupils(p, s), f"pupils_{s}.png") for s in ("on", "off")}))
    a.add_layer(Layer(id="mouth", part="mouth", default_state="closed", states={
        s: save(_mouth(p, s), f"mouth_{s}.png") for s in ("closed", "smile", "open", "frown")}))
    a.add_layer(Layer(id="hair", part="hair", states={"base": save(_hair(p), "hair.png")}, default_state="base"))

    for expr, (eyes, mouth, blush) in _EXPR.items():
        mapping = {"eyes": eyes, "mouth": mouth}
        mapping["blush"] = "on" if blush else ""
        mapping["pupils"] = "off" if eyes == "closed" else "on"      # hide pupils when the eyes shut
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
