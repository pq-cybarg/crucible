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
    # Default = the reference look: warm tan skin, a near-black BOB, brown eyes behind ROUND GLASSES,
    # a black ribbed sweater, thin brows, calm expression.
    skin: tuple = (236, 206, 174, 255)
    hair: tuple = (42, 34, 38, 255)                # near-black bob
    hair_hi: tuple = (84, 70, 76, 255)
    iris: tuple = (126, 90, 64, 255)               # warm brown iris
    iris_hi: tuple = (182, 140, 104, 255)
    line: tuple = (36, 28, 30, 255)                # near-black outline / pupil core
    cloth: tuple = (34, 32, 40, 255)               # black sweater
    blush: tuple = (228, 172, 156, 90)             # very subtle
    mouth: tuple = (176, 118, 108, 255)
    glass: tuple = (28, 22, 24, 255)               # round-glasses frame
    brow: tuple = (54, 44, 46, 255)


# ART-STYLE presets (palette swaps) — pick the whole look; user/agent can tune further. Tunable proof.
PALETTES: dict[str, Palette] = {
    "ink":   Palette(),                            # the reference (dark hair, brown eyes)
    "ash":   Palette(hair=(70, 64, 74, 255), hair_hi=(120, 114, 126, 255),
                     iris=(96, 132, 150, 255), iris_hi=(168, 200, 214, 255)),
    "cocoa": Palette(hair=(74, 52, 44, 255), hair_hi=(126, 96, 80, 255),
                     iris=(150, 96, 60, 255), iris_hi=(206, 156, 110, 255)),
    "plum":  Palette(hair=(58, 44, 62, 255), hair_hi=(110, 88, 118, 255),
                     iris=(140, 92, 140, 255), iris_hi=(200, 156, 200, 255)),
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


# The eye is authored as THREE layers so the iris can be OCCLUDED correctly (a glance can't spill over the
# lash or outside the white):
#   • SCLERA (part 'eyes')  — the white + a thin round outline. It's the clip mask for the iris.
#   • IRIS   (part 'pupils')— the big sparkly iris; clip='eyes' keeps it inside the white; gaze moves it.
#   • LASH   (part 'eyelash')— the thick upper lid, drawn ABOVE the iris so it covers the top of it.
# 'K' outline, '.' white; iris: 'a' light rim, 'b' iris, 'p' iris-dark, 'K' pupil core, '*' highlight.
_SCLERA_OPEN = [
    "   KKKKKK   ",
    "  K......K  ",
    " K........K ",
    "K..........K",
    "K..........K",
    "K..........K",
    " K........K ",
    "  K......K  ",
    "   KKKKKK   ",
]
_SCLERA_WIDE = [
    "   KKKKKK   ",
    "  K......K  ",
    " K........K ",
    "K..........K",
    "K..........K",
    "K..........K",
    "K..........K",
    " K........K ",
    "  K......K  ",
    "   KKKKKK   ",
]
_LASH_OPEN = [
    "   KKKKKK   ",
    "  KKKKKKKK  ",
    " KKKKKKKKKK ",
]
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
    return {"K": p.line, ".": (255, 253, 250, 255),
            "a": p.iris_hi, "b": p.iris, "p": _dark(p.iris), "*": (255, 255, 255, 255)}


def _sclera_sprite(p: Palette, state: str):
    from PIL import Image
    if state == "closed":                                   # shut → no white (the lash draws the closed line)
        return Image.new("RGBA", (EYE_W, 2), (0, 0, 0, 0))
    return _grid(_SCLERA_WIDE if state == "wide" else _SCLERA_OPEN, _eye_pal(p))


def _lash_sprite(p: Palette, state: str):
    if state == "closed":                                   # a gentle downward lash — the happy closed eye
        return _draw(lambda dr: dr.arc([0, 0, EYE_W - 1, 12], 198, 342, fill=p.line, width=3))
    return _grid(_LASH_OPEN, _eye_pal(p))


def _iris_sprite(p: Palette):
    return _grid(_IRIS, _eye_pal(p))


def _eye_centres(spacing: int) -> tuple:
    """Where the two eyes (and thus the glasses lenses) sit, derived from the mirror-pair spacing."""
    cx = W // 2
    return (cx - spacing // 2 - EYE_W // 2, cx + spacing // 2 + EYE_W // 2, EYE_Y + 4)


def _skin(p: Palette):
    # Back-to-front so the round chin overlaps the neck (no seam): sweater + ribbed collar, neck, then head.
    def d(dr):
        nose = (216, 176, 150, 255)
        dr.polygon([(10, 78), (18, 68), (46, 68), (54, 78)], fill=p.cloth, outline=p.line)  # black sweater
        for x in range(20, 45, 3):                                           # ribbed collar
            dr.line([(x, 70), (x, 77)], fill=(54, 52, 62, 255))
        dr.rectangle([29, 58, 35, 70], fill=p.skin)                          # slim neck
        dr.ellipse([9, 8, 55, 66], fill=p.skin, outline=p.line, width=2)     # warm tan head
        dr.ellipse([31, 49, 33, 51], fill=nose)                              # tiny nose dot
    return _draw(d)


def _brows(p: Palette):
    def d(dr):
        dr.line([(20, 27), (28, 26)], fill=p.brow, width=2)
        dr.line([(36, 26), (44, 27)], fill=p.brow, width=2)
    return _draw(d)


def _glasses(p: Palette, spacing: int):
    # ROUND glasses (the reference's signature): a thin frame ring on each eye, a bridge, and short temples.
    lcx, rcx, cy = _eye_centres(spacing)
    r = 8

    def d(dr):
        for ccx in (lcx, rcx):
            dr.ellipse([ccx - r, cy - r, ccx + r, cy + r], outline=p.glass, width=2)
        dr.line([(lcx + r - 1, cy - 2), (rcx - r + 1, cy - 2)], fill=p.glass, width=2)      # bridge
        dr.line([(lcx - r, cy - 2), (lcx - r - 5, cy - 4)], fill=p.glass, width=2)          # temples
        dr.line([(rcx + r, cy - 2), (rcx + r + 5, cy - 4)], fill=p.glass, width=2)
    return _draw(d)


def _mouth(p: Palette, state: str):
    # calm/subtle by default (the reference is deadpan); still shifts for the emotive expressions
    def d(dr):
        if state == "open":
            dr.ellipse([29, 54, 35, 61], fill=(150, 92, 84, 255), outline=p.line)
        elif state == "smile":
            dr.arc([28, 52, 36, 59], 20, 160, fill=p.line, width=2)
        elif state == "frown":
            dr.arc([28, 57, 36, 64], 200, 340, fill=p.line, width=2)
        else:
            dr.line([30, 55, 34, 55], fill=p.line, width=2)
    return _draw(d)


def _blush(p: Palette):
    def d(dr):
        dr.ellipse([14, 44, 22, 48], fill=p.blush)
        dr.ellipse([42, 44, 50, 48], fill=p.blush)
    return _draw(d)


def _hair(p: Palette, style: str):
    # a near-black BOB: a rounded scalp + a straight fringe that stops at BROW level (so the eyes + glasses
    # show below it) + side locks that frame the cheeks WITHOUT covering the eyes. `style` varies the lock
    # length (short crop → chin-length → longer).
    def d(dr):
        dr.pieslice([6, 3, 58, 48], 180, 360, fill=p.hair)                   # scalp — top half only (to ~y25)
        # straight fringe with a soft centre part; bottom edge ~y26 = brow level, clearing the eyes at y31
        dr.polygon([(6, 12), (6, 26), (16, 23), (22, 27), (28, 22), (32, 26), (36, 22), (42, 27),
                    (48, 23), (58, 26), (58, 12)], fill=p.hair)
        lock = 46 if style == "short" else (68 if style == "long" else 60)   # chin-length bob by default
        # side locks: inner edge kept OUTSIDE the eyes (x≈14 / x≈50) so they only frame the cheeks
        dr.polygon([(6, 18), (3, lock), (16, lock), (15, 30), (10, 22)], fill=p.hair, outline=p.line)
        dr.polygon([(58, 18), (61, lock), (48, lock), (49, 30), (54, 22)], fill=p.hair, outline=p.line)
        for sx in (18, 30, 40):                                             # subtle strand highlights
            dr.line([(sx, 8), (sx - 2, 22)], fill=p.hair_hi, width=1)
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


def generate_avatar(name: str, out_dir: str, style: str = "ink", spacing: int = 6,
                    hairstyle: str = "bangs", palette: Palette | None = None) -> Avatar:
    """Author the pixel-art part sprites into out_dir and assemble an editable, MODULAR rig. Customization
    is first-class: `style` picks a colour palette (art style), `spacing` sets the eye distance, `hairstyle`
    picks the default hair — and every part stays swappable/tunable by the agent/user afterwards (the eyes
    are a mirror PAIR so `spacing` moves them apart; the iris is its own layer so gaze moves it)."""
    p = palette or PALETTES.get(style, PALETTES["ink"])
    os.makedirs(out_dir, exist_ok=True)

    def save(img, fn) -> str:
        path = os.path.join(out_dir, fn)
        img.save(path)
        return path

    a = Avatar(name=name, kind="sprites", size=(W, H), meta={"style": style, "hairstyle": hairstyle})
    a.add_layer(Layer(id="skin", part="skin", states={"base": save(_skin(p), "skin.png")}, default_state="base"))
    a.add_layer(Layer(id="blush", part="blush", states={"on": save(_blush(p), "blush.png")}, default_state=""))
    a.add_layer(Layer(id="brows", part="brows", states={"base": save(_brows(p), "brows.png")}, default_state="base"))
    # eyes = the SCLERA (white): a mirror PAIR separated by `spacing` (the eye-distance knob). Doubles as
    # the clip mask for the iris.
    a.add_layer(Layer(id="eyes", part="eyes", default_state="open", mirror=True, spacing=spacing,
                      pos=(0, EYE_Y), states={
                          s: save(_sclera_sprite(p, s), f"eyes_{s}.png") for s in ("open", "closed", "wide")}))
    # pupils = the iris: its own mirror pair, CLIPPED to the eyes' sclera so a glance can't spill out; gaze
    # shifts this layer.
    pupil_spacing = spacing + (EYE_W - IRIS_W)
    from PIL import Image as _Img
    a.add_layer(Layer(id="pupils", part="pupils", default_state="on", mirror=True, spacing=pupil_spacing,
                      pos=(0, EYE_Y + 1), clip="eyes", states={
                          "on": save(_iris_sprite(p), "iris.png"),
                          "off": save(_Img.new("RGBA", (IRIS_W, 8), (0, 0, 0, 0)), "iris_off.png")}))
    # eyelash = the upper lid, drawn ABOVE the iris so it OCCLUDES the top of it (proper lidded look)
    a.add_layer(Layer(id="eyelash", part="eyelash", default_state="open", mirror=True, spacing=spacing,
                      pos=(0, EYE_Y), states={
                          s: save(_lash_sprite(p, s), f"lash_{s}.png") for s in ("open", "closed")}))
    # round glasses (the reference's signature), aligned to the eye spacing, over the eyes but under the hair
    a.add_layer(Layer(id="glasses", part="glasses",
                      states={"on": save(_glasses(p, spacing), "glasses.png"), "off": ""},
                      default_state="on"))
    a.add_layer(Layer(id="mouth", part="mouth", default_state="closed", states={
        s: save(_mouth(p, s), f"mouth_{s}.png") for s in ("closed", "smile", "open", "frown")}))
    # hair: all styles kept as STATES so the user/agent can swap hairstyle by re-picking the default state
    a.add_layer(Layer(id="hair", part="hair", default_state=hairstyle, states={
        h: save(_hair(p, h), f"hair_{h}.png") for h in HAIRSTYLES}))

    for expr, (eyes, mouth, blush) in _EXPR.items():
        shut = eyes == "closed"
        mapping = {"eyes": eyes, "mouth": mouth, "blush": "on" if blush else "",
                   "pupils": "off" if shut else "on", "eyelash": "closed" if shut else "open"}
        a.set_expression(expr, mapping)
    a.save(os.path.join(out_dir, "avatar.json"))
    return a


# Feature geometry measured for template_face.webp at 128px: the round-glasses BOX (both lenses), the two
# LENS centres, and the small MOUTH box. Other imports can pass their own.
TEMPLATE_GLASSES_BOX = (43, 50, 95, 76)
TEMPLATE_LENS = [(56, 63), (81, 63)]
TEMPLATE_MOUTH = (60, 82, 68, 87)


def rig_portrait(image_path: str, out_dir: str, name: str = "kiri", native: int = 128,
                 lens: list | None = None, glasses_box: tuple | None = None, mouth_box: tuple | None = None):
    """DECONSTRUCT a hand-drawn portrait (e.g. template_face.webp) into real, separable part LAYERS and rig
    them — so it looks EXACTLY like the reference AND animates, with no overlay hacks:

      • face  — the portrait with the MOUTH genuinely removed (its pixels filled with the local skin), so
                the mouth layer isn't imprinting a patch over an existing mouth. PROTECTED (custom import).
      • eyes  — the whole glasses region is LIFTED OUT as its own layer. `open` shows the base's eyes; the
                `closed` state redraws the eye interiors as lids with the FRAME kept on top, so a blink sits
                BEHIND the glasses, never over them.
      • mouth — a small drawn mouth shape only (neutral/smile/open/frown); the face already has clean skin
                where the original mouth was, so there's nothing to imprint.

    A light posterize flattens the busy hair + raises contrast (also helps the TUI). `lens`/`glasses_box`/
    `mouth_box` locate the features at `native` px (defaults are measured for the bundled template)."""
    from PIL import Image, ImageDraw, ImageEnhance
    import statistics
    os.makedirs(out_dir, exist_ok=True)
    gx0, gy0, gx1, gy1 = glasses_box or TEMPLATE_GLASSES_BOX
    mx0, my0, mx1, my1 = mouth_box or TEMPLATE_MOUTH
    mcx, mcy = (mx0 + mx1) // 2, (my0 + my1) // 2

    base = Image.open(image_path).convert("RGB").resize((native, native), Image.LANCZOS)
    # STRONG simplify + contrast → a clean, stylised pixel look: the busy hair flattens to a bold mass, the
    # features pop, and it reads far better shrunk into the TUI box.
    base = ImageEnhance.Contrast(base).enhance(1.35).quantize(colors=14, dither=Image.NONE).convert("RGBA")

    def save(img, fn) -> str:
        path = os.path.join(out_dir, fn)
        img.save(path)
        return path

    def lum(x, y) -> float:
        r, g, b = base.getpixel((max(0, min(native - 1, x)), max(0, min(native - 1, y))))[:3]
        return 0.3 * r + 0.6 * g + 0.1 * b

    # DETECT each eye's centre (centroid of the dark iris/eyeliner inside its lens) so the lids land on the
    # ACTUAL eyes — per eye, so an uneven hand-drawn pair stays symmetric. Overridable via `lens`.
    def detect(x0, x1, y0, y1) -> tuple:
        xs = ys = n = 0
        for y in range(y0, y1):
            for x in range(x0, x1):
                if lum(x, y) < 105:
                    xs += x; ys += y; n += 1
        return (xs // n, ys // n) if n else ((x0 + x1) // 2, (y0 + y1) // 2)
    cxm = (gx0 + gx1) // 2
    lens = lens or [detect(gx0 + 5, cxm - 3, gy0 + 5, gy1 - 4), detect(cxm + 3, gx1 - 5, gy0 + 5, gy1 - 4)]

    # ONE robust skin tone (median rejects shadows/hair) for BOTH eyelids — so the closed eyes are symmetric
    skin_pts = [(46, 76), (50, 80), (54, 82), (74, 82), (78, 80), (82, 76), (64, 90), (60, 86), (68, 86)]
    cols = [base.getpixel((max(0, min(native - 1, x)), max(0, min(native - 1, y))))[:3] for x, y in skin_pts]
    skin = tuple(int(statistics.median(c[i] for c in cols)) for i in range(3))
    dark = (56, 40, 42, 255)

    # a tiny catchlight so the eyes aren't dead (the posterize can drop the original reflection)
    bd = ImageDraw.Draw(base)
    for (cx, cy) in lens:
        bd.point([(cx - 2, cy - 1)], fill=(250, 248, 244, 255))

    # EYES layer: lift the whole glasses region as a swappable layer. `open` is transparent (the base's own
    # eyes show). The `closed` state KEEPS the eye MAKEUP (the dark eyeliner) and fills only the light
    # EYEBALL (white+iris) with skin — so a blink shuts the eye without erasing her look, and the frame
    # (outside the eye) is untouched, so the lids stay BEHIND the glasses.
    eyes_closed = Image.new("RGBA", (native, native), (0, 0, 0, 0))
    eyes_closed.paste(base.crop((gx0, gy0, gx1, gy1)), (gx0, gy0))
    ecpx = eyes_closed.load()
    for (cx, cy) in lens:
        for y in range(cy - 6, cy + 9):
            for x in range(cx - 9, cx + 10):
                r, g, b, aa = eyes_closed.getpixel((x, y))
                if aa and (0.3 * r + 0.6 * g + 0.1 * b) > 82:               # eyeball → skin; eyeliner kept
                    ecpx[x, y] = (*skin, 255)
        ImageDraw.Draw(eyes_closed).arc([cx - 7, cy + 1, cx + 7, cy + 8], 204, 336, fill=dark, width=1)  # lower lash

    # MOUTH: genuinely erase the drawn mouth from the FACE (fill the mouth pixels with the chin skin tone).
    # The mouth layer then supplies just the shape — no patch imprinted over an existing mouth.
    ImageDraw.Draw(base).rectangle([mx0, my0, mx1, my1], fill=(*skin, 255))

    def mouth_sprite(state: str):
        im = Image.new("RGBA", (native, native), (0, 0, 0, 0))
        d = ImageDraw.Draw(im)                                              # ONLY the mouth shape, no skin
        if state == "neutral":
            d.line([(mcx - 3, mcy), (mcx + 3, mcy)], fill=dark, width=1)
        elif state == "smile":
            d.arc([mcx - 4, mcy - 2, mcx + 4, mcy + 3], 25, 155, fill=dark, width=2)
        elif state == "open":
            d.ellipse([mcx - 3, mcy - 1, mcx + 3, mcy + 5], fill=(112, 66, 64, 255), outline=dark)
        elif state == "frown":
            d.arc([mcx - 4, mcy + 2, mcx + 4, mcy + 8], 205, 335, fill=dark, width=2)
        return im

    # face_box: what the tiny TUI box zooms into (the head/face — not the big dark sweater/hair bust, which
    # shrinks to a dark blob). The web keeps the full portrait.
    fb = (max(0, gx0 - 20), max(0, gy0 - 44), min(native, gx1 + 20), min(native, my1 + 22))
    a = Avatar(name=name, kind="sprites", size=(native, native),
               meta={"imported_from": os.path.basename(image_path), "rigged": True, "face_box": list(fb)})
    a.add_layer(Layer(id="face", part="face", protected=True,
                      states={"base": save(base, "base.png")}, default_state="base"))
    a.add_layer(Layer(id="eyes", part="eyes", default_state="open",
                      states={"open": "", "closed": save(eyes_closed, "eyes_closed.png")}))
    a.add_layer(Layer(id="mouth", part="mouth", default_state="neutral", states={
        s: save(mouth_sprite(s), f"mouth_{s}.png") for s in ("neutral", "smile", "open", "frown")}))
    expr = {"neutral": ("open", "neutral"), "happy": ("open", "smile"), "laughing": ("closed", "open"),
            "surprised": ("open", "open"), "sad": ("open", "frown"), "angry": ("open", "frown"),
            "love": ("closed", "smile"), "curious": ("open", "neutral")}
    for e, (ey, m) in expr.items():
        a.set_expression(e, {"eyes": ey, "mouth": m})
    a.save(os.path.join(out_dir, "avatar.json"))
    return a


# A character supplied as SEPARATE, pre-aligned part sprites (each on a painted checkerboard = transparency,
# except the eyes+glasses which sit on the tan skin). role → filename.
PART_FILES = {
    "side": "side hair bob and back.jpg", "head": "blank head.jpg", "chin": "chin and neck.jpg",
    "eyes": "eyes+glasses overlaid on face with loops.jpg", "bangs": "bangs front.jpg",
    "mouth": "mouth.jpg", "sweater": "sweater.jpg", "necklace": "necklace.jpg",
    "headphones": "headphones.jpg",
}


def _despeckle(rgba):
    """A 3×3 median on the alpha drops isolated leftover speckles (and fills pinholes) after bg removal."""
    from PIL import Image, ImageFilter
    r, g, b, al = rgba.split()
    return Image.merge("RGBA", (r, g, b, al.filter(ImageFilter.MedianFilter(3))))


def _dechecker(im):
    """Remove a painted checkerboard 'transparency' background → real alpha (the two checker greys, plus a
    light-grey JPEG-fringe catch-all), then despeckle."""
    import numpy as np
    from PIL import Image
    a = np.asarray(im.convert("RGB")).astype(int)
    cor = a[:60, :60].reshape(-1, 3)
    u, c = np.unique(cor, axis=0, return_counts=True)
    m = np.zeros(a.shape[:2], bool)
    for cc in u[np.argsort(-c)[:2]]:
        m |= np.abs(a - cc).sum(2) < 70
    r, g, b = a[..., 0], a[..., 1], a[..., 2]
    m |= (np.abs(r - g) < 16) & (np.abs(g - b) < 16) & (np.minimum(np.minimum(r, g), b) > 172)
    return _despeckle(Image.fromarray(np.dstack([a.astype("uint8"), np.where(m, 0, 255).astype("uint8")]), "RGBA"))


def _detan(im):
    """The eyes+glasses sit on the flat tan skin — knock that out so only the eyes/glasses/brows remain."""
    import numpy as np
    from PIL import Image
    a = np.asarray(im.convert("RGB")).astype(int)
    m = np.abs(a - a[3, 3]).sum(2) < 44
    return _despeckle(Image.fromarray(np.dstack([a.astype("uint8"), np.where(m, 0, 255).astype("uint8")]), "RGBA"))


def _dechecker_light(im, sq: int = 16):
    """Checker removal for a LIGHT part (white headphones, grey chain) where the plain colour-match would
    eat the part itself: only pixels that ALTERNATE (a checker colour whose ±sq neighbour is the OTHER
    checker colour) are the background; a solid light region (a cup) has no alternating neighbour → kept."""
    import numpy as np
    from PIL import Image
    a = np.asarray(im.convert("RGB")).astype(int)
    near = lambda c, t: np.abs(a - c).sum(2) < t
    wm, gm = near((252, 252, 252), 58), near((190, 190, 190), 62)
    roll = lambda m, dx, dy: np.roll(np.roll(m, dy, 0), dx, 1)
    ga = roll(gm, sq, 0) | roll(gm, -sq, 0) | roll(gm, 0, sq) | roll(gm, 0, -sq)
    wa = roll(wm, sq, 0) | roll(wm, -sq, 0) | roll(wm, 0, sq) | roll(wm, 0, -sq)
    ch = (wm & ga) | (gm & wa)
    for _ in range(2):                                       # grow into the adjacent checker-coloured fringe
        ch = ch | ((roll(ch, 1, 0) | roll(ch, -1, 0) | roll(ch, 0, 1) | roll(ch, 0, -1)) & (wm | gm))
    return _despeckle(Image.fromarray(np.dstack([a.astype("uint8"), np.where(ch, 0, 255).astype("uint8")]), "RGBA"))


def _lift_shadows(im, black: int = 44, gain: float = 0.78):
    """Lift the blacks + gently compress → a brighter, minimal-shadow CEL look (BotW-ish) that also reads
    in the low-res TUI box. Alpha preserved."""
    from PIL import Image
    r, g, b, al = im.split()
    f = [max(0, min(255, int(black + p * gain))) for p in range(256)]
    return Image.merge("RGBA", (r.point(f), g.point(f), b.point(f), al))


def _scale_about_centroid(im, s: float):
    """Scale a part by `s` about its own centre of mass (keeps it aligned) — e.g. shrink an oversized
    necklace without breaking the shared coordinate frame."""
    import numpy as np
    from PIL import Image
    ys, xs = np.where(np.asarray(im.split()[-1]) > 60)
    if not len(xs):
        return im
    cx, cy = int(xs.mean()), int(ys.mean())
    w, h = im.size
    sm = im.resize((max(1, int(w * s)), max(1, int(h * s))), Image.LANCZOS)
    out = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    out.alpha_composite(sm, (cx - int(cx * s), cy - int(cy * s)))
    return out


def build_from_parts(parts_dir: str, out_dir: str, name: str = "kiri", native: int = 200,
                     files: dict | None = None):
    """COMPOSE a companion from pre-separated, pre-aligned part sprites and rig it — the true modular
    deconstruction. Backgrounds are removed, parts are grouped by z-order into layers (hair+head+chin
    behind → eyes → bangs → mouth → sweater+necklace), and the EYES and MOUTH stay as riggable layers:
    a blink shuts the eyes (keeping the glasses/brows/eyeliner); the mouth swaps shape to emote. Looks
    exactly like the source art, animates, and every part is genuinely separate."""
    import numpy as np
    from PIL import Image, ImageDraw
    files = files or PART_FILES
    os.makedirs(out_dir, exist_ok=True)
    P = {}
    for role, fn in files.items():
        if role == "necklace":
            continue                                        # supplied necklace reads as a stethoscope — drawn below
        path = os.path.join(parts_dir, fn)
        if os.path.exists(path):
            src = Image.open(path)
            if role == "eyes":
                P[role] = _detan(src)
            elif role == "headphones":                      # white cups: colour-match would erase them
                P[role] = _dechecker_light(src)
            else:
                P[role] = _dechecker(src)
    if "head" not in P or "eyes" not in P:
        raise RuntimeError("need at least a head + eyes part")
    W0 = next(iter(P.values())).width

    # robust union bounding box (columns/rows with real coverage — ignore stray fringe pixels)
    al = np.zeros((W0, W0), int)
    for im in P.values():
        al = np.maximum(al, np.asarray(im.split()[-1]))
    op = al > 40
    cols, rows = np.where(op.sum(0) > 8)[0], np.where(op.sum(1) > 8)[0]
    bbox = (int(cols.min()), int(rows.min()), int(cols.max()) + 1, int(rows.max()) + 1)
    nh = round(native * (bbox[3] - bbox[1]) / (bbox[2] - bbox[0]))

    def grp(*roles):
        c = Image.new("RGBA", (W0, W0), (0, 0, 0, 0))
        for r in roles:
            if r in P:
                c.alpha_composite(P[r])
        return c.crop(bbox).resize((native, nh), Image.LANCZOS)

    base = _lift_shadows(grp("side", "head", "chin"))
    eyes = _lift_shadows(grp("eyes"))
    bangs = _lift_shadows(grp("bangs"))
    mouth = grp("mouth")
    body = _lift_shadows(grp("sweater"))
    headphones = _lift_shadows(grp("headphones")) if "headphones" in P else None

    # a SUBTLE necklace drawn at the sweater collar (thin chain + small pendant) — matches the reference,
    # avoids the supplied part's stethoscope shape
    import numpy as _np
    srows = _np.where(_np.asarray(body.split()[-1]).sum(1) > 5)[0]
    if len(srows):
        collar, cx = int(srows[0]), native // 2
        py = collar + max(3, nh // 20)
        dn = ImageDraw.Draw(body)
        dn.line([(cx - native // 11, collar + 1), (cx, py)], fill=(198, 198, 202, 255), width=1)
        dn.line([(cx + native // 11, collar + 1), (cx, py)], fill=(198, 198, 202, 255), width=1)
        dn.ellipse([cx - 2, py - 1, cx + 3, py + 4], fill=(182, 182, 188, 255), outline=(120, 120, 126, 255))

    def save(img, fn) -> str:
        path = os.path.join(out_dir, fn)
        img.save(path)
        return path

    # a robust skin tone from the cheeks (for the closed-eye lids)
    def px(im, x, y):
        return im.convert("RGB").getpixel((max(0, min(native - 1, x)), max(0, min(native - 1, y))))
    cheeks = [px(base, int(native * f), nh * 55 // 100) for f in (0.32, 0.68, 0.4, 0.6)]
    import statistics
    skin = tuple(int(statistics.median(c[i] for c in cheeks)) for i in range(3))
    dark = (52, 38, 40, 255)

    # detect each eye centre (dark iris centroid inside its lens) on the eyes sprite → symmetric lids
    ea = np.asarray(eyes.convert("RGB")).astype(int)
    lumf = 0.3 * ea[..., 0] + 0.6 * ea[..., 1] + 0.1 * ea[..., 2]
    alpha = np.asarray(eyes.split()[-1])
    def eye_centre(x0, x1):
        m = (lumf < 110) & (alpha > 60)
        m[:, :x0] = False; m[:, x1:] = False
        ys, xs = np.where(m)
        return (int(xs.mean()), int(ys.mean())) if len(xs) else ((x0 + x1) // 2, nh // 2)
    mid = native // 2
    lens = [eye_centre(mid - native // 3, mid), eye_centre(mid, mid + native // 3)]

    # CLOSED eyes: fill the eyeball opening with skin (KEEP eyeliner/brows/glasses), then draw a clean,
    # CUTE happy closed-eye curve (^‿^) per eye — consistent + intentional, not a squinty half-lidded patch.
    eyes_closed = eyes.copy()
    ecp = eyes_closed.load()
    rr = max(8, native // 13)
    for (cx, cy) in lens:
        for y in range(cy - rr, cy + rr + 2):
            for x in range(cx - rr - 2, cx + rr + 3):
                if 0 <= x < native and 0 <= y < nh and (x - cx) ** 2 + ((y - cy) * 1.15) ** 2 <= rr * rr:
                    r, g, b, aa = eyes_closed.getpixel((x, y))
                    if aa and (0.3 * r + 0.6 * g + 0.1 * b) > 52:         # eyeball (sclera/iris) → skin
                        ecp[x, y] = (*skin, 255)
        ew = max(9, native // 15)
        ImageDraw.Draw(eyes_closed).arc([cx - ew, cy - 3, cx + ew, cy + rr], 200, 340, fill=dark, width=3)

    # PUPILS layer. This eye style is "dead-fish": a flat BROWN IRIS with eyeliner on top and almost no
    # white sclera — so you can't lift the iris out and fill with white (that made a pale patch + a
    # wandering dot, i.e. the eye's "style" changing every frame). Instead the gaze cue is the PUPIL +
    # CATCHLIGHT shifting inside the brown iris. So: flatten each iris core to smooth brown in eyes_open
    # (erasing the baked pupil/catchlight, keeping the dark eyeliner), and DRAW one clean, IDENTICAL
    # pupil+catchlight per eye on the pupils layer that gaze can move a couple px. Symmetric → no drift.
    # The eye must survive the TUI: at ~30-40 cols in a 6-level sepia palette, a dark brown iris + dark
    # pupil + eyeliner all collapse to the darkest level → a solid black blob. So the flat iris is made a
    # LIGHT warm tan (lands on a mid sepia, not black), with a DARK pupil and a GENEROUS bright catchlight
    # — three tones that stay distinct when posterized small. High contrast + simplification, per the brief.
    # Structure the eye like a proper anime eye so it reads at every size: a LIGHT sclera that stays put
    # (eyes_open) + a BROWN IRIS unit (iris ring + dark pupil + bright catchlight) that GAZE moves inside
    # it (pupils layer, clipped to the eye). The light sclera gives the TUI its contrast (it maps to the
    # lightest sepia, the pupil to the darkest); the brown iris keeps the reference look at full size.
    eyes_open = eyes.copy()
    eop = eyes_open.load()
    pupils = Image.new("RGBA", (native, nh), (0, 0, 0, 0))
    pdraw = ImageDraw.Draw(pupils)
    ir = max(5, native // 15)                                # iris sampling / sclera-fill radius
    irad = max(3, native // 26)                             # brown iris radius
    prad = max(2, native // 52)                             # dark pupil radius
    for (cx, cy) in lens:
        cols = []                                           # iris hue = median of the mid-tone iris band
        for y in range(cy - ir, cy + ir + 1):
            for x in range(cx - ir, cx + ir + 1):
                if 0 <= x < native and 0 <= y < nh and (x - cx) ** 2 + ((y - cy) * 1.1) ** 2 <= ir * ir:
                    r, g, b, aa = eyes.getpixel((x, y))
                    lm = 0.3 * r + 0.6 * g + 0.1 * b
                    if aa > 120 and 40 < lm < 175:          # brown iris (skip near-black liner + light corner)
                        cols.append((r, g, b))
        base_iris = tuple(int(statistics.median(c[i] for c in cols)) for i in range(3)) if cols else (120, 82, 66)
        sclera = tuple(min(255, 210 + int(c * 0.3)) for c in base_iris)   # near-white warm sclera → lightest sepia
        core = ir * 0.82
        # HALF-LIDDED "dead-fish" archetype: the eye only OPENS below the lid line; above it is a heavy
        # dark upper lid, so she reads sleepy/half-closed instead of wide-eyed.
        lid_y = cy - core * 0.15
        for y in range(cy - ir, cy + ir + 1):
            for x in range(cx - ir, cx + ir + 1):
                if 0 <= x < native and 0 <= y < nh and (x - cx) ** 2 + ((y - cy) * 1.1) ** 2 <= core * core:
                    r, g, b, aa = eyes.getpixel((x, y))
                    if aa > 120 and (0.3 * r + 0.6 * g + 0.1 * b) > 30:   # keep the dark eyeliner
                        eop[x, y] = (*sclera, 255) if y >= lid_y else dark    # sclera below the lid, dark lid above
        pupc = tuple(min(255, int(c * 0.5) + 22) for c in base_iris)     # SOFT dark-brown pupil that BLENDS
        #                                                                  into the iris — not a hard black dot
        icy = cy + max(1, irad // 2)                        # the iris sits LOW in the opening
        pdraw.ellipse([cx - irad, icy - irad, cx + irad, icy + irad], fill=(*base_iris, 255))        # MATTE brown iris
        pdraw.ellipse([cx - prad, icy - prad, cx + prad, icy + prad], fill=(*pupc, 255))             # dark pupil — NO catchlight (unreflective dead-fish eye)
        pdraw.rectangle([cx - irad - 2, 0, cx + irad + 2, int(lid_y)], fill=(0, 0, 0, 0))            # cut the iris top flat under the lid
    _blank = Image.new("RGBA", (native, nh), (0, 0, 0, 0))

    # mouth: the supplied 'mouth' part is only a COLOUR swatch (a brown block) → use it for POSITION only,
    # and DRAW proper lips. A dedicated moderate 'talk' state makes the lip-flap subtle (closed↔talk),
    # not the awful closed↔brown-rectangle it was.
    ma = np.asarray(mouth.split()[-1])
    mys, mxs = np.where(ma > 60)
    mcx, mcy = (int(mxs.mean()), int(mys.mean())) if len(mxs) else (native // 2, nh * 72 // 100)
    mw = max(4, native // 30)
    DK, INN, LO = (74, 40, 36, 255), (96, 52, 48, 255), (190, 122, 110, 255)

    def mouth_sprite(state):
        im = Image.new("RGBA", (native, nh), (0, 0, 0, 0))
        d = ImageDraw.Draw(im)
        x, y = mcx, mcy
        if state in ("neutral", "closed"):                  # a small BOLD closed mouth (a thin 1px line
            d.rectangle([x - mw + 1, y, x + mw - 1, y + 1], fill=DK)   # vanishes when shrunk to the TUI box)
            d.line([(x - mw + 2, y + 2), (x + mw - 2, y + 2)], fill=(*LO[:3], 170), width=1)
        elif state == "talk":                               # small open — the lip-flap frame
            d.ellipse([x - mw + 1, y - 1, x + mw - 1, y + mw], fill=INN, outline=DK)
            d.chord([x - mw + 2, y + 1, x + mw - 2, y + mw], 0, 180, fill=LO)
        elif state == "open":                               # wider open (surprised)
            d.ellipse([x - mw, y - 2, x + mw, y + mw + 2], fill=INN, outline=DK)
            d.chord([x - mw + 1, y + 2, x + mw - 1, y + mw + 2], 0, 180, fill=LO)
        elif state == "smile":
            d.arc([x - mw - 1, y - mw + 1, x + mw + 1, y + 3], 20, 160, fill=DK, width=2)
        elif state == "frown":
            d.arc([x - mw - 1, y + 2, x + mw + 1, y + mw + 3], 200, 340, fill=DK, width=2)
        return im

    fb = [max(0, lens[0][0] - native // 4), max(0, min(l[1] for l in lens) - nh // 4),
          min(native, lens[1][0] + native // 4), min(nh, mcy + nh // 5)]
    a = Avatar(name=name, kind="sprites", size=(native, nh),
               meta={"rigged": True, "from_parts": True, "face_box": fb,
                     "gaze_px": max(2, native // 55)})    # how far the irises may travel inside the eye
    a.add_layer(Layer(id="base", part="skin", protected=True, z=2,
                      states={"base": save(base, "base.png")}, default_state="base"))
    # eyes = the whites/eyeliner/glasses with the iris LIFTED OUT (eyes_open); the iris rides on its own
    # 'pupils' layer above, clipped to the eyes so a glance moves it a couple px without spilling.
    a.add_layer(Layer(id="eyes", part="eyes", z=5, default_state="open",
                      states={"open": save(eyes_open, "eyes_open.png"), "closed": save(eyes_closed, "eyes_closed.png")}))
    a.add_layer(Layer(id="pupils", part="pupils", z=6, clip="eyes", default_state="on",
                      states={"on": save(pupils, "pupils.png"), "off": save(_blank, "pupils_off.png")}))
    a.add_layer(Layer(id="bangs", part="hair", protected=True, z=7,
                      states={"base": save(bangs, "bangs.png")}, default_state="base"))
    a.add_layer(Layer(id="mouth", part="mouth", z=8, default_state="neutral", states={
        s: save(mouth_sprite(s), f"mouth_{s}.png") for s in ("neutral", "closed", "talk", "smile", "open", "frown")}))
    a.add_layer(Layer(id="body", part="clothes_front", protected=True, z=9,
                      states={"base": save(body, "body.png")}, default_state="base"))
    if headphones is not None:                                       # over the hair, at the ears
        a.add_layer(Layer(id="headphones", part="accessory", protected=True, z=10,
                          states={"base": save(headphones, "headphones.png")}, default_state="base"))
    expr = {"neutral": ("open", "neutral"), "happy": ("open", "smile"), "laughing": ("closed", "open"),
            "surprised": ("open", "open"), "sad": ("open", "frown"), "angry": ("open", "frown"),
            "love": ("closed", "smile"), "curious": ("open", "neutral")}
    for e, (ey, m) in expr.items():                          # iris hidden when the eyes are shut
        a.set_expression(e, {"eyes": ey, "pupils": "off" if ey == "closed" else "on", "mouth": m})
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


def _seed_from_default(active: str):
    """Seed the active dir from the repo's VENDORED default avatar (crucible/avatars/default — the real
    cute-anime companion, checked into the repo so a fresh clone HAS her). Copies the sprites in and writes
    an avatar.json whose state paths point at the now-populated active dir. Returns None if not vendored."""
    import json
    import shutil
    default = os.path.join(os.path.dirname(__file__), "avatars", "default")
    dspec = os.path.join(default, "avatar.json")
    if not os.path.exists(dspec):
        return None
    os.makedirs(active, exist_ok=True)
    for f in os.listdir(default):
        if f.endswith(".png"):
            shutil.copy2(os.path.join(default, f), os.path.join(active, f))
    spec = json.loads(open(dspec).read())
    for lyr in spec.get("layers", []):                      # relative filenames → absolute active-dir paths
        st = lyr.get("states") or {}
        for k, v in list(st.items()):
            if isinstance(v, str):
                st[k] = os.path.join(active, os.path.basename(v))
    with open(os.path.join(active, "avatar.json"), "w") as fh:
        json.dump(spec, fh, indent=2)
    try:
        return Avatar.load(os.path.join(active, "avatar.json"))
    except (OSError, ValueError):
        return None


def ensure_default_avatar(data_dir: str) -> Avatar:
    """Load the active avatar from <data_dir>/avatars/active. On first run SEED it from the vendored default
    companion (checked into the repo) so a fresh clone shows the real avatar; only if that's missing do we
    fall back to generating a bare procedural one."""
    active = os.path.join(data_dir, "avatars", "active")
    spec = os.path.join(active, "avatar.json")
    if os.path.exists(spec):
        try:
            return Avatar.load(spec)
        except (OSError, ValueError):
            pass
    seeded = _seed_from_default(active)
    if seeded is not None:
        return seeded
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
