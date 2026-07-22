"""A library of special EYE-SHAPE overlays for the companion — heart / star / cat-slit / swirl /
concentric / Oshi-no-Ko star-with-bloom — drawn as pixel-art over the open eye's iris+pupil.

Each `draw_*` takes the target IMAGE (RGBA), an iris centre, an iris radius `r`, and an intensity
`amt` in [0,1] (used to fade a shape in). Shapes replace the round iris/pupil while the eye is open;
the caller (draw_eyes) decides when to invoke them.

HARD RULE (user): any glow/bloom must stay as PIXELATED as the character — never a smooth gaussian
halo. `_pixelate()` blocks a soft layer down to the sprite's pixel grid before compositing, so bloom
reads as chunky pixel-art light, matching the model.
"""
from __future__ import annotations
import math

from PIL import Image, ImageDraw, ImageFilter

IRIS = (111, 87, 77, 255)          # the deadpan brown iris
PUPIL = (58, 44, 44, 255)
HILITE = (236, 214, 206, 230)
ROSE = (150, 84, 86, 255)          # iris-brown nudged red/pink — a FLAT, anime-friendly heart tone
ROSE_DK = (104, 52, 56, 255)       # darker rose = the reshaped heart PUPIL (so it matches the iris shape)
ROSE_HI = (210, 150, 148, 235)
PINK = (214, 96, 112, 255)         # louder pink (heart_pink variant)
PINK_HI = (255, 232, 236, 235)
GOLD = (240, 206, 90, 255)
BLOOM = (150, 210, 255, 255)       # Oshi-no-Ko cool star bloom


def _pixelate(layer: Image.Image, block: int = 3) -> Image.Image:
    """Blockify a soft layer to the sprite pixel grid so glow stays pixel-art (rule: bloom must be as
    pixelated as the model)."""
    w, h = layer.size
    small = layer.resize((max(1, w // block), max(1, h // block)), Image.BILINEAR)
    return small.resize((w, h), Image.NEAREST)


def _heart_poly(cx, cy, r):
    pts = []
    for i in range(24):                                   # parametric heart, sampled + scaled to r
        t = math.pi * (i / 23) * 2 - math.pi
        x = 16 * math.sin(t) ** 3
        y = -(13 * math.cos(t) - 5 * math.cos(2 * t) - 2 * math.cos(3 * t) - math.cos(4 * t))
        pts.append((cx + x / 16 * r, cy + y / 16 * r * 0.92 + r * 0.05))
    return pts


def _star_poly(cx, cy, r, points=5, inner=0.44, rot=-math.pi / 2):
    pts = []
    for i in range(points * 2):
        ang = rot + i * math.pi / points
        rr = r if i % 2 == 0 else r * inner
        pts.append((cx + rr * math.cos(ang), cy + rr * math.sin(ang)))
    return pts


def draw_cat(img, cx, cy, r, amt=1.0):
    d = ImageDraw.Draw(img, "RGBA")
    d.ellipse([cx - r, cy - r * 0.86, cx + r, cy + r * 0.86], fill=IRIS)
    sw = max(1.2, r * 0.22)
    d.ellipse([cx - sw, cy - r * 0.92, cx + sw, cy + r * 0.92], fill=PUPIL)   # vertical slit
    d.ellipse([cx - r * 0.5, cy - r * 0.55, cx - r * 0.5 + 2, cy - r * 0.55 + 2], fill=HILITE)


def _heart_fill(d, cx, cy, r, col):
    """A SOLID heart (two overlapping lobe-circles + a point) — no notch gap that would show a hole."""
    lobe = r * 0.54
    ly = cy - r * 0.36                                    # lobe-centres row
    d.ellipse([cx - r * 0.92, ly - lobe, cx + r * 0.04, ly + lobe], fill=col)   # left lobe
    d.ellipse([cx - r * 0.04, ly - lobe, cx + r * 0.92, ly + lobe], fill=col)   # right lobe
    d.polygon([(cx - r * 0.9, ly - r * 0.05), (cx + r * 0.9, ly - r * 0.05),
               (cx, cy + r * 0.7)], fill=col)                                   # bottom point


def draw_heart(img, cx, cy, r, amt=1.0, pink=False):
    # a FLAT rose (iris-brown + red/pink tint) so it stays anime, not a loud sticker pink. CONCENTRIC hearts:
    # the iris-heart, a darker heart PUPIL (so the pupil matches the iris shape, not a round pupil), + shine.
    d = ImageDraw.Draw(img, "RGBA")
    col = PINK if pink else ROSE
    pup = (150, 40, 60, 255) if pink else ROSE_DK
    hi = PINK_HI if pink else ROSE_HI
    _heart_fill(d, cx, cy, r, col)                        # iris heart
    _heart_fill(d, cx, cy - r * 0.02, r * 0.52, pup)      # reshaped heart PUPIL (matches the iris shape)
    d.ellipse([cx - r * 0.34, cy - r * 0.36, cx - r * 0.06, cy - r * 0.08], fill=hi)   # small catchlight


def draw_star(img, cx, cy, r, amt=1.0, col=GOLD):
    d = ImageDraw.Draw(img, "RGBA")
    d.polygon(_star_poly(cx, cy, r * 1.1), fill=col)
    d.ellipse([cx - r * 0.24, cy - r * 0.24, cx + r * 0.24, cy + r * 0.24], fill=PUPIL)
    d.point((cx - r * 0.3, cy - r * 0.3), fill=(255, 255, 255, 255))


def draw_swirl(img, cx, cy, r, amt=1.0):
    """Dizzy/confused spiral."""
    d = ImageDraw.Draw(img, "RGBA")
    d.ellipse([cx - r, cy - r * 0.86, cx + r, cy + r * 0.86], fill=(232, 226, 220, 255))
    pts = []
    for deg in range(0, 900, 18):
        a = math.radians(deg)
        rr = r * (1 - deg / 900.0) * 0.95
        pts.append((cx + rr * math.cos(a), cy + rr * math.sin(a) * 0.86))
    d.line(pts, fill=PUPIL, width=1, joint="curve")


def draw_concentric(img, cx, cy, r, amt=1.0):
    """Hypnotic concentric rings."""
    d = ImageDraw.Draw(img, "RGBA")
    for k, frac in enumerate((1.0, 0.72, 0.46, 0.22)):
        rr = r * frac
        col = PUPIL if k % 2 == 0 else (206, 158, 120, 255)
        d.ellipse([cx - rr, cy - rr * 0.86, cx + rr, cy + rr * 0.86], fill=col)


def draw_star_bloom(img, cx, cy, r, amt=1.0):
    """Oshi-no-Ko style: a 4-point light star in a bright iris, wrapped in PIXELATED bloom."""
    # 1) bloom layer — a soft blob, then blockified so it's chunky pixel light (never a smooth halo)
    glow = Image.new("RGBA", img.size, (0, 0, 0, 0))
    gd = ImageDraw.Draw(glow, "RGBA")
    gr = r * 1.8
    gd.ellipse([cx - gr, cy - gr, cx + gr, cy + gr], fill=(*BLOOM[:3], 90))
    glow = glow.filter(ImageFilter.GaussianBlur(r * 0.5))
    img.alpha_composite(_pixelate(glow, block=3))
    # 2) iris + 4-point star sparkle
    d = ImageDraw.Draw(img, "RGBA")
    d.ellipse([cx - r, cy - r * 0.86, cx + r, cy + r * 0.86], fill=(70, 96, 150, 255))
    d.polygon(_star_poly(cx, cy, r * 1.15, points=4, inner=0.3, rot=-math.pi / 2), fill=(230, 244, 255, 255))
    d.polygon(_star_poly(cx, cy, r * 0.6, points=4, inner=0.3, rot=0), fill=(255, 255, 255, 255))


def draw_sparkle(img, cx, cy, r, amt=1.0):
    """Kirakira shoujo eyes — a glossy jewel iris with big catchlights + sparkles (admiring/excited)."""
    d = ImageDraw.Draw(img, "RGBA")
    d.ellipse([cx - r, cy - r * 0.86, cx + r, cy + r * 0.86], fill=(126, 100, 158, 255))
    d.ellipse([cx - r * 0.34, cy - r * 0.08, cx + r * 0.34, cy + r * 0.52], fill=(74, 54, 106, 255))  # pupil low
    d.ellipse([cx - r * 0.54, cy - r * 0.56, cx - r * 0.08, cy - r * 0.1], fill=(255, 255, 255, 255))  # big shine
    d.ellipse([cx + r * 0.16, cy + r * 0.14, cx + r * 0.44, cy + r * 0.42], fill=(255, 255, 255, 235))  # small
    for sx, sy, sr in ((cx + r * 0.86, cy - r * 0.74, r * 0.3), (cx - r * 0.82, cy + r * 0.66, r * 0.22)):
        d.polygon(_star_poly(sx, sy, sr, points=4, inner=0.28), fill=(255, 255, 255, 255))         # sparkles


def draw_x_eyes(img, cx, cy, r, amt=1.0):
    """Comedic KO / dead-tired ✕ eyes."""
    d = ImageDraw.Draw(img, "RGBA")
    col = (52, 40, 40, 255)
    d.line([(cx - r * 0.72, cy - r * 0.72), (cx + r * 0.72, cy + r * 0.72)], fill=col, width=3, joint="curve")
    d.line([(cx - r * 0.72, cy + r * 0.72), (cx + r * 0.72, cy - r * 0.72)], fill=col, width=3, joint="curve")


def draw_flame(img, cx, cy, r, amt=1.0):
    """Fired-up / determined — a flame burning in the eye."""
    d = ImageDraw.Draw(img, "RGBA")
    d.polygon([(cx - r * 0.7, cy + r * 0.8), (cx - r * 0.82, cy - r * 0.1), (cx - r * 0.32, cy + r * 0.12),
               (cx, cy - r * 0.92), (cx + r * 0.32, cy + r * 0.12), (cx + r * 0.82, cy - r * 0.1),
               (cx + r * 0.7, cy + r * 0.8)], fill=(230, 116, 40, 255))                             # outer flame
    d.polygon([(cx - r * 0.36, cy + r * 0.7), (cx - r * 0.42, cy), (cx, cy - r * 0.5),
               (cx + r * 0.42, cy), (cx + r * 0.36, cy + r * 0.7)], fill=(246, 206, 92, 255))        # inner


def draw_money(img, cx, cy, r, amt=1.0):
    """Greedy $ eyes."""
    d = ImageDraw.Draw(img, "RGBA")
    d.ellipse([cx - r, cy - r * 0.86, cx + r, cy + r * 0.86], fill=(66, 138, 82, 255))              # green iris
    g = (238, 240, 214, 255)
    d.line([(cx, cy - r * 0.82), (cx, cy + r * 0.82)], fill=g, width=2)                             # $ vertical
    d.arc([cx - r * 0.52, cy - r * 0.66, cx + r * 0.52, cy + r * 0.06], 300, 150, fill=g, width=2)  # top S curve
    d.arc([cx - r * 0.52, cy - r * 0.06, cx + r * 0.52, cy + r * 0.66], 120, 330, fill=g, width=2)  # bottom S


def draw_dots(img, cx, cy, r, amt=1.0):
    """Shock / pinprick — tiny pupils on a wide pale eye."""
    d = ImageDraw.Draw(img, "RGBA")
    d.ellipse([cx - r, cy - r * 0.9, cx + r, cy + r * 0.9], fill=(238, 236, 233, 255))
    d.ellipse([cx - r * 0.2, cy - r * 0.2, cx + r * 0.2, cy + r * 0.2], fill=(42, 32, 32, 255))     # tiny pupil


TEAR = (196, 226, 248, 225)        # pale watery blue for the shine + streaks


def draw_tears(img, cx, cy, r, amt=1.0):
    """Crying — her NORMAL brown eye, WET: big glossy welling highlights + thin tears streaming down.
    (Keeps her eye colour; the water is the light, not a recolour.)"""
    d = ImageDraw.Draw(img, "RGBA")
    d.ellipse([cx - r, cy - r * 0.86, cx + r, cy + r * 0.86], fill=IRIS)                    # her brown iris
    d.ellipse([cx - r * 0.4, cy - r * 0.15, cx + r * 0.4, cy + r * 0.55], fill=PUPIL)       # pupil
    d.ellipse([cx - r * 0.52, cy - r * 0.55, cx + r * 0.02, cy], fill=(255, 255, 255, 245))  # BIG teary shine
    d.ellipse([cx + r * 0.2, cy + r * 0.18, cx + r * 0.5, cy + r * 0.48], fill=(255, 255, 255, 220))  # 2nd shine
    d.arc([cx - r, cy + r * 0.4, cx + r, cy + r * 1.1], 200, 340, fill=TEAR, width=2)       # water welling at lid
    for sx in (cx - r * 0.55, cx + r * 0.5):                                                # thin tears streaming
        d.line([(sx, cy + r * 0.75), (sx - r * 0.06, cy + r * 2.7)], fill=TEAR, width=2, joint="curve")
        d.ellipse([sx - r * 0.13, cy + r * 2.5, sx + r * 0.13, cy + r * 2.9], fill=TEAR)    # bead at the tip


SHAPES = {
    "cat": draw_cat,
    "heart": draw_heart,
    "heart_pink": lambda *a, **k: draw_heart(*a, pink=True, **{kk: vv for kk, vv in k.items() if kk != "pink"}),
    "star": draw_star,
    "swirl": draw_swirl,
    "concentric": draw_concentric,
    "star_bloom": draw_star_bloom,
    "sparkle": draw_sparkle,
    "x_eyes": draw_x_eyes,
    "flame": draw_flame,
    "money": draw_money,
    "dots": draw_dots,
    "tears": draw_tears,
}
