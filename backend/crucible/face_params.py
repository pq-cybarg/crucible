"""Parametric facial expressions — blend in PARAMETER space, then MORPH the
features, instead of alpha-cross-fading whole rendered faces.

Each named expression is a point in a small continuous parameter space (mouth
openness/curve/width, eye openness, brow, blush…). A mood MIX is combined in
that space — a weighted base plus NONLINEAR interaction rules (emotions don't
just average: happy+surprised reads as *excited* with more than the mean mouth
opening; happy+sad reads as a tense *bittersweet*, not a flat neutral). The
resulting params drive procedural drawing of the mouth (and eyelids), so the
in-between of a smile and a frown is a genuinely re-shaped mouth, not a ghost of
both. Matches the avatar's simple procedural feature style.
"""
from __future__ import annotations

# continuous facial parameters (all default 0 except where noted)
DEFAULTS = {
    "mouth_open": 0.0,     # 0 shut … 1 wide
    "mouth_curve": 0.0,    # -1 frown … +1 smile
    "mouth_width": 1.0,    # 0.6 pursed … 1.3 wide
    "eye_open": 1.0,       # 0 shut … 1.2 wide
    "eye_happy": 0.0,      # 0 normal … 1 happy ^ arc
    "brow": 0.0,           # -1 furrowed … +1 raised
    "blush": 0.0,          # 0 … 1
}

# named expression → parameter targets (only the non-default keys)
EXPRESSION_PARAMS = {
    "neutral":   {},
    "happy":     {"mouth_curve": 0.7, "mouth_open": 0.12, "eye_open": 0.85},
    "laughing":  {"mouth_open": 0.75, "mouth_curve": 0.5, "eye_open": 0.34, "eye_happy": 1.0},
    "surprised": {"mouth_open": 0.7, "mouth_width": 0.8, "eye_open": 1.18, "brow": 0.9},
    "sad":       {"mouth_curve": -0.6, "eye_open": 0.82, "brow": -0.4},
    "angry":     {"mouth_curve": -0.35, "mouth_open": 0.1, "eye_open": 0.95, "brow": -0.9},
    "love":      {"mouth_curve": 0.8, "mouth_open": 0.15, "eye_open": 0.4, "eye_happy": 1.0, "blush": 1.0},
    "curious":   {"mouth_curve": 0.15, "eye_open": 1.05, "brow": 0.4},
    "smug":      {"mouth_curve": 0.4, "mouth_width": 0.8, "eye_open": 0.8, "brow": -0.2},
    "shy":       {"mouth_curve": 0.2, "mouth_open": 0.05, "eye_open": 0.75, "blush": 0.9},
    "teasing":   {"mouth_curve": 0.5, "mouth_open": 0.2, "eye_open": 0.7, "eye_happy": 0.4},
    "scared":    {"mouth_open": 0.4, "mouth_curve": -0.3, "eye_open": 1.1, "brow": 0.3},
    "talk":      {"mouth_open": 0.45},
}


def _clamp(v, lo, hi):
    return max(lo, min(hi, v))


def blend_params(weights: dict) -> dict:
    """Weighted blend of expression params + nonlinear emotion interactions."""
    items = [(n, float(w)) for n, w in (weights or {}).items() if w and w > 0]
    if not items:
        return dict(DEFAULTS)
    tot = sum(w for _, w in items)
    wn = {n: w / tot for n, w in items}                       # normalized weights

    p = dict(DEFAULTS)
    for key in DEFAULTS:
        p[key] = sum(wn[n] * EXPRESSION_PARAMS.get(n, {}).get(key, DEFAULTS[key]) for n in wn)

    # ---- NONLINEAR interaction rules (super-/sub-additive emotion combos) ----
    def g(name):
        return wn.get(name, 0.0)

    # EXCITED: happy + surprised → delighted, MORE open than the linear mean
    exc = min(g("happy"), g("surprised")) + min(g("laughing"), g("surprised"))
    if exc > 0:
        p["mouth_open"] = min(1.0, p["mouth_open"] + 0.5 * exc)
        p["eye_open"] = min(1.3, p["eye_open"] + 0.25 * exc)
        p["mouth_curve"] = min(1.0, p["mouth_curve"] + 0.3 * exc)

    # BITTERSWEET: happy + sad conflict → the smile doesn't fully commit (tense),
    # NOT a flat average-to-neutral. Pull the curve toward a faint, uneasy smile.
    conf = min(g("happy") + g("love"), g("sad"))
    if conf > 0:
        p["mouth_curve"] = p["mouth_curve"] * (1 - 0.5 * conf) + 0.12 * conf
        p["mouth_open"] *= (1 - 0.4 * conf)
        p["eye_open"] = min(p["eye_open"], 0.85)              # slightly downcast
        p["blush"] *= (1 - 0.5 * conf)

    # FURIOUS: anger with anything → sharper furrow, tighter mouth
    ang = g("angry")
    if ang > 0.2:
        p["brow"] = min(p["brow"], -0.5 * ang) - 0.3 * ang
        p["mouth_curve"] = min(p["mouth_curve"], -0.2 * ang)

    p["mouth_open"] = _clamp(p["mouth_open"], 0.0, 1.0)
    p["mouth_curve"] = _clamp(p["mouth_curve"], -1.0, 1.0)
    p["eye_open"] = _clamp(p["eye_open"], 0.0, 1.3)
    p["eye_happy"] = _clamp(p["eye_happy"], 0.0, 1.0)
    p["brow"] = _clamp(p["brow"], -1.2, 1.2)
    p["blush"] = _clamp(p["blush"], 0.0, 1.0)
    return p


def draw_mouth(draw, cx: int, cy: int, p: dict, s: float = 1.0,
               lips: bool = True, inside: bool = True):
    """Draw a MORPHING mouth from params (continuous frown↔smile, shut↔open) in the
    avatar's simple-lip style. `s` scales for the sprite size (native ~1.0). `lips`/
    `inside` toggle the lip outline / inner-mouth fill (part-hierarchy toggles)."""
    w = 7.0 * s * p.get("mouth_width", 1.0)
    op = _clamp(p.get("mouth_open", 0.0), 0.0, 1.0)
    cv = _clamp(p.get("mouth_curve", 0.0), -1.0, 1.0)
    DK, INN, LO = (74, 40, 36, 255), (96, 52, 48, 255), (188, 120, 110, 255)
    lw = max(1, int(2 * s))

    # ONE continuous shape (no open/closed branch → no pop when talking/blending), densely sampled so it
    # reads as a smooth curve. KEY: the smile's centre-dip (the ‿) only applies to a CLOSED mouth; as the
    # mouth OPENS the upper lip FLATTENS, so the smile comes from the lifted corners + the lower lip — an
    # open smile is NOT a downward-pointing heart/V.
    N = 15
    tsx = [(-1.0 + 2.0 * i / (N - 1)) for i in range(N)]
    corner_y = cy - cv * 2.5 * s                              # mouth corners lift on a smile
    center_dip = cv * 5.0 * s * max(0.0, 1.0 - op * 1.3)      # ‿ depth: full when shut → 0 when open
    upper = [(cx + t * w, corner_y + center_dip * (1 - t * t)) for t in tsx]
    gap = op * 10.0 * s                                       # opening (0 → shut, grows smoothly)
    lower = [(x, y + gap * (0.5 + 0.5 * (1 - t * t))) for t, (x, y) in zip(tsx, upper)]

    if gap > 1.4 * s:                                         # visibly open
        if inside:                                           # inner mouth (cavity / tongue+teeth stand-in)
            draw.polygon(upper + lower[::-1], fill=INN)
            draw.line([(x, y - 1.2 * s) for x, y in lower[2:-2]], fill=LO, width=1)   # lower-lip sheen
        if lips:
            draw.line(lower, fill=DK, width=1, joint="curve")
    if lips:
        draw.line(upper, fill=DK, width=lw, joint="curve")   # the lip line
        if gap <= 1.4 * s and cv > 0.4:                      # lower-lip hint on a closed big smile
            draw.line([(x, y + 1.6 * s) for x, y in upper[3:-3]], fill=LO, width=1)


def draw_eyes(img, centers, p: dict, blink: float = 0.0, glasses=None,
              skin=(219, 179, 147, 255), half_w: int = 15, top_h: int = 19, bot_h: int = 11):
    """CONTINUOUS eye close by DEFORMING the real eye art. Each eye box (whose frame + lashes have
    been separated into their own layers, pre-composited by the caller) is squashed vertically
    toward the lower lid by `eye_open` — the eyeball + the pre-composited lashes compress down as
    one — and the RIGID glasses (`glasses`, same-size RGBA) are re-composited on top AFTER, so they
    never deform. `blink` (0..1) closes on top. `centers` = [(cx,cy), …]."""
    from PIL import Image, ImageDraw
    eo = _clamp(p.get("eye_open", 1.0) * (1.0 - _clamp(blink, 0.0, 1.0)), 0.06, 1.15)
    if eo < 1.0:
        d = ImageDraw.Draw(img, "RGBA")
        for (cx, cy) in centers:
            x0, y0, x1, y1 = cx - half_w, cy - top_h, cx + half_w, cy + bot_h
            bw, bh = x1 - x0, y1 - y0
            region = img.crop((x0, y0, x1, y1))
            nh = max(2, round(bh * eo))
            squ = region.resize((bw, nh), Image.BILINEAR)     # squash the eye toward the lower lid
            d.rectangle([x0, y0, x1 - 1, y1 - 1], fill=skin)  # clear to lid skin
            img.alpha_composite(squ, (x0, y1 - nh))           # re-seat, anchored at the bottom lid
    if glasses is not None:
        img.alpha_composite(glasses)                          # rigid real-art frames back on top


def draw_brows(draw, centers, p: dict, base_y: int = 101, color=(60, 46, 44, 255)):
    """Parametric eyebrows (the baked brows were stripped from the eye sprite, so
    these are the ONLY brows and are drawn always). brow=0 is a relaxed near-flat
    brow; `brow`>0 raises/arches (surprised), `brow`<0 furrows + drops the inner
    ends (angry). Drawn in the BELOW band so the bangs overlap it like the art."""
    brow = _clamp(p.get("brow", 0.0), -1.2, 1.2)
    by = base_y - brow * 3.0
    for (cx, cy) in centers:
        toward = 1 if cx < 100 else -1                       # +x points toward the nose
        inner = (cx + toward * 7, by + max(0.0, -brow) * 5.0)    # furrow drops the inner end
        mid = (cx + toward * 1, by - 1.0 - max(0.0, brow) * 1.5)  # slight relaxed arch at rest
        outer = (cx - toward * 9, by + 0.5 - max(0.0, brow) * 2.5)  # raise lifts the outer end
        draw.line([inner, mid, outer], fill=color, width=2, joint="curve")


def draw_blush(img, cheeks, p: dict, color=(232, 120, 120)):
    """Soft pink cheeks by the `blush` param (0..1). `cheeks` = [(cx,cy), …]."""
    from PIL import Image, ImageDraw
    b = _clamp(p.get("blush", 0.0), 0.0, 1.0)
    if b <= 0.03:
        return
    layer = Image.new("RGBA", img.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(layer, "RGBA")
    a = int(150 * b)
    for (cx, cy) in cheeks:
        d.ellipse([cx - 9, cy - 4, cx + 9, cy + 4], fill=(*color, a))
        d.ellipse([cx - 6, cy - 3, cx + 6, cy + 3], fill=(*color, min(255, a + 40)))
    img.alpha_composite(layer)
