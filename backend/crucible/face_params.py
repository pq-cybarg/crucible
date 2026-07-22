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
    "cat": 0.0,            # 0 human … 1 CAT MODE (ω mouth + cat ears; pairs with eye_shape "cat")
}

# named expression → parameter targets (only the non-default keys)
EXPRESSION_PARAMS = {
    "neutral":   {},
    "happy":     {"mouth_curve": 0.7, "mouth_open": 0.12, "eye_open": 0.85},
    "laughing":  {"mouth_open": 0.75, "mouth_curve": 0.5, "eye_open": 0.34, "eye_happy": 1.0},
    "surprised": {"mouth_open": 0.7, "mouth_width": 0.8, "eye_open": 1.18, "brow": 0.9},
    "sad":       {"mouth_curve": -0.6, "eye_open": 0.82, "brow": -0.4},
    "angry":     {"mouth_curve": -0.35, "mouth_open": 0.1, "eye_open": 0.95, "brow": -0.9},
    # love softened: a gentle happy squint, NOT always fully shut (eye_open 0.55 / eye_happy 0.85) — mixed
    # moods open it further; heart-eyes are a SEPARATE transient effect ("lovestruck"), not baked in here.
    "love":      {"mouth_curve": 0.8, "mouth_open": 0.15, "eye_open": 0.55, "eye_happy": 0.45, "blush": 1.0},
    "curious":   {"mouth_curve": 0.15, "eye_open": 1.05, "brow": 0.4},
    "smug":      {"mouth_curve": 0.4, "mouth_width": 0.8, "eye_open": 0.8, "brow": -0.2, "eye_shape": "cat"},
    "shy":       {"mouth_curve": 0.2, "mouth_open": 0.05, "eye_open": 0.75, "blush": 0.9},
    "teasing":   {"mouth_curve": 0.5, "mouth_open": 0.2, "eye_open": 0.7, "eye_happy": 0.4, "eye_shape": "cat"},
    "scared":    {"mouth_open": 0.4, "mouth_curve": -0.3, "eye_open": 1.1, "brow": 0.3},
    "talk":      {"mouth_open": 0.45},
    # EFFECT moods — special eye shapes for intense / temporary beats; blend one in for the moment.
    "lovestruck": {"mouth_curve": 0.85, "mouth_open": 0.2, "eye_open": 1.05, "blush": 1.0, "eye_shape": "heart"},
    "starstruck": {"mouth_curve": 0.6, "mouth_open": 0.45, "eye_open": 1.1, "brow": 0.6, "eye_shape": "star_bloom"},
    "dizzy":      {"mouth_curve": -0.1, "mouth_open": 0.2, "eye_open": 1.0, "eye_shape": "swirl"},
    "mesmerized": {"mouth_curve": 0.1, "mouth_open": 0.15, "eye_open": 1.0, "eye_shape": "concentric"},
    "sparkly":    {"mouth_curve": 0.75, "mouth_open": 0.3, "eye_open": 1.15, "brow": 0.4, "blush": 0.5, "eye_shape": "sparkle"},
    "ko":         {"mouth_curve": -0.1, "mouth_open": 0.28, "eye_open": 1.0, "eye_shape": "x_eyes"},
    "fired_up":   {"mouth_curve": 0.35, "mouth_open": 0.4, "eye_open": 1.1, "brow": -0.3, "eye_shape": "flame"},
    "greedy":     {"mouth_curve": 0.5, "mouth_width": 0.9, "eye_open": 1.05, "eye_shape": "money"},
    "shock":      {"mouth_open": 0.5, "mouth_width": 0.7, "eye_open": 1.2, "brow": 0.7, "eye_shape": "dots"},
    "crying":     {"mouth_curve": -0.5, "mouth_open": 0.25, "eye_open": 0.95, "brow": -0.2, "eye_shape": "tears"},
    # CAT MODE — slit eyes + ω mouth + cat ears. Variants for cat emotions.
    "cat":        {"mouth_curve": 0.35, "eye_open": 1.0, "eye_shape": "cat", "cat": 1.0},
    "cat_meow":   {"mouth_open": 0.6, "mouth_curve": 0.2, "eye_open": 0.9, "eye_shape": "cat", "cat": 1.0},
    "cat_smug":   {"mouth_curve": 0.55, "mouth_width": 0.9, "eye_open": 0.65, "eye_happy": 0.3, "brow": -0.2, "eye_shape": "cat", "cat": 1.0},
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
    p["cat"] = _clamp(p["cat"], 0.0, 1.0)

    # ---- EYE-SHAPE (categorical): the strongest active mood that declares an `eye_shape` wins; its weight
    # is the INTENSITY, so special eyes (heart/star/cat/…) only show when that mood is strong, not always.
    shape, samt = "", 0.0
    for n in wn:
        sh = EXPRESSION_PARAMS.get(n, {}).get("eye_shape")
        if sh and wn[n] > samt:
            shape, samt = sh, wn[n]
    p["eye_shape"] = shape
    p["eye_shape_amt"] = round(samt, 3)
    return p


def _draw_cat_mouth(draw, cx, cy, p, s=1.0):
    """The cat ω / :3 mouth — two humps meeting at a centre peak; opens into a meow with a little tongue."""
    op = _clamp(p.get("mouth_open", 0.0), 0.0, 1.0)
    cv = _clamp(p.get("mouth_curve", 0.0), -1.0, 1.0)
    DK, TONGUE, INN = (74, 40, 36, 255), (214, 116, 120, 255), (120, 54, 56, 255)
    w = 6.0 * s
    base = cy - cv * 1.5 * s                                # a smile lifts the whole ω
    # the ω outline: corner → dip → CENTRE PEAK → dip → corner (a rounded W = the cat mouth)
    omega = [(cx - w, base - 0.5 * s), (cx - w * 0.5, base + 3.0 * s), (cx, base - 2.6 * s),
             (cx + w * 0.5, base + 3.0 * s), (cx + w, base - 0.5 * s)]
    if op > 0.22:                                           # MEOW — open mouth under the ω
        mh = op * 8.0 * s
        draw.polygon([(cx - w * 0.5, base + 1.0 * s), (cx + w * 0.5, base + 1.0 * s),
                      (cx + w * 0.32, base + mh), (cx - w * 0.32, base + mh)], fill=INN)
        draw.ellipse([cx - w * 0.3, base + mh * 0.45, cx + w * 0.3, base + mh * 1.05], fill=TONGUE)
    draw.line(omega, fill=DK, width=max(1, int(2 * s)), joint="curve")


def draw_mouth(draw, cx: int, cy: int, p: dict, s: float = 1.0,
               lips: bool = True, inside: bool = True, teeth: bool = True, tongue: bool = True):
    """Draw a MORPHING mouth from params (continuous frown↔smile, shut↔open) in the
    avatar's simple-lip style. `s` scales for the sprite size (native ~1.0). `lips`/`inside`
    toggle the lip outline / inner-mouth cavity; `teeth`/`tongue` add a subtle upper-teeth band
    and a soft tongue that only appear once the mouth is CLEARLY open (part-hierarchy toggles)."""
    if _clamp(p.get("cat", 0.0), 0.0, 1.0) > 0.5:          # CAT MODE → the ω / :3 cat mouth
        _draw_cat_mouth(draw, cx, cy, p, s)
        return
    w = 7.0 * s * p.get("mouth_width", 1.0)
    op = _clamp(p.get("mouth_open", 0.0), 0.0, 1.0)
    cv = _clamp(p.get("mouth_curve", 0.0), -1.0, 1.0)
    DK, INN, LO = (74, 40, 36, 255), (96, 52, 48, 255), (188, 120, 110, 255)
    TEETH, TONGUE = (240, 234, 228, 255), (206, 116, 116, 255)
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
        if inside:                                           # inner mouth cavity
            draw.polygon(upper + lower[::-1], fill=INN)
            # TONGUE: a soft mound rising from the lower lip, only when clearly open
            if tongue and gap > 4.0 * s:
                base = lower[3:-3]
                rise = min(0.6, 0.28 + 0.4 * op)
                top = [(x, y - gap * rise) for x, y in base]
                draw.polygon(top + base[::-1], fill=TONGUE)
            # UPPER TEETH: a thin bright band tucked under the upper lip, only when clearly open
            if teeth and gap > 3.2 * s:
                band = [(x, y + 1.1 * s) for x, y in upper[2:-2]]
                draw.line(band, fill=TEETH, width=max(1, int(1.6 * s)), joint="curve")
            draw.line([(x, y - 1.2 * s) for x, y in lower[2:-2]], fill=LO, width=1)   # lower-lip sheen
        if lips:
            draw.line(lower, fill=DK, width=1, joint="curve")
    if lips:
        draw.line(upper, fill=DK, width=lw, joint="curve")   # the lip line
        if gap <= 1.4 * s and cv > 0.4:                      # lower-lip hint on a closed big smile
            draw.line([(x, y + 1.6 * s) for x, y in upper[3:-3]], fill=LO, width=1)


def draw_eyes(img, centers, p: dict, blink: float = 0.0, glasses=None, lashes=None,
              skin=(219, 179, 147, 255), half_w: int = 15, top_h: int = 19, bot_h: int = 11):
    """Eye close = the upper EYELID coming DOWN over the eye (a real blink), NOT the eyeball squishing.
    The full-size iris/whites stay put; a local-skin LID descends from the eye-top to its lower edge and
    OCCLUDES the eye top-down, and the separate `lashes` (same-size RGBA, passed in — not pre-composited)
    ride that lower edge (translated down). `eye_open`/`blink` set how far the lid is down; the RIGID
    `glasses` re-composite on top last. `centers` = [(cx,cy), …]."""
    from PIL import Image, ImageDraw
    eo = _clamp(p.get("eye_open", 1.0) * (1.0 - _clamp(blink, 0.0, 1.0)), 0.06, 1.15)
    blink_c = _clamp(blink, 0.0, 1.0)
    _shape = p.get("eye_shape", "")
    _samt = _clamp(p.get("eye_shape_amt", 0.0), 0.0, 1.0)
    # MORPH (continuous): a special shape forms OUT of the eye and back as its mood eases in/out; it fades
    # with the closing lids (open_f) so a blink shows the normal lid close, then it re-forms on reopen.
    morph = _clamp((_samt - 0.3) * 1.8, 0.0, 1.0) if _shape else 0.0
    open_f = _clamp((eo - 0.3) / 0.6, 0.0, 1.0)
    shape_alpha = morph * open_f
    show_shape = shape_alpha > 0.02
    eop = _clamp(p.get("eye_open", 1.0), 0.0, 1.3)
    eh = _clamp(p.get("eye_happy", 0.0), 0.0, 1.0)
    arc = eh * _clamp((1.0 - eop) * 1.5, 0.0, 1.0)

    # BLINK / CLOSE — SQUASH the eye toward its CENTRE: the whole eye (iris + the pre-composited lashes +
    # the bordering lash/lid lines) compresses vertically as it closes, so its outline SQUEEZES together
    # (the SIDES compress too) rather than the lid sliding down over a static eye. Centred (not bottom-
    # anchored) so it doesn't read as the eye sliding downward. `close`: 0 open … 1 shut.
    close = _clamp(1.0 - eo, 0.0, 1.0)
    if not show_shape and arc <= 0.5 and close > 0.02:
        d = ImageDraw.Draw(img, "RGBA")
        H = img.height
        for (cx, cy) in centers:
            x0, y0, x1, y1 = cx - half_w, cy - top_h, cx + half_w, cy + bot_h
            nh = max(2, round((y1 - y0) * eo))
            squ = img.crop((x0, y0, x1, y1)).resize((x1 - x0, nh), Image.BILINEAR)
            lid = img.getpixel((cx, min(H - 1, cy + bot_h + 5)))   # local cheek skin so the cleared lid matches
            if not (isinstance(lid, tuple) and len(lid) == 4 and lid[3] > 200):
                lid = skin
            d.rectangle([x0, y0, x1 - 1, y1 - 1], fill=lid)
            img.alpha_composite(squ, (x0, (y0 + y1) // 2 - nh // 2))   # re-seat CENTRED → squeezes in place

    # HAPPY ^ ARC: a STRONG happy squint (laughing) closes into an upward ^_^ rather than a flat lid.
    if not show_shape and arc > 0.5 and blink_c < 0.3:
        d = ImageDraw.Draw(img, "RGBA")
        aw, peak = 13, 8.0
        for (cx, cy) in centers:
            base_y = cy + 4
            n = 15
            pts = [(cx + (-1 + 2 * i / (n - 1)) * aw,
                    base_y - peak * (1 - (-1 + 2 * i / (n - 1)) ** 2)) for i in range(n)]
            lid = img.getpixel((cx, max(0, cy - top_h - 2)))
            if len(lid) == 4 and lid[3] < 40:
                lid = skin
            d.rectangle([cx - half_w, cy - top_h, cx + half_w, cy + bot_h + 1], fill=lid)   # hide the eyeball
            d.line(pts, fill=(52, 40, 40, 255), width=3, joint="curve")
            d.line([(x, y + 1.6) for x, y in pts[3:-3]], fill=(72, 55, 53, 255), width=1)

    if show_shape:
        from crucible.eye_shapes import SHAPES
        fn = SHAPES.get(_shape)
        if fn is not None:
            # Draw the shape on its OWN layer and fade it in by `morph`, then composite. The CALLER fades the
            # real round iris/pupil OUT by the same morph, so the shape crossfades from the eye ON the real
            # sclera — NO white erase ellipse (that used to paint over the skin / eye outlines / lids).
            sl = Image.new("RGBA", img.size, (0, 0, 0, 0))
            r = 8.0 * (0.62 + 0.38 * morph)                   # size grows as it FORMS (morph)…
            for (cx, cy) in centers:
                fn(sl, cx, cy, r, 1.0)
            sl.putalpha(sl.split()[-1].point(lambda v: int(v * shape_alpha)))   # …opacity fades with the lids
            img.alpha_composite(sl)

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
