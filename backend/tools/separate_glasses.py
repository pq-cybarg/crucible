"""Occlusion-aware separation of the ROUND GLASSES frame from the eye sprites.

The baked art (`eyes_*.pre_glass.png`) entangles the frame and the eyes: the ring
is a THIN circumference, the lashes are a solid FILL inside the top of that circle,
and the sclera is the fill below. They only touch at the top, where the ring is
HIDDEN behind the lashes — so no ring pixels exist there to copy.

Method (deterministic, no hand-tuned colour thresholds beyond dark/skin):
  1. Fit a circle per eye to the DARK OUTER arc (bottom 270°, excluding the top
     lash zone) — this is the true hand-drawn ring geometry, not an assumed radius.
  2. GLASSES = the thin real circumference (|dist-R| <= 1.7) on the visible arcs,
     plus bridge + arm/hinge, PLUS an IMAGINED 1px stroke in the frame colour only
     where the circle has no real pixel (the top arc occluded by the lashes). This
     completes the ring without scooping the thick lash fill into the frame.
  3. EYES = the original with ONLY the thin ring band (ring-centred dist, radius ~20)
     knocked to transparent. The eyeball fill (radius < 13) and the whole detailed
     lash fill are never touched, so the eye stays complete and the lashes stay whole.

Re-run after any change to the source eye art. Idempotent: reads `*.pre_glass.png`.
"""
from __future__ import annotations
import os
import numpy as np
from PIL import Image

ACTIVE = os.path.expanduser("~/.crucible/avatars/active")
EYES = [(70, 127), (134, 127)]           # eye centres (for side split only)
STATES = ["open", "half", "blink", "closed", "wide"]


def _prep(a):
    h, w = a.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w]
    r, g, b, al = a[:, :, 0], a[:, :, 1], a[:, :, 2], a[:, :, 3]
    lum = r * 0.3 + g * 0.59 + b * 0.11
    opq = al > 40
    return h, w, yy, xx, lum, opq


def _fit_circle(pts):
    x, y = pts[:, 0], pts[:, 1]
    M = np.c_[x, y, np.ones_like(x)]
    a0, b0, c0 = np.linalg.lstsq(M, -(x ** 2 + y ** 2), rcond=None)[0]
    xc, yc = -a0 / 2, -b0 / 2
    return xc, yc, float(np.sqrt(xc ** 2 + yc ** 2 - c0))


def separate(active_dir: str = ACTIVE) -> None:
    src = np.asarray(Image.open(os.path.join(active_dir, "eyes_open.pre_glass.png"))
                     .convert("RGBA")).astype(int)
    h, w, yy, xx, lum, opq = _prep(src)
    r, b = src[:, :, 0], src[:, :, 2]
    neutral = r <= b + 11                     # the frame is neutral-dark; lashes/iris/skin are WARM (r>>b)
    region = (xx >= 28) & (xx <= 176) & (yy >= 95) & (yy <= 168)
    dark = opq & region & (lum < 110)
    frame_px = dark & neutral                 # frame candidates only — never warm lash/iris pixels

    rings = []
    for ex, ey in EYES:
        dEye = np.sqrt((xx - ex) ** 2 + (yy - ey) ** 2)
        side = (xx < 100) if ex < 100 else (xx >= 100)
        seed = dark & side & (dEye >= 15) & (dEye <= 27) & (yy > ey - 8)
        rings.append(_fit_circle(np.c_[xx[seed], yy[seed]]))

    def dC(XX, YY, i):
        xc, yc, _ = rings[i]
        return np.sqrt((XX - xc) ** 2 + (YY - yc) ** 2)

    # ---- glasses: thin real circumference + bridge/arm (NEUTRAL frame pixels only) ----
    glassm = np.zeros((h, w), bool)
    for i, (xc, yc, R) in enumerate(rings):
        side = (xx < 100) if xc < 100 else (xx >= 100)
        glassm |= frame_px & side & (np.abs(dC(xx, yy, i) - R) <= 1.7)
    # bridge = the narrow connector BETWEEN the lenses only (not into the lens interiors)
    glassm |= frame_px & (xx >= 92) & (xx <= 112) & (yy >= 122) & (yy <= 133)
    # arms/hinges = outer side pixels ON or OUTSIDE the ring only (dist>=R-2) — never the eye's inner
    # corner, which sits ~15px INSIDE the lens and would otherwise leak in as a speck.
    for i, (xc, yc, R) in enumerate(rings):
        side = (xx <= 58) if xc < 100 else (xx >= 146)
        glassm |= frame_px & side & (yy >= 108) & (yy <= 144) & (dC(xx, yy, i) >= R - 2)
    ecol = tuple(int(np.median(src[:, :, c][glassm])) for c in range(3))

    gl = np.zeros_like(src)
    gl[glassm] = src[glassm]
    gl[glassm, 3] = 255
    # COMPLETE the thin ring stroke in neutral frame colour wherever a pixel is missing (the top arc
    # occluded by lashes, plus any warm-antialiased edge we dropped) — keeps the ring continuous + clean.
    for i, (xc, yc, R) in enumerate(rings):
        side = (xx < 100) if xc < 100 else (xx >= 100)
        fill = side & (np.abs(dC(xx, yy, i) - R) <= 1.15) & ~glassm
        for c in range(3):
            gl[:, :, c][fill] = ecol[c]
        gl[:, :, 3][fill] = 255
    Image.fromarray(gl.astype("uint8")).save(os.path.join(active_dir, "glasses.png"))

    # ---- lashes = SEPARATE part (isolated by position: the thick dark fill INSIDE the top of each lens,
    #      dist<R-2 so the ring is excluded); saved from the OPEN state (the live render squashes 'open').
    #      Eyes then become LASH-FREE (eyeball/lids only) so the lashes layer doesn't double up. ----
    def lash_mask(arr, LUM, YY, XX):
        m = np.zeros((h, w), bool)
        for i, (xc, yc, R) in enumerate(rings):
            m |= (arr[:, :, 3] > 40) & (LUM < 120) & (dC(XX, YY, i) < R - 2) & (YY <= yc + 4)
        return m

    for st in STATES:
        p = os.path.join(active_dir, f"eyes_{st}.pre_glass.png")
        if not os.path.exists(p):
            continue
        e = np.asarray(Image.open(p).convert("RGBA")).astype(int).copy()
        _, _, YY, XX, elum, _ = _prep(e)
        lash = lash_mask(e, elum, YY, XX)
        lashes_path = os.path.join(active_dir, "lashes.png")
        if st == "open" and not os.path.exists(lashes_path):  # don't clobber a hand-edited lashes.png
            lay = np.zeros_like(e)
            lay[lash] = e[lash]
            lay[lash, 3] = 255
            Image.fromarray(lay.astype("uint8")).save(lashes_path)
        # eyes = eyeball/lids only. Lashes are a SEPARATE layer now, so strip the frame THOROUGHLY:
        # clear the whole thin ring band, AND any residual NEUTRAL-dark pixel in a wider ring vicinity
        # (the antialiased ring edge that read as a faint "glasses outline around the eyes").
        eneut = (e[:, :, 0] <= e[:, :, 2] + 11) & (elum < 140)
        band = np.zeros((h, w), bool)
        for i, (xc, yc, R) in enumerate(rings):
            side = (XX < 100) if xc < 100 else (XX >= 100)
            d = dC(XX, YY, i)
            band |= side & (np.abs(d - R) <= 3.2)                             # full ring band
            band |= side & eneut & (d > R - 6) & (d < R + 6)                  # + neutral frame vicinity
        band |= (XX >= 92) & (XX <= 112) & (YY >= 122) & (YY <= 133)          # bridge connector
        for i, (xc, yc, R) in enumerate(rings):                              # outer hinges
            side = (XX <= 58) if xc < 100 else (XX >= 146)
            band |= side & (YY >= 108) & (YY <= 144) & (dC(XX, YY, i) >= R - 2)
        e[band | lash] = 0
        Image.fromarray(e.astype("uint8")).save(os.path.join(active_dir, f"eyes_{st}.png"))

    print(f"rings={[tuple(round(v,1) for v in r) for r in rings]} frame_colour={ecol}")


if __name__ == "__main__":
    separate()
