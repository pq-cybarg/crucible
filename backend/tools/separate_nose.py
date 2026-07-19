"""Split the little nose mark out of the head sprite (`base.png`) into its own part.

The dead-fish face has a small warm nose shadow (~a dozen px) below and between the eyes.
This lifts it into `nose.png` and fills the hole in `base.png` with the surrounding skin
colour, so the nose becomes a toggleable part. With the nose shown it composites back over
the exact same pixels (no visible change); toggling it off reveals clean skin.

Backs up `base.png` to `base.pre_nose.png` once and reads from that on re-runs.
"""
from __future__ import annotations
import os
import numpy as np
from PIL import Image, ImageFilter

ACTIVE = os.path.expanduser("~/.crucible/avatars/active")
# nose search box (below/between the eyes) — centre of the face
BOX = (94, 111, 137, 158)      # x0, x1(exclusive-ish via <=), y0, y1


def separate(active_dir: str = ACTIVE) -> None:
    cur = os.path.join(active_dir, "base.png")
    backup = os.path.join(active_dir, "base.pre_nose.png")
    if not os.path.exists(backup):
        Image.open(cur).convert("RGBA").save(backup)

    a = np.asarray(Image.open(backup).convert("RGBA")).astype(int)
    h, w = a.shape[:2]
    r, g, b, al = a[:, :, 0], a[:, :, 1], a[:, :, 2], a[:, :, 3]
    lum = r * 0.3 + g * 0.59 + b * 0.11
    opq = al > 40
    yy, xx = np.mgrid[0:h, 0:w]
    x0, x1, y0, y1 = BOX
    region = opq & (xx >= x0) & (xx <= x1) & (yy >= y0) & (yy <= y1)

    skin_lum = float(np.median(lum[region]))                  # local skin brightness
    warm = r > b + 6
    core = region & warm & (lum < skin_lum - 10)              # the darker warm nose shadow
    grown = np.asarray(Image.fromarray((core * 255).astype("uint8"))
                       .filter(ImageFilter.MaxFilter(3))) > 128   # dilate 1px to catch the AA edge
    nose = region & grown & (lum < skin_lum - 3)              # include the lighter blob edge, not flat skin
    # local skin colour to fill the hole (median of the light, non-nose region pixels)
    skinpx = region & ~nose & (lum >= skin_lum - 4)
    skin_col = tuple(int(np.median(a[:, :, c][skinpx])) for c in range(3)) + (255,)

    nose_img = np.zeros_like(a)
    nose_img[nose] = a[nose]
    nose_img[nose, 3] = 255
    Image.fromarray(nose_img.astype("uint8")).save(os.path.join(active_dir, "nose.png"))

    base = a.copy()
    for c in range(4):
        base[:, :, c][nose] = skin_col[c]
    Image.fromarray(base.astype("uint8")).save(cur)
    print(f"nose={int(nose.sum())} skin_fill={skin_col}")


if __name__ == "__main__":
    separate()
