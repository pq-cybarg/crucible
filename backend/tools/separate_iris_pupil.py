"""Split the baked iris+pupil disc (`pupils.png`) into two independent parts:

  irises.png  — the COMPLETE iris disc (the pupil area is filled with the iris colour,
                i.e. the iris behind the pupil is IMAGINED, per the occlusion rule) so the
                iris is a whole disc that reads correctly with the pupil hidden.
  pupils.png  — ONLY the dark central pupil (real pixels), composited on top of the iris.

The dead-fish eye has a mid-brown iris ring (lum ~70-130) with a darker pupil (lum<70) and
no catchlight. Both parts sit in the eye box so they squash-close with the eye; the render
composites iris then pupil (z: iris under pupil), both under the lashes/glasses.

Idempotent-ish: backs up the original disc to `pupils.pre_split.png` on first run and always
reads from that backup, so re-running re-derives both parts from the untouched disc.
"""
from __future__ import annotations
import os
import numpy as np
from PIL import Image

ACTIVE = os.path.expanduser("~/.crucible/avatars/active")
EYES = [(70, 127), (134, 127)]


def separate(active_dir: str = ACTIVE) -> None:
    src_path = os.path.join(active_dir, "pupils.png")
    backup = os.path.join(active_dir, "pupils.pre_split.png")
    if not os.path.exists(backup):                       # preserve the original combined disc once
        Image.open(src_path).convert("RGBA").save(backup)

    a = np.asarray(Image.open(backup).convert("RGBA")).astype(int)
    h, w = a.shape[:2]
    r, g, b, al = a[:, :, 0], a[:, :, 1], a[:, :, 2], a[:, :, 3]
    lum = r * 0.3 + g * 0.59 + b * 0.11
    opq = al > 40
    yy, xx = np.mgrid[0:h, 0:w]

    disc = np.zeros((h, w), bool)                         # the two iris discs (a box around each eye)
    for cx, cy in EYES:
        disc |= (np.abs(xx - cx) < 14) & (np.abs(yy - cy) < 11)
    disc &= opq

    pupil = disc & (lum < 75)                             # dark centre
    iris_ring = disc & ~pupil                             # brown ring (visible iris)
    iris_col = tuple(int(np.median(a[:, :, c][iris_ring])) for c in range(3))

    # irises.png = the whole disc, pupil area filled with iris colour (imagine the iris behind pupil)
    iris_img = np.zeros_like(a)
    iris_img[iris_ring] = a[iris_ring]
    iris_img[iris_ring, 3] = 255
    for c in range(3):
        iris_img[:, :, c][pupil] = iris_col[c]
    iris_img[:, :, 3][pupil] = 255
    Image.fromarray(iris_img.astype("uint8")).save(os.path.join(active_dir, "irises.png"))

    # pupils.png = the dark pupil only (real pixels)
    pup_img = np.zeros_like(a)
    pup_img[pupil] = a[pupil]
    pup_img[pupil, 3] = 255
    Image.fromarray(pup_img.astype("uint8")).save(os.path.join(active_dir, "pupils.png"))

    print(f"iris_ring={int(iris_ring.sum())} pupil={int(pupil.sum())} iris_colour={iris_col}")


if __name__ == "__main__":
    separate()
