"""Subdivide the single `hair.png` bob into SUBSECTION layers — groundwork for 2.5D head-turn (#26).

The bob splits into the pieces that move independently as the head turns:
  hair_crown.png  — the top/parted crown (and the back mass behind the head)
  hair_bangs.png  — the front fringe over the forehead (stays in FRONT of the face on a turn)
  hair_left.png   — the left side-lock / curtain hanging down
  hair_right.png  — the right side-lock / curtain

Split by position around the head centre (meta head-centre x, default 103). Non-overlapping, so
re-compositing all four reproduces `hair.png` exactly. Backs `hair.png` up to `hair.pre_sub.png`.

This is the SEGMENTATION step. Wiring each piece to its own physics rig / per-subsection toggle and the
2.5D depth ordering is the next phase; these layers are the data it will build on.
"""
from __future__ import annotations
import os
import numpy as np
from PIL import Image

ACTIVE = os.path.expanduser("~/.crucible/avatars/active")
CX = 103          # head centre x on the 200-wide canvas (avatar meta)
CROWN_Y = 54      # crown/top band ends here
SIDE_HALF = 19    # the centre band (bangs) is CX ± SIDE_HALF; outside it → left/right locks


def subdivide(active_dir: str = ACTIVE) -> None:
    cur = os.path.join(active_dir, "hair.png")
    backup = os.path.join(active_dir, "hair.pre_sub.png")
    if not os.path.exists(backup):
        Image.open(cur).convert("RGBA").save(backup)

    a = np.asarray(Image.open(backup).convert("RGBA")).astype("uint8")
    h, w = a.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w]
    opq = a[:, :, 3] > 40                                           # (only for the px counts printed)

    # Partition EVERY pixel by position (not gated on opacity), so the four layers recompose to the exact
    # original incl. faint anti-aliased edges (transparent pixels just copy as transparent).
    crown = yy <= CROWN_Y                                           # top/parted crown + back
    below = yy > CROWN_Y
    bangs = below & (np.abs(xx - CX) <= SIDE_HALF)                  # centre fringe over the forehead
    left = below & (xx < CX - SIDE_HALF)                            # left side-lock
    right = below & (xx > CX + SIDE_HALF)                           # right side-lock

    for name, mask in (("crown", crown), ("bangs", bangs), ("left", left), ("right", right)):
        out = np.zeros_like(a)
        out[mask] = a[mask]
        Image.fromarray(out).save(os.path.join(active_dir, f"hair_{name}.png"))
        print(f"hair_{name}: {int((mask & opq).sum())} hair px")


if __name__ == "__main__":
    subdivide()
