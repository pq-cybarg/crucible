"""Split the eye whites (sclera) out of the eyeball sprites into their own part.

The lash-free eye sprite (`eyes_*.png`, produced by separate_glasses.py) is the eyeball:
a BRIGHT sclera fill plus a warm lower-lid / waterline rim. This pulls the bright sclera
into `whites.png` (a separate toggleable part, composited under the iris) and leaves the
warm rim in the eye sprite. The live render squashes `eyes_open`, so `whites.png` is taken
from the open state; every state is stripped of its sclera for consistency.

Backs each eye state up to `eyes_<st>.pre_sclera.png` once and reads from that, so re-running
re-derives from the untouched eyeball.
"""
from __future__ import annotations
import os
import numpy as np
from PIL import Image

ACTIVE = os.path.expanduser("~/.crucible/avatars/active")
STATES = ["open", "half", "blink", "closed", "wide"]
BRIGHT = 165          # sclera is bright (~200+); the warm rim/waterline is < 165


def separate(active_dir: str = ACTIVE) -> None:
    for st in STATES:
        cur = os.path.join(active_dir, f"eyes_{st}.png")
        if not os.path.exists(cur):
            continue
        backup = os.path.join(active_dir, f"eyes_{st}.pre_sclera.png")
        if not os.path.exists(backup):
            Image.open(cur).convert("RGBA").save(backup)

        a = np.asarray(Image.open(backup).convert("RGBA")).astype(int)
        r, g, b, al = a[:, :, 0], a[:, :, 1], a[:, :, 2], a[:, :, 3]
        lum = r * 0.3 + g * 0.59 + b * 0.11
        opq = al > 40
        sclera = opq & (lum >= BRIGHT)

        if st == "open":                                   # the whites layer (open shape → squashes live)
            w = np.zeros_like(a)
            w[sclera] = a[sclera]
            w[sclera, 3] = 255
            Image.fromarray(w.astype("uint8")).save(os.path.join(active_dir, "whites.png"))

        e = a.copy()                                       # eye sprite = the warm rim (sclera removed)
        e[sclera] = 0
        Image.fromarray(e.astype("uint8")).save(cur)
        if st == "open":
            print(f"sclera={int(sclera.sum())} rim={int((opq & ~sclera).sum())}")


if __name__ == "__main__":
    separate()
