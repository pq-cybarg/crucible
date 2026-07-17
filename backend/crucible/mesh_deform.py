"""Mesh-deformation + soft-body physics for the companion rig.

The 2D avatar's motion must DEFORM a continuous surface, not translate flat
"cardboard" cutouts — a rotated cutout would reveal geometry that was hidden
(a shoulder joint, the inner side of a lock of hair) and hide what was visible,
leaving holes and seams. A mesh warp instead moves control points and smoothly
resamples the art between them, so nothing pops apart. The same engine drives
hair sway, chest/skirt jiggle, and (with bones) head-turn.

Two pieces:
  * `warp_triangles` — forward triangle-affine image warp (numpy only).
  * `SoftStrandSolver` — a semi-implicit spring/damper solver for a grid mesh
    whose top rows are PINNED to a rigid driver (the skull) and whose lower
    rows lag and bounce (springy, with viscous resistance).

Kept dependency-light (numpy + PIL) so it runs in the render path.
"""
from __future__ import annotations

import numpy as np


def build_grid_mesh(bbox: tuple[int, int, int, int], cols: int, rows: int):
    """A regular `cols`×`rows` grid of vertices over `bbox`=(x0,y0,x1,y1) plus the
    triangle list (two per cell) and each vertex's row index. Returns
    (rest[N,2] float, tris[list of (i,j,k)], row_of[N] int, cols, rows)."""
    x0, y0, x1, y1 = bbox
    gx = np.linspace(x0, x1, cols)
    gy = np.linspace(y0, y1, rows)
    rest = np.array([[gx[c], gy[r]] for r in range(rows) for c in range(cols)], float)
    row_of = np.array([r for r in range(rows) for c in range(cols)], int)

    def vid(r, c):
        return r * cols + c

    tris = []
    for r in range(rows - 1):
        for c in range(cols - 1):
            a, b, cc, d = vid(r, c), vid(r, c + 1), vid(r + 1, c), vid(r + 1, c + 1)
            tris += [(a, b, cc), (b, d, cc)]
    return rest, tris, row_of, cols, rows


def warp_triangles(src_rgba: np.ndarray, rest: np.ndarray, dst: np.ndarray,
                   tris: list[tuple[int, int, int]]) -> np.ndarray:
    """Forward-warp an RGBA image: for each triangle, map its REST (source) shape
    onto its DST (deformed) shape with an affine sample. Overlapping coverage is
    averaged so shared edges don't seam. `rest`/`dst` are Nx2 vertex arrays in
    pixel coords; returns a uint8 HxWx4 array the same size as `src_rgba`."""
    h, w = src_rgba.shape[:2]
    src = src_rgba.astype(np.float32)
    out = np.zeros((h, w, 4), np.float32)
    acc = np.zeros((h, w), np.float32)
    for (i, j, k) in tris:
        dt = dst[[i, j, k]]
        st = rest[[i, j, k]]
        minx = max(0, int(np.floor(dt[:, 0].min())))
        maxx = min(w - 1, int(np.ceil(dt[:, 0].max())))
        miny = max(0, int(np.floor(dt[:, 1].min())))
        maxy = min(h - 1, int(np.ceil(dt[:, 1].max())))
        if maxx < minx or maxy < miny:
            continue
        ys, xs = np.mgrid[miny:maxy + 1, minx:maxx + 1]
        px = np.stack([xs.ravel(), ys.ravel()], 1).astype(np.float32)
        v0 = dt[1] - dt[0]
        v1 = dt[2] - dt[0]
        v2 = px - dt[0]
        den = v0[0] * v1[1] - v1[0] * v0[1]
        if abs(den) < 1e-6:
            continue
        u = (v2[:, 0] * v1[1] - v1[0] * v2[:, 1]) / den
        v = (v0[0] * v2[:, 1] - v2[:, 0] * v0[1]) / den
        ww = 1 - u - v
        inside = (u >= -0.02) & (v >= -0.02) & (ww >= -0.02)
        if not inside.any():
            continue
        sc = ww[:, None] * st[0] + u[:, None] * st[1] + v[:, None] * st[2]
        sx = np.clip(sc[:, 0].round().astype(int), 0, w - 1)
        sy = np.clip(sc[:, 1].round().astype(int), 0, h - 1)
        ox = px[:, 0].astype(int)
        oy = px[:, 1].astype(int)
        m = inside
        out[oy[m], ox[m]] += src[sy[m], sx[m]]
        acc[oy[m], ox[m]] += 1
    acc[acc == 0] = 1
    return (out / acc[:, :, None]).astype(np.uint8)


class SoftStrandSolver:
    """Spring/damper physics for a grid mesh pinned to a rigid driver at the top.

    Vertices with `pin`≈1 (the roots / skull-attached rows) snap to the driver's
    rigid transform each step; lower vertices lag via a spring toward that rigid
    anchor plus a chain spring that preserves each vertex's offset from the one
    above — so a lock of hair keeps its shape while its tip swings and settles.

    Semi-implicit (symplectic) Euler with a speed clamp + max-deviation clamp:
    explicit Euler with stiff springs diverges, so those clamps + damping keep it
    unconditionally stable at interactive rates.
    """

    def __init__(self, rest: np.ndarray, pin: np.ndarray, pivot: tuple[float, float],
                 up_idx: np.ndarray, *, k_anchor=0.05, k_anchor_gain=0.20, k_chain=0.30,
                 damp=0.28, dt=0.6, vmax=6.0, max_dev=60.0):
        self.rest = np.asarray(rest, float)
        self.pin = np.asarray(pin, float)
        self.pivot = np.asarray(pivot, float)
        self.up_idx = np.asarray(up_idx, int)
        self.k_anchor = k_anchor + k_anchor_gain * self.pin
        self.k_chain = k_chain
        self.damp = damp
        self.dt = dt
        self.vmax = vmax
        self.max_dev = max_dev
        self.pos = self.rest.copy()
        self.vel = np.zeros_like(self.rest)
        self._root = self.pin > 0.85

    def _rigid(self, deg: float, bob: float = 0.0) -> np.ndarray:
        """Where every vertex would be if rigidly attached to the head: rotate the
        rest grid about the pivot by `deg`, then translate by `bob` (vertical).
        NOTE: sign matches PIL's `Image.rotate(deg)` (CCW) used for the head crop,
        so the hair rotates the SAME visual direction as the head it's pinned to —
        NOT the opposite (which made the head appear to phase through the hair)."""
        th = np.radians(-deg)
        c, s = np.cos(th), np.sin(th)
        d = self.rest - self.pivot
        rot = self.pivot + np.stack([d[:, 0] * c - d[:, 1] * s,
                                     d[:, 0] * s + d[:, 1] * c], 1)
        rot[:, 1] += bob
        return rot

    def step(self, deg: float, bob: float = 0.0, dt: float | None = None) -> np.ndarray:
        """Advance one physics step for head pose (deg tilt, bob px); returns the
        deformed vertex positions (Nx2) to feed `warp_triangles`."""
        dt = self.dt if dt is None else dt
        anchor = self._rigid(deg, bob)
        chain_target = self.pos[self.up_idx] + (self.rest - self.rest[self.up_idx])
        force = self.k_anchor[:, None] * (anchor - self.pos) + self.k_chain * (chain_target - self.pos)
        self.vel = (self.vel + force * dt) * (1 - self.damp)
        sp = np.linalg.norm(self.vel, axis=1)
        over = sp > self.vmax
        if over.any():
            self.vel[over] *= (self.vmax / sp[over])[:, None]
        self.pos = self.pos + self.vel * dt
        dev = self.pos - anchor
        dl = np.linalg.norm(dev, axis=1)
        far = dl > self.max_dev
        if far.any():
            self.pos[far] = anchor[far] + dev[far] * (self.max_dev / dl[far])[:, None]
        self.pos[self._root] = anchor[self._root]
        self.vel[self._root] = 0
        return self.pos

    def reset(self):
        self.pos = self.rest.copy()
        self.vel = np.zeros_like(self.rest)


def pin_weights(row_of: np.ndarray, rows: int, exponent: float = 1.3) -> np.ndarray:
    """Per-vertex pin weight: 1 at the top row (rigid root), →0 at the bottom
    (free tip). `exponent`>1 keeps more of the strand rigid near the root."""
    return np.array([max(0.0, 1.0 - (r / (rows - 1)) ** exponent) for r in row_of])


def up_indices(row_of: np.ndarray, cols: int, rows: int) -> np.ndarray:
    """For each vertex, the index of the vertex one row ABOVE in the same column
    (itself for the top row) — the chain-spring parent."""
    return np.array([((max(0, r - 1)) * cols + c) for r in range(rows) for c in range(cols)], int)


class HairLayerRig:
    """A ready-to-drive hair (or any pinned-strand) layer: builds a grid mesh over
    an RGBA sprite's opaque bbox, pins the top rows to the skull, and holds a
    `SoftStrandSolver` so each `deform(angle, bob, dt)` returns the warped sprite
    with the tips lagging/bouncing. Stateful — one per live session."""

    def __init__(self, rgba: np.ndarray, cols: int = 11, rows: int = 10,
                 pin_exp: float = 1.3, pivot_override: tuple[float, float] | None = None,
                 **solver_kw):
        self.rgba = rgba
        h, w = rgba.shape[:2]
        ys, xs = np.where(rgba[:, :, 3] > 40)
        if len(xs) == 0:                                     # nothing to rig
            self.empty = True
            return
        self.empty = False
        bbox = (int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max()))
        self.rest, self.tris, row_of, c, r = build_grid_mesh(bbox, cols, rows)
        pin = pin_weights(row_of, r, pin_exp)
        up = up_indices(row_of, c, r)
        # pivot = the head-rotation centre (share the head-bob pivot so the roots track the skull),
        # else just below the layer's own bbox.
        pivot = pivot_override or ((bbox[0] + bbox[2]) / 2, bbox[3] + (bbox[3] - bbox[1]) * 0.15)
        # tight sway limit: hair is ATTACHED — the tips jiggle a little, they don't swing across the face
        max_dev = max(6.0, (bbox[3] - bbox[1]) * 0.12)
        self.solver = SoftStrandSolver(self.rest, pin, pivot, up, max_dev=max_dev, **solver_kw)

    def deform(self, angle_deg: float, bob: float = 0.0, dt: float | None = None) -> np.ndarray:
        """Step the physics for the current head pose and return the warped RGBA."""
        if self.empty:
            return self.rgba
        dst = self.solver.step(angle_deg, bob, dt)
        return warp_triangles(self.rgba, self.rest, dst, self.tris)
