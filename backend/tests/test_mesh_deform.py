"""Mesh-deformation engine: warp correctness + physics stability/behavior."""
import numpy as np

from crucible.mesh_deform import (
    build_grid_mesh, warp_triangles, SoftStrandSolver, pin_weights, up_indices,
)


def _checker(h=40, w=40):
    a = np.zeros((h, w, 4), np.uint8)
    a[..., 3] = 255
    yy, xx = np.mgrid[0:h, 0:w]
    a[..., 0] = np.where((xx // 5 + yy // 5) % 2 == 0, 220, 40)
    return a


def test_identity_warp_reproduces_image():
    img = _checker()
    rest, tris, _, _, _ = build_grid_mesh((0, 0, 39, 39), 5, 5)
    out = warp_triangles(img, rest, rest.copy(), tris)
    # identity mesh → essentially the same image (allow tiny resample noise at edges)
    diff = np.abs(out[..., :3].astype(int) - img[..., :3].astype(int))
    assert diff.mean() < 6


def test_displacement_moves_content():
    img = _checker()
    rest, tris, _, _, _ = build_grid_mesh((0, 0, 39, 39), 5, 5)
    dst = rest.copy()
    dst[:, 0] += 8  # shift every vertex right by 8px
    out = warp_triangles(img, rest, dst, tris)
    # content should have moved right: left edge column now mostly empty(ish) vs original
    assert out[:, 0:4, 3].mean() < img[:, 0:4, 3].mean()


def test_solver_is_stable_and_settles():
    rest, tris, row_of, cols, rows = build_grid_mesh((0, 0, 40, 60), 5, 7)
    pin = pin_weights(row_of, rows)
    up = up_indices(row_of, cols, rows)
    solver = SoftStrandSolver(rest, pin, pivot=(20, 70), up_idx=up)
    # a damped shake, then let it settle
    for f in range(60):
        deg = 12 * np.sin(f / 3) * np.exp(-f / 20)
        pos = solver.step(deg)
        assert np.isfinite(pos).all(), "physics diverged (NaN/inf)"
        assert np.abs(pos - rest).max() < 200, "physics exploded"
    # after the shake decays, it returns near rest
    for _ in range(40):
        pos = solver.step(0.0)
    assert np.abs(pos - rest).max() < 3.0


def test_roots_pinned_tips_lag():
    rest, tris, row_of, cols, rows = build_grid_mesh((0, 0, 40, 60), 5, 7)
    pin = pin_weights(row_of, rows)
    up = up_indices(row_of, cols, rows)
    solver = SoftStrandSolver(rest, pin, pivot=(20, 70), up_idx=up)
    pos = solver.step(15.0)  # sudden tilt
    root = pin > 0.85
    tip = row_of == (rows - 1)
    root_move = np.abs(pos[root] - rest[root]).sum(1).mean()
    tip_move = np.abs(pos[tip] - rest[tip]).sum(1).mean()
    # on the first frame the pinned roots have already moved to the rigid pose,
    # while the springy tips lag behind (moved less than a rigid tip would)
    rigid_tip = solver._rigid(15.0)[tip]
    rigid_tip_move = np.abs(rigid_tip - rest[tip]).sum(1).mean()
    assert root_move > 0.0
    assert tip_move < rigid_tip_move
