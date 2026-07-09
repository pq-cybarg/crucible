from __future__ import annotations
# Background removal for avatar PARTS — a part sprite needs transparency around it so it composites on
# the rig. Dependency-free (no rembg): sample the corners as the background color and make pixels within
# a tolerance transparent, then crop to the content. Good for generated art on a flat background or for a
# drawn part. (A learned matting model can be swapped in later for hair/soft edges.)


def remove_background(img, tolerance: int = 32, autocrop: bool = True, feather: bool = True):
    """Return an RGBA image with the flat background knocked out. `tolerance` = colour distance from the
    sampled background that still counts as background. `autocrop` trims to the remaining content."""
    import numpy as np
    from PIL import Image

    rgba = img.convert("RGBA")
    arr = np.asarray(rgba).astype(np.int16)
    h, w, _ = arr.shape
    corners = np.array([arr[0, 0, :3], arr[0, w - 1, :3], arr[h - 1, 0, :3], arr[h - 1, w - 1, :3]])
    bg = np.median(corners, axis=0)
    dist = np.sqrt(((arr[..., :3] - bg) ** 2).sum(axis=2))
    if feather:                                           # soft edge: fade alpha over the tolerance band
        alpha = np.clip((dist - tolerance) / max(1.0, tolerance * 0.5), 0.0, 1.0)
        arr[..., 3] = (arr[..., 3] * alpha).astype(np.int16)
    else:
        arr[..., 3][dist <= tolerance] = 0
    out = Image.fromarray(arr.clip(0, 255).astype("uint8"), "RGBA")
    if autocrop:
        bbox = out.getbbox()
        if bbox:
            out = out.crop(bbox)
    return out


def remove_background_file(src: str, dst: str, tolerance: int = 32) -> str:
    from PIL import Image
    remove_background(Image.open(src), tolerance=tolerance).save(dst)
    return dst
