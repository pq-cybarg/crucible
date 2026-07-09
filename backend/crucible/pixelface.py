from __future__ import annotations
# Render a pixel-art sprite (the avatar's face/body) into the TERMINAL as low-res color blocks — the
# look from the reference: a limited-palette, dithered anime portrait with a retro-terminal vibe. This is
# the TUI avatar's rendering foundation: any expression sprite → a compact ANSI image the sidebar shows,
# low-resolution and low-framerate by design so it's cheap to redraw as reactions fire.
#
# Technique: each terminal cell is drawn with the upper-half-block ▀ — foreground color = the TOP pixel,
# background color = the BOTTOM pixel — so one character row shows TWO pixel rows (square-ish pixels).
# Palette reduction + Floyd–Steinberg dithering give the two-/few-color halftone feel; an optional duotone
# maps luminance onto a two-color ramp (e.g. dark→cream) for the sepia terminal-waifu aesthetic.
from typing import Optional

RESET = "\x1b[0m"

# a few ready palettes (as (dark, light) duotone ramps); extend freely
DUOTONES: dict[str, tuple[tuple[int, int, int], tuple[int, int, int]]] = {
    "terminal-sepia": ((26, 20, 16), (232, 220, 196)),
    "amber": ((18, 10, 0), (255, 176, 0)),
    "green": ((2, 12, 4), (140, 240, 150)),
    "mono": ((16, 16, 16), (235, 235, 235)),
}


def render_image(img, cols: int = 44, rows: Optional[int] = None, palette_size: int = 0,
                 dither: bool = True, duotone: Optional[str] = None,
                 bg: tuple = (26, 20, 16), blocks: str = "half") -> list[str]:
    """Render an in-memory PIL image (e.g. a composited avatar) to terminal pixel blocks. Transparent
    pixels are flattened onto `bg` (the box background). `blocks`: 'half' = ▀ half-blocks (1×2 px/cell,
    full color); 'quad' = quadrant blocks (2×2 px/cell — DOUBLE the resolution in the same box width, at
    2 colors/cell). Both keep pixel art crisp (nearest-neighbour resize)."""
    from PIL import Image

    if img.mode in ("RGBA", "LA", "P"):
        flat = Image.new("RGB", img.size, bg)
        rgba = img.convert("RGBA")
        flat.paste(rgba, mask=rgba.split()[-1])
        img = flat
    else:
        img = img.convert("RGB")
    w, h = img.size
    sx = 2 if blocks == "quad" else 1                  # quad packs 2 subpixels per cell horizontally
    # A character cell is ~1 wide : 2 tall. So (with a cell holding 2 rows of subpixels) the row count
    # that keeps the image UNDISTORTED is cols·(h/w)·0.5 — independent of horizontal subpixels. Using
    # sx here (the old bug) doubled the rows for quad and squished the picture horizontally.
    if rows is None:
        rows = max(1, round(cols * (h / w) * 0.5))
    img = img.resize((cols * sx, rows * 2), Image.NEAREST)
    if duotone and duotone in DUOTONES:
        img = _apply_duotone(img, DUOTONES[duotone], palette_size if palette_size >= 2 else 0, dither)
    elif palette_size and palette_size >= 2:
        img = img.quantize(colors=palette_size,
                           dither=Image.FLOYDSTEINBERG if dither else Image.NONE).convert("RGB")
    return _to_ansi_quad(img) if blocks == "quad" else _to_ansi(img)


def render_file(path: str, cols: int = 44, rows: Optional[int] = None, palette_size: int = 0,
                dither: bool = True, duotone: Optional[str] = None) -> list[str]:
    """Return ANSI lines rendering the image FILE as terminal pixel blocks. `cols` = width in characters;
    `rows` (character rows) defaults to preserve aspect (each row = 2 pixels). `palette_size`>=2 reduces
    to that many colors (dithered); `duotone` maps luminance onto a named two-color ramp."""
    from PIL import Image

    img = Image.open(path).convert("RGB")
    w, h = img.size
    if rows is None:
        rows = max(1, round(cols * (h / w) * 0.5))
    img = img.resize((cols, rows * 2), Image.LANCZOS)
    if duotone and duotone in DUOTONES:
        img = _apply_duotone(img, DUOTONES[duotone], palette_size if palette_size >= 2 else 0, dither)
    elif palette_size and palette_size >= 2:
        img = img.quantize(colors=palette_size,
                           dither=Image.FLOYDSTEINBERG if dither else Image.NONE).convert("RGB")
    return _to_ansi(img)


# 4×4 Bayer matrix (centred to [-0.5, 0.5)) for ORDERED dithering — a fixed, position-deterministic
# pattern, unlike Floyd–Steinberg error-diffusion whose result depends on the whole image.
import numpy as _np
_BAYER4 = _np.array([[0, 8, 2, 10], [12, 4, 14, 6], [3, 11, 1, 9], [15, 7, 13, 5]],
                    dtype=_np.float32) / 16.0 - 0.5


def _apply_duotone(img, ramp, levels: int, dither: bool):
    from PIL import Image
    import numpy as np

    g = np.asarray(img.convert("L"), dtype=np.float32) / 255.0
    if levels >= 2:
        n = levels - 1
        # FIXED-threshold posterization (a given luminance ALWAYS maps to the same level) + an ORDERED
        # dither. Both are frame-INDEPENDENT, so animating one part (a blink, a glance) can never re-bucket
        # another part's shade — the old adaptive quantize + error-diffusion made the hair flicker on blink.
        if dither:
            h, w = g.shape
            tile = np.tile(_BAYER4, (h // 4 + 1, w // 4 + 1))[:h, :w]
            g = g + tile / n
        g = np.clip(np.round(g * n) / n, 0.0, 1.0)
    lo = np.array(ramp[0], dtype=np.float32)
    hi = np.array(ramp[1], dtype=np.float32)
    out = (lo + (hi - lo) * g[..., None]).astype("uint8")
    return Image.fromarray(out, "RGB")


def _to_ansi(img) -> list[str]:
    import numpy as np

    arr = np.asarray(img.convert("RGB"))
    hh, ww, _ = arr.shape
    lines: list[str] = []
    for y in range(0, hh - 1, 2):
        parts = []
        prev = None
        for x in range(ww):
            tr, tg, tb = (int(v) for v in arr[y, x])
            br, bg, bb = (int(v) for v in arr[y + 1, x])
            cell = (tr, tg, tb, br, bg, bb)
            if cell != prev:                            # only re-emit color codes when they change (smaller)
                parts.append(f"\x1b[38;2;{tr};{tg};{tb};48;2;{br};{bg};{bb}m")
                prev = cell
            parts.append("▀")                      # ▀ upper half block
        parts.append(RESET)
        lines.append("".join(parts))
    return lines


# quadrant blocks by 2×2 bitmask: bit0=top-left, bit1=top-right, bit2=bottom-left, bit3=bottom-right
_QUAD = [" ", "▘", "▝", "▀", "▖", "▌", "▞", "▛", "▗", "▚", "▐", "▜", "▄", "▙", "▟", "█"]


def _to_ansi_quad(img) -> list[str]:
    """2×2-pixels-per-cell renderer: for each block, split the 4 subpixels into two colors (bright vs
    dark) and pick the quadrant glyph — twice the spatial resolution of half-blocks per character."""
    import numpy as np

    arr = np.asarray(img.convert("RGB")).astype(int)
    hh, ww, _ = arr.shape
    lum = (0.299 * arr[..., 0] + 0.587 * arr[..., 1] + 0.114 * arr[..., 2])
    lines: list[str] = []
    for y in range(0, hh - 1, 2):
        parts = []
        prev = None
        for x in range(0, ww - 1, 2):
            px = [arr[y, x], arr[y, x + 1], arr[y + 1, x], arr[y + 1, x + 1]]
            ls = [lum[y, x], lum[y, x + 1], lum[y + 1, x], lum[y + 1, x + 1]]
            hi, lo = int(np.argmax(ls)), int(np.argmin(ls))
            fg, bg = px[hi], px[lo]
            mid = (ls[hi] + ls[lo]) / 2
            mask = sum((1 << i) for i in range(4) if ls[i] >= mid)   # bright subpixels → fg
            key = (fg[0], fg[1], fg[2], bg[0], bg[1], bg[2])
            if key != prev:
                parts.append(f"\x1b[38;2;{fg[0]};{fg[1]};{fg[2]};48;2;{bg[0]};{bg[1]};{bg[2]}m")
                prev = key
            parts.append(_QUAD[mask])
        parts.append(RESET)
        lines.append("".join(parts))
    return lines


def strip_ansi(lines: list[str]) -> list[str]:
    import re
    a = re.compile(r"\x1b\[[0-9;]*m")
    return [a.sub("", ln) for ln in lines]
