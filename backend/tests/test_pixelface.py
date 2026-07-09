"""Pixel-art terminal renderer — image → ANSI half-block blocks (the TUI avatar's look)."""


def _make_png(path: str, w: int = 60, h: int = 80) -> None:
    from PIL import Image
    img = Image.new("RGB", (w, h))
    px = img.load()
    for y in range(h):
        for x in range(w):
            px[x, y] = (x * 255 // w, y * 255 // h, 128)   # a gradient with structure
    img.save(path)


def test_renders_ansi_halfblocks_at_target_width(tmp_path):
    from crucible.pixelface import render_file, strip_ansi
    p = tmp_path / "s.png"
    _make_png(str(p))
    lines = render_file(str(p), cols=32)
    assert lines and all("\x1b[" in ln for ln in lines)          # ANSI color codes present
    assert all("▀" in ln for ln in lines)                        # upper-half-block used
    # each visible row is exactly `cols` blocks wide (two-pixel-tall cells)
    stripped = strip_ansi(lines)
    assert all(len(s) == 32 for s in stripped)


def test_duotone_limits_the_palette(tmp_path):
    import re
    from crucible.pixelface import render_file
    p = tmp_path / "s.png"
    _make_png(str(p))
    lines = render_file(str(p), cols=24, duotone="terminal-sepia", palette_size=3)
    colors = set(re.findall(r"38;2;(\d+;\d+;\d+)", "".join(lines)))
    # a duotone ramp posterized to 3 levels → only a few distinct foreground colors
    assert 1 <= len(colors) <= 8


def test_quad_blocks_add_horizontal_detail_without_distorting(tmp_path):
    from crucible.pixelface import render_image, strip_ansi
    from PIL import Image
    # a fine checkerboard has mixed 2x2 blocks → exercises the partial quadrant glyphs
    img = Image.new("RGB", (80, 100))
    for y in range(100):
        for x in range(80):
            img.putpixel((x, y), (240, 240, 240) if (x + y) % 2 == 0 else (10, 10, 10))
    half = render_image(img, cols=30, blocks="half")
    quad = render_image(img, cols=30, blocks="quad")
    # same on-screen character dimensions for both (aspect-correct — quad no longer doubles the rows,
    # which was the squish bug); quad packs 2x horizontal SUBpixels per cell instead.
    assert all(len(s) == 30 for s in strip_ansi(half))
    assert all(len(s) == 30 for s in strip_ansi(quad))
    assert len(quad) == len(half)                              # same rows → same aspect, no squish
    assert any(ch in "".join(quad) for ch in "▘▝▖▗▚▞▛▜▙▟▌▐")   # quadrant glyphs = finer horizontal detail


def test_duotone_posterize_is_frame_stable(tmp_path):
    # a given luminance must ALWAYS map to the same duotone shade, independent of the rest of the image —
    # otherwise animating one region (a blink) re-buckets another (the hair flickers colour).
    import numpy as np
    from PIL import Image
    from crucible.pixelface import _apply_duotone, DUOTONES
    flat = Image.new("RGB", (24, 12), (110, 110, 110))
    withpatch = flat.copy()
    for x in range(4):                                       # a bright patch, like eyes opening
        for y in range(4):
            withpatch.putpixel((x, y), (255, 255, 255))
    a = np.asarray(_apply_duotone(flat, DUOTONES["terminal-sepia"], 5, False))
    b = np.asarray(_apply_duotone(withpatch, DUOTONES["terminal-sepia"], 5, False))
    assert np.array_equal(a[6:, 8:], b[6:, 8:])             # the far region is untouched by the patch


def test_two_color_palette(tmp_path):
    import re
    from crucible.pixelface import render_file
    p = tmp_path / "s.png"
    _make_png(str(p))
    lines = render_file(str(p), cols=24, duotone="mono", palette_size=2)
    fg = set(re.findall(r"38;2;(\d+;\d+;\d+)", "".join(lines)))
    bg = set(re.findall(r"48;2;(\d+;\d+;\d+)", "".join(lines)))
    assert len(fg | bg) <= 2                                      # genuinely two colors
