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


def test_two_color_palette(tmp_path):
    import re
    from crucible.pixelface import render_file
    p = tmp_path / "s.png"
    _make_png(str(p))
    lines = render_file(str(p), cols=24, duotone="mono", palette_size=2)
    fg = set(re.findall(r"38;2;(\d+;\d+;\d+)", "".join(lines)))
    bg = set(re.findall(r"48;2;(\d+;\d+;\d+)", "".join(lines)))
    assert len(fg | bg) <= 2                                      # genuinely two colors
