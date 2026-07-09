"""Background removal for avatar parts — flat background → transparent, then crop to content."""


def test_removes_flat_background_and_crops(tmp_path):
    from PIL import Image
    from crucible.bgremove import remove_background
    # a white canvas with a small solid red square in the middle
    img = Image.new("RGB", (40, 40), (255, 255, 255))
    for y in range(16, 24):
        for x in range(16, 24):
            img.putpixel((x, y), (200, 30, 30))
    out = remove_background(img, tolerance=30, feather=False)
    # the white background is gone → cropped down to (roughly) the 8x8 red square
    assert out.mode == "RGBA"
    assert out.width <= 12 and out.height <= 12
    # centre pixel is opaque red; there is real content
    px = out.getpixel((out.width // 2, out.height // 2))
    assert px[3] > 0 and px[0] > 120


def test_all_background_yields_empty_or_tiny(tmp_path):
    from PIL import Image
    from crucible.bgremove import remove_background
    img = Image.new("RGB", (30, 30), (10, 10, 10))     # uniform → all background
    out = remove_background(img, tolerance=30, feather=False)
    # everything transparent; getbbox() None → returns the original (nothing to crop)
    alpha = out.split()[-1]
    assert alpha.getextrema()[1] == 0                  # max alpha is 0 (fully transparent)
