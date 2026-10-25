"""Draw deja's app icon (smartreplay/frontend/static/deja.ico) from icon.svg's shapes.

    python tools/make_deja_icon.py

Windows wants a raster .ico for the window and taskbar, and the standard
library cannot render SVG, so this redraws the same geometry - a violet tile
with the play-button D - with Pillow, supersampled so the small sizes stay crisp.
Run it again after changing icon.svg.
"""

from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw

OUT = Path(__file__).resolve().parents[1] / "smartreplay" / "frontend" / "static" / "deja.ico"
SIZES = (16, 20, 24, 32, 40, 48, 64, 128, 256)
TOP, BOTTOM = (0x8B, 0x5C, 0xF6), (0xC0, 0x26, 0xD3)
SS = 16  # supersampling factor over the 32-unit design grid of icon.svg


def gradient(size: int) -> Image.Image:
    """The tile's diagonal gradient, top-left to bottom-right."""
    img = Image.new("RGB", (size, size))
    px = img.load()
    for y in range(size):
        for x in range(size):
            t = (x + y) / (2 * (size - 1))
            px[x, y] = tuple(round(a + (b - a) * t) for a, b in zip(TOP, BOTTOM))
    return img


def draw() -> Image.Image:
    n = 32 * SS
    s = lambda v: v * SS  # icon.svg units to pixels
    tile = gradient(n).convert("RGBA")
    mask = Image.new("L", (n, n), 0)
    ImageDraw.Draw(mask).rounded_rectangle((0, 0, n - 1, n - 1), radius=s(8), fill=255)
    tile.putalpha(mask)
    d = ImageDraw.Draw(tile)
    # The D: flat left edge with rounded corners, a semicircular bowl on the right.
    d.rounded_rectangle((s(8.5), s(7), s(16.5), s(25)), radius=s(2), fill="white",
                        corners=(True, False, False, True))
    d.pieslice((s(7.5), s(7), s(25.5), s(25)), -90, 90, fill="white")
    # The play triangle, cut back through to the tile.
    hole = Image.new("L", (n, n), 0)
    ImageDraw.Draw(hole).polygon([(s(13.5), s(11.5)), (s(20), s(16)), (s(13.5), s(20.5))], fill=255)
    tile.paste(gradient(n).convert("RGBA"), (0, 0), hole)
    return tile


def main() -> None:
    big = draw()
    frames = [big.resize((size, size), Image.LANCZOS) for size in SIZES]
    frames[-1].save(OUT, format="ICO", sizes=[(size, size) for size in SIZES], append_images=frames[:-1])
    print(f"wrote {OUT} ({', '.join(map(str, SIZES))} px)")


if __name__ == "__main__":
    main()
