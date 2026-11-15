"""Draw the work-mat texture for the hands-on scene: ``scenes/handson_mat.png``.

Laid out like the session's workspace mat (without its printing or logos): a black 350 x 300 mm
mat with a 10 mm grid and a 50 mm major grid, an X axis along the long edge and a Y axis along
the short edge, both in millimetres, and a dashed PLACE AREA.

Mat coordinates, as at the session: the arm stands at the mat's long edge (Y = 0) at X = 100 mm,
facing +Y. In world coordinates (the arm's base at the origin, facing +x)::

    world x = 0.06 + Y / 1000        world y = 0.10 - X / 1000

The PNG is committed; run this only to change the mat::

    python make_mat_texture.py
"""

from __future__ import annotations

import math
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

OUT = Path(__file__).resolve().parent / "scenes" / "handson_mat.png"

# Mat size in mm. Must match the `mat` geom in scenes/handson_pick_place.xml:
# centre (0.06 + 0.150, 0.10 - 0.175) = (0.21, -0.075), half sizes (0.150, 0.175).
MAT_X, MAT_Y = 350, 300
PX_PER_MM = 4

# PLACE AREA centre and side in mat mm. Must match the `place_area` body: world (0.285, -0.15).
PLACE_X, PLACE_Y, PLACE_SIDE = 250, 225, 53

FONT_BOLD = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"

BACKGROUND = (24, 24, 27)
MINOR = (62, 62, 67)
MAJOR = (110, 110, 118)
INK = (236, 236, 236)
EDGE = (8, 8, 9)


def px(x_mm: float, y_mm: float) -> tuple[float, float]:
    """Drawing pixel of a mat point: +X to the right, +Y up (the arm's side at the bottom).

    Seen from behind the arm, this is how the mat lies in the overview camera.
    """
    return (x_mm * PX_PER_MM, (MAT_Y - y_mm) * PX_PER_MM)


def font(path: str, size_mm: float) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    try:
        return ImageFont.truetype(path, round(size_mm * PX_PER_MM))
    except OSError:
        return ImageFont.load_default(round(size_mm * PX_PER_MM))


def text(draw: ImageDraw.ImageDraw, xy_mm: tuple[float, float], label: str, fnt, anchor: str = "mm") -> None:
    draw.text(px(*xy_mm), label, fill=INK, font=fnt, anchor=anchor)


def arrow(draw: ImageDraw.ImageDraw, start: tuple[float, float], end: tuple[float, float]) -> None:
    """Axis line from *start* to *end* (mat mm) with an arrowhead at *end*."""
    draw.line([px(*start), px(*end)], fill=INK, width=round(0.7 * PX_PER_MM))
    (sx, sy), (ex, ey) = px(*start), px(*end)
    angle = math.atan2(ey - sy, ex - sx)
    size = 5 * PX_PER_MM
    left = (ex - size * math.cos(angle - 0.4), ey - size * math.sin(angle - 0.4))
    right = (ex - size * math.cos(angle + 0.4), ey - size * math.sin(angle + 0.4))
    draw.polygon([(ex, ey), left, right], fill=INK)


def rounded_square_path(cx: float, cy: float, side: float, radius: float, steps: int = 12) -> list[tuple[float, float]]:
    """Closed outline of a rounded square in mat mm."""
    half = side / 2
    corners = [(half - radius, half - radius), (-half + radius, half - radius),
               (-half + radius, -half + radius), (half - radius, -half + radius)]
    points = []
    for i, (ox, oy) in enumerate(corners):
        for k in range(steps + 1):
            a = math.radians(i * 90 + 90 * k / steps)
            points.append((cx + ox + radius * math.cos(a), cy + oy + radius * math.sin(a)))
    points.append(points[0])
    return points


def dashed(draw: ImageDraw.ImageDraw, path_mm: list[tuple[float, float]], dash: float, gap: float) -> None:
    """Draw *path_mm* as a dashed line (lengths in mm)."""
    width = round(0.6 * PX_PER_MM)
    on, left = True, dash
    for (x0, y0), (x1, y1) in zip(path_mm, path_mm[1:]):
        seg = math.hypot(x1 - x0, y1 - y0)
        pos = 0.0
        while pos < seg:
            step = min(left, seg - pos)
            a, b = pos / seg, (pos + step) / seg
            if on:
                p0 = (x0 + (x1 - x0) * a, y0 + (y1 - y0) * a)
                p1 = (x0 + (x1 - x0) * b, y0 + (y1 - y0) * b)
                draw.line([px(*p0), px(*p1)], fill=INK, width=width)
            pos += step
            left -= step
            if left <= 1e-9:
                on = not on
                left = dash if on else gap


def cube_icon(draw: ImageDraw.ImageDraw, cx: float, cy: float, size: float) -> None:
    """A small line drawing of a cube, centred on mat (cx, cy) mm."""
    ccx, ccy = px(cx, cy)
    s = size * PX_PER_MM / 2
    w = round(0.6 * PX_PER_MM)
    top = [(ccx, ccy - s), (ccx + s, ccy - s / 2), (ccx, ccy), (ccx - s, ccy - s / 2)]
    draw.line([*top, top[0]], fill=INK, width=w)
    draw.line([(ccx - s, ccy - s / 2), (ccx - s, ccy + s / 2), (ccx, ccy + s),
               (ccx + s, ccy + s / 2), (ccx + s, ccy - s / 2)], fill=INK, width=w)
    draw.line([(ccx, ccy), (ccx, ccy + s)], fill=INK, width=w)


def main() -> None:
    width, height = MAT_X * PX_PER_MM, MAT_Y * PX_PER_MM
    image = Image.new("RGB", (width, height), BACKGROUND)
    draw = ImageDraw.Draw(image)

    # Grid inside the axes, as on the session mat.
    gx0, gx1, gy0, gy1 = 30, MAT_X - 30, 20, MAT_Y - 20
    for x in range(gx0, gx1 + 1, 10):
        major = x % 50 == 0
        draw.line([px(x, gy0), px(x, gy1)], fill=MAJOR if major else MINOR, width=2 if major else 1)
    for y in range(gy0, gy1 + 1, 10):
        major = y % 50 == 0
        draw.line([px(gx0, y), px(gx1, y)], fill=MAJOR if major else MINOR, width=2 if major else 1)

    label = font(FONT, 6)
    title = font(FONT_BOLD, 6.5)
    # X axis along the long edge on the arm's side, pointing away from the arm's left.
    arrow(draw, (gx0, gy0), (gx1 + 8, gy0))
    for x in range(100, gx1 + 1, 50):
        draw.line([px(x, gy0), px(x, gy0 + 3)], fill=INK, width=3)
        text(draw, (x, gy0 + 5), str(x), label, anchor="mb")
    text(draw, (gx1 + 5, gy0 - 4), "X (mm)", title, anchor="rt")
    # Y axis along the short edge, pointing away from the arm.
    arrow(draw, (gx0, gy0), (gx0, gy1 + 8))
    for y in range(50, gy1 + 1, 50):
        draw.line([px(gx0, y), px(gx0 + 3, y)], fill=INK, width=3)
        text(draw, (gx0 + 5, y), str(y), label, anchor="lm")
    text(draw, (gx0 + 4, gy1 + 6), "Y (mm)", title, anchor="lm")

    # PLACE AREA, as on the session mat: a plain panel (no grid inside) in a dashed rounded
    # square, a large cube icon in the middle, and the name in bold along the far side, across X.
    half = PLACE_SIDE / 2
    left, top = px(PLACE_X - half, PLACE_Y + half)
    right, bottom = px(PLACE_X + half, PLACE_Y - half)
    draw.rounded_rectangle([left, top, right, bottom], radius=6 * PX_PER_MM, fill=BACKGROUND)
    dashed(draw, rounded_square_path(PLACE_X, PLACE_Y, PLACE_SIDE, 6), dash=1.6, gap=1.2)
    cube_icon(draw, PLACE_X, PLACE_Y - 4, 15)
    text(draw, (PLACE_X, PLACE_Y + 16), "PLACE AREA", font(FONT_BOLD, 5.5), anchor="mm")

    # Stitched edge, as on a cloth mat.
    draw.rectangle([0, 0, width - 1, height - 1], outline=EDGE, width=round(3 * PX_PER_MM))

    # MuJoCo maps this box face's texture columns to world +x and rows to world -y (checked with a
    # corner-colour test texture). The drawing has +Y (world +x) up and +X (world -y) right, so
    # turn it 90 degrees clockwise to land it as drawn.
    image = image.transpose(Image.Transpose.ROTATE_270)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    image.save(OUT, optimize=True)
    print(f"Wrote {OUT} ({image.width}x{image.height} px)")


if __name__ == "__main__":
    main()
