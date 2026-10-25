#!/usr/bin/env python3
"""Bouncing, rotating shaded 3D sphere rendered with Unicode braille characters.

Each terminal cell is a 2x4 grid of braille dots.  For every braille dot we
shoot a ray through the sphere, compute the surface normal and Lambert shading,
and decide whether the dot is "on" using an ordered dithering threshold.  That
turns brightness into dot density, giving a smooth shaded sphere whose latitude
and longitude bands visibly rotate.
"""

import math
import shutil
import signal
import sys
import time

BRAILLE_BASE = 0x2800
# Dot bit for (dx, dy): dx in 0..1 (column within cell), dy in 0..3 (row in cell)
DOT_BITS = (
    (0x01, 0x02, 0x04, 0x40),
    (0x08, 0x10, 0x20, 0x80),
)

# 4x4 Bayer matrix for ordered dithering, normalised to 0..1.
BAYER = (
    (0, 8, 2, 10),
    (12, 4, 14, 6),
    (3, 11, 1, 9),
    (15, 7, 13, 5),
)
BAYER = tuple(tuple((v + 0.5) / 16.0 for v in row) for row in BAYER)

FPS = 30


def render_frame(cols, rows, cx, cy, r, light, angle):
    """Render one frame to a string."""
    px_w = cols * 2
    px_h = rows * 4
    lx, ly, lz = light
    inv_r = 1.0 / r

    lines = []
    for row in range(rows):
        chars = []
        for col in range(cols):
            bits = 0
            for dy in range(4):
                py = row * 4 + dy
                # Sphere-space y (up is positive), normalised to [-1, 1]
                ny = -(py + 0.5 - cy) * inv_r
                for dx in range(2):
                    px = col * 2 + dx
                    nx = (px + 0.5 - cx) * inv_r
                    rr = nx * nx + ny * ny
                    if rr >= 1.0:
                        continue
                    nz = math.sqrt(1.0 - rr)
                    # Lambert shading, light from upper-left-front.
                    shade = nx * lx + ny * ly + nz * lz
                    if shade <= 0.0:
                        continue
                    shade = min(1.0, shade)
                    # Latitude / longitude bands that rotate with the sphere.
                    lat = math.asin(max(-1.0, min(1.0, ny)))
                    lon = math.atan2(nz, nx) + angle
                    band = 0.5 + 0.5 * math.cos(lat * 9.0)
                    stripe = 0.5 + 0.5 * math.cos(lon * 6.0)
                    pattern = 0.72 + 0.28 * (0.6 * band + 0.4 * stripe)
                    value = shade * pattern
                    if value > BAYER[dy][dx]:
                        bits |= DOT_BITS[dx][dy]
            chars.append(chr(BRAILLE_BASE + bits) if bits else " ")
        lines.append("".join(chars))
    return "\n".join(lines)


def main():
    out = sys.stdout
    out.write("\x1b[?25l")  # hide cursor
    out.write("\x1b[2J\x1b[H")
    out.flush()

    def restore(*_):
        out.write("\x1b[?25h\x1b[0m\n")
        out.flush()
        sys.exit(0)

    signal.signal(signal.SIGINT, restore)
    signal.signal(signal.SIGTERM, restore)

    light = (-0.42, 0.52, 0.74)
    ln = math.sqrt(sum(c * c for c in light))
    light = tuple(c / ln for c in light)

    angle = 0.0
    sx, sy = 0.5, 0.5
    vx, vy = 0.012, 0.008  # fractional units per frame

    try:
        while True:
            t0 = time.monotonic()
            size = shutil.get_terminal_size(fallback=(80, 24))
            cols, rows = max(8, size.columns), max(4, size.lines)
            px_w, px_h = cols * 2, rows * 4

            r = max(3.0, min(px_w, px_h) * 0.32)

            angle += 0.05

            cx = sx * px_w + vx * px_w
            cy = sy * px_h + vy * px_h
            minx, maxx = r + 1, px_w - r - 1
            miny, maxy = r + 1, px_h - r - 1
            if maxx < minx:
                minx = maxx = px_w / 2.0
            if maxy < miny:
                miny = maxy = px_h / 2.0
            if cx < minx:
                cx, vx = minx, abs(vx)
            elif cx > maxx:
                cx, vx = maxx, -abs(vx)
            if cy < miny:
                cy, vy = miny, abs(vy)
            elif cy > maxy:
                cy, vy = maxy, -abs(vy)
            sx, sy = cx / px_w, cy / px_h

            frame = render_frame(cols, rows, cx, cy, r, light, angle)
            out.write("\x1b[H" + frame)
            out.flush()

            dt = time.monotonic() - t0
            sleep = (1.0 / FPS) - dt
            if sleep > 0:
                time.sleep(sleep)
    except KeyboardInterrupt:
        restore()


if __name__ == "__main__":
    main()
