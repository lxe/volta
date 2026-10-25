#!/usr/bin/env python3
"""Animated shaded 3D cube rendered with Unicode braille characters."""

import math
import shutil
import sys
import time

FPS = 30

# Braille dot bit values for each (row, col) in a 2x4 cell.
BRAILLE_BITS = (
    (0x01, 0x08),
    (0x02, 0x10),
    (0x04, 0x20),
    (0x40, 0x80),
)

# Cube corners (unit cube centred at origin).
CORNERS = [
    (-1, -1, -1), (1, -1, -1), (1, 1, -1), (-1, 1, -1),
    (-1, -1, 1), (1, -1, 1), (1, 1, 1), (-1, 1, 1),
]

# Faces as (corner indices, normal) — normals point outward.
FACES = [
    ((0, 1, 2, 3), (0, 0, -1)),
    ((4, 5, 6, 7), (0, 0, 1)),
    ((0, 1, 5, 4), (0, -1, 0)),
    ((3, 2, 6, 7), (0, 1, 0)),
    ((0, 3, 7, 4), (-1, 0, 0)),
    ((1, 2, 6, 5), (1, 0, 0)),
]

# 4x4 ordered dither matrix for brightness -> dot coverage.
BAYER = (
    (0, 8, 2, 10),
    (12, 4, 14, 6),
    (3, 11, 1, 9),
    (15, 7, 13, 5),
)


def rotate(p, ax, ay):
    x, y, z = p
    cx, sx = math.cos(ay), math.sin(ay)
    x, z = x * cx + z * sx, -x * sx + z * cx
    cy, sy = math.cos(ax), math.sin(ax)
    y, z = y * cy - z * sy, y * sy + z * cy
    return x, y, z


def main():
    out = sys.stdout
    out.write("\x1b[?25l")  # hide cursor
    out.flush()

    frame = 0
    t0 = time.monotonic()

    try:
        while True:
            cols, rows = shutil.get_terminal_size((80, 24))
            cols = max(cols, 10)
            rows = max(rows, 5)

            px_w = cols * 2   # dot columns
            px_h = rows * 4   # dot rows

            # Braille cells are ~2x as tall as wide; scale so cube fills nicely.
            scale = min(px_w / 2.2, px_h / 2.2)
            cx, cy = px_w / 2.0, px_h / 2.0

            ax = frame * 0.031
            ay = frame * 0.047

            proj = []
            for c in CORNERS:
                x, y, z = rotate(c, ax, ay)
                proj.append((cx + x * scale, cy - y * scale, z))

            light = (0.4, 0.55, -0.75)
            ln = math.sqrt(sum(v * v for v in light))
            light = tuple(v / ln for v in light)

            faces = []
            for idx, normal in FACES:
                nx, ny, nz = rotate(normal, ax, ay)
                if -nz <= 0.0:
                    continue
                lam = max(0.0, nx * light[0] + ny * light[1] + nz * light[2])
                shade = 0.25 + 0.75 * lam
                faces.append((idx, shade, nz))

            # Back-to-front painter's algorithm.
            faces.sort(key=lambda f: f[2], reverse=True)

            # grid holds brightness per dot (0.0 = empty).
            grid = [[0.0] * px_w for _ in range(px_h)]
            for idx, shade, _ in faces:
                pts = [proj[i] for i in idx]
                xs = [p[0] for p in pts]
                ys = [p[1] for p in pts]
                x0 = max(0, int(min(xs)))
                x1 = min(px_w - 1, int(max(xs)) + 1)
                y0 = max(0, int(min(ys)))
                y1 = min(px_h - 1, int(max(ys)) + 1)
                if x1 < x0 or y1 < y0:
                    continue
                n = len(pts)
                for py in range(y0, y1 + 1):
                    yc = py + 0.5
                    xints = []
                    for k in range(n):
                        ax0, ay0, _ = pts[k]
                        ax1, ay1, _ = pts[(k + 1) % n]
                        if (ay0 <= yc < ay1) or (ay1 <= yc < ay0):
                            t = (yc - ay0) / (ay1 - ay0)
                            xints.append(ax0 + t * (ax1 - ax0))
                    if len(xints) < 2:
                        continue
                    xa, xb = min(xints), max(xints)
                    pa = max(0, int(math.ceil(xa - 0.5)))
                    pb = min(px_w - 1, int(math.floor(xb - 0.5)))
                    row = grid[py]
                    for pxx in range(pa, pb + 1):
                        row[pxx] = shade

            lines = []
            for r in range(rows):
                chars = []
                for c in range(cols):
                    bits = 0
                    for dr in range(4):
                        gy = r * 4 + dr
                        for dc in range(2):
                            b = grid[gy][c * 2 + dc]
                            threshold = (BAYER[gy & 3][(c * 2 + dc) & 3] + 0.5) / 16.0
                            if b > threshold:
                                bits |= BRAILLE_BITS[dr][dc]
                    chars.append(chr(0x2800 + bits))
                lines.append("".join(chars))

            out.write("\x1b[H" + "\n".join(lines))
            out.flush()

            frame += 1
            target = t0 + frame / FPS
            delay = target - time.monotonic()
            if delay > 0:
                time.sleep(delay)
    except KeyboardInterrupt:
        pass
    finally:
        out.write("\x1b[?25h\x1b[0m\n")
        out.flush()


if __name__ == "__main__":
    main()
