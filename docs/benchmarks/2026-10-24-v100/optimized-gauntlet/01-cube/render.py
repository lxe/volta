#!/usr/bin/env python3
"""Rotating shaded 3D cube rendered with Unicode braille characters."""

import math
import shutil
import signal
import sys
import time

# Braille dot bit values: 2 columns x 4 rows per cell.
DOT = (
    (0x01, 0x08),
    (0x02, 0x10),
    (0x04, 0x20),
    (0x40, 0x80),
)

# Cube vertices (unit cube centered at origin).
VERTS = [
    (-1, -1, -1), (1, -1, -1), (1, 1, -1), (-1, 1, -1),
    (-1, -1, 1), (1, -1, 1), (1, 1, 1), (-1, 1, 1),
]

# Faces: vertex indices (counter-clockwise seen from outside) and base shade.
FACES = [
    ((0, 1, 2, 3), 0.55),  # -z
    ((4, 5, 6, 7), 0.55),  # +z
    ((0, 1, 5, 4), 0.75),  # -y
    ((3, 2, 6, 7), 0.75),  # +y
    ((0, 3, 7, 4), 0.95),  # -x
    ((1, 2, 6, 5), 0.95),  # +x
]

SHADES = " .:-=+*#%@"


def rot(axis, angle, m):
    c, s = math.cos(angle), math.sin(angle)
    if axis == "x":
        r = ((1, 0, 0), (0, c, -s), (0, s, c))
    elif axis == "y":
        r = ((c, 0, s), (0, 1, 0), (-s, 0, c))
    else:
        r = ((c, -s, 0), (s, c, 0), (0, 0, 1))
    return tuple(
        tuple(sum(r[i][k] * m[k][j] for k in range(3)) for j in range(3))
        for i in range(3)
    )


def apply(m, v):
    return tuple(sum(m[i][k] * v[k] for k in range(3)) for i in range(3))


def edge_func(a, b, p):
    return (b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0])


def render(cols, rows, angle_x, angle_y):
    px_w, px_h = cols * 2, rows * 4
    m = rot("x", angle_x, ((1, 0, 0), (0, 1, 0), (0, 0, 1)))
    m = rot("y", angle_y, m)

    # Project vertices to pixel space with perspective, sized to the viewport.
    # Camera distance chosen so the cube occupies ~70% of the smaller axis.
    cam = 6.0
    base = min(px_w, px_h) * 0.30  # cube half-size in pixels
    proj = []
    for v in VERTS:
        x, y, z = apply(m, v)
        f = cam / (cam - z * 0.9)
        px = x * base * f + px_w * 0.5
        py = y * base * f + px_h * 0.5
        proj.append((px, py, z))

    # Per-pixel brightness buffer (-1 = empty), plus depth for z-buffer.
    buf = [-1.0] * (px_w * px_h)
    zbuf = [1e9] * (px_w * px_h)
    light = (-0.4, -0.6, -0.7)  # light direction (toward viewer, upper-left)

    for idx, base in FACES:
        a, b, c, d = (proj[i] for i in idx)
        zavg = sum(p[2] for p in (a, b, c, d)) / 4.0
        n = _normal(apply(m, VERTS[idx[0]]), apply(m, VERTS[idx[1]]),
                    apply(m, VERTS[idx[2]]))
        # Lambert shading: how much the normal faces the light.
        lam = max(0.0, n[0] * light[0] + n[1] * light[1] + n[2] * light[2])
        facing = max(0.0, -n[2])
        shade = base * (0.12 + 0.88 * lam) * (0.35 + 0.65 * facing)

        minx = max(0, int(min(a[0], b[0], c[0], d[0])))
        maxx = min(px_w - 1, int(max(a[0], b[0], c[0], d[0])) + 1)
        miny = max(0, int(min(a[1], b[1], c[1], d[1])))
        maxy = min(px_h - 1, int(max(a[1], b[1], c[1], d[1])) + 1)

        for py in range(miny, maxy + 1):
            yc = py + 0.5
            for px in range(minx, maxx + 1):
                p = (px + 0.5, yc)
                w0 = edge_func(b, c, p)
                w1 = edge_func(c, d, p)
                w2 = edge_func(d, a, p)
                w3 = edge_func(a, b, p)
                if (w0 >= 0 and w1 >= 0 and w2 >= 0 and w3 >= 0) or (
                    w0 <= 0 and w1 <= 0 and w2 <= 0 and w3 <= 0
                ):
                    i = py * px_w + px
                    if zavg < zbuf[i]:
                        zbuf[i] = zavg
                        buf[i] = shade

    # Pack pixels into braille cells; brightness -> dot density (Bayer dither).
    bayer = (
        (0, 32, 8, 40, 2, 34, 10, 42),
        (48, 16, 56, 24, 50, 18, 58, 26),
        (12, 44, 4, 36, 14, 46, 6, 38),
        (60, 28, 52, 20, 62, 30, 54, 22),
        (3, 35, 11, 43, 1, 33, 9, 41),
        (51, 19, 59, 27, 49, 17, 57, 25),
        (15, 47, 7, 39, 13, 45, 5, 37),
        (63, 31, 55, 23, 61, 29, 53, 21),
    )
    lines = []
    for cy in range(rows):
        chars = []
        for cx in range(cols):
            bits = 0
            for dy in range(4):
                for dx in range(2):
                    px = cx * 2 + dx
                    py = cy * 4 + dy
                    s = buf[py * px_w + px]
                    if s >= 0:
                        threshold = (bayer[dy][dx] + 0.5) / 64.0
                        if s >= threshold:
                            bits |= DOT[dy][dx]
            chars.append(chr(0x2800 + bits) if bits else " ")
        lines.append("".join(chars))
    return lines


def _normal(p0, p1, p2):
    ux, uy, uz = p1[0] - p0[0], p1[1] - p0[1], p1[2] - p0[2]
    vx, vy, vz = p2[0] - p0[0], p2[1] - p0[1], p2[2] - p0[2]
    nx = uy * vz - uz * vy
    ny = uz * vx - ux * vz
    nz = ux * vy - uy * vx
    ln = math.sqrt(nx * nx + ny * ny + nz * nz) or 1.0
    return (nx / ln, ny / ln, nz / ln)


def main():
    out = sys.stdout
    out.write("\x1b[?25l")  # hide cursor
    out.write("\x1b[2J")
    out.flush()

    def restore(*_):
        out.write("\x1b[?25h\x1b[0m\n")
        out.flush()
        sys.exit(0)

    signal.signal(signal.SIGINT, restore)
    signal.signal(signal.SIGTERM, restore)

    frame_time = 1.0 / 30.0
    t0 = time.monotonic()
    try:
        while True:
            start = time.monotonic()
            size = shutil.get_terminal_size((80, 24))
            cols, rows = size.columns, size.lines
            t = start - t0
            lines = render(cols, rows, t * 0.9, t * 1.3)
            frame = ["\x1b[H"] + [ln + "\x1b[K" for ln in lines]
            out.write("\n".join(frame))
            out.flush()
            elapsed = time.monotonic() - start
            if elapsed < frame_time:
                time.sleep(frame_time - elapsed)
    finally:
        out.write("\x1b[?25h\x1b[0m\n")
        out.flush()


if __name__ == "__main__":
    main()
