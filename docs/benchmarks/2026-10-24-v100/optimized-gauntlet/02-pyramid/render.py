#!/usr/bin/env python3
"""Braille-rendered spinning, tilting, shaded square pyramid."""

import math
import shutil
import signal
import sys
import time

# Braille dot bit values for a 2x4 cell, indexed [row][col].
DOTS = ((0x01, 0x08), (0x02, 0x10), (0x04, 0x20), (0x40, 0x80))
BLANK = 0x2800

# 4x4 ordered-dither threshold matrix (values in 0..1).
BAYER4 = [[(v + 0.5) / 16.0 for v in row] for row in (
    (0, 8, 2, 10),
    (12, 4, 14, 6),
    (3, 11, 1, 9),
    (15, 7, 13, 5),
)]

# Pyramid geometry: square base of half-width 1 at y=-1, apex at y=+1.
HALF = 1.0
APEX_Y = 1.6
BASE_Y = -1.0
BASE = [(-HALF, BASE_Y, -HALF), (HALF, BASE_Y, -HALF),
        (HALF, BASE_Y, HALF), (-HALF, BASE_Y, HALF)]
APEX = (0.0, APEX_Y, 0.0)

# Faces: (vertices, normal computed from winding).
FACES = [
    (BASE[0], BASE[1], APEX),
    (BASE[1], BASE[2], APEX),
    (BASE[2], BASE[3], APEX),
    (BASE[3], BASE[0], APEX),
    (BASE[3], BASE[2], BASE[1], BASE[0]),  # base, wound so normal points down
]


def sub(a, b):
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def cross(a, b):
    return (a[1] * b[2] - a[2] * b[1],
            a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0])


def dot(a, b):
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def norm(a):
    l = math.sqrt(dot(a, a)) or 1.0
    return (a[0] / l, a[1] / l, a[2] / l)


def rotate(p, spin, tilt):
    """Spin about vertical (y) axis, then tilt about horizontal (x) axis."""
    c, s = math.cos(spin), math.sin(spin)
    x, y, z = p[0] * c - p[2] * s, p[1], p[0] * s + p[2] * c
    c, s = math.cos(tilt), math.sin(tilt)
    y, z = y * c - z * s, y * s + z * c
    return (x, y, z)


def transform(p, spin, tilt, dist, zoom):
    x, y, z = rotate(p, spin, tilt)
    z += dist
    f = zoom / z
    return (x * f, y * f, z)


def shade(normal, spin, tilt):
    """Shade in 0..1: how much the face faces the viewer, plus a soft light."""
    n = rotate(normal, spin, tilt)
    facing = max(0.0, n[2])
    light = norm((-0.4, 0.5, 0.9))
    lam = max(0.0, dot(n, light))
    # Keep a floor so silhouette edges stay visible, then blend facing+light.
    return min(1.0, 0.28 + 0.72 * (0.55 * facing + 0.45 * lam) * (0.5 + 0.5 * facing))

def triangle_coverage(px, py, tri):
    """Barycentric inside test; returns (inside, w0, w1, w2)."""
    (x0, y0), (x1, y1), (x2, y2) = tri
    d = (y1 - y2) * (x0 - x2) + (x2 - x1) * (y0 - y2)
    if abs(d) < 1e-12:
        return False, 0.0, 0.0, 0.0
    w0 = ((y1 - y2) * (px - x2) + (x2 - x1) * (py - y2)) / d
    w1 = ((y2 - y0) * (px - x2) + (x0 - x2) * (py - y2)) / d
    w2 = 1.0 - w0 - w1
    return (w0 >= 0 and w1 >= 0 and w2 >= 0), w0, w1, w2


def render(cols, rows, spin, tilt):
    dot_w = cols * 2
    dot_h = rows * 4
    dist = 5.0

    # First pass with unit focal length to find the projected bounding box.
    raw = {v: transform(v, spin, tilt, dist, 1.0) for v in BASE + [APEX]}
    xs = [p[0] for p in raw.values()]
    ys = [p[1] for p in raw.values()]
    cx = (min(xs) + max(xs)) * 0.5
    cy = (min(ys) + max(ys)) * 0.5
    ext_x = max(1e-6, max(xs) - min(xs))
    ext_y = max(1e-6, max(ys) - min(ys))

    # Fit to 80% of the viewport, preserving aspect (dot cells are 2x4).
    zoom = min(0.72 * dot_w / ext_x, 0.72 * dot_h / ext_y)

    # Project all vertices, centered and scaled to the dot grid.
    proj = {}
    for v in BASE + [APEX]:
        p = raw[v]
        proj[v] = ((p[0] - cx) * zoom + dot_w * 0.5,
                   dot_h * 0.5 - (p[1] - cy) * zoom,
                   p[2])
    # Build a list of screen triangles with shade, sorted back-to-front.
    tris = []
    for face in FACES:
        n = norm(cross(sub(face[1], face[0]), sub(face[2], face[0])))
        n_view = rotate(n, spin, tilt)
        if n_view[2] <= 0.0:
            continue  # back-facing: hidden by the solid body
        s = shade(n, spin, tilt)
        # Widen contrast so faces read as distinct planes.
        s = max(0.12, min(0.97, (s - 0.28) / 0.72 * 0.85 + 0.12))
        pts = [proj[v] for v in face]
        depth = sum(p[2] for p in pts) / len(pts)
        scr = [(p[0], p[1]) for p in pts]
        if len(scr) == 3:
            tris.append((depth, s, [scr]))
        else:  # quad -> two triangles
            tris.append((depth, s, [[scr[0], scr[1], scr[2]],
                                    [scr[0], scr[2], scr[3]]]))
    tris.sort(key=lambda t: -t[0])

    # Rasterize into a dot buffer with shade values.
    buf = [0.0] * (dot_w * dot_h)
    for depth, s, parts in tris:
        for tri in parts:
            xs = [p[0] for p in tri]
            ys = [p[1] for p in tri]
            x0 = max(0, int(math.floor(min(xs))))
            x1 = min(dot_w - 1, int(math.ceil(max(xs))))
            y0 = max(0, int(math.floor(min(ys))))
            y1 = min(dot_h - 1, int(math.ceil(max(ys))))
            for yy in range(y0, y1 + 1):
                py = yy + 0.5
                row_off = yy * dot_w
                for xx in range(x0, x1 + 1):
                    inside, _, _, _ = triangle_coverage(xx + 0.5, py, tri)
                    if inside:
                        buf[row_off + xx] = s

    # Draw the pyramid's edges as bright lines so faces read as separate planes.
    edges = {(APEX, BASE[0]), (APEX, BASE[1]), (APEX, BASE[2]), (APEX, BASE[3]),
             (BASE[0], BASE[1]), (BASE[1], BASE[2]),
             (BASE[2], BASE[3]), (BASE[3], BASE[0])}
    for a, b in edges:
        pa, pb = proj[a], proj[b]
        dx = pb[0] - pa[0]
        dy = pb[1] - pa[1]
        steps = int(max(abs(dx), abs(dy))) + 1
        for i in range(steps + 1):
            t = i / steps
            ex = int(round(pa[0] + dx * t))
            ey = int(round(pa[1] + dy * t))
            if 0 <= ex < dot_w and 0 <= ey < dot_h:
                buf[ey * dot_w + ex] = 1.0

    # Pack dot buffer into braille characters using ordered dithering so the
    # per-face shade shows up as dot density.
    lines = []
    for cy in range(rows):
        chars = []
        for cx in range(cols):
            bits = 0
            for r in range(4):
                yy = cy * 4 + r
                if yy >= dot_h:
                    break
                row_off = yy * dot_w
                for c in range(2):
                    xx = cx * 2 + c
                    if xx >= dot_w:
                        break
                    s = buf[row_off + xx]
                    if s > 0.0 and s > BAYER4[(yy & 3)][(xx & 3)]:
                        bits |= DOTS[r][c]
            chars.append(chr(BLANK + bits))
        lines.append("".join(chars))
    return lines


def main():
    out = sys.stdout
    try:
        out.write("\x1b[?25l")  # hide cursor
        out.flush()
        start = time.monotonic()
        while True:
            t = time.monotonic() - start
            size = shutil.get_terminal_size(fallback=(80, 24))
            cols = max(4, size.columns)
            rows = max(4, size.lines)
            spin = t * 0.9
            tilt = 0.38 + 0.30 * math.sin(t * 0.37)
            lines = render(cols, rows, spin, tilt)
            out.write("\x1b[H" + "\n".join(lines))
            out.flush()
            target = 1.0 / 30.0
            elapsed = time.monotonic() - start - t
            delay = target - elapsed
            if delay > 0:
                time.sleep(delay)
    except KeyboardInterrupt:
        pass
    finally:
        out.write("\x1b[?25h\x1b[0m\n")
        out.flush()


if __name__ == "__main__":
    signal.signal(signal.SIGTERM, lambda *a: (_ for _ in ()).throw(KeyboardInterrupt))
    main()
