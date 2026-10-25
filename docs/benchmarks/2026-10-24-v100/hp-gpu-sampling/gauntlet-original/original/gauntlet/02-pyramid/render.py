#!/usr/bin/env python3
"""Spinning, tilting, shaded square pyramid rendered with braille characters.

Each terminal cell is a 2x4 grid of dots (U+2800..U+28FF).  Every frame the
terminal size is re-read, the scene is rasterised at dot resolution into a
depth buffer, and only the characters that changed are emitted.
"""

import math
import shutil
import signal
import sys
import time

# ---------------------------------------------------------------------------
# Braille bit layout: dot (x, y) -> bit index, x in 0..1 (left..right),
# y in 0..3 (top..bottom).
# ---------------------------------------------------------------------------
DOT_BITS = (
    (0x01, 0x02, 0x04, 0x40),  # left column, rows 0..3
    (0x08, 0x10, 0x20, 0x80),  # right column, rows 0..3
)

CLEAR = "\x1b[H\x1b[2J"
HIDE_CURSOR = "\x1b[?25l"
SHOW_CURSOR = "\x1b[?25h"
RESET = "\x1b[0m"
# 256-colour ramp used for shading: dark blue-grey up to bright white.
SHADE_COLOURS = (234, 238, 242, 245, 248, 250, 252, 253, 255)
TARGET_FPS = 30.0

# ---------------------------------------------------------------------------
# Geometry: a square pyramid centred on the origin, sitting on y = -h/2.
# ---------------------------------------------------------------------------
BASE_HALF = 0.78
APEX_Y = 1.05
BASE_Y = -0.72
APEX = (0.0, APEX_Y, 0.0)
BASE = (
    (-BASE_HALF, BASE_Y, -BASE_HALF),
    (BASE_HALF, BASE_Y, -BASE_HALF),
    (BASE_HALF, BASE_Y, BASE_HALF),
    (-BASE_HALF, BASE_Y, BASE_HALF),
)

# Faces: (vertices, outward normal hint).  The apex is repeated so each side
# face is one triangle.
SIDE_FACES = [
    ((APEX, BASE[0], BASE[1]), (0.0, 0.5, -1.0)),
    ((APEX, BASE[1], BASE[2]), (1.0, 0.5, 0.0)),
    ((APEX, BASE[2], BASE[3]), (0.0, 0.5, 1.0)),
    ((APEX, BASE[3], BASE[0]), (-1.0, 0.5, 0.0)),
]
BOTTOM_FACE = ((BASE[0], BASE[1], BASE[2], BASE[3]), (0.0, -1.0, 0.0))


def rot_x(v, a):
    c, s = math.cos(a), math.sin(a)
    x, y, z = v
    return (x, y * c - z * s, y * s + z * c)


def rot_y(v, a):
    c, s = math.cos(a), math.sin(a)
    x, y, z = v
    return (x * c + z * s, y, -x * s + z * c)


def sub(a, b):
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def cross(a, b):
    return (
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    )


def dot(a, b):
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def normalise(v):
    n = math.sqrt(dot(v, v)) or 1.0
    return (v[0] / n, v[1] / n, v[2] / n)


def build_faces(spin, tilt):
    """Transform the pyramid and return (vertices, colour_index) per face."""
    def tx(p):
        return rot_x(rot_y(p, spin), tilt)

    faces = []
    light = normalise((-0.45, 0.72, -0.52))
    for verts, hint in SIDE_FACES + [BOTTOM_FACE]:
        tv = [tx(p) for p in verts]
        n = normalise(cross(sub(tv[1], tv[0]), sub(tv[2], tv[0])))
        if dot(n, tx(hint)) < 0.0:      # make the normal point outwards
            n = (-n[0], -n[1], -n[2])
        # View vector points from the face towards the eye (looking along +z
        # towards the viewer at z = +inf), so facing-ness is n.z.
        facing = n[2]
        if facing <= 0.02:              # back-facing: hidden
            continue
        lambert = max(0.0, dot(n, light))
        # Brightness: half diffuse light, half how squarely the face meets the
        # viewer, so faces fade as they turn edge-on and the tilt reveals the
        # changing shading as the pyramid spins.
        shade = 0.10 + 0.55 * lambert + 0.55 * facing
        shade = min(1.0, shade)
        idx = int(round(shade * (len(SHADE_COLOURS) - 1)))
        faces.append((tv, idx))
    return faces


def render(cols, rows, spin, tilt, prev):
    """Rasterise the scene into braille cells; return list of output rows."""
    px_w, px_h = cols * 2, rows * 4
    if px_w <= 0 or px_h <= 0:
        return []

    # Fit the object (radius ~1.6) into the dot grid, keeping aspect ratio
    # square so the pyramid is not stretched.
    scale = min(px_w / 3.9, px_h / 3.9)
    cx, cy = px_w * 0.5, px_h * 0.5

    zbuf = [0.0] * (px_w * px_h)   # 0 = empty, else depth (larger = nearer)
    cbuf = [0] * (px_w * px_h)     # shade index per dot

    def project(p):
        sx = cx + p[0] * scale
        sy = cy - p[1] * scale
        return sx, sy, p[2]

    faces = build_faces(spin, tilt)
    for verts, shade in faces:
        pts = [project(v) for v in verts]
        # Screen-space signed area decides winding; fill both orientations.
        area = 0.0
        for i in range(len(pts)):
            x0, y0, _ = pts[i]
            x1, y1, _ = pts[(i + 1) % len(pts)]
            area += x0 * y1 - x1 * y0
        if abs(area) < 1e-6:
            continue

        min_x = max(0, int(math.floor(min(p[0] for p in pts))))
        max_x = min(px_w - 1, int(math.ceil(max(p[0] for p in pts))))
        min_y = max(0, int(math.floor(min(p[1] for p in pts))))
        max_y = min(px_h - 1, int(math.ceil(max(p[1] for p in pts))))
        if min_x > max_x or min_y > max_y:
            continue

        n = len(pts)
        for py in range(min_y, max_y + 1):
            sy = py + 0.5
            for px in range(min_x, max_x + 1):
                sx = px + 0.5
                inside = True
                zs = []
                for i in range(n):
                    ax, ay, az = pts[i]
                    bx, by, bz = pts[(i + 1) % n]
                    ex, ey = bx - ax, by - ay
                    w = ex * (sy - ay) - ey * (sx - ax)
                    if (w > 0.0) != (area > 0.0) and w != 0.0:
                        inside = False
                        break
                    seg = ex * ex + ey * ey
                    t = 0.0 if seg == 0 else ((sx - ax) * ex + (sy - ay) * ey) / seg
                    t = 0.0 if t < 0.0 else (1.0 if t > 1.0 else t)
                    zs.append(az + (bz - az) * t)
                if not inside:
                    continue
                z = sum(zs) / len(zs)
                i = py * px_w + px
                if z > zbuf[i]:
                    zbuf[i] = z
                    cbuf[i] = shade

    # Compose rows of braille characters.
    out = []
    for row in range(rows):
        chars = []
        colours = []
        for col in range(cols):
            bits = 0
            shade_sum = 0
            count = 0
            base_x = col * 2
            base_y = row * 4
            for dx in range(2):
                for dy in range(4):
                    i = (base_y + dy) * px_w + base_x + dx
                    if zbuf[i] > 0.0:
                        bits |= DOT_BITS[dx][dy]
                        shade_sum += cbuf[i]
                        count += 1
            if count:
                chars.append(chr(0x2800 + bits))
                colours.append(SHADE_COLOURS[shade_sum // count])
            else:
                chars.append(" ")
                colours.append(-1)
        out.append((chars, colours))

    return out


def encode(rows_data, prev_rows, rows, cols):
    """Build the escape-coded frame, only repainting changed cells."""
    parts = []
    last_colour = None
    for r, (chars, colours) in enumerate(rows_data):
        old = prev_rows[r] if r < len(prev_rows) else None
        col = 0
        while col < cols:
            if old is not None and col < len(old) and old[col] == (chars[col], colours[col]):
                col += 1
                continue
            start = col
            buf = []
            while col < cols and (old is None or col >= len(old) or old[col] != (chars[col], colours[col])):
                if colours[col] != last_colour:
                    buf.append("\x1b[38;5;%dm" % colours[col] if colours[col] >= 0 else RESET)
                    last_colour = colours[col]
                buf.append(chars[col])
                col += 1
            parts.append("\x1b[%d;%dH" % (r + 1, start + 1))
            parts.append("".join(buf))
    return "".join(parts)


def main():
    out = sys.stdout
    out.write(HIDE_CURSOR)
    out.flush()

    def restore(signum=None, frame=None):
        out.write(RESET + SHOW_CURSOR)
        out.flush()
        sys.exit(0)

    signal.signal(signal.SIGINT, restore)
    signal.signal(signal.SIGTERM, restore)

    frame_time = 1.0 / TARGET_FPS
    spin = 0.0
    tilt = 0.0
    t0 = time.monotonic()
    prev_rows = []
    prev_size = None
    next_tick = t0

    try:
        while True:
            size = shutil.get_terminal_size()
            cols, rows = size.columns, size.lines
            if (cols, rows) != prev_size:
                out.write(CLEAR)
                prev_rows = []
                prev_size = (cols, rows)

            now = time.monotonic() - t0
            spin = now * 1.15
            tilt = 0.30 + 0.26 * math.sin(now * 0.35)

            rows_data = render(cols, rows, spin, tilt, prev_rows)
            frame = encode(rows_data, prev_rows, rows, cols)
            if frame:
                out.write(frame)
                out.flush()
            prev_rows = [(list(c), list(k)) for c, k in rows_data]

            next_tick += frame_time
            delay = next_tick - time.monotonic()
            if delay > 0:
                time.sleep(delay)
            else:
                next_tick = time.monotonic()
    except (KeyboardInterrupt, SystemExit):
        pass
    finally:
        out.write(RESET + SHOW_CURSOR)
        out.flush()


if __name__ == "__main__":
    main()
