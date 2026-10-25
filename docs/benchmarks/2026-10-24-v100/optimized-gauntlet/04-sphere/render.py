#!/usr/bin/env python3
"""Shaded 3D sphere bouncing inside the terminal, drawn with braille.

Each character cell is a 2x4 dot grid, so the pixel resolution is
(cols*2) x (rows*4).  The sphere is ray-traced per dot with a rotating
latitude/longitude grid, Lambert shading and a moving highlight.  It
bounces off all four terminal edges at ~30 fps.  Ctrl-C restores the
cursor and clears the screen.
"""

import math
import shutil
import sys
import time

FPS = 30
TWO_PI = 2.0 * math.pi

# Braille dot bit for each (row, col) inside a 2x4 cell.
DOT_BITS = [
    (0x01, 0x08),  # row 0: dots 1, 4
    (0x02, 0x10),  # row 1: dots 2, 5
    (0x04, 0x20),  # row 2: dots 3, 6
    (0x40, 0x80),  # row 3: dots 7, 8
]

SHADE = " .:-=+*#%@"  # brightness ramp (dark -> bright)


def rot_y(x, y, z, a):
    c, s = math.cos(a), math.sin(a)
    return x * c + z * s, y, -x * s + z * c


def rot_x(x, y, z, a):
    c, s = math.cos(a), math.sin(a)
    return x, y * c - z * s, y * s + z * c


def render(cols, rows, cx, cy, radius, angle, light):
    """Return the frame as a list of strings.

    cx, cy are the sphere centre in pixel coordinates; pixels are
    cols*2 wide and rows*4 tall.
    """
    inv_r = 1.0 / radius
    lx, ly, lz = light
    ca, sa = math.cos(angle), math.sin(angle)
    tilt = 0.5
    ct, st = math.cos(tilt), math.sin(tilt)

    lines = []
    for row in range(rows):
        chars = []
        for col in range(cols):
            bits = 0
            for sub_y in range(4):
                dy = (row * 4 + sub_y + 0.5 - cy) * inv_r
                if dy < -1.0 or dy > 1.0:
                    continue
                for sub_x in range(2):
                    dx = (col * 2 + sub_x + 0.5 - cx) * inv_r
                    q = dx * dx + dy * dy
                    if q > 1.0:
                        continue
                    dz = math.sqrt(1.0 - q)

                    # Rotate into sphere space (spin about Y, tilt about X).
                    x, y, z = dx, dy, dz
                    x, y, z = x * ca + z * sa, y, -x * sa + z * ca
                    y, z = y * ct - z * st, y * st + z * ct

                    lam = dx * lx + dy * ly + dz * lz
                    shade = max(lam, 0.0)

                    lat = math.asin(max(-1.0, min(1.0, y)))
                    lon = math.atan2(x, z)
                    grid = False
                    if abs(math.sin(lat * 3.0)) < 0.08:      # parallels
                        grid = True
                    elif abs(math.sin(lon * 4.0)) < 0.08:    # meridians
                        grid = True
                    if grid:
                        shade = min(1.0, shade + 0.6)
                    shade = min(1.0, shade + 0.8 * max(lam, 0.0) ** 24)

                    dither = ((sub_x * 2 + sub_y * 3) % 5) / 5.0 * 0.18
                    if grid or shade > 0.42 + dither:
                        bits |= DOT_BITS[sub_y][sub_x]
            chars.append(chr(0x2800 + bits))
        lines.append("".join(chars))
    return lines


def main():
    out = sys.stdout
    out.write("\x1b[?25l")  # hide cursor
    out.flush()
    try:
        t0 = time.monotonic()
        frame_time = 1.0 / FPS
        next_t = t0
        last_size = (0, 0)
        prev_lines = None

        while True:
            now = time.monotonic()
            t = now - t0

            size = shutil.get_terminal_size(fallback=(80, 24))
            cols, rows = size.columns, size.lines
            if (cols, rows) != last_size:
                out.write("\x1b[0m\x1b[2J\x1b[H")
                prev_lines = None
                last_size = (cols, rows)

            pw, ph = cols * 2, rows * 4
            # Sphere radius grows with the smaller screen dimension; it
            # must fit fully inside the pane for clean edge bounces.
            radius = max(2.0, min(pw, ph) * 0.30)

            # Bounce position: travel across the free range each axis.
            span_x = max(1.0, pw - 2.0 * radius)
            span_y = max(1.0, ph - 2.0 * radius)
            sx = abs((t * 0.55 + 0.13) % 2.0 - 1.0)   # triangle wave 0..1
            sy = abs((t * 0.41 + 0.61) % 2.0 - 1.0)
            cx = radius + sx * span_x
            cy = radius + sy * span_y

            angle = t * 1.6
            # Light orbits a little so the terminator sweeps around.
            la = 0.6 * math.sin(t * 0.7)
            light = (0.6 * math.cos(la), 0.5, 0.6 * math.sin(la) + 0.35)
            ln = math.sqrt(sum(c * c for c in light))
            light = (light[0] / ln, light[1] / ln, light[2] / ln)

            lines = render(cols, rows, cx, cy, radius, angle, light)

            # Compose with cursor-home and full repaint each frame.
            buf = ["\x1b[H"]
            if prev_lines is None:
                buf.append("\n".join(lines))
            else:
                for i, line in enumerate(lines):
                    if i < len(prev_lines) and prev_lines[i] == line:
                        buf.append("\x1b[%d;1H" % (i + 1))
                        buf.append(line)
                    else:
                        buf.append("\x1b[%d;1H%s" % (i + 1, line))
            prev_lines = lines
            out.write("".join(buf))
            out.flush()

            next_t += frame_time
            delay = next_t - time.monotonic()
            if delay > 0:
                time.sleep(delay)
            else:
                next_t = time.monotonic()  # fell behind; resync
    except KeyboardInterrupt:
        pass
    finally:
        out.write("\x1b[0m\x1b[?25h\x1b[2J\x1b[H")  # restore cursor, clear
        out.flush()


if __name__ == "__main__":
    main()
