#!/usr/bin/env python3
"""Rotating, normal-shaded 3D torus rendered with Unicode braille characters."""

import math
import shutil
import signal
import sys
import time

# Braille dot bit values for a 2x4 grid:
#   (0,0)=0x01 (1,0)=0x08
#   (0,1)=0x02 (1,1)=0x10
#   (0,2)=0x04 (1,2)=0x20
#   (0,3)=0x40 (1,3)=0x80
DOT_BITS = (
    (0x01, 0x08),
    (0x02, 0x10),
    (0x04, 0x20),
    (0x40, 0x80),
)

# Shading ramp from dark to bright (braille blank is skipped).
RAMP = " .:-=+*#%@"

MAJOR_R = 1.0    # torus ring radius
MINOR_R = 0.42   # tube radius
FPS = 30.0


def main():
    # Hide cursor, clear screen.
    out = sys.stdout
    out.write("\x1b[?25l\x1b[2J")
    out.flush()

    def restore(*_):
        out.write("\x1b[?25h\x1b[0m\n")
        out.flush()
        sys.exit(0)

    signal.signal(signal.SIGINT, restore)
    signal.signal(signal.SIGTERM, restore)

    light = (-0.45, 0.55, 0.70)
    ln = math.sqrt(sum(c * c for c in light))
    lx, ly, lz = (c / ln for c in light)

    start = time.monotonic()
    frame = 0
    try:
        while True:
            frame_start = time.monotonic()
            elapsed = frame_start - start

            cols, rows = shutil.get_terminal_size(fallback=(80, 24))
            cols = max(cols, 1)
            rows = max(rows, 1)

            W = cols * 2          # pixel width
            H = rows * 4          # pixel height
            aspect = W / H        # pixels per pixel; braille dots are ~2:1 tall

            # Rotation angles about two axes.
            a = elapsed * 1.1     # around vertical (Y)
            b = elapsed * 0.7     # around horizontal (X)
            ca, sa = math.cos(a), math.sin(a)
            cb, sb = math.cos(b), math.sin(b)

            # Fit torus (outer radius ~ MAJOR_R + MINOR_R) into view.
            extent = MAJOR_R + MINOR_R
            scale = 0.94 * min(W / 2.0, H / 2.0) / extent

            cx = W / 2.0
            cy = H / 2.0

            # Character grid: (char_code, brightness) per cell.
            grid = [[0] * cols for _ in range(rows)]
            zbuf = [[-1e18] * cols for _ in range(rows)]

            # Sample the torus surface on a (u, v) grid.
            # Sample density: enough points to cover every pixel dot.
            NU = max(64, int(2 * math.pi * extent * scale * 1.6))
            NV = max(32, NU // 2)
            NU = min(NU, 400)
            NV = min(NV, 200)
            du = 2 * math.pi / NU
            dv = 2 * math.pi / NV

            for i in range(NU):
                u = i * du
                cu, su = math.cos(u), math.sin(u)
                for j in range(NV):
                    v = j * dv
                    cv, sv = math.cos(v), math.sin(v)

                    # Point on torus.
                    rx = (MAJOR_R + MINOR_R * cv) * cu
                    ry = MINOR_R * sv
                    rz = (MAJOR_R + MINOR_R * cv) * su

                    # Normal (before rotation): (cv*cu, sv, cv*su)
                    nx0 = cv * cu
                    ny0 = sv
                    nz0 = cv * su

                    # Rotate about Y then X.
                    # Y rotation
                    rx, rz = rx * ca + rz * sa, -rx * sa + rz * ca
                    nx0, nz0 = nx0 * ca + nz0 * sa, -nx0 * sa + nz0 * ca
                    # X rotation
                    ry, rz = ry * cb - rz * sb, ry * sb + rz * cb
                    ny0, nz0 = ny0 * cb - nz0 * sb, ny0 * sb + nz0 * cb

                    # Project (perspective-lite: scale by depth).
                    depth = rz
                    persp = 1.0 / (1.0 + 0.18 * (2.2 - depth))
                    px = cx + rx * scale * persp
                    py = cy - ry * scale * persp

                    ix = int(px)
                    iy = int(py)
                    if ix < 0 or ix >= W or iy < 0 or iy >= H:
                        continue
                    col, row = ix >> 1, iy >> 2
                    if depth <= zbuf[row][col]:
                        continue
                    zbuf[row][col] = depth

                    # Shade by normal dot light.
                    lum = nx0 * lx + ny0 * ly + nz0 * lz
                    lum = max(0.0, min(1.0, lum * 0.85 + 0.15))
                    lum *= 0.55 + 0.45 * persp  # depth cue

                    dot = DOT_BITS[iy & 3][ix & 1]
                    grid[row][col] |= dot | (int(lum * 7) << 8)

            # Compose output.
            lines = []
            for row in range(rows):
                chars = []
                rowdata = grid[row]
                for cell in rowdata:
                    bits = cell & 0xFF
                    if not bits:
                        chars.append(" ")
                    else:
                        shade = (cell >> 8) & 7
                        chars.append(chr(0x2800 + bits))
                lines.append("".join(chars))

            out.write("\x1b[H" + "\n".join(lines))
            out.flush()
            frame += 1

            # Pace to ~30 fps.
            spent = time.monotonic() - frame_start
            delay = (1.0 / FPS) - spent
            if delay > 0:
                time.sleep(delay)
    finally:
        out.write("\x1b[?25h\x1b[0m\n")
        out.flush()


if __name__ == "__main__":
    main()
