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

# 4x4 Bayer ordered-dither matrix, normalized to (0, 1).
BAYER4 = (
    (0, 8, 2, 10),
    (12, 4, 14, 6),
    (3, 11, 1, 9),
    (15, 7, 13, 5),
)
BAYER = tuple(tuple((v + 0.5) / 16.0 for v in row) for row in BAYER4)

MAJOR_R = 1.0    # torus ring radius
MINOR_R = 0.42   # tube radius
FPS = 30.0


def main():
    out = sys.stdout
    out.write("\x1b[?25l\x1b[2J")
    out.flush()

    def restore(*_):
        out.write("\x1b[?25h\x1b[0m\n")
        out.flush()
        sys.exit(0)

    signal.signal(signal.SIGINT, restore)
    signal.signal(signal.SIGTERM, restore)

    # Normalized light direction.
    light = (-0.45, 0.55, 0.70)
    ln = math.sqrt(sum(c * c for c in light))
    lx, ly, lz = (c / ln for c in light)

    start = time.monotonic()
    try:
        while True:
            frame_start = time.monotonic()
            elapsed = frame_start - start

            cols, rows = shutil.get_terminal_size(fallback=(80, 24))
            cols = max(cols, 1)
            rows = max(rows, 1)

            W = cols * 2          # pixel (dot) width
            H = rows * 4          # pixel (dot) height
            extent = MAJOR_R + MINOR_R
            scale = 0.94 * min(W / 2.0, H / 2.0) / extent
            cx = W / 2.0
            cy = H / 2.0

            # Per-dot depth buffer and per-dot brightness buffer.
            zbuf = [[-1e18] * W for _ in range(H)]
            lbuf = [[0.0] * W for _ in range(H)]

            # Rotation angles about two axes.
            a = elapsed * 1.1     # around vertical (Y)
            b = elapsed * 0.7     # around horizontal (X)
            ca, sa = math.cos(a), math.sin(a)
            cb, sb = math.cos(b), math.sin(b)

            # Sample density: enough points to cover every dot.
            NU = max(64, int(2 * math.pi * extent * scale * 1.7))
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

                    rx = (MAJOR_R + MINOR_R * cv) * cu
                    ry = MINOR_R * sv
                    rz = (MAJOR_R + MINOR_R * cv) * su

                    # Normal (before rotation): (cv*cu, sv, cv*su)
                    nx0 = cv * cu
                    ny0 = sv
                    nz0 = cv * su

                    # Rotate about Y then X (positions and normals).
                    rx, rz = rx * ca + rz * sa, -rx * sa + rz * ca
                    nx0, nz0 = nx0 * ca + nz0 * sa, -nx0 * sa + nz0 * ca
                    ry, rz = ry * cb - rz * sb, ry * sb + rz * cb
                    ny0, nz0 = ny0 * cb - nz0 * sb, ny0 * sb + nz0 * cb

                    depth = rz
                    persp = 1.0 / (1.0 + 0.18 * (2.2 - depth))
                    px = cx + rx * scale * persp
                    py = cy - ry * scale * persp

                    ix = int(px)
                    iy = int(py)
                    if ix < 0 or ix >= W or iy < 0 or iy >= H:
                        continue
                    if depth <= zbuf[iy][ix]:
                        continue
                    zbuf[iy][ix] = depth

                    # Actual visible normal lighting.
                    lum = nx0 * lx + ny0 * ly + nz0 * lz
                    lum = max(0.0, min(1.0, lum * 0.85 + 0.15))
                    lum *= 0.55 + 0.45 * persp  # depth cue
                    lbuf[iy][ix] = lum

            # Compose: per-dot ordered dither decides whether the dot is drawn.
            lines = []
            for row in range(rows):
                chars = []
                for col in range(cols):
                    bits = 0
                    for dy in range(4):
                        py = row * 4 + dy
                        brow = lbuf[py]
                        zrow = zbuf[py]
                        for dx in range(2):
                            px = col * 2 + dx
                            if zrow[px] <= -1e17:
                                continue
                            if brow[px] > BAYER[py & 3][px & 3]:
                                bits |= DOT_BITS[dy][dx]
                    chars.append(chr(0x2800 + bits) if bits else " ")
                lines.append("".join(chars))

            out.write("\x1b[H" + "\n".join(lines))
            out.flush()

            spent = time.monotonic() - frame_start
            delay = (1.0 / FPS) - spent
            if delay > 0:
                time.sleep(delay)
    finally:
        out.write("\x1b[?25h\x1b[0m\n")
        out.flush()


if __name__ == "__main__":
    main()
