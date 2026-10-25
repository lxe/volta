#!/usr/bin/env python3
"""Animate a shaded, rotating 3D torus using Unicode braille characters."""

import math
import shutil
import sys
import time

# Braille dot bit for each (row, col) inside a 2x4 cell.
DOT_BITS = (
    (0x01, 0x08),
    (0x02, 0x10),
    (0x04, 0x20),
    (0x40, 0x80),
)

# 4x4 Bayer matrix for ordered dithering (values 0..15).
BAYER = (
    (0, 8, 2, 10),
    (12, 4, 14, 6),
    (3, 11, 1, 9),
    (15, 7, 13, 5),
)

R1 = 1.0   # tube radius
R2 = 2.0   # ring radius
K2 = 5.0   # perspective distance factor
# Light direction (normalized), from upper-front-left.
LX, LY, LZ = -0.5, 0.7, 0.5
_ln = math.sqrt(LX * LX + LY * LY + LZ * LZ)
LX, LY, LZ = LX / _ln, LY / _ln, LZ / _ln

# Coverage remap: luminance in [0,1] -> dot coverage in [MIN_COV, 1].
MIN_COV = 0.40
COV_SPAN = 1.0 - MIN_COV


def render_frame(cols, rows, theta, phi):
    """Return the frame as a list of strings."""
    width = cols * 2
    height = rows * 4
    cx = width / 2.0
    cy = height / 2.0
    # Scale so the torus roughly fills the viewport.
    scale = min(width, height) / (2.0 * (R2 + R1) * 1.12)

    cos_t, sin_t = math.cos(theta), math.sin(theta)
    cos_p, sin_p = math.cos(phi), math.sin(phi)

    zbuf = [None] * (width * height)
    lum_buf = [0.0] * (width * height)

    # Oversample relative to sub-pixel resolution so coverage is accurate.
    n_u = max(120, width * 3)
    n_v = max(60, height * 2)

    # Precompute rotation of the torus point and normal in two stages.
    # Stage 1 (about Y): combine with stage 2 (about X) into a direct matrix
    # so each sample is just a few multiplies.
    # Direct rotation matrix R = Rx(phi) * Ry(theta).
    ct, st = cos_t, sin_t
    cp, sp = cos_p, sin_p
    # Ry(theta): (x,y,z) -> (x*ct + z*st, y, -x*st + z*ct)
    # Rx(phi):  (x,y,z) -> (x, y*cp - z*sp, y*sp + z*cp)
    # Combined coefficients:
    m00 = ct
    m02 = st
    m10 = sp * st
    m11 = cp
    m12 = -sp * ct
    m20 = -cp * st
    m21 = sp
    m22 = cp * ct

    # Precompute trig tables for u and v to avoid recomputing sin/cos.
    cu_tab = [math.cos(2.0 * math.pi * i / n_u) for i in range(n_u)]
    su_tab = [math.sin(2.0 * math.pi * i / n_u) for i in range(n_u)]
    cv_tab = [math.cos(2.0 * math.pi * j / n_v) for j in range(n_v)]
    sv_tab = [math.sin(2.0 * math.pi * j / n_v) for j in range(n_v)]

    # Bind locals for speed inside the hot loop.
    zb = zbuf
    lb = lum_buf
    W = width
    H = height

    for i in range(n_u):
        cos_u = cu_tab[i]
        sin_u = su_tab[i]
        for j in range(n_v):
            cos_v = cv_tab[j]
            sin_v = sv_tab[j]

            # Point on torus (before rotation).
            rr = R2 + R1 * cos_v
            x = rr * cos_u
            y = rr * sin_u
            z = R1 * sin_v

            # Normal (before rotation).
            cvcu = cos_v * cos_u
            cvsu = cos_v * sin_u

            # Rotate point.
            x2 = m00 * x + m02 * z
            y2 = m10 * x + m11 * y + m12 * z
            z2 = m20 * x + m21 * y + m22 * z
            # Rotate normal.
            nx2 = m00 * cvcu + m02 * sin_v
            ny2 = m10 * cvcu + m11 * cvsu + m12 * sin_v
            nz2 = m20 * cvcu + m21 * cvsu + m22 * sin_v

            # Perspective projection.
            denom = K2 - z2
            if denom <= 0.2:
                continue
            f = K2 / denom
            sx = int(cx + x2 * f * scale + 0.5)
            sy = int(cy - y2 * f * scale + 0.5)
            if sx < 0 or sx >= W or sy < 0 or sy >= H:
                continue

            idx = sy * W + sx
            prev = zb[idx]
            if prev is not None and prev >= z2:
                continue
            zb[idx] = z2

            # Lambert shading.
            lum = nx2 * LX + ny2 * LY + nz2 * LZ
            if lum < 0.0:
                lum = 0.0
            # Ambient so back-facing areas aren't pure black.
            lb[idx] = 0.10 + 0.90 * lum

    # Compose braille cells.  Each sub-pixel is drawn if its luminance
    # coverage exceeds the ordered-dither threshold, giving a solid body
    # whose brightness reads as dot density.
    # Precompute a threshold per (sub-pixel row, col) inside a cell.
    THR = [[(BAYER[r][c] + 0.5) / 16.0 for c in range(2)] for r in range(4)]
    DBITS = DOT_BITS
    zb = zbuf
    lb = lum_buf
    W = width
    lines = []
    for cy_row in range(rows):
        chars = []
        base_y = cy_row * 4
        for cx_col in range(cols):
            bits = 0
            base_x = cx_col * 2
            for r in range(4):
                row_off = (base_y + r) * W + base_x
                drow = THR[r]
                dbits = DBITS[r]
                for c in range(2):
                    idx = row_off + c
                    if zb[idx] is None:
                        continue
                    cov = MIN_COV + COV_SPAN * lb[idx]
                    if cov > drow[c]:
                        bits |= dbits[c]
            if bits == 0:
                chars.append(" ")
            else:
                chars.append(chr(0x2800 + bits))
        lines.append("".join(chars))
    return lines


def main():
    out = sys.stdout
    try:
        out.write("\x1b[?25l")  # hide cursor
        out.write("\x1b[?7l")   # disable line wrap
        out.flush()
        target_dt = 1.0 / 30.0
        start = time.monotonic()
        while True:
            t0 = time.monotonic()
            size = shutil.get_terminal_size()
            cols, rows = size.columns, size.lines
            if cols <= 0 or rows <= 0:
                cols, rows = 80, 24

            elapsed = time.monotonic() - start
            theta = elapsed * 1.1
            phi = elapsed * 0.7

            lines = render_frame(cols, rows, theta, phi)

            buf = ["\x1b[H"]
            for line in lines:
                buf.append(line)
                buf.append("\r\n")
            buf.append("\x1b[0J")
            out.write("".join(buf))
            out.flush()

            dt = time.monotonic() - t0
            sleep = target_dt - dt
            if sleep > 0:
                time.sleep(sleep)
    except KeyboardInterrupt:
        pass
    finally:
        try:
            out.write("\x1b[?7h")   # restore line wrap
            out.write("\x1b[?25h")  # restore cursor
            out.write("\x1b[0m")
            out.flush()
        except Exception:
            pass


if __name__ == "__main__":
    main()
