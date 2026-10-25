# Repaired torus source review

The final torus uses H×W depth and intensity buffers, with one entry for each Braille dot. When a nearer projected surface sample wins its depth test, it replaces that dot's stored normal-derived intensity. Output tests the intensity against a 4×4 Bayer threshold and sets the corresponding Braille bit only when the threshold is exceeded. The calculated lighting now directly controls visible dot density, and the old per-character depth/retained-bit mechanism is gone.

The original analytic torus parameterization, transformed surface normals, two-axis rotation and depth-dependent projection remain present. The renderer rereads terminal size each frame, has frame pacing and restores the cursor through signal handlers and finally. These source checks establish that both reported torus defects were repaired; they do not measure FPS or prove every possible sampling/resize condition. A finite surface sample grid can leave small holes. This review did not import or execute render.py. Live process and three changed-frame checks are recorded separately by the parent's validator.

The cube and pyramid were not repaired. The original review's cube clipping and pyramid depth/coverage limitations remain, so this repair does not establish universal satisfaction of every four-shape requirement. No private desktop screenshot is included in this archive.
