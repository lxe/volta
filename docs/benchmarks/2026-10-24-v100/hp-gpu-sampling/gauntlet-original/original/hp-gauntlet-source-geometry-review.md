# HP gauntlet renderer source review

All four final files contain actual 3D geometry or analytic 3D surface reconstruction, correct Unicode Braille packing, animation, per-frame terminal-size reads and Ctrl-C cursor restoration. The torus does not satisfy its requested surface-normal shading: its calculated luminance never reaches the visible output. Source review therefore does not support an unqualified claim that all four completed every requested rendering requirement. The parent separately owns live-process and pixel validation.

## Cube

Eight corners and six explicit face normals define a solid cube. Points and normals rotate about two axes. Orthographic projection, front-face culling and scanline polygon fills produce solid faces. A fixed light drives face brightness, then ordered dithering converts brightness into Braille dot coverage. This is visible normal-dependent shading, although the task literally asks for shading by viewer-facing orientation. The selected scale fits the unrotated cube but can clip its larger rotated extent. The painter sort uses normal z rather than face depth; for a convex cube with back-face culling, distinct front-face interiors should not overlap. The program rereads terminal dimensions, targets 30 FPS and restores the cursor in finally on Ctrl-C. Very small terminals are clamped upward to ten columns and five rows, so resizing below that size is not faithfully fitted.

## Square pyramid

An apex, square base, four triangular sides and a bottom face define a genuine square pyramid. Spin and slowly varying tilt transform its vertices and normals. Orthographic projection and convex-polygon tests rasterize the faces, with viewer-facing and Lambert terms selecting visible terminal colours. The standard Braille mapping is correct. Its depth approximation has limits: the buffer starts at zero, so valid surface samples with nonpositive z are rejected, and averaging projected edge depths is not plane interpolation. The changed-cell optimization also compares the previous row's two component lists against per-cell tuples, so it effectively repaints cells rather than reliably suppressing unchanged output. It still rereads dimensions and handles Ctrl-C/SIGTERM plus finally cleanup.

## Torus

The parameterization uses a major radius of 1.0 and tube radius of 0.42, with correct analytic surface normals. Both positions and normals rotate about two axes before depth-dependent perspective projection. However, lines 129–135 store the normal-derived luminance only in upper grid bits. Lines 143–148 emit the lower eight dot bits, while the extracted shade variable is unused. Neither dot visibility nor terminal colour depends on that luminance, so the requested shading is absent. The depth buffer is also one value per character cell rather than per dot. A nearer sample can suppress a different dot in that cell, while OR operations retain previously accepted dots. This can produce sampling/occlusion artifacts even though the underlying surface is a torus. Size adaptation, frame pacing and signal/finally cursor cleanup are present.

## Sphere

For each projected disk point, nz=sqrt(1-nx²-ny²) reconstructs the visible unit-sphere hemisphere. Its normal supplies Lambert lighting; latitude bands and an angle-offset longitude pattern make rotation visible. Ordered dithering controls the standard Braille dot bits. Fractional centre coordinates and reflected velocities implement motion against all four boundaries, with radius-aware bounds recomputed after resizing. Ctrl-C/SIGTERM restore the cursor. This program has no finally block for unrelated exceptions, and minimum dimension clamps can exceed extremely small panes, but no material sphere geometry or requested shading defect was identified statically.

## Evidence boundary

The JSON companion records SHA-256 hashes of the exact four final render.py files. This review read source and TASK.md only. It did not import, execute, repair or restart the renderers, inspect remote state, or run GPU work. Source presence of a 30 FPS target does not prove measured frame rate, and source geometry does not establish visible quality in a live pane.
