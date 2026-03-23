# WebGPU Flocking Optimization — Autoresearch Program

## Objective
Maximize the number of 3D boids that can be simulated AND rendered at 60 FPS (p99 < 16.6ms).
The compute side is already well-optimized. **Focus on rendering performance.**

## The Loop
1. Read this file and the current `shader.wgsl`, `render.wgsl`, and `simulation.js`
2. Pick **ONE** optimization to try (see ideas below)
3. Edit the target files
4. `git add shader.wgsl render.wgsl simulation.js && git commit -m "experiment: <description>"`
5. Run: `python3 evaluate.py`
6. Parse the last line of stdout for `max_boids: NNNNN`
7. If max_boids **>** previous best → **KEEP** the commit, append to `results.tsv`
8. If max_boids **≤** previous best → `git revert --no-edit HEAD`, append to `results.tsv`
9. **REPEAT.** Do NOT pause to ask the human. Loop until interrupted.

## What You Can Edit
- `shader.wgsl` — compute kernels
- `render.wgsl` — vertex/fragment shaders (THE MAIN BOTTLENECK)
- `simulation.js` — buffer layout, dispatch config
- `renderer.js` — render pipeline config, draw calls

## What You CANNOT Edit
- `benchmark.ts`, `evaluate.py` — locked evaluation
- `index.html` — UI
- `dashboard.py`, `serve.py`, `bloom.wgsl` — infrastructure
- `program.md` — only humans edit this

## CRITICAL: git revert
When reverting, use `git revert --no-edit HEAD`. Do NOT use `git reset --hard`.

## Results Tracking
Append to `results.tsv`:
`<experiment_number>\t<max_boids>\t<description>\t<kept|reverted>`

## Rendering Optimization Ideas (priority order)
1. **Simplify colormap in vertex shader** — the 10-way switch with 5 stops each is expensive. Precompute a lookup or use a simpler formula.
2. **Reduce vertex shader math** — the billboard vs_billboard does projection, NDC direction, perpendicular, stretch, colormap, gain, HDR boost. Simplify.
3. **Move colormap to fragment shader** — compute color per-pixel instead of per-vertex (fewer invocations for large triangles).
4. **Reduce vertex count** — billboards use 6 verts. Could use 3 (single triangle covering a quad area).
5. **Skip invisible boids** — boids behind camera produce degenerate quads but still run the vertex shader. Add clip_center.w < 0 early exit.
6. **Reduce overdraw** — make distant billboards smaller so they cover fewer pixels.
7. **Simplify the tetrahedron vertex shader** — the rotation matrix construction + face normal computation is heavy. Could precompute or simplify.
8. **Use flat shading** — avoid per-vertex normal computation, use provoking vertex.
9. **Reduce precision** — use f16 where possible for colors/UVs.

## Constraints
- Boid struct: pos(vec3f), size_factor(f32), vel(vec3f), speed(f32), heading(vec3f), neighbor_count(f32), dir_change(f32), flock_alignment(f32), sep_pressure(f32), density(f32) — 64 bytes
- SimParams: 24 fields, 96 bytes
- CameraUniforms in render.wgsl: 128 bytes
- Grid size auto-scales: `round(cbrt(numBoids) * 0.8)`, clamped 8-128
- Must produce correct visual output (colored boids in 3D)
- All changes must be a single atomic experiment
