# Full-Pipeline Performance Optimization — Autoresearch Program

## Objective
Maximize the number of boids that render at 60 FPS in a REAL BROWSER with full rendering (compute + vertex shaders + fragment shaders + MSAA + present).

The benchmark uses Playwright to open Chrome, load the simulation with clustered boids (15s warmup), and measure actual FPS over 8 seconds. This captures the REAL bottleneck — not just compute.

## The Loop
1. Read `shader.wgsl`, `render.wgsl`, `simulation.js`, `renderer.js`
2. Pick ONE optimization to try
3. Edit the file(s)
4. `git add shader.wgsl render.wgsl simulation.js renderer.js && git commit -m "experiment: <description>"`
5. Run: `python3 evaluate.py`
6. Parse the last line for `max_boids: NNNNN`
7. If max_boids > previous best → KEEP, append to results.tsv
8. If max_boids ≤ previous best → `git revert --no-edit HEAD`, append to results.tsv
9. REPEAT. Do NOT pause. Loop until interrupted.

## What You Can Edit
- `shader.wgsl` — compute kernels (flock_radius is the active one for perf testing)
- `render.wgsl` — vertex and fragment shaders (THE MAIN BOTTLENECK)
- `simulation.js` — buffer layout, grid sizing, pipeline config
- `renderer.js` — render pipeline config, MSAA, draw calls

## What You CANNOT Edit
- `bench_browser.js`, `evaluate.py` — locked evaluation
- `index.html` — UI (the benchmark sets mode via page.evaluate)
- `dashboard.py`, `serve.py`, `bloom.wgsl` — infrastructure
- `program.md` — only humans edit this
- Do NOT add roost attractors or global orbit forces

## CRITICAL: git revert
Use `git revert --no-edit HEAD`. NOT `git reset --hard`.

## Results Tracking
Append to results.tsv: `<experiment_number>\t<max_boids>\t<description>\t<kept|reverted>`

## Performance Analysis
The benchmark tests with Billboard Additive + Radius mode. Current bottlenecks:
1. **Vertex shader** in render.wgsl (vs_billboard) — runs per-vertex with colormap switch
2. **Fragment overdraw** — additive blending means every pixel drawn, no early-Z
3. **Compute: flock_radius** — neighbor iteration in dense cells
4. **Grid sizing** — worldSize/gridSize ratio affects cell density
5. **writeBuffer calls** — uniform upload overhead

## Optimization Ideas
### Render (highest impact):
1. Simplify vs_billboard — remove pow(), reduce colormap to 3 stops instead of 5
2. Reduce vertex count — 3 verts (triangle) instead of 6 (quad)
3. Shrink distant billboards — scale by 1/clip.w to reduce pixel coverage
4. Skip behind-camera boids — already done but verify
5. Move colormap to fragment shader (fewer invocations for large particles)

### Compute:
6. Reduce neighbor cap in flock_radius (16 → 8)
7. Optimize grid: ensure cells aren't too dense when boids cluster
8. Remove unnecessary per-boid metric writes (speed, dir_change, etc.)
9. Simplify turn rate limiter

### Pipeline:
10. Remove MSAA for billboard modes entirely
11. Use render bundles for static pipeline state
12. Batch uniform writes

## Constraints
- Billboard Additive mode is what's tested (no MSAA, no depth)
- Radius neighbor mode is what's tested (not topological)
- Must produce visible colored boids (not blank screen)
- The benchmark runs `page.evaluate` to set modes — don't rename dropdown IDs
