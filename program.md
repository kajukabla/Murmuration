# Compute Shader Optimization: Dense Cluster Performance

## Objective
Maximize boid count at 60 FPS when boids are **clustered together** (worst case).
The rendering is already lean (billboard additive, no MSAA). The bottleneck is the
**compute shader** when boids form dense clusters — the flock_radius pass slows down
because cells become densely packed and each boid iterates too many neighbors.

Current baseline: **118,000 boids** (browser-measured, clustered steady state).

## The Loop
1. Read `shader.wgsl` and `simulation.js`
2. Pick ONE compute optimization
3. Edit the file(s)
4. `git add shader.wgsl simulation.js && git commit -m "experiment: <description>"`
5. Run: `python3 evaluate.py`
6. Parse the last line for `max_boids: NNNNN`
7. If max_boids > previous best → KEEP, append to `results_perf.tsv`
8. If max_boids ≤ previous best → `git revert --no-edit HEAD`, append to `results_perf.tsv`
9. REPEAT. Do NOT pause. Loop until interrupted.

## What You Can Edit
- `shader.wgsl` — compute kernels, especially `flock_radius`
- `simulation.js` — grid sizing, buffer layout, pipeline config

## What You CANNOT Edit
- `bench_browser.js`, `evaluate.py` — locked evaluation
- `render.wgsl`, `renderer.js`, `index.html` — rendering/UI
- `dashboard.py`, `serve.py` — infrastructure
- `program.md` — only humans edit this
- Do NOT add roost attractors or global orbit forces

## CRITICAL RULES
- Use `git revert --no-edit HEAD` to undo. NOT `git reset --hard`.
- Results go in `results_perf.tsv` (NOT results.tsv)

## The Problem
When boids cluster (which they always do after 30 seconds), cells become dense.
With gridSize based on `cbrt(numBoids/4)`, average is ~4 boids/cell. But in a
cluster, cells can have 50-100 boids. The flock_radius pass iterates ALL boids
in 27 neighboring cells. At 50 boids/cell × 27 = 1350 distance checks per boid.

## Optimization Ideas (priority order)
1. **Hard neighbor cap** — break out of the inner loop after finding 8-10 neighbors.
   Currently caps at 16 but that's still too many per cell iteration.
2. **Early cell skip** — skip cells with 0 boids immediately (already uses `start >= end_val`)
3. **Reduce per-neighbor work** — the distance check + separation calculation has branches.
   Simplify to branchless min/max operations.
4. **Smaller visual range** — clamp visual range so fewer cells need checking.
   Currently clamped to cellSize*1.4 but could be tighter.
5. **Workgroup-local caching** — load cell data into shared memory before iterating
6. **Sort boids by cell** — the scatter pass already does this but the flock pass
   reads from boids_src[] not sorted order. Could restructure to read sorted.
7. **Reduce output writes** — flock_radius writes 10 fields per boid. Some (dir_change,
   flock_alignment, sep_pressure, density) are only for visualization. Skip them.
8. **Simpler turn rate limiter** — the normalize+mix+clamp chain is expensive.
   Could use a fast approximation.
9. **Grid sizing** — experiment with different grid formulas. Current: cbrt(N/4).
   Try: based on sphere_radius and visual_range instead of boid count.
10. **Merge passes** — combine clear_grid + assign_cells into one dispatch.

## Constraints
- The `flock_radius` function must produce correct flocking (separation, alignment, cohesion)
- Boid struct: 64 bytes (16 floats) — don't change the layout (render.wgsl depends on it)
- Must work with billboard additive render mode + radius neighbor mode
- The benchmark runs in a real browser with 30s warmup for clustering
