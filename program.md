# Compute Shader Optimization: Dense Cluster Performance

## Objective
Maximize boid count at 60 FPS when boids are clustered (worst case).
Current baseline: **291,000 boids** (browser-measured, clustered steady state).

## The Loop
1. Read `shader.wgsl` and `simulation.js`
2. Pick ONE compute optimization
3. Edit the file(s)
4. `git add shader.wgsl simulation.js && git commit -m "experiment: <description>"`
5. Run: `python3 evaluate.py 2>/dev/null`
6. Parse the last line for `max_boids: NNNNN`
7. If max_boids > previous best → KEEP, append to `results_perf.tsv`
8. If max_boids ≤ previous best → `git revert --no-edit HEAD`, append to `results_perf.tsv`
9. REPEAT. Do NOT pause. Loop until interrupted.

## What You Can Edit
- `shader.wgsl` — compute kernels, especially `flock_radius`
- `simulation.js` — grid sizing, buffer layout, pipeline config

## What You CANNOT Edit
- `bench_browser.js`, `evaluate.py`, `render.wgsl`, `renderer.js`, `index.html`
- `dashboard.py`, `serve.py`, `program.md`
- Do NOT add roost attractors or global orbit forces

## CRITICAL: Use `git revert --no-edit HEAD` to undo. NOT `git reset --hard`.
## Results go in `results_perf.tsv`

## The Problem
When boids cluster, cells become dense. At 50+ boids/cell × 27 cells = 1350+
distance checks per boid, with random memory access to boids_src[].

## Research-Backed Optimization Ideas (priority order)

### 1. Read from sorted order (HIGHEST IMPACT)
The scatter pass already sorts boid indices by cell into `sorted_indices[]`.
But flock_radius reads `boids_src[other_idx]` — random access across global memory.
Instead, read neighbor data from a SORTED buffer where particles in the same cell
are contiguous. This dramatically improves cache coherence.
Implementation: After scatter, copy boid data into cell-sorted order. Then flock
reads neighbors from the sorted buffer sequentially.

### 2. Reduce per-boid output writes
The agent already removed expensive viz-only outputs (experiment 44). Check if
there are more: dir_change, flock_alignment, sep_pressure, density calculations
that can be skipped or simplified.

### 3. Hard neighbor cap at 8
With sorted/cached reads, a lower neighbor cap becomes viable because the
iteration itself is cheaper. Try capping at 6-8 instead of 16.

### 4. Grid cell size tuning
Current: `cbrt(numBoids/4)` with worldSize = sphereRadius * 2.5.
Try: make cellSize = visualRange (so 3x3x3 covers exactly the interaction range).
This means gridSize = worldSize / visualRange.

### 5. Merge clear_grid + assign_cells
Each boid can clear its own target cell before writing. Saves one full dispatch.

### 6. Simplify the turn rate limiter further
Current: normalize + mix + clamp chain. Try: just lerp velocity directly
without the direction/speed decomposition.

### 7. Reduce visual range when dense
If a boid finds 8+ neighbors quickly, it doesn't need the full visual range.
Early-exit the cell loop after finding enough neighbors.

### 8. Use workgroup shared memory for cell data
Load particles from a cell into workgroup shared memory (var<workgroup>),
then all threads in the workgroup read from shared mem instead of global.
Only helps if multiple boids in the workgroup are in the same/adjacent cells.

## New Idea from Research: Branchless Neighbor Accumulation
Replace branching in the inner loop:
```wgsl
// BAD (warp divergence):
if (d2 < params.visual_range_sq) { ali += other.vel; n_align++; }

// GOOD (all threads compute, mask result):
let in_range = f32(d2 < params.visual_range_sq && d2 > 0.0001);
ali += other.vel * in_range;
coh += other_pos * in_range;
n_align += u32(in_range);
let in_sep = f32(d2 < params.separation_dist_sq) * in_range;
sep += diff * (1.0 - d2 / params.separation_dist_sq) * in_sep;
```
This eliminates branch divergence in dense cells where some threads find neighbors and others don't.

## Also Try: Tighter grid cell = visual range
Set cellSize = visualRange so 3x3x3 search exactly covers the interaction radius.
gridSize = floor(worldSize / visualRange). No wasted empty cells.

## Note
291k boids in 3D with 27-cell neighbor search is already competitive with published
demos. Most "million boid" demos are 2D. In 3D, 500k at 60fps is approximately
the ceiling for uniform grid + spatial hashing on consumer GPUs.

## Constraints
- flock_radius must produce correct flocking (separation, alignment, cohesion)
- Boid struct: 64 bytes (16 floats) — don't change the layout
- Billboard additive + radius neighbor mode is what's benchmarked
- Benchmark runs in real browser with 30s warmup for clustering
