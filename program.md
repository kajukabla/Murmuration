# WebGPU Flocking Optimization — Autoresearch Program

## Objective
Maximize the number of 3D boids that can be simulated at 60 FPS (p99 frame time < 16.6ms).
Current baseline: **170,000 boids**.

## The Loop
1. Read this file and the current `shader.wgsl` + `simulation.js`
2. Pick **ONE** optimization to try (see ideas below)
3. Edit `shader.wgsl` and/or `simulation.js`
4. `git add shader.wgsl simulation.js && git commit -m "experiment: <description>"`
5. Run: `python3 evaluate.py`
6. Parse the last line of stdout for `max_boids: NNNNN`
7. If max_boids **>** previous best → **KEEP** the commit, append to `results.tsv`
8. If max_boids **≤** previous best → `git reset --hard HEAD~1`, append to `results.tsv`
9. **REPEAT.** Do NOT pause to ask the human. Loop until interrupted.

## What You Can Edit
- `shader.wgsl` — compute kernels, workgroup sizes, algorithms, struct layout
- `simulation.js` — buffer layout, dispatch counts, pipeline config, grid size, constants

## What You CANNOT Edit
- `benchmark.ts` — locked evaluation harness
- `evaluate.py` — locked metric extraction
- `render.wgsl`, `renderer.js`, `index.html` — rendering/UI
- `program.md` — only humans edit this

## Results Tracking
After each experiment, append ONE line to `results.tsv`:
```
<experiment_number>\t<max_boids>\t<description>\t<kept|reverted>
```

## Important Notes
- The Deno benchmark in `benchmark.ts` reads WORKGROUP_SIZE, PARAMS_SIZE, and gridSize from `simulation.js` via regex. If you rename these constants, the benchmark breaks.
- The Boid struct (pos, vel, heading — each vec3f with padding = 48 bytes) must stay consistent between `shader.wgsl` and `simulation.js`. The benchmark mirrors this layout.
- If you change buffer sizes or add/remove buffers, `benchmark.ts` will break. Only change what's inside the existing structure.
- The spatial hashing approach (grid binning + 27-cell neighbor lookup) must be preserved.
- The simulation MUST produce correct flocking behavior (separation, alignment, cohesion).

## Optimization Ideas (roughly priority order)
1. **Parallel prefix sum** — the serial `workgroup_size(1)` prefix_sum is the #1 bottleneck at high boid counts. Replace with Blelloch scan or Hillis-Steele scan.
2. **Workgroup size tuning** — try 128, 256 for compute passes (clear_grid, assign_cells, scatter, flock).
3. **Grid resolution tuning** — GRID_SIZE=32 may not be optimal. Try 16, 24, 48, 64. Smaller grid = faster prefix sum but denser cells.
4. **Reduce per-boid math** — `acos()` in the turn-rate limiter is expensive. Use a fast approximation or skip it if the angle is small.
5. **Combine passes** — merge clear_grid + assign_cells into one dispatch (clear your cell first, then assign).
6. **Early cell skip** — in the flock pass, skip cells with count=0 immediately.
7. **Local workgroup accumulation** — aggregate atomics within a workgroup before global atomicAdd.
8. **Struct packing** — heading could be 2 floats (spherical coords) instead of 3, shrinking Boid to 40 bytes. (Must update benchmark.ts Boid size too — CAREFUL.)
9. **Loop unrolling** — manually unroll the 27-cell neighbor loop or the inner boid loop.
10. **Memory coalescing** — ensure flock pass reads from sorted_indices are sequential.
11. **Use u16 for sorted_indices** — if boid count < 65536, halve the index buffer size.
12. **Reduce visual_range** — smaller range = fewer neighbors = faster flock pass. But must still produce valid flocking.

## Constraints
- Do NOT change `benchmark.ts` or `evaluate.py`
- Do NOT change the CLI interface (--boids, --frames args, JSON stdout format)
- Keep the 5-pass structure (clear, assign, prefix_sum, scatter, flock) unless you can prove a merged version is correct
- All changes must be a single atomic experiment — one idea per commit
