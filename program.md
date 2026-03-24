# Classic Performance Optimization

## Objective
Maximize boid count at 60 FPS (browser-measured, clustered steady state).
The evaluate command launches a browser, ramps boids, and reports `max_boids`.

## The Loop
1. Read `shader.wgsl` and `simulation.js`
2. Pick ONE optimization for the `flock_radius` function (or supporting code)
3. Edit the file(s)
4. `git add shader.wgsl simulation.js && git commit -m "experiment: <description>"`
5. Run: `python3 evaluate.py 2>/dev/null`
6. Parse the last line for `max_boids: NNNNN`
7. If max_boids improves over previous best -> KEEP, append result to `results_classic_perf.tsv`
8. If max_boids does not improve -> `git revert --no-edit HEAD`, append result to `results_classic_perf.tsv`
9. REPEAT. Do NOT pause. Loop until interrupted.

## What You Can Edit
- `shader.wgsl`
- `simulation.js`

Focus on the `flock_radius` compute kernel in `shader.wgsl` (and pipeline/grid
config in `simulation.js` if it is editable).

## LOCKED - DO NOT MODIFY
The `flock` function in `shader.wgsl` is **LOCKED**. Do NOT change it in any way.
Do NOT rename, rewrite, reorder, or touch any line inside `flock`.

## What You CANNOT Edit
- `bench_browser.js`, `evaluate.py`, `render.wgsl`, `renderer.js`, `index.html`
- `dashboard.py`, `serve.py`, `program.md`, `scheduler.py`, `schedule_config.json`
- Do NOT add roost attractors or global orbit forces

## CRITICAL: Use `git revert --no-edit HEAD` to undo. NOT `git reset --hard`.

## Results go in `results_classic_perf.tsv`

## Constraints
- `flock_radius` must produce correct flocking (separation, alignment, cohesion)
- Boid struct: 64 bytes (16 floats) -- don't change the layout
- Billboard additive + radius neighbor mode is what's benchmarked
- Do NOT modify `flock` under any circumstances

## MANDATORY: Flocking Correctness Checks
Before committing ANY experiment, verify these are preserved:
1. **Heading update**: boids_dst[i].heading MUST be written with a smoothly
   tracked direction based on velocity. Without this, boids face the wrong way.
2. **size_factor copy**: boids_dst[i].size_factor = boid.size_factor MUST be written.
3. **27-cell neighbor search**: The flock function MUST search neighboring grid cells
   (3x3x3), not just the own cell. Own-cell-only breaks flocking.
4. **Separation + Alignment + Cohesion**: All three forces must be computed.
   Removing any one breaks the flocking behavior.
5. **If a drift() pass exists**: It MUST copy heading, size_factor, and all viz
   metrics (speed, neighbor_count, etc.) from src to dst.

If you remove any of these for performance, the experiment WILL be reverted
because the visual result will be broken even if the benchmark passes.
