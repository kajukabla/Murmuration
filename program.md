# Murmuration Quality Optimization — Autoresearch Program

## Objective
Maximize the murmuration quality score (0-1) which measures how closely the simulation matches real starling murmuration behavior based on empirical measurements from Cavagna, Ballerini, and Attanasi et al.

## The Score
The composite score combines 5 metrics from real murmurations:
- **Cohesion ratio** (target >0.90): fraction of boids with 5+ of 7 neighbors
- **Velocity correlation** (target 0.75-0.95): avg alignment with neighbors (1.0 = too static)
- **Aspect ratio** (target 2-5): flock elongation (1=sphere, bad; 3-4=ideal)
- **Density uniformity** (target CV<0.5): uniform spacing, not clumpy
- **Shape dynamics** (target 0.05-0.2): flock is actively morphing, not frozen

**Current baseline score: ~0.70**

## The Loop
1. Read the current `shader.wgsl` and `index.html` (Murmuration preset values)
2. Pick ONE parameter change to try
3. Edit the file(s)
4. `git add shader.wgsl index.html && git commit -m "experiment: <description>"`
5. Run: `python3 evaluate.py`
6. Parse the last line of stdout for `score: X.XXXX`
7. If score > previous best → KEEP, append to results.tsv
8. If score ≤ previous best → `git revert --no-edit HEAD`, append to results.tsv
9. REPEAT. Do NOT pause. Loop until interrupted.

## What You Can Edit
- `shader.wgsl` — hardcoded constants in the flock() function:
  - `K_NEIGHBORS` (line ~127): try 5, 6, 7, 8
  - Gravity: `new_vel.y -= 0.03` — try 0.0 to 0.1
  - Center-seeking: `normalize(to_center) * 0.08` — try 0.0 to 0.3
  - Roost attractor: `normalize(to_roost) * 0.3` — try 0.1 to 1.0
  - Roost orbit speed: `f32(params.frame_count) * 0.002` — try 0.001 to 0.01
  - Roost orbit radius: `params.sphere_radius * 0.4` — try 0.2 to 0.6
  - Noise amplitude: the `* 0.15` — try 0.0 to 0.3
  - Separation multiplier: `* 2.0` — try 1.0 to 4.0
- `index.html` — Murmuration preset values (the agent reads these via benchmark.ts):
  - visualRange, separationDist, alignFactor, cohesionFactor, separationFactor
  - maxSpeed, minSpeed, turnFactor, smoothing, sphereRadius

## What You CANNOT Edit
- `benchmark.ts`, `evaluate.py` — locked evaluation
- `render.wgsl`, `renderer.js` — rendering
- `simulation.js` — compute pipeline setup
- `program.md` — only humans edit this

## CRITICAL: git revert
Use `git revert --no-edit HEAD` to undo failed experiments. NOT `git reset --hard`.

## Results Tracking
Append to results.tsv: `<experiment_number>\t<score>\t<description>\t<kept|reverted>`

## Optimization Strategy
The score has 5 components. Focus on whichever is weakest:
- Low cohesion → increase cohesionFactor, reduce sphereRadius, increase visualRange
- Low velocity corr → increase alignFactor
- Bad aspect ratio → adjust gravity, roost attractor shape
- High density CV → adjust separationFactor, K_NEIGHBORS
- Low dynamics → increase roost speed, noise, reduce smoothing

Try ONE change at a time. Small increments. The evaluation takes ~90 seconds.
