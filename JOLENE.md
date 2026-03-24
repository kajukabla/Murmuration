---
phase: app
views:
  - name: Sim
    port: 8080
  - name: Dashboard
    port: 8050
dev: python serve.py
spec: program.md
---

# Murmuration

Boid flocking simulation with automated performance optimization. WebGPU compute shaders, real-time rendering, and a Python benchmark coordination server.

## Status
Running auto-optimization loop iteration 47. Frame rate improved from 42fps to 61fps since last session.

## Attention
- MSAA 4x test showed a regression — review the trace?
- Shader variant B outperformed A by 18% — ready to commit?

## Recent
- Switched from instanced rendering to indirect draw calls
- Added bloom post-processing pass
- Neighbor cap changed to 10 for spatial hash
