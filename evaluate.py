#!/usr/bin/env python3
"""
Evaluate murmuration quality using headless WebGPU benchmark.
Runs simulation for ~1 minute to reach steady state, then measures
behavioral quality metrics derived from real starling murmuration data.

Outputs: score: X.XXXX (0 to 1, higher = more murmuration-like)
"""

import subprocess
import json
import sys
import os
from datetime import datetime

DENO = os.path.expanduser("~/.deno/bin/deno")
BENCHMARK = "benchmark.ts"
CSV_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "run_metrics.csv")
NUM_BOIDS = 20000


def run_benchmark() -> dict | None:
    try:
        result = subprocess.run(
            [DENO, "run", "--unstable-webgpu", "--allow-read",
             BENCHMARK, "--boids", str(NUM_BOIDS), "--frames", "30"],
            capture_output=True, text=True, timeout=300,
        )
        if result.returncode != 0:
            print(f"  benchmark failed: {result.stderr[:200]}", file=sys.stderr)
            return None
        return json.loads(result.stdout.strip())
    except (subprocess.TimeoutExpired, json.JSONDecodeError) as e:
        print(f"  benchmark error: {e}", file=sys.stderr)
        return None


def append_csv(iteration: int, data: dict):
    write_header = not os.path.exists(CSV_FILE) or os.path.getsize(CSV_FILE) == 0
    with open(CSV_FILE, "a") as f:
        if write_header:
            f.write("Timestamp,Iteration,Score,Cohesion,VelocityCorr,AspectRatio,DensityCV,Dynamics,AvgNeighbors,AvgMs\n")
        f.write(f"{datetime.now().isoformat()},{iteration},{data.get('score',0):.4f},"
                f"{data.get('cohesion_ratio',0):.3f},{data.get('velocity_corr',0):.3f},"
                f"{data.get('aspect_ratio',0):.2f},{data.get('density_cv',0):.3f},"
                f"{data.get('dynamics',0):.4f},{data.get('avg_neighbors',0):.1f},"
                f"{data.get('avg_ms',0):.1f}\n")


def main():
    iteration = 0
    if os.path.exists(CSV_FILE):
        with open(CSV_FILE) as f:
            iteration = max(0, sum(1 for _ in f) - 1)

    print(f"Evaluating murmuration quality ({NUM_BOIDS} boids, ~90s)...", file=sys.stderr)
    result = run_benchmark()

    if result is None:
        print("score: 0.0000")
        return

    score = result.get("score", 0)
    print(f"  cohesion={result.get('cohesion_ratio',0):.3f} "
          f"vel_corr={result.get('velocity_corr',0):.3f} "
          f"aspect={result.get('aspect_ratio',0):.2f} "
          f"density_cv={result.get('density_cv',0):.3f} "
          f"dynamics={result.get('dynamics',0):.4f} "
          f"avg_neighbors={result.get('avg_neighbors',0):.1f}", file=sys.stderr)
    append_csv(iteration, result)
    print(f"score: {score:.4f}")


if __name__ == "__main__":
    main()
