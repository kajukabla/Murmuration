#!/usr/bin/env python3
"""
Locked evaluation script for the autoresearch optimization loop.
Binary-searches for the maximum boid count that maintains 60 FPS (p99 < 16.6ms).
Outputs: max_boids: NNNNN
Appends each probe to run_metrics.csv.
"""

import subprocess
import json
import sys
import os
from datetime import datetime

DENO = os.path.expanduser("~/.deno/bin/deno")
BENCHMARK = "benchmark.ts"
TARGET_MS = 16.6  # 60 FPS
FRAMES = 100
CSV_FILE = "run_metrics.csv"

# Binary search bounds (in thousands)
LO = 1    # 1,000 boids
HI = 500  # 500,000 boids
STEP = 1  # search in increments of 1,000


def run_benchmark(num_boids: int) -> dict | None:
    """Run the Deno benchmark and return parsed JSON, or None on failure."""
    try:
        result = subprocess.run(
            [DENO, "run", "--unstable-webgpu", "--allow-read",
             BENCHMARK, "--boids", str(num_boids), "--frames", str(FRAMES)],
            capture_output=True, text=True, timeout=120,
        )
        if result.returncode != 0:
            print(f"  benchmark failed (exit {result.returncode}): {result.stderr[:200]}", file=sys.stderr)
            return None
        return json.loads(result.stdout.strip())
    except (subprocess.TimeoutExpired, json.JSONDecodeError) as e:
        print(f"  benchmark error: {e}", file=sys.stderr)
        return None


def append_csv(iteration: int, num_boids: int, avg_ms: float, p99_ms: float, passed: bool):
    """Append a row to run_metrics.csv."""
    write_header = not os.path.exists(CSV_FILE) or os.path.getsize(CSV_FILE) == 0
    with open(CSV_FILE, "a") as f:
        if write_header:
            f.write("Timestamp,Iteration,Particle_Count,Avg_Frame_Time,P99_Frame_Time,Result\n")
        result = "Pass" if passed else "Fail"
        f.write(f"{datetime.now().isoformat()},{iteration},{num_boids},{avg_ms:.2f},{p99_ms:.2f},{result}\n")


def main():
    # Read iteration number from run_metrics.csv
    iteration = 0
    if os.path.exists(CSV_FILE):
        with open(CSV_FILE) as f:
            iteration = max(0, sum(1 for _ in f) - 1)  # subtract header

    print(f"Evaluating: binary search for max boids @ p99 < {TARGET_MS}ms", file=sys.stderr)

    lo, hi = LO, HI
    best = 0
    probe_num = 0

    while lo <= hi:
        mid = (lo + hi) // 2
        num_boids = mid * 1000
        probe_num += 1

        print(f"  probe {probe_num}: {num_boids} boids ...", end="", file=sys.stderr, flush=True)
        result = run_benchmark(num_boids)

        if result is None:
            # Benchmark failed — treat as too many boids
            print(f" FAILED", file=sys.stderr)
            append_csv(iteration, num_boids, 0, 0, False)
            hi = mid - 1
            continue

        p99 = result["p99_ms"]
        avg = result["avg_ms"]
        passed = p99 < TARGET_MS

        print(f" avg={avg:.1f}ms p99={p99:.1f}ms {'PASS' if passed else 'FAIL'}", file=sys.stderr)
        append_csv(iteration, num_boids, avg, p99, passed)

        if passed:
            best = max(best, num_boids)
            lo = mid + 1
        else:
            hi = mid - 1

    print(f"max_boids: {best}")


if __name__ == "__main__":
    main()
