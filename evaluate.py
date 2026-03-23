#!/usr/bin/env python3
"""
Evaluate full-pipeline performance using Playwright browser benchmark.
Binary-searches for max boid count that maintains 60 FPS with REAL rendering.
Measures compute + vertex shaders + fragment overdraw + MSAA + present.

Outputs: max_boids: NNNNN
"""

import subprocess
import json
import sys
import os
from datetime import datetime

BENCH_SCRIPT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "bench_browser.js")
CSV_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "run_metrics.csv")
TARGET_FPS = 55  # slightly below 60 to account for variance

LO = 10   # 10k
HI = 500  # 500k
WARMUP = 10  # seconds
MEASURE = 8  # seconds


def run_benchmark(num_boids: int) -> dict | None:
    try:
        result = subprocess.run(
            ["node", BENCH_SCRIPT,
             "--boids", str(num_boids),
             "--warmup-sec", str(WARMUP),
             "--measure-sec", str(MEASURE)],
            capture_output=True, text=True, timeout=120,
        )
        if result.returncode != 0:
            print(f"  bench failed: {result.stderr[:200]}", file=sys.stderr)
            return None
        # stdout is the JSON result line
        for line in result.stdout.strip().split('\n'):
            try:
                return json.loads(line)
            except json.JSONDecodeError:
                continue
        return None
    except (subprocess.TimeoutExpired, Exception) as e:
        print(f"  bench error: {e}", file=sys.stderr)
        return None


def append_csv(iteration: int, num_boids: int, avg_fps: float, avg_ms: float, passed: bool):
    write_header = not os.path.exists(CSV_FILE) or os.path.getsize(CSV_FILE) == 0
    with open(CSV_FILE, "a") as f:
        if write_header:
            f.write("Timestamp,Iteration,Particle_Count,Avg_Frame_Time,P99_Frame_Time,Result\n")
        f.write(f"{datetime.now().isoformat()},{iteration},{num_boids},"
                f"{avg_ms:.1f},{avg_ms:.1f},{'Pass' if passed else 'Fail'}\n")


def main():
    iteration = 0
    if os.path.exists(CSV_FILE):
        with open(CSV_FILE) as f:
            iteration = max(0, sum(1 for _ in f) - 1)

    print(f"Browser benchmark: binary search for max boids @ {TARGET_FPS}+ fps", file=sys.stderr)

    lo, hi = LO, HI
    best = 0
    probe = 0

    while lo <= hi:
        mid = (lo + hi) // 2
        num_boids = mid * 1000
        probe += 1

        print(f"  probe {probe}: {num_boids} boids ...", end="", file=sys.stderr, flush=True)
        result = run_benchmark(num_boids)

        if result is None:
            print(f" FAILED", file=sys.stderr)
            append_csv(iteration, num_boids, 0, 999, False)
            hi = mid - 1
            continue

        avg_fps = result.get("avg_fps", 0)
        min_fps = result.get("min_fps", 0)
        avg_ms = result.get("avg_ms", 999)
        passed = min_fps >= TARGET_FPS

        print(f" avg={avg_fps:.0f}fps min={min_fps}fps {avg_ms:.0f}ms {'PASS' if passed else 'FAIL'}", file=sys.stderr)
        append_csv(iteration, num_boids, avg_fps, avg_ms, passed)

        if passed:
            best = max(best, num_boids)
            lo = mid + 1
        else:
            hi = mid - 1

    print(f"max_boids: {best}")


if __name__ == "__main__":
    main()
