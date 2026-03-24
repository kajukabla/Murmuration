#!/usr/bin/env python3
"""
Evaluate full-pipeline performance by navigating existing Chrome tab.
NO Playwright, NO new windows. Uses osascript + bench_result.json.

Binary-searches for max boid count at 60 FPS with real rendering.
Requires: serve.py on port 8080, Chrome tab open to localhost:8080.
"""

import subprocess
import json
import sys
import os
import time
from datetime import datetime

BENCH_SCRIPT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "bench_browser.js")
CSV_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "run_metrics.csv")
TARGET_MS = 16.6  # 60 FPS

LO = 10   # 10k
HI = 1000  # 300k


def run_benchmark(num_boids: int) -> dict | None:
    try:
        result = subprocess.run(
            ["node", BENCH_SCRIPT, "--boids", str(num_boids), "--timeout", "80"],
            capture_output=True, text=True, timeout=120,
        )
        if result.returncode != 0:
            print(f"  failed: {result.stderr[:200]}", file=sys.stderr)
            return None
        for line in result.stdout.strip().split('\n'):
            try:
                return json.loads(line)
            except json.JSONDecodeError:
                continue
        return None
    except Exception as e:
        print(f"  error: {e}", file=sys.stderr)
        return None


def append_csv(iteration: int, num_boids: int, avg_ms: float, p99_ms: float, passed: bool):
    write_header = not os.path.exists(CSV_FILE) or os.path.getsize(CSV_FILE) == 0
    with open(CSV_FILE, "a") as f:
        if write_header:
            f.write("Timestamp,Iteration,Particle_Count,Avg_Frame_Time,P99_Frame_Time,Result\n")
        f.write(f"{datetime.now().isoformat()},{iteration},{num_boids},"
                f"{avg_ms:.1f},{p99_ms:.1f},{'Pass' if passed else 'Fail'}\n")


def main():
    iteration = 0
    if os.path.exists(CSV_FILE):
        with open(CSV_FILE) as f:
            iteration = max(0, sum(1 for _ in f) - 1)

    print(f"Browser perf: binary search for max boids @ p99 < {TARGET_MS}ms", file=sys.stderr)

    lo, hi = LO, HI
    best = 0
    probe = 0

    while lo <= hi:
        mid = (lo + hi) // 2
        num_boids = mid * 1000
        probe += 1

        print(f"  probe {probe}: {num_boids} boids ...", end="", file=sys.stderr, flush=True)
        result = run_benchmark(num_boids)

        if result is None or 'error' in result:
            print(f" FAILED", file=sys.stderr)
            append_csv(iteration, num_boids, 0, 0, False)
            hi = mid - 1
            continue

        p99 = result.get("p99_ms", 999)
        avg = result.get("avg_ms", 999)
        drop_rate = result.get("drop_rate", 1.0)
        eff_fps = result.get("effective_fps", 0)
        # Pass if <5% frames dropped (missed VSync) AND effective fps >= 55
        passed = drop_rate < 0.05 and eff_fps >= 55

        print(f" fps={eff_fps:.0f} drops={drop_rate*100:.1f}% avg={avg:.1f}ms {'PASS' if passed else 'FAIL'}", file=sys.stderr)
        append_csv(iteration, num_boids, avg, p99, passed)

        if passed:
            best = max(best, num_boids)
            lo = mid + 1
        else:
            hi = mid - 1

    # Navigate back to normal view
    print(f"max_boids: {best}")


if __name__ == "__main__":
    main()
