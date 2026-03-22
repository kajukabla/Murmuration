#!/usr/bin/env python3
"""
Locked evaluation script for the autoresearch optimization loop.
Binary-searches for the maximum boid count that maintains 60 FPS.
Uses the LIVE BROWSER (Chrome on macOS) for benchmarking with full rendering.

The browser navigates to localhost:8080?boids=N, runs 120 measured frames,
and POSTs results to localhost:8080/api/bench_result.

Outputs: max_boids: NNNNN
Appends each probe to run_metrics.csv.
"""

import subprocess
import json
import sys
import os
import time

SERVER_URL = "http://localhost:8080"
TARGET_MS = 16.6  # 60 FPS
RESULT_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "bench_result.json")
CSV_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "run_metrics.csv")
TIMEOUT = 30  # max seconds to wait for benchmark

# Binary search bounds (in thousands)
LO = 1     # 1,000 boids
HI = 500   # 500,000 boids


def navigate_chrome(url: str):
    """Tell the frontmost Chrome tab to navigate to a URL (macOS)."""
    script = f'''
    tell application "Google Chrome"
        if (count of windows) > 0 then
            set URL of active tab of front window to "{url}"
        else
            open location "{url}"
        end if
    end tell
    '''
    subprocess.run(["osascript", "-e", script], capture_output=True, timeout=5)


def wait_for_result(timeout: int = TIMEOUT) -> dict | None:
    """Poll bench_result.json until it appears or timeout."""
    # Clear any stale result
    if os.path.exists(RESULT_FILE):
        os.remove(RESULT_FILE)

    deadline = time.time() + timeout
    while time.time() < deadline:
        if os.path.exists(RESULT_FILE):
            try:
                with open(RESULT_FILE) as f:
                    data = json.load(f)
                os.remove(RESULT_FILE)  # consume it
                return data
            except (json.JSONDecodeError, IOError):
                pass
        time.sleep(0.5)
    return None


def append_csv(iteration: int, num_boids: int, avg_ms: float, p99_ms: float, passed: bool):
    from datetime import datetime
    write_header = not os.path.exists(CSV_FILE) or os.path.getsize(CSV_FILE) == 0
    with open(CSV_FILE, "a") as f:
        if write_header:
            f.write("Timestamp,Iteration,Particle_Count,Avg_Frame_Time,P99_Frame_Time,Result\n")
        result = "Pass" if passed else "Fail"
        f.write(f"{datetime.now().isoformat()},{iteration},{num_boids},{avg_ms:.2f},{p99_ms:.2f},{result}\n")


def run_benchmark(num_boids: int) -> dict | None:
    """Navigate Chrome to benchmark URL and wait for results."""
    url = f"{SERVER_URL}?boids={num_boids}"
    navigate_chrome(url)
    # Wait for warmup (30 frames) + measurement (120 frames) + buffer
    # At 60fps that's ~2.5s, at 15fps ~10s. Give it generous time.
    return wait_for_result(TIMEOUT)


def main():
    # Read iteration number
    iteration = 0
    if os.path.exists(CSV_FILE):
        with open(CSV_FILE) as f:
            iteration = max(0, sum(1 for _ in f) - 1)

    print(f"Evaluating: binary search for max boids @ p99 < {TARGET_MS}ms (browser)", file=sys.stderr)

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
            print(f" TIMEOUT", file=sys.stderr)
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

    # Navigate back to normal viewing mode with best count
    navigate_chrome(SERVER_URL)
    print(f"max_boids: {best}")


if __name__ == "__main__":
    main()
