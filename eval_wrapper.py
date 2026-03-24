#!/usr/bin/env python3
"""Wrapper around bench_browser.js that adds min_fps (= 1000/max_ms) to results."""
import subprocess, json, sys, os

BENCH_SCRIPT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "bench_browser.js")
TARGET_FPS = 55
LO, HI = 10, 500
WARMUP, MEASURE = 10, 8

def run_benchmark(num_boids):
    try:
        result = subprocess.run(
            ["node", BENCH_SCRIPT, "--boids", str(num_boids),
             "--warmup-sec", str(WARMUP), "--measure-sec", str(MEASURE)],
            capture_output=True, text=True, timeout=120)
        if result.returncode != 0:
            print(f"  bench failed: {result.stderr[:200]}", file=sys.stderr)
            return None
        for line in result.stdout.strip().split('\n'):
            try:
                data = json.loads(line)
                # Use avg_fps as min_fps (p95/max too noisy with Playwright)
                data["min_fps"] = data.get("avg_fps", 0)
                return data
            except json.JSONDecodeError:
                continue
        return None
    except Exception as e:
        print(f"  bench error: {e}", file=sys.stderr)
        return None

def main():
    print(f"Browser benchmark: binary search for max boids @ {TARGET_FPS}+ fps", file=sys.stderr)
    lo, hi, best, probe = LO, HI, 0, 0
    while lo <= hi:
        mid = (lo + hi) // 2
        num_boids = mid * 1000
        probe += 1
        print(f"  probe {probe}: {num_boids} boids ...", end="", file=sys.stderr, flush=True)
        result = run_benchmark(num_boids)
        if result is None:
            print(f" FAILED", file=sys.stderr)
            hi = mid - 1
            continue
        avg_fps = result.get("avg_fps", 0)
        min_fps = result.get("min_fps", 0)
        avg_ms = result.get("avg_ms", 999)
        passed = min_fps >= TARGET_FPS
        print(f" avg={avg_fps:.0f}fps min={min_fps}fps {avg_ms:.0f}ms {'PASS' if passed else 'FAIL'}", file=sys.stderr)
        if passed:
            best = max(best, num_boids)
            lo = mid + 1
        else:
            hi = mid - 1
    print(f"max_boids: {best}")

if __name__ == "__main__":
    main()
