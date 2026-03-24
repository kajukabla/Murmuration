#!/usr/bin/env python3
"""
Phase scheduler for Murmuration optimization.

Reads schedule_config.json, cycles through phases, spawns a Claude agent
for each phase with a generated program.md, and rotates after the configured
duration.

Usage:
    python3 scheduler.py            # production (uses durations from config)
    python3 scheduler.py --test     # test mode (2-minute phases)
"""

import json
import os
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
CONFIG_PATH = BASE_DIR / "schedule_config.json"
PROGRAM_MD_PATH = BASE_DIR / "program.md"
STATUS_PATH = BASE_DIR / "scheduler_status.json"
POLL_INTERVAL = 30       # seconds between agent health checks
STATUS_INTERVAL = 10     # seconds between status file writes

# ---------------------------------------------------------------------------
# Globals for signal handling
# ---------------------------------------------------------------------------
_agent_pgid: int | None = None
_running = True


def log(msg: str) -> None:
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    print(f"[scheduler {ts}] {msg}", file=sys.stderr, flush=True)


# ---------------------------------------------------------------------------
# program.md generation
# ---------------------------------------------------------------------------

def _objective_text(phase: dict) -> str:
    mf = phase["metric_format"]
    if mf == "max_boids":
        if "perf" in phase["name"]:
            return (
                "Maximize boid count at 60 FPS (browser-measured, clustered steady state).\n"
                "The evaluate command launches a browser, ramps boids, and reports `max_boids`."
            )
        else:
            return (
                "Maximize boid throughput on the headless WebGPU compute benchmark.\n"
                "The evaluate command runs a Deno compute-only benchmark and reports `max_boids`."
            )
    elif mf == "score":
        return (
            "Maximize flocking quality score on a fixed boid count.\n"
            "The evaluate command runs a Deno benchmark that scores cohesion, alignment,\n"
            "and separation quality. Higher `score` is better."
        )
    return "Optimize the metric reported by the evaluate command."


def _metric_parse_hint(phase: dict) -> str:
    mf = phase["metric_format"]
    if mf == "max_boids":
        return "Parse the last line for `max_boids: NNNNN`"
    elif mf == "score":
        return "Parse the last line for `score: NNNNN`"
    return "Parse the metric from the last line of output"


def generate_program_md(phase: dict) -> str:
    editable = phase["editable_function"]
    locked = phase["locked_function"]
    editable_files = phase["editable_files"]
    results_file = phase["results_file"]
    evaluate_cmd = phase["evaluate_cmd"]
    label = phase["label"]
    metric = phase["metric_format"]

    editable_list = "\n".join(f"- `{f}`" for f in editable_files)
    git_add_files = " ".join(editable_files)

    return f"""\
# {label} Optimization

## Objective
{_objective_text(phase)}

## The Loop
1. Read `shader.wgsl` and `simulation.js`
2. Pick ONE optimization for the `{editable}` function (or supporting code)
3. Edit the file(s)
4. `git add {git_add_files} && git commit -m "experiment: <description>"`
5. Run: `{evaluate_cmd}`
6. {_metric_parse_hint(phase)}
7. If {metric} improves over previous best -> KEEP, append result to `{results_file}`
8. If {metric} does not improve -> `git revert --no-edit HEAD`, append result to `{results_file}`
9. REPEAT. Do NOT pause. Loop until interrupted.

## What You Can Edit
{editable_list}

Focus on the `{editable}` compute kernel in `shader.wgsl` (and pipeline/grid
config in `simulation.js` if it is editable).

## LOCKED - DO NOT MODIFY
The `{locked}` function in `shader.wgsl` is **LOCKED**. Do NOT change it in any way.
Do NOT rename, rewrite, reorder, or touch any line inside `{locked}`.

## What You CANNOT Edit
- `bench_browser.js`, `evaluate.py`, `render.wgsl`, `renderer.js`, `index.html`
- `dashboard.py`, `serve.py`, `program.md`, `scheduler.py`, `schedule_config.json`
- Do NOT add roost attractors or global orbit forces

## CRITICAL: Use `git revert --no-edit HEAD` to undo. NOT `git reset --hard`.

## Results go in `{results_file}`

## Constraints
- `{editable}` must produce correct flocking (separation, alignment, cohesion)
- Boid struct: 64 bytes (16 floats) -- don't change the layout
- Billboard additive + radius neighbor mode is what's benchmarked
- Do NOT modify `{locked}` under any circumstances

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
"""


# ---------------------------------------------------------------------------
# Status file
# ---------------------------------------------------------------------------

def write_status(phase: dict, phase_start: datetime, experiments: int,
                 total_completed: int, next_phase: str | None,
                 running: bool = True) -> None:
    status = {
        "running": running,
        "current_phase": phase["name"],
        "phase_start": phase_start.isoformat(),
        "duration_min": phase["duration_minutes"],
        "experiments": experiments,
        "total_completed": total_completed,
        "next_phase": next_phase,
    }
    tmp = STATUS_PATH.with_suffix(".tmp")
    try:
        tmp.write_text(json.dumps(status, indent=2) + "\n")
        tmp.replace(STATUS_PATH)
    except Exception as exc:
        log(f"Failed to write status: {exc}")


# ---------------------------------------------------------------------------
# Experiment counting
# ---------------------------------------------------------------------------

def count_experiments(results_file: str) -> int:
    p = BASE_DIR / results_file
    if not p.exists():
        return 0
    try:
        lines = p.read_text().strip().splitlines()
        # skip header if present
        return max(0, len(lines) - 1) if lines else 0
    except Exception:
        return 0


# ---------------------------------------------------------------------------
# Process management
# ---------------------------------------------------------------------------

def kill_stray_processes() -> None:
    """Kill stray chromium / bench_browser / eval_wrapper processes."""
    for pattern in ["chromium", "bench_browser", "eval_wrapper"]:
        try:
            subprocess.run(
                ["pkill", "-f", pattern],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        except Exception:
            pass


def start_agent() -> subprocess.Popen:
    cmd = [
        "claude",
        "--verbose",
        "--output-format", "stream-json",
        "--dangerously-skip-permissions",
        "Read program.md and begin the optimization loop. Do not stop or ask questions.",
    ]
    log(f"Starting agent: {' '.join(cmd)}")
    proc = subprocess.Popen(
        cmd,
        cwd=str(BASE_DIR),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        preexec_fn=os.setsid,
    )
    log(f"Agent started, pid={proc.pid}, pgid={os.getpgid(proc.pid)}")
    return proc


def kill_agent(proc: subprocess.Popen) -> None:
    global _agent_pgid
    if proc is None:
        return
    try:
        pgid = os.getpgid(proc.pid)
        log(f"Killing agent process group pgid={pgid}")
        os.killpg(pgid, signal.SIGTERM)
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            log("Agent did not exit after SIGTERM, sending SIGKILL")
            os.killpg(pgid, signal.SIGKILL)
            proc.wait(timeout=5)
    except ProcessLookupError:
        log("Agent already exited")
    except Exception as exc:
        log(f"Error killing agent: {exc}")
    finally:
        _agent_pgid = None
    kill_stray_processes()


# ---------------------------------------------------------------------------
# Signal handling
# ---------------------------------------------------------------------------

def _handle_signal(signum, frame):
    global _running
    log(f"Received signal {signum}, shutting down...")
    _running = False


# ---------------------------------------------------------------------------
# Main scheduler loop
# ---------------------------------------------------------------------------

def run_phase(phase: dict, duration_override: int | None,
              total_completed: int, next_phase_name: str | None) -> int:
    global _agent_pgid

    duration = duration_override if duration_override is not None else phase["duration_minutes"]
    label = phase["label"]
    log(f"=== Starting phase: {label} ({duration} min) ===")

    # Generate and write program.md
    md = generate_program_md(phase)
    PROGRAM_MD_PATH.write_text(md)
    log("Wrote program.md")

    phase_start = datetime.now(timezone.utc)
    deadline = time.monotonic() + duration * 60

    # Start agent
    proc = start_agent()
    _agent_pgid = os.getpgid(proc.pid)

    last_status_write = 0.0
    initial_experiments = count_experiments(phase["results_file"])

    try:
        while _running and time.monotonic() < deadline:
            # Write status periodically
            now_mono = time.monotonic()
            if now_mono - last_status_write >= STATUS_INTERVAL:
                current_experiments = count_experiments(phase["results_file"])
                write_status(
                    phase, phase_start,
                    experiments=current_experiments - initial_experiments,
                    total_completed=total_completed,
                    next_phase=next_phase_name,
                )
                last_status_write = now_mono

            # Sleep in small increments so we can respond to signals
            sleep_end = min(time.monotonic() + POLL_INTERVAL, deadline)
            while _running and time.monotonic() < sleep_end:
                time.sleep(min(1.0, sleep_end - time.monotonic()))

            if not _running:
                break

            # Check if agent is still alive
            ret = proc.poll()
            if ret is not None:
                log(f"Agent exited with code {ret}, restarting...")
                kill_stray_processes()
                if time.monotonic() < deadline and _running:
                    proc = start_agent()
                    _agent_pgid = os.getpgid(proc.pid)
            else:
                # Agent process is alive but might be hung (context exhausted)
                # Check if agent.log has grown recently
                log_path = BASE_DIR / "agent.log"
                if log_path.exists():
                    log_age = time.time() - log_path.stat().st_mtime
                    if log_age > 180:  # No log output in 3 minutes = hung
                        log(f"Agent hung (no output for {int(log_age)}s), killing and restarting...")
                        kill_agent(proc)
                        kill_stray_processes()
                        if time.monotonic() < deadline and _running:
                            proc = start_agent()
                            _agent_pgid = os.getpgid(proc.pid)
    finally:
        log(f"Phase {label} ending, killing agent...")
        kill_agent(proc)

    final_experiments = count_experiments(phase["results_file"])
    phase_experiments = final_experiments - initial_experiments
    log(f"Phase {label} completed. Experiments this phase: {phase_experiments}")
    return total_completed + 1


def main() -> None:
    global _running

    # Parse args
    test_mode = "--test" in sys.argv
    if test_mode:
        log("TEST MODE: 2-minute phases")
    duration_override = 2 if test_mode else None

    # Install signal handlers
    signal.signal(signal.SIGTERM, _handle_signal)
    signal.signal(signal.SIGINT, _handle_signal)

    # Load config
    try:
        config = json.loads(CONFIG_PATH.read_text())
    except Exception as exc:
        log(f"Failed to read config: {exc}")
        sys.exit(1)

    phases = config["phases"]
    loop = config.get("loop", False)

    if not phases:
        log("No phases configured, exiting.")
        sys.exit(1)

    log(f"Loaded {len(phases)} phases, loop={loop}")

    total_completed = 0
    cycle = 0

    while _running:
        cycle += 1
        log(f"--- Cycle {cycle} ---")

        for i, phase in enumerate(phases):
            if not _running:
                break

            next_phase_name = phases[i + 1]["name"] if i + 1 < len(phases) else (
                phases[0]["name"] if loop else None
            )

            try:
                total_completed = run_phase(
                    phase, duration_override, total_completed, next_phase_name
                )
            except Exception as exc:
                log(f"Error in phase {phase['name']}: {exc}")
                kill_stray_processes()
                # Continue to next phase
                total_completed += 1

        if not loop:
            break

    # Final status
    if phases:
        write_status(
            phases[-1],
            datetime.now(timezone.utc),
            experiments=0,
            total_completed=total_completed,
            next_phase=None,
            running=False,
        )

    log("Scheduler exiting.")


if __name__ == "__main__":
    main()
