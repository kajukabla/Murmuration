#!/bin/bash
# Watchdog: check if scheduler is running, restart if dead
cd /Users/aaronlemke/AI/Murmuration

if ! pgrep -f "scheduler.py" > /dev/null; then
    echo "$(date): Scheduler dead, restarting..." >> watchdog.log
    rm -f bench_request.json bench_result.json
    echo '{}' > scheduler_status.json
    python3 scheduler.py >> scheduler.log 2>&1 &
    echo "$(date): Scheduler restarted, PID $!" >> watchdog.log
else
    echo "$(date): Scheduler alive" >> watchdog.log
fi
