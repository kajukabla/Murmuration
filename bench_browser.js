#!/usr/bin/env node
/**
 * Browser-based benchmark that reuses an existing Chrome tab.
 * NO Playwright, NO new windows, NO focus stealing.
 *
 * Uses osascript to navigate the existing localhost:8080 tab,
 * waits for benchmark mode to complete, reads results from bench_result.json.
 *
 * Requires: serve.py running on port 8080 with POST /api/bench_result
 * Requires: An existing Chrome tab open to localhost:8080
 *
 * Usage: node bench_browser.js [--boids N] [--warmup-sec N] [--measure-sec N]
 */

const { execSync, execFileSync } = require('child_process');
const fs = require('fs');
const path = require('path');

const args = {};
for (let i = 2; i < process.argv.length; i += 2) {
  if (process.argv[i].startsWith('--')) {
    args[process.argv[i].slice(2)] = parseInt(process.argv[i + 1]);
  }
}
const NUM_BOIDS = args.boids || 40000;
const TIMEOUT = args.timeout || 60; // seconds

const RESULT_FILE = path.join(__dirname, 'bench_result.json');

// Clear any stale result
try { fs.unlinkSync(RESULT_FILE); } catch {}

// Navigate existing Chrome tab to benchmark URL (no focus steal)
const url = `http://localhost:8080?boids=${NUM_BOIDS}&mode=radius&benchmark=1`;
const script = `
tell application "Google Chrome"
  if (count of windows) > 0 then
    set found to false
    repeat with w in windows
      repeat with t in tabs of w
        if URL of t contains "localhost:8080" then
          set URL of t to "${url}"
          set found to true
          exit repeat
        end if
      end repeat
      if found then exit repeat
    end repeat
    if not found then
      -- Don't open new tab, just use active tab
      set URL of active tab of front window to "${url}"
    end if
  end if
end tell
`;

try {
  execFileSync('osascript', ['-e', script], { timeout: 5000 });
} catch (e) {
  console.error('Failed to navigate Chrome:', e.message);
  process.exit(1);
}

// Poll for bench_result.json (the page POSTs results when benchmark completes)
console.error(`Waiting for ${NUM_BOIDS} boids benchmark (timeout ${TIMEOUT}s)...`);
const deadline = Date.now() + TIMEOUT * 1000;

function poll() {
  while (Date.now() < deadline) {
    try {
      const data = fs.readFileSync(RESULT_FILE, 'utf8');
      const result = JSON.parse(data);
      if (result.num_boids === NUM_BOIDS) {
        fs.unlinkSync(RESULT_FILE); // consume it
        console.log(JSON.stringify(result));
        process.exit(0);
      }
    } catch {}
    // Sleep 500ms
    execSync('sleep 0.5');
  }
  console.error('Timeout waiting for benchmark result');
  console.log(JSON.stringify({ error: 'timeout', num_boids: NUM_BOIDS }));
  process.exit(1);
}

poll();
