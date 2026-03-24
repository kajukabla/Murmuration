#!/usr/bin/env node
/**
 * Browser benchmark coordinator.
 * Writes a bench_request.json, the page polls for it and runs the benchmark,
 * then posts results to bench_result.json.
 *
 * NO Playwright, NO osascript, NO new windows, NO focus stealing.
 * The page handles everything via polling.
 *
 * Usage: node bench_browser.js [--boids N] [--timeout N]
 */

const fs = require('fs');
const path = require('path');
const { execSync } = require('child_process');

const args = {};
for (let i = 2; i < process.argv.length; i += 2) {
  if (process.argv[i].startsWith('--')) {
    args[process.argv[i].slice(2)] = parseInt(process.argv[i + 1]);
  }
}
const NUM_BOIDS = args.boids || 40000;
const TIMEOUT = args.timeout || 90; // 30s warmup + 120 frames measurement + margin

const DIR = __dirname;
const REQUEST_FILE = path.join(DIR, 'bench_request.json');
const RESULT_FILE = path.join(DIR, 'bench_result.json');

// Clear stale files
try { fs.unlinkSync(RESULT_FILE); } catch {}
try { fs.unlinkSync(REQUEST_FILE); } catch {}

// Write benchmark request — the page will pick this up via polling
fs.writeFileSync(REQUEST_FILE, JSON.stringify({ boids: NUM_BOIDS }));
console.error(`Requesting benchmark: ${NUM_BOIDS} boids (timeout ${TIMEOUT}s)`);

// Poll for result
const deadline = Date.now() + TIMEOUT * 1000;
while (Date.now() < deadline) {
  try {
    const data = fs.readFileSync(RESULT_FILE, 'utf8');
    const result = JSON.parse(data);
    if (result.num_boids === NUM_BOIDS || result.num_boids > 0) {
      try { fs.unlinkSync(RESULT_FILE); } catch {}
      console.log(JSON.stringify(result));
      process.exit(0);
    }
  } catch {}
  execSync('sleep 0.5');
}

// Timeout — clean up request
try { fs.unlinkSync(REQUEST_FILE); } catch {}
console.error('Timeout waiting for benchmark');
console.log(JSON.stringify({ error: 'timeout', num_boids: NUM_BOIDS }));
process.exit(1);
