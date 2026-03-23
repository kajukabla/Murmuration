#!/usr/bin/env node
/**
 * Browser-based benchmark using Playwright.
 * Launches Chrome with GPU, loads the simulation, waits for steady state,
 * then reads actual FPS from the HUD. Measures FULL pipeline (compute + render).
 *
 * Usage: node bench_browser.js [--boids N] [--warmup-sec N] [--measure-sec N]
 *
 * Requires: npx playwright install chromium (first time)
 * Requires: serve.py running on port 8080
 */

const { chromium } = require('playwright');

// Parse args
const args = {};
for (let i = 2; i < process.argv.length; i += 2) {
  if (process.argv[i].startsWith('--')) {
    args[process.argv[i].slice(2)] = parseInt(process.argv[i + 1]);
  }
}
const NUM_BOIDS = args.boids || 40000;
const WARMUP_SEC = args['warmup-sec'] || 15; // seconds to let boids cluster
const MEASURE_SEC = args['measure-sec'] || 10; // seconds to measure

async function run() {
  const browser = await chromium.launch({
    headless: false, // need real GPU for WebGPU
    args: [
      '--enable-unsafe-webgpu',
      '--enable-features=Vulkan,WebGPU',
      '--use-angle=metal',
      '--ignore-gpu-blocklist',
      '--disable-frame-rate-limit', // bypass VSync for accurate timing
      '--disable-gpu-vsync',
    ],
  });

  const page = await browser.newPage();
  // Set viewport to consistent size
  await page.setViewportSize({ width: 1280, height: 800 });

  // Navigate — add neighborMode=1 to default to radius mode
  const url = `http://localhost:8080?boids=${NUM_BOIDS}&mode=radius`;
  await page.goto(url, { waitUntil: 'domcontentloaded' });

  // Wait for WebGPU to init and first frame
  await page.waitForFunction(() => {
    const hud = document.getElementById('hud');
    return hud && hud.textContent.includes('fps');
  }, { timeout: 30000 });

  // Switch to radius mode + billboard additive via JS
  await page.evaluate(() => {
    // Find and set neighbor mode dropdown
    const nm = document.getElementById('neighbor-mode-select');
    if (nm) { nm.value = '1'; nm.dispatchEvent(new Event('change')); }
    // Find and set render mode dropdown
    const rm = document.getElementById('render-mode-select');
    if (rm) { rm.value = '1'; rm.dispatchEvent(new Event('change')); }
  });

  console.error(`Warming up ${NUM_BOIDS} boids for ${WARMUP_SEC}s...`);
  await page.waitForTimeout(WARMUP_SEC * 1000);

  // Measure: clear frame times, wait, then read raw timing data
  // First clear the accumulated frame times
  await page.evaluate(() => {
    // Expose a way to get raw frame times without VSync cap
    // Use requestAnimationFrame timing delta which IS capped by VSync
    // Instead, measure GPU submit time via performance.now() around the encoder
    window.__benchSamples = [];
    window.__benchActive = true;
  });

  // Inject measurement hook into the frame loop
  await page.evaluate((sec) => {
    const origRAF = window.requestAnimationFrame;
    let lastT = performance.now();
    const samples = window.__benchSamples;
    const maxSamples = sec * 120; // oversample
    function hookedFrame(cb) {
      origRAF(function(ts) {
        const now = performance.now();
        if (window.__benchActive && samples.length < maxSamples) {
          samples.push(now - lastT);
        }
        lastT = now;
        cb(ts);
      });
    }
    window.requestAnimationFrame = hookedFrame;
  }, MEASURE_SEC);

  console.error(`Measuring for ${MEASURE_SEC}s...`);
  await page.waitForTimeout(MEASURE_SEC * 1000);

  // Read results
  const samples = await page.evaluate(() => {
    window.__benchActive = false;
    return window.__benchSamples || [];
  });

  await browser.close();

  if (samples.length < 10) {
    console.log(JSON.stringify({ error: 'too few samples: ' + samples.length }));
    process.exit(1);
  }

  // Drop first 10% (warmup noise) and outliers > 200ms
  const cleaned = samples.slice(Math.floor(samples.length * 0.1)).filter(t => t < 200);
  cleaned.sort((a, b) => a - b);

  const avgMs = cleaned.reduce((a, b) => a + b, 0) / cleaned.length;
  const p50Ms = cleaned[Math.floor(cleaned.length * 0.5)];
  const p95Ms = cleaned[Math.floor(cleaned.length * 0.95)];
  const p99Ms = cleaned[Math.floor(cleaned.length * 0.99)];
  const maxMs = cleaned[cleaned.length - 1];

  const result = {
    num_boids: NUM_BOIDS,
    samples: cleaned.length,
    avg_fps: +(1000 / avgMs).toFixed(1),
    avg_ms: +avgMs.toFixed(1),
    p50_ms: +p50Ms.toFixed(1),
    p95_ms: +p95Ms.toFixed(1),
    p99_ms: +p99Ms.toFixed(1),
    max_ms: +maxMs.toFixed(1),
  };

  console.log(JSON.stringify(result));
}

run().catch(e => {
  console.error(e);
  process.exit(1);
});
