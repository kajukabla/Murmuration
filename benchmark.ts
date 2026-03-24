#!/usr/bin/env -S deno run --unstable-webgpu --allow-read
/**
 * Headless WebGPU benchmark with murmuration quality metrics.
 * Runs simulation to steady state, then measures behavioral quality.
 *
 * Output JSON includes:
 * - Frame timing (avg_ms, p99_ms)
 * - Murmuration quality score (0-1) and sub-metrics
 *
 * Usage: deno run --unstable-webgpu --allow-read benchmark.ts [--boids N] [--frames N]
 */

const args = parseArgs(Deno.args);
const NUM_BOIDS = args.boids ?? 20000;
const NUM_FRAMES = args.frames ?? 60;
const WARMUP_FRAMES = 200; // enough for compute-only steady state

const simCode = await Deno.readTextFile("simulation.js");
const WORKGROUP_SIZE = parseInt(simCode.match(/const WORKGROUP_SIZE\s*=\s*(\d+)/)?.[1] ?? "64");
const PARAMS_SIZE = parseInt(simCode.match(/const PARAMS_SIZE\s*=\s*(\d+)/)?.[1] ?? "96");

const GRID_SIZE = Math.max(16, Math.min(80, Math.round(Math.cbrt(NUM_BOIDS / 4))));
const GRID_CELLS = GRID_SIZE ** 3;
const SPHERE_R = parseFloat((await Deno.readTextFile('index.html')).match(/sphereRadius:\s*(\d+)/)?.[1] ?? '100');
const WORLD_SIZE = SPHERE_R * 2.5;
const CELL_SIZE = WORLD_SIZE / GRID_SIZE;

const adapter = await navigator.gpu?.requestAdapter();
if (!adapter) { console.error("No WebGPU adapter"); Deno.exit(1); }
const device = await adapter.requestDevice();

const shaderCode = await Deno.readTextFile("shader.wgsl");
const module = device.createShaderModule({ code: shaderCode });

// --- Params ---
const paramsData = new ArrayBuffer(PARAMS_SIZE);
const u32 = new Uint32Array(paramsData);
const f32 = new Float32Array(paramsData);

u32[0] = NUM_BOIDS;
u32[1] = GRID_SIZE;
u32[2] = GRID_CELLS;
f32[3] = WORLD_SIZE;
f32[4] = CELL_SIZE;
// Read preset values from index.html (Murmuration preset)
const htmlCode = await Deno.readTextFile("index.html");
const presetMatch = htmlCode.match(/'Murmuration':\s*\{([^}]+)\}/);
const preset: Record<string, number> = {};
if (presetMatch) {
  for (const m of presetMatch[1].matchAll(/(\w+):\s*([\d.]+)/g)) {
    preset[m[1]] = parseFloat(m[2]);
  }
}
f32[5]  = preset.visualRange ?? 15.0;
f32[6]  = (preset.visualRange ?? 15.0) ** 2;
f32[7]  = preset.separationDist ?? 2.0;
f32[8]  = (preset.separationDist ?? 2.0) ** 2;
f32[9]  = preset.alignFactor ?? 0.12;
f32[10] = preset.cohesionFactor ?? 0.003;
f32[11] = preset.separationFactor ?? 0.08;
f32[12] = preset.maxSpeed ?? 8.0;
f32[13] = preset.minSpeed ?? 3.0;
f32[14] = 0.016;
f32[15] = preset.turnFactor ?? 0.3;
f32[16] = preset.smoothing ?? 0.10;
f32[17] = 1.0;  // sim_speed
f32[18] = 0.0;  // size_randomness
f32[19] = 0.0;  // drag_factor
u32[20] = 0;
u32[21] = 0;
f32[22] = preset.sphereRadius ?? 120.0;
u32[23] = 0;  // frame_count — updated each step

const paramsBuffer = device.createBuffer({
  size: PARAMS_SIZE,
  usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
});
// Write full params once at init
device.queue.writeBuffer(paramsBuffer, 0, new Uint8Array(paramsData));

// --- Boid buffers ---
const BOID_FLOATS = 16;
const BOID_BYTES = NUM_BOIDS * BOID_FLOATS * 4;
const initData = new Float32Array(NUM_BOIDS * BOID_FLOATS);
const spawnR = SPHERE_R * 0.3;
for (let i = 0; i < NUM_BOIDS; i++) {
  const o = i * BOID_FLOATS;
  // Spawn in sphere
  let px, py, pz;
  do { px = (Math.random()-0.5)*2; py = (Math.random()-0.5)*2; pz = (Math.random()-0.5)*2; }
  while (px*px+py*py+pz*pz > 1);
  initData[o] = px * spawnR;
  initData[o+1] = py * spawnR;
  initData[o+2] = pz * spawnR;
  initData[o+3] = 1.0;
  const vx = (Math.random()-0.5)*4, vy = (Math.random()-0.5)*4, vz = (Math.random()-0.5)*4;
  initData[o+4] = vx; initData[o+5] = vy; initData[o+6] = vz;
  initData[o+7] = Math.hypot(vx,vy,vz);
  const vl = initData[o+7] || 1;
  initData[o+8] = vx/vl; initData[o+9] = vy/vl; initData[o+10] = vz/vl;
}

function makeBuf(size: number, usage: number, data?: Float32Array) {
  const buf = device.createBuffer({ size, usage });
  if (data) device.queue.writeBuffer(buf, 0, data);
  return buf;
}

const boidA = makeBuf(BOID_BYTES, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST, initData);
const boidB = makeBuf(BOID_BYTES, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST);
const cellCounts = makeBuf(GRID_CELLS * 4, GPUBufferUsage.STORAGE);
const cellOffsets = makeBuf(GRID_CELLS * 4, GPUBufferUsage.STORAGE);
const boidCells = makeBuf(NUM_BOIDS * 4, GPUBufferUsage.STORAGE);
const sortedIndices = makeBuf(NUM_BOIDS * 4, GPUBufferUsage.STORAGE);
const scatterCounters = makeBuf(GRID_CELLS * 4, GPUBufferUsage.STORAGE);

// --- Metrics buffers (group 1) ---
const statsBuf = makeBuf(8, GPUBufferUsage.STORAGE);  // auto-range stats (binding 0)
const metricsBuf = makeBuf(64, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC);  // quality metrics (binding 1)
const metricsReadBuf = device.createBuffer({
  size: 64, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
});

// --- Bind group layouts ---
const bgl = device.createBindGroupLayout({
  entries: Array.from({ length: 8 }, (_, i) => ({
    binding: i, visibility: GPUShaderStage.COMPUTE,
    buffer: { type: (i === 0 ? "uniform" : "storage") as GPUBufferBindingType },
  })),
});

const metricsBGL = device.createBindGroupLayout({
  entries: [
    { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" as GPUBufferBindingType } },
    { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" as GPUBufferBindingType } },
  ],
});

// Sim pipelines use group(0) only
const simPipeLayout = device.createPipelineLayout({ bindGroupLayouts: [bgl] });
// Metrics pipelines use group(0) + group(1)
const metricsPipeLayout = device.createPipelineLayout({ bindGroupLayouts: [bgl, metricsBGL] });

function makeBG(src: GPUBuffer, dst: GPUBuffer) {
  return device.createBindGroup({
    layout: bgl,
    entries: [
      { binding: 0, resource: { buffer: paramsBuffer } },
      { binding: 1, resource: { buffer: src } },
      { binding: 2, resource: { buffer: dst } },
      { binding: 3, resource: { buffer: cellCounts } },
      { binding: 4, resource: { buffer: cellOffsets } },
      { binding: 5, resource: { buffer: boidCells } },
      { binding: 6, resource: { buffer: sortedIndices } },
      { binding: 7, resource: { buffer: scatterCounters } },
    ],
  });
}
const bgA = makeBG(boidA, boidB);
const bgB = makeBG(boidB, boidA);

const metricsBG = device.createBindGroup({
  layout: metricsBGL,
  entries: [
    { binding: 0, resource: { buffer: statsBuf } },
    { binding: 1, resource: { buffer: metricsBuf } },
  ],
});

// --- Compute pipelines ---
const makeSP = (ep: string) => device.createComputePipeline({ layout: simPipeLayout, compute: { module, entryPoint: ep } });
const makeMP = (ep: string) => device.createComputePipeline({ layout: metricsPipeLayout, compute: { module, entryPoint: ep } });

const clearPipe = makeSP("clear_grid_linked");
const assignPipe = makeSP("assign_linked");
const flockRadiusPipe = makeSP("flock_radius_linked");
const driftPipe = makeSP("drift");
const clearMetricsPipe = makeMP("clear_metrics");
const computeMetricsPipe = makeMP("compute_metrics");

const gridWG = Math.ceil(GRID_CELLS / 128); // clear_grid_linked uses wg=128
const boidWG = Math.ceil(NUM_BOIDS / 128); // assign_linked, flock_radius_linked, drift use wg=128

const frameCountBuf = new Uint32Array(1);
function encodeFrame(encoder: GPUCommandEncoder, step: number) {
  // Update only frame_count (4 bytes at offset 92) instead of full 96-byte buffer
  frameCountBuf[0] = step;
  device.queue.writeBuffer(paramsBuffer, 92, frameCountBuf);

  const bg = step % 2 === 0 ? bgA : bgB;

  // 2-tier schedule: grid+flock_radius 1/8, drift 7/8 (matches simulation.js)
  if (step % 8 === 0) {
    for (const [pipe, wg] of [[clearPipe, gridWG], [assignPipe, boidWG], [flockRadiusPipe, boidWG]] as [GPUComputePipeline, number][]) {
      const p = encoder.beginComputePass();
      p.setPipeline(pipe);
      p.setBindGroup(0, bg);
      p.dispatchWorkgroups(wg);
      p.end();
    }
  } else {
    const p = encoder.beginComputePass();
    p.setPipeline(driftPipe);
    p.setBindGroup(0, bg);
    p.dispatchWorkgroups(boidWG);
    p.end();
  }
}

function encodeMetrics(encoder: GPUCommandEncoder, step: number) {
  const bg = step % 2 === 0 ? bgA : bgB;
  // Clear
  const c = encoder.beginComputePass();
  c.setPipeline(clearMetricsPipe);
  c.setBindGroup(0, bg);
  c.setBindGroup(1, metricsBG);
  c.dispatchWorkgroups(1);
  c.end();
  // Compute
  const s = encoder.beginComputePass();
  s.setPipeline(computeMetricsPipe);
  s.setBindGroup(0, bg);
  s.setBindGroup(1, metricsBG);
  s.dispatchWorkgroups(boidWG);
  s.end();
  // Copy to readback
  encoder.copyBufferToBuffer(metricsBuf, 0, metricsReadBuf, 0, 64);
}

async function readMetrics(): Promise<Float32Array> {
  await metricsReadBuf.mapAsync(GPUMapMode.READ);
  const data = new Float32Array(metricsReadBuf.getMappedRange().slice(0));
  metricsReadBuf.unmap();
  return data;
}

function computeScore(raw: Float32Array, raw2: Float32Array | null) {
  const n = raw[10] || 1; // total boids processed (slot 10, as u32 → read via f32)
  // Reinterpret slot 10 as u32
  const nBoids = new Uint32Array(raw.buffer)[10] || 1;
  const cohesionCount = new Uint32Array(raw.buffer)[0]; // slot 0: boids with 5+ neighbors (u32)

  const cohesion_ratio = cohesionCount / nBoids;
  const avg_neighbors = raw[1] / nBoids; // slot 1: sum of neighbor_count
  const velocity_corr = raw[2] / nBoids; // slot 2: sum of flock_alignment

  // Position variance for aspect ratio
  const mx = raw[3] / nBoids, my = raw[4] / nBoids, mz = raw[5] / nBoids;
  const vx = raw[6] / nBoids - mx*mx;
  const vy = raw[7] / nBoids - my*my;
  const vz = raw[8] / nBoids - mz*mz;
  const variances = [Math.max(vx, 0.01), Math.max(vy, 0.01), Math.max(vz, 0.01)].sort((a,b) => b-a);
  const aspect_ratio = variances[0] / variances[2]; // largest / smallest

  // Density uniformity (CV of neighbor counts)
  const avg_n = raw[1] / nBoids;
  const avg_n2 = raw[9] / nBoids;
  const var_n = Math.max(0, avg_n2 - avg_n * avg_n);
  const density_cv = avg_n > 0.01 ? Math.sqrt(var_n) / avg_n : 10.0;

  // Shape dynamics (compare two measurements)
  let dynamics = 0.05; // default if no second measurement
  if (raw2) {
    const n2 = new Uint32Array(raw2.buffer)[10] || 1;
    const mx2 = raw2[3]/n2, my2 = raw2[4]/n2, mz2 = raw2[5]/n2;
    const vx2 = raw2[6]/n2 - mx2*mx2, vy2 = raw2[7]/n2 - my2*my2, vz2 = raw2[8]/n2 - mz2*mz2;
    const vars2 = [Math.max(vx2,0.01), Math.max(vy2,0.01), Math.max(vz2,0.01)].sort((a,b) => b-a);
    const ar2 = vars2[0] / vars2[2];
    dynamics = Math.abs(aspect_ratio - ar2) / Math.max(aspect_ratio, 0.01);
  }

  // Composite score
  const cr = Math.min(cohesion_ratio, 1.0);
  const vc = Math.max(velocity_corr, 0.0);
  const ar_norm = Math.min(aspect_ratio, 5.0) / 5.0;
  const du_norm = 1.0 / (1.0 + density_cv);
  const dyn_norm = Math.min(dynamics * 20, 1.0);

  const score = cr * cr * vc * ar_norm * du_norm * dyn_norm;

  return {
    score: +score.toFixed(4),
    cohesion_ratio: +cr.toFixed(3),
    velocity_corr: +vc.toFixed(3),
    aspect_ratio: +aspect_ratio.toFixed(2),
    density_cv: +density_cv.toFixed(3),
    dynamics: +dynamics.toFixed(4),
    avg_neighbors: +avg_neighbors.toFixed(1),
  };
}

// === Main ===
// Warmup (batched)
const BATCH = 50;
for (let i = 0; i < WARMUP_FRAMES; i += BATCH) {
  const n = Math.min(BATCH, WARMUP_FRAMES - i);
  const encoder = device.createCommandEncoder();
  for (let j = 0; j < n; j++) encodeFrame(encoder, i + j);
  device.queue.submit([encoder.finish()]);
  await device.queue.onSubmittedWorkDone();
}

// Measure metrics at steady state
let step = WARMUP_FRAMES;
{
  const encoder = device.createCommandEncoder();
  encodeMetrics(encoder, step);
  device.queue.submit([encoder.finish()]);
  await device.queue.onSubmittedWorkDone();
}
const metrics1 = await readMetrics();

// Run 100 more steps, measure again for dynamics
for (let i = 0; i < 100; i++) {
  const encoder = device.createCommandEncoder();
  encodeFrame(encoder, step + i);
  device.queue.submit([encoder.finish()]);
}
step += 100;
await device.queue.onSubmittedWorkDone();

{
  const encoder = device.createCommandEncoder();
  encodeMetrics(encoder, step);
  device.queue.submit([encoder.finish()]);
  await device.queue.onSubmittedWorkDone();
}
const metrics2 = await readMetrics();

// Timing (quick)
const frameTimes: number[] = [];
for (let b = 0; b < Math.ceil(NUM_FRAMES/BATCH); b++) {
  const n = Math.min(BATCH, NUM_FRAMES - b*BATCH);
  const t0 = performance.now();
  const encoder = device.createCommandEncoder();
  for (let i = 0; i < n; i++) encodeFrame(encoder, step++);
  device.queue.submit([encoder.finish()]);
  await device.queue.onSubmittedWorkDone();
  const pf = (performance.now() - t0) / n;
  for (let i = 0; i < n; i++) frameTimes.push(pf);
}
frameTimes.sort((a,b) => a-b);
const avg = frameTimes.reduce((a,b) => a+b, 0) / frameTimes.length;

const result = computeScore(metrics1, metrics2);

const max_boids = Math.round(NUM_BOIDS * (16.6 / avg) / 1000) * 1000;
console.log(JSON.stringify({
  num_boids: NUM_BOIDS,
  avg_ms: +avg.toFixed(2),
  p99_ms: +frameTimes[Math.floor(frameTimes.length * 0.99)].toFixed(2),
  ...result,
}));
console.log(`max_boids: ${max_boids}`);

device.destroy();
Deno.exit(0);

function parseArgs(args: string[]): Record<string, number> {
  const result: Record<string, number> = {};
  for (let i = 0; i < args.length; i++) {
    if (args[i].startsWith("--") && i + 1 < args.length) {
      result[args[i].slice(2)] = parseInt(args[i + 1]);
      i++;
    }
  }
  return result;
}
