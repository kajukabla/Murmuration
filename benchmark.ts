#!/usr/bin/env -S deno run --unstable-webgpu --allow-read
/**
 * Headless WebGPU benchmark for the flocking simulation.
 * Mirrors the exact compute pipeline from simulation.js.
 * Outputs JSON with frame timing stats to stdout.
 *
 * Usage: deno run --unstable-webgpu --allow-read benchmark.ts [--boids N] [--frames N]
 */

// --- Parse CLI args ---
const args = parseArgs(Deno.args);
const NUM_BOIDS = args.boids ?? 5000;
const NUM_FRAMES = args.frames ?? 120;
const WARMUP_FRAMES = 20;

// --- Read simulation constants from simulation.js ---
const simCode = await Deno.readTextFile("simulation.js");
const WORKGROUP_SIZE = parseInt(simCode.match(/const WORKGROUP_SIZE\s*=\s*(\d+)/)?.[1] ?? "64");
const PARAMS_SIZE = parseInt(simCode.match(/const PARAMS_SIZE\s*=\s*(\d+)/)?.[1] ?? "80");

// Extract gridSize default from simulation.js
const gridSizeMatch = simCode.match(/gridSize\s*=\s*(\d+)/);
const GRID_SIZE = parseInt(gridSizeMatch?.[1] ?? "32");
const GRID_CELLS = GRID_SIZE ** 3;
const WORLD_SIZE = 100.0;
const CELL_SIZE = WORLD_SIZE / GRID_SIZE;

// --- WebGPU init ---
const adapter = await navigator.gpu?.requestAdapter();
if (!adapter) {
  console.error("No WebGPU adapter found");
  Deno.exit(1);
}
const device = await adapter.requestDevice();

// --- Load shader ---
const shaderCode = await Deno.readTextFile("shader.wgsl");
const module = device.createShaderModule({ code: shaderCode });

// --- Params uniform (matches simulation.js layout exactly) ---
const paramsData = new ArrayBuffer(PARAMS_SIZE);
const u32 = new Uint32Array(paramsData);
const f32 = new Float32Array(paramsData);

u32[0] = NUM_BOIDS;
u32[1] = GRID_SIZE;
u32[2] = GRID_CELLS;
f32[3] = WORLD_SIZE;
f32[4] = CELL_SIZE;
// Murmuration preset defaults
f32[5] = 5.0;          // visual_range
f32[6] = 25.0;         // visual_range_sq
f32[7] = 1.5;          // separation_dist
f32[8] = 2.25;         // separation_dist_sq
f32[9] = 0.08;         // align_factor
f32[10] = 0.006;       // cohesion_factor
f32[11] = 0.5;         // separation_factor
f32[12] = 8.0;         // max_speed
f32[13] = 2.5;         // min_speed
f32[14] = 0.016;       // dt
f32[15] = 0.5;         // turn_factor
f32[16] = 0.04;        // smoothing

const paramsBuffer = device.createBuffer({
  size: PARAMS_SIZE,
  usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
});
device.queue.writeBuffer(paramsBuffer, 0, new Uint8Array(paramsData));

// --- Boid buffers (48 bytes per boid: pos+pad, vel+pad, heading+pad) ---
const BOID_BYTES = NUM_BOIDS * 48;
const initData = new Float32Array(NUM_BOIDS * 12);
for (let i = 0; i < NUM_BOIDS; i++) {
  const o = i * 12;
  const r = WORLD_SIZE * 0.35;
  initData[o] = (Math.random() - 0.5) * 2 * r;
  initData[o + 1] = (Math.random() - 0.5) * 2 * r;
  initData[o + 2] = (Math.random() - 0.5) * 2 * r;
  const vx = (Math.random() - 0.5) * 4;
  const vy = (Math.random() - 0.5) * 4;
  const vz = (Math.random() - 0.5) * 4;
  initData[o + 4] = vx;
  initData[o + 5] = vy;
  initData[o + 6] = vz;
  const vlen = Math.hypot(vx, vy, vz) || 1;
  initData[o + 8] = vx / vlen;
  initData[o + 9] = vy / vlen;
  initData[o + 10] = vz / vlen;
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

// --- Bind group layout + pipeline layout ---
const bgl = device.createBindGroupLayout({
  entries: Array.from({ length: 8 }, (_, i) => ({
    binding: i,
    visibility: GPUShaderStage.COMPUTE,
    buffer: { type: (i === 0 ? "uniform" : "storage") as GPUBufferBindingType },
  })),
});
const pipeLayout = device.createPipelineLayout({ bindGroupLayouts: [bgl] });

// --- Bind groups (ping-pong) ---
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

// --- Compute pipelines ---
function makePipe(entryPoint: string) {
  return device.createComputePipeline({
    layout: pipeLayout,
    compute: { module, entryPoint },
  });
}
const clearPipe = makePipe("clear_grid");
const assignPipe = makePipe("assign_cells");
const prefixPipe = makePipe("prefix_sum");
const scatterPipe = makePipe("scatter");
const flockPipe = makePipe("flock");

const gridWG = Math.ceil(GRID_CELLS / WORKGROUP_SIZE);
const boidWG = Math.ceil(NUM_BOIDS / WORKGROUP_SIZE);

// --- Run one simulation frame ---
function encodeFrame(encoder: GPUCommandEncoder, step: number) {
  const bg = step % 2 === 0 ? bgA : bgB;
  const passes: [GPUComputePipeline, number][] = [
    [clearPipe, gridWG],
    [assignPipe, boidWG],
    [prefixPipe, 1],
    [scatterPipe, boidWG],
    [flockPipe, boidWG],
  ];
  for (const [pipeline, wg] of passes) {
    const p = encoder.beginComputePass();
    p.setPipeline(pipeline);
    p.setBindGroup(0, bg);
    p.dispatchWorkgroups(wg);
    p.end();
  }
}

// --- Benchmark loop ---
// Batch multiple frames per submit to amortize Deno WebGPU dispatch overhead.
// Then divide total time by batch size to get per-frame cost.
const BATCH_SIZE = 10;
const frameTimes: number[] = [];

// Warmup
{
  const encoder = device.createCommandEncoder();
  for (let i = 0; i < WARMUP_FRAMES; i++) {
    encodeFrame(encoder, i);
  }
  device.queue.submit([encoder.finish()]);
  await device.queue.onSubmittedWorkDone();
}

// Measured frames (in batches)
const totalBatches = Math.ceil(NUM_FRAMES / BATCH_SIZE);
let step = WARMUP_FRAMES;
for (let b = 0; b < totalBatches; b++) {
  const batchFrames = Math.min(BATCH_SIZE, NUM_FRAMES - b * BATCH_SIZE);
  const t0 = performance.now();
  const encoder = device.createCommandEncoder();
  for (let i = 0; i < batchFrames; i++) {
    encodeFrame(encoder, step++);
  }
  device.queue.submit([encoder.finish()]);
  await device.queue.onSubmittedWorkDone();
  const t1 = performance.now();
  const perFrame = (t1 - t0) / batchFrames;
  for (let i = 0; i < batchFrames; i++) {
    frameTimes.push(perFrame);
  }
}

// --- Compute stats ---
frameTimes.sort((a, b) => a - b);
const avg = frameTimes.reduce((a, b) => a + b, 0) / frameTimes.length;
const p99Index = Math.floor(frameTimes.length * 0.99);
const p99 = frameTimes[p99Index];
const min = frameTimes[0];
const max = frameTimes[frameTimes.length - 1];

console.log(JSON.stringify({
  num_boids: NUM_BOIDS,
  frames: NUM_FRAMES,
  avg_ms: +avg.toFixed(2),
  p99_ms: +p99.toFixed(2),
  min_ms: +min.toFixed(2),
  max_ms: +max.toFixed(2),
}));

// Cleanup
device.destroy();
Deno.exit(0);

// --- Arg parser ---
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
