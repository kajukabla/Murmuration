// === WebGPU Compute Simulation (Spatial-Hash Flocking) ===

const WORKGROUP_SIZE = 64;
const PARAMS_SIZE = 96; // 24 fields × 4 bytes = 96 (16-byte aligned)

export async function createSimulation(device, {
  numBoids = 5000,
  worldSize = 100.0,
  gridSize = 0, // 0 = auto-scale to boid count
} = {}) {
  // Auto-scale grid: ~1-2 boids per cell on average
  if (gridSize <= 0) {
    gridSize = Math.max(8, Math.min(128, Math.round(Math.cbrt(numBoids) * 0.8)));
  }
  const GRID_CELLS = gridSize ** 3;
  const cellSize = worldSize / gridSize;

  // --- Shader module ---
  const code = await (await fetch('shader.wgsl')).text();
  const module = device.createShaderModule({ code });

  // --- Params uniform ---
  const paramsData = new ArrayBuffer(PARAMS_SIZE);
  const u = new Uint32Array(paramsData);
  const f = new Float32Array(paramsData);

  // Fixed fields
  u[0]  = numBoids;
  u[1]  = gridSize;
  u[2]  = GRID_CELLS;
  f[3]  = worldSize;
  f[4]  = cellSize;
  f[14] = 0.016;  // dt

  function writeParams(p) {
    f[5]  = p.visualRange;
    f[6]  = p.visualRange * p.visualRange;
    f[7]  = p.separationDist;
    f[8]  = p.separationDist * p.separationDist;
    f[9]  = p.alignFactor;
    f[10] = p.cohesionFactor;
    f[11] = p.separationFactor;
    f[12] = p.maxSpeed;
    f[13] = p.minSpeed;
    f[15] = p.turnFactor;
    f[16] = p.smoothing;
    f[17] = p.simSpeed ?? 1.0;
    f[18] = p.sizeRandomness ?? 0.0;
    f[19] = p.dragFactor ?? 0.3;
    u[20] = p.gradientId ?? 0;
    u[21] = p.colorSource ?? 0;
    f[22] = p.sphereRadius ?? 100.0;
    device.queue.writeBuffer(paramsBuffer, 0, paramsData);
  }

  const paramsBuffer = device.createBuffer({
    size: PARAMS_SIZE,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });

  // --- Boid buffers (ping-pong) ---
  // Boid struct: 64 bytes = 16 floats
  // pos(3) + size_factor(1) + vel(3) + speed(1) + heading(3) + neighbor_count(1) +
  // dir_change(1) + flock_alignment(1) + sep_pressure(1) + density(1)
  const BOID_FLOATS = 16;
  const boidBytes = numBoids * BOID_FLOATS * 4;
  const initData = new Float32Array(numBoids * BOID_FLOATS);
  for (let i = 0; i < numBoids; i++) {
    const o = i * BOID_FLOATS;
    const r = worldSize * 0.35;
    // pos
    // Spawn inside sphere using rejection sampling
    let px, py, pz;
    do {
      px = (Math.random() - 0.5) * 2;
      py = (Math.random() - 0.5) * 2;
      pz = (Math.random() - 0.5) * 2;
    } while (px*px + py*py + pz*pz > 1);
    initData[o]     = px * r;
    initData[o + 1] = py * r;
    initData[o + 2] = pz * r;
    // size_factor (default 1.0, randomized via sizeRandomness param)
    initData[o + 3] = 1.0;
    // vel
    const vx = (Math.random() - 0.5) * 4;
    const vy = (Math.random() - 0.5) * 4;
    const vz = (Math.random() - 0.5) * 4;
    initData[o + 4] = vx;
    initData[o + 5] = vy;
    initData[o + 6] = vz;
    // speed
    initData[o + 7] = Math.hypot(vx, vy, vz);
    // heading = normalize(vel)
    const vlen = initData[o + 7] || 1;
    initData[o + 8]  = vx / vlen;
    initData[o + 9]  = vy / vlen;
    initData[o + 10] = vz / vlen;
    // neighbor_count, dir_change, flock_alignment, sep_pressure, density = 0
  }

  // Apply size randomness
  function applySizeRandomness(randomness) {
    for (let i = 0; i < numBoids; i++) {
      const sf = 1.0 + (Math.random() - 0.5) * 2 * randomness;
      initData[i * BOID_FLOATS + 3] = Math.max(0.3, Math.min(2.0, sf));
    }
    // Write size_factor to both ping-pong buffers
    // (only the size_factor field at offset 3 per boid matters, but writing full buffer is simpler)
    device.queue.writeBuffer(boidA, 0, initData);
    device.queue.writeBuffer(boidB, 0, initData);
  }

  const makeBoidBuf = (label, data) => {
    const buf = device.createBuffer({
      label,
      size: boidBytes,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    if (data) device.queue.writeBuffer(buf, 0, data);
    return buf;
  };
  const boidA = makeBoidBuf('boids-A', initData);
  const boidB = makeBoidBuf('boids-B', null);

  // --- Grid buffers ---
  const makeGridBuf = (label, count) => device.createBuffer({
    label,
    size: count * 4,
    usage: GPUBufferUsage.STORAGE,
  });
  const cellCounts      = makeGridBuf('cell-counts', GRID_CELLS);
  const cellOffsets     = makeGridBuf('cell-offsets', GRID_CELLS);
  const boidCells       = makeGridBuf('boid-cells', numBoids);
  const sortedIndices   = makeGridBuf('sorted-idx', numBoids);
  const scatterCounters = makeGridBuf('scatter-ctr', GRID_CELLS);

  // --- Bind group layout ---
  const bgl = device.createBindGroupLayout({
    entries: Array.from({ length: 8 }, (_, i) => ({
      binding: i,
      visibility: GPUShaderStage.COMPUTE,
      buffer: { type: i === 0 ? 'uniform' : 'storage' },
    })),
  });
  const pipeLayout = device.createPipelineLayout({ bindGroupLayouts: [bgl] });

  const makeBG = (src, dst) => device.createBindGroup({
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
  const bgA = makeBG(boidA, boidB);
  const bgB = makeBG(boidB, boidA);

  // --- Compute pipelines ---
  const pipe = (entryPoint) => device.createComputePipeline({
    layout: pipeLayout,
    compute: { module, entryPoint },
  });
  const clearPipe   = pipe('clear_grid');
  const assignPipe  = pipe('assign_cells');
  const prefixPipe  = pipe('prefix_sum');
  const scatterPipe = pipe('scatter');
  const flockPipe   = pipe('flock');

  const gridWG = Math.ceil(GRID_CELLS / WORKGROUP_SIZE);
  const boidWG = Math.ceil(numBoids / WORKGROUP_SIZE);

  // --- Auto-range stats ---
  const statsBuf = device.createBuffer({
    label: 'stats',
    size: 8, // 2 × u32 (min, max as bitcast f32)
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
  });
  const statsReadBuf = device.createBuffer({
    label: 'stats-read',
    size: 8,
    usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
  });

  // Stats uses group(0) for params+boids and group(1) for stats buffer
  const statsBGL = device.createBindGroupLayout({
    entries: [
      { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
    ],
  });
  const statsBG = device.createBindGroup({
    layout: statsBGL,
    entries: [{ binding: 0, resource: { buffer: statsBuf } }],
  });

  const statsPipeLayout = device.createPipelineLayout({
    bindGroupLayouts: [bgl, statsBGL],
  });
  const clearStatsPipe = device.createComputePipeline({
    layout: statsPipeLayout,
    compute: { module, entryPoint: 'clear_stats' },
  });
  const computeStatsPipe = device.createComputePipeline({
    layout: statsPipeLayout,
    compute: { module, entryPoint: 'compute_stats' },
  });

  let step = 0;
  let autoRangeEnabled = false;
  let statsReading = false;
  let smoothMin = 0, smoothMax = 1;
  let emaInitialized = false;

  return {
    numBoids,
    boidA,
    boidB,
    setParams: writeParams,
    applySizeRandomness,

    set autoRange(v) { autoRangeEnabled = v; if (!v) emaInitialized = false; },
    get autoMin() { return smoothMin; },
    get autoMax() { return smoothMax; },

    /** Run one sim step. Set lastStep=true on the final step of the frame. */
    update(encoder, lastStep = true) {
      const bg = step % 2 === 0 ? bgA : bgB;
      const passes = [
        [clearPipe,   gridWG],
        [assignPipe,  boidWG],
        [prefixPipe,  1],
        [scatterPipe, boidWG],
        [flockPipe,   boidWG],
      ];
      for (const [pipeline, wg] of passes) {
        const p = encoder.beginComputePass();
        p.setPipeline(pipeline);
        p.setBindGroup(0, bg);
        p.dispatchWorkgroups(wg);
        p.end();
      }

      // Only compute stats on the last step of the frame (avoids races with multi-step)
      if (false && autoRangeEnabled && lastStep && !statsReading) {
        const c = encoder.beginComputePass();
        c.setPipeline(clearStatsPipe);
        c.setBindGroup(0, bg);
        c.setBindGroup(1, statsBG);
        c.dispatchWorkgroups(1);
        c.end();
        const s = encoder.beginComputePass();
        s.setPipeline(computeStatsPipe);
        s.setBindGroup(0, bg);
        s.setBindGroup(1, statsBG);
        s.dispatchWorkgroups(boidWG);
        s.end();
        encoder.copyBufferToBuffer(statsBuf, 0, statsReadBuf, 0, 8);
      }

      step++;
    },

    /** Call after queue.submit to read back stats (non-blocking) */
    async readStats() {
      if (!autoRangeEnabled || statsReading) return;
      statsReading = true;
      try {
        await statsReadBuf.mapAsync(GPUMapMode.READ);
        const mapped = statsReadBuf.getMappedRange();
        const data = new Uint32Array(mapped.slice(0));
        statsReadBuf.unmap();
        // Bitcast u32 back to f32
        const f = new Float32Array(data.buffer);
        const rawMin = f[0], rawMax = f[1];
        if (isFinite(rawMin) && isFinite(rawMax) && rawMax >= rawMin) {
          if (!emaInitialized) {
            // Snap to first valid reading
            smoothMin = rawMin;
            smoothMax = rawMax;
            emaInitialized = true;
          } else {
            // Very slow EMA to avoid flickering
            const alpha = 0.005;
            smoothMin = smoothMin * (1 - alpha) + rawMin * alpha;
            smoothMax = smoothMax * (1 - alpha) + rawMax * alpha;
          }
        }
      } catch {}
      statsReading = false;
    },

    currentBuffer() {
      return step % 2 === 1 ? boidB : boidA;
    },
  };
}
