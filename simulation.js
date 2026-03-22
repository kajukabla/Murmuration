// === WebGPU Compute Simulation (Spatial-Hash Flocking) ===

const WORKGROUP_SIZE = 64;

export async function createSimulation(device, {
  numBoids = 5000,
  worldSize = 50.0,
  gridSize = 32,
} = {}) {
  const GRID_CELLS = gridSize ** 3;
  const cellSize = worldSize / gridSize;

  // --- Shader module ---
  const code = await (await fetch('shader.wgsl')).text();
  const module = device.createShaderModule({ code });

  // --- Params uniform ---
  const paramsData = new ArrayBuffer(64);
  const u = new Uint32Array(paramsData);
  const f = new Float32Array(paramsData);
  u[0]  = numBoids;          // num_boids
  u[1]  = gridSize;          // grid_size
  u[2]  = GRID_CELLS;        // grid_cells
  f[3]  = worldSize;         // world_size
  f[4]  = cellSize;          // cell_size
  f[5]  = 3.0;               // visual_range
  f[6]  = 9.0;               // visual_range_sq
  f[7]  = 1.2;               // separation_dist
  f[8]  = 1.44;              // separation_dist_sq
  f[9]  = 0.04;              // align_factor
  f[10] = 0.003;             // cohesion_factor
  f[11] = 0.4;               // separation_factor
  f[12] = 6.0;               // max_speed
  f[13] = 1.5;               // min_speed
  f[14] = 0.016;             // dt
  f[15] = 0.35;              // turn_factor

  const paramsBuffer = device.createBuffer({
    size: 64,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });
  device.queue.writeBuffer(paramsBuffer, 0, paramsData);

  // --- Boid buffers (ping-pong) ---
  const boidBytes = numBoids * 32; // Boid struct = 32 bytes (vec3f + pad + vec3f + pad)
  const initData = new Float32Array(numBoids * 8);
  for (let i = 0; i < numBoids; i++) {
    const o = i * 8;
    const r = worldSize * 0.35;
    initData[o]     = (Math.random() - 0.5) * 2 * r;
    initData[o + 1] = (Math.random() - 0.5) * 2 * r;
    initData[o + 2] = (Math.random() - 0.5) * 2 * r;
    // [o+3] padding
    initData[o + 4] = (Math.random() - 0.5) * 4;
    initData[o + 5] = (Math.random() - 0.5) * 4;
    initData[o + 6] = (Math.random() - 0.5) * 4;
    // [o+7] padding
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

  // --- Bind group layout (shared by all 5 compute pipelines) ---
  const bgl = device.createBindGroupLayout({
    entries: Array.from({ length: 8 }, (_, i) => ({
      binding: i,
      visibility: GPUShaderStage.COMPUTE,
      buffer: { type: i === 0 ? 'uniform' : 'storage' },
    })),
  });
  const pipeLayout = device.createPipelineLayout({ bindGroupLayouts: [bgl] });

  // --- Two bind groups for ping-pong ---
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

  let step = 0;

  return {
    numBoids,
    boidA,
    boidB,

    /** Encode all 5 compute passes into the command encoder. */
    update(encoder) {
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
      step++;
    },

    /** Return the buffer that holds the latest computed boid state. */
    currentBuffer() {
      // After step N completes, dst of that step has the result.
      // step 0 (even): src=A dst=B → result in B, step is now 1 → odd → return B ✓
      return step % 2 === 1 ? boidB : boidA;
    },
  };
}
