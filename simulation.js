// === WebGPU Compute Simulation (Spatial-Hash Flocking) ===

const WORKGROUP_SIZE = 64;
const PARAMS_SIZE = 80; // 17 fields × 4 bytes = 68, rounded to 80 (16-byte align)

export async function createSimulation(device, {
  numBoids = 5000,
  worldSize = 100.0,
  gridSize = 64,
} = {}) {
  const GRID_CELLS = gridSize ** 3;
  const cellSize = worldSize / gridSize;

  // --- Shader module ---
  const code = await (await fetch('shader.wgsl')).text();
  const module = device.createShaderModule({ code });

  // --- Params uniform ---
  const paramsData = new ArrayBuffer(PARAMS_SIZE);
  const u = new Uint32Array(paramsData);
  const f = new Float32Array(paramsData);

  // Fixed fields (never change at runtime)
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
    device.queue.writeBuffer(paramsBuffer, 0, paramsData);
  }

  const paramsBuffer = device.createBuffer({
    size: PARAMS_SIZE,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });

  // --- Boid buffers (ping-pong) ---
  // Boid struct: pos(vec3f+pad) + vel(vec3f+pad) + heading(vec3f+pad) = 48 bytes = 12 floats
  const boidBytes = numBoids * 48;
  const initData = new Float32Array(numBoids * 12);
  for (let i = 0; i < numBoids; i++) {
    const o = i * 12;
    const r = worldSize * 0.35;
    // pos
    initData[o]     = (Math.random() - 0.5) * 2 * r;
    initData[o + 1] = (Math.random() - 0.5) * 2 * r;
    initData[o + 2] = (Math.random() - 0.5) * 2 * r;
    // [o+3] padding
    // vel
    const vx = (Math.random() - 0.5) * 4;
    const vy = (Math.random() - 0.5) * 4;
    const vz = (Math.random() - 0.5) * 4;
    initData[o + 4] = vx;
    initData[o + 5] = vy;
    initData[o + 6] = vz;
    // [o+7] padding
    // heading = normalize(vel)
    const vlen = Math.hypot(vx, vy, vz) || 1;
    initData[o + 8]  = vx / vlen;
    initData[o + 9]  = vy / vlen;
    initData[o + 10] = vz / vlen;
    // [o+11] padding
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

  let step = 0;

  return {
    numBoids,
    boidA,
    boidB,

    /** Update tunable params at runtime. */
    setParams: writeParams,

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

    currentBuffer() {
      return step % 2 === 1 ? boidB : boidA;
    },
  };
}
