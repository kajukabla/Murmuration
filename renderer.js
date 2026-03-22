// === WebGPU Renderer: Tetra + Billboard, Frustum Culling, Indirect Draw ===

const SAMPLE_COUNT = 4;
const HDR_FORMAT = 'rgba16float';
const UNIFORM_SIZE = 112;

export async function createRenderer(device, context, simulation) {
  context.configure({
    device,
    format: HDR_FORMAT,
    alphaMode: 'opaque',
    toneMapping: { mode: 'extended' },
  });

  const renderCode = await (await fetch('render.wgsl')).text();
  const renderModule = device.createShaderModule({ code: renderCode });

  const shaderCode = await (await fetch('shader.wgsl')).text();
  const computeModule = device.createShaderModule({ code: shaderCode });

  // GPU error logging
  device.addEventListener('uncapturederror', e => {
    console.error('GPU ERROR:', e.error.message);
    fetch('/api/bench_result', { method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ gpu_error: e.error.message }),
    }).catch(() => {});
  });

  // --- Camera uniform ---
  const uniformBuf = device.createBuffer({
    size: UNIFORM_SIZE,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });

  // --- Visible indices buffer (for frustum culling) ---
  let maxBoids = simulation.numBoids;

  function createCullBuffers(count) {
    return {
      visibleIndices: device.createBuffer({
        label: 'visible-indices',
        size: count * 4,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
      }),
      indirectBuf: device.createBuffer({
        label: 'indirect-args',
        size: 16, // 4 x u32: vertexCount, instanceCount, firstVertex, firstInstance
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.INDIRECT,
      }),
      visibleCountBuf: device.createBuffer({
        label: 'visible-count',
        size: 4,
        usage: GPUBufferUsage.STORAGE,
      }),
      cullUniformBuf: device.createBuffer({
        label: 'cull-uniforms',
        size: 160, // mat4x4f(64) + 6 * vec4f(96) = 160
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
      }),
    };
  }
  let cullBufs = createCullBuffers(maxBoids);

  // Initialize visible_indices to identity (no culling fallback)
  const identityIndices = new Uint32Array(maxBoids);
  for (let i = 0; i < maxBoids; i++) identityIndices[i] = i;
  device.queue.writeBuffer(cullBufs.visibleIndices, 0, identityIndices);

  // --- Render bind group layout (uniform + boids + visible_indices) ---
  const renderBGL = device.createBindGroupLayout({
    entries: [
      { binding: 0, visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT, buffer: { type: 'uniform' } },
      { binding: 1, visibility: GPUShaderStage.VERTEX, buffer: { type: 'read-only-storage' } },
      { binding: 2, visibility: GPUShaderStage.VERTEX, buffer: { type: 'read-only-storage' } },
    ],
  });

  function makeRenderBGs(sim, visBuf) {
    const make = (boidBuf) => device.createBindGroup({
      layout: renderBGL,
      entries: [
        { binding: 0, resource: { buffer: uniformBuf } },
        { binding: 1, resource: { buffer: boidBuf } },
        { binding: 2, resource: { buffer: visBuf } },
      ],
    });
    return { bgA: make(sim.boidA), bgB: make(sim.boidB) };
  }
  let { bgA, bgB } = makeRenderBGs(simulation, cullBufs.visibleIndices);

  const renderPipeLayout = device.createPipelineLayout({ bindGroupLayouts: [renderBGL] });

  // --- Tetrahedron pipeline (MSAA, depth, opaque) ---
  const tetraPipeline = device.createRenderPipeline({
    layout: renderPipeLayout,
    vertex: { module: renderModule, entryPoint: 'vs_main' },
    fragment: { module: renderModule, entryPoint: 'fs_main', targets: [{ format: HDR_FORMAT }] },
    primitive: { topology: 'triangle-list' },
    depthStencil: { format: 'depth24plus', depthWriteEnabled: true, depthCompare: 'less' },
    multisample: { count: SAMPLE_COUNT },
  });

  // --- Billboard pipeline (no MSAA, no depth, additive) ---
  let billboardPipeline = null;
  if (renderCode.includes('fn vs_billboard')) {
    billboardPipeline = device.createRenderPipeline({
      layout: renderPipeLayout,
      vertex: { module: renderModule, entryPoint: 'vs_billboard' },
      fragment: {
        module: renderModule, entryPoint: 'fs_billboard',
        targets: [{
          format: HDR_FORMAT,
          blend: {
            color: { srcFactor: 'one', dstFactor: 'one', operation: 'add' },
            alpha: { srcFactor: 'one', dstFactor: 'one', operation: 'add' },
          },
        }],
      },
      primitive: { topology: 'triangle-list' },
    });
  }

  // --- Frustum cull compute pipelines ---
  // Uses group(0) for sim params+boids, group(2) for cull buffers
  // We need the sim bind group layout from simulation.js — reuse it via the compute module
  const simBGL = device.createBindGroupLayout({
    entries: Array.from({ length: 8 }, (_, i) => ({
      binding: i,
      visibility: GPUShaderStage.COMPUTE,
      buffer: { type: i === 0 ? 'uniform' : 'storage' },
    })),
  });
  const cullBGL = device.createBindGroupLayout({
    entries: [
      { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
      { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
      { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
      { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
    ],
  });
  // Stats BGL (group 1) - same as simulation.js uses
  const statsBGL = device.createBindGroupLayout({
    entries: [
      { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
    ],
  });

  let hasCullShader = shaderCode.includes('fn frustum_cull');
  let clearCullPipe = null, cullPipe = null, cullBG = null;

  if (hasCullShader) {
    const cullPipeLayout = device.createPipelineLayout({
      bindGroupLayouts: [simBGL, statsBGL, cullBGL],
    });
    clearCullPipe = device.createComputePipeline({
      layout: cullPipeLayout,
      compute: { module: computeModule, entryPoint: 'clear_cull' },
    });
    cullPipe = device.createComputePipeline({
      layout: cullPipeLayout,
      compute: { module: computeModule, entryPoint: 'frustum_cull' },
    });
    cullBG = device.createBindGroup({
      layout: cullBGL,
      entries: [
        { binding: 0, resource: { buffer: cullBufs.cullUniformBuf } },
        { binding: 1, resource: { buffer: cullBufs.visibleCountBuf } },
        { binding: 2, resource: { buffer: cullBufs.visibleIndices } },
        { binding: 3, resource: { buffer: cullBufs.indirectBuf } },
      ],
    });
  }

  let { msaaTex, depthTex } = createMSAATargets(device, context.canvas);
  const uniformData = new ArrayBuffer(UNIFORM_SIZE);

  // Frustum plane extraction from VP matrix
  function extractFrustumPlanes(vp) {
    // Row-major extraction from column-major Float32Array
    const planes = new Float32Array(24); // 6 planes * 4 floats
    for (let i = 0; i < 6; i++) {
      const sign = (i % 2 === 0) ? 1 : -1;
      const row = Math.floor(i / 2);
      for (let j = 0; j < 4; j++) {
        // vp is column-major: element [row][col] = vp[col*4 + row]
        planes[i * 4 + j] = vp[j * 4 + 3] + sign * vp[j * 4 + row];
      }
      // Normalize plane
      const nx = planes[i*4], ny = planes[i*4+1], nz = planes[i*4+2];
      const len = Math.hypot(nx, ny, nz) || 1;
      planes[i*4] /= len; planes[i*4+1] /= len; planes[i*4+2] /= len; planes[i*4+3] /= len;
    }
    return planes;
  }

  return {
    rebindSimulation(sim) {
      maxBoids = sim.numBoids;
      cullBufs = createCullBuffers(maxBoids);
      const identity = new Uint32Array(maxBoids);
      for (let i = 0; i < maxBoids; i++) identity[i] = i;
      device.queue.writeBuffer(cullBufs.visibleIndices, 0, identity);
      ({ bgA, bgB } = makeRenderBGs(sim, cullBufs.visibleIndices));
      if (hasCullShader) {
        cullBG = device.createBindGroup({
          layout: cullBGL,
          entries: [
            { binding: 0, resource: { buffer: cullBufs.cullUniformBuf } },
            { binding: 1, resource: { buffer: cullBufs.visibleCountBuf } },
            { binding: 2, resource: { buffer: cullBufs.visibleIndices } },
            { binding: 3, resource: { buffer: cullBufs.indirectBuf } },
          ],
        });
      }
    },

    render(encoder, viewProj, opts = {}) {
      const {
        gradientId = 0, colorSource = 0, gain = 0.5,
        autoRange = false, autoMin = 0, autoMax = 1,
        falloff = 1.0, brightness = 1.0, sphereRadius = 100,
        renderMode = 0,
        numBoids = simulation.numBoids, sim = simulation,
        frustumCull = true,
      } = opts;

      // Pack render uniforms
      new Float32Array(uniformData, 0, 16).set(viewProj);
      new Uint32Array(uniformData, 64, 2).set([gradientId, colorSource]);
      new Float32Array(uniformData, 72, 1).set([gain]);
      new Uint32Array(uniformData, 76, 1).set([autoRange ? 1 : 0]);
      new Float32Array(uniformData, 80, 2).set([autoMin, autoMax]);
      new Float32Array(uniformData, 88, 1).set([falloff]);
      new Float32Array(uniformData, 92, 1).set([brightness]);
      new Float32Array(uniformData, 96, 1).set([sphereRadius]);
      device.queue.writeBuffer(uniformBuf, 0, new Uint8Array(uniformData));

      const curBuf = sim.currentBuffer();
      const bg = curBuf === sim.boidA ? bgA : bgB;
      const canvasTex = context.getCurrentTexture().createView();

      // Frustum culling (if available and enabled)
      const useCull = frustumCull && hasCullShader && cullPipe && sim._simBG;
      if (useCull) {
        // Write cull uniforms (VP matrix + frustum planes)
        const cullData = new ArrayBuffer(160);
        new Float32Array(cullData, 0, 16).set(viewProj);
        new Float32Array(cullData, 64, 24).set(extractFrustumPlanes(viewProj));
        device.queue.writeBuffer(cullBufs.cullUniformBuf, 0, new Uint8Array(cullData));

        // Set indirect args vertex count
        const vertCount = renderMode === 0 ? 12 : 6;
        const argsInit = new Uint32Array([vertCount, 0, 0, 0]);
        device.queue.writeBuffer(cullBufs.indirectBuf, 0, argsInit);

        const simBG = sim._simBG(); // get current sim bind group
        const wg = Math.ceil(numBoids / 64);

        // Clear
        const c = encoder.beginComputePass();
        c.setPipeline(clearCullPipe);
        c.setBindGroup(0, simBG);
        c.setBindGroup(1, sim._statsBG);
        c.setBindGroup(2, cullBG);
        c.dispatchWorkgroups(1);
        c.end();

        // Cull
        const s = encoder.beginComputePass();
        s.setPipeline(cullPipe);
        s.setBindGroup(0, simBG);
        s.setBindGroup(1, sim._statsBG);
        s.setBindGroup(2, cullBG);
        s.dispatchWorkgroups(wg);
        s.end();
      } else {
        // No culling: write identity indices + full instance count
        // (already initialized at creation, just set indirect args)
      }

      if (renderMode === 0) {
        // Tetrahedron mode (MSAA)
        const pass = encoder.beginRenderPass({
          colorAttachments: [{
            view: msaaTex.createView(),
            resolveTarget: canvasTex,
            clearValue: { r: 0.003, g: 0.003, b: 0.01, a: 1 },
            loadOp: 'clear', storeOp: 'discard',
          }],
          depthStencilAttachment: {
            view: depthTex.createView(),
            depthClearValue: 1.0, depthLoadOp: 'clear', depthStoreOp: 'discard',
          },
        });
        pass.setPipeline(tetraPipeline);
        pass.setBindGroup(0, bg);
        if (useCull) {
          pass.drawIndirect(cullBufs.indirectBuf, 0);
        } else {
          pass.draw(12, numBoids);
        }
        pass.end();
      } else if (billboardPipeline) {
        // Billboard mode (additive, no MSAA)
        const pass = encoder.beginRenderPass({
          colorAttachments: [{
            view: canvasTex,
            clearValue: { r: 0.0, g: 0.0, b: 0.0, a: 1 },
            loadOp: 'clear', storeOp: 'store',
          }],
        });
        pass.setPipeline(billboardPipeline);
        pass.setBindGroup(0, bg);
        if (useCull) {
          pass.drawIndirect(cullBufs.indirectBuf, 0);
        } else {
          pass.draw(6, numBoids);
        }
        pass.end();
      }
    },

    resize() {
      msaaTex.destroy();
      depthTex.destroy();
      ({ msaaTex, depthTex } = createMSAATargets(device, context.canvas));
    },
  };
}

function createMSAATargets(device, canvas) {
  const size = [canvas.width, canvas.height];
  return {
    msaaTex: device.createTexture({
      size, format: HDR_FORMAT,
      usage: GPUTextureUsage.RENDER_ATTACHMENT, sampleCount: SAMPLE_COUNT,
    }),
    depthTex: device.createTexture({
      size, format: 'depth24plus',
      usage: GPUTextureUsage.RENDER_ATTACHMENT, sampleCount: SAMPLE_COUNT,
    }),
  };
}
