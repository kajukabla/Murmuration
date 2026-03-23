// === WebGPU Renderer: Tetrahedron + Billboard modes ===

const SAMPLE_COUNT = 4;
const HDR_FORMAT = 'rgba16float';
const UNIFORM_SIZE = 128; // + camera_pos(12) + pad(4)

export async function createRenderer(device, context, simulation) {
  context.configure({
    device,
    format: HDR_FORMAT,
    alphaMode: 'opaque',
    toneMapping: { mode: 'extended' },
  });

  const renderCode = await (await fetch('render.wgsl')).text();
  const renderModule = device.createShaderModule({ code: renderCode });

  const gpuErrors = [];
  device.addEventListener('uncapturederror', e => {
    console.error('GPU ERROR:', e.error.message);
    gpuErrors.push(e.error.message);
    // Only send the first few errors (the first one is the root cause)
    if (gpuErrors.length <= 3) {
      fetch('/api/bench_result', { method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ gpu_errors: gpuErrors }),
      }).catch(() => {});
    }
  });

  const uniformBuf = device.createBuffer({
    size: UNIFORM_SIZE,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });

  const bgl = device.createBindGroupLayout({
    entries: [
      { binding: 0, visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT, buffer: { type: 'uniform' } },
      { binding: 1, visibility: GPUShaderStage.VERTEX, buffer: { type: 'read-only-storage' } },
    ],
  });

  function makeBoidBGs(sim) {
    return {
      bgA: device.createBindGroup({
        layout: bgl,
        entries: [
          { binding: 0, resource: { buffer: uniformBuf } },
          { binding: 1, resource: { buffer: sim.boidA } },
        ],
      }),
      bgB: device.createBindGroup({
        layout: bgl,
        entries: [
          { binding: 0, resource: { buffer: uniformBuf } },
          { binding: 1, resource: { buffer: sim.boidB } },
        ],
      }),
    };
  }
  let { bgA, bgB } = makeBoidBGs(simulation);
  const pipelineLayout = device.createPipelineLayout({ bindGroupLayouts: [bgl] });

  const tetraPipeline = device.createRenderPipeline({
    layout: pipelineLayout,
    vertex: { module: renderModule, entryPoint: 'vs_main' },
    fragment: { module: renderModule, entryPoint: 'fs_main', targets: [{ format: HDR_FORMAT }] },
    primitive: { topology: 'triangle-list' },
    depthStencil: { format: 'depth24plus', depthWriteEnabled: true, depthCompare: 'less' },
    multisample: { count: SAMPLE_COUNT },
  });

  let billboardPipeline = null;
  if (renderCode.includes('fn vs_billboard')) {
    billboardPipeline = device.createRenderPipeline({
      layout: pipelineLayout,
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

  // Opaque billboard pipeline (with depth, no blend)
  let opaqueBlbPipeline = null;
  if (renderCode.includes('fn vs_billboard')) {
    opaqueBlbPipeline = device.createRenderPipeline({
      layout: pipelineLayout,
      vertex: { module: renderModule, entryPoint: 'vs_billboard' },
      fragment: {
        module: renderModule, entryPoint: 'fs_billboard',
        targets: [{ format: HDR_FORMAT }],
      },
      primitive: { topology: 'triangle-list' },
      depthStencil: { format: 'depth24plus', depthWriteEnabled: true, depthCompare: 'less' },
      multisample: { count: SAMPLE_COUNT },
    });
  }

  let { msaaTex, depthTex } = createMSAATargets(device, context.canvas);
  const uniformData = new ArrayBuffer(UNIFORM_SIZE);

  return {
    rebindSimulation(sim) {
      ({ bgA, bgB } = makeBoidBGs(sim));
    },

    render(encoder, viewProj, opts = {}) {
      const {
        gradientId = 0, colorSource = 0, gain = 0.5,
        autoRange = false, autoMin = 0, autoMax = 1,
        falloff = 1.0, brightness = 1.0, sphereRadius = 100,
        particleScale = 1.0, renderMode = 0, cameraPos = [0,0,0],
        numBoids = simulation.numBoids, sim = simulation,
      } = opts;

      new Float32Array(uniformData, 0, 16).set(viewProj);
      new Uint32Array(uniformData, 64, 2).set([gradientId, colorSource]);
      new Float32Array(uniformData, 72, 1).set([gain]);
      new Uint32Array(uniformData, 76, 1).set([autoRange ? 1 : 0]);
      new Float32Array(uniformData, 80, 2).set([autoMin, autoMax]);
      new Float32Array(uniformData, 88, 1).set([falloff]);
      new Float32Array(uniformData, 92, 1).set([brightness]);
      new Float32Array(uniformData, 96, 1).set([sphereRadius]);
      new Float32Array(uniformData, 100, 1).set([particleScale]);
      new Uint32Array(uniformData, 104, 1).set([renderMode]);
      new Float32Array(uniformData, 112, 3).set(cameraPos);
      device.queue.writeBuffer(uniformBuf, 0, new Uint8Array(uniformData));

      const curBuf = sim.currentBuffer();
      const bg = curBuf === sim.boidA ? bgA : bgB;
      const canvasTex = context.getCurrentTexture().createView();

      if (renderMode === 0 || renderMode === 3) {
        // Tetrahedron or Reflective Tetrahedron (MSAA + depth)
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
        pass.draw(12, numBoids);
        pass.end();
      } else if (renderMode === 1 && billboardPipeline) {
        // Billboard additive (no MSAA, no depth)
        const pass = encoder.beginRenderPass({
          colorAttachments: [{
            view: canvasTex,
            clearValue: { r: 0.0, g: 0.0, b: 0.0, a: 1 },
            loadOp: 'clear', storeOp: 'store',
          }],
        });
        pass.setPipeline(billboardPipeline);
        pass.setBindGroup(0, bg);
        pass.draw(6, numBoids);
        pass.end();
      } else if ((renderMode === 2 || renderMode === 4) && opaqueBlbPipeline) {
        // Billboard opaque or Reflective Billboard (MSAA + depth)
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
        pass.setPipeline(opaqueBlbPipeline);
        pass.setBindGroup(0, bg);
        pass.draw(6, numBoids);
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
