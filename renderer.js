// === WebGPU Renderer: Tetrahedron + Billboard modes ===

const SAMPLE_COUNT = 4;
const HDR_FORMAT = 'rgba16float';
const UNIFORM_SIZE = 96;

export async function createRenderer(device, context, simulation) {
  context.configure({
    device,
    format: HDR_FORMAT,
    alphaMode: 'opaque',
    toneMapping: { mode: 'extended' },
  });

  const renderCode = await (await fetch('render.wgsl')).text();
  const renderModule = device.createShaderModule({ code: renderCode });

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

  // Tetrahedron pipeline (MSAA, depth, opaque)
  const tetraPipeline = device.createRenderPipeline({
    layout: pipelineLayout,
    vertex: { module: renderModule, entryPoint: 'vs_main' },
    fragment: { module: renderModule, entryPoint: 'fs_main', targets: [{ format: HDR_FORMAT }] },
    primitive: { topology: 'triangle-list' },
    depthStencil: { format: 'depth24plus', depthWriteEnabled: true, depthCompare: 'less' },
    multisample: { count: SAMPLE_COUNT },
  });

  // Billboard pipeline (no MSAA, no depth, additive blend)
  // Created lazily — only if the shader has the entry points
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

  // Log GPU errors
  device.addEventListener('uncapturederror', e => {
    console.error('GPU ERROR:', e.error.message);
    fetch('/api/bench_result', { method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ gpu_error: e.error.message }),
    }).catch(() => {});
  });

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
        falloff = 1.0, renderMode = 0,
        numBoids = simulation.numBoids, sim = simulation,
      } = opts;

      // Pack uniforms
      new Float32Array(uniformData, 0, 16).set(viewProj);
      new Uint32Array(uniformData, 64, 2).set([gradientId, colorSource]);
      new Float32Array(uniformData, 72, 1).set([gain]);
      new Uint32Array(uniformData, 76, 1).set([autoRange ? 1 : 0]);
      new Float32Array(uniformData, 80, 2).set([autoMin, autoMax]);
      new Float32Array(uniformData, 88, 1).set([falloff]);
      device.queue.writeBuffer(uniformBuf, 0, new Uint8Array(uniformData));

      const curBuf = sim.currentBuffer();
      const bg = curBuf === sim.boidA ? bgA : bgB;
      const canvasTex = context.getCurrentTexture().createView();

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
        pass.draw(12, numBoids);
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
