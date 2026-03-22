// === WebGPU Instanced Boid Renderer (4× MSAA, HDR) ===

const SAMPLE_COUNT = 4;
const HDR_FORMAT = 'rgba16float';
// Camera uniform: mat4x4f(64) + gradient_id(4) + color_source(4) + gain(4) + auto_range(4)
//                 + auto_min(4) + auto_max(4) + pad(8) = 96 bytes
const UNIFORM_SIZE = 96;

export async function createRenderer(device, context, simulation) {
  context.configure({
    device,
    format: HDR_FORMAT,
    alphaMode: 'opaque',
    toneMapping: { mode: 'extended' },
  });

  const code = await (await fetch('render.wgsl')).text();
  const module = device.createShaderModule({ code });

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

  const bgA = device.createBindGroup({
    layout: bgl,
    entries: [
      { binding: 0, resource: { buffer: uniformBuf } },
      { binding: 1, resource: { buffer: simulation.boidA } },
    ],
  });
  const bgB = device.createBindGroup({
    layout: bgl,
    entries: [
      { binding: 0, resource: { buffer: uniformBuf } },
      { binding: 1, resource: { buffer: simulation.boidB } },
    ],
  });

  const pipeline = device.createRenderPipeline({
    layout: device.createPipelineLayout({ bindGroupLayouts: [bgl] }),
    vertex:   { module, entryPoint: 'vs_main' },
    fragment: { module, entryPoint: 'fs_main', targets: [{ format: HDR_FORMAT }] },
    primitive: { topology: 'triangle-list' },
    depthStencil: {
      format: 'depth24plus',
      depthWriteEnabled: true,
      depthCompare: 'less',
    },
    multisample: { count: SAMPLE_COUNT },
  });

  let { msaaTex, depthTex } = createTargets(device, context.canvas);

  // Reusable uniform data buffer
  const uniformData = new ArrayBuffer(UNIFORM_SIZE);

  return {
    render(encoder, viewProj, gradientId = 0, colorSource = 0, gain = 0.5, autoRange = false, autoMin = 0, autoMax = 1) {
      new Float32Array(uniformData, 0, 16).set(viewProj);
      new Uint32Array(uniformData, 64, 2).set([gradientId, colorSource]);
      new Float32Array(uniformData, 72, 1).set([gain]);
      new Uint32Array(uniformData, 76, 1).set([autoRange ? 1 : 0]);
      new Float32Array(uniformData, 80, 2).set([autoMin, autoMax]);
      device.queue.writeBuffer(uniformBuf, 0, new Uint8Array(uniformData));

      const curBuf = simulation.currentBuffer();
      const bg = curBuf === simulation.boidA ? bgA : bgB;

      const pass = encoder.beginRenderPass({
        colorAttachments: [{
          view: msaaTex.createView(),
          resolveTarget: context.getCurrentTexture().createView(),
          clearValue: { r: 0.003, g: 0.003, b: 0.01, a: 1 },
          loadOp: 'clear',
          storeOp: 'discard',
        }],
        depthStencilAttachment: {
          view: depthTex.createView(),
          depthClearValue: 1.0,
          depthLoadOp: 'clear',
          depthStoreOp: 'discard',
        },
      });
      pass.setPipeline(pipeline);
      pass.setBindGroup(0, bg);
      pass.draw(12, simulation.numBoids);
      pass.end();
    },

    resize() {
      msaaTex.destroy();
      depthTex.destroy();
      ({ msaaTex, depthTex } = createTargets(device, context.canvas));
    },
  };
}

function createTargets(device, canvas) {
  const size = [canvas.width, canvas.height];
  const msaaTex = device.createTexture({
    size,
    format: HDR_FORMAT,
    usage: GPUTextureUsage.RENDER_ATTACHMENT,
    sampleCount: SAMPLE_COUNT,
  });
  const depthTex = device.createTexture({
    size,
    format: 'depth24plus',
    usage: GPUTextureUsage.RENDER_ATTACHMENT,
    sampleCount: SAMPLE_COUNT,
  });
  return { msaaTex, depthTex };
}
