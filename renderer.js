// === WebGPU Renderer: Tetrahedron + Billboard modes, Bloom post-process ===

const SAMPLE_COUNT = 4;
const HDR_FORMAT = 'rgba16float';
const UNIFORM_SIZE = 96;
const BLOOM_MIP_LEVELS = 5;

export async function createRenderer(device, context, simulation) {
  context.configure({
    device,
    format: HDR_FORMAT,
    alphaMode: 'opaque',
    toneMapping: { mode: 'extended' },
  });

  const renderCode = await (await fetch('render.wgsl')).text();
  const renderModule = device.createShaderModule({ code: renderCode });

  const bloomCode = await (await fetch('bloom.wgsl')).text();
  const bloomModule = device.createShaderModule({ code: bloomCode });

  // --- Camera uniform ---
  const uniformBuf = device.createBuffer({
    size: UNIFORM_SIZE,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });

  // --- Bind group layout (shared by tetra + billboard pipelines) ---
  const bgl = device.createBindGroupLayout({
    entries: [
      { binding: 0, visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT, buffer: { type: 'uniform' } },
      { binding: 1, visibility: GPUShaderStage.VERTEX, buffer: { type: 'read-only-storage' } },
    ],
  });

  // --- Bind groups for ping-pong ---
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

  // --- Tetrahedron pipeline (MSAA, depth, opaque) ---
  const tetraPipeline = device.createRenderPipeline({
    layout: pipelineLayout,
    vertex: { module: renderModule, entryPoint: 'vs_main' },
    fragment: { module: renderModule, entryPoint: 'fs_main', targets: [{ format: HDR_FORMAT }] },
    primitive: { topology: 'triangle-list' },
    depthStencil: { format: 'depth24plus', depthWriteEnabled: true, depthCompare: 'less' },
    multisample: { count: SAMPLE_COUNT },
  });

  // --- Billboard pipeline (no MSAA, no depth, additive blend) ---
  const billboardPipeline = device.createRenderPipeline({
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
    // No depth stencil for additive
  });

  // --- Bloom resources ---
  const bloomSampler = device.createSampler({
    minFilter: 'linear', magFilter: 'linear',
    addressModeU: 'clamp-to-edge', addressModeV: 'clamp-to-edge',
  });

  const bloomParamsBuf = device.createBuffer({
    size: 16, // threshold(4) + intensity(4) + texel_size(8)
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });

  const bloomBGL = device.createBindGroupLayout({
    entries: [
      { binding: 0, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: 'float' } },
      { binding: 1, visibility: GPUShaderStage.FRAGMENT, sampler: {} },
      { binding: 2, visibility: GPUShaderStage.FRAGMENT, buffer: { type: 'uniform' } },
    ],
  });
  const bloomPipeLayout = device.createPipelineLayout({ bindGroupLayouts: [bloomBGL] });

  function makeBloomPipeline(entryPoint, blend = false) {
    return device.createRenderPipeline({
      layout: bloomPipeLayout,
      vertex: { module: bloomModule, entryPoint: 'vs_fullscreen' },
      fragment: {
        module: bloomModule, entryPoint,
        targets: [{
          format: HDR_FORMAT,
          ...(blend ? {
            blend: {
              color: { srcFactor: 'one', dstFactor: 'one', operation: 'add' },
              alpha: { srcFactor: 'one', dstFactor: 'one', operation: 'add' },
            },
          } : {}),
        }],
      },
    });
  }

  const thresholdPipe = makeBloomPipeline('fs_threshold');
  const downsamplePipe = makeBloomPipeline('fs_downsample');
  const upsamplePipe = makeBloomPipeline('fs_upsample');
  const compositePipe = makeBloomPipeline('fs_composite', true); // additive onto scene

  // --- Render targets ---
  let targets = null;

  function createAllTargets(canvas) {
    const w = canvas.width, h = canvas.height;
    // MSAA + depth for tetrahedron mode
    const msaaTex = device.createTexture({
      size: [w, h], format: HDR_FORMAT,
      usage: GPUTextureUsage.RENDER_ATTACHMENT, sampleCount: SAMPLE_COUNT,
    });
    const depthTex = device.createTexture({
      size: [w, h], format: 'depth24plus',
      usage: GPUTextureUsage.RENDER_ATTACHMENT, sampleCount: SAMPLE_COUNT,
    });
    // Scene HDR texture (non-MSAA, for billboard mode and bloom input)
    const sceneTex = device.createTexture({
      size: [w, h], format: HDR_FORMAT,
      usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING,
    });
    // Bloom mip chain
    const bloomMips = [];
    let mw = Math.max(1, w >> 1), mh = Math.max(1, h >> 1);
    for (let i = 0; i < BLOOM_MIP_LEVELS; i++) {
      bloomMips.push(device.createTexture({
        size: [Math.max(1, mw), Math.max(1, mh)], format: HDR_FORMAT,
        usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING,
      }));
      mw = Math.max(1, mw >> 1);
      mh = Math.max(1, mh >> 1);
    }
    return { msaaTex, depthTex, sceneTex, bloomMips, w, h };
  }

  targets = createAllTargets(context.canvas);

  // Uniform data buffer
  const uniformData = new ArrayBuffer(UNIFORM_SIZE);

  // --- Public API ---
  return {
    /** Recreate bind groups when simulation buffers change (e.g., particle count change) */
    rebindSimulation(sim) {
      ({ bgA, bgB } = makeBoidBGs(sim));
    },

    render(encoder, viewProj, opts = {}) {
      const {
        gradientId = 0, colorSource = 0, gain = 0.5,
        autoRange = false, autoMin = 0, autoMax = 1,
        falloff = 1.0, renderMode = 0, // 0=tetra, 1=billboard
        bloomEnabled = false, bloomThreshold = 0.8, bloomIntensity = 0.5,
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
        // === Tetrahedron mode (MSAA) ===
        const renderTarget = bloomEnabled ? targets.sceneTex.createView() : canvasTex;
        const pass = encoder.beginRenderPass({
          colorAttachments: [{
            view: targets.msaaTex.createView(),
            resolveTarget: renderTarget,
            clearValue: { r: 0.003, g: 0.003, b: 0.01, a: 1 },
            loadOp: 'clear', storeOp: 'discard',
          }],
          depthStencilAttachment: {
            view: targets.depthTex.createView(),
            depthClearValue: 1.0, depthLoadOp: 'clear', depthStoreOp: 'discard',
          },
        });
        pass.setPipeline(tetraPipeline);
        pass.setBindGroup(0, bg);
        pass.draw(12, numBoids);
        pass.end();
      } else {
        // === Billboard mode (no MSAA, additive) ===
        const renderTarget = bloomEnabled ? targets.sceneTex.createView() : canvasTex;
        const pass = encoder.beginRenderPass({
          colorAttachments: [{
            view: renderTarget,
            clearValue: { r: 0.0, g: 0.0, b: 0.0, a: 1 },
            loadOp: 'clear', storeOp: 'store',
          }],
        });
        pass.setPipeline(billboardPipeline);
        pass.setBindGroup(0, bg);
        pass.draw(6, numBoids);
        pass.end();
      }

      // === Bloom post-process ===
      if (bloomEnabled) {
        const { bloomMips, w, h } = targets;

        // Helper: create bind group for a texture
        const makeBloomBG = (tex, texelW, texelH) => {
          const paramsData = new Float32Array([bloomThreshold, bloomIntensity, 1/texelW, 1/texelH]);
          device.queue.writeBuffer(bloomParamsBuf, 0, paramsData);
          return device.createBindGroup({
            layout: bloomBGL,
            entries: [
              { binding: 0, resource: tex.createView ? tex.createView() : tex },
              { binding: 1, resource: bloomSampler },
              { binding: 2, resource: { buffer: bloomParamsBuf } },
            ],
          });
        };

        // 1. Threshold: scene → bloomMips[0]
        const mip0 = bloomMips[0];
        {
          const bg0 = makeBloomBG(targets.sceneTex, w, h);
          const pass = encoder.beginRenderPass({
            colorAttachments: [{ view: mip0.createView(), loadOp: 'clear', clearValue: {r:0,g:0,b:0,a:0}, storeOp: 'store' }],
          });
          pass.setPipeline(thresholdPipe);
          pass.setBindGroup(0, bg0);
          pass.draw(3);
          pass.end();
        }

        // 2. Downsample chain
        let mw = Math.max(1, w >> 1), mh = Math.max(1, h >> 1);
        for (let i = 1; i < BLOOM_MIP_LEVELS; i++) {
          const srcTex = bloomMips[i - 1];
          const dstTex = bloomMips[i];
          const bgD = makeBloomBG(srcTex, mw, mh);
          mw = Math.max(1, mw >> 1); mh = Math.max(1, mh >> 1);
          const pass = encoder.beginRenderPass({
            colorAttachments: [{ view: dstTex.createView(), loadOp: 'clear', clearValue: {r:0,g:0,b:0,a:0}, storeOp: 'store' }],
          });
          pass.setPipeline(downsamplePipe);
          pass.setBindGroup(0, bgD);
          pass.draw(3);
          pass.end();
        }

        // 3. Upsample chain (blend back up)
        mw = Math.max(1, w >> BLOOM_MIP_LEVELS);
        mh = Math.max(1, h >> BLOOM_MIP_LEVELS);
        for (let i = BLOOM_MIP_LEVELS - 1; i > 0; i--) {
          const srcTex = bloomMips[i];
          const dstTex = bloomMips[i - 1];
          mw = Math.max(1, mw << 1); mh = Math.max(1, mh << 1);
          const paramsData = new Float32Array([bloomThreshold, bloomIntensity, 1/mw, 1/mh]);
          device.queue.writeBuffer(bloomParamsBuf, 0, paramsData);
          const bgU = device.createBindGroup({
            layout: bloomBGL,
            entries: [
              { binding: 0, resource: srcTex.createView() },
              { binding: 1, resource: bloomSampler },
              { binding: 2, resource: { buffer: bloomParamsBuf } },
            ],
          });
          const pass = encoder.beginRenderPass({
            colorAttachments: [{ view: dstTex.createView(), loadOp: 'load', storeOp: 'store' }],
          });
          pass.setPipeline(upsamplePipe);
          pass.setBindGroup(0, bgU);
          pass.draw(3);
          pass.end();
        }

        // 4. Composite: copy scene to canvas, then additively blend bloom on top
        // First: copy scene to canvas
        {
          const sceneBG = makeBloomBG(targets.sceneTex, w, h);
          // Use a simple fullscreen blit (composite with intensity=1 as passthrough)
          const paramsData = new Float32Array([bloomThreshold, 1.0, 1/w, 1/h]);
          device.queue.writeBuffer(bloomParamsBuf, 0, paramsData);
          const pass = encoder.beginRenderPass({
            colorAttachments: [{ view: canvasTex, loadOp: 'clear', clearValue: {r:0,g:0,b:0,a:1}, storeOp: 'store' }],
          });
          pass.setPipeline(compositePipe); // additive, but on cleared canvas = just copy
          pass.setBindGroup(0, sceneBG);
          pass.draw(3);
          pass.end();
        }
        // Then: add bloom
        {
          const paramsData = new Float32Array([bloomThreshold, bloomIntensity, 1/(w>>1), 1/(h>>1)]);
          device.queue.writeBuffer(bloomParamsBuf, 0, paramsData);
          const bloomBG2 = device.createBindGroup({
            layout: bloomBGL,
            entries: [
              { binding: 0, resource: bloomMips[0].createView() },
              { binding: 1, resource: bloomSampler },
              { binding: 2, resource: { buffer: bloomParamsBuf } },
            ],
          });
          const pass = encoder.beginRenderPass({
            colorAttachments: [{ view: canvasTex, loadOp: 'load', storeOp: 'store' }],
          });
          pass.setPipeline(compositePipe);
          pass.setBindGroup(0, bloomBG2);
          pass.draw(3);
          pass.end();
        }
      }
    },

    resize() {
      if (targets) {
        targets.msaaTex.destroy();
        targets.depthTex.destroy();
        targets.sceneTex.destroy();
        targets.bloomMips.forEach(t => t.destroy());
      }
      targets = createAllTargets(context.canvas);
    },
  };
}
