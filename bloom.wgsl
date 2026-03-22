// bloom.wgsl — Multi-pass dual Kawase bloom post-process shader
//
// Entry points:
//   vs_fullscreen   — shared fullscreen triangle vertex shader
//   fs_threshold    — bright-pixel extraction with soft knee
//   fs_downsample   — Kawase 4-tap diamond downsample
//   fs_upsample     — 9-tap tent filter upsample
//   fs_composite    — apply bloom intensity (additive blend in pipeline)

// ---------------------------------------------------------------------------
// Bind group layout
// ---------------------------------------------------------------------------

@group(0) @binding(0) var inputTex : texture_2d<f32>;
@group(0) @binding(1) var samp     : sampler;

struct BloomParams {
    threshold  : f32,
    intensity  : f32,
    texel_size : vec2f,
};

@group(0) @binding(2) var<uniform> params : BloomParams;

// ---------------------------------------------------------------------------
// Vertex output
// ---------------------------------------------------------------------------

struct VsOut {
    @builtin(position) position : vec4f,
    @location(0)       uv      : vec2f,
};

// ---------------------------------------------------------------------------
// Shared fullscreen triangle vertex shader
// Three vertices cover the entire screen without an index buffer.
//   vid 0 -> (-1, -1)   uv (0, 1)
//   vid 1 -> ( 3, -1)   uv (2, 1)
//   vid 2 -> (-1,  3)   uv (0,-1)
// The rasteriser clips to the viewport, producing a full-screen quad.
// UV y is flipped so that (0,0) is top-left matching WebGPU texture coords.
// ---------------------------------------------------------------------------

@vertex
fn vs_fullscreen(@builtin(vertex_index) vid : u32) -> VsOut {
    var out : VsOut;
    let x = f32(i32(vid & 1u) * 4 - 1);
    let y = f32(i32(vid >> 1u) * 4 - 1);
    out.position = vec4f(x, y, 0.0, 1.0);
    out.uv = vec2f((x + 1.0) * 0.5, 1.0 - (y + 1.0) * 0.5);
    return out;
}

// ---------------------------------------------------------------------------
// 1. Threshold — extract bright pixels with a soft knee
// ---------------------------------------------------------------------------

@fragment
fn fs_threshold(in : VsOut) -> @location(0) vec4f {
    let color = textureSample(inputTex, samp, in.uv);
    let lum   = dot(color.rgb, vec3f(0.2126, 0.7152, 0.0722));
    let soft  = clamp((lum - params.threshold) / 0.5, 0.0, 1.0);
    return vec4f(color.rgb * soft, 1.0);
}

// ---------------------------------------------------------------------------
// 2. Downsample — Kawase 4-tap diamond blur
// ---------------------------------------------------------------------------

@fragment
fn fs_downsample(in : VsOut) -> @location(0) vec4f {
    let d = params.texel_size;
    let c = textureSample(inputTex, samp, in.uv);
    let a = textureSample(inputTex, samp, in.uv + vec2f(-d.x, -d.y));
    let b = textureSample(inputTex, samp, in.uv + vec2f( d.x, -d.y));
    let e = textureSample(inputTex, samp, in.uv + vec2f(-d.x,  d.y));
    let f = textureSample(inputTex, samp, in.uv + vec2f( d.x,  d.y));
    return (c * 4.0 + a + b + e + f) / 8.0;
}

// ---------------------------------------------------------------------------
// 3. Upsample — 9-tap tent filter
// ---------------------------------------------------------------------------

@fragment
fn fs_upsample(in : VsOut) -> @location(0) vec4f {
    let d = params.texel_size;
    var result = textureSample(inputTex, samp, in.uv + vec2f(-d.x, -d.y)) * 1.0
               + textureSample(inputTex, samp, in.uv + vec2f( 0.0, -d.y)) * 2.0
               + textureSample(inputTex, samp, in.uv + vec2f( d.x, -d.y)) * 1.0
               + textureSample(inputTex, samp, in.uv + vec2f(-d.x,  0.0)) * 2.0
               + textureSample(inputTex, samp, in.uv)                      * 4.0
               + textureSample(inputTex, samp, in.uv + vec2f( d.x,  0.0)) * 2.0
               + textureSample(inputTex, samp, in.uv + vec2f(-d.x,  d.y)) * 1.0
               + textureSample(inputTex, samp, in.uv + vec2f( 0.0,  d.y)) * 2.0
               + textureSample(inputTex, samp, in.uv + vec2f( d.x,  d.y)) * 1.0;
    return result / 16.0;
}

// ---------------------------------------------------------------------------
// 4. Composite — scale bloom by intensity (additive blend set in pipeline)
// ---------------------------------------------------------------------------

@fragment
fn fs_composite(in : VsOut) -> @location(0) vec4f {
    let bloom = textureSample(inputTex, samp, in.uv);
    return vec4f(bloom.rgb * params.intensity, 1.0);
}
