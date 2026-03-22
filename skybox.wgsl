// === Procedural Gradient Skybox ===

struct SkyUniforms {
  view_proj: mat4x4f,
  inv_view_proj: mat4x4f,
}

@group(0) @binding(0) var<uniform> u: SkyUniforms;

struct VsOut {
  @builtin(position) pos: vec4f,
  @location(0) clip_xy: vec2f,
}

// Fullscreen triangle (3 verts cover entire screen)
@vertex
fn vs_main(@builtin(vertex_index) vid: u32) -> VsOut {
  var out: VsOut;
  let x = f32(i32(vid & 1u)) * 4.0 - 1.0;
  let y = f32(i32(vid >> 1u)) * 4.0 - 1.0;
  out.pos = vec4f(x, y, 1.0, 1.0); // z=1 → far plane
  out.clip_xy = vec2f(x, y);
  return out;
}

@fragment
fn fs_main(@location(0) clip_xy: vec2f) -> @location(0) vec4f {
  // Reconstruct world-space ray direction from clip coords
  let clip_near = vec4f(clip_xy, 0.0, 1.0);
  let clip_far  = vec4f(clip_xy, 1.0, 1.0);
  let world_near = u.inv_view_proj * clip_near;
  let world_far  = u.inv_view_proj * clip_far;
  let ray = normalize(world_far.xyz / world_far.w - world_near.xyz / world_near.w);

  let y = ray.y;

  // Dusk / twilight palette
  let deep_sky  = vec3f(0.01, 0.01, 0.04);   // zenith
  let mid_sky   = vec3f(0.04, 0.04, 0.12);    // upper sky
  let low_sky   = vec3f(0.12, 0.06, 0.16);    // purple haze
  let horizon   = vec3f(0.55, 0.22, 0.10);    // warm orange glow
  let below     = vec3f(0.03, 0.02, 0.02);    // dark ground

  var color: vec3f;
  if (y > 0.0) {
    // Sky
    let t1 = smoothstep(0.0, 0.08, y);   // horizon → low_sky
    let t2 = smoothstep(0.08, 0.35, y);   // low_sky → mid_sky
    let t3 = smoothstep(0.35, 0.7, y);    // mid_sky → deep_sky
    color = mix(horizon, low_sky, t1);
    color = mix(color, mid_sky, t2);
    color = mix(color, deep_sky, t3);
  } else {
    // Below horizon
    let t = smoothstep(0.0, -0.15, y);
    color = mix(horizon * 0.7, below, t);
  }

  // Subtle glow band at horizon
  let glow = exp(-abs(y) * 25.0) * 0.15;
  color += vec3f(glow * 0.9, glow * 0.5, glow * 0.2);

  return vec4f(color, 1.0);
}
