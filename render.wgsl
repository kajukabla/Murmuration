// render.wgsl — Instanced boid rendering with colormaps and color data sources

struct CameraUniforms {
  view_proj: mat4x4f,
  gradient_id: u32,
  color_source: u32,
  gain: f32,
  auto_range: u32,   // 0=off, 1=on
  auto_min: f32,
  auto_max: f32,
  falloff: f32,
  brightness: f32,
  sphere_radius: f32,
  particle_scale: f32,
  render_mode: u32,
  _pad0: u32,
  camera_pos: vec3f,
  _pad1: u32,
}

struct Boid {
  pos: vec3f,
  size_factor: f32,
  vel: vec3f,
  speed: f32,
  heading: vec3f,
  neighbor_count: f32,
  dir_change: f32,
  flock_alignment: f32,
  sep_pressure: f32,
  density: f32,
}

struct BoidBuffer {
  boids: array<Boid>,
}

@group(0) @binding(0) var<uniform> camera: CameraUniforms;
@group(0) @binding(1) var<storage, read> boid_buf: BoidBuffer;

struct VertexOutput {
  @builtin(position) pos: vec4f,
  @location(0) color: vec3f,
  @location(1) lighting: f32,
  @location(2) world_normal: vec3f,
  @location(3) world_pos: vec3f,
}

// ---------------------------------------------------------------------------
// Piecewise-linear ramp with 5 color stops at t = 0, 0.25, 0.5, 0.75, 1.0
// ---------------------------------------------------------------------------
fn ramp5(t: f32, c0: vec3f, c1: vec3f, c2: vec3f, c3: vec3f, c4: vec3f) -> vec3f {
  let tc = clamp(t, 0.0, 1.0);
  if (tc < 0.25) {
    return mix(c0, c1, tc / 0.25);
  } else if (tc < 0.5) {
    return mix(c1, c2, (tc - 0.25) / 0.25);
  } else if (tc < 0.75) {
    return mix(c2, c3, (tc - 0.5) / 0.25);
  } else {
    return mix(c3, c4, (tc - 0.75) / 0.25);
  }
}

// ---------------------------------------------------------------------------
// Colormaps (10 palettes)
// ---------------------------------------------------------------------------
fn colormap(t: f32, id: u32) -> vec3f {
  switch (id) {
    // 0 - Viridis
    case 0u: {
      return ramp5(t,
        vec3f(0.267, 0.004, 0.329),
        vec3f(0.282, 0.140, 0.458),
        vec3f(0.127, 0.566, 0.551),
        vec3f(0.544, 0.774, 0.247),
        vec3f(0.993, 0.906, 0.144));
    }
    // 1 - Inferno
    case 1u: {
      return ramp5(t,
        vec3f(0.001, 0.000, 0.014),
        vec3f(0.259, 0.038, 0.406),
        vec3f(0.578, 0.148, 0.404),
        vec3f(0.895, 0.412, 0.188),
        vec3f(0.988, 0.998, 0.645));
    }
    // 2 - Magma
    case 2u: {
      return ramp5(t,
        vec3f(0.001, 0.000, 0.014),
        vec3f(0.232, 0.059, 0.437),
        vec3f(0.550, 0.161, 0.506),
        vec3f(0.929, 0.412, 0.365),
        vec3f(0.987, 0.991, 0.749));
    }
    // 3 - Plasma
    case 3u: {
      return ramp5(t,
        vec3f(0.050, 0.030, 0.528),
        vec3f(0.415, 0.009, 0.658),
        vec3f(0.698, 0.165, 0.564),
        vec3f(0.930, 0.411, 0.305),
        vec3f(0.940, 0.975, 0.131));
    }
    // 4 - Turbo
    case 4u: {
      return ramp5(t,
        vec3f(0.190, 0.072, 0.232),
        vec3f(0.133, 0.569, 0.920),
        vec3f(0.246, 0.876, 0.406),
        vec3f(0.928, 0.679, 0.068),
        vec3f(0.659, 0.106, 0.096));
    }
    // 5 - Cividis
    case 5u: {
      return ramp5(t,
        vec3f(0.000, 0.135, 0.305),
        vec3f(0.166, 0.272, 0.385),
        vec3f(0.420, 0.420, 0.420),
        vec3f(0.651, 0.580, 0.349),
        vec3f(0.995, 0.764, 0.210));
    }
    // 6 - Coolwarm
    case 6u: {
      return ramp5(t,
        vec3f(0.230, 0.299, 0.754),
        vec3f(0.552, 0.691, 0.996),
        vec3f(0.866, 0.866, 0.866),
        vec3f(0.956, 0.604, 0.486),
        vec3f(0.706, 0.016, 0.150));
    }
    // 7 - Spectral
    case 7u: {
      return ramp5(t,
        vec3f(0.620, 0.004, 0.259),
        vec3f(0.957, 0.427, 0.263),
        vec3f(0.998, 0.999, 0.746),
        vec3f(0.529, 0.808, 0.502),
        vec3f(0.369, 0.310, 0.635));
    }
    // 8 - Hot
    case 8u: {
      return ramp5(t,
        vec3f(0.042, 0.000, 0.000),
        vec3f(0.586, 0.000, 0.000),
        vec3f(1.000, 0.437, 0.000),
        vec3f(1.000, 0.878, 0.000),
        vec3f(1.000, 1.000, 1.000));
    }
    // 9 - Ocean
    case 9u: {
      return ramp5(t,
        vec3f(0.000, 0.125, 0.000),
        vec3f(0.000, 0.208, 0.271),
        vec3f(0.000, 0.417, 0.542),
        vec3f(0.471, 0.745, 0.784),
        vec3f(1.000, 1.000, 1.000));
    }
    // Default fallback to Viridis
    default: {
      return ramp5(t,
        vec3f(0.267, 0.004, 0.329),
        vec3f(0.282, 0.140, 0.458),
        vec3f(0.127, 0.566, 0.551),
        vec3f(0.544, 0.774, 0.247),
        vec3f(0.993, 0.906, 0.144));
    }
  }
}

// ---------------------------------------------------------------------------
// Color data sources (10 sources)
// ---------------------------------------------------------------------------
// Returns raw value for the selected data source (not yet normalized).
// Gain is applied at the call site.
fn get_color_raw(boid: Boid, instance_id: u32, source: u32) -> f32 {
  switch (source) {
    case 0u: { return dot(normalize(boid.heading), vec3f(0.3, 0.4, 0.3)) + 0.5; } // Velocity Dir [~0-1]
    case 1u: { return boid.speed; }                    // Speed [0-max_speed]
    case 2u: { return boid.dir_change; }               // Direction Change [0-~2]
    case 3u: { return boid.neighbor_count; }            // Neighbor Count [0-16+]
    case 4u: { return (boid.pos.y + 50.0) / 100.0; }   // Altitude [0-1]
    case 5u: { return length(boid.pos) / 50.0; }        // Distance from Center [0-~1.4]
    case 6u: { return boid.flock_alignment * 0.5 + 0.5; } // Flock Alignment [0-1]
    case 7u: { return boid.sep_pressure; }              // Separation Pressure [0-~5]
    case 8u: { return boid.density; }                   // Local Density [0-1]
    case 9u: { return fract(f32(instance_id) * 0.618034); } // Boid ID [0-1]
    default: { return dot(normalize(boid.heading), vec3f(0.3, 0.4, 0.3)) + 0.5; }
  }
}

// ---------------------------------------------------------------------------
// Vertex Shader
// ---------------------------------------------------------------------------
@vertex
fn vs_main(
  @builtin(vertex_index) vid: u32,
  @builtin(instance_index) iid: u32,
) -> VertexOutput {
  
  let boid = boid_buf.boids[iid];

  // Scale factor
  let s = 0.25 * boid.size_factor * camera.particle_scale;

  // Tetrahedron vertices in local space (nose points along +Z)
  let nose  = vec3f( 0.0,  0.0,  1.5) * s;
  let left  = vec3f(-0.5,  0.0, -0.5) * s;
  let right = vec3f( 0.5,  0.0, -0.5) * s;
  let top   = vec3f( 0.0,  0.5, -0.3) * s;

  // 4 faces x 3 vertices = 12 vertices
  var local_pos: vec3f;
  var face_id: u32;

  switch (vid) {
    // Face 0: bottom (nose, left, right)
    case 0u  { local_pos = nose;  face_id = 0u; }
    case 1u  { local_pos = left;  face_id = 0u; }
    case 2u  { local_pos = right; face_id = 0u; }
    // Face 1: left side (nose, top, left)
    case 3u  { local_pos = nose;  face_id = 1u; }
    case 4u  { local_pos = top;   face_id = 1u; }
    case 5u  { local_pos = left;  face_id = 1u; }
    // Face 2: right side (nose, right, top)
    case 6u  { local_pos = nose;  face_id = 2u; }
    case 7u  { local_pos = right; face_id = 2u; }
    case 8u  { local_pos = top;   face_id = 2u; }
    // Face 3: back cap (left, top, right)
    case 9u  { local_pos = left;  face_id = 3u; }
    case 10u { local_pos = top;   face_id = 3u; }
    case 11u { local_pos = right; face_id = 3u; }
    default  { local_pos = vec3f(0.0); face_id = 0u; }
  }

  // Compute face normals in local space
  var face_normal: vec3f;
  switch (face_id) {
    case 0u { face_normal = normalize(cross(left - nose, right - nose)); }
    case 1u { face_normal = normalize(cross(top - nose, left - nose)); }
    case 2u { face_normal = normalize(cross(right - nose, top - nose)); }
    case 3u { face_normal = normalize(cross(top - left, right - left)); }
    default { face_normal = vec3f(0.0, 1.0, 0.0); }
  }

  // Build orientation matrix from boid heading (guard against zero)
  var fwd = boid.heading;
  let h_len = length(fwd);
  if (h_len < 0.001) { fwd = vec3f(1.0, 0.0, 0.0); } else { fwd = fwd / h_len; }

  // Smooth up-vector blend to avoid gimbal lock near vertical
  let world_up = vec3f(0.0, 1.0, 0.0);
  let alt_up   = vec3f(0.0, 0.0, 1.0);
  let up_dot   = abs(dot(fwd, world_up));
  let ref_up   = mix(world_up, alt_up, smoothstep(0.9, 0.99, up_dot));

  let right_dir = normalize(cross(ref_up, fwd));
  let up_dir    = cross(fwd, right_dir);

  // Rotation matrix: columns are right, up, forward
  let rot = mat3x3f(right_dir, up_dir, fwd);

  // Transform to world space
  let world_pos = rot * local_pos + boid.pos;
  let rotated_normal = rot * face_normal;

  // Lighting: abs dot for two-sided illumination
  let light_dir = normalize(vec3f(0.4, 0.8, 0.6));
  let ndotl = abs(dot(rotated_normal, light_dir));
  let lighting = 0.3 + 0.7 * ndotl;

  // Color: map data source through auto-range + gain + colormap
  var raw = get_color_raw(boid, iid, camera.color_source);
  var t: f32;
  if (camera.auto_range > 0u) {
    // Auto-range: normalize to [0,1], then apply gentle contrast curve
    let range = camera.auto_max - camera.auto_min;
    if (range > 0.0001) {
      raw = (raw - camera.auto_min) / range;
    }
    // Gentle power curve for contrast: gain 0.5=linear, <0.5=compress, >0.5=expand
    let gamma = pow(3.0, (0.5 - camera.gain) * 3.0);
    t = clamp(pow(clamp(raw, 0.0, 1.0), gamma), 0.0, 1.0);
  } else {
    // No auto-range: full logarithmic gain as divisor
    let gainMul = pow(10.0, (0.5 - camera.gain) * 4.0);
    t = clamp(raw / gainMul, 0.0, 1.0);
  }
  let base = colormap(t, camera.gradient_id);

  // HDR bloom boost
  let lum = dot(base, vec3f(0.299, 0.587, 0.114));
  var hdr = base;
  if (lum > 0.8) {
    hdr = base * (1.0 + (lum - 0.8) * 1.5);
  }

  var out: VertexOutput;
  out.pos = camera.view_proj * vec4f(world_pos, 1.0);
  out.color = hdr;
  out.lighting = lighting;
  out.world_normal = rotated_normal;
  out.world_pos = world_pos;
  return out;
}

// ---------------------------------------------------------------------------
// Fragment Shader
// ---------------------------------------------------------------------------
// Sample the invisible gradient skybox as an environment map
fn env_reflect(normal: vec3f, view_dir: vec3f) -> vec3f {
  let refl = reflect(view_dir, normal);
  // Map reflection Y to [0,1] for gradient lookup (latitude-based env map)
  let t = clamp(refl.y * 0.5 + 0.5, 0.0, 1.0);
  let env = colormap(t, camera.gradient_id);
  // Add some Fresnel-like rim brightness
  let fresnel = pow(1.0 - max(dot(normal, -view_dir), 0.0), 3.0);
  return env * (0.6 + fresnel * 0.8);
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4f {
  if (camera.render_mode == 3u || camera.render_mode == 4u) {
    // Reflective modes (3=reflective tetra, 4=reflective billboard)
    let view_dir = normalize(in.world_pos - camera.camera_pos);
    let n = normalize(in.world_normal);
    let refl_color = env_reflect(n, view_dir);
    return vec4f(refl_color * camera.brightness, 1.0);
  }
  return vec4f(in.color * in.lighting * camera.brightness, 1.0);
}

// ===========================================================================
// Billboard Mode — Stretched additive sprites
// ===========================================================================

struct BillboardOut {
  @builtin(position) pos: vec4f,
  @location(0) color: vec3f,
  @location(1) uv: vec2f,
}

@vertex
fn vs_billboard(
  @builtin(vertex_index) vid: u32,
  @builtin(instance_index) iid: u32,
) -> BillboardOut {
  
  let boid = boid_buf.boids[iid];

  var fwd = boid.heading;
  let fwd_len = length(fwd);
  if (fwd_len < 0.001) { fwd = vec3f(1.0, 0.0, 0.0); }
  else { fwd = fwd / fwd_len; }

  let clip_center = camera.view_proj * vec4f(boid.pos, 1.0);
  let clip_ahead = camera.view_proj * vec4f(boid.pos + fwd, 1.0);

  let ndc_center = clip_center.xy / clip_center.w;
  let ndc_ahead = clip_ahead.xy / clip_ahead.w;
  var screen_dir = ndc_ahead - ndc_center;
  let screen_len = length(screen_dir);
  if (screen_len < 0.0001) { screen_dir = vec2f(1.0, 0.0); }
  else { screen_dir = screen_dir / screen_len; }

  let screen_perp = vec2f(-screen_dir.y, screen_dir.x);

  let base_size = 0.004 * boid.size_factor * camera.particle_scale;
  let stretch = 1.0 + clamp(boid.speed * 0.15, 0.0, 3.0);
  let half_long = base_size * stretch;
  let half_short = base_size * 0.4;

  var local_uv: vec2f;
  switch (vid) {
    case 0u { local_uv = vec2f(-1.0, -1.0); }
    case 1u { local_uv = vec2f( 1.0, -1.0); }
    case 2u { local_uv = vec2f( 1.0,  1.0); }
    case 3u { local_uv = vec2f(-1.0, -1.0); }
    case 4u { local_uv = vec2f( 1.0,  1.0); }
    case 5u { local_uv = vec2f(-1.0,  1.0); }
    default { local_uv = vec2f(0.0); }
  }

  let offset_ndc = screen_dir * local_uv.x * half_long
                 + screen_perp * local_uv.y * half_short;

  var out: BillboardOut;
  out.pos = clip_center;
  out.pos.x += offset_ndc.x * clip_center.w;
  out.pos.y += offset_ndc.y * clip_center.w;
  out.uv = local_uv;

  let raw = get_color_raw(boid, iid, camera.color_source);
  var t: f32;
  if (camera.auto_range > 0u) {
    let range = camera.auto_max - camera.auto_min;
    if (range > 0.0001) {
      let norm = (raw - camera.auto_min) / range;
      let gamma = pow(3.0, (0.5 - camera.gain) * 3.0);
      t = clamp(pow(clamp(norm, 0.0, 1.0), gamma), 0.0, 1.0);
    } else { t = raw; }
  } else {
    let gainMul = pow(10.0, (0.5 - camera.gain) * 4.0);
    t = clamp(raw / gainMul, 0.0, 1.0);
  }
  let base = colormap(t, camera.gradient_id);
  let lum = dot(base, vec3f(0.299, 0.587, 0.114));
  var hdr = base;
  if (lum > 0.8) { hdr = base * (1.0 + (lum - 0.8) * 1.5); }
  out.color = hdr;

  return out;
}

@fragment
fn fs_billboard(in: BillboardOut) -> @location(0) vec4f {
  let dist = length(in.uv * vec2f(1.0, camera.falloff));
  let alpha = smoothstep(1.0, 0.3, dist);

  if (camera.render_mode == 4u) {
    // Reflective billboard: fake sphere normal from UV
    let nz = sqrt(max(0.0, 1.0 - dot(in.uv, in.uv)));
    if (nz < 0.01) { discard; }
    // Construct view-space normal, approximate world normal
    let sphere_normal = normalize(vec3f(in.uv.x, in.uv.y, nz));
    let view_dir = vec3f(0.0, 0.0, -1.0); // approximate
    let refl_color = env_reflect(sphere_normal, view_dir);
    let sphere_alpha = smoothstep(0.0, 0.1, nz);
    return vec4f(refl_color * camera.brightness * sphere_alpha, sphere_alpha);
  }

  return vec4f(in.color * alpha * camera.brightness, alpha);
}
// CACHE BUST 1774222319
