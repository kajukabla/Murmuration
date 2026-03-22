// === 3D Boid Render Shader (instanced) ===

struct CameraUniforms {
  view_proj: mat4x4f,
  // inv_view_proj removed — no skybox
}

struct Boid {
  pos: vec3f,
  vel: vec3f,
  heading: vec3f,
}

@group(0) @binding(0) var<uniform> camera: CameraUniforms;
@group(0) @binding(1) var<storage, read> boids: array<Boid>;

struct VsOut {
  @builtin(position) pos: vec4f,
  @location(0) color: vec3f,
  @location(1) lighting: f32,
}

@vertex
fn vs_main(
  @builtin(vertex_index) vid: u32,
  @builtin(instance_index) iid: u32,
) -> VsOut {
  let boid = boids[iid];

  // Build local frame from smoothed heading (decoupled from velocity)
  var fwd = boid.heading;
  let spd = length(fwd);
  if (spd < 0.5) { fwd = normalize(boid.vel); } else { fwd = fwd / spd; }
  let fwd_len = length(fwd);
  if (fwd_len < 0.001) { fwd = vec3f(1.0, 0.0, 0.0); } else { fwd = fwd / fwd_len; }

  // Smooth up-vector: blend from Y-up to Z-up as fwd approaches vertical
  var ref_up = vec3f(0.0, 1.0, 0.0);
  let dot_y = abs(fwd.y);
  if (dot_y > 0.9) {
    let t = smoothstep(0.9, 0.99, dot_y);
    ref_up = normalize(mix(vec3f(0.0, 1.0, 0.0), vec3f(0.0, 0.0, 1.0), t));
  }
  let right = normalize(cross(fwd, ref_up));
  let real_up = normalize(cross(right, fwd));

  let s = 0.25;

  // Tetrahedron vertices
  let tip   = fwd * s * 1.8;
  let left  = -fwd * s + right * s * 0.7;
  let rgt   = -fwd * s - right * s * 0.7;
  let top   = -fwd * s + real_up * s * 0.7;

  // 4 faces × 3 vertices = 12 verts
  var lp: vec3f;
  var face_normal: vec3f;
  switch (vid) {
    // Face 0: bottom (tip, left, right)
    case 0u  { lp = tip; }
    case 1u  { lp = left; }
    case 2u  { lp = rgt; }
    // Face 1: right side (tip, right, top)
    case 3u  { lp = tip; }
    case 4u  { lp = rgt; }
    case 5u  { lp = top; }
    // Face 2: left side (tip, top, left)
    case 6u  { lp = tip; }
    case 7u  { lp = top; }
    case 8u  { lp = left; }
    // Face 3: back cap (rgt, top, left) — wound outward
    case 9u  { lp = rgt; }
    case 10u { lp = top; }
    case 11u { lp = left; }
    default  { lp = vec3f(0.0); }
  }

  // Compute face normal for lighting
  let face_id = vid / 3u;
  var v0: vec3f; var v1: vec3f; var v2: vec3f;
  switch (face_id) {
    case 0u { v0 = tip; v1 = left; v2 = rgt; }
    case 1u { v0 = tip; v1 = rgt;  v2 = top; }
    case 2u { v0 = tip; v1 = top;  v2 = left; }
    case 3u { v0 = rgt; v1 = top; v2 = left; }
    default { v0 = vec3f(0.0); v1 = vec3f(0.0); v2 = vec3f(0.0); }
  }
  face_normal = normalize(cross(v1 - v0, v2 - v0));

  let world = boid.pos + lp;
  let light_dir = normalize(vec3f(0.4, 0.8, 0.6));
  let ndotl = abs(dot(face_normal, light_dir));

  var out: VsOut;
  out.pos = camera.view_proj * vec4f(world, 1.0);
  out.lighting = 0.45 + ndotl * 0.8;

  // Viridis-inspired palette: map velocity direction to [0,1] then sample viridis
  // Viridis goes: dark purple → blue → teal → green → yellow
  let d = fwd;
  let t = clamp(d.x * 0.3 + d.y * 0.4 + d.z * 0.3 + 0.5, 0.0, 1.0);

  // Piecewise viridis approximation (5 stops)
  let c0 = vec3f(0.267, 0.004, 0.329);  // dark purple (t=0)
  let c1 = vec3f(0.282, 0.140, 0.458);  // blue-purple (t=0.25)
  let c2 = vec3f(0.127, 0.566, 0.551);  // teal (t=0.5)
  let c3 = vec3f(0.544, 0.774, 0.247);  // green (t=0.75)
  let c4 = vec3f(0.993, 0.906, 0.144);  // yellow (t=1) — bloom candidate

  var base: vec3f;
  if (t < 0.25) {
    base = mix(c0, c1, t / 0.25);
  } else if (t < 0.5) {
    base = mix(c1, c2, (t - 0.25) / 0.25);
  } else if (t < 0.75) {
    base = mix(c2, c3, (t - 0.5) / 0.25);
  } else {
    base = mix(c3, c4, (t - 0.75) / 0.25);
  }

  // HDR bloom: push bright yellows/greens slightly above 1.0 for HDR glow
  let bloom = smoothstep(0.7, 1.0, t) * 0.3;
  out.color = base + vec3f(bloom * 0.5, bloom * 0.3, 0.0);

  return out;
}

@fragment
fn fs_main(@location(0) color: vec3f, @location(1) lighting: f32) -> @location(0) vec4f {
  return vec4f(color * lighting, 1.0);
}
