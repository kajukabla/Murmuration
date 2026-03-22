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
  out.lighting = 0.5 + ndotl * 1.0;

  // HDR color: blue-purple palette with values > 1.0 for bright highlights
  let d = fwd;
  out.color = vec3f(
    0.5 + d.x * 0.35 + d.y * 0.15,
    0.55 + d.y * 0.25 + d.z * 0.15,
    1.1 + d.z * 0.25 - d.x * 0.15,
  );

  return out;
}

@fragment
fn fs_main(@location(0) color: vec3f, @location(1) lighting: f32) -> @location(0) vec4f {
  return vec4f(color * lighting, 1.0);
}
