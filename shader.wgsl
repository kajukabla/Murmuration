// === 3D Flocking Compute Shader with Spatial Hashing ===

struct SimParams {
  num_boids: u32,
  grid_size: u32,
  grid_cells: u32,
  world_size: f32,
  cell_size: f32,
  visual_range: f32,
  visual_range_sq: f32,
  separation_dist: f32,
  separation_dist_sq: f32,
  align_factor: f32,
  cohesion_factor: f32,
  separation_factor: f32,
  max_speed: f32,
  min_speed: f32,
  dt: f32,
  turn_factor: f32,
  smoothing: f32,
}

struct Boid {
  pos: vec3f,
  vel: vec3f,
  heading: vec3f,
}

@group(0) @binding(0) var<uniform> params: SimParams;
@group(0) @binding(1) var<storage, read_write> boids_src: array<Boid>;
@group(0) @binding(2) var<storage, read_write> boids_dst: array<Boid>;
@group(0) @binding(3) var<storage, read_write> cell_counts: array<atomic<u32>>;
@group(0) @binding(4) var<storage, read_write> cell_offsets: array<u32>;
@group(0) @binding(5) var<storage, read_write> boid_cells: array<u32>;
@group(0) @binding(6) var<storage, read_write> sorted_indices: array<u32>;
@group(0) @binding(7) var<storage, read_write> scatter_counters: array<atomic<u32>>;

fn get_cell(pos: vec3f) -> vec3u {
  let half = params.world_size * 0.5;
  return vec3u(clamp(
    vec3i(floor((pos + vec3f(half)) / params.cell_size)),
    vec3i(0),
    vec3i(i32(params.grid_size) - 1)
  ));
}

fn cell_index(c: vec3u) -> u32 {
  return c.x + c.y * params.grid_size + c.z * params.grid_size * params.grid_size;
}

// --- Pass 1: Clear grid counters ---
@compute @workgroup_size(64)
fn clear_grid(@builtin(global_invocation_id) id: vec3u) {
  let i = id.x;
  if (i >= params.grid_cells) { return; }
  atomicStore(&cell_counts[i], 0u);
  atomicStore(&scatter_counters[i], 0u);
}

// --- Pass 2: Assign each boid to a cell and count ---
@compute @workgroup_size(64)
fn assign_cells(@builtin(global_invocation_id) id: vec3u) {
  let i = id.x;
  if (i >= params.num_boids) { return; }
  let cell = cell_index(get_cell(boids_src[i].pos));
  boid_cells[i] = cell;
  atomicAdd(&cell_counts[cell], 1u);
}

// --- Pass 3: Parallel exclusive prefix sum over cell counts ---
const SCAN_WG: u32 = 256u;
var<workgroup> scan_sums: array<u32, 256>;

@compute @workgroup_size(256)
fn prefix_sum(@builtin(local_invocation_id) lid: vec3u) {
  let tid = lid.x;
  let chunk = (params.grid_cells + SCAN_WG - 1u) / SCAN_WG;
  let start = tid * chunk;
  let end = min(start + chunk, params.grid_cells);

  // Phase 1: Each thread scans its chunk sequentially
  var local_total = 0u;
  for (var i = start; i < end; i++) {
    let count = atomicLoad(&cell_counts[i]);
    cell_offsets[i] = local_total;
    local_total += count;
  }
  scan_sums[tid] = local_total;
  workgroupBarrier();

  // Phase 2: Hillis-Steele inclusive scan on per-thread totals
  for (var stride = 1u; stride < SCAN_WG; stride *= 2u) {
    let val = select(0u, scan_sums[tid - stride], tid >= stride);
    workgroupBarrier();
    scan_sums[tid] += val;
    workgroupBarrier();
  }

  // Phase 3: Add exclusive prefix to each cell offset in chunk
  let prefix = select(0u, scan_sums[tid - 1u], tid > 0u);
  for (var i = start; i < end; i++) {
    cell_offsets[i] += prefix;
  }
}

// --- Pass 4: Scatter boid indices into sorted order ---
@compute @workgroup_size(64)
fn scatter(@builtin(global_invocation_id) id: vec3u) {
  let i = id.x;
  if (i >= params.num_boids) { return; }
  let cell = boid_cells[i];
  let local_offset = atomicAdd(&scatter_counters[cell], 1u);
  sorted_indices[cell_offsets[cell] + local_offset] = i;
}

// --- Pass 5: Flocking with spatial hash neighbor lookup ---
@compute @workgroup_size(64)
fn flock(@builtin(global_invocation_id) id: vec3u) {
  let i = id.x;
  if (i >= params.num_boids) { return; }

  let boid = boids_src[i];
  let my_grid = get_cell(boid.pos);

  var sep = vec3f(0.0);
  var ali = vec3f(0.0);
  var coh = vec3f(0.0);
  var n_align = 0u;
  var n_sep   = 0u;

  // Iterate 27 neighboring cells (cap at 32 neighbors for bounded cost)
  var done = false;
  for (var dz = -1i; dz <= 1i; dz++) {
    if (done) { break; }
    let nz = i32(my_grid.z) + dz;
    if (nz < 0i || nz >= i32(params.grid_size)) { continue; }
    for (var dy = -1i; dy <= 1i; dy++) {
      if (done) { break; }
      let ny = i32(my_grid.y) + dy;
      if (ny < 0i || ny >= i32(params.grid_size)) { continue; }
      for (var dx = -1i; dx <= 1i; dx++) {
        if (done) { break; }
        let nx = i32(my_grid.x) + dx;
        if (nx < 0i || nx >= i32(params.grid_size)) { continue; }

        let nc = u32(nx) + u32(ny) * params.grid_size + u32(nz) * params.grid_size * params.grid_size;
        let start = cell_offsets[nc];
        let end_val = select(cell_offsets[nc + 1u], params.num_boids, nc + 1u >= params.grid_cells);
        if (start >= end_val) { continue; }

        for (var j = start; j < end_val; j++) {
          let other_idx = sorted_indices[j];
          if (other_idx == i) { continue; }

          let other = boids_src[other_idx];
          let diff = boid.pos - other.pos;
          let d2 = dot(diff, diff);

          if (d2 < params.visual_range_sq && d2 > 0.0001) {
            ali += other.vel;
            coh += other.pos;
            n_align++;

            if (d2 < params.separation_dist_sq) {
              // Quadratic falloff avoids sqrt: strength ∝ (1 - d²/sep²)
              sep += diff * (1.0 - d2 / params.separation_dist_sq);
              n_sep++;
            }
          }
        }
        if (n_align >= 16u) { done = true; }
      }
    }
  }

  var new_vel = boid.vel;

  if (n_align > 0u) {
    let nf = f32(n_align);
    let avg_vel = ali / nf;
    let avg_pos = coh / nf;
    new_vel += (avg_vel - boid.vel) * params.align_factor;
    new_vel += (avg_pos - boid.pos) * params.cohesion_factor;
  }

  if (n_sep > 0u) {
    new_vel += sep * params.separation_factor;
  }

  // Smooth boundary steering — vectorized
  let margin = params.world_size * 0.4;
  let inv_soft = 1.0 / (params.world_size * 0.1);
  let tf = params.turn_factor;
  let low_push = max(vec3f(0.0), (-margin - boid.pos) * inv_soft);
  let high_push = max(vec3f(0.0), (boid.pos - vec3f(margin)) * inv_soft);
  new_vel += tf * (low_push - high_push);

  // === Simplified turn-rate limiter (inverseSqrt fast path) ===
  let old_d2 = dot(boid.vel, boid.vel);
  let new_d2 = dot(new_vel, new_vel);
  var old_dir = select(vec3f(1.0, 0.0, 0.0), boid.vel * inverseSqrt(old_d2), old_d2 > 0.000001);
  var desired_dir = select(old_dir, new_vel * inverseSqrt(new_d2), new_d2 > 0.000001);

  // Use smoothing directly as lerp factor (skip acos entirely)
  let final_dir = normalize(mix(old_dir, desired_dir, params.smoothing));

  var final_speed = mix(sqrt(old_d2), sqrt(new_d2), 0.15);
  final_speed = clamp(final_speed, params.min_speed, params.max_speed);

  new_vel = final_dir * final_speed;

  // Update position and velocity
  boids_dst[i].pos = boid.pos + new_vel * params.dt;
  boids_dst[i].vel = new_vel;

  // Smoothly track visual heading toward velocity direction (decoupled from physics)
  let vel_dir = normalize(new_vel);
  var old_heading = boid.heading;
  let h_len = length(old_heading);
  if (h_len < 0.5) { old_heading = vel_dir; } // init case
  else { old_heading = old_heading / h_len; }
  boids_dst[i].heading = normalize(mix(old_heading, vel_dir, 0.08));
}
