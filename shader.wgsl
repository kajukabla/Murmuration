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
  sim_speed: f32,
  size_randomness: f32,
  drag_factor: f32,
  gradient_id: u32,
  color_source: u32,
  _pad0: u32,
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

@compute @workgroup_size(64)
fn clear_grid(@builtin(global_invocation_id) id: vec3u) {
  let i = id.x;
  if (i >= params.grid_cells) { return; }
  atomicStore(&cell_counts[i], 0u);
  atomicStore(&scatter_counters[i], 0u);
}

@compute @workgroup_size(64)
fn assign_cells(@builtin(global_invocation_id) id: vec3u) {
  let i = id.x;
  if (i >= params.num_boids) { return; }
  let cell = cell_index(get_cell(boids_src[i].pos));
  boid_cells[i] = cell;
  atomicAdd(&cell_counts[cell], 1u);
}

const SCAN_WG: u32 = 256u;
var<workgroup> scan_sums: array<u32, 256>;

@compute @workgroup_size(256)
fn prefix_sum(@builtin(local_invocation_id) lid: vec3u) {
  let tid = lid.x;
  let chunk = (params.grid_cells + SCAN_WG - 1u) / SCAN_WG;
  let start = tid * chunk;
  let end = min(start + chunk, params.grid_cells);
  var local_total = 0u;
  for (var i = start; i < end; i++) {
    let count = atomicLoad(&cell_counts[i]);
    cell_offsets[i] = local_total;
    local_total += count;
  }
  scan_sums[tid] = local_total;
  workgroupBarrier();
  for (var stride = 1u; stride < SCAN_WG; stride *= 2u) {
    let val = select(0u, scan_sums[tid - stride], tid >= stride);
    workgroupBarrier();
    scan_sums[tid] += val;
    workgroupBarrier();
  }
  let prefix = select(0u, scan_sums[tid - 1u], tid > 0u);
  for (var i = start; i < end; i++) {
    cell_offsets[i] += prefix;
  }
}

@compute @workgroup_size(64)
fn scatter(@builtin(global_invocation_id) id: vec3u) {
  let i = id.x;
  if (i >= params.num_boids) { return; }
  let cell = boid_cells[i];
  let local_offset = atomicAdd(&scatter_counters[cell], 1u);
  sorted_indices[cell_offsets[cell] + local_offset] = i;
}

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

  let gs = i32(params.grid_size);
  let mg = vec3i(my_grid);
  let lo = max(mg - vec3i(1), vec3i(0));
  let hi = min(mg + vec3i(1), vec3i(gs - 1));

  for (var nz = lo.z; nz <= hi.z; nz++) {
    let zoff = u32(nz) * params.grid_size * params.grid_size;
    for (var ny = lo.y; ny <= hi.y; ny++) {
      let yzoff = u32(ny) * params.grid_size + zoff;
      for (var nx = lo.x; nx <= hi.x; nx++) {
        let nc = u32(nx) + yzoff;
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
              sep += diff * (1.0 - d2 / params.separation_dist_sq);
              n_sep++;
            }
          }
        }
        if (n_align >= 16u) { break; }
      }
      if (n_align >= 16u) { break; }
    }
    if (n_align >= 16u) { break; }
  }

  var new_vel = boid.vel;
  var alignment_metric = 0.0;
  if (n_align > 0u) {
    let nf = f32(n_align);
    let avg_vel = ali / nf;
    let avg_pos = coh / nf;
    new_vel += (avg_vel - boid.vel) * params.align_factor;
    new_vel += (avg_pos - boid.pos) * params.cohesion_factor;
    let my_dir = normalize(boid.vel + vec3f(0.0001, 0.0, 0.0));
    let avg_dir = normalize(avg_vel + vec3f(0.0001, 0.0, 0.0));
    alignment_metric = dot(my_dir, avg_dir);
  }
  if (n_sep > 0u) {
    new_vel += sep * params.separation_factor;
  }

  let margin = params.world_size * 0.4;
  let inv_soft = 1.0 / (params.world_size * 0.1);
  let btf = params.turn_factor;
  let low_push = max(vec3f(0.0), (-margin - boid.pos) * inv_soft);
  let high_push = max(vec3f(0.0), (boid.pos - vec3f(margin)) * inv_soft);
  new_vel += btf * (low_push - high_push);

  let old_speed = length(boid.vel);
  var old_dir = boid.vel;
  if (old_speed > 0.001) { old_dir = old_dir / old_speed; }
  else { old_dir = vec3f(1.0, 0.0, 0.0); }
  let desired_speed = length(new_vel);
  var desired_dir = new_vel;
  if (desired_speed > 0.001) { desired_dir = desired_dir / desired_speed; }
  else { desired_dir = old_dir; }
  let final_dir = normalize(mix(old_dir, desired_dir, params.smoothing));

  let drag_scale = 1.0 / mix(1.0, boid.size_factor, params.drag_factor);
  var final_speed = mix(old_speed, desired_speed, 0.15);
  final_speed = clamp(final_speed, params.min_speed * drag_scale, params.max_speed * drag_scale);
  new_vel = final_dir * final_speed;

  let dir_change_val = 1.0 - clamp(dot(old_dir, final_dir), -1.0, 1.0);
  let effective_dt = params.dt * params.sim_speed;

  boids_dst[i].pos = boid.pos + new_vel * effective_dt;
  boids_dst[i].vel = new_vel;
  boids_dst[i].size_factor = boid.size_factor;

  let vel_dir = normalize(new_vel);
  var old_heading = boid.heading;
  let h_len = length(old_heading);
  if (h_len < 0.5) { old_heading = vel_dir; }
  else { old_heading = old_heading / h_len; }
  boids_dst[i].heading = normalize(mix(old_heading, vel_dir, 0.08));

  boids_dst[i].speed = length(new_vel);
  boids_dst[i].neighbor_count = f32(n_align);
  boids_dst[i].dir_change = dir_change_val;
  boids_dst[i].flock_alignment = alignment_metric;
  boids_dst[i].sep_pressure = length(sep);
  let vol = params.visual_range * params.visual_range * params.visual_range;
  boids_dst[i].density = clamp(f32(n_align) / max(vol * 0.01, 1.0), 0.0, 1.0);
}
