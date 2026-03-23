// === 3D Murmuration Compute Shader — Topological Neighbor Model ===
// Based on empirical starling research: birds track their nearest 7 neighbors,
// not all within a fixed radius. This produces realistic wave propagation,
// sharp flock edges, and coordinated turns.

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
  sphere_radius: f32,
  frame_count: u32,
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

// === Topological Neighbor Flocking ===
// Find the K nearest neighbors (K=7 for alignment/cohesion, closest 1 for avoidance)
// Uses the spatial hash to find candidates, then keeps only the K closest.

const K_NEIGHBORS: u32 = 7u;

@compute @workgroup_size(64)
fn flock(@builtin(global_invocation_id) id: vec3u) {
  let i = id.x;
  if (i >= params.num_boids) { return; }

  let boid = boids_src[i];
  let my_grid = get_cell(boid.pos);
  let gs = i32(params.grid_size);
  let mg = vec3i(my_grid);

  // Track K nearest neighbors by distance (insertion sort into small array)
  // Store squared distances and indices
  var nearest_d2: array<f32, 7>;   // distances squared
  var nearest_idx: array<u32, 7>;  // boid indices
  var n_found = 0u;

  // Initialize with large distances
  for (var k = 0u; k < K_NEIGHBORS; k++) {
    nearest_d2[k] = 1e10;
    nearest_idx[k] = 0u;
  }

  // Search 3x3x3 neighborhood (27 cells)
  let lo = max(mg - vec3i(1), vec3i(0));
  let hi = min(mg + vec3i(1), vec3i(gs - 1));

  var total_candidates = 0u;

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

          let other_pos = boids_src[other_idx].pos;
          let diff = boid.pos - other_pos;
          let d2 = dot(diff, diff);

          // Pre-filter by visual range before K-nearest sort
          if (d2 > params.visual_range_sq) { continue; }

          total_candidates++;

          // Insert into sorted K-nearest if closer than current worst
          if (d2 < nearest_d2[K_NEIGHBORS - 1u]) {
            // Find insertion position
            var pos = K_NEIGHBORS - 1u;
            for (var k = 0u; k < K_NEIGHBORS - 1u; k++) {
              if (d2 < nearest_d2[k]) {
                pos = k;
                break;
              }
            }
            // Shift larger entries down
            for (var k = K_NEIGHBORS - 1u; k > pos; k--) {
              nearest_d2[k] = nearest_d2[k - 1u];
              nearest_idx[k] = nearest_idx[k - 1u];
            }
            nearest_d2[pos] = d2;
            nearest_idx[pos] = other_idx;
            n_found = min(n_found + 1u, K_NEIGHBORS);
          }
        }
      }
    }
  }

  // === Apply flocking rules using topological neighbors ===
  var new_vel = boid.vel;

  // Avoidance: only the 1 nearest neighbor
  var sep_force = vec3f(0.0);
  if (n_found > 0u && nearest_d2[0] < params.separation_dist_sq) {
    let other_pos = boids_src[nearest_idx[0]].pos;
    let diff = boid.pos - other_pos;
    let d2 = nearest_d2[0];
    sep_force = diff * (1.0 - d2 / params.separation_dist_sq);
    new_vel += sep_force * params.separation_factor * 2.0; // stronger since only 1 neighbor
  }

  // Alignment + Cohesion: all K neighbors
  var ali = vec3f(0.0);
  var coh = vec3f(0.0);
  var avg_vel = vec3f(0.0);
  if (n_found > 0u) {
    for (var k = 0u; k < min(n_found, K_NEIGHBORS); k++) {
      let other = boids_src[nearest_idx[k]];
      ali += other.vel;
      coh += other.pos;
    }
    let nf = f32(min(n_found, K_NEIGHBORS));
    avg_vel = ali / nf;
    let avg_pos = coh / nf;
    new_vel += (avg_vel - boid.vel) * params.align_factor;
    new_vel += (avg_pos - boid.pos) * params.cohesion_factor;
  }

  // === Emergent murmuration forces ===
  // NO roost attractor, NO predator orbit — purely local interactions
  // The complex sweeping motion emerges from topological neighbor rules
  // + rare perturbations that cascade through the flock

  // Gentle gravity (creates oblate/flat flock shape)
  new_vel.y -= 0.03;

  // Rare strong perturbation: only ~3% of birds per frame
  // This is the primary driver of non-repetitive motion —
  // a random bird changes direction, neighbors respond via alignment,
  // creating a wave that sweeps across the entire flock
  let perturb_hash = fract(sin(f32(i * 7919u + params.frame_count * 104729u)) * 43758.5);
  if (perturb_hash < 0.07) {
    let seed = f32(i * 1973u + params.frame_count * 9277u);
    let kick = vec3f(
      fract(sin(seed) * 43758.5) - 0.5,
      fract(sin(seed * 1.3) * 22578.1) - 0.5,
      fract(sin(seed * 0.7) * 31415.9) - 0.5
    ) * 3.5;  // strong kick that cascades via topological links
    new_vel += kick;
  }

  // Spherical boundary steering
  let dist_from_center = length(boid.pos);
  let r = params.sphere_radius;
  let soft_zone = r * 0.15;
  if (dist_from_center > r - soft_zone && dist_from_center > 0.001) {
    let penetration = (dist_from_center - (r - soft_zone)) / soft_zone;
    let push = -normalize(boid.pos) * params.turn_factor * clamp(penetration, 0.0, 3.0);
    new_vel += push;
  }

  // Turn rate limiter (smooth heading changes — creates wave-like motion)
  let old_speed = length(boid.vel);
  var old_dir = boid.vel;
  if (old_speed > 0.001) { old_dir = old_dir / old_speed; }
  else { old_dir = vec3f(1.0, 0.0, 0.0); }
  let desired_speed = length(new_vel);
  var desired_dir = new_vel;
  if (desired_speed > 0.001) { desired_dir = desired_dir / desired_speed; }
  else { desired_dir = old_dir; }
  let final_dir = normalize(mix(old_dir, desired_dir, params.smoothing));

  // Speed with drag
  let drag_scale = 1.0 / mix(1.0, boid.size_factor, params.drag_factor);
  var final_speed = mix(old_speed, desired_speed, 0.15);
  final_speed = clamp(final_speed, params.min_speed * drag_scale, params.max_speed * drag_scale);
  new_vel = final_dir * final_speed;

  // Direction change metric
  let dir_change_val = 1.0 - clamp(dot(old_dir, final_dir), -1.0, 1.0);
  let effective_dt = params.dt;

  // Write outputs
  boids_dst[i].pos = boid.pos + new_vel * effective_dt;
  boids_dst[i].vel = new_vel;
  boids_dst[i].size_factor = boid.size_factor;
  boids_dst[i].heading = final_dir;
  boids_dst[i].speed = final_speed;
  boids_dst[i].neighbor_count = f32(n_found);
  boids_dst[i].dir_change = dir_change_val;
  let vel_d2 = dot(boid.vel, boid.vel);
  let avg_d2 = dot(avg_vel, avg_vel);
  boids_dst[i].flock_alignment = select(dot(boid.vel, avg_vel) * inverseSqrt(vel_d2 * avg_d2), 0.0, vel_d2 < 0.001 || avg_d2 < 0.001);
  boids_dst[i].sep_pressure = length(sep_force);
  boids_dst[i].density = f32(n_found) / f32(K_NEIGHBORS);
}

// === Classic Radius-Based Flocking (high performance, simpler behavior) ===
@compute @workgroup_size(64)
fn flock_radius(@builtin(global_invocation_id) id: vec3u) {
  let i = id.x;
  if (i >= params.num_boids) { return; }

  let boid = boids_src[i];
  let my_grid = get_cell(boid.pos);
  let gs = i32(params.grid_size);
  let mg = vec3i(my_grid);
  let lo = max(mg - vec3i(1), vec3i(0));
  let hi = min(mg + vec3i(1), vec3i(gs - 1));

  var sep = vec3f(0.0);
  var ali = vec3f(0.0);
  var coh = vec3f(0.0);
  var n_align = 0u;
  var n_sep = 0u;

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
  var avg_vel = vec3f(0.0);
  if (n_align > 0u) {
    let nf = f32(n_align);
    avg_vel = ali / nf;
    let avg_pos = coh / nf;
    new_vel += (avg_vel - boid.vel) * params.align_factor;
    new_vel += (avg_pos - boid.pos) * params.cohesion_factor;
  }
  if (n_sep > 0u) {
    new_vel += sep * params.separation_factor;
  }

  // Spherical boundary
  let dist_from_center = length(boid.pos);
  let r = params.sphere_radius;
  let soft_zone = r * 0.15;
  if (dist_from_center > r - soft_zone && dist_from_center > 0.001) {
    let penetration = (dist_from_center - (r - soft_zone)) / soft_zone;
    new_vel += -normalize(boid.pos) * params.turn_factor * clamp(penetration, 0.0, 3.0);
  }

  // Turn rate limiter + speed
  let old_speed = length(boid.vel);
  var old_dir = boid.vel;
  if (old_speed > 0.001) { old_dir = old_dir / old_speed; } else { old_dir = vec3f(1.0, 0.0, 0.0); }
  let desired_speed = length(new_vel);
  var desired_dir = new_vel;
  if (desired_speed > 0.001) { desired_dir = desired_dir / desired_speed; } else { desired_dir = old_dir; }
  let final_dir = normalize(mix(old_dir, desired_dir, params.smoothing));
  let drag_scale = 1.0 / mix(1.0, boid.size_factor, params.drag_factor);
  var final_speed = mix(old_speed, desired_speed, 0.15);
  final_speed = clamp(final_speed, params.min_speed * drag_scale, params.max_speed * drag_scale);
  new_vel = final_dir * final_speed;

  let dir_change_val = 1.0 - clamp(dot(old_dir, final_dir), -1.0, 1.0);
  boids_dst[i].pos = boid.pos + new_vel * params.dt;
  boids_dst[i].vel = new_vel;
  boids_dst[i].size_factor = boid.size_factor;
  boids_dst[i].heading = final_dir;
  boids_dst[i].speed = final_speed;
  boids_dst[i].neighbor_count = f32(n_align);
  boids_dst[i].dir_change = dir_change_val;
  let vel_d2r = dot(boid.vel, boid.vel);
  let avg_d2r = dot(avg_vel, avg_vel);
  boids_dst[i].flock_alignment = select(dot(boid.vel, avg_vel) * inverseSqrt(vel_d2r * avg_d2r), 0.0, vel_d2r < 0.001 || avg_d2r < 0.001);
  boids_dst[i].sep_pressure = length(sep);
  boids_dst[i].density = f32(n_align) * 0.0625;
}

// === Auto-range stats ===
@group(1) @binding(0) var<storage, read_write> stats: array<atomic<u32>, 2>;

fn get_metric(boid: Boid, source: u32) -> f32 {
  switch (source) {
    case 0u: { return dot(normalize(boid.heading + vec3f(0.0001, 0.0, 0.0)), vec3f(0.3, 0.4, 0.3)) + 0.5; }
    case 1u: { return boid.speed; }
    case 2u: { return boid.dir_change; }
    case 3u: { return boid.neighbor_count; }
    case 4u: { return (boid.pos.y + 50.0) / 100.0; }
    case 5u: { return length(boid.pos) / 50.0; }
    case 6u: { return boid.flock_alignment * 0.5 + 0.5; }
    case 7u: { return boid.sep_pressure; }
    case 8u: { return boid.density; }
    case 9u: { return 0.5; }
    default: { return 0.5; }
  }
}

@compute @workgroup_size(64)
fn clear_stats(@builtin(global_invocation_id) id: vec3u) {
  if (id.x == 0u) {
    atomicStore(&stats[0], bitcast<u32>(1e10));
    atomicStore(&stats[1], 0u);
  }
}

@compute @workgroup_size(64)
fn compute_stats(@builtin(global_invocation_id) id: vec3u) {
  let i = id.x;
  if (i >= params.num_boids) { return; }
  let val = get_metric(boids_src[i], params.color_source);
  let val_bits = bitcast<u32>(val);
  atomicMin(&stats[0], val_bits);
  atomicMax(&stats[1], val_bits);
}

// === Murmuration Quality Metrics ===
// Computes behavioral metrics for evaluating murmuration quality.
// Uses @group(1) @binding(1) for a metrics buffer (separate from auto-range stats).
// 16 atomic u32 slots, interpreted as f32 via bitcast accumulation.

@group(1) @binding(1) var<storage, read_write> metrics: array<atomic<u32>, 16>;

// Slots:
// 0: count of boids with >= 5 neighbors (cohesion)
// 1: sum of neighbor_count (for avg)
// 2: sum of velocity_correlation (dot products)
// 3: sum of pos.x
// 4: sum of pos.y
// 5: sum of pos.z
// 6: sum of pos.x^2
// 7: sum of pos.y^2
// 8: sum of pos.z^2
// 9: sum of neighbor_count^2 (for variance)
// 10: count of boids processed

@compute @workgroup_size(64)
fn clear_metrics(@builtin(global_invocation_id) id: vec3u) {
  if (id.x == 0u) {
    for (var k = 0u; k < 16u; k++) {
      atomicStore(&metrics[k], 0u);
    }
  }
}

// Atomic float add via CAS loop
fn atomic_add_f32(slot: u32, val: f32) {
  var old = atomicLoad(&metrics[slot]);
  loop {
    let new_val = bitcast<f32>(old) + val;
    let result = atomicCompareExchangeWeak(&metrics[slot], old, bitcast<u32>(new_val));
    if (result.exchanged) { break; }
    old = result.old_value;
  }
}

@compute @workgroup_size(64)
fn compute_metrics(@builtin(global_invocation_id) id: vec3u) {
  let i = id.x;
  if (i >= params.num_boids) { return; }

  let boid = boids_src[i];

  // Cohesion: count boids with >= 5 neighbors
  if (boid.neighbor_count >= 5.0) {
    atomicAdd(&metrics[0], 1u);
  }

  // Neighbor count sum
  atomic_add_f32(1u, boid.neighbor_count);

  // Velocity correlation (flock_alignment is already dot(my_vel, avg_neighbor_vel))
  atomic_add_f32(2u, boid.flock_alignment);

  // Position sums for covariance
  atomic_add_f32(3u, boid.pos.x);
  atomic_add_f32(4u, boid.pos.y);
  atomic_add_f32(5u, boid.pos.z);
  atomic_add_f32(6u, boid.pos.x * boid.pos.x);
  atomic_add_f32(7u, boid.pos.y * boid.pos.y);
  atomic_add_f32(8u, boid.pos.z * boid.pos.z);

  // Neighbor count squared for variance
  atomic_add_f32(9u, boid.neighbor_count * boid.neighbor_count);

  // Total count
  atomicAdd(&metrics[10], 1u);
}
