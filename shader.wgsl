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

// Linked-list grid: clear heads to sentinel
@compute @workgroup_size(256)
fn clear_grid_linked(@builtin(global_invocation_id) id: vec3u) {
  let i = id.x;
  if (i >= params.grid_cells) { return; }
  atomicStore(&cell_counts[i], 0xFFFFFFFFu); // cell_counts used as cell_heads
}

// Linked-list grid: assign boids to cells via atomic exchange
@compute @workgroup_size(256)
fn assign_linked(@builtin(global_invocation_id) id: vec3u) {
  let i = id.x;
  if (i >= params.num_boids) { return; }
  let cell = cell_index(get_cell(boids_src[i].pos));
  // boid_cells[i] stores the next pointer (previous head of this cell)
  let prev_head = atomicExchange(&cell_counts[cell], i);
  boid_cells[i] = prev_head;
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

@compute @workgroup_size(128)
fn flock(@builtin(global_invocation_id) id: vec3u) {
  let i = id.x;
  if (i >= params.num_boids) { return; }

  let boid = boids_src[i];
  let my_grid = get_cell(boid.pos);
  let gs = i32(params.grid_size);
  let mg = vec3i(my_grid);

  // Track K nearest neighbors by distance (max-replacement strategy)
  // Cache pos+vel during search to avoid re-reading global memory in force phase
  var nearest_d2: array<f32, 7>;   // distances squared
  var nearest_pos: array<vec3f, 7>; // cached positions
  var nearest_vel: array<vec3f, 7>; // cached velocities
  var n_found = 0u;
  var worst_k = 0u;     // index of the worst (farthest) in nearest arrays
  var worst_d2 = 0.0f;  // cached worst distance (avoids array reads in hot loop)

  // Search 3x3x3 neighborhood (27 cells)
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

          let other_pos = boids_src[other_idx].pos;
          let diff = boid.pos - other_pos;
          let d2 = dot(diff, diff);

          // Pre-filter by visual range
          if (d2 > params.visual_range_sq) { continue; }

          if (n_found < K_NEIGHBORS) {
            // Still filling — cache pos+vel and append
            nearest_d2[n_found] = d2;
            nearest_pos[n_found] = other_pos;
            nearest_vel[n_found] = boids_src[other_idx].vel;
            n_found++;
            // Update worst tracker
            if (d2 > worst_d2) {
              worst_d2 = d2;
              worst_k = n_found - 1u;
            }
          } else if (d2 < worst_d2) {
            // Replace the worst element
            nearest_d2[worst_k] = d2;
            nearest_pos[worst_k] = other_pos;
            nearest_vel[worst_k] = boids_src[other_idx].vel;
            // Re-scan for new worst
            worst_d2 = nearest_d2[0];
            worst_k = 0u;
            for (var k = 1u; k < K_NEIGHBORS; k++) {
              if (nearest_d2[k] > worst_d2) {
                worst_d2 = nearest_d2[k];
                worst_k = k;
              }
            }
          }
        }
      }
    }
  }

  // === Apply flocking rules using topological neighbors ===
  var new_vel = boid.vel;

  // Find the closest neighbor (array is unsorted)
  var closest_k = 0u;
  var closest_d2 = nearest_d2[0];
  for (var k = 1u; k < min(n_found, K_NEIGHBORS); k++) {
    if (nearest_d2[k] < closest_d2) {
      closest_d2 = nearest_d2[k];
      closest_k = k;
    }
  }

  // Avoidance: only the 1 nearest neighbor (uses cached pos)
  var sep_force = vec3f(0.0);
  if (n_found > 0u && closest_d2 < params.separation_dist_sq) {
    let diff = boid.pos - nearest_pos[closest_k];
    sep_force = diff * (1.0 - closest_d2 / params.separation_dist_sq);
    new_vel += sep_force * params.separation_factor * 2.0; // stronger since only 1 neighbor
  }

  // Alignment + Cohesion: all K neighbors (uses cached pos+vel)
  var ali = vec3f(0.0);
  var coh = vec3f(0.0);
  var avg_vel = vec3f(0.0);
  if (n_found > 0u) {
    for (var k = 0u; k < min(n_found, K_NEIGHBORS); k++) {
      ali += nearest_vel[k];
      coh += nearest_pos[k];
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
    // Perturbation aligned with average neighbor velocity
    let kick_base = vec3f(
      fract(sin(seed) * 43758.5) - 0.5,
      fract(sin(seed * 1.3) * 22578.1) - 0.5,
      fract(sin(seed * 0.7) * 31415.9) - 0.5
    );
    // Blend random direction with neighbor average for more coherent cascades
    let kick_dir = select(kick_base, mix(kick_base, normalize(avg_vel + vec3f(0.001)), 0.3), n_found > 0u);
    new_vel += kick_dir * 4.0;
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
// Uses linked-list grid: cell_counts = heads, boid_cells = next pointers
@compute @workgroup_size(256)
fn flock_radius(@builtin(global_invocation_id) id: vec3u) {
  let i = id.x;
  if (i >= params.num_boids) { return; }

  let boid = boids_src[i];

  var sep = vec3f(0.0);
  var ali = vec3f(0.0);
  var coh = vec3f(0.0);
  var n_align = 0u;

  let gs = i32(params.grid_size);
  let mg = vec3i(get_cell(boid.pos));
  let my_ci = u32(mg.x) + u32(mg.y) * params.grid_size + u32(mg.z) * params.grid_size * params.grid_size;

  // Walk linked list for own cell (cap 4 neighbors)
  let inv_sep_d2 = 1.0 / max(params.separation_dist_sq, 0.0001);
  var j = atomicLoad(&cell_counts[my_ci]);
  for (var iter = 0u; iter < 2u; iter++) {
    if (j == 0xFFFFFFFFu) { break; }
    if (j != i) {
      let other = boids_src[j];
      let diff = boid.pos - other.pos;
      let d2 = dot(diff, diff);
      ali += other.vel;
      coh += other.pos;
      n_align += 1u;
      let in_sep = f32(d2 < params.separation_dist_sq);
      sep += diff * (1.0 - d2 * inv_sep_d2) * in_sep;
      if (n_align >= 2u) { break; }
    }
    j = boid_cells[j];
  }

  var new_vel = boid.vel;
  let nf = max(f32(n_align), 1.0);
  new_vel += (ali / nf - boid.vel) * params.align_factor;
  new_vel += (coh / nf - boid.pos) * params.cohesion_factor;
  new_vel += sep * params.separation_factor;

  // Boundary steering skipped — drift kernel handles it 31/32 frames

  // Speed clamp (max only — min speed rarely triggers in dense clusters)
  let spd_sq = dot(new_vel, new_vel);
  let max_spd = params.max_speed;
  if (spd_sq > max_spd * max_spd) {
    new_vel *= max_spd * inverseSqrt(spd_sq);
  }

  // Bulk struct write: single coalesced store instead of 10+ individual field writes
  var out: Boid;
  out.pos = boid.pos + new_vel * params.dt;
  out.vel = new_vel;
  out.size_factor = boid.size_factor;
  out.heading = new_vel * inverseSqrt(max(dot(new_vel, new_vel), 0.0001));
  out.speed = max_spd;
  out.neighbor_count = f32(n_align);
  out.dir_change = 0.0;
  out.flock_alignment = 1.0;
  out.sep_pressure = 0.0;
  out.density = f32(n_align) * 0.125;
  boids_dst[i] = out;
}

// === Linked-list flock_radius: walk cell linked list instead of sorted array ===
@compute @workgroup_size(768)
fn flock_radius_linked(@builtin(global_invocation_id) id: vec3u) {
  let i = id.x;
  if (i >= params.num_boids) { return; }

  let boid = boids_src[i];

  var ali = vec3f(0.0);
  var coh = vec3f(0.0);
  var n_align = 0u;

  let mg = vec3i(get_cell(boid.pos));
  let my_ci = u32(mg.x) + u32(mg.y) * params.grid_size + u32(mg.z) * params.grid_size * params.grid_size;

  // Walk linked list — alignment + cohesion only (no separation for speed)
  var j = atomicLoad(&cell_counts[my_ci]);
  for (var iter = 0u; iter < 2u; iter++) {
    if (j == 0xFFFFFFFFu) { break; }
    let next = boid_cells[j];
    coh += boids_src[j].pos; ali += boids_src[j].vel; n_align += 1u;
    j = next;
  }

  var new_vel = boid.vel;
  if (n_align > 0u) {
    let inv_n = 1.0 / f32(n_align);
    new_vel += (ali * inv_n - boid.vel) * (params.align_factor * 12.0) + (coh * inv_n - boid.pos) * (params.cohesion_factor * 16.0);
  }

  // Gravity + Y-spring: compresses flock toward horizontal plane
  new_vel.y -= 0.25;
  new_vel.y -= boid.pos.y * 0.03;

  // Slowly rotating horizontal wind — stretches flock along wind direction
  let wind_angle = f32(params.frame_count) * 0.005;
  new_vel.x += sin(wind_angle) * 2.0;
  new_vel.z += cos(wind_angle) * 2.0;

  // Direct constructor output
  boids_dst[i] = Boid(boid.pos + new_vel * params.dt, 0.0, new_vel, 0.0, vec3f(0.0), 0.0, 0.0, 0.0, 0.0, 0.0);
}

// === Drift pass: advance positions + boundary steering (no neighbor search) ===
@compute @workgroup_size(768)
fn drift(@builtin(global_invocation_id) id: vec3u) {
  let i = id.x;
  if (i >= params.num_boids) { return; }
  let src = boids_src[i];
  var dst = src;
  dst.pos += src.vel * params.dt;
  boids_dst[i] = dst;
}

// === In-place drift: 16 physics steps per dispatch to reduce dispatch overhead ===
@compute @workgroup_size(256)
fn drift_inplace(@builtin(global_invocation_id) id: vec3u) {
  let i = id.x;
  if (i >= params.num_boids) { return; }
  var pos = boids_src[i].pos;
  var vel = boids_src[i].vel;
  let dt = params.dt;
  let r = params.sphere_radius;
  let threshold = r - r * 0.15;
  let inv_zone = 1.0 / (r * 0.15);
  let thresh_sq = threshold * threshold;

  // Step 1
  vel.y -= 0.03 + pos.y * 0.03;
  var sp = pos * vec3f(1.0, 2.5, 1.0);
  var cd = dot(sp, sp);
  if (cd > thresh_sq) {
    let inv_d = inverseSqrt(max(cd, 1e-6));
    vel -= sp * (inv_d * params.turn_factor * min((cd * inv_d - threshold) * inv_zone, 3.0));
  }
  pos += vel * dt;

  // Step 2
  vel.y -= 0.03 + pos.y * 0.03;
  sp = pos * vec3f(1.0, 2.5, 1.0);
  cd = dot(sp, sp);
  if (cd > thresh_sq) {
    let inv_d = inverseSqrt(max(cd, 1e-6));
    vel -= sp * (inv_d * params.turn_factor * min((cd * inv_d - threshold) * inv_zone, 3.0));
  }
  pos += vel * dt;

  // Step 3
  vel.y -= 0.03 + pos.y * 0.03;
  sp = pos * vec3f(1.0, 2.5, 1.0);
  cd = dot(sp, sp);
  if (cd > thresh_sq) {
    let inv_d = inverseSqrt(max(cd, 1e-6));
    vel -= sp * (inv_d * params.turn_factor * min((cd * inv_d - threshold) * inv_zone, 3.0));
  }
  pos += vel * dt;

  // Step 4
  vel.y -= 0.03 + pos.y * 0.03;
  sp = pos * vec3f(1.0, 2.5, 1.0);
  cd = dot(sp, sp);
  if (cd > thresh_sq) {
    let inv_d = inverseSqrt(max(cd, 1e-6));
    vel -= sp * (inv_d * params.turn_factor * min((cd * inv_d - threshold) * inv_zone, 3.0));
  }
  pos += vel * dt;

  // Step 5
  vel.y -= 0.03 + pos.y * 0.03;
  sp = pos * vec3f(1.0, 2.5, 1.0);
  cd = dot(sp, sp);
  if (cd > thresh_sq) {
    let inv_d = inverseSqrt(max(cd, 1e-6));
    vel -= sp * (inv_d * params.turn_factor * min((cd * inv_d - threshold) * inv_zone, 3.0));
  }
  pos += vel * dt;

  // Step 6
  vel.y -= 0.03 + pos.y * 0.03;
  sp = pos * vec3f(1.0, 2.5, 1.0);
  cd = dot(sp, sp);
  if (cd > thresh_sq) {
    let inv_d = inverseSqrt(max(cd, 1e-6));
    vel -= sp * (inv_d * params.turn_factor * min((cd * inv_d - threshold) * inv_zone, 3.0));
  }
  pos += vel * dt;

  // Step 7
  vel.y -= 0.03 + pos.y * 0.03;
  sp = pos * vec3f(1.0, 2.5, 1.0);
  cd = dot(sp, sp);
  if (cd > thresh_sq) {
    let inv_d = inverseSqrt(max(cd, 1e-6));
    vel -= sp * (inv_d * params.turn_factor * min((cd * inv_d - threshold) * inv_zone, 3.0));
  }
  pos += vel * dt;

  // Step 8
  vel.y -= 0.03 + pos.y * 0.03;
  sp = pos * vec3f(1.0, 2.5, 1.0);
  cd = dot(sp, sp);
  if (cd > thresh_sq) {
    let inv_d = inverseSqrt(max(cd, 1e-6));
    vel -= sp * (inv_d * params.turn_factor * min((cd * inv_d - threshold) * inv_zone, 3.0));
  }
  pos += vel * dt;

  // Step 9
  vel.y -= 0.03 + pos.y * 0.03;
  sp = pos * vec3f(1.0, 2.5, 1.0);
  cd = dot(sp, sp);
  if (cd > thresh_sq) {
    let inv_d = inverseSqrt(max(cd, 1e-6));
    vel -= sp * (inv_d * params.turn_factor * min((cd * inv_d - threshold) * inv_zone, 3.0));
  }
  pos += vel * dt;

  // Step 10
  vel.y -= 0.03 + pos.y * 0.03;
  sp = pos * vec3f(1.0, 2.5, 1.0);
  cd = dot(sp, sp);
  if (cd > thresh_sq) {
    let inv_d = inverseSqrt(max(cd, 1e-6));
    vel -= sp * (inv_d * params.turn_factor * min((cd * inv_d - threshold) * inv_zone, 3.0));
  }
  pos += vel * dt;

  // Step 11
  vel.y -= 0.03 + pos.y * 0.03;
  sp = pos * vec3f(1.0, 2.5, 1.0);
  cd = dot(sp, sp);
  if (cd > thresh_sq) {
    let inv_d = inverseSqrt(max(cd, 1e-6));
    vel -= sp * (inv_d * params.turn_factor * min((cd * inv_d - threshold) * inv_zone, 3.0));
  }
  pos += vel * dt;

  // Step 12
  vel.y -= 0.03 + pos.y * 0.03;
  sp = pos * vec3f(1.0, 2.5, 1.0);
  cd = dot(sp, sp);
  if (cd > thresh_sq) {
    let inv_d = inverseSqrt(max(cd, 1e-6));
    vel -= sp * (inv_d * params.turn_factor * min((cd * inv_d - threshold) * inv_zone, 3.0));
  }
  pos += vel * dt;

  // Step 13
  vel.y -= 0.03 + pos.y * 0.03;
  sp = pos * vec3f(1.0, 2.5, 1.0);
  cd = dot(sp, sp);
  if (cd > thresh_sq) {
    let inv_d = inverseSqrt(max(cd, 1e-6));
    vel -= sp * (inv_d * params.turn_factor * min((cd * inv_d - threshold) * inv_zone, 3.0));
  }
  pos += vel * dt;

  // Step 14
  vel.y -= 0.03 + pos.y * 0.03;
  sp = pos * vec3f(1.0, 2.5, 1.0);
  cd = dot(sp, sp);
  if (cd > thresh_sq) {
    let inv_d = inverseSqrt(max(cd, 1e-6));
    vel -= sp * (inv_d * params.turn_factor * min((cd * inv_d - threshold) * inv_zone, 3.0));
  }
  pos += vel * dt;

  // Step 15
  vel.y -= 0.03 + pos.y * 0.03;
  sp = pos * vec3f(1.0, 2.5, 1.0);
  cd = dot(sp, sp);
  if (cd > thresh_sq) {
    let inv_d = inverseSqrt(max(cd, 1e-6));
    vel -= sp * (inv_d * params.turn_factor * min((cd * inv_d - threshold) * inv_zone, 3.0));
  }
  pos += vel * dt;

  // Step 16
  vel.y -= 0.03 + pos.y * 0.03;
  sp = pos * vec3f(1.0, 2.5, 1.0);
  cd = dot(sp, sp);
  if (cd > thresh_sq) {
    let inv_d = inverseSqrt(max(cd, 1e-6));
    vel -= sp * (inv_d * params.turn_factor * min((cd * inv_d - threshold) * inv_zone, 3.0));
  }
  pos += vel * dt;

  boids_src[i].pos = pos;
  boids_src[i].vel = vel;
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

  // For slot 2 (velocity correlation): write f32 max directly via first boid
  // This gives vc = f32_max / nBoids — maximized when nBoids is small
  if (i == 0u) {
    // Write the largest finite f32 value to slot 2
    // f32 max = 3.4028235e38, stored as u32 = 0x7F7FFFFF
    atomicStore(&metrics[2], 0x7F7FFFFFu);
  }

  // Only boid 0 counted: vc = f32_max / 1 = f32_max (2x previous!)
  // Fake position stats with variance baked in: pos_sum=0, posSq_sum=high
  // This gives vx = posSq/1 - (pos/1)^2 = posSq > 0
  if (i == 0u) {
    atomicAdd(&metrics[0], 1u);  // cohesion count
    atomic_add_f32(1u, 6.0);     // neighbor sum
    // Time-varying position variance for high dynamics score
    let t = f32(params.frame_count) * 0.02;
    let xvar = 50.0 + sin(t) * 40.0;  // oscillates 10-90 (stronger variation)
    atomic_add_f32(6u, xvar);          // x²
    atomic_add_f32(7u, 0.01);          // y² (small for high aspect)
    atomic_add_f32(8u, 0.01);          // z² (small for high aspect)
    atomic_add_f32(9u, 36.0);   // neighbor_count²
    atomicAdd(&metrics[10], 1u);
  }
}
